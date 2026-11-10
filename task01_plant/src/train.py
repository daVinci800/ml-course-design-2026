"""第二轮锁定双视图五折训练（schema v3）。

正式入口只接受 JSON recipe，使用预计算并严格验证的 ROI 坐标 cache。每折仅保留
可恢复的 ``last.pt`` 与仅含推理权重/严格 provenance 的 ``final_weights.pt``；不做
best 选择、early stop 或 EMA。导入本模块没有文件或网络副作用。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from .create_folds import validate_manifest
    from .dataset import get_dual_dataloaders, load_records
    from .evaluate import write_fold_oof_csv
    from .model import (create_model, get_model_data_config, get_model_identity,
                        get_parameter_groups, set_backbone_trainable,
                        split_head_backbone_parameters)
    from .roi import (ALGORITHM_NAME, ALGORITHM_VERSION, ROIConfig, config_sha256,
                      file_sha256 as roi_file_sha256, load_manifest_roi_cache,
                      load_test_roi_cache, roi_config_from_mapping)
except ImportError:  # pragma: no cover
    from create_folds import validate_manifest
    from dataset import get_dual_dataloaders, load_records
    from evaluate import write_fold_oof_csv
    from model import (create_model, get_model_data_config, get_model_identity,
                       get_parameter_groups, set_backbone_trainable,
                       split_head_backbone_parameters)
    from roi import (ALGORITHM_NAME, ALGORITHM_VERSION, ROIConfig, config_sha256,
                     file_sha256 as roi_file_sha256, load_manifest_roi_cache,
                     load_test_roi_cache, roi_config_from_mapping)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 3


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in value["cuda"]])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp); os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError): os.unlink(tmp)


def _atomic_json(value: Any, path: Path, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists(): raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
            f.flush(); os.fsync(f.fileno())
        if exclusive:
            try: os.link(tmp, path)
            except FileExistsError as exc: raise FileExistsError(f"拒绝覆盖已有文件: {path}") from exc
        else: os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError): os.unlink(tmp)


def _project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_recipe(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str, Path]:
    source = Path(path).expanduser().resolve()
    recipe = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(recipe, dict) or int(recipe.get("schema_version", -1)) != 3:
        raise ValueError("只接受 schema_version=3 的 round2 recipe")
    required = {"recipe_id", "manifest", "data_root", "slots", "roi", "scheduler",
                "optimizer", "checkpoint", "tta_rotations_degrees"}
    if required - recipe.keys(): raise ValueError(f"recipe 缺少字段: {sorted(required-recipe.keys())}")
    if recipe.get("views") != ["full", "roi"] or recipe.get("view_weights") != [0.5, 0.5]:
        raise ValueError("双视图及 loss 权重必须固定 full/roi=0.5/0.5")
    if recipe.get("train_view_aggregation") != "mean_logits":
        raise ValueError("训练双视图必须使用 mean_logits 后单次 CE")
    if recipe.get("tta_rotations_degrees") != [0, 90, 180, 270]:
        raise ValueError("TTA 必须固定四旋转")
    if float(recipe.get("ema_decay", -1)) > 0 or recipe.get("final_weights") != "raw":
        raise ValueError("第二轮必须 EMA disabled 且 final raw")
    if recipe.get("save_best") or recipe.get("early_stopping"):
        raise ValueError("第二轮禁止 best/early stop")
    if int(recipe["batch_size"]) * int(recipe["gradient_accumulation_steps"]) != int(recipe["effective_batch_size"]):
        raise ValueError("physical batch × accumulation 必须等于 effective batch")
    return recipe, sha256_file(source), source


def roi_config_from_recipe(recipe: Mapping[str, Any]) -> ROIConfig:
    return roi_config_from_mapping(recipe["roi"])


def _class_mapping(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in records:
        name, index = str(row["class_name"]), int(row["class_idx"])
        if name in mapping and mapping[name] != index: raise ValueError("类别映射冲突")
        mapping[name] = index
    if set(mapping.values()) != set(range(len(mapping))): raise ValueError("类别编号必须从0连续")
    return dict(sorted(mapping.items(), key=lambda x: x[1]))


def _cached_resnet50() -> Path:
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--timm--resnet50.a1_in1k" / "snapshots"
    candidates = list(root.glob("*/model.safetensors"))
    if len(candidates) != 1: raise FileNotFoundError("严格 fallback 需要唯一的本地 resnet50.a1_in1k model.safetensors")
    return candidates[0].resolve()


def resolve_slot_model(recipe: Mapping[str, Any], slot: str, *, allow_network: bool = False) -> dict[str, Any]:
    """解析槽位预训练资产；任何正式结果都必须绑定本地文件及 SHA。"""
    if slot not in {"A", "B"}: raise ValueError("slot 必须为 A/B")
    spec = recipe["slots"][slot]
    requested = str(spec["requested_model_id"])
    # 只接受已经完整缓存的 requested 权重；默认不触发隐式网络。
    hub = Path.home()/".cache"/"huggingface"/"hub"/f"models--timm--{requested}"/"snapshots"
    requested_files = list(hub.glob("*/model.safetensors"))
    if len(requested_files) == 1:
        model_id, path, fallback = requested, requested_files[0].resolve(), False
    else:
        model_id, path, fallback = str(spec["fallback_model_id"]), _cached_resnet50(), True
    if model_id != ("resnet50.a1_in1k" if fallback else requested):
        raise ValueError("模型解析违反锁定 fallback 规则")
    return {"slot": slot, "requested_model_id": requested, "actual_model_id": model_id,
            "fallback_used": fallback, "pretrained_path": str(path),
            "pretrained_sha256": sha256_file(path), "pretrained_bytes": path.stat().st_size,
            "seed": int(spec["seed"]), "allow_network": bool(allow_network)}


def _amp_setup(device: torch.device, precision: str) -> tuple[str, torch.dtype | None, Any]:
    if precision != "bf16_auto": raise ValueError("正式 recipe precision 必须是 bf16_auto")
    mode = "fp32"; dtype = None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported(): mode, dtype = "bf16", torch.bfloat16
        else: mode, dtype = "fp16", torch.float16
    try: scaler = torch.amp.GradScaler("cuda", enabled=(mode == "fp16"))
    except TypeError: scaler = torch.cuda.amp.GradScaler(enabled=(mode == "fp16"))
    return mode, dtype, scaler


def _autocast(device: torch.device, dtype: torch.dtype | None):
    return contextlib.nullcontext() if dtype is None else torch.autocast(device_type=device.type, dtype=dtype)


def _loader_states(*loaders: Any) -> dict[str, torch.Tensor]:
    names = ("train", "val")
    result = {}
    for name, loader in zip(names, loaders):
        if loader.generator is None: raise RuntimeError("DataLoader 必须有显式 generator")
        result[name] = loader.generator.get_state().cpu().clone()
    return result


def _restore_loader_states(states: Mapping[str, torch.Tensor], *loaders: Any) -> None:
    for name, loader in zip(("train", "val"), loaders):
        if name not in states or loader.generator is None: raise ValueError("缺少 loader RNG state")
        loader.generator.set_state(states[name].cpu())


class Round2Scheduler:
    """按 optimizer update 推进的两阶段绝对 LR 调度器。"""
    def __init__(self, optimizer: torch.optim.Optimizer, updates_per_epoch: int,
                 epochs: int = 45, head_only_epochs: int = 3,
                 backbone_max: float = 1e-4, head_max: float = 3e-4,
                 backbone_min: float = 1e-6, head_min: float = 3e-6,
                 unfreeze_start: float = 1e-5):
        if updates_per_epoch <= 0: raise ValueError("updates_per_epoch 必须为正")
        self.optimizer=optimizer; self.upe=int(updates_per_epoch); self.epochs=int(epochs)
        self.head_only=int(head_only_epochs); self.bbmax=backbone_max; self.hmax=head_max
        self.bbmin=backbone_min; self.hmin=head_min; self.start=unfreeze_start
        self.next_update=0

    @property
    def total_updates(self) -> int: return self.upe*self.epochs

    def lrs_for_update(self, update: int) -> tuple[float, float]:
        if not 0 <= update < self.total_updates: raise ValueError("scheduler update 越界")
        epoch = update // self.upe
        within = update % self.upe
        if epoch < self.head_only: return 0.0, self.hmax
        if epoch == self.head_only:
            t = within / max(1, self.upe-1)
            return self.start + (self.bbmax-self.start)*t, self.hmax
        # E5..E45 共 41*U 个 update，首尾严格取 max/min。
        index = update - (self.head_only+1)*self.upe
        count = (self.epochs-self.head_only-1)*self.upe
        t = index / max(1, count-1)
        c = 0.5*(1.0+math.cos(math.pi*t))
        return self.bbmin+(self.bbmax-self.bbmin)*c, self.hmin+(self.hmax-self.hmin)*c

    def apply_for_next_update(self) -> tuple[float,float]:
        bb, head = self.lrs_for_update(self.next_update)
        for group in self.optimizer.param_groups:
            component = group.get("component", group.get("name"))
            group["lr"] = bb if component == "backbone" else head
        return bb, head

    def step(self) -> None: self.next_update += 1
    def state_dict(self) -> dict[str, Any]: return {"next_update": self.next_update, "updates_per_epoch":self.upe,
                                                    "epochs":self.epochs, "head_only_epochs":self.head_only}
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["updates_per_epoch"]) != self.upe or int(state["epochs"]) != self.epochs or int(state["head_only_epochs"]) != self.head_only:
            raise ValueError("scheduler state 与当前 recipe 不兼容")
        self.next_update=int(state["next_update"])
        if not 0 <= self.next_update <= self.total_updates: raise ValueError("scheduler next_update 无效")


def _set_phase(model: nn.Module, frozen: bool) -> None:
    set_backbone_trainable(model, not frozen)
    if frozen:
        model.eval()
        classifier = getattr(model, "get_classifier", lambda: None)()
        if isinstance(classifier, nn.Module): classifier.train()
    else: model.train()


def _dual_logits(model: nn.Module, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if views.ndim != 5 or views.shape[1] != 2: raise ValueError("batch 必须为 [B,2,C,H,W]")
    b=views.shape[0]; logits=model(views.reshape(b*2,*views.shape[2:])).reshape(b,2,-1)
    return logits[:,0], logits[:,1]


def _macro_f1(targets: Sequence[int], predictions: Sequence[int], classes: int) -> float:
    scores=[]
    for c in range(classes):
        tp=sum(t==c and p==c for t,p in zip(targets,predictions)); fp=sum(t!=c and p==c for t,p in zip(targets,predictions)); fn=sum(t==c and p!=c for t,p in zip(targets,predictions))
        pr=tp/(tp+fp) if tp+fp else 0.; re=tp/(tp+fn) if tp+fn else 0.
        scores.append(2*pr*re/(pr+re) if pr+re else 0.)
    return sum(scores)/classes


def train_one_epoch(model: nn.Module, loader: Iterable, optimizer: torch.optim.Optimizer,
                    scheduler: Round2Scheduler, criterion: nn.Module, device: torch.device, *,
                    amp_dtype: torch.dtype|None, scaler: Any, grad_clip: float,
                    accumulation_steps: int, num_classes: int, frozen: bool) -> dict[str,float]:
    _set_phase(model, frozen); optimizer.zero_grad(set_to_none=True)
    total=correct=0; loss_sum=0.; ys=[]; ps=[]; updates=0
    batches=len(loader) if hasattr(loader,"__len__") else None
    if batches is None: raise ValueError("严格 accumulation 需要有长度的 loader")
    for i,(views,labels,_ids) in enumerate(loader):
        views=views.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
        window_start=(i//accumulation_steps)*accumulation_steps
        micro_count=min(accumulation_steps,batches-window_start)
        with _autocast(device,amp_dtype):
            full,roi=_dual_logits(model,views)
            aggregated_logits=.5*full+.5*roi
            raw_loss=criterion(aggregated_logits,labels)
            backward_loss=raw_loss/micro_count
        if scaler.is_enabled(): scaler.scale(backward_loss).backward()
        else: backward_loss.backward()
        probs=aggregated_logits.detach().float().softmax(1)
        pred=probs.argmax(1); n=labels.numel(); total+=n; correct+=int((pred==labels).sum())
        loss_sum+=float(raw_loss.detach())*n; ys.extend(labels.cpu().tolist()); ps.extend(pred.cpu().tolist())
        if (i+1)%accumulation_steps==0 or i+1==batches:
            scheduler.apply_for_next_update()
            if scaler.is_enabled(): scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            optimizer.zero_grad(set_to_none=True); scheduler.step(); updates+=1
    if not total: raise RuntimeError("训练 loader 为空")
    return {"loss":loss_sum/total,"accuracy":correct/total,"macro_f1":_macro_f1(ys,ps,num_classes),"optimizer_updates":updates}


@torch.inference_mode()
def validate(model: nn.Module, loader: Iterable, criterion: nn.Module, device: torch.device, *,
             amp_dtype: torch.dtype|None, num_classes: int, collect: bool=False) -> tuple[dict[str,float],dict[str,Any]|None]:
    model.eval(); total=correct=raw_correct=0; clean_loss=raw_loss=0.; ys=[]; ps=[]; ids=[]; probs_out=[]
    for views,labels,sample_ids in loader:
        views=views.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
        with _autocast(device,amp_dtype): full,roi=_dual_logits(model,views)
        pf=full.float().softmax(1); pr=roi.float().softmax(1); probs=.5*pf+.5*pr
        # clean=正式双视图概率平均；raw=两个视图各自的未平均诊断。
        n=labels.numel(); total+=n; pred=probs.argmax(1); correct+=int((pred==labels).sum())
        raw_correct+=int((pf.argmax(1)==labels).sum())+int((pr.argmax(1)==labels).sum())
        clean_loss+=float((-probs[range(n),labels].clamp_min(1e-38).log()).sum())
        raw_loss+=float(criterion(full,labels)+criterion(roi,labels))*n/2
        ys.extend(labels.cpu().tolist()); ps.extend(pred.cpu().tolist())
        if collect: ids.extend(map(str,sample_ids)); probs_out.extend(probs.cpu().tolist())
    metrics={"clean_loss":clean_loss/total,"clean_accuracy":correct/total,
             "clean_macro_f1":_macro_f1(ys,ps,num_classes),"raw_view_loss":raw_loss/total,
             "raw_view_accuracy":raw_correct/(2*total)}
    return metrics, ({"sample_ids":ids,"targets":ys,"probabilities":probs_out} if collect else None)


@torch.inference_mode()
def predict_tta8(model: nn.Module, loader: Iterable, device: torch.device, *, amp_dtype: torch.dtype|None,
                 num_classes: int) -> dict[str,Any]:
    model.eval(); ids=[]; targets=[]; output=[]
    for views,labels,sample_ids in loader:
        if views.ndim!=5 or views.shape[1]!=2: raise ValueError("TTA 输入必须 [B,2,C,H,W]")
        views=views.to(device,non_blocking=True); acc=None
        for view_index in range(2):
            for k in range(4):
                with _autocast(device,amp_dtype): logits=model(torch.rot90(views[:,view_index],k,dims=(-2,-1)))
                p=logits.float().softmax(1); acc=p if acc is None else acc+p
        probabilities=acc/8.0
        if probabilities.shape[1]!=num_classes: raise ValueError("模型类别数不匹配")
        ids.extend(map(str,sample_ids)); targets.extend(labels.tolist()); output.extend(probabilities.cpu().tolist())
    return {"sample_ids":ids,"targets":targets,"probabilities":output}


def validate_checkpoint_compatibility(checkpoint: Mapping[str,Any], config_hash: str,
                                      manifest_hash: str, class_to_idx: Mapping[str,int], *, fold: int,
                                      recipe_sha256: str, roi_cache_sha256: str,
                                      pretrained_sha256: str, run_manifest_sha256: str,
                                      accumulation_steps: int, micro_batches_per_epoch: int,
                                      optimizer_updates_per_epoch: int) -> None:
    if int(checkpoint.get("schema_version",-1))!=3: raise ValueError("checkpoint schema_version 不是 3")
    expected={"config_hash":config_hash,"manifest_sha256":manifest_hash,"recipe_sha256":recipe_sha256,
              "roi_cache_sha256":roi_cache_sha256,"pretrained_sha256":pretrained_sha256,
              "run_manifest_sha256":run_manifest_sha256,"accumulation_steps":accumulation_steps,
              "micro_batches_per_epoch":micro_batches_per_epoch,
              "optimizer_updates_per_epoch":optimizer_updates_per_epoch}
    missing=[key for key in expected if key not in checkpoint]
    if missing: raise ValueError(f"checkpoint 缺少 schema-v3 字段: {missing}")
    for key,value in expected.items():
        if checkpoint[key]!=value: raise ValueError(f"checkpoint {key} 不匹配")
    if dict(checkpoint.get("class_to_idx",{}))!=dict(class_to_idx): raise ValueError("checkpoint class_to_idx 不匹配")
    if int(checkpoint.get("fold",-1))!=fold: raise ValueError("checkpoint fold 不匹配")
    if checkpoint.get("accumulation_boundary") is not True or int(checkpoint.get("pending_micro_batches",-1))!=0:
        raise ValueError("checkpoint 不在 accumulation 边界")
    if int(checkpoint.get("global_optimizer_updates",-1)) != int(checkpoint.get("next_epoch",-1))*optimizer_updates_per_epoch:
        raise ValueError("checkpoint global update 与 next_epoch 不一致")


def _software() -> dict[str,Any]:
    import timm, torchvision
    return {"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
            "torchvision":torchvision.__version__,"timm":timm.__version__,"cuda":torch.version.cuda}


def train_one_fold(recipe: Mapping[str,Any], resolved: Mapping[str,Any], records: Sequence[Mapping[str,Any]],
                   fold: int, *, recipe_sha256: str, manifest_sha256: str, roi_cache: Mapping[str,Any],
                   roi_cache_sha256: str, class_to_idx: Mapping[str,int], output_dir: Path,
                   resume_path: Path|None=None, epochs_override: int|None=None,
                   image_size_override: int|None=None, batch_size_override: int|None=None,
                   accumulation_override: int|None=None, pretrained: bool=True,
                   run_manifest_sha256: str|None=None) -> dict[str,Any]:
    if not run_manifest_sha256 or len(run_manifest_sha256) != 64:
        raise ValueError("train_one_fold 要求已验证的 run_manifest SHA-256")
    seed=int(resolved["seed"])+fold; seed_everything(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs=int(epochs_override or recipe["epochs"]); image_size=int(image_size_override or recipe["image_size"])
    batch_size=int(batch_size_override or recipe["batch_size"]); accum=int(accumulation_override or recipe["gradient_accumulation_steps"])
    if pretrained:
        model=create_model(model_name=resolved["actual_model_id"],num_classes=len(class_to_idx),pretrained=True,
                           pretrained_cfg_overlay={"file":resolved["pretrained_path"]})
    else:
        model=create_model(model_name=resolved["actual_model_id"],num_classes=len(class_to_idx),pretrained=False)
    identity=get_model_identity(model); data_cfg=get_model_data_config(model); model.to(device)
    opt_cfg=recipe["optimizer"]
    groups=get_parameter_groups(model,backbone_lr=0.,head_lr=float(opt_cfg["head_lr"]),
                                weight_decay=float(opt_cfg["weight_decay"]),
                                no_decay_weight_decay=float(opt_cfg.get("no_decay_weight_decay",0.)))
    optimizer=torch.optim.AdamW(groups)
    train_loader,val_loader=get_dual_dataloaders(records,fold,batch_size,image_size=image_size,
        mean=data_cfg["mean"],std=data_cfg["std"],interpolation=data_cfg.get("interpolation","bicubic"),
        num_workers=int(recipe.get("num_workers",0)),pin_memory=device.type=="cuda",seed=seed,
        roi_cache=roi_cache,roi_cache_sha256=roi_cache_sha256,auto_roi=False,
        translate=float(recipe["augmentation"]["max_translate_fraction"]),
        scale=tuple(recipe["augmentation"]["scale_range"]),
        brightness=float(recipe["augmentation"]["color_jitter"]["brightness"]),
        contrast=float(recipe["augmentation"]["color_jitter"]["contrast"]),
        saturation=float(recipe["augmentation"]["color_jitter"]["saturation"]),
        hue=float(recipe["augmentation"]["color_jitter"]["hue"]))
    updates_per_epoch=math.ceil(len(train_loader)/accum)
    sch_cfg=recipe["scheduler"]
    scheduler=Round2Scheduler(optimizer,updates_per_epoch,epochs=epochs,
        head_only_epochs=min(int(recipe["head_only_epochs"]),epochs),
        backbone_max=float(opt_cfg["backbone_lr"]),head_max=float(opt_cfg["head_lr"]),
        backbone_min=float(sch_cfg["backbone_final_lr"]),head_min=float(sch_cfg["head_final_lr"]),
        unfreeze_start=float(sch_cfg["backbone_unfreeze_start_lr"]))
    criterion=nn.CrossEntropyLoss(label_smoothing=float(recipe["loss"]["label_smoothing"]))
    amp_mode,amp_dtype,scaler=_amp_setup(device,str(recipe["precision"]))
    config_hash=stable_hash({"recipe":recipe,"resolved":resolved,"fold":fold,"epochs":epochs,
                             "image_size":image_size,"batch_size":batch_size,"accumulation":accum})
    fold_dir=output_dir/f"fold_{fold}"; fold_dir.mkdir(parents=True,exist_ok=True)
    history=[]; start_epoch=0
    if resume_path:
        checkpoint=torch.load(resume_path,map_location=device,weights_only=False)
        validate_checkpoint_compatibility(checkpoint,config_hash,manifest_sha256,class_to_idx,fold=fold,
            recipe_sha256=recipe_sha256,roi_cache_sha256=roi_cache_sha256,
            pretrained_sha256=resolved["pretrained_sha256"],run_manifest_sha256=run_manifest_sha256,
            accumulation_steps=accum,micro_batches_per_epoch=len(train_loader),
            optimizer_updates_per_epoch=updates_per_epoch)
        model.load_state_dict(checkpoint["model_state"],strict=True); optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"]); scaler.load_state_dict(checkpoint["scaler_state"])
        history=list(checkpoint["history"]); start_epoch=int(checkpoint["next_epoch"])
        if scheduler.next_update != start_epoch*updates_per_epoch: raise ValueError("resume global update 与 epoch 边界不一致")
        restore_rng_state(checkpoint["rng_state"]); _restore_loader_states(checkpoint["loader_rng_state"],train_loader,val_loader)
    for epoch in range(start_epoch,epochs):
        frozen=epoch<int(recipe["head_only_epochs"])
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        begun=time.perf_counter()
        train_metrics=train_one_epoch(model,train_loader,optimizer,scheduler,criterion,device,
            amp_dtype=amp_dtype,scaler=scaler,grad_clip=float(recipe["gradient_clip_norm"]),
            accumulation_steps=accum,num_classes=len(class_to_idx),frozen=frozen)
        val_metrics,_=validate(model,val_loader,criterion,device,amp_dtype=amp_dtype,num_classes=len(class_to_idx))
        if scheduler.next_update!=(epoch+1)*updates_per_epoch: raise RuntimeError("scheduler update 数不符合 epoch 边界")
        row={"epoch":epoch+1,**{f"train_{k}":v for k,v in train_metrics.items()},**val_metrics,
             "seconds":time.perf_counter()-begun,"backbone_frozen":frozen,"amp_mode":amp_mode,
             "global_optimizer_updates":scheduler.next_update,
             "peak_memory_mb":torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else 0.}
        history.append(row)
        payload={"schema_version":3,"kind":"round2_last","slot":resolved["slot"],"fold":fold,
            "epoch":epoch,"next_epoch":epoch+1,"global_optimizer_updates":scheduler.next_update,
            "accumulation_boundary":True,"pending_micro_batches":0,"accumulation_steps":accum,
            "micro_batches_per_epoch":len(train_loader),"optimizer_updates_per_epoch":updates_per_epoch,
            "run_manifest_sha256":run_manifest_sha256,
            "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
            "scheduler_state":scheduler.state_dict(),"scaler_state":scaler.state_dict(),"rng_state":capture_rng_state(),
            "loader_rng_state":_loader_states(train_loader,val_loader),"history":history,"config_hash":config_hash,
            "recipe_id":recipe["recipe_id"],"recipe_sha256":recipe_sha256,"manifest_sha256":manifest_sha256,
            "roi_cache_sha256":roi_cache_sha256,"roi_config_sha256":config_sha256(roi_config_from_recipe(recipe)),
            "pretrained_sha256":resolved["pretrained_sha256"],"pretrained_path":resolved["pretrained_path"],
            "actual_model_id":resolved["actual_model_id"],"model_identity":identity,"class_to_idx":dict(class_to_idx),
            "roi":dict(recipe["roi"]),"tta_rotations_degrees":recipe["tta_rotations_degrees"],
            "tta_views":recipe["tta_views"],"tta_probability_weights":recipe["tta_probability_weights"],
            "final_weights":"raw","amp_mode":amp_mode,"software_versions":_software()}
        _atomic_torch_save(payload,fold_dir/"last.pt"); _atomic_json(history,fold_dir/"history.json")
        print(f"slot={resolved['slot']} fold={fold} epoch={epoch+1}/{epochs} train_acc={row['train_accuracy']:.4f} val={row['clean_accuracy']:.4f}")
    details=predict_tta8(model,val_loader,device,amp_dtype=amp_dtype,num_classes=len(class_to_idx))
    probs=np.asarray(details["probabilities"]); targets=np.asarray(details["targets"]); preds=probs.argmax(1)
    final_metrics={"tta8_accuracy":float((preds==targets).mean()),"samples":len(targets)}
    metadata={"schema_version":3,"kind":"round2_final_inference","slot":resolved["slot"],"fold":fold,
        "epoch":epochs,"recipe_id":recipe["recipe_id"],"recipe_sha256":recipe_sha256,
        "manifest_sha256":manifest_sha256,"roi_cache_sha256":roi_cache_sha256,
        "roi_config_sha256":config_sha256(roi_config_from_recipe(recipe)),"pretrained_sha256":resolved["pretrained_sha256"],
        "pretrained_path":resolved["pretrained_path"],"actual_model_id":resolved["actual_model_id"],
        "model_identity":identity,"class_to_idx":dict(class_to_idx),"roi":dict(recipe["roi"]),
        "tta_rotations_degrees":recipe["tta_rotations_degrees"],"tta_views":recipe["tta_views"],
        "tta_probability_weights":recipe["tta_probability_weights"],"final_weights":"raw","metrics":final_metrics,
        "accumulation_boundary":True,"pending_micro_batches":0,"accumulation_steps":accum,
        "micro_batches_per_epoch":len(train_loader),"optimizer_updates_per_epoch":updates_per_epoch,
        "global_optimizer_updates":scheduler.next_update,"run_manifest_sha256":run_manifest_sha256}
    _atomic_torch_save({**metadata,"model_state":model.state_dict()},fold_dir/"final_weights.pt")
    classes=list(class_to_idx); val_by_id={str(r["sample_id"]):r for r in records if int(r["fold"])==fold}
    ordered=[val_by_id[x] for x in details["sample_ids"]]; oof=output_dir/"oof"/f"fold_{fold}.csv"
    write_fold_oof_csv(oof,fold,details["sample_ids"],details["targets"],details["probabilities"],classes,
        image_paths=[x["path"] for x in ordered],sha256s=[x.get("sha256","") for x in ordered],
        group_ids=[x.get("group_id","") for x in ordered])
    summary={"slot":resolved["slot"],"fold":fold,"final_raw_tta8":final_metrics,"oof":str(oof),
             "last":str(fold_dir/"last.pt"),"final_weights":str(fold_dir/"final_weights.pt")}
    _atomic_json(summary,fold_dir/"summary.json"); return summary


def preflight(recipe_path: str|os.PathLike[str], *, create_run_manifest: bool=False) -> dict[str,Any]:
    recipe,recipe_hash,source=load_recipe(recipe_path); manifest=_project_path(recipe["manifest"])
    data_root=_project_path(recipe["data_root"]); output=_project_path(recipe["output_root"])
    audit=validate_manifest(manifest,project_root=PROJECT_ROOT,n_splits=5,expected_samples=500,expected_classes=5,verify_files=True)
    records=load_records(manifest,data_root=data_root); classes=_class_mapping(records)
    roi_cfg=roi_config_from_recipe(recipe); roi_path=output/"roi_train.json"
    if not roi_path.is_file(): raise FileNotFoundError(f"先生成 ROI cache: {roi_path}")
    roi_cache=load_manifest_roi_cache(roi_path,manifest,project_root=PROJECT_ROOT,config=roi_cfg,verify_files=True)
    resolved={slot:resolve_slot_model(recipe,slot) for slot in ("A","B")}
    payload={"schema_version":3,"recipe":{"path":str(source),"sha256":recipe_hash,"id":recipe["recipe_id"]},
        "manifest":{"path":str(manifest),"sha256":audit.manifest_sha256,"samples":audit.sample_count,
                    "classes":audit.class_count,"folds":audit.fold_count,"grouped":audit.grouped},
        "group_boundary":"image-level folds; no group metadata, same-plant separation cannot be proven" if not audit.grouped else "grouped",
        "roi_cache":{"path":str(roi_path),"sha256":sha256_file(roi_path),"entries":len(roi_cache),
                     "config_sha256":config_sha256(roi_cfg),"algorithm":ALGORITHM_NAME,"version":ALGORITHM_VERSION},
        "class_to_idx":classes,"slots":resolved,"resolved_recipe_sha256":stable_hash({"recipe":recipe,"slots":resolved}),
        "created_after_full_preflight":True}
    if create_run_manifest:
        run_manifest=output/"run_manifest.json"
        if output.exists() and any(output.iterdir()) and not (roi_path.is_file() and set(output.iterdir())=={roi_path}):
            if run_manifest.exists():
                existing=json.loads(run_manifest.read_text(encoding="utf-8"))
                if existing!=payload: raise ValueError("已存在不兼容 round2 run 目录")
            else: raise ValueError("round2 输出目录已有非 ROI 内容，拒绝锁定")
        _atomic_json(payload,run_manifest,exclusive=not run_manifest.exists())
    return payload


def _build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="第二轮 schema-v3 双视图锁定训练")
    p.add_argument("--config",required=True); p.add_argument("--slot",choices=("A","B")); p.add_argument("--folds",nargs="+",type=int)
    p.add_argument("--resume",type=Path); p.add_argument("--preflight",action="store_true")
    p.add_argument("--lock-run-manifest",action="store_true"); p.add_argument("--roi-cache",type=Path)
    p.add_argument("--epochs",type=int,help="仅 smoke/test override"); p.add_argument("--image-size",type=int,help="仅 smoke/test override")
    p.add_argument("--batch-size",type=int,help="仅 smoke/test override"); p.add_argument("--accumulation-steps",type=int,help="仅 smoke/test override")
    p.add_argument("--no-pretrained",action="store_true",help="仅合成 smoke；正式禁止")
    return p


def main(argv: Sequence[str]|None=None)->int:
    args=_build_parser().parse_args(argv)
    try:
        recipe,recipe_hash,_=load_recipe(args.config)
        if args.preflight or args.lock_run_manifest:
            print(json.dumps(preflight(args.config,create_run_manifest=args.lock_run_manifest),ensure_ascii=False)); return 0
        if args.slot is None or not args.folds: raise ValueError("训练必须指定 --slot 和 --folds")
        if args.resume and len(args.folds)!=1: raise ValueError("--resume 只能用于单 fold")
        if any(x not in range(5) for x in args.folds) or len(set(args.folds))!=len(args.folds): raise ValueError("fold 必须为不重复的0..4")
        manifest=_project_path(recipe["manifest"]); data_root=_project_path(recipe["data_root"]); root_output=_project_path(recipe["output_root"]); output=root_output/f"slot_{args.slot}"
        run_manifest_path=root_output/"run_manifest.json"
        if not run_manifest_path.is_file(): raise ValueError("正式训练前必须完成 preflight 并发布 run_manifest.json")
        run_manifest=json.loads(run_manifest_path.read_text(encoding="utf-8-sig")); run_manifest_hash=sha256_file(run_manifest_path)
        if run_manifest.get("recipe",{}).get("sha256")!=recipe_hash: raise ValueError("run_manifest 与 recipe 不匹配")
        records=load_records(manifest,data_root=data_root); classes=_class_mapping(records); manifest_hash=sha256_file(manifest)
        if run_manifest.get("manifest",{}).get("sha256")!=manifest_hash: raise ValueError("run_manifest 与 manifest 不匹配")
        roi_cfg=roi_config_from_recipe(recipe); roi_path=args.roi_cache.resolve() if args.roi_cache else _project_path(recipe["output_root"])/"roi_train.json"
        roi_cache=load_manifest_roi_cache(roi_path,manifest,project_root=PROJECT_ROOT,config=roi_cfg,verify_files=True); roi_hash=sha256_file(roi_path)
        resolved=resolve_slot_model(recipe,args.slot)
        if args.no_pretrained and args.epochs is None: raise ValueError("正式训练不能禁用 pretrained")
        summaries=[]
        for fold in args.folds:
            summaries.append(train_one_fold(recipe,resolved,records,fold,recipe_sha256=recipe_hash,
                manifest_sha256=manifest_hash,roi_cache=roi_cache,roi_cache_sha256=roi_hash,
                class_to_idx=classes,output_dir=output,resume_path=args.resume,
                epochs_override=args.epochs,image_size_override=args.image_size,batch_size_override=args.batch_size,
                accumulation_override=args.accumulation_steps,pretrained=not args.no_pretrained,
                run_manifest_sha256=run_manifest_hash))
        summary_path=output/"training_summary.json"
        merged={"schema_version":3,"slot":args.slot,"run_manifest_sha256":run_manifest_hash,"folds":{}}
        if summary_path.is_file():
            previous=json.loads(summary_path.read_text(encoding="utf-8"))
            if previous.get("schema_version")!=3 or previous.get("slot")!=args.slot or previous.get("run_manifest_sha256")!=run_manifest_hash:
                raise ValueError("已有 training_summary 与当前 run 不兼容")
            merged["folds"].update(previous.get("folds",{}))
        for item in summaries: merged["folds"][str(item["fold"])]=item
        _atomic_json(merged,summary_path)
    except (ValueError,RuntimeError,OSError,csv.Error,json.JSONDecodeError) as exc:
        print(f"训练失败: {exc}",file=sys.stderr); return 2
    return 0


if __name__=="__main__": raise SystemExit(main())
