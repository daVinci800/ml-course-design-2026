"""第三轮 EfficientNet-B3：单视图、固定末轮 raw 权重的短时五折训练入口。

模块导入没有数据扫描、模型构造、网络访问或文件写入等副作用。正式模式只从显式的
本地 safetensors 文件加载预训练权重，并在构造模型前校验固定 SHA-256。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm

from src.evaluate import write_fold_oof_csv


SCHEMA = "round3_effnet_b3_v1"
NUM_CLASSES = 5
FORMAL_MODEL = "efficientnet_b3.ra2_in1k"
FORMAL_WEIGHT_SHA256 = "279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469"
DEFAULT_WEIGHT_PATH = Path("pretrained/efficientnet_b3.ra2_in1k/model.safetensors")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEAN_FILL = tuple(round(x * 255) for x in IMAGENET_MEAN)
FORMAL_OUTPUT = Path("runs/round3_effnet_b3")


@dataclass(frozen=True)
class Precision:
    name: str
    autocast_dtype: torch.dtype | None
    scaler_enabled: bool


class SquarePad:
    """用 ImageNet 均值颜色把 PIL 图像补成正方形。"""

    def __init__(self, fill: tuple[int, int, int] = MEAN_FILL) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, (left, top, right, bottom), fill=self.fill)


class RandomD4:
    """均匀采样四次旋转与可选水平反射构成的八个 D4 元素。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        code = int(torch.randint(0, 8, ()).item())
        rotations = code % 4
        reflected = code >= 4
        if rotations:
            operations = (
                Image.Transpose.ROTATE_90,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270,
            )
            image = image.transpose(operations[rotations - 1])
        if reflected:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


def build_transform(image_size: int, training: bool, augment: bool = True) -> transforms.Compose:
    operations: list[Any] = [
        SquarePad(),
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
    ]
    if training and augment:
        operations.extend(
            [
                RandomD4(),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=MEAN_FILL,
                ),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.02,
                ),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


class ManifestDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        paths: Sequence[Path],
        transform: Any,
    ) -> None:
        if len(rows) != len(paths):
            raise ValueError("rows 与 paths 长度不一致")
        self.rows = list(rows)
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        with Image.open(self.paths[index]) as opened:
            image = opened.convert("RGB")
        image = self.transform(image)
        return image, int(self.rows[index]["class_idx"]), index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        saved = state["torch_cuda"]
        if len(saved) != torch.cuda.device_count():
            raise RuntimeError("checkpoint 的 CUDA RNG 设备数与当前环境不一致")
        torch.cuda.set_rng_state_all(saved)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    return device


def choose_precision(device: torch.device, force_fp32: bool = False) -> Precision:
    if force_fp32 or device.type != "cuda":
        return Precision("fp32", None, False)
    if torch.cuda.is_bf16_supported():
        return Precision("bf16", torch.bfloat16, False)
    return Precision("fp16", torch.float16, True)


def autocast_context(device: torch.device, precision: Precision) -> Any:
    if precision.autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.autocast_dtype)


def read_manifest(path: Path) -> tuple[list[dict[str, str]], dict[int, str]]:
    required = {"sample_id", "path", "filename", "class_name", "class_idx", "fold"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少列: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("manifest 为空")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest sample_id 不唯一")
    classes: dict[int, str] = {}
    folds: set[int] = set()
    for row in rows:
        class_idx = int(row["class_idx"])
        fold = int(row["fold"])
        if not 0 <= class_idx < NUM_CLASSES:
            raise ValueError(f"非法 class_idx: {class_idx}")
        if class_idx in classes and classes[class_idx] != row["class_name"]:
            raise ValueError(f"class_idx {class_idx} 对应多个类名")
        classes[class_idx] = row["class_name"]
        folds.add(fold)
    if set(classes) != set(range(NUM_CLASSES)):
        raise ValueError(f"manifest 必须正好包含 {NUM_CLASSES} 个类别")
    if folds != set(range(5)):
        raise ValueError("正式五折 manifest 的 fold 必须正好为 0,1,2,3,4")
    return rows, classes


def resolve_image_path(row: Mapping[str, str], manifest: Path, data_root: Path) -> Path:
    recorded = Path(row["path"])
    candidates: list[Path] = []
    if recorded.is_absolute():
        candidates.append(recorded)
    else:
        candidates.extend(
            [
                Path.cwd() / recorded,
                manifest.parent.parent / recorded,
                data_root / recorded,
                data_root / "train" / row["class_name"] / row["filename"],
            ]
        )
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False)).casefold()
        if normalized not in seen and candidate.is_file():
            return candidate.resolve()
        seen.add(normalized)
    raise FileNotFoundError(f"找不到样本图像 {row['sample_id']}: {row['path']}")


def resolve_all_paths(
    rows: Sequence[Mapping[str, str]], manifest: Path, data_root: Path
) -> list[Path]:
    return [resolve_image_path(row, manifest, data_root) for row in rows]


def create_model(model_name: str, pretrained_path: Path | None, num_classes: int = NUM_CLASSES) -> nn.Module:
    if pretrained_path is None:
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    # file overlay 让 timm 直接读取已校验的本地文件，不经过下载路径。
    return timm.create_model(
        FORMAL_MODEL,
        pretrained=True,
        pretrained_cfg_overlay={"file": str(pretrained_path)},
        num_classes=num_classes,
    )


def is_head_parameter(name: str) -> bool:
    return name == "classifier.weight" or name == "classifier.bias" or name.startswith("classifier.")


def set_head_only(model: nn.Module, enabled: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (not enabled) or is_head_parameter(name)
    if enabled:
        model.eval()
        classifier = model.get_classifier()
        classifier.train()
    else:
        model.train()


def split_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[nn.Parameter]] = {
        ("backbone", "decay"): [],
        ("backbone", "no_decay"): [],
        ("head", "decay"): [],
        ("head", "no_decay"): [],
    }
    for name, parameter in model.named_parameters():
        scope = "head" if is_head_parameter(name) else "backbone"
        decay_kind = "no_decay" if parameter.ndim <= 1 or name.endswith(".bias") else "decay"
        buckets[(scope, decay_kind)].append(parameter)
    if any(not values for values in buckets.values()):
        empty = [f"{key[0]}/{key[1]}" for key, values in buckets.items() if not values]
        raise RuntimeError(f"AdamW 参数组为空: {empty}")
    return [
        {
            "params": buckets[("backbone", "decay")],
            "lr": 0.0,
            "weight_decay": weight_decay,
            "group_name": "backbone_decay",
        },
        {
            "params": buckets[("backbone", "no_decay")],
            "lr": 0.0,
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        },
        {
            "params": buckets[("head", "decay")],
            "lr": 1e-3,
            "weight_decay": weight_decay,
            "group_name": "head_decay",
        },
        {
            "params": buckets[("head", "no_decay")],
            "lr": 1e-3,
            "weight_decay": 0.0,
            "group_name": "head_no_decay",
        },
    ]


class Round3Scheduler:
    """以实际 optimizer update 为单位的固定两阶段学习率计划。"""

    def __init__(self, optimizer: torch.optim.Optimizer, updates_per_epoch: int, epochs: int) -> None:
        if updates_per_epoch <= 0:
            raise ValueError("updates_per_epoch 必须为正")
        self.optimizer = optimizer
        self.updates_per_epoch = updates_per_epoch
        self.epochs = epochs
        self.num_updates = 0

    @staticmethod
    def _linear(index: int, count: int, start: float, end: float) -> float:
        if count <= 1:
            return end
        return start + (end - start) * index / (count - 1)

    @staticmethod
    def _cosine(index: int, count: int, start: float, end: float) -> float:
        if count <= 1:
            return end
        fraction = index / (count - 1)
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * fraction))

    def rates_for_next_update(self) -> tuple[float, float]:
        epoch = self.num_updates // self.updates_per_epoch
        within = self.num_updates % self.updates_per_epoch
        if epoch < 2:
            return 0.0, 1e-3
        if epoch == 2:
            backbone = self._linear(within, self.updates_per_epoch, 1e-5, 1e-4)
            return backbone, 3e-4
        cosine_count = max(1, (self.epochs - 3) * self.updates_per_epoch)
        cosine_index = (epoch - 3) * self.updates_per_epoch + within
        backbone = self._cosine(cosine_index, cosine_count, 1e-4, 1e-6)
        head = self._cosine(cosine_index, cosine_count, 3e-4, 3e-6)
        return backbone, head

    def apply_for_next_update(self) -> tuple[float, float]:
        backbone_lr, head_lr = self.rates_for_next_update()
        for group in self.optimizer.param_groups:
            scope = str(group["group_name"]).split("_", maxsplit=1)[0]
            group["lr"] = backbone_lr if scope == "backbone" else head_lr
        return backbone_lr, head_lr

    def step(self) -> None:
        self.num_updates += 1

    def state_dict(self) -> dict[str, int]:
        return {
            "num_updates": self.num_updates,
            "updates_per_epoch": self.updates_per_epoch,
            "epochs": self.epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["updates_per_epoch"]) != self.updates_per_epoch:
            raise RuntimeError("scheduler updates_per_epoch 与当前配置不一致")
        if int(state["epochs"]) != self.epochs:
            raise RuntimeError("scheduler epochs 与当前配置不一致")
        self.num_updates = int(state["num_updates"])


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def confusion_metrics(confusion: torch.Tensor) -> tuple[float, float]:
    confusion = confusion.to(dtype=torch.float64)
    total = confusion.sum().item()
    accuracy = confusion.diag().sum().item() / total if total else 0.0
    denominator = 2 * confusion.diag() + confusion.sum(dim=0) - confusion.diag() + confusion.sum(dim=1) - confusion.diag()
    f1 = torch.where(denominator > 0, 2 * confusion.diag() / denominator, torch.zeros_like(denominator))
    return accuracy, f1.mean().item()


def update_confusion(confusion: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor) -> None:
    encoded = targets.detach().to("cpu", torch.int64) * NUM_CLASSES + predictions.detach().to("cpu", torch.int64)
    confusion += torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Round3Scheduler,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    precision: Precision,
    accumulation_steps: int,
    grad_clip: float,
) -> dict[str, Any]:
    num_batches = len(loader)
    if num_batches == 0:
        raise RuntimeError("训练 DataLoader 为空")
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_sum = 0.0
    sample_count = 0
    optimizer_steps = 0
    skipped_steps = 0
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    consumed = 0
    while consumed < num_batches:
        window = min(accumulation_steps, num_batches - consumed)
        for _ in range(window):
            images, targets, _indices = next(iterator)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, precision):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss / window).backward()
            batch_size = targets.shape[0]
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            update_confusion(confusion, targets, logits.argmax(dim=1))
            consumed += 1
        scheduler.apply_for_next_update()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        # FP16 溢出时 GradScaler 跳过了 optimizer.step；此时不得推进 scheduler。
        if not scaler.is_enabled() or new_scale >= old_scale:
            scheduler.step()
            optimizer_steps += 1
        else:
            skipped_steps += 1
        optimizer.zero_grad(set_to_none=True)
    accuracy, macro_f1 = confusion_metrics(confusion)
    return {
        "loss": loss_sum / sample_count,
        "acc": accuracy,
        "macro_f1": macro_f1,
        "samples": sample_count,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
    }


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    precision: Precision,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_sum = 0.0
    sample_count = 0
    for images, targets, _indices in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, precision):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = targets.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        update_confusion(confusion, targets, logits.argmax(dim=1))
    accuracy, macro_f1 = confusion_metrics(confusion)
    return {
        "loss": loss_sum / sample_count,
        "acc": accuracy,
        "macro_f1": macro_f1,
        "samples": sample_count,
    }


@torch.inference_mode()
def predict_rotation_tta(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    precision: Precision,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = np.zeros((len(loader.dataset), NUM_CLASSES), dtype=np.float32)
    seen = np.zeros(len(loader.dataset), dtype=np.bool_)
    for images, _targets, indices in loader:
        images = images.to(device, non_blocking=True)
        mean_probability = torch.zeros((images.shape[0], NUM_CLASSES), device=device, dtype=torch.float32)
        for k in range(4):
            rotated = torch.rot90(images, k=k, dims=(-2, -1))
            with autocast_context(device, precision):
                logits = model(rotated)
            # softmax 必须在 float32 中执行；仅把低精度 softmax 结果 cast 为
            # float32 会保留其概率和误差，无法满足标准 OOF 的严格校验。
            mean_probability.add_(torch.softmax(logits.float(), dim=1), alpha=0.25)
        numpy_indices = indices.numpy()
        if seen[numpy_indices].any():
            raise RuntimeError("OOF DataLoader 返回了重复索引")
        probabilities[numpy_indices] = mean_probability.cpu().numpy()
        seen[numpy_indices] = True
    if not seen.all():
        raise RuntimeError("OOF DataLoader 未覆盖全部验证样本")
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    return probabilities, predictions


def make_loader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # 每个 epoch 重建 worker，使 loader generator 的边界状态足以严格恢复；
        # persistent worker 的内部 RNG 无法由主进程 checkpoint 完整捕获。
        persistent_workers=False,
        generator=generator,
        drop_last=False,
    )


def formal_recipe(args: argparse.Namespace, manifest_sha: str, weight_sha: str, mode: str) -> dict[str, Any]:
    smoke = mode == "smoke"
    return {
        "schema": SCHEMA,
        "mode": mode,
        "model": "resnet18" if smoke else FORMAL_MODEL,
        "pretrained": not smoke,
        "weight_sha256": weight_sha,
        "manifest_sha256": manifest_sha,
        "num_classes": NUM_CLASSES,
        "num_folds": 5,
        "image_size": 64 if smoke else 288,
        "epochs": 1 if smoke else 30,
        "head_only_epochs": 0 if smoke else 2,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accum_steps,
        "optimizer": "AdamW",
        "weight_decay": 1e-2,
        "no_decay_rule": "bias_or_ndim_le_1",
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
        "gradient_clip_norm": 1.0,
        "scheduler": {
            "epochs_1_2": {"backbone": 0.0, "head": 1e-3},
            "epoch_3": {"backbone_linear": [1e-5, 1e-4], "head": 3e-4},
            "epochs_4_30": {"backbone_cosine": [1e-4, 1e-6], "head_cosine": [3e-4, 3e-6]},
            "unit": "successful_optimizer_update",
        },
        "augmentation": "none" if smoke else {
            "square_pad_fill": list(MEAN_FILL),
            "resize": 288,
            "random_d4": True,
            "affine_translate": 0.05,
            "affine_scale": [0.9, 1.1],
            "color_jitter": [0.15, 0.15, 0.15, 0.02],
        },
        "validation": "deterministic_full_square_pad_resize",
        "oof_tta_rotations": [0, 90, 180, 270],
        "selection": "fixed_final_epoch_raw",
        "ema": False,
        "precision": "bf16",
        "seed": args.seed,
        "num_workers": args.num_workers,
    }


def checkpoint_payload(
    *,
    fold: int,
    next_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Round3Scheduler,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    config: Mapping[str, Any],
    config_hash: str,
    manifest_info: Mapping[str, Any],
    weight_sha: str,
    history: Sequence[Mapping[str, Any]],
    precision: Precision,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "epoch_boundary": True,
        "fold": fold,
        "next_epoch": next_epoch,
        "global_update": scheduler.num_updates,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(),
        "loader_rng": loader_generator.get_state(),
        "config": dict(config),
        "config_hash": config_hash,
        "manifest": dict(manifest_info),
        "weight_sha256": weight_sha,
        "history": list(history),
        "precision": precision.name,
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    # RNG state 必须保持 CPU ByteTensor；optimizer.load_state_dict 会按参数设备迁移
    # optimizer state，因此统一先加载到 CPU，亦可降低恢复瞬间的显存峰值。
    del device
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 compatibility
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint 顶层必须是 dict")
    return checkpoint


def strict_restore(
    checkpoint: Mapping[str, Any],
    *,
    fold: int,
    config_hash: str,
    manifest_sha: str,
    weight_sha: str,
    epochs: int,
    precision: Precision,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Round3Scheduler,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
) -> tuple[int, list[dict[str, Any]]]:
    checks = {
        "schema": (checkpoint.get("schema"), SCHEMA),
        "epoch_boundary": (checkpoint.get("epoch_boundary"), True),
        "fold": (checkpoint.get("fold"), fold),
        "config_hash": (checkpoint.get("config_hash"), config_hash),
        "manifest_sha256": ((checkpoint.get("manifest") or {}).get("sha256"), manifest_sha),
        "weight_sha256": (checkpoint.get("weight_sha256"), weight_sha),
        "precision": (checkpoint.get("precision"), precision.name),
    }
    mismatches = [f"{name}: checkpoint={actual!r}, current={expected!r}" for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise RuntimeError("严格 resume 校验失败：\n" + "\n".join(mismatches))
    next_epoch = int(checkpoint["next_epoch"])
    if not 0 <= next_epoch <= epochs:
        raise RuntimeError(f"checkpoint next_epoch 越界: {next_epoch}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scheduler.num_updates != int(checkpoint["global_update"]):
        raise RuntimeError("checkpoint global_update 与 scheduler 不一致")
    scaler.load_state_dict(checkpoint["scaler"])
    loader_generator.set_state(checkpoint["loader_rng"])
    restore_rng_state(checkpoint["rng"])
    history = [dict(item) for item in checkpoint.get("history", [])]
    if len(history) != next_epoch:
        raise RuntimeError("checkpoint history 长度与 next_epoch 不一致")
    return next_epoch, history


def resolve_resume_path(resume: str | None, output_dir: Path, fold: int, folds: Sequence[int]) -> Path | None:
    if resume is None:
        return None
    if resume == "auto":
        candidate = output_dir / f"fold_{fold}" / "last.pt"
        # auto 表示“有 checkpoint 就恢复，没有就从头开始”。这样同一条命令既能
        # 用于全新实验，也能在中断后恢复已开始的折并继续尚未开始的折。
        return candidate if candidate.is_file() else None
    if len(folds) != 1:
        raise ValueError("显式 --resume PATH 只能与单个 --folds 值一起使用；多折请用 --resume auto")
    return Path(resume).expanduser().resolve()


def update_training_summary(output_dir: Path, fold: int, fold_result: Mapping[str, Any], config_hash: str) -> None:
    path = output_dir / "training_summary.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            current = json.load(handle)
        if not isinstance(current, dict) or current.get("schema") != SCHEMA:
            raise RuntimeError("现有 training_summary.json schema 不兼容，拒绝覆盖")
        if current.get("config_hash") != config_hash:
            raise RuntimeError("现有 training_summary.json config_hash 不兼容，拒绝合并")
    else:
        current = {"schema": SCHEMA, "config_hash": config_hash, "folds": {}}
    folds = dict(current.get("folds", {}))
    folds[str(fold)] = dict(fold_result)
    current["folds"] = folds
    current["completed_folds"] = sorted(int(key) for key in folds)
    atomic_json_dump(current, path)


def call_oof_writer(
    rows: Sequence[Mapping[str, str]],
    fold: int,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    class_names: Mapping[int, str],
    output_path: Path,
) -> None:
    """调用项目标准 writer；参数适配集中在这里，避免自行定义 CSV 格式。"""
    # 具体调用按 evaluate.write_fold_oof_csv 的公开签名组织；prediction 由标准
    # writer 对概率 argmax 得到，避免维护两套 CSV 语义。
    expected = probabilities.argmax(axis=1)
    if not np.array_equal(expected, predictions):
        raise RuntimeError("传入的 OOF predictions 与 probabilities argmax 不一致")
    write_fold_oof_csv(
        output_path=output_path,
        fold=fold,
        sample_ids=[row["sample_id"] for row in rows],
        y_true=[int(row["class_idx"]) for row in rows],
        probabilities=probabilities,
        class_names=[class_names[index] for index in range(NUM_CLASSES)],
        image_paths=[row["path"] for row in rows],
        sha256s=[row.get("sha256") or None for row in rows],
        group_ids=[row.get("group_id") or None for row in rows],
    )


def run_fold(
    *,
    args: argparse.Namespace,
    fold: int,
    rows: Sequence[dict[str, str]],
    all_paths: Sequence[Path],
    class_names: Mapping[int, str],
    manifest_path: Path,
    manifest_sha: str,
    weight_path: Path | None,
    weight_sha: str,
    output_dir: Path,
    device: torch.device,
    mode: str,
    config: Mapping[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    image_size = int(config["image_size"])
    epochs = int(config["epochs"])
    training_indices = [index for index, row in enumerate(rows) if int(row["fold"]) != fold]
    validation_indices = [index for index, row in enumerate(rows) if int(row["fold"]) == fold]
    if not training_indices or not validation_indices:
        raise RuntimeError(f"fold {fold} 的训练集或验证集为空")
    training_rows = [rows[index] for index in training_indices]
    training_paths = [all_paths[index] for index in training_indices]
    validation_rows = [rows[index] for index in validation_indices]
    validation_paths = [all_paths[index] for index in validation_indices]
    augment = mode == "formal"
    train_dataset = ManifestDataset(training_rows, training_paths, build_transform(image_size, True, augment))
    validation_dataset = ManifestDataset(validation_rows, validation_paths, build_transform(image_size, False))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(fold_seed + 10_000)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, pin_memory, loader_generator
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, args.num_workers, pin_memory
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    model_name = "resnet18" if mode == "smoke" else FORMAL_MODEL
    model = create_model(model_name, weight_path).to(device)
    groups = split_parameter_groups(model, 1e-2)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    scheduler = Round3Scheduler(optimizer, updates_per_epoch, epochs)
    precision = choose_precision(device)
    if mode == "formal" and precision.name != "bf16":
        raise RuntimeError(f"正式训练要求 CUDA BF16，当前精度为 {precision.name}")
    scaler = make_grad_scaler(precision.scaler_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_path = fold_dir / "last.pt"
    resume_path = resolve_resume_path(args.resume, output_dir, fold, args.folds)
    next_epoch = 0
    history: list[dict[str, Any]] = []
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_path}")
        checkpoint = load_checkpoint(resume_path, device)
        next_epoch, history = strict_restore(
            checkpoint,
            fold=fold,
            config_hash=config_hash,
            manifest_sha=manifest_sha,
            weight_sha=weight_sha,
            epochs=epochs,
            precision=precision,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=loader_generator,
        )
        print(f"fold={fold} 严格恢复成功，next_epoch={next_epoch}, global_update={scheduler.num_updates}", flush=True)
    manifest_info = {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "row_count": len(rows),
        "class_names": {str(key): value for key, value in class_names.items()},
    }
    for epoch in range(next_epoch, epochs):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        head_only = mode == "formal" and epoch < 2
        set_head_only(model, head_only)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            precision,
            args.accum_steps,
            1.0,
        )
        validation_metrics = validate_one_epoch(model, validation_loader, criterion, device, precision)
        elapsed = time.perf_counter() - started
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        )
        backbone_lr = next(
            float(group["lr"]) for group in optimizer.param_groups if group["group_name"] == "backbone_decay"
        )
        head_lr = next(
            float(group["lr"]) for group in optimizer.param_groups if group["group_name"] == "head_decay"
        )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": validation_metrics,
            "lr": {"backbone": backbone_lr, "head": head_lr},
            "seconds": elapsed,
            "peak_memory_mb": peak_memory_mb,
            "precision": precision.name,
            "global_update": scheduler.num_updates,
        }
        history.append(record)
        payload = checkpoint_payload(
            fold=fold,
            next_epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=loader_generator,
            config=config,
            config_hash=config_hash,
            manifest_info=manifest_info,
            weight_sha=weight_sha,
            history=history,
            precision=precision,
        )
        atomic_torch_save(payload, checkpoint_path)
        atomic_json_dump(history, fold_dir / "metrics.json")
        print(
            f"fold={fold} epoch={epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.5f} train_acc={train_metrics['acc']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} val_loss={validation_metrics['loss']:.5f} "
            f"val_acc={validation_metrics['acc']:.4f} val_f1={validation_metrics['macro_f1']:.4f} "
            f"lr={backbone_lr:.3g}/{head_lr:.3g} time={elapsed:.1f}s memory={peak_memory_mb:.0f}MiB",
            flush=True,
        )
    if len(history) != epochs:
        raise RuntimeError("训练结束时 history 不完整")
    final_path = fold_dir / "final_weights.pt"
    atomic_torch_save(
        {
            "schema": SCHEMA,
            "fold": fold,
            "epoch": epochs,
            "model": model.state_dict(),
            "config_hash": config_hash,
            "manifest_sha256": manifest_sha,
            "weight_sha256": weight_sha,
            "class_names": {str(key): value for key, value in class_names.items()},
        },
        final_path,
    )
    probabilities, predictions = predict_rotation_tta(model, validation_loader, device, precision)
    oof_path = output_dir / "oof" / f"fold_{fold}.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    call_oof_writer(validation_rows, fold, probabilities, predictions, class_names, oof_path)
    oof_confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    targets = torch.tensor([int(row["class_idx"]) for row in validation_rows])
    update_confusion(oof_confusion, targets, torch.from_numpy(predictions))
    oof_acc, oof_f1 = confusion_metrics(oof_confusion)
    result = {
        "fold": fold,
        "epochs": epochs,
        "final_val": history[-1]["val"],
        "oof_tta": {"acc": oof_acc, "macro_f1": oof_f1},
        "checkpoint": str(checkpoint_path.resolve()),
        "final_weights": str(final_path.resolve()),
        "oof": str(oof_path.resolve()),
        "peak_memory_mb": max(float(item["peak_memory_mb"]) for item in history),
    }
    update_training_summary(output_dir, fold, result, config_hash)
    return result


def run_sanity(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    all_paths: Sequence[Path],
    class_names: Mapping[int, str],
    weight_path: Path,
    weight_sha: str,
    manifest_sha: str,
    output_dir: Path,
    device: torch.device,
) -> bool:
    selected_indices: list[int] = []
    for class_idx in range(NUM_CLASSES):
        candidates = [index for index, row in enumerate(rows) if int(row["class_idx"]) == class_idx]
        candidates.sort(key=lambda index: rows[index]["sample_id"])
        if len(candidates) < 10:
            raise RuntimeError(f"类别 {class_names[class_idx]} 不足 10 张，无法执行 sanity")
        selected_indices.extend(candidates[:10])
    selected_indices.sort(key=lambda index: (int(rows[index]["class_idx"]), rows[index]["sample_id"]))
    selected_rows = [rows[index] for index in selected_indices]
    selected_paths = [all_paths[index] for index in selected_indices]
    dataset = ManifestDataset(selected_rows, selected_paths, build_transform(288, False))
    generator = torch.Generator().manual_seed(args.seed + 90_000)
    train_loader = make_loader(dataset, args.batch_size, True, 0, device.type == "cuda", generator)
    eval_loader = make_loader(dataset, args.batch_size, False, 0, device.type == "cuda")
    seed_everything(args.seed)
    model = create_model(FORMAL_MODEL, weight_path).to(device)
    set_head_only(model, False)
    optimizer = torch.optim.AdamW(split_parameter_groups(model, 0.0), lr=3e-4, weight_decay=0.0)
    for group in optimizer.param_groups:
        group["lr"] = 3e-4
        group["weight_decay"] = 0.0
    precision = choose_precision(device, force_fp32=args.sanity_fp32)
    scaler = make_grad_scaler(precision.scaler_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    history: list[dict[str, Any]] = []
    success = False
    for epoch in range(30):
        started = time.perf_counter()
        model.train()
        confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        loss_sum = 0.0
        count = 0
        for images, targets, _indices in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * targets.shape[0]
            count += targets.shape[0]
            update_confusion(confusion, targets, logits.argmax(dim=1))
        clean_metrics = validate_one_epoch(model, eval_loader, criterion, device, precision)
        train_acc, train_f1 = confusion_metrics(confusion)
        record = {
            "epoch": epoch + 1,
            "optimization_train_loss": loss_sum / count,
            "optimization_train_acc": train_acc,
            "optimization_train_macro_f1": train_f1,
            "clean_train": clean_metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"sanity epoch={epoch + 1}/30 clean_train_loss={clean_metrics['loss']:.5f} "
            f"clean_train_acc={clean_metrics['acc']:.4f}",
            flush=True,
        )
        if clean_metrics["acc"] >= 0.98:
            success = True
            break
    output = {
        "schema": SCHEMA,
        "mode": "sanity",
        "success": success,
        "target_clean_train_acc": 0.98,
        "selected_sample_ids": [row["sample_id"] for row in selected_rows],
        "manifest_sha256": manifest_sha,
        "weight_sha256": weight_sha,
        "precision": precision.name,
        "history": history,
    }
    atomic_json_dump(output, output_dir / "sanity_result.json")
    atomic_torch_save(
        {
            "schema": SCHEMA,
            "mode": "sanity",
            "success": success,
            "model": model.state_dict(),
            "selected_sample_ids": output["selected_sample_ids"],
            "manifest_sha256": manifest_sha,
            "weight_sha256": weight_sha,
        },
        output_dir / "sanity_weights.pt",
    )
    return success


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/folds_v2.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--resume", nargs="?", const="auto", default=None, metavar="PATH|auto")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:N")
    parser.add_argument("--seed", type=int, default=2026)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="resnet18、无预训练、64px、1 epoch")
    modes.add_argument("--sanity", action="store_true", help="固定每类 10 张的全模型过拟合检查")
    parser.add_argument("--sanity-fp32", action="store_true", help="sanity 强制 FP32（正式模式无效）")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.accum_steps <= 0:
        parser.error("--batch-size 和 --accum-steps 必须为正")
    if args.num_workers < 0:
        parser.error("--num-workers 不得为负")
    if len(args.folds) != len(set(args.folds)) or any(fold not in range(5) for fold in args.folds):
        parser.error("--folds 必须是不重复的 0..4")
    if args.sanity_fp32 and not args.sanity:
        parser.error("--sanity-fp32 只能与 --sanity 一起使用")
    if (args.smoke or args.sanity) and args.output_dir is None:
        parser.error("--smoke/--sanity 必须显式指定临时 --output-dir，防止污染正式输出")
    if args.sanity and args.resume is not None:
        parser.error("sanity 不支持 --resume")
    if args.sanity and args.folds != [0, 1, 2, 3, 4]:
        parser.error("sanity 固定按类别取样，不使用 --folds；请不要传入该参数")
    if args.output_dir is None:
        args.output_dir = FORMAL_OUTPUT
    try:
        if (args.smoke or args.sanity) and args.output_dir.resolve() == FORMAL_OUTPUT.resolve():
            parser.error("--smoke/--sanity 不得写入正式输出目录")
    except OSError:
        pass
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    rows, class_names = read_manifest(manifest_path)
    all_paths = resolve_all_paths(rows, manifest_path, data_root)
    device = choose_device(args.device)
    mode = "sanity" if args.sanity else "smoke" if args.smoke else "formal"
    if mode == "smoke":
        weight_path = None
        weight_sha = "none"
    else:
        weight_path = args.weights.expanduser().resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(f"本地预训练权重不存在: {weight_path}")
        weight_sha = sha256_file(weight_path)
        if weight_sha != FORMAL_WEIGHT_SHA256:
            raise RuntimeError(
                "本地预训练权重 SHA-256 不匹配："
                f"expected={FORMAL_WEIGHT_SHA256}, actual={weight_sha}, path={weight_path}"
            )
        # 即使第三方库内部行为变化，也明确禁止 Hugging Face Hub 联网。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "formal" and args.resume is None:
        unexpected = [path for path in output_dir.iterdir()]
        if unexpected:
            raise RuntimeError("正式输出目录已非空；请使用 --resume auto 严格恢复，或改用空目录")
    print(
        f"mode={mode} device={device} manifest_sha256={manifest_sha} "
        f"weight_sha256={weight_sha} output={output_dir}",
        flush=True,
    )
    if mode == "sanity":
        assert weight_path is not None
        success = run_sanity(
            args,
            rows,
            all_paths,
            class_names,
            weight_path,
            weight_sha,
            manifest_sha,
            output_dir,
            device,
        )
        if not success:
            print("sanity 失败：30 epoch 内 clean train accuracy 未达到 98%", file=sys.stderr)
            return 2
        print("sanity 通过：clean train accuracy >= 98%", flush=True)
        return 0
    config = formal_recipe(args, manifest_sha, weight_sha, mode)
    config_hash = canonical_hash(config)
    run_metadata = {
        "schema": SCHEMA,
        "config": config,
        "config_hash": config_hash,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha, "rows": len(rows)},
        "weights": {"path": str(weight_path) if weight_path else None, "sha256": weight_sha},
        "output_dir": str(output_dir),
    }
    metadata_path = output_dir / "run_config.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("schema") != SCHEMA or existing.get("config_hash") != config_hash:
            raise RuntimeError("输出目录已有不兼容的 run_config.json，拒绝混写")
    else:
        atomic_json_dump(run_metadata, metadata_path)
    results = []
    for fold in args.folds:
        results.append(
            run_fold(
                args=args,
                fold=fold,
                rows=rows,
                all_paths=all_paths,
                class_names=class_names,
                manifest_path=manifest_path,
                manifest_sha=manifest_sha,
                weight_path=weight_path,
                weight_sha=weight_sha,
                output_dir=output_dir,
                device=device,
                mode=mode,
                config=config,
                config_hash=config_hash,
            )
        )
    print(json.dumps({"schema": SCHEMA, "completed": results}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
