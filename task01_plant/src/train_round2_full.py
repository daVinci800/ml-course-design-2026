"""按 round2_fast 配方在全部 500 张训练图上训练一个 final model。"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .train_round2_fast import (
    DEFAULT_WEIGHT_PATH,
    FORMAL_MODEL,
    FORMAL_WEIGHT_SHA256,
    NUM_CLASSES,
    ManifestDataset,
    Round2Scheduler,
    atomic_json_dump,
    atomic_torch_save,
    build_transform,
    canonical_hash,
    capture_rng_state,
    choose_device,
    choose_precision,
    create_model,
    load_checkpoint,
    make_grad_scaler,
    make_loader,
    read_manifest,
    resolve_all_paths,
    restore_rng_state,
    seed_everything,
    set_head_only,
    sha256_file,
    split_parameter_groups,
    train_one_epoch,
)

SCHEMA = "round2_fast_full500_v1"
OUTPUT_DIR = Path("runs/round2_fast_full500")


def recipe(args: argparse.Namespace, manifest_sha: str, weight_sha: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "training_scope": "all_500_manifest_samples",
        "trained_on_all_500": True,
        "model": FORMAL_MODEL,
        "pretrained": True,
        "weight_sha256": weight_sha,
        "manifest_sha256": manifest_sha,
        "num_classes": NUM_CLASSES,
        "image_size": 288,
        "epochs": 45,
        "head_only_epochs": 3,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accum_steps,
        "optimizer": "AdamW",
        "weight_decay": 1e-2,
        "no_decay_rule": "bias_or_ndim_le_1",
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
        "gradient_clip_norm": 1.0,
        "scheduler": {
            "epochs_1_3": {"backbone": 0.0, "head": 1e-3},
            "epoch_4": {"backbone_linear": [1e-5, 1e-4], "head": 3e-4},
            "epochs_5_45": {"backbone_cosine": [1e-4, 1e-6], "head_cosine": [3e-4, 3e-6]},
            "unit": "successful_optimizer_update",
        },
        "augmentation": {
            "square_pad_fill": [124, 116, 104],
            "resize": 288,
            "random_d4": True,
            "affine_translate": 0.05,
            "affine_scale": [0.9, 1.1],
            "color_jitter": [0.15, 0.15, 0.15, 0.02],
        },
        "validation": None,
        "early_stopping": False,
        "ema": False,
        "selection": "fixed_final_epoch_raw",
        "seed": args.seed,
        "num_workers": args.num_workers,
    }


def checkpoint_payload(
    *,
    next_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Round2Scheduler,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    config: Mapping[str, Any],
    config_hash: str,
    manifest_info: Mapping[str, Any],
    weight_sha: str,
    history: Sequence[Mapping[str, Any]],
    precision_name: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "epoch_boundary": True,
        "trained_on_all_500": True,
        "is_oof_model": False,
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
        "precision": precision_name,
    }


def restore(
    checkpoint: Mapping[str, Any], *, config_hash: str, manifest_sha: str,
    weight_sha: str, precision_name: str, model: nn.Module,
    optimizer: torch.optim.Optimizer, scheduler: Round2Scheduler,
    scaler: torch.amp.GradScaler, loader_generator: torch.Generator,
) -> tuple[int, list[dict[str, Any]]]:
    checks = {
        "schema": (checkpoint.get("schema"), SCHEMA),
        "epoch_boundary": (checkpoint.get("epoch_boundary"), True),
        "trained_on_all_500": (checkpoint.get("trained_on_all_500"), True),
        "is_oof_model": (checkpoint.get("is_oof_model"), False),
        "config_hash": (checkpoint.get("config_hash"), config_hash),
        "manifest_sha256": ((checkpoint.get("manifest") or {}).get("sha256"), manifest_sha),
        "weight_sha256": (checkpoint.get("weight_sha256"), weight_sha),
        "precision": (checkpoint.get("precision"), precision_name),
    }
    mismatches = [f"{name}: checkpoint={actual!r}, current={expected!r}"
                  for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise RuntimeError("严格 resume 校验失败：\n" + "\n".join(mismatches))
    next_epoch = int(checkpoint["next_epoch"])
    if not 0 <= next_epoch <= 45:
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/folds_v2.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", nargs="?", const="auto", default="auto", metavar="PATH|auto")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.accum_steps <= 0 or args.num_workers < 0:
        parser.error("batch-size/accum-steps 必须为正，num-workers 不能为负")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    weight_path = args.weights.expanduser().resolve()
    if not manifest_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError("manifest 或本地预训练权重不存在")
    manifest_sha = sha256_file(manifest_path)
    weight_sha = sha256_file(weight_path)
    if weight_sha != FORMAL_WEIGHT_SHA256:
        raise RuntimeError(f"预训练权重 SHA 不匹配: {weight_sha}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows, class_names = read_manifest(manifest_path)
    if len(rows) != 500:
        raise RuntimeError(f"full500 要求 manifest 恰为 500 行，实际 {len(rows)}")
    all_paths = resolve_all_paths(rows, manifest_path, data_root)
    config = recipe(args, manifest_sha, weight_sha)
    config_hash = canonical_hash(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    run_metadata = {
        "schema": SCHEMA,
        "trained_on_all_500": True,
        "is_oof_model": False,
        "config": config,
        "config_hash": config_hash,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha, "rows": len(rows)},
        "weights": {"path": str(weight_path), "sha256": weight_sha},
        "output_dir": str(output_dir),
    }
    if run_config_path.exists():
        with run_config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("schema") != SCHEMA or existing.get("config_hash") != config_hash:
            raise RuntimeError("输出目录已有不兼容 run_config.json，拒绝混写")
    else:
        atomic_json_dump(run_metadata, run_config_path)

    seed_everything(args.seed)
    device = choose_device(args.device)
    precision = choose_precision(device)
    dataset = ManifestDataset(rows, all_paths, build_transform(288, training=True, augment=True))
    loader_generator = torch.Generator().manual_seed(args.seed + 10_000)
    loader = make_loader(dataset, args.batch_size, True, args.num_workers,
                         device.type == "cuda", loader_generator)
    updates_per_epoch = math.ceil(len(loader) / args.accum_steps)
    model = create_model(FORMAL_MODEL, weight_path).to(device)
    optimizer = torch.optim.AdamW(split_parameter_groups(model, 1e-2), betas=(0.9, 0.999))
    scheduler = Round2Scheduler(optimizer, updates_per_epoch, 45)
    scaler = make_grad_scaler(precision.scaler_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    checkpoint_path = output_dir / "last.pt"
    final_path = output_dir / "final_weights.pt"
    next_epoch = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        resume_path = checkpoint_path if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        if resume_path.is_file():
            checkpoint = load_checkpoint(resume_path, device)
            next_epoch, history = restore(
                checkpoint, config_hash=config_hash, manifest_sha=manifest_sha,
                weight_sha=weight_sha, precision_name=precision.name, model=model,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                loader_generator=loader_generator,
            )
            print(f"full500 严格恢复成功 next_epoch={next_epoch} global_update={scheduler.num_updates}", flush=True)
        elif args.resume != "auto":
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_path}")
    if next_epoch == 45 and final_path.is_file():
        print(f"full500 已完成，不覆盖现有 final_weights.pt: {final_path}", flush=True)
        return 0
    manifest_info = {
        "path": str(manifest_path), "sha256": manifest_sha, "row_count": len(rows),
        "class_names": {str(key): value for key, value in class_names.items()},
    }
    for epoch in range(next_epoch, 45):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        set_head_only(model, epoch < 3)
        train_metrics = train_one_epoch(
            model, loader, optimizer, scheduler, scaler, criterion, device,
            precision, args.accum_steps, 1.0,
        )
        elapsed = time.perf_counter() - started
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        backbone_lr = next(float(group["lr"]) for group in optimizer.param_groups
                           if group["group_name"] == "backbone_decay")
        head_lr = next(float(group["lr"]) for group in optimizer.param_groups
                       if group["group_name"] == "head_decay")
        record = {
            "epoch": epoch + 1, "train": train_metrics,
            "lr": {"backbone": backbone_lr, "head": head_lr},
            "seconds": elapsed, "peak_memory_mb": peak_memory_mb,
            "precision": precision.name, "global_update": scheduler.num_updates,
            "validation": None,
        }
        history.append(record)
        payload = checkpoint_payload(
            next_epoch=epoch + 1, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, loader_generator=loader_generator,
            config=config, config_hash=config_hash, manifest_info=manifest_info,
            weight_sha=weight_sha, history=history, precision_name=precision.name,
        )
        atomic_torch_save(payload, checkpoint_path)
        atomic_json_dump(history, output_dir / "metrics.json")
        print(
            f"full500 epoch={epoch + 1}/45 train_loss={train_metrics['loss']:.5f} "
            f"train_acc={train_metrics['acc']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"lr={backbone_lr:.3g}/{head_lr:.3g} time={elapsed:.1f}s memory={peak_memory_mb:.0f}MiB",
            flush=True,
        )
    if final_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 final_weights.pt: {final_path}")
    atomic_torch_save({
        "schema": SCHEMA,
        "trained_on_all_500": True,
        "is_oof_model": False,
        "training_sample_count": 500,
        "epoch": 45,
        "model": model.state_dict(),
        "config_hash": config_hash,
        "manifest_sha256": manifest_sha,
        "weight_sha256": weight_sha,
        "class_names": {str(key): value for key, value in class_names.items()},
        "validation": None,
        "selection": "fixed_final_epoch_raw",
    }, final_path)
    atomic_json_dump({
        "schema": SCHEMA, "trained_on_all_500": True, "is_oof_model": False,
        "epochs": 45, "training_sample_count": 500,
        "final_weights": str(final_path), "final_weights_sha256": sha256_file(final_path),
        "note": "该模型使用全部训练样本，不是 OOF 模型且无独立验证。",
    }, output_dir / "training_summary.json")
    print(json.dumps({"completed": True, "final_weights": str(final_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
