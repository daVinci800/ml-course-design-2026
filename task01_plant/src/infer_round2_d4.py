"""对 round2_fast 最终权重执行严格 D4 OOF 或测试集推理。"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate import evaluate_oof
from .infer_round2_fast import (
    TestDataset,
    atomic_csv,
    atomic_json,
    checkpoint_state,
    load_template,
    sha256_file,
)
from .train_round2_fast import (
    FORMAL_MODEL,
    NUM_CLASSES,
    ManifestDataset,
    build_transform,
    call_oof_writer,
    create_model,
    read_manifest,
    resolve_all_paths,
)


@torch.inference_mode()
def predict_d4(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[list[str] | None, np.ndarray, np.ndarray]:
    """八视图逐一以 float32 softmax，再作严格等权概率平均。"""
    model.eval()
    probabilities = np.zeros((len(loader.dataset), NUM_CLASSES), dtype=np.float32)
    seen = np.zeros(len(loader.dataset), dtype=np.bool_)
    ordered_ids: list[str] = []
    indexed = None
    offset = 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        metadata = batch[2] if len(batch) == 3 else batch[1]
        probability_sum = torch.zeros((images.shape[0], NUM_CLASSES), device=device, dtype=torch.float32)
        for reflected in (False, True):
            base = torch.flip(images, dims=(-1,)) if reflected else images
            for rotation in range(4):
                view = torch.rot90(base, k=rotation, dims=(-2, -1))
                autocast = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_dtype is not None
                    else contextlib.nullcontext()
                )
                with autocast:
                    logits = model(view)
                probability_sum.add_(torch.softmax(logits.float(), dim=1), alpha=0.125)
        batch_probabilities = probability_sum.cpu().numpy()
        if torch.is_tensor(metadata):
            indices = metadata.numpy()
            if indexed is False or seen[indices].any():
                raise RuntimeError("D4 DataLoader 索引重复或元数据类型不一致")
            indexed = True
            probabilities[indices] = batch_probabilities
            seen[indices] = True
        else:
            ids = [str(value) for value in metadata]
            if indexed is True:
                raise RuntimeError("D4 DataLoader 元数据类型不一致")
            indexed = False
            end = offset + len(ids)
            probabilities[offset:end] = batch_probabilities
            seen[offset:end] = True
            ordered_ids.extend(ids)
            offset = end
    if not seen.all():
        raise RuntimeError("D4 DataLoader 未完整覆盖数据集")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=2e-6):
        raise RuntimeError("D4 概率含非有限值或概率和不为 1")
    return (None if indexed else ordered_ids), probabilities, probabilities.argmax(1).astype(np.int64)


def device_and_amp(requested: str) -> tuple[torch.device, torch.dtype | None]:
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available() else
        ("cpu" if requested == "auto" else requested)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 CUDA 不可用")
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, amp_dtype


def ensure_empty(path: Path, description: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{description} 已存在且非空，拒绝覆盖或混写: {path}")


def run_oof(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_dir = args.evaluation_dir.resolve()
    ensure_empty(output_dir, "D4 OOF 输出目录")
    ensure_empty(evaluation_dir, "D4 评估输出目录")
    rows, class_names = read_manifest(manifest)
    all_paths = resolve_all_paths(rows, manifest, args.data_root.resolve())
    manifest_sha = sha256_file(manifest)
    device, amp_dtype = device_and_amp(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        indices = [index for index, row in enumerate(rows) if int(row["fold"]) == fold]
        fold_rows = [rows[index] for index in indices]
        fold_paths = [all_paths[index] for index in indices]
        dataset = ManifestDataset(fold_rows, fold_paths, build_transform(288, training=False))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=False)
        checkpoint_path = run_dir / f"fold_{fold}" / "final_weights.pt"
        state, checkpoint_classes = checkpoint_state(checkpoint_path, fold, manifest_sha)
        if checkpoint_classes != {str(key): value for key, value in class_names.items()}:
            raise ValueError(f"fold {fold} checkpoint 类别映射与 manifest 不一致")
        model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
        model.load_state_dict(state, strict=True)
        model.to(device)
        ids, probabilities, predictions = predict_d4(model, loader, device, amp_dtype)
        if ids is not None:
            raise RuntimeError("OOF 推理应返回索引而非 ID")
        call_oof_writer(fold_rows, fold, probabilities, predictions, class_names,
                        output_dir / f"fold_{fold}.csv")
        print(f"fold={fold} D4 OOF 完成 samples={len(fold_rows)}", flush=True)
        del model, state, probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = evaluate_oof(manifest, [output_dir / f"fold_{fold}.csv" for fold in range(5)], evaluation_dir)
    summary = {
        "samples": metrics["total"],
        "correct": metrics["correct"],
        "pooled_accuracy": metrics["pooled_accuracy"],
        "output_dir": str(output_dir),
        "evaluation_dir": str(evaluation_dir),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    manifest = args.manifest.resolve()
    template = args.template.resolve()
    output_dir = args.output_dir.resolve()
    ensure_empty(output_dir, "D4 测试推理输出目录")
    manifest_sha = sha256_file(manifest)
    ids, headers = load_template(template)
    files = sorted((path for path in args.test_dir.resolve().glob("*.png") if path.is_file()), key=lambda p: p.name)
    by_name = {path.name: path for path in files}
    if len(files) != len(by_name) or set(ids) != set(by_name):
        raise ValueError("模板 ID 与测试 PNG 集合不完全一致")
    dataset = TestDataset([by_name[sample_id] for sample_id in ids], build_transform(288, training=False))
    device, amp_dtype = device_and_amp(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=False)
    fold_sum: np.ndarray | None = None
    class_names: dict[str, str] | None = None
    checkpoints: list[dict[str, Any]] = []
    for fold in range(5):
        checkpoint_path = run_dir / f"fold_{fold}" / "final_weights.pt"
        state, fold_classes = checkpoint_state(checkpoint_path, fold, manifest_sha)
        if class_names is None:
            class_names = fold_classes
        elif fold_classes != class_names:
            raise ValueError("五折 checkpoint 类别映射不一致")
        model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
        model.load_state_dict(state, strict=True)
        model.to(device)
        predicted_ids, probabilities, _ = predict_d4(model, loader, device, amp_dtype)
        if predicted_ids != ids:
            raise ValueError(f"fold {fold} 测试 ID 顺序错误")
        fold_sum = probabilities if fold_sum is None else fold_sum + probabilities
        checkpoints.append({"fold": fold, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)})
        print(f"fold={fold} D4 test 完成 samples={len(ids)}", flush=True)
        del model, state, probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert fold_sum is not None and class_names is not None
    probabilities = (fold_sum / np.float32(5.0)).astype(np.float32, copy=False)
    probabilities /= probabilities.sum(1, keepdims=True)
    predictions = probabilities.argmax(1)
    ordered_classes = [class_names[str(index)] for index in range(NUM_CLASSES)]
    probability_headers = [f"prob_{index}" for index in range(NUM_CLASSES)]
    probability_rows = []
    submission_rows = []
    for row_index, sample_id in enumerate(ids):
        category = ordered_classes[int(predictions[row_index])]
        row: dict[str, Any] = {"ID": sample_id, "prediction": int(predictions[row_index]), "Category": category}
        row.update({name: float(probabilities[row_index, index]) for index, name in enumerate(probability_headers)})
        probability_rows.append(row)
        submission_rows.append({"ID": sample_id, "Category": category})
    atomic_csv(output_dir / "test_probabilities.csv", ["ID", "prediction", "Category", *probability_headers], probability_rows)
    atomic_csv(output_dir / "submission_round2_fast_d4.csv", headers, submission_rows)
    counts = {name: sum(row["Category"] == name for row in submission_rows) for name in ordered_classes}
    atomic_json(output_dir / "provenance.json", {
        "schema": "round2_fast_test_d4_v1",
        "run_dir": str(run_dir),
        "manifest": {"path": str(manifest), "sha256": manifest_sha},
        "template": {"path": str(template), "sha256": sha256_file(template), "category_values_ignored": True},
        "test_dir": str(args.test_dir.resolve()),
        "sample_count": len(ids),
        "model": FORMAL_MODEL,
        "image_size": 288,
        "tta_views": ["rot0", "rot90", "rot180", "rot270", "hflip_rot0", "hflip_rot90", "hflip_rot180", "hflip_rot270"],
        "within_model_tta": "eight_view_equal_float32_softmax_probability_mean",
        "across_fold_ensemble": "five_fold_equal_probability_mean",
        "class_names": ordered_classes,
        "prediction_counts": counts,
        "checkpoints": checkpoints,
    })
    summary = {"samples": len(ids), "counts": counts, "output_dir": str(output_dir)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("oof", "test"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/round2_fast_resnet50"))
    parser.add_argument("--manifest", type=Path, default=Path("data/folds_v2.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--template", type=Path, default=Path("data/submission_fortask1.csv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-dir", type=Path, default=Path("runs/round2_fast_resnet50/evaluation_d4"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size 必须为正，num-workers 不能为负")
    if args.output_dir is None:
        args.output_dir = Path("runs/round2_fast_resnet50/oof_d4" if args.mode == "oof" else
                               "runs/round2_fast_resnet50/test_inference_d4")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_oof(args) if args.mode == "oof" else run_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
