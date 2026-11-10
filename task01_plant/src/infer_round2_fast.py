"""使用 round2_fast 的五折最终权重生成测试集概率和提交文件。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .train_round2_fast import (
    FORMAL_WEIGHT_SHA256,
    FORMAL_MODEL,
    NUM_CLASSES,
    SCHEMA,
    build_transform,
    create_model,
)


class TestDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, paths: Sequence[Path], transform: Any) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有推理输出: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有推理输出: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_template(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "Category"]:
            raise ValueError(f"提交模板表头必须为 ID,Category，实际为 {reader.fieldnames}")
        ids = [str(row["ID"]).strip() for row in reader]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("提交模板 ID 为空或重复")
    return ids, ["ID", "Category"]


def checkpoint_state(path: Path, fold: int, expected_manifest_sha: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {
        "schema": SCHEMA,
        "fold": fold,
        "epoch": 45,
        "manifest_sha256": expected_manifest_sha,
        "weight_sha256": FORMAL_WEIGHT_SHA256,
    }
    for name, expected in required.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"fold {fold} checkpoint {name} 不匹配: {checkpoint.get(name)!r} != {expected!r}")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"fold {fold} checkpoint 缺少 model state")
    classes = checkpoint.get("class_names")
    if not isinstance(classes, dict):
        raise ValueError(f"fold {fold} checkpoint 缺少 class_names")
    return state, {str(key): str(value) for key, value in classes.items()}


@torch.inference_mode()
def predict_one_fold(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[list[str], np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    chunks: list[np.ndarray] = []
    for images, ids in loader:
        images = images.to(device, non_blocking=True)
        probability_sum = torch.zeros((images.shape[0], NUM_CLASSES), device=device, dtype=torch.float32)
        for rotation in range(4):
            rotated = torch.rot90(images, k=rotation, dims=(-2, -1))
            context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else __import__("contextlib").nullcontext()
            )
            with context:
                logits = model(rotated)
            probability_sum.add_(torch.softmax(logits.float(), dim=1), alpha=0.25)
        chunks.append(probability_sum.cpu().numpy())
        sample_ids.extend(str(value) for value in ids)
    probabilities = np.concatenate(chunks).astype(np.float32, copy=False)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=2e-6):
        raise ValueError("测试概率含非有限值或概率和不为 1")
    return sample_ids, probabilities


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/round2_fast_resnet50"))
    parser.add_argument("--manifest", type=Path, default=Path("data/folds_v2.csv"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--template", type=Path, default=Path("data/submission_fortask1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/round2_fast_resnet50/test_inference"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    manifest = args.manifest.resolve()
    test_dir = args.test_dir.resolve()
    template = args.template.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == template.parent or template.is_relative_to(output_dir):
        raise ValueError("推理输出目录不能覆盖提交模板")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空，拒绝混合推理结果: {output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size 必须为正，num-workers 不能为负")

    expected_manifest_sha = sha256_file(manifest)
    ids, headers = load_template(template)
    files = sorted((path for path in test_dir.glob("*.png") if path.is_file()), key=lambda path: path.name)
    by_name = {path.name: path for path in files}
    if len(by_name) != len(files) or set(ids) != set(by_name):
        raise ValueError("提交模板 ID 与测试目录 PNG 文件集合不完全一致")
    ordered_paths = [by_name[sample_id] for sample_id in ids]

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dataset = TestDataset(ordered_paths, build_transform(288, training=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    fold_sum: np.ndarray | None = None
    class_names: dict[str, str] | None = None
    checkpoint_sources: list[dict[str, Any]] = []
    for fold in range(5):
        checkpoint_path = run_dir / f"fold_{fold}" / "final_weights.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"缺少 fold {fold} 最终权重: {checkpoint_path}")
        state, fold_classes = checkpoint_state(checkpoint_path, fold, expected_manifest_sha)
        if class_names is None:
            class_names = fold_classes
        elif fold_classes != class_names:
            raise ValueError("五折 checkpoint 类别映射不一致")
        model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
        model.load_state_dict(state, strict=True)
        model.to(device)
        predicted_ids, probabilities = predict_one_fold(model, loader, device, amp_dtype)
        if predicted_ids != ids:
            raise ValueError(f"fold {fold} 推理 ID 顺序与模板不一致")
        fold_sum = probabilities if fold_sum is None else fold_sum + probabilities
        checkpoint_sources.append({
            "fold": fold,
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        })
        del model, state, probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert fold_sum is not None and class_names is not None
    probabilities = (fold_sum / np.float32(5.0)).astype(np.float32, copy=False)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    ordered_classes = [class_names[str(index)] for index in range(NUM_CLASSES)]

    probability_headers = [f"prob_{index}" for index in range(NUM_CLASSES)]
    probability_rows: list[dict[str, Any]] = []
    submission_rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(ids):
        predicted_class = ordered_classes[int(predictions[row_index])]
        probability_row: dict[str, Any] = {
            "ID": sample_id,
            "prediction": int(predictions[row_index]),
            "Category": predicted_class,
        }
        probability_row.update({name: float(probabilities[row_index, index]) for index, name in enumerate(probability_headers)})
        probability_rows.append(probability_row)
        submission_rows.append({"ID": sample_id, "Category": predicted_class})

    atomic_csv(output_dir / "test_probabilities.csv", ["ID", "prediction", "Category", *probability_headers], probability_rows)
    atomic_csv(output_dir / "submission_round2_fast.csv", headers, submission_rows)
    counts = {name: int(sum(category == name for category in (row["Category"] for row in submission_rows))) for name in ordered_classes}
    provenance = {
        "schema": "round2_fast_test_inference_v1",
        "run_dir": str(run_dir),
        "manifest": {"path": str(manifest), "sha256": expected_manifest_sha},
        "template": {"path": str(template), "sha256": sha256_file(template), "category_values_ignored": True},
        "test_dir": str(test_dir),
        "sample_count": len(ids),
        "model": FORMAL_MODEL,
        "image_size": 288,
        "tta_rotations": [0, 90, 180, 270],
        "within_fold_tta": "equal_probability_mean",
        "across_fold_ensemble": "five_fold_equal_probability_mean",
        "class_names": ordered_classes,
        "prediction_counts": counts,
        "checkpoints": checkpoint_sources,
    }
    atomic_json(output_dir / "provenance.json", provenance)
    print(json.dumps({"samples": len(ids), "counts": counts, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
