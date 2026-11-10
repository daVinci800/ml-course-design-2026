"""推理 full500，并按预先固定的 0.5/0.5 规则与五折 ensemble 融合。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .infer_round2_d4 import device_and_amp, predict_d4
from .infer_round2_fast import TestDataset, atomic_csv, atomic_json, load_template, predict_one_fold, sha256_file
from .train_round2_fast import FORMAL_MODEL, FORMAL_WEIGHT_SHA256, NUM_CLASSES, SCHEMA, build_transform, create_model
from .train_round2_full import SCHEMA as FULL_SCHEMA


def read_probability_csv(path: Path, expected_ids: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        expected_columns = [f"prob_{index}" for index in range(NUM_CLASSES)]
        required = {"ID", "prediction", "Category", *expected_columns}
        if not required.issubset(headers):
            raise ValueError(f"五折概率 CSV 缺列: {sorted(required - set(headers))}")
        rows = list(reader)
    ids = [str(row["ID"]).strip() for row in rows]
    if ids != list(expected_ids) or len(ids) != len(set(ids)):
        raise ValueError("五折概率 CSV 的 ID 顺序、数量或唯一性错误")
    probabilities = np.asarray([[float(row[column]) for column in expected_columns] for row in rows], dtype=np.float32)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=2e-6):
        raise ValueError("五折概率 CSV 含非法概率")
    categories = [str(row["Category"]) for row in rows]
    return probabilities, categories


def load_full_checkpoint(path: Path, manifest_sha: str) -> tuple[dict[str, torch.Tensor], list[str]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    checks = {
        "schema": (checkpoint.get("schema"), FULL_SCHEMA),
        "trained_on_all_500": (checkpoint.get("trained_on_all_500"), True),
        "is_oof_model": (checkpoint.get("is_oof_model"), False),
        "training_sample_count": (checkpoint.get("training_sample_count"), 500),
        "epoch": (checkpoint.get("epoch"), 45),
        "manifest_sha256": (checkpoint.get("manifest_sha256"), manifest_sha),
        "weight_sha256": (checkpoint.get("weight_sha256"), FORMAL_WEIGHT_SHA256),
    }
    mismatches = [f"{name}: {actual!r} != {expected!r}"
                  for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError("full500 checkpoint 元数据错误：\n" + "\n".join(mismatches))
    state = checkpoint.get("model")
    classes = checkpoint.get("class_names")
    if not isinstance(state, dict) or not isinstance(classes, dict):
        raise ValueError("full500 checkpoint 缺 model/class_names")
    class_names = [str(classes[str(index)]) for index in range(NUM_CLASSES)]
    return state, class_names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/folds_v2.csv"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--template", type=Path, default=Path("data/submission_fortask1.csv"))
    parser.add_argument("--full-checkpoint", type=Path, default=Path("runs/round2_fast_full500/final_weights.pt"))
    parser.add_argument("--fivefold-probabilities", type=Path,
                        default=Path("runs/round2_fast_resnet50/test_inference/test_probabilities.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/round2_fast_final_blend"))
    parser.add_argument("--tta", choices=("4rot", "d4"), default="4rot")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size 必须为正，num-workers 不能为负")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"最终融合目录已存在且非空，拒绝覆盖: {output_dir}")
    manifest = args.manifest.resolve()
    template = args.template.resolve()
    template_sha_before = sha256_file(template)
    ids, headers = load_template(template)
    if len(ids) != 378:
        raise ValueError(f"测试模板必须为 378 个 ID，实际 {len(ids)}")
    files = sorted((path for path in args.test_dir.resolve().glob("*.png") if path.is_file()), key=lambda p: p.name)
    by_name = {path.name: path for path in files}
    if len(files) != 378 or len(by_name) != 378 or set(ids) != set(by_name):
        raise ValueError("378 个模板 ID 与测试 PNG 集合不完全一致或不唯一")
    fivefold_path = args.fivefold_probabilities.resolve()
    fivefold_probabilities, fivefold_categories = read_probability_csv(fivefold_path, ids)
    manifest_sha = sha256_file(manifest)
    checkpoint_path = args.full_checkpoint.resolve()
    state, class_names = load_full_checkpoint(checkpoint_path, manifest_sha)
    if any(category not in set(class_names) for category in fivefold_categories):
        raise ValueError("五折概率文件含非法类别名")
    device, amp_dtype = device_and_amp(args.device)
    dataset = TestDataset([by_name[sample_id] for sample_id in ids], build_transform(288, training=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=False)
    model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
    model.load_state_dict(state, strict=True)
    model.to(device)
    if args.tta == "d4":
        predicted_ids, full_probabilities, _ = predict_d4(model, loader, device, amp_dtype)
        tta_views = ["rot0", "rot90", "rot180", "rot270", "hflip_rot0", "hflip_rot90", "hflip_rot180", "hflip_rot270"]
    else:
        predicted_ids, full_probabilities = predict_one_fold(model, loader, device, amp_dtype)
        tta_views = ["rot0", "rot90", "rot180", "rot270"]
    if predicted_ids != ids:
        raise ValueError("full500 推理 ID 顺序与模板不一致")
    blend = (np.float32(0.5) * fivefold_probabilities + np.float32(0.5) * full_probabilities).astype(np.float32)
    blend /= blend.sum(1, keepdims=True)
    predictions = blend.argmax(1)
    if not np.isfinite(blend).all() or not np.allclose(blend.sum(1), 1.0, atol=2e-6):
        raise ValueError("融合概率非法")
    probability_headers = [f"prob_{index}" for index in range(NUM_CLASSES)]
    probability_rows: list[dict[str, Any]] = []
    submission_rows: list[dict[str, str]] = []
    for row_index, sample_id in enumerate(ids):
        prediction = int(predictions[row_index])
        category = class_names[prediction]
        if prediction != int(np.argmax(blend[row_index])) or category not in set(class_names):
            raise RuntimeError("argmax 或类别合法性检查失败")
        row: dict[str, Any] = {"ID": sample_id, "prediction": prediction, "Category": category}
        row.update({name: float(blend[row_index, index]) for index, name in enumerate(probability_headers)})
        probability_rows.append(row)
        submission_rows.append({"ID": sample_id, "Category": category})
    atomic_csv(output_dir / "test_probabilities.csv", ["ID", "prediction", "Category", *probability_headers], probability_rows)
    atomic_csv(output_dir / "submission_final_blend.csv", headers, submission_rows)
    template_sha_after = sha256_file(template)
    if template_sha_after != template_sha_before:
        raise RuntimeError("原提交模板 hash 在推理期间发生变化")
    counts = {name: sum(row["Category"] == name for row in submission_rows) for name in class_names}
    atomic_json(output_dir / "provenance.json", {
        "schema": "round2_fast_final_blend_v1",
        "sample_count": 378,
        "id_order_source": str(template),
        "template_sha256_before": template_sha_before,
        "template_sha256_after": template_sha_after,
        "template_unchanged": True,
        "manifest": {"path": str(manifest), "sha256": manifest_sha},
        "class_names": class_names,
        "model": FORMAL_MODEL,
        "image_size": 288,
        "tta_rule_selected_by_oof": args.tta,
        "tta_views": tta_views,
        "full500": {
            "checkpoint": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
            "trained_on_all_500": True, "is_oof_model": False, "weight": 0.5,
        },
        "fivefold_ensemble": {
            "probabilities": str(fivefold_path), "sha256": sha256_file(fivefold_path),
            "ensemble_as_one_component": True, "weight": 0.5,
        },
        "blend_rule": "0.5 * fivefold_ensemble_probability + 0.5 * full500_probability",
        "weight_selection": "precommitted; no test labels or leaderboard used",
        "validation": {
            "sample_count": 378, "ids_unique": len(ids) == len(set(ids)),
            "id_order_matches_template": True, "probabilities_finite": True,
            "probability_sums_one": True, "prediction_equals_argmax": True,
            "categories_legal": True, "template_hash_unchanged": True,
        },
        "prediction_counts": counts,
    })
    print(json.dumps({"samples": 378, "counts": counts, "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
