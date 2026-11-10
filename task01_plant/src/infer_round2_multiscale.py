"""零训练多尺度 4rot OOF 选择，并生成固定 0.5/0.5 full500 测试融合。"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate import evaluate_oof
from .infer_round2_d4 import device_and_amp
from .infer_round2_fast import (
    TestDataset,
    atomic_csv,
    atomic_json,
    checkpoint_state,
    load_template,
    predict_one_fold,
    sha256_file,
)
from .infer_round2_final import load_full_checkpoint
from .train_round2_fast import (
    FORMAL_MODEL,
    NUM_CLASSES,
    ManifestDataset,
    build_transform,
    call_oof_writer,
    choose_precision,
    create_model,
    predict_rotation_tta,
    read_manifest,
    resolve_all_paths,
)

SCALES = (256, 288, 320)
# 候选及顺序均为预声明；最后的顺序只用于解决用户未指定的完全等价平局。
CANDIDATES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("single_256", (256,)),
    ("single_288", (288,)),
    ("single_320", (320,)),
    ("mean_256_288", (256, 288)),
    ("mean_288_320", (288, 320)),
    ("mean_256_288_320", (256, 288, 320)),
)
EXPECTED_TEST_COUNT = 378


def _loader(dataset: Any, batch_size: int, num_workers: int, cuda: bool) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cuda,
        persistent_workers=False,
    )


def _normalize_and_validate(probabilities: np.ndarray, rows: int, context: str) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (rows, NUM_CLASSES):
        raise ValueError(f"{context} 概率形状错误: {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{context} 概率含非有限值或负值")
    sums = probabilities.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError(f"{context} 概率和非正")
    probabilities = probabilities / sums
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError(f"{context} 概率和不为 1")
    return probabilities


def _mean_scales(per_scale: Mapping[int, np.ndarray], scales: Sequence[int], context: str) -> np.ndarray:
    if not scales or any(scale not in per_scale for scale in scales):
        raise ValueError(f"{context} 缺少预声明尺度概率")
    result = np.mean(np.stack([per_scale[scale] for scale in scales]), axis=0, dtype=np.float32)
    return _normalize_and_validate(result, result.shape[0], context)


def _probability_rows(
    ids: Sequence[str], probabilities: np.ndarray, class_names: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    probabilities = _normalize_and_validate(probabilities, len(ids), "输出")
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    probability_rows: list[dict[str, Any]] = []
    submission_rows: list[dict[str, str]] = []
    for row_index, sample_id in enumerate(ids):
        prediction = int(predictions[row_index])
        category = class_names[prediction]
        row: dict[str, Any] = {"ID": sample_id, "prediction": prediction, "Category": category}
        row.update({f"prob_{index}": float(probabilities[row_index, index]) for index in range(NUM_CLASSES)})
        if int(np.argmax(probabilities[row_index])) != prediction:
            raise RuntimeError("prediction 与概率 argmax 不一致")
        probability_rows.append(row)
        submission_rows.append({"ID": sample_id, "Category": category})
    counts = dict(Counter(row["Category"] for row in submission_rows))
    return probability_rows, submission_rows, {name: int(counts.get(name, 0)) for name in class_names}


def _write_probability_csv(path: Path, ids: Sequence[str], probabilities: np.ndarray,
                           class_names: Sequence[str]) -> dict[str, int]:
    rows, _submission, counts = _probability_rows(ids, probabilities, class_names)
    atomic_csv(path, ["ID", "prediction", "Category", *(f"prob_{i}" for i in range(NUM_CLASSES))], rows)
    return counts


def _validate_template_and_test(template: Path, test_dir: Path) -> tuple[list[str], list[str], list[Path]]:
    ids, headers = load_template(template)
    files = sorted((path for path in test_dir.glob("*.png") if path.is_file()), key=lambda path: path.name)
    by_name = {path.name: path for path in files}
    if len(ids) != EXPECTED_TEST_COUNT or len(set(ids)) != EXPECTED_TEST_COUNT:
        raise ValueError(f"模板必须严格包含 {EXPECTED_TEST_COUNT} 个唯一 ID")
    if len(files) != EXPECTED_TEST_COUNT or len(by_name) != EXPECTED_TEST_COUNT or set(ids) != set(by_name):
        raise ValueError(f"模板 ID 与 {EXPECTED_TEST_COUNT} 个唯一测试 PNG 不完全一致")
    return ids, headers, [by_name[sample_id] for sample_id in ids]


def _selection_key(item: dict[str, Any]) -> tuple[float, int, int, int]:
    # pooled accuracy 最大；平局先少尺度，再含 288，最后保持预声明顺序。
    return (
        float(item["pooled_accuracy"]),
        -len(item["scales"]),
        int(288 in item["scales"]),
        -int(item["declaration_index"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    full_checkpoint = args.full_checkpoint.resolve()
    manifest = args.manifest.resolve()
    data_root = args.data_root.resolve()
    test_dir = args.test_dir.resolve()
    template = args.template.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖或混写: {output_dir}")
    if output_dir == template.parent or template.is_relative_to(output_dir):
        raise ValueError("输出目录不能包含或覆盖原提交模板")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size 必须为正，num-workers 不能为负")

    template_sha_before = sha256_file(template)
    manifest_sha = sha256_file(manifest)
    rows, manifest_classes = read_manifest(manifest)
    if len(rows) != 500:
        raise ValueError(f"正式 manifest 必须为 500 行，实际 {len(rows)}")
    expected_checkpoint_classes = {str(index): manifest_classes[index] for index in range(NUM_CLASSES)}
    class_names = [manifest_classes[index] for index in range(NUM_CLASSES)]
    all_paths = resolve_all_paths(rows, manifest, data_root)
    device, amp_dtype = device_and_amp(args.device)
    precision = choose_precision(device)
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_records: list[dict[str, Any]] = []
    per_fold_scale: dict[int, dict[int, np.ndarray]] = {}
    fold_rows_by_fold: dict[int, list[dict[str, str]]] = {}

    # 阶段 1a：每个样本严格只由其 held-out fold 模型预测。
    for fold in range(5):
        fold_indices = [index for index, row in enumerate(rows) if int(row["fold"]) == fold]
        fold_rows = [rows[index] for index in fold_indices]
        fold_paths = [all_paths[index] for index in fold_indices]
        if len(fold_rows) != 100:
            raise ValueError(f"fold {fold} held-out 样本数应为 100，实际 {len(fold_rows)}")
        checkpoint_path = run_dir / f"fold_{fold}" / "final_weights.pt"
        state, checkpoint_classes = checkpoint_state(checkpoint_path, fold, manifest_sha)
        if checkpoint_classes != expected_checkpoint_classes:
            raise ValueError(f"fold {fold} checkpoint 类别映射与 manifest 不一致")
        model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
        model.load_state_dict(state, strict=True)
        model.to(device)
        scale_probabilities: dict[int, np.ndarray] = {}
        for scale in SCALES:
            dataset = ManifestDataset(fold_rows, fold_paths, build_transform(scale, training=False))
            loader = _loader(dataset, args.batch_size, args.num_workers, device.type == "cuda")
            probabilities, predictions = predict_rotation_tta(model, loader, device, precision)
            probabilities = _normalize_and_validate(probabilities, len(fold_rows), f"fold {fold} scale {scale} OOF")
            if not np.array_equal(predictions, probabilities.argmax(1)):
                raise RuntimeError(f"fold {fold} scale {scale} OOF argmax 不一致")
            scale_probabilities[scale] = probabilities
            print(f"fold={fold} held-out 4rot scale={scale} 完成", flush=True)
        per_fold_scale[fold] = scale_probabilities
        fold_rows_by_fold[fold] = fold_rows
        checkpoint_records.append({"fold": fold, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)})
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 阶段 1b：只完整评估六个预声明候选。
    candidate_results: list[dict[str, Any]] = []
    for declaration_index, (name, scales) in enumerate(CANDIDATES):
        candidate_oof_dir = output_dir / "oof_candidates" / name
        candidate_eval_dir = output_dir / "evaluations" / name
        oof_paths: list[Path] = []
        for fold in range(5):
            probabilities = _mean_scales(per_fold_scale[fold], scales, f"{name} fold {fold}")
            predictions = probabilities.argmax(axis=1).astype(np.int64)
            oof_path = candidate_oof_dir / f"fold_{fold}.csv"
            call_oof_writer(fold_rows_by_fold[fold], fold, probabilities, predictions,
                            manifest_classes, oof_path)
            oof_paths.append(oof_path)
        metrics = evaluate_oof(manifest, oof_paths, candidate_eval_dir)
        result = {
            "name": name,
            "scales": list(scales),
            "declaration_index": declaration_index,
            "total": int(metrics["total"]),
            "correct": int(metrics["correct"]),
            "pooled_accuracy": float(metrics["pooled_accuracy"]),
            "multiclass_log_loss": float(metrics["multiclass_log_loss"]),
            "multiclass_brier_score": float(metrics["multiclass_brier_score"]),
            "oof_dir": str(candidate_oof_dir),
            "evaluation_dir": str(candidate_eval_dir),
        }
        candidate_results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    selected = max(candidate_results, key=_selection_key)
    selection_payload = {
        "schema": "round2_multiscale_selection_v1",
        "development_selection_acknowledgement": (
            "The rule was selected on held-out OOF labels; this is development-set selection, "
            "not an untouched final evaluation. No test labels or leaderboard feedback were used."
        ),
        "strict_oof": True,
        "oof_prediction_rule": "each sample predicted only by its held-out fold final_weights checkpoint",
        "predeclared_scales": list(SCALES),
        "predeclared_candidates": [{"name": name, "scales": list(scales)} for name, scales in CANDIDATES],
        "selection_metric": "pooled_accuracy",
        "tie_break": "fewer scales, then contains 288, then declaration order",
        "candidates": candidate_results,
        "selected": {"name": selected["name"], "scales": selected["scales"],
                     "pooled_accuracy": selected["pooled_accuracy"]},
        "test_labels_or_leaderboard_used": False,
    }
    atomic_json(output_dir / "selection.json", selection_payload)

    # 阶段 1c：按选中规则推理测试集；五折和 full500 仍为固定 0.5/0.5。
    ids, submission_headers, ordered_test_paths = _validate_template_and_test(template, test_dir)
    selected_scales = tuple(int(scale) for scale in selected["scales"])
    fivefold_sum = np.zeros((len(ids), NUM_CLASSES), dtype=np.float32)
    for fold in range(5):
        checkpoint_path = run_dir / f"fold_{fold}" / "final_weights.pt"
        state, checkpoint_classes = checkpoint_state(checkpoint_path, fold, manifest_sha)
        if checkpoint_classes != expected_checkpoint_classes:
            raise ValueError(f"fold {fold} test checkpoint 类别映射与 manifest 不一致")
        model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
        model.load_state_dict(state, strict=True)
        model.to(device)
        scale_probabilities: dict[int, np.ndarray] = {}
        for scale in selected_scales:
            dataset = TestDataset(ordered_test_paths, build_transform(scale, training=False))
            loader = _loader(dataset, args.batch_size, args.num_workers, device.type == "cuda")
            predicted_ids, probabilities = predict_one_fold(model, loader, device, amp_dtype)
            if predicted_ids != ids:
                raise ValueError(f"fold {fold} scale {scale} 测试 ID 顺序与模板不一致")
            scale_probabilities[scale] = _normalize_and_validate(
                probabilities, len(ids), f"fold {fold} scale {scale} test"
            )
        fivefold_sum += _mean_scales(scale_probabilities, selected_scales, f"fold {fold} selected test")
        print(f"fold={fold} selected multiscale test 完成", flush=True)
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fivefold_probabilities = _normalize_and_validate(
        fivefold_sum / np.float32(5.0), len(ids), "fivefold test"
    )

    full_state, full_classes = load_full_checkpoint(full_checkpoint, manifest_sha)
    if full_classes != class_names:
        raise ValueError("full500 checkpoint 类别映射与 manifest 不一致")
    full_model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
    full_model.load_state_dict(full_state, strict=True)
    full_model.to(device)
    full_per_scale: dict[int, np.ndarray] = {}
    for scale in selected_scales:
        dataset = TestDataset(ordered_test_paths, build_transform(scale, training=False))
        loader = _loader(dataset, args.batch_size, args.num_workers, device.type == "cuda")
        predicted_ids, probabilities = predict_one_fold(full_model, loader, device, amp_dtype)
        if predicted_ids != ids:
            raise ValueError(f"full500 scale {scale} 测试 ID 顺序与模板不一致")
        full_per_scale[scale] = _normalize_and_validate(probabilities, len(ids), f"full500 scale {scale}")
    full_probabilities = _mean_scales(full_per_scale, selected_scales, "full500 selected test")
    del full_model, full_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    blend = _normalize_and_validate(
        np.float32(0.5) * fivefold_probabilities + np.float32(0.5) * full_probabilities,
        len(ids),
        "fixed final blend",
    )
    fivefold_counts = _write_probability_csv(
        output_dir / "fivefold_test_probabilities.csv", ids, fivefold_probabilities, class_names
    )
    full_counts = _write_probability_csv(
        output_dir / "full500_test_probabilities.csv", ids, full_probabilities, class_names
    )
    final_rows, submission_rows, final_counts = _probability_rows(ids, blend, class_names)
    atomic_csv(output_dir / "test_probabilities.csv",
               ["ID", "prediction", "Category", *(f"prob_{i}" for i in range(NUM_CLASSES))], final_rows)
    submission_path = output_dir / "submission_multiscale_final.csv"
    atomic_csv(submission_path, submission_headers, submission_rows)

    template_sha_after = sha256_file(template)
    if template_sha_after != template_sha_before:
        raise RuntimeError("原提交模板 hash 在执行期间发生变化")
    if len(submission_rows) != EXPECTED_TEST_COUNT or [row["ID"] for row in submission_rows] != ids:
        raise RuntimeError("最终 submission 数量或 ID 顺序错误")

    provenance = {
        "schema": "round2_multiscale_final_v1",
        "zero_training": True,
        "manifest": {"path": str(manifest), "sha256": manifest_sha, "rows": len(rows)},
        "template": {"path": str(template), "sha256_before": template_sha_before,
                     "sha256_after": template_sha_after, "unchanged": True,
                     "category_values_ignored": True},
        "test_dir": str(test_dir),
        "sample_count": len(ids),
        "class_names": class_names,
        "model": FORMAL_MODEL,
        "tta": "4rot equal float32-softmax probability mean",
        "selected_multiscale_rule": {"name": selected["name"], "scales": list(selected_scales),
                                     "selection_json": str(output_dir / "selection.json")},
        "fivefold": {"checkpoints": checkpoint_records, "equal_fold_probability_mean": True,
                     "weight_in_final_blend": 0.5, "prediction_counts": fivefold_counts},
        "full500": {"checkpoint": str(full_checkpoint), "sha256": sha256_file(full_checkpoint),
                    "trained_on_all_500": True, "is_oof_model": False,
                    "weight_in_final_blend": 0.5, "prediction_counts": full_counts},
        "blend_rule": "0.5 * fivefold_probability + 0.5 * full500_probability",
        "blend_weights": "precommitted; not tuned on test labels or leaderboard",
        "validation": {
            "expected_sample_count": EXPECTED_TEST_COUNT,
            "actual_sample_count": len(ids),
            "ids_unique": len(ids) == len(set(ids)),
            "ids_match_test_png_set": True,
            "id_order_matches_template": True,
            "probabilities_finite_nonnegative": True,
            "probability_sums_one": True,
            "prediction_equals_argmax": True,
            "categories_legal": True,
            "template_hash_unchanged": True,
        },
        "prediction_counts": final_counts,
    }
    atomic_json(output_dir / "provenance.json", provenance)
    # 文件落盘后记录最终可交付物 hash，另存校验清单，避免自引用 hash。
    atomic_json(output_dir / "artifact_hashes.json", {
        "template_sha256": template_sha_after,
        "selection_sha256": sha256_file(output_dir / "selection.json"),
        "fivefold_probabilities_sha256": sha256_file(output_dir / "fivefold_test_probabilities.csv"),
        "full500_probabilities_sha256": sha256_file(output_dir / "full500_test_probabilities.csv"),
        "final_probabilities_sha256": sha256_file(output_dir / "test_probabilities.csv"),
        "submission_sha256": sha256_file(submission_path),
        "provenance_sha256": sha256_file(output_dir / "provenance.json"),
    })
    summary = {
        "selected": selection_payload["selected"],
        "candidates": [{"name": item["name"], "accuracy": item["pooled_accuracy"]}
                       for item in candidate_results],
        "samples": len(ids),
        "submission": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "counts": final_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/round2_fast_resnet50"))
    parser.add_argument("--full-checkpoint", type=Path,
                        default=Path("runs/round2_fast_full500/final_weights.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/folds_v2.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--template", type=Path, default=Path("data/submission_fortask1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/round2_multiscale_final"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
