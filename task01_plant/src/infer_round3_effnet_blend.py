"""评估 EfficientNet-B3 OOF，并与 ResNet320 做预声明 50:50 固定融合。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate import evaluate_oof
from .infer_round2_fast import TestDataset, load_template, predict_one_fold
from .train_round3_effnet import (
    FORMAL_MODEL,
    FORMAL_WEIGHT_SHA256,
    NUM_CLASSES,
    SCHEMA,
    build_transform,
    call_oof_writer,
    canonical_hash,
    create_model,
    read_manifest,
    sha256_file,
)

EXPECTED_TEST_COUNT = 378
RESNET320_EXPECTED_ACCURACY = 0.908
CANDIDATE_ORDER = ("resnet320", "efficientnet_b3", "fixed_50_50")


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_probabilities(values: Any, rows: int, context: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float32)
    if probabilities.shape != (rows, NUM_CLASSES):
        raise ValueError(f"{context} 概率形状错误: {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{context} 概率包含非有限值或负值")
    sums = probabilities.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError(f"{context} 概率和非正")
    probabilities = probabilities / sums
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError(f"{context} 概率和不为 1")
    return probabilities


def read_standard_oof(path: Path, class_names: Sequence[str]) -> dict[str, dict[str, Any]]:
    probability_columns = [f"prob::{quote(name, safe='')}" for name in class_names]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "fold", "label", *probability_columns}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"OOF {path} 缺少列: {sorted(missing)}")
        result: dict[str, dict[str, Any]] = {}
        for line_number, row in enumerate(reader, start=2):
            sample_id = str(row["sample_id"]).strip()
            if not sample_id or sample_id in result:
                raise ValueError(f"OOF {path}:{line_number} sample_id 为空或重复")
            try:
                probabilities = np.asarray([float(row[name]) for name in probability_columns], dtype=np.float32)
                fold = int(row["fold"])
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"OOF {path}:{line_number} 数值非法") from exc
            probabilities = normalize_probabilities(probabilities[None, :], 1, f"{path}:{line_number}")[0]
            result[sample_id] = {"fold": fold, "label": label, "probabilities": probabilities}
    return result


def read_test_probabilities(path: Path, expected_ids: Sequence[str], context: str) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        probability_columns = [f"prob_{index}" for index in range(NUM_CLASSES)]
        required = {"ID", *probability_columns}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{context} 缺少列: {sorted(missing)}")
        ids: list[str] = []
        values: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            ids.append(str(row["ID"]).strip())
            try:
                values.append([float(row[name]) for name in probability_columns])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}:{line_number} 概率非法") from exc
    if ids != list(expected_ids):
        raise ValueError(f"{context} ID 或顺序与提交模板不一致")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} ID 重复")
    return normalize_probabilities(values, len(expected_ids), context)


def probability_rows(
    ids: Sequence[str], probabilities: np.ndarray, class_names: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    probabilities = normalize_probabilities(probabilities, len(ids), "最终输出")
    predictions = probabilities.argmax(axis=1)
    probability_output: list[dict[str, Any]] = []
    submission: list[dict[str, str]] = []
    for row_index, sample_id in enumerate(ids):
        prediction = int(predictions[row_index])
        category = class_names[prediction]
        row: dict[str, Any] = {"ID": sample_id, "prediction": prediction, "Category": category}
        row.update({f"prob_{index}": float(probabilities[row_index, index]) for index in range(NUM_CLASSES)})
        probability_output.append(row)
        submission.append({"ID": sample_id, "Category": category})
    counts = Counter(row["Category"] for row in submission)
    return probability_output, submission, {name: int(counts.get(name, 0)) for name in class_names}


def validate_template(template: Path, test_dir: Path) -> tuple[list[str], list[str], list[Path]]:
    ids, headers = load_template(template)
    files = [path for path in test_dir.glob("*.png") if path.is_file()]
    by_name = {path.name: path for path in files}
    if len(ids) != EXPECTED_TEST_COUNT or len(set(ids)) != EXPECTED_TEST_COUNT:
        raise ValueError(f"提交模板必须严格包含 {EXPECTED_TEST_COUNT} 个唯一 ID")
    if len(files) != EXPECTED_TEST_COUNT or len(by_name) != EXPECTED_TEST_COUNT or set(ids) != set(by_name):
        raise ValueError(f"提交模板 ID 与 {EXPECTED_TEST_COUNT} 个唯一测试 PNG 不完全一致")
    return ids, headers, [by_name[sample_id] for sample_id in ids]


def load_effnet_checkpoint(
    path: Path,
    fold: int,
    manifest_sha: str,
    expected_config_hash: str,
    expected_classes: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    checks = {
        "schema": SCHEMA,
        "fold": fold,
        "epoch": 30,
        "manifest_sha256": manifest_sha,
        "weight_sha256": FORMAL_WEIGHT_SHA256,
        "config_hash": expected_config_hash,
    }
    for name, expected in checks.items():
        if checkpoint.get(name) != expected:
            raise ValueError(
                f"EffNet fold {fold} checkpoint {name} 不匹配: "
                f"{checkpoint.get(name)!r} != {expected!r}"
            )
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes != dict(expected_classes):
        raise ValueError(f"EffNet fold {fold} checkpoint 类别映射不一致")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"EffNet fold {fold} checkpoint 缺少 model state")
    return state


def validate_run_config(
    run_dir: Path, manifest_sha: str, expected_args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    path = run_dir / "run_config.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("EffNet run_config 缺少 config")
    expected = {
        "schema": SCHEMA,
        "model": FORMAL_MODEL,
        "manifest_sha256": manifest_sha,
        "weight_sha256": FORMAL_WEIGHT_SHA256,
        "image_size": 288,
        "batch_size": 16,
        "accumulation_steps": 2,
        "epochs": 30,
        "head_only_epochs": 2,
        "precision": "bf16",
        "ema": False,
        "selection": "fixed_final_epoch_raw",
    }
    mismatches = [
        f"{name}: {config.get(name)!r} != {value!r}"
        for name, value in expected.items()
        if config.get(name) != value
    ]
    if mismatches:
        raise ValueError("EffNet 正式配置不匹配:\n" + "\n".join(mismatches))
    config_hash = canonical_hash(config)
    if payload.get("config_hash") != config_hash:
        raise ValueError("EffNet run_config config_hash 与配置内容不一致")
    del expected_args
    return config_hash, payload


def metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "total": int(metrics["total"]),
        "correct": int(metrics["correct"]),
        "pooled_accuracy": float(metrics["pooled_accuracy"]),
        "macro_f1": float(metrics["macro"]["f1"]),
        "multiclass_log_loss": float(metrics["multiclass_log_loss"]),
        "multiclass_brier_score": float(metrics["multiclass_brier_score"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    manifest = args.manifest.resolve()
    resnet_oof_dir = args.resnet_oof_dir.resolve()
    resnet_evaluation = args.resnet_evaluation.resolve()
    resnet_test_path = args.resnet_test_probabilities.resolve()
    data_root = args.data_root.resolve()
    test_dir = args.test_dir.resolve()
    template = args.template.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"固定融合输出目录已存在，拒绝覆盖或混写: {output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size 必须为正，num-workers 不能为负")

    template_sha_before = sha256_file(template)
    manifest_sha = sha256_file(manifest)
    rows, class_mapping = read_manifest(manifest)
    if len(rows) != 500:
        raise ValueError(f"正式 manifest 必须为 500 行，实际 {len(rows)}")
    class_names = [class_mapping[index] for index in range(NUM_CLASSES)]
    expected_classes = {str(index): class_names[index] for index in range(NUM_CLASSES)}
    config_hash, run_config = validate_run_config(run_dir, manifest_sha, args)

    eff_oof_paths = [run_dir / "oof" / f"fold_{fold}.csv" for fold in range(5)]
    if any(not path.is_file() for path in eff_oof_paths):
        missing = [str(path) for path in eff_oof_paths if not path.is_file()]
        raise FileNotFoundError(f"缺少 EffNet OOF: {missing}")
    eff_evaluation_dir = run_dir / "evaluation"
    if eff_evaluation_dir.exists():
        raise FileExistsError(f"EffNet evaluation 已存在，拒绝覆盖: {eff_evaluation_dir}")
    eff_metrics = evaluate_oof(manifest, eff_oof_paths, eff_evaluation_dir)

    with (resnet_evaluation / "metrics.json").open("r", encoding="utf-8") as handle:
        resnet_metrics = json.load(handle)
    if int(resnet_metrics.get("total", -1)) != 500 or float(resnet_metrics.get("pooled_accuracy", -1)) != RESNET320_EXPECTED_ACCURACY:
        raise ValueError("ResNet single_320 指标不是预声明的 454/500 = 90.8%")
    if (resnet_metrics.get("provenance") or {}).get("manifest", {}).get("sha256") != manifest_sha:
        raise ValueError("ResNet single_320 OOF manifest hash 不匹配")

    output_dir.mkdir(parents=True, exist_ok=False)
    blend_oof_dir = output_dir / "oof_50_50"
    blend_oof_paths: list[Path] = []
    for fold in range(5):
        expected_rows = [row for row in rows if int(row["fold"]) == fold]
        resnet_path = resnet_oof_dir / f"fold_{fold}.csv"
        effnet_path = eff_oof_paths[fold]
        resnet_by_id = read_standard_oof(resnet_path, class_names)
        effnet_by_id = read_standard_oof(effnet_path, class_names)
        expected_ids = [row["sample_id"] for row in expected_rows]
        expected_set = set(expected_ids)
        if set(resnet_by_id) != expected_set or set(effnet_by_id) != expected_set:
            raise ValueError(f"fold {fold} 的 ResNet/EffNet OOF 未严格覆盖 held-out 样本")
        blend_values: list[np.ndarray] = []
        for row in expected_rows:
            sample_id = row["sample_id"]
            target = int(row["class_idx"])
            for model_name, record in (("ResNet", resnet_by_id[sample_id]), ("EffNet", effnet_by_id[sample_id])):
                if int(record["fold"]) != fold or int(record["label"]) != target:
                    raise ValueError(f"{model_name} OOF {sample_id} 的 fold/真值与 manifest 不一致")
            blend_values.append(
                np.float32(0.5) * resnet_by_id[sample_id]["probabilities"]
                + np.float32(0.5) * effnet_by_id[sample_id]["probabilities"]
            )
        probabilities = normalize_probabilities(blend_values, len(expected_rows), f"fold {fold} 50:50 OOF")
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        output_path = blend_oof_dir / f"fold_{fold}.csv"
        call_oof_writer(expected_rows, fold, probabilities, predictions, class_mapping, output_path)
        blend_oof_paths.append(output_path)
    blend_metrics = evaluate_oof(manifest, blend_oof_paths, output_dir / "evaluation_50_50")

    candidates = [
        {"name": "resnet320", **metric_summary(resnet_metrics)},
        {"name": "efficientnet_b3", **metric_summary(eff_metrics)},
        {"name": "fixed_50_50", **metric_summary(blend_metrics)},
    ]
    maximum_accuracy = max(float(candidate["pooled_accuracy"]) for candidate in candidates)
    blend_candidate = candidates[2]
    if float(blend_candidate["pooled_accuracy"]) == maximum_accuracy:
        selected_name = "fixed_50_50"
        selection_reason = "50:50 达到最高 OOF accuracy（并列也保留预声明融合）"
    else:
        singles = candidates[:2]
        selected_name = max(singles, key=lambda item: (float(item["pooled_accuracy"]), -CANDIDATE_ORDER.index(str(item["name"]))))["name"]
        selection_reason = "50:50 不是最高，按预声明选择 OOF accuracy 最高的单模型"

    ids, submission_headers, ordered_test_paths = validate_template(template, test_dir)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("EffNet 正式测试推理要求可用的 CUDA BF16")
    amp_dtype = torch.bfloat16
    dataset = TestDataset(ordered_test_paths, build_transform(288, training=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    fold_sum = np.zeros((len(ids), NUM_CLASSES), dtype=np.float32)
    checkpoint_sources: list[dict[str, Any]] = []
    for fold in range(5):
        checkpoint_path = run_dir / f"fold_{fold}" / "final_weights.pt"
        state = load_effnet_checkpoint(
            checkpoint_path, fold, manifest_sha, config_hash, expected_classes
        )
        model = create_model(FORMAL_MODEL, pretrained_path=None, num_classes=NUM_CLASSES)
        model.load_state_dict(state, strict=True)
        model.to(device)
        predicted_ids, probabilities = predict_one_fold(model, loader, device, amp_dtype)
        if predicted_ids != ids:
            raise ValueError(f"EffNet fold {fold} 测试 ID 顺序与模板不一致")
        fold_sum += normalize_probabilities(probabilities, len(ids), f"EffNet fold {fold} test")
        checkpoint_sources.append(
            {"fold": fold, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
        )
        print(f"EffNet fold={fold} 288px 4rot test 完成", flush=True)
        del model, state, probabilities
        torch.cuda.empty_cache()
    effnet_test = normalize_probabilities(fold_sum / np.float32(5.0), len(ids), "EffNet fivefold test")

    effnet_inference_dir = run_dir / "test_inference"
    if effnet_inference_dir.exists():
        raise FileExistsError(f"EffNet test_inference 已存在，拒绝覆盖: {effnet_inference_dir}")
    effnet_rows, _effnet_submission, effnet_counts = probability_rows(ids, effnet_test, class_names)
    effnet_probability_path = effnet_inference_dir / "test_probabilities.csv"
    atomic_csv(
        effnet_probability_path,
        ["ID", "prediction", "Category", *(f"prob_{index}" for index in range(NUM_CLASSES))],
        effnet_rows,
    )
    atomic_json(
        effnet_inference_dir / "provenance.json",
        {
            "schema": "round3_effnet_test_inference_v1",
            "model": FORMAL_MODEL,
            "image_size": 288,
            "tta": "4rot equal float32-softmax probability mean",
            "across_folds": "fivefold equal probability mean",
            "manifest": {"path": str(manifest), "sha256": manifest_sha},
            "template": {"path": str(template), "sha256": template_sha_before},
            "sample_count": len(ids),
            "class_names": class_names,
            "checkpoints": checkpoint_sources,
            "prediction_counts": effnet_counts,
        },
    )

    resnet_test = read_test_probabilities(resnet_test_path, ids, "ResNet320 fivefold test")
    fixed_test_blend = normalize_probabilities(
        np.float32(0.5) * resnet_test + np.float32(0.5) * effnet_test,
        len(ids),
        "fixed 50:50 test",
    )
    selected_probabilities = {
        "resnet320": resnet_test,
        "efficientnet_b3": effnet_test,
        "fixed_50_50": fixed_test_blend,
    }[str(selected_name)]
    probability_output, submission_rows, selected_counts = probability_rows(
        ids, selected_probabilities, class_names
    )
    probabilities_path = output_dir / "probabilities.csv"
    submission_path = output_dir / "submission.csv"
    atomic_csv(
        probabilities_path,
        ["ID", "prediction", "Category", *(f"prob_{index}" for index in range(NUM_CLASSES))],
        probability_output,
    )
    atomic_csv(submission_path, submission_headers, submission_rows)

    template_sha_after = sha256_file(template)
    if template_sha_after != template_sha_before:
        raise RuntimeError("提交模板在运行期间发生变化")
    if len(submission_rows) != EXPECTED_TEST_COUNT or [row["ID"] for row in submission_rows] != ids:
        raise RuntimeError("最终 submission 行数或 ID 顺序错误")
    if any(row["Category"] not in class_names for row in submission_rows):
        raise RuntimeError("最终 submission 含非法类别")

    selection_payload = {
        "schema": "round3_effnet_resnet_selection_v1",
        "predeclared_candidates": list(CANDIDATE_ORDER),
        "selection_metric": "pooled_accuracy",
        "fixed_blend_weights": {"resnet320": 0.5, "efficientnet_b3": 0.5},
        "candidates": candidates,
        "selected": str(selected_name),
        "selection_reason": selection_reason,
        "test_labels_used": False,
        "leaderboard_used": False,
        "external_old_resnet_final_blend_score_92_06_excluded": True,
    }
    atomic_json(output_dir / "selection.json", selection_payload)
    provenance = {
        "schema": "round3_effnet_resnet_blend_v1",
        "elapsed_seconds": time.perf_counter() - started,
        "manifest": {"path": str(manifest), "sha256": manifest_sha, "rows": len(rows)},
        "template": {
            "path": str(template),
            "sha256_before": template_sha_before,
            "sha256_after": template_sha_after,
            "unchanged": True,
            "category_values_ignored": True,
        },
        "test_dir": str(test_dir),
        "sample_count": len(ids),
        "class_names": class_names,
        "oof": {
            "strict": True,
            "resnet320": {
                "oof_dir": str(resnet_oof_dir),
                "evaluation": str(resnet_evaluation),
                "metrics": metric_summary(resnet_metrics),
            },
            "efficientnet_b3": {
                "oof_dir": str(run_dir / "oof"),
                "evaluation": str(eff_evaluation_dir),
                "metrics": metric_summary(eff_metrics),
            },
            "fixed_50_50": {
                "oof_dir": str(blend_oof_dir),
                "evaluation": str(output_dir / "evaluation_50_50"),
                "metrics": metric_summary(blend_metrics),
            },
        },
        "selection": selection_payload,
        "test_components": {
            "resnet320_fivefold": {
                "path": str(resnet_test_path),
                "sha256": sha256_file(resnet_test_path),
                "full500_included": False,
            },
            "efficientnet_b3_fivefold": {
                "path": str(effnet_probability_path),
                "sha256": sha256_file(effnet_probability_path),
                "checkpoints": checkpoint_sources,
                "full500_included": False,
            },
        },
        "selected_test_rule": str(selected_name),
        "prediction_counts": selected_counts,
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
            "no_full500_component": True,
            "no_weight_tuning": True,
            "no_test_or_leaderboard_labels_used": True,
        },
        "artifacts": {
            "probabilities": {"path": str(probabilities_path), "sha256": sha256_file(probabilities_path)},
            "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
            "selection": {"path": str(output_dir / "selection.json"), "sha256": sha256_file(output_dir / "selection.json")},
        },
        "effnet_run_config": run_config,
    }
    atomic_json(output_dir / "provenance.json", provenance)
    result = {
        "candidates": candidates,
        "selected": str(selected_name),
        "samples": len(ids),
        "submission": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "probabilities": str(probabilities_path),
        "probabilities_sha256": sha256_file(probabilities_path),
        "elapsed_seconds": provenance["elapsed_seconds"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/round3_effnet_b3"))
    parser.add_argument("--manifest", type=Path, default=Path("data/folds_v2.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--resnet-oof-dir",
        type=Path,
        default=Path("runs/round2_multiscale_final/oof_candidates/single_320"),
    )
    parser.add_argument(
        "--resnet-evaluation",
        type=Path,
        default=Path("runs/round2_multiscale_final/evaluations/single_320"),
    )
    parser.add_argument(
        "--resnet-test-probabilities",
        type=Path,
        default=Path("runs/round2_multiscale_final/fivefold_test_probabilities.csv"),
    )
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--template", type=Path, default=Path("data/submission_fortask1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/round3_effnet_resnet_blend"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
