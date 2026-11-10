"""严格、可复现的五折 OOF 与测试集概率集成。

本模块没有导入时副作用。正式协议固定为两个模型槽 A/B：OOF 中每个样本只使用
其所属 fold 的 checkpoint，测试集先在每个槽内对五折等权平均，再以 A/B 各 0.5
平均。输入 recipe、manifest、类别顺序、ROI、TTA 与每个 checkpoint 的来源均会
校验并写入 provenance。

CSV 与 NPZ OOF 均受支持；文件操作只在公开函数或 CLI 被显式调用时发生。
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
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote, unquote

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import DEFAULT_MEAN, DEFAULT_STD, TestDualViewDataset, build_dual_view_transforms
    from .model import create_model, get_model_data_config
    from .roi import load_test_roi_cache, roi_config_from_mapping
except ImportError:  # pragma: no cover - direct script compatibility
    from dataset import DEFAULT_MEAN, DEFAULT_STD, TestDualViewDataset, build_dual_view_transforms
    from model import create_model, get_model_data_config
    from roi import load_test_roi_cache, roi_config_from_mapping


PathLike = str | os.PathLike[str]
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SLOTS = ("A", "B")
EXPECTED_SLOT_WEIGHTS = {"A": 0.5, "B": 0.5}
EXPECTED_TTA_ROTATIONS = (0, 90, 180, 270)
EXPECTED_TTA_VIEWS = ("full", "roi")
DEFAULT_EXPECTED_OOF_SAMPLES = 500
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


class PredictionError(ValueError):
    """预测输入或 provenance 不满足锁定协议。"""


@dataclass(frozen=True)
class PredictionBatch:
    """按输入顺序返回的 ID 与 float32 类别概率。"""

    sample_ids: list[str]
    probabilities: np.ndarray

    def __iter__(self):
        # 允许 ``ids, probabilities = result``，同时保留具名属性。
        yield self.sample_ids
        yield self.probabilities


@dataclass(frozen=True)
class OOFData:
    sample_ids: list[str]
    folds: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    class_names: list[str]
    sources: list[dict[str, str]]


def _sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: PathLike) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise PredictionError(f"JSON 不存在: {source}")
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PredictionError(f"JSON 无效: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise PredictionError(f"JSON 顶层必须是对象: {source}")
    return value


def _normalise_classes(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        mapping = {str(name): int(index) for name, index in value.items()}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        mapping = {str(name): index for index, name in enumerate(value)}
    else:
        raise PredictionError("类别必须是 class_to_idx 映射或有序名称列表")
    if not mapping or set(mapping.values()) != set(range(len(mapping))):
        raise PredictionError("类别编号必须从 0 开始连续且不重复")
    if len(mapping) != len(set(mapping)):
        raise PredictionError("类别名重复")
    return dict(sorted(mapping.items(), key=lambda item: item[1]))


def _classes_from_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    reverse: dict[int, str] = {}
    for index, row in enumerate(rows):
        if "class_idx" not in row:
            raise PredictionError(f"manifest 第 {index + 2} 行缺少 class_idx")
        class_index = _integer(row["class_idx"], f"manifest 第 {index + 2} 行 class_idx")
        class_name = str(row.get("class_name", class_index)).strip()
        if not class_name:
            raise PredictionError(f"manifest 第 {index + 2} 行 class_name 为空")
        if class_name in mapping and mapping[class_name] != class_index:
            raise PredictionError("manifest 类别名映射不唯一")
        if class_index in reverse and reverse[class_index] != class_name:
            raise PredictionError("manifest class_idx 映射不唯一")
        mapping[class_name] = class_index
        reverse[class_index] = class_name
    return _normalise_classes(mapping)


def _integer(value: Any, description: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"{description} 必须是整数") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise PredictionError(f"{description} 必须是有限整数")
    return int(number)


def _load_manifest(path: PathLike) -> tuple[list[dict[str, str]], dict[str, int], str]:
    source = Path(path)
    if not source.is_file():
        raise PredictionError(f"manifest 不存在: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        required = {"sample_id", "fold", "class_idx"}
        missing = required.difference(headers)
        if missing:
            raise PredictionError(f"manifest 缺少列: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise PredictionError("manifest 为空")
    ids = [str(row["sample_id"]).strip() for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise PredictionError("manifest sample_id 为空或重复")
    folds = {_integer(row["fold"], "manifest fold") for row in rows}
    if folds != set(EXPECTED_FOLDS):
        raise PredictionError(f"正式预测要求恰好五 fold 0..4，实际为 {sorted(folds)}")
    return rows, _classes_from_manifest(rows), _sha256_file(source)


def load_and_validate_recipe(recipe_path: PathLike) -> tuple[dict[str, Any], str]:
    """读取并验证固定双模型、五折、full+ROI、D4 TTA recipe。"""

    recipe = _read_json(recipe_path)
    if _integer(recipe.get("num_folds"), "recipe num_folds") != 5:
        raise PredictionError("recipe 必须固定为五 fold")
    if set((recipe.get("slots") or {}).keys()) != set(EXPECTED_SLOTS):
        raise PredictionError("recipe 必须且只能包含模型槽 A/B")
    ensemble = recipe.get("ensemble")
    if not isinstance(ensemble, Mapping):
        raise PredictionError("recipe 缺少 ensemble")
    weights = ensemble.get("slot_weights")
    try:
        normalized_weights = {str(key): float(value) for key, value in weights.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise PredictionError("ensemble.slot_weights 无效") from exc
    if normalized_weights != EXPECTED_SLOT_WEIGHTS:
        raise PredictionError("A/B 权重必须固定为 0.5/0.5")
    if ensemble.get("fold_weights") != "equal":
        raise PredictionError("测试集五 fold 权重必须为 equal")
    rotations = tuple(_integer(value, "TTA rotation") for value in recipe.get("tta_rotations_degrees", []))
    if rotations != EXPECTED_TTA_ROTATIONS:
        raise PredictionError(f"TTA 旋转必须严格为 {list(EXPECTED_TTA_ROTATIONS)}")
    views = tuple(str(value) for value in recipe.get("tta_views", recipe.get("views", [])))
    if views != EXPECTED_TTA_VIEWS:
        raise PredictionError("TTA views 必须按 full, roi 排列")
    if recipe.get("tta_probability_weights") != "equal_float32":
        raise PredictionError("TTA 概率必须以 float32 等权平均")
    view_weights = [float(value) for value in recipe.get("view_weights", [])]
    if view_weights != [0.5, 0.5]:
        raise PredictionError("full/roi view_weights 必须为 0.5/0.5")
    roi = recipe.get("roi")
    try:
        roi_config_from_mapping(roi)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"recipe ROI canonical schema 无效: {exc}") from exc
    if _integer(recipe.get("num_classes"), "recipe num_classes") <= 1:
        raise PredictionError("recipe num_classes 必须大于 1")
    recipe_id = str(recipe.get("recipe_id", "")).strip()
    if not recipe_id:
        raise PredictionError("recipe_id 为空")
    return recipe, _sha256_file(recipe_path)


def _validate_probabilities(probabilities: Any, rows: int | None = None, classes: int | None = None) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float32)
    if array.ndim != 2:
        raise PredictionError(f"概率必须是二维数组，实际 shape={array.shape}")
    if rows is not None and array.shape[0] != rows:
        raise PredictionError(f"概率行数应为 {rows}，实际为 {array.shape[0]}")
    if classes is not None and array.shape[1] != classes:
        raise PredictionError(f"概率类别数应为 {classes}，实际为 {array.shape[1]}")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise PredictionError("概率必须是 [0,1] 内有限值")
    sums = array.astype(np.float64).sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=2e-6):
        bad = int(np.flatnonzero(np.abs(sums - 1.0) > 2e-6)[0])
        raise PredictionError(f"第 {bad} 行概率和不为 1: {sums[bad]}")
    return array


def _batch_parts(batch: Any) -> tuple[torch.Tensor, list[str]]:
    if isinstance(batch, Mapping):
        images = batch.get("images", batch.get("image"))
        ids = batch.get("sample_ids", batch.get("sample_id", batch.get("ids")))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images = batch[0]
        ids = batch[-1]
    else:
        raise PredictionError("DataLoader batch 必须含 images 与 sample_ids")
    if not torch.is_tensor(images):
        raise PredictionError("batch images 必须是 Tensor")
    if isinstance(ids, (str, bytes)):
        id_list = [str(ids)]
    else:
        try:
            id_list = [str(value) for value in ids]
        except TypeError as exc:
            raise PredictionError("batch sample_ids 不是序列") from exc
    if images.shape[0] != len(id_list):
        raise PredictionError("batch 图像数与 sample_id 数不一致")
    if any(not value for value in id_list):
        raise PredictionError("sample_id 不允许为空")
    return images, id_list


def _rotate_tensor(images: torch.Tensor, degrees: int) -> torch.Tensor:
    if degrees % 90:
        raise PredictionError("Tensor TTA 仅支持 90 度倍数")
    return torch.rot90(images, k=(degrees // 90) % 4, dims=(-2, -1))


@torch.inference_mode()
def predict_fold_probabilities(
    model: torch.nn.Module,
    loader: Iterable[Any],
    device: str | torch.device | None = None,
    *,
    num_classes: int | None = None,
    tta_rotations: Sequence[int] | None = EXPECTED_TTA_ROTATIONS,
    expected_samples: int | None = None,
    amp_dtype: torch.dtype | None = None,
) -> PredictionBatch:
    """对一个 checkpoint 推理并等权平均 TTA 概率。

    loader 可给 ``[B,C,H,W]``，此时按 ``tta_rotations`` 生成旋转；也可给已经展开的
    ``[B,V,C,H,W]``（例如 full/ROI × D4），此时 V 个 view 直接等权。始终先对每个
    view 的 logits 做 float32 softmax，再平均，绝不平均 logits。返回 ID 和 float32
    ``[N,C]`` 概率，顺序与 loader 一致。
    """

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rotations = tuple(int(value) for value in (tta_rotations or (0,)))
    if not rotations:
        raise PredictionError("TTA rotations 不能为空")
    model.eval()
    all_ids: list[str] = []
    chunks: list[np.ndarray] = []
    seen: set[str] = set()
    for batch in loader:
        images, sample_ids = _batch_parts(batch)
        for sample_id in sample_ids:
            if sample_id in seen:
                raise PredictionError(f"推理 sample_id 重复: {sample_id!r}")
            seen.add(sample_id)
        images = images.to(target_device, non_blocking=True)
        if images.ndim != 5 or images.shape[1] != 2:
            raise PredictionError(f"锁定推理要求 [B,2,C,H,W]，实际 shape={tuple(images.shape)}")
        if rotations != EXPECTED_TTA_ROTATIONS:
            raise PredictionError("锁定推理必须使用 0/90/180/270 四旋转")
        views = [
            _rotate_tensor(images[:, view_index], rotation)
            for view_index in range(2)
            for rotation in rotations
        ]
        probability_sum: torch.Tensor | None = None
        for view in views:
            autocast = (
                torch.autocast(device_type=target_device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else contextlib.nullcontext()
            )
            with autocast:
                logits = model(view)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise PredictionError("模型输出必须是 [B,C] logits Tensor")
            if logits.shape[0] != len(sample_ids):
                raise PredictionError("模型输出 batch 大小不匹配")
            if num_classes is not None and logits.shape[1] != num_classes:
                raise PredictionError("模型输出类别数与契约不匹配")
            probabilities = logits.float().softmax(dim=1)
            probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
        assert probability_sum is not None
        averaged = (probability_sum / float(len(views))).cpu().numpy().astype(np.float32, copy=False)
        chunks.append(averaged)
        all_ids.extend(sample_ids)
    if not chunks:
        raise PredictionError("推理 DataLoader 为空")
    result = _validate_probabilities(np.concatenate(chunks), expected_samples, num_classes)
    return PredictionBatch(all_ids, result)


def _csv_probability_columns(headers: Sequence[str]) -> tuple[list[str], list[str]]:
    columns: list[str] = []
    names: list[str] = []
    for header in headers:
        folded = header.casefold()
        if folded.startswith("prob::"):
            columns.append(header)
            names.append(unquote(header[6:]))
    if not columns:
        indexed: list[tuple[int, str]] = []
        for header in headers:
            folded = header.casefold()
            if folded.startswith("prob_") and header[5:].isdigit():
                indexed.append((int(header[5:]), header))
        indexed.sort()
        if [index for index, _ in indexed] != list(range(len(indexed))):
            raise PredictionError("CSV prob_0...prob_N 必须连续")
        columns = [column for _, column in indexed]
        names = [str(index) for index, _ in indexed]
    if not columns or len(names) != len(set(names)):
        raise PredictionError("CSV 缺少有效且唯一的概率列")
    return columns, names


def _first_column(headers: Sequence[str], aliases: Sequence[str]) -> str | None:
    lookup = {name.casefold(): name for name in headers}
    return next((lookup[name.casefold()] for name in aliases if name.casefold() in lookup), None)


def _load_oof_csv(path: Path) -> OOFData:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    if not rows:
        raise PredictionError(f"OOF CSV 为空: {path}")
    id_column = _first_column(headers, ("sample_id", "id"))
    fold_column = _first_column(headers, ("fold", "fold_id", "cv_fold"))
    target_column = _first_column(headers, ("target", "label", "class_idx", "y_true", "true_label"))
    if id_column is None or fold_column is None or target_column is None:
        raise PredictionError(f"OOF CSV 缺少 sample_id/fold/target: {path}")
    probability_columns, names = _csv_probability_columns(headers)
    ids = [str(row[id_column]).strip() for row in rows]
    folds = np.asarray([_integer(row[fold_column], "OOF fold") for row in rows], dtype=np.int64)
    targets = np.asarray([_integer(row[target_column], "OOF target") for row in rows], dtype=np.int64)
    probabilities = _validate_probabilities(
        [[row[column] for column in probability_columns] for row in rows],
        rows=len(rows), classes=len(names),
    )
    return OOFData(ids, folds, targets, probabilities, names, [{"path": str(path.resolve()), "sha256": _sha256_file(path)}])


def _npz_scalar_strings(value: Any) -> list[str]:
    array = np.asarray(value)
    return [str(item) for item in array.reshape(-1).tolist()]


def _load_oof_npz(path: Path) -> OOFData:
    try:
        archive_context = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise PredictionError(f"NPZ 不能依赖 pickle: {path}") from exc
    with archive_context as archive:
        keys = set(archive.files)
        def choose(*aliases: str) -> str:
            match = next((name for name in aliases if name in keys), None)
            if match is None:
                raise PredictionError(f"OOF NPZ 缺少 {aliases}: {path}")
            return match
        ids = _npz_scalar_strings(archive[choose("sample_ids", "sample_id", "ids")])
        probabilities = _validate_probabilities(archive[choose("probabilities", "probs")], rows=len(ids))
        folds = np.asarray(archive[choose("folds", "fold")], dtype=np.int64).reshape(-1)
        targets = np.asarray(archive[choose("targets", "target", "y_true", "labels")], dtype=np.int64).reshape(-1)
        if "class_names" in keys:
            names = _npz_scalar_strings(archive["class_names"])
        elif "classes" in keys:
            names = _npz_scalar_strings(archive["classes"])
        else:
            names = [str(index) for index in range(probabilities.shape[1])]
    if len(folds) != len(ids) or len(targets) != len(ids):
        raise PredictionError(f"OOF NPZ 数组长度不一致: {path}")
    return OOFData(ids, folds, targets, probabilities, names, [{"path": str(path.resolve()), "sha256": _sha256_file(path)}])


def _oof_paths(source: PathLike | Sequence[PathLike]) -> list[Path]:
    values = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    paths: list[Path] = []
    for raw in values:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(
                (item for item in path.iterdir() if item.is_file() and item.suffix.casefold() in {".csv", ".npz"}),
                key=lambda item: item.name.casefold(),
            )
            # 避免把评估目录中的 errors/confusion 等 CSV 当作 OOF。
            preferred = [item for item in candidates if "fold" in item.stem.casefold() or "oof" in item.stem.casefold()]
            paths.extend(preferred or candidates)
        else:
            paths.append(path)
    if not paths:
        raise PredictionError("没有 OOF CSV/NPZ")
    if any(not path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise PredictionError(f"OOF 文件不存在: {missing}")
    return paths


def _load_oof_source(source: PathLike | Sequence[PathLike]) -> OOFData:
    pieces = []
    for path in _oof_paths(source):
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            pieces.append(_load_oof_csv(path))
        elif suffix == ".npz":
            pieces.append(_load_oof_npz(path))
        else:
            raise PredictionError(f"OOF 仅支持 CSV/NPZ: {path}")
    first = pieces[0]
    for piece in pieces[1:]:
        if piece.class_names != first.class_names:
            raise PredictionError("同一模型各 OOF 文件类别顺序不一致")
    return OOFData(
        [item for piece in pieces for item in piece.sample_ids],
        np.concatenate([piece.folds for piece in pieces]),
        np.concatenate([piece.targets for piece in pieces]),
        np.concatenate([piece.probabilities for piece in pieces]),
        first.class_names,
        [source_item for piece in pieces for source_item in piece.sources],
    )


def _align_oof(
    data: OOFData,
    manifest_rows: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    slot: str,
    expected_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if expected_samples is not None and len(data.sample_ids) != expected_samples:
        raise PredictionError(f"模型 {slot} OOF 必须为 {expected_samples} 条，实际 {len(data.sample_ids)}")
    if len(data.sample_ids) != len(set(data.sample_ids)):
        raise PredictionError(f"模型 {slot} OOF sample_id 重复")
    manifest_ids = [str(row["sample_id"]).strip() for row in manifest_rows]
    if set(data.sample_ids) != set(manifest_ids) or len(data.sample_ids) != len(manifest_ids):
        missing = set(manifest_ids) - set(data.sample_ids)
        extra = set(data.sample_ids) - set(manifest_ids)
        raise PredictionError(f"模型 {slot} OOF 未恰好覆盖 manifest：缺失 {len(missing)}，额外 {len(extra)}")
    numeric_names = [str(index) for index in range(len(class_names))]
    if data.class_names not in (list(class_names), numeric_names):
        raise PredictionError(f"模型 {slot} OOF 类别顺序不匹配: {data.class_names}")
    by_id = {sample_id: index for index, sample_id in enumerate(data.sample_ids)}
    order = np.asarray([by_id[sample_id] for sample_id in manifest_ids], dtype=np.int64)
    folds = data.folds[order]
    targets = data.targets[order]
    probabilities = data.probabilities[order]
    expected_folds = np.asarray([_integer(row["fold"], "manifest fold") for row in manifest_rows])
    expected_targets = np.asarray([_integer(row["class_idx"], "manifest class_idx") for row in manifest_rows])
    if not np.array_equal(folds, expected_folds):
        raise PredictionError(f"模型 {slot} OOF 含非对应 fold 的预测")
    if not np.array_equal(targets, expected_targets):
        raise PredictionError(f"模型 {slot} OOF 真值与 manifest 不一致")
    for fold in EXPECTED_FOLDS:
        if not np.any(folds == fold):
            raise PredictionError(f"模型 {slot} OOF 缺少 fold {fold}")
    return folds, targets, probabilities


def _exclusive_destination(path: PathLike, forbidden: Sequence[PathLike] = ()) -> Path:
    destination = Path(path).expanduser().resolve()
    forbidden_paths = {Path(value).expanduser().resolve() for value in forbidden}
    if destination in forbidden_paths:
        raise PredictionError(f"拒绝覆盖只读输入/template: {destination}")
    if os.path.lexists(destination):
        raise FileExistsError(f"拒绝覆盖已存在的输出: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _write_csv_exclusive(path: PathLike, headers: Sequence[str], rows: Iterable[Mapping[str, Any]], *, forbidden: Sequence[PathLike] = ()) -> Path:
    destination = _exclusive_destination(path, forbidden)
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(headers), extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"拒绝覆盖并发创建的输出: {destination}") from exc
        return destination
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def _write_json_exclusive(path: PathLike, payload: Mapping[str, Any], *, forbidden: Sequence[PathLike] = ()) -> Path:
    destination = _exclusive_destination(path, forbidden)
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"拒绝覆盖并发创建的输出: {destination}") from exc
        return destination
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def _write_npz_exclusive(path: PathLike, *, forbidden: Sequence[PathLike] = (), **arrays: Any) -> Path:
    destination = _exclusive_destination(path, forbidden)
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
        os.close(fd)
        np.savez_compressed(temporary, **arrays)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"拒绝覆盖并发创建的输出: {destination}") from exc
        return destination
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def ensemble_oof_runs(
    run_a: PathLike | Sequence[PathLike],
    run_b: PathLike | Sequence[PathLike],
    manifest_path: PathLike,
    output_dir: PathLike,
    *,
    expected_samples: int | None = DEFAULT_EXPECTED_OOF_SAMPLES,
) -> dict[str, Any]:
    """严格合并 A/B OOF；每个样本仅接受其 manifest fold，固定各 0.5。

    每个模型源可以是一个 CSV/NPZ、五个逐折文件序列或目录。写出
    ``oof_probabilities.csv/.npz`` 与 ``provenance.json``，且拒绝覆盖已有产物。
    """

    manifest_rows, class_to_idx, manifest_hash = _load_manifest(manifest_path)
    if expected_samples is not None and len(manifest_rows) != expected_samples:
        raise PredictionError(f"manifest 应为 {expected_samples} 条，实际 {len(manifest_rows)}")
    class_names = list(class_to_idx)
    a = _load_oof_source(run_a)
    b = _load_oof_source(run_b)
    folds_a, targets_a, probabilities_a = _align_oof(a, manifest_rows, class_names, "A", expected_samples)
    folds_b, targets_b, probabilities_b = _align_oof(b, manifest_rows, class_names, "B", expected_samples)
    if not np.array_equal(folds_a, folds_b) or not np.array_equal(targets_a, targets_b):
        raise PredictionError("A/B OOF fold 或真值不一致")
    probabilities = _validate_probabilities(
        probabilities_a * np.float32(0.5) + probabilities_b * np.float32(0.5),
        rows=len(manifest_rows), classes=len(class_names),
    )
    output = Path(output_dir).expanduser().resolve()
    ids = [str(row["sample_id"]).strip() for row in manifest_rows]
    headers = ["sample_id", "fold", "target", "prediction", *[f"prob::{quote(name, safe='')}" for name in class_names]]
    rows = []
    for index, sample_id in enumerate(ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "fold": int(folds_a[index]),
            "target": int(targets_a[index]),
            "prediction": int(np.argmax(probabilities[index])),
        }
        row.update({headers[4 + class_index]: float(probabilities[index, class_index]) for class_index in range(len(class_names))})
        rows.append(row)
    csv_path = _write_csv_exclusive(output / "oof_probabilities.csv", headers, rows)
    npz_path = _write_npz_exclusive(
        output / "oof_probabilities.npz",
        sample_ids=np.asarray(ids), folds=folds_a, targets=targets_a,
        probabilities=probabilities, class_names=np.asarray(class_names),
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "kind": "oof_ab_ensemble",
        "manifest": {"path": str(Path(manifest_path).resolve()), "sha256": manifest_hash},
        "class_to_idx": class_to_idx,
        "folds": list(EXPECTED_FOLDS),
        "sample_count_per_model": len(ids),
        "oof_rule": "each sample uses only its manifest fold",
        "slot_weights": EXPECTED_SLOT_WEIGHTS,
        "inputs": {"A": a.sources, "B": b.sources},
        "outputs": {
            "csv": {"path": str(csv_path), "sha256": _sha256_file(csv_path)},
            "npz": {"path": str(npz_path), "sha256": _sha256_file(npz_path)},
        },
    }
    provenance_path = _write_json_exclusive(output / "provenance.json", provenance)
    provenance["provenance_path"] = str(provenance_path)
    return provenance


def _provenance_value(checkpoint: Mapping[str, Any], *names: str) -> Any:
    containers = [checkpoint]
    for key in ("provenance", "metadata", "contract"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        for name in names:
            if name in container:
                return container[name]
    return None


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    *,
    slot: str,
    fold: int,
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    manifest_sha256: str,
    class_to_idx: Mapping[str, int],
) -> str:
    """严格验证 checkpoint，并返回实际模型 ID。"""

    if slot not in EXPECTED_SLOTS or fold not in EXPECTED_FOLDS:
        raise PredictionError("checkpoint slot/fold 超出锁定协议")
    checks = (
        ("slot", _provenance_value(checkpoint, "slot", "model_slot"), slot),
        ("fold", _provenance_value(checkpoint, "fold"), fold),
        ("recipe_id", _provenance_value(checkpoint, "recipe_id"), recipe["recipe_id"]),
        ("recipe_sha256", _provenance_value(checkpoint, "recipe_sha256", "recipe_hash"), recipe_sha256),
        ("manifest_sha256", _provenance_value(checkpoint, "manifest_sha256", "manifest_hash"), manifest_sha256),
    )
    for name, actual, expected in checks:
        if actual is None:
            raise PredictionError(f"checkpoint 缺少 provenance 字段 {name}")
        if name == "fold":
            actual = _integer(actual, "checkpoint fold")
        if actual != expected:
            raise PredictionError(f"checkpoint {name} 不匹配: {actual!r} != {expected!r}")
    raw_classes = _provenance_value(checkpoint, "class_to_idx", "classes")
    if raw_classes is None or _normalise_classes(raw_classes) != _normalise_classes(class_to_idx):
        raise PredictionError("checkpoint 类别顺序不匹配")
    roi = _provenance_value(checkpoint, "roi", "roi_config")
    if roi is None or _json_hash(roi) != _json_hash(recipe["roi"]):
        raise PredictionError("checkpoint ROI 配置不匹配")
    rotations = _provenance_value(checkpoint, "tta_rotations_degrees", "tta_rotations")
    views = _provenance_value(checkpoint, "tta_views", "views")
    if tuple(int(value) for value in (rotations or ())) != EXPECTED_TTA_ROTATIONS:
        raise PredictionError("checkpoint TTA rotations 不匹配")
    if tuple(str(value) for value in (views or ())) != EXPECTED_TTA_VIEWS:
        raise PredictionError("checkpoint TTA views 不匹配")
    weights_kind = _provenance_value(checkpoint, "tta_probability_weights")
    if weights_kind != "equal_float32":
        raise PredictionError("checkpoint TTA 概率平均契约不匹配")
    actual_model_id = _provenance_value(checkpoint, "actual_model_id", "resolved_model_id", "model_name")
    if not isinstance(actual_model_id, str) or not actual_model_id:
        raise PredictionError("checkpoint 缺少实际 model ID")
    slot_recipe = recipe["slots"][slot]
    allowed = {str(slot_recipe.get("requested_model_id", "")), str(slot_recipe.get("fallback_model_id", ""))}
    allowed.discard("")
    if actual_model_id not in allowed:
        raise PredictionError(f"checkpoint 实际 model ID 不属于槽 {slot}: {actual_model_id!r}")
    final_weights = _provenance_value(checkpoint, "final_weights")
    if final_weights is not None and final_weights != recipe.get("final_weights"):
        raise PredictionError("checkpoint final_weights 类型与 recipe 不匹配")
    return actual_model_id


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise PredictionError(f"checkpoint 顶层不是映射: {path}")
    return checkpoint


def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state", "state_dict", "model_state_dict", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    raise PredictionError("checkpoint 缺少模型 state_dict")


def _checkpoint_paths(source: PathLike | Sequence[PathLike], slot: str, final_filename: str) -> dict[int, Path]:
    if isinstance(source, (str, os.PathLike)):
        root = Path(source)
        if root.is_dir():
            candidates = list(root.rglob(final_filename))
        else:
            candidates = [root]
    else:
        candidates = [Path(value) for value in source]
    if len(candidates) != 5:
        raise PredictionError(f"槽 {slot} 必须恰好提供 5 个 checkpoint，实际 {len(candidates)}")
    result: dict[int, Path] = {}
    for path in candidates:
        if not path.is_file():
            raise PredictionError(f"checkpoint 不存在: {path}")
        # fold 最终以 checkpoint 内容验证；这里只在常见路径名中推断排序键。
        inferred = None
        for part in reversed(path.parts):
            folded = part.casefold()
            if folded.startswith("fold_") and folded[5:].isdigit():
                inferred = int(folded[5:])
                break
        if inferred is None:
            # 无路径 fold 信息时稍后加载小型元数据；torch checkpoint 无法仅读 header。
            checkpoint = _load_checkpoint(path, torch.device("cpu"))
            inferred = _integer(_provenance_value(checkpoint, "fold"), "checkpoint fold")
            del checkpoint
        if inferred in result:
            raise PredictionError(f"槽 {slot} checkpoint fold 重复: {inferred}")
        result[inferred] = path
    if set(result) != set(EXPECTED_FOLDS):
        raise PredictionError(f"槽 {slot} checkpoint 必须覆盖 fold 0..4，实际 {sorted(result)}")
    return result


def _read_template(template_path: PathLike) -> tuple[str, str, list[str], list[str]]:
    path = Path(template_path)
    if not path.is_file():
        raise PredictionError(f"submission template 不存在: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        id_column = _first_column(headers, ("ID", "id"))
        category_column = _first_column(headers, ("Category", "category"))
        if id_column is None or category_column is None:
            raise PredictionError("template 必须包含 ID 与 Category 列")
        rows = [dict(row) for row in reader]
    ids = [str(row[id_column]).strip() for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise PredictionError("template ID 为空或重复")
    # Category 有意完全忽略；它可能是占位值，绝不能决定预测类别或顺序。
    return id_column, category_column, ids, headers


def _test_paths(ids: Sequence[str], test_dir: PathLike) -> list[Path]:
    directory = Path(test_dir)
    if not directory.is_dir():
        raise PredictionError(f"test_dir 不存在或不是目录: {directory}")
    images = sorted((path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() in _IMAGE_EXTENSIONS), key=lambda path: path.name.casefold())
    by_name: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for image in images:
        for key in {image.name, image.stem}:
            if key in by_name and by_name[key] != image:
                ambiguous.add(key)
            by_name[key] = image
    paths = []
    for sample_id in ids:
        if sample_id in ambiguous or sample_id not in by_name:
            raise PredictionError(f"template ID 无法唯一映射到测试图像: {sample_id!r}")
        paths.append(by_name[sample_id])
    if len(set(paths)) != len(paths):
        raise PredictionError("多个 template ID 映射到同一测试图像")
    if set(paths) != set(images):
        raise PredictionError(f"template 与 test_dir 不完全对应：template {len(paths)}，图像 {len(images)}")
    return paths


def _amp_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "bf16_auto" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        return torch.bfloat16
    if precision in {"bf16_auto", "fp16"}:
        return torch.float16
    return None


def predict_test_ensemble(
    recipe_path: PathLike,
    checkpoints: Mapping[str, PathLike | Sequence[PathLike]],
    template_path: PathLike,
    output_dir: PathLike,
    *,
    manifest_path: PathLike | None = None,
    test_dir: PathLike | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    roi_cache_path: PathLike | None = None,
    device: str | torch.device | None = None,
    model_factory: Callable[..., torch.nn.Module] = create_model,
) -> dict[str, Any]:
    """执行锁定测试集集成并输出概率、submission 与 provenance。

    每槽必须恰好五个 final checkpoint。每个 checkpoint 单独加载、推理、释放；先在
    槽内五折等权，再以 A/B 各 0.5 合并。template 仅提供 ID 顺序，原 Category
    完全忽略；任何输出都不得覆盖 template，且默认拒绝覆盖已有文件。
    """

    recipe, recipe_hash = load_and_validate_recipe(recipe_path)
    recipe_root = Path(recipe_path).expanduser().resolve().parent.parent
    manifest = Path(manifest_path) if manifest_path is not None else recipe_root / str(recipe["manifest"])
    manifest_rows, class_to_idx, manifest_hash = _load_manifest(manifest)
    if len(class_to_idx) != _integer(recipe["num_classes"], "recipe num_classes"):
        raise PredictionError("recipe num_classes 与 manifest 不一致")
    if set(checkpoints) != set(EXPECTED_SLOTS):
        raise PredictionError("checkpoints 必须且只能提供 A/B")
    final_filename = str((recipe.get("checkpoint") or {}).get("final_filename", "final_weights.pt"))
    checkpoint_paths = {
        slot: _checkpoint_paths(checkpoints[slot], slot, final_filename)
        for slot in EXPECTED_SLOTS
    }
    id_column, category_column, ids, template_headers = _read_template(template_path)
    resolved_test_dir = Path(test_dir) if test_dir is not None else recipe_root / str(recipe["test_dir"])
    image_paths = _test_paths(ids, resolved_test_dir)
    roi_config = roi_config_from_mapping(recipe["roi"])
    resolved_roi_cache = Path(roi_cache_path) if roi_cache_path is not None else recipe_root / str(recipe["output_root"]) / "roi_test.json"
    roi_cache = load_test_roi_cache(resolved_roi_cache, resolved_test_dir, config=roi_config, verify_files=True)
    roi_cache_sha256 = _sha256_file(resolved_roi_cache)
    # cache key 是测试目录相对路径；模板顺序映射为 dataset sample_id。
    cache_by_id = {sample_id: roi_cache[path.relative_to(resolved_test_dir).as_posix()]
                   for sample_id, path in zip(ids, image_paths)}
    if num_workers < 0:
        raise PredictionError("num_workers 不能为负")
    effective_batch_size = int(batch_size or recipe.get("batch_size", 16))
    if effective_batch_size <= 0:
        raise PredictionError("batch_size 必须为正")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_names = list(class_to_idx)
    slot_probabilities: dict[str, np.ndarray] = {}
    checkpoint_provenance: dict[str, list[dict[str, Any]]] = {slot: [] for slot in EXPECTED_SLOTS}

    for slot in EXPECTED_SLOTS:
        fold_sum: np.ndarray | None = None
        for fold in EXPECTED_FOLDS:
            checkpoint_path = checkpoint_paths[slot][fold]
            checkpoint = _load_checkpoint(checkpoint_path, target_device)
            model_id = validate_checkpoint_provenance(
                checkpoint, slot=slot, fold=fold, recipe=recipe,
                recipe_sha256=recipe_hash, manifest_sha256=manifest_hash,
                class_to_idx=class_to_idx,
            )
            model = model_factory(model_name=model_id, num_classes=len(class_names), pretrained=False)
            model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
            model.to(target_device)
            data_config = get_model_data_config(model)
            input_size = data_config.get("input_size", (3, recipe["image_size"], recipe["image_size"]))
            model_image_size = int(input_size[-1])
            if model_image_size != int(recipe["image_size"]):
                # recipe 是本实验锁定的实际尺寸；timm 默认配置不能悄悄改变它。
                model_image_size = int(recipe["image_size"])
            _, deterministic_transform = build_dual_view_transforms(
                model_image_size, mean=data_config.get("mean", DEFAULT_MEAN),
                std=data_config.get("std", DEFAULT_STD),
                interpolation=data_config.get("interpolation", "bicubic"),
            )
            dataset = TestDualViewDataset(
                [{"path": str(path), "sample_id": sample_id} for sample_id, path in zip(ids, image_paths)],
                deterministic_transform, roi_cache=cache_by_id,
                roi_cache_sha256=roi_cache_sha256, auto_roi=False,
            )
            loader = DataLoader(
                dataset, batch_size=effective_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=target_device.type == "cuda",
                persistent_workers=False,
            )
            prediction = predict_fold_probabilities(
                model, loader, target_device, num_classes=len(class_names),
                tta_rotations=EXPECTED_TTA_ROTATIONS, expected_samples=len(ids),
                amp_dtype=_amp_dtype(target_device, str(recipe.get("precision", "fp32"))),
            )
            if prediction.sample_ids != ids:
                raise PredictionError(f"槽 {slot} fold {fold} 推理 ID 顺序改变")
            fold_sum = prediction.probabilities if fold_sum is None else fold_sum + prediction.probabilities
            checkpoint_provenance[slot].append({
                "fold": fold,
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256_file(checkpoint_path),
                "actual_model_id": model_id,
            })
            # 保证任何时刻仅驻留一个模型/checkpoint，降低测试推理显存峰值。
            del prediction, loader, dataset, model, checkpoint
            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
        assert fold_sum is not None
        slot_probabilities[slot] = _validate_probabilities(
            fold_sum / np.float32(5.0), rows=len(ids), classes=len(class_names)
        )

    probabilities = _validate_probabilities(
        slot_probabilities["A"] * np.float32(0.5) + slot_probabilities["B"] * np.float32(0.5),
        rows=len(ids), classes=len(class_names),
    )
    predicted = probabilities.argmax(axis=1)
    output = Path(output_dir).expanduser().resolve()
    template = Path(template_path).expanduser().resolve()
    probability_headers = [f"prob::{quote(name, safe='')}" for name in class_names]
    probability_rows = []
    for row_index, sample_id in enumerate(ids):
        row: dict[str, Any] = {"ID": sample_id, "prediction": int(predicted[row_index]), "Category": class_names[int(predicted[row_index])]}
        row.update({header: float(probabilities[row_index, index]) for index, header in enumerate(probability_headers)})
        probability_rows.append(row)
    probability_csv = _write_csv_exclusive(
        output / "test_probabilities.csv", ["ID", "prediction", "Category", *probability_headers],
        probability_rows, forbidden=[template],
    )
    probability_npz = _write_npz_exclusive(
        output / "test_probabilities.npz", forbidden=[template],
        ids=np.asarray(ids), probabilities=probabilities, class_names=np.asarray(class_names),
    )
    # 输出 submission 严格保留 template 的 ID 顺序和列名；非 ID/Category 附加列若有，
    # 写空值而不传播 template 内容，避免把占位 Category 或其他泄漏字段带入结果。
    submission_rows = []
    for sample_id, class_index in zip(ids, predicted.tolist()):
        row = {header: "" for header in template_headers}
        row[id_column] = sample_id
        row[category_column] = class_names[int(class_index)]
        submission_rows.append(row)
    submission = _write_csv_exclusive(
        output / "submission.csv", template_headers, submission_rows, forbidden=[template]
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "kind": "test_ab_five_fold_ensemble",
        "recipe": {"path": str(Path(recipe_path).resolve()), "sha256": recipe_hash, "recipe_id": recipe["recipe_id"]},
        "manifest": {"path": str(manifest.resolve()), "sha256": manifest_hash},
        "template": {"path": str(template), "sha256": _sha256_file(template), "id_order_only": True, "category_ignored": True},
        "class_to_idx": class_to_idx,
        "roi": recipe["roi"],
        "roi_cache": {"path": str(resolved_roi_cache.resolve()), "sha256": roi_cache_sha256},
        "tta": {"views": list(EXPECTED_TTA_VIEWS), "rotations_degrees": list(EXPECTED_TTA_ROTATIONS), "probability_weights": "equal_float32"},
        "ensemble": {"within_slot": "five folds equal weight 0.2", "slot_weights": EXPECTED_SLOT_WEIGHTS},
        "checkpoints": checkpoint_provenance,
        "sample_count": len(ids),
        "outputs": {
            "probabilities_csv": {"path": str(probability_csv), "sha256": _sha256_file(probability_csv)},
            "probabilities_npz": {"path": str(probability_npz), "sha256": _sha256_file(probability_npz)},
            "submission": {"path": str(submission), "sha256": _sha256_file(submission)},
        },
    }
    provenance_path = _write_json_exclusive(output / "provenance.json", provenance, forbidden=[template])
    provenance["provenance_path"] = str(provenance_path)
    return provenance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="严格五折双模型 OOF/测试集概率预测与集成。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    oof = subparsers.add_parser("ensemble-oof", help="固定 0.5/0.5 合并 A/B OOF CSV/NPZ")
    oof.add_argument("--run-a", nargs="+", required=True)
    oof.add_argument("--run-b", nargs="+", required=True)
    oof.add_argument("--manifest", required=True)
    oof.add_argument("--output-dir", required=True)
    oof.add_argument("--expected-samples", type=int, default=DEFAULT_EXPECTED_OOF_SAMPLES)

    test = subparsers.add_parser("test", help="五折槽内平均后固定 0.5/0.5 测试集成")
    test.add_argument("--recipe", required=True)
    test.add_argument("--checkpoints-a", nargs="+", required=True)
    test.add_argument("--checkpoints-b", nargs="+", required=True)
    test.add_argument("--template", required=True)
    test.add_argument("--output-dir", required=True)
    test.add_argument("--manifest")
    test.add_argument("--test-dir")
    test.add_argument("--batch-size", type=int)
    test.add_argument("--num-workers", type=int, default=0)
    test.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "ensemble-oof":
            result = ensemble_oof_runs(
                arguments.run_a, arguments.run_b, arguments.manifest, arguments.output_dir,
                expected_samples=arguments.expected_samples,
            )
        else:
            result = predict_test_ensemble(
                arguments.recipe,
                {"A": arguments.checkpoints_a, "B": arguments.checkpoints_b},
                arguments.template, arguments.output_dir,
                manifest_path=arguments.manifest, test_dir=arguments.test_dir,
                batch_size=arguments.batch_size, num_workers=arguments.num_workers,
                device=arguments.device,
            )
    except (PredictionError, FileExistsError, OSError, csv.Error, RuntimeError) as exc:
        print(f"预测失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


__all__ = [
    "DEFAULT_EXPECTED_OOF_SAMPLES",
    "EXPECTED_FOLDS",
    "EXPECTED_SLOTS",
    "OOFData",
    "PredictionBatch",
    "PredictionError",
    "ensemble_oof_runs",
    "load_and_validate_recipe",
    "predict_fold_probabilities",
    "predict_test_ensemble",
    "validate_checkpoint_provenance",
]


if __name__ == "__main__":
    raise SystemExit(main())
