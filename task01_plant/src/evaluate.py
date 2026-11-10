"""严格合并并评估交叉验证的 OOF（out-of-fold）预测。

推荐训练代码在每个 fold 结束时调用 :func:`write_fold_oof_csv`。全部 fold
完成后调用 :func:`evaluate_oof`，或直接运行本模块的 CLI。模块导入时不会读写
文件，也不会执行评估。

标准 OOF CSV 的概率列写作 ``prob::<URL 编码后的类别名>``；列的物理顺序就是
类别顺序。评估器也兼容常见的 ``prob_<类别>``、``probability_<类别>``、
``p_<类别>``，以及在显式给出 class_names 时直接以类别名命名的概率列。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import quote, unquote


PathLike = Union[str, os.PathLike]

_ID_COLUMNS = ("sample_id", "id")
_PATH_COLUMNS = ("image_path", "path", "filepath", "file_path", "filename")
_LABEL_COLUMNS = ("label", "class_idx", "target", "y_true", "true_label")
_CLASS_COLUMNS = ("class_name", "true_class", "target_class")
_FOLD_COLUMNS = ("fold", "fold_id", "cv_fold")
_SHA_COLUMNS = ("sha256", "sha_256")
_GROUP_COLUMNS = ("group_id", "group", "patient_id", "plant_id")
_NON_FINITE_TEXT = {
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


class EvaluationError(ValueError):
    """输入 OOF 或 manifest 不满足完整性约束。"""


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _require_finite_token(value: Any, description: str) -> str:
    text = _as_text(value)
    if not text:
        raise EvaluationError(f"{description} 为空")
    if text.casefold() in _NON_FINITE_TEXT:
        raise EvaluationError(f"{description} 不是有限值: {text!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise EvaluationError(f"{description} 不是有限值: {value!r}")
    return text


def _normalise_image_path(value: Any) -> str:
    text = _require_finite_token(value, "image_path")
    # 路径只用作跨 CSV 的稳定键，不访问文件系统。统一 Windows/POSIX 分隔符，
    # 并折叠重复分隔符；casefold 让 Windows 路径大小写差异不造成假缺失。
    normalised = text.replace("\\", "/")
    while "//" in normalised:
        normalised = normalised.replace("//", "/")
    return normalised.casefold()


def make_sample_id(
    *,
    sample_id: Any = None,
    sha256: Any = None,
    image_path: Any = None,
) -> str:
    """按 sample_id、sha256、image_path 的优先级生成稳定样本键。

    显式 sample_id 原样（去首尾空白）使用；哈希转小写；路径统一分隔符并
    casefold。三个字段至少有一个非空，数值型 NaN/Inf 或文本 NaN/Inf 会失败。
    """

    if _as_text(sample_id):
        return _require_finite_token(sample_id, "sample_id")
    if _as_text(sha256):
        return _require_finite_token(sha256, "sha256").casefold()
    if _as_text(image_path):
        return _normalise_image_path(image_path)
    raise EvaluationError("无法确定 sample_id：sample_id、sha256、image_path 均为空")


def _parse_fold(value: Any, description: str = "fold") -> int:
    text = _require_finite_token(value, description)
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{description} 必须是有限整数，实际为 {text!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise EvaluationError(f"{description} 必须是有限整数，实际为 {text!r}")
    return int(number)


def _normalise_headers(fieldnames: Optional[Sequence[str]], source: PathLike) -> List[str]:
    if not fieldnames:
        raise EvaluationError(f"CSV 没有表头: {source}")
    result = []
    seen = set()
    for raw in fieldnames:
        if raw is None:
            raise EvaluationError(f"CSV 表头含空列名: {source}")
        name = raw.lstrip("﻿").strip()
        key = name.casefold()
        if not name or key in seen:
            raise EvaluationError(f"CSV 表头存在空列名或重复列名 {name!r}: {source}")
        seen.add(key)
        result.append(name)
    return result


def _find_column(headers: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {name.casefold(): name for name in headers}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    return None


def _value(row: Mapping[str, Any], column: Optional[str]) -> str:
    return _as_text(row.get(column)) if column is not None else ""


def _read_csv(path: PathLike, encoding: str) -> Tuple[List[str], List[Dict[str, str]]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise EvaluationError(f"CSV 不存在或不是文件: {csv_path}")
    try:
        with csv_path.open("r", newline="", encoding=encoding) as handle:
            reader = csv.DictReader(handle)
            raw_headers = reader.fieldnames
            headers = _normalise_headers(raw_headers, csv_path)
            # DictReader 仍以原始表头为键；在这里统一键名，并拒绝多余字段。
            original = list(raw_headers or [])
            rows: List[Dict[str, str]] = []
            for line_number, raw in enumerate(reader, start=2):
                if None in raw:
                    raise EvaluationError(f"{csv_path}:{line_number} 的字段数多于表头")
                clean = {
                    headers[index]: _as_text(raw.get(original[index]))
                    for index in range(len(headers))
                }
                rows.append(clean)
    except UnicodeError as exc:
        raise EvaluationError(f"无法用 {encoding!r} 解码 CSV {csv_path}: {exc}") from exc
    if not rows:
        raise EvaluationError(f"CSV 没有数据行: {csv_path}")
    return headers, rows


def _probability_columns(
    headers: Sequence[str], explicit_class_names: Optional[Sequence[str]] = None
) -> Tuple[List[str], List[str]]:
    columns: List[str] = []
    names: List[str] = []

    for header in headers:
        folded = header.casefold()
        encoded_name: Optional[str] = None
        for prefix in ("prob::", "probability::"):
            if folded.startswith(prefix):
                encoded_name = header[len(prefix) :]
                break
        if encoded_name is not None:
            name = unquote(encoded_name).strip()
            if not name:
                raise EvaluationError(f"概率列类别名为空: {header!r}")
            columns.append(header)
            names.append(name)

    if not columns:
        for header in headers:
            folded = header.casefold()
            name = None
            for prefix in ("probability_", "prob_", "p_"):
                if folded.startswith(prefix):
                    name = header[len(prefix) :].strip()
                    break
            if name:
                columns.append(header)
                names.append(name)

    if not columns and explicit_class_names:
        lookup = {header: header for header in headers}
        if all(name in lookup for name in explicit_class_names):
            columns = [lookup[name] for name in explicit_class_names]
            names = list(explicit_class_names)

    if not columns:
        raise EvaluationError(
            "未找到类别概率列；请使用 prob::<类别>、prob_<类别>，"
            "或通过 class_names 明确直接类别列"
        )
    if len(set(names)) != len(names):
        raise EvaluationError(f"概率列解析出重复类别: {names!r}")
    # prob_0...prob_N 是常见格式；显式类别顺序给出时将数字后缀解释为索引，
    # 而不是字面类别名，同时仍保持 CSV 中的实际列顺序。
    if explicit_class_names and all(name.isdigit() for name in names):
        indices = [int(name) for name in names]
        if indices != list(range(len(explicit_class_names))) or len(columns) != len(explicit_class_names):
            raise EvaluationError(
                f"数字概率列必须按 prob_0...prob_N 连续排列，实际索引为 {indices!r}"
            )
        names = list(explicit_class_names)
    return columns, names


def _atomic_write_csv(
    path: PathLike, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        return destination
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _sha256_file(path: PathLike) -> str:
    """流式计算输入/输出产物哈希，不读取任何图像。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: PathLike, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="\n",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        return destination
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _coerce_class_names(class_names: Sequence[Any]) -> List[str]:
    names = [_require_finite_token(name, "class_name") for name in class_names]
    if not names:
        raise EvaluationError("class_names 不能为空")
    if len(set(names)) != len(names):
        raise EvaluationError(f"class_names 含重复类别: {names!r}")
    return names


def _truth_index(value: Any, class_names: Sequence[str], description: str) -> int:
    text = _require_finite_token(value, description)
    if text in class_names:
        return class_names.index(text)
    try:
        number = float(text)
    except ValueError as exc:
        raise EvaluationError(f"{description} {text!r} 不在类别顺序中") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise EvaluationError(f"{description} 必须是类别名或有限整数索引: {text!r}")
    index = int(number)
    if index < 0 or index >= len(class_names):
        raise EvaluationError(
            f"{description} 类别索引越界: {index}，类别数为 {len(class_names)}"
        )
    return index


def _validate_probability_vector(
    values: Sequence[Any], description: str, sum_tolerance: float
) -> List[float]:
    if not math.isfinite(sum_tolerance) or sum_tolerance < 0.0:
        raise EvaluationError("probability_sum_tolerance 必须是非负有限值")
    probabilities: List[float] = []
    for index, raw_probability in enumerate(values):
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise EvaluationError(
                f"{description} 第 {index} 个概率不是数值: {raw_probability!r}"
            ) from exc
        if not math.isfinite(probability):
            raise EvaluationError(f"{description} 第 {index} 个概率不是有限值")
        if probability < 0.0 or probability > 1.0:
            raise EvaluationError(
                f"{description} 第 {index} 个概率超出 [0, 1]: {probability}"
            )
        probabilities.append(probability)
    probability_sum = math.fsum(probabilities)
    if abs(probability_sum - 1.0) > sum_tolerance:
        raise EvaluationError(
            f"{description} 概率和应为 1（容差 {sum_tolerance:g}），实际为 {probability_sum:.17g}"
        )
    return probabilities


def _records_list(value: Any, description: str) -> List[Dict[str, Any]]:
    """把记录序列或 pandas 风格 DataFrame 转为独立字典列表。"""

    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        try:
            value = value.to_dict(orient="records")
        except TypeError:
            value = value.to_dict("records")
    try:
        records = list(value)
    except TypeError as exc:
        raise EvaluationError(f"{description} 必须是记录序列或 DataFrame") from exc
    if not records:
        raise EvaluationError(f"{description} 不能为空")
    result: List[Dict[str, Any]] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise EvaluationError(f"{description} 第 {index} 项不是映射记录")
        result.append(dict(row))
    return result


def validate_oof(
    rows_or_df: Any,
    manifest_records: Any,
    *,
    probability_columns: Optional[Sequence[str]] = None,
    probability_sum_tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    """校验内存中的 OOF，并返回按 manifest 顺序排列的记录副本。

    两个参数均可为 ``Sequence[Mapping]`` 或具有 ``to_dict(orient="records")``
    的 DataFrame。标准字段为 sample_id/fold/target/prediction/prob_0...prob_N；
    manifest 的真值也兼容 label/class_idx/y_true/true_label。严格检查 ID 唯一且
    完整覆盖、fold/真值一致、预测及概率有限、概率范围与概率和，并要求 prediction
    等于概率 argmax。该 helper 只校验内存，不产生任何文件。
    """

    rows = _records_list(rows_or_df, "OOF rows")
    manifest = _records_list(manifest_records, "manifest_records")
    tolerance = float(probability_sum_tolerance)
    headers = [str(key) for key in rows[0].keys()]
    if probability_columns is None:
        numeric_probability_columns: List[Tuple[int, str]] = []
        for header in headers:
            if header.casefold().startswith("prob_"):
                suffix = header[5:]
                if suffix.isdigit():
                    numeric_probability_columns.append((int(suffix), header))
        numeric_probability_columns.sort()
        if not numeric_probability_columns:
            raise EvaluationError("OOF rows 缺少 prob_0...prob_N 概率列")
        indices = [index for index, _ in numeric_probability_columns]
        if indices != list(range(len(indices))):
            raise EvaluationError(f"概率列索引必须从 0 连续排列，实际为 {indices!r}")
        probability_columns = [column for _, column in numeric_probability_columns]
    else:
        probability_columns = list(probability_columns)
        if not probability_columns or len(set(probability_columns)) != len(probability_columns):
            raise EvaluationError("probability_columns 不能为空或含重复列")

    def resolve_columns(record_headers: Sequence[str], source: str) -> Tuple[str, str, str]:
        sample_column = _find_column(record_headers, ("sample_id",))
        fold_column = _find_column(record_headers, _FOLD_COLUMNS)
        target_column = _find_column(record_headers, ("target", *_LABEL_COLUMNS))
        missing = [
            name
            for name, column in (
                ("sample_id", sample_column),
                ("fold", fold_column),
                ("target", target_column),
            )
            if column is None
        ]
        if missing:
            raise EvaluationError(f"{source} 缺少列: {missing!r}")
        return str(sample_column), str(fold_column), str(target_column)

    row_id_col, row_fold_col, row_target_col = resolve_columns(headers, "OOF rows")
    prediction_col = _find_column(headers, ("prediction", "predicted_label", "y_pred"))
    if prediction_col is None:
        raise EvaluationError("OOF rows 缺少 prediction 列")
    for column in probability_columns:
        if column not in headers:
            raise EvaluationError(f"OOF rows 缺少概率列 {column!r}")

    manifest_headers = [str(key) for key in manifest[0].keys()]
    manifest_id_col, manifest_fold_col, manifest_target_col = resolve_columns(
        manifest_headers, "manifest_records"
    )

    manifest_by_id: Dict[str, Tuple[int, str, Dict[str, Any]]] = {}
    manifest_order: List[str] = []
    for index, record in enumerate(manifest):
        sample_id = _require_finite_token(record.get(manifest_id_col), f"manifest 第 {index} 行 sample_id")
        if sample_id in manifest_by_id:
            raise EvaluationError(f"manifest sample_id 重复: {sample_id!r}")
        fold = _parse_fold(record.get(manifest_fold_col), f"manifest 第 {index} 行 fold")
        target = _label_key(record.get(manifest_target_col), f"manifest 第 {index} 行 target")
        manifest_by_id[sample_id] = (fold, target, record)
        manifest_order.append(sample_id)

    validated_by_id: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(rows):
        sample_id = _require_finite_token(record.get(row_id_col), f"OOF 第 {index} 行 sample_id")
        if sample_id in validated_by_id:
            raise EvaluationError(f"OOF sample_id 重复: {sample_id!r}")
        if sample_id not in manifest_by_id:
            raise EvaluationError(f"OOF 含 manifest 外 sample_id: {sample_id!r}")
        fold = _parse_fold(record.get(row_fold_col), f"OOF 第 {index} 行 fold")
        target = _label_key(record.get(row_target_col), f"OOF 第 {index} 行 target")
        expected_fold, expected_target, _ = manifest_by_id[sample_id]
        if fold != expected_fold:
            raise EvaluationError(
                f"sample_id {sample_id!r} 的 fold 不一致: {fold} != {expected_fold}"
            )
        if target != expected_target:
            raise EvaluationError(
                f"sample_id {sample_id!r} 的 target 不一致: {target!r} != {expected_target!r}"
            )
        probabilities = _validate_probability_vector(
            [record.get(column) for column in probability_columns],
            f"OOF 第 {index} 行",
            tolerance,
        )
        prediction = _parse_fold(record.get(prediction_col), f"OOF 第 {index} 行 prediction")
        if prediction < 0 or prediction >= len(probabilities):
            raise EvaluationError(f"OOF 第 {index} 行 prediction 越界: {prediction}")
        expected_prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        if prediction != expected_prediction:
            raise EvaluationError(
                f"OOF 第 {index} 行 prediction 与概率 argmax 不一致: "
                f"{prediction} != {expected_prediction}"
            )
        validated_by_id[sample_id] = dict(record)

    missing = set(manifest_by_id) - set(validated_by_id)
    if missing:
        raise EvaluationError(
            f"OOF 缺失 {len(missing)} 个 manifest 样本；示例: {sorted(missing)[:10]!r}"
        )
    return [validated_by_id[sample_id] for sample_id in manifest_order]


def _optional_values(values: Optional[Sequence[Any]], size: int, name: str) -> List[Any]:
    if values is None:
        return [None] * size
    result = list(values)
    if len(result) != size:
        raise EvaluationError(f"{name} 长度为 {len(result)}，应为 {size}")
    return result


def write_fold_oof_csv(
    output_path: PathLike,
    fold: Any,
    sample_ids: Sequence[Any],
    y_true: Sequence[Any],
    probabilities: Sequence[Sequence[Any]],
    class_names: Sequence[Any],
    *,
    image_paths: Optional[Sequence[Any]] = None,
    sha256s: Optional[Sequence[Any]] = None,
    group_ids: Optional[Sequence[Any]] = None,
) -> Path:
    """以原子替换方式写一个 fold 的标准 OOF CSV。

    ``y_true`` 可使用类别名或 class_names 中的零基索引。``probabilities`` 每行
    必须与 class_names 等长，且每个值都有限并位于 [0, 1]。本函数严格拒绝
    当前 fold 内的重复 sample_id。可选元数据序列若提供，长度必须一致。
    """

    fold_number = _parse_fold(fold)
    names = _coerce_class_names(class_names)
    ids = list(sample_ids)
    truths = list(y_true)
    probability_rows = [list(row) for row in probabilities]
    size = len(ids)
    if not size:
        raise EvaluationError("不能写入空的 fold OOF")
    if len(truths) != size or len(probability_rows) != size:
        raise EvaluationError(
            "sample_ids、y_true、probabilities 长度必须一致："
            f"{size}, {len(truths)}, {len(probability_rows)}"
        )
    paths = _optional_values(image_paths, size, "image_paths")
    hashes = _optional_values(sha256s, size, "sha256s")
    groups = _optional_values(group_ids, size, "group_ids")

    probability_headers = [f"prob::{quote(name, safe='')}" for name in names]
    fieldnames = [
        "sample_id",
        "image_path",
        "sha256",
        "group_id",
        "fold",
        "label",
        "class_name",
        "predicted_label",
        "predicted_class_name",
        *probability_headers,
    ]
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    for index in range(size):
        sample_key = make_sample_id(
            sample_id=ids[index], sha256=hashes[index], image_path=paths[index]
        )
        if sample_key in seen_ids:
            raise EvaluationError(
                f"fold {fold_number} 内 sample_id 重复: {sample_key!r}"
            )
        seen_ids.add(sample_key)
        truth = _truth_index(truths[index], names, f"第 {index} 行真值")
        raw_probabilities = probability_rows[index]
        if len(raw_probabilities) != len(names):
            raise EvaluationError(
                f"第 {index} 行概率数为 {len(raw_probabilities)}，应为 {len(names)}"
            )
        numeric_probabilities = _validate_probability_vector(
            raw_probabilities, f"第 {index} 行", 1e-6
        )

        row: Dict[str, Any] = {
            "sample_id": sample_key,
            "image_path": _as_text(paths[index]),
            "sha256": _as_text(hashes[index]).casefold(),
            "group_id": _as_text(groups[index]),
            "fold": fold_number,
            "label": truth,
            "class_name": names[truth],
            "predicted_label": max(
                range(len(numeric_probabilities)), key=numeric_probabilities.__getitem__
            ),
            "predicted_class_name": names[
                max(
                    range(len(numeric_probabilities)),
                    key=numeric_probabilities.__getitem__,
                )
            ],
        }
        row.update(dict(zip(probability_headers, numeric_probabilities)))
        rows.append(row)

    return _atomic_write_csv(output_path, fieldnames, rows)


def _label_key(value: Any, description: str) -> str:
    text = _require_finite_token(value, description)
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        raise EvaluationError(f"{description} 不是有限值: {text!r}")
    if number.is_integer():
        return str(int(number))
    return text


def _resolve_truth(
    row: Mapping[str, str],
    label_column: Optional[str],
    class_column: Optional[str],
    class_names: Sequence[str],
    label_to_class: Mapping[str, str],
    description: str,
) -> int:
    label_text = _value(row, label_column)
    class_text = _value(row, class_column)
    if not label_text and not class_text:
        raise EvaluationError(f"{description} 缺少真值（label/class_name）")

    class_index: Optional[int] = None
    if class_text:
        _require_finite_token(class_text, f"{description} class_name")
        if class_text not in class_names:
            raise EvaluationError(
                f"{description} class_name {class_text!r} 不在概率类别顺序 {list(class_names)!r} 中"
            )
        class_index = class_names.index(class_text)

    label_index: Optional[int] = None
    if label_text:
        key = _label_key(label_text, f"{description} label")
        if key in label_to_class:
            mapped_name = label_to_class[key]
            if mapped_name not in class_names:
                raise EvaluationError(
                    f"{description} label {label_text!r} 映射到未知类别 {mapped_name!r}"
                )
            label_index = class_names.index(mapped_name)
        else:
            label_index = _truth_index(label_text, class_names, f"{description} label")

    if class_index is not None and label_index is not None and class_index != label_index:
        raise EvaluationError(
            f"{description} 的 label 与 class_name 不一致: {label_text!r} / {class_text!r}"
        )
    return class_index if class_index is not None else int(label_index)


def _wilson_interval(correct: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        raise EvaluationError("无法对空数据计算置信区间")
    z = 1.959963984540054  # 标准正态 97.5% 分位数，对应双侧 95% CI
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def evaluate_oof(
    manifest_path: PathLike,
    oof_paths: Sequence[PathLike],
    output_dir: PathLike,
    *,
    class_names: Optional[Sequence[Any]] = None,
    check_group_leakage: bool = False,
    target_accuracy: float = 0.95,
    encoding: str = "utf-8-sig",
) -> Dict[str, Any]:
    """合并、严格校验 OOF，写评估产物并返回与 metrics.json 相同的字典。

    覆盖检查以 manifest 为全集，要求 manifest 和 OOF 的 sample_id 都唯一且 OOF
    恰好覆盖一次；当 sample_id 缺失时，自动依次以 sha256、image_path/path 生成。
    fold、真值、可用元数据必须匹配。``check_group_leakage=True`` 时同一非空
    group_id 只能属于一个 fold，且每个 manifest 样本必须有 group_id。
    """

    try:
        target = float(target_accuracy)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"target_accuracy 不是数值: {target_accuracy!r}") from exc
    if not math.isfinite(target) or target < 0.0 or target > 1.0:
        raise EvaluationError("target_accuracy 必须是 [0, 1] 内的有限值")
    if not oof_paths:
        raise EvaluationError("至少需要一个 OOF CSV")

    explicit_names = _coerce_class_names(class_names) if class_names is not None else None
    loaded_oof: List[Tuple[Path, List[str], List[Dict[str, str]], List[str], List[str]]] = []
    canonical_names: Optional[List[str]] = explicit_names
    for raw_path in oof_paths:
        path = Path(raw_path)
        headers, rows = _read_csv(path, encoding)
        probability_columns, inferred_names = _probability_columns(headers, explicit_names)
        if canonical_names is None:
            canonical_names = inferred_names
        if inferred_names != canonical_names:
            raise EvaluationError(
                f"OOF 类别概率顺序不一致: {path} 为 {inferred_names!r}，"
                f"期望 {canonical_names!r}"
            )
        loaded_oof.append((path, headers, rows, probability_columns, inferred_names))
    assert canonical_names is not None
    names = canonical_names

    manifest_headers, manifest_rows = _read_csv(manifest_path, encoding)
    manifest_id_col = _find_column(manifest_headers, _ID_COLUMNS)
    manifest_path_col = _find_column(manifest_headers, _PATH_COLUMNS)
    manifest_sha_col = _find_column(manifest_headers, _SHA_COLUMNS)
    manifest_group_col = _find_column(manifest_headers, _GROUP_COLUMNS)
    manifest_fold_col = _find_column(manifest_headers, _FOLD_COLUMNS)
    manifest_label_col = _find_column(manifest_headers, _LABEL_COLUMNS)
    manifest_class_col = _find_column(manifest_headers, _CLASS_COLUMNS)
    if manifest_fold_col is None:
        raise EvaluationError("manifest 缺少 fold 列")
    if manifest_label_col is None and manifest_class_col is None:
        raise EvaluationError("manifest 缺少真值列（label/class_idx/class_name）")
    if manifest_id_col is None and manifest_sha_col is None and manifest_path_col is None:
        raise EvaluationError("manifest 缺少 sample_id、sha256 或 image_path/path")
    # grouped manifest 绝不能靠调用者忘记一个开关而绕过泄漏检查。只要存在
    # group 列便强制完整、非空并检查同组仅属于一个 fold；旧的非 grouped
    # manifest 仍兼容，显式开关则继续要求该列存在。
    grouped_manifest = manifest_group_col is not None
    if check_group_leakage and not grouped_manifest:
        raise EvaluationError("启用 group 泄漏检查时 manifest 必须包含 group_id/group 列")
    enforce_group_check = bool(check_group_leakage or grouped_manifest)

    # 对常见 prob_0...prob_N，若未显式给类别名，则从 manifest 的
    # label/class_name 一一映射恢复类别顺序。没有该映射时数字本身就是类别名。
    if (
        explicit_names is None
        and len(names) > 0
        and all(name.isdigit() for name in names)
        and [int(name) for name in names] == list(range(len(names)))
        and manifest_label_col is not None
        and manifest_class_col is not None
    ):
        indexed_names: Dict[int, str] = {}
        for line_number, row in enumerate(manifest_rows, start=2):
            label_text = _label_key(
                _value(row, manifest_label_col), f"manifest:{line_number} label"
            )
            try:
                label_index = int(label_text)
            except ValueError:
                indexed_names = {}
                break
            class_text = _require_finite_token(
                _value(row, manifest_class_col), f"manifest:{line_number} class_name"
            )
            previous = indexed_names.get(label_index)
            if previous is not None and previous != class_text:
                raise EvaluationError(
                    f"manifest 标签 {label_index} 同时映射到 {previous!r} 和 {class_text!r}"
                )
            indexed_names[label_index] = class_text
        if set(indexed_names) == set(range(len(names))):
            inferred_manifest_names = [indexed_names[index] for index in range(len(names))]
            if len(set(inferred_manifest_names)) != len(inferred_manifest_names):
                raise EvaluationError("manifest 的 label/class_name 映射不是一一对应")
            names = inferred_manifest_names

    # 有 class_name 与 label 时先建立映射，兼容零基、一起基或字符串标签。
    label_to_class: Dict[str, str] = {}
    class_to_label: Dict[str, str] = {}
    if manifest_label_col is not None and manifest_class_col is not None:
        for line_number, row in enumerate(manifest_rows, start=2):
            label_text = _value(row, manifest_label_col)
            class_text = _value(row, manifest_class_col)
            if not label_text or not class_text:
                continue
            key = _label_key(label_text, f"manifest:{line_number} label")
            _require_finite_token(class_text, f"manifest:{line_number} class_name")
            previous_class = label_to_class.get(key)
            previous_label = class_to_label.get(class_text)
            if previous_class is not None and previous_class != class_text:
                raise EvaluationError(
                    f"manifest 标签 {label_text!r} 同时映射到 {previous_class!r} 和 {class_text!r}"
                )
            if previous_label is not None and previous_label != key:
                raise EvaluationError(
                    f"manifest 类别 {class_text!r} 同时映射到标签 {previous_label!r} 和 {key!r}"
                )
            label_to_class[key] = class_text
            class_to_label[class_text] = key

    manifest_by_id: Dict[str, Dict[str, Any]] = {}
    groups_to_folds: Dict[str, set] = defaultdict(set)
    for line_number, row in enumerate(manifest_rows, start=2):
        sample_key = make_sample_id(
            sample_id=_value(row, manifest_id_col),
            sha256=_value(row, manifest_sha_col),
            image_path=_value(row, manifest_path_col),
        )
        if sample_key in manifest_by_id:
            raise EvaluationError(
                f"manifest sample_id 重复: {sample_key!r}（第 {line_number} 行）"
            )
        fold = _parse_fold(_value(row, manifest_fold_col), f"manifest:{line_number} fold")
        truth = _resolve_truth(
            row,
            manifest_label_col,
            manifest_class_col,
            names,
            label_to_class,
            f"manifest:{line_number}",
        )
        group = _value(row, manifest_group_col)
        if enforce_group_check:
            _require_finite_token(group, f"manifest:{line_number} group_id")
            groups_to_folds[group].add(fold)
        manifest_by_id[sample_key] = {
            "sample_id": sample_key,
            "image_path": _value(row, manifest_path_col),
            "sha256": _value(row, manifest_sha_col).casefold(),
            "group_id": group,
            "fold": fold,
            "truth": truth,
        }

    if enforce_group_check:
        leaked = {group: sorted(folds) for group, folds in groups_to_folds.items() if len(folds) > 1}
        if leaked:
            preview = list(leaked.items())[:10]
            raise EvaluationError(
                f"检测到 group 跨 fold 泄漏，共 {len(leaked)} 个；示例: {preview!r}"
            )

    predictions_by_id: Dict[str, Dict[str, Any]] = {}
    for path, headers, rows, probability_columns, _ in loaded_oof:
        id_col = _find_column(headers, _ID_COLUMNS)
        image_col = _find_column(headers, _PATH_COLUMNS)
        sha_col = _find_column(headers, _SHA_COLUMNS)
        group_col = _find_column(headers, _GROUP_COLUMNS)
        fold_col = _find_column(headers, _FOLD_COLUMNS)
        label_col = _find_column(headers, _LABEL_COLUMNS)
        class_col = _find_column(headers, _CLASS_COLUMNS)
        if id_col is None and sha_col is None and image_col is None:
            raise EvaluationError(f"OOF 缺少 sample_id、sha256 或 image_path/path: {path}")
        if fold_col is None:
            raise EvaluationError(f"OOF 缺少 fold 列: {path}")
        if label_col is None and class_col is None:
            raise EvaluationError(f"OOF 缺少真值列: {path}")

        for line_number, row in enumerate(rows, start=2):
            location = f"{path}:{line_number}"
            sample_key = make_sample_id(
                sample_id=_value(row, id_col),
                sha256=_value(row, sha_col),
                image_path=_value(row, image_col),
            )
            if sample_key in predictions_by_id:
                previous = predictions_by_id[sample_key]["source"]
                raise EvaluationError(
                    f"OOF sample_id 重复: {sample_key!r}，位于 {previous} 和 {location}"
                )
            if sample_key not in manifest_by_id:
                raise EvaluationError(f"OOF 含 manifest 外样本 {sample_key!r}（{location}）")
            manifest_record = manifest_by_id[sample_key]
            fold = _parse_fold(_value(row, fold_col), f"{location} fold")
            if fold != manifest_record["fold"]:
                raise EvaluationError(
                    f"{location} fold 与 manifest 不一致: {fold} != {manifest_record['fold']}"
                )
            truth = _resolve_truth(
                row, label_col, class_col, names, label_to_class, location
            )
            if truth != manifest_record["truth"]:
                raise EvaluationError(
                    f"{location} 真值与 manifest 不一致: {names[truth]!r} != "
                    f"{names[manifest_record['truth']]!r}"
                )

            oof_group = _value(row, group_col)
            if oof_group and manifest_record["group_id"] and oof_group != manifest_record["group_id"]:
                raise EvaluationError(f"{location} group_id 与 manifest 不一致")
            oof_sha = _value(row, sha_col).casefold()
            if oof_sha and manifest_record["sha256"] and oof_sha != manifest_record["sha256"]:
                raise EvaluationError(f"{location} sha256 与 manifest 不一致")

            probabilities = _validate_probability_vector(
                [_value(row, column) for column in probability_columns],
                location,
                1e-6,
            )
            predicted = max(range(len(names)), key=probabilities.__getitem__)
            predictions_by_id[sample_key] = {
                **manifest_record,
                "probabilities": probabilities,
                "predicted": predicted,
                "confidence": probabilities[predicted],
                "source": location,
            }

    manifest_ids = set(manifest_by_id)
    predicted_ids = set(predictions_by_id)
    missing = manifest_ids - predicted_ids
    extra = predicted_ids - manifest_ids
    if missing or extra:
        missing_preview = sorted(missing)[:10]
        extra_preview = sorted(extra)[:10]
        raise EvaluationError(
            "OOF 未完整且唯一覆盖 manifest："
            f"缺失 {len(missing)} 个（示例 {missing_preview!r}），"
            f"额外 {len(extra)} 个（示例 {extra_preview!r}）"
        )

    # 按 fold、sample_id 排序，使所有输出可重现。
    records = sorted(predictions_by_id.values(), key=lambda item: (item["fold"], item["sample_id"]))
    class_count = len(names)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    fold_counts: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    log_loss_sum = 0.0
    brier_sum = 0.0
    # 仅为 log(0) 提供机器精度下界；概率本身及 Brier 不做裁剪。
    log_epsilon = sys.float_info.min
    for record in records:
        truth = record["truth"]
        predicted = record["predicted"]
        probabilities = record["probabilities"]
        confusion[truth][predicted] += 1
        fold_counts[record["fold"]][1] += 1
        log_loss_sum -= math.log(max(probabilities[truth], log_epsilon))
        brier_sum += math.fsum(
            (probability - (1.0 if index == truth else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
        if truth == predicted:
            correct += 1
            fold_counts[record["fold"]][0] += 1

    total = len(records)
    accuracy = correct / total
    multiclass_log_loss = log_loss_sum / total
    # 多分类 Brier score 采用每样本对全部类别的平方误差和（范围 [0, 2]）。
    multiclass_brier_score = brier_sum / total
    per_class: List[Dict[str, Any]] = []
    for index, class_name in enumerate(names):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(confusion[row][index] for row in range(class_count))
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro = {
        "precision": sum(item["precision"] for item in per_class) / class_count,
        "recall": sum(item["recall"] for item in per_class) / class_count,
        "f1": sum(item["f1"] for item in per_class) / class_count,
    }
    per_fold = []
    fold_accuracies = []
    for fold in sorted(fold_counts):
        fold_correct, fold_total = fold_counts[fold]
        fold_accuracy = fold_correct / fold_total
        fold_accuracies.append(fold_accuracy)
        per_fold.append(
            {
                "fold": fold,
                "total": fold_total,
                "correct": fold_correct,
                "accuracy": fold_accuracy,
            }
        )
    fold_variance = statistics.pvariance(fold_accuracies) if fold_accuracies else 0.0
    fold_sample_variance = statistics.variance(fold_accuracies) if len(fold_accuracies) > 1 else 0.0
    ci_low, ci_high = _wilson_interval(correct, total)

    metrics: Dict[str, Any] = {
        "schema_version": 1,
        "class_order": names,
        "total": total,
        "correct": correct,
        "pooled_accuracy": accuracy,
        "multiclass_log_loss": multiclass_log_loss,
        "multiclass_brier_score": multiclass_brier_score,
        "brier_definition": "mean per-sample sum over classes of (p_k - 1[y=k])^2",
        "target": {
            "name": "95% accuracy target" if target == 0.95 else "accuracy target",
            "accuracy": target,
            "met": accuracy >= target,
            "margin": accuracy - target,
        },
        # 扁平字段便于自动验收和下游脚本直接读取。
        "target_accuracy_95": target,
        "meets_target_95": accuracy >= target,
        "accuracy_ci_95": {
            "method": "Wilson score interval",
            "confidence_level": 0.95,
            "lower": ci_low,
            "upper": ci_high,
        },
        "macro": macro,
        "per_class": per_class,
        "per_fold": per_fold,
        "fold_accuracy_variance": fold_variance,
        "fold_accuracy_sample_variance": fold_sample_variance,
        "fold_variance_definition": "population variance across unweighted fold accuracies",
        "group_leakage_checked": enforce_group_check,
        "grouped_manifest": grouped_manifest,
        "provenance": {
            "manifest": {"path": str(Path(manifest_path).resolve()), "sha256": _sha256_file(manifest_path)},
            "oof_inputs": [
                {"path": str(Path(path).resolve()), "sha256": _sha256_file(path)}
                for path in oof_paths
            ],
            "class_order": names,
            "probability_column_order": [
                f"prob::{quote(name, safe='')}" for name in names
            ],
            "sample_id_precedence": ["sample_id", "sha256", "image_path"],
        },
    }

    destination = Path(output_dir)
    probability_headers = [f"prob::{quote(name, safe='')}" for name in names]
    merged_headers = [
        "sample_id",
        "image_path",
        "sha256",
        "group_id",
        "fold",
        "label",
        "class_name",
        "predicted_label",
        "predicted_class_name",
        "confidence",
        "correct",
        *probability_headers,
    ]

    def output_row(record: Mapping[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "sample_id": record["sample_id"],
            "image_path": record["image_path"],
            "sha256": record["sha256"],
            "group_id": record["group_id"],
            "fold": record["fold"],
            "label": record["truth"],
            "class_name": names[record["truth"]],
            "predicted_label": record["predicted"],
            "predicted_class_name": names[record["predicted"]],
            "confidence": record["confidence"],
            "correct": int(record["truth"] == record["predicted"]),
        }
        row.update(dict(zip(probability_headers, record["probabilities"])))
        return row

    merged_rows = [output_row(record) for record in records]
    error_rows = [row for row in merged_rows if row["correct"] == 0]
    confusion_headers = ["true_class", *names, "total"]
    confusion_rows = [
        {
            "true_class": names[index],
            **{names[column]: confusion[index][column] for column in range(class_count)},
            "total": sum(confusion[index]),
        }
        for index in range(class_count)
    ]
    fold_headers = ["fold", "total", "correct", "accuracy"]
    class_headers = [
        "class_index",
        "class_name",
        "support",
        "predicted_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
    ]

    # 所有验证和计算均在写文件之前完成；每个文件再通过同目录临时文件原子替换。
    _atomic_write_csv(destination / "oof_merged.csv", merged_headers, merged_rows)
    _atomic_write_csv(destination / "errors.csv", merged_headers, error_rows)
    _atomic_write_csv(destination / "confusion_matrix.csv", confusion_headers, confusion_rows)
    _atomic_write_csv(destination / "fold_metrics.csv", fold_headers, per_fold)
    _atomic_write_csv(destination / "per_class_metrics.csv", class_headers, per_class)
    _atomic_write_json(destination / "metrics.json", metrics)
    _atomic_write_json(destination / "provenance.json", metrics["provenance"])
    return metrics


# 语义清晰的兼容别名，方便不同训练脚本复用。
write_fold_oof = write_fold_oof_csv
merge_and_evaluate_oof = evaluate_oof


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="严格合并各 fold OOF CSV，并生成分类评估产物。"
    )
    parser.add_argument("--manifest", required=True, help="含 fold 和真值的 manifest CSV")
    oof_source = parser.add_mutually_exclusive_group(required=True)
    oof_source.add_argument(
        "--oof",
        nargs="+",
        metavar="CSV",
        help="一个或多个 fold OOF CSV（按给定顺序读取）",
    )
    oof_source.add_argument(
        "--oof-dir",
        help="包含各 fold OOF CSV 的目录；按文件名排序读取目录直属的 *.csv",
    )
    parser.add_argument("--output-dir", required=True, help="评估输出目录")
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="按概率列顺序重复指定类别名；通常可由 OOF 表头自动推断",
    )
    parser.add_argument(
        "--check-group-leakage",
        action="store_true",
        help="要求 group_id 完整，且同一 group 不跨 fold",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="目标准确率，默认 0.95",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="输入 CSV 编码，默认 utf-8-sig",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口；成功返回 0，输入校验失败返回 2。"""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.oof_dir is not None:
            oof_directory = Path(arguments.oof_dir)
            if not oof_directory.is_dir():
                raise EvaluationError(f"OOF 目录不存在或不是目录: {oof_directory}")
            oof_paths: Sequence[PathLike] = sorted(
                (path for path in oof_directory.iterdir() if path.is_file() and path.suffix.casefold() == ".csv"),
                key=lambda path: path.name.casefold(),
            )
            if not oof_paths:
                raise EvaluationError(f"OOF 目录中没有 CSV: {oof_directory}")
        else:
            oof_paths = arguments.oof
        metrics = evaluate_oof(
            manifest_path=arguments.manifest,
            oof_paths=oof_paths,
            output_dir=arguments.output_dir,
            class_names=arguments.class_names,
            check_group_leakage=arguments.check_group_leakage,
            target_accuracy=arguments.target_accuracy,
            encoding=arguments.encoding,
        )
    except (EvaluationError, OSError, csv.Error) as exc:
        print(f"评估失败: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "total": metrics["total"],
                "correct": metrics["correct"],
                "pooled_accuracy": metrics["pooled_accuracy"],
                "macro": metrics["macro"],
                "metrics_json": str(Path(arguments.output_dir) / "metrics.json"),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
