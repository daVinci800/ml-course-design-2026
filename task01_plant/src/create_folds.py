"""创建并验证可复现的交叉验证 fold manifest。

默认从项目根目录下的 ``data/train/<class>/<image>`` 读取图像，并写入
``data/folds_v2.csv``。模块导入时不会扫描数据、打印内容或写文件；所有操作都由
函数调用或 :func:`main` 触发。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = Path("data/train")
DEFAULT_OUTPUT = Path("data/folds_v2.csv")
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 2026
DEFAULT_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# 保留原项目使用的类别编号，便于其他模块显式导入；默认发现逻辑仍会按类别名排序，
# 因而当前五类数据会得到完全相同的编号。
CLASS_TO_IDX = {
    "Black-grass": 0,
    "Common wheat": 1,
    "Loose Silky-bent": 2,
    "Scentless Mayweed": 3,
    "Sugar beet": 4,
}

REQUIRED_FIELDS = (
    "sample_id",
    "path",
    "filename",
    "class_name",
    "class_idx",
    "sha256",
    "fold",
)
GROUP_FIELD = "group_id"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")


class ManifestError(ValueError):
    """表示数据或 manifest 不满足约束。"""


@dataclass(frozen=True)
class ManifestValidation:
    """manifest 验证结果及可复用的摘要信息。"""

    path: Path | None
    manifest_sha256: str | None
    sample_count: int
    class_count: int
    fold_count: int
    class_counts: dict[str, int]
    fold_counts: dict[int, int]
    grouped: bool


def resolve_project_path(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    """将路径按项目根目录解析，并拒绝项目目录之外的路径。"""

    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"路径必须位于项目根目录内: {candidate}") from exc
    return candidate


def project_relative_path(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> str:
    """返回跨平台稳定的项目根相对 POSIX 路径。"""

    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"样本路径不在项目根目录内: {candidate}") from exc
    return relative.as_posix()


def make_sample_id(relative_path: str) -> str:
    """由规范化的项目相对路径生成稳定的 64 位十六进制 sample_id。"""

    normalized = _validate_relative_manifest_path(relative_path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件内容 SHA-256。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: str | Path) -> None:
    """使用 Pillow 同时执行格式检查和完整像素解码。"""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            image.verify()
        # verify() 后必须重新打开；load() 确保像素数据确实能够完整解码。
        with Image.open(image_path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
        raise ManifestError(f"Pillow 无法完整解码图像: {image_path}: {exc}") from exc


def discover_samples(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    *,
    project_root: str | Path = PROJECT_ROOT,
    class_to_idx: Mapping[str, int] | None = None,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> list[dict[str, object]]:
    """发现、解码并哈希训练图像，返回路径排序稳定的样本记录。

    类别由 ``train_dir`` 的直接子目录表示，图像必须直接位于类别目录中。
    未提供 ``class_to_idx`` 时，类别名按 Unicode 字典序映射到连续编号。
    """

    root = Path(project_root).expanduser().resolve()
    directory = resolve_project_path(train_dir, root)
    if not directory.is_dir():
        raise ManifestError(f"训练目录不存在或不是目录: {directory}")

    normalized_extensions = _normalize_extensions(extensions)
    class_dirs = sorted(
        (entry for entry in directory.iterdir() if entry.is_dir()),
        key=lambda entry: entry.name,
    )
    if not class_dirs:
        raise ManifestError(f"训练目录中没有类别子目录: {directory}")

    discovered_names = [entry.name for entry in class_dirs]
    mapping = _class_mapping(discovered_names, class_to_idx)
    records: list[dict[str, object]] = []

    for class_dir in class_dirs:
        image_paths = sorted(
            (
                entry
                for entry in class_dir.iterdir()
                if entry.is_file() and entry.suffix.lower() in normalized_extensions
            ),
            key=lambda entry: entry.name,
        )
        if not image_paths:
            raise ManifestError(f"类别目录中没有支持的图像: {class_dir}")

        for image_path in image_paths:
            relative_path = project_relative_path(image_path, root)
            validate_image(image_path)
            records.append(
                {
                    "sample_id": make_sample_id(relative_path),
                    "path": relative_path,
                    "filename": image_path.name,
                    "class_name": class_dir.name,
                    "class_idx": mapping[class_dir.name],
                    "sha256": compute_file_sha256(image_path),
                }
            )

    records.sort(key=lambda row: str(row["path"]))
    _validate_unique_identity(records)
    return records


def load_group_mapping(
    mapping_path: str | Path,
    records: Sequence[Mapping[str, object]],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, str]:
    """读取可选的 CSV 分组映射，返回 ``sample_id -> group_id``。

    CSV 必须含 ``group_id``（也接受列名 ``group``），并且使用 ``sample_id``、
    ``path`` 或 ``filename`` 中恰好一个标识列。每个样本必须映射一次，不允许额外、
    重复或空的映射。仅在文件名全局唯一时才可使用 ``filename``。
    """

    path = resolve_project_path(mapping_path, project_root)
    if not path.is_file():
        raise ManifestError(f"group 映射文件不存在: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        group_columns = [name for name in (GROUP_FIELD, "group") if name in headers]
        id_columns = [name for name in ("sample_id", "path", "filename") if name in headers]
        if len(group_columns) != 1 or len(id_columns) != 1:
            raise ManifestError(
                "group CSV 必须含 group_id（或 group）以及 sample_id/path/filename "
                "中的恰好一个标识列"
            )
        group_column = group_columns[0]
        id_column = id_columns[0]
        rows = list(reader)

    lookup: dict[str, str] = {}
    ambiguous: set[str] = set()
    for record in records:
        key = str(record[id_column])
        if id_column == "path":
            key = _normalize_mapping_path(key, project_root)
        if key in lookup:
            ambiguous.add(key)
        else:
            lookup[key] = str(record["sample_id"])
    if ambiguous:
        example = sorted(ambiguous)[0]
        raise ManifestError(f"group 映射标识不唯一（请改用 sample_id 或 path）: {example}")

    result: dict[str, str] = {}
    seen_mapping_keys: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        raw_key = (row.get(id_column) or "").strip()
        group_id = (row.get(group_column) or "").strip()
        if not raw_key or not group_id:
            raise ManifestError(f"group CSV 第 {line_number} 行含空标识或空 group_id")
        key = _normalize_mapping_path(raw_key, project_root) if id_column == "path" else raw_key
        if key in seen_mapping_keys:
            raise ManifestError(f"group CSV 中标识重复: {raw_key}")
        seen_mapping_keys.add(key)
        if key not in lookup:
            raise ManifestError(f"group CSV 含未知样本: {raw_key}")
        result[lookup[key]] = group_id

    missing = sorted(set(lookup.values()) - set(result))
    if missing:
        raise ManifestError(f"group CSV 缺少 {len(missing)} 个样本的映射，例如: {missing[0]}")
    return result


def assign_folds(
    records: Sequence[Mapping[str, object]],
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    groups: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """稳定地分配 fold；有 groups 时使用 StratifiedGroupKFold。"""

    if n_splits < 2:
        raise ManifestError("n_splits 必须至少为 2")
    if not records:
        raise ManifestError("没有可划分的样本")

    ordered = [dict(record) for record in sorted(records, key=lambda row: str(row["path"]))]
    _validate_unique_identity(ordered)
    labels = [int(record["class_idx"]) for record in ordered]
    class_counts = Counter(labels)

    if groups is None:
        too_small = {label: count for label, count in class_counts.items() if count < n_splits}
        if too_small:
            raise ManifestError(f"每类样本数必须不少于 n_splits，当前不足: {too_small}")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = splitter.split(ordered, labels)
    else:
        sample_ids = {str(record["sample_id"]) for record in ordered}
        supplied_ids = {str(sample_id) for sample_id in groups}
        missing = sorted(sample_ids - supplied_ids)
        extra = sorted(supplied_ids - sample_ids)
        if missing or extra:
            raise ManifestError(
                f"group 映射与样本不一致: 缺少 {len(missing)} 个，多出 {len(extra)} 个"
            )
        group_values = [str(groups[str(record["sample_id"])]) for record in ordered]
        if any(not value for value in group_values):
            raise ManifestError("group_id 不允许为空")
        if len(set(group_values)) < n_splits:
            raise ManifestError("不同 group 的数量必须不少于 n_splits")
        groups_by_class: dict[int, set[str]] = defaultdict(set)
        for label, group_id in zip(labels, group_values):
            groups_by_class[label].add(group_id)
        insufficient = {
            label: len(values)
            for label, values in groups_by_class.items()
            if len(values) < n_splits
        }
        if insufficient:
            raise ManifestError(
                f"每个类别涉及的不同 group 数必须不少于 n_splits，当前不足: {insufficient}"
            )
        for record, group_id in zip(ordered, group_values):
            record[GROUP_FIELD] = group_id
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        split_iterator = splitter.split(ordered, labels, group_values)

    assigned: set[int] = set()
    for fold, (_, validation_indices) in enumerate(split_iterator):
        for index_value in validation_indices:
            index = int(index_value)
            if index in assigned:
                raise ManifestError(f"样本被重复分配到 fold: {ordered[index]['sample_id']}")
            ordered[index]["fold"] = fold
            assigned.add(index)

    if len(assigned) != len(ordered):
        raise ManifestError(f"有 {len(ordered) - len(assigned)} 个样本未获得 fold")
    return sorted(
        ordered,
        key=lambda row: (int(row["fold"]), int(row["class_idx"]), str(row["path"])),
    )


def validate_records(
    records: Sequence[Mapping[str, object]],
    *,
    project_root: str | Path = PROJECT_ROOT,
    n_splits: int | None = None,
    expected_samples: int | None = None,
    expected_classes: int | None = None,
    verify_files: bool = True,
) -> ManifestValidation:
    """验证内存中的 manifest 记录；可选择重新解码图像并核对内容哈希。"""

    if not records:
        raise ManifestError("manifest 不能为空")
    root = Path(project_root).expanduser().resolve()
    normalized: list[dict[str, object]] = []

    for row_number, raw_record in enumerate(records, start=2):
        missing_fields = [field for field in REQUIRED_FIELDS if field not in raw_record]
        if missing_fields:
            raise ManifestError(f"第 {row_number} 行缺少字段: {', '.join(missing_fields)}")
        try:
            class_idx = int(raw_record["class_idx"])
            fold = int(raw_record["fold"])
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"第 {row_number} 行 class_idx 或 fold 不是整数") from exc

        relative_path = _validate_relative_manifest_path(str(raw_record["path"]))
        sample_id = str(raw_record["sample_id"])
        content_sha256 = str(raw_record["sha256"])
        filename = str(raw_record["filename"])
        class_name = str(raw_record["class_name"])
        if not filename or not class_name:
            raise ManifestError(f"第 {row_number} 行 filename 或 class_name 为空")
        if filename != PurePosixPath(relative_path).name:
            raise ManifestError(f"第 {row_number} 行 filename 与 path 不一致")
        if PurePosixPath(relative_path).parent.name != class_name:
            raise ManifestError(f"第 {row_number} 行 class_name 与 path 的父目录不一致")
        expected_sample_id = make_sample_id(relative_path)
        if sample_id != expected_sample_id:
            raise ManifestError(f"第 {row_number} 行 sample_id 与稳定路径 ID 不一致")
        if not _SHA256_RE.fullmatch(content_sha256):
            raise ManifestError(f"第 {row_number} 行 sha256 不是小写 64 位十六进制")
        if class_idx < 0 or fold < 0:
            raise ManifestError(f"第 {row_number} 行 class_idx 和 fold 必须为非负整数")

        image_path = resolve_project_path(relative_path, root)
        if not image_path.is_file():
            raise ManifestError(f"第 {row_number} 行图像不存在: {image_path}")
        if verify_files:
            validate_image(image_path)
            actual_sha256 = compute_file_sha256(image_path)
            if actual_sha256 != content_sha256:
                raise ManifestError(f"第 {row_number} 行图像 SHA-256 不匹配: {relative_path}")

        row: dict[str, object] = {
            "sample_id": sample_id,
            "path": relative_path,
            "filename": filename,
            "class_name": class_name,
            "class_idx": class_idx,
            "sha256": content_sha256,
            "fold": fold,
        }
        if GROUP_FIELD in raw_record:
            group_id = str(raw_record[GROUP_FIELD])
            if not group_id:
                raise ManifestError(f"第 {row_number} 行 group_id 为空")
            row[GROUP_FIELD] = group_id
        normalized.append(row)

    _validate_unique_identity(normalized)
    if expected_samples is not None and len(normalized) != expected_samples:
        raise ManifestError(f"样本数应为 {expected_samples}，实际为 {len(normalized)}")

    name_to_idx: dict[str, int] = {}
    idx_to_name: dict[int, str] = {}
    for row in normalized:
        name = str(row["class_name"])
        index = int(row["class_idx"])
        if name in name_to_idx and name_to_idx[name] != index:
            raise ManifestError(f"类别 {name} 对应多个 class_idx")
        if index in idx_to_name and idx_to_name[index] != name:
            raise ManifestError(f"class_idx {index} 对应多个类别")
        name_to_idx[name] = index
        idx_to_name[index] = name
    expected_indices = set(range(len(name_to_idx)))
    if set(idx_to_name) != expected_indices:
        raise ManifestError("class_idx 必须从 0 开始连续编号")
    if expected_classes is not None and len(name_to_idx) != expected_classes:
        raise ManifestError(f"类别数应为 {expected_classes}，实际为 {len(name_to_idx)}")

    observed_folds = {int(row["fold"]) for row in normalized}
    inferred_n_splits = max(observed_folds) + 1
    target_n_splits = inferred_n_splits if n_splits is None else n_splits
    if target_n_splits < 2:
        raise ManifestError("fold 数必须至少为 2")
    if observed_folds != set(range(target_n_splits)):
        raise ManifestError(
            f"fold 必须从 0 到 {target_n_splits - 1} 且每折非空，实际为 {sorted(observed_folds)}"
        )

    grouped_flags = [GROUP_FIELD in row for row in normalized]
    if any(grouped_flags) and not all(grouped_flags):
        raise ManifestError("group_id 列必须为所有样本提供值，或完全不存在")
    grouped = all(grouped_flags)
    if grouped:
        group_folds: dict[str, set[int]] = defaultdict(set)
        for row in normalized:
            group_folds[str(row[GROUP_FIELD])].add(int(row["fold"]))
        leaking = sorted(group for group, folds in group_folds.items() if len(folds) != 1)
        if leaking:
            raise ManifestError(f"group 跨 fold 泄漏，例如: {leaking[0]}")
    else:
        # StratifiedKFold 对每个类别给出的各折数量最多相差 1。
        class_fold_counts = Counter(
            (int(row["class_idx"]), int(row["fold"])) for row in normalized
        )
        for class_index in expected_indices:
            counts = [class_fold_counts[(class_index, fold)] for fold in range(target_n_splits)]
            if max(counts) - min(counts) > 1:
                raise ManifestError(f"类别 {idx_to_name[class_index]} 的 fold 分布不是分层划分")

    class_counts_counter = Counter(str(row["class_name"]) for row in normalized)
    fold_counts_counter = Counter(int(row["fold"]) for row in normalized)
    return ManifestValidation(
        path=None,
        manifest_sha256=None,
        sample_count=len(normalized),
        class_count=len(name_to_idx),
        fold_count=target_n_splits,
        class_counts=dict(sorted(class_counts_counter.items())),
        fold_counts=dict(sorted(fold_counts_counter.items())),
        grouped=grouped,
    )


def manifest_sha256(path: str | Path) -> str:
    """计算 manifest 文件原始字节的 SHA-256。"""

    return compute_file_sha256(path)


def write_manifest_atomic(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> str:
    """以原子、排他方式发布 CSV，绝不覆盖任何已存在的路径。

    临时文件与目标文件位于同一目录，并通过硬链接进行原子发布；若目标在检查后由
    其他进程创建，链接操作仍会安全失败而不会覆盖它。
    """

    if not records:
        raise ManifestError("不能写入空 manifest")
    output = resolve_project_path(output_path, project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"拒绝覆盖已存在的输出: {output}")

    grouped_flags = [GROUP_FIELD in record for record in records]
    if any(grouped_flags) and not all(grouped_flags):
        raise ManifestError("group_id 必须在所有记录中存在，或在所有记录中都不存在")
    fieldnames = list(REQUIRED_FIELDS)
    if all(grouped_flags):
        fieldnames.append(GROUP_FIELD)

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow({field: record[field] for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(temporary_path, output)
        except FileExistsError as exc:
            raise FileExistsError(f"拒绝覆盖并发创建的输出: {output}") from exc
        except OSError as exc:
            raise OSError(
                f"无法原子且排他地发布 manifest（目标文件未被覆盖）: {output}: {exc}"
            ) from exc
        return manifest_sha256(output)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def load_manifest(path: str | Path) -> list[dict[str, object]]:
    """读取 v2 manifest，并进行表头和基本值类型检查。"""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest 不存在: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        allowed_headers = [list(REQUIRED_FIELDS), [*REQUIRED_FIELDS, GROUP_FIELD]]
        if headers not in allowed_headers:
            raise ManifestError(
                f"manifest 表头应为 {allowed_headers[0]}，可选末列 {GROUP_FIELD}；实际为 {headers}"
            )
        records: list[dict[str, object]] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ManifestError(f"manifest 第 {line_number} 行列数多于表头")
            if any(row[field] is None for field in headers):
                raise ManifestError(f"manifest 第 {line_number} 行列数少于表头")
            record: dict[str, object] = dict(row)
            try:
                record["class_idx"] = int(str(record["class_idx"]))
                record["fold"] = int(str(record["fold"]))
            except ValueError as exc:
                raise ManifestError(f"manifest 第 {line_number} 行整数列无效") from exc
            records.append(record)
    return records


def validate_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    n_splits: int | None = None,
    expected_samples: int | None = None,
    expected_classes: int | None = None,
    expected_manifest_sha256: str | None = None,
    verify_files: bool = True,
) -> ManifestValidation:
    """读取并完整验证现有 manifest，返回其文件哈希和摘要。"""

    path = resolve_project_path(manifest_path, project_root)
    digest = manifest_sha256(path)
    if expected_manifest_sha256 is not None:
        expected = expected_manifest_sha256.strip().lower()
        if not _SHA256_RE.fullmatch(expected):
            raise ManifestError("期望的 manifest SHA-256 格式无效")
        if digest != expected:
            raise ManifestError(f"manifest SHA-256 不匹配: 期望 {expected}，实际 {digest}")

    result = validate_records(
        load_manifest(path),
        project_root=project_root,
        n_splits=n_splits,
        expected_samples=expected_samples,
        expected_classes=expected_classes,
        verify_files=verify_files,
    )
    return ManifestValidation(
        path=path,
        manifest_sha256=digest,
        sample_count=result.sample_count,
        class_count=result.class_count,
        fold_count=result.fold_count,
        class_counts=result.class_counts,
        fold_counts=result.fold_counts,
        grouped=result.grouped,
    )


def build_manifest(
    train_dir: Path,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    groups_path: Path | None = None,
) -> list[dict[str, object]]:
    """构建但不写入 manifest，供 dataset/train 或测试代码复用。

    ``train_dir`` 与 ``groups_path`` 可以是项目根相对路径，也可以是项目内绝对路径。
    本函数没有写文件或打印等副作用；默认使用稳定的 ``StratifiedKFold``，提供
    ``groups_path`` 时改用 ``StratifiedGroupKFold``。
    """

    samples = discover_samples(train_dir, project_root=PROJECT_ROOT)
    groups = None
    if groups_path is not None:
        groups = load_group_mapping(groups_path, samples, project_root=PROJECT_ROOT)
    records = assign_folds(samples, n_splits=n_splits, seed=seed, groups=groups)
    validate_records(
        records,
        project_root=PROJECT_ROOT,
        n_splits=n_splits,
        verify_files=True,
    )
    return records


def write_manifest(
    records: Sequence[Mapping[str, object]],
    output_path: Path,
    overwrite: bool = False,
) -> str:
    """原子写入新 manifest 并返回文件 SHA-256。

    ``overwrite`` 仅为兼容显式调用的公开参数；为遵守数据保护约束，即使传入
    ``True`` 也会拒绝覆盖，而不会删除或替换任何已有文件。
    """

    if overwrite:
        raise ManifestError("安全策略禁止覆盖已有输出；请改用一个尚不存在的新路径")
    return write_manifest_atomic(records, output_path, project_root=PROJECT_ROOT)


def create_folds_manifest(
    *,
    project_root: str | Path = PROJECT_ROOT,
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    group_mapping_path: str | Path | None = None,
    class_to_idx: Mapping[str, int] | None = None,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    expected_samples: int | None = None,
    expected_classes: int | None = None,
) -> ManifestValidation:
    """发现数据、分 fold、完整验证并原子写入一个新 manifest。"""

    root = Path(project_root).expanduser().resolve()
    output = resolve_project_path(output_path, root)
    # 在昂贵的图像验证前快速失败；最终发布仍由 write_manifest_atomic 排他保护。
    if os.path.lexists(output):
        raise FileExistsError(f"拒绝覆盖已存在的输出: {output}")

    samples = discover_samples(
        train_dir,
        project_root=root,
        class_to_idx=class_to_idx,
        extensions=extensions,
    )
    groups = None
    if group_mapping_path is not None:
        groups = load_group_mapping(group_mapping_path, samples, project_root=root)
    records = assign_folds(samples, n_splits=n_splits, seed=seed, groups=groups)

    # 再次读取文件，防止图像在发现、哈希与正式写入之间发生改变。
    summary = validate_records(
        records,
        project_root=root,
        n_splits=n_splits,
        expected_samples=expected_samples,
        expected_classes=expected_classes,
        verify_files=True,
    )
    digest = write_manifest_atomic(records, output, project_root=root)
    return ManifestValidation(
        path=output,
        manifest_sha256=digest,
        sample_count=summary.sample_count,
        class_count=summary.class_count,
        fold_count=summary.fold_count,
        class_counts=summary.class_counts,
        fold_counts=summary.fold_counts,
        grouped=summary.grouped,
    )


def _normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    normalized: set[str] = set()
    for extension in extensions:
        value = str(extension).strip().lower()
        if not value:
            continue
        normalized.add(value if value.startswith(".") else f".{value}")
    if not normalized:
        raise ManifestError("至少需要一个图像扩展名")
    return frozenset(normalized)


def _class_mapping(
    discovered_names: Sequence[str], class_to_idx: Mapping[str, int] | None
) -> dict[str, int]:
    if class_to_idx is None:
        return {name: index for index, name in enumerate(sorted(discovered_names))}
    mapping = {str(name): int(index) for name, index in class_to_idx.items()}
    if set(mapping) != set(discovered_names):
        missing = sorted(set(discovered_names) - set(mapping))
        extra = sorted(set(mapping) - set(discovered_names))
        raise ManifestError(f"类别映射与目录不一致: 缺少 {missing}，多出 {extra}")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ManifestError("class_to_idx 的编号必须从 0 开始连续且不重复")
    return mapping


def _validate_relative_manifest_path(value: str) -> str:
    if not value or "\\" in value or _WINDOWS_ABSOLUTE_RE.match(value):
        raise ManifestError(f"path 必须是规范的项目相对 POSIX 路径: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ManifestError(f"path 必须是规范的项目相对 POSIX 路径: {value!r}")
    normalized = pure.as_posix()
    if normalized != value:
        raise ManifestError(f"path 不是规范形式: {value!r}")
    return normalized


def _normalize_mapping_path(value: str, project_root: str | Path) -> str:
    text = value.strip().replace("\\", "/")
    if _WINDOWS_ABSOLUTE_RE.match(text) or Path(value).is_absolute():
        return project_relative_path(Path(value), project_root)
    return _validate_relative_manifest_path(text)


def _validate_unique_identity(records: Sequence[Mapping[str, object]]) -> None:
    fields = ["sample_id", "path"]
    if records and all("sha256" in record for record in records):
        # 内容相同但文件名不同的图片跨 fold 会形成确定的数据泄漏。当前数据不含
        # 这类副本；未来若确实需要保留，应先显式去重，而不是静默分折。
        fields.append("sha256")
    for field in fields:
        values = [str(record[field]) for record in records]
        if len(values) != len(set(values)):
            duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
            raise ManifestError(f"{field} 存在重复，例如: {duplicates[0]}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="创建可复现、带图像 SHA-256 的分层交叉验证 manifest。",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="项目根目录（默认由本文件位置确定）",
    )
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="项目根相对训练目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="新 manifest 的项目根相对路径")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS, help="fold 数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="稳定随机种子")
    parser.add_argument(
        "--groups",
        "--group-map",
        dest="group_map",
        help="可选的项目根相对 group CSV；启用 StratifiedGroupKFold",
    )
    parser.add_argument(
        "--validate-existing",
        metavar="MANIFEST",
        help="只验证一个现有 v2 manifest，不创建新文件",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        help="验证现有 manifest 时可选的预期 SHA-256",
    )
    parser.add_argument("--expected-samples", type=int, help="可选的预期样本数（当前数据可设 500）")
    parser.add_argument("--expected-classes", type=int, help="可选的预期类别数（当前数据可设 5）")
    parser.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help="允许的图像扩展名；可重复指定（默认支持常用 Pillow 格式）",
    )
    return parser


def _print_summary(result: ManifestValidation) -> None:
    action = "验证通过" if result.path is not None else "检查通过"
    print(f"{action}: {result.path}")
    print(
        f"样本={result.sample_count}, 类别={result.class_count}, "
        f"fold={result.fold_count}, grouped={result.grouped}"
    )
    print(f"各 fold 样本数: {result.fold_counts}")
    print(f"各类别样本数: {result.class_counts}")
    if result.manifest_sha256 is not None:
        print(f"manifest_sha256={result.manifest_sha256}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；成功返回 0，输入/数据错误返回 2。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_existing:
            if args.group_map:
                raise ManifestError("--validate-existing 不能与 --group-map 同时使用")
            result = validate_manifest(
                args.validate_existing,
                project_root=args.project_root,
                n_splits=args.n_splits,
                expected_samples=args.expected_samples,
                expected_classes=args.expected_classes,
                expected_manifest_sha256=args.expected_manifest_sha256,
                verify_files=True,
            )
        else:
            if args.expected_manifest_sha256:
                raise ManifestError("--expected-manifest-sha256 仅可与 --validate-existing 一起使用")
            result = create_folds_manifest(
                project_root=args.project_root,
                train_dir=args.train_dir,
                output_path=args.output,
                n_splits=args.n_splits,
                seed=args.seed,
                group_mapping_path=args.group_map,
                extensions=args.extensions or DEFAULT_IMAGE_EXTENSIONS,
                expected_samples=args.expected_samples,
                expected_classes=args.expected_classes,
            )
    except (ManifestError, FileExistsError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
