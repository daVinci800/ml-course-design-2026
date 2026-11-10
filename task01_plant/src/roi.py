"""固定、无标签泄漏的绿色植株 ROI 与坐标缓存。

算法只使用图像像素和 :class:`ROIConfig` 中的固定阈值：ExG 与 HSV 联合
分割，随后用 ``scipy.ndimage`` 做 opening/closing，并对保留连通域取 union
bbox。bbox 扩为 1.4 倍方框；方框越界时先整体平移，图像本身不足以容纳方框
时才以整幅图像的 RGB 均值填充。任何不可靠的分割都安全回退到完整图像。

模块导入没有文件系统副作用。缓存只保存坐标和校验元数据，不生成 ROI 图片。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage


ALGORITHM_NAME = "fixed-green-exg-hsv-roi"
ALGORITHM_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
DEFAULT_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")


class ROIError(ValueError):
    """ROI 输入、配置或缓存不满足契约。"""


class ROICacheError(ROIError):
    """坐标缓存不完整、过期或被篡改。"""


@dataclass(frozen=True)
class ROIConfig:
    """固定绿色 ROI 参数；所有阈值均与类别标签无关。

    RGB/HSV 值在 ``[0, 1]`` 上计算，色相使用角度。默认算法要求像素同时满足
    ExG 和 HSV 条件，避免仅靠某一个宽松规则引入背景。所有面积阈值都在形态学
    处理后的原图分辨率上应用。
    """

    exg_threshold: float = 0.08
    hue_min_degrees: float = 30.0
    hue_max_degrees: float = 170.0
    saturation_min: float = 0.15
    value_min: float = 0.10
    opening_size: int = 3
    closing_size: int = 5
    min_component_pixels: int = 16
    min_foreground_pixels: int = 32
    min_foreground_fraction: float = 0.001
    max_foreground_fraction: float = 0.98
    min_bbox_fraction: float = 0.01
    min_crop_fraction: float = 0.01
    max_crop_fraction: float = 0.90
    expansion_factor: float = 1.4

    def __post_init__(self) -> None:
        integer_values = {
            "opening_size": self.opening_size,
            "closing_size": self.closing_size,
            "min_component_pixels": self.min_component_pixels,
            "min_foreground_pixels": self.min_foreground_pixels,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ROIError(f"{name} 必须是整数")
        finite_values = {
            "exg_threshold": self.exg_threshold,
            "hue_min_degrees": self.hue_min_degrees,
            "hue_max_degrees": self.hue_max_degrees,
            "saturation_min": self.saturation_min,
            "value_min": self.value_min,
            "min_foreground_fraction": self.min_foreground_fraction,
            "max_foreground_fraction": self.max_foreground_fraction,
            "min_bbox_fraction": self.min_bbox_fraction,
            "min_crop_fraction": self.min_crop_fraction,
            "max_crop_fraction": self.max_crop_fraction,
            "expansion_factor": self.expansion_factor,
        }
        for name, value in finite_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ROIError(f"{name} 必须是数值")
            if not math.isfinite(float(value)):
                raise ROIError(f"{name} 必须是有限数")
        if not 0.0 <= self.hue_min_degrees <= 360.0:
            raise ROIError("hue_min_degrees 必须位于 [0, 360]")
        if not 0.0 <= self.hue_max_degrees <= 360.0:
            raise ROIError("hue_max_degrees 必须位于 [0, 360]")
        if not 0.0 <= self.saturation_min <= 1.0:
            raise ROIError("saturation_min 必须位于 [0, 1]")
        if not 0.0 <= self.value_min <= 1.0:
            raise ROIError("value_min 必须位于 [0, 1]")
        if self.opening_size != 3 or self.closing_size != 5:
            raise ROIError("锁定算法要求 3x3 opening 和 5x5 closing")
        if self.min_component_pixels < 1 or self.min_foreground_pixels < 1:
            raise ROIError("像素数阈值必须至少为 1")
        if not 0.0 <= self.min_foreground_fraction < self.max_foreground_fraction <= 1.0:
            raise ROIError("前景比例必须满足 0 <= min < max <= 1")
        if not 0.0 <= self.min_bbox_fraction <= self.min_crop_fraction < self.max_crop_fraction <= 1.0:
            raise ROIError("bbox/crop 比例阈值无效")
        if self.expansion_factor < 1.0:
            raise ROIError("expansion_factor 必须至少为 1")

    def to_dict(self) -> dict[str, Any]:
        """返回可稳定序列化的参数字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ROIConfig":
        """从严格字段集合构造配置，拒绝拼错或未来未知字段。"""

        if not isinstance(values, Mapping):
            raise ROIError("ROI config 必须是 JSON object")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ROIError(f"ROI config 含未知字段: {', '.join(sorted(unknown))}")
        try:
            return cls(**dict(values))
        except TypeError as exc:
            raise ROIError(f"ROI config 字段类型或结构无效: {exc}") from exc


@dataclass(frozen=True)
class ROIResult:
    """一个图像的可复现 ROI 坐标结果。

    ``box`` 和 ``union_bbox`` 均采用 Pillow 风格的半开坐标
    ``(left, top, right, bottom)``。``box`` 始终为方形，允许在图像边界之外；
    ``padding`` 为应用该方框时需做的 ``(left, top, right, bottom)`` 均值填充。
    ``union_bbox`` 是所有通过过滤的绿色连通域的联合框，fallback 时为 ``None``。
    """

    box: tuple[int, int, int, int]
    union_bbox: tuple[int, int, int, int] | None
    image_size: tuple[int, int]
    padding: tuple[int, int, int, int]
    fill_rgb: tuple[int, int, int]
    foreground_pixels: int
    foreground_fraction: float
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        left, top, right, bottom = self.box
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ROIError("ROIResult.image_size 必须为正")
        if right <= left or bottom <= top or right - left != bottom - top:
            raise ROIError("ROIResult.box 必须是非空方框")
        if any(value < 0 for value in self.padding):
            raise ROIError("ROIResult.padding 不能为负数")
        if len(self.fill_rgb) != 3 or any(not 0 <= value <= 255 for value in self.fill_rgb):
            raise ROIError("ROIResult.fill_rgb 必须是三个 [0, 255] 整数")
        if self.foreground_pixels < 0:
            raise ROIError("ROIResult.foreground_pixels 不能为负数")
        if not math.isfinite(self.foreground_fraction) or not 0.0 <= self.foreground_fraction <= 1.0:
            raise ROIError("ROIResult.foreground_fraction 必须位于 [0, 1]")
        expected_padding = (
            max(0, -left),
            max(0, -top),
            max(0, right - width),
            max(0, bottom - height),
        )
        if self.padding != expected_padding:
            raise ROIError("ROIResult.padding 与 box/image_size 不一致")
        if self.union_bbox is not None:
            x0, y0, x1, y1 = self.union_bbox
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ROIError("ROIResult.union_bbox 超出图像或为空")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """``box`` 的兼容别名。"""

        return self.box

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None

    @property
    def side(self) -> int:
        return self.box[2] - self.box[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "union_bbox": None if self.union_bbox is None else list(self.union_bbox),
            "image_size": list(self.image_size),
            "padding": list(self.padding),
            "fill_rgb": list(self.fill_rgb),
            "foreground_pixels": self.foreground_pixels,
            "foreground_fraction": self.foreground_fraction,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ROIResult":
        if not isinstance(value, Mapping):
            raise ROICacheError("ROI result 必须是 JSON object")
        required = {
            "box",
            "union_bbox",
            "image_size",
            "padding",
            "fill_rgb",
            "foreground_pixels",
            "foreground_fraction",
            "fallback_reason",
        }
        if set(value) != required:
            raise ROICacheError(
                f"ROI result 字段不匹配: 缺少 {sorted(required - set(value))}，"
                f"多出 {sorted(set(value) - required)}"
            )
        try:
            union = value["union_bbox"]
            return cls(
                box=_integer_tuple(value["box"], 4, "box"),
                union_bbox=None if union is None else _integer_tuple(union, 4, "union_bbox"),
                image_size=_integer_tuple(value["image_size"], 2, "image_size"),
                padding=_integer_tuple(value["padding"], 4, "padding"),
                fill_rgb=_integer_tuple(value["fill_rgb"], 3, "fill_rgb"),
                foreground_pixels=_strict_int(value["foreground_pixels"], "foreground_pixels"),
                foreground_fraction=float(value["foreground_fraction"]),
                fallback_reason=(
                    None if value["fallback_reason"] is None else str(value["fallback_reason"])
                ),
            )
        except (TypeError, ValueError, ROIError) as exc:
            raise ROICacheError(f"ROI result 无效: {exc}") from exc


DEFAULT_CONFIG = ROIConfig()


def roi_config_from_mapping(value: Mapping[str, Any]) -> ROIConfig:
    """解析 canonical ``{algorithm, algorithm_version, config}`` ROI recipe。"""
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "algorithm_version", "config"}:
        raise ROIError("ROI recipe 必须且只能含 algorithm/algorithm_version/config")
    if value["algorithm"] != ALGORITHM_NAME or value["algorithm_version"] != ALGORITHM_VERSION:
        raise ROIError("ROI recipe 算法身份或版本不匹配")
    return ROIConfig.from_dict(value["config"])


def config_sha256(config: ROIConfig = DEFAULT_CONFIG) -> str:
    """返回配置规范 JSON 的 SHA-256。"""

    return hashlib.sha256(_canonical_json(config.to_dict()).encode("utf-8")).hexdigest()


def compute_roi(image: Image.Image | np.ndarray, config: ROIConfig = DEFAULT_CONFIG) -> ROIResult:
    """计算固定绿色 ROI 坐标，不读取或接受任何类别信息。"""

    rgb = _as_rgb_array(image)
    height, width, _ = rgb.shape
    fill_rgb = tuple(int(value) for value in np.rint(rgb.mean(axis=(0, 1))).clip(0, 255))
    rgb_unit = rgb.astype(np.float32) / np.float32(255.0)
    red, green, blue = np.moveaxis(rgb_unit, -1, 0)
    exg = np.float32(2.0) * green - red - blue
    hue, saturation, value = _rgb_to_hsv(rgb_unit)

    if config.hue_min_degrees <= config.hue_max_degrees:
        hue_ok = (hue >= config.hue_min_degrees) & (hue <= config.hue_max_degrees)
    else:  # 允许跨越 0° 的色相区间，尽管默认绿色区间不需要。
        hue_ok = (hue >= config.hue_min_degrees) | (hue <= config.hue_max_degrees)
    candidate = (
        (exg >= config.exg_threshold)
        & hue_ok
        & (saturation >= config.saturation_min)
        & (value >= config.value_min)
    )
    candidate_count = int(np.count_nonzero(candidate))
    total_pixels = width * height
    candidate_fraction = candidate_count / total_pixels

    if candidate_count == 0:
        return _fallback_result(width, height, fill_rgb, 0, 0.0, "no_green_pixels")
    if candidate_count < config.min_foreground_pixels:
        return _fallback_result(
            width, height, fill_rgb, candidate_count, candidate_fraction,
            "insufficient_green_pixels",
        )
    if candidate_fraction < config.min_foreground_fraction:
        return _fallback_result(
            width, height, fill_rgb, candidate_count, candidate_fraction,
            "green_fraction_too_small",
        )
    if candidate_fraction > config.max_foreground_fraction:
        return _fallback_result(
            width, height, fill_rgb, candidate_count, candidate_fraction,
            "green_fraction_too_large",
        )

    mask = ndimage.binary_opening(
        candidate,
        structure=np.ones((config.opening_size, config.opening_size), dtype=bool),
        iterations=1,
        border_value=0,
    )
    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((config.closing_size, config.closing_size), dtype=bool),
        iterations=1,
        border_value=0,
    )
    morph_count = int(np.count_nonzero(mask))
    if morph_count == 0:
        return _fallback_result(
            width, height, fill_rgb, 0, 0.0, "morphology_removed_all_pixels"
        )

    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    component_sizes = np.bincount(labels.ravel())
    # 固定像素阈值在 50×50 与 2800×2500 图上含义完全不同。按锁定方案
    # 使用 max(16px, image_area×0.0001)，避免高分辨率砂砾中的零散绿色噪声
    # 把所有连通域 union bbox 拉到整幅图。
    minimum_component_area = max(
        config.min_component_pixels,
        int(math.ceil(total_pixels * 0.0001)),
    )
    keep_labels = np.flatnonzero(component_sizes >= minimum_component_area)
    keep_labels = keep_labels[keep_labels != 0]
    if component_count == 0 or keep_labels.size == 0:
        return _fallback_result(
            width, height, fill_rgb, morph_count, morph_count / total_pixels,
            "no_component_large_enough",
        )
    kept = np.isin(labels, keep_labels)
    foreground_pixels = int(np.count_nonzero(kept))
    foreground_fraction = foreground_pixels / total_pixels
    if foreground_pixels < config.min_foreground_pixels:
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "insufficient_component_pixels",
        )
    if foreground_fraction < config.min_foreground_fraction:
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "component_fraction_too_small",
        )
    if foreground_fraction > config.max_foreground_fraction:
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "component_fraction_too_large",
        )

    ys, xs = np.nonzero(kept)
    if xs.size == 0 or ys.size == 0:  # 防御 scipy/NumPy 异常结果。
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "empty_union_bbox",
        )
    union_bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    bbox_width = union_bbox[2] - union_bbox[0]
    bbox_height = union_bbox[3] - union_bbox[1]
    bbox_fraction = (bbox_width * bbox_height) / total_pixels
    touches_horizontal_pair = union_bbox[0] == 0 and union_bbox[2] == width
    touches_vertical_pair = union_bbox[1] == 0 and union_bbox[3] == height
    if bbox_fraction < config.min_bbox_fraction:
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "bbox_fraction_too_small",
        )
    if touches_horizontal_pair or touches_vertical_pair:
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "component_touches_opposite_edges",
        )
    side = max(1, int(math.ceil(config.expansion_factor * max(bbox_width, bbox_height))))
    crop_fraction = (side * side) / total_pixels
    if crop_fraction < config.min_crop_fraction:
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "crop_fraction_too_small",
        )
    if crop_fraction > config.max_crop_fraction:
        return _fallback_result(
            width, height, fill_rgb, foreground_pixels, foreground_fraction,
            "crop_fraction_too_large",
        )
    center_x = (union_bbox[0] + union_bbox[2]) / 2.0
    center_y = (union_bbox[1] + union_bbox[3]) / 2.0
    box = _square_window(center_x, center_y, side, width, height)
    return ROIResult(
        box=box,
        union_bbox=union_bbox,
        image_size=(width, height),
        padding=_padding_for_box(box, width, height),
        fill_rgb=fill_rgb,
        foreground_pixels=foreground_pixels,
        foreground_fraction=foreground_fraction,
        fallback_reason=None,
    )


def crop_with_roi(image: Image.Image | np.ndarray, result: ROIResult) -> Image.Image:
    """按已算好的坐标裁剪，越界区域使用整图均值颜色填充。"""

    rgb = _as_rgb_array(image)
    height, width, _ = rgb.shape
    if result.image_size != (width, height):
        raise ROIError(
            f"ROI 尺寸 {result.image_size} 与图像尺寸 {(width, height)} 不一致"
        )
    left, top, right, bottom = result.box
    side = right - left
    output = Image.new("RGB", (side, side), result.fill_rgb)
    source_box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
    if source_box[0] < source_box[2] and source_box[1] < source_box[3]:
        source = Image.fromarray(rgb, mode="RGB").crop(source_box)
        output.paste(source, (source_box[0] - left, source_box[1] - top))
    return output


def extract_roi(
    image: Image.Image | np.ndarray,
    config: ROIConfig = DEFAULT_CONFIG,
) -> tuple[Image.Image, ROIResult]:
    """计算并应用 ROI，返回 ``(RGB 方形图像, ROIResult)``。"""

    result = compute_roi(image, config)
    if result.is_fallback:
        return Image.fromarray(_as_rgb_array(image), mode="RGB"), result
    return crop_with_roi(image, result), result


def build_manifest_roi_cache(
    manifest_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
    config: ROIConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """从训练 CSV manifest 构建内存坐标缓存，不写图或缓存文件。

    manifest 至少必须有 ``path``，推荐含稳定且唯一的 ``sample_id`` 与图像
    ``sha256``。若有 manifest SHA，则生成时同时核对实际文件，防止基于已变化图像
    生成坐标。类别、fold 等字段从不参与算法，也不会写入缓存。
    """

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise ROICacheError(f"训练 manifest 不存在: {manifest}")
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    rows = _read_manifest_rows(manifest)
    samples: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        raw_path = (row.get("path") or "").strip()
        if not raw_path:
            raise ROICacheError(f"manifest 第 {line_number} 行 path 为空")
        cache_path = _normalize_manifest_cache_path(raw_path, root)
        image_path = _resolve_manifest_image_path(raw_path, root)
        sample_id = (row.get("sample_id") or cache_path).strip()
        if not sample_id:
            raise ROICacheError(f"manifest 第 {line_number} 行 sample_id 为空")
        if sample_id in seen_ids:
            raise ROICacheError(f"manifest sample_id 重复: {sample_id}")
        if cache_path in seen_paths:
            raise ROICacheError(f"manifest path 重复: {cache_path}")
        seen_ids.add(sample_id)
        seen_paths.add(cache_path)
        expected_sha = (row.get("sha256") or "").strip().lower()
        if expected_sha and not _SHA256_RE.fullmatch(expected_sha):
            raise ROICacheError(f"manifest 第 {line_number} 行 sha256 格式无效")
        samples.append(
            {
                "key": sample_id,
                "path": cache_path,
                "absolute_path": str(image_path),
                "expected_sha256": expected_sha,
            }
        )
    if not samples:
        raise ROICacheError("训练 manifest 不能为空")

    entries = _build_entries(samples, config)
    return _make_cache(
        source={
            "kind": "training_manifest",
            "manifest_sha256": file_sha256(manifest),
        },
        config=config,
        entries=entries,
    )


def build_test_roi_cache(
    test_dir: str | os.PathLike[str],
    *,
    config: ROIConfig = DEFAULT_CONFIG,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> dict[str, Any]:
    """递归发现测试目录图像并构建内存坐标缓存；key 是相对 POSIX 路径。"""

    directory = Path(test_dir).expanduser().resolve()
    paths = _discover_test_images(directory, extensions)
    samples = [
        {
            "key": path.relative_to(directory).as_posix(),
            "path": path.relative_to(directory).as_posix(),
            "absolute_path": str(path),
            "expected_sha256": "",
        }
        for path in paths
    ]
    entries = _build_entries(samples, config)
    return _make_cache(
        source={"kind": "test_directory"},
        config=config,
        entries=entries,
    )


def write_roi_cache(
    cache: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> str:
    """原子写入坐标 JSON 并返回文件 SHA-256；默认拒绝覆盖。"""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已存在的 ROI cache: {output}")
    payload = json.dumps(cache, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output)
            temporary = None
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as exc:
                raise FileExistsError(f"拒绝覆盖并发创建的 ROI cache: {output}") from exc
            except OSError as exc:
                raise OSError(f"无法原子且排他地发布 ROI cache: {output}: {exc}") from exc
        return file_sha256(output)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def generate_manifest_roi_cache(
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
    config: ROIConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
) -> str:
    """构建并写入训练 manifest ROI cache，返回 JSON 文件 SHA-256。"""

    cache = build_manifest_roi_cache(
        manifest_path,
        project_root=project_root,
        config=config,
    )
    return write_roi_cache(cache, output_path, overwrite=overwrite)


def generate_test_roi_cache(
    test_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    config: ROIConfig = DEFAULT_CONFIG,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    overwrite: bool = False,
) -> str:
    """构建并写入测试目录 ROI cache，返回 JSON 文件 SHA-256。"""

    cache = build_test_roi_cache(test_dir, config=config, extensions=extensions)
    return write_roi_cache(cache, output_path, overwrite=overwrite)


def load_manifest_roi_cache(
    cache_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
    config: ROIConfig = DEFAULT_CONFIG,
    verify_files: bool = True,
) -> dict[str, ROIResult]:
    """加载训练缓存并严格验证全覆盖、SHA、尺寸、算法版本与 config hash。"""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise ROICacheError(f"训练 manifest 不存在: {manifest}")
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    rows = _read_manifest_rows(manifest)
    expected: dict[str, dict[str, str]] = {}
    seen_paths: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        raw_path = (row.get("path") or "").strip()
        if not raw_path:
            raise ROICacheError(f"manifest 第 {line_number} 行 path 为空")
        cache_relative = _normalize_manifest_cache_path(raw_path, root)
        key = (row.get("sample_id") or cache_relative).strip()
        if not key or key in expected:
            raise ROICacheError(f"manifest sample_id 为空或重复: {key!r}")
        if cache_relative in seen_paths:
            raise ROICacheError(f"manifest path 重复: {cache_relative}")
        seen_paths.add(cache_relative)
        expected_sha = (row.get("sha256") or "").strip().lower()
        if expected_sha and not _SHA256_RE.fullmatch(expected_sha):
            raise ROICacheError(f"manifest 第 {line_number} 行 sha256 格式无效")
        expected[key] = {
            "path": cache_relative,
            "absolute_path": str(_resolve_manifest_image_path(raw_path, root)),
            "expected_sha256": expected_sha,
        }
    cache = _read_and_validate_cache_header(cache_path, config, "training_manifest")
    source = cache["source"]
    actual_manifest_sha = file_sha256(manifest)
    if source.get("manifest_sha256") != actual_manifest_sha:
        raise ROICacheError("训练 manifest SHA-256 与 cache 不匹配")
    return _validate_entries(cache, expected, verify_files=verify_files)


def load_test_roi_cache(
    cache_path: str | os.PathLike[str],
    test_dir: str | os.PathLike[str],
    *,
    config: ROIConfig = DEFAULT_CONFIG,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    verify_files: bool = True,
) -> dict[str, ROIResult]:
    """加载测试缓存并严格验证目录全覆盖、SHA、尺寸、算法版本与 config hash。"""

    directory = Path(test_dir).expanduser().resolve()
    paths = _discover_test_images(directory, extensions)
    expected = {
        path.relative_to(directory).as_posix(): {
            "path": path.relative_to(directory).as_posix(),
            "absolute_path": str(path),
            "expected_sha256": "",
        }
        for path in paths
    }
    cache = _read_and_validate_cache_header(cache_path, config, "test_directory")
    return _validate_entries(cache, expected, verify_files=verify_files)


def load_roi_cache(
    cache_path: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    test_dir: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
    config: ROIConfig = DEFAULT_CONFIG,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    verify_files: bool = True,
) -> dict[str, ROIResult]:
    """统一严格加载入口；必须且只能提供 manifest_path 或 test_dir。"""

    if (manifest_path is None) == (test_dir is None):
        raise ROICacheError("必须且只能提供 manifest_path 或 test_dir")
    if manifest_path is not None:
        return load_manifest_roi_cache(
            cache_path,
            manifest_path,
            project_root=project_root,
            config=config,
            verify_files=verify_files,
        )
    return load_test_roi_cache(
        cache_path,
        test_dir,  # type: ignore[arg-type]
        config=config,
        extensions=extensions,
        verify_files=verify_files,
    )


def file_sha256(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""

    if chunk_size <= 0:
        raise ROIError("chunk_size 必须大于 0")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        if image.width <= 0 or image.height <= 0:
            raise ROIError("图像尺寸必须为正")
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ROIError("NumPy 图像必须为 H x W x 3/4")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ROIError("图像尺寸必须为正")
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ROIError("NumPy 图像必须含有限数值")
        if np.issubdtype(array.dtype, np.floating):
            minimum = float(array.min())
            maximum = float(array.max())
            if minimum < 0.0 or maximum > 255.0:
                raise ROIError("浮点 NumPy 图像值必须位于 [0, 255]")
            # [0, 1] 浮点图按常用图像约定缩放，否则视作 [0, 255]。
            if maximum <= 1.0:
                array = array * 255.0
        elif float(array.min()) < 0.0 or float(array.max()) > 255.0:
            raise ROIError("整数 NumPy 图像值必须位于 [0, 255]")
        array = np.rint(array[..., :3]).astype(np.uint8)
    else:
        raise TypeError("image 必须是 PIL.Image.Image 或 NumPy ndarray")
    return np.ascontiguousarray(array)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red, green, blue = np.moveaxis(rgb, -1, 0)
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    delta = maximum - minimum
    hue = np.zeros(maximum.shape, dtype=np.float32)
    nonzero = delta > 0
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue *= 60.0
    saturation = np.zeros_like(maximum, dtype=np.float32)
    positive = maximum > 0
    saturation[positive] = delta[positive] / maximum[positive]
    return hue, saturation, maximum


def _disk_structure(radius: int) -> np.ndarray:
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def _axis_window(center: float, side: int, length: int) -> int:
    # floor(x + .5) 实现稳定的 half-up，而不是 Python 的 bankers rounding。
    start = math.floor(center - side / 2.0 + 0.5)
    if side <= length:
        return min(max(start, 0), length - side)
    # 无法完全容纳时仍先尽量平移：保证 start 位于 [length-side, 0]，
    # 仅剩不可避免的越界交给 mean fill。
    return min(max(start, length - side), 0)


def _square_window(
    center_x: float,
    center_y: float,
    side: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = _axis_window(center_x, side, width)
    top = _axis_window(center_y, side, height)
    return left, top, left + side, top + side


def _padding_for_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return max(0, -left), max(0, -top), max(0, right - width), max(0, bottom - height)


def _fallback_result(
    width: int,
    height: int,
    fill_rgb: tuple[int, int, int],
    foreground_pixels: int,
    foreground_fraction: float,
    reason: str,
) -> ROIResult:
    side = max(width, height)
    box = _square_window(width / 2.0, height / 2.0, side, width, height)
    return ROIResult(
        box=box,
        union_bbox=None,
        image_size=(width, height),
        padding=_padding_for_box(box, width, height),
        fill_rgb=fill_rgb,
        foreground_pixels=foreground_pixels,
        foreground_fraction=foreground_fraction,
        fallback_reason=reason,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if "path" not in headers:
                raise ROICacheError("训练 manifest 缺少 path 列")
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(row.get(header) is None for header in headers):
                    raise ROICacheError(f"manifest 第 {line_number} 行列数与表头不一致")
                rows.append({key: str(value) for key, value in row.items()})
            return rows
    except UnicodeError as exc:
        raise ROICacheError(f"manifest 不是有效 UTF-8: {path}") from exc


def _resolve_manifest_image_path(raw_path: str, root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if _WINDOWS_ABSOLUTE_RE.match(raw_path) and os.name != "nt":
        raise ROICacheError(f"当前平台无法解析 Windows 绝对路径: {raw_path}")
    if not candidate.is_absolute():
        normalized = PurePosixPath(raw_path.replace("\\", "/"))
        if normalized.is_absolute() or any(part in ("", ".", "..") for part in normalized.parts):
            raise ROICacheError(f"manifest path 不是安全相对路径: {raw_path!r}")
        candidate = root.joinpath(*normalized.parts)
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        # 明确给出的绝对路径可位于项目根外；相对路径已在上面安全拼接到 root。
        if not Path(raw_path).expanduser().is_absolute():
            raise ROICacheError(f"manifest 相对路径逃逸项目根目录: {raw_path!r}")
    if not candidate.is_file():
        raise ROICacheError(f"manifest 图像不存在: {candidate}")
    return candidate


def _normalize_manifest_cache_path(raw_path: str, root: Path) -> str:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute() or _WINDOWS_ABSOLUTE_RE.match(raw_path):
        resolved = candidate.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            # 旧 manifest 可能是另一台 Windows 机器的绝对路径。其路径仍可作为
            # 稳定 cache 元数据，但实际读取必须由当前路径能够解析。
            return raw_path.replace("\\", "/")
    normalized = PurePosixPath(raw_path.replace("\\", "/"))
    if normalized.is_absolute() or any(part in ("", ".", "..") for part in normalized.parts):
        raise ROICacheError(f"manifest path 不是安全相对路径: {raw_path!r}")
    return normalized.as_posix()


def _normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    result: set[str] = set()
    for extension in extensions:
        value = str(extension).strip().lower()
        if value:
            result.add(value if value.startswith(".") else f".{value}")
    if not result:
        raise ROICacheError("至少需要一个图像扩展名")
    return frozenset(result)


def _discover_test_images(directory: Path, extensions: Iterable[str]) -> list[Path]:
    if not directory.is_dir():
        raise ROICacheError(f"测试目录不存在或不是目录: {directory}")
    allowed = _normalize_extensions(extensions)
    paths = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not paths:
        raise ROICacheError(f"测试目录中没有支持的图像: {directory}")
    return paths


def _build_entries(
    samples: Sequence[Mapping[str, str]], config: ROIConfig
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = sample["key"]
        path = Path(sample["absolute_path"])
        actual_sha = file_sha256(path)
        expected_sha = sample.get("expected_sha256", "")
        if expected_sha and actual_sha != expected_sha:
            raise ROICacheError(f"图像 SHA-256 与 manifest 不匹配: {sample['path']}")
        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB")
        except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
            raise ROICacheError(f"无法完整解码图像 {path}: {exc}") from exc
        result = compute_roi(image, config)
        if key in entries:
            raise ROICacheError(f"cache key 重复: {key}")
        entries[key] = {
            "path": sample["path"],
            "sha256": actual_sha,
            "width": image.width,
            "height": image.height,
            "roi": result.to_dict(),
        }
    return dict(sorted(entries.items()))


def _make_cache(
    *,
    source: Mapping[str, Any],
    config: ROIConfig,
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "algorithm": ALGORITHM_NAME,
        "algorithm_version": ALGORITHM_VERSION,
        "config": config.to_dict(),
        "config_sha256": config_sha256(config),
        "source": dict(source),
        "entry_count": len(entries),
        "entries": dict(entries),
    }


def _read_and_validate_cache_header(
    cache_path: str | os.PathLike[str],
    config: ROIConfig,
    expected_source_kind: str,
) -> dict[str, Any]:
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise ROICacheError(f"ROI cache 不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ROICacheError(f"无法读取 ROI cache {path}: {exc}") from exc
    if not isinstance(cache, dict):
        raise ROICacheError("ROI cache 顶层必须是 JSON object")
    required = {
        "schema_version",
        "algorithm",
        "algorithm_version",
        "config",
        "config_sha256",
        "source",
        "entry_count",
        "entries",
    }
    if set(cache) != required:
        raise ROICacheError(
            f"ROI cache 顶层字段不匹配: 缺少 {sorted(required - set(cache))}，"
            f"多出 {sorted(set(cache) - required)}"
        )
    if cache["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ROICacheError(
            f"cache schema 版本不匹配: {cache['schema_version']} != {CACHE_SCHEMA_VERSION}"
        )
    if cache["algorithm"] != ALGORITHM_NAME:
        raise ROICacheError("ROI 算法名称不匹配")
    if cache["algorithm_version"] != ALGORITHM_VERSION:
        raise ROICacheError(
            f"ROI 算法版本不匹配: {cache['algorithm_version']} != {ALGORITHM_VERSION}"
        )
    expected_config_hash = config_sha256(config)
    if cache["config_sha256"] != expected_config_hash:
        raise ROICacheError("ROI config hash 不匹配")
    try:
        cached_config = ROIConfig.from_dict(cache["config"])
    except ROIError as exc:
        raise ROICacheError(f"cache 内 config 无效: {exc}") from exc
    if config_sha256(cached_config) != cache["config_sha256"]:
        raise ROICacheError("cache 内 config 内容与 config_sha256 不匹配")
    if cached_config != config:
        raise ROICacheError("cache 内 ROI config 与请求配置不一致")
    if not isinstance(cache["source"], dict):
        raise ROICacheError("cache source 必须是 JSON object")
    if cache["source"].get("kind") != expected_source_kind:
        raise ROICacheError(
            f"cache 来源类型不是 {expected_source_kind}: {cache['source'].get('kind')}"
        )
    if not isinstance(cache["entries"], dict):
        raise ROICacheError("cache entries 必须是 JSON object")
    entry_count = _strict_int(cache["entry_count"], "entry_count")
    if entry_count != len(cache["entries"]):
        raise ROICacheError("cache entry_count 与 entries 数量不一致")
    return cache


def _validate_entries(
    cache: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, str]],
    *,
    verify_files: bool,
) -> dict[str, ROIResult]:
    entries = cache["entries"]
    cached_keys = set(entries)
    expected_keys = set(expected)
    missing = sorted(expected_keys - cached_keys)
    extra = sorted(cached_keys - expected_keys)
    if missing or extra:
        examples: list[str] = []
        if missing:
            examples.append(f"缺少 {len(missing)} 个（如 {missing[0]}）")
        if extra:
            examples.append(f"多出 {len(extra)} 个（如 {extra[0]}）")
        raise ROICacheError("ROI cache 未严格全覆盖: " + "，".join(examples))

    results: dict[str, ROIResult] = {}
    required_entry_fields = {"path", "sha256", "width", "height", "roi"}
    for key in sorted(expected):
        entry = entries[key]
        if not isinstance(entry, dict) or set(entry) != required_entry_fields:
            raise ROICacheError(f"cache entry 字段无效: {key}")
        specification = expected[key]
        if entry["path"] != specification["path"]:
            raise ROICacheError(f"cache path 不匹配: {key}")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ROICacheError(f"cache SHA-256 格式无效: {key}")
        expected_digest = specification.get("expected_sha256", "")
        if expected_digest and digest != expected_digest:
            raise ROICacheError(f"cache SHA-256 与 manifest 不匹配: {key}")
        width = _strict_int(entry["width"], "width")
        height = _strict_int(entry["height"], "height")
        if width <= 0 or height <= 0:
            raise ROICacheError(f"cache 图像尺寸无效: {key}")
        result = ROIResult.from_dict(entry["roi"])
        if result.image_size != (width, height):
            raise ROICacheError(f"cache ROI 尺寸与 entry 尺寸不一致: {key}")

        if verify_files:
            image_path = Path(specification["absolute_path"])
            if not image_path.is_file():
                raise ROICacheError(f"图像不存在: {image_path}")
            if file_sha256(image_path) != digest:
                raise ROICacheError(f"图像 SHA-256 与 cache 不匹配: {key}")
            try:
                with Image.open(image_path) as image:
                    image.load()
                    actual_size = image.size
            except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
                raise ROICacheError(f"无法完整解码图像 {image_path}: {exc}") from exc
            if actual_size != (width, height):
                raise ROICacheError(
                    f"图像尺寸与 cache 不匹配: {key}: {actual_size} != {(width, height)}"
                )
        results[key] = result
    return results


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ROICacheError(f"{name} 必须是整数")
    return value


def _integer_tuple(value: Any, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ROICacheError(f"{name} 必须含 {length} 个整数")
    return tuple(_strict_int(item, name) for item in value)


def _load_config_json(path: str | os.PathLike[str] | None) -> ROIConfig:
    if path is None:
        return DEFAULT_CONFIG
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ROIError(f"无法读取 config JSON {config_path}: {exc}") from exc
    if isinstance(value, Mapping) and "roi" in value:
        value = value["roi"]
    if isinstance(value, Mapping) and set(value) == {"algorithm", "algorithm_version", "config"}:
        return roi_config_from_mapping(value)
    return ROIConfig.from_dict(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成或严格校验固定绿色 ROI 坐标 JSON cache（不生成图像）。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="从训练 CSV manifest 生成 cache")
    manifest.add_argument("--manifest", required=True, help="训练 manifest CSV")
    manifest.add_argument("--output", required=True, help="输出坐标 JSON")
    manifest.add_argument("--project-root", help="manifest 相对图像路径的根目录")
    manifest.add_argument("--config", help="可选 ROIConfig JSON")
    manifest.add_argument("--overwrite", action="store_true", help="显式允许原子替换已有 cache")

    test = subparsers.add_parser("test-dir", help="从测试图像目录生成 cache")
    test.add_argument("--test-dir", required=True, help="测试图像目录（递归发现）")
    test.add_argument("--output", required=True, help="输出坐标 JSON")
    test.add_argument("--config", help="可选 ROIConfig JSON")
    test.add_argument("--extension", action="append", dest="extensions", help="允许扩展名，可重复")
    test.add_argument("--overwrite", action="store_true", help="显式允许原子替换已有 cache")

    validate_manifest = subparsers.add_parser("validate-manifest", help="严格校验训练 cache")
    validate_manifest.add_argument("--cache", required=True, help="坐标 JSON")
    validate_manifest.add_argument("--manifest", required=True, help="训练 manifest CSV")
    validate_manifest.add_argument("--project-root", help="manifest 相对图像路径的根目录")
    validate_manifest.add_argument("--config", help="可选 ROIConfig JSON")

    validate_test = subparsers.add_parser("validate-test-dir", help="严格校验测试 cache")
    validate_test.add_argument("--cache", required=True, help="坐标 JSON")
    validate_test.add_argument("--test-dir", required=True, help="测试图像目录")
    validate_test.add_argument("--config", help="可选 ROIConfig JSON")
    validate_test.add_argument("--extension", action="append", dest="extensions", help="允许扩展名，可重复")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；成功为 0，数据/配置错误为 2。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config_json(args.config)
        if args.command == "manifest":
            digest = generate_manifest_roi_cache(
                args.manifest,
                args.output,
                project_root=args.project_root,
                config=config,
                overwrite=args.overwrite,
            )
            count = len(
                load_manifest_roi_cache(
                    args.output,
                    args.manifest,
                    project_root=args.project_root,
                    config=config,
                    verify_files=True,
                )
            )
            print(f"生成成功: {args.output}")
            print(f"样本={count}, cache_sha256={digest}, config_sha256={config_sha256(config)}")
        elif args.command == "test-dir":
            extensions = args.extensions or DEFAULT_IMAGE_EXTENSIONS
            digest = generate_test_roi_cache(
                args.test_dir,
                args.output,
                config=config,
                extensions=extensions,
                overwrite=args.overwrite,
            )
            count = len(
                load_test_roi_cache(
                    args.output,
                    args.test_dir,
                    config=config,
                    extensions=extensions,
                    verify_files=True,
                )
            )
            print(f"生成成功: {args.output}")
            print(f"样本={count}, cache_sha256={digest}, config_sha256={config_sha256(config)}")
        elif args.command == "validate-manifest":
            results = load_manifest_roi_cache(
                args.cache,
                args.manifest,
                project_root=args.project_root,
                config=config,
                verify_files=True,
            )
            print(f"验证通过: {args.cache}")
            print(f"样本={len(results)}, config_sha256={config_sha256(config)}")
        else:
            extensions = args.extensions or DEFAULT_IMAGE_EXTENSIONS
            results = load_test_roi_cache(
                args.cache,
                args.test_dir,
                config=config,
                extensions=extensions,
                verify_files=True,
            )
            print(f"验证通过: {args.cache}")
            print(f"样本={len(results)}, config_sha256={config_sha256(config)}")
    except (ROIError, FileExistsError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "ALGORITHM_NAME",
    "ALGORITHM_VERSION",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CONFIG",
    "DEFAULT_IMAGE_EXTENSIONS",
    "ROICacheError",
    "ROIConfig",
    "ROIError",
    "ROIResult",
    "roi_config_from_mapping",
    "build_manifest_roi_cache",
    "build_test_roi_cache",
    "compute_roi",
    "config_sha256",
    "crop_with_roi",
    "extract_roi",
    "file_sha256",
    "generate_manifest_roi_cache",
    "generate_test_roi_cache",
    "load_manifest_roi_cache",
    "load_roi_cache",
    "load_test_roi_cache",
    "main",
    "write_roi_cache",
]


if __name__ == "__main__":
    raise SystemExit(main())
