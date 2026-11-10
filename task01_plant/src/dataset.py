"""Dataset, transforms, and reproducible data-loader helpers."""

from __future__ import annotations

import csv
import importlib
import inspect
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

try:  # InterpolationMode is unavailable in very old torchvision releases.
    from torchvision.transforms import InterpolationMode
except ImportError:  # pragma: no cover - compatibility path
    InterpolationMode = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_FOLDS_CSV = DEFAULT_DATA_ROOT / "folds_v2.csv"
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _resolve_record_path(row: Mapping[str, str], data_root: Path) -> Path:
    """Resolve a CSV image path against an explicit, relocatable data root."""
    raw_path = Path(row.get("path", ""))
    filename = row.get("filename") or raw_path.name
    class_name = row.get("class_name")

    # folds.csv may contain absolute paths from the machine that generated it.
    # Prefer its structured columns so moving the whole data directory still works.
    if filename and class_name:
        return data_root / "train" / class_name / filename

    if raw_path.is_absolute():
        if raw_path.exists():
            return raw_path
        parts = raw_path.parts
        if "train" in parts:
            return data_root.joinpath(*parts[parts.index("train") :])
        return raw_path

    parts = raw_path.parts
    if parts and parts[0].lower() == data_root.name.lower():
        parts = parts[1:]
    return data_root.joinpath(*parts)


def load_records(
    csv_path: str | os.PathLike[str] | None = None,
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> list[dict[str, Any]]:
    """Load fold records and resolve every image path from ``data_root``.

    ``sample_id`` is read from the CSV when present, otherwise ``filename`` (or
    the image filename) is used. Labels and folds are converted to integers.
    """
    root = Path(data_root).expanduser().resolve()
    folds_path = (
        Path(csv_path).expanduser()
        if csv_path is not None
        else root / DEFAULT_FOLDS_CSV.name
    )
    folds_path = folds_path.resolve()

    records: list[dict[str, Any]] = []
    with folds_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"path", "class_idx", "fold"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{folds_path} 缺少必需列: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            try:
                image_path = _resolve_record_path(row, root)
                label = int(row["class_idx"])
                fold = int(row["fold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{folds_path} 第 {line_number} 行数据无效") from exc

            sample_id = row.get("sample_id") or row.get("filename") or image_path.name
            record: dict[str, Any] = dict(row)
            record.update(
                path=str(image_path),
                class_idx=label,
                fold=fold,
                sample_id=str(sample_id),
            )
            records.append(record)

    return records


def split_by_fold(
    records: Sequence[Mapping[str, Any]], val_fold: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split records without mutating them, using ``val_fold`` for validation."""
    train_records: list[Mapping[str, Any]] = []
    val_records: list[Mapping[str, Any]] = []
    for record in records:
        (val_records if int(record["fold"]) == val_fold else train_records).append(record)
    return train_records, val_records


def _pil_interpolation(name: str) -> Any:
    resampling = getattr(Image, "Resampling", Image)
    modes = {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "lanczos": resampling.LANCZOS,
    }
    try:
        return modes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"不支持的插值方式: {name!r}") from exc


def resolve_interpolation(interpolation: Any = "bicubic") -> Any:
    """Accept a torchvision/Pillow interpolation value or a readable name."""
    if not isinstance(interpolation, str):
        return interpolation

    name = interpolation.lower()
    if InterpolationMode is not None:
        aliases = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
        }
        if hasattr(InterpolationMode, "LANCZOS"):
            aliases["lanczos"] = InterpolationMode.LANCZOS
        if name in aliases:
            return aliases[name]
    return _pil_interpolation(name)


class SquarePad:
    """Pad the shorter side symmetrically to make an image square."""

    def __init__(self, fill: int | tuple[int, int, int] = 0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        size = max(width, height)
        horizontal = size - width
        vertical = size - height
        padding = (
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        )
        return TF.pad(image, padding, fill=self.fill)


def build_transforms(
    image_size: int | tuple[int, int] = 224,
    *,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    interpolation: Any = "bicubic",
    pad_fill: int | tuple[int, int, int] | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build light train augmentation and deterministic validation transforms.

    Both pipelines preserve the complete object by square-padding then resizing;
    neither performs a crop.
    """
    interpolation_mode = resolve_interpolation(interpolation)
    if pad_fill is None:
        # 让 padding 在 Normalize 后接近 0，避免黑框成为额外的类别捷径。
        pad_fill = tuple(round(float(channel) * 255) for channel in mean)
    resize = lambda: transforms.Resize(image_size, interpolation=interpolation_mode)
    normalize = lambda: transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose(
        [
            SquarePad(fill=pad_fill),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                interpolation=interpolation_mode,
                fill=pad_fill,
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            resize(),
            transforms.ToTensor(),
            normalize(),
        ]
    )
    val_transform = transforms.Compose(
        [
            SquarePad(fill=pad_fill),
            resize(),
            transforms.ToTensor(),
            normalize(),
        ]
    )
    return train_transform, val_transform


def _size_pair(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size 必须为正数")
        return image_size, image_size
    if len(image_size) != 2 or min(int(value) for value in image_size) <= 0:
        raise ValueError("image_size 必须是正整数或两个正整数")
    return int(image_size[0]), int(image_size[1])


def _uniform(low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(torch.empty(1).uniform_(float(low), float(high)).item())


class PairedDualViewTransform:
    """Apply one sampled augmentation to a full image and its ROI.

    Training uses one shared D4 element, affine scale/translation, and one shared
    ColorJitter parameter set (including operation order).  It deliberately has
    no RandomResizedCrop, RandomErasing, Mixup, or CutMix.  Validation mode only
    pads, resizes, converts, and normalizes.  The result always has shape
    ``[2, C, H, W]`` in ``(full, roi)`` order.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        *,
        training: bool = True,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        interpolation: Any = "bicubic",
        pad_fill: int | tuple[int, int, int] | None = None,
        translate: float = 0.04,
        scale: tuple[float, float] = (0.95, 1.05),
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.0,
    ) -> None:
        self.image_size = _size_pair(image_size)
        self.training = bool(training)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean 和 std 必须各含三个通道值")
        if any(value <= 0 for value in self.std):
            raise ValueError("std 必须为正数")
        self.interpolation = resolve_interpolation(interpolation)
        self.pad_fill = (
            tuple(round(channel * 255) for channel in self.mean)
            if pad_fill is None
            else pad_fill
        )
        if not 0.0 <= float(translate) <= 1.0:
            raise ValueError("translate 必须位于 [0, 1]")
        if len(scale) != 2 or not 0 < float(scale[0]) <= float(scale[1]):
            raise ValueError("scale 必须是两个递增的正数")
        if min(brightness, contrast, saturation) < 0 or not 0 <= hue <= 0.5:
            raise ValueError("ColorJitter 参数无效")
        self.translate = float(translate)
        self.scale = (float(scale[0]), float(scale[1]))
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def _prepare(self, image: Image.Image) -> Image.Image:
        image = SquarePad(fill=self.pad_fill)(image)
        # Functional resize takes (height, width), matching tensor shape notation.
        return TF.resize(image, list(self.image_size), interpolation=self.interpolation)

    @staticmethod
    def _d4(image: Image.Image, rotation: int, reflect: bool) -> Image.Image:
        if reflect:
            image = TF.hflip(image)
        if rotation:
            image = TF.rotate(image, 90 * rotation)
        return image

    def _jitter_parameters(self) -> tuple[list[int], tuple[float, float, float, float]]:
        factors = (
            _uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness),
            _uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast),
            _uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation),
            _uniform(-self.hue, self.hue),
        )
        return torch.randperm(4).tolist(), factors

    @staticmethod
    def _jitter(
        image: Image.Image,
        order: Sequence[int],
        factors: tuple[float, float, float, float],
    ) -> Image.Image:
        operations = (
            TF.adjust_brightness,
            TF.adjust_contrast,
            TF.adjust_saturation,
            TF.adjust_hue,
        )
        for operation_index in order:
            image = operations[operation_index](image, factors[operation_index])
        return image

    def __call__(
        self,
        full_image: Image.Image | Sequence[Image.Image],
        roi_image: Image.Image | None = None,
    ) -> torch.Tensor:
        # Also accept transform((full, roi)), which makes the object convenient in
        # generic paired-data pipelines.
        if roi_image is None:
            if isinstance(full_image, Sequence) and len(full_image) == 2:
                full_image, roi_image = full_image
            else:
                raise TypeError("paired transform 需要 full_image 和 roi_image")
        if not isinstance(full_image, Image.Image) or not isinstance(roi_image, Image.Image):
            raise TypeError("paired transform 的两个输入都必须是 PIL.Image")

        images = [self._prepare(full_image), self._prepare(roi_image)]
        if self.training:
            rotation = int(torch.randint(4, (1,)).item())
            reflect = bool(torch.randint(2, (1,)).item())
            scale = _uniform(*self.scale)
            height, width = self.image_size
            translation = (
                round(_uniform(-self.translate, self.translate) * width),
                round(_uniform(-self.translate, self.translate) * height),
            )
            order, factors = self._jitter_parameters()
            images = [
                self._jitter(
                    TF.affine(
                        self._d4(image, rotation, reflect),
                        angle=0.0,
                        translate=translation,
                        scale=scale,
                        shear=(0.0, 0.0),
                        interpolation=self.interpolation,
                        fill=self.pad_fill,
                    ),
                    order,
                    factors,
                )
                for image in images
            ]

        tensors = [TF.normalize(TF.to_tensor(image), self.mean, self.std) for image in images]
        return torch.stack(tensors, dim=0)


# Readable compatibility alias for callers that used a shorter draft name.
PairedTransform = PairedDualViewTransform


def build_dual_view_transforms(
    image_size: int | tuple[int, int] = 224,
    *,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    interpolation: Any = "bicubic",
    pad_fill: int | tuple[int, int, int] | None = None,
    translate: float = 0.04,
    scale: tuple[float, float] = (0.95, 1.05),
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.0,
) -> tuple[PairedDualViewTransform, PairedDualViewTransform]:
    """Build stochastic train and deterministic validation paired transforms."""
    common = dict(
        image_size=image_size,
        mean=mean,
        std=std,
        interpolation=interpolation,
        pad_fill=pad_fill,
    )
    train_transform = PairedDualViewTransform(
        training=True,
        translate=translate,
        scale=scale,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        **common,
    )
    val_transform = PairedDualViewTransform(training=False, **common)
    return train_transform, val_transform


# Alias chosen to describe the operation rather than the consuming model.
build_paired_transforms = build_dual_view_transforms


def _as_rgb_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_file():
            return None
        with Image.open(path) as source:
            return source.convert("RGB")
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            return TF.to_pil_image(tensor).convert("RGB")
    # NumPy is optional.  Pillow accepts ndarray-like objects through fromarray.
    if hasattr(value, "__array_interface__"):
        try:
            return Image.fromarray(value).convert("RGB")
        except (TypeError, ValueError):
            return None
    return None


def _bbox_from_mapping(value: Mapping[str, Any]) -> tuple[Sequence[Any], bool, str] | None:
    for key in ("bbox", "box", "roi_bbox", "coordinates", "coords"):
        if value.get(key) is not None:
            return value[key], bool(value.get("normalized", value.get("is_normalized", False))), str(
                value.get("bbox_format", value.get("format", "xyxy"))
            )
    key_sets = (
        (("left", "top", "right", "bottom"), "xyxy"),
        (("x1", "y1", "x2", "y2"), "xyxy"),
        (("xmin", "ymin", "xmax", "ymax"), "xyxy"),
        (("x", "y", "width", "height"), "xywh"),
    )
    for keys, bbox_format in key_sets:
        if all(key in value and value[key] not in (None, "") for key in keys):
            return [value[key] for key in keys], bool(
                value.get("normalized", value.get("is_normalized", False))
            ), bbox_format
    return None


def _crop_bbox(
    image: Image.Image,
    coordinates: Sequence[Any],
    *,
    normalized: bool = False,
    bbox_format: str = "xyxy",
) -> Image.Image | None:
    if isinstance(coordinates, (str, bytes)) or len(coordinates) != 4:
        return None
    try:
        values = [float(value) for value in coordinates]
    except (TypeError, ValueError):
        return None
    if not all(torch.isfinite(torch.tensor(values)).tolist()):
        return None
    if normalized or all(0.0 <= value <= 1.0 for value in values):
        values[0] *= image.width
        values[2] *= image.width
        values[1] *= image.height
        values[3] *= image.height
    if bbox_format.lower().replace("-", "") in {"xywh", "coco"}:
        values[2] += values[0]
        values[3] += values[1]
    left, top, right, bottom = values
    left = max(0, min(image.width, int(left // 1)))
    top = max(0, min(image.height, int(top // 1)))
    right = max(0, min(image.width, int(-(-right // 1))))
    bottom = max(0, min(image.height, int(-(-bottom // 1))))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def _roi_from_value(value: Any, full_image: Image.Image) -> Image.Image | None:
    image = _as_rgb_image(value)
    if image is not None:
        return image
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("roi", "roi_image", "image", "crop"):
            if key in value:
                image = _as_rgb_image(value[key])
                if image is not None:
                    return image
        bbox = _bbox_from_mapping(value)
        if bbox is not None:
            return _crop_bbox(full_image, bbox[0], normalized=bbox[1], bbox_format=bbox[2])
    for attribute in ("roi", "roi_image", "image", "crop"):
        if hasattr(value, attribute):
            image = _as_rgb_image(getattr(value, attribute))
            if image is not None:
                return image
    for attribute in ("bbox", "box", "roi_bbox", "coordinates", "coords"):
        if hasattr(value, attribute):
            return _crop_bbox(
                full_image,
                getattr(value, attribute),
                normalized=bool(getattr(value, "normalized", False)),
                bbox_format=str(getattr(value, "bbox_format", "xyxy")),
            )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return _crop_bbox(full_image, value)
        # Common detector APIs return (crop, metadata) or (bbox, score).
        if value:
            image = _roi_from_value(value[0], full_image)
            if image is not None:
                return image
    return None


def _discover_roi_provider() -> Any:
    module_names = (f"{__package__}.roi", "src.roi") if __package__ else ("roi", "src.roi")
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        for name in ("extract_roi", "get_roi", "detect_roi", "make_roi", "crop_roi"):
            provider = getattr(module, name, None)
            if callable(provider):
                return provider
        for name in ("ROIExtractor", "RoiExtractor"):
            provider_type = getattr(module, name, None)
            if callable(provider_type):
                try:
                    return provider_type()
                except TypeError:
                    pass
    return None


def _provider_callable(provider: Any) -> Any:
    if callable(provider):
        return provider
    for name in ("extract_roi", "extract", "get_roi", "detect", "crop"):
        candidate = getattr(provider, name, None)
        if callable(candidate):
            return candidate
    raise TypeError("roi_provider 必须可调用或提供 extract/get_roi/detect/crop 方法")


def _call_roi_provider(provider: Any, image: Image.Image, record: Mapping[str, Any]) -> Any:
    function = _provider_callable(provider)
    path = record.get("path")
    candidates = (
        ((image, record), {}),
        ((), {"image": image, "record": record}),
        ((image,), {}),
        ((), {"image": image}),
        ((), {"source": image, "record": record}),
        ((path, record), {}),
        ((), {"path": path, "record": record}),
    )
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(image, record)
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return function(*args, **kwargs)
    raise TypeError("无法用受支持的 image/record/path 参数调用 roi_provider")


def _record_roi(record: Mapping[str, Any], full_image: Image.Image) -> Image.Image | None:
    for key in ("roi", "roi_image", "roi_path", "crop_path"):
        if record.get(key) is not None:
            image = _roi_from_value(record[key], full_image)
            if image is not None:
                return image
    bbox = _bbox_from_mapping(record)
    if bbox is not None:
        return _crop_bbox(full_image, bbox[0], normalized=bbox[1], bbox_format=bbox[2])
    return None


class DualViewPlantDataset(Dataset):
    """Labeled dataset returning ``(views, label, sample_id)``.

    ``views`` is ``[2, C, H, W]`` with full image first and ROI second.  ROI may
    come from record image/path fields, bbox mappings, or a duck-typed provider.
    If no valid ROI is produced, the already-decoded full image is copied as the
    ROI pixel source; paired transforms therefore keep both fallback views equal.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        transform: Any = None,
        roi_provider: Any = None,
        *,
        roi_getter: Any = None,
        roi_cache: Mapping[str, Any] | None = None,
        roi_cache_sha256: str | None = None,
        auto_roi: bool = False,
    ) -> None:
        if auto_roi:
            raise ValueError("dual-view 正式管线禁止动态 ROI；必须传入严格验证的预计算 roi_cache")
        if roi_provider is not None or roi_getter is not None:
            raise ValueError("dual-view 正式管线不接受动态 roi_provider/roi_getter")
        if roi_cache is None:
            raise ValueError("dual-view dataset 必须消费预计算且已验证的 roi_cache")
        if not isinstance(roi_cache_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", roi_cache_sha256):
            raise ValueError("必须提供小写 64 位十六进制 ROI cache 文件 SHA-256")
        self.records = [dict(record) for record in records]
        self.transform = transform
        self.roi_provider = None
        self.roi_cache = dict(roi_cache)
        self.roi_cache_sha256 = str(roi_cache_sha256).lower()
        expected_ids = {str(row.get("sample_id") or row.get("filename") or Path(row["path"]).name) for row in self.records}
        if set(self.roi_cache) != expected_ids:
            raise ValueError("roi_cache 必须恰好全覆盖 dataset records，且不能有额外项")

    def __len__(self) -> int:
        return len(self.records)

    def _load_views(self, record: Mapping[str, Any]) -> tuple[Image.Image, Image.Image, Path]:
        image_path = Path(record["path"])
        with Image.open(image_path) as source:
            full_image = source.convert("RGB")
        sample_id = str(record.get("sample_id") or record.get("filename") or image_path.name)
        result = self.roi_cache[sample_id]
        if bool(getattr(result, "is_fallback", False)):
            # 锁定契约：fallback 的两视图像素源完全相同，不应用均值方形 padding。
            roi_image = full_image.copy()
        else:
            try:
                from .roi import ROIResult, crop_with_roi
            except ImportError:  # pragma: no cover
                from roi import ROIResult, crop_with_roi
            if not isinstance(result, ROIResult):
                raise TypeError("roi_cache value 必须是 src.roi.ROIResult")
            roi_image = crop_with_roi(full_image, result)
        return full_image, roi_image, image_path

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        record = self.records[index]
        full_image, roi_image, image_path = self._load_views(record)
        if self.transform is not None:
            views = self.transform(full_image, roi_image)
        else:
            views = torch.stack([TF.to_tensor(full_image), TF.to_tensor(roi_image)])
        if not torch.is_tensor(views) or views.ndim != 4 or views.shape[0] != 2:
            raise ValueError("dual-view transform 必须返回形状 [2, C, H, W] 的 Tensor")
        label = int(record["class_idx"])
        sample_id = str(record.get("sample_id") or record.get("filename") or image_path.name)
        return views, label, sample_id


class TestDualViewDataset(DualViewPlantDataset):
    """Unlabeled dual-view dataset returning ``(views, sample_id)``."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any] | str | os.PathLike[str]],
        transform: Any = None,
        roi_provider: Any = None,
        *,
        roi_getter: Any = None,
        roi_cache: Mapping[str, Any] | None = None,
        roi_cache_sha256: str | None = None,
        auto_roi: bool = False,
    ) -> None:
        normalized_records: list[Mapping[str, Any]] = []
        for record in records:
            if isinstance(record, (str, os.PathLike)):
                path = Path(record)
                normalized_records.append({"path": str(path), "sample_id": path.name})
            elif isinstance(record, Mapping):
                normalized_records.append(record)
            else:
                raise TypeError("测试记录必须是路径或 mapping")
        super().__init__(
            normalized_records,
            transform,
            roi_provider,
            roi_getter=roi_getter,
            roi_cache=roi_cache,
            roi_cache_sha256=roi_cache_sha256,
            auto_roi=auto_roi,
        )

    def __getitem__(self, index: int) -> tuple[Any, str]:
        record = self.records[index]
        full_image, roi_image, image_path = self._load_views(record)
        if self.transform is not None:
            views = self.transform(full_image, roi_image)
        else:
            views = torch.stack([TF.to_tensor(full_image), TF.to_tensor(roi_image)])
        if not torch.is_tensor(views) or views.ndim != 4 or views.shape[0] != 2:
            raise ValueError("dual-view transform 必须返回形状 [2, C, H, W] 的 Tensor")
        sample_id = str(record.get("sample_id") or record.get("filename") or image_path.name)
        return views, sample_id


class PlantDataset(Dataset):
    """Image dataset returning ``(image, label, sample_id)``."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        transform: Any = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        record = self.records[index]
        image_path = Path(record["path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(record["class_idx"])
        sample_id = str(record.get("sample_id") or record.get("filename") or image_path.name)
        return image, label, sample_id


def _seed_python_libraries(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # NumPy is optional for this module.
        pass


def _seed_worker(worker_id: int) -> None:
    del worker_id
    _seed_python_libraries(torch.initial_seed() % (2**32))


def get_dual_dataloaders(
    records: Sequence[Mapping[str, Any]],
    val_fold: int,
    batch_size: int = 32,
    *,
    image_size: int | tuple[int, int] = 224,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    interpolation: Any = "bicubic",
    pad_fill: int | tuple[int, int, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 2026,
    persistent_workers: bool = False,
    roi_provider: Any = None,
    roi_getter: Any = None,
    roi_cache: Mapping[str, Any] | None = None,
    roi_cache_sha256: str | None = None,
    auto_roi: bool = False,
    translate: float = 0.04,
    scale: tuple[float, float] = (0.95, 1.05),
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.0,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible labeled dual-view train/validation loaders.

    Each batch is ``(views, labels, sample_ids)`` and ``views`` has shape
    ``[B, 2, C, H, W]``.  Separate explicit generators preserve the checkpoint
    contract used by ``train._loader_rng_state``.  On Windows, keep the default
    ``num_workers=0`` or construct loaders beneath the process main guard.
    ``persistent_workers=False`` is recommended for exact epoch-boundary resume,
    because worker-internal RNG state is not captured by a DataLoader generator.
    """
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    train_records, val_records = split_by_fold(records, val_fold)
    if roi_cache is None:
        raise ValueError("dual dataloaders 必须提供预计算 roi_cache")
    all_ids = {str(row.get("sample_id") or row.get("filename") or Path(row["path"]).name) for row in records}
    if set(roi_cache) != all_ids:
        raise ValueError("roi_cache 必须先严格全覆盖全部 records")
    train_ids = {str(row.get("sample_id") or row.get("filename") or Path(row["path"]).name) for row in train_records}
    val_ids = all_ids - train_ids
    train_cache = {key: roi_cache[key] for key in train_ids}
    val_cache = {key: roi_cache[key] for key in val_ids}
    train_transform, val_transform = build_dual_view_transforms(
        image_size,
        mean=mean,
        std=std,
        interpolation=interpolation,
        pad_fill=pad_fill,
        translate=translate,
        scale=scale,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    dataset_kwargs = dict(
        roi_provider=roi_provider,
        roi_getter=roi_getter,
        roi_cache_sha256=roi_cache_sha256,
        auto_roi=auto_roi,
    )
    train_dataset = DualViewPlantDataset(train_records, train_transform, roi_cache=train_cache, **dataset_kwargs)
    val_dataset = DualViewPlantDataset(val_records, val_transform, roi_cache=val_cache, **dataset_kwargs)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    _seed_python_libraries(seed)
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=val_generator,
        **common,
    )
    return train_loader, val_loader


# Naming variants kept intentionally: downstream branches used both forms.
get_dual_view_dataloaders = get_dual_dataloaders
get_dualview_dataloaders = get_dual_dataloaders


def get_test_dual_dataloader(
    records: Sequence[Mapping[str, Any] | str | os.PathLike[str]],
    batch_size: int = 32,
    *,
    image_size: int | tuple[int, int] = 224,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    interpolation: Any = "bicubic",
    pad_fill: int | tuple[int, int, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 2026,
    persistent_workers: bool = False,
    roi_provider: Any = None,
    roi_getter: Any = None,
    roi_cache: Mapping[str, Any] | None = None,
    roi_cache_sha256: str | None = None,
    auto_roi: bool = False,
) -> DataLoader:
    """Create a deterministic dual-view test loader with an explicit generator."""
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    _, transform = build_dual_view_transforms(
        image_size,
        mean=mean,
        std=std,
        interpolation=interpolation,
        pad_fill=pad_fill,
    )
    dataset = TestDualViewDataset(
        records,
        transform,
        roi_provider,
        roi_getter=roi_getter,
        roi_cache=roi_cache,
        roi_cache_sha256=roi_cache_sha256,
        auto_roi=auto_roi,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    _seed_python_libraries(seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        persistent_workers=persistent_workers and num_workers > 0,
        generator=generator,
    )


get_test_dual_view_dataloader = get_test_dual_dataloader


def get_dataloaders(
    records: Sequence[Mapping[str, Any]],
    val_fold: int,
    batch_size: int = 32,
    *,
    image_size: int | tuple[int, int] = 224,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    interpolation: Any = "bicubic",
    pad_fill: int | tuple[int, int, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 2026,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible train/validation loaders.

    ``num_workers=0`` is deliberately the default so Windows callers are safe
    even before opting into multiprocessing. When increasing it on Windows,
    call this function beneath an ``if __name__ == "__main__":`` guard.
    """
    train_records, val_records = split_by_fold(records, val_fold)
    train_transform, val_transform = build_transforms(
        image_size,
        mean=mean,
        std=std,
        interpolation=interpolation,
        pad_fill=pad_fill,
    )
    train_dataset = PlantDataset(train_records, train_transform)
    val_dataset = PlantDataset(val_records, val_transform)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数")

    _seed_python_libraries(seed)
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=val_generator,
        **common,
    )
    return train_loader, val_loader


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_FOLDS_CSV",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "DualViewPlantDataset",
    "PairedDualViewTransform",
    "PairedTransform",
    "PlantDataset",
    "SquarePad",
    "TestDualViewDataset",
    "build_dual_view_transforms",
    "build_paired_transforms",
    "build_transforms",
    "get_dataloaders",
    "get_dual_dataloaders",
    "get_dual_view_dataloaders",
    "get_dualview_dataloaders",
    "get_test_dual_dataloader",
    "get_test_dual_view_dataloader",
    "load_records",
    "resolve_interpolation",
    "split_by_fold",
]
