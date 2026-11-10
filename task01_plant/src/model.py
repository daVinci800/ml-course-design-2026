"""timm 模型构造、身份记录以及 AdamW 参数分组工具。"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch.nn as nn
import timm

try:
    from timm.data import resolve_model_data_config
except ImportError:  # pragma: no cover - compatibility with older timm
    from timm.data import resolve_data_config

    def resolve_model_data_config(model: nn.Module, **kwargs: Any) -> dict[str, Any]:
        return resolve_data_config(model=model, **kwargs)


_IDENTITY_ATTRIBUTE = "_plant_classifier_model_identity"


def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 of a local file without loading it into memory."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"模型权重文件不存在或不是普通文件: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Make timm config values deterministic and safe for checkpoint metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key != "state_dict"
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_model_id(config: Mapping[str, Any], requested_model_id: str) -> str:
    architecture = str(config.get("architecture") or requested_model_id.split(".", 1)[0])
    tag = config.get("tag")
    return f"{architecture}.{tag}" if tag else architecture


def _registry_pretrained_source(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the registered source without touching a cache or the network."""
    explicit_source = str(config.get("source") or "")
    local_file = config.get("file")
    if local_file:
        path = Path(str(local_file)).expanduser().resolve()
        source: dict[str, Any] = {"kind": "local_file", "path": str(path)}
        if path.is_file():
            source["sha256"] = compute_file_sha256(path)
        return source
    if explicit_source == "local-dir":
        return {"kind": "local_directory", "path": str(local_file or "")}
    if explicit_source == "hf-hub" and config.get("hf_hub_id"):
        return {
            "kind": "hf_hub",
            "location": str(config["hf_hub_id"]),
            "filename": config.get("hf_hub_filename"),
        }
    # timm may select HF Hub or URL according to the installed optional
    # dependencies. Preserve both registered locators rather than claiming a
    # transport that cannot be observed reliably from the returned model.
    return {
        "kind": "timm_registry",
        "hf_hub_id": config.get("hf_hub_id"),
        "hf_hub_filename": config.get("hf_hub_filename"),
        "url": config.get("url"),
    }


def _local_source(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    actual_sha256 = compute_file_sha256(resolved)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError("expected_sha256 必须是 64 位十六进制 SHA-256")
        if actual_sha256 != expected:
            raise ValueError(
                f"模型权重 SHA-256 不匹配: expected={expected}, actual={actual_sha256}"
            )
    return {"kind": "local_file", "path": str(resolved), "sha256": actual_sha256}


def create_model(
    num_classes: int = 5,
    *,
    model_name: str = "resnet50",
    pretrained: bool = True,
    return_data_config: bool = False,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    **model_kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Create a timm classifier and attach reproducible model identity metadata.

    ``model_name`` is passed to timm unchanged, so tagged IDs such as
    ``efficientnet_b3.ra2_in1k`` retain their exact pretrained configuration.
    ``checkpoint_path`` remains compatible with :func:`timm.create_model`; when
    supplied, its absolute path and SHA-256 are recorded and may be checked with
    ``checkpoint_sha256``. Set ``pretrained=False`` for wholly offline random
    construction.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name 必须是非空的精确 timm 模型 ID")
    requested_model_id = model_name.strip()
    if requested_model_id != model_name:
        raise ValueError("model_name 不能包含首尾空白；请提供精确 timm 模型 ID")

    local_checkpoint = None
    if checkpoint_path not in (None, ""):
        local_checkpoint = _local_source(checkpoint_path, checkpoint_sha256)
    elif checkpoint_sha256 is not None:
        raise ValueError("checkpoint_sha256 只能与 checkpoint_path 一起使用")

    create_kwargs = dict(model_kwargs)
    if checkpoint_path not in (None, ""):
        # Resolve once so the path used by timm and the path in metadata agree.
        create_kwargs["checkpoint_path"] = local_checkpoint["path"]

    model = timm.create_model(
        requested_model_id,
        pretrained=pretrained,
        num_classes=num_classes,
        **create_kwargs,
    )

    actual_config = _jsonable(
        getattr(model, "pretrained_cfg", None)
        or getattr(model, "default_cfg", None)
        or {}
    )
    if not isinstance(actual_config, dict):  # defensive for nonstandard timm models
        actual_config = {"value": actual_config}
    sources: list[dict[str, Any]] = []
    if pretrained:
        sources.append(_registry_pretrained_source(actual_config))
    if local_checkpoint is not None:
        checkpoint_source = dict(local_checkpoint)
        checkpoint_source["role"] = "checkpoint_path"
        sources.append(checkpoint_source)

    identity = {
        "requested_model_id": requested_model_id,
        "resolved_model_id": _resolved_model_id(actual_config, requested_model_id),
        "architecture": actual_config.get("architecture"),
        "pretrained_cfg_tag": actual_config.get("tag"),
        "pretrained_cfg": actual_config,
        "pretrained_cfg_sha256": _stable_sha256(actual_config),
        "pretrained": bool(pretrained),
        "pretrained_sources": sources,
        # Singular form is convenient for the common one-source case and is
        # retained alongside the lossless list for checkpoint compatibility.
        "pretrained_source": sources[-1] if sources else {"kind": "random_init"},
        "checkpoint_path": local_checkpoint["path"] if local_checkpoint else None,
        "checkpoint_sha256": local_checkpoint["sha256"] if local_checkpoint else None,
        "num_classes": int(num_classes),
        "timm_version": str(getattr(timm, "__version__", "unknown")),
    }
    setattr(model, _IDENTITY_ATTRIBUTE, identity)

    if return_data_config:
        return model, get_model_data_config(model)
    return model


def get_model_data_config(model: nn.Module) -> dict[str, Any]:
    """Resolve timm preprocessing metadata for an already-created model."""
    return dict(resolve_model_data_config(model))


def get_model_identity(model: nn.Module) -> dict[str, Any]:
    """Return a detached, checkpoint-safe description of a model's identity.

    Models not created by :func:`create_model` still receive an inferred config
    identity, but their requested ID and weight-loading history are necessarily
    unknown.
    """
    recorded = getattr(model, _IDENTITY_ATTRIBUTE, None)
    if isinstance(recorded, Mapping):
        return copy.deepcopy(dict(recorded))

    actual_config = _jsonable(
        getattr(model, "pretrained_cfg", None)
        or getattr(model, "default_cfg", None)
        or {}
    )
    if not isinstance(actual_config, dict):
        actual_config = {"value": actual_config}
    architecture = actual_config.get("architecture")
    requested = str(architecture or model.__class__.__qualname__)
    return {
        "requested_model_id": None,
        "resolved_model_id": _resolved_model_id(actual_config, requested),
        "architecture": architecture,
        "pretrained_cfg_tag": actual_config.get("tag"),
        "pretrained_cfg": actual_config,
        "pretrained_cfg_sha256": _stable_sha256(actual_config),
        "pretrained": None,
        "pretrained_sources": [],
        "pretrained_source": {"kind": "unknown"},
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "num_classes": getattr(model, "num_classes", None),
        "timm_version": str(getattr(timm, "__version__", "unknown")),
    }


def _get_submodule(model: nn.Module, path: str) -> nn.Module | None:
    try:
        if hasattr(model, "get_submodule"):
            return model.get_submodule(path)
        current: Any = model
        for component in path.split("."):
            current = getattr(current, component)
        return current if isinstance(current, nn.Module) else None
    except (AttributeError, KeyError, TypeError):
        return None


def _flatten_modules(value: Any) -> Iterator[nn.Module]:
    if isinstance(value, nn.Module):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_modules(item)


def _iter_classifier_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Yield classifier modules, preferring timm's precise public metadata."""
    seen: set[int] = set()
    candidates: list[nn.Module] = []

    config = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None)
    classifier_paths = config.get("classifier") if isinstance(config, Mapping) else None
    if isinstance(classifier_paths, str):
        classifier_paths = (classifier_paths,)
    if isinstance(classifier_paths, (tuple, list)):
        for path in classifier_paths:
            if isinstance(path, str):
                module = _get_submodule(model, path)
                if module is not None:
                    candidates.append(module)

    get_classifier = getattr(model, "get_classifier", None)
    if callable(get_classifier):
        try:
            candidates.extend(_flatten_modules(get_classifier()))
        except (AttributeError, NotImplementedError, TypeError):
            pass

    # Config/get_classifier avoid accidentally treating pooling or head norms as
    # classifier parameters. Fall back to conventional names only when neither
    # public mechanism found a parameter-bearing module.
    if not any(tuple(module.parameters()) for module in candidates):
        for attribute in ("fc", "classifier", "head", "heads", "last_linear"):
            candidates.extend(_flatten_modules(getattr(model, attribute, None)))

    for module in candidates:
        if id(module) not in seen:
            seen.add(id(module))
            yield module


def split_head_backbone_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(backbone_params, head_params)`` without duplicating parameters.

    Shared/tied parameters are classified as head if a recognized classifier
    owns them. Frozen parameters remain included so an optimizer created before
    freeze/unfreeze transitions stays valid.
    """
    head_ids = {
        id(parameter)
        for module in _iter_classifier_modules(model)
        for parameter in module.parameters()
    }
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        (head_params if parameter_id in head_ids else backbone_params).append(parameter)
    return backbone_params, head_params


def _normalization_parameter_ids(model: nn.Module) -> set[int]:
    norm_ids: set[int] = set()
    norm_base = getattr(nn.modules.batchnorm, "_NormBase", ())
    instance_norm_base = getattr(nn.modules.instancenorm, "_InstanceNorm", ())
    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm)
    for module in model.modules():
        # The name check covers timm normalization implementations while the
        # isinstance checks cover all standard PyTorch normalization layers.
        is_norm = (
            isinstance(module, norm_types)
            or (norm_base and isinstance(module, norm_base))
            or (instance_norm_base and isinstance(module, instance_norm_base))
            or "norm" in module.__class__.__name__.lower()
        )
        if is_norm:
            norm_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    return norm_ids


def _assert_trainable_parameters_grouped_once(
    model: nn.Module, groups: list[dict[str, Any]]
) -> None:
    membership = Counter(
        id(parameter)
        for group in groups
        for parameter in group.get("params", ())
    )
    trainable = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = [name for parameter_id, name in trainable.items() if membership[parameter_id] == 0]
    duplicated = [name for parameter_id, name in trainable.items() if membership[parameter_id] != 1]
    if missing or duplicated:
        raise AssertionError(
            "每个 trainable parameter 必须恰好属于一个 AdamW 参数组; "
            f"missing={missing}, duplicated={duplicated}"
        )


def get_parameter_groups(
    model: nn.Module,
    *,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    weight_decay: float | None = None,
    no_decay_weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Build four AdamW groups: backbone/head × decay/no-decay.

    Biases, normalization parameters, and every parameter with at most one
    dimension receive ``no_decay_weight_decay`` (zero by default). The legacy
    ``name`` values remain ``backbone``/``head``; ``group_name`` uniquely names
    all four groups.
    """
    backbone_params, head_params = split_head_backbone_parameters(model)
    head_ids = {id(parameter) for parameter in head_params}
    if not head_ids:
        raise RuntimeError("无法可靠识别模型分类头，拒绝创建差分学习率参数组")

    norm_ids = _normalization_parameter_ids(model)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("backbone", True): [],
        ("backbone", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    for parameter in (*backbone_params, *head_params):
        parameter_id = id(parameter)
        name = parameter_names.get(parameter_id, "")
        no_decay = (
            parameter.ndim <= 1
            or name.rsplit(".", 1)[-1] == "bias"
            or parameter_id in norm_ids
        )
        component = "head" if parameter_id in head_ids else "backbone"
        buckets[(component, not no_decay)].append(parameter)

    groups: list[dict[str, Any]] = []
    for component, learning_rate in (("backbone", backbone_lr), ("head", head_lr)):
        for use_decay, suffix in ((True, "decay"), (False, "no_decay")):
            group: dict[str, Any] = {
                "params": buckets[(component, use_decay)],
                "name": component,  # backwards compatible with train.py
                "group_name": f"{component}_{suffix}",
                "component": component,
                "decay": use_decay,
            }
            if learning_rate is not None:
                group["lr"] = learning_rate
            if use_decay:
                if weight_decay is not None:
                    group["weight_decay"] = weight_decay
            else:
                group["weight_decay"] = no_decay_weight_decay
            groups.append(group)

    _assert_trainable_parameters_grouped_once(model, groups)
    return groups


def set_backbone_trainable(
    model: nn.Module,
    trainable: bool,
    *,
    keep_head_trainable: bool = True,
) -> nn.Module:
    """Freeze or unfreeze the recognized backbone and return ``model``."""
    backbone, head = split_head_backbone_parameters(model)
    if not head:
        raise RuntimeError("无法可靠识别模型分类头，拒绝冻结或解冻 backbone")
    for parameter in backbone:
        parameter.requires_grad_(trainable)
    if keep_head_trainable:
        for parameter in head:
            parameter.requires_grad_(True)
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze backbone parameters while leaving the classifier trainable."""
    return set_backbone_trainable(model, False, keep_head_trainable=True)


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """Unfreeze backbone parameters while leaving the classifier trainable."""
    return set_backbone_trainable(model, True, keep_head_trainable=True)


# Explicit alias documents that these groups are intended for AdamW while the
# established project API remains unchanged.
build_adamw_parameter_groups = get_parameter_groups


__all__ = [
    "build_adamw_parameter_groups",
    "compute_file_sha256",
    "create_model",
    "freeze_backbone",
    "get_model_data_config",
    "get_model_identity",
    "get_parameter_groups",
    "set_backbone_trainable",
    "split_head_backbone_parameters",
    "unfreeze_backbone",
]
