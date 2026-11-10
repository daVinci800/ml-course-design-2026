"""Pipeline contract tests using only temporary synthetic data.

These tests deliberately never read from or write to the repository's ``data/``
directory and never request pretrained weights.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _save_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (48, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_imports_have_no_runtime_side_effects(tmp_path: Path) -> None:
    """Importing modules must not train, parse CLI args, print, or create files."""

    script = """
import contextlib
import importlib
import io
import json
from pathlib import Path

before = sorted(str(p.relative_to(Path.cwd())) for p in Path.cwd().rglob('*'))
stdout, stderr = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    for name in ('src.create_folds', 'src.dataset', 'src.model', 'src.train', 'src.evaluate'):
        importlib.import_module(name)
after = sorted(str(p.relative_to(Path.cwd())) for p in Path.cwd().rglob('*'))
print(json.dumps({'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(),
                  'created': sorted(set(after) - set(before))}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report == {"stdout": "", "stderr": "", "created": []}


def test_folds_are_disjoint_complete_stable_and_never_touch_project_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.create_folds as folds

    # The public API resolves paths relative to its configured project root. Point
    # that root at pytest's sandbox so repository data is structurally unreachable.
    monkeypatch.setattr(folds, "PROJECT_ROOT", tmp_path)
    build_manifest = folds.build_manifest
    write_manifest = folds.write_manifest

    # Three classes x ten samples permits a balanced five-fold split.
    train_dir = tmp_path / "synthetic" / "train"
    for class_index, class_name in enumerate(("alpha", "beta", "gamma")):
        for image_index in range(10):
            _save_image(
                train_dir / class_name / f"{image_index:02d}.png",
                (30 + 60 * class_index, 10 * image_index, 120),
            )

    first = build_manifest(train_dir, n_splits=5, seed=2026)
    second = build_manifest(train_dir, n_splits=5, seed=2026)
    assert [(row["sample_id"], row["fold"]) for row in first] == [
        (row["sample_id"], row["fold"]) for row in second
    ]

    all_ids = {str(row["sample_id"]) for row in first}
    ids_by_fold = {
        fold: {str(row["sample_id"]) for row in first if row["fold"] == fold}
        for fold in range(5)
    }
    assert set().union(*ids_by_fold.values()) == all_ids
    assert sum(map(len, ids_by_fold.values())) == len(all_ids)
    for left in range(5):
        for right in range(left + 1, 5):
            assert ids_by_fold[left].isdisjoint(ids_by_fold[right])
    assert Counter((row["fold"], row["class_idx"]) for row in first) == {
        (fold, class_index): 2
        for fold in range(5)
        for class_index in range(3)
    }
    # Generation is in-memory until an explicit output is supplied, and writes
    # are exclusive rather than silently replacing an existing manifest.
    output = tmp_path / "folds.csv"
    digest = write_manifest(first, output)
    assert len(digest) == 64
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="覆盖|存在"):
        write_manifest(first, output)
    assert output.read_bytes() == original


def test_deterministic_transform_preserves_subject_touching_image_edge() -> None:
    from src.dataset import build_transforms

    image = Image.new("RGB", (96, 40), "black")
    ImageDraw.Draw(image).rectangle((0, 5, 11, 34), fill="white")
    _, deterministic = build_transforms(
        image_size=64,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    first = deterministic(image)
    second = deterministic(image)
    assert first.shape == (3, 64, 64)
    assert torch.equal(first, second)
    # Square padding may add rows, but a crop would remove this left-edge object.
    assert first[:, :, 0].max().item() > 0.5


def test_dataset_model_and_one_batch_forward_backward(tmp_path: Path) -> None:
    from src.dataset import PlantDataset, build_transforms, get_dataloaders
    from src.model import create_model, get_model_data_config

    records: list[dict[str, Any]] = []
    for index in range(6):
        image_path = tmp_path / f"sample-{index}.png"
        _save_image(image_path, (20 * index, 80, 160), size=(40, 52))
        records.append(
            {
                "path": str(image_path),
                "class_idx": index % 2,
                "fold": index % 2,
                "sample_id": f"sample-{index}",
            }
        )

    _, val_transform = build_transforms(image_size=64)
    image, label, sample_id = PlantDataset(records, val_transform)[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert sample_id == "sample-0"

    train_loader, val_loader = get_dataloaders(
        records,
        val_fold=0,
        batch_size=2,
        image_size=64,
        num_workers=0,
        pin_memory=False,
        seed=7,
    )
    train_ids = {row["sample_id"] for row in train_loader.dataset.records}
    val_ids = {row["sample_id"] for row in val_loader.dataset.records}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {row["sample_id"] for row in records}
    images, labels, sample_ids = next(iter(train_loader))
    assert images.shape == (2, 3, 64, 64)
    assert labels.shape == (2,)
    assert len(sample_ids) == 2

    # A small, offline model keeps this a CPU-friendly unit-level smoke test.
    model = create_model(model_name="resnet18", num_classes=2, pretrained=False)
    config = get_model_data_config(model)
    assert tuple(config["input_size"])[0] == 3
    logits = model(images)
    assert logits.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("failure", ["duplicate", "missing"])
def test_oof_rejects_duplicate_or_missing_samples(failure: str) -> None:
    from src.evaluate import EvaluationError, validate_oof

    manifest = [
        {"sample_id": "a", "fold": 0, "class_idx": 0},
        {"sample_id": "b", "fold": 1, "class_idx": 1},
    ]
    rows = [
        {"sample_id": "a", "fold": 0, "target": 0, "prediction": 0, "prob_0": 0.9, "prob_1": 0.1},
        {"sample_id": "b", "fold": 1, "target": 1, "prediction": 1, "prob_0": 0.2, "prob_1": 0.8},
    ]
    if failure == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows.pop()

    with pytest.raises(EvaluationError, match="重复|缺失|覆盖"):
        validate_oof(rows, manifest)


def test_checkpoint_config_and_manifest_compatibility_contract() -> None:
    """Exercise the final train helper when available during API integration."""

    try:
        train = importlib.import_module("src.train")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("训练模块仍在 API 集成中，当前尚不能作为包安全导入")
    validator = getattr(train, "validate_checkpoint_compatibility", None)
    if validator is None:
        pytest.skip("训练模块尚未公开 checkpoint 兼容性校验 API")

    checkpoint = {
        "config_hash": "config-v1",
        "manifest_hash": "manifest-v1",
        "class_to_idx": {"zero": 0, "one": 1},
    }
    arguments = {
        "checkpoint": checkpoint,
        "config_hash": "config-v1",
        "manifest_hash": "manifest-v1",
        "class_to_idx": {"zero": 0, "one": 1},
    }
    signature = inspect.signature(validator)
    validator(**{name: value for name, value in arguments.items() if name in signature.parameters})

    for field, incompatible in (
        ("config_hash", "other-config"),
        ("manifest_hash", "other-manifest"),
        ("class_to_idx", {"zero": 1, "one": 0}),
    ):
        changed = dict(arguments)
        changed[field] = incompatible
        with pytest.raises((ValueError, RuntimeError), match="config|配置|manifest|清单|class|类别|不兼容|匹配"):
            validator(**{name: value for name, value in changed.items() if name in signature.parameters})
