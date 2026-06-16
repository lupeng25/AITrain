#!/usr/bin/env python3
"""Lightweight SMP adapter checks that do not require torch/SMP installation."""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "python_trainers" / "semantic_segmentation" / "smp_trainer.py"
EVALUATOR = ROOT / "python_trainers" / "semantic_segmentation" / "smp_evaluator.py"


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("aitrain_smp_trainer_test", TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load trainer module spec: {TRAINER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_py_compile() -> None:
    py_compile.compile(str(TRAINER), doraise=True)
    py_compile.compile(str(EVALUATOR), doraise=True)


def test_public_presets_registered() -> None:
    module = load_trainer_module()
    expected = {
        "smp_unet_resnet34",
        "smp_unetplusplus_resnet34",
        "smp_fpn_resnet34",
        "smp_deeplabv3plus_resnet50",
        "smp_segformer_mit_b0",
    }
    assert expected.issubset(set(module.PRESETS))
    for preset_id in expected:
        assert module.PRESETS[preset_id]["encoder_candidates"]


if __name__ == "__main__":
    test_py_compile()
    test_public_presets_registered()
