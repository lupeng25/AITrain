#!/usr/bin/env python3
"""Lightweight SMP adapter checks that do not require torch/SMP installation."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import py_compile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "python_trainers" / "semantic_segmentation" / "smp_trainer.py"
EVALUATOR = ROOT / "python_trainers" / "semantic_segmentation" / "smp_evaluator.py"
EXPORTER = ROOT / "python_trainers" / "semantic_segmentation" / "smp_exporter.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trainer_module():
    return load_module("aitrain_smp_trainer_test", TRAINER)


def test_py_compile() -> None:
    py_compile.compile(str(TRAINER), doraise=True)
    py_compile.compile(str(EVALUATOR), doraise=True)
    py_compile.compile(str(EXPORTER), doraise=True)


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


def test_semantic_masks_preserve_raw_palette_indexes() -> None:
    trainer_source = TRAINER.read_text(encoding="utf-8")
    evaluator_source = EVALUATOR.read_text(encoding="utf-8")
    for source in (trainer_source, evaluator_source):
        assert "def load_mask_ids(" in source
        assert 'mask.mode not in {"L", "P"}' in source
        assert "convert(\"L\")" not in source

    from PIL import Image

    evaluator = load_module("aitrain_smp_evaluator_test", EVALUATOR)
    with tempfile.TemporaryDirectory() as directory:
        mask_path = Path(directory) / "palette.png"
        mask = Image.new("P", (2, 2))
        palette = [0] * 768
        palette[3:6] = [255, 0, 0]
        mask.putpalette(palette)
        mask.putdata([1, 1, 1, 1])
        mask.save(mask_path)
        assert load_trainer_module().load_mask_ids(mask_path).tolist() == [[1, 1], [1, 1]]
        assert evaluator.load_mask_ids(mask_path).tolist() == [[1, 1], [1, 1]]


def test_smp_adapters_materialize_verified_dataset_snapshot() -> None:
    trainer = load_trainer_module()
    evaluator = load_module("aitrain_smp_evaluator_snapshot_test", EVALUATOR)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        classes = source / "classes.txt"
        classes.write_text("background\ndefect\n", encoding="utf-8")
        digest = hashlib.sha256(classes.read_bytes()).hexdigest()
        manifest = root / "dataset_snapshot.json"
        manifest.write_text(json.dumps({
            "schemaVersion": 2,
            "complete": True,
            "files": [{"relativePath": "classes.txt", "sha256": digest}],
        }), encoding="utf-8")

        trainer_result, trainer_manifest = trainer.materialize_request_dataset({
            "datasetSnapshotManifest": str(manifest),
            "datasetSnapshotStagingPath": str(root / "trainer-staging"),
        }, {}, source)
        evaluator_result, evaluator_manifest = evaluator.materialize_request_dataset({}, {
            "datasetSnapshotManifest": str(manifest),
            "datasetSnapshotStagingPath": str(root / "evaluator-staging"),
        }, source)

        assert trainer_manifest == str(manifest)
        assert evaluator_manifest == str(manifest)
        assert (trainer_result / "classes.txt").read_text(encoding="utf-8") == "background\ndefect\n"
        assert (evaluator_result / "classes.txt").read_text(encoding="utf-8") == "background\ndefect\n"


def test_smp_adapter_events_use_sdk_and_preserve_backend() -> None:
    for name, path, backend in (
        ("trainer", TRAINER, "smp_semantic_segmentation"),
        ("evaluator", EVALUATOR, "smp_semantic_segmentation_eval"),
        ("exporter", EXPORTER, "smp_semantic_segmentation_export"),
    ):
        module = load_module(f"aitrain_smp_{name}_events_test", path)
        events: list[dict] = []
        module._adapter = module.AdapterSdk(backend, event_sink=events.append)
        module._event_channel = None
        module.emit("artifact", backend="spoofed", kind="report", path="out/report.json")
        module.emit("completed", reportPath="out/report.json")
        assert [event["type"] for event in events] == ["artifact", "completed"]
        assert all(event["backend"] == backend for event in events)


def test_smp_exporter_writes_complete_contract_and_exact_artifacts() -> None:
    exporter = load_module("aitrain_smp_exporter_contract_test", EXPORTER)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "best.onnx"
        source.write_bytes(b"fake-onnx")
        source_sidecar = root / "semantic_segmentation_sidecar.json"
        source_sidecar.write_text(json.dumps({
            "classNames": ["stale"],
            "normalization": {"mean": exporter.MEAN, "std": exporter.STD, "scale": 1.0 / 255.0},
        }), encoding="utf-8")
        evaluation = root / "evaluation_report.json"
        evaluation.write_text(json.dumps({
            "perClass": [
                {"classId": 1, "className": "defect"},
                {"classId": 0, "className": "background"},
            ],
        }), encoding="utf-8")
        output = root / "export" / "model.onnx"
        events: list[dict] = []
        exporter._adapter = exporter.AdapterSdk(exporter.BACKEND_ID, event_sink=events.append)
        exporter._event_channel = None
        original_inspect = exporter.inspect_onnx
        exporter.inspect_onnx = lambda _path: (
            [{"name": "images", "layout": "NCHW", "shape": [1, 3, 256, 256]}],
            [{"name": "logits", "layout": "NCHW", "shape": [1, 2, 256, 256]}],
            13,
        )
        try:
            assert exporter.run({
                "modelPath": str(source),
                "outputPath": str(output),
                "sidecarPath": str(source_sidecar),
                "evaluationReportPath": str(evaluation),
            }) == 0
        finally:
            exporter.inspect_onnx = original_inspect

        sidecar_path = output.with_suffix(".aitrain-export.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        contract = sidecar["modelContract"]
        assert output.read_bytes() == b"fake-onnx"
        assert sidecar["backend"] == "smp_semantic_segmentation_export"
        assert sidecar["format"] == "onnx"
        assert sidecar["decoder"] == "smp_semantic_segmentation"
        assert sidecar["opset"] == 13
        assert sidecar["exporterVersion"]
        assert contract["modelFamily"] == "semantic_segmentation"
        assert contract["taskType"] == "semantic_segmentation"
        assert contract["decoder"] == "smp_semantic_segmentation"
        assert contract["classNames"] == ["background", "defect"]
        assert contract["runtimeRoutes"] == ["aitrain_onnxruntime"]
        assert contract["inputs"][0]["shape"] == [1, 3, 256, 256]
        assert contract["outputs"][0]["shape"] == [1, 2, 256, 256]
        assert [event["type"] for event in events] == ["progress", "artifact", "artifact", "progress", "completed"]
        assert [event.get("kind") for event in events if event["type"] == "artifact"] == ["export", "export_sidecar"]


def test_smp_evaluator_resubmits_required_artifact_kinds() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    for kind in ("onnx_model", "model_sidecar", "checkpoint", "evaluation_report"):
        assert f'("{kind}",' in source


if __name__ == "__main__":
    test_py_compile()
    test_public_presets_registered()
    test_semantic_masks_preserve_raw_palette_indexes()
    test_smp_adapters_materialize_verified_dataset_snapshot()
    test_smp_adapter_events_use_sdk_and_preserve_backend()
    test_smp_exporter_writes_complete_contract_and_exact_artifacts()
    test_smp_evaluator_resubmits_required_artifact_kinds()
