#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pytest
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER_DIR = ROOT / "python_trainers" / "detection"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))
YOLO_DIR = ROOT / "python_trainers" / "yolo"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

import ultralytics_trainer as trainer  # noqa: E402
import ultralytics_evaluator as evaluator  # noqa: E402
from yolo import ultralytics_exporter as exporter  # noqa: E402
from adapter_sdk import AdapterSdk  # noqa: E402
from dataset_snapshot import materialize_dataset_snapshot  # noqa: E402


@pytest.fixture(autouse=True)
def standalone_protocol(monkeypatch):
    """Unit tests invoke adapters without a Worker Host."""
    monkeypatch.setenv("AITRAIN_STANDALONE_ADAPTER_PROTOCOL", "1")


class FakeModel:
    def __init__(self) -> None:
        self.callbacks: dict[str, list] = {}

    def add_callback(self, event: str, callback) -> None:
        self.callbacks.setdefault(event, []).append(callback)


class FakeTrainer:
    def __init__(self) -> None:
        self.epoch = 0
        self.train_loader = [object(), object(), object(), object()]
        self.tloss = [1.0, 2.0, 3.0]
        self.metrics = {
            "metrics/precision(B)": 0.25,
            "metrics/recall(B)": 0.5,
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": 0.125,
        }

    def label_loss_items(self, loss_items) -> dict[str, float]:
        return {
            "train/box_loss": float(loss_items[0]),
            "train/cls_loss": float(loss_items[1]),
            "train/dfl_loss": float(loss_items[2]),
        }


def test_sanitize_log_line_removes_ansi_tqdm_noise() -> None:
    dirty = "\x1b[34mtrain:\x1b[0m 50% ━━━━━ 2/4 1.0it/s\r\nuseful message"
    cleaned = trainer.sanitize_log_line(dirty)
    assert "━━━━" not in cleaned
    assert "\x1b" not in cleaned
    assert "useful message" in cleaned


def test_snapshot_data_yaml_rejects_paths_outside_materialized_root() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        dataset = root / "snapshot"
        output = root / "output"
        dataset.mkdir()
        output.mkdir()
        (dataset / "data.yaml").write_text(
            f"path: {(root / 'external').as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n",
            encoding="utf-8",
        )
        try:
            trainer.normalize_data_yaml(dataset, output, require_snapshot_containment=True)
        except ValueError as exc:
            assert "escapes the immutable dataset snapshot" in str(exc)
        else:
            raise AssertionError("snapshot data.yaml path escape was accepted")


def test_snapshot_data_yaml_keeps_paths_inside_materialized_root() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        dataset = root / "snapshot"
        output = root / "output"
        (dataset / "images" / "train").mkdir(parents=True)
        (dataset / "images" / "val").mkdir(parents=True)
        output.mkdir()
        (dataset / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n",
            encoding="utf-8",
        )
        normalized = trainer.normalize_data_yaml(dataset, output, require_snapshot_containment=True)
        text = normalized.read_text(encoding="utf-8")
        assert f'path: "{dataset.resolve().as_posix()}"' in text
        assert 'train: "images/train"' in text
        assert 'val: "images/val"' in text


def test_evaluator_rejects_snapshot_data_yaml_paths_outside_materialized_root() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        dataset = root / "snapshot"
        output = root / "output"
        dataset.mkdir()
        output.mkdir()
        (dataset / "data.yaml").write_text(
            f"path: {(root / 'external').as_posix()}\ntrain: images/train\nval: images/val\nnames: [item]\n",
            encoding="utf-8",
        )
        try:
            evaluator.normalize_evaluation_data_yaml(dataset, output, root / "dataset_snapshot.json")
        except ValueError as exc:
            assert "escapes the immutable dataset snapshot" in str(exc)
        else:
            raise AssertionError("evaluator accepted a snapshot data.yaml path escape")


def test_yolo_callbacks_emit_structured_progress_and_epoch_metrics() -> None:
    events: list[tuple[str, dict]] = []
    original_emit = trainer.emit
    trainer.emit = lambda event_type, **payload: events.append((event_type, payload))
    try:
        model = FakeModel()
        fake = FakeTrainer()
        trainer.register_training_callbacks(model, epochs=2, device="cpu")

        model.callbacks["on_train_start"][0](fake)
        model.callbacks["on_train_epoch_start"][0](fake)
        for _ in range(4):
            model.callbacks["on_train_batch_end"][0](fake)
        model.callbacks["on_train_epoch_end"][0](fake)
        model.callbacks["on_fit_epoch_end"][0](fake)
        model.callbacks["on_fit_epoch_end"][0](fake)
    finally:
        trainer.emit = original_emit

    progress = [payload for event_type, payload in events if event_type == "progress"]
    metrics = [payload for event_type, payload in events if event_type == "metric"]

    assert any(item["phase"] == "train" and item["batch"] == 4 and item["batches"] == 4 for item in progress)
    assert any(item["phase"] == "validate" and "liveMetrics" in item for item in progress)
    assert any(item["name"] == "boxLoss" and item["epoch"] == 1 for item in metrics)
    assert any(item["name"] == "mAP50" and item["value"] == 0.75 for item in metrics)
    assert sum(1 for item in metrics if item["name"] == "mAP50" and item["epoch"] == 1) == 1


def test_ultralytics_event_adapter_uses_sdk_and_preserves_backend() -> None:
    events: list[dict] = []
    original_adapter = trainer._adapter
    original_backend = trainer._adapter_backend
    original_channel = trainer._event_channel
    trainer._adapter = AdapterSdk("ultralytics_yolo_detect", event_sink=events.append)
    trainer._adapter_backend = "ultralytics_yolo_detect"
    trainer._event_channel = None
    try:
        trainer.emit("log", backend="spoofed", level="info", message="sdk event")
        trainer.emit("progress", percent=5, message="running", epoch=1)
        trainer.emit("artifact", kind="report", path="out/report.json", message="report")
        trainer.emit("completed", reportPath="out/report.json")
    finally:
        trainer._adapter = original_adapter
        trainer._adapter_backend = original_backend
        trainer._event_channel = original_channel

    assert [event["type"] for event in events] == ["log", "progress", "artifact", "completed"]
    assert all(event["backend"] == "ultralytics_yolo_detect" for event in events)
    assert events[1]["percent"] == 5.0


def test_official_evaluator_event_adapter_uses_sdk_and_preserves_backend() -> None:
    events: list[dict] = []
    original_adapter = evaluator._adapter
    original_backend = evaluator._adapter_backend
    original_channel = evaluator._event_channel
    evaluator._adapter = AdapterSdk("ultralytics_yolo_eval", event_sink=events.append)
    evaluator._adapter_backend = "ultralytics_yolo_eval"
    evaluator._event_channel = None
    try:
        evaluator.emit("artifact", backend="spoofed", kind="evaluation_report", path="out/evaluation_report.json")
        evaluator.emit("completed", reportPath="out/evaluation_report.json")
    finally:
        evaluator._adapter = original_adapter
        evaluator._adapter_backend = original_backend
        evaluator._event_channel = original_channel

    assert [event["type"] for event in events] == ["artifact", "completed"]
    assert all(event["backend"] == "ultralytics_yolo_eval" for event in events)


def test_official_evaluator_expands_directory_outputs_for_candidates() -> None:
    events: list[tuple[str, dict]] = []
    original_emit = evaluator.emit
    original_channel = evaluator._event_channel
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plot = root / "results.png"
        plot.write_bytes(b"png")
        evaluator.emit = lambda event_type, **payload: events.append((event_type, payload))
        evaluator._event_channel = object()  # type: ignore[assignment]
        try:
            evaluator.emit_official_artifacts(
                [{"name": "results.png", "kind": "official_plot", "relativePath": "results.png"}], root)
        finally:
            evaluator.emit = original_emit
            evaluator._event_channel = original_channel

    assert events == [("artifact", {
        "name": "results.png",
            "kind": "official_val_001_official_plot",
            "path": str(plot),
            "relativePath": "official_artifacts/results.png",
            "message": "Official Ultralytics validation output",
    })]


def test_official_evaluator_never_emits_directory_candidates() -> None:
    events: list[tuple[str, dict]] = []
    original_emit = evaluator.emit
    original_channel = evaluator._event_channel
    evaluator.emit = lambda event_type, **payload: events.append((event_type, payload))
    evaluator._event_channel = None
    try:
        evaluator.emit_official_artifacts([], Path("out/official_val"))
    finally:
        evaluator.emit = original_emit
        evaluator._event_channel = original_channel

    assert events == []


def test_dataset_snapshot_materialization_copies_only_verified_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        image = source / "images" / "sample.jpg"
        image.parent.mkdir()
        image.write_bytes(b"snapshot-data")
        digest = __import__("hashlib").sha256(image.read_bytes()).hexdigest()
        manifest = root / "dataset_snapshot.json"
        manifest.write_text(json.dumps({
            "schemaVersion": 2,
            "complete": True,
            "files": [{"relativePath": "images/sample.jpg", "sha256": digest}],
        }), encoding="utf-8")
        destination = root / "staging"
        result = materialize_dataset_snapshot(source, manifest, destination)

        assert result == destination.resolve()
        assert (destination / "images" / "sample.jpg").read_bytes() == b"snapshot-data"
        image.write_bytes(b"mutated")
        try:
            materialize_dataset_snapshot(source, manifest, root / "changed")
        except ValueError as exc:
            assert "does not match manifest" in str(exc)
        else:
            raise AssertionError("mutated dataset source was accepted")


def test_official_exporter_event_adapter_uses_sdk_and_preserves_backend() -> None:
    events: list[dict] = []
    original_adapter = exporter._adapter
    original_backend = exporter._adapter_backend
    original_channel = exporter._event_channel
    exporter._adapter = AdapterSdk("ultralytics_yolo_export", event_sink=events.append)
    exporter._adapter_backend = "ultralytics_yolo_export"
    exporter._event_channel = None
    try:
        exporter.emit("artifact", backend="spoofed", kind="export", path="out/model.onnx")
        exporter.emit("completed", exportPath="out/model.onnx")
    finally:
        exporter._adapter = original_adapter
        exporter._adapter_backend = original_backend
        exporter._event_channel = original_channel

    assert [event["type"] for event in events] == ["artifact", "completed"]
    assert all(event["backend"] == "ultralytics_yolo_export" for event in events)


def test_official_exporter_does_not_emit_legacy_model_export_frame() -> None:
    assert "modelExport" not in exporter.emit.__code__.co_consts


def test_ultralytics_train_args_are_sanitized_and_merged() -> None:
    kwargs = trainer.build_ultralytics_train_kwargs(
        {
            "trainingBackend": "ultralytics_yolo_detect",
            "epochs": 3,
            "batchSize": 2,
            "imageSize": 128,
            "seed": 42,
            "ultralyticsTrainArgs": {
                "device": "cpu",
                "optimizer": "AdamW",
                "lr0": "0.002",
                "classes": "0, 2",
                "copy_paste": "0.5",
            },
        },
        Path("data.yaml"),
        Path("runs"),
        "ultralytics_yolo_detect",
    )

    assert kwargs["epochs"] == 3
    assert kwargs["batch"] == 2
    assert kwargs["imgsz"] == 128
    assert kwargs["seed"] == 42
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == 0.002
    assert kwargs["classes"] == [0, 2]
    assert "copy_paste" not in kwargs


def test_ultralytics_segment_args_allow_mask_parameters() -> None:
    kwargs = trainer.build_ultralytics_train_kwargs(
        {
            "trainingBackend": "ultralytics_yolo_segment",
            "ultralyticsTrainArgs": {
                "copy_paste": "0.5",
                "overlap_mask": "false",
                "mask_ratio": "4",
            },
        },
        Path("data.yaml"),
        Path("runs"),
        "ultralytics_yolo_segment",
    )

    assert kwargs["copy_paste"] == 0.5
    assert kwargs["overlap_mask"] is False
    assert kwargs["mask_ratio"] == 4


def test_ultralytics_obb_args_use_official_backend_without_mask_parameters() -> None:
    kwargs = trainer.build_ultralytics_train_kwargs(
        {
            "trainingBackend": "ultralytics_yolo_obb",
            "model": "yolo11n-obb.pt",
            "epochs": 2,
            "batchSize": 4,
            "imageSize": 640,
            "device": "cpu",
            "ultralyticsTrainArgs": {
                "degrees": "15",
                "copy_paste": "0.5",
            },
        },
        Path("data.yaml"),
        Path("runs"),
        "ultralytics_yolo_obb",
    )

    assert kwargs["epochs"] == 2
    assert kwargs["batch"] == 4
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == "cpu"
    assert kwargs["modelName"] == "yolo11n-obb.pt"
    assert kwargs["degrees"] == 15.0
    assert "copy_paste" not in kwargs


def test_ultralytics_train_args_reject_unknown_keys() -> None:
    try:
        trainer.build_ultralytics_train_kwargs(
            {"ultralyticsTrainArgs": {"unknown_arg": 1}},
            Path("data.yaml"),
            Path("runs"),
            "ultralytics_yolo_detect",
        )
    except ValueError as exc:
        assert "unknown_arg" in str(exc)
    else:
        raise AssertionError("unknown Ultralytics train argument was accepted")


def test_official_val_metric_extraction_detection_and_segmentation() -> None:
    raw = {
        "metrics/precision(B)": 0.8,
        "metrics/recall(B)": 0.7,
        "metrics/mAP50(B)": 0.6,
        "metrics/mAP50-95(B)": 0.5,
        "metrics/precision(M)": 0.55,
        "metrics/recall(M)": 0.45,
        "metrics/mAP50(M)": 0.35,
        "metrics/mAP50-95(M)": 0.25,
    }

    detection = evaluator.extract_metrics(raw, "detection")
    segmentation = evaluator.extract_metrics(raw, "segmentation")

    assert detection["mAP50"] == 0.6
    assert detection["mAP50_95"] == 0.5
    assert segmentation["maskMap50"] == 0.35
    assert segmentation["maskMap50_95"] == 0.25


def test_official_val_metric_extraction_obb_uses_box_metrics() -> None:
    raw = {
        "metrics/precision(B)": 0.9,
        "metrics/recall(B)": 0.8,
        "metrics/mAP50(B)": 0.7,
        "metrics/mAP50-95(B)": 0.6,
    }

    obb = evaluator.extract_metrics(raw, "obb")

    assert obb["precision"] == 0.9
    assert obb["recall"] == 0.8
    assert obb["mAP50"] == 0.7
    assert obb["mAP50_95"] == 0.6


def test_ultralytics_export_args_are_sanitized() -> None:
    plan = exporter.build_export_plan(
        {
            "ultralyticsExportArgs": {
                "format": "onnx",
                "dynamic": "true",
                "half": "false",
                "int8": False,
                "imgsz": "640",
                "batch": "2",
                "device": "cpu",
            }
        }
    )

    assert plan["productFormat"] == "onnx"
    assert plan["officialFormat"] == "onnx"
    assert plan["kwargs"]["dynamic"] is True
    assert plan["kwargs"]["half"] is False
    assert plan["kwargs"]["int8"] is False
    assert plan["kwargs"]["imgsz"] == 640
    assert plan["kwargs"]["batch"] == 2
    assert plan["kwargs"]["device"] == "cpu"


def test_training_export_defaults_to_single_image_batch() -> None:
    plan = exporter.build_export_plan(
        {},
        default_format="onnx",
        default_imgsz=128,
        default_batch=1,
        default_device="cpu",
    )

    assert plan["productFormat"] == "onnx"
    assert plan["kwargs"]["batch"] == 1
    assert plan["normalized"]["batch"] == 1


def test_cpu_device_environment_uses_default_cpu() -> None:
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        exporter.apply_cpu_device_environment({}, default_device="cpu")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def test_yolo26_detection_auto_end2end_defaults_true() -> None:
    plan = exporter.build_export_plan(
        {"ultralyticsExportArgs": {"format": "onnx", "end2end": "auto"}},
        model_name="yolo26n.yaml",
        model_family="yolo_detection",
    )

    assert plan["normalized"]["end2end"] is True
    assert plan["kwargs"]["end2end"] is True


def test_yolo26_segmentation_auto_end2end_defaults_false() -> None:
    plan = exporter.build_export_plan(
        {"ultralyticsExportArgs": {"format": "onnx", "end2end": "auto"}},
        model_name="yolo26n-seg.pt",
        model_family="yolo_segmentation",
    )

    assert plan["normalized"]["end2end"] is False
    assert plan["kwargs"]["end2end"] is False


def test_yolo26_segmentation_auto_uses_model_end2end_hint() -> None:
    plan = exporter.build_export_plan(
        {"ultralyticsExportArgs": {"format": "onnx", "end2end": "auto"}},
        model_name="yolo26n-seg.yaml",
        model_family="yolo_segmentation",
        auto_end2end=True,
    )

    assert plan["normalized"]["end2end"] is True
    assert plan["kwargs"]["end2end"] is True
    assert plan["modelSeries"] == "yolo26"
    assert plan["task"] == "segmentation"


def test_non_yolo26_auto_false_does_not_emit_end2end_kwarg() -> None:
    plan = exporter.build_export_plan(
        {"ultralyticsExportArgs": {"format": "onnx", "end2end": "auto"}},
        model_name="yolov8n.yaml",
        model_family="yolo_detection",
        auto_end2end=False,
    )

    assert plan["normalized"]["end2end"] is False
    assert "end2end" not in plan["kwargs"]


def test_ultralytics_export_args_accept_bool_end2end_override() -> None:
    plan = exporter.build_export_plan(
        {"ultralyticsExportArgs": {"format": "onnx", "end2end": True}},
        model_name="yolo26n-seg.yaml",
        model_family="yolo_segmentation",
    )

    assert plan["normalized"]["end2end"] is True
    assert plan["kwargs"]["end2end"] is True


def test_yolo26_ncnn_export_is_rejected() -> None:
    try:
        exporter.build_export_plan(
            {"ultralyticsExportArgs": {"format": "ncnn", "end2end": "auto"}},
            model_name="yolo26n.yaml",
            model_family="yolo_detection",
        )
    except ValueError as exc:
        assert "YOLO26 NCNN export is not supported" in str(exc)
    else:
        raise AssertionError("YOLO26 NCNN export was accepted")


def test_yolo_obb_export_defaults_to_obb_task_and_onnx() -> None:
    plan = exporter.build_export_plan(
        {"model": "yolo11n-obb.pt"},
        default_format="onnx",
        default_imgsz=640,
        default_batch=1,
        default_device="cpu",
    )

    assert plan["productFormat"] == "onnx"
    assert plan["officialFormat"] == "onnx"
    assert plan["modelFamily"] == "yolo_obb"
    assert plan["task"] == "obb_detection"
    assert plan["kwargs"]["imgsz"] == 640
    assert plan["kwargs"]["batch"] == 1


def test_yolo_obb_ncnn_export_is_rejected() -> None:
    try:
        exporter.build_export_plan(
            {"ultralyticsExportArgs": {"format": "ncnn"}, "model": "yolo11n-obb.pt"},
            model_family="yolo_obb",
        )
    except ValueError as exc:
        assert "OBB NCNN export is not supported" in str(exc)
    else:
        raise AssertionError("YOLO OBB NCNN export was accepted")


def test_ultralytics_export_args_reject_unknown_keys() -> None:
    try:
        exporter.build_export_plan({"ultralyticsExportArgs": {"unknown": 1}})
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown Ultralytics export argument was accepted")


def test_ultralytics_export_args_reject_unsupported_combinations() -> None:
    for raw in [
        {"format": "onnx", "int8": True},
        {"format": "ncnn", "dynamic": True},
        {"format": "ncnn", "half": True},
        {"format": "ncnn", "int8": True},
        {"format": "ncnn", "end2end": True},
        {"format": "tensorrt", "int8": True},
    ]:
        try:
            exporter.build_export_plan({"ultralyticsExportArgs": raw})
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported export combination was accepted: {raw}")


def test_ultralytics_export_args_accept_tensorrt_int8_with_data() -> None:
    plan = exporter.build_export_plan(
        {
            "ultralyticsExportArgs": {
                "format": "tensorrt",
                "int8": True,
                "data": "aitrain_yolo_data.yaml",
                "imgsz": 640,
                "batch": 1,
                "device": "0",
            }
        }
    )

    assert plan["productFormat"] == "tensorrt"
    assert plan["officialFormat"] == "engine"
    assert plan["kwargs"]["int8"] is True
    assert plan["kwargs"]["data"] == "aitrain_yolo_data.yaml"


def test_exporter_infers_segmentation_family_from_training_report() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        weights_dir = root / "ultralytics_runs" / "aitrain-yolo-segment" / "weights"
        weights_dir.mkdir(parents=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_text("fake checkpoint\n", encoding="utf-8")
        report_path = root / "ultralytics_training_report.json"
        report_path.write_text(
            json.dumps({"backend": "ultralytics_yolo_segment", "model": "yolov8n-seg.yaml"}),
            encoding="utf-8",
        )

        family, found_report, report = exporter.infer_model_family(best_pt, None, {})

    assert family == "yolo_segmentation"
    assert found_report == report_path
    assert report["backend"] == "ultralytics_yolo_segment"


def test_exporter_infers_obb_family_from_training_report() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        weights_dir = root / "ultralytics_runs" / "aitrain-yolo-obb" / "weights"
        weights_dir.mkdir(parents=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_text("fake checkpoint\n", encoding="utf-8")
        report_path = root / "ultralytics_training_report.json"
        report_path.write_text(
            json.dumps({"backend": "ultralytics_yolo_obb", "model": "yolo11n-obb.pt"}),
            encoding="utf-8",
        )

        family, found_report, report = exporter.infer_model_family(best_pt, None, {})

    assert family == "yolo_obb"
    assert found_report == report_path
    assert report["backend"] == "ultralytics_yolo_obb"


def test_exporter_builds_contract_only_from_official_evaluation_evidence() -> None:
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        evaluation_path = root / "evaluation_report.json"
        evaluation_path.write_text(json.dumps({
            "runtime": "ultralytics_official_val",
            "perClass": [
                {"classId": 1, "className": "scratch"},
                {"classId": 0, "className": "part"},
            ],
        }), encoding="utf-8")
        contract = exporter.model_contract("yolo_detection", {
            "available": True,
            "inputs": [{"name": "images", "shape": [1, 3, 640, 640]}],
            "outputs": [{"name": "output0", "shape": [1, 84, "anchors"]}],
        }, evaluation_path)

    assert contract["modelFamily"] == "yolo_detection"
    assert contract["taskType"] == "detection"
    assert contract["inputs"] == [{"name": "images", "layout": "NCHW", "shape": [1, 3, 640, 640]}]
    assert contract["outputs"] == [{"name": "output0", "layout": "NCN", "shape": [1, 84, -1]}]
    assert contract["classNames"] == ["part", "scratch"]
    assert contract["runtimeRoutes"] == ["aitrain_onnxruntime"]


def test_exporter_builds_variant_specific_contracts() -> None:
    segmentation = exporter.model_contract("yolo_segmentation", {
        "inputs": [{"name": "images", "shape": [1, 3, 640, 640]}],
        "outputs": [
            {"name": "output0", "shape": [1, 37, "anchors"]},
            {"name": "output1", "shape": [1, 32, 160, 160]},
        ],
    }, None)
    assert segmentation["taskType"] == "segmentation"
    assert segmentation["decoder"] == "yolo_segmentation_v8"
    assert segmentation["postprocessing"] == {"id": "yolo_segmentation_masks_v8"}
    assert [item["layout"] for item in segmentation["outputs"]] == ["NCN", "NCHW"]

    obb = exporter.model_contract("yolo_obb", {
        "inputs": [{"name": "images", "shape": [1, 3, 640, 640]}],
        "outputs": [{"name": "output0", "shape": [1, 6, "anchors"]}],
    }, None)
    assert obb["taskType"] == "obb_detection"
    assert obb["decoder"] == "yolo_obb_v8"
    assert obb["postprocessing"] == {"id": "yolo_obb_nms"}
    assert obb["runtimeRoutes"] == ["aitrain_onnxruntime"]


if __name__ == "__main__":
    os.environ.setdefault("AITRAIN_STANDALONE_ADAPTER_PROTOCOL", "1")
    test_sanitize_log_line_removes_ansi_tqdm_noise()
    test_yolo_callbacks_emit_structured_progress_and_epoch_metrics()
    test_ultralytics_train_args_are_sanitized_and_merged()
    test_ultralytics_segment_args_allow_mask_parameters()
    test_ultralytics_obb_args_use_official_backend_without_mask_parameters()
    test_ultralytics_train_args_reject_unknown_keys()
    test_official_val_metric_extraction_detection_and_segmentation()
    test_official_val_metric_extraction_obb_uses_box_metrics()
    test_ultralytics_export_args_are_sanitized()
    test_training_export_defaults_to_single_image_batch()
    test_cpu_device_environment_uses_default_cpu()
    test_yolo26_detection_auto_end2end_defaults_true()
    test_yolo26_segmentation_auto_end2end_defaults_false()
    test_yolo26_segmentation_auto_uses_model_end2end_hint()
    test_non_yolo26_auto_false_does_not_emit_end2end_kwarg()
    test_ultralytics_export_args_accept_bool_end2end_override()
    test_yolo26_ncnn_export_is_rejected()
    test_yolo_obb_export_defaults_to_obb_task_and_onnx()
    test_yolo_obb_ncnn_export_is_rejected()
    test_ultralytics_export_args_reject_unknown_keys()
    test_ultralytics_export_args_reject_unsupported_combinations()
    test_ultralytics_export_args_accept_tensorrt_int8_with_data()
    test_exporter_infers_segmentation_family_from_training_report()
    test_exporter_infers_obb_family_from_training_report()
    test_exporter_builds_contract_only_from_official_evaluation_evidence()
