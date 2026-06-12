#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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


def test_ultralytics_export_args_accept_bool_end2end_override() -> None:
    plan = exporter.build_export_plan(
        {"ultralyticsExportArgs": {"format": "onnx", "end2end": True}},
        model_name="yolo26n-seg.yaml",
        model_family="yolo_segmentation",
    )

    assert plan["normalized"]["end2end"] is True
    assert plan["kwargs"]["end2end"] is True


def test_yolo26_ncnn_auto_forces_traditional_end2end_false() -> None:
    plan = exporter.build_export_plan(
        {"ultralyticsExportArgs": {"format": "ncnn", "end2end": "auto"}},
        model_name="yolo26n.yaml",
        model_family="yolo_detection",
    )

    assert plan["productFormat"] == "ncnn"
    assert plan["normalized"]["end2end"] is False
    assert plan["kwargs"]["end2end"] is False


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


if __name__ == "__main__":
    test_sanitize_log_line_removes_ansi_tqdm_noise()
    test_yolo_callbacks_emit_structured_progress_and_epoch_metrics()
    test_ultralytics_train_args_are_sanitized_and_merged()
    test_ultralytics_segment_args_allow_mask_parameters()
    test_ultralytics_train_args_reject_unknown_keys()
    test_official_val_metric_extraction_detection_and_segmentation()
    test_ultralytics_export_args_are_sanitized()
    test_training_export_defaults_to_single_image_batch()
    test_cpu_device_environment_uses_default_cpu()
    test_yolo26_detection_auto_end2end_defaults_true()
    test_yolo26_segmentation_auto_end2end_defaults_false()
    test_ultralytics_export_args_accept_bool_end2end_override()
    test_yolo26_ncnn_auto_forces_traditional_end2end_false()
    test_ultralytics_export_args_reject_unknown_keys()
    test_ultralytics_export_args_reject_unsupported_combinations()
    test_ultralytics_export_args_accept_tensorrt_int8_with_data()
    test_exporter_infers_segmentation_family_from_training_report()
