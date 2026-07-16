#!/usr/bin/env python3
"""SMP semantic segmentation ONNX evaluator for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from adapter_event_channel_v2 import AdapterEventChannelV2, event_channel_from_environment  # noqa: E402
from adapter_sdk import AdapterSdk  # noqa: E402
from dataset_snapshot_v2 import materialize_dataset_snapshot_v2  # noqa: E402
from trainer_protocol import configure_stdio, exception_details  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BACKEND_ID = "smp_semantic_segmentation_eval"

configure_stdio()

_adapter: AdapterSdk | None = None
_event_channel: AdapterEventChannelV2 | None = None


def configure_adapter() -> None:
    global _adapter, _event_channel
    if _event_channel is None and os.environ.get("AITRAIN_EVENT_PORT"):
        _event_channel = event_channel_from_environment()
        _event_channel.connect()
    if _adapter is None:
        sink = _event_channel.emit_legacy_event if _event_channel is not None else None
        _adapter = AdapterSdk(BACKEND_ID, event_sink=sink)


def close_adapter() -> None:
    global _event_channel
    if _event_channel is not None:
        _event_channel.close()
        _event_channel = None


def active_adapter() -> AdapterSdk:
    configure_adapter()
    assert _adapter is not None
    return _adapter


def emit(event_type: str, **payload: Any) -> None:
    payload.pop("backend", None)
    adapter = active_adapter()
    if event_type == "log":
        adapter.emit_log(str(payload.pop("message", "")), level=str(payload.pop("level", "info")), **payload)
    elif event_type == "progress":
        adapter.emit_progress(float(payload.pop("percent", 0)), message=str(payload.pop("message", "")), **payload)
    elif event_type == "metric":
        adapter.emit_metric(str(payload.pop("name", "")), float(payload.pop("value", 0)), **payload)
    elif event_type == "artifact":
        adapter.emit_artifact_candidate(
            str(payload.pop("kind", "artifact")),
            str(payload.pop("path", "")),
            message=str(payload.pop("message", "")),
            **payload,
        )
    elif event_type == "completed":
        adapter.emit_completed(str(payload.pop("message", "SMP evaluation completed")), **payload)
    elif event_type == "failed":
        adapter.emit_failed(
            str(payload.pop("message", "SMP evaluation failed")),
            str(payload.pop("code", "smp_evaluation_failed")),
            payload.pop("details", {}),
        )
    else:
        raise ValueError(f"unsupported adapter event type: {event_type}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def materialize_request_dataset(request: dict[str, Any], options: dict[str, Any], dataset_path: Path) -> tuple[Path, str]:
    snapshot_manifest = str(options.get("datasetSnapshotManifest") or request.get("datasetSnapshotManifest") or "").strip()
    snapshot_staging = str(options.get("datasetSnapshotStagingPath") or request.get("datasetSnapshotStagingPath") or "").strip()
    if bool(snapshot_manifest) != bool(snapshot_staging):
        raise ValueError("Dataset Snapshot V2 requires both manifest and staging paths.")
    if snapshot_manifest:
        dataset_path = materialize_dataset_snapshot_v2(dataset_path, snapshot_manifest, snapshot_staging)
    return dataset_path, snapshot_manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return to_jsonable(value.item())
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
    except Exception:
        pass
    return str(value)


def read_classes(dataset_path: Path) -> list[str]:
    classes_path = dataset_path / "classes.txt"
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not classes:
        raise ValueError("classes.txt must contain at least one class name")
    return classes


def load_mask_ids(mask_path: Path) -> Any:
    """Load raw semantic class IDs without converting palette colors to grayscale."""
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    mask = Image.open(mask_path)
    if mask.mode not in {"L", "P"}:
        raise ValueError(f"Semantic mask must use L or P mode: {mask_path}")
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Semantic mask must be single-channel: {mask_path}")
    return values.astype(np.int64, copy=False)


def split_samples(dataset_path: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_path / "images" / split
    mask_dir = dataset_path / "masks" / split
    if not image_dir.exists() or not mask_dir.exists():
        return []
    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(item for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            samples.append((image_path, mask_path))
    return samples


def resolve_onnx(model_path: Path) -> Path:
    if model_path.suffix.lower() == ".onnx":
        return model_path
    sibling = model_path.with_suffix(".onnx")
    if sibling.exists():
        return sibling
    sibling = model_path.parent / "best.onnx"
    if sibling.exists():
        return sibling
    raise ValueError(f"SMP evaluation requires an ONNX model or best.onnx beside the checkpoint: {model_path}")


def sidecar_for_onnx(onnx_path: Path, explicit_path: Path | None = None) -> dict[str, Any]:
    if explicit_path is not None and explicit_path.is_file():
        return read_json(explicit_path)
    for candidate in sidecar_candidates(onnx_path):
        if candidate.exists():
            return read_json(candidate)
    return {}


def sidecar_candidates(onnx_path: Path) -> list[Path]:
    return [
        onnx_path.with_suffix(".aitrain-export.json"),
        onnx_path.parent / "semantic_segmentation_sidecar.json",
    ]


def existing_sidecar(onnx_path: Path, explicit_path: Path | None = None) -> Path | None:
    if explicit_path is not None and explicit_path.is_file():
        return explicit_path
    return next((candidate for candidate in sidecar_candidates(onnx_path) if candidate.is_file()), None)


def metrics_from_confusion(confusion: Any, class_names: list[str]) -> dict[str, Any]:
    import numpy as np  # type: ignore

    cm = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    dice = np.divide(2 * tp, denom_dice, out=np.zeros_like(tp), where=denom_dice > 0)
    total = cm.sum()
    per_class = []
    for index, name in enumerate(class_names):
        per_class.append(
            {
                "classId": index,
                "className": name,
                "iou": float(iou[index]) if index < iou.size else 0.0,
                "dice": float(dice[index]) if index < dice.size else 0.0,
                "tp": int(tp[index]) if index < tp.size else 0,
                "fp": int(fp[index]) if index < fp.size else 0,
                "fn": int(fn[index]) if index < fn.size else 0,
            }
        )
    return {
        "mIoU": float(iou.mean()) if iou.size else 0.0,
        "meanDice": float(dice.mean()) if dice.size else 0.0,
        "pixelAccuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "perClass": per_class,
    }


def overlay_image(image_path: Path, prediction: Any, class_names: list[str], output_path: Path) -> None:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    palette = [
        (0, 0, 0, 0),
        (220, 40, 40, 95),
        (36, 145, 255, 95),
        (15, 174, 102, 95),
        (180, 92, 220, 95),
        (245, 158, 11, 95),
        (14, 165, 233, 95),
        (236, 72, 153, 95),
    ]
    image = Image.open(image_path).convert("RGBA")
    pred = np.asarray(prediction, dtype=np.uint8)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    for y in range(pred.shape[0]):
        for x in range(pred.shape[1]):
            class_id = int(pred[y, x])
            if class_id <= 0:
                continue
            overlay_pixels[x, y] = palette[class_id % len(palette)]
    blended = Image.alpha_composite(image, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)


def evaluate(request: dict[str, Any]) -> int:
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    from PIL import Image  # type: ignore

    model_path = Path(str(request.get("modelPath") or "")).resolve()
    dataset_path = Path(str(request.get("datasetPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or "")).resolve()
    options = request.get("options") if isinstance(request.get("options"), dict) else {}
    split = str(options.get("split") or "val")
    ignore_index = int(options.get("ignoreIndex", 255))
    max_overlays = max(0, int(options.get("maxOverlays", 12)))
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path, snapshot_manifest = materialize_request_dataset(request, options, dataset_path)

    onnx_path = resolve_onnx(model_path)
    requested_sidecar = str(request.get("sidecarPath") or options.get("sidecarPath") or "").strip()
    explicit_sidecar = Path(requested_sidecar).resolve() if requested_sidecar else None
    sidecar = sidecar_for_onnx(onnx_path, explicit_sidecar)
    sidecar_path = existing_sidecar(onnx_path, explicit_sidecar)
    class_names = [str(item) for item in sidecar.get("classNames", []) if str(item)]
    if not class_names:
        class_names = read_classes(dataset_path)
    class_count = len(class_names)
    input_width = int(sidecar.get("inputWidth") or sidecar.get("imageSize") or 0)
    input_height = int(sidecar.get("inputHeight") or sidecar.get("imageSize") or input_width)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    if not input_width or not input_height:
        input_height = int(input_shape[2]) if len(input_shape) >= 4 and isinstance(input_shape[2], int) else 256
        input_width = int(input_shape[3]) if len(input_shape) >= 4 and isinstance(input_shape[3], int) else input_height

    samples = split_samples(dataset_path, split)
    if not samples:
        raise ValueError(f"No semantic segmentation samples found for split '{split}'")

    mean = np.asarray(sidecar.get("normalization", {}).get("mean", MEAN), dtype=np.float32)
    std = np.asarray(sidecar.get("normalization", {}).get("std", STD), dtype=np.float32)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    low_quality: list[dict[str, Any]] = []
    overlay_dir = output_path / "overlays"
    overlay_count = 0

    for image_path, mask_path in samples:
        image = Image.open(image_path).convert("RGB")
        source_size = image.size
        resized = image.resize((input_width, input_height), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - mean) / std
        tensor = array.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        logits = session.run(None, {input_name: tensor})[0]
        pred = np.asarray(logits).argmax(axis=1)[0].astype(np.uint8)
        pred_image = Image.fromarray(pred, mode="L").resize(source_size, Image.NEAREST)
        pred_array = np.asarray(pred_image, dtype=np.int64)
        target = load_mask_ids(mask_path)
        valid = target != ignore_index
        encoded = target[valid] * class_count + pred_array[valid]
        counts = np.bincount(encoded, minlength=class_count * class_count).reshape(class_count, class_count)
        confusion += counts
        sample_metrics = metrics_from_confusion(counts, class_names)
        sample_miou = float(sample_metrics["mIoU"])
        if sample_miou < float(options.get("lowQualityThreshold", 0.5)):
            item = {
                "imagePath": str(image_path),
                "maskPath": str(mask_path),
                "mIoU": sample_miou,
                "meanDice": sample_metrics["meanDice"],
                "reason": "low_semantic_iou",
            }
            if overlay_count < max_overlays:
                overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
                overlay_image(image_path, pred_array, class_names, overlay_path)
                item["overlayPath"] = str(overlay_path)
                overlay_count += 1
            low_quality.append(item)

    metrics = metrics_from_confusion(confusion, class_names)
    per_class_path = output_path / "per_class_metrics.json"
    confusion_path = output_path / "confusion_matrix.json"
    low_quality_path = output_path / "low_quality_samples.json"
    write_json(per_class_path, {"perClass": metrics["perClass"]})
    write_json(confusion_path, {"matrix": confusion.tolist(), "classNames": class_names})
    write_json(low_quality_path, {"samples": low_quality})

    summary_path = output_path / "evaluation_summary.md"
    write_text(
        summary_path,
        "# SMP Semantic Segmentation Evaluation\n\n"
        f"- Samples: {len(samples)}\n"
        f"- mIoU: {metrics['mIoU']:.6f}\n"
        f"- meanDice: {metrics['meanDice']:.6f}\n"
        f"- pixelAccuracy: {metrics['pixelAccuracy']:.6f}\n",
    )

    report = {
        "ok": True,
        "kind": "evaluation_report",
        "createdAt": now_iso(),
        "modelPath": str(model_path),
        "onnxPath": str(onnx_path),
        "datasetPath": str(dataset_path),
        "datasetSnapshotId": str(options.get("datasetSnapshotId") or request.get("datasetSnapshotId") or ""),
        "datasetSnapshotHash": str(options.get("datasetSnapshotHash") or request.get("datasetSnapshotHash") or ""),
        "datasetSnapshotManifest": snapshot_manifest,
        "taskType": "semantic_segmentation",
        "datasetFormat": "semantic_segmentation_mask",
        "runtime": "onnxruntime",
        "evaluationSource": "smp_onnxruntime",
        "status": "completed",
        "scaffold": False,
        "sampleCount": len(samples),
        "split": split,
        "classNames": class_names,
        "ignoreIndex": ignore_index,
        "metrics": {key: value for key, value in metrics.items() if key != "perClass"},
        "perClass": metrics["perClass"],
        "confusionMatrix": confusion.tolist(),
        "lowQualitySamples": low_quality,
        "errorSamples": low_quality,
        "perClassMetricsPath": str(per_class_path),
        "confusionMatrixPath": str(confusion_path),
        "errorSamplesPath": str(low_quality_path),
        "overlayDir": str(overlay_dir),
        "evaluationSummaryPath": str(summary_path),
    }
    report_path = output_path / "evaluation_report.json"
    report["reportPath"] = str(report_path)
    write_json(report_path, report)
    artifacts = [
        ("onnx_model", onnx_path, "Verified SMP ONNX model for downstream export"),
        ("evaluation_report", report_path, "SMP semantic segmentation evaluation report"),
        ("per_class_metrics", per_class_path, "SMP per-class evaluation metrics"),
        ("confusion_matrix", confusion_path, "SMP evaluation confusion matrix"),
        ("error_samples", low_quality_path, "SMP low-quality sample inventory"),
        ("evaluation_summary", summary_path, "SMP evaluation summary"),
    ]
    if sidecar_path is not None:
        artifacts.insert(1, ("model_sidecar", sidecar_path, "Verified SMP model sidecar for downstream export"))
    if model_path.is_file() and model_path != onnx_path:
        artifacts.insert(0, ("checkpoint", model_path, "Verified SMP checkpoint for downstream export"))
    for kind, path, message in artifacts:
        emit("artifact", kind=kind, path=str(path), message=message)
    emit("completed", reportPath=str(report_path), onnxPath=str(onnx_path), metrics=report["metrics"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="Path to AITrain SMP evaluation request JSON")
    args = parser.parse_args(argv)
    try:
        configure_adapter()
        try:
            request = read_json(Path(args.request))
        except Exception as exc:
            emit("failed", code="bad_request", message=f"failed to read evaluation request: {exc}", details=exception_details(exc))
            return 2
        try:
            return evaluate(request)
        except Exception as exc:
            output_path = Path(str(request.get("outputPath") or ".")).resolve()
            report = {
                "ok": False,
                "kind": "evaluation_report",
                "createdAt": now_iso(),
                "taskType": "semantic_segmentation",
                "runtime": "onnxruntime",
                "status": "failed",
                "failureCategory": "smp_evaluation_failed",
                "message": str(exc),
            }
            report_path = output_path / "evaluation_report.json"
            write_json(report_path, report)
            emit(
                "failed",
                code="smp_evaluation_failed",
                message=str(exc),
                details={**exception_details(exc), "reportPath": str(report_path)},
            )
            return 1
    finally:
        close_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
