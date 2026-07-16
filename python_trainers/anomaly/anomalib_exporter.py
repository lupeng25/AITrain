#!/usr/bin/env python3
"""Package an Anomalib checkpoint for the Worker-managed Python runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from adapter_event_channel_v2 import AdapterEventChannelV2, event_channel_from_environment  # noqa: E402
from adapter_sdk import AdapterSdk  # noqa: E402
from trainer_protocol import configure_stdio, exception_details  # noqa: E402


BACKEND_ID = "anomalib_bundle_export"
EXPORTER_VERSION = "aitrain-anomalib-bundle-exporter-v2"
SUPPORTED_BACKENDS = {"anomalib_patchcore", "anomalib_efficientad"}

configure_stdio()
_adapter: AdapterSdk | None = None
_event_channel: AdapterEventChannelV2 | None = None


def configure_adapter() -> AdapterSdk:
    global _adapter, _event_channel
    if _event_channel is None and os.environ.get("AITRAIN_EVENT_PORT"):
        _event_channel = event_channel_from_environment()
        _event_channel.connect()
    if _adapter is None:
        sink = _event_channel.emit_legacy_event if _event_channel is not None else None
        _adapter = AdapterSdk(BACKEND_ID, event_sink=sink)
    return _adapter


def close_adapter() -> None:
    global _adapter, _event_channel
    if _event_channel is not None:
        _event_channel.close()
    _adapter = None
    _event_channel = None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def locate_sidecar(request: dict[str, Any], model_path: Path) -> tuple[Path, dict[str, Any]]:
    options = request.get("options") if isinstance(request.get("options"), dict) else {}
    explicit = request.get("sidecarPath") or options.get("sidecarPath")
    candidates = [Path(str(explicit)).resolve()] if explicit else []
    if model_path.name == "anomaly_sidecar.json":
        candidates.append(model_path)
    candidates.extend((model_path.parent / "anomaly_sidecar.json", model_path.parent / "model" / "anomaly_sidecar.json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate, read_json(candidate)
    raise FileNotFoundError("Anomalib export requires anomaly_sidecar.json from the training step.")


def locate_checkpoint(request: dict[str, Any], model_path: Path, sidecar_path: Path, sidecar: dict[str, Any]) -> Path:
    options = request.get("options") if isinstance(request.get("options"), dict) else {}
    raw_value = request.get("checkpointPath") or options.get("checkpointPath") or sidecar.get("checkpointPath")
    if raw_value:
        raw = Path(str(raw_value))
        candidate = raw if raw.is_absolute() else sidecar_path.parent / raw
        if candidate.is_file():
            return candidate.resolve()
    if model_path.is_file() and model_path.suffix.lower() == ".ckpt":
        return model_path
    raise FileNotFoundError("Anomalib export requires an existing .ckpt checkpoint.")


def run(request: dict[str, Any]) -> int:
    adapter = configure_adapter()
    model_path = Path(str(request.get("modelPath") or request.get("checkpointPath") or "")).resolve()
    requested_output = Path(str(request.get("outputPath") or "")).resolve()
    if not str(request.get("outputPath") or "").strip():
        raise ValueError("outputPath is required")
    sidecar_path, source = locate_sidecar(request, model_path)
    checkpoint = locate_checkpoint(request, model_path, sidecar_path, source)
    source_backend = str(source.get("trainingBackend") or request.get("sourceTrainingBackend") or "").strip().lower()
    if source_backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported Anomalib training backend: {source_backend}")

    if requested_output.suffix.lower() == ".json":
        package_dir = requested_output.parent
        output_sidecar = requested_output
    else:
        package_dir = requested_output
        output_sidecar = package_dir / "anomaly_sidecar.json"
    package_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = package_dir / "model.ckpt"
    if checkpoint != output_checkpoint:
        shutil.copy2(checkpoint, output_checkpoint)

    threshold = float(source.get("threshold") or 0.5)
    preprocessing = source.get("preprocessing") if isinstance(source.get("preprocessing"), dict) else {}
    if not preprocessing:
        preprocessing = {
            "id": "anomalib_folder_rgb",
            "colorSpace": "RGB",
            "imageSize": int((source.get("parameters") or {}).get("imageSize") or 256),
        }
    postprocessing = source.get("postprocessing") if isinstance(source.get("postprocessing"), dict) else {}
    if not postprocessing:
        postprocessing = {
            "id": "anomalib_score_threshold_v1",
            "threshold": threshold,
            "decision": {"ok": "score <= threshold", "ng": "score > threshold"},
        }
    exported_at = now_iso()
    bundle = {
        "schemaVersion": 2,
        "kind": "anomalib_bundle",
        "artifactFormat": "anomalib_bundle",
        "modelFamily": "anomaly_detection",
        "taskType": "anomaly_detection",
        "sourceTrainingBackend": source_backend,
        "trainingBackend": source_backend,
        "runtimeRoutes": ["anomalib_python"],
        "runtime": "anomalib_python",
        "decoder": "anomalib_python_sidecar_v1",
        "exporterVersion": EXPORTER_VERSION,
        "checkpointPath": "model.ckpt",
        "classNames": ["normal", "anomaly"],
        "preprocessing": preprocessing,
        "postprocessing": postprocessing,
        "threshold": threshold,
        "parameters": source.get("parameters") if isinstance(source.get("parameters"), dict) else {},
        "sourceTrainingReportPath": str(source.get("trainingReportPath") or ""),
        "exportedAt": exported_at,
        "limitations": [
            "This bundle runs only through the Worker-managed Python/Anomalib runtime.",
            "It does not claim AITrain C++ ONNX Runtime, TensorRT, or NCNN support.",
        ],
        "artifactContract": {
            "sidecar": "anomaly_sidecar.json",
            "checkpoint": "model.ckpt",
            "pathsArePackageRelative": True,
        },
    }
    write_json(output_sidecar, bundle)
    report_path = package_dir / "anomalib_export_report.json"
    write_json(
        report_path,
        {
            "schemaVersion": 2,
            "kind": "anomalib_export_report",
            "ok": True,
            "artifactFormat": "anomalib_bundle",
            "runtime": "anomalib_python",
            "sourceTrainingBackend": source_backend,
            "sidecarPath": str(output_sidecar),
            "checkpointPath": str(output_checkpoint),
            "exportedAt": exported_at,
            "runtimeBoundary": "Worker-managed Python/Anomalib only; no C++ ONNX/TensorRT/NCNN runtime claim.",
        },
    )
    adapter.emit_progress(90, message="Anomalib bundle packaged.")
    adapter.emit_artifact_candidate("export", output_sidecar, message="Anomalib bundle contract sidecar")
    adapter.emit_artifact_candidate("export", output_checkpoint, message="Anomalib bundle checkpoint")
    adapter.emit_artifact_candidate("export_report", report_path, message="Anomalib bundle export report")
    adapter.emit_progress(100, message="Anomalib bundle export completed.")
    adapter.emit_completed(
        "Anomalib bundle export completed.",
        exportPath=str(output_sidecar),
        sidecarPath=str(output_sidecar),
        checkpointPath=str(output_checkpoint),
        reportPath=str(report_path),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        try:
            request = read_json(args.request)
        except Exception as exc:
            configure_adapter().emit_failed(f"Failed to read export request: {exc}", "bad_request", exception_details(exc))
            return 2
        try:
            return run(request)
        except Exception as exc:
            configure_adapter().emit_failed(str(exc), "anomalib_export_failed", exception_details(exc))
            return 1
    finally:
        close_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
