#!/usr/bin/env python3
"""Normalize an SMP ONNX export and write its verified V2 model contract."""

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


BACKEND_ID = "smp_semantic_segmentation_export"
EXPORTER_VERSION = "aitrain-smp-exporter-v2"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
    if event_type == "artifact":
        adapter.emit_artifact_candidate(
            str(payload.pop("kind", "artifact")),
            str(payload.pop("path", "")),
            message=str(payload.pop("message", "")),
            **payload,
        )
    elif event_type == "progress":
        adapter.emit_progress(float(payload.pop("percent", 0)), message=str(payload.pop("message", "")), **payload)
    elif event_type == "completed":
        adapter.emit_completed(str(payload.pop("message", "SMP export completed")), **payload)
    elif event_type == "failed":
        adapter.emit_failed(
            str(payload.pop("message", "SMP export failed")),
            str(payload.pop("code", "smp_export_failed")),
            payload.pop("details", {}),
        )
    else:
        raise ValueError(f"unsupported adapter event type: {event_type}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_onnx(model_path: Path) -> Path:
    if model_path.is_file() and model_path.suffix.lower() == ".onnx":
        return model_path
    for candidate in (model_path.with_suffix(".onnx"), model_path.parent / "best.onnx"):
        if candidate.is_file():
            return candidate
    raise ValueError(f"SMP export requires an existing ONNX model: {model_path}")


def source_sidecar(onnx_path: Path, requested_path: Any = None) -> dict[str, Any]:
    candidates = [Path(str(requested_path)).resolve()] if requested_path else []
    candidates.extend((onnx_path.with_suffix(".aitrain-export.json"), onnx_path.parent / "semantic_segmentation_sidecar.json"))
    for candidate in candidates:
        if candidate.is_file():
            return read_json(candidate)
    return {}


def class_names_from_evaluation(path_value: Any) -> list[str]:
    path = Path(str(path_value or "")).expanduser()
    if not path.is_file():
        return []
    report = read_json(path)
    rows = report.get("perClass")
    if isinstance(rows, list):
        indexed: list[tuple[int, str]] = []
        for fallback_index, row in enumerate(rows):
            if not isinstance(row, dict) or not str(row.get("className") or "").strip():
                continue
            try:
                class_id = int(row.get("classId", fallback_index))
            except (TypeError, ValueError):
                class_id = fallback_index
            indexed.append((class_id, str(row["className"]).strip()))
        if indexed:
            return [name for _, name in sorted(indexed)]
    return [str(item) for item in report.get("classNames", []) if str(item).strip()]


def _shape(dimensions: Any) -> list[int]:
    if not isinstance(dimensions, list):
        return []
    return [dimension if isinstance(dimension, int) and dimension > 0 else -1 for dimension in dimensions]


def inspect_onnx(onnx_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    try:
        import onnx  # type: ignore
    except Exception:
        return [], [], 13
    model = onnx.load(str(onnx_path))

    def tensor(value: Any, layout: str) -> dict[str, Any]:
        tensor_type = value.type.tensor_type
        dimensions: list[Any] = []
        for dimension in tensor_type.shape.dim:
            dimensions.append(int(dimension.dim_value) if dimension.dim_value else (str(dimension.dim_param) or None))
        return {"name": str(value.name), "layout": layout, "shape": _shape(dimensions)}

    opset = max((int(item.version) for item in model.opset_import if not str(item.domain)), default=13)
    return (
        [tensor(value, "NCHW") for value in model.graph.input],
        [tensor(value, "NCHW") for value in model.graph.output],
        opset,
    )


def build_model_contract(
    onnx_path: Path,
    sidecar: dict[str, Any],
    evaluation_report_path: Any,
) -> tuple[dict[str, Any], int]:
    inputs, outputs, inspected_opset = inspect_onnx(onnx_path)
    source_contract = sidecar.get("modelContract") if isinstance(sidecar.get("modelContract"), dict) else {}
    input_width = int(sidecar.get("inputWidth") or sidecar.get("imageSize") or 256)
    input_height = int(sidecar.get("inputHeight") or sidecar.get("imageSize") or input_width)
    class_names = class_names_from_evaluation(evaluation_report_path)
    if not class_names:
        class_names = [str(item) for item in sidecar.get("classNames", []) if str(item).strip()]
    if not inputs:
        inputs = source_contract.get("inputs") if isinstance(source_contract.get("inputs"), list) else []
    if not outputs:
        outputs = source_contract.get("outputs") if isinstance(source_contract.get("outputs"), list) else []
    if not inputs:
        inputs = [{"name": "images", "layout": "NCHW", "shape": [1, 3, input_height, input_width]}]
    if not outputs:
        outputs = [{"name": "logits", "layout": "NCHW", "shape": [1, len(class_names), input_height, input_width]}]
    normalization = sidecar.get("normalization") if isinstance(sidecar.get("normalization"), dict) else {}
    preprocessing = {
        "id": "smp_rgb_mean_std",
        "colorSpace": "RGB",
        "layout": "NCHW",
        "scale": float(normalization.get("scale", 1.0 / 255.0)),
        "mean": normalization.get("mean", MEAN),
        "std": normalization.get("std", STD),
    }
    contract = {
        "source": "smp_semantic_segmentation_export",
        "modelFamily": "semantic_segmentation",
        "taskType": "semantic_segmentation",
        "inputs": inputs,
        "outputs": outputs,
        "preprocessing": preprocessing,
        "postprocessing": {"id": "smp_semantic_segmentation", "operation": "argmax"},
        "decoder": "smp_semantic_segmentation",
        "classNames": class_names,
        "runtimeRoutes": ["aitrain_onnxruntime"],
        "evaluationReportPath": str(evaluation_report_path or ""),
    }
    # The trainer intentionally exports opset 13; reject a divergent model
    # instead of advertising an inaccurate product contract.
    if inspected_opset != 13:
        raise ValueError(f"SMP V2 export requires ONNX opset 13, got {inspected_opset}")
    return contract, inspected_opset


def run(request: dict[str, Any]) -> int:
    model_path = Path(str(request.get("modelPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or "")).resolve()
    options = request.get("options") if isinstance(request.get("options"), dict) else {}
    requested_sidecar = request.get("sidecarPath") or options.get("sidecarPath")
    evaluation_report_path = request.get("evaluationReportPath") or options.get("evaluationReportPath")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("SMP V2 exporter outputPath must name the destination .onnx file")
    source_onnx = resolve_onnx(model_path)
    source_metadata = source_sidecar(source_onnx, requested_sidecar)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emit("progress", percent=10, message="Normalizing SMP ONNX export")
    if source_onnx != output_path:
        shutil.copy2(source_onnx, output_path)
    contract, opset = build_model_contract(output_path, source_metadata, evaluation_report_path)
    sidecar_path = output_path.with_suffix(".aitrain-export.json")
    sidecar = {
        "schemaVersion": 2,
        "backend": BACKEND_ID,
        "format": "onnx",
        "modelFamily": "semantic_segmentation",
        "taskType": "semantic_segmentation",
        "decoder": "smp_semantic_segmentation",
        "opset": opset,
        "runtimeRoutes": ["aitrain_onnxruntime"],
        "sourceModelPath": str(model_path),
        "exportPath": str(output_path),
        "evaluationReportPath": str(evaluation_report_path or ""),
        "exporterVersion": EXPORTER_VERSION,
        "exportedAt": now_iso(),
        "modelContract": contract,
        "limitations": [
            "SMP semantic segmentation product deployment is ONNX Runtime-only.",
            "NCNN and TensorRT export/runtime are outside the SMP v1 scope.",
        ],
    }
    write_json(sidecar_path, sidecar)
    emit("artifact", kind="export", path=str(output_path), message="Normalized SMP semantic segmentation ONNX")
    emit("artifact", kind="export_sidecar", path=str(sidecar_path), message="SMP V2 model contract sidecar")
    emit("progress", percent=100, message="SMP ONNX export completed")
    emit("completed", exportPath=str(output_path), reportPath=str(sidecar_path), sidecarPath=str(sidecar_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        configure_adapter()
        try:
            request = read_json(args.request)
        except Exception as exc:
            emit("failed", code="bad_request", message=f"failed to read export request: {exc}", details=exception_details(exc))
            return 2
        try:
            return run(request)
        except Exception as exc:
            emit("failed", code="smp_export_failed", message=str(exc), details=exception_details(exc))
            return 1
    finally:
        close_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
