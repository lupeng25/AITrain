#!/usr/bin/env python3
"""Official Ultralytics YOLO export helper for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from trainer_protocol import configure_stdio, emit_failed, exception_details  # noqa: E402


BACKEND_ID = "ultralytics_yolo_export"
LICENSE_NOTE = "Ultralytics YOLO is executed through the installed official Python package. Review its license before redistribution."
SUPPORTED_EXPORT_ARGS = {"format", "dynamic", "half", "int8", "imgsz", "batch", "device", "data"}


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time()}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def fail(message: str, code: str = "ultralytics_export_failed", details: dict[str, Any] | None = None) -> int:
    return emit_failed(BACKEND_ID, message, code, details)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("export request must be a JSON object")
    return value


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Ultralytics export argument '{name}' must be a boolean")


def parse_int(name: str, value: Any, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Ultralytics export argument '{name}' must be an integer") from None
    if parsed < minimum:
        raise ValueError(f"Ultralytics export argument '{name}' must be >= {minimum}")
    return parsed


def normalize_product_format(value: Any, default: str = "onnx") -> str:
    text = str(value or default or "onnx").strip().lower()
    if text in {"engine", "trt", "tensorrt", "tensorrt_engine"}:
        return "tensorrt"
    if text in {"onnx", "ncnn"}:
        return text
    raise ValueError(f"Unsupported Ultralytics export format: {text}")


def official_format_for(product_format: str) -> str:
    return "engine" if product_format == "tensorrt" else product_format


def export_report_path(export_path: Path) -> Path:
    return export_path.with_name(f"{export_path.stem}.aitrain-export.json")


def build_export_plan(
    parameters: dict[str, Any] | None,
    default_format: str = "onnx",
    default_imgsz: int | None = None,
    default_batch: int | None = None,
    default_device: str | None = None,
    data_yaml: str | Path | None = None,
) -> dict[str, Any]:
    parameters = parameters or {}
    raw_args = parameters.get("ultralyticsExportArgs")
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        raise ValueError("parameters.ultralyticsExportArgs must be a JSON object")

    unknown = sorted(str(key) for key in raw_args.keys() if str(key) not in SUPPORTED_EXPORT_ARGS)
    if unknown:
        raise ValueError(f"Unsupported Ultralytics export argument: {unknown[0]}")

    product_format = normalize_product_format(raw_args.get("format", default_format), default_format)
    dynamic = parse_bool("dynamic", raw_args.get("dynamic", False))
    half = parse_bool("half", raw_args.get("half", False))
    int8 = parse_bool("int8", raw_args.get("int8", False))

    if product_format == "onnx" and int8:
        raise ValueError("ONNX export does not support int8 in AITrain; use TensorRT export for INT8.")
    if product_format == "ncnn" and (dynamic or half or int8):
        raise ValueError("NCNN export requires a static FP32 ONNX intermediate; dynamic/half/int8 are unsupported.")
    if int8 and product_format != "tensorrt":
        raise ValueError("INT8 export is only supported for TensorRT engine export.")

    imgsz = raw_args.get("imgsz", default_imgsz)
    batch = raw_args.get("batch", default_batch)
    device = raw_args.get("device", default_device)

    normalized: dict[str, Any] = {
        "format": product_format,
        "dynamic": dynamic,
        "half": half,
        "int8": int8,
    }
    kwargs: dict[str, Any] = {
        "format": official_format_for(product_format),
        "dynamic": dynamic,
        "half": half,
        "int8": int8,
    }
    if imgsz not in {None, ""}:
        normalized["imgsz"] = parse_int("imgsz", imgsz, 32)
        kwargs["imgsz"] = normalized["imgsz"]
    if batch not in {None, ""}:
        normalized["batch"] = parse_int("batch", batch, 1)
        kwargs["batch"] = normalized["batch"]
    if device not in {None, ""}:
        normalized["device"] = str(device).strip()
        kwargs["device"] = normalized["device"]

    requested_data = raw_args.get("data", data_yaml)
    if product_format == "tensorrt" and int8:
        if requested_data in {None, ""}:
            raise ValueError("TensorRT INT8 export requires a calibration data yaml.")
        normalized["data"] = str(requested_data)
        kwargs["data"] = str(requested_data)

    return {
        "productFormat": product_format,
        "officialFormat": kwargs["format"],
        "normalized": normalized,
        "kwargs": kwargs,
    }


def onnx_kwargs_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(plan.get("normalized") or {})
    normalized["format"] = "onnx"
    normalized["int8"] = False
    nested = {"ultralyticsExportArgs": normalized}
    return build_export_plan(nested, default_format="onnx")["kwargs"]


def copy_exported_artifact(exported_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() == output_path.resolve():
        return output_path
    shutil.copyfile(exported_path, output_path)
    return output_path


def run_official_export(request: dict[str, Any]) -> int:
    task_id = str(request.get("taskId") or "")
    model_path = Path(str(request.get("modelPath") or request.get("checkpointPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or "")).resolve()
    product_format = normalize_product_format(request.get("format") or "onnx")
    options = request.get("options") or {}
    if not isinstance(options, dict):
        options = {}
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    export_parameters: dict[str, Any] = {}
    export_parameters.update(parameters)
    if "ultralyticsExportArgs" in options:
        export_parameters["ultralyticsExportArgs"] = options.get("ultralyticsExportArgs")
    export_parameters.setdefault("ultralyticsExportArgs", {})
    if isinstance(export_parameters["ultralyticsExportArgs"], dict):
        export_parameters["ultralyticsExportArgs"].setdefault("format", product_format)

    if not model_path.exists():
        return fail(f"model path does not exist: {model_path}", "model_missing")
    if output_path.name == "":
        return fail("outputPath is required for official YOLO export", "output_missing")

    try:
        plan = build_export_plan(export_parameters, default_format=product_format)
    except ValueError as exc:
        return fail(str(exc), "ultralytics_export_args_invalid")

    try:
        import ultralytics  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        return fail(
            "Ultralytics is not available. Install it with: python -m pip install ultralytics",
            "ultralytics_missing",
            {"exception": str(exc)},
        )

    emit("progress", taskId=task_id, backend=BACKEND_ID, phase="export", percent=0, message="official YOLO export started")
    emit("log", taskId=task_id, backend=BACKEND_ID, level="info", message=f"Using Ultralytics module: {getattr(ultralytics, '__file__', 'built-in')}")
    emit("log", taskId=task_id, backend=BACKEND_ID, level="info", message=f"Exporting {model_path} as {plan['productFormat']}")

    try:
        model = YOLO(str(model_path))
        exported = model.export(**plan["kwargs"])
    except Exception as exc:
        return fail("Ultralytics official export failed.", "ultralytics_export_failed", exception_details(exc))

    if not exported:
        return fail("Ultralytics official export completed without returning an artifact path.", "official_export_missing")
    exported_path = Path(str(exported)).resolve()
    if not exported_path.exists():
        return fail(f"Ultralytics official export artifact does not exist: {exported_path}", "official_export_missing")

    final_path = copy_exported_artifact(exported_path, output_path)
    report_path = export_report_path(final_path)
    report = {
        "ok": True,
        "backend": BACKEND_ID,
        "format": plan["productFormat"],
        "officialFormat": plan["officialFormat"],
        "modelFamily": "yolo_segmentation" if "-seg" in model_path.stem else "yolo_detection",
        "sourceCheckpoint": str(model_path),
        "officialExportPath": str(exported_path),
        "exportPath": str(final_path),
        "ultralyticsVersion": getattr(ultralytics, "__version__", "unknown"),
        "ultralyticsExportArgs": plan["normalized"],
        "exportedAt": now_iso(),
        "licenseNote": LICENSE_NOTE,
    }
    write_json(report_path, report)

    emit("artifact", taskId=task_id, backend=BACKEND_ID, kind="export", path=str(final_path), message="Official Ultralytics export")
    emit("artifact", taskId=task_id, backend=BACKEND_ID, kind="export_sidecar", path=str(report_path), message="AITrain export sidecar")
    emit("progress", taskId=task_id, backend=BACKEND_ID, phase="completed", percent=100, message="official YOLO export completed")
    emit(
        "modelExport",
        taskId=task_id,
        backend=BACKEND_ID,
        ok=True,
        format=plan["productFormat"],
        checkpointPath=str(model_path),
        exportPath=str(final_path),
        reportPath=str(report_path),
        config=report,
    )
    emit("completed", taskId=task_id, backend=BACKEND_ID, checkpointPath=str(model_path), exportPath=str(final_path), reportPath=str(report_path))
    return 0


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = read_request(args.request)
    except Exception as exc:
        return fail(f"failed to read export request: {exc}", "bad_request", exception_details(exc))
    try:
        return run_official_export(request)
    except Exception as exc:
        return fail("Unhandled official YOLO export failure.", "unhandled_exception", exception_details(exc))


if __name__ == "__main__":
    raise SystemExit(main())
