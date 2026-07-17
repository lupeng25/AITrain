#!/usr/bin/env python3
"""Official Ultralytics YOLO export helper for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from adapter_event_channel import AdapterEventChannel, event_channel_from_environment, standalone_protocol_enabled  # noqa: E402
from adapter_sdk import AdapterSdk  # noqa: E402
from trainer_protocol import configure_stdio, exception_details  # noqa: E402


BACKEND_ID = "ultralytics_yolo_export"
LICENSE_NOTE = "Ultralytics YOLO is executed through the installed official Python package. Review its license before redistribution."
SUPPORTED_EXPORT_ARGS = {"format", "dynamic", "half", "int8", "imgsz", "batch", "device", "data", "end2end"}


_adapter: AdapterSdk | None = None
_adapter_backend = ""
_event_channel: AdapterEventChannel | None = None


def configure_adapter(backend: str | None = None) -> None:
    """Select the authenticated event channel once."""
    global _adapter, _adapter_backend, _event_channel
    selected_backend = backend or BACKEND_ID
    if _event_channel is None and not standalone_protocol_enabled() and _adapter is None:
        _event_channel = event_channel_from_environment()
        _event_channel.connect()
    if _adapter is None or _adapter_backend != selected_backend:
        sink = _event_channel.emit_event if _event_channel is not None else None
        _adapter = AdapterSdk(selected_backend, event_sink=sink)
        _adapter_backend = selected_backend


def close_adapter() -> None:
    global _event_channel
    if _event_channel is not None:
        _event_channel.close()
        _event_channel = None


def active_adapter() -> AdapterSdk:
    configure_adapter(BACKEND_ID)
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
        adapter.emit_completed(str(payload.pop("message", "official export completed")), **payload)
    elif event_type == "failed":
        adapter.emit_failed(
            str(payload.pop("message", "official export failed")),
            str(payload.pop("code", "ultralytics_export_failed")),
            payload.pop("details", {}),
        )
    else:
        raise ValueError(f"unsupported adapter event type: {event_type}")


def fail(message: str, code: str = "ultralytics_export_failed", details: dict[str, Any] | None = None) -> int:
    return active_adapter().emit_failed(message, code, details)


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


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def prepend_python_paths(parameters: dict[str, Any]) -> None:
    raw_paths = parameters.get("pythonPathPrepend") or parameters.get("pythonPath")
    if raw_paths is None:
        return
    if isinstance(raw_paths, str):
        paths = [raw_paths]
    elif isinstance(raw_paths, list):
        paths = [str(item) for item in raw_paths]
    else:
        return
    for path in reversed(paths):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def apply_cpu_device_environment(parameters: dict[str, Any], default_device: str | None = None) -> None:
    raw_args = parameters.get("ultralyticsExportArgs") if isinstance(parameters.get("ultralyticsExportArgs"), dict) else {}
    device = raw_args.get("device", parameters.get("device", default_device))
    if device is None or device == "":
        device = default_device
    if str(device or "").strip().lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def find_ultralytics_training_report(model_path: Path) -> tuple[Path | None, dict[str, Any]]:
    directories = [model_path.parent, *model_path.parents]
    seen: set[Path] = set()
    for directory in directories[:8]:
        try:
            resolved = directory.resolve()
        except Exception:
            resolved = directory
        if resolved in seen:
            continue
        seen.add(resolved)
        candidate = directory / "ultralytics_training_report.json"
        report = read_json_object(candidate)
        backend = str(report.get("backend") or "")
        if backend in {"ultralytics_yolo_detect", "ultralytics_yolo_segment", "ultralytics_yolo_obb"}:
            return candidate, report
    return None, {}


def model_family_from_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"segmentation", "segment", "yolo_segmentation", "ultralytics_yolo_segment"}:
        return "yolo_segmentation"
    if text in {"obb", "obb_detection", "yolo_obb", "ultralytics_yolo_obb"}:
        return "yolo_obb"
    if text in {"detection", "detect", "yolo_detection", "ultralytics_yolo_detect"}:
        return "yolo_detection"
    return ""


def infer_model_family(model_path: Path, model: Any | None, request: dict[str, Any]) -> tuple[str, Path | None, dict[str, Any]]:
    options = request.get("options") if isinstance(request.get("options"), dict) else {}
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    for source in (options, parameters, request):
        family = model_family_from_text(source.get("modelFamily"))
        if family:
            return family, None, {}
        family = model_family_from_text(source.get("taskType"))
        if family:
            return family, None, {}
        family = model_family_from_text(source.get("trainingBackend"))
        if family:
            return family, None, {}

    report_path, report = find_ultralytics_training_report(model_path)
    family = model_family_from_text(report.get("backend"))
    if family:
        return family, report_path, report
    family = model_family_from_text(report.get("taskType"))
    if family:
        return family, report_path, report
    report_model = str(report.get("model") or "")
    if "-obb" in report_model.lower():
        return "yolo_obb", report_path, report
    if "-seg" in report_model.lower():
        return "yolo_segmentation", report_path, report

    for candidate in (
        getattr(model, "task", None),
        getattr(getattr(model, "model", None), "task", None),
        getattr(getattr(model, "predictor", None), "task", None),
    ):
        family = model_family_from_text(candidate)
        if family:
            return family, report_path, report

    stem = model_path.stem.lower()
    if "-obb" in stem:
        return "yolo_obb", report_path, report
    return ("yolo_segmentation" if "-seg" in stem else "yolo_detection"), report_path, report


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


def parse_end2end(name: str, value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto", "default"}:
            return None
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Ultralytics export argument '{name}' must be auto, true, or false")


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


def is_yolo26_model_name(value: Any) -> bool:
    text = str(value or "").replace("\\", "/").lower()
    return "yolo26" in Path(text).name or "yolo26" in Path(text).stem


def model_name_hint(parameters: dict[str, Any], explicit: str | Path | None) -> str:
    if explicit is not None and str(explicit) != "":
        return str(explicit)
    for key in ("model", "modelPreset", "checkpointPath", "modelPath"):
        value = parameters.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def model_family_hint(parameters: dict[str, Any], explicit: str | None, model_name: str) -> str:
    family = model_family_from_text(explicit)
    if family:
        return family
    for key in ("modelFamily", "trainingBackend", "backend", "taskType"):
        family = model_family_from_text(parameters.get(key))
        if family:
            return family
    lower_name = model_name.lower()
    if "-obb" in lower_name:
        return "yolo_obb"
    return "yolo_segmentation" if "-seg" in lower_name else "yolo_detection"


def normalize_end2end(
    raw_value: Any,
    *,
    product_format: str,
    model_name: str,
    model_family: str,
    auto_end2end: bool | None = None,
) -> tuple[bool, bool]:
    parsed = parse_end2end("end2end", raw_value)
    explicitly_requested = parsed is not None
    if product_format == "ncnn":
        if parsed is True:
            raise ValueError("NCNN export requires end2end=false because AITrain NCNN runtime uses traditional YOLO decoding.")
        return False, explicitly_requested

    if parsed is not None:
        return parsed, explicitly_requested

    if auto_end2end is not None:
        return auto_end2end, explicitly_requested

    if is_yolo26_model_name(model_name) and model_family == "yolo_detection":
        return True, explicitly_requested
    return False, explicitly_requested


def model_end2end_default(model: Any | None) -> bool | None:
    if model is None:
        return None
    candidates: list[Any] = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "end2end"):
            try:
                return parse_end2end("end2end", getattr(candidate, "end2end"))
            except ValueError:
                pass
        for attr_name in ("yaml", "args", "overrides"):
            container = getattr(candidate, attr_name, None)
            if isinstance(container, dict) and "end2end" in container:
                try:
                    return parse_end2end("end2end", container.get("end2end"))
                except ValueError:
                    pass
    return None


def model_series_from_name(value: Any) -> str:
    name = Path(str(value or "").replace("\\", "/")).name.lower()
    match = re.search(r"(yolo26|yolo12|yolo11|yolov8|yolov5u?|pp-ocrv\d+)", name)
    return match.group(1) if match else ""


def task_from_model_family(model_family: str) -> str:
    if model_family == "yolo_segmentation":
        return "segmentation"
    if model_family == "yolo_obb":
        return "obb_detection"
    if model_family == "yolo_detection":
        return "detection"
    return ""


def tensor_shape_from_value_info(value_info: Any) -> dict[str, Any]:
    shape: list[Any] = []
    tensor_type = getattr(getattr(value_info, "type", None), "tensor_type", None)
    for dim in getattr(getattr(tensor_type, "shape", None), "dim", []):
        if getattr(dim, "dim_value", 0):
            shape.append(int(dim.dim_value))
        elif getattr(dim, "dim_param", ""):
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return {"name": str(getattr(value_info, "name", "")), "shape": shape}


def inspect_onnx_io_shapes(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".onnx" or not path.exists():
        return {"available": False, "reason": "not_onnx"}
    try:
        import onnx  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": "onnx_python_missing", "error": str(exc)}
    try:
        model = onnx.load(str(path))
        graph = model.graph
        return {
            "available": True,
            "inputs": [tensor_shape_from_value_info(value) for value in graph.input],
            "outputs": [tensor_shape_from_value_info(value) for value in graph.output],
        }
    except Exception as exc:
        return {"available": False, "reason": "onnx_shape_inspection_failed", "error": str(exc)}


def _contract_shape(value: Any) -> list[int]:
    """Turn ONNX dimensions into the  contract's positive-or--1 form."""
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for dimension in value:
        if isinstance(dimension, int) and dimension > 0:
            result.append(dimension)
        else:
            result.append(-1)
    return result


def class_names_from_evaluation_report(path_value: Any) -> list[str]:
    """Read class names only from the preceding official evaluation evidence."""
    path = Path(str(path_value or "")).expanduser()
    report = read_json_object(path)
    rows = report.get("perClass")
    if not isinstance(rows, list):
        return []
    indexed: list[tuple[int, str]] = []
    for fallback_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = str(row.get("className") or "").strip()
        if not name:
            continue
        raw_index = row.get("classId", fallback_index)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = fallback_index
        indexed.append((index, name))
    return [name for _, name in sorted(indexed)]


def model_contract(model_family: str, output_shapes: dict[str, Any], evaluation_report_path: Any) -> dict[str, Any]:
    """Build only verifiable model facts for C++  registration.

    Identity, source artifact hash and the final verified flag are intentionally
    absent: they are assigned by the  Workspace after immutable artifact
    submission, never trusted from this Python sidecar.
    """
    task_type = task_from_model_family(model_family)
    inputs = output_shapes.get("inputs") if isinstance(output_shapes, dict) else []
    outputs = output_shapes.get("outputs") if isinstance(output_shapes, dict) else []
    return {
        "source": "ultralytics_official_export",
        "modelFamily": model_family,
        "taskType": task_type,
        "inputs": [
            {"name": str(item.get("name") or "images"), "layout": "NCHW", "shape": _contract_shape(item.get("shape"))}
            for item in inputs if isinstance(item, dict)
        ],
        "outputs": [
            {
                "name": str(item.get("name") or "output0"),
                "layout": (
                    "NCHW"
                    if model_family == "yolo_segmentation" and len(_contract_shape(item.get("shape"))) == 4
                    else "NCN"
                ),
                "shape": _contract_shape(item.get("shape")),
            }
            for item in outputs if isinstance(item, dict)
        ],
        "preprocessing": {"id": "letterbox_rgb_0_1"},
        "postprocessing": {
            "id": (
                "yolo_obb_nms"
                if model_family == "yolo_obb"
                else ("yolo_segmentation_masks_v8" if model_family == "yolo_segmentation" else "yolo_detection_nms")
            )
        },
        "decoder": "yolo_obb_v8" if model_family == "yolo_obb" else ("yolo_segmentation_v8" if model_family == "yolo_segmentation" else "yolo_detection_v8"),
        "classNames": class_names_from_evaluation_report(evaluation_report_path),
        "runtimeRoutes": ["aitrain_onnxruntime"] if model_family in {"yolo_detection", "yolo_segmentation", "yolo_obb"} else [],
        "evaluationReportPath": str(evaluation_report_path or ""),
    }


def export_report_path(export_path: Path) -> Path:
    return export_path.with_name(f"{export_path.stem}.aitrain-export.json")


def build_export_plan(
    parameters: dict[str, Any] | None,
    default_format: str = "onnx",
    default_imgsz: int | None = None,
    default_batch: int | None = None,
    default_device: str | None = None,
    data_yaml: str | Path | None = None,
    model_name: str | Path | None = None,
    model_family: str | None = None,
    auto_end2end: bool | None = None,
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
    model_hint = model_name_hint(parameters, model_name)
    family_hint = model_family_hint(parameters, model_family, model_hint)
    if product_format == "ncnn" and family_hint == "yolo_obb":
        raise ValueError("OBB NCNN export is not supported in AITrain v1; use ONNX Runtime for OBB deployment.")
    if product_format == "ncnn" and is_yolo26_model_name(model_hint):
        raise ValueError("YOLO26 NCNN export is not supported by AITrain; use ONNX or TensorRT for YOLO26 deployment.")
    dynamic = parse_bool("dynamic", raw_args.get("dynamic", False))
    half = parse_bool("half", raw_args.get("half", False))
    int8 = parse_bool("int8", raw_args.get("int8", False))
    end2end, explicit_end2end = normalize_end2end(
        raw_args.get("end2end", "auto"),
        product_format=product_format,
        model_name=model_hint,
        model_family=family_hint,
        auto_end2end=auto_end2end,
    )

    if product_format == "onnx" and int8:
        raise ValueError("ONNX export does not support int8 in AITrain; use TensorRT export for INT8.")
    if product_format == "ncnn" and (dynamic or half or int8):
        raise ValueError("NCNN export requires a static FP32 ONNX intermediate; dynamic/half/int8/end2end are unsupported.")
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
        "end2end": end2end,
    }
    kwargs: dict[str, Any] = {
        "format": official_format_for(product_format),
        "dynamic": dynamic,
        "half": half,
        "int8": int8,
    }
    if is_yolo26_model_name(model_hint) or explicit_end2end or auto_end2end is True:
        kwargs["end2end"] = end2end
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
        "modelName": model_hint,
        "modelFamily": family_hint,
        "modelSeries": model_series_from_name(model_hint),
        "task": task_from_model_family(family_hint),
        "normalized": normalized,
        "kwargs": kwargs,
    }


def onnx_kwargs_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(plan.get("kwargs") or {})
    kwargs["format"] = "onnx"
    kwargs["int8"] = False
    kwargs.pop("data", None)
    return kwargs


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
    prepend_python_paths(options)
    prepend_python_paths(parameters)
    export_parameters: dict[str, Any] = {}
    export_parameters.update(parameters)
    if "ultralyticsExportArgs" in options:
        export_parameters["ultralyticsExportArgs"] = options.get("ultralyticsExportArgs")
    export_parameters.setdefault("ultralyticsExportArgs", {})
    if isinstance(export_parameters["ultralyticsExportArgs"], dict):
        export_parameters["ultralyticsExportArgs"].setdefault("format", product_format)
    apply_cpu_device_environment(export_parameters)

    if not model_path.exists():
        return fail(f"model path does not exist: {model_path}", "model_missing")
    if output_path.name == "":
        return fail("outputPath is required for official YOLO export", "output_missing")

    try:
        import ultralytics  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        return fail(
            "Ultralytics is not available. Install it with: python -m pip install ultralytics",
            "ultralytics_missing",
            {"exception": str(exc)},
        )

    try:
        model = YOLO(str(model_path))
        model_family, source_report_path, source_report = infer_model_family(model_path, model, request)
    except Exception as exc:
        return fail("Ultralytics model load failed.", "ultralytics_model_load_failed", exception_details(exc))

    model_hint = str(source_report.get("model") or model_path.name)
    try:
        plan = build_export_plan(
            export_parameters,
            default_format=product_format,
            model_name=model_hint,
            model_family=model_family,
            auto_end2end=model_end2end_default(model),
        )
    except ValueError as exc:
        return fail(str(exc), "ultralytics_export_args_invalid")

    emit("progress", taskId=task_id, backend=BACKEND_ID, phase="export", percent=0, message="official YOLO export started")
    emit("log", taskId=task_id, backend=BACKEND_ID, level="info", message=f"Using Ultralytics module: {getattr(ultralytics, '__file__', 'built-in')}")
    emit("log", taskId=task_id, backend=BACKEND_ID, level="info", message=f"Exporting {model_path} as {plan['productFormat']}")

    try:
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
    output_shapes = inspect_onnx_io_shapes(final_path)
    evaluation_report_path = request.get("evaluationReportPath")
    report = {
        "ok": True,
        "backend": BACKEND_ID,
        "format": plan["productFormat"],
        "officialFormat": plan["officialFormat"],
        "modelFamily": model_family,
        "modelSeries": plan["modelSeries"] or model_series_from_name(model_hint),
        "task": plan["task"],
        "sourceTrainingBackend": str(source_report.get("backend") or ""),
        "sourceTrainingReport": str(source_report_path) if source_report_path else "",
        "sourceCheckpoint": str(model_path),
        "officialExportPath": str(exported_path),
        "exportPath": str(final_path),
        "ultralyticsVersion": getattr(ultralytics, "__version__", "unknown"),
        "ultralyticsExportArgs": plan["normalized"],
        "outputShapes": output_shapes,
        "modelContract": model_contract(model_family, output_shapes, evaluation_report_path),
        "exportedAt": now_iso(),
        "licenseNote": LICENSE_NOTE,
    }
    write_json(report_path, report)

    emit("artifact", taskId=task_id, backend=BACKEND_ID, kind="export", path=str(final_path), message="Official Ultralytics export")
    emit("artifact", taskId=task_id, backend=BACKEND_ID, kind="export_sidecar", path=str(report_path), message="AITrain export sidecar")
    emit("progress", taskId=task_id, backend=BACKEND_ID, phase="completed", percent=100, message="official YOLO export completed")
    emit("completed", taskId=task_id, backend=BACKEND_ID, checkpointPath=str(model_path), exportPath=str(final_path), reportPath=str(report_path))
    return 0


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        configure_adapter(BACKEND_ID)
        try:
            request = read_request(args.request)
        except Exception as exc:
            return fail(f"failed to read export request: {exc}", "bad_request", exception_details(exc))
        try:
            return run_official_export(request)
        except Exception as exc:
            return fail("Unhandled official YOLO export failure.", "unhandled_exception", exception_details(exc))
    finally:
        close_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
