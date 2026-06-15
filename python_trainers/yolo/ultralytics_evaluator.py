#!/usr/bin/env python3
"""Official Ultralytics YOLO validation adapter for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

DETECTION_ADAPTER_DIR = TRAINER_ROOT / "detection"
if str(DETECTION_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_ADAPTER_DIR))

import ultralytics_trainer as shared  # type: ignore  # noqa: E402
from trainer_protocol import configure_stdio, exception_details  # noqa: E402


configure_stdio()
EVALUATION_SOURCE = "ultralytics_official_val"


VAL_ARG_TYPES: dict[str, str] = {
    "split": "str",
    "batch": "int_positive",
    "imgsz": "int_positive",
    "device": "str",
    "workers": "int_nonnegative",
    "conf": "float_0_1",
    "iou": "float_0_1",
    "max_det": "int_positive",
    "half": "bool",
    "dnn": "bool",
    "plots": "bool",
    "save_json": "bool",
    "save_txt": "bool",
    "save_conf": "bool",
    "rect": "bool",
    "classes": "int_list",
    "single_cls": "bool",
    "augment": "bool",
    "agnostic_nms": "bool",
    "visualize": "bool",
    "end2end": "bool",
}


METRIC_ALIASES: dict[str, list[str]] = {
    "precision": ["metrics/precision(B)", "precision(B)", "box/precision"],
    "recall": ["metrics/recall(B)", "recall(B)", "box/recall"],
    "mAP50": ["metrics/mAP50(B)", "mAP50(B)", "box/mAP50"],
    "mAP50_95": ["metrics/mAP50-95(B)", "mAP50-95(B)", "box/mAP50-95"],
    "maskPrecision": ["metrics/precision(M)", "precision(M)", "mask/precision"],
    "maskRecall": ["metrics/recall(M)", "recall(M)", "mask/recall"],
    "maskMap50": ["metrics/mAP50(M)", "mAP50(M)", "mask/mAP50"],
    "maskMap50_95": ["metrics/mAP50-95(M)", "mAP50-95(M)", "mask/mAP50-95"],
    "fitness": ["fitness"],
}


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time()}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("evaluation request must be a JSON object")
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            return to_jsonable(value.detach().cpu().tolist())
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def coerce_val_arg(name: str, value: Any) -> Any:
    kind = VAL_ARG_TYPES[name]
    if value is None or value == "":
        return None
    if kind == "str":
        return str(value)
    if kind == "bool":
        return shared.parse_bool_arg(name, value)
    if kind == "int_positive":
        return shared.parse_int_arg(name, value, 1)
    if kind == "int_nonnegative":
        return shared.parse_int_arg(name, value, 0)
    if kind == "float_0_1":
        return shared.parse_float_arg(name, value, 0.0, 1.0)
    if kind == "int_list":
        return shared.parse_list_arg(name, value, "int")
    raise ValueError(f"Unsupported Ultralytics val argument schema for '{name}'")


def yolo_split_exists(dataset_path: Path, split: str) -> bool:
    existing = shared.read_existing_data_yaml(dataset_path)
    yaml_base = shared._resolve_yaml_path(dataset_path, existing.get("path", ""))
    split_value = existing.get(split, f"images/{split}")
    image_dir = shared._resolve_yaml_path(yaml_base, split_value)
    return image_dir.exists()


def select_split(dataset_path: Path, requested: str | None = None) -> str:
    if requested:
        return requested
    for split in ("val", "test", "train"):
        if yolo_split_exists(dataset_path, split):
            return split
    return "val"


def count_split_images(dataset_path: Path, split: str) -> int:
    existing = shared.read_existing_data_yaml(dataset_path)
    yaml_base = shared._resolve_yaml_path(dataset_path, existing.get("path", ""))
    split_value = existing.get(split, f"images/{split}")
    image_dir = shared._resolve_yaml_path(yaml_base, split_value)
    if not image_dir.exists():
        return 0
    return sum(1 for item in image_dir.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def class_names_from_data_yaml(dataset_path: Path) -> list[str]:
    existing = shared.read_existing_data_yaml(dataset_path)
    names = existing.get("names")
    if isinstance(names, list):
        return [str(item) for item in names]
    nc = existing.get("nc")
    try:
        count = int(nc)
    except (TypeError, ValueError):
        count = 0
    return [f"class_{index}" for index in range(max(0, count))]


def build_val_kwargs(options: dict[str, Any], data_yaml: Path, output_path: Path, dataset_path: Path) -> dict[str, Any]:
    raw_args = options.get("ultralyticsValArgs") or {}
    if not isinstance(raw_args, dict):
        raise ValueError("options.ultralyticsValArgs must be a JSON object")

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "project": str(output_path / "ultralytics_val"),
        "name": "official_val",
        "exist_ok": True,
        "verbose": False,
        "plots": True,
        "save_json": True,
    }
    if "batch" in options:
        kwargs["batch"] = shared.parse_int_arg("batch", options["batch"], 1)
    if "imageSize" in options:
        kwargs["imgsz"] = shared.parse_int_arg("imgsz", options["imageSize"], 32)

    for name, value in raw_args.items():
        key = str(name).strip()
        if not key:
            continue
        if key not in VAL_ARG_TYPES:
            raise ValueError(f"Unsupported Ultralytics val argument: {key}")
        coerced = coerce_val_arg(key, value)
        if coerced is not None:
            kwargs[key] = coerced

    kwargs["split"] = select_split(dataset_path, str(kwargs.get("split") or "").strip() or None)
    return kwargs


def results_dict_from_metrics(metrics: Any) -> dict[str, Any]:
    results = getattr(metrics, "results_dict", None)
    if isinstance(results, dict):
        return dict(results)
    try:
        return dict(metrics)
    except Exception:
        return {}


def extract_metrics(results: dict[str, Any], task_type: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for output_name, source_names in METRIC_ALIASES.items():
        for source_name in source_names:
            if source_name not in results:
                continue
            try:
                value = float(results[source_name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                output[output_name] = round(value, 6)
                break
    if task_type == "segmentation":
        if "maskMap50" in output:
            output.setdefault("mAP50", output.get("mAP50", output["maskMap50"]))
        if "maskMap50_95" in output:
            output.setdefault("mAP50_95", output.get("mAP50_95", output["maskMap50_95"]))
    return output


def metric_maps(metric_group: Any) -> list[float]:
    maps = getattr(metric_group, "maps", None)
    if maps is None:
        return []
    converted = to_jsonable(maps)
    if isinstance(converted, list):
        result: list[float] = []
        for item in converted:
            try:
                value = float(item)
            except (TypeError, ValueError):
                value = 0.0
            result.append(round(value if math.isfinite(value) else 0.0, 6))
        return result
    return []


def extract_per_class(metrics: Any, task_type: str, class_names: list[str]) -> list[dict[str, Any]]:
    box_maps = metric_maps(getattr(metrics, "box", None))
    seg_maps = metric_maps(getattr(metrics, "seg", None))
    count = max(len(class_names), len(box_maps), len(seg_maps))
    rows: list[dict[str, Any]] = []
    for class_id in range(count):
        row: dict[str, Any] = {
            "classId": class_id,
            "className": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
            "official": True,
        }
        if class_id < len(box_maps):
            row["mAP50_95"] = box_maps[class_id]
        if task_type == "segmentation" and class_id < len(seg_maps):
            row["maskMap50_95"] = seg_maps[class_id]
        rows.append(row)
    return rows


def official_artifacts(save_dir: Path) -> list[dict[str, str]]:
    if not save_dir.exists():
        return []
    artifacts: list[dict[str, str]] = []
    interesting_suffixes = {".json", ".png", ".jpg", ".jpeg", ".txt", ".csv"}
    for path in sorted(save_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in interesting_suffixes:
            continue
        kind = "official_plot" if path.suffix.lower() in {".png", ".jpg", ".jpeg"} else "official_artifact"
        if path.name == "predictions.json":
            kind = "official_predictions"
        artifacts.append({"name": path.name, "kind": kind, "path": str(path)})
        if len(artifacts) >= 120:
            break
    return artifacts


def decision_summary(task_type: str, metrics: dict[str, float], sample_count: int) -> dict[str, Any]:
    primary = "maskMap50" if task_type == "segmentation" and "maskMap50" in metrics else "mAP50"
    return {
        "schemaVersion": 1,
        "taskType": task_type,
        "status": "official_evaluation_completed" if sample_count > 0 else "official_evaluation_completed_no_sample_count",
        "primaryMetric": primary,
        "primaryMetricValue": metrics.get(primary, 0.0),
        "sampleCount": sample_count,
        "errorSampleCount": 0,
        "recommendedActions": ["inspect_ultralytics_official_plots", "run_benchmark", "generate_delivery_report"],
    }


def empty_error_taxonomy(task_type: str) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "taskType": task_type,
        "source": EVALUATION_SOURCE,
        "sampleErrorCount": 0,
        "lowConfidenceCount": 0,
        "reasonCounts": {},
        "falsePositiveCount": 0,
        "falseNegativeCount": 0,
        "truePositiveCount": 0,
    }


def write_summary(path: Path, report: dict[str, Any]) -> None:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    lines = [
        "# Evaluation Summary",
        "",
        f"- Source: {EVALUATION_SOURCE}",
        f"- Task type: {report.get('taskType', '')}",
        f"- Status: {report.get('status', '')}",
        f"- Split: {report.get('split', '')}",
        f"- Official run dir: {report.get('officialRunDir', '')}",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(metrics.keys()):
        lines.append(f"- {key}: {metrics[key]}")
    lines.append("")
    lines.append("Metrics are produced by Ultralytics official val(). AITrain does not compute local AP/mAP or mask IoU for this report.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_report(output_path: Path, request: dict[str, Any], message: str, code: str, details: dict[str, Any]) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "evaluation_report.json"
    task_type = str(request.get("taskType") or "detection")
    report = {
        "ok": False,
        "status": "failed",
        "failureCategory": "official-evaluation",
        "errorCode": code,
        "message": message,
        "details": details,
        "kind": "evaluation_report",
        "createdAt": now_iso(),
        "modelPath": str(request.get("modelPath") or ""),
        "datasetPath": str(request.get("datasetPath") or ""),
        "taskType": task_type,
        "runtime": EVALUATION_SOURCE,
        "evaluationSource": EVALUATION_SOURCE,
        "scaffold": False,
        "metrics": {},
        "perClass": [],
        "errorSamples": [],
        "lowConfidenceSamples": [],
        "limitations": "YOLO detection/segmentation evaluation is official-only. This report is failed because Ultralytics val() could not complete; AITrain local AP/mAP fallback is disabled.",
    }
    write_json(report_path, report)
    return report_path


def run(request: dict[str, Any]) -> int:
    model_path = Path(str(request.get("modelPath") or "")).resolve()
    dataset_path = Path(str(request.get("datasetPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or "aitrain-yolo-evaluation")).resolve()
    task_type = str(request.get("taskType") or "detection").strip().lower()
    if task_type in {"yolo_detection", "detect"}:
        task_type = "detection"
    if task_type in {"yolo_segmentation", "segment"}:
        task_type = "segmentation"
    options = request.get("options")
    if not isinstance(options, dict):
        options = {}
    shared.prepend_python_paths(options)

    output_path.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        report_path = write_failure_report(output_path, request, f"model path does not exist: {model_path}", "model_missing", {})
        emit("failed", code="model_missing", reportPath=str(report_path), message="Model path does not exist.")
        return 2
    if not dataset_path.exists():
        report_path = write_failure_report(output_path, request, f"dataset path does not exist: {dataset_path}", "dataset_missing", {})
        emit("failed", code="dataset_missing", reportPath=str(report_path), message="Dataset path does not exist.")
        return 2

    try:
        data_yaml = shared.normalize_data_yaml(dataset_path, output_path)
        val_kwargs = build_val_kwargs(options, data_yaml, output_path, dataset_path)
    except Exception as exc:
        report_path = write_failure_report(output_path, request, str(exc), "official_val_args_invalid", exception_details(exc))
        emit("failed", code="official_val_args_invalid", reportPath=str(report_path), message=str(exc))
        return 2

    try:
        import ultralytics  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        report_path = write_failure_report(
            output_path,
            request,
            "Ultralytics is not available. Install it with: python -m pip install ultralytics",
            "ultralytics_missing",
            exception_details(exc),
        )
        emit("failed", code="ultralytics_missing", reportPath=str(report_path), message="Ultralytics is unavailable.")
        return 3

    emit("log", level="info", message=f"Using Ultralytics module: {getattr(ultralytics, '__file__', 'built-in')}")
    emit("log", level="info", message=f"Starting official Ultralytics val: model={model_path}, split={val_kwargs.get('split')}")

    try:
        model = YOLO(str(model_path))
        metrics_obj = model.val(**val_kwargs)
    except Exception as exc:
        report_path = write_failure_report(
            output_path,
            request,
            "Ultralytics official val() failed.",
            "ultralytics_val_failed",
            exception_details(exc),
        )
        emit("failed", code="ultralytics_val_failed", reportPath=str(report_path), message="Ultralytics official val() failed.")
        return 4

    results = to_jsonable(results_dict_from_metrics(metrics_obj))
    metrics = extract_metrics(results if isinstance(results, dict) else {}, task_type)
    class_names = class_names_from_data_yaml(dataset_path)
    per_class = extract_per_class(metrics_obj, task_type, class_names)
    save_dir = Path(str(getattr(metrics_obj, "save_dir", "") or Path(val_kwargs["project"]) / str(val_kwargs["name"]))).resolve()
    artifacts = official_artifacts(save_dir)
    metrics_path = output_path / "ultralytics_official_metrics.json"
    write_json(metrics_path, {
        "source": EVALUATION_SOURCE,
        "taskType": task_type,
        "resultsDict": results if isinstance(results, dict) else {},
        "metrics": metrics,
        "valArgs": {key: to_jsonable(value) for key, value in val_kwargs.items()},
        "saveDir": str(save_dir),
    })

    split = str(val_kwargs.get("split") or "val")
    sample_count = count_split_images(dataset_path, split)
    report = {
        "ok": True,
        "status": "ok",
        "kind": "evaluation_report",
        "createdAt": now_iso(),
        "modelPath": str(model_path),
        "datasetPath": str(dataset_path),
        "taskType": task_type,
        "split": split,
        "runtime": EVALUATION_SOURCE,
        "evaluationSource": EVALUATION_SOURCE,
        "datasetSnapshotId": int(options.get("datasetSnapshotId") or 0),
        "datasetSnapshotHash": str(options.get("datasetSnapshotHash") or ""),
        "datasetSnapshotManifest": str(options.get("datasetSnapshotManifest") or ""),
        "scaffold": False,
        "metrics": metrics,
        "perClass": per_class,
        "samples": [],
        "errorSamples": [],
        "lowConfidenceSamples": [],
        "sampleCount": sample_count,
        "officialRunDir": str(save_dir),
        "officialMetricsPath": str(metrics_path),
        "officialResultsDict": results if isinstance(results, dict) else {},
        "officialArtifacts": artifacts,
        "parameters": {key: to_jsonable(value) for key, value in val_kwargs.items() if key not in {"data", "project", "name"}},
        "decisionSummary": decision_summary(task_type, metrics, sample_count),
        "errorTaxonomy": empty_error_taxonomy(task_type),
        "limitations": "YOLO detection/segmentation evaluation metrics are produced exclusively by Ultralytics official val(). AITrain local AP/mAP, mask IoU, TP/FP/FN, local error samples, and local overlays are disabled for this report.",
    }

    report_path = output_path / "evaluation_report.json"
    summary_path = output_path / "evaluation_summary.md"
    write_json(report_path, report)
    write_summary(summary_path, report)
    emit("artifact", name="evaluation_report.json", kind="evaluation_report", path=str(report_path))
    emit("artifact", name="ultralytics_official_metrics.json", kind="official_metrics", path=str(metrics_path))
    emit("artifact", name="ultralytics_official_val", kind="official_run_dir", path=str(save_dir))
    emit("completed", reportPath=str(report_path), metrics=metrics, officialRunDir=str(save_dir))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = read_request(args.request)
    except Exception as exc:
        output_path = Path("aitrain-yolo-evaluation").resolve()
        report_path = write_failure_report(output_path, {}, f"failed to read evaluation request: {exc}", "bad_request", exception_details(exc))
        emit("failed", code="bad_request", reportPath=str(report_path), message=f"failed to read evaluation request: {exc}")
        return 2
    try:
        return run(request)
    except Exception as exc:
        output_path = Path(str(request.get("outputPath") or "aitrain-yolo-evaluation")).resolve()
        report_path = write_failure_report(output_path, request, f"Unhandled Ultralytics evaluation failure: {exc}", "ultralytics_evaluator_failed", exception_details(exc))
        emit("failed", code="ultralytics_evaluator_failed", reportPath=str(report_path), message=f"Unhandled Ultralytics evaluation failure: {exc}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
