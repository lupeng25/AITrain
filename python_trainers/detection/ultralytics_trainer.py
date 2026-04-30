#!/usr/bin/env python3
"""Ultralytics YOLO detection trainer adapter for AITrain Studio.

This adapter intentionally keeps the official training implementation in
Python while AITrain Worker owns process lifetime, request routing, and JSONL
event forwarding.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


BACKEND_ID = "ultralytics_yolo_detect"


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time()}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def fail(message: str, code: str = "ultralytics_trainer_failed", details: dict[str, Any] | None = None) -> int:
    emit("failed", backend=BACKEND_ID, code=code, message=message, details=details or {})
    return 1


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trainer request must be a JSON object")
    return value


def as_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
    return f"\"{escaped}\""


def read_existing_data_yaml(dataset_path: Path) -> tuple[int | None, list[str] | None]:
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        return None, None

    text = yaml_path.read_text(encoding="utf-8-sig")
    nc_match = re.search(r"(?m)^\s*nc\s*:\s*(\d+)\s*$", text)
    nc = int(nc_match.group(1)) if nc_match else None

    names_match = re.search(r"(?ms)^\s*names\s*:\s*(\[[^\]]*\])", text)
    names: list[str] | None = None
    if names_match:
        raw_names = names_match.group(1).strip()[1:-1]
        names = []
        for part in raw_names.split(","):
            name = part.strip().strip("'\"")
            if name:
                names.append(name)
    else:
        block_match = re.search(r"(?ms)^\s*names\s*:\s*\n((?:\s+\d+\s*:\s*.+\n?)+)", text)
        if block_match:
            names = []
            for line in block_match.group(1).splitlines():
                item = re.match(r"\s*(\d+)\s*:\s*(.+)\s*$", line)
                if item:
                    names.append(item.group(2).strip().strip("'\""))

    return nc, names


def normalize_data_yaml(dataset_path: Path, output_path: Path) -> Path:
    nc, names = read_existing_data_yaml(dataset_path)
    if names:
        class_names = names
    else:
        inferred_nc = nc if nc and nc > 0 else 1
        class_names = [f"class_{index}" for index in range(inferred_nc)]

    if nc is not None and nc > 0 and len(class_names) != nc:
        if len(class_names) < nc:
            class_names.extend(f"class_{index}" for index in range(len(class_names), nc))
        else:
            class_names = class_names[:nc]

    data_yaml = output_path / "aitrain_yolo_data.yaml"
    lines = [
        f"path: {yaml_scalar(dataset_path.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {yaml_scalar(name)}" for index, name in enumerate(class_names))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


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


def resolve_save_dir(train_result: Any, project_dir: Path, run_name: str) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir:
        return Path(save_dir)

    expected = project_dir / run_name
    if expected.exists():
        return expected

    candidates = [item for item in project_dir.iterdir() if item.is_dir()] if project_dir.exists() else []
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_mtime)
    return expected


def parse_results_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    last = rows[-1]
    aliases = {
        "boxLoss": ["train/box_loss", "box_loss"],
        "classLoss": ["train/cls_loss", "cls_loss"],
        "dflLoss": ["train/dfl_loss", "dfl_loss"],
        "precision": ["metrics/precision(B)", "precision"],
        "recall": ["metrics/recall(B)", "recall"],
        "mAP50": ["metrics/mAP50(B)", "mAP50"],
        "mAP50_95": ["metrics/mAP50-95(B)", "mAP50-95"],
    }

    metrics: dict[str, float] = {}
    for output_name, source_names in aliases.items():
        for source_name in source_names:
            if source_name not in last:
                continue
            try:
                metrics[output_name] = float(str(last[source_name]).strip())
                break
            except ValueError:
                continue

    loss_parts = [metrics.get("boxLoss"), metrics.get("classLoss"), metrics.get("dflLoss")]
    available_loss_parts = [value for value in loss_parts if value is not None]
    if available_loss_parts:
        metrics["loss"] = float(sum(available_loss_parts))
    return metrics


def emit_metrics(metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        emit("metric", backend=BACKEND_ID, name=name, value=value)


def emit_artifact(name: str, path: Path, artifact_kind: str) -> None:
    if path.exists():
        emit("artifact", backend=BACKEND_ID, name=name, path=str(path), kind=artifact_kind)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(request: dict[str, Any]) -> int:
    parameters = request.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}

    prepend_python_paths(parameters)

    dataset_path = Path(str(request.get("datasetPath") or parameters.get("datasetPath") or "")).resolve()
    if not dataset_path.exists():
        return fail(f"dataset path does not exist: {dataset_path}", "dataset_missing")

    output_path = Path(str(request.get("outputPath") or parameters.get("outputPath") or "aitrain-yolo-output")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    data_yaml = normalize_data_yaml(dataset_path, output_path)
    emit("log", backend=BACKEND_ID, level="info", message=f"Prepared Ultralytics data yaml: {data_yaml}")

    try:
        import ultralytics  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        return fail(
            "Ultralytics is not available. Install it with: python -m pip install ultralytics",
            "ultralytics_missing",
            {"exception": str(exc)},
        )
    emit(
        "log",
        backend=BACKEND_ID,
        level="info",
        message=f"Using Ultralytics module: {getattr(ultralytics, '__file__', 'built-in')}",
    )

    model_name = str(parameters.get("model") or "yolov8n.pt")
    epochs = as_int(parameters.get("epochs"), 1, 1)
    batch = as_int(parameters.get("batchSize", parameters.get("batch")), 1, 1)
    image_size = as_int(parameters.get("imageSize", parameters.get("imgsz")), 320, 32)
    workers = as_int(parameters.get("workers"), 0, 0)
    device = str(parameters.get("device") or "cpu")
    run_name = str(parameters.get("runName") or f"aitrain-{int(time.time())}")
    export_onnx = as_bool(parameters.get("exportOnnx"), True)

    project_dir = output_path / "ultralytics_runs"
    emit(
        "log",
        backend=BACKEND_ID,
        level="info",
        message=f"Starting official Ultralytics YOLO detection training: model={model_name}, epochs={epochs}, device={device}",
    )
    emit("progress", backend=BACKEND_ID, value=0.0, message="training started")

    try:
        model = YOLO(model_name)
        train_result = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            device=device,
            workers=workers,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            verbose=False,
        )
    except Exception as exc:
        return fail("Ultralytics training failed.", "ultralytics_train_failed", {"exception": str(exc)})

    save_dir = resolve_save_dir(train_result, project_dir, run_name)
    emit("log", backend=BACKEND_ID, level="info", message=f"Ultralytics training returned save_dir={save_dir}")
    weights_dir = save_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    results_csv = save_dir / "results.csv"
    args_yaml = save_dir / "args.yaml"

    metrics = parse_results_csv(results_csv)
    emit("log", backend=BACKEND_ID, level="info", message=f"Parsed {len(metrics)} training metrics from {results_csv}")
    emit_metrics(metrics)
    emit_artifact("best.pt", best_path, "checkpoint")
    emit_artifact("last.pt", last_path, "checkpoint")
    emit_artifact("results.csv", results_csv, "report")
    emit_artifact("args.yaml", args_yaml, "config")
    emit_artifact("aitrain_yolo_data.yaml", data_yaml, "config")

    onnx_path: Path | None = None
    if export_onnx:
        try:
            emit("log", backend=BACKEND_ID, level="info", message="Starting Ultralytics ONNX export")
            export_model = YOLO(str(best_path if best_path.exists() else model_name))
            exported = export_model.export(format="onnx", imgsz=image_size, device=device)
            emit("log", backend=BACKEND_ID, level="info", message=f"Ultralytics ONNX export returned {exported}")
            if exported:
                onnx_path = Path(str(exported))
            elif best_path.exists():
                onnx_path = best_path.with_suffix(".onnx")
            if onnx_path and onnx_path.exists():
                emit_artifact("model.onnx", onnx_path, "onnx")
            else:
                return fail("Ultralytics ONNX export completed without producing an ONNX file.", "onnx_missing")
        except Exception as exc:
            return fail("Ultralytics ONNX export failed.", "onnx_export_failed", {"exception": str(exc)})

    report_path = output_path / "ultralytics_training_report.json"
    report = {
        "ok": True,
        "backend": BACKEND_ID,
        "model": model_name,
        "datasetPath": str(dataset_path),
        "dataYaml": str(data_yaml),
        "saveDir": str(save_dir),
        "checkpointPath": str(best_path if best_path.exists() else last_path),
        "onnxPath": str(onnx_path) if onnx_path else "",
        "metrics": metrics,
        "licenseNote": "Ultralytics YOLO is executed through the installed official Python package. Review its license before redistribution.",
    }
    write_report(report_path, report)
    emit_artifact("ultralytics_training_report.json", report_path, "report")
    emit("progress", backend=BACKEND_ID, value=1.0, message="training completed")
    emit(
        "completed",
        backend=BACKEND_ID,
        checkpointPath=report["checkpointPath"],
        onnxPath=report["onnxPath"],
        reportPath=str(report_path),
        metrics=metrics,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()

    try:
        request = read_request(args.request)
    except Exception as exc:
        return fail(f"failed to read trainer request: {exc}", "bad_request")
    return run(request)


if __name__ == "__main__":
    raise SystemExit(main())
