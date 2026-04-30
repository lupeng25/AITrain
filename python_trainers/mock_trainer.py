#!/usr/bin/env python3
"""AITrain Studio Phase 8 mock Python trainer.

This is a protocol fixture, not real model training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def emit(message_type: str, payload: dict) -> None:
    sys.stdout.write(json.dumps({"type": message_type, "payload": payload}, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    task_id = request.get("taskId", "")
    task_type = request.get("taskType", "train")
    backend = request.get("backend", "python_mock")
    output_path = Path(request.get("outputPath") or request_path.parent)
    parameters = request.get("parameters") or {}
    epochs = max(1, int(parameters.get("epochs", 1)))
    steps_per_epoch = max(1, int(parameters.get("mockStepsPerEpoch", 2)))
    sleep_ms = max(0, int(parameters.get("mockSleepMs", 5)))
    mode = str(parameters.get("mockMode", "ok")).lower()

    output_path.mkdir(parents=True, exist_ok=True)

    emit("log", {
        "taskId": task_id,
        "message": f"Starting mock Python trainer backend={backend} taskType={task_type}",
        "backend": backend,
        "scaffold": True,
    })

    if mode == "fail":
        emit("failed", {
            "taskId": task_id,
            "message": "Mock Python trainer failure requested",
            "backend": backend,
        })
        return 2

    total_steps = epochs * steps_per_epoch
    step = 0
    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            step += 1
            loss = max(0.05, 1.0 - (0.75 * step / total_steps))
            quality = min(0.95, 0.2 + (0.65 * step / total_steps))
            percent = min(99, round(100.0 * step / total_steps))
            emit("progress", {
                "taskId": task_id,
                "percent": percent,
                "step": step,
                "epoch": epoch,
                "backend": backend,
            })
            emit("metric", {
                "taskId": task_id,
                "name": "loss",
                "value": loss,
                "step": step,
                "epoch": epoch,
            })
            emit("metric", {
                "taskId": task_id,
                "name": "mAP50" if task_type != "ocr_recognition" else "accuracy",
                "value": quality,
                "step": step,
                "epoch": epoch,
            })
            emit("log", {
                "taskId": task_id,
                "message": f"epoch={epoch} step={step} loss={loss:.4f}",
                "backend": backend,
            })
            if sleep_ms:
                time.sleep(sleep_ms / 1000.0)

    checkpoint_path = output_path / "python_mock_checkpoint.json"
    report_path = output_path / "python_mock_training_report.json"
    checkpoint = {
        "type": "python_mock_trainer_checkpoint",
        "taskId": task_id,
        "taskType": task_type,
        "backend": backend,
        "scaffold": True,
        "steps": step,
    }
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps({
        "ok": True,
        "taskId": task_id,
        "backend": backend,
        "steps": step,
        "finalLoss": loss,
    }, indent=2), encoding="utf-8")

    emit("artifact", {
        "taskId": task_id,
        "kind": "checkpoint",
        "path": str(checkpoint_path),
        "message": "Mock Python trainer checkpoint",
        "backend": backend,
        "scaffold": True,
    })
    emit("artifact", {
        "taskId": task_id,
        "kind": "training_report",
        "path": str(report_path),
        "message": "Mock Python trainer report",
        "backend": backend,
        "scaffold": True,
    })
    emit("progress", {
        "taskId": task_id,
        "percent": 100,
        "step": step,
        "epoch": epochs,
        "backend": backend,
    })
    emit("completed", {
        "taskId": task_id,
        "message": "Mock Python trainer completed",
        "backend": backend,
        "scaffold": True,
        "checkpointPath": str(checkpoint_path),
        "reportPath": str(report_path),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
