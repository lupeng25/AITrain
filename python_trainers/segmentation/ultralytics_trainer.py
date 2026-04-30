#!/usr/bin/env python3
"""Ultralytics YOLO segmentation trainer adapter for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DETECTION_ADAPTER_DIR = Path(__file__).resolve().parents[1] / "detection"
if str(DETECTION_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_ADAPTER_DIR))

import ultralytics_trainer as shared  # type: ignore  # noqa: E402


BACKEND_ID = "ultralytics_yolo_segment"


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trainer request must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()

    try:
        request = read_request(args.request)
    except Exception as exc:
        shared.BACKEND_ID = BACKEND_ID
        return shared.fail(f"failed to read trainer request: {exc}", "bad_request")

    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
        request["parameters"] = parameters
    parameters.setdefault("trainingBackend", BACKEND_ID)
    parameters.setdefault("model", "yolov8n-seg.yaml")
    parameters.setdefault("runName", "aitrain-yolo-segment")

    shared.BACKEND_ID = BACKEND_ID
    return shared.run(request)


if __name__ == "__main__":
    raise SystemExit(main())
