#!/usr/bin/env python3
"""Ultralytics YOLO OBB trainer adapter for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from trainer_protocol import configure_stdio, exception_details  # noqa: E402


DETECTION_ADAPTER_DIR = Path(__file__).resolve().parents[1] / "detection"
if str(DETECTION_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_ADAPTER_DIR))

import ultralytics_trainer as shared  # type: ignore  # noqa: E402


BACKEND_ID = "ultralytics_yolo_obb"
configure_stdio()


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
        shared.configure_adapter(BACKEND_ID)
        return shared.fail(f"failed to read trainer request: {exc}", "bad_request", exception_details(exc))

    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
        request["parameters"] = parameters
    parameters.setdefault("trainingBackend", BACKEND_ID)
    parameters.setdefault("model", "yolo11n-obb.pt")
    parameters.setdefault("runName", "aitrain-yolo-obb")
    parameters.setdefault("modelFamily", "yolo_obb")

    shared.BACKEND_ID = BACKEND_ID
    try:
        shared.configure_adapter(BACKEND_ID)
        return shared.run(request)
    except Exception as exc:
        return shared.fail(
            "Python trainer failed with an unhandled exception (trainer_unhandled_exception).",
            "trainer_unhandled_exception",
            exception_details(exc),
        )
    finally:
        shared.close_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
