#!/usr/bin/env python3
"""PaddleOCR DET V2 predictor entry point."""

from __future__ import annotations

import sys
from pathlib import Path

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from ocr.paddleocr_workflow_common import cli_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli_main("det", "infer"))
