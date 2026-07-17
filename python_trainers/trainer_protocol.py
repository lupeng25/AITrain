#!/usr/bin/env python3
"""Shared runtime helpers for AITrain Python adapters."""

from __future__ import annotations

import sys
import traceback
from typing import Any


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def exception_details(exc: BaseException, *, include_traceback: bool = False) -> dict[str, Any]:
    details: dict[str, Any] = {
        "exceptionType": type(exc).__name__,
        "exception": str(exc),
    }
    if include_traceback:
        details["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return details
