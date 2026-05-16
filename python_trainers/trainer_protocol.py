#!/usr/bin/env python3
"""Shared JSONL protocol helpers for AITrain Python trainers."""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any


UNHANDLED_EXCEPTION_CODE = "trainer_unhandled_exception"


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def emit_event(backend: str, event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time(), "backend": backend}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def exception_details(exc: BaseException, *, include_traceback: bool = False) -> dict[str, Any]:
    details: dict[str, Any] = {
        "exceptionType": type(exc).__name__,
        "exception": str(exc),
    }
    if include_traceback:
        details["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return details


def emit_failed(backend: str, message: str, code: str, details: dict[str, Any] | None = None) -> int:
    emit_event(backend, "failed", code=code, message=message, details=details or {})
    return 1


def unhandled_failure(backend: str, exc: BaseException) -> int:
    return emit_failed(
        backend,
        "Python trainer failed with an unhandled exception (trainer_unhandled_exception).",
        UNHANDLED_EXCEPTION_CODE,
        exception_details(exc),
    )
