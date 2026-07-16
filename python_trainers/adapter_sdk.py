#!/usr/bin/env python3
"""V2 Python Adapter SDK.

The SDK keeps the existing JSONL event shape while giving adapters one small,
testable boundary for structured events, cooperative cancellation and child
process execution.  Process-tree ownership remains with the V2 Worker Host;
this module only owns the direct child it starts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from trainer_protocol import emit_event


EventSink = Callable[[dict[str, Any]], None]
CancellationCheck = Callable[[], bool]

# Protocol V2 allows a complete event.log envelope up to 64 KiB.  Keep the
# message itself below 48 KiB so envelope metadata and multibyte JSON escaping
# cannot push a child-process log frame over the wire limit.
MAX_STRUCTURED_LOG_MESSAGE_BYTES = 48 * 1024
MAX_BUFFERED_OUTPUT_LINES = 256


class AdapterCanceled(RuntimeError):
    """Raised when an adapter observes a cooperative cancellation request."""


@dataclass(frozen=True)
class ChildProcessResult:
    """Immutable outcome of one SDK-managed direct child process."""

    exit_code: int
    canceled: bool
    line_count: int
    tail_lines: tuple[str, ...]
    log_path: Path | None


class AdapterSdk:
    """Emit V2-compatible adapter events and run a cancellable direct child."""

    def __init__(
        self,
        backend: str,
        *,
        event_sink: EventSink | None = None,
        cancellation_check: CancellationCheck | None = None,
        cancel_file: str | Path | None = None,
    ) -> None:
        if not backend.strip():
            raise ValueError("backend must not be empty")
        self._backend = backend
        self._event_sink = event_sink
        self._cancellation_check = cancellation_check
        self._cancel_file = Path(cancel_file) if cancel_file is not None else None

    @property
    def backend(self) -> str:
        return self._backend

    def _emit(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload, "backend": self._backend}
        if self._event_sink is not None:
            self._event_sink(event)
            return
        emit_event(self._backend, event_type, **payload)

    def emit_log(self, message: str, *, level: str = "info", **details: Any) -> None:
        encoded = message.encode("utf-8")
        if len(encoded) > MAX_STRUCTURED_LOG_MESSAGE_BYTES:
            shortened = encoded[:MAX_STRUCTURED_LOG_MESSAGE_BYTES]
            while shortened:
                try:
                    message = shortened.decode("utf-8")
                    break
                except UnicodeDecodeError as exc:
                    shortened = shortened[:exc.start]
            details = {
                **details,
                "truncated": True,
                "originalUtf8Bytes": len(encoded),
            }
        self._emit("log", level=level, message=message, **details)

    def emit_progress(self, percent: float, *, message: str, **details: Any) -> None:
        if percent < 0 or percent > 100:
            raise ValueError("progress percent must be within [0, 100]")
        self._emit("progress", percent=float(percent), message=message, **details)

    def emit_metric(self, name: str, value: float, **details: Any) -> None:
        if not name.strip():
            raise ValueError("metric name must not be empty")
        self._emit("metric", name=name, value=float(value), **details)

    def emit_artifact_candidate(self, kind: str, path: str | Path, *, message: str = "", **details: Any) -> None:
        if not kind.strip():
            raise ValueError("artifact kind must not be empty")
        artifact_path = str(path)
        if not artifact_path:
            raise ValueError("artifact path must not be empty")
        self._emit("artifact", kind=kind, path=artifact_path, message=message, **details)

    def emit_completed(self, message: str, **details: Any) -> None:
        self._emit("completed", message=message, **details)

    def emit_failed(self, message: str, code: str, details: Mapping[str, Any] | None = None) -> int:
        if not code.strip():
            raise ValueError("failure code must not be empty")
        self._emit("failed", message=message, code=code, details=dict(details or {}))
        return 1

    def emit_canceled(self, message: str = "Canceled by request", *, force: bool = False, **details: Any) -> None:
        self._emit("canceled", message=message, force=force, **details)

    def check_canceled(self) -> bool:
        if self._cancellation_check is not None and self._cancellation_check():
            return True
        return self._cancel_file is not None and self._cancel_file.exists()

    def raise_if_canceled(self) -> None:
        if self.check_canceled():
            raise AdapterCanceled("adapter cancellation requested")

    def run_child_process(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        log_path: str | Path | None = None,
        poll_interval_seconds: float = 0.2,
        tail_line_limit: int = 100,
        on_line: Callable[[str], None] | None = None,
    ) -> ChildProcessResult:
        """Run one direct child, stream combined output and honor cancellation.

        Command arguments and environment values are intentionally not emitted.
        Callers can persist their own redacted command/environment summaries in a
        report.  A V2 Worker Host must attach the root process to its Job Object
        to guarantee cleanup of descendants.
        """
        if not command or not command[0]:
            raise ValueError("command must contain an executable")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if tail_line_limit < 1:
            raise ValueError("tail_line_limit must be at least 1")

        resolved_log_path = Path(log_path) if log_path is not None else None
        if resolved_log_path is not None:
            resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.emit_log("Starting SDK-managed child process.", level="info")
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None

        # A noisy official backend must apply pipe backpressure instead of
        # growing the Adapter process indefinitely.  The complete stream is
        # still written to log_path by the consumer below.
        output: Queue[str | None] = Queue(maxsize=MAX_BUFFERED_OUTPUT_LINES)

        def read_output() -> None:
            try:
                for line in process.stdout:
                    output.put(line)
            finally:
                output.put(None)

        reader = threading.Thread(target=read_output, name="aitrain-adapter-output", daemon=True)
        reader.start()

        tail: deque[str] = deque(maxlen=tail_line_limit)
        line_count = 0
        canceled = False
        reader_done = False
        log_file = resolved_log_path.open("w", encoding="utf-8") if resolved_log_path is not None else None
        try:
            while not reader_done or process.poll() is None:
                if not canceled and self.check_canceled():
                    canceled = True
                    self.emit_log("Cancellation requested; terminating direct child process.", level="warning")
                    process.terminate()
                try:
                    line = output.get(timeout=poll_interval_seconds)
                except Empty:
                    continue
                if line is None:
                    reader_done = True
                    continue
                text = line.rstrip("\r\n")
                if not text:
                    continue
                line_count += 1
                tail_text = text
                tail_bytes = tail_text.encode("utf-8")
                if len(tail_bytes) > MAX_STRUCTURED_LOG_MESSAGE_BYTES:
                    tail_text = tail_bytes[:MAX_STRUCTURED_LOG_MESSAGE_BYTES].decode("utf-8", errors="ignore")
                tail.append(tail_text)
                if log_file is not None:
                    log_file.write(text + "\n")
                    log_file.flush()
                if on_line is not None:
                    on_line(text)
                self.emit_log(text, level="info", childProcess=True)

            try:
                exit_code = process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                canceled = True
                process.kill()
                exit_code = process.wait()
        finally:
            if log_file is not None:
                log_file.close()
            if process.poll() is None:
                process.kill()
                process.wait()
            reader.join(timeout=1.0)

        if resolved_log_path is not None and resolved_log_path.is_file():
            self.emit_artifact_candidate(
                "official_process_log",
                resolved_log_path,
                message="Complete SDK-managed child-process log.",
            )
        self.emit_log(
            "SDK-managed child process finished.",
            level="warning" if canceled else ("info" if exit_code == 0 else "error"),
            exitCode=exit_code,
            canceled=canceled,
            lineCount=line_count,
            logPath=str(resolved_log_path) if resolved_log_path is not None else "",
        )
        return ChildProcessResult(exit_code, canceled, line_count, tuple(tail), resolved_log_path)


def sdk_from_environment(backend: str, *, event_sink: EventSink | None = None) -> AdapterSdk:
    """Create an SDK that observes the V2 host's optional cancel-file signal."""
    cancel_file = os.environ.get("AITRAIN_CANCEL_FILE")
    return AdapterSdk(backend, event_sink=event_sink, cancel_file=cancel_file)
