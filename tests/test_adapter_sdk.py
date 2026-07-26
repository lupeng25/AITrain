#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import socket
import threading
import uuid
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINERS = ROOT / "python_trainers"
if str(TRAINERS) not in sys.path:
    sys.path.insert(0, str(TRAINERS))

from adapter_event_channel import (  # noqa: E402
    AdapterEventChannelError,
    AdapterEventChannel,
    domain_failure_code,
)
from adapter_sdk import (  # noqa: E402
    MAX_BUFFERED_OUTPUT_LINES,
    MAX_STRUCTURED_LOG_MESSAGE_BYTES,
    AdapterSdk,
    AdapterEventTransportUnavailable,
    sdk_from_environment,
)


def test_structured_events_preserve_existing_jsonl_shape() -> None:
    events: list[dict] = []
    sdk = AdapterSdk("official_test", event_sink=events.append)
    sdk.emit_log("hello")
    sdk.emit_progress(12.5, message="running", phase="train")
    sdk.emit_metric("loss", 0.25, epoch=1)
    sdk.emit_artifact_candidate("report", Path("out/report.json"), message="report")
    sdk.emit_completed("done", reportPath="out/report.json")

    assert [item["type"] for item in events] == ["log", "progress", "metric", "artifact", "completed"]
    assert all(item["backend"] == "official_test" for item in events)
    assert events[1]["percent"] == 12.5
    assert events[3]["path"] == "out\\report.json" or events[3]["path"] == "out/report.json"


def test_terminal_sink_failure_cannot_trigger_second_terminal() -> None:
    attempts: list[dict] = []

    def failing_sink(event: dict) -> None:
        attempts.append(event)
        raise RuntimeError("sink failed")

    sdk = AdapterSdk("official_test", event_sink=failing_sink)
    with pytest.raises(RuntimeError, match="sink failed"):
        sdk.emit_completed("done")
    with pytest.raises(RuntimeError, match="already attempted"):
        sdk.emit_failed("fallback", "fallback_failed")
    assert len(attempts) == 1


def test_events_are_rejected_after_terminal_attempt() -> None:
    events: list[dict] = []
    sdk = AdapterSdk("official_test", event_sink=events.append)
    sdk.emit_canceled("canceled")
    with pytest.raises(RuntimeError, match="after terminal attempt"):
        sdk.emit_progress(100, message="late")
    assert [event["type"] for event in events] == ["canceled"]


def test_run_child_process_streams_log_and_records_exit_code() -> None:
    events: list[dict] = []
    with tempfile.TemporaryDirectory() as temporary:
        log_path = Path(temporary) / "adapter.log"
        sdk = AdapterSdk("official_test", event_sink=events.append)
        result = sdk.run_child_process(
            [sys.executable, "-c", "print('first'); print('second')"],
            log_path=log_path,
            poll_interval_seconds=0.05,
        )

        assert result.exit_code == 0
        assert result.canceled is False
        assert result.line_count == 2
        assert result.tail_lines == ("first", "second")
        assert log_path.read_text(encoding="utf-8") == "first\nsecond\n"
        assert any(item["type"] == "log" and item.get("childProcess") for item in events)
        assert any(
            item["type"] == "artifact"
            and item["kind"] == "official_process_log"
            and Path(item["path"]) == log_path
            for item in events
        )
        assert events[-1]["exitCode"] == 0


def test_large_child_log_is_streamed_in_full_but_structured_tail_is_bounded() -> None:
    events: list[dict] = []
    payload = "日志" * (MAX_STRUCTURED_LOG_MESSAGE_BYTES // 2)
    with tempfile.TemporaryDirectory() as temporary:
        log_path = Path(temporary) / "complete.log"
        sdk = AdapterSdk("official_test", event_sink=events.append)
        result = sdk.run_child_process(
            [sys.executable, "-c", f"print('日志' * {MAX_STRUCTURED_LOG_MESSAGE_BYTES // 2})"],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            log_path=log_path,
            poll_interval_seconds=0.05,
        )

        assert result.exit_code == 0
        assert log_path.read_text(encoding="utf-8") == payload + "\n"
        child_log = next(item for item in events if item["type"] == "log" and item.get("childProcess"))
        assert len(child_log["message"].encode("utf-8")) <= MAX_STRUCTURED_LOG_MESSAGE_BYTES
        assert child_log["truncated"] is True
        assert child_log["originalUtf8Bytes"] == len(payload.encode("utf-8"))
        assert len(result.tail_lines[0].encode("utf-8")) <= MAX_STRUCTURED_LOG_MESSAGE_BYTES


def test_child_output_queue_has_a_fixed_memory_window() -> None:
    assert MAX_BUFFERED_OUTPUT_LINES == 256


def test_cancel_file_requests_direct_child_termination() -> None:
    events: list[dict] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        cancel_file = root / "cancel.request"
        cancel_file.write_text("cancel", encoding="utf-8")
        sdk = AdapterSdk("official_test", event_sink=events.append, cancel_file=cancel_file)
        result = sdk.run_child_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            poll_interval_seconds=0.05,
        )

        assert result.canceled is True
        assert result.exit_code != 0
        assert any("Cancellation requested" in item.get("message", "") for item in events)


def test_sdk_from_environment_observes_cancel_file(monkeypatch=None) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        cancel_file = Path(temporary) / "cancel.request"
        cancel_file.write_text("cancel", encoding="utf-8")
        previous = __import__("os").environ.get("AITRAIN_CANCEL_FILE")
        try:
            __import__("os").environ["AITRAIN_CANCEL_FILE"] = str(cancel_file)
            assert sdk_from_environment("official_test", event_sink=lambda _: None).check_canceled() is True
        finally:
            if previous is None:
                __import__("os").environ.pop("AITRAIN_CANCEL_FILE", None)
            else:
                __import__("os").environ["AITRAIN_CANCEL_FILE"] = previous


def test_event_channel_authenticates_and_emits_protocol_envelopes() -> None:
    received: list[dict] = []
    ready = threading.Event()
    result: dict[str, int] = {}

    def server() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            result["port"] = listener.getsockname()[1]
            ready.set()
            connection, _ = listener.accept()
            with connection:
                reader = connection.makefile("rb")
                received.append(__import__("json").loads(reader.readline().decode("utf-8")))
                connection.sendall(b'{"status":"accepted"}\n')
                received.append(__import__("json").loads(reader.readline().decode("utf-8")))

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)
    channel = AdapterEventChannel(
        host="127.0.0.1",
        port=result["port"],
        token="0123456789abcdef",
        request_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
    )
    with channel:
        envelope = channel.emit_event({"type": "progress", "backend": "official_test", "percent": 12.5})
    thread.join(timeout=2)

    assert received[0] == {"channel": "aitrain.adapter", "token": "0123456789abcdef"}
    assert received[1] == envelope
    assert envelope["protocol"] == 2
    assert envelope["kind"] == "event.progress"
    assert envelope["sequence"] == "1"
    assert envelope["payload"]["percent"] == 12.5
    assert envelope["payload"]["taskId"] == channel._task_id


def test_event_channel_marks_sdk_artifacts_as_uncommitted_candidates() -> None:
    channel = AdapterEventChannel(
        host="127.0.0.1",
        port=1,
        token="0123456789abcdef",
        request_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
    )
    captured: list[tuple[str, dict]] = []
    channel.send_event = lambda kind, payload: captured.append((kind, dict(payload))) or {"kind": kind}  # type: ignore[method-assign]
    result = channel.emit_event({"type": "artifact", "kind": "report", "path": "out/report.json"})

    assert result["kind"] == "event.artifact_candidate"
    assert captured == [("event.artifact_candidate", {"kind": "report", "path": "out/report.json"})]


def test_event_channel_preserves_adapter_failure_code_and_maps_domain_code() -> None:
    channel = AdapterEventChannel(
        host="127.0.0.1", port=1, token="0123456789abcdef",
        request_id=str(uuid.uuid4()), task_id=str(uuid.uuid4()),
    )
    captured: list[tuple[str, dict]] = []
    channel.send_event = lambda kind, payload: captured.append((kind, dict(payload))) or {"kind": kind}  # type: ignore[method-assign]
    channel.emit_event({"type": "failed", "code": "ultralytics_missing", "message": "missing"})

    assert captured == [("event.failed", {
        "message": "missing", "adapterCode": "ultralytics_missing",
        "originCode": "ultralytics_missing", "failureCode": "dependency_missing",
    })]


def test_adapter_failure_code_mapping_keeps_known_domain_categories() -> None:
    assert domain_failure_code("bad_request") == "invalid_request"
    assert domain_failure_code("dataset_snapshot_invalid") == "invalid_dataset"
    assert domain_failure_code("ultralytics_missing") == "dependency_missing"
    assert domain_failure_code("anomalib_evaluation_blocked") == "internal_error"


def test_sdk_rejects_non_finite_metrics_and_progress() -> None:
    sdk = AdapterSdk("official_test", event_sink=lambda _: None)
    try:
        sdk.emit_metric("loss", float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN metric was accepted")
    try:
        sdk.emit_progress(float("inf"), message="invalid")
    except ValueError:
        pass
    else:
            raise AssertionError("infinite progress was accepted")


def test_sdk_requires_authenticated_transport_without_standalone_opt_in(monkeypatch=None) -> None:
    previous = os.environ.pop("AITRAIN_STANDALONE_ADAPTER_PROTOCOL", None)
    try:
        sdk = AdapterSdk("official_test")
        try:
            sdk.emit_log("transport must be explicit")
        except AdapterEventTransportUnavailable as exc:
            assert "authenticated Worker event channel" in str(exc)
        else:
            raise AssertionError("SDK silently emitted an unauthenticated stdout event")
    finally:
        if previous is not None:
            os.environ["AITRAIN_STANDALONE_ADAPTER_PROTOCOL"] = previous


def test_event_channel_rejects_non_loopback_and_bad_handshake() -> None:
    try:
        AdapterEventChannel(
            host="192.168.1.10", port=1234, token="0123456789abcdef",
            request_id=str(uuid.uuid4()), task_id=str(uuid.uuid4()),
        )
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("non-loopback host was accepted")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def reject() -> None:
            connection, _ = listener.accept()
            with connection:
                connection.recv(4096)
                connection.sendall(b'{"status":"rejected"}\n')

        thread = threading.Thread(target=reject, daemon=True)
        thread.start()
        channel = AdapterEventChannel(
            host="127.0.0.1", port=port, token="0123456789abcdef",
            request_id=str(uuid.uuid4()), task_id=str(uuid.uuid4()),
        )
        try:
            channel.connect()
        except AdapterEventChannelError as exc:
            assert "rejected" in str(exc)
        else:
            raise AssertionError("rejected handshake was accepted")
        thread.join(timeout=2)


if __name__ == "__main__":
    os.environ.setdefault("AITRAIN_STANDALONE_ADAPTER_PROTOCOL", "1")
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
