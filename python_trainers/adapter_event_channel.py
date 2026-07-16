#!/usr/bin/env python3
"""Authenticated loopback event client for a future AITrain  Worker Host.

Wire contract:
1. the first JSONL frame is ``{"channel":"aitrain.adapter","token":"..."}``;
2. the host replies ``{"status":"accepted"}`` before any task event is sent;
3. all following frames are Protocol envelopes, one compact JSON object per
   line.  The token is deliberately never repeated in task events.

This module is a client-side contract only.  It does not make the existing V1
Worker accept  adapter connections.
"""

from __future__ import annotations

from datetime import datetime, timezone
import ipaddress
import json
import os
import socket
from typing import Any, Mapping
from uuid import UUID, uuid4


PROTOCOL_VERSION = 1
CHANNEL_NAME = "aitrain.adapter"
MAX_CONTROL_MESSAGE_BYTES = 1024 * 1024
MAX_LOG_MESSAGE_BYTES = 64 * 1024
_KNOWN_KINDS = frozenset({
    "event.ready",
    "event.progress",
    "event.metric",
    "event.artifact",
    "event.artifact_candidate",
    "event.log",
    "event.succeeded",
    "event.failed",
    "event.canceled",
})
_LEGACY_EVENT_KIND_MAP = {
    "log": "event.log",
    "progress": "event.progress",
    "metric": "event.metric",
    "artifact": "event.artifact_candidate",
    "completed": "event.succeeded",
    "failed": "event.failed",
    "canceled": "event.canceled",
}


class AdapterEventChannelError(RuntimeError):
    """The  adapter event channel could not be established or used."""


def _validate_uuid(value: str, field_name: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc
    return str(parsed)


def _validate_loopback_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("event channel host must be a literal loopback address") from exc
    if not address.is_loopback:
        raise ValueError("event channel host must be a loopback address")
    return str(address)


def _compact_json_line(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class AdapterEventChannel:
    """One authenticated loopback connection for exactly one request/task pair."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str,
        request_id: str,
        task_id: str,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._host = _validate_loopback_host(host)
        if not 1 <= port <= 65535:
            raise ValueError("event channel port must be within [1, 65535]")
        if len(token) < 16:
            raise ValueError("event channel token must contain at least 16 characters")
        if timeout_seconds <= 0:
            raise ValueError("event channel timeout must be positive")
        self._port = port
        self._token = token
        self._request_id = _validate_uuid(request_id, "request_id")
        self._task_id = _validate_uuid(task_id, "task_id")
        self._timeout_seconds = timeout_seconds
        self._sequence = 0
        self._socket: socket.socket | None = None
        self._reader = None

    def connect(self) -> None:
        if self._socket is not None:
            raise AdapterEventChannelError("event channel is already connected")
        connection = socket.create_connection((self._host, self._port), timeout=self._timeout_seconds)
        connection.settimeout(self._timeout_seconds)
        reader = connection.makefile("rb")
        try:
            connection.sendall(_compact_json_line({"channel": CHANNEL_NAME, "token": self._token}))
            reply_line = reader.readline(MAX_CONTROL_MESSAGE_BYTES + 1)
            if not reply_line or len(reply_line) > MAX_CONTROL_MESSAGE_BYTES:
                raise AdapterEventChannelError("event channel did not return a valid handshake response")
            reply = json.loads(reply_line.decode("utf-8"))
            if not isinstance(reply, dict) or reply.get("status") != "accepted":
                raise AdapterEventChannelError("event channel authentication was rejected")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            connection.close()
            raise AdapterEventChannelError("event channel handshake failed") from exc
        except Exception:
            connection.close()
            raise
        self._socket = connection
        self._reader = reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "AdapterEventChannel":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def send_event(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            raise AdapterEventChannelError("event channel is not connected")
        if kind not in _KNOWN_KINDS:
            raise ValueError(f"unsupported Protocol event kind: {kind}")
        self._sequence += 1
        event = {
            "protocol": PROTOCOL_VERSION,
            "messageId": str(uuid4()),
            "requestId": self._request_id,
            "taskId": self._task_id,
            "sequence": str(self._sequence),
            "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "payload": dict(payload),
        }
        wire = _compact_json_line(event)
        max_bytes = MAX_LOG_MESSAGE_BYTES if kind == "event.log" else MAX_CONTROL_MESSAGE_BYTES
        if len(wire) > max_bytes:
            self._sequence -= 1
            raise ValueError(f"Protocol message exceeds maximum size for kind {kind}")
        try:
            self._socket.sendall(wire)
        except OSError as exc:
            raise AdapterEventChannelError("event channel write failed") from exc
        return event

    def emit_legacy_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Bridge the SDK's current event naming to a Protocol envelope."""
        event_type = str(event.get("type") or "")
        kind = _LEGACY_EVENT_KIND_MAP.get(event_type)
        if kind is None:
            raise ValueError(f"unsupported adapter SDK event type: {event_type}")
        payload = dict(event)
        payload.pop("type", None)
        if event_type == "failed":
            adapter_code = str(payload.pop("code", ""))
            if adapter_code:
                payload["adapterCode"] = adapter_code
            payload["failureCode"] = "internal_error"
        return self.send_event(kind, payload)


def event_channel_from_environment() -> AdapterEventChannel:
    """Load a  event channel only from explicit Worker Host environment."""
    required = {
        "host": os.environ.get("AITRAIN_EVENT_HOST"),
        "port": os.environ.get("AITRAIN_EVENT_PORT"),
        "token": os.environ.get("AITRAIN_EVENT_TOKEN"),
        "request_id": os.environ.get("AITRAIN_REQUEST_ID"),
        "task_id": os.environ.get("AITRAIN_TASK_ID"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise AdapterEventChannelError(f"missing  event channel environment values: {', '.join(missing)}")
    try:
        port = int(str(required["port"]))
    except ValueError as exc:
        raise AdapterEventChannelError("AITRAIN_EVENT_PORT must be an integer") from exc
    return AdapterEventChannel(
        host=str(required["host"]),
        port=port,
        token=str(required["token"]),
        request_id=str(required["request_id"]),
        task_id=str(required["task_id"]),
    )
