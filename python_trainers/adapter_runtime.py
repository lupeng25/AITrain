#!/usr/bin/env python3
"""AITrain Python Adapter 的轻量进程生命周期边界。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapter_event_channel import (
    AdapterEventChannel,
    event_channel_from_environment,
    standalone_protocol_enabled,
)
from adapter_sdk import AdapterSdk


class AdapterRuntime:
    """统一请求读取、事件通道、SDK 创建和关闭，不封装算法实现。"""

    def __init__(self) -> None:
        self._channel: AdapterEventChannel | None = None
        self._sdk: AdapterSdk | None = None

    @staticmethod
    def read_request(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("adapter request root must be a JSON object")
        return value

    def sdk(self, backend: str) -> AdapterSdk:
        selected = str(backend).strip()
        if not selected:
            raise ValueError("backend must not be empty")
        if self._sdk is not None:
            if self._sdk.backend != selected:
                raise RuntimeError(
                    f"adapter runtime backend is already bound to {self._sdk.backend}"
                )
            return self._sdk
        if not standalone_protocol_enabled():
            self._channel = event_channel_from_environment()
            self._channel.connect()
        sink = self._channel.emit_event if self._channel is not None else None
        self._sdk = AdapterSdk(selected, event_sink=sink)
        return self._sdk

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._sdk = None

    def __enter__(self) -> "AdapterRuntime":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
