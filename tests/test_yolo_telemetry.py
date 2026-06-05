#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER_DIR = ROOT / "python_trainers" / "detection"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

import ultralytics_trainer as trainer  # noqa: E402


class FakeModel:
    def __init__(self) -> None:
        self.callbacks: dict[str, list] = {}

    def add_callback(self, event: str, callback) -> None:
        self.callbacks.setdefault(event, []).append(callback)


class FakeTrainer:
    def __init__(self) -> None:
        self.epoch = 0
        self.train_loader = [object(), object(), object(), object()]
        self.tloss = [1.0, 2.0, 3.0]
        self.metrics = {
            "metrics/precision(B)": 0.25,
            "metrics/recall(B)": 0.5,
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": 0.125,
        }

    def label_loss_items(self, loss_items) -> dict[str, float]:
        return {
            "train/box_loss": float(loss_items[0]),
            "train/cls_loss": float(loss_items[1]),
            "train/dfl_loss": float(loss_items[2]),
        }


def test_sanitize_log_line_removes_ansi_tqdm_noise() -> None:
    dirty = "\x1b[34mtrain:\x1b[0m 50% ━━━━━ 2/4 1.0it/s\r\nuseful message"
    cleaned = trainer.sanitize_log_line(dirty)
    assert "━━━━" not in cleaned
    assert "\x1b" not in cleaned
    assert "useful message" in cleaned


def test_yolo_callbacks_emit_structured_progress_and_epoch_metrics() -> None:
    events: list[tuple[str, dict]] = []
    original_emit = trainer.emit
    trainer.emit = lambda event_type, **payload: events.append((event_type, payload))
    try:
        model = FakeModel()
        fake = FakeTrainer()
        trainer.register_training_callbacks(model, epochs=2, device="cpu")

        model.callbacks["on_train_start"][0](fake)
        model.callbacks["on_train_epoch_start"][0](fake)
        for _ in range(4):
            model.callbacks["on_train_batch_end"][0](fake)
        model.callbacks["on_train_epoch_end"][0](fake)
        model.callbacks["on_fit_epoch_end"][0](fake)
        model.callbacks["on_fit_epoch_end"][0](fake)
    finally:
        trainer.emit = original_emit

    progress = [payload for event_type, payload in events if event_type == "progress"]
    metrics = [payload for event_type, payload in events if event_type == "metric"]

    assert any(item["phase"] == "train" and item["batch"] == 4 and item["batches"] == 4 for item in progress)
    assert any(item["phase"] == "validate" and "liveMetrics" in item for item in progress)
    assert any(item["name"] == "boxLoss" and item["epoch"] == 1 for item in metrics)
    assert any(item["name"] == "mAP50" and item["value"] == 0.75 for item in metrics)
    assert sum(1 for item in metrics if item["name"] == "mAP50" and item["epoch"] == 1) == 1


if __name__ == "__main__":
    test_sanitize_log_line_removes_ansi_tqdm_noise()
    test_yolo_callbacks_emit_structured_progress_and_epoch_metrics()
