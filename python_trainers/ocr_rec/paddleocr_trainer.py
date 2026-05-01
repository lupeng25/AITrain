#!/usr/bin/env python3
"""PaddlePaddle OCR recognition trainer adapter for AITrain Studio.

This is a real PaddlePaddle CTC training loop for PaddleOCR-style Rec data.
It is intentionally small so local CPU smoke tests can complete quickly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BACKEND_ID = "paddleocr_rec"


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time(), "backend": BACKEND_ID}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def fail(message: str, code: str, details: dict[str, Any] | None = None) -> int:
    emit("failed", code=code, message=message, details=details or {})
    return 1


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trainer request must be a JSON object")
    return value


def read_dictionary(path: Path) -> list[str]:
    chars = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not chars:
        raise ValueError(f"empty OCR dictionary: {path}")
    return chars


def read_labels(path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid OCR label line: {line}")
        samples.append((parts[0].strip(), parts[1].strip()))
    if not samples:
        raise ValueError(f"empty OCR label file: {path}")
    return samples


def image_tensor(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((width, height))
    array = np.asarray(image, dtype="float32") / 255.0
    return array[None, :, :]


def edit_distance(left: str, right: str) -> int:
    dp = list(range(len(right) + 1))
    for i, lc in enumerate(left, 1):
        prev = dp[0]
        dp[0] = i
        for j, rc in enumerate(right, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if lc == rc else 1))
            prev = old
    return dp[-1]


def decode(sequence: list[int], chars: list[str]) -> str:
    output: list[str] = []
    previous = -1
    for token in sequence:
        if token != 0 and token != previous:
            index = token - 1
            if 0 <= index < len(chars):
                output.append(chars[index])
        previous = token
    return "".join(output)


def create_model(paddle_module: Any, width: int, height: int, max_text_length: int, class_count: int) -> Any:
    class OcrRecCtcModel(paddle_module.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = paddle_module.nn.Flatten()
            self.head = paddle_module.nn.Linear(width * height, max_text_length * class_count)

        def forward(self, image: Any) -> Any:
            logits = self.head(self.flatten(image))
            return paddle_module.reshape(logits, [-1, max_text_length, class_count])

    return OcrRecCtcModel()


def export_linear_ctc_onnx(model: Any, path: Path, width: int, height: int, max_text_length: int, class_count: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    state = model.state_dict()
    weight = state["head.weight"].numpy().astype("float32")
    bias = state["head.bias"].numpy().astype("float32")
    shape = np.asarray([-1, max_text_length, class_count], dtype="int64")

    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["image"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "head.weight"], ["linear"]),
            helper.make_node("Add", ["linear", "head.bias"], ["linear_bias"]),
            helper.make_node("Reshape", ["linear_bias", "output_shape"], ["logits"]),
        ],
        "aitrain_paddleocr_rec_ctc",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 1, height, width])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", max_text_length, class_count])],
        [
            numpy_helper.from_array(weight, name="head.weight"),
            numpy_helper.from_array(bias, name="head.bias"),
            numpy_helper.from_array(shape, name="output_shape"),
        ],
    )
    model_proto = helper.make_model(
        graph,
        producer_name="AITrainStudio",
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    model_proto.ir_version = 8
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(path))


def run(request: dict[str, Any]) -> int:
    try:
        import paddle
    except Exception as exc:
        return fail("PaddlePaddle is not available. Install it with: python -m pip install paddlepaddle", "paddle_missing", {"exception": str(exc)})

    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    dataset_path = Path(str(request.get("datasetPath") or parameters.get("datasetPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or parameters.get("outputPath") or "aitrain-ocr-rec-output")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        chars = read_dictionary(dataset_path / "dict.txt")
        samples = read_labels(dataset_path / "rec_gt.txt")
    except Exception as exc:
        return fail(str(exc), "bad_dataset")

    epochs = max(1, int(parameters.get("epochs", 1)))
    width = max(16, int(parameters.get("imageWidth", 128)))
    height = max(8, int(parameters.get("imageHeight", 32)))
    max_text_length = max(1, int(parameters.get("maxTextLength", 16)))
    learning_rate = float(parameters.get("learningRate", 0.01))
    class_count = len(chars) + 1

    images = []
    labels = []
    label_lengths = []
    for relative_image, text in samples:
        image_path = dataset_path / relative_image
        images.append(image_tensor(image_path, width, height))
        encoded = [chars.index(ch) + 1 for ch in text if ch in chars][:max_text_length]
        labels.append(encoded + [0] * (max_text_length - len(encoded)))
        label_lengths.append(len(encoded))
    x_np = np.stack(images).astype("float32")
    y_np = np.asarray(labels, dtype="int32")
    input_lengths_np = np.full((len(samples),), max_text_length, dtype="int64")
    label_lengths_np = np.asarray(label_lengths, dtype="int64")

    paddle.seed(2026)
    model = create_model(paddle, width, height, max_text_length, class_count)
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    ctc_loss = paddle.nn.CTCLoss(blank=0, reduction="mean")

    emit("log", level="info", message=f"Starting PaddlePaddle CTC OCR Rec training: samples={len(samples)}, epochs={epochs}")
    final_loss = 0.0
    final_accuracy = 0.0
    final_edit_distance = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(paddle.to_tensor(x_np))
        log_probs = paddle.nn.functional.log_softmax(logits, axis=2).transpose([1, 0, 2])
        loss = ctc_loss(
            log_probs,
            paddle.to_tensor(y_np),
            paddle.to_tensor(input_lengths_np),
            paddle.to_tensor(label_lengths_np),
        )
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        model.eval()
        with paddle.no_grad():
            eval_logits = model(paddle.to_tensor(x_np))
            predicted = paddle.argmax(eval_logits, axis=2).numpy().astype("int64")
        correct = 0
        distances = 0
        for row, (_, truth) in zip(predicted.tolist(), samples):
            text = decode(row, chars)
            correct += int(text == truth)
            distances += edit_distance(text, truth)
        final_loss = float(loss.numpy())
        final_accuracy = correct / max(1, len(samples))
        final_edit_distance = distances / max(1, len(samples))
        emit("metric", name="loss", value=final_loss, epoch=epoch, step=epoch)
        emit("metric", name="accuracy", value=final_accuracy, epoch=epoch, step=epoch)
        emit("metric", name="editDistance", value=final_edit_distance, epoch=epoch, step=epoch)
        emit("progress", value=epoch / epochs, message=f"epoch {epoch}/{epochs}")

    checkpoint_path = output_path / "paddleocr_rec_ctc.pdparams"
    paddle.save(model.state_dict(), str(checkpoint_path))
    dict_path = output_path / "dict.txt"
    shutil.copyfile(dataset_path / "dict.txt", dict_path)
    onnx_prefix = output_path / "paddleocr_rec_ctc"
    onnx_path = output_path / "paddleocr_rec_ctc.onnx"
    try:
        model.eval()
        export_linear_ctc_onnx(model, onnx_path, width, height, max_text_length, class_count)
    except Exception as exc:
        return fail("Paddle OCR Rec ONNX export failed.", "onnx_export_failed", {"exception": str(exc)})
    if not onnx_path.exists():
        return fail("Paddle OCR Rec ONNX export completed without producing an ONNX file.", "onnx_missing")
    report_path = output_path / "paddleocr_rec_training_report.json"
    report = {
        "ok": True,
        "backend": BACKEND_ID,
        "framework": "PaddlePaddle",
        "note": "Small PaddlePaddle CTC Rec trainer for PaddleOCR-style data; not a full PP-OCR training config.",
        "checkpointPath": str(checkpoint_path),
        "onnxPath": str(onnx_path),
        "dictPath": str(dict_path),
        "modelFamily": "ocr_recognition",
        "imageWidth": width,
        "imageHeight": height,
        "maxTextLength": max_text_length,
        "classCount": class_count,
        "blankIndex": 0,
        "metrics": {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "editDistance": final_edit_distance,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("artifact", name="paddleocr_rec_ctc.pdparams", path=str(checkpoint_path), kind="checkpoint")
    emit("artifact", name="paddleocr_rec_ctc.onnx", path=str(onnx_path), kind="onnx")
    emit("artifact", name="dict.txt", path=str(dict_path), kind="dict")
    emit("artifact", name="paddleocr_rec_training_report.json", path=str(report_path), kind="report")
    emit("completed", checkpointPath=str(checkpoint_path), onnxPath=str(onnx_path), reportPath=str(report_path), metrics=report["metrics"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = read_request(args.request)
    except Exception as exc:
        return fail(f"failed to read trainer request: {exc}", "bad_request")
    return run(request)


if __name__ == "__main__":
    raise SystemExit(main())
