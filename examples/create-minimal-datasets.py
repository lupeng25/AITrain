#!/usr/bin/env python3
"""Generate tiny AITrain Studio smoke datasets without third-party packages."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytearray()
    x0, y0, x1, y1 = rect
    for y in range(height):
        pixels.append(0)
        for x in range(width):
            if x0 <= x < x1 and y0 <= y < y1:
                pixels.extend(color)
            else:
                pixels.extend((245, 245, 245))
    payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", payload) + _chunk(b"IDAT", zlib.compress(bytes(pixels), 9)) + _chunk(b"IEND", b"")
    path.write_bytes(data)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_yolo_detection(root: Path) -> None:
    write_png(root / "images/train/a.png", 64, 64, (18, 20, 46, 48), (36, 145, 255))
    write_png(root / "images/val/b.png", 64, 64, (16, 18, 44, 46), (36, 145, 255))
    write_text(root / "labels/train/a.txt", "0 0.500000 0.531250 0.437500 0.437500\n")
    write_text(root / "labels/val/b.txt", "0 0.468750 0.500000 0.437500 0.437500\n")
    write_text(
        root / "data.yaml",
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n",
    )


def make_yolo_segmentation(root: Path) -> None:
    write_png(root / "images/train/a.png", 64, 64, (20, 20, 44, 44), (15, 174, 102))
    write_png(root / "images/val/b.png", 64, 64, (18, 18, 46, 46), (15, 174, 102))
    polygon_a = "0 0.312500 0.312500 0.687500 0.312500 0.687500 0.687500 0.312500 0.687500\n"
    polygon_b = "0 0.281250 0.281250 0.718750 0.281250 0.718750 0.718750 0.281250 0.718750\n"
    write_text(root / "labels/train/a.txt", polygon_a)
    write_text(root / "labels/val/b.txt", polygon_b)
    write_text(
        root / "data.yaml",
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [part]\n",
    )


def make_ocr_rec(root: Path) -> None:
    write_png(root / "images/a.png", 96, 32, (8, 8, 42, 24), (40, 40, 40))
    write_png(root / "images/b.png", 96, 32, (52, 8, 88, 24), (40, 40, 40))
    write_text(root / "dict.txt", "a\nb\n1\n2\n")
    write_text(root / "rec_gt.txt", "images/a.png\tab12\nimages/b.png\tba\n")


def make_ocr_det(root: Path) -> None:
    write_png(root / "images/a.png", 96, 48, (8, 12, 42, 30), (40, 40, 40))
    write_png(root / "images/b.png", 96, 48, (52, 12, 88, 30), (40, 40, 40))
    label_a = [{"transcription": "ab12", "points": [[8, 12], [42, 12], [42, 30], [8, 30]]}]
    label_b = [{"transcription": "###", "points": [[52, 12], [88, 12], [88, 30], [52, 30]]}]
    write_text(
        root / "det_gt.txt",
        f"images/a.png\t{json.dumps(label_a, separators=(',', ':'))}\n"
        f"images/b.png\t{json.dumps(label_b, separators=(',', ':'))}\n",
    )


def make_requests(root: Path) -> None:
    requests = {
        "yolo_detect_request.json": {
            "protocolVersion": 1,
            "taskId": "example-yolo-detect",
            "taskType": "detection",
            "datasetPath": str(root / "yolo_detect"),
            "outputPath": str(root / "runs/yolo_detect"),
            "backend": "ultralytics_yolo_detect",
            "parameters": {"model": "yolov8n.yaml", "epochs": 1, "batchSize": 1, "imageSize": 64, "device": "cpu", "workers": 0},
        },
        "yolo_segment_request.json": {
            "protocolVersion": 1,
            "taskId": "example-yolo-segment",
            "taskType": "segmentation",
            "datasetPath": str(root / "yolo_segment"),
            "outputPath": str(root / "runs/yolo_segment"),
            "backend": "ultralytics_yolo_segment",
            "parameters": {"model": "yolov8n-seg.yaml", "epochs": 1, "batchSize": 1, "imageSize": 64, "device": "cpu", "workers": 0},
        },
        "paddleocr_rec_request.json": {
            "protocolVersion": 1,
            "taskId": "example-paddleocr-rec",
            "taskType": "ocr_recognition",
            "datasetPath": str(root / "paddleocr_rec"),
            "outputPath": str(root / "runs/paddleocr_rec"),
            "backend": "paddleocr_rec",
            "parameters": {"epochs": 1, "batchSize": 2, "imageWidth": 96, "imageHeight": 32, "maxTextLength": 8, "learningRate": 0.01},
        },
        "paddleocr_det_official_request.json": {
            "protocolVersion": 1,
            "taskId": "example-paddleocr-det-official",
            "taskType": "ocr_detection",
            "datasetPath": str(root / "paddleocr_det"),
            "outputPath": str(root / "runs/paddleocr_det_official"),
            "backend": "paddleocr_det_official",
            "parameters": {"trainingBackend": "paddleocr_det_official", "prepareOnly": True, "epochs": 1, "batchSize": 1, "imageSize": 64, "useGpu": False},
        },
        "paddleocr_system_official_request.json": {
            "protocolVersion": 1,
            "taskId": "example-paddleocr-system-official",
            "taskType": "ocr",
            "datasetPath": str(root / "paddleocr_det/images/a.png"),
            "outputPath": str(root / "runs/paddleocr_system_official"),
            "backend": "paddleocr_system_official",
            "parameters": {
                "trainingBackend": "paddleocr_system_official",
                "prepareOnly": True,
                "detModelDir": str(root / "runs/paddleocr_det_official/official_inference"),
                "recModelDir": str(root / "runs/paddleocr_rec_official/official_inference"),
                "dictionaryFile": str(root / "paddleocr_rec/dict.txt"),
                "inferenceImage": str(root / "paddleocr_det/images/a.png"),
                "useGpu": False,
            },
        },
    }
    for name, request in requests.items():
        write_text(root / name, json.dumps(request, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="minimal-datasets", help="Output directory")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    make_yolo_detection(root / "yolo_detect")
    make_yolo_segmentation(root / "yolo_segment")
    make_ocr_det(root / "paddleocr_det")
    make_ocr_rec(root / "paddleocr_rec")
    make_requests(root)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
