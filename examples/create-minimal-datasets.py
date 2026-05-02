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


def yolo_box_label(class_id: int, rect: tuple[int, int, int, int], width: int, height: int) -> str:
    x0, y0, x1, y1 = rect
    x_center = ((x0 + x1) / 2.0) / width
    y_center = ((y0 + y1) / 2.0) / height
    box_width = (x1 - x0) / width
    box_height = (y1 - y0) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"


def make_yolo_detection_cpu_smoke(root: Path) -> None:
    width = 128
    height = 128
    class_colors = [(36, 145, 255), (220, 80, 80)]
    for split, count, offset in (("train", 32, 0), ("val", 8, 100)):
        for index in range(count):
            class_id = index % 2
            box_width = 24 + ((index + offset) % 4) * 6
            box_height = 22 + ((index * 3 + offset) % 5) * 5
            x0 = 8 + ((index * 11 + offset) % (width - box_width - 16))
            y0 = 10 + ((index * 7 + offset) % (height - box_height - 18))
            rect = (x0, y0, x0 + box_width, y0 + box_height)
            stem = f"{split}_{index:02d}"
            write_png(root / f"images/{split}/{stem}.png", width, height, rect, class_colors[class_id])
            write_text(root / f"labels/{split}/{stem}.txt", yolo_box_label(class_id, rect, width, height))
    write_text(
        root / "data.yaml",
        "path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames: [item, marker]\n",
    )


def make_yolo_segmentation_cpu_smoke(root: Path) -> None:
    width = 128
    height = 128
    class_colors = [(15, 174, 102), (180, 92, 220)]
    for split, count, offset in (("train", 24, 0), ("val", 8, 80)):
        for index in range(count):
            class_id = index % 2
            box_width = 28 + ((index + offset) % 5) * 5
            box_height = 26 + ((index * 2 + offset) % 4) * 7
            x0 = 9 + ((index * 13 + offset) % (width - box_width - 18))
            y0 = 11 + ((index * 9 + offset) % (height - box_height - 20))
            rect = (x0, y0, x0 + box_width, y0 + box_height)
            stem = f"{split}_{index:02d}"
            write_png(root / f"images/{split}/{stem}.png", width, height, rect, class_colors[class_id])
            x1, y1 = x0 + box_width, y0 + box_height
            polygon = (
                f"{class_id} "
                f"{x0 / width:.6f} {y0 / height:.6f} "
                f"{x1 / width:.6f} {y0 / height:.6f} "
                f"{x1 / width:.6f} {y1 / height:.6f} "
                f"{x0 / width:.6f} {y1 / height:.6f}\n"
            )
            write_text(root / f"labels/{split}/{stem}.txt", polygon)
    write_text(
        root / "data.yaml",
        "path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames: [part, scratch]\n",
    )


def make_ocr_rec_cpu_smoke(root: Path) -> None:
    chars = "ab12cd34"
    labels = []
    for index in range(96):
        x0 = 6 + (index * 5) % 52
        x1 = min(122, x0 + 22 + (index % 6) * 7)
        y0 = 5 + (index * 3) % 12
        y1 = min(30, y0 + 13 + (index % 3) * 3)
        stem = f"ocr_{index:03d}"
        text_length = 2 + (index % 5)
        text = "".join(chars[(index + step * 3) % len(chars)] for step in range(text_length))
        write_png(root / f"images/{stem}.png", 128, 32, (x0, y0, x1, y1), (35 + (index % 3) * 35, 35, 35))
        labels.append(f"images/{stem}.png\t{text}")
    write_text(root / "dict.txt", "\n".join(chars) + "\n")
    write_text(root / "rec_gt.txt", "\n".join(labels) + "\n")


def make_requests(root: Path, profile: str = "minimal") -> None:
    if profile == "cpu-smoke":
        detect_parameters = {
            "model": "yolov8n.yaml",
            "epochs": 3,
            "batchSize": 2,
            "imageSize": 128,
            "device": "cpu",
            "workers": 0,
            "runName": "cpu-yolo-detect",
            "compactEvents": True,
        }
        segment_parameters = {
            "model": "yolov8n-seg.yaml",
            "epochs": 3,
            "batchSize": 2,
            "imageSize": 128,
            "device": "cpu",
            "workers": 0,
            "runName": "cpu-yolo-segment",
            "compactEvents": True,
        }
        ocr_parameters = {
            "epochs": 8,
            "batchSize": 8,
            "imageWidth": 128,
            "imageHeight": 32,
            "maxTextLength": 10,
            "learningRate": 0.01,
        }
    else:
        detect_parameters = {"model": "yolov8n.yaml", "epochs": 1, "batchSize": 1, "imageSize": 64, "device": "cpu", "workers": 0}
        segment_parameters = {"model": "yolov8n-seg.yaml", "epochs": 1, "batchSize": 1, "imageSize": 64, "device": "cpu", "workers": 0}
        ocr_parameters = {"epochs": 1, "batchSize": 2, "imageWidth": 96, "imageHeight": 32, "maxTextLength": 8, "learningRate": 0.01}

    requests = {
        "yolo_detect_request.json": {
            "protocolVersion": 1,
            "taskId": "example-yolo-detect",
            "taskType": "detection",
            "datasetPath": str(root / "yolo_detect"),
            "outputPath": str(root / "runs/yolo_detect"),
            "backend": "ultralytics_yolo_detect",
            "parameters": detect_parameters,
        },
        "yolo_segment_request.json": {
            "protocolVersion": 1,
            "taskId": "example-yolo-segment",
            "taskType": "segmentation",
            "datasetPath": str(root / "yolo_segment"),
            "outputPath": str(root / "runs/yolo_segment"),
            "backend": "ultralytics_yolo_segment",
            "parameters": segment_parameters,
        },
        "paddleocr_rec_request.json": {
            "protocolVersion": 1,
            "taskId": "example-paddleocr-rec",
            "taskType": "ocr_recognition",
            "datasetPath": str(root / "paddleocr_rec"),
            "outputPath": str(root / "runs/paddleocr_rec"),
            "backend": "paddleocr_rec",
            "parameters": ocr_parameters,
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
    parser.add_argument("--profile", choices=("minimal", "cpu-smoke"), default="minimal", help="Dataset size/profile")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.profile == "cpu-smoke":
        make_yolo_detection_cpu_smoke(root / "yolo_detect")
        make_yolo_segmentation_cpu_smoke(root / "yolo_segment")
        make_ocr_rec_cpu_smoke(root / "paddleocr_rec")
    else:
        make_yolo_detection(root / "yolo_detect")
        make_yolo_segmentation(root / "yolo_segment")
        make_ocr_rec(root / "paddleocr_rec")
    make_ocr_det(root / "paddleocr_det")
    make_requests(root, args.profile)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
