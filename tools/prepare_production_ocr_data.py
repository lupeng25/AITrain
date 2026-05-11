import argparse
import hashlib
import json
import os
import random
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

from PIL import Image


TOTAL_TEXT_URL = "https://paddleocr.bj.bcebos.com/dataset/total_text.tar"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp.replace(path)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    marker = destination / "total_text" / "train" / "train.txt"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(root)):
                continue
            try:
                tar.extract(member, destination)
            except OSError:
                # The upstream archive contains redundant entries that Windows
                # cannot materialize. The image and label files are still usable.
                continue


def read_label_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            rel_image, payload = line.split("\t", 1)
            try:
                boxes = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(boxes, list):
                rows.append((rel_image, boxes))
    return rows


def normalize_points(points):
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        normalized.append([float(point[0]), float(point[1])])
    return normalized


def split_train_rows(rows, val_ratio: float, seed: int):
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def write_det_split(rows, source_root: Path, output_root: Path, split: str):
    label_path = output_root / "det_dataset" / f"det_gt_{split}.txt"
    image_dir = output_root / "det_dataset" / "images" / split
    label_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    box_count = 0
    with label_path.open("w", encoding="utf-8", newline="\n") as handle:
        for rel_image, boxes in rows:
            source_image = source_root / rel_image
            if not source_image.exists():
                continue
            target_name = f"totaltext_{Path(rel_image).name}"
            target_image = image_dir / target_name
            copy_image(source_image, target_image)
            cleaned = []
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                points = normalize_points(box.get("points", []))
                if len(points) < 4:
                    continue
                cleaned.append({
                    "transcription": str(box.get("transcription", "")),
                    "points": points,
                })
            if not cleaned:
                continue
            handle.write(f"images/{split}/{target_name}\t{json.dumps(cleaned, ensure_ascii=False, separators=(',', ':'))}\n")
            written += 1
            box_count += len(cleaned)
    return {"images": written, "boxes": box_count, "labelFile": str(label_path)}


def crop_rec_samples(rows, source_root: Path, output_root: Path, split: str, start_index: int, max_samples: int):
    label_path = output_root / "rec_dataset" / f"rec_gt_{split}.txt"
    image_dir = output_root / "rec_dataset" / "images" / split
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_labels = set()
    with label_path.open("w", encoding="utf-8", newline="\n") as handle:
        for rel_image, boxes in rows:
            if count >= max_samples:
                break
            source_image = source_root / rel_image
            if not source_image.exists():
                continue
            try:
                image = Image.open(source_image).convert("RGB")
            except Exception:
                continue
            width, height = image.size
            for box_index, box in enumerate(boxes):
                if count >= max_samples:
                    break
                text = str(box.get("transcription", "")).strip()
                if not text or text in {"###", "*"}:
                    continue
                points = normalize_points(box.get("points", []))
                if len(points) < 4:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                left = max(0, int(min(xs)) - 2)
                top = max(0, int(min(ys)) - 2)
                right = min(width, int(max(xs)) + 2)
                bottom = min(height, int(max(ys)) + 2)
                if right - left < 4 or bottom - top < 4:
                    continue
                crop = image.crop((left, top, right, bottom))
                target_name = f"totaltext_{start_index + count:06d}_{Path(rel_image).stem}_{box_index}.jpg"
                target_rel = f"images/{split}/{target_name}"
                crop.save(image_dir / target_name, quality=92)
                handle.write(f"{target_rel}\t{text}\n")
                seen_labels.update(text)
                count += 1
    return {
        "samples": count,
        "labelFile": str(label_path),
        "characters": "".join(sorted(seen_labels)),
        "nextIndex": start_index + count,
    }


def combine_rec_labels(output_root: Path):
    rec_root = output_root / "rec_dataset"
    combined = rec_root / "rec_gt.txt"
    total = 0
    with combined.open("w", encoding="utf-8", newline="\n") as output:
        for name in ["rec_gt_train.txt", "rec_gt_val.txt", "rec_gt_test.txt"]:
            path = rec_root / name
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        output.write(line)
                        total += 1
    return total


def write_dict(output_root: Path, characters: str):
    dict_path = output_root / "rec_dataset" / "dict.txt"
    unique = sorted({char for char in characters if not char.isspace()})
    with dict_path.open("w", encoding="utf-8", newline="\n") as handle:
        for char in unique:
            handle.write(char + "\n")
    return {"path": str(dict_path), "characters": len(unique)}


def copy_system_images(rows, source_root: Path, output_root: Path, minimum: int):
    system_dir = output_root / "system_images"
    system_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel_image, _boxes in rows:
        if copied >= minimum:
            break
        source_image = source_root / rel_image
        if not source_image.exists():
            continue
        target = system_dir / f"totaltext_{Path(rel_image).name}"
        copy_image(source_image, target)
        copied += 1
    return {"images": copied, "path": str(system_dir)}


def main():
    parser = argparse.ArgumentParser(description="Prepare public Total-Text data for AITrain production OCR acceptance gates.")
    parser.add_argument("--work-dir", default=".deps/production-ocr-data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-rec-train", type=int, default=2400)
    parser.add_argument("--max-rec-val", type=int, default=300)
    parser.add_argument("--max-rec-test", type=int, default=300)
    parser.add_argument("--minimum-system-images", type=int, default=100)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    sources = work_dir / "sources"
    extracted = sources / "extracted"
    archive = sources / "total_text.tar"

    if not args.skip_download:
        download_file(TOTAL_TEXT_URL, archive)
    if not archive.exists():
        raise FileNotFoundError(f"Total-Text archive not found: {archive}")
    safe_extract_tar(archive, extracted)

    dataset_root = extracted / "total_text"
    train_root = dataset_root / "train"
    test_root = dataset_root / "test"
    train_rows = read_label_file(train_root / "train.txt")
    test_rows = read_label_file(test_root / "test.txt")
    train_rows, val_rows = split_train_rows(train_rows, args.val_ratio, args.seed)

    output_root = work_dir
    det_stats = {
        "train": write_det_split(train_rows, train_root, output_root, "train"),
        "val": write_det_split(val_rows, train_root, output_root, "val"),
        "test": write_det_split(test_rows, test_root, output_root, "test"),
    }

    next_index = 0
    rec_train = crop_rec_samples(train_rows, train_root, output_root, "train", next_index, args.max_rec_train)
    next_index = rec_train["nextIndex"]
    rec_val = crop_rec_samples(val_rows, train_root, output_root, "val", next_index, args.max_rec_val)
    next_index = rec_val["nextIndex"]
    rec_test = crop_rec_samples(test_rows, test_root, output_root, "test", next_index, args.max_rec_test)
    rec_total = combine_rec_labels(output_root)
    dict_stats = write_dict(output_root, rec_train["characters"] + rec_val["characters"] + rec_test["characters"])
    system_stats = copy_system_images(test_rows + val_rows + train_rows, test_root, output_root, args.minimum_system_images)

    reports_dir = output_root / "reports"
    manifests_dir = output_root / "manifests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ok": True,
        "status": "prepared",
        "preparedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "name": "Total-Text",
            "url": TOTAL_TEXT_URL,
            "archive": str(archive),
            "sha256": sha256_file(archive),
            "note": "Public dataset link listed by PaddleOCR OCR datasets documentation.",
        },
        "thresholdIntent": {
            "minimumDetImages": 100,
            "minimumRecSamples": 1000,
            "minimumSystemImages": args.minimum_system_images,
        },
        "det": det_stats,
        "rec": {
            "train": {k: v for k, v in rec_train.items() if k != "characters"},
            "val": {k: v for k, v in rec_val.items() if k != "characters"},
            "test": {k: v for k, v in rec_test.items() if k != "characters"},
            "combinedSamples": rec_total,
            "dictionary": dict_stats,
        },
        "system": system_stats,
        "reportsDir": str(reports_dir),
        "limitations": [
            "This prepares public representative OCR data only; it is not customer-domain production evidence.",
            "Recognition samples are cropped from Total-Text detection annotations for gate preparation.",
            "Official PaddleOCR train/export/system reports are still required before production OCR acceptance can pass.",
        ],
    }
    manifest_path = manifests_dir / "production_ocr_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
