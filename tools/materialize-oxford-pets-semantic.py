#!/usr/bin/env python3
"""Materialize Oxford-IIIT Pet trimaps as AITrain semantic mask data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SOURCE_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
IMAGE_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATION_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
HF_IMAGE_MIRROR_PAGE = "https://huggingface.co/datasets/enterprise-explorers/oxford-pets"
HF_IMAGE_MIRROR_URL = (
    "https://huggingface.co/datasets/enterprise-explorers/oxford-pets/resolve/main/"
    "data/train-00000-of-00001-ecc2afb43dedd5e0.parquet"
)
LICENSE = "CC BY-SA 4.0"
DATASET_NOTE = (
    "Oxford-IIIT Pet is a public quality-comparison dataset with pet trimap masks. "
    "It is not industrial defect/customer-domain production evidence."
)
CLASS_NAMES = ["background", "pet"]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    trimap_path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def expected_content_length(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=120) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_file(url: str, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_content_length(url)
    if destination.exists() and destination.stat().st_size > 0 and (
        expected is None or destination.stat().st_size == expected
    ):
        return
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists() and destination.stat().st_size > 0:
        if temp_path.exists():
            temp_path.unlink()
        destination.replace(temp_path)
    if expected is not None and temp_path.exists() and temp_path.stat().st_size > expected:
        temp_path.unlink()

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            start = temp_path.stat().st_size if temp_path.exists() else 0
            request = urllib.request.Request(url)
            if start > 0:
                request.add_header("Range", f"bytes={start}-")
            with urllib.request.urlopen(request, timeout=300) as response:
                if start > 0 and getattr(response, "status", 200) != 206:
                    start = 0
                    if temp_path.exists():
                        temp_path.unlink()
                with temp_path.open("ab" if start > 0 else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected is None:
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    temp_path.replace(destination)
                    return
            elif temp_path.exists() and temp_path.stat().st_size == expected:
                temp_path.replace(destination)
                return
            elif expected is not None and temp_path.exists() and temp_path.stat().st_size > expected:
                temp_path.unlink()
        except Exception as exc:
            last_error = exc
        if attempt < attempts - 1:
            time.sleep(min(30, 5 * (attempt + 1)))

    actual = temp_path.stat().st_size if temp_path.exists() else 0
    detail = f"incomplete download for {url}: {actual}/{expected if expected is not None else 'unknown'} bytes"
    if last_error is not None:
        detail += f"; last error: {last_error}"
    raise OSError(detail)


def safe_extract(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_target = (destination / member.name).resolve()
            if destination_root != member_target and destination_root not in member_target.parents:
                raise ValueError(f"Refusing to extract path outside destination: {member.name}")
        archive.extractall(destination)


def materialize_hf_image_mirror(
    download_dir: Path,
    materialized_dir: Path,
    annotation_archive: Path,
    allow_download: bool,
    reason: str,
) -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Official Oxford Pets images archive was unavailable and the Hugging Face image "
            "mirror fallback requires pyarrow in the selected Python environment."
        ) from exc

    if not annotation_archive.exists():
        raise FileNotFoundError(
            "Official Oxford Pets annotations archive is required for trimaps before using the image mirror: "
            f"{annotation_archive}"
        )

    parquet_path = download_dir / "hf-enterprise" / "train.parquet"
    if not parquet_path.exists():
        if not allow_download:
            raise FileNotFoundError(
                "Hugging Face Oxford Pets image mirror parquet is missing and --skip-download was set: "
                f"{parquet_path}"
            )
        download_file(HF_IMAGE_MIRROR_URL, parquet_path)

    if materialized_dir.exists():
        shutil.rmtree(materialized_dir)
    safe_extract(annotation_archive, materialized_dir)

    images_dir = materialized_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(parquet_path)
    rows = 0
    images_written = 0
    missing_bytes = 0
    for batch in parquet_file.iter_batches(columns=["path", "image"], batch_size=256):
        path_index = batch.schema.get_field_index("path")
        image_index = batch.schema.get_field_index("image")
        paths = batch.column(path_index).to_pylist()
        images = batch.column(image_index).to_pylist()
        for source_path, image_value in zip(paths, images):
            rows += 1
            image_bytes = image_value.get("bytes") if isinstance(image_value, dict) else None
            image_path = image_value.get("path") if isinstance(image_value, dict) else ""
            filename = Path(str(source_path or image_path)).name
            if not filename:
                filename = f"mirror_image_{rows:05d}.jpg"
            if not image_bytes:
                missing_bytes += 1
                continue
            (images_dir / filename).write_bytes(image_bytes)
            images_written += 1

    mirror_manifest = {
        "ok": True,
        "createdAt": now_iso(),
        "source": "Hugging Face dataset enterprise-explorers/oxford-pets image mirror",
        "sourcePage": HF_IMAGE_MIRROR_PAGE,
        "sourceParquet": str(parquet_path),
        "officialAnnotationsArchive": str(annotation_archive),
        "rows": rows,
        "imagesWritten": images_written,
        "missingImageBytes": missing_bytes,
        "materializedDir": str(materialized_dir),
        "reason": reason,
        "note": (
            "Images mirror was used because Oxford official images.tar.gz was unavailable "
            "or corrupt in this environment; annotations/trimaps are from official annotations.tar.gz."
        ),
    }
    write_json(materialized_dir / "mirror_manifest.json", mirror_manifest)
    if not materialized_layout_ready(materialized_dir):
        raise FileNotFoundError(f"Oxford Pets mirror materialization did not produce the expected layout: {materialized_dir}")


def materialized_layout_ready(materialized_dir: Path) -> bool:
    return (
        (materialized_dir / "images").is_dir()
        and (materialized_dir / "annotations" / "trimaps").is_dir()
        and (materialized_dir / "annotations" / "trainval.txt").is_file()
        and (materialized_dir / "annotations" / "test.txt").is_file()
    )


def ensure_materialized(download_dir: Path, materialized_dir: Path, skip_download: bool) -> None:
    if materialized_layout_ready(materialized_dir):
        return

    image_archive = download_dir / "images.tar.gz"
    annotation_archive = download_dir / "annotations.tar.gz"
    image_download_error: Exception | None = None
    if not skip_download:
        try:
            download_file(IMAGE_URL, image_archive)
        except Exception as exc:
            image_download_error = exc
        download_file(ANNOTATION_URL, annotation_archive)
    if not annotation_archive.exists():
        raise FileNotFoundError(
            "Oxford Pets official annotations archive is missing and is required for trimaps: "
            f"{annotation_archive}"
        )
    if not image_archive.exists():
        try:
            materialize_hf_image_mirror(
                download_dir=download_dir,
                materialized_dir=materialized_dir,
                annotation_archive=annotation_archive,
                allow_download=not skip_download,
                reason=(
                    str(image_download_error)
                    if image_download_error is not None
                    else "Official Oxford Pets images archive was not present."
                ),
            )
            return
        except Exception as mirror_exc:
            if image_download_error is not None:
                raise RuntimeError(
                    "Official Oxford Pets images download failed and Hugging Face image mirror fallback failed: "
                    f"{mirror_exc}"
                ) from mirror_exc
            raise

    missing = [str(path) for path in (image_archive, annotation_archive) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Oxford Pets archives are missing and --skip-download was set: " + ", ".join(missing)
        )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            safe_extract(image_archive, materialized_dir)
            safe_extract(annotation_archive, materialized_dir)
            if materialized_layout_ready(materialized_dir):
                return
            raise FileNotFoundError(f"Oxford Pets layout was not found after extraction: {materialized_dir}")
        except (EOFError, OSError, tarfile.TarError) as exc:
            last_error = exc
            if skip_download or attempt >= 1:
                break
            for archive_path, url in ((image_archive, IMAGE_URL), (annotation_archive, ANNOTATION_URL)):
                expected = expected_content_length(url)
                if archive_path.exists() and (expected is None or archive_path.stat().st_size >= expected):
                    archive_path.unlink()
            if materialized_dir.exists():
                shutil.rmtree(materialized_dir)
            try:
                download_file(IMAGE_URL, image_archive)
            except Exception as exc:
                last_error = exc
            download_file(ANNOTATION_URL, annotation_archive)
    if last_error is not None:
        try:
            materialize_hf_image_mirror(
                download_dir=download_dir,
                materialized_dir=materialized_dir,
                annotation_archive=annotation_archive,
                allow_download=not skip_download,
                reason=str(last_error),
            )
            return
        except Exception as mirror_exc:
            raise RuntimeError(
                "Official Oxford Pets archive extraction failed and Hugging Face image mirror fallback failed: "
                f"{mirror_exc}"
            ) from mirror_exc
    raise FileNotFoundError(f"Oxford Pets layout was not found after extraction: {materialized_dir}")


def parse_split_file(path: Path) -> list[str]:
    sample_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sample_ids.append(line.split()[0])
    if not sample_ids:
        raise ValueError(f"No sample ids found in split file: {path}")
    return sample_ids


def find_image(images_dir: Path, sample_id: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for Oxford Pets sample id '{sample_id}'")


def make_sample(materialized_dir: Path, sample_id: str) -> Sample:
    images_dir = materialized_dir / "images"
    trimap_path = materialized_dir / "annotations" / "trimaps" / f"{sample_id}.png"
    if not trimap_path.exists():
        raise FileNotFoundError(f"Trimap not found for Oxford Pets sample id '{sample_id}': {trimap_path}")
    return Sample(sample_id=sample_id, image_path=find_image(images_dir, sample_id), trimap_path=trimap_path)


def derive_splits(materialized_dir: Path, seed: int) -> dict[str, list[Sample]]:
    trainval_ids = parse_split_file(materialized_dir / "annotations" / "trainval.txt")
    test_ids = parse_split_file(materialized_dir / "annotations" / "test.txt")
    trainval_ids = sorted(dict.fromkeys(trainval_ids))
    test_ids = sorted(dict.fromkeys(test_ids))
    rng = random.Random(seed)
    rng.shuffle(trainval_ids)
    if len(trainval_ids) <= 1:
        raise ValueError("Oxford Pets trainval split must contain at least two samples to derive train/val.")
    val_count = max(1, int(round(len(trainval_ids) * 0.2)))
    val_count = min(val_count, len(trainval_ids) - 1)
    val_ids = sorted(trainval_ids[:val_count])
    train_ids = sorted(trainval_ids[val_count:])
    return {
        "train": [make_sample(materialized_dir, sample_id) for sample_id in train_ids],
        "val": [make_sample(materialized_dir, sample_id) for sample_id in val_ids],
        "test": [make_sample(materialized_dir, sample_id) for sample_id in test_ids],
    }


def apply_max_samples(splits: dict[str, list[Sample]], max_samples_per_split: int) -> dict[str, list[Sample]]:
    if max_samples_per_split <= 0:
        return splits
    return {split: samples[:max_samples_per_split] for split, samples in splits.items()}


def convert_trimap(trimap_path: Path, mask_path: Path) -> dict[str, int]:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    raw = np.asarray(Image.open(trimap_path).convert("L"), dtype=np.uint8)
    converted = np.full(raw.shape, 255, dtype=np.uint8)
    converted[raw == 1] = 1
    converted[raw == 2] = 0
    converted[raw == 3] = 255
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(converted, mode="L").save(mask_path)
    return {
        "background": int((converted == 0).sum()),
        "pet": int((converted == 1).sum()),
        "ignore": int((converted == 255).sum()),
    }


def clear_output(output_dir: Path) -> None:
    for child in ("images", "masks", "classes.txt", "dataset_manifest.json", "split_manifest.json", "conversion_log.txt"):
        path = output_dir / child
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def materialize_dataset(
    work_dir: Path,
    output_dir: Path,
    download_dir: Path,
    materialized_dir: Path,
    skip_download: bool,
    max_samples_per_split: int,
    seed: int,
) -> dict[str, object]:
    ensure_materialized(download_dir, materialized_dir, skip_download)
    mirror_manifest_path = materialized_dir / "mirror_manifest.json"
    image_source = (
        "huggingface_image_mirror_with_official_annotations"
        if mirror_manifest_path.is_file()
        else "official_archive"
    )
    splits = apply_max_samples(derive_splits(materialized_dir, seed), max_samples_per_split)
    clear_output(output_dir)
    write_text(output_dir / "classes.txt", "background\npet\n")

    split_manifest: dict[str, object] = {
        "seed": seed,
        "sourceTrainValSplit": "Oxford Pets annotations/trainval.txt",
        "sourceTestSplit": "Oxford Pets annotations/test.txt",
        "trainValDerivation": "trainval shuffled with seed and split 80/20 into train/val",
        "maxSamplesPerSplit": max_samples_per_split,
        "splits": {},
    }
    pixel_totals = {
        "background": 0,
        "pet": 0,
        "ignore": 0,
    }
    split_stats: dict[str, dict[str, int]] = {}
    conversion_lines = [
        f"createdAt={now_iso()}",
        f"sourcePage={SOURCE_PAGE}",
        f"license={LICENSE}",
        f"imageSource={image_source}",
        "trimapMapping=1->pet, 2->background, 3->ignoreIndex(255)",
    ]

    for split, samples in splits.items():
        image_dir = output_dir / "images" / split
        mask_dir = output_dir / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        split_pixel_counts = {"background": 0, "pet": 0, "ignore": 0}
        split_ids: list[str] = []
        for sample in samples:
            target_image = image_dir / sample.image_path.name
            shutil.copy2(sample.image_path, target_image)
            target_mask = mask_dir / f"{sample.sample_id}.png"
            counts = convert_trimap(sample.trimap_path, target_mask)
            for key, value in counts.items():
                split_pixel_counts[key] += value
                pixel_totals[key] += value
            split_ids.append(sample.sample_id)
        split_stats[split] = {
            "sampleCount": len(samples),
            "backgroundPixels": split_pixel_counts["background"],
            "petPixels": split_pixel_counts["pet"],
            "ignorePixels": split_pixel_counts["ignore"],
        }
        split_manifest["splits"][split] = {
            "sampleCount": len(samples),
            "sampleIds": split_ids,
            "pixelCounts": split_pixel_counts,
        }
        conversion_lines.append(f"{split}: samples={len(samples)} ids={','.join(split_ids[:10])}")

    manifest = {
        "ok": True,
        "kind": "oxford_pets_semantic_dataset_manifest",
        "createdAt": now_iso(),
        "datasetPath": str(output_dir),
        "workDir": str(work_dir),
        "sourcePage": SOURCE_PAGE,
        "downloadUrls": {
            "images": IMAGE_URL,
            "annotations": ANNOTATION_URL,
            "imageMirror": HF_IMAGE_MIRROR_URL,
        },
        "imageSource": image_source,
        "imageMirrorPage": HF_IMAGE_MIRROR_PAGE,
        "imageMirrorManifestPath": str(mirror_manifest_path) if mirror_manifest_path.is_file() else "",
        "license": LICENSE,
        "sourceDescription": "Oxford-IIIT Pet public dataset: 37 pet categories with trimap segmentation annotations.",
        "note": DATASET_NOTE,
        "classNames": CLASS_NAMES,
        "ignoreIndex": 255,
        "trimapMapping": {
            "1": "pet",
            "2": "background",
            "3": "ignoreIndex",
        },
        "seed": seed,
        "maxSamplesPerSplit": max_samples_per_split,
        "downloadDir": str(download_dir),
        "materializedDir": str(materialized_dir),
        "splitStats": split_stats,
        "totalSamples": sum(stats["sampleCount"] for stats in split_stats.values()),
        "pixelTotals": pixel_totals,
        "splitManifestPath": str(output_dir / "split_manifest.json"),
        "conversionLogPath": str(output_dir / "conversion_log.txt"),
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_text(output_dir / "conversion_log.txt", "\n".join(conversion_lines) + "\n")
    return manifest


def create_fake_source(root: Path) -> Path:
    from PIL import Image  # type: ignore

    materialized = root / "materialized"
    images_dir = materialized / "images"
    trimaps_dir = materialized / "annotations" / "trimaps"
    images_dir.mkdir(parents=True)
    trimaps_dir.mkdir(parents=True)
    sample_ids = [f"Pet_{index:02d}" for index in range(1, 8)]
    for index, sample_id in enumerate(sample_ids):
        color = (40 + index * 20, 90 + index * 10, 140 + index * 5)
        Image.new("RGB", (8, 8), color).save(images_dir / f"{sample_id}.jpg")
        trimap = Image.new("L", (8, 8), 2)
        pixels = trimap.load()
        for y in range(2, 6):
            for x in range(2, 6):
                pixels[x, y] = 1
        for x in range(1, 7):
            pixels[x, 1] = 3
            pixels[x, 6] = 3
        trimap.save(trimaps_dir / f"{sample_id}.png")
    write_text(materialized / "annotations" / "trainval.txt", "\n".join(sample_ids[:5]) + "\n")
    write_text(materialized / "annotations" / "test.txt", "\n".join(sample_ids[5:]) + "\n")
    return materialized


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="aitrain-oxford-pets-selftest-") as temp_name:
        temp_root = Path(temp_name)
        materialized = create_fake_source(temp_root)
        manifest = materialize_dataset(
            work_dir=temp_root / "work",
            output_dir=temp_root / "work" / "semantic_mask_oxford_pets",
            download_dir=temp_root / "downloads",
            materialized_dir=materialized,
            skip_download=True,
            max_samples_per_split=0,
            seed=42,
        )
        output_dir = Path(str(manifest["datasetPath"]))
        classes = (output_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
        if classes != CLASS_NAMES:
            raise AssertionError(f"Unexpected classes.txt: {classes}")
        split_manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
        split_counts = {key: value["sampleCount"] for key, value in split_manifest["splits"].items()}
        if split_counts != {"train": 4, "val": 1, "test": 2}:
            raise AssertionError(f"Unexpected split counts: {split_counts}")
        first_train_id = split_manifest["splits"]["train"]["sampleIds"][0]
        mask_path = output_dir / "masks" / "train" / f"{first_train_id}.png"
        if not mask_path.exists():
            raise AssertionError(f"Missing converted mask: {mask_path}")
        print(json.dumps({"ok": True, "status": "passed", "datasetPath": str(output_dir)}, ensure_ascii=False))
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=r".deps\smp-quality\oxford-pets")
    parser.add_argument("--output", default="")
    parser.add_argument("--download-dir", default=r".deps\datasets\downloads\oxford-pets")
    parser.add_argument("--materialized-dir", default=r".deps\datasets\materialized\oxford-pets")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    try:
        work_dir = resolve_path(args.work_dir)
        output_dir = resolve_path(args.output) if args.output else work_dir / "semantic_mask_oxford_pets"
        manifest = materialize_dataset(
            work_dir=work_dir,
            output_dir=output_dir,
            download_dir=resolve_path(args.download_dir),
            materialized_dir=resolve_path(args.materialized_dir),
            skip_download=bool(args.skip_download),
            max_samples_per_split=max(0, int(args.max_samples_per_split)),
            seed=int(args.seed),
        )
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "failed", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
