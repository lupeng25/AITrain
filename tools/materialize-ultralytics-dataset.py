#!/usr/bin/env python3
"""Materialize a small Ultralytics dataset into a stable local YOLO layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


def write_result(path: Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def quote_yaml(value: object) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not an object: {path}")
    return value


def ultralytics_dataset_yaml(name: str) -> tuple[Path, str]:
    import ultralytics  # type: ignore

    package_root = Path(ultralytics.__file__).resolve().parent
    candidates = [
        package_root / "cfg" / "datasets" / name,
        package_root.parent / "ultralytics" / "cfg" / "datasets" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, getattr(ultralytics, "__version__", "")
    raise FileNotFoundError(f"Ultralytics dataset YAML not found: {name}")


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())


def find_extracted_dataset(extract_root: Path, dataset_name: str) -> Path:
    candidates = [
        extract_root / dataset_name,
        extract_root / dataset_name.replace("-seg", ""),
        extract_root,
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    for candidate in extract_root.rglob("*"):
        if candidate.is_dir() and (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    raise FileNotFoundError(f"Could not find images/labels under extracted dataset: {extract_root}")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    yaml_path, ultralytics_version = ultralytics_dataset_yaml(args.yaml)
    data = load_yaml(yaml_path)
    download_url = str(data.get("download") or "").strip()
    if not download_url:
        raise ValueError(f"Dataset YAML has no download URL: {yaml_path}")

    dataset_name = Path(args.yaml).stem
    destination = Path(args.destination).resolve()
    downloads = Path(args.downloads).resolve()
    downloaded_zip = downloads / f"{dataset_name}.zip"
    extract_root = downloads / f"{dataset_name}-extract"
    materialized_root = Path(args.materialized_root).resolve() if args.materialized_root else destination.parent

    download_file(download_url, downloaded_zip)
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(downloaded_zip) as archive:
        archive.extractall(extract_root)

    extracted_dataset = find_extracted_dataset(extract_root, dataset_name)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(extracted_dataset, destination)

    names = data.get("names") or {0: "item"}
    if isinstance(names, dict):
        ordered_names = [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    else:
        ordered_names = [str(item) for item in names]
    lines = [
        f"path: {quote_yaml(destination.as_posix())}",
        f"train: {quote_yaml(data.get('train', 'images/train'))}",
        f"val: {quote_yaml(data.get('val', 'images/val'))}",
    ]
    if "test" in data:
        lines.append(f"test: {quote_yaml(data.get('test'))}")
    lines.extend([
        f"nc: {len(ordered_names)}",
        "names:",
    ])
    lines.extend(f"  {index}: {quote_yaml(name)}" for index, name in enumerate(ordered_names))
    (destination / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "fallback": False,
        "yamlName": args.yaml,
        "sourceYaml": str(yaml_path),
        "ultralyticsVersion": ultralytics_version,
        "downloadUrl": download_url,
        "downloadedZip": str(downloaded_zip),
        "materializedRoot": str(materialized_root),
        "path": str(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--downloads", default=".deps/datasets/downloads")
    parser.add_argument("--materialized-root", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report_path = Path(args.report).resolve() if args.report else None
    try:
        write_result(report_path, materialize(args))
        return 0
    except Exception as exc:
        write_result(report_path, {"ok": False, "fallback": False, "yamlName": args.yaml, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
