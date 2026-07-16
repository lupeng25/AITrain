#!/usr/bin/env python3
"""Materialize a verified Dataset Snapshot  for an official Python adapter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: Any) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Dataset Snapshot  contains an unsafe relative path: {value}")
    return relative


def materialize_dataset_snapshot(dataset_root: str | Path, manifest_path: str | Path, staging_path: str | Path) -> Path:
    """Copy exactly the manifest files into an empty staging directory.

    The copied bytes are hashed after the copy, so a source mutation cannot
    silently become the official adapter input. The caller owns cleanup of the
    task-local staging directory; this helper never mutates the source dataset.
    """
    root = Path(dataset_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    destination_root = Path(staging_path).resolve()
    if not root.is_dir() or not manifest_file.is_file():
        raise ValueError("Dataset Snapshot  requires an existing dataset root and manifest file")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError("Dataset Snapshot  manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2 or manifest.get("complete") is not True:
        raise ValueError("Dataset Snapshot  manifest is incomplete or unsupported")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Dataset Snapshot  manifest does not contain files")
    if destination_root.exists():
        if any(destination_root.iterdir()):
            raise ValueError("Dataset Snapshot  staging directory must be empty")
    else:
        destination_root.mkdir(parents=True, exist_ok=False)

    normalized_root = os.path.normcase(str(root))
    copied: set[str] = set()
    try:
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("Dataset Snapshot  contains an invalid file entry")
            relative = _relative_path(entry.get("relativePath"))
            expected_hash = str(entry.get("sha256") or "").lower()
            if len(expected_hash) != 64:
                raise ValueError(f"Dataset Snapshot  file is missing SHA-256: {relative}")
            normalized_relative = relative.as_posix()
            if normalized_relative in copied:
                raise ValueError(f"Dataset Snapshot  repeats a file: {relative}")
            copied.add(normalized_relative)
            source = (root / relative).resolve()
            if os.path.commonpath([normalized_root, os.path.normcase(str(source))]) != normalized_root:
                raise ValueError(f"Dataset Snapshot  source escapes dataset root: {relative}")
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"Dataset Snapshot  source is missing or is a symlink: {relative}")
            target = destination_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if _sha256(target) != expected_hash:
                raise ValueError(f"Dataset Snapshot  source changed or does not match manifest: {relative}")
        return destination_root
    except Exception:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise
