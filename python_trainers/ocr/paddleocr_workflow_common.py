#!/usr/bin/env python3
"""PaddleOCR Det/Rec  workflow primitives.

The module deliberately treats the PaddleOCR checkout as read-only.  Every
official command is started through :class:`AdapterSdk`; generated configs,
checkpoints and inference bundles live below the task-local output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from adapter_event_channel import AdapterEventChannel, event_channel_from_environment, standalone_protocol_enabled  # noqa: E402
from adapter_sdk import AdapterCanceled, AdapterSdk  # noqa: E402
from dataset_snapshot import materialize_dataset_snapshot  # noqa: E402
from trainer_protocol import configure_stdio, exception_details  # noqa: E402

configure_stdio()

EXPORTER_VERSION = "aitrain-paddleocr-exporter"
RUNTIME_ROUTE = "paddleocr_official"
ARTIFACT_FORMAT = "paddleocr_inference_bundle"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

_adapter: AdapterSdk | None = None
_event_channel: AdapterEventChannel | None = None


def backend_id(component: str, operation: str) -> str:
    if component not in {"det", "rec"}:
        raise ValueError(f"unsupported PaddleOCR component: {component}")
    suffix = {"train": "official", "evaluate": "official_eval", "export": "official_export", "infer": "official_runtime"}[operation]
    return f"paddleocr_{component}_{suffix}"


def task_type(component: str) -> str:
    return "ocr_detection" if component == "det" else "ocr_recognition"


def decoder(component: str) -> str:
    return "paddleocr_official_det_v1" if component == "det" else "paddleocr_official_rec_v1"


def class_names(component: str) -> list[str]:
    return ["text_region"] if component == "det" else ["text"]


def configure_adapter(component: str, operation: str) -> AdapterSdk:
    global _adapter, _event_channel
    expected = backend_id(component, operation)
    if _event_channel is None and not standalone_protocol_enabled() and _adapter is None:
        _event_channel = event_channel_from_environment()
        _event_channel.connect()
    if _adapter is None or _adapter.backend != expected:
        sink = _event_channel.emit_event if _event_channel is not None else None
        _adapter = AdapterSdk(expected, event_sink=sink, cancel_file=os.environ.get("AITRAIN_CANCEL_FILE"))
    return _adapter


def close_adapter() -> None:
    global _adapter, _event_channel
    if _event_channel is not None:
        _event_channel.close()
    _adapter = None
    _event_channel = None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def merged_options(request: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("parameters", "options"):
        value = request.get(key)
        if isinstance(value, dict):
            result.update(value)
    return result


def first_value(request: dict[str, Any], options: dict[str, Any], *keys: str) -> Any:
    for source in (request, options):
        for key in keys:
            value = source.get(key)
            if value is not None and str(value).strip():
                return value
    return None


def resolve_repo(request: dict[str, Any], options: dict[str, Any]) -> Path:
    value = first_value(request, options, "paddleOcrRepoPath", "paddleOCRRepoPath", "officialRepoPath")
    value = value or os.environ.get("AITRAIN_PADDLEOCR_REPO", "")
    repo = Path(str(value)).expanduser().resolve()
    if not value or not repo.is_dir():
        raise FileNotFoundError("PaddleOCR checkout is required (paddleOcrRepoPath or AITRAIN_PADDLEOCR_REPO)")
    return repo


def official_script(repo: Path, relative: str) -> Path:
    script = (repo / relative).resolve()
    if repo not in script.parents or not script.is_file():
        raise FileNotFoundError(f"official PaddleOCR script is missing: {relative}")
    return script


def python_program(request: dict[str, Any], options: dict[str, Any]) -> str:
    return str(first_value(request, options, "pythonProgram", "officialPython") or sys.executable)


def resolve_config(component: str, request: dict[str, Any], options: dict[str, Any], repo: Path) -> Path:
    value = first_value(request, options, "configPath", "officialConfig")
    if value:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = repo / path
    else:
        relative = ("configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml" if component == "det"
                    else "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")
        path = repo / relative
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"official PaddleOCR config is missing: {path}")
    return path


def require_snapshot(request: dict[str, Any], options: dict[str, Any]) -> tuple[Path, str]:
    dataset_root = Path(str(first_value(request, options, "datasetPath") or "")).resolve()
    manifest_value = first_value(request, options, "datasetSnapshotManifest")
    staging_value = first_value(request, options, "datasetSnapshotStagingPath")
    if not manifest_value or not staging_value:
        raise ValueError("PaddleOCR Train/Evaluate requires Dataset Snapshot  manifest and staging path")
    manifest = Path(str(manifest_value)).resolve()
    staging = Path(str(staging_value)).resolve()
    return materialize_dataset_snapshot(dataset_root, manifest, staging), str(manifest)


def _safe_relative(name: str) -> Path:
    normalized = name.replace("\\", "/")
    path = Path(normalized)
    if (not normalized or normalized.startswith("/") or path.is_absolute()
            or re.match(r"^[A-Za-z]:", normalized) or ".." in path.parts):
        raise ValueError(f"unsafe archive entry: {name}")
    return path


def safe_extract_zip(archive: Path, destination: Path) -> list[Path]:
    if not archive.is_file():
        raise FileNotFoundError(f"archive is missing: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive, "r") as package:
        if len(package.infolist()) > 10000:
            raise ValueError("PaddleOCR bundle contains too many archive entries")
        if sum(info.file_size for info in package.infolist()) > 8 * 1024 * 1024 * 1024:
            raise ValueError("PaddleOCR bundle exceeds the safe uncompressed-size limit")
        seen: set[str] = set()
        for info in package.infolist():
            relative = _safe_relative(info.filename)
            relative_text = relative.as_posix()
            if relative_text in seen:
                raise ValueError(f"PaddleOCR bundle repeats an archive entry: {info.filename}")
            seen.add(relative_text)
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted PaddleOCR bundle entries are forbidden: {info.filename}")
            mode = info.external_attr >> 16
            if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"special files are forbidden in PaddleOCR bundles: {info.filename}")
            target = (root / relative).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive entry escapes extraction root: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    return extracted


def deterministic_zip(source_root: Path, output: Path) -> list[dict[str, Any]]:
    files = sorted((path for path in source_root.rglob("*") if path.is_file() and not path.is_symlink()),
                   key=lambda path: path.relative_to(source_root).as_posix())
    if not files:
        raise ValueError(f"cannot package an empty PaddleOCR directory: {source_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in files:
            relative = path.relative_to(source_root).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            package.writestr(info, data)
            inventory.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return inventory


def verify_inventory(extracted_root: Path, expected: Any) -> None:
    if not isinstance(expected, list) or not expected:
        raise ValueError("PaddleOCR bundle sidecar has no inventory")
    expected_paths: set[str] = set()
    for item in expected:
        if not isinstance(item, dict):
            raise ValueError("PaddleOCR bundle inventory contains an invalid row")
        relative = _safe_relative(str(item.get("path") or ""))
        relative_text = relative.as_posix()
        if relative_text in expected_paths:
            raise ValueError(f"PaddleOCR bundle inventory repeats {relative_text}")
        expected_paths.add(relative_text)
        path = extracted_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"PaddleOCR bundle inventory file is missing: {relative_text}")
        if path.stat().st_size != int(item.get("bytes", -1)) or sha256_file(path) != str(item.get("sha256") or "").lower():
            raise ValueError(f"PaddleOCR bundle inventory mismatch: {relative_text}")
    actual_paths = {path.relative_to(extracted_root).as_posix() for path in extracted_root.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise ValueError("PaddleOCR bundle contains files outside its signed inventory")


def copy_file(source: Path, target: Path) -> Path:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"required regular file is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def required_output_path(request: dict[str, Any]) -> Path:
    value = request.get("outputPath")
    if value is None or not str(value).strip():
        raise ValueError("outputPath is required")
    output = Path(str(value)).expanduser().resolve()
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError(f"outputPath must be a regular directory path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _find_dataset_file(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _dictionary(component: str, root: Path, request: dict[str, Any], options: dict[str, Any], repo: Path) -> Path | None:
    if component != "rec":
        return None
    value = first_value(request, options, "dictionaryPath", "characterDictPath")
    candidates = [Path(str(value)).expanduser()] if value else []
    candidates.extend([root / "dict.txt", root / "dictionary.txt", repo / "ppocr/utils/ppocr_keys_v1.txt"])
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = repo / candidate
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    raise FileNotFoundError("PaddleOCR Rec requires an explicit or official character dictionary")


def _dataset_overrides(component: str, root: Path, dictionary_path: Path | None) -> list[str]:
    train_labels = _find_dataset_file(root, ("train.txt", "train_list.txt", "rec_gt_train.txt", "det_gt_train.txt"))
    eval_labels = _find_dataset_file(root, ("val.txt", "test.txt", "val_list.txt", "rec_gt_val.txt", "det_gt_val.txt"))
    overrides = [f"Train.dataset.data_dir={root}", f"Eval.dataset.data_dir={root}"]
    if train_labels:
        overrides.append(f"Train.dataset.label_file_list=[{train_labels}]")
    if eval_labels:
        overrides.append(f"Eval.dataset.label_file_list=[{eval_labels}]")
    if component == "rec" and dictionary_path:
        overrides.append(f"Global.character_dict_path={dictionary_path}")
    return overrides


def _run_official(sdk: AdapterSdk, command: list[str], repo: Path, log_path: Path) -> tuple[str, ...]:
    child_environment = dict(os.environ)
    # Running an official script must not populate a shared checkout with
    # __pycache__ files. Model/download caches remain controlled by the Worker
    # environment and the task-local command arguments.
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = sdk.run_child_process(command, cwd=repo, env=child_environment, log_path=log_path)
    if result.canceled:
        raise AdapterCanceled("PaddleOCR official subprocess was canceled")
    if result.exit_code != 0:
        raise RuntimeError(f"PaddleOCR official subprocess failed with exit code {result.exit_code}: " + " | ".join(result.tail_lines[-5:]))
    return result.tail_lines


def _artifact(sdk: AdapterSdk, kind: str, path: Path, message: str) -> None:
    sdk.emit_artifact_candidate(kind, path, message=message)


def run_train(component: str, request: dict[str, Any]) -> int:
    sdk = configure_adapter(component, "train")
    options = merged_options(request)
    repo = resolve_repo(request, options)
    dataset, manifest = require_snapshot(request, options)
    output = required_output_path(request)
    config_source = resolve_config(component, request, options, repo)
    config_path = copy_file(config_source, output / "train.yml")
    dictionary_source = _dictionary(component, dataset, request, options, repo)
    dictionary_path = copy_file(dictionary_source, output / "dict.txt") if dictionary_source else None
    official_output = output / "official-checkpoint"
    command = [python_program(request, options), str(official_script(repo, "tools/train.py")), "-c", str(config_path), "-o",
               f"Global.save_model_dir={official_output}", *_dataset_overrides(component, dataset, dictionary_path)]
    sdk.emit_progress(10, message="Materialized verified PaddleOCR Dataset Snapshot ", datasetSnapshotManifest=manifest)
    tail = _run_official(sdk, command, repo, output / "official_train.log")
    inventory = deterministic_zip(official_output, output / "model.zip")
    report = {
        "schemaVersion": 2, "backend": backend_id(component, "train"), "sourceTrainingBackend": backend_id(component, "train"),
        "taskType": task_type(component), "component": component, "datasetSnapshotManifest": manifest,
        "configPath": "train.yml", "checkpointPath": "model.zip", "dictionaryPath": "dict.txt" if dictionary_path else "",
        "checkpointInventory": inventory, "officialOutputTail": list(tail), "completedAt": now_iso(),
    }
    report_path = output / "training_report.json"
    write_json(report_path, report)
    _artifact(sdk, "checkpoint", output / "model.zip", "Official PaddleOCR checkpoint package")
    _artifact(sdk, "config", config_path, "Task-local PaddleOCR training config")
    if dictionary_path:
        _artifact(sdk, "dictionary", dictionary_path, "PaddleOCR Rec character dictionary")
    _artifact(sdk, "training_report", report_path, "Official PaddleOCR training report")
    sdk.emit_progress(100, message="Official PaddleOCR training completed")
    sdk.emit_completed("Official PaddleOCR training completed", checkpointPath=str(output / "model.zip"), configPath=str(config_path),
                       dictionaryPath=str(dictionary_path or ""), reportPath=str(report_path))
    return 0


def _input_path(request: dict[str, Any], options: dict[str, Any], key: str, fallback_model: bool = False) -> Path:
    value = first_value(request, options, key)
    if not value and fallback_model:
        value = request.get("modelPath")
    path = Path(str(value or "")).resolve()
    if not value or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required {key} is missing")
    return path


def _pretrained_base(extracted: Path) -> Path:
    params = sorted(extracted.rglob("*.pdparams"))
    if params:
        return params[0].with_suffix("")
    files = sorted(path for path in extracted.rglob("*") if path.is_file())
    if not files:
        raise ValueError("PaddleOCR checkpoint package is empty")
    return files[0]


def _parse_metrics(lines: Iterable[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in lines:
        for name, value in re.findall(r"([A-Za-z][A-Za-z0-9_./-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)", line):
            try:
                metrics[name] = float(value)
            except ValueError:
                pass
    return metrics


def run_evaluate(component: str, request: dict[str, Any]) -> int:
    sdk = configure_adapter(component, "evaluate")
    options = merged_options(request)
    repo = resolve_repo(request, options)
    dataset, manifest = require_snapshot(request, options)
    output = required_output_path(request)
    checkpoint = _input_path(request, options, "checkpointPath", fallback_model=True)
    config = _input_path(request, options, "configPath")
    dictionary = _input_path(request, options, "dictionaryPath") if component == "rec" else None
    checkpoint_copy = copy_file(checkpoint, output / "model.zip")
    config_copy = copy_file(config, output / "train.yml")
    dictionary_copy = copy_file(dictionary, output / "dict.txt") if dictionary else None
    extracted = output / "checkpoint-input"
    safe_extract_zip(checkpoint_copy, extracted)
    command = [python_program(request, options), str(official_script(repo, "tools/eval.py")), "-c", str(config_copy), "-o",
               f"Global.pretrained_model={_pretrained_base(extracted)}", *_dataset_overrides(component, dataset, dictionary_copy)]
    sdk.emit_progress(15, message="Materialized verified PaddleOCR evaluation snapshot", datasetSnapshotManifest=manifest)
    tail = _run_official(sdk, command, repo, output / "official_eval.log")
    metrics = _parse_metrics(tail)
    report = {"schemaVersion": 2, "backend": backend_id(component, "evaluate"), "taskType": task_type(component),
              "component": component, "datasetSnapshotManifest": manifest, "metrics": metrics,
              "officialOutputTail": list(tail), "evaluatedAt": now_iso()}
    report_path = output / "evaluation_report.json"
    write_json(report_path, report)
    for name, value in metrics.items():
        sdk.emit_metric(name, value)
    _artifact(sdk, "checkpoint", checkpoint_copy, "Propagated PaddleOCR checkpoint")
    _artifact(sdk, "config", config_copy, "Propagated PaddleOCR config")
    if dictionary_copy:
        _artifact(sdk, "dictionary", dictionary_copy, "Propagated PaddleOCR dictionary")
    _artifact(sdk, "evaluation_report", report_path, "Official PaddleOCR evaluation report")
    sdk.emit_progress(100, message="Official PaddleOCR evaluation completed")
    sdk.emit_completed("Official PaddleOCR evaluation completed", checkpointPath=str(checkpoint_copy), configPath=str(config_copy),
                       dictionaryPath=str(dictionary_copy or ""), reportPath=str(report_path))
    return 0


def run_export(component: str, request: dict[str, Any]) -> int:
    sdk = configure_adapter(component, "export")
    options = merged_options(request)
    repo = resolve_repo(request, options)
    output = required_output_path(request)
    checkpoint = _input_path(request, options, "checkpointPath", fallback_model=True)
    config = _input_path(request, options, "configPath")
    dictionary = _input_path(request, options, "dictionaryPath") if component == "rec" else None
    evaluation = _input_path(request, options, "evaluationReportPath")
    extracted = output / "checkpoint-input"
    safe_extract_zip(checkpoint, extracted)
    inference = output / "official-inference"
    config_copy = copy_file(config, output / "export-input.yml")
    dictionary_copy = copy_file(dictionary, output / "export-dict.txt") if dictionary else None
    command = [python_program(request, options), str(official_script(repo, "tools/export_model.py")), "-c", str(config_copy), "-o",
               f"Global.pretrained_model={_pretrained_base(extracted)}", f"Global.save_inference_dir={inference}"]
    if dictionary_copy:
        command.append(f"Global.character_dict_path={dictionary_copy}")
    sdk.emit_progress(20, message="Running official PaddleOCR inference export")
    tail = _run_official(sdk, command, repo, output / "official_export.log")
    if dictionary_copy:
        copy_file(dictionary_copy, inference / "dict.txt")
    bundle_path = output / "paddleocr_inference.zip"
    inventory = deterministic_zip(inference, bundle_path)
    bundle_hash = sha256_file(bundle_path)
    sidecar = {
        "schemaVersion": 2,
        "kind": "paddleocr_bundle",
        "artifactFormat": ARTIFACT_FORMAT,
        "sourceTrainingBackend": backend_id(component, "train"),
        "modelFamily": task_type(component),
        "taskType": task_type(component),
        "decoder": decoder(component),
        "exporterVersion": EXPORTER_VERSION,
        "classNames": class_names(component),
        "preprocessing": {"id": f"paddleocr_official_{component}_preprocess", "ownedBy": "PaddleOCR"},
        "postprocessing": {"id": decoder(component), "ownedBy": "PaddleOCR"},
        "runtimeRoutes": [RUNTIME_ROUTE],
        "inventory": inventory,
        "bundlePath": "paddleocr_inference.zip",
        "bundleSha256": bundle_hash,
        "bundleByteCount": bundle_path.stat().st_size,
        "evaluationReportSha256": sha256_file(evaluation),
        "officialOutputTail": list(tail),
        "exportedAt": now_iso(),
        "limitations": ["Only the official PaddleOCR Python runtime is declared.",
                        "A single Det or Rec workflow is not a PaddleOCR System acceptance result."],
    }
    sidecar_path = output / "paddleocr_bundle.json"
    write_json(sidecar_path, sidecar)
    _artifact(sdk, "export", bundle_path, "Deterministic official PaddleOCR inference bundle")
    # Both files intentionally share the ``export`` kind so the immutable
    # Artifact layout is export/<name>, matching the Profile's exact contract.
    _artifact(sdk, "export", sidecar_path, "Verified PaddleOCR bundle sidecar")
    sdk.emit_progress(100, message="Official PaddleOCR export completed")
    sdk.emit_completed("Official PaddleOCR export completed", exportPath=str(bundle_path), sidecarPath=str(sidecar_path), reportPath=str(sidecar_path))
    return 0


def run_predict(component: str, request: dict[str, Any]) -> int:
    sdk = configure_adapter(component, "infer")
    options = merged_options(request)
    repo = resolve_repo(request, options)
    output = required_output_path(request)
    sidecar_path = _input_path(request, options, "sidecarPath", fallback_model=True)
    bundle_path = _input_path(request, options, "bundlePath")
    image_path = _input_path(request, options, "imagePath")
    sidecar = read_json(sidecar_path)
    if sidecar.get("kind") != "paddleocr_bundle" or sidecar.get("artifactFormat") != ARTIFACT_FORMAT:
        raise ValueError("PaddleOCR deployment sidecar has an invalid bundle contract")
    if (sidecar.get("sourceTrainingBackend") != backend_id(component, "train")
            or sidecar.get("modelFamily") != task_type(component)
            or sidecar.get("taskType") != task_type(component)
            or sidecar.get("decoder") != decoder(component)
            or sidecar.get("runtimeRoutes") != [RUNTIME_ROUTE]):
        raise ValueError("PaddleOCR deployment sidecar does not match the requested component/runtime")
    if bundle_path.stat().st_size != int(sidecar.get("bundleByteCount", -1)) or sha256_file(bundle_path) != str(sidecar.get("bundleSha256") or ""):
        raise ValueError("PaddleOCR inference bundle hash or byte count does not match its sidecar")
    extracted = output / "verified-inference"
    safe_extract_zip(bundle_path, extracted)
    verify_inventory(extracted, sidecar.get("inventory"))
    script_name = "tools/infer/predict_det.py" if component == "det" else "tools/infer/predict_rec.py"
    model_arg = "--det_model_dir" if component == "det" else "--rec_model_dir"
    official_preview_root = output / "official-preview"
    command = [python_program(request, options), str(official_script(repo, script_name)), model_arg, str(extracted),
               "--image_dir", str(image_path), "--draw_img_save_dir", str(official_preview_root)]
    if component == "rec":
        bundled_dictionary = extracted / "dict.txt"
        if not bundled_dictionary.is_file():
            raise ValueError("PaddleOCR Rec inference bundle is missing dict.txt")
        command.extend(["--rec_char_dict_path", str(bundled_dictionary)])
    sdk.emit_progress(30, message="Verified PaddleOCR inference bundle and inventory")
    tail = _run_official(sdk, command, repo, output / "official_predict.log")
    prediction = {"schemaVersion": 2, "taskType": task_type(component), "runtimeRoute": RUNTIME_ROUTE,
                  "imagePath": str(image_path), "officialOutputTail": list(tail), "predictedAt": now_iso()}
    prediction_path = output / "inference_predictions.json"
    write_json(prediction_path, prediction)
    rendered_previews = sorted(path for path in official_preview_root.rglob("*")
                               if path.is_file() and not path.is_symlink()) if official_preview_root.is_dir() else []
    preview_source = rendered_previews[0] if rendered_previews else image_path
    preview_path = output / ("preview" + (preview_source.suffix.lower() or image_path.suffix.lower() or ".bin"))
    copy_file(preview_source, preview_path)
    report = {"schemaVersion": 2, "status": "passed", "backend": backend_id(component, "infer"),
              "taskType": task_type(component), "runtimeRoute": RUNTIME_ROUTE,
              "bundleSha256": sidecar["bundleSha256"], "bundleByteCount": sidecar["bundleByteCount"],
              "inventoryVerified": True, "predictionPath": prediction_path.name, "previewPath": preview_path.name,
              "validatedAt": now_iso()}
    report_path = output / "deployment_report.json"
    write_json(report_path, report)
    _artifact(sdk, "deployment_report", report_path, "Official PaddleOCR deployment validation report")
    _artifact(sdk, "prediction", prediction_path, "Official PaddleOCR prediction")
    _artifact(sdk, "preview", preview_path, "PaddleOCR deployment sample preview")
    sdk.emit_progress(100, message="Official PaddleOCR deployment validation completed")
    sdk.emit_completed("Official PaddleOCR deployment validation completed", reportPath=str(report_path),
                       predictionPath=str(prediction_path), previewPath=str(preview_path))
    return 0


def cli_main(component: str, operation: str, argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    sdk: AdapterSdk | None = None
    try:
        sdk = configure_adapter(component, operation)
        request = read_json(args.request)
        return {"train": run_train, "evaluate": run_evaluate, "export": run_export, "infer": run_predict}[operation](component, request)
    except AdapterCanceled as exc:
        if sdk is not None:
            sdk.emit_canceled(str(exc))
        return 130
    except Exception as exc:
        if sdk is None:
            sdk = configure_adapter(component, operation)
        return sdk.emit_failed(str(exc), f"paddleocr_{component}_{operation}_failed", exception_details(exc))
    finally:
        close_adapter()
