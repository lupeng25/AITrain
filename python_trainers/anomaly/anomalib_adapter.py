#!/usr/bin/env python3
"""Worker-managed Anomalib adapter for AITrain anomaly_detection v1."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))
REPO_ROOT = TRAINER_ROOT.parent

from trainer_protocol import configure_stdio, emit_event, emit_failed, unhandled_failure


configure_stdio()


BACKEND_PATCHCORE = "anomalib_patchcore"
BACKEND_EFFICIENTAD = "anomalib_efficientad"
TASK_TYPE = "anomaly_detection"
DATASET_FORMAT = "anomaly_folder"
RUNTIME = "anomalib_python"
_DATASET_VIEW_TEMPDIRS: List[Any] = []


@dataclass(frozen=True)
class Preset:
    backend: str
    name: str
    defaults: Dict[str, Any]


@dataclass
class DatamoduleLayout:
    root: Path
    eval_split: str
    tempdir: Optional[Any] = None
    materialized_alias: bool = False


PRESETS: Dict[str, Preset] = {
    "anomalib_patchcore_wide_resnet50_2": Preset(
        backend=BACKEND_PATCHCORE,
        name="anomalib_patchcore_wide_resnet50_2",
        defaults={
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coresetSamplingRatio": 0.1,
            "numNeighbors": 9,
        },
    ),
    "anomalib_efficientad_s": Preset(
        backend=BACKEND_EFFICIENTAD,
        name="anomalib_efficientad_s",
        defaults={
            "modelSize": "small",
            "lr": 0.0001,
            "weightDecay": 0.00001,
        },
    ),
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Request JSON must be an object: {path}")
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def image_suffixes() -> Tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes())


def mvtec_defect_dirs(dataset_path: Path) -> List[Path]:
    test_root = dataset_path / "test"
    if not test_root.exists():
        return []
    return [
        path
        for path in sorted(test_root.iterdir())
        if path.is_dir() and path.name.lower() not in {"good", "anomaly"}
    ]


def mask_stem_matches(mask_path: Path, image_stem: str) -> bool:
    mask_stem = mask_path.stem
    return mask_stem == image_stem or mask_stem == f"{image_stem}_mask" or mask_stem.startswith(f"{image_stem}_")


def mask_for_image(canonical_mask_dir: Path, mvtec_mask_dir: Optional[Path], image_path: Path) -> Optional[Path]:
    image_stem = image_path.stem
    for candidate in (
        canonical_mask_dir / f"{image_stem}.png",
        canonical_mask_dir / f"{image_stem}_mask.png",
    ):
        if candidate.exists():
            return candidate
    if mvtec_mask_dir is not None and mvtec_mask_dir.exists():
        for mask_path in sorted(mvtec_mask_dir.glob("*.png")):
            if mask_stem_matches(mask_path, image_stem):
                return mask_path
    return None


def link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_image_dir(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.exists():
        return
    for image_path in list_images(source_dir):
        relative = image_path.relative_to(source_dir)
        link_or_copy_file(image_path, destination_dir / relative)


def alias_image_name(defect_type: str, image_path: Path, defect_dir: Path) -> str:
    relative = image_path.relative_to(defect_dir)
    flattened = "__".join(relative.parts)
    return f"{defect_type}__{flattened}"


def materialize_anomaly_dir(
    source_dir: Path,
    image_destination_dir: Path,
    mask_destination_dir: Path,
    canonical_mask_dir: Path,
    mvtec_mask_dir: Optional[Path],
    defect_type: str = "",
) -> None:
    if not source_dir.exists():
        return
    for image_path in list_images(source_dir):
        if defect_type:
            target_name = alias_image_name(defect_type, image_path, source_dir)
            image_destination = image_destination_dir / target_name
        else:
            relative = image_path.relative_to(source_dir)
            image_destination = image_destination_dir / relative
        link_or_copy_file(image_path, image_destination)
        mask_path = mask_for_image(canonical_mask_dir, mvtec_mask_dir, image_path)
        if mask_path is not None:
            link_or_copy_file(mask_path, mask_destination_dir / f"{image_destination.stem}.png")


def create_dataset_view_root(params: Dict[str, Any]) -> Tuple[Path, Optional[Any]]:
    requested = params.get("_aitrainDatasetViewRoot") or params.get("_aitrain_dataset_view_root")
    if requested:
        view_root = Path(str(requested)).resolve() / "anomaly_folder_canonical"
        if view_root.exists():
            shutil.rmtree(view_root)
        view_root.mkdir(parents=True, exist_ok=True)
        return view_root, None
    tempdir = tempfile.TemporaryDirectory(prefix="aitrain_anomaly_folder_")
    return Path(tempdir.name).resolve(), tempdir


def select_eval_split(dataset_path: Path, defect_dirs: Sequence[Path]) -> str:
    if (dataset_path / "test" / "anomaly").exists() or (dataset_path / "test" / "good").exists() or defect_dirs:
        return "test"
    if (dataset_path / "val" / "anomaly").exists() or (dataset_path / "val" / "good").exists():
        return "val"
    return "train"


def datamodule_layout(dataset_path: Path, params: Dict[str, Any]) -> DatamoduleLayout:
    defect_dirs = mvtec_defect_dirs(dataset_path)
    eval_split = select_eval_split(dataset_path, defect_dirs)
    if not defect_dirs:
        return DatamoduleLayout(root=dataset_path, eval_split=eval_split)

    view_root, tempdir = create_dataset_view_root(params)
    materialize_image_dir(dataset_path / "train" / "good", view_root / "train" / "good")
    for split in ("val", "test"):
        materialize_image_dir(dataset_path / split / "good", view_root / split / "good")
        materialize_anomaly_dir(
            dataset_path / split / "anomaly",
            view_root / split / "anomaly",
            view_root / "masks" / split / "anomaly",
            dataset_path / "masks" / split / "anomaly",
            None,
        )

    for defect_dir in defect_dirs:
        materialize_anomaly_dir(
            defect_dir,
            view_root / "test" / "anomaly",
            view_root / "masks" / "test" / "anomaly",
            dataset_path / "masks" / "test" / "anomaly",
            dataset_path / "ground_truth" / defect_dir.name,
            defect_type=defect_dir.name,
        )

    for split in ("val", "test"):
        (view_root / split / "anomaly").mkdir(parents=True, exist_ok=True)
        (view_root / "masks" / split / "anomaly").mkdir(parents=True, exist_ok=True)
    return DatamoduleLayout(root=view_root, eval_split=eval_split, tempdir=tempdir, materialized_alias=True)


def dataset_inventory(dataset_path: Path) -> Dict[str, Any]:
    splits: Dict[str, Dict[str, int]] = {}
    normal_count = 0
    anomaly_count = 0
    mask_count = 0
    for split in ("train", "val", "test"):
        good = list_images(dataset_path / split / "good")
        anomaly = list_images(dataset_path / split / "anomaly")
        if split == "test":
            test_root = dataset_path / "test"
            if test_root.exists():
                for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
                    if defect_dir.name.lower() in {"good", "anomaly"}:
                        continue
                    anomaly.extend(list_images(defect_dir))
        split_masks = list_images(dataset_path / "masks" / split / "anomaly")
        if split == "test":
            gt_root = dataset_path / "ground_truth"
            if gt_root.exists():
                for defect_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
                    split_masks.extend(list_images(defect_dir))
        splits[split] = {
            "good": len(good),
            "anomaly": len(anomaly),
            "mask": len(split_masks),
        }
        normal_count += len(good)
        anomaly_count += len(anomaly)
        mask_count += len(split_masks)
    return {
        "normalCount": normal_count,
        "anomalyCount": anomaly_count,
        "maskCount": mask_count,
        "splits": splits,
        "evaluationLimited": anomaly_count == 0,
        "pixelEvaluationAvailable": mask_count > 0,
    }


def merge_preset(backend: str, params: Dict[str, Any]) -> Dict[str, Any]:
    preset_name = str(params.get("modelPreset") or "").strip()
    if not preset_name:
        preset_name = (
            "anomalib_efficientad_s"
            if backend == BACKEND_EFFICIENTAD
            else "anomalib_patchcore_wide_resnet50_2"
        )
    preset = PRESETS.get(preset_name)
    merged: Dict[str, Any] = {}
    if preset:
        merged.update(preset.defaults)
    merged.update(params)
    merged["modelPreset"] = preset_name
    return merged


def layers_value(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return ["layer2", "layer3"]


def resolve_imagenet_dir(params: Dict[str, Any]) -> Optional[Path]:
    candidates = [
        str(params.get("imagenetDir") or "").strip(),
        os.environ.get("AITRAIN_ANOMALIB_IMAGENET_DIR", "").strip(),
        str(REPO_ROOT / ".deps" / "anomalib" / "imagenette"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    return None


def require_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Required Python module is unavailable: {module_name}: {exc}") from exc


def import_anomalib_symbols():
    anomalib = require_module("anomalib")
    data_mod = require_module("anomalib.data")
    models_mod = require_module("anomalib.models")
    engine_mod = require_module("anomalib.engine")
    folder_cls = getattr(data_mod, "Folder", None)
    engine_cls = getattr(engine_mod, "Engine", None)
    patchcore_cls = getattr(models_mod, "Patchcore", None) or getattr(models_mod, "PatchCore", None)
    efficientad_cls = (
        getattr(models_mod, "EfficientAd", None)
        or getattr(models_mod, "EfficientAD", None)
        or getattr(models_mod, "Efficientad", None)
    )
    if folder_cls is None or engine_cls is None:
        raise RuntimeError("Anomalib Folder datamodule or Engine class was not found.")
    return anomalib, folder_cls, engine_cls, patchcore_cls, efficientad_cls


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def first_prediction(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            prediction = first_prediction(item)
            if prediction is not None:
                return prediction
        return None
    return value


def scalar_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return default


def extract_metrics(raw_result: Any, threshold: float) -> Dict[str, Any]:
    safe = json_safe(raw_result)
    source: Dict[str, Any] = {}
    if isinstance(safe, list) and safe and isinstance(safe[0], dict):
        source = safe[0]
    elif isinstance(safe, dict):
        source = safe
    metrics: Dict[str, Any] = {"threshold": threshold, "rawTestResult": safe}
    key_map = {
        "imageAUROC": ("image_AUROC", "image_auroc", "image/auroc", "auroc"),
        "imageF1": ("image_F1Score", "image_f1", "image/f1", "f1"),
        "pixelAUROC": ("pixel_AUROC", "pixel_auroc", "pixel/auroc"),
        "pixelF1": ("pixel_F1Score", "pixel_f1", "pixel/f1"),
    }
    for output_key, candidates in key_map.items():
        for candidate in candidates:
            if candidate in source:
                metrics[output_key] = source[candidate]
                break
    return metrics


def build_datamodule(folder_cls, dataset_path: Path, params: Dict[str, Any]):
    batch_size = int(params.get("batchSize") or params.get("batch_size") or 1)
    workers = int(params.get("workers") or 0)
    image_size = int(params.get("imageSize") or params.get("image_size") or 256)
    layout = datamodule_layout(dataset_path, params)
    eval_split = layout.eval_split
    abnormal_dir = f"{eval_split}/anomaly"
    normal_test_dir = f"{eval_split}/good"
    mask_dir = f"masks/{eval_split}/anomaly"
    attempts = [
        {
            "name": "aitrain_anomaly",
            "root": str(layout.root),
            "normal_dir": "train/good",
            "abnormal_dir": abnormal_dir,
            "normal_test_dir": normal_test_dir,
            "mask_dir": mask_dir,
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "num_workers": workers,
        },
        {
            "root": str(layout.root),
            "normal_dir": "train/good",
            "abnormal_dir": abnormal_dir,
            "normal_test_dir": normal_test_dir,
            "mask_dir": mask_dir,
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "num_workers": workers,
            "image_size": image_size,
        },
        {
            "root": str(layout.root),
            "normal_dir": "train/good",
            "abnormal_dir": abnormal_dir,
            "normal_test_dir": normal_test_dir,
            "mask_dir": mask_dir,
        },
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            datamodule = folder_cls(**kwargs)
            if layout.tempdir is not None:
                try:
                    setattr(datamodule, "_aitrain_dataset_view_tempdir", layout.tempdir)
                except Exception:
                    _DATASET_VIEW_TEMPDIRS.append(layout.tempdir)
            if layout.materialized_alias:
                try:
                    setattr(datamodule, "_aitrain_dataset_view_root", str(layout.root))
                except Exception:
                    pass
            return datamodule
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Cannot construct Anomalib Folder datamodule: {last_error}") from last_error


def normalize_efficientad_model_size(value: Any) -> str:
    normalized = str(value or "small").strip().lower()
    if normalized in {"s", "small"}:
        return "small"
    if normalized in {"m", "medium"}:
        return "medium"
    return "small"


def build_model(backend: str, params: Dict[str, Any], patchcore_cls, efficientad_cls):
    if backend == BACKEND_PATCHCORE:
        if patchcore_cls is None:
            raise RuntimeError("Anomalib PatchCore model class was not found.")
        return patchcore_cls(
            backbone=str(params.get("backbone") or "wide_resnet50_2"),
            layers=layers_value(params.get("layers", ["layer2", "layer3"])),
            coreset_sampling_ratio=float(params.get("coresetSamplingRatio") or 0.1),
            num_neighbors=int(params.get("numNeighbors") or 9),
        )
    if backend == BACKEND_EFFICIENTAD:
        if efficientad_cls is None:
            raise RuntimeError("Anomalib EfficientAD model class was not found.")
        imagenet_dir = resolve_imagenet_dir(params)
        if imagenet_dir is None:
            raise FileNotFoundError("efficientad_imagenet_dir_missing")
        model_size = normalize_efficientad_model_size(params.get("modelSize"))
        attempts = [
            {
                "model_size": model_size,
                "lr": float(params.get("lr") or 0.0001),
                "weight_decay": float(params.get("weightDecay") or 0.00001),
                "imagenet_dir": str(imagenet_dir),
            },
            {
                "model_size": model_size,
                "imagenet_dir": str(imagenet_dir),
            },
        ]
        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return efficientad_cls(**kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Cannot construct Anomalib EfficientAD model: {last_error}") from last_error
    raise RuntimeError(f"Unsupported anomaly backend: {backend}")


def accelerator_and_devices(device: str) -> Tuple[str, Any]:
    normalized = (device or "cpu").strip().lower()
    if normalized in {"cpu", ""}:
        return "cpu", 1
    if normalized == "cuda":
        return "gpu", 1
    if normalized.isdigit():
        return "gpu", [int(normalized)]
    return "auto", "auto"


def inferencer_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized == "cpu":
        return "cpu"
    if normalized == "xpu":
        return "xpu"
    if normalized == "gpu" or normalized == "cuda" or normalized.startswith("cuda:") or normalized.isdigit():
        return "cuda"
    return "auto"


def make_engine(engine_cls, output_path: Path, params: Dict[str, Any]):
    device = str(params.get("device") or "cpu")
    accelerator, devices = accelerator_and_devices(device)
    epochs = int(params.get("epochs") or 1)
    attempts = [
        {
            "max_epochs": epochs,
            "default_root_dir": str(output_path),
            "accelerator": accelerator,
            "devices": devices,
        },
        {
            "default_root_dir": str(output_path),
            "accelerator": accelerator,
            "devices": devices,
        },
        {"default_root_dir": str(output_path)},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return engine_cls(**kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Cannot construct Anomalib Engine: {last_error}") from last_error


def latest_checkpoint(output_path: Path) -> Optional[Path]:
    checkpoints = sorted(output_path.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return checkpoints[-1] if checkpoints else None


def training_report(
    request: Dict[str, Any],
    params: Dict[str, Any],
    inventory: Dict[str, Any],
    status: str,
    ok: bool,
    message: str = "",
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    backend = str(request.get("backend") or params.get("trainingBackend") or BACKEND_PATCHCORE)
    report = {
        "schemaVersion": 1,
        "kind": "anomalib_training_report",
        "createdAt": now_iso(),
        "ok": ok,
        "status": status,
        "taskType": TASK_TYPE,
        "datasetFormat": DATASET_FORMAT,
        "trainingBackend": backend,
        "modelFamily": "anomaly_detection",
        "runtime": RUNTIME,
        "datasetPath": request.get("datasetPath", ""),
        "outputPath": request.get("outputPath", ""),
        "modelPreset": params.get("modelPreset", ""),
        "parameters": params,
        "datasetInventory": inventory,
        "evaluationLimited": bool(inventory.get("evaluationLimited")),
        "pixelEvaluationAvailable": bool(inventory.get("pixelEvaluationAvailable")),
        "metrics": metrics or {},
        "message": message,
        "scaffold": False,
        "runtimeBoundary": "Worker-managed Python/Anomalib artifacts; not AITrain C++ ONNX/TensorRT/NCNN runtime.",
    }
    if inventory.get("evaluationLimited"):
        report["limitations"] = ["Only normal samples were found; evaluation metrics are limited."]
    return report


def write_failed_report(
    output_path: Path,
    request: Dict[str, Any],
    params: Dict[str, Any],
    inventory: Dict[str, Any],
    message: str,
    error_code: str,
    status: str = "failed",
) -> Path:
    report = training_report(request, params, inventory, status, False, message)
    report["errorCode"] = error_code
    report["failureCategory"] = "anomalib"
    report_path = output_path / "anomalib_training_report.json"
    write_json(report_path, report)
    return report_path


def sidecar_payload(
    request: Dict[str, Any],
    params: Dict[str, Any],
    checkpoint: Optional[Path],
    report_path: Path,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    threshold = float(params.get("threshold") or params.get("quantile") or 0.5)
    return {
        "schemaVersion": 1,
        "kind": "anomaly_sidecar",
        "createdAt": now_iso(),
        "taskType": TASK_TYPE,
        "datasetFormat": DATASET_FORMAT,
        "trainingBackend": request.get("backend", BACKEND_PATCHCORE),
        "modelFamily": "anomaly_detection",
        "runtime": RUNTIME,
        "checkpointPath": str(checkpoint) if checkpoint else "",
        "trainingReportPath": str(report_path),
        "threshold": threshold,
        "thresholdStrategy": params.get("thresholdStrategy", "quantile"),
        "quantile": float(params.get("quantile") or 0.995),
        "okNgDecision": {"ok": "score <= threshold", "ng": "score > threshold"},
        "datasetInventory": inventory,
        "parameters": params,
    }


def train(request: Dict[str, Any]) -> int:
    output_path = Path(str(request.get("outputPath") or ".")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    backend = str(request.get("backend") or BACKEND_PATCHCORE).strip().lower()
    params = merge_preset(backend, dict(request.get("parameters") or {}))
    if backend == BACKEND_EFFICIENTAD:
        params["batchSize"] = 1
        params["batch_size"] = 1
    datamodule_params = dict(params)
    datamodule_params["_aitrainDatasetViewRoot"] = str(output_path / "_aitrain_dataset_views")
    dataset_path = Path(str(request.get("datasetPath") or "")).resolve()
    inventory = dataset_inventory(dataset_path)

    emit_event(backend, "progress", message="Starting Anomalib training.", percent=1)
    if backend == BACKEND_EFFICIENTAD and resolve_imagenet_dir(params) is None:
        report_path = write_failed_report(
            output_path,
            request,
            params,
            inventory,
            "EfficientAD requires imagenetDir from parameters, AITRAIN_ANOMALIB_IMAGENET_DIR, or .deps/anomalib/imagenette. AITrain will not auto-download external data.",
            "efficientad_imagenet_dir_missing",
            "blocked",
        )
        emit_event(backend, "artifact", kind="report", path=str(report_path), message="Blocked Anomalib training report")
        emit_failed(backend, "EfficientAD ImageNet directory is missing.", "efficientad_imagenet_dir_missing", {"reportPath": str(report_path)})
        return 1

    try:
        _, folder_cls, engine_cls, patchcore_cls, efficientad_cls = import_anomalib_symbols()
        datamodule = build_datamodule(folder_cls, dataset_path, datamodule_params)
        model = build_model(backend, params, patchcore_cls, efficientad_cls)
        engine = make_engine(engine_cls, output_path, params)
        emit_event(backend, "progress", message="Running Anomalib fit().", percent=8)
        engine.fit(model=model, datamodule=datamodule)
        emit_event(backend, "progress", message="Collecting Anomalib artifacts.", percent=90)
        checkpoint = latest_checkpoint(output_path)
        metrics: Dict[str, Any] = {}
        if not inventory.get("evaluationLimited"):
            try:
                test_result = engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint) if checkpoint else None)
                threshold = float(params.get("threshold") or params.get("quantile") or 0.5)
                metrics.update(extract_metrics(test_result, threshold))
            except Exception as exc:
                metrics["evaluationStatus"] = "limited"
                metrics["evaluationMessage"] = str(exc)
        report = training_report(request, params, inventory, "completed", True, "Anomalib training completed.", metrics)
        if checkpoint:
            report["checkpointPath"] = str(checkpoint)
        report_path = output_path / "anomalib_training_report.json"
        write_json(report_path, report)
        sidecar_path = output_path / "anomaly_sidecar.json"
        write_json(sidecar_path, sidecar_payload(request, params, checkpoint, report_path, inventory))
        emit_event(backend, "artifact", kind="report", path=str(report_path), message="Anomalib training report")
        emit_event(backend, "artifact", kind="anomaly_sidecar", path=str(sidecar_path), message="Anomaly model sidecar")
        if checkpoint:
            emit_event(backend, "artifact", kind="checkpoint", path=str(checkpoint), message="Anomalib checkpoint")
        emit_event(backend, "metric", name="anomalySampleCount", value=float(inventory.get("anomalyCount") or 0), step=1, epoch=1)
        emit_event(backend, "completed", message="Anomalib training completed.", reportPath=str(report_path))
        return 0
    except FileNotFoundError as exc:
        code = str(exc) or "file_missing"
        report_path = write_failed_report(output_path, request, params, inventory, code, code, "blocked")
        emit_event(backend, "artifact", kind="report", path=str(report_path), message="Blocked Anomalib training report")
        emit_failed(backend, code, code, {"reportPath": str(report_path)})
        return 1
    except Exception as exc:
        message = f"{exc}\n{traceback.format_exc()}"
        error_code = "anomalib_missing" if "Required Python module is unavailable" in str(exc) else "anomalib_training_failed"
        report_path = write_failed_report(output_path, request, params, inventory, message, error_code)
        emit_event(backend, "artifact", kind="report", path=str(report_path), message="Failed Anomalib training report")
        emit_failed(backend, str(exc), error_code, {"reportPath": str(report_path)})
        return 1


def load_sidecar(model_path: Path) -> Dict[str, Any]:
    if model_path.name == "anomaly_sidecar.json":
        return read_json(model_path)
    sidecar = model_path.parent / "anomaly_sidecar.json"
    if sidecar.exists():
        return read_json(sidecar)
    return {"checkpointPath": str(model_path), "threshold": 0.5, "trainingBackend": BACKEND_PATCHCORE}


def save_prediction_images(image_path: Path, output_path: Path, result: Any) -> Tuple[str, str, str]:
    from PIL import Image, ImageEnhance

    output_path.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    heatmap_path = output_path / "anomaly_heatmap.png"
    overlay_path = output_path / "anomaly_overlay.png"
    mask_path = output_path / "anomaly_mask.png"

    heatmap = getattr(result, "heat_map", None)
    if heatmap is None:
        heatmap = getattr(result, "heatmap", None)
    mask = getattr(result, "pred_mask", None)
    if mask is None:
        mask = getattr(result, "mask", None)
    if mask is None:
        mask = getattr(result, "segmentations", None)
    if heatmap is not None:
        import numpy as np
        from PIL import ImageOps

        heat = np.asarray(heatmap)
        if heat.ndim == 3:
            heat = heat[..., 0]
        heat = heat.astype("float32")
        if heat.max() > heat.min():
            heat = (heat - heat.min()) / (heat.max() - heat.min())
        heat_img = Image.fromarray((heat * 255).clip(0, 255).astype("uint8")).resize(image.size)
        heat_rgb = ImageOps.colorize(heat_img, black="navy", white="red")
        heat_rgb.save(heatmap_path)
        Image.blend(image, heat_rgb, 0.45).save(overlay_path)
    else:
        image.save(heatmap_path)
        Image.blend(image, ImageEnhance.Color(image).enhance(0.3), 0.35).save(overlay_path)
    if mask is not None:
        import numpy as np

        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_img = Image.fromarray((mask_array > 0).astype("uint8") * 255).resize(image.size)
        mask_img.save(mask_path)
    else:
        Image.new("L", image.size, 0).save(mask_path)
    return str(heatmap_path), str(overlay_path), str(mask_path)


def infer(request: Dict[str, Any]) -> int:
    output_path = Path(str(request.get("outputPath") or ".")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(str(request.get("modelPath") or request.get("checkpointPath") or "")).resolve()
    image_path = Path(str(request.get("imagePath") or request.get("sampleImagePath") or "")).resolve()
    backend = str(request.get("backend") or BACKEND_PATCHCORE)
    try:
        sidecar = load_sidecar(model_path)
        backend = str(sidecar.get("trainingBackend") or backend).strip().lower()
        checkpoint = Path(str(sidecar.get("checkpointPath") or model_path)).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint_missing: {checkpoint}")
        if not image_path.exists():
            raise FileNotFoundError(f"sample_image_missing: {image_path}")
        _, _, engine_cls, patchcore_cls, efficientad_cls = import_anomalib_symbols()
        params = merge_preset(
            backend,
            dict(sidecar.get("parameters") or request.get("parameters") or {}),
        )
        requested_device = str((request.get("options") or {}).get("device") or params.get("device") or "auto")
        params["device"] = requested_device
        model = build_model(backend, params, patchcore_cls, efficientad_cls)
        engine = make_engine(engine_cls, output_path, params)
        started = time.perf_counter()
        predictions = engine.predict(
            model=model,
            data_path=str(image_path),
            ckpt_path=str(checkpoint),
            return_predictions=True,
        )
        result = first_prediction(predictions)
        if result is None:
            raise RuntimeError("Anomalib predict returned no predictions.")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        score = scalar_float(getattr(result, "pred_score", getattr(result, "anomaly_score", 0.0)), 0.0)
        threshold = float(sidecar.get("threshold") or (request.get("options") or {}).get("threshold") or 0.5)
        decision = "ng" if score > threshold else "ok"
        heatmap_path, overlay_path, mask_path = save_prediction_images(image_path, output_path, result)
        prediction = {
            "imagePath": str(image_path),
            "anomalyScore": score,
            "threshold": threshold,
            "decision": decision,
            "heatmapPath": heatmap_path,
            "overlayPath": overlay_path,
            "maskPath": mask_path,
        }
        report = {
            "schemaVersion": 1,
            "kind": "inference_predictions",
            "createdAt": now_iso(),
            "ok": True,
            "taskType": TASK_TYPE,
            "runtime": RUNTIME,
            "modelPath": str(model_path),
            "checkpointPath": str(checkpoint),
            "imagePath": str(image_path),
            "elapsedMs": int(elapsed_ms),
            "predictions": [prediction],
        }
        predictions_path = output_path / "inference_predictions.json"
        write_json(predictions_path, report)
        emit_event(backend, "artifact", kind="inference_predictions", path=str(predictions_path), message="Anomaly predictions")
        emit_event(backend, "artifact", kind="inference_overlay", path=overlay_path, message="Anomaly overlay")
        emit_event(backend, "completed", message="Anomalib inference completed.", predictionsPath=str(predictions_path), overlayPath=overlay_path)
        return 0
    except Exception as exc:
        failure_path = output_path / "inference_predictions.json"
        write_json(
            failure_path,
            {
                "schemaVersion": 1,
                "kind": "inference_predictions",
                "createdAt": now_iso(),
                "ok": False,
                "status": "failed",
                "taskType": TASK_TYPE,
                "runtime": RUNTIME,
                "modelPath": str(model_path),
                "imagePath": str(image_path),
                "errorCode": "anomalib_inference_failed",
                "message": str(exc),
                "predictions": [],
            },
        )
        emit_failed(backend, str(exc), "anomalib_inference_failed", {"predictionsPath": str(failure_path)})
        return 1


def evaluate(request: Dict[str, Any]) -> int:
    output_path = Path(str(request.get("outputPath") or ".")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(str(request.get("datasetPath") or "")).resolve()
    inventory = dataset_inventory(dataset_path)
    backend = str(request.get("backend") or BACKEND_PATCHCORE)
    params = merge_preset(backend, dict(request.get("parameters") or {}))
    model_path = Path(str(request.get("modelPath") or "")).resolve()
    threshold = float(params.get("threshold") or params.get("quantile") or 0.5)
    report = {
        "schemaVersion": 1,
        "kind": "evaluation_report",
        "createdAt": now_iso(),
        "ok": False,
        "status": "limited" if inventory.get("evaluationLimited") else "blocked",
        "taskType": TASK_TYPE,
        "datasetFormat": DATASET_FORMAT,
        "runtime": RUNTIME,
        "modelPath": request.get("modelPath", ""),
        "datasetPath": str(dataset_path),
        "datasetInventory": inventory,
        "metrics": {},
        "sampleCount": inventory.get("anomalyCount", 0) + inventory.get("normalCount", 0),
        "threshold": threshold,
        "scaffold": False,
    }
    if inventory.get("evaluationLimited"):
        report["message"] = "Only normal samples were found; image-level AUROC/F1 are limited until anomaly samples are provided."
        report["limitations"] = ["good-only dataset"]
    else:
        try:
            sidecar = load_sidecar(model_path)
            backend = str(sidecar.get("trainingBackend") or backend)
            params = merge_preset(backend, dict(sidecar.get("parameters") or params))
            datamodule_params = dict(params)
            datamodule_params["_aitrainDatasetViewRoot"] = str(output_path / "_aitrain_dataset_views")
            threshold = float(sidecar.get("threshold") or params.get("threshold") or params.get("quantile") or 0.5)
            checkpoint = Path(str(sidecar.get("checkpointPath") or model_path)).resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(f"checkpoint_missing: {checkpoint}")
            _, folder_cls, engine_cls, patchcore_cls, efficientad_cls = import_anomalib_symbols()
            datamodule = build_datamodule(folder_cls, dataset_path, datamodule_params)
            model = build_model(backend, params, patchcore_cls, efficientad_cls)
            engine = make_engine(engine_cls, output_path, params)
            raw_result = engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint))
            report["ok"] = True
            report["status"] = "completed"
            report["trainingBackend"] = backend
            report["checkpointPath"] = str(checkpoint)
            report["threshold"] = threshold
            report["metrics"] = extract_metrics(raw_result, threshold)
            report["message"] = "Anomalib evaluation completed."
            if not inventory.get("pixelEvaluationAvailable"):
                report["limitations"] = ["No masks were found; pixel metrics are unavailable."]
        except Exception as exc:
            report["status"] = "blocked"
            report["message"] = str(exc)
            report["errorCode"] = "anomalib_evaluation_blocked"
            report["failureCategory"] = "anomalib_evaluation_requires_runtime"
    report_path = output_path / "evaluation_report.json"
    write_json(report_path, report)
    emit_event(backend, "artifact", kind="evaluation_report", path=str(report_path), message="Anomaly evaluation report")
    return 0


def benchmark(request: Dict[str, Any]) -> int:
    output_path = Path(str(request.get("outputPath") or ".")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    options = dict(request.get("options") or {})
    warmup = max(0, int(options.get("warmupIterations", 2)))
    iterations = max(1, int(options.get("iterations", 10)))
    timings: List[float] = []
    status = "completed"
    message = ""
    for index in range(warmup + iterations):
        started = time.perf_counter()
        code = infer({**request, "outputPath": str(output_path / f"run_{index:03d}")})
        elapsed = (time.perf_counter() - started) * 1000.0
        if code != 0:
            status = "blocked"
            message = "Anomalib benchmark requires successful Python infer runs."
            break
        if index >= warmup:
            timings.append(elapsed)
    avg = statistics.mean(timings) if timings else 0.0
    sorted_timings = sorted(timings)

    def percentile(pct: float) -> float:
        if not sorted_timings:
            return 0.0
        index = int(round((pct / 100.0) * (len(sorted_timings) - 1)))
        return sorted_timings[max(0, min(index, len(sorted_timings) - 1))]

    report = {
        "schemaVersion": 1,
        "kind": "benchmark_report",
        "createdAt": now_iso(),
        "ok": status == "completed",
        "status": status,
        "modelFamily": "anomaly_detection",
        "runtime": RUNTIME,
        "runtimeUsable": status == "completed",
        "timedInference": bool(timings),
        "warmupIterations": warmup,
        "iterations": iterations,
        "averageMs": avg,
        "p50Ms": percentile(50),
        "p95Ms": percentile(95),
        "p99Ms": percentile(99),
        "throughput": 1000.0 / avg if avg > 0 else 0.0,
        "message": message,
        "scaffold": False,
        "deploymentConclusion": "local-runtime-available" if status == "completed" else "anomalib-python-blocked",
    }
    report_path = output_path / "benchmark_report.json"
    write_json(report_path, report)
    return 0 if status == "completed" else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--mode", choices=["train", "evaluate", "infer", "benchmark"], default="")
    args = parser.parse_args(argv)
    request = read_json(Path(args.request))
    mode = args.mode or str(request.get("mode") or "train")
    try:
        if mode == "train":
            return train(request)
        if mode == "evaluate":
            return evaluate(request)
        if mode == "infer":
            return infer(request)
        if mode == "benchmark":
            return benchmark(request)
        raise ValueError(f"Unsupported mode: {mode}")
    except Exception:
        backend = str(request.get("backend") or BACKEND_PATCHCORE)
        unhandled_failure(backend, sys.exc_info()[1])
        return 1


if __name__ == "__main__":
    sys.exit(main())
