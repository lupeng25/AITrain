#!/usr/bin/env python3
"""SMP semantic segmentation trainer adapter for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from adapter_event_channel import AdapterEventChannel, event_channel_from_environment  # noqa: E402
from adapter_sdk import AdapterCanceled, AdapterSdk  # noqa: E402
from dataset_snapshot import materialize_dataset_snapshot  # noqa: E402
from trainer_protocol import configure_stdio, exception_details  # noqa: E402


BACKEND_ID = "smp_semantic_segmentation"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PRESETS: dict[str, dict[str, Any]] = {
    "smp_unet_resnet34": {"arch": "Unet", "encoder_candidates": ["resnet34"]},
    "smp_unetplusplus_resnet34": {"arch": "UnetPlusPlus", "encoder_candidates": ["resnet34"]},
    "smp_fpn_resnet34": {"arch": "FPN", "encoder_candidates": ["resnet34"]},
    "smp_deeplabv3plus_resnet50": {"arch": "DeepLabV3Plus", "encoder_candidates": ["resnet50"]},
    "smp_segformer_mit_b0": {"arch": "Segformer", "encoder_candidates": ["mit_b0", "tu-mit_b0"]},
}

configure_stdio()

_adapter: AdapterSdk | None = None
_event_channel: AdapterEventChannel | None = None


def configure_adapter() -> None:
    """Use authenticated  events when the Worker Host provides a channel."""
    global _adapter, _event_channel
    if _event_channel is None and os.environ.get("AITRAIN_EVENT_PORT"):
        _event_channel = event_channel_from_environment()
        _event_channel.connect()
    if _adapter is None:
        sink = _event_channel.emit_event if _event_channel is not None else None
        _adapter = AdapterSdk(BACKEND_ID, event_sink=sink)


def close_adapter() -> None:
    global _event_channel
    if _event_channel is not None:
        _event_channel.close()
        _event_channel = None


def active_adapter() -> AdapterSdk:
    configure_adapter()
    assert _adapter is not None
    return _adapter


def check_canceled() -> None:
    active_adapter().raise_if_canceled()


def emit(event_type: str, **payload: Any) -> None:
    payload.pop("backend", None)
    adapter = active_adapter()
    if event_type == "log":
        adapter.emit_log(str(payload.pop("message", "")), level=str(payload.pop("level", "info")), **payload)
    elif event_type == "progress":
        adapter.emit_progress(float(payload.pop("percent", 0)), message=str(payload.pop("message", "")), **payload)
    elif event_type == "metric":
        adapter.emit_metric(str(payload.pop("name", "")), float(payload.pop("value", 0)), **payload)
    elif event_type == "artifact":
        adapter.emit_artifact_candidate(
            str(payload.pop("kind", "artifact")),
            str(payload.pop("path", "")),
            message=str(payload.pop("message", "")),
            **payload,
        )
    elif event_type == "completed":
        adapter.emit_completed(str(payload.pop("message", "SMP training completed")), **payload)
    elif event_type == "failed":
        adapter.emit_failed(
            str(payload.pop("message", "SMP training failed")),
            str(payload.pop("code", "smp_trainer_failed")),
            payload.pop("details", {}),
        )
    else:
        raise ValueError(f"unsupported adapter event type: {event_type}")


def fail(message: str, code: str = "smp_trainer_failed", details: dict[str, Any] | None = None) -> int:
    return active_adapter().emit_failed(message, code, details)


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trainer request must be a JSON object")
    return value


def materialize_request_dataset(request: dict[str, Any], params: dict[str, Any], dataset_path: Path) -> tuple[Path, str]:
    snapshot_manifest = str(params.get("datasetSnapshotManifest") or request.get("datasetSnapshotManifest") or "").strip()
    snapshot_staging = str(params.get("datasetSnapshotStagingPath") or request.get("datasetSnapshotStagingPath") or "").strip()
    if bool(snapshot_manifest) != bool(snapshot_staging):
        raise ValueError("Dataset Snapshot  requires both manifest and staging paths.")
    if snapshot_manifest:
        dataset_path = materialize_dataset_snapshot(dataset_path, snapshot_manifest, snapshot_staging)
    return dataset_path, snapshot_manifest


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def as_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def as_float(value: Any, default: float, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return to_jsonable(value.item())
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            return to_jsonable(value.detach().cpu().tolist())
    except Exception:
        pass
    return str(value)


def read_classes(dataset_path: Path) -> list[str]:
    classes_path = dataset_path / "classes.txt"
    if not classes_path.exists():
        raise ValueError(f"classes.txt is required for semantic segmentation mask datasets: {classes_path}")
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not classes:
        raise ValueError("classes.txt must contain at least one class name")
    return classes


def load_mask_ids(mask_path: Path, target_size: tuple[int, int] | None = None) -> Any:
    """Load raw semantic class IDs without converting palette colors to grayscale."""
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    mask = Image.open(mask_path)
    if mask.mode not in {"L", "P"}:
        raise ValueError(f"Semantic mask must use L or P mode, got {mask.mode}: {mask_path}")
    if target_size is not None:
        mask = mask.resize(target_size, Image.NEAREST)
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Semantic mask must be single-channel: {mask_path}")
    return values.astype(np.int64, copy=False)


class SemanticMaskDataset:
    def __init__(self, samples: list[tuple[Path, Path]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(load_mask_ids(mask_path, (self.image_size, self.image_size))).long()
        return image_tensor, mask_tensor, image_path.name


def split_samples(dataset_path: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_path / "images" / split
    mask_dir = dataset_path / "masks" / split
    if not image_dir.exists() or not mask_dir.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(item for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in suffixes):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            samples.append((image_path, mask_path))
    return samples


def resolve_preset(preset_id: str, smp: Any) -> tuple[str, str]:
    if preset_id not in PRESETS:
        raise ValueError(f"Unsupported SMP preset '{preset_id}'. Supported presets: {', '.join(PRESETS)}")
    preset = PRESETS[preset_id]
    available = set(smp.encoders.get_encoder_names())
    for encoder in preset["encoder_candidates"]:
        if encoder in available:
            return str(preset["arch"]), encoder
    raise ValueError(
        f"SMP preset '{preset_id}' has no available encoder among {preset['encoder_candidates']}. "
        "Check segmentation-models-pytorch/timm installation."
    )


def build_model(preset_id: str, class_count: int, encoder_weights: str | None) -> tuple[Any, str, str]:
    import segmentation_models_pytorch as smp  # type: ignore

    arch, encoder = resolve_preset(preset_id, smp)
    weights = None if not encoder_weights or encoder_weights.lower() == "none" else encoder_weights
    model_class = getattr(smp, arch, None)
    if model_class is None:
        raise ValueError(f"SMP architecture '{arch}' is unavailable in the installed segmentation-models-pytorch package")
    try:
        model = model_class(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=class_count)
    except TypeError:
        model = model_class(encoder_name=encoder, encoder_weights=weights, classes=class_count)
    return model, arch, encoder


def normalize_device_name(raw_device: Any, torch_module: Any) -> str:
    value = str(raw_device or "cpu").strip().lower()
    if not value or value == "cpu":
        return "cpu"
    if value.isdigit():
        return f"cuda:{value}"
    if value in {"gpu", "cuda"}:
        return "cuda:0"
    if value.startswith("cuda"):
        return value
    if value.startswith("gpu:") and value.split(":", 1)[1].isdigit():
        return f"cuda:{value.split(':', 1)[1]}"
    if not torch_module.cuda.is_available():
        return "cpu"
    return value


def load_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name, package_name in (
        ("segmentation_models_pytorch", "segmentation_models_pytorch"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
    ):
        try:
            module = __import__(module_name)
            versions[package_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[package_name] = "missing"
    return versions


def make_dataset_class() -> type:
    return SemanticMaskDataset


def dice_ce_loss(logits: Any, targets: Any, ignore_index: int) -> Any:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    ce = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    class_count = logits.shape[1]
    valid = targets != ignore_index
    safe_targets = targets.clone()
    safe_targets[~valid] = 0
    target_onehot = F.one_hot(safe_targets, num_classes=class_count).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    probs = torch.softmax(logits, dim=1) * valid_float
    target_onehot = target_onehot * valid_float
    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dims)
    cardinality = probs.sum(dims) + target_onehot.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    return ce + (1.0 - dice.mean())


def metrics_from_confusion(confusion: Any) -> dict[str, Any]:
    import numpy as np  # type: ignore

    cm = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    dice = np.divide(2 * tp, denom_dice, out=np.zeros_like(tp), where=denom_dice > 0)
    total = cm.sum()
    return {
        "mIoU": float(iou.mean()) if iou.size else 0.0,
        "meanDice": float(dice.mean()) if dice.size else 0.0,
        "pixelAccuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "perClassIoU": iou.tolist(),
        "perClassDice": dice.tolist(),
    }


def update_confusion(confusion: Any, targets: Any, predictions: Any, class_count: int, ignore_index: int) -> Any:
    import torch  # type: ignore

    valid = targets != ignore_index
    if valid.sum().item() == 0:
        return confusion
    encoded = targets[valid].view(-1) * class_count + predictions[valid].view(-1)
    counts = torch.bincount(encoded, minlength=class_count * class_count)
    return confusion + counts.reshape(class_count, class_count)


def run_training(request: dict[str, Any]) -> int:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    task_id = str(request.get("taskId") or "smp-semantic-segmentation")
    dataset_path = Path(str(request.get("datasetPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or "")).resolve()
    params = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    output_path.mkdir(parents=True, exist_ok=True)
    check_canceled()

    try:
        dataset_path, snapshot_manifest = materialize_request_dataset(request, params, dataset_path)
    except Exception as exc:
        code = "dataset_snapshot_request_invalid" if "requires both" in str(exc) else "dataset_snapshot_invalid"
        return fail("Dataset Snapshot  materialization failed.", code, exception_details(exc))

    if not dataset_path.exists():
        return fail(f"Dataset path does not exist: {dataset_path}", "dataset_missing")
    class_names = read_classes(dataset_path)
    class_count = len(class_names)
    train_samples = split_samples(dataset_path, "train")
    val_samples = split_samples(dataset_path, "val")
    if not train_samples or not val_samples:
        return fail("SMP training requires images/train, masks/train, images/val, and masks/val samples.", "semantic_dataset_empty")

    seed = as_int(params.get("seed"), 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    epochs = as_int(params.get("epochs"), 10, 1)
    batch_size = as_int(params.get("batchSize"), 4, 1)
    image_size = as_int(params.get("imageSize"), 256, 32)
    workers = as_int(params.get("workers"), 0, 0)
    learning_rate = as_float(params.get("learningRate"), 1e-3, 1e-8)
    optimizer_name = str(params.get("optimizer") or "adamw").lower()
    loss_name = str(params.get("loss") or "dice_ce").lower()
    ignore_index = as_int(params.get("ignoreIndex"), 255, 0)
    requested_device_name = str(params.get("device") or "cpu")
    device_name = normalize_device_name(requested_device_name, torch)
    if device_name != "cpu" and not torch.cuda.is_available():
        emit("log", taskId=task_id, backend=BACKEND_ID, message=f"Requested device '{requested_device_name}' is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    preset_id = str(params.get("modelPreset") or params.get("model") or params.get("preset") or "smp_unet_resnet34")
    encoder_weights = str(params.get("encoderWeights") or "none")
    model, arch, encoder = build_model(preset_id, class_count, encoder_weights)
    model.to(device)

    DatasetCls = make_dataset_class()
    train_loader = DataLoader(DatasetCls(train_samples, image_size), batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(DatasetCls(val_samples, image_size), batch_size=batch_size, shuffle=False, num_workers=workers)

    if optimizer_name != "adamw":
        return fail("SMP first release supports optimizer=adamw only.", "unsupported_optimizer", {"optimizer": optimizer_name})
    if loss_name != "dice_ce":
        return fail("SMP first release supports loss=dice_ce only.", "unsupported_loss", {"loss": loss_name})
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_miou = -1.0
    best_checkpoint_path = output_path / "best.pt"
    best_onnx_path = output_path / "best.onnx"
    last_metrics: dict[str, Any] = {}

    emit("log", taskId=task_id, backend=BACKEND_ID, message=f"Starting SMP training preset={preset_id} encoder={encoder} classes={class_count}")
    for epoch in range(1, epochs + 1):
        check_canceled()
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, masks, _names in train_loader:
            check_canceled()
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = dice_ce_loss(logits, masks, ignore_index)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item()) * images.shape[0]
            sample_count += int(images.shape[0])

        model.eval()
        confusion = torch.zeros((class_count, class_count), dtype=torch.int64, device="cpu")
        with torch.no_grad():
            for images, masks, _names in val_loader:
                check_canceled()
                images = images.to(device)
                logits = model(images).detach().cpu()
                predictions = logits.argmax(dim=1)
                confusion = update_confusion(confusion, masks.cpu(), predictions, class_count, ignore_index)
        last_metrics = metrics_from_confusion(confusion.numpy())
        train_loss = running_loss / max(1, sample_count)
        last_metrics["loss"] = train_loss
        last_metrics["epoch"] = epoch
        emit("metric", taskId=task_id, backend=BACKEND_ID, name="loss", value=train_loss, epoch=epoch, step=epoch)
        emit("metric", taskId=task_id, backend=BACKEND_ID, name="mIoU", value=last_metrics["mIoU"], epoch=epoch, step=epoch)
        emit("metric", taskId=task_id, backend=BACKEND_ID, name="meanDice", value=last_metrics["meanDice"], epoch=epoch, step=epoch)
        emit("progress", taskId=task_id, backend=BACKEND_ID, percent=round(100.0 * epoch / epochs), epoch=epoch, step=epoch)

        if float(last_metrics["mIoU"]) >= best_miou:
            best_miou = float(last_metrics["mIoU"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "preset": preset_id,
                    "arch": arch,
                    "encoder": encoder,
                    "encoderWeights": encoder_weights,
                    "classNames": class_names,
                    "ignoreIndex": ignore_index,
                    "imageSize": image_size,
                    "mean": MEAN,
                    "std": STD,
                    "metrics": last_metrics,
                    "backend": BACKEND_ID,
                },
                best_checkpoint_path,
            )

    try:
        check_canceled()
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        best_onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=13,
    )

    sidecar = {
        "schemaVersion": 1,
        "backend": BACKEND_ID,
        "modelFamily": "semantic_segmentation",
        "taskType": "semantic_segmentation",
        "datasetFormat": "semantic_segmentation_mask",
        "preset": preset_id,
        "architecture": arch,
        "encoder": encoder,
        "encoderWeights": encoder_weights,
        "classNames": class_names,
        "classCount": class_count,
        "ignoreIndex": ignore_index,
        "inputWidth": image_size,
        "inputHeight": image_size,
        "normalization": {"mean": MEAN, "std": STD, "scale": 1.0 / 255.0},
        "decoder": "argmax_semantic_mask",
        "sourceCheckpoint": str(best_checkpoint_path),
        "exportPath": str(best_onnx_path),
    }
    sidecar_path = output_path / "semantic_segmentation_sidecar.json"
    export_sidecar_path = best_onnx_path.with_suffix(".aitrain-export.json")
    write_json(sidecar_path, sidecar)
    write_json(export_sidecar_path, sidecar)

    versions = load_versions()
    report = {
        "ok": True,
        "kind": "smp_training_report",
        "createdAt": now_iso(),
        "backend": BACKEND_ID,
        "modelFamily": "semantic_segmentation",
        "taskType": "semantic_segmentation",
        "datasetFormat": "semantic_segmentation_mask",
        "datasetPath": str(dataset_path),
        "datasetSnapshotId": str(params.get("datasetSnapshotId") or request.get("datasetSnapshotId") or ""),
        "datasetSnapshotHash": str(params.get("datasetSnapshotHash") or request.get("datasetSnapshotHash") or ""),
        "datasetSnapshotManifest": snapshot_manifest,
        "checkpointPath": str(best_checkpoint_path),
        "onnxPath": str(best_onnx_path),
        "sidecarPath": str(sidecar_path),
        "preset": preset_id,
        "architecture": arch,
        "encoder": encoder,
        "classNames": class_names,
        "ignoreIndex": ignore_index,
        "normalization": {"mean": MEAN, "std": STD, "scale": 1.0 / 255.0},
        "seed": seed,
        "epochs": epochs,
        "batchSize": batch_size,
        "imageSize": image_size,
        "device": device_name,
        "workers": workers,
        "learningRate": learning_rate,
        "optimizer": optimizer_name,
        "loss": loss_name,
        "metrics": last_metrics,
        "versions": versions,
        "licenseNote": "segmentation-models-pytorch, torch, timm, onnx, and onnxruntime licenses must be reviewed before commercial redistribution.",
        "scaffold": False,
    }
    report_path = output_path / "smp_training_report.json"
    write_json(report_path, report)

    artifacts = [
        {"kind": "checkpoint", "path": str(best_checkpoint_path), "message": "SMP best PyTorch checkpoint"},
        {"kind": "onnx_model", "path": str(best_onnx_path), "message": "SMP semantic segmentation ONNX"},
        {"kind": "training_report", "path": str(report_path), "message": "SMP training report"},
        {"kind": "model_sidecar", "path": str(sidecar_path), "message": "SMP semantic segmentation sidecar"},
    ]
    for artifact in artifacts:
        emit("artifact", taskId=task_id, backend=BACKEND_ID, **artifact)
    emit(
        "completed",
        taskId=task_id,
        backend=BACKEND_ID,
        message="SMP semantic segmentation training completed",
        checkpointPath=str(best_checkpoint_path),
        onnxPath=str(best_onnx_path),
        reportPath=str(report_path),
        artifacts=artifacts,
        metrics=last_metrics,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="Path to AITrain Python trainer request JSON")
    args = parser.parse_args(argv)
    try:
        configure_adapter()
        try:
            request = read_request(Path(args.request))
        except Exception as exc:
            return fail(f"failed to read training request: {exc}", "bad_request", exception_details(exc))
        try:
            return run_training(request)
        except Exception as exc:
            if isinstance(exc, AdapterCanceled):
                active_adapter().emit_canceled("SMP training canceled by request")
                return 2
            return fail("Unhandled SMP training failure.", "unhandled_exception", exception_details(exc))
    finally:
        close_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
