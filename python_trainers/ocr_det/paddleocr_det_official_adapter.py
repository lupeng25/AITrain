#!/usr/bin/env python3
"""Official PaddleOCR detection trainer adapter for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from trainer_protocol import configure_stdio, emit_failed, exception_details, unhandled_failure  # noqa: E402


BACKEND_ID = "paddleocr_det_official"
DEFAULT_MODEL_PRESET = "PP-OCRv5_mobile_det"
DET_PRESETS: dict[str, dict[str, Any]] = {
    "PP-OCRv4_mobile_det": {
        "ocrVersion": "PP-OCRv4",
        "config": "configs/det/PP-OCRv4/PP-OCRv4_mobile_det.yml",
        "outputConfigName": "aitrain_ppocrv4_det.yml",
        "modelName": "PP-OCRv4_mobile_det",
        "requiresRepo": False,
    },
    "PP-OCRv5_mobile_det": {
        "ocrVersion": "PP-OCRv5",
        "config": "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml",
        "outputConfigName": "aitrain_ppocrv5_mobile_det.yml",
        "modelName": "PP-OCRv5_mobile_det",
        "requiresRepo": True,
    },
    "PP-OCRv5_server_det": {
        "ocrVersion": "PP-OCRv5",
        "config": "configs/det/PP-OCRv5/PP-OCRv5_server_det.yml",
        "outputConfigName": "aitrain_ppocrv5_server_det.yml",
        "modelName": "PP-OCRv5_server_det",
        "requiresRepo": True,
    },
    "PP-OCRv6_tiny_det": {
        "ocrVersion": "PP-OCRv6",
        "config": "configs/det/PP-OCRv6/PP-OCRv6_tiny_det.yml",
        "outputConfigName": "aitrain_ppocrv6_tiny_det.yml",
        "modelName": "PP-OCRv6_tiny_det",
        "requiresRepo": True,
    },
    "PP-OCRv6_small_det": {
        "ocrVersion": "PP-OCRv6",
        "config": "configs/det/PP-OCRv6/PP-OCRv6_small_det.yml",
        "outputConfigName": "aitrain_ppocrv6_small_det.yml",
        "modelName": "PP-OCRv6_small_det",
        "requiresRepo": True,
    },
    "PP-OCRv6_medium_det": {
        "ocrVersion": "PP-OCRv6",
        "config": "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml",
        "outputConfigName": "aitrain_ppocrv6_medium_det.yml",
        "modelName": "PP-OCRv6_medium_det",
        "requiresRepo": True,
    },
}

configure_stdio()


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time(), "backend": BACKEND_ID}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def fail(message: str, code: str, details: dict[str, Any] | None = None) -> int:
    return emit_failed(BACKEND_ID, message, code, details)


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trainer request must be a JSON object")
    return value


def bool_param(parameters: dict[str, Any], key: str, default: bool) -> bool:
    value = parameters.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def int_param(parameters: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(parameters.get(key, default)))
    except (TypeError, ValueError):
        return max(minimum, default)


def official_log_options(parameters: dict[str, Any]) -> tuple[str, int, int]:
    verbosity = str(parameters.get("officialLogVerbosity") or "summary").strip().lower()
    if verbosity not in {"summary", "full", "quiet"}:
        verbosity = "summary"
    interval_seconds = int_param(parameters, "officialLogEventIntervalSeconds", 30, 1)
    tail_lines = int_param(parameters, "officialLogTailLines", 200, 1)
    return verbosity, interval_seconds, tail_lines


def is_important_official_line(line: str) -> bool:
    lower = line.lower()
    important_tokens = ("traceback", "error", "exception", "failed", "warning", "fatal")
    return any(token in lower for token in important_tokens)


def resolve_preset(parameters: dict[str, Any]) -> dict[str, Any]:
    requested = str(parameters.get("modelPreset") or parameters.get("model") or DEFAULT_MODEL_PRESET).strip()
    if not requested:
        requested = DEFAULT_MODEL_PRESET
    preset = dict(DET_PRESETS.get(requested, DET_PRESETS[DEFAULT_MODEL_PRESET]))
    preset["modelPreset"] = requested if requested in DET_PRESETS else DEFAULT_MODEL_PRESET
    official_config = str(parameters.get("officialConfig") or "").strip().replace("\\", "/")
    if official_config:
        preset["config"] = official_config
        preset["configSource"] = "officialConfig_override"
        preset["requiresRepo"] = True
    else:
        preset["configSource"] = "builtin_preset"
    return preset


def config_model_name(config: dict[str, Any], fallback: str) -> str:
    global_config = config.get("Global") if isinstance(config, dict) else {}
    if isinstance(global_config, dict):
        value = str(global_config.get("model_name") or "").strip()
        if value:
            return value
    return fallback


def resolve_repo_file(repo: Path, relative: str, description: str) -> tuple[Path, str]:
    normalized = relative.replace("\\", "/")
    candidates = [normalized]
    if normalized.endswith(".yml"):
        candidates.append(normalized[:-4] + ".yaml")
    elif normalized.endswith(".yaml"):
        candidates.append(normalized[:-5] + ".yml")
    for candidate in candidates:
        path = (repo / candidate).resolve()
        if path.exists():
            return path, candidate
    tried = ", ".join(str((repo / candidate).resolve()) for candidate in candidates)
    raise FileNotFoundError(f"{description} not found. Tried: {tried}")


def resolve_dataset_file(dataset_path: Path, value: Any, default_name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        return dataset_path / default_name
    path = Path(text)
    return path if path.is_absolute() else dataset_path / path


def find_repo(parameters: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    for key in ("paddleOcrRepoPath", "paddleOCRRepoPath", "officialRepoPath"):
        value = str(parameters.get(key) or "").strip()
        if value:
            candidates.append(Path(value))
    env_value = os.environ.get("AITRAIN_PADDLEOCR_REPO", "").strip()
    if env_value:
        candidates.append(Path(env_value))
    cwd = Path.cwd()
    script = Path(__file__).resolve()
    candidates.extend(
        [
            cwd / ".deps" / "PaddleOCR",
            script.parents[3] / ".deps" / "PaddleOCR" if len(script.parents) > 3 else script.parent,
            script.parents[2] / ".deps" / "PaddleOCR" if len(script.parents) > 2 else script.parent,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "tools" / "train.py").exists() and (resolved / "tools" / "export_model.py").exists():
            return resolved
    return None


def read_det_labels(path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid PaddleOCR Det label line {line_number}: {line}")
        json.loads(parts[1])
        samples.append((parts[0].strip().replace("\\", "/"), parts[1].strip()))
    if not samples:
        raise ValueError(f"empty PaddleOCR Det label file: {path}")
    return samples


def split_samples(samples: list[tuple[str, str]], validation_ratio: float) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if len(samples) == 1:
        return samples, samples
    ratio = min(0.5, max(0.0, validation_ratio))
    val_count = max(1, int(round(len(samples) * ratio))) if ratio > 0 else 1
    val_count = min(val_count, len(samples) - 1)
    return samples[:-val_count], samples[-val_count:]


def write_label_file(path: Path, samples: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{image}\t{label}\n" for image, label in samples), encoding="utf-8")


def yaml_dump(value: Any) -> str:
    import yaml

    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


def simplify_tiny_det_transforms(transforms: Any, epochs: int, image_size: int, training: bool) -> Any:
    if not isinstance(transforms, list):
        return transforms
    simplified: list[Any] = []
    unstable_train_transforms = {"IaaAugment", "CopyPaste"}
    for transform in transforms:
        if isinstance(transform, dict):
            if training and any(name in transform for name in unstable_train_transforms):
                continue
            if training and "EastRandomCropData" in transform and isinstance(transform["EastRandomCropData"], dict):
                transform["EastRandomCropData"]["size"] = [image_size, image_size]
                transform["EastRandomCropData"]["keep_ratio"] = True
            if "MakeBorderMap" in transform and isinstance(transform["MakeBorderMap"], dict):
                transform["MakeBorderMap"]["total_epoch"] = epochs
            if "MakeShrinkMap" in transform and isinstance(transform["MakeShrinkMap"], dict):
                transform["MakeShrinkMap"]["total_epoch"] = epochs
            if "DetResizeForTest" in transform and isinstance(transform["DetResizeForTest"], dict):
                transform["DetResizeForTest"]["image_shape"] = [image_size, image_size]
        simplified.append(transform)
    return simplified


def checkpoint_base_exists(path: Path) -> bool:
    return path.with_suffix(".pdparams").exists()


def select_checkpoint_base(output_path: Path, explicit_base: Any) -> Path:
    if explicit_base:
        return Path(str(explicit_base))
    model_dir = output_path / "official_model"
    for name in ("best_accuracy", "latest", "iter_epoch_1"):
        candidate = model_dir / name
        if checkpoint_base_exists(candidate):
            return candidate
    epoch_candidates = []
    for candidate in model_dir.glob("iter_epoch_*.pdparams"):
        match = re.search(r"iter_epoch_(\d+)\.pdparams$", candidate.name)
        epoch = int(match.group(1)) if match else -1
        epoch_candidates.append((epoch, candidate))
    for _, candidate in sorted(epoch_candidates, key=lambda item: item[0], reverse=True):
        return candidate.with_suffix("")
    return model_dir / "best_accuracy"


def build_config(
    repo: Path | None,
    parameters: dict[str, Any],
    preset: dict[str, Any],
    dataset_path: Path,
    output_path: Path,
    train_list_path: Path,
    val_list_path: Path,
    first_image: str,
) -> dict[str, Any]:
    import yaml

    template_relative = str(preset["config"]).replace("\\", "/")
    if repo:
        template_path, resolved_relative = resolve_repo_file(repo, template_relative, "PaddleOCR det config template")
        preset["resolvedConfig"] = resolved_relative
        with template_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        config = {
            "Global": {},
            "Architecture": {"model_type": "det", "algorithm": "DB"},
            "Loss": {"name": "DBLoss"},
            "Optimizer": {"name": "Adam", "lr": {"name": "Cosine", "learning_rate": 0.001}},
            "PostProcess": {"name": "DBPostProcess"},
            "Metric": {"name": "DetMetric", "main_indicator": "hmean"},
            "Train": {"dataset": {}, "loader": {}},
            "Eval": {"dataset": {}, "loader": {}},
        }

    epochs = max(1, int(parameters.get("epochs", 1)))
    batch_size = max(1, int(parameters.get("batchSize", 1)))
    image_size = max(32, int(parameters.get("imageSize", parameters.get("detImageSize", 640))))
    eval_every_steps = max(1, int(parameters.get("evalEverySteps", 1000000)))
    print_batch_step = int_param(parameters, "officialPrintBatchStep", 20, 1)
    save_epoch_step = min(epochs, int_param(parameters, "officialSaveEpochStep", 10, 1))
    save_model_dir = output_path / "official_model"
    save_inference_dir = output_path / "official_inference"

    global_config = config.setdefault("Global", {})
    global_config.update(
        {
            "use_gpu": bool_param(parameters, "useGpu", False),
            "epoch_num": epochs,
            "print_batch_step": print_batch_step,
            "save_model_dir": str(save_model_dir),
            "save_epoch_step": save_epoch_step,
            "eval_batch_step": [0, eval_every_steps],
            "pretrained_model": parameters.get("pretrainedModel") or None,
            "checkpoints": parameters.get("resumeCheckpoint") or None,
            "save_inference_dir": str(save_inference_dir),
            "infer_img": str((dataset_path / first_image).resolve()),
            "distributed": False,
            "d2s_train_image_shape": [3, image_size, image_size],
            "cal_metric_during_train": bool_param(parameters, "calMetricDuringTrain", False),
            "export_with_pir": bool_param(parameters, "exportWithPir", True),
        }
    )

    for section, label_path, shuffle in (
        ("Train", train_list_path, True),
        ("Eval", val_list_path, False),
    ):
        section_config = config.setdefault(section, {})
        dataset_config = section_config.setdefault("dataset", {})
        dataset_config["data_dir"] = str(dataset_path)
        dataset_config["label_file_list"] = [str(label_path)]
        loader_config = section_config.setdefault("loader", {})
        loader_config.update({"batch_size_per_card": batch_size, "drop_last": False, "num_workers": 0, "shuffle": shuffle})
        dataset_config["transforms"] = simplify_tiny_det_transforms(
            dataset_config.get("transforms", []), epochs, image_size, section == "Train"
        )

    return config


def write_command_file(path: Path, command: list[str], cwd: Path | None) -> None:
    quoted = " ".join(f'"{part}"' if " " in part else part for part in command)
    lines = []
    if cwd is not None:
        lines.append(f'Set-Location "{cwd}"')
    lines.append(quoted)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_metrics(line: str, metrics: dict[str, float]) -> None:
    name_map = {"hmean": "hmean", "precision": "precision", "recall": "recall", "loss": "loss"}
    for raw_name, raw_value in re.findall(r"\b(hmean|precision|recall|loss):\s*([-+0-9.eE]+)", line):
        try:
            value = float(raw_value)
        except ValueError:
            continue
        name = name_map.get(raw_name, raw_name)
        metrics[name] = value
        emit("metric", name=name, value=value)


def run_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    metrics: dict[str, float] | None,
    log_path: Path,
    parameters: dict[str, Any],
) -> int:
    emit("log", level="info", message=f"Running official PaddleOCR command: {' '.join(command)}")
    verbosity, interval_seconds, tail_lines = official_log_options(parameters)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=tail_lines)
    line_count = 0
    last_event_at = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            stripped = line.rstrip()
            if not stripped:
                continue
            line_count += 1
            log_file.write(stripped + "\n")
            log_file.flush()
            tail.append(stripped)
            if metrics is not None:
                parse_metrics(stripped, metrics)
            now = time.monotonic()
            if verbosity == "full" or is_important_official_line(stripped):
                emit("log", level="info", message=stripped[:2000])
                last_event_at = now
            elif verbosity == "summary" and now - last_event_at >= interval_seconds:
                emit("log", level="info", message=f"Official PaddleOCR command still running; lines={line_count}; latest={stripped[:500]}")
                last_event_at = now
    exit_code = process.wait()
    if verbosity != "full":
        emit(
            "log",
            level="info" if exit_code == 0 else "error",
            message=f"Official PaddleOCR command finished with exitCode={exit_code}; logPath={log_path}; tailLines={len(tail)}",
        )
    return exit_code


def prune_intermediate_checkpoints(output_path: Path, parameters: dict[str, Any]) -> int:
    retention = str(parameters.get("checkpointRetention") or "").strip().lower()
    if retention not in {"latest_best_inference", "latest-best-inference"}:
        return 0
    model_dir = output_path / "official_model"
    if not model_dir.exists():
        return 0
    removed = 0
    for path in model_dir.glob("iter_epoch_*"):
        try:
            if path.is_file():
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def patch_random_crop_numpy_choice(repo: Path | None) -> bool:
    if repo is None:
        return False
    path = repo / "ppocr" / "data" / "imaug" / "random_crop_data.py"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    target = "int(np.random.choice(axis, size=1))"
    replacement = "int(np.random.choice(axis, size=1)[0])"
    if replacement in text:
        return False
    if target not in text:
        return False
    backup = path.with_suffix(path.suffix + ".aitrain.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(target, replacement), encoding="utf-8")
    return True


def git_head(repo: Path | None) -> str:
    if repo is None:
        return ""
    try:
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return ""
    return result.stdout.strip()


def module_version(module_name: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(module_name)
    except Exception:
        return ""


def run(request: dict[str, Any]) -> int:
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    dataset_path = Path(str(request.get("datasetPath") or parameters.get("datasetPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or parameters.get("outputPath") or "aitrain-ppocr-det-output")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    preset = resolve_preset(parameters)
    repo = find_repo(parameters)
    if repo is None and preset.get("requiresRepo"):
        return fail(
            "Official PaddleOCR source checkout is required for this PaddleOCR detection preset or officialConfig override.",
            "paddleocr_repo_missing",
            {"modelPreset": preset["modelPreset"], "officialConfig": preset["config"]},
        )

    try:
        train_label_source = resolve_dataset_file(dataset_path, parameters.get("trainLabelFile"), "det_gt.txt")
        train_samples_source = read_det_labels(train_label_source)
        val_label_value = str(parameters.get("valLabelFile") or "").strip()
        if val_label_value:
            val_label_source = resolve_dataset_file(dataset_path, val_label_value, "det_gt_val.txt")
            val_samples_source = read_det_labels(val_label_source)
            samples = train_samples_source + val_samples_source
        else:
            samples = train_samples_source
    except Exception as exc:
        return fail(str(exc), "bad_dataset")

    run_official = bool_param(parameters, "runOfficial", False)
    prepare_only = bool_param(parameters, "prepareOnly", not run_official)
    if repo is None and not prepare_only:
        return fail(
            "Official PaddleOCR source checkout was not found. Set paddleOcrRepoPath or AITRAIN_PADDLEOCR_REPO, or use prepareOnly=true.",
            "paddleocr_repo_missing",
        )

    official_data_dir = output_path / "official_data"
    official_data_dir.mkdir(parents=True, exist_ok=True)
    if "val_samples_source" in locals():
        train_samples, val_samples = train_samples_source, val_samples_source
    else:
        train_samples, val_samples = split_samples(samples, float(parameters.get("validationRatio", 0.2)))
    train_list_path = official_data_dir / "train_det_list.txt"
    val_list_path = official_data_dir / "val_det_list.txt"
    write_label_file(train_list_path, train_samples)
    write_label_file(val_list_path, val_samples)

    try:
        config = build_config(repo, parameters, preset, dataset_path, output_path, train_list_path, val_list_path, samples[0][0])
    except Exception as exc:
        return fail(f"Failed to build official PaddleOCR det config: {exc}", "config_failed")

    config_path = output_path / str(preset["outputConfigName"])
    config_path.write_text(yaml_dump(config), encoding="utf-8")
    resolved_model_name = config_model_name(config, str(preset["modelName"]))
    export_only = bool_param(parameters, "exportOnly", False)
    train_command = [sys.executable, "tools/train.py", "-c", str(config_path)]
    pretrained_base = select_checkpoint_base(output_path, parameters.get("pretrainedModel"))
    export_command = [
        sys.executable,
        "tools/export_model.py",
        "-c",
        str(config_path),
        "-o",
        f"Global.pretrained_model={pretrained_base}",
        f"Global.save_inference_dir={output_path / 'official_inference'}",
    ]
    write_command_file(output_path / "run_official_det_train.ps1", train_command, repo)
    write_command_file(output_path / "run_official_det_export.ps1", export_command, repo)

    report_path = output_path / "paddleocr_official_det_report.json"
    train_log_path = output_path / "official_det_train.log"
    export_log_path = output_path / "official_det_export.log"
    report: dict[str, Any] = {
        "ok": True,
        "backend": BACKEND_ID,
        "framework": "PaddleOCR official tools",
        "modelFamily": "ocr_detection",
        "mode": "prepareOnly" if prepare_only else ("exportOnly" if export_only else "officialTrain"),
        "note": "PaddleOCR PP-OCRv4/PP-OCRv5/PP-OCRv6 official detection adapter. It validates official train/export wiring, not OCR accuracy.",
        "ocrVersion": preset["ocrVersion"],
        "modelPreset": preset["modelPreset"],
        "resolvedModelName": resolved_model_name,
        "resolvedOfficialConfig": str(preset.get("resolvedConfig", preset["config"])),
        "configSource": preset["configSource"],
        "presetDictionaryPath": "",
        "pythonVersion": sys.version.split()[0],
        "paddleVersion": module_version("paddlepaddle"),
        "paddleOcrPackageVersion": module_version("paddleocr"),
        "paddleOcrRepoPath": str(repo) if repo else "",
        "paddleOcrRequestedRef": str(parameters.get("paddleOcrRef") or ""),
        "paddleOcrResolvedRef": git_head(repo),
        "configPath": str(config_path),
        "trainListPath": str(train_list_path),
        "valListPath": str(val_list_path),
        "sourceTrainLabelPath": str(train_label_source),
        "sourceValLabelPath": str(val_label_source) if "val_label_source" in locals() else "",
        "trainCommand": train_command,
        "exportCommand": export_command,
        "trainLogPath": str(train_log_path),
        "exportLogPath": str(export_log_path),
        "metrics": {},
    }
    compatibility_patches: list[str] = []
    if bool_param(parameters, "patchPaddleOcrNumpyChoice", True) and patch_random_crop_numpy_choice(repo):
        compatibility_patches.append("paddleocr_random_crop_numpy_choice_scalar")
    report["compatibilityPatches"] = compatibility_patches

    emit("artifact", name=config_path.name, path=str(config_path), kind="config")
    emit("artifact", name="train_det_list.txt", path=str(train_list_path), kind="dataset")
    emit("artifact", name="val_det_list.txt", path=str(val_list_path), kind="dataset")

    if not prepare_only:
        assert repo is not None
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        official_metrics: dict[str, float] = {}
        if not export_only:
            train_exit = run_process(train_command, repo, env, official_metrics, train_log_path, parameters)
            report["metrics"] = official_metrics
            report["trainExitCode"] = train_exit
            if train_exit != 0:
                report["ok"] = False
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                emit("artifact", name="paddleocr_official_det_report.json", path=str(report_path), kind="report")
                return fail("Official PaddleOCR detection training failed.", "official_train_failed", {"exitCode": train_exit, "logPath": str(train_log_path)})
        pretrained_base = select_checkpoint_base(output_path, parameters.get("pretrainedModel"))
        export_command = [
            sys.executable,
            "tools/export_model.py",
            "-c",
            str(config_path),
            "-o",
            f"Global.pretrained_model={pretrained_base}",
            f"Global.save_inference_dir={output_path / 'official_inference'}",
        ]
        report["exportCommand"] = export_command
        report["checkpointBasePath"] = str(pretrained_base)
        write_command_file(output_path / "run_official_det_export.ps1", export_command, repo)
        export_exit = run_process(export_command, repo, env, None, export_log_path, parameters)
        report["exportExitCode"] = export_exit
        if export_exit != 0:
            report["ok"] = False
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit("artifact", name="paddleocr_official_det_report.json", path=str(report_path), kind="report")
            return fail("Official PaddleOCR detection export failed.", "official_export_failed", {"exitCode": export_exit, "logPath": str(export_log_path)})
        report["checkpointPath"] = str(pretrained_base.with_suffix(".pdparams"))
        report["inferenceModelDir"] = str(output_path / "official_inference")
        pruned_count = prune_intermediate_checkpoints(output_path, parameters)
        report["checkpointRetention"] = str(parameters.get("checkpointRetention") or "")
        report["prunedCheckpointCount"] = pruned_count
        emit("artifact", name="official_det_model", path=str(output_path / "official_model"), kind="checkpoint_dir")
        emit("artifact", name="official_det_inference", path=str(output_path / "official_inference"), kind="model_dir")

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("artifact", name="paddleocr_official_det_report.json", path=str(report_path), kind="report")
    emit(
        "completed",
        checkpointPath=report.get("checkpointPath", ""),
        inferenceModelDir=report.get("inferenceModelDir", ""),
        reportPath=str(report_path),
        configPath=str(config_path),
        mode=report["mode"],
        metrics=report["metrics"],
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = read_request(args.request)
    except Exception as exc:
        return fail(f"failed to read trainer request: {exc}", "bad_request", exception_details(exc))
    try:
        return run(request)
    except Exception as exc:
        return unhandled_failure(BACKEND_ID, exc)


if __name__ == "__main__":
    raise SystemExit(main())
