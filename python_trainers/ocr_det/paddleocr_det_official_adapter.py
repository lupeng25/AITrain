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
from pathlib import Path
from typing import Any


BACKEND_ID = "paddleocr_det_official"
DEFAULT_CONFIG_RELATIVE = "configs/det/PP-OCRv4/PP-OCRv4_mobile_det.yml"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time(), "backend": BACKEND_ID}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def fail(message: str, code: str, details: dict[str, Any] | None = None) -> int:
    emit("failed", code=code, message=message, details=details or {})
    return 1


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
    unstable_train_transforms = {"IaaAugment", "CopyPaste", "EastRandomCropData"}
    for transform in transforms:
        if isinstance(transform, dict):
            if training and any(name in transform for name in unstable_train_transforms):
                continue
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
    for candidate in sorted(model_dir.glob("iter_epoch_*.pdparams"), reverse=True):
        return candidate.with_suffix("")
    return model_dir / "best_accuracy"


def build_config(
    repo: Path | None,
    parameters: dict[str, Any],
    dataset_path: Path,
    output_path: Path,
    train_list_path: Path,
    val_list_path: Path,
    first_image: str,
) -> dict[str, Any]:
    import yaml

    template_relative = str(parameters.get("officialConfig") or DEFAULT_CONFIG_RELATIVE).replace("\\", "/")
    if repo:
        template_path = (repo / template_relative).resolve()
        if not template_path.exists():
            raise FileNotFoundError(f"PaddleOCR det config template not found: {template_path}")
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
    save_model_dir = output_path / "official_model"
    save_inference_dir = output_path / "official_inference"

    global_config = config.setdefault("Global", {})
    global_config.update(
        {
            "use_gpu": bool_param(parameters, "useGpu", False),
            "epoch_num": epochs,
            "print_batch_step": 1,
            "save_model_dir": str(save_model_dir),
            "save_epoch_step": 1,
            "eval_batch_step": [0, 1],
            "pretrained_model": parameters.get("pretrainedModel") or None,
            "checkpoints": parameters.get("resumeCheckpoint") or None,
            "save_inference_dir": str(save_inference_dir),
            "infer_img": str((dataset_path / first_image).resolve()),
            "distributed": False,
            "d2s_train_image_shape": [3, image_size, image_size],
            "cal_metric_during_train": bool_param(parameters, "calMetricDuringTrain", False),
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


def run_process(command: list[str], cwd: Path, env: dict[str, str], metrics: dict[str, float] | None, log_path: Path) -> int:
    emit("log", level="info", message=f"Running official PaddleOCR command: {' '.join(command)}")
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
    lines: list[str] = []
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            lines.append(stripped)
            emit("log", level="info", message=stripped)
            if metrics is not None:
                parse_metrics(stripped, metrics)
    exit_code = process.wait()
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return exit_code


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

    repo = find_repo(parameters)
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
        config = build_config(repo, parameters, dataset_path, output_path, train_list_path, val_list_path, samples[0][0])
    except Exception as exc:
        return fail(f"Failed to build official PaddleOCR det config: {exc}", "config_failed")

    config_path = output_path / "aitrain_ppocrv4_det.yml"
    config_path.write_text(yaml_dump(config), encoding="utf-8")
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
        "note": "PP-OCRv4 official detection adapter. It validates official train/export wiring, not OCR accuracy.",
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

    emit("artifact", name="aitrain_ppocrv4_det.yml", path=str(config_path), kind="config")
    emit("artifact", name="train_det_list.txt", path=str(train_list_path), kind="dataset")
    emit("artifact", name="val_det_list.txt", path=str(val_list_path), kind="dataset")

    if not prepare_only:
        assert repo is not None
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        official_metrics: dict[str, float] = {}
        if not export_only:
            train_exit = run_process(train_command, repo, env, official_metrics, train_log_path)
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
        export_exit = run_process(export_command, repo, env, None, export_log_path)
        report["exportExitCode"] = export_exit
        if export_exit != 0:
            report["ok"] = False
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit("artifact", name="paddleocr_official_det_report.json", path=str(report_path), kind="report")
            return fail("Official PaddleOCR detection export failed.", "official_export_failed", {"exitCode": export_exit, "logPath": str(export_log_path)})
        report["checkpointPath"] = str(pretrained_base.with_suffix(".pdparams"))
        report["inferenceModelDir"] = str(output_path / "official_inference")
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
        return fail(f"failed to read trainer request: {exc}", "bad_request")
    return run(request)


if __name__ == "__main__":
    raise SystemExit(main())
