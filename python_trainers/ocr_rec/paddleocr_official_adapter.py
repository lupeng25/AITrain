#!/usr/bin/env python3
"""Official PaddleOCR recognition trainer adapter for AITrain Studio.

This adapter prepares a PaddleOCR PP-OCRv4 recognition config from an
AITrain PaddleOCR-style Rec dataset. When a PaddleOCR source checkout is
available, it can also launch the official tools/train.py and export_model.py
entry points. The lightweight smoke path uses prepareOnly=true so CI and local
CPU checks do not depend on long official training runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BACKEND_ID = "paddleocr_rec_official"
DEFAULT_CONFIG_RELATIVE = "configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def emit(backend: str, event_type: str, **payload: Any) -> None:
    message = {"type": event_type, "timestamp": time.time(), "backend": backend}
    message.update(payload)
    print(json.dumps(message, ensure_ascii=False), flush=True)


def fail(backend: str, message: str, code: str, details: dict[str, Any] | None = None) -> int:
    emit(backend, "failed", code=code, message=message, details=details or {})
    return 1


def read_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("trainer request must be a JSON object")
    return value


def read_labels(path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid OCR label line: {line}")
        samples.append((parts[0].strip().replace("\\", "/"), parts[1].strip()))
    if not samples:
        raise ValueError(f"empty OCR label file: {path}")
    return samples


def read_dictionary(path: Path) -> list[str]:
    chars = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not chars:
        raise ValueError(f"empty OCR dictionary: {path}")
    return chars


def bool_param(parameters: dict[str, Any], key: str, default: bool) -> bool:
    value = parameters.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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


def split_samples(samples: list[tuple[str, str]], validation_ratio: float) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if len(samples) == 1:
        return samples, samples
    ratio = min(0.5, max(0.0, validation_ratio))
    val_count = max(1, int(round(len(samples) * ratio))) if ratio > 0 else 1
    val_count = min(val_count, len(samples) - 1)
    return samples[:-val_count], samples[-val_count:]


def write_label_file(path: Path, samples: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{image}\t{text}\n" for image, text in samples), encoding="utf-8")


def yaml_dump(value: Any) -> str:
    import yaml

    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


def build_config(
    repo: Path | None,
    parameters: dict[str, Any],
    dataset_path: Path,
    output_path: Path,
    dict_path: Path,
    train_list_path: Path,
    val_list_path: Path,
    first_image: str,
) -> dict[str, Any]:
    import yaml

    template_relative = str(parameters.get("officialConfig") or DEFAULT_CONFIG_RELATIVE).replace("\\", "/")
    if repo:
        template_path = (repo / template_relative).resolve()
        if not template_path.exists():
            raise FileNotFoundError(f"PaddleOCR config template not found: {template_path}")
        with template_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        config = {
            "Global": {},
            "Optimizer": {"name": "Adam", "lr": {"name": "Cosine", "learning_rate": 0.0005}},
            "Architecture": {"model_type": "rec", "algorithm": "SVTR_LCNet"},
            "Loss": {"name": "MultiLoss"},
            "PostProcess": {"name": "CTCLabelDecode"},
            "Metric": {"name": "RecMetric", "main_indicator": "acc"},
            "Train": {"dataset": {}, "loader": {}, "sampler": {}},
            "Eval": {"dataset": {}, "loader": {}},
        }

    epochs = max(1, int(parameters.get("epochs", 1)))
    batch_size = max(1, int(parameters.get("batchSize", 1)))
    image_height = max(16, int(parameters.get("imageHeight", 48)))
    image_width = max(32, int(parameters.get("imageWidth", 320)))
    max_text_length = max(1, int(parameters.get("maxTextLength", 25)))
    use_gpu = bool_param(parameters, "useGpu", False)
    save_model_dir = output_path / "official_model"
    save_inference_dir = output_path / "official_inference"

    global_config = config.setdefault("Global", {})
    global_config.update(
        {
            "use_gpu": use_gpu,
            "epoch_num": epochs,
            "print_batch_step": 1,
            "save_model_dir": str(save_model_dir),
            "save_epoch_step": 1,
            "eval_batch_step": [0, 1],
            "pretrained_model": parameters.get("pretrainedModel") or None,
            "checkpoints": parameters.get("resumeCheckpoint") or None,
            "save_inference_dir": str(save_inference_dir),
            "infer_img": str((dataset_path / first_image).resolve()),
            "character_dict_path": str(dict_path),
            "max_text_length": max_text_length,
            "use_space_char": bool_param(parameters, "useSpaceChar", True),
            "distributed": False,
            "d2s_train_image_shape": [3, image_height, image_width],
        }
    )

    train = config.setdefault("Train", {})
    train_dataset = train.setdefault("dataset", {})
    train_dataset["data_dir"] = str(dataset_path)
    train_dataset["label_file_list"] = [str(train_list_path)]
    for transform in train_dataset.get("transforms", []):
        if isinstance(transform, dict) and "RecConAug" in transform and isinstance(transform["RecConAug"], dict):
            transform["RecConAug"]["image_shape"] = [image_height, image_width, 3]
            transform["RecConAug"]["max_text_length"] = max_text_length
    train_loader = train.setdefault("loader", {})
    train_loader.update({"batch_size_per_card": batch_size, "drop_last": False, "num_workers": 0})
    train_sampler = train.setdefault("sampler", {})
    if isinstance(train_sampler, dict):
        train_sampler.update({"first_bs": batch_size, "fix_bs": True, "is_training": True})

    eval_config = config.setdefault("Eval", {})
    eval_dataset = eval_config.setdefault("dataset", {})
    eval_dataset["data_dir"] = str(dataset_path)
    eval_dataset["label_file_list"] = [str(val_list_path)]
    for transform in eval_dataset.get("transforms", []):
        if isinstance(transform, dict) and "RecResizeImg" in transform and isinstance(transform["RecResizeImg"], dict):
            transform["RecResizeImg"]["image_shape"] = [3, image_height, image_width]
    eval_loader = eval_config.setdefault("loader", {})
    eval_loader.update({"batch_size_per_card": batch_size, "drop_last": False, "num_workers": 0})

    return config


def write_command_file(path: Path, command: list[str], cwd: Path | None) -> None:
    quoted = " ".join(f'"{part}"' if " " in part else part for part in command)
    lines = []
    if cwd is not None:
        lines.append(f'Set-Location "{cwd}"')
    lines.append(quoted)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_official_metrics(backend: str, line: str, metrics: dict[str, float]) -> None:
    name_map = {
        "acc": "accuracy",
        "norm_edit_dis": "normalizedEditDistance",
        "CTCLoss": "ctcLoss",
        "NRTRLoss": "nrtrLoss",
        "loss": "loss",
    }
    for raw_name, raw_value in re.findall(r"\b(acc|norm_edit_dis|CTCLoss|NRTRLoss|loss):\s*([-+0-9.eE]+)", line):
        try:
            value = float(raw_value)
        except ValueError:
            continue
        name = name_map.get(raw_name, raw_name)
        metrics[name] = value
        emit(backend, "metric", name=name, value=value)


def run_process(backend: str, command: list[str], cwd: Path, env: dict[str, str], metrics: dict[str, float] | None = None) -> int:
    emit(backend, "log", level="info", message=f"Running official PaddleOCR command: {' '.join(command)}")
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
    for line in process.stdout:
        if line.strip():
            stripped = line.rstrip()
            emit(backend, "log", level="info", message=stripped)
            if metrics is not None:
                parse_official_metrics(backend, stripped, metrics)
    return process.wait()


def run(request: dict[str, Any]) -> int:
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    backend = str(request.get("backend") or parameters.get("trainingBackend") or DEFAULT_BACKEND_ID)
    dataset_path = Path(str(request.get("datasetPath") or parameters.get("datasetPath") or "")).resolve()
    output_path = Path(str(request.get("outputPath") or parameters.get("outputPath") or "aitrain-ppocr-output")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        chars = read_dictionary(dataset_path / "dict.txt")
        samples = read_labels(dataset_path / "rec_gt.txt")
    except Exception as exc:
        return fail(backend, str(exc), "bad_dataset")

    repo = find_repo(parameters)
    run_official = bool_param(parameters, "runOfficial", False)
    prepare_only = bool_param(parameters, "prepareOnly", not run_official)
    if repo is None and not prepare_only:
        return fail(
            backend,
            "Official PaddleOCR source checkout was not found. Set paddleOcrRepoPath or AITRAIN_PADDLEOCR_REPO, or use prepareOnly=true.",
            "paddleocr_repo_missing",
        )

    official_data_dir = output_path / "official_data"
    official_data_dir.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples = split_samples(samples, float(parameters.get("validationRatio", 0.2)))
    train_list_path = official_data_dir / "train_list.txt"
    val_list_path = official_data_dir / "val_list.txt"
    dict_path = official_data_dir / "dict.txt"
    write_label_file(train_list_path, train_samples)
    write_label_file(val_list_path, val_samples)
    shutil.copyfile(dataset_path / "dict.txt", dict_path)

    try:
        config = build_config(repo, parameters, dataset_path, output_path, dict_path, train_list_path, val_list_path, samples[0][0])
    except Exception as exc:
        return fail(backend, f"Failed to build official PaddleOCR config: {exc}", "config_failed")

    config_path = output_path / "aitrain_ppocrv4_rec.yml"
    config_path.write_text(yaml_dump(config), encoding="utf-8")
    train_command = [sys.executable, "tools/train.py", "-c", str(config_path)]
    export_command = [
        sys.executable,
        "tools/export_model.py",
        "-c",
        str(config_path),
        "-o",
        f"Global.pretrained_model={output_path / 'official_model' / 'best_accuracy'}",
        f"Global.save_inference_dir={output_path / 'official_inference'}",
    ]
    write_command_file(output_path / "run_official_train.ps1", train_command, repo)
    write_command_file(output_path / "run_official_export.ps1", export_command, repo)

    report_path = output_path / "paddleocr_official_rec_report.json"
    report: dict[str, Any] = {
        "ok": True,
        "backend": backend,
        "framework": "PaddleOCR official tools",
        "modelFamily": "ocr_recognition",
        "mode": "prepareOnly" if prepare_only else "officialTrain",
        "note": "PP-OCRv4 official config adapter. prepareOnly=true validates dataset/config generation without running official training.",
        "paddleOcrRepoPath": str(repo) if repo else "",
        "configPath": str(config_path),
        "trainListPath": str(train_list_path),
        "valListPath": str(val_list_path),
        "dictPath": str(dict_path),
        "classCount": len(chars) + 1,
        "blankIndex": 0,
        "trainCommand": train_command,
        "exportCommand": export_command,
        "metrics": {},
    }

    emit(backend, "artifact", name="aitrain_ppocrv4_rec.yml", path=str(config_path), kind="config")
    emit(backend, "artifact", name="train_list.txt", path=str(train_list_path), kind="dataset")
    emit(backend, "artifact", name="val_list.txt", path=str(val_list_path), kind="dataset")
    emit(backend, "artifact", name="dict.txt", path=str(dict_path), kind="dict")

    if not prepare_only:
        assert repo is not None
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        official_metrics: dict[str, float] = {}
        train_exit = run_process(backend, train_command, repo, env, official_metrics)
        report["metrics"] = official_metrics
        if train_exit != 0:
            report["ok"] = False
            report["trainExitCode"] = train_exit
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit(backend, "artifact", name="paddleocr_official_rec_report.json", path=str(report_path), kind="report")
            return fail(backend, "Official PaddleOCR training failed.", "official_train_failed", {"exitCode": train_exit})
        export_exit = run_process(backend, export_command, repo, env)
        if export_exit != 0:
            report["ok"] = False
            report["exportExitCode"] = export_exit
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit(backend, "artifact", name="paddleocr_official_rec_report.json", path=str(report_path), kind="report")
            return fail(backend, "Official PaddleOCR export failed.", "official_export_failed", {"exitCode": export_exit})
        report["checkpointPath"] = str(output_path / "official_model" / "best_accuracy.pdparams")
        report["inferenceModelDir"] = str(output_path / "official_inference")
        emit(backend, "artifact", name="official_model", path=str(output_path / "official_model"), kind="checkpoint_dir")
        emit(backend, "artifact", name="official_inference", path=str(output_path / "official_inference"), kind="model_dir")

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit(backend, "artifact", name="paddleocr_official_rec_report.json", path=str(report_path), kind="report")
    emit(
        backend,
        "completed",
        checkpointPath=report.get("checkpointPath", ""),
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
        return fail(DEFAULT_BACKEND_ID, f"failed to read trainer request: {exc}", "bad_request")
    return run(request)


if __name__ == "__main__":
    raise SystemExit(main())
