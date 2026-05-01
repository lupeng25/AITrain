#!/usr/bin/env python3
"""Official PaddleOCR system inference adapter for AITrain Studio."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BACKEND_ID = "paddleocr_system_official"

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
        if (resolved / "tools" / "infer" / "predict_system.py").exists():
            return resolved
    return None


def resolve_path(parameters: dict[str, Any], key: str, fallback: str = "") -> Path:
    value = str(parameters.get(key) or fallback).strip()
    return Path(value).resolve() if value else Path()


def prepare_compatible_model_dir(source_dir: Path, output_path: Path, name: str) -> Path:
    inference_config = source_dir / "inference.yml"
    if not inference_config.exists():
        return source_dir
    import yaml

    target_dir = output_path / f"official_system_{name}_model"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_file in source_dir.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, target_dir / source_file.name)
    config = yaml.safe_load(inference_config.read_text(encoding="utf-8")) or {}
    if isinstance(config, dict):
        global_config = config.get("Global")
        if isinstance(global_config, dict):
            global_config.pop("model_name", None)
    (target_dir / "inference.yml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target_dir


def write_command_file(path: Path, command: list[str], cwd: Path | None) -> None:
    quoted = " ".join(f'"{part}"' if " " in part else part for part in command)
    lines = []
    if cwd is not None:
        lines.append(f'Set-Location "{cwd}"')
    lines.append(quoted)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def run_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, list[str]]:
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
    exit_code = process.wait()
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return exit_code, lines


def parse_system_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    predictions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        image_name, _, payload = line.partition("\t")
        predictions.append({"image": image_name, "raw": payload})
    return predictions


def run(request: dict[str, Any]) -> int:
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    output_path = Path(str(request.get("outputPath") or parameters.get("outputPath") or "aitrain-ppocr-system-output")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    repo = find_repo(parameters)
    prepare_only = bool_param(parameters, "prepareOnly", False)
    if repo is None and not prepare_only:
        return fail(
            "Official PaddleOCR source checkout was not found. Set paddleOcrRepoPath or AITRAIN_PADDLEOCR_REPO, or use prepareOnly=true.",
            "paddleocr_repo_missing",
        )

    det_model_dir = resolve_path(parameters, "detModelDir")
    rec_model_dir = resolve_path(parameters, "recModelDir")
    dictionary_file = resolve_path(parameters, "dictionaryFile")
    inference_image = resolve_path(parameters, "inferenceImage", str(request.get("datasetPath") or ""))
    command_det_model_dir = det_model_dir
    command_rec_model_dir = rec_model_dir
    if not prepare_only:
        command_det_model_dir = prepare_compatible_model_dir(det_model_dir, output_path, "det")
        command_rec_model_dir = prepare_compatible_model_dir(rec_model_dir, output_path, "rec")
    draw_dir = output_path / "official_system_visualization"
    log_path = output_path / "official_system_predict.log"
    prediction_path = output_path / "official_system_prediction.json"
    report_path = output_path / "paddleocr_official_system_report.json"

    command = [
        sys.executable,
        "tools/infer/predict_system.py",
        f"--image_dir={inference_image}",
        f"--det_model_dir={command_det_model_dir}",
        f"--rec_model_dir={command_rec_model_dir}",
        f"--rec_char_dict_path={dictionary_file}",
        "--det_algorithm=DB",
        "--rec_algorithm=SVTR_LCNet",
        "--use_angle_cls=False",
        f"--use_gpu={str(bool_param(parameters, 'useGpu', False))}",
        "--enable_mkldnn=False",
        f"--drop_score={float(parameters.get('dropScore', 0.0))}",
        f"--draw_img_save_dir={draw_dir}",
    ]
    write_command_file(output_path / "run_official_system_predict.ps1", command, repo)

    report: dict[str, Any] = {
        "ok": True,
        "backend": BACKEND_ID,
        "framework": "PaddleOCR official tools",
        "modelFamily": "ocr",
        "mode": "prepareOnly" if prepare_only else "officialSystemPredict",
        "note": "Official PaddleOCR predict_system.py adapter. Angle classifier is disabled in Phase 31.",
        "pythonVersion": sys.version.split()[0],
        "paddleVersion": module_version("paddlepaddle"),
        "paddleOcrPackageVersion": module_version("paddleocr"),
        "paddleOcrRepoPath": str(repo) if repo else "",
        "paddleOcrRequestedRef": str(parameters.get("paddleOcrRef") or ""),
        "paddleOcrResolvedRef": git_head(repo),
        "detModelDir": str(det_model_dir),
        "recModelDir": str(rec_model_dir),
        "commandDetModelDir": str(command_det_model_dir),
        "commandRecModelDir": str(command_rec_model_dir),
        "dictionaryFile": str(dictionary_file),
        "inferenceImage": str(inference_image),
        "predictCommand": command,
        "predictLogPath": str(log_path),
        "drawImageSaveDir": str(draw_dir),
        "predictionPath": str(prediction_path),
    }

    emit("artifact", name="run_official_system_predict.ps1", path=str(output_path / "run_official_system_predict.ps1"), kind="command")
    if not prepare_only:
        missing = []
        for label, path in (
            ("detModelDir", det_model_dir),
            ("recModelDir", rec_model_dir),
            ("dictionaryFile", dictionary_file),
            ("inferenceImage", inference_image),
        ):
            if not path.exists():
                missing.append(f"{label}={path}")
        if missing:
            return fail("Missing PaddleOCR system inference inputs.", "missing_inputs", {"missing": missing})
        assert repo is not None
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        draw_dir.mkdir(parents=True, exist_ok=True)
        exit_code, lines = run_process(command, repo, env, log_path)
        report["predictExitCode"] = exit_code
        results_path = draw_dir / "system_results.txt"
        predictions = parse_system_results(results_path)
        prediction_payload = {
            "ok": exit_code == 0,
            "taskType": "ocr",
            "backend": BACKEND_ID,
            "imagePath": str(inference_image),
            "detModelDir": str(det_model_dir),
            "recModelDir": str(rec_model_dir),
            "dictionaryFile": str(dictionary_file),
            "systemResultsPath": str(results_path),
            "visualizationDir": str(draw_dir),
            "predictions": predictions,
            "output": lines,
        }
        prediction_path.write_text(json.dumps(prediction_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["systemResultsPath"] = str(results_path)
        report["predictionCount"] = len(predictions)
        emit("artifact", name="official_system_prediction.json", path=str(prediction_path), kind="prediction")
        emit("artifact", name="official_system_visualization", path=str(draw_dir), kind="preview_dir")
        if exit_code != 0:
            report["ok"] = False
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit("artifact", name="paddleocr_official_system_report.json", path=str(report_path), kind="report")
            return fail("Official PaddleOCR system prediction failed.", "official_predict_failed", {"exitCode": exit_code, "logPath": str(log_path)})

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("artifact", name="paddleocr_official_system_report.json", path=str(report_path), kind="report")
    emit("completed", reportPath=str(report_path), predictionPath=str(prediction_path) if prediction_path.exists() else "", mode=report["mode"])
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
