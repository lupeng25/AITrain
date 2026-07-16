#!/usr/bin/env python3
"""Official PaddleOCR system inference adapter for AITrain Studio."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from adapter_event_channel_v2 import AdapterEventChannelV2, event_channel_from_environment  # noqa: E402
from adapter_sdk import AdapterCanceled, AdapterSdk  # noqa: E402
from trainer_protocol import configure_stdio, exception_details  # noqa: E402


BACKEND_ID = "paddleocr_system_official"

configure_stdio()

_adapter: AdapterSdk | None = None
_event_channel: AdapterEventChannelV2 | None = None


def configure_adapter() -> None:
    global _adapter, _event_channel
    if _event_channel is None and os.environ.get("AITRAIN_EVENT_PORT"):
        _event_channel = event_channel_from_environment()
        _event_channel.connect()
    if _adapter is None:
        sink = _event_channel.emit_legacy_event if _event_channel is not None else None
        _adapter = AdapterSdk(BACKEND_ID, event_sink=sink, cancel_file=os.environ.get("AITRAIN_CANCEL_FILE"))


def close_adapter() -> None:
    global _event_channel
    if _event_channel is not None:
        _event_channel.close()
        _event_channel = None


def active_adapter() -> AdapterSdk:
    configure_adapter()
    assert _adapter is not None
    return _adapter


def emit(event_type: str, **payload: Any) -> None:
    payload.pop("backend", None)
    adapter = active_adapter()
    if event_type == "artifact":
        adapter.emit_artifact_candidate(
            str(payload.pop("kind", "artifact")),
            str(payload.pop("path", "")),
            message=str(payload.pop("message", "")),
            **payload,
        )
    elif event_type == "log":
        adapter.emit_log(str(payload.pop("message", "")), level=str(payload.pop("level", "info")), **payload)
    elif event_type == "completed":
        adapter.emit_completed(str(payload.pop("message", "PaddleOCR System adapter completed")), **payload)
    elif event_type == "failed":
        adapter.emit_failed(
            str(payload.pop("message", "PaddleOCR System adapter failed")),
            str(payload.pop("code", "paddleocr_system_failed")),
            payload.pop("details", {}),
        )
    elif event_type == "canceled":
        adapter.emit_canceled(str(payload.pop("message", "PaddleOCR System adapter canceled")), **payload)
    else:
        raise ValueError(f"unsupported adapter event type: {event_type}")


def fail(message: str, code: str, details: dict[str, Any] | None = None) -> int:
    return active_adapter().emit_failed(message, code, details)


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
            cwd / ".deps" / "repos" / "PaddleOCR",
            cwd / ".deps" / "PaddleOCR",
            script.parents[3] / ".deps" / "repos" / "PaddleOCR" if len(script.parents) > 3 else script.parent,
            script.parents[3] / ".deps" / "PaddleOCR" if len(script.parents) > 3 else script.parent,
            script.parents[2] / ".deps" / "repos" / "PaddleOCR" if len(script.parents) > 2 else script.parent,
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


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def rec_algorithm_for_preset(preset: str) -> str:
    normalized = preset.strip().lower()
    if normalized == "pp-ocrv5_server_rec":
        return "SVTR_HGNet"
    if normalized.startswith("pp-ocrv6"):
        return ""
    if normalized:
        return "SVTR_LCNet"
    return ""


def is_ppocrv6_metadata(parameters: dict[str, Any], metadata: dict[str, str]) -> bool:
    values = [
        str(parameters.get("recModelPreset") or ""),
        str(parameters.get("modelPreset") or ""),
        str(metadata.get("recModelPreset") or ""),
        str(metadata.get("recModelName") or ""),
        str(metadata.get("modelName") or ""),
        str(metadata.get("ocrVersion") or ""),
    ]
    return any(value.strip().lower().startswith("pp-ocrv6") for value in values)


def infer_rec_metadata_from_inference_yml(model_dir: Path) -> dict[str, str]:
    inference_config = model_dir / "inference.yml"
    if not inference_config.exists():
        return {}
    try:
        import yaml

        config = yaml.safe_load(inference_config.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}
    metadata: dict[str, str] = {}
    global_config = config.get("Global")
    if isinstance(global_config, dict):
        model_name = str(global_config.get("model_name") or "").strip()
        if model_name:
            metadata["modelName"] = model_name
            algorithm = rec_algorithm_for_preset(model_name)
            if algorithm:
                metadata["recAlgorithm"] = algorithm
    architecture = config.get("Architecture")
    if isinstance(architecture, dict):
        algorithm = str(architecture.get("algorithm") or "").strip()
        if algorithm:
            metadata["recAlgorithm"] = algorithm
    return metadata


def resolve_rec_metadata(parameters: dict[str, Any], rec_model_dir: Path) -> dict[str, str]:
    metadata = infer_rec_metadata_from_inference_yml(rec_model_dir)
    rec_report_path = resolve_path(parameters, "recReportPath")
    report = read_json_object(rec_report_path) if str(rec_report_path) != "." else {}
    if report:
        for source_key, target_key in (
            ("modelPreset", "recModelPreset"),
            ("resolvedModelName", "recModelName"),
            ("recAlgorithm", "recAlgorithm"),
            ("ocrVersion", "ocrVersion"),
        ):
            value = str(report.get(source_key) or "").strip()
            if value:
                metadata[target_key] = value
    for key in ("recModelPreset", "modelPreset"):
        value = str(parameters.get(key) or "").strip()
        if value:
            metadata["recModelPreset"] = value
            algorithm = rec_algorithm_for_preset(value)
            if algorithm:
                metadata["recAlgorithm"] = algorithm
            break
    explicit_algorithm = str(parameters.get("recAlgorithm") or "").strip()
    if explicit_algorithm:
        metadata["recAlgorithm"] = explicit_algorithm
    if not metadata.get("recAlgorithm") and not is_ppocrv6_metadata(parameters, metadata):
        metadata["recAlgorithm"] = "SVTR_LCNet"
    return metadata


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


def run_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path, parameters: dict[str, Any]) -> tuple[int, list[str], bool]:
    emit("log", level="info", message=f"Running official PaddleOCR command: {' '.join(command)}")
    _, _, tail_lines = official_log_options(parameters)

    result = active_adapter().run_child_process(
        command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        tail_line_limit=tail_lines,
    )
    return result.exit_code, list(result.tail_lines), result.canceled


def parse_system_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"PaddleOCR system_results.txt was not produced: {path}")
    predictions: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if not line.strip():
            continue
        image_name, _, payload = line.partition("\t")
        if not image_name.strip() or not payload.strip():
            raise ValueError(f"invalid system_results.txt line {line_number}: expected image and payload")
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"invalid system_results.txt payload at line {line_number}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"invalid system_results.txt payload at line {line_number}: expected a list")
        predictions.append({"image": image_name, "results": parsed})
    return predictions


def create_preview_archive(draw_dir: Path, archive_path: Path) -> list[str]:
    files = sorted(path for path in draw_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"PaddleOCR visualization directory contains no files: {draw_dir}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(draw_dir).as_posix())
    return [path.relative_to(draw_dir).as_posix() for path in files]


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
    rec_metadata = resolve_rec_metadata(parameters, rec_model_dir)
    rec_algorithm = rec_metadata.get("recAlgorithm", "")
    if not rec_algorithm:
        return fail(
            "PP-OCRv6 System inference requires recAlgorithm from Rec inference.yml, Rec report, or explicit recAlgorithm.",
            "rec_algorithm_missing",
            {
                "recModelPreset": str(parameters.get("recModelPreset") or parameters.get("modelPreset") or ""),
                "recReportPath": str(resolve_path(parameters, "recReportPath")),
            },
        )
    det_model_preset = str(parameters.get("detModelPreset") or "").strip()
    rec_model_preset = rec_metadata.get("recModelPreset", str(parameters.get("recModelPreset") or "").strip())
    command_det_model_dir = det_model_dir
    command_rec_model_dir = rec_model_dir
    if not prepare_only:
        command_det_model_dir = prepare_compatible_model_dir(det_model_dir, output_path, "det")
        command_rec_model_dir = prepare_compatible_model_dir(rec_model_dir, output_path, "rec")
    draw_dir = output_path / "official_system_visualization"
    log_path = output_path / "official_system_predict.log"
    prediction_path = output_path / "official_system_prediction.json"
    report_path = output_path / "paddleocr_official_system_report.json"
    preview_archive_path = output_path / "official_system_visualization.zip"

    command = [
        sys.executable,
        "tools/infer/predict_system.py",
        f"--image_dir={inference_image}",
        f"--det_model_dir={command_det_model_dir}",
        f"--rec_model_dir={command_rec_model_dir}",
        f"--rec_char_dict_path={dictionary_file}",
        "--det_algorithm=DB",
        f"--rec_algorithm={rec_algorithm}",
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
        "note": "Official PaddleOCR predict_system.py wiring for an independently supplied Det model plus Rec model. This report proves official Det+Rec composition and artifact generation only; it is not customer-domain OCR quality acceptance evidence. Angle classifier is disabled in this product route.",
        "acceptanceBoundary": "official_det_rec_system_wiring_only_not_customer_domain_quality_acceptance",
        "detModelPreset": det_model_preset,
        "recModelPreset": rec_model_preset,
        "recModelName": rec_metadata.get("recModelName", rec_metadata.get("modelName", "")),
        "recAlgorithm": rec_algorithm,
        "ocrVersion": rec_metadata.get("ocrVersion", ""),
        "recReportPath": str(resolve_path(parameters, "recReportPath")),
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
        exit_code, lines, canceled = run_process(command, repo, env, log_path, parameters)
        report["predictExitCode"] = exit_code
        report["canceled"] = canceled
        if canceled:
            report["ok"] = False
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit("artifact", name="paddleocr_official_system_report.json", path=str(report_path), kind="report")
            emit("canceled", message="Official PaddleOCR system prediction was canceled.")
            return 1
        if exit_code != 0:
            report["ok"] = False
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit("artifact", name="paddleocr_official_system_report.json", path=str(report_path), kind="report")
            return fail("Official PaddleOCR system prediction failed.", "official_predict_failed", {"exitCode": exit_code, "logPath": str(log_path)})
        results_path = draw_dir / "system_results.txt"
        try:
            predictions = parse_system_results(results_path)
            preview_files = create_preview_archive(draw_dir, preview_archive_path)
        except (OSError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
            report["ok"] = False
            report["systemResultsPath"] = str(results_path)
            report["resultValidationError"] = str(exc)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit("artifact", name="paddleocr_official_system_report.json", path=str(report_path), kind="report")
            return fail(
                "Official PaddleOCR command succeeded but system_results.txt or visualization artifacts are invalid.",
                "official_results_invalid",
                {"systemResultsPath": str(results_path), "error": str(exc)},
            )
        prediction_payload = {
            "ok": True,
            "taskType": "ocr",
            "backend": BACKEND_ID,
            "imagePath": str(inference_image),
            "detModelDir": str(det_model_dir),
            "recModelDir": str(rec_model_dir),
            "dictionaryFile": str(dictionary_file),
            "systemResultsPath": str(results_path),
            "visualizationDir": str(draw_dir),
            "visualizationArchivePath": str(preview_archive_path),
            "predictions": predictions,
            "output": lines,
            "acceptanceBoundary": "official_det_rec_system_wiring_only_not_customer_domain_quality_acceptance",
        }
        prediction_path.write_text(json.dumps(prediction_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["systemResultsPath"] = str(results_path)
        report["predictionCount"] = len(predictions)
        report["visualizationArchivePath"] = str(preview_archive_path)
        report["visualizationFiles"] = preview_files
        emit("artifact", name="official_system_prediction.json", path=str(prediction_path), kind="prediction")
        emit("artifact", name="official_system_visualization.zip", path=str(preview_archive_path), kind="preview")

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("artifact", name="paddleocr_official_system_report.json", path=str(report_path), kind="report")
    emit("completed", reportPath=str(report_path), predictionPath=str(prediction_path) if prediction_path.exists() else "", mode=report["mode"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        configure_adapter()
        try:
            request = read_request(args.request)
        except Exception as exc:
            return fail(f"failed to read trainer request: {exc}", "bad_request", exception_details(exc))
        try:
            return run(request)
        except AdapterCanceled:
            emit("canceled", message="PaddleOCR System adapter was canceled.")
            return 1
        except Exception as exc:
            return fail(f"Unhandled PaddleOCR System adapter error: {exc}", "unhandled_exception", exception_details(exc))
    except Exception as exc:
        # Channel/bootstrap failures cannot be reported through the authenticated
        # channel; keep a concise stderr diagnostic for the Worker Host.
        print(f"PaddleOCR System adapter bootstrap failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        close_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
