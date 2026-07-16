from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = REPO_ROOT / "python_trainers" / "ocr_system" / "paddleocr_system_official_adapter.py"


def write_fake_repo(root: Path, *, write_results: bool = True, invalid_results: bool = False) -> Path:
    repo = root / "PaddleOCR"
    script = repo / "tools" / "infer" / "predict_system.py"
    script.parent.mkdir(parents=True)
    result_action = ""
    if write_results:
        payload = "not-valid" if invalid_results else "[{'transcription': 'AITrain', 'points': [[0, 0], [4, 0], [4, 2], [0, 2]], 'score': 0.98}]"
        result_action = f"(draw / 'system_results.txt').write_text({('sample.png' + chr(9) + payload + chr(10))!r}, encoding='utf-8')"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "draw_arg = next(arg for arg in sys.argv if arg.startswith('--draw_img_save_dir='))",
                "draw = Path(draw_arg.split('=', 1)[1])",
                "draw.mkdir(parents=True, exist_ok=True)",
                result_action,
                "(draw / 'sample_preview.png').write_bytes(b'preview')",
                "print('official system fake completed')",
            ]
        ),
        encoding="utf-8",
    )
    return repo


def make_request(root: Path, repo: Path, *, prepare_only: bool = False) -> tuple[Path, Path]:
    output = root / "output"
    det_model = root / "det-model"
    rec_model = root / "rec-model"
    det_model.mkdir()
    rec_model.mkdir()
    dictionary = root / "dict.txt"
    dictionary.write_text("A\nI\nT\nr\na\ni\nn\n", encoding="utf-8")
    image = root / "sample.png"
    image.write_bytes(b"image")
    request = {
        "protocolVersion": 2,
        "taskId": "ocr-system-test",
        "taskType": "ocr",
        "datasetPath": str(image),
        "outputPath": str(output),
        "backend": "paddleocr_system_official",
        "parameters": {
            "prepareOnly": prepare_only,
            "paddleOcrRepoPath": str(repo),
            "detModelDir": str(det_model),
            "recModelDir": str(rec_model),
            "dictionaryFile": str(dictionary),
            "inferenceImage": str(image),
            "recModelPreset": "PP-OCRv5_mobile_rec",
            "recAlgorithm": "SVTR_LCNet",
        },
    }
    request_path = root / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path, output


def run_adapter(request_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ADAPTER), "--request", str(request_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def events_from(result: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]


def test_official_system_success_requires_structured_results_and_emits_file_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo = write_fake_repo(root)
        request_path, output = make_request(root, repo)

        result = run_adapter(request_path)

        assert result.returncode == 0, result.stdout + result.stderr
        prediction = json.loads((output / "official_system_prediction.json").read_text(encoding="utf-8"))
        assert prediction["predictions"][0]["results"][0]["transcription"] == "AITrain"
        assert prediction["acceptanceBoundary"] == "official_det_rec_system_wiring_only_not_customer_domain_quality_acceptance"
        archive_path = output / "official_system_visualization.zip"
        assert archive_path.is_file()
        with zipfile.ZipFile(archive_path) as archive:
            assert set(archive.namelist()) == {"sample_preview.png", "system_results.txt"}
        events = events_from(result)
        artifact_events = [event for event in events if event.get("type") == "artifact"]
        assert any(event.get("name") == archive_path.name and event.get("kind") == "preview" for event in artifact_events)
        assert not any(event.get("kind") == "preview_dir" for event in artifact_events)
        report = json.loads((output / "paddleocr_official_system_report.json").read_text(encoding="utf-8"))
        assert report["predictionCount"] == 1
        assert "not customer-domain OCR quality acceptance" in report["note"]


def test_official_system_rejects_missing_system_results_after_zero_exit() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo = write_fake_repo(root, write_results=False)
        request_path, output = make_request(root, repo)

        result = run_adapter(request_path)

        assert result.returncode != 0
        assert "official_results_invalid" in result.stdout
        report = json.loads((output / "paddleocr_official_system_report.json").read_text(encoding="utf-8"))
        assert report["ok"] is False
        assert "was not produced" in report["resultValidationError"]
        assert not (output / "official_system_prediction.json").exists()


def test_official_system_rejects_unparseable_system_results() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo = write_fake_repo(root, invalid_results=True)
        request_path, output = make_request(root, repo)

        result = run_adapter(request_path)

        assert result.returncode != 0
        assert "official_results_invalid" in result.stdout
        report = json.loads((output / "paddleocr_official_system_report.json").read_text(encoding="utf-8"))
        assert "invalid system_results.txt payload" in report["resultValidationError"]


def test_prepare_only_keeps_algorithm_derivation_and_does_not_require_repo_execution() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        repo = write_fake_repo(root, write_results=False)
        request_path, output = make_request(root, repo, prepare_only=True)

        result = run_adapter(request_path)

        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((output / "paddleocr_official_system_report.json").read_text(encoding="utf-8"))
        assert report["mode"] == "prepareOnly"
        assert report["recAlgorithm"] == "SVTR_LCNet"
        assert "--rec_algorithm=SVTR_LCNet" in report["predictCommand"]
        assert not (output / "official_system_prediction.json").exists()
