#!/usr/bin/env python3
"""PaddleOCR Det/Rec  contract tests using a fake official checkout."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import py_compile
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER_ROOT = ROOT / "python_trainers"
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from adapter_sdk import AdapterSdk
from ocr import paddleocr_workflow_common as common


SCRIPTS = [
    TRAINER_ROOT / f"ocr_{component}" / f"paddleocr_{component}_{operation}.py"
    for component in ("det", "rec")
    for operation in ("trainer", "evaluator", "exporter", "predictor")
]


def write_script(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def fake_repo(root: Path) -> Path:
    repo = root / "PaddleOCR"
    write_script(repo / "tools/train.py", """
import pathlib, sys
value = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('Global.save_model_dir='))
out = pathlib.Path(value); out.mkdir(parents=True, exist_ok=True)
(out / 'best_accuracy.pdparams').write_bytes(b'official-checkpoint')
(out / 'best_accuracy.pdopt').write_bytes(b'optimizer')
print('loss: 0.25')
""".strip() + "\n")
    write_script(repo / "tools/eval.py", "print('accuracy: 0.95, precision: 0.90')\n")
    write_script(repo / "tools/export_model.py", """
import pathlib, sys
value = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('Global.save_inference_dir='))
out = pathlib.Path(value); out.mkdir(parents=True, exist_ok=True)
(out / 'inference.pdmodel').write_bytes(b'official-inference-model')
(out / 'inference.pdiparams').write_bytes(b'official-inference-params')
(out / 'inference.yml').write_text('Global: {}\\n', encoding='utf-8')
print('exported: 1')
""".strip() + "\n")
    for component in ("det", "rec"):
        write_script(repo / f"tools/infer/predict_{component}.py", f"print('official_{component}_prediction: ok')\n")
    for relative in (
        "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml",
        "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Global: {}\n", encoding="utf-8")
    dictionary = repo / "ppocr/utils/ppocr_keys_v1.txt"
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    dictionary.write_text("a\nb\n", encoding="utf-8")
    return repo


def dataset_snapshot(root: Path, component: str) -> tuple[Path, Path]:
    dataset = root / f"dataset-{component}"
    dataset.mkdir()
    contents = {
        "train.txt": "images/a.png\tlabel\n",
        "val.txt": "images/a.png\tlabel\n",
        "images/a.png": "not-a-real-image",
    }
    if component == "rec":
        contents["dict.txt"] = "a\nb\n"
    files = []
    for relative, text in contents.items():
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        files.append({"relativePath": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = root / f"snapshot-{component}.json"
    manifest.write_text(json.dumps({"schemaVersion": 2, "complete": True, "files": files}), encoding="utf-8")
    return dataset, manifest


def inject_events(component: str, operation: str, events: list[dict]) -> None:
    common._adapter = AdapterSdk(common.backend_id(component, operation), event_sink=events.append)
    common._event_channel = None


def tree_digest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_all_entry_points_compile_and_delegate_to_common_sdk_boundary() -> None:
    for script in SCRIPTS:
        py_compile.compile(str(script), doraise=True)
        source = script.read_text(encoding="utf-8")
        assert "paddleocr_workflow_common import cli_main" in source
    source = (TRAINER_ROOT / "ocr/paddleocr_workflow_common.py").read_text(encoding="utf-8")
    assert "event_channel_from_environment" in source
    assert "sdk.run_child_process(" in source
    assert "subprocess." not in source


def test_det_and_rec_execute_official_eight_step_adapter_contracts() -> None:
    for component in ("det", "rec"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = fake_repo(root)
            repo_before = tree_digest(repo)
            dataset, manifest = dataset_snapshot(root, component)
            events: list[dict] = []
            train_out = root / "train-out"
            inject_events(component, "train", events)
            assert common.run_train(component, {
                "datasetPath": str(dataset), "datasetSnapshotManifest": str(manifest),
                "datasetSnapshotStagingPath": str(root / "train-snapshot"), "outputPath": str(train_out),
                "parameters": {"paddleOcrRepoPath": str(repo)},
            }) == 0
            assert (train_out / "model.zip").is_file()
            assert (train_out / "train.yml").is_file()
            assert (train_out / "training_report.json").is_file()
            if component == "rec":
                assert (train_out / "dict.txt").is_file()

            eval_events: list[dict] = []
            eval_out = root / "eval-out"
            inject_events(component, "evaluate", eval_events)
            evaluate_request = {
                "datasetPath": str(dataset), "datasetSnapshotManifest": str(manifest),
                "datasetSnapshotStagingPath": str(root / "eval-snapshot"), "outputPath": str(eval_out),
                "modelPath": str(train_out / "model.zip"), "checkpointPath": str(train_out / "model.zip"),
                "configPath": str(train_out / "train.yml"), "options": {"paddleOcrRepoPath": str(repo)},
            }
            if component == "rec":
                evaluate_request["dictionaryPath"] = str(train_out / "dict.txt")
            assert common.run_evaluate(component, evaluate_request) == 0
            report = json.loads((eval_out / "evaluation_report.json").read_text(encoding="utf-8"))
            assert report["metrics"]["accuracy"] == 0.95

            export_events: list[dict] = []
            export_out = root / "export-out"
            inject_events(component, "export", export_events)
            export_request = {
                "outputPath": str(export_out), "modelPath": str(eval_out / "model.zip"),
                "checkpointPath": str(eval_out / "model.zip"), "configPath": str(eval_out / "train.yml"),
                "evaluationReportPath": str(eval_out / "evaluation_report.json"),
                "parameters": {"paddleOcrRepoPath": str(repo)},
            }
            if component == "rec":
                export_request["dictionaryPath"] = str(eval_out / "dict.txt")
            assert common.run_export(component, export_request) == 0
            bundle = export_out / "paddleocr_inference.zip"
            sidecar_path = export_out / "paddleocr_bundle.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            assert sidecar["kind"] == "paddleocr_bundle"
            assert sidecar["artifactFormat"] == "paddleocr_inference_bundle"
            assert sidecar["sourceTrainingBackend"] == common.backend_id(component, "train")
            assert sidecar["modelFamily"] == common.task_type(component)
            assert sidecar["taskType"] == common.task_type(component)
            assert sidecar["decoder"] == common.decoder(component)
            assert sidecar["runtimeRoutes"] == ["paddleocr_official"]
            assert sidecar["bundleSha256"] == common.sha256_file(bundle)
            assert sidecar["bundleByteCount"] == bundle.stat().st_size
            assert all(set(row) == {"path", "sha256", "bytes"} for row in sidecar["inventory"])
            assert [event["kind"] for event in export_events if event["type"] == "artifact"] == [
                "official_process_log", "export", "export",
            ]

            predict_events: list[dict] = []
            predict_out = root / "predict-out"
            inject_events(component, "infer", predict_events)
            assert common.run_predict(component, {
                "outputPath": str(predict_out), "sidecarPath": str(sidecar_path), "bundlePath": str(bundle),
                "imagePath": str(dataset / "images/a.png"), "options": {"paddleOcrRepoPath": str(repo)},
            }) == 0
            deployment = json.loads((predict_out / "deployment_report.json").read_text(encoding="utf-8"))
            assert deployment["status"] == "passed"
            assert deployment["inventoryVerified"] is True
            assert (predict_out / "inference_predictions.json").is_file()
            assert (predict_out / "preview.png").read_bytes() == (dataset / "images/a.png").read_bytes()
            assert tree_digest(repo) == repo_before
            assert all(event["backend"] == common.backend_id(component, "infer") for event in predict_events)


def test_predictor_rejects_bundle_traversal_before_official_process() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "malicious.zip"
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("../escaped.txt", b"bad")
        try:
            common.safe_extract_zip(archive, root / "out")
        except ValueError as exc:
            assert "unsafe archive entry" in str(exc)
        else:
            raise AssertionError("unsafe PaddleOCR bundle was accepted")


def test_predictor_recomputes_bundle_hash_and_inventory() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repo = fake_repo(root)
        image = root / "sample.png"
        image.write_bytes(b"image")
        inference = root / "inference"
        inference.mkdir()
        (inference / "model.pdmodel").write_bytes(b"model")
        bundle = root / "paddleocr_inference.zip"
        inventory = common.deterministic_zip(inference, bundle)
        sidecar = root / "paddleocr_bundle.json"
        common.write_json(sidecar, {
            "kind": "paddleocr_bundle", "artifactFormat": common.ARTIFACT_FORMAT,
            "sourceTrainingBackend": common.backend_id("det", "train"),
            "modelFamily": common.task_type("det"), "taskType": common.task_type("det"),
            "decoder": common.decoder("det"),
            "runtimeRoutes": [common.RUNTIME_ROUTE], "inventory": inventory,
            "bundleSha256": "0" * 64, "bundleByteCount": bundle.stat().st_size,
        })
        inject_events("det", "infer", [])
        try:
            common.run_predict("det", {"outputPath": str(root / "out"), "sidecarPath": str(sidecar),
                                             "bundlePath": str(bundle), "imagePath": str(image),
                                             "options": {"paddleOcrRepoPath": str(repo)}})
        except ValueError as exc:
            assert "hash or byte count" in str(exc)
        else:
            raise AssertionError("tampered PaddleOCR sidecar was accepted")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
