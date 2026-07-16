import importlib.util
import json
import py_compile
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "python_trainers" / "anomaly" / "anomalib_adapter.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("aitrain_anomalib_adapter", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_py_compile():
    py_compile.compile(str(ADAPTER_PATH), doraise=True)


def test_preset_registry_and_defaults():
    adapter = load_adapter()
    params = adapter.merge_preset("anomalib_patchcore", {})
    assert params["modelPreset"] == "anomalib_patchcore_wide_resnet50_2"
    assert params["backbone"] == "wide_resnet50_2"
    assert params["layers"] == ["layer2", "layer3"]
    efficient = adapter.merge_preset("anomalib_efficientad", {})
    assert efficient["modelPreset"] == "anomalib_efficientad_s"
    assert efficient["modelSize"] == "small"


def test_dataset_inventory_good_only_limited():
    adapter = load_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train" / "good").mkdir(parents=True)
        (root / "train" / "good" / "a.png").write_bytes(b"not-a-real-image")
        inventory = adapter.dataset_inventory(root)
        assert inventory["normalCount"] == 1
        assert inventory["anomalyCount"] == 0
        assert inventory["evaluationLimited"] is True


def test_training_report_schema_helpers():
    adapter = load_adapter()
    request = {
        "backend": "anomalib_patchcore",
        "datasetPath": "dataset",
        "outputPath": "out",
    }
    params = adapter.merge_preset("anomalib_patchcore", {"quantile": 0.99})
    inventory = {"normalCount": 1, "anomalyCount": 0, "maskCount": 0, "evaluationLimited": True}
    report = adapter.training_report(request, params, inventory, "completed", True, "done")
    assert report["kind"] == "anomalib_training_report"
    assert report["taskType"] == "anomaly_detection"
    assert report["datasetFormat"] == "anomaly_folder"
    assert report["runtime"] == "anomalib_python"
    assert report["evaluationLimited"] is True


def test_metric_extraction_schema_helpers():
    adapter = load_adapter()
    metrics = adapter.extract_metrics(
        [{"image_AUROC": 0.91, "image_F1Score": 0.82, "pixel_AUROC": 0.73}],
        0.995,
    )
    assert metrics["threshold"] == 0.995
    assert metrics["imageAUROC"] == 0.91
    assert metrics["imageF1"] == 0.82
    assert metrics["pixelAUROC"] == 0.73


def test_datamodule_uses_val_split_when_test_split_is_absent():
    adapter = load_adapter()

    class FakeFolder:
        last_kwargs = None

        def __init__(self, **kwargs):
            FakeFolder.last_kwargs = kwargs

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train" / "good").mkdir(parents=True)
        (root / "val" / "good").mkdir(parents=True)
        (root / "val" / "anomaly").mkdir(parents=True)
        adapter.build_datamodule(FakeFolder, root, {})
        assert FakeFolder.last_kwargs["abnormal_dir"] == "val/anomaly"
        assert FakeFolder.last_kwargs["normal_test_dir"] == "val/good"
        assert FakeFolder.last_kwargs["mask_dir"] == "masks/val/anomaly"


def test_datamodule_prefers_test_split_when_available():
    adapter = load_adapter()

    class FakeFolder:
        last_kwargs = None

        def __init__(self, **kwargs):
            FakeFolder.last_kwargs = kwargs

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train" / "good").mkdir(parents=True)
        (root / "val" / "anomaly").mkdir(parents=True)
        (root / "test" / "anomaly").mkdir(parents=True)
        adapter.build_datamodule(FakeFolder, root, {})
        assert FakeFolder.last_kwargs["abnormal_dir"] == "test/anomaly"


def test_datamodule_materializes_mvtec_alias_layout():
    adapter = load_adapter()

    class FakeFolder:
        last_kwargs = None

        def __init__(self, **kwargs):
            FakeFolder.last_kwargs = kwargs

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        view_root = Path(tmp) / "views"
        (root / "train" / "good").mkdir(parents=True)
        (root / "test" / "good").mkdir(parents=True)
        (root / "test" / "scratch").mkdir(parents=True)
        (root / "ground_truth" / "scratch").mkdir(parents=True)
        (root / "train" / "good" / "a.png").write_bytes(b"fake")
        (root / "test" / "good" / "b.png").write_bytes(b"fake")
        (root / "test" / "scratch" / "ng.png").write_bytes(b"fake")
        (root / "ground_truth" / "scratch" / "ng_mask.png").write_bytes(b"mask")

        datamodule = adapter.build_datamodule(FakeFolder, root, {"_aitrainDatasetViewRoot": str(view_root)})
        canonical = (view_root / "anomaly_folder_canonical").resolve()
        assert FakeFolder.last_kwargs["root"] == str(canonical)
        assert FakeFolder.last_kwargs["abnormal_dir"] == "test/anomaly"
        assert FakeFolder.last_kwargs["normal_test_dir"] == "test/good"
        assert FakeFolder.last_kwargs["mask_dir"] == "masks/test/anomaly"
        assert (canonical / "test" / "anomaly" / "scratch__ng.png").exists()
        assert (canonical / "masks" / "test" / "anomaly" / "scratch__ng.png").exists()
        assert getattr(datamodule, "_aitrain_dataset_view_root") == str(canonical)


def test_efficientad_imagenet_dir_resolution():
    adapter = load_adapter()
    adapter.resolve_imagenet_dir({})
    with tempfile.TemporaryDirectory() as tmp:
        resolved = adapter.resolve_imagenet_dir({"imagenetDir": tmp})
        assert resolved == Path(tmp).resolve()


def test_efficientad_constructor_never_drops_imagenet_dir():
    adapter = load_adapter()

    class FakeEfficientAD:
        calls = []

        def __init__(self, **kwargs):
            FakeEfficientAD.calls.append(kwargs)
            raise TypeError("unsupported signature")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            adapter.build_model("anomalib_efficientad", {"imagenetDir": tmp}, None, FakeEfficientAD)
        except RuntimeError:
            pass
        else:
            raise AssertionError("build_model should fail when all EfficientAD signatures are unsupported")

    assert FakeEfficientAD.calls
    assert all("imagenet_dir" in call for call in FakeEfficientAD.calls)


def test_request_file_parsing():
    adapter = load_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "request.json"
        path.write_text(json.dumps({"mode": "evaluate"}), encoding="utf-8")
        assert adapter.read_json(path)["mode"] == "evaluate"


def test_request_file_parsing_accepts_utf8_bom():
    adapter = load_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "request.json"
        path.write_text("\ufeff" + json.dumps({"mode": "train"}), encoding="utf-8")
        assert adapter.read_json(path)["mode"] == "train"


def test_efficientad_model_size_normalization():
    adapter = load_adapter()
    assert adapter.normalize_efficientad_model_size("s") == "small"
    assert adapter.normalize_efficientad_model_size("small") == "small"
    assert adapter.normalize_efficientad_model_size("m") == "medium"
    assert adapter.normalize_efficientad_model_size("medium") == "medium"
    assert adapter.normalize_efficientad_model_size("") == "small"
    assert adapter.normalize_efficientad_model_size("unexpected") == "small"


def test_train_forces_efficientad_batch_size_one():
    adapter = load_adapter()

    class FakeEngine:
        def fit(self, model=None, datamodule=None):
            return None

    captured = {}

    def fake_build_datamodule(folder_cls, dataset_path, params):
        captured["datamodule_params"] = dict(params)
        return object()

    def fake_build_model(backend, params, patchcore_cls, efficientad_cls):
        captured["model_params"] = dict(params)
        return object()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "dataset"
        output = root / "out"
        imagenet = root / "imagenette"
        checkpoint = output / "weights" / "model.ckpt"
        (dataset / "train" / "good").mkdir(parents=True)
        (dataset / "train" / "good" / "good.png").write_bytes(b"fake")
        imagenet.mkdir()
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")

        adapter.import_anomalib_symbols = lambda: (None, object, object, object, object)
        adapter.build_datamodule = fake_build_datamodule
        adapter.build_model = fake_build_model
        adapter.make_engine = lambda engine_cls, output_path, params: FakeEngine()
        adapter.latest_checkpoint = lambda output_path: checkpoint

        code = adapter.train(
            {
                "backend": "anomalib_efficientad",
                "datasetPath": str(dataset),
                "outputPath": str(output),
                "parameters": {"batchSize": 4, "batch_size": 4, "imagenetDir": str(imagenet)},
            }
        )

        assert code == 0
        assert captured["datamodule_params"]["batchSize"] == 1
        assert captured["datamodule_params"]["batch_size"] == 1
        assert captured["model_params"]["batchSize"] == 1
        assert captured["model_params"]["batch_size"] == 1
        sidecar = json.loads((output / "anomaly_sidecar.json").read_text(encoding="utf-8"))
        assert sidecar["parameters"]["batchSize"] == 1
        assert sidecar["parameters"]["batch_size"] == 1


def test_infer_uses_engine_predict_for_lightning_checkpoint():
    adapter = load_adapter()

    class FakePrediction:
        pred_score = 0.75

    class FakeEngine:
        def predict(self, **kwargs):
            captured["predict_kwargs"] = kwargs
            return [FakePrediction()]

    captured = {}

    def fake_save_prediction_images(image_path, output_path, result):
        heatmap = output_path / "anomaly_heatmap.png"
        overlay = output_path / "anomaly_overlay.png"
        mask = output_path / "anomaly_mask.png"
        heatmap.write_bytes(b"heat")
        overlay.write_bytes(b"overlay")
        mask.write_bytes(b"mask")
        return str(heatmap), str(overlay), str(mask)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "infer"
        model_dir = root / "model"
        checkpoint = model_dir / "model.ckpt"
        sidecar = model_dir / "anomaly_sidecar.json"
        image = root / "sample.png"
        model_dir.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        image.write_bytes(b"fake-image")
        sidecar.write_text(
            json.dumps(
                {
                    "trainingBackend": "anomalib_efficientad",
                    "checkpointPath": str(checkpoint),
                    "threshold": 0.5,
                    "parameters": {"modelSize": "small", "quantile": 0.995},
                }
            ),
            encoding="utf-8",
        )

        adapter.import_anomalib_symbols = lambda: (None, None, object, object, object)
        def fake_build_model(backend, params, patchcore_cls, efficientad_cls):
            captured["backend"] = backend
            captured["model_params"] = params
            return object()

        adapter.build_model = fake_build_model
        adapter.make_engine = lambda engine_cls, output_path, params: FakeEngine()
        adapter.save_prediction_images = fake_save_prediction_images

        code = adapter.infer(
            {
                "modelPath": str(sidecar),
                "imagePath": str(image),
                "outputPath": str(output),
                "options": {"device": "cpu"},
            }
        )

        assert code == 0
        assert captured["backend"] == "anomalib_efficientad"
        assert captured["model_params"]["modelPreset"] == "anomalib_efficientad_s"
        assert captured["predict_kwargs"]["ckpt_path"] == str(checkpoint.resolve())
        assert captured["predict_kwargs"]["data_path"] == str(image.resolve())
        assert captured["predict_kwargs"]["return_predictions"] is True
        report = json.loads((output / "deployment_validation_report.json").read_text(encoding="utf-8"))
        assert report["ok"] is True
        assert report["kind"] == "deployment_validation_report"
        assert report["predictions"][0]["decision"] == "ng"


def test_bundle_exporter_writes_relative_python_runtime_contract():
    exporter_path = ROOT / "python_trainers" / "anomaly" / "anomalib_exporter.py"
    spec = importlib.util.spec_from_file_location("aitrain_anomalib_exporter", exporter_path)
    assert spec and spec.loader
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        training = root / "training"
        package = root / "package"
        training.mkdir()
        checkpoint = training / "source.ckpt"
        checkpoint.write_bytes(b"anomalib-checkpoint")
        sidecar = training / "anomaly_sidecar.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "trainingBackend": "anomalib_patchcore",
                    "checkpointPath": str(checkpoint),
                    "threshold": 0.42,
                    "parameters": {"imageSize": 224},
                }
            ),
            encoding="utf-8",
        )

        assert exporter.run(
            {
                "modelPath": str(checkpoint),
                "sidecarPath": str(sidecar),
                "outputPath": str(package),
            }
        ) == 0

        contract = json.loads((package / "anomaly_sidecar.json").read_text(encoding="utf-8"))
        assert contract["schemaVersion"] == 2
        assert contract["kind"] == "anomalib_bundle"
        assert contract["artifactFormat"] == "anomalib_bundle"
        assert contract["modelFamily"] == "anomaly_detection"
        assert contract["taskType"] == "anomaly_detection"
        assert contract["sourceTrainingBackend"] == "anomalib_patchcore"
        assert contract["runtimeRoutes"] == ["anomalib_python"]
        assert contract["decoder"] == "anomalib_python_sidecar_v1"
        assert contract["exporterVersion"] == "aitrain-anomalib-bundle-exporter-v2"
        assert contract["checkpointPath"] == "model.ckpt"
        assert contract["classNames"] == ["normal", "anomaly"]
        assert contract["preprocessing"]
        assert contract["postprocessing"]
        assert (package / "model.ckpt").read_bytes() == b"anomalib-checkpoint"
        assert (package / "anomalib_export_report.json").is_file()


def test_bundle_sidecar_relative_checkpoint_is_resolved_from_package():
    adapter = load_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp)
        checkpoint = package / "model.ckpt"
        sidecar_path = package / "anomaly_sidecar.json"
        checkpoint.write_bytes(b"checkpoint")
        sidecar = {"checkpointPath": "model.ckpt", "trainingBackend": "anomalib_patchcore"}
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        assert adapter.checkpoint_from_sidecar(sidecar_path, sidecar) == checkpoint.resolve()


if __name__ == "__main__":
    test_adapter_py_compile()
    test_preset_registry_and_defaults()
    test_dataset_inventory_good_only_limited()
    test_training_report_schema_helpers()
    test_metric_extraction_schema_helpers()
    test_datamodule_uses_val_split_when_test_split_is_absent()
    test_datamodule_prefers_test_split_when_available()
    test_datamodule_materializes_mvtec_alias_layout()
    test_efficientad_imagenet_dir_resolution()
    test_efficientad_constructor_never_drops_imagenet_dir()
    test_request_file_parsing()
    test_request_file_parsing_accepts_utf8_bom()
    test_efficientad_model_size_normalization()
    test_train_forces_efficientad_batch_size_one()
    test_infer_uses_engine_predict_for_lightning_checkpoint()
