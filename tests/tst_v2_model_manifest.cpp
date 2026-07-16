#include "aitrain/v2/ModelManifestV2.h"
#include "aitrain/v2/RuntimeAdapterV2.h"
#include "aitrain/v2/OnnxRuntimeAdapterV2.h"
#include "aitrain/v2/RuntimeCapabilityMatrixV2.h"
#include "aitrain/v2/RuntimeInvocationV2.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>

namespace {
aitrain::v2::ModelManifestV2 validManifest()
{
    aitrain::v2::ModelManifestV2 manifest;
    manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    manifest.modelFamily = QStringLiteral("yolo_detection");
    manifest.taskType = QStringLiteral("detection");
    manifest.sourceBackend = QStringLiteral("ultralytics_yolo_detect");
    manifest.sourceTaskId = aitrain::v2::TaskId::create();
    manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
    manifest.sourceArtifactSha256 = QString(64, QLatin1Char('a'));
    manifest.artifactEntryPath = QStringLiteral("model/model.onnx");
    manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 84, -1}}};
    manifest.preprocessing = QJsonObject{{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    manifest.postprocessing = QJsonObject{{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    manifest.decoder = QStringLiteral("yolo_detection_v8");
    manifest.classNames.append(QStringLiteral("part"));
    manifest.opset = 17;
    manifest.exporterVersion = QStringLiteral("ultralytics-8.4.45");
    manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    manifest.verified = true;
    return manifest;
}

QString testPythonExecutable()
{
    const QStringList candidates{
        QDir::current().filePath(QStringLiteral(".deps/python-3.13.13-embed-amd64/python.exe")),
        QStandardPaths::findExecutable(QStringLiteral("python")),
        QStandardPaths::findExecutable(QStringLiteral("python3"))};
    for (const QString& candidate : candidates) {
        if (candidate.isEmpty()) continue;
        QProcess process;
        process.start(candidate, {QStringLiteral("-c"), QStringLiteral("import onnx")});
        if (process.waitForStarted(2000) && process.waitForFinished(10000)
            && process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
            return candidate;
        }
    }
    return {};
}

bool writeTwoClassObbOnnx(const QString& python, const QString& path)
{
    const QString script = QStringLiteral(
        "import sys, onnx\n"
        "from onnx import TensorProto, helper\n"
        "input_info = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 32, 32])\n"
        "output_info = helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 7, 1])\n"
        "values = helper.make_tensor('values', TensorProto.FLOAT, [1, 7, 1], "
        "[16.0, 16.0, 8.0, 4.0, 0.10, 0.90, 0.25])\n"
        "node = helper.make_node('Constant', inputs=[], outputs=['output0'], value=values)\n"
        "graph = helper.make_graph([node], 'aitrain_v2_obb_manifest_classes', [input_info], [output_info])\n"
        "model = helper.make_model(graph, producer_name='aitrain-test', opset_imports=[helper.make_opsetid('', 13)])\n"
        "model.ir_version = 8\n"
        "onnx.save(model, sys.argv[1])\n");
    QProcess process;
    process.start(python, {QStringLiteral("-c"), script, path});
    return process.waitForStarted(2000) && process.waitForFinished(20000)
        && process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0
        && QFileInfo::exists(path);
}
}

class V2ModelManifestTests : public QObject {
    Q_OBJECT
private slots:
    void validManifestRoundTripsAndAllowsDeclaredRuntime();
    void anomalibBundleDoesNotRequireOrPermitOnnxContract();
    void paddleOcrBundleUsesOnlyOfficialPythonRuntime();
    void importDraftAssignsInternalProvenanceFields();
    void missingOrInvalidManifestCannotEnterRuntime();
    void runtimeAdmissionRequiresDeclaredRouteAndUntamperedArtifact();
    void runtimeInvocationCarriesOnlyAdmittedUntamperedModel();
    void runtimeDeliveryProtocolCarriesOnlyModelIdentityAndOptions();
    void modelImportV2ProtocolCarriesDraftInsteadOfRuntimePath();
    void onnxAdapterUsesManifestFamilyInsteadOfShapeGuessing();
    void onnxAdapterUsesManifestClassesForMultiClassObb();
    void runtimeMatrixSeparatesProductSdkAndEvidenceStates();
};

void V2ModelManifestTests::validManifestRoundTripsAndAllowsDeclaredRuntime()
{
    const aitrain::v2::ModelManifestV2 source = validManifest();
    QString error;
    const QJsonObject encoded = aitrain::v2::encodeModelManifestV2(source, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    QCOMPARE(aitrain::v2::modelManifestStatusV2(&encoded, &error), aitrain::v2::ModelManifestStatusV2::Valid);
    QVERIFY2(aitrain::v2::canUseModelManifestV2ForRuntime(&encoded, QStringLiteral("aitrain_onnxruntime"), &error), qPrintable(error));
    QVERIFY(!aitrain::v2::canUseModelManifestV2ForRuntime(&encoded, QStringLiteral("aitrain_ncnn"), &error));
}

void V2ModelManifestTests::anomalibBundleDoesNotRequireOrPermitOnnxContract()
{
    aitrain::v2::ModelManifestV2 manifest = validManifest();
    manifest.modelFamily = QStringLiteral("anomaly_detection");
    manifest.taskType = QStringLiteral("anomaly_detection");
    manifest.sourceBackend = QStringLiteral("anomalib_patchcore");
    manifest.artifactEntryPath = QStringLiteral("export/anomaly_sidecar.json");
    manifest.artifactFormat = QStringLiteral("anomalib_bundle");
    manifest.inputs.clear();
    manifest.outputs.clear();
    manifest.decoder = QStringLiteral("anomalib_python_sidecar_v1");
    manifest.classNames = QStringList{QStringLiteral("normal"), QStringLiteral("anomaly")};
    manifest.opset = 0;
    manifest.exporterVersion = QStringLiteral("aitrain-anomalib-bundle-exporter-v2");
    manifest.runtimeRoutes = QStringList{QStringLiteral("anomalib_python")};
    QString error;
    const QJsonObject encoded = aitrain::v2::encodeModelManifestV2(manifest, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    aitrain::v2::ModelManifestV2 decoded;
    QVERIFY2(aitrain::v2::decodeModelManifestV2(encoded, &decoded, &error), qPrintable(error));
    QCOMPARE(decoded.artifactFormat, QStringLiteral("anomalib_bundle"));
    QVERIFY(decoded.inputs.isEmpty());
    QVERIFY(decoded.outputs.isEmpty());
    QVERIFY2(aitrain::v2::canUseModelManifestV2ForRuntime(&encoded, QStringLiteral("anomalib_python"), &error), qPrintable(error));

    manifest.opset = 17;
    QVERIFY(aitrain::v2::encodeModelManifestV2(manifest, &error).isEmpty());
    manifest.opset = 0;
    manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    QVERIFY(aitrain::v2::encodeModelManifestV2(manifest, &error).isEmpty());
}

void V2ModelManifestTests::paddleOcrBundleUsesOnlyOfficialPythonRuntime()
{
    aitrain::v2::ModelManifestV2 manifest = validManifest();
    manifest.modelFamily = QStringLiteral("ocr_recognition");
    manifest.taskType = QStringLiteral("ocr_recognition");
    manifest.sourceBackend = QStringLiteral("paddleocr_rec_official");
    manifest.artifactEntryPath = QStringLiteral("export/paddleocr_bundle.json");
    manifest.artifactFormat = QStringLiteral("paddleocr_inference_bundle");
    manifest.inputs.clear();
    manifest.outputs.clear();
    manifest.decoder = QStringLiteral("paddleocr_official_rec_v1");
    manifest.classNames = QStringList{QStringLiteral("text")};
    manifest.opset = 0;
    manifest.exporterVersion = QStringLiteral("aitrain-paddleocr-exporter-v2");
    manifest.runtimeRoutes = QStringList{QStringLiteral("paddleocr_official")};
    QString error;
    const QJsonObject encoded = aitrain::v2::encodeModelManifestV2(manifest, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    QVERIFY2(aitrain::v2::canUseModelManifestV2ForRuntime(
        &encoded, QStringLiteral("paddleocr_official"), &error), qPrintable(error));
    manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    QVERIFY(aitrain::v2::encodeModelManifestV2(manifest, &error).isEmpty());
}

void V2ModelManifestTests::importDraftAssignsInternalProvenanceFields()
{
    const aitrain::v2::ModelManifestV2 source = validManifest();
    QJsonObject draft = aitrain::v2::encodeModelManifestV2(source);
    draft.remove(QStringLiteral("sourceTaskId"));
    draft.remove(QStringLiteral("sourceArtifactSha256"));
    aitrain::v2::ModelManifestV2 decoded;
    QString error;
    QVERIFY2(aitrain::v2::decodeModelManifestImportDraftV2(draft, &decoded, &error), qPrintable(error));
    QVERIFY(decoded.sourceTaskId.isValid());
    QCOMPARE(decoded.sourceArtifactSha256, QString(64, QLatin1Char('0')));
}

void V2ModelManifestTests::missingOrInvalidManifestCannotEnterRuntime()
{
    QString error;
    QCOMPARE(aitrain::v2::modelManifestStatusV2(nullptr, &error), aitrain::v2::ModelManifestStatusV2::Unclassified);
    QVERIFY(!aitrain::v2::canUseModelManifestV2ForRuntime(nullptr, QStringLiteral("aitrain_onnxruntime"), &error));
    QJsonObject invalid = aitrain::v2::encodeModelManifestV2(validManifest(), &error);
    invalid.insert(QStringLiteral("verified"), false);
    QCOMPARE(aitrain::v2::modelManifestStatusV2(&invalid, &error), aitrain::v2::ModelManifestStatusV2::Invalid);
}

void V2ModelManifestTests::runtimeAdmissionRequiresDeclaredRouteAndUntamperedArtifact()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString entryPath = directory.filePath(QStringLiteral("model.onnx"));
    QFile entry(entryPath);
    QVERIFY(entry.open(QIODevice::WriteOnly));
    QVERIFY(entry.write("official-model") > 0);
    entry.close();

    aitrain::v2::RuntimeModelLocationV2 location;
    location.manifest = validManifest();
    location.manifest.artifactEntryPath = QStringLiteral("model.onnx");
    location.manifest.sourceArtifactSha256 = QString::fromLatin1(QCryptographicHash::hash("official-model", QCryptographicHash::Sha256).toHex());
    location.artifactDirectory = directory.path();
    QCOMPARE(aitrain::v2::validateRuntimeModelV2(location, QStringLiteral("aitrain_onnxruntime")).status,
        aitrain::v2::RuntimeStatusV2::Available);
    QCOMPARE(aitrain::v2::validateRuntimeModelV2(location, QStringLiteral("aitrain_ncnn")).status,
        aitrain::v2::RuntimeStatusV2::ArtifactIncompatible);

    QVERIFY(entry.open(QIODevice::Append));
    QVERIFY(entry.write("tamper") > 0);
    entry.close();
    QCOMPARE(aitrain::v2::validateRuntimeModelV2(location, QStringLiteral("aitrain_onnxruntime")).status,
        aitrain::v2::RuntimeStatusV2::ArtifactIncompatible);
}

void V2ModelManifestTests::runtimeInvocationCarriesOnlyAdmittedUntamperedModel()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString entryPath = directory.filePath(QStringLiteral("model.onnx"));
    QFile entry(entryPath);
    QVERIFY(entry.open(QIODevice::WriteOnly));
    QVERIFY(entry.write("registered-model") > 0);
    entry.close();

    aitrain::v2::RuntimeInvocationV2 invocation;
    invocation.model.manifest = validManifest();
    invocation.model.manifest.artifactEntryPath = QStringLiteral("model.onnx");
    invocation.model.manifest.sourceArtifactSha256 = QString::fromLatin1(QCryptographicHash::hash("registered-model", QCryptographicHash::Sha256).toHex());
    invocation.model.artifactDirectory = directory.path();
    invocation.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    invocation.imagePath = QStringLiteral("sample.png");
    invocation.outputPath = directory.filePath(QStringLiteral("output"));
    invocation.options = QJsonObject{{QStringLiteral("confidenceThreshold"), 0.25}};

    QString error;
    const QJsonObject encoded = aitrain::v2::encodeRuntimeInvocationV2(invocation, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    aitrain::v2::RuntimeInvocationV2 decoded;
    QVERIFY2(aitrain::v2::decodeRuntimeInvocationV2(encoded, &decoded, &error), qPrintable(error));
    QCOMPARE(decoded.model.manifest.modelPackageId.toString(), invocation.model.manifest.modelPackageId.toString());
    QCOMPARE(decoded.model.artifactDirectory, QDir::cleanPath(directory.path()));

    QVERIFY(entry.open(QIODevice::Append));
    QVERIFY(entry.write("tamper") > 0);
    entry.close();
    QVERIFY(!aitrain::v2::decodeRuntimeInvocationV2(encoded, &decoded, &error));
}

void V2ModelManifestTests::runtimeDeliveryProtocolCarriesOnlyModelIdentityAndOptions()
{
    QJsonObject options;
    options.insert(QStringLiteral("benchmarkIterations"), 3);
    const QJsonObject payload = aitrain::worker_protocol::runtimeDeliveryWorkflowV2Request(
        QStringLiteral("task-v2"), QStringLiteral("C:/project"), QStringLiteral("package-v2"),
        QStringLiteral("aitrain_onnxruntime"), QStringLiteral("C:/samples/part.png"), options);
    QCOMPARE(payload.value(QStringLiteral("taskId")).toString(), QStringLiteral("task-v2"));
    QCOMPARE(payload.value(QStringLiteral("projectRoot")).toString(), QStringLiteral("C:/project"));
    QCOMPARE(payload.value(QStringLiteral("modelPackageId")).toString(), QStringLiteral("package-v2"));
    QCOMPARE(payload.value(QStringLiteral("runtimeRoute")).toString(), QStringLiteral("aitrain_onnxruntime"));
    QCOMPARE(payload.value(QStringLiteral("sampleImagePath")).toString(), QStringLiteral("C:/samples/part.png"));
    QCOMPARE(payload.value(QStringLiteral("options")).toObject(), options);
    QVERIFY(!payload.contains(QStringLiteral("runtimeInvocation")));
    QVERIFY(!payload.contains(QStringLiteral("modelPath")));
    QVERIFY(!payload.contains(QStringLiteral("checkpointPath")));
}

void V2ModelManifestTests::modelImportV2ProtocolCarriesDraftInsteadOfRuntimePath()
{
    const QJsonObject draft{{QStringLiteral("schemaVersion"), 2}, {QStringLiteral("verified"), true}};
    const QJsonObject payload = aitrain::worker_protocol::modelImportV2Request(
        QStringLiteral("task-import"), QStringLiteral("C:/project"), QStringLiteral("C:/external/model.onnx"), draft);
    QCOMPARE(payload.value(QStringLiteral("taskId")).toString(), QStringLiteral("task-import"));
    QCOMPARE(payload.value(QStringLiteral("projectRoot")).toString(), QStringLiteral("C:/project"));
    QCOMPARE(payload.value(QStringLiteral("sourceFilePath")).toString(), QStringLiteral("C:/external/model.onnx"));
    QCOMPARE(payload.value(QStringLiteral("manifestDraft")).toObject(), draft);
    QVERIFY(!payload.contains(QStringLiteral("runtimeInvocation")));
}

void V2ModelManifestTests::onnxAdapterUsesManifestFamilyInsteadOfShapeGuessing()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    QFile entry(directory.filePath(QStringLiteral("model.onnx")));
    QVERIFY(entry.open(QIODevice::WriteOnly));
    QVERIFY(entry.write("manifest-only-routing") > 0);
    entry.close();
    aitrain::v2::RuntimeModelLocationV2 location;
    location.manifest = validManifest();
    location.manifest.artifactEntryPath = QStringLiteral("model.onnx");
    location.manifest.sourceArtifactSha256 = QString::fromLatin1(QCryptographicHash::hash("manifest-only-routing", QCryptographicHash::Sha256).toHex());
    location.manifest.modelFamily = QStringLiteral("ocr_detection");
    location.manifest.decoder = QStringLiteral("paddle_db");
    location.artifactDirectory = directory.path();
    aitrain::v2::OnnxRuntimeAdapterV2 adapter;
    QCOMPARE(adapter.probe(location).status, aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
    const aitrain::v2::RuntimeOperationResultV2 result = adapter.infer(location,
        QJsonObject{{QStringLiteral("imagePath"), QStringLiteral("unused.png")}});
    QCOMPARE(result.status, aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
}

void V2ModelManifestTests::onnxAdapterUsesManifestClassesForMultiClassObb()
{
    if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
        QSKIP("ONNX Runtime is not enabled in this build.");
    }
    const QString python = testPythonExecutable();
    if (python.isEmpty()) {
        QSKIP("Python with the onnx package is unavailable.");
    }

    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString modelPath = directory.filePath(QStringLiteral("model.onnx"));
    QVERIFY2(writeTwoClassObbOnnx(python, modelPath), "Unable to create the two-class OBB ONNX fixture.");
    const QString imagePath = directory.filePath(QStringLiteral("sample.png"));
    QImage image(32, 32, QImage::Format_RGB888);
    image.fill(Qt::white);
    QVERIFY(image.save(imagePath));

    QFile modelFile(modelPath);
    QVERIFY(modelFile.open(QIODevice::ReadOnly));
    const QByteArray modelBytes = modelFile.readAll();
    modelFile.close();

    aitrain::v2::RuntimeModelLocationV2 location;
    location.manifest = validManifest();
    location.manifest.modelFamily = QStringLiteral("yolo_obb");
    location.manifest.taskType = QStringLiteral("obb");
    location.manifest.sourceBackend = QStringLiteral("ultralytics_yolo_export");
    location.manifest.sourceArtifactSha256 = QString::fromLatin1(
        QCryptographicHash::hash(modelBytes, QCryptographicHash::Sha256).toHex());
    location.manifest.artifactEntryPath = QStringLiteral("model.onnx");
    location.manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 32, 32}}};
    location.manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 7, 1}}};
    location.manifest.postprocessing = QJsonObject{{QStringLiteral("id"), QStringLiteral("yolo_obb_nms")}};
    location.manifest.decoder = QStringLiteral("yolo_obb_v8");
    location.manifest.classNames = QStringList()
        << QStringLiteral("ship") << QStringLiteral("plane");
    location.artifactDirectory = directory.path();

    const QString outputPath = directory.filePath(QStringLiteral("output"));
    aitrain::v2::OnnxRuntimeAdapterV2 adapter;
    const aitrain::v2::RuntimeOperationResultV2 result = adapter.infer(location,
        QJsonObject{{QStringLiteral("imagePath"), imagePath}, {QStringLiteral("outputPath"), outputPath},
            {QStringLiteral("options"), QJsonObject{{QStringLiteral("confidenceThreshold"), 0.25}}}});
    QCOMPARE(result.status, aitrain::v2::RuntimeStatusV2::Available);
    QCOMPARE(result.details.value(QStringLiteral("predictionCount")).toInt(), 1);

    QFile predictions(result.details.value(QStringLiteral("predictionsPath")).toString());
    QVERIFY(predictions.open(QIODevice::ReadOnly));
    const QJsonObject report = QJsonDocument::fromJson(predictions.readAll()).object();
    const QJsonArray values = report.value(QStringLiteral("predictions")).toArray();
    QCOMPARE(values.size(), 1);
    const QJsonObject prediction = values.first().toObject();
    QCOMPARE(prediction.value(QStringLiteral("taskType")).toString(), QStringLiteral("obb_detection"));
    QCOMPARE(prediction.value(QStringLiteral("classId")).toInt(), 1);
    QCOMPARE(prediction.value(QStringLiteral("className")).toString(), QStringLiteral("plane"));
    QCOMPARE(prediction.value(QStringLiteral("xywhr")).toArray().size(), 5);
    QVERIFY(QFileInfo::exists(result.details.value(QStringLiteral("overlayPath")).toString()));
}

void V2ModelManifestTests::runtimeMatrixSeparatesProductSdkAndEvidenceStates()
{
    aitrain::v2::RuntimeCapabilityMatrixV2 matrix;
    const aitrain::v2::RuntimeCapabilityV2 obbNcnn = matrix.query({QStringLiteral("yolo_obb"), QStringLiteral("aitrain_ncnn")});
    QCOMPARE(obbNcnn.status, aitrain::v2::RuntimeCapabilityStatusV2::UnsupportedByProduct);
    const aitrain::v2::RuntimeCapabilityV2 anomaly = matrix.query({QStringLiteral("anomaly_detection"), QStringLiteral("anomalib_python")});
    QCOMPARE(anomaly.status, aitrain::v2::RuntimeCapabilityStatusV2::RequiresExternalEvidence);
    const aitrain::v2::RuntimeCapabilityV2 unsupported = matrix.query({QStringLiteral("ocr_recognition"), QStringLiteral("aitrain_onnxruntime")});
    QCOMPARE(unsupported.status, aitrain::v2::RuntimeCapabilityStatusV2::UnsupportedByProduct);
    QVERIFY(!matrix.toJson().value(QStringLiteral("entries")).toArray().isEmpty());
}

QTEST_MAIN(V2ModelManifestTests)
#include "tst_v2_model_manifest.moc"
