#include "aitrain/v2/ProjectWorkspaceV2.h"

#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>

namespace {

class FixtureRuntimeAdapter final : public aitrain::v2::RuntimeAdapterV2 {
public:
    explicit FixtureRuntimeAdapter(aitrain::v2::RuntimeStatusV2 probeStatus)
        : probeStatus_(probeStatus) {}

    QString runtimeRoute() const override { return QStringLiteral("aitrain_onnxruntime"); }
    aitrain::v2::RuntimeOperationResultV2 validateModel(
        const aitrain::v2::RuntimeModelLocationV2&) const override
    {
        return {aitrain::v2::RuntimeStatusV2::Available, QStringLiteral("测试 Manifest 有效。"), {}};
    }
    aitrain::v2::RuntimeOperationResultV2 probe(
        const aitrain::v2::RuntimeModelLocationV2&) const override
    {
        return {probeStatus_, probeStatus_ == aitrain::v2::RuntimeStatusV2::Available
                ? QStringLiteral("测试 Runtime 可用。") : QStringLiteral("测试 SDK 缺失。"), {}};
    }
    aitrain::v2::RuntimeOperationResultV2 infer(
        const aitrain::v2::RuntimeModelLocationV2&, const QJsonObject& request) const override
    {
        return writeOperation(request);
    }
    aitrain::v2::RuntimeOperationResultV2 benchmark(
        const aitrain::v2::RuntimeModelLocationV2&, const QJsonObject& request) const override
    {
        return writeOperation(request);
    }
    aitrain::v2::RuntimeOperationResultV2 deploymentValidate(
        const aitrain::v2::RuntimeModelLocationV2&, const QJsonObject& request) const override
    {
        return writeOperation(request);
    }

private:
    static aitrain::v2::RuntimeOperationResultV2 writeOperation(const QJsonObject& request)
    {
        const QString outputPath = request.value(QStringLiteral("outputPath")).toString();
        if (!QDir().mkpath(outputPath)) {
            return {aitrain::v2::RuntimeStatusV2::ArtifactIncompatible,
                QStringLiteral("测试输出目录创建失败。"), {}};
        }
        const QString predictions = QDir(outputPath).filePath(QStringLiteral("predictions.json"));
        const QString overlay = QDir(outputPath).filePath(QStringLiteral("overlay.png"));
        QFile predictionFile(predictions);
        if (!predictionFile.open(QIODevice::WriteOnly)
            || predictionFile.write("{\"detections\":[]}") <= 0
            || !QImage(8, 8, QImage::Format_RGB32).save(overlay)) {
            return {aitrain::v2::RuntimeStatusV2::ArtifactIncompatible,
                QStringLiteral("测试输出写入失败。"), {}};
        }
        predictionFile.close();
        return {aitrain::v2::RuntimeStatusV2::Available, QStringLiteral("测试推理成功。"),
            {{QStringLiteral("predictionsPath"), predictions},
                {QStringLiteral("overlayPath"), overlay}, {QStringLiteral("elapsedMs"), 2.0}}};
    }

    aitrain::v2::RuntimeStatusV2 probeStatus_;
};

bool writeFile(const QString& path, const QByteArray& bytes)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}

aitrain::v2::ModelImportResultV2 importFixtureModel(
    aitrain::v2::ProjectWorkspaceV2* workspace, const QString& sourcePath, QString* error)
{
    aitrain::v2::ModelImportRequestV2 request;
    request.taskId = aitrain::v2::TaskId::create();
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("workflow_fixture");
    request.manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
    request.manifest.artifactEntryPath = QStringLiteral("model/model.onnx");
    request.manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 32, 32}}};
    request.manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 5, 1}}};
    request.manifest.preprocessing = {{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    request.manifest.postprocessing = {{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    request.manifest.decoder = QStringLiteral("yolo_detection_v8");
    request.manifest.classNames.append(QStringLiteral("part"));
    request.manifest.opset = 17;
    request.manifest.exporterVersion = QStringLiteral("workflow-fixture");
    request.manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    request.manifest.verified = true;
    aitrain::v2::ModelImportResultV2 imported;
    workspace->importModel(request, &imported, error);
    return imported;
}

bool evidenceHasAllFormats(const aitrain::v2::EvidenceArtifactBundleV2& evidence)
{
    const QStringList names{QStringLiteral("evidence.json"), QStringLiteral("evidence.md"),
        QStringLiteral("evidence.html"), QStringLiteral("model_card.json")};
    for (const QString& name : names) {
        if (!QFileInfo::exists(evidence.pathsByKind.value(name))) return false;
    }
    return true;
}

} // namespace

class V2RuntimeDeliveryWorkflowTests : public QObject {
    Q_OBJECT
private slots:
    void successCommitsSixChainedStepsAndFourEvidenceFormats();
    void cancellationProducesOneTerminalStateAndEvidence();
    void sdkMissingProducesPreciseFailureAndEvidence();
};

void V2RuntimeDeliveryWorkflowTests::successCommitsSixChainedStepsAndFourEvidenceFormats()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString modelPath = directory.filePath(QStringLiteral("fixture.onnx"));
    const QString imagePath = directory.filePath(QStringLiteral("sample.png"));
    QVERIFY(writeFile(modelPath, "fixture-model"));
    QVERIFY(QImage(32, 32, QImage::Format_RGB32).save(imagePath));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("workspace")), &error), qPrintable(error));
    const auto imported = importFixtureModel(&workspace, modelPath, &error);
    QVERIFY2(imported.modelPackage.manifest.modelPackageId.isValid(), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("runtime.delivery"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    aitrain::v2::RuntimeDeliveryWorkflowRequestV2 request;
    request.modelPackageId = imported.modelPackage.manifest.modelPackageId;
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    request.sampleImagePath = imagePath;
    request.options.insert(QStringLiteral("benchmarkWarmup"), 1);
    request.options.insert(QStringLiteral("benchmarkIterations"), 3);
    aitrain::v2::RuntimeDeliveryWorkflowResultV2 result;
    QVERIFY2(workspace.runRuntimeDeliveryWorkflow(taskId, request, &result, &error, {},
        [](const QString&) { return std::make_unique<FixtureRuntimeAdapter>(aitrain::v2::RuntimeStatusV2::Available); }), qPrintable(error));
    QCOMPARE(result.state, aitrain::v2::WorkflowStepState::Succeeded);
    QCOMPARE(result.runtimeStatus, aitrain::v2::RuntimeStatusV2::Available);
    QVERIFY(evidenceHasAllFormats(result.evidence));
    const auto steps = workspace.workflowSteps(result.workflowRunId, &error);
    QCOMPARE(steps.size(), 6);
    aitrain::v2::StorageV2 lineageStorage;
    QVERIFY2(lineageStorage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")),
        &error), qPrintable(error));
    aitrain::v2::WorkflowInputBindingV2 modelInput;
    QVERIFY2(lineageStorage.workflowInput(result.workflowRunId, QStringLiteral("model_package"),
        &modelInput, &error), qPrintable(error));
    QCOMPARE(modelInput.modelPackageId, imported.modelPackage.manifest.modelPackageId);
    QCOMPARE(modelInput.sourceArtifactId, imported.modelPackage.sourceArtifactId);
    QCOMPARE(modelInput.sourceTaskId, imported.modelPackage.manifest.sourceTaskId);
    aitrain::v2::ArtifactId previous = imported.modelPackage.sourceArtifactId;
    for (const auto& step : steps) {
        QCOMPARE(step.state, aitrain::v2::WorkflowStepState::Succeeded);
        QCOMPARE(step.inputArtifactId, previous);
        QVERIFY(step.outputArtifactId.isValid());
        previous = step.outputArtifactId;
    }
    QFile evidenceFile(result.evidence.pathsByKind.value(QStringLiteral("evidence.json")));
    QVERIFY(evidenceFile.open(QIODevice::ReadOnly));
    const QJsonObject evidence = QJsonDocument::fromJson(evidenceFile.readAll()).object();
    QCOMPARE(evidence.value(QStringLiteral("benchmark")).toObject()
        .value(QStringLiteral("benchmarkKind")).toString(), QStringLiteral("smoke_timing"));
    QCOMPARE(evidence.value(QStringLiteral("benchmark")).toObject()
        .value(QStringLiteral("measuredIterations")).toInt(), 3);
    QCOMPARE(evidence.value(QStringLiteral("runtimeStatus")).toObject()
        .value(QStringLiteral("modelProducerTaskId")).toString(),
        imported.modelPackage.manifest.sourceTaskId.toString());
}

void V2RuntimeDeliveryWorkflowTests::cancellationProducesOneTerminalStateAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString modelPath = directory.filePath(QStringLiteral("fixture.onnx"));
    const QString imagePath = directory.filePath(QStringLiteral("sample.png"));
    QVERIFY(writeFile(modelPath, "fixture-model"));
    QVERIFY(QImage(32, 32, QImage::Format_RGB32).save(imagePath));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("workspace")), &error), qPrintable(error));
    const auto imported = importFixtureModel(&workspace, modelPath, &error);
    QVERIFY2(imported.modelPackage.manifest.modelPackageId.isValid(), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("runtime.delivery"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    aitrain::v2::RuntimeDeliveryWorkflowResultV2 result;
    aitrain::v2::RuntimeDeliveryWorkflowRequestV2 request;
    request.modelPackageId = imported.modelPackage.manifest.modelPackageId;
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    request.sampleImagePath = imagePath;
    QVERIFY2(workspace.runRuntimeDeliveryWorkflow(taskId, request,
        &result, &error, []() { return true; }), qPrintable(error));
    QCOMPARE(result.state, aitrain::v2::WorkflowStepState::Canceled);
    QCOMPARE(result.failure.code, aitrain::v2::FailureCode::Canceled);
    QVERIFY(evidenceHasAllFormats(result.evidence));
    QCOMPARE(workspace.workflowRunsForTask(taskId, &error).size(), 1);
    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(workspace.task(taskId, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::v2::TaskState::Canceled);
    QCOMPARE(stored.failure.code, aitrain::v2::FailureCode::Canceled);
    const auto steps = workspace.workflowSteps(result.workflowRunId, &error);
    QCOMPARE(steps.size(), 6);
    QCOMPARE(steps.first().state, aitrain::v2::WorkflowStepState::Canceled);
    for (int index = 1; index < steps.size(); ++index) {
        QCOMPARE(steps.at(index).state, aitrain::v2::WorkflowStepState::Skipped);
    }
}

void V2RuntimeDeliveryWorkflowTests::sdkMissingProducesPreciseFailureAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString modelPath = directory.filePath(QStringLiteral("fixture.onnx"));
    const QString imagePath = directory.filePath(QStringLiteral("sample.png"));
    QVERIFY(writeFile(modelPath, "fixture-model"));
    QVERIFY(QImage(32, 32, QImage::Format_RGB32).save(imagePath));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("workspace")), &error), qPrintable(error));
    const auto imported = importFixtureModel(&workspace, modelPath, &error);
    QVERIFY2(imported.modelPackage.manifest.modelPackageId.isValid(), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("runtime.delivery"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    aitrain::v2::RuntimeDeliveryWorkflowResultV2 result;
    aitrain::v2::RuntimeDeliveryWorkflowRequestV2 request;
    request.modelPackageId = imported.modelPackage.manifest.modelPackageId;
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    request.sampleImagePath = imagePath;
    QVERIFY2(workspace.runRuntimeDeliveryWorkflow(taskId, request,
        &result, &error, {}, [](const QString&) {
            return std::make_unique<FixtureRuntimeAdapter>(aitrain::v2::RuntimeStatusV2::SdkMissing);
        }), qPrintable(error));
    QCOMPARE(result.state, aitrain::v2::WorkflowStepState::Failed);
    QCOMPARE(result.runtimeStatus, aitrain::v2::RuntimeStatusV2::SdkMissing);
    QCOMPARE(result.failure.code, aitrain::v2::FailureCode::SdkMissing);
    QVERIFY(evidenceHasAllFormats(result.evidence));
    QCOMPARE(workspace.workflowRunsForTask(taskId, &error).size(), 1);
    QFile evidenceFile(result.evidence.pathsByKind.value(QStringLiteral("evidence.json")));
    QVERIFY(evidenceFile.open(QIODevice::ReadOnly));
    const QJsonObject evidence = QJsonDocument::fromJson(evidenceFile.readAll()).object();
    QCOMPARE(evidence.value(QStringLiteral("runtimeStatus")).toObject()
        .value(QStringLiteral("status")).toString(), QStringLiteral("sdk_missing"));
    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(workspace.task(taskId, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::v2::TaskState::Failed);
    QCOMPARE(stored.failure.code, aitrain::v2::FailureCode::SdkMissing);
}

QTEST_MAIN(V2RuntimeDeliveryWorkflowTests)
#include "tst_v2_runtime_delivery_workflow.moc"
