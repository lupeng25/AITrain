#include "aitrain/workflow/ProjectWorkspace.h"

#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>

namespace {

class FixtureRuntimeAdapter final : public aitrain::RuntimeAdapter {
public:
    explicit FixtureRuntimeAdapter(aitrain::RuntimeStatus probeStatus)
        : probeStatus_(probeStatus) {}

    QString runtimeRoute() const override { return QStringLiteral("aitrain_onnxruntime"); }
    aitrain::RuntimeOperationResult validateModel(
        const aitrain::RuntimeModelLocation&) const override
    {
        return {aitrain::RuntimeStatus::Available, QStringLiteral("测试 Manifest 有效。"), {}};
    }
    aitrain::RuntimeOperationResult probe(
        const aitrain::RuntimeModelLocation&) const override
    {
        return {probeStatus_, probeStatus_ == aitrain::RuntimeStatus::Available
                ? QStringLiteral("测试 Runtime 可用。") : QStringLiteral("测试 SDK 缺失。"), {}};
    }
    aitrain::RuntimeOperationResult infer(
        const aitrain::RuntimeModelLocation&, const QJsonObject& request) const override
    {
        return writeOperation(request);
    }
    aitrain::RuntimeOperationResult benchmark(
        const aitrain::RuntimeModelLocation&, const QJsonObject& request) const override
    {
        return writeOperation(request);
    }
    aitrain::RuntimeOperationResult deploymentValidate(
        const aitrain::RuntimeModelLocation&, const QJsonObject& request) const override
    {
        return writeOperation(request);
    }

private:
    static aitrain::RuntimeOperationResult writeOperation(const QJsonObject& request)
    {
        const QString outputPath = request.value(QStringLiteral("outputPath")).toString();
        if (!QDir().mkpath(outputPath)) {
            return {aitrain::RuntimeStatus::ArtifactIncompatible,
                QStringLiteral("测试输出目录创建失败。"), {}};
        }
        const QString predictions = QDir(outputPath).filePath(QStringLiteral("predictions.json"));
        const QString overlay = QDir(outputPath).filePath(QStringLiteral("overlay.png"));
        QFile predictionFile(predictions);
        if (!predictionFile.open(QIODevice::WriteOnly)
            || predictionFile.write("{\"detections\":[]}") <= 0
            || !QImage(8, 8, QImage::Format_RGB32).save(overlay)) {
            return {aitrain::RuntimeStatus::ArtifactIncompatible,
                QStringLiteral("测试输出写入失败。"), {}};
        }
        predictionFile.close();
        return {aitrain::RuntimeStatus::Available, QStringLiteral("测试推理成功。"),
            {{QStringLiteral("predictionsPath"), predictions},
                {QStringLiteral("overlayPath"), overlay}, {QStringLiteral("elapsedMs"), 2.0}}};
    }

    aitrain::RuntimeStatus probeStatus_;
};

bool writeFile(const QString& path, const QByteArray& bytes)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}

aitrain::ModelImportResult importFixtureModel(
    aitrain::ProjectWorkspace* workspace, const QString& sourcePath, QString* error)
{
    aitrain::ModelImportRequest request;
    request.taskId = aitrain::TaskId::create();
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("workflow_fixture");
    request.manifest.sourceSnapshotId = aitrain::SnapshotId::create();
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
    aitrain::ModelImportResult imported;
    workspace->importModel(request, &imported, error);
    return imported;
}

bool evidenceHasAllFormats(const aitrain::EvidenceArtifactBundle& evidence)
{
    const QStringList names{QStringLiteral("evidence.json"), QStringLiteral("evidence.md"),
        QStringLiteral("evidence.html"), QStringLiteral("model_card.json")};
    for (const QString& name : names) {
        if (!QFileInfo::exists(evidence.pathsByKind.value(name))) return false;
    }
    return true;
}

} // namespace

class RuntimeDeliveryWorkflowTests : public QObject {
    Q_OBJECT
private slots:
    void successCommitsSixChainedStepsAndFourEvidenceFormats();
    void cancellationProducesOneTerminalStateAndEvidence();
    void sdkMissingProducesPreciseFailureAndEvidence();
};

void RuntimeDeliveryWorkflowTests::successCommitsSixChainedStepsAndFourEvidenceFormats()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString modelPath = directory.filePath(QStringLiteral("fixture.onnx"));
    const QString imagePath = directory.filePath(QStringLiteral("sample.png"));
    QVERIFY(writeFile(modelPath, "fixture-model"));
    QVERIFY(QImage(32, 32, QImage::Format_RGB32).save(imagePath));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("workspace")), &error), qPrintable(error));
    const auto imported = importFixtureModel(&workspace, modelPath, &error);
    QVERIFY2(imported.modelPackage.manifest.modelPackageId.isValid(), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("runtime.delivery"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    aitrain::RuntimeDeliveryWorkflowRequest request;
    request.modelPackageId = imported.modelPackage.manifest.modelPackageId;
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    request.sampleImagePath = imagePath;
    request.options.insert(QStringLiteral("benchmarkWarmup"), 1);
    request.options.insert(QStringLiteral("benchmarkIterations"), 3);
    aitrain::RuntimeDeliveryWorkflowResult result;
    QVERIFY2(workspace.runRuntimeDeliveryWorkflow(taskId, request, &result, &error, {},
        [](const QString&) { return std::make_unique<FixtureRuntimeAdapter>(aitrain::RuntimeStatus::Available); }), qPrintable(error));
    QCOMPARE(result.state, aitrain::WorkflowStepState::Succeeded);
    QCOMPARE(result.runtimeStatus, aitrain::RuntimeStatus::Available);
    QVERIFY(evidenceHasAllFormats(result.evidence));
    const auto steps = workspace.workflowSteps(result.workflowRunId, &error);
    QCOMPARE(steps.size(), 6);
    aitrain::ProjectStore lineageStorage;
    QVERIFY2(lineageStorage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")),
        &error), qPrintable(error));
    aitrain::WorkflowInputBinding modelInput;
    QVERIFY2(lineageStorage.workflowInput(result.workflowRunId, QStringLiteral("model_package"),
        &modelInput, &error), qPrintable(error));
    QCOMPARE(modelInput.modelPackageId, imported.modelPackage.manifest.modelPackageId);
    QCOMPARE(modelInput.sourceArtifactId, imported.modelPackage.sourceArtifactId);
    QCOMPARE(modelInput.sourceTaskId, imported.modelPackage.manifest.sourceTaskId);
    aitrain::ArtifactId previous = imported.modelPackage.sourceArtifactId;
    for (const auto& step : steps) {
        QCOMPARE(step.state, aitrain::WorkflowStepState::Succeeded);
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

void RuntimeDeliveryWorkflowTests::cancellationProducesOneTerminalStateAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString modelPath = directory.filePath(QStringLiteral("fixture.onnx"));
    const QString imagePath = directory.filePath(QStringLiteral("sample.png"));
    QVERIFY(writeFile(modelPath, "fixture-model"));
    QVERIFY(QImage(32, 32, QImage::Format_RGB32).save(imagePath));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("workspace")), &error), qPrintable(error));
    const auto imported = importFixtureModel(&workspace, modelPath, &error);
    QVERIFY2(imported.modelPackage.manifest.modelPackageId.isValid(), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("runtime.delivery"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    aitrain::RuntimeDeliveryWorkflowResult result;
    aitrain::RuntimeDeliveryWorkflowRequest request;
    request.modelPackageId = imported.modelPackage.manifest.modelPackageId;
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    request.sampleImagePath = imagePath;
    QVERIFY2(workspace.runRuntimeDeliveryWorkflow(taskId, request,
        &result, &error, []() { return true; }), qPrintable(error));
    QCOMPARE(result.state, aitrain::WorkflowStepState::Canceled);
    QCOMPARE(result.failure.code, aitrain::FailureCode::Canceled);
    QVERIFY(evidenceHasAllFormats(result.evidence));
    QCOMPARE(workspace.workflowRunsForTask(taskId, &error).size(), 1);
    aitrain::TaskSnapshot stored;
    QVERIFY2(workspace.task(taskId, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Canceled);
    QCOMPARE(stored.failure.code, aitrain::FailureCode::Canceled);
    const auto steps = workspace.workflowSteps(result.workflowRunId, &error);
    QCOMPARE(steps.size(), 6);
    QCOMPARE(steps.first().state, aitrain::WorkflowStepState::Canceled);
    for (int index = 1; index < steps.size(); ++index) {
        QCOMPARE(steps.at(index).state, aitrain::WorkflowStepState::Skipped);
    }
}

void RuntimeDeliveryWorkflowTests::sdkMissingProducesPreciseFailureAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString modelPath = directory.filePath(QStringLiteral("fixture.onnx"));
    const QString imagePath = directory.filePath(QStringLiteral("sample.png"));
    QVERIFY(writeFile(modelPath, "fixture-model"));
    QVERIFY(QImage(32, 32, QImage::Format_RGB32).save(imagePath));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("workspace")), &error), qPrintable(error));
    const auto imported = importFixtureModel(&workspace, modelPath, &error);
    QVERIFY2(imported.modelPackage.manifest.modelPackageId.isValid(), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("runtime.delivery"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    aitrain::RuntimeDeliveryWorkflowResult result;
    aitrain::RuntimeDeliveryWorkflowRequest request;
    request.modelPackageId = imported.modelPackage.manifest.modelPackageId;
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    request.sampleImagePath = imagePath;
    QVERIFY2(workspace.runRuntimeDeliveryWorkflow(taskId, request,
        &result, &error, {}, [](const QString&) {
            return std::make_unique<FixtureRuntimeAdapter>(aitrain::RuntimeStatus::SdkMissing);
        }), qPrintable(error));
    QCOMPARE(result.state, aitrain::WorkflowStepState::Failed);
    QCOMPARE(result.runtimeStatus, aitrain::RuntimeStatus::SdkMissing);
    QCOMPARE(result.failure.code, aitrain::FailureCode::SdkMissing);
    QVERIFY(evidenceHasAllFormats(result.evidence));
    QCOMPARE(workspace.workflowRunsForTask(taskId, &error).size(), 1);
    QFile evidenceFile(result.evidence.pathsByKind.value(QStringLiteral("evidence.json")));
    QVERIFY(evidenceFile.open(QIODevice::ReadOnly));
    const QJsonObject evidence = QJsonDocument::fromJson(evidenceFile.readAll()).object();
    QCOMPARE(evidence.value(QStringLiteral("runtimeStatus")).toObject()
        .value(QStringLiteral("status")).toString(), QStringLiteral("sdk_missing"));
    aitrain::TaskSnapshot stored;
    QVERIFY2(workspace.task(taskId, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Failed);
    QCOMPARE(stored.failure.code, aitrain::FailureCode::SdkMissing);
}

QTEST_MAIN(RuntimeDeliveryWorkflowTests)
#include "tst_runtime_delivery_workflow.moc"
