#include "TestSupport.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/TaskModels.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/core/WorkerRequests.h"

class PlatformTests final : public QObject {
    Q_OBJECT

private slots:
    void builtinCapabilityRegistryIsAuthoritative()
    {
        const auto& registry = aitrain::BuiltinCapabilityRegistry::instance();
        QVERIFY(!registry.capabilities().isEmpty());
        QVERIFY(!registry.backends().isEmpty());
        QVERIFY(registry.supports(
            QStringLiteral("yolo"),
            QStringLiteral("detection"),
            QStringLiteral("yolo_detection"),
            QStringLiteral("ultralytics_yolo_detect")));
        QVERIFY(registry.supports(
            QStringLiteral("yolo"),
            QStringLiteral("detection"),
            QStringLiteral("yolo_txt"),
            QStringLiteral("ultralytics_yolo")));
        QVERIFY(registry.supports(
            QStringLiteral("paddleocr"),
            QStringLiteral("ocr_recognition"),
            QStringLiteral("paddleocr_rec"),
            QStringLiteral("paddleocr_ppocrv4_rec")));
        QCOMPARE(registry.backendsForTask(QStringLiteral("obb"), QStringLiteral("yolo_obb")),
            QStringList() << QStringLiteral("ultralytics_yolo_obb"));
        QCOMPARE(registry.backend(QStringLiteral("ultralytics_yolo")).id,
            QStringLiteral("ultralytics_yolo_detect"));
        QVERIFY(!registry.supports(
            QStringLiteral("yolo"),
            QStringLiteral("detection"),
            QStringLiteral("yolo_detection"),
            QStringLiteral("paddleocr_rec_official")));
        QVERIFY(registry.capability(QStringLiteral("dataset_interop")).backendIds.isEmpty());
        QCOMPARE(registry.backend(QStringLiteral("ultralytics_yolo_obb")).exportFormats,
            QStringList() << QStringLiteral("onnx"));
        QVERIFY(registry.backend(QStringLiteral("ultralytics_yolo_detect")).modelPresets.contains(
            QStringLiteral("yolov8n.yaml")));
        QVERIFY(registry.backend(QStringLiteral("ultralytics_yolo_detect")).modelPresets.contains(
            QStringLiteral("yolo26x.pt")));
        QCOMPARE(registry.backend(QStringLiteral("ultralytics_yolo_obb")).modelPresets.size(), 10);
        QCOMPARE(registry.backend(QStringLiteral("smp_semantic_segmentation")).modelPresets.size(), 5);
        QCOMPARE(registry.backend(QStringLiteral("anomalib_efficientad")).modelPresets,
            QStringList() << QStringLiteral("anomalib_efficientad_s"));
        QCOMPARE(registry.backend(QStringLiteral("paddleocr_det_official")).modelPresets.size(), 6);
        QCOMPARE(registry.backend(QStringLiteral("paddleocr_rec_official")).modelPresets.size(), 7);

        const QJsonObject json = registry.toJson();
        QCOMPARE(json.value(QStringLiteral("schemaVersion")).toInt(), 1);
        QCOMPARE(json.value(QStringLiteral("aliases")).toObject()
                     .value(QStringLiteral("datasetFormats")).toObject()
                     .value(QStringLiteral("yolo_txt")).toString(),
            QStringLiteral("yolo_detection"));
        QVERIFY(json.value(QStringLiteral("capabilities")).toArray().size() >= 4);
        QVERIFY(json.value(QStringLiteral("backends")).toArray().size() >= 8);
    }

    void protocolEnvelopeIsVersionedAndSequenced()
    {
        const QJsonObject payload{{QStringLiteral("taskId"), QStringLiteral("task-1")}};
        const QByteArray first = aitrain::protocol::encodeMessage(QStringLiteral("progress"), payload);
        const QByteArray second = aitrain::protocol::encodeMessage(QStringLiteral("progress"), payload);
        const QJsonObject firstWire = QJsonDocument::fromJson(first.trimmed()).object();
        const QJsonObject secondWire = QJsonDocument::fromJson(second.trimmed()).object();
        QCOMPARE(firstWire.value(QStringLiteral("protocolVersion")).toInt(), aitrain::protocol::kCurrentProtocolVersion);
        QVERIFY(secondWire.value(QStringLiteral("sequence")).toVariant().toULongLong()
            > firstWire.value(QStringLiteral("sequence")).toVariant().toULongLong());

        QString type;
        QJsonObject decoded;
        QString requestId;
        QString error;
        QVERIFY(aitrain::protocol::decodeMessage(first, &type, &decoded, &requestId, &error));
        QCOMPARE(type, QStringLiteral("progress"));
        QCOMPARE(decoded.value(QStringLiteral("taskId")).toString(), QStringLiteral("task-1"));
        QVERIFY(error.isEmpty());

        QJsonObject invalid = firstWire;
        invalid.insert(QStringLiteral("protocolVersion"), 1);
        QVERIFY(!aitrain::protocol::decodeMessage(
            QJsonDocument(invalid).toJson(QJsonDocument::Compact), &type, &decoded, &requestId, &error));
        QVERIFY(error.contains(QStringLiteral("protocolVersion")));
    }

    void taskStateMachineHasNoPauseState()
    {
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Running));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Failed));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Completed));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Canceled));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Queued));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Completed, aitrain::TaskState::Running));
        QCOMPARE(aitrain::taskStateFromString(QStringLiteral("paused")), aitrain::TaskState::Queued);
        QCOMPARE(aitrain::taskStateToString(aitrain::TaskState::Running), QStringLiteral("running"));
    }

    void workerProtocolBuildersStayTyped()
    {
        namespace wp = aitrain::worker_protocol;
        namespace wr = aitrain::worker_requests;

        const QJsonObject options{{QStringLiteral("dryRun"), true}};
        const QJsonObject request = wp::datasetValidationRequest(
            QStringLiteral("task-1"), QStringLiteral("dataset"), QStringLiteral("yolo_detection"), options);
        const wr::DatasetPathRequest parsed = wr::parseDatasetValidationRequest(request);
        QCOMPARE(parsed.taskId, QStringLiteral("task-1"));
        QCOMPARE(parsed.datasetPath, QStringLiteral("dataset"));
        QCOMPARE(parsed.format, QStringLiteral("yolo_detection"));
        QVERIFY(parsed.options.value(QStringLiteral("dryRun")).toBool());
        QVERIFY(wp::isControlCommand(wp::command::cancel()));
        QVERIFY(!wp::isControlCommand(wp::command::exportModel()));
        QVERIFY(wp::isTerminalEvent(wp::event::completed()));
        QCOMPARE(wp::taskStateForEvent(wp::event::completed()), aitrain::TaskState::Completed);
        QCOMPARE(wp::taskStateForEvent(wp::event::failed()), aitrain::TaskState::Failed);
    }

    void trainingRequestRoundTripsBuiltinCapabilityId()
    {
        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("task-1");
        request.capabilityId = QStringLiteral("yolo");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = QStringLiteral("dataset");
        request.outputPath = QStringLiteral("run");
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect"));

        const aitrain::TrainingRequest parsed = aitrain::TrainingRequest::fromJson(request.toJson());
        QCOMPARE(parsed.capabilityId, QStringLiteral("yolo"));
        QCOMPARE(parsed.parameters.value(QStringLiteral("trainingBackend")).toString(),
            QStringLiteral("ultralytics_yolo_detect"));
    }

    void packagingLayoutContainsOnlyProductDirectories()
    {
        const aitrain::PackagingLayout layout = aitrain::packagingLayoutForRoot(QStringLiteral("C:/AITrain"));
        QVERIFY(!layout.rootPath.isEmpty());
        QVERIFY(!layout.appExecutablePath.isEmpty());
        QVERIFY(!layout.workerExecutablePath.isEmpty());
        QVERIFY(!layout.runtimesDirectory.isEmpty());
        QVERIFY(!layout.toJson().contains(QStringLiteral("pluginModelsDirectory")));
    }
};

QTEST_MAIN(PlatformTests)
#include "tst_platform.moc"
