#include "TestSupport.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/v2/ProtocolV2.h"

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

    void workerProtocolBuildersStayTyped()
    {
        namespace wp = aitrain::worker_protocol;
        const QJsonObject options{{QStringLiteral("dryRun"), true}};
        const QJsonObject request = wp::dataQualityWorkflowV2Request(
            QStringLiteral("task-1"), QStringLiteral("project"), QStringLiteral("dataset-id"),
            QStringLiteral("version-id"), QStringLiteral("snapshot-id"),
            QStringLiteral("artifact-id"), options);
        QCOMPARE(request.value(QStringLiteral("taskId")).toString(), QStringLiteral("task-1"));
        QCOMPARE(request.value(QStringLiteral("projectRoot")).toString(), QStringLiteral("project"));
        QVERIFY(request.value(QStringLiteral("options")).toObject().value(QStringLiteral("dryRun")).toBool());
        QVERIFY(!request.contains(QStringLiteral("datasetPath")));
        QVERIFY(!request.contains(QStringLiteral("outputPath")));
        QVERIFY(wp::isTerminalEvent(wp::event::completed()));
    }

    void workerControlBridgeUsesProtocolV2Envelope()
    {
        namespace wp = aitrain::worker_protocol;
        const aitrain::v2::RequestId requestId = aitrain::v2::RequestId::create();
        const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
        const QJsonObject businessPayload{{QStringLiteral("taskId"), taskId.toString()}};

        const aitrain::v2::ProtocolEnvelope start = wp::control_v2::startTaskEnvelope(
            requestId, taskId, 1, wp::command::runDataQualityWorkflowV2(), businessPayload);
        QCOMPARE(start.kind, QStringLiteral("command.start_task"));
        QString command;
        QJsonObject decodedPayload;
        QString error;
        QVERIFY2(wp::control_v2::unpackStartTask(start, &command, &decodedPayload, &error), qPrintable(error));
        QCOMPARE(command, wp::command::runDataQualityWorkflowV2());
        QCOMPARE(decodedPayload, businessPayload);

        const aitrain::v2::ProtocolEnvelope result = wp::control_v2::eventEnvelope(
            requestId, taskId, 2, wp::event::dataQualityWorkflowV2(), businessPayload);
        QCOMPARE(result.kind, QStringLiteral("event.result"));
        QVERIFY(aitrain::v2::isKnownProtocolV2Kind(result.kind));
        QString event;
        QVERIFY2(wp::control_v2::unpackBusinessEvent(result, &event, &decodedPayload, &error), qPrintable(error));
        QCOMPARE(event, wp::event::dataQualityWorkflowV2());
        QCOMPARE(decodedPayload, businessPayload);
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
