#include "TestSupport.h"

class DetectionInferenceTests : public QObject {
    Q_OBJECT

private slots:
    void exportRejectsLegacyNonOnnxArtifacts()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString legacyPath = dir.filePath(QStringLiteral("legacy.aitrain"));
        writeTextFile(legacyPath, QStringLiteral("{\"type\":\"legacy\"}\n"));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            legacyPath,
            dir.filePath(QStringLiteral("model.onnx")),
            QStringLiteral("onnx"));

        QVERIFY(!exported.ok);
        QVERIFY(exported.error.contains(QStringLiteral("official ONNX")));
    }

    void onnxExportKeepsOfficialSidecarMetadata()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourceOnnx = dir.filePath(QStringLiteral("source.onnx"));
        writeTextFile(sourceOnnx, QStringLiteral("fake official onnx\n"));

        QJsonObject sidecar;
        sidecar.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        sidecar.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_detect"));
        sidecar.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_detection"));
        sidecar.insert(QStringLiteral("scaffold"), false);
        sidecar.insert(QStringLiteral("classNames"), QJsonArray{QStringLiteral("item")});
        QFile sidecarFile(dir.filePath(QStringLiteral("source.aitrain-export.json")));
        QVERIFY(sidecarFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
        sidecarFile.write(QJsonDocument(sidecar).toJson(QJsonDocument::Indented));
        sidecarFile.close();

        QCOMPARE(aitrain::inferOnnxModelFamily(sourceOnnx), QStringLiteral("yolo_detection"));

        const QString outputOnnx = dir.filePath(QStringLiteral("export/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            outputOnnx,
            QStringLiteral("onnx"));

        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("onnx"));
        QCOMPARE(exported.exportPath, outputOnnx);
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_detect"));
        QCOMPARE(exported.config.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_detection"));
        QVERIFY(!exported.config.value(QStringLiteral("scaffold")).toBool(true));
    }

    void onnxExportFollowsSourceTrainingReportForClassNames()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourceOnnx = dir.filePath(QStringLiteral("runs/weights/best.onnx"));
        const QString dataYaml = dir.filePath(QStringLiteral("dataset/data.yaml"));
        const QString reportPath = dir.filePath(QStringLiteral("runs/ultralytics_training_report.json"));
        writeTextFile(sourceOnnx, QStringLiteral("fake official onnx\n"));
        writeTextFile(dataYaml, QStringLiteral("nc: 2\nnames: [widget, defect]\n"));

        QJsonObject trainingReport;
        trainingReport.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_detect"));
        trainingReport.insert(QStringLiteral("dataYaml"), dataYaml);
        writeTextFile(reportPath, QString::fromUtf8(QJsonDocument(trainingReport).toJson(QJsonDocument::Indented)));

        QJsonObject sidecar;
        sidecar.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        sidecar.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_detect"));
        sidecar.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_detection"));
        sidecar.insert(QStringLiteral("scaffold"), false);
        sidecar.insert(QStringLiteral("sourceTrainingReport"), reportPath);
        writeTextFile(
            dir.filePath(QStringLiteral("runs/weights/best.aitrain-export.json")),
            QString::fromUtf8(QJsonDocument(sidecar).toJson(QJsonDocument::Indented)));

        const QString outputOnnx = dir.filePath(QStringLiteral("export/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            outputOnnx,
            QStringLiteral("onnx"));

        QVERIFY2(exported.ok, qPrintable(exported.error));
        const QJsonArray classNames = exported.config.value(QStringLiteral("classNames")).toArray();
        QCOMPARE(classNames.size(), 2);
        QCOMPARE(classNames.at(0).toString(), QStringLiteral("widget"));
        QCOMPARE(classNames.at(1).toString(), QStringLiteral("defect"));
    }

    void onnxExportUsesOfficialSiblingFromYoloCheckpoint()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString checkpointPath = dir.filePath(QStringLiteral("weights/best.pt"));
        const QString sourceOnnx = dir.filePath(QStringLiteral("weights/best.onnx"));
        writeTextFile(checkpointPath, QStringLiteral("fake yolo checkpoint\n"));
        writeTextFile(sourceOnnx, QStringLiteral("fake official onnx\n"));

        QJsonObject sidecar;
        sidecar.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        sidecar.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_detect"));
        sidecar.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_detection"));
        sidecar.insert(QStringLiteral("scaffold"), false);
        sidecar.insert(QStringLiteral("classNames"), QJsonArray{QStringLiteral("item")});
        writeTextFile(
            dir.filePath(QStringLiteral("weights/best.aitrain-export.json")),
            QString::fromUtf8(QJsonDocument(sidecar).toJson(QJsonDocument::Indented)));

        const QString outputOnnx = dir.filePath(QStringLiteral("export/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            checkpointPath,
            outputOnnx,
            QStringLiteral("onnx"));

        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("onnx"));
        QCOMPARE(exported.sourceCheckpointPath, checkpointPath);
        QCOMPARE(exported.exportPath, outputOnnx);
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_detect"));
        QCOMPARE(exported.config.value(QStringLiteral("sourceCheckpoint")).toString(), checkpointPath);
        QCOMPARE(QFileInfo(exported.config.value(QStringLiteral("sourceOnnx")).toString()).absoluteFilePath(), QFileInfo(sourceOnnx).absoluteFilePath());
    }

    void postprocessFiltersOfficialDetectionPredictions()
    {
        QVector<aitrain::DetectionPrediction> predictions;
        aitrain::DetectionPrediction first;
        first.box.classId = 0;
        first.box.xCenter = 0.5;
        first.box.yCenter = 0.5;
        first.box.width = 0.4;
        first.box.height = 0.4;
        first.confidence = 0.9;
        first.className = QStringLiteral("item");
        predictions.append(first);

        aitrain::DetectionPrediction overlapping = first;
        overlapping.confidence = 0.5;
        predictions.append(overlapping);

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.25;
        options.iouThreshold = 0.45;
        const QVector<aitrain::DetectionPrediction> filtered =
            aitrain::postProcessDetectionPredictions(predictions, options);
        QCOMPARE(filtered.size(), 1);
        QCOMPARE(filtered.first().className, QStringLiteral("item"));
        QCOMPARE(filtered.first().confidence, 0.9);
    }

    void inferenceOptionsUseProductionConfidenceDefault()
    {
        const aitrain::DetectionInferenceOptions options;
        QCOMPARE(options.confidenceThreshold, 0.25);
        QCOMPARE(options.iouThreshold, 0.45);
        QCOMPARE(options.maxDetections, 100);
    }
};

QTEST_MAIN(DetectionInferenceTests)
#include "tst_detection_inference.moc"
