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
};

QTEST_MAIN(DetectionInferenceTests)
#include "tst_detection_inference.moc"
