#include "TestSupport.h"

#include "../src/core/src/VisionRuntimeInternal.h"

#include "aitrain/core/VisionPostprocess.h"

#include <QtMath>

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
        sidecar.insert(QStringLiteral("modelSeries"), QStringLiteral("yolo26"));
        sidecar.insert(QStringLiteral("task"), QStringLiteral("detection"));
        sidecar.insert(QStringLiteral("ultralyticsExportArgs"), QJsonObject{
            {QStringLiteral("format"), QStringLiteral("onnx")},
            {QStringLiteral("end2end"), true}
        });
        sidecar.insert(QStringLiteral("outputShapes"), QJsonObject{
            {QStringLiteral("available"), true}
        });
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
        QCOMPARE(exported.config.value(QStringLiteral("modelSeries")).toString(), QStringLiteral("yolo26"));
        QCOMPARE(exported.config.value(QStringLiteral("task")).toString(), QStringLiteral("detection"));
        QVERIFY(exported.config.value(QStringLiteral("ultralyticsExportArgs")).toObject().value(QStringLiteral("end2end")).toBool());
        QVERIFY(exported.config.value(QStringLiteral("outputShapes")).toObject().value(QStringLiteral("available")).toBool());
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

    void onnxExportKeepsOfficialSegmentationMetadata()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourceOnnx = dir.filePath(QStringLiteral("runs/weights/best.onnx"));
        const QString reportPath = dir.filePath(QStringLiteral("runs/ultralytics_training_report.json"));
        writeTextFile(sourceOnnx, QStringLiteral("fake official segmentation onnx\n"));

        QJsonObject trainingReport;
        trainingReport.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_segment"));
        writeTextFile(reportPath, QString::fromUtf8(QJsonDocument(trainingReport).toJson(QJsonDocument::Indented)));

        QJsonObject sidecar;
        sidecar.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        sidecar.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_segment"));
        sidecar.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_segmentation"));
        sidecar.insert(QStringLiteral("scaffold"), false);
        sidecar.insert(QStringLiteral("sourceTrainingReport"), reportPath);
        sidecar.insert(QStringLiteral("classNames"), QJsonArray{QStringLiteral("part"), QStringLiteral("scratch")});
        writeTextFile(
            dir.filePath(QStringLiteral("runs/weights/best.aitrain-export.json")),
            QString::fromUtf8(QJsonDocument(sidecar).toJson(QJsonDocument::Indented)));

        QCOMPARE(aitrain::inferOnnxModelFamily(sourceOnnx), QStringLiteral("yolo_segmentation"));

        const QString outputOnnx = dir.filePath(QStringLiteral("export/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            outputOnnx,
            QStringLiteral("onnx"));

        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_segment"));
        QCOMPARE(exported.config.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_segmentation"));
        const QJsonObject postprocess = exported.config.value(QStringLiteral("postprocess")).toObject();
        QCOMPARE(postprocess.value(QStringLiteral("decoder")).toString(), QStringLiteral("yolo_v8_segmentation"));
    }

    void onnxExportKeepsOfficialObbMetadata()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourceOnnx = dir.filePath(QStringLiteral("runs/weights/best.onnx"));
        const QString reportPath = dir.filePath(QStringLiteral("runs/ultralytics_training_report.json"));
        writeTextFile(sourceOnnx, QStringLiteral("fake official obb onnx\n"));

        QJsonObject trainingReport;
        trainingReport.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_obb"));
        trainingReport.insert(QStringLiteral("task"), QStringLiteral("obb"));
        writeTextFile(reportPath, QString::fromUtf8(QJsonDocument(trainingReport).toJson(QJsonDocument::Indented)));

        QJsonObject sidecar;
        sidecar.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        sidecar.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_obb"));
        sidecar.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_obb"));
        sidecar.insert(QStringLiteral("task"), QStringLiteral("obb"));
        sidecar.insert(QStringLiteral("scaffold"), false);
        sidecar.insert(QStringLiteral("sourceTrainingReport"), reportPath);
        sidecar.insert(QStringLiteral("classNames"), QJsonArray{QStringLiteral("ship"), QStringLiteral("plane")});
        writeTextFile(
            dir.filePath(QStringLiteral("runs/weights/best.aitrain-export.json")),
            QString::fromUtf8(QJsonDocument(sidecar).toJson(QJsonDocument::Indented)));

        QCOMPARE(aitrain::inferOnnxModelFamily(sourceOnnx), QStringLiteral("yolo_obb"));

        const QString outputOnnx = dir.filePath(QStringLiteral("export/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            outputOnnx,
            QStringLiteral("onnx"));

        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_obb"));
        QCOMPARE(exported.config.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_obb"));
        QCOMPARE(exported.config.value(QStringLiteral("task")).toString(), QStringLiteral("obb"));
        QCOMPARE(exported.config.value(QStringLiteral("classNames")).toArray().size(), 2);
        const QJsonObject postprocess = exported.config.value(QStringLiteral("postprocess")).toObject();
        QCOMPARE(postprocess.value(QStringLiteral("decoder")).toString(), QStringLiteral("yolo_obb"));
        QVERIFY(postprocess.value(QStringLiteral("nms")).toString().contains(QStringLiteral("rotated")));
    }

    void ncnnExportUsesPnnxForYoloOnnxAndKeepsInputSize()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourceOnnx = dir.filePath(QStringLiteral("runs/weights/best.onnx"));
        const QString reportPath = dir.filePath(QStringLiteral("runs/ultralytics_training_report.json"));
        const QString dataYaml = dir.filePath(QStringLiteral("dataset/data.yaml"));
        writeTextFile(sourceOnnx, QStringLiteral("fake official onnx\n"));
        writeTextFile(dataYaml, QStringLiteral("nc: 2\nnames: [square, circle]\n"));

        QJsonObject trainingReport;
        trainingReport.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_detect"));
        trainingReport.insert(QStringLiteral("dataYaml"), dataYaml);
        trainingReport.insert(QStringLiteral("ultralyticsExportArgs"), QJsonObject{
            {QStringLiteral("format"), QStringLiteral("onnx")},
            {QStringLiteral("imgsz"), 128},
            {QStringLiteral("batch"), 1}
        });
        writeTextFile(reportPath, QString::fromUtf8(QJsonDocument(trainingReport).toJson(QJsonDocument::Indented)));

        const QString outputParam = dir.filePath(QStringLiteral("export/model.param"));
        const QString outputBin = dir.filePath(QStringLiteral("export/model.bin"));
#ifdef Q_OS_WIN
        const QString fakePnnx = dir.filePath(QStringLiteral("fake_pnnx.cmd"));
        writeTextFile(fakePnnx,
            QStringLiteral("@echo off\r\n"
                           "echo 7767517>\"%1\"\r\n"
                           "echo 2 2>>\"%1\"\r\n"
                           "echo Input in0 0 1 in0>>\"%1\"\r\n"
                           "echo MemoryData out0 1 1 in0 out0 0=1>>\"%1\"\r\n"
                           "echo fake>\"%2\"\r\n")
                .arg(QDir::toNativeSeparators(outputParam), QDir::toNativeSeparators(outputBin)));
#else
        const QString fakePnnx = dir.filePath(QStringLiteral("fake_pnnx.sh"));
        writeTextFile(fakePnnx,
            QStringLiteral("#!/bin/sh\n"
                           "printf '7767517\\n2 2\\nInput in0 0 1 in0\\nMemoryData out0 1 1 in0 out0 0=1\\n' > '%1'\n"
                           "printf 'fake\\n' > '%2'\n")
                .arg(outputParam, outputBin));
        QFile::setPermissions(fakePnnx, QFile::permissions(fakePnnx) | QFileDevice::ExeOwner | QFileDevice::ExeUser);
#endif
        const ScopedEnvVar pnnxEnv("AITRAIN_NCNN_PNNX", QFile::encodeName(fakePnnx));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            outputParam,
            QStringLiteral("ncnn"));

        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.exportPath, outputParam);
        QVERIFY(QFileInfo::exists(outputParam));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("export/model.bin"))));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_detect"));
        QCOMPARE(exported.config.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_detection"));
        const QJsonObject ncnn = exported.config.value(QStringLiteral("ncnn")).toObject();
        QCOMPARE(QFileInfo(ncnn.value(QStringLiteral("converter")).toString()).absoluteFilePath(), QFileInfo(fakePnnx).absoluteFilePath());
        QCOMPARE(ncnn.value(QStringLiteral("inputBlob")).toString(), QStringLiteral("in0"));
        QCOMPARE(ncnn.value(QStringLiteral("outputBlobs")).toArray().size(), 1);
        QCOMPARE(ncnn.value(QStringLiteral("outputBlobs")).toArray().first().toString(), QStringLiteral("out0"));
        QCOMPARE(ncnn.value(QStringLiteral("inputSize")).toInt(), 128);
        QCOMPARE(exported.config.value(QStringLiteral("classNames")).toArray().size(), 2);
    }

    void ncnnExportRejectsYolo26Onnx()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourceOnnx = dir.filePath(QStringLiteral("runs/weights/best.onnx"));
        writeTextFile(sourceOnnx, QStringLiteral("fake yolo26 official onnx\n"));

        QJsonObject sidecar;
        sidecar.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        sidecar.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_detect"));
        sidecar.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_detection"));
        sidecar.insert(QStringLiteral("modelSeries"), QStringLiteral("yolo26"));
        sidecar.insert(QStringLiteral("scaffold"), false);
        writeTextFile(
            dir.filePath(QStringLiteral("runs/weights/best.aitrain-export.json")),
            QString::fromUtf8(QJsonDocument(sidecar).toJson(QJsonDocument::Indented)));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            dir.filePath(QStringLiteral("export/model.param")),
            QStringLiteral("ncnn"));

        QVERIFY(!exported.ok);
        QVERIFY(exported.error.contains(QStringLiteral("YOLO26 NCNN export is not supported")));
    }

    void ncnnExportRejectsYoloObbOnnx()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString sourceOnnx = dir.filePath(QStringLiteral("runs/weights/best.onnx"));
        writeTextFile(sourceOnnx, QStringLiteral("fake yolo obb official onnx\n"));

        QJsonObject sidecar;
        sidecar.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        sidecar.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_obb"));
        sidecar.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_obb"));
        sidecar.insert(QStringLiteral("task"), QStringLiteral("obb"));
        sidecar.insert(QStringLiteral("scaffold"), false);
        writeTextFile(
            dir.filePath(QStringLiteral("runs/weights/best.aitrain-export.json")),
            QString::fromUtf8(QJsonDocument(sidecar).toJson(QJsonDocument::Indented)));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            dir.filePath(QStringLiteral("export/model.param")),
            QStringLiteral("ncnn"));

        QVERIFY(!exported.ok);
        QVERIFY(exported.error.contains(QStringLiteral("OBB NCNN export is not supported")));
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

    void yolo26EndToEndDetectionUsesCornerBoxesWithoutLocalNms()
    {
        const std::vector<float> output = {
            10.0f, 20.0f, 50.0f, 60.0f, 0.90f, 0.0f,
            12.0f, 22.0f, 52.0f, 62.0f, 0.80f, 0.0f,
        };
        const std::vector<int64_t> shape = {1, 2, 6};

        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(100, 100);
        transform.targetSize = QSize(100, 100);
        transform.scale = 1.0;

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.25;
        options.iouThreshold = 0.01;
        options.maxDetections = 100;
        QString error;
        const QVector<aitrain::DetectionPrediction> predictions =
            aitrain::detection_detail::yoloEndToEndPredictionsFromOutput(
                output.data(),
                shape,
                QStringList{QStringLiteral("item")},
                QSize(100, 100),
                transform,
                options,
                &error);

        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 2);
        QCOMPARE(predictions.first().className, QStringLiteral("item"));
        QVERIFY(qAbs(predictions.first().confidence - 0.90) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().box.xCenter - 0.30) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().box.yCenter - 0.40) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().box.width - 0.40) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().box.height - 0.40) < 1.0e-6);
    }

    void yolo26EndToEndDetectionRejectsAttributeFirstShape()
    {
        const std::vector<float> output = {
            10.0f, 12.0f,
            20.0f, 22.0f,
            50.0f, 52.0f,
            60.0f, 62.0f,
            0.90f, 0.80f,
            0.0f, 0.0f,
        };
        const std::vector<int64_t> shape = {1, 6, 2};

        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(100, 100);
        transform.targetSize = QSize(100, 100);
        transform.scale = 1.0;

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.25;
        options.iouThreshold = 0.45;
        options.maxDetections = 100;
        QString error;
        const QVector<aitrain::DetectionPrediction> predictions =
            aitrain::detection_detail::yoloEndToEndPredictionsFromOutput(
                output.data(),
                shape,
                QStringList{QStringLiteral("item")},
                QSize(100, 100),
                transform,
                options,
                &error);

        QVERIFY(predictions.isEmpty());
        QVERIFY(error.contains(QStringLiteral("shape")));
    }

    void yolo26EndToEndSegmentationBuildsMaskFromPrototype()
    {
        const std::vector<float> boxesAndMasks = {
            0.0f, 0.0f, 4.0f, 4.0f, 0.95f, 0.0f, 8.0f, 0.0f,
        };
        const std::vector<int64_t> boxesShape = {1, 1, 8};
        std::vector<float> prototypes(2 * 4 * 4, 0.0f);
        for (int index = 0; index < 16; ++index) {
            prototypes[index] = 1.0f;
        }
        const std::vector<int64_t> prototypeShape = {1, 2, 4, 4};

        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(4, 4);
        transform.targetSize = QSize(4, 4);
        transform.scale = 1.0;

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.25;
        options.iouThreshold = 0.45;
        options.maxDetections = 100;
        QString error;
        const QVector<aitrain::SegmentationPrediction> predictions =
            aitrain::detection_detail::yoloEndToEndSegmentationPredictionsFromOutputs(
                boxesAndMasks.data(),
                boxesShape,
                prototypes.data(),
                prototypeShape,
                QStringList{QStringLiteral("part")},
                QSize(4, 4),
                transform,
                options,
                &error);

        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QCOMPARE(predictions.first().detection.className, QStringLiteral("part"));
        QVERIFY(qAbs(predictions.first().detection.confidence - 0.95) < 1.0e-6);
        QVERIFY(!predictions.first().mask.isNull());
        QVERIFY(predictions.first().maskArea > 0.90);
    }

    void yolo26EndToEndSegmentationRejectsAttributeFirstShape()
    {
        const std::vector<float> boxesAndMasks = {
            0.0f, 0.0f, 4.0f, 4.0f, 0.95f, 0.0f, 8.0f, 0.0f,
        };
        const std::vector<int64_t> boxesShape = {1, 8, 1};
        std::vector<float> prototypes(2 * 4 * 4, 1.0f);
        const std::vector<int64_t> prototypeShape = {1, 2, 4, 4};

        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(4, 4);
        transform.targetSize = QSize(4, 4);
        transform.scale = 1.0;

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.25;
        options.iouThreshold = 0.45;
        options.maxDetections = 100;
        QString error;
        const QVector<aitrain::SegmentationPrediction> predictions =
            aitrain::detection_detail::yoloEndToEndSegmentationPredictionsFromOutputs(
                boxesAndMasks.data(),
                boxesShape,
                prototypes.data(),
                prototypeShape,
                QStringList{QStringLiteral("part")},
                QSize(4, 4),
                transform,
                options,
                &error);

        QVERIFY(predictions.isEmpty());
        QVERIFY(error.contains(QStringLiteral("outputs must be")));
    }

    void yoloObbPostprocessOutputsRotatedJsonAndOverlay()
    {
        const std::vector<float> output = {
            50.0f, 50.0f, 40.0f, 20.0f, 0.90f, 0.10f, 0.0f,
            51.0f, 50.0f, 40.0f, 20.0f, 0.80f, 0.20f, 0.0f,
        };
        const std::vector<int64_t> shape = {1, 2, 7};

        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(100, 100);
        transform.targetSize = QSize(100, 100);
        transform.scale = 1.0;

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.25;
        options.iouThreshold = 0.01;
        options.maxDetections = 100;
        QString error;
        const QVector<aitrain::ObbPrediction> predictions =
            aitrain::detection_detail::yoloObbPredictionsFromOutput(
                output.data(),
                shape,
                QStringList{QStringLiteral("ship"), QStringLiteral("plane")},
                QSize(100, 100),
                transform,
                options,
                &error);

        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QCOMPARE(predictions.first().detection.className, QStringLiteral("ship"));
        QCOMPARE(predictions.first().points.size(), 4);
        QVERIFY(qAbs(predictions.first().detection.confidence - 0.90) < 1.0e-6);

        const QJsonObject json = aitrain::obbPredictionToJson(predictions.first());
        QCOMPARE(json.value(QStringLiteral("taskType")).toString(), QStringLiteral("obb_detection"));
        QCOMPARE(json.value(QStringLiteral("xywhr")).toArray().size(), 5);
        QCOMPARE(json.value(QStringLiteral("points")).toArray().size(), 4);
        QVERIFY(json.value(QStringLiteral("bbox")).isObject());

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString imagePath = dir.filePath(QStringLiteral("sample.png"));
        QImage image(100, 100, QImage::Format_RGB888);
        image.fill(Qt::white);
        QVERIFY(image.save(imagePath));
        const QImage overlay = aitrain::renderObbPredictions(imagePath, predictions, &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
        QCOMPARE(overlay.size(), QSize(100, 100));
    }

    void yoloObbPostprocessHandlesAnchorsFirstWhenBothDimsLookValid()
    {
        std::vector<float> output;
        output.reserve(8 * 7);
        output.insert(output.end(), {60.0f, 40.0f, 20.0f, 10.0f, 0.91f, 0.05f, 0.0f});
        for (int anchor = 1; anchor < 8; ++anchor) {
            output.insert(output.end(), {10.0f, 10.0f, 1.0f, 1.0f, 0.02f, 0.01f, 0.0f});
        }
        const std::vector<int64_t> shape = {1, 8, 7};

        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(100, 100);
        transform.targetSize = QSize(100, 100);
        transform.scale = 1.0;

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.25;
        options.iouThreshold = 0.45;
        options.maxDetections = 100;
        QString error;
        const QVector<aitrain::ObbPrediction> predictions =
            aitrain::detection_detail::yoloObbPredictionsFromOutput(
                output.data(),
                shape,
                QStringList{QStringLiteral("ship"), QStringLiteral("plane")},
                QSize(100, 100),
                transform,
                options,
                &error);

        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QCOMPARE(predictions.first().detection.className, QStringLiteral("ship"));
        QVERIFY(qAbs(predictions.first().detection.confidence - 0.91) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().detection.box.xCenter - 0.60) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().detection.box.yCenter - 0.40) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().detection.box.width - 0.20) < 1.0e-6);
        QVERIFY(qAbs(predictions.first().detection.box.height - 0.10) < 1.0e-6);
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
