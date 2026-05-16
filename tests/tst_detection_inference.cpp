#include "TestSupport.h"
#include "../src/core/src/DetectionTrainerInternal.h"

class DetectionInferenceTests : public QObject {
    Q_OBJECT

private slots:
    void detectionTrainingBackendStatusMarksPhase8Scaffold()
    {
        const QJsonObject status = aitrain::detectionTrainingBackendStatus();
        QCOMPARE(status.value(QStringLiteral("phase")).toInt(), 8);
        QCOMPARE(status.value(QStringLiteral("activeBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QVERIFY(status.value(QStringLiteral("activeBackendScaffold")).toBool());
        QVERIFY(!status.value(QStringLiteral("realYoloStyleTrainingAvailable")).toBool(true));
        QVERIFY(!status.value(QStringLiteral("availableBackends")).toArray().isEmpty());
    }

    void detectionTrainerWritesTinyDetectorCheckpoint()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/b.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/c.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/b.txt")), QStringLiteral("1 0.4 0.4 0.20 0.30\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/c.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 2;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));

        int callbackCount = 0;
        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(
            root,
            options,
            [&callbackCount](const aitrain::DetectionTrainingMetrics& metrics) {
                ++callbackCount;
                return metrics.epoch >= 1 && metrics.step >= 1 && metrics.loss >= 0.0;
            });

        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.trainingBackend, QStringLiteral("tiny_linear_detector"));
        QCOMPARE(result.modelFamily, QStringLiteral("yolo_style_detection_scaffold"));
        QVERIFY(result.scaffold);
        QCOMPARE(result.modelArchitecture.value(QStringLiteral("family")).toString(), QStringLiteral("tiny_linear_detector_scaffold"));
        QCOMPARE(result.steps, 4);
        QCOMPARE(callbackCount, 4);
        QVERIFY(result.finalLoss >= 0.0);
        QVERIFY(QFileInfo::exists(result.checkpointPath));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("type")).toString(), QStringLiteral("tiny_linear_detector"));
        QCOMPARE(json.value(QStringLiteral("checkpointSchemaVersion")).toInt(), 2);
        QCOMPARE(json.value(QStringLiteral("trainingBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QCOMPARE(json.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_style_detection_scaffold"));
        QVERIFY(json.value(QStringLiteral("scaffold")).toBool());
        QCOMPARE(json.value(QStringLiteral("modelArchitecture")).toObject().value(QStringLiteral("family")).toString(), QStringLiteral("tiny_linear_detector_scaffold"));
        QVERIFY(!json.value(QStringLiteral("phase8")).toObject().value(QStringLiteral("realYoloStyleTraining")).toBool(true));
        QCOMPARE(json.value(QStringLiteral("steps")).toInt(), 4);
        QCOMPARE(json.value(QStringLiteral("gridSize")).toInt(), 4);
        QVERIFY(json.value(QStringLiteral("featureCount")).toInt() > 0);
        QVERIFY(!json.value(QStringLiteral("objectnessWeights")).toArray().isEmpty());
        QVERIFY(!json.value(QStringLiteral("classWeights")).toArray().isEmpty());
        QVERIFY(!json.value(QStringLiteral("boxWeights")).toArray().isEmpty());
        QVERIFY(json.contains(QStringLiteral("mAP50")));
        QVERIFY(json.value(QStringLiteral("precision")).toDouble() >= 0.0);
        QVERIFY(json.value(QStringLiteral("recall")).toDouble() >= 0.0);
        QVERIFY(json.value(QStringLiteral("mAP50")).toDouble() >= 0.0);
        QVERIFY(json.value(QStringLiteral("mAP50")).toDouble() <= 1.0);

        QString error;
        aitrain::DetectionBaselineCheckpoint loaded;
        QVERIFY2(aitrain::loadDetectionBaselineCheckpoint(result.checkpointPath, &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.type, QStringLiteral("tiny_linear_detector"));
        QCOMPARE(loaded.checkpointSchemaVersion, 2);
        QCOMPARE(loaded.trainingBackend, QStringLiteral("tiny_linear_detector"));
        QCOMPARE(loaded.modelFamily, QStringLiteral("yolo_style_detection_scaffold"));
        QVERIFY(loaded.scaffold);
        QCOMPARE(loaded.modelArchitecture.value(QStringLiteral("family")).toString(), QStringLiteral("tiny_linear_detector_scaffold"));
        QCOMPARE(loaded.steps, 4);
        QCOMPARE(loaded.gridSize, 4);
        QVERIFY(loaded.featureCount > 0);
        QVERIFY(!loaded.objectnessWeights.isEmpty());
        QVERIFY(!loaded.classWeights.isEmpty());
        QVERIFY(!loaded.boxWeights.isEmpty());
        QCOMPARE(loaded.classNames.size(), 2);
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(
            loaded,
            QDir(root).filePath(QStringLiteral("images/val/c.png")),
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QVERIFY(predictions.first().objectness >= 0.0);
        QVERIFY(predictions.first().confidence >= 0.0);
        QVERIFY(predictions.first().box.width > 0.0);
        const QJsonObject predictionJson = aitrain::detectionPredictionToJson(predictions.first());
        QCOMPARE(predictionJson.value(QStringLiteral("className")).toString(), predictions.first().className);
        const QImage rendered = aitrain::renderDetectionPredictions(
            QDir(root).filePath(QStringLiteral("images/val/c.png")),
            predictions,
            &error);
        QVERIFY2(!rendered.isNull(), qPrintable(error));
        QCOMPARE(rendered.size(), QSize(8, 8));
    }

    void detectionTrainerRejectsUnavailableYoloStyleBackend()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::DetectionTrainingOptions options;
        options.trainingBackend = QStringLiteral("yolo_style_libtorch");
        options.outputPath = dir.filePath(QStringLiteral("run"));

        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(root, options);
        QVERIFY(!result.ok);
        QCOMPARE(result.trainingBackend, QStringLiteral("yolo_style_libtorch"));
        QVERIFY(result.error.contains(QStringLiteral("not available")));
        QVERIFY(result.error.contains(QStringLiteral("YOLO-style")));
    }

    void detectionPostProcessAppliesThresholdNmsAndLimit()
    {
        QVector<aitrain::DetectionPrediction> predictions;

        aitrain::DetectionPrediction high;
        high.box.classId = 0;
        high.box.xCenter = 0.5;
        high.box.yCenter = 0.5;
        high.box.width = 0.4;
        high.box.height = 0.4;
        high.confidence = 0.9;
        high.objectness = 0.9;
        high.className = QStringLiteral("item");
        predictions.append(high);

        aitrain::DetectionPrediction overlap = high;
        overlap.box.xCenter = 0.52;
        overlap.box.yCenter = 0.52;
        overlap.confidence = 0.8;
        predictions.append(overlap);

        aitrain::DetectionPrediction otherClass = overlap;
        otherClass.box.classId = 1;
        otherClass.className = QStringLiteral("other");
        otherClass.confidence = 0.7;
        predictions.append(otherClass);

        aitrain::DetectionPrediction low = high;
        low.box.xCenter = 0.1;
        low.box.yCenter = 0.1;
        low.confidence = 0.05;
        predictions.append(low);

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.1;
        options.iouThreshold = 0.5;
        options.maxDetections = 10;
        const QVector<aitrain::DetectionPrediction> filtered = aitrain::postProcessDetectionPredictions(predictions, options);
        QCOMPARE(filtered.size(), 2);
        QCOMPARE(filtered.at(0).confidence, 0.9);
        QCOMPARE(filtered.at(0).box.classId, 0);
        QCOMPARE(filtered.at(1).box.classId, 1);

        options.maxDetections = 1;
        const QVector<aitrain::DetectionPrediction> limited = aitrain::postProcessDetectionPredictions(predictions, options);
        QCOMPARE(limited.size(), 1);
        QCOMPARE(limited.first().confidence, 0.9);
    }

    void detectionTrainerLossDecreasesOnSimpleDataset()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.70 0.30 0.40 0.20\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.70 0.30 0.40 0.20\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 8;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.learningRate = 0.2;
        options.outputPath = dir.filePath(QStringLiteral("run"));

        double firstLoss = -1.0;
        double lastLoss = -1.0;
        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(
            root,
            options,
            [&firstLoss, &lastLoss](const aitrain::DetectionTrainingMetrics& metrics) {
                if (firstLoss < 0.0) {
                    firstLoss = metrics.loss;
                }
                lastLoss = metrics.loss;
                return true;
            });

        QVERIFY2(result.ok, qPrintable(result.error));
        QVERIFY(firstLoss >= 0.0);
        QVERIFY(lastLoss >= 0.0);
        QVERIFY2(lastLoss < firstLoss, qPrintable(QStringLiteral("first=%1 last=%2").arg(firstLoss).arg(lastLoss)));
    }

    void detectionTrainerSupportsAugmentationOptions()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.25 0.50 0.30 0.30\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.25 0.50 0.30 0.30\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.gridSize = 2;
        options.horizontalFlip = true;
        options.colorJitter = true;
        options.outputPath = dir.filePath(QStringLiteral("run"));

        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(result.ok, qPrintable(result.error));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("gridSize")).toInt(), 2);
        QVERIFY(json.value(QStringLiteral("horizontalFlip")).toBool());
        QVERIFY(json.value(QStringLiteral("colorJitter")).toBool());
    }

    void detectionTrainerResumesTinyDetectorCheckpoint()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.60 0.40 0.30 0.20\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.60 0.40 0.30 0.20\n"));

        aitrain::DetectionTrainingOptions firstOptions;
        firstOptions.epochs = 1;
        firstOptions.batchSize = 1;
        firstOptions.imageSize = QSize(32, 32);
        firstOptions.outputPath = dir.filePath(QStringLiteral("run1"));
        const aitrain::DetectionTrainingResult first = aitrain::trainDetectionBaseline(root, firstOptions);
        QVERIFY2(first.ok, qPrintable(first.error));
        QCOMPARE(first.steps, 1);

        aitrain::DetectionTrainingOptions resumeOptions = firstOptions;
        resumeOptions.outputPath = dir.filePath(QStringLiteral("run2"));
        resumeOptions.resumeCheckpointPath = first.checkpointPath;
        const aitrain::DetectionTrainingResult resumed = aitrain::trainDetectionBaseline(root, resumeOptions);
        QVERIFY2(resumed.ok, qPrintable(resumed.error));
        QCOMPARE(resumed.steps, 2);

        QFile checkpoint(resumed.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("resumeFrom")).toString(), first.checkpointPath);
    }

    void detectionExportWritesTinyDetectorJson()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPath = dir.filePath(QStringLiteral("exports/model.aitrain-export.json"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(training.checkpointPath, exportPath);
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("tiny_detector_json"));
        QVERIFY(QFileInfo::exists(exported.exportPath));

        QFile file(exported.exportPath);
        QVERIFY(file.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(file.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("format")).toString(), QStringLiteral("tiny_detector_json"));
        QCOMPARE(json.value(QStringLiteral("note")).toString(), QStringLiteral("Scaffold export for AITrain tiny detector. This is not ONNX."));
        QCOMPARE(json.value(QStringLiteral("trainingBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QVERIFY(json.value(QStringLiteral("scaffold")).toBool());
        QVERIFY(!json.value(QStringLiteral("objectnessWeights")).toArray().isEmpty());
        QVERIFY(!json.value(QStringLiteral("classLogits")).toArray().isEmpty());
        QVERIFY(json.value(QStringLiteral("priorBox")).isObject());

        QString error;
        aitrain::DetectionBaselineCheckpoint loadedExport;
        QVERIFY2(aitrain::loadDetectionBaselineCheckpoint(exported.exportPath, &loadedExport, &error), qPrintable(error));
        QCOMPARE(loadedExport.trainingBackend, QStringLiteral("tiny_linear_detector"));
        QVERIFY(loadedExport.scaffold);
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(
            loadedExport,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
    }

    void detectionExportHonorsCancellation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString exportPath = dir.filePath(QStringLiteral("exports/model.aitrain-export.json"));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            dir.filePath(QStringLiteral("missing.aitrain")),
            exportPath,
            QStringLiteral("tiny_detector_json"),
            []() {
                return true;
            });
        QVERIFY(!exported.ok);
        QCOMPARE(exported.error, QStringLiteral("Canceled by user"));
        QVERIFY(exported.exportPath.isEmpty());
        QVERIFY(!QFileInfo::exists(exportPath));
    }

    void detectionExportWritesTinyDetectorOnnx()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPath = dir.filePath(QStringLiteral("exports/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            exportPath,
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("onnx"));
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));
        QCOMPARE(exported.config.value(QStringLiteral("format")).toString(), QStringLiteral("onnx"));
        QCOMPARE(exported.config.value(QStringLiteral("trainingBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QVERIFY(exported.config.value(QStringLiteral("scaffold")).toBool());
        QCOMPARE(exported.config.value(QStringLiteral("classNames")).toArray().first().toString(), QStringLiteral("item"));
        QCOMPARE(exported.config.value(QStringLiteral("input")).toObject().value(QStringLiteral("name")).toString(), QStringLiteral("features"));
        QCOMPARE(exported.config.value(QStringLiteral("outputs")).toArray().size(), 3);

        QFile file(exported.exportPath);
        QVERIFY(file.open(QIODevice::ReadOnly));
        const QByteArray bytes = file.readAll();
        QVERIFY(bytes.size() > 128);
        QVERIFY(bytes.contains("AITrain Studio"));
        QVERIFY(bytes.contains("Gemm"));
        QVERIFY(bytes.contains("Softmax"));
        QVERIFY(bytes.contains("objectness"));
        QVERIFY(bytes.contains("class_probabilities"));
        QVERIFY(bytes.contains("boxes"));
    }

    void detectionExportWritesNcnnParamBinWithConfiguredConverter()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString converterPath = dir.filePath(
#ifdef Q_OS_WIN
            QStringLiteral("onnx2ncnn.cmd")
#else
            QStringLiteral("onnx2ncnn")
#endif
        );
#ifdef Q_OS_WIN
        writeTextFile(converterPath,
            QStringLiteral("@echo off\r\n"
                           "echo fake ncnn param from %~1> \"%~2\"\r\n"
                           "echo fake ncnn bin from %~1> \"%~3\"\r\n"
                           "exit /b 0\r\n"));
#else
        writeTextFile(converterPath,
            QStringLiteral("#!/bin/sh\n"
                           "printf 'fake ncnn param from %s\\n' \"$1\" > \"$2\"\n"
                           "printf 'fake ncnn bin from %s\\n' \"$1\" > \"$3\"\n"));
        QFile::setPermissions(converterPath, QFile::permissions(converterPath)
            | QFileDevice::ExeOwner | QFileDevice::ExeUser | QFileDevice::ExeGroup | QFileDevice::ExeOther);
#endif
        ScopedEnvVar converterEnv("AITRAIN_NCNN_ONNX2NCNN", QFile::encodeName(converterPath));

        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPrefix = dir.filePath(QStringLiteral("exports/mobile-model"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            exportPrefix,
            QStringLiteral("ncnn"));
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("ncnn"));
        QCOMPARE(exported.exportPath, QDir(dir.filePath(QStringLiteral("exports"))).filePath(QStringLiteral("mobile-model.param")));
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));

        const QJsonObject ncnn = exported.config.value(QStringLiteral("ncnn")).toObject();
        const QString binPath = ncnn.value(QStringLiteral("binPath")).toString();
        QCOMPARE(exported.config.value(QStringLiteral("format")).toString(), QStringLiteral("ncnn"));
        QCOMPARE(ncnn.value(QStringLiteral("paramPath")).toString(), exported.exportPath);
        QCOMPARE(ncnn.value(QStringLiteral("runtime")).toString(), QStringLiteral("ncnn"));
        QCOMPARE(ncnn.value(QStringLiteral("runtimeValidation")).toString(), QStringLiteral("runtime-inference"));
        QVERIFY(QFileInfo::exists(binPath));
        QCOMPARE(QFileInfo(binPath).fileName(), QStringLiteral("mobile-model.bin"));

        QFile paramFile(exported.exportPath);
        QVERIFY(paramFile.open(QIODevice::ReadOnly));
        QVERIFY(paramFile.readAll().contains("fake ncnn param"));
    }

    void detectionNcnnExportPreservesYoloSegmentationSidecar()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString converterPath = dir.filePath(
#ifdef Q_OS_WIN
            QStringLiteral("onnx2ncnn.cmd")
#else
            QStringLiteral("onnx2ncnn")
#endif
        );
#ifdef Q_OS_WIN
        writeTextFile(converterPath,
            QStringLiteral("@echo off\r\n"
                           "echo fake ncnn param from %~1> \"%~2\"\r\n"
                           "echo fake ncnn bin from %~1> \"%~3\"\r\n"
                           "exit /b 0\r\n"));
#else
        writeTextFile(converterPath,
            QStringLiteral("#!/bin/sh\n"
                           "printf 'fake ncnn param from %s\\n' \"$1\" > \"$2\"\n"
                           "printf 'fake ncnn bin from %s\\n' \"$1\" > \"$3\"\n"));
        QFile::setPermissions(converterPath, QFile::permissions(converterPath)
            | QFileDevice::ExeOwner | QFileDevice::ExeUser | QFileDevice::ExeGroup | QFileDevice::ExeOther);
#endif
        ScopedEnvVar converterEnv("AITRAIN_NCNN_ONNX2NCNN", QFile::encodeName(converterPath));

        const QString sourceOnnx = dir.filePath(QStringLiteral("yolov8n-seg.onnx"));
        writeTextFile(sourceOnnx, QStringLiteral("fake segmentation onnx\n"));
        writeTextFile(dir.filePath(QStringLiteral("yolov8n-seg.aitrain-export.json")),
            QStringLiteral("{\n"
                           "  \"format\": \"onnx\",\n"
                           "  \"backend\": \"ultralytics_yolo_segment\",\n"
                           "  \"modelFamily\": \"yolo_segmentation\",\n"
                           "  \"classNames\": [\"person\", \"car\"],\n"
                           "  \"postprocess\": {\"decoder\": \"yolo_v8_segmentation\"}\n"
                           "}\n"));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            dir.filePath(QStringLiteral("exports/model.param")),
            QStringLiteral("ncnn"));
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_segment"));
        QCOMPARE(exported.config.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_segmentation"));
        QCOMPARE(exported.config.value(QStringLiteral("postprocess")).toObject().value(QStringLiteral("decoder")).toString(),
            QStringLiteral("yolo_v8_segmentation"));
        QCOMPARE(exported.config.value(QStringLiteral("ncnn")).toObject().value(QStringLiteral("runtimeValidation")).toString(),
            QStringLiteral("runtime-inference"));
    }

    void detectionNcnnExportReportsMissingConverter()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const ScopedEnvVar converterEnv("AITRAIN_NCNN_ONNX2NCNN", QFile::encodeName(dir.filePath(QStringLiteral("missing-onnx2ncnn.exe"))));
        const QString sourceOnnx = dir.filePath(QStringLiteral("source.onnx"));
        writeTextFile(sourceOnnx, QStringLiteral("fake onnx\n"));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            dir.filePath(QStringLiteral("exports/model.param")),
            QStringLiteral("ncnn"));
        QVERIFY(!exported.ok);
        QCOMPARE(exported.format, QStringLiteral("ncnn"));
        QVERIFY(exported.error.contains(QStringLiteral("NCNN")));
        QVERIFY(exported.error.contains(QStringLiteral("not found")));
    }

    void detectionNcnnBackendStatusAndSidecarMetadataAreExplicit()
    {
        const aitrain::NcnnBackendStatus status = aitrain::ncnnBackendStatus();
        QVERIFY(!status.message.isEmpty());
        QVERIFY(status.status == QStringLiteral("sdk_missing")
            || status.status == QStringLiteral("backend_available"));
        QCOMPARE(status.toJson().value(QStringLiteral("inferenceAvailable")).toBool(), status.inferenceAvailable);

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString paramPath = dir.filePath(QStringLiteral("model.param"));
        const QString binPath = dir.filePath(QStringLiteral("model.bin"));
        writeTextFile(paramPath,
            QStringLiteral("7767517\n"
                           "3 3\n"
                           "Input data 0 1 data 0=32 1=32 2=3\n"
                           "Dummy layer0 1 1 data hidden\n"
                           "Dummy layer1 1 1 hidden prob\n"));
        writeTextFile(binPath, QStringLiteral("fake bin\n"));
        writeTextFile(dir.filePath(QStringLiteral("model.aitrain-export.json")),
            QStringLiteral("{\n"
                           "  \"format\": \"ncnn\",\n"
                           "  \"modelFamily\": \"yolo_detection\",\n"
                           "  \"classNames\": [\"item\"],\n"
                           "  \"ncnn\": {\n"
                           "    \"inputBlob\": \"data\",\n"
                           "    \"outputBlobs\": [\"prob\"],\n"
                           "    \"decoder\": \"dfl\",\n"
                           "    \"inputSize\": 32,\n"
                           "    \"strides\": [8, 16, 32],\n"
                           "    \"regMax\": 16\n"
                           "  }\n"
                           "}\n"));

        QCOMPARE(aitrain::inferNcnnModelFamily(paramPath), QStringLiteral("yolo_detection"));
    }

    void detectionNcnnUltralyticsOutputDecoderGeneratesBox()
    {
        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(32, 32);
        transform.targetSize = QSize(32, 32);
        transform.scale = 1.0;

        const float output[] = {16.0f, 16.0f, 12.0f, 10.0f, 0.90f};
        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.50;
        QString error;
        const QVector<aitrain::DetectionPrediction> predictions =
            aitrain::detection_detail::yoloPredictionsFromOutput(
                output,
                std::vector<int64_t>{1, 5, 1},
                QStringList{QStringLiteral("item")},
                QSize(32, 32),
                transform,
                options,
                &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QCOMPARE(predictions.first().className, QStringLiteral("item"));
        QVERIFY(predictions.first().box.width > 0.0);
        QVERIFY(predictions.first().box.height > 0.0);
    }

    void detectionNcnnDflDecoderGeneratesBox()
    {
        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(4, 4);
        transform.targetSize = QSize(4, 4);
        transform.scale = 1.0;

        QVector<float> output;
        output.fill(0.0f, 9);
        output[8] = 8.0f;

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.50;
        QString error;
        const QVector<aitrain::DetectionPrediction> predictions =
            aitrain::detection_detail::ncnnDflPredictionsFromOutput(
                output.constData(),
                std::vector<int64_t>{1, 1, 9},
                QStringList{QStringLiteral("item")},
                QVector<int>{4},
                2,
                QSize(4, 4),
                transform,
                options,
                &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QCOMPARE(predictions.first().className, QStringLiteral("item"));
        QVERIFY(predictions.first().confidence > 0.50);
        QVERIFY(predictions.first().box.width > 0.0);
        QVERIFY(predictions.first().box.height > 0.0);
    }

    void detectionNcnnSyntheticSegmentationOutputsGenerateMask()
    {
        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(4, 4);
        transform.targetSize = QSize(4, 4);
        transform.scale = 1.0;

        const float boxesAndMasks[] = {2.0f, 2.0f, 4.0f, 4.0f, 0.95f, 8.0f};
        QVector<float> prototypes;
        prototypes.fill(1.0f, 16);

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.50;
        QString error;
        const QVector<aitrain::SegmentationPrediction> predictions =
            aitrain::detection_detail::yoloSegmentationPredictionsFromOutputs(
                boxesAndMasks,
                std::vector<int64_t>{1, 6, 1},
                prototypes.constData(),
                std::vector<int64_t>{1, 1, 4, 4},
                QStringList{QStringLiteral("item")},
                QSize(4, 4),
                transform,
                options,
                &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QVERIFY(!predictions.first().mask.isNull());
        QVERIFY(predictions.first().maskArea > 0.0);
    }

    void detectionNcnnDflSegmentationOutputsGenerateMask()
    {
        aitrain::LetterboxTransform transform;
        transform.sourceSize = QSize(4, 4);
        transform.targetSize = QSize(4, 4);
        transform.scale = 1.0;

        QVector<float> boxes;
        boxes.fill(0.0f, 9);
        boxes[8] = 8.0f;
        const float maskFeatures[] = {8.0f};
        QVector<float> prototypes;
        prototypes.fill(1.0f, 16);

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.50;
        QString error;
        const QVector<aitrain::SegmentationPrediction> predictions =
            aitrain::detection_detail::ncnnDflSegmentationPredictionsFromOutputs(
                boxes.constData(),
                std::vector<int64_t>{1, 1, 9},
                maskFeatures,
                1,
                1,
                prototypes.constData(),
                std::vector<int64_t>{1, 1, 4, 4},
                QStringList{QStringLiteral("item")},
                QVector<int>{4},
                2,
                QSize(4, 4),
                transform,
                options,
                &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QVERIFY(!predictions.first().mask.isNull());
        QVERIFY(predictions.first().maskArea > 0.0);
    }

    void detectionNcnnRuntimeReportsUnavailableWhenSdkMissing()
    {
        if (aitrain::isNcnnInferenceAvailable()) {
            QSKIP("NCNN SDK is enabled in this build.");
        }

        QString error;
        aitrain::DetectionInferenceOptions options;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionNcnnRuntime(
            QStringLiteral("model.param"),
            QStringLiteral("image.png"),
            options,
            &error);
        QVERIFY(predictions.isEmpty());
        QVERIFY(error.contains(QStringLiteral("NCNN")));
        QVERIFY(error.contains(QStringLiteral("not enabled")));
    }

    void deploymentValidationNcnnFailsWhenRuntimeMissing()
    {
        if (aitrain::isNcnnInferenceAvailable()) {
            QSKIP("NCNN SDK is enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString paramPath = dir.filePath(QStringLiteral("model.param"));
        const QString binPath = dir.filePath(QStringLiteral("model.bin"));
        const QString samplePath = dir.filePath(QStringLiteral("sample.png"));
        writeTextFile(paramPath, QStringLiteral("7767517\n1 1\nInput in0 0 1 in0 0=32 1=32 2=3\n"));
        writeTextFile(binPath, QStringLiteral("fake bin\n"));
        writeTinyPng(samplePath);

        QJsonObject options;
        options.insert(QStringLiteral("sampleImagePath"), samplePath);
        const aitrain::WorkflowResult result = aitrain::validateDeploymentArtifactReport(
            paramPath,
            dir.filePath(QStringLiteral("deployment-validation")),
            QStringLiteral("ncnn"),
            options);
        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.payload.value(QStringLiteral("status")).toString(), QStringLiteral("failed"));
        QCOMPARE(result.payload.value(QStringLiteral("runtime")).toString(), QStringLiteral("ncnn"));
        QVERIFY(result.payload.value(QStringLiteral("runtimeValidation")).toString() != QStringLiteral("artifact-only"));

        bool sawRuntimeCheck = false;
        for (const QJsonValue& value : result.payload.value(QStringLiteral("checks")).toArray()) {
            const QJsonObject check = value.toObject();
            if (check.value(QStringLiteral("name")).toString() == QStringLiteral("ncnn_runtime_available")) {
                sawRuntimeCheck = true;
                QCOMPARE(check.value(QStringLiteral("status")).toString(), QStringLiteral("failed"));
            }
        }
        QVERIFY(sawRuntimeCheck);
    }

    void deploymentValidationNcnnBlocksWhenSampleMissing()
    {
        if (!aitrain::isNcnnInferenceAvailable()) {
            QSKIP("NCNN SDK is not enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString paramPath = dir.filePath(QStringLiteral("model.param"));
        const QString binPath = dir.filePath(QStringLiteral("model.bin"));
        writeTextFile(paramPath, QStringLiteral("7767517\n1 1\nInput in0 0 1 in0 0=32 1=32 2=3\n"));
        writeTextFile(binPath, QStringLiteral("fake bin\n"));

        const aitrain::WorkflowResult result = aitrain::validateDeploymentArtifactReport(
            paramPath,
            dir.filePath(QStringLiteral("deployment-validation")),
            QStringLiteral("ncnn"),
            QJsonObject());
        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.payload.value(QStringLiteral("status")).toString(), QStringLiteral("blocked"));

        bool sawRuntimeCheck = false;
        bool sawSampleCheck = false;
        for (const QJsonValue& value : result.payload.value(QStringLiteral("checks")).toArray()) {
            const QJsonObject check = value.toObject();
            const QString name = check.value(QStringLiteral("name")).toString();
            if (name == QStringLiteral("ncnn_runtime_available")) {
                sawRuntimeCheck = true;
                QCOMPARE(check.value(QStringLiteral("status")).toString(), QStringLiteral("passed"));
            } else if (name == QStringLiteral("ncnn_sample_image_present")) {
                sawSampleCheck = true;
                QCOMPARE(check.value(QStringLiteral("status")).toString(), QStringLiteral("blocked"));
            }
        }
        QVERIFY(sawRuntimeCheck);
        QVERIFY(sawSampleCheck);
    }

    void deploymentValidationNcnnReportsUnsupportedLayerWithoutCrash()
    {
        if (!aitrain::isNcnnInferenceAvailable()) {
            QSKIP("NCNN SDK is not enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString paramPath = dir.filePath(QStringLiteral("model.param"));
        const QString binPath = dir.filePath(QStringLiteral("model.bin"));
        const QString samplePath = dir.filePath(QStringLiteral("sample.png"));
        writeTextFile(paramPath,
            QStringLiteral("7767517\n"
                           "2 2\n"
                           "Input images 0 1 images 0=32 1=32 2=3\n"
                           "Shape Shape_0 1 1 images output\n"));
        writeTextFile(binPath, QStringLiteral("fake bin\n"));
        writeTinyPng(samplePath);

        QJsonObject options;
        options.insert(QStringLiteral("sampleImagePath"), samplePath);
        options.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_segmentation"));
        options.insert(QStringLiteral("decoder"), QStringLiteral("ultralytics_output"));
        options.insert(QStringLiteral("classNames"), QJsonArray{QStringLiteral("item")});
        const aitrain::WorkflowResult result = aitrain::validateDeploymentArtifactReport(
            paramPath,
            dir.filePath(QStringLiteral("deployment-validation")),
            QStringLiteral("ncnn"),
            options);
        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.payload.value(QStringLiteral("status")).toString(), QStringLiteral("failed"));

        bool sawInferenceCheck = false;
        for (const QJsonValue& value : result.payload.value(QStringLiteral("checks")).toArray()) {
            const QJsonObject check = value.toObject();
            if (check.value(QStringLiteral("name")).toString() == QStringLiteral("ncnn_runtime_inference")) {
                sawInferenceCheck = true;
                QCOMPARE(check.value(QStringLiteral("status")).toString(), QStringLiteral("failed"));
                const QString message = check.value(QStringLiteral("message")).toString();
                QVERIFY(message.contains(QStringLiteral("unsupported layer type")));
                QVERIFY(message.contains(QStringLiteral("Shape")));
            }
        }
        QVERIFY(sawInferenceCheck);
    }

    void detectionTensorRtBackendStatusIsExplicit()
    {
        const aitrain::TensorRtBackendStatus status = aitrain::tensorRtBackendStatus();
        QVERIFY(!status.message.isEmpty());
        QVERIFY(status.status == QStringLiteral("sdk_missing")
            || status.status == QStringLiteral("backend_not_implemented")
            || status.status == QStringLiteral("backend_available"));
        QCOMPARE(status.toJson().value(QStringLiteral("inferenceAvailable")).toBool(), status.inferenceAvailable);

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            QStringLiteral("missing.onnx"),
            QStringLiteral("model.engine"),
            QStringLiteral("tensorrt"));
        QVERIFY(!exported.ok);
        QCOMPARE(exported.format, QStringLiteral("tensorrt"));
        if (status.exportAvailable) {
            QVERIFY(exported.error.contains(QStringLiteral("ONNX"))
                || exported.error.contains(QStringLiteral("onnx")));
        } else {
            QVERIFY(exported.error.contains(QStringLiteral("TensorRT")));
            QVERIFY(exported.error.contains(QStringLiteral("not available")));
            QVERIFY(exported.error.contains(status.message));
        }
    }

    void detectionTensorRtInferenceReportsClearMissingInput()
    {
        QString error;
        aitrain::DetectionInferenceOptions options;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionTensorRt(
            QStringLiteral("model.engine"),
            QStringLiteral("image.png"),
            options,
            &error);
        QVERIFY(predictions.isEmpty());
        QVERIFY(error.contains(QStringLiteral("TensorRT")));
        if (aitrain::isTensorRtInferenceAvailable()) {
            QVERIFY(error.contains(QStringLiteral("engine")));
        } else {
            QVERIFY(error.contains(QStringLiteral("not available")));
            QVERIFY(error.contains(aitrain::tensorRtBackendStatus().message));
        }
    }

    void detectionOnnxRuntimeReportsUnavailableWhenSdkMissing()
    {
        if (aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is enabled in this build.");
        }

        QString error;
        aitrain::DetectionInferenceOptions options;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(
            QStringLiteral("model.onnx"),
            QStringLiteral("image.png"),
            options,
            &error);
        QVERIFY(predictions.isEmpty());
        QVERIFY(error.contains(QStringLiteral("ONNX Runtime")));
        QVERIFY(error.contains(QStringLiteral("not enabled")));
    }

    void detectionOnnxRuntimeRunsExportedTinyDetector()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::DetectionTrainingOptions trainingOptions;
        trainingOptions.epochs = 1;
        trainingOptions.batchSize = 1;
        trainingOptions.imageSize = QSize(32, 32);
        trainingOptions.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, trainingOptions);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPath = dir.filePath(QStringLiteral("exports/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            exportPath,
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));

        QString error;
        aitrain::DetectionInferenceOptions inferenceOptions;
        inferenceOptions.confidenceThreshold = 0.0;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(
            exported.exportPath,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QVERIFY(!predictions.isEmpty());
        QVERIFY(predictions.first().confidence >= 0.0);
        QVERIFY(predictions.first().objectness >= 0.0);
        QCOMPARE(predictions.first().className, QStringLiteral("item"));
        QVERIFY(predictions.first().box.width > 0.0);
        QVERIFY(predictions.first().box.height > 0.0);
    }

    void detectionOnnxRuntimeMatchesCheckpointPrediction()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::DetectionTrainingOptions trainingOptions;
        trainingOptions.epochs = 2;
        trainingOptions.batchSize = 1;
        trainingOptions.imageSize = QSize(32, 32);
        trainingOptions.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, trainingOptions);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString imagePath = QDir(root).filePath(QStringLiteral("images/val/a.png"));
        QString error;
        aitrain::DetectionBaselineCheckpoint checkpoint;
        QVERIFY2(aitrain::loadDetectionBaselineCheckpoint(training.checkpointPath, &checkpoint, &error), qPrintable(error));

        aitrain::DetectionInferenceOptions inferenceOptions;
        inferenceOptions.confidenceThreshold = 0.0;
        inferenceOptions.maxDetections = 1;
        const QVector<aitrain::DetectionPrediction> checkpointPredictions = aitrain::predictDetectionBaseline(
            checkpoint,
            imagePath,
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(checkpointPredictions.size(), 1);

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            dir.filePath(QStringLiteral("exports/model.onnx")),
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));

        const QVector<aitrain::DetectionPrediction> onnxPredictions = aitrain::predictDetectionOnnxRuntime(
            exported.exportPath,
            imagePath,
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(onnxPredictions.size(), 1);

        const aitrain::DetectionPrediction& checkpointPrediction = checkpointPredictions.first();
        const aitrain::DetectionPrediction& onnxPrediction = onnxPredictions.first();
        QCOMPARE(onnxPrediction.box.classId, checkpointPrediction.box.classId);
        QCOMPARE(onnxPrediction.className, checkpointPrediction.className);
        QVERIFY(qAbs(onnxPrediction.objectness - checkpointPrediction.objectness) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.confidence - checkpointPrediction.confidence) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.xCenter - checkpointPrediction.box.xCenter) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.yCenter - checkpointPrediction.box.yCenter) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.width - checkpointPrediction.box.width) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.height - checkpointPrediction.box.height) <= 1.0e-5);
    }

    void detectionOnnxRuntimeRunsUltralyticsYoloDetectionSmokeModel()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString smokeRoot = phase9RealSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("Phase 9 real Ultralytics smoke artifacts are not available.");
        }
        QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("out/ultralytics_runs/phase9-real-smoke/weights/best.onnx"));
        QString imagePath = QDir(smokeRoot).filePath(QStringLiteral("dataset/images/val/a.png"));
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_detect/ultralytics_runs/acceptance-yolo-detect/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("generated/yolo_detect/images/val/b.png"));
        }
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_detect/ultralytics_runs/cpu-yolo-detect/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("yolo_detect/images/val/val_00.png"));
        }
        if (!QFileInfo::exists(onnxPath) || !QFileInfo::exists(imagePath)) {
            QSKIP("Phase 9 real Ultralytics smoke artifacts are not available.");
        }

        QString error;
        aitrain::DetectionInferenceOptions inferenceOptions;
        inferenceOptions.confidenceThreshold = 0.0;
        inferenceOptions.maxDetections = 10;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(
            onnxPath,
            imagePath,
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QVERIFY(!predictions.isEmpty());
        for (const aitrain::DetectionPrediction& prediction : predictions) {
            QVERIFY(prediction.confidence >= 0.0);
            QVERIFY(prediction.box.classId >= 0);
            QVERIFY(prediction.box.xCenter >= 0.0 && prediction.box.xCenter <= 1.0);
            QVERIFY(prediction.box.yCenter >= 0.0 && prediction.box.yCenter <= 1.0);
            QVERIFY(prediction.box.width > 0.0 && prediction.box.width <= 1.0);
            QVERIFY(prediction.box.height > 0.0 && prediction.box.height <= 1.0);
        }
    }

    void detectionExportCopiesUltralyticsYoloOnnxWithSidecar()
    {
        const QString smokeRoot = phase9RealSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("Phase 9 real Ultralytics smoke ONNX artifact is not available.");
        }
        const QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("out/ultralytics_runs/phase9-real-smoke/weights/best.onnx"));
        if (!QFileInfo::exists(onnxPath)) {
            QSKIP("Phase 9 real Ultralytics smoke ONNX artifact is not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString exportPath = dir.filePath(QStringLiteral("exports/copied-yolo.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            onnxPath,
            exportPath,
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("onnx"));
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_detect"));
        QCOMPARE(exported.config.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_detection"));
        QVERIFY(!exported.config.value(QStringLiteral("scaffold")).toBool(true));
        QCOMPARE(exported.config.value(QStringLiteral("classNames")).toArray().first().toString(), QStringLiteral("item"));
        QCOMPARE(exported.config.value(QStringLiteral("postprocess")).toObject().value(QStringLiteral("decoder")).toString(), QStringLiteral("yolo_v8_detection"));
    }

    void segmentationOnnxRuntimeRunsUltralyticsYoloSegmentationSmokeModel()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString smokeRoot = phase11SegSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("YOLO segmentation smoke ONNX artifact is not available.");
        }
        QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("out/ultralytics_runs/phase11-seg-smoke/weights/best.onnx"));
        QString imagePath = QDir(smokeRoot).filePath(QStringLiteral("dataset/images/val/a.png"));
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/aitrain-yolo-segment/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/acceptance-yolo-segment/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("generated/yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/cpu-yolo-segment/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("yolo_segment/images/val/val_00.png"));
        }
        if (!QFileInfo::exists(onnxPath) || !QFileInfo::exists(imagePath)) {
            QSKIP("YOLO segmentation smoke ONNX artifact is not available.");
        }

        QCOMPARE(aitrain::inferOnnxModelFamily(onnxPath), QStringLiteral("yolo_segmentation"));
        QString error;
        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.0;
        options.maxDetections = 5;
        const QVector<aitrain::SegmentationPrediction> predictions = aitrain::predictSegmentationOnnxRuntime(
            onnxPath,
            imagePath,
            options,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QVERIFY(!predictions.isEmpty());
        const QJsonObject predictionJson = aitrain::segmentationPredictionToJson(predictions.first());
        QCOMPARE(predictionJson.value(QStringLiteral("taskType")).toString(), QStringLiteral("segmentation"));
        QVERIFY(predictionJson.value(QStringLiteral("box")).toObject().value(QStringLiteral("xCenter")).toDouble() >= 0.0);
        QVERIFY(predictionJson.value(QStringLiteral("maskArea")).toDouble() >= 0.0);
        const QImage overlay = aitrain::renderSegmentationPredictions(imagePath, predictions, &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
    }

    void ocrRecOnnxRuntimeDecodesSmokeModel()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString smokeRoot = phase14OcrSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("OCR Rec smoke ONNX artifact is not available.");
        }
        const QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/paddleocr_rec/paddleocr_rec_ctc.onnx"));
        const QString imagePath = QFileInfo::exists(QDir(smokeRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png")))
            ? QDir(smokeRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png"))
            : QDir(smokeRoot).filePath(QStringLiteral("dataset/images/a.png"));
        if (!QFileInfo::exists(onnxPath) || !QFileInfo::exists(imagePath)) {
            QSKIP("OCR Rec smoke ONNX artifact is not available.");
        }

        QCOMPARE(aitrain::inferOnnxModelFamily(onnxPath), QStringLiteral("ocr_recognition"));
        QString error;
        const aitrain::OcrRecPrediction prediction = aitrain::predictOcrRecOnnxRuntime(onnxPath, imagePath, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        const QJsonObject predictionJson = aitrain::ocrRecPredictionToJson(prediction);
        QCOMPARE(predictionJson.value(QStringLiteral("taskType")).toString(), QStringLiteral("ocr_recognition"));
        QVERIFY(predictionJson.value(QStringLiteral("tokens")).toArray().size() > 0);
        QVERIFY(prediction.confidence >= 0.0);
        const QImage overlay = aitrain::renderOcrRecPrediction(imagePath, prediction, &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
    }

    void paddleOcrDetDbPostProcessExtractsTextBoxes()
    {
        QVector<float> probabilityMap(16 * 12, 0.01f);
        for (int y = 3; y <= 7; ++y) {
            for (int x = 4; x <= 10; ++x) {
                probabilityMap[y * 16 + x] = 0.92f;
            }
        }
        probabilityMap[0] = 0.99f;

        aitrain::OcrDetPostprocessOptions options;
        options.binaryThreshold = 0.3;
        options.boxThreshold = 0.5;
        options.minArea = 6;
        options.maxDetections = 5;
        QString error;
        const QVector<aitrain::OcrDetPrediction> predictions = aitrain::postProcessPaddleOcrDetDbMap(
            probabilityMap,
            QSize(16, 12),
            QSize(160, 120),
            options,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        const aitrain::OcrDetPrediction& prediction = predictions.first();
        QVERIFY(prediction.confidence > 0.9);
        QCOMPARE(prediction.pixelArea, 35);
        QCOMPARE(prediction.polygon.size(), 4);
        QCOMPARE(qRound(prediction.polygon.at(0).x()), 40);
        QCOMPARE(qRound(prediction.polygon.at(0).y()), 30);
        QCOMPARE(qRound(prediction.polygon.at(2).x()), 110);
        QCOMPARE(qRound(prediction.polygon.at(2).y()), 80);
        QVERIFY(prediction.box.xCenter > 0.45 && prediction.box.xCenter < 0.5);
        QVERIFY(prediction.box.yCenter > 0.45 && prediction.box.yCenter < 0.5);

        const QJsonObject predictionJson = aitrain::ocrDetPredictionToJson(prediction);
        QCOMPARE(predictionJson.value(QStringLiteral("taskType")).toString(), QStringLiteral("ocr_detection"));
        QCOMPARE(predictionJson.value(QStringLiteral("points")).toArray().size(), 4);

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString imagePath = dir.filePath(QStringLiteral("sample.png"));
        QImage image(160, 120, QImage::Format_RGB888);
        image.fill(Qt::white);
        QVERIFY(image.save(imagePath));
        const QImage overlay = aitrain::renderOcrDetPredictions(imagePath, predictions, &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
        QCOMPARE(overlay.size(), image.size());
    }

};

QTEST_MAIN(DetectionInferenceTests)
#include "tst_detection_inference.moc"
