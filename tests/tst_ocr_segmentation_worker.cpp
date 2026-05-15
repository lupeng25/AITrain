#include "TestSupport.h"

class OcrSegmentationWorkerTests : public QObject {
    Q_OBJECT

private slots:
    void workerRunsOnnxInferenceEndToEnd()
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

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            dir.filePath(QStringLiteral("exports/model.onnx")),
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));

        const QString outputPath = dir.filePath(QStringLiteral("worker-inference"));
        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.requestInference(
            workerExecutablePath(),
            exported.exportPath,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            outputPath,
            &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            60000);
        QVERIFY2(ok, qPrintable(QStringLiteral("%1 messages=%2 outputExists=%3")
            .arg(finishedMessage)
            .arg(messages.size())
            .arg(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"))))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        QVERIFY(!messages.isEmpty());
        QCOMPARE(messages.last().first, QStringLiteral("completed"));

        bool sawInferenceResult = false;
        bool sawPredictionsArtifact = false;
        bool sawOverlayArtifact = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("inferenceResult")) {
                sawInferenceResult = true;
                QCOMPARE(message.second.value(QStringLiteral("predictionCount")).toInt(), 1);
                QVERIFY(message.second.value(QStringLiteral("elapsedMs")).toInt() >= 0);
            } else if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawPredictionsArtifact = sawPredictionsArtifact || kind == QStringLiteral("inference_predictions");
                sawOverlayArtifact = sawOverlayArtifact || kind == QStringLiteral("inference_overlay");
            }
        }
        QVERIFY(sawInferenceResult);
        QVERIFY(sawPredictionsArtifact);
        QVERIFY(sawOverlayArtifact);

        const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
        const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
        QVERIFY(QFileInfo::exists(predictionsPath));
        QVERIFY(QFileInfo::exists(overlayPath));
        QFile predictionsFile(predictionsPath);
        QVERIFY(predictionsFile.open(QIODevice::ReadOnly));
        const QJsonObject predictionsJson = QJsonDocument::fromJson(predictionsFile.readAll()).object();
        QCOMPARE(predictionsJson.value(QStringLiteral("runtime")).toString(), QStringLiteral("onnxruntime"));
        QCOMPARE(predictionsJson.value(QStringLiteral("predictions")).toArray().size(), 1);
        QCOMPARE(predictionsJson.value(QStringLiteral("predictions")).toArray().first().toObject().value(QStringLiteral("className")).toString(), QStringLiteral("item"));
    }

    void workerRoutesSegmentationAndOcrOnnxInferenceEndToEnd()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString segRoot = phase11SegSmokeRoot();
        const QString ocrRoot = phase14OcrSmokeRoot();
        if (segRoot.isEmpty() || ocrRoot.isEmpty()) {
            QSKIP("Segmentation/OCR ONNX smoke artifacts are not available.");
        }

        QString segOnnx = QDir(segRoot).filePath(QStringLiteral("out/ultralytics_runs/phase11-seg-smoke/weights/best.onnx"));
        QString segImage = QDir(segRoot).filePath(QStringLiteral("dataset/images/val/a.png"));
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/aitrain-yolo-segment/weights/best.onnx"));
            segImage = QDir(segRoot).filePath(QStringLiteral("yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/acceptance-yolo-segment/weights/best.onnx"));
            segImage = QDir(segRoot).filePath(QStringLiteral("generated/yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/cpu-yolo-segment/weights/best.onnx"));
            segImage = QDir(segRoot).filePath(QStringLiteral("yolo_segment/images/val/val_00.png"));
        }
        const QString ocrOnnx = QDir(ocrRoot).filePath(QStringLiteral("runs/paddleocr_rec/paddleocr_rec_ctc.onnx"));
        const QString ocrImage = QFileInfo::exists(QDir(ocrRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png")))
            ? QDir(ocrRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png"))
            : QDir(ocrRoot).filePath(QStringLiteral("dataset/images/a.png"));
        if (!QFileInfo::exists(segOnnx) || !QFileInfo::exists(segImage)
            || !QFileInfo::exists(ocrOnnx) || !QFileInfo::exists(ocrImage)) {
            QSKIP("Segmentation/OCR ONNX smoke artifacts are not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        auto runInference = [this, &dir](const QString& modelPath, const QString& imagePath, const QString& outputName, const QString& expectedTaskType) {
            const QString outputPath = dir.filePath(outputName);
            WorkerClient client;
            QVector<QPair<QString, QJsonObject>> messages;
            QStringList logs;
            bool finished = false;
            bool ok = false;
            QString finishedMessage;
            connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
                messages.append(qMakePair(type, payload));
            });
            connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
                logs.append(line);
            });
            connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
                finished = true;
                ok = result;
                finishedMessage = message;
            });

            QString error;
            QVERIFY2(client.requestInference(workerExecutablePath(), modelPath, imagePath, outputPath, &error), qPrintable(error));
            QTRY_VERIFY2_WITH_TIMEOUT(
                finished,
                qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
                20000);
            QVERIFY2(ok, qPrintable(finishedMessage));
            QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

            bool sawInferenceResult = false;
            bool sawPredictionsArtifact = false;
            bool sawOverlayArtifact = false;
            for (const auto& message : messages) {
                if (message.first == QStringLiteral("inferenceResult")) {
                    sawInferenceResult = true;
                    QCOMPARE(message.second.value(QStringLiteral("taskType")).toString(), expectedTaskType);
                    QVERIFY(message.second.value(QStringLiteral("predictionCount")).toInt() > 0);
                } else if (message.first == QStringLiteral("artifact")) {
                    const QString kind = message.second.value(QStringLiteral("kind")).toString();
                    sawPredictionsArtifact = sawPredictionsArtifact || kind == QStringLiteral("inference_predictions");
                    sawOverlayArtifact = sawOverlayArtifact || kind == QStringLiteral("inference_overlay");
                }
            }
            QVERIFY(sawInferenceResult);
            QVERIFY(sawPredictionsArtifact);
            QVERIFY(sawOverlayArtifact);

            const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
            const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
            QVERIFY(QFileInfo::exists(predictionsPath));
            QVERIFY(QFileInfo::exists(overlayPath));
            QFile predictionsFile(predictionsPath);
            QVERIFY(predictionsFile.open(QIODevice::ReadOnly));
            const QJsonObject predictionsJson = QJsonDocument::fromJson(predictionsFile.readAll()).object();
            QCOMPARE(predictionsJson.value(QStringLiteral("taskType")).toString(), expectedTaskType);
            QVERIFY(predictionsJson.value(QStringLiteral("predictions")).toArray().size() > 0);
        };

        runInference(segOnnx, segImage, QStringLiteral("worker-seg-onnx-inference"), QStringLiteral("segmentation"));
        runInference(ocrOnnx, ocrImage, QStringLiteral("worker-ocr-onnx-inference"), QStringLiteral("ocr_recognition"));
    }

    void ocrRecTrainerWritesScaffoldCheckpointAndPreview()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);

        aitrain::OcrRecTrainingOptions options;
        options.epochs = 3;
        options.batchSize = 1;
        options.imageSize = QSize(32, 16);
        options.learningRate = 0.1;
        options.maxTextLength = 8;
        options.outputPath = dir.filePath(QStringLiteral("run"));

        double firstLoss = -1.0;
        double lastLoss = -1.0;
        int callbackCount = 0;
        const aitrain::OcrRecTrainingResult result = aitrain::trainOcrRecBaseline(
            root,
            options,
            [&firstLoss, &lastLoss, &callbackCount](const aitrain::OcrRecTrainingMetrics& metrics) {
                ++callbackCount;
                if (firstLoss < 0.0) {
                    firstLoss = metrics.loss;
                }
                lastLoss = metrics.loss;
                return metrics.accuracy > 0.0;
            });

        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.steps, 6);
        QCOMPARE(callbackCount, 6);
        QVERIFY(firstLoss >= 0.0);
        QVERIFY(lastLoss >= 0.0);
        QVERIFY(lastLoss < firstLoss);
        QCOMPARE(result.accuracy, 1.0);
        QCOMPARE(result.editDistance, 0.0);
        QVERIFY(QFileInfo::exists(result.checkpointPath));
        QVERIFY(QFileInfo::exists(result.previewPath));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("type")).toString(), QStringLiteral("tiny_ocr_recognition_scaffold"));
        QVERIFY(json.value(QStringLiteral("note")).toString().contains(QStringLiteral("Scaffold")));
        QCOMPARE(json.value(QStringLiteral("steps")).toInt(), 6);
        QCOMPARE(json.value(QStringLiteral("accuracy")).toDouble(), 1.0);
        QCOMPARE(json.value(QStringLiteral("editDistance")).toDouble(), 0.0);
        QCOMPARE(json.value(QStringLiteral("modelHead")).toString(), QStringLiteral("label_echo_ctc_scaffold"));

        QFile preview(result.previewPath);
        QVERIFY(preview.open(QIODevice::ReadOnly));
        const QJsonObject previewJson = QJsonDocument::fromJson(preview.readAll()).object();
        QCOMPARE(previewJson.value(QStringLiteral("label")).toString(), QStringLiteral("ab12"));
        QCOMPARE(previewJson.value(QStringLiteral("prediction")).toString(), QStringLiteral("ab12"));
    }

    void workerRunsSegmentationTrainingScaffoldEndToEnd()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinySegmentationDataset(root);
        const QString outputPath = dir.filePath(QStringLiteral("worker-segmentation"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("seg-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("segmentation");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("epochs"), 2);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageSize"), 16);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawCheckpoint = false;
        bool sawPreview = false;
        bool sawMaskPreview = false;
        bool sawMaskLoss = false;
        bool sawMaskCoverage = false;
        bool sawMaskIou = false;
        bool sawPrecision = false;
        bool sawRecall = false;
        bool sawSegmentationMap50 = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawCheckpoint = sawCheckpoint || kind == QStringLiteral("checkpoint");
                sawPreview = sawPreview || kind == QStringLiteral("preview");
                sawMaskPreview = sawMaskPreview || kind == QStringLiteral("mask_preview");
            } else if (message.first == QStringLiteral("metric")) {
                const QString name = message.second.value(QStringLiteral("name")).toString();
                sawMaskLoss = sawMaskLoss || name == QStringLiteral("maskLoss");
                sawMaskCoverage = sawMaskCoverage || name == QStringLiteral("maskCoverage");
                sawMaskIou = sawMaskIou || name == QStringLiteral("maskIoU");
                sawPrecision = sawPrecision || name == QStringLiteral("precision");
                sawRecall = sawRecall || name == QStringLiteral("recall");
                sawSegmentationMap50 = sawSegmentationMap50 || name == QStringLiteral("segmentationMap50");
            }
        }
        QVERIFY(sawCheckpoint);
        QVERIFY(sawPreview);
        QVERIFY(sawMaskPreview);
        QVERIFY(sawMaskLoss);
        QVERIFY(sawMaskCoverage);
        QVERIFY(sawMaskIou);
        QVERIFY(sawPrecision);
        QVERIFY(sawRecall);
        QVERIFY(sawSegmentationMap50);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("checkpoint_latest.aitrain"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("preview_latest.png"))));
        QCOMPARE(messages.last().first, QStringLiteral("completed"));
    }

    void workerRunsPythonTrainerMockEndToEnd()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        const QString trainerScript = mockPythonTrainerScriptPath();
        QVERIFY2(!trainerScript.isEmpty(), "Mock Python trainer script is not available.");

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);
        const QString outputPath = dir.filePath(QStringLiteral("worker-python-mock"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("python-mock-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("python_mock"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("pythonTrainerScript"), trainerScript);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("mockStepsPerEpoch"), 2);
        request.parameters.insert(QStringLiteral("mockSleepMs"), 0);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool success, const QString& message) {
            finished = true;
            ok = success;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawLoss = false;
        bool sawCheckpoint = false;
        bool sawReport = false;
        bool sawPythonBackend = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("metric")) {
                sawLoss = sawLoss || message.second.value(QStringLiteral("name")).toString() == QStringLiteral("loss");
            } else if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawCheckpoint = sawCheckpoint || kind == QStringLiteral("checkpoint");
                sawReport = sawReport || kind == QStringLiteral("training_report");
                sawPythonBackend = sawPythonBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("python_mock");
            } else if (message.first == QStringLiteral("completed")) {
                sawPythonBackend = sawPythonBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("python_mock");
            }
        }
        QVERIFY(sawLoss);
        QVERIFY(sawCheckpoint);
        QVERIFY(sawReport);
        QVERIFY(sawPythonBackend);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("python_trainer_request.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("python_mock_checkpoint.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("python_mock_training_report.json"))));
    }

    void workerPropagatesPythonTrainerMockFailure()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        const QString trainerScript = mockPythonTrainerScriptPath();
        QVERIFY2(!trainerScript.isEmpty(), "Mock Python trainer script is not available.");

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("python-mock-fail-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = root;
        request.outputPath = dir.filePath(QStringLiteral("worker-python-mock-fail"));
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("python_mock"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("pythonTrainerScript"), trainerScript);
        request.parameters.insert(QStringLiteral("mockMode"), QStringLiteral("fail"));

        WorkerClient client;
        bool finished = false;
        bool ok = true;
        QString finishedMessage;
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool success, const QString& message) {
            finished = true;
            ok = success;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY(!ok);
        QVERIFY(finishedMessage.contains(QStringLiteral("Mock Python trainer failure requested")));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);
    }

    void workerRunsUltralyticsYoloDetectionAdapterWithFakeOfficialPackage()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        const QDir rootDir(root);
        writeTextFile(rootDir.filePath(QStringLiteral("data.yaml")),
            QStringLiteral("path: raw\ntrain: custom/images/train\nval: custom/images/val\nnc: 1\nnames: [item]\n"));
        writeTinyPng(rootDir.filePath(QStringLiteral("raw/custom/images/train/a.png")));
        writeTinyPng(rootDir.filePath(QStringLiteral("raw/custom/images/val/b.png")));
        writeTextFile(rootDir.filePath(QStringLiteral("raw/custom/labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(rootDir.filePath(QStringLiteral("raw/custom/labels/val/b.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        const QString fakePackageRoot = dir.filePath(QStringLiteral("fake-pythonpath"));
        writeFakeUltralyticsPackage(fakePackageRoot);
        const QString outputPath = dir.filePath(QStringLiteral("worker-ultralytics-yolo"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("ultralytics-yolo-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("pythonPathPrepend"), fakePackageRoot);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageSize"), 64);
        request.parameters.insert(QStringLiteral("device"), QStringLiteral("cpu"));
        request.parameters.insert(QStringLiteral("model"), QStringLiteral("fake-yolo.pt"));
        request.parameters.insert(QStringLiteral("runName"), QStringLiteral("fake-run"));
        request.parameters.insert(QStringLiteral("exportOnnx"), true);
        request.parameters.insert(QStringLiteral("compactEvents"), true);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawCompletedOnnx = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("ultralytics_yolo_detect");
            if (message.first == QStringLiteral("completed")) {
                sawCompletedOnnx = !message.second.value(QStringLiteral("onnxPath")).toString().isEmpty();
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawCompletedOnnx);
        const QString normalizedDataYamlPath = QDir(outputPath).filePath(QStringLiteral("aitrain_yolo_data.yaml"));
        QVERIFY(QFileInfo::exists(normalizedDataYamlPath));
        QFile normalizedDataYaml(normalizedDataYamlPath);
        QVERIFY(normalizedDataYaml.open(QIODevice::ReadOnly | QIODevice::Text));
        const QString normalizedDataYamlText = QString::fromUtf8(normalizedDataYaml.readAll());
        QVERIFY(normalizedDataYamlText.contains(QStringLiteral("raw")));
        QVERIFY(normalizedDataYamlText.contains(QStringLiteral("custom/images/train")));
        QVERIFY(normalizedDataYamlText.contains(QStringLiteral("custom/images/val")));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("ultralytics_training_report.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("ultralytics_runs/fake-run/weights/best.pt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("ultralytics_runs/fake-run/weights/best.onnx"))));
    }

    void workerRunsPaddleOcrOfficialAdapterPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetPath = QDir(dir.path()).filePath(QStringLiteral("ocr-rec"));
        writeTinyOcrRecDataset(datasetPath);
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-official-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_recognition");
        request.datasetPath = datasetPath;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_rec_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageWidth"), 96);
        request.parameters.insert(QStringLiteral("imageHeight"), 32);
        request.parameters.insert(QStringLiteral("maxTextLength"), 8);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawPrepareOnly = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_rec_official");
            if (message.first == QStringLiteral("completed")) {
                sawPrepareOnly = message.second.value(QStringLiteral("mode")).toString() == QStringLiteral("prepareOnly");
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawPrepareOnly);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("aitrain_ppocrv4_rec.yml"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/train_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/val_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_rec_report.json"))));
    }

    void workerRunsPaddleOcrDetOfficialAdapterPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR det adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetPath = QDir(dir.path()).filePath(QStringLiteral("ocr-det"));
        writeTinyOcrDetDataset(datasetPath);
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-det-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-det-official-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_detection");
        request.datasetPath = datasetPath;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_det_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageSize"), 64);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawPrepareOnly = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_det_official");
            if (message.first == QStringLiteral("completed")) {
                sawPrepareOnly = message.second.value(QStringLiteral("mode")).toString() == QStringLiteral("prepareOnly");
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawPrepareOnly);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("aitrain_ppocrv4_det.yml"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/train_det_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/val_det_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_det_report.json"))));
    }

    void workerRunsPaddleOcrSystemOfficialAdapterPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR system adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-system-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-system-official-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr");
        request.datasetPath = dir.filePath(QStringLiteral("image.png"));
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_system_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("detModelDir"), dir.filePath(QStringLiteral("det_model")));
        request.parameters.insert(QStringLiteral("recModelDir"), dir.filePath(QStringLiteral("rec_model")));
        request.parameters.insert(QStringLiteral("dictionaryFile"), dir.filePath(QStringLiteral("dict.txt")));
        request.parameters.insert(QStringLiteral("inferenceImage"), dir.filePath(QStringLiteral("image.png")));

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawPrepareOnly = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_system_official");
            if (message.first == QStringLiteral("completed")) {
                sawPrepareOnly = message.second.value(QStringLiteral("mode")).toString() == QStringLiteral("prepareOnly");
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawPrepareOnly);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("run_official_system_predict.ps1"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_system_report.json"))));
    }

    void workerEnvironmentCheckReportsPythonTrainerBackends()
    {
        QTemporaryDir reportDir;
        QVERIFY(reportDir.isValid());
        ScopedEnvVar reportDirEnv("AITRAIN_ENVIRONMENT_REPORT_DIR", reportDir.path().toLocal8Bit());

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::idle, this, [&finished]() {
            finished = true;
        });

        QString error;
        QVERIFY2(client.requestEnvironmentCheck(workerExecutablePath(), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);

        bool sawPython = false;
        bool sawUltralytics = false;
        bool sawPaddleOcr = false;
        bool sawPaddle = false;
        bool sawProfiles = false;
        bool sawYoloProfile = false;
        bool sawOcrProfile = false;
        bool sawTensorRtProfile = false;
        bool sawProfileArtifact = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("artifact")
                && message.second.value(QStringLiteral("kind")).toString() == QStringLiteral("environment_profiles_report")) {
                sawProfileArtifact = true;
                QVERIFY(!message.second.value(QStringLiteral("path")).toString().isEmpty());
            }
            if (message.first != QStringLiteral("environmentCheck")) {
                continue;
            }
            const QJsonArray checks = message.second.value(QStringLiteral("checks")).toArray();
            for (const QJsonValue& value : checks) {
                const QJsonObject check = value.toObject();
                const QString name = check.value(QStringLiteral("name")).toString();
                sawPython = sawPython || name == QStringLiteral("Python");
                sawUltralytics = sawUltralytics || name == QStringLiteral("Ultralytics YOLO");
                sawPaddleOcr = sawPaddleOcr || name == QStringLiteral("PaddleOCR");
                sawPaddle = sawPaddle || name == QStringLiteral("PaddlePaddle");
            }

            const QJsonObject profiles = message.second.value(QStringLiteral("profiles")).toObject();
            sawProfiles = !profiles.isEmpty();
            const auto validateProfile = [](const QJsonObject& profile) {
                QVERIFY(!profile.value(QStringLiteral("status")).toString().isEmpty());
                QVERIFY(profile.value(QStringLiteral("checks")).isArray());
                QVERIFY(profile.value(QStringLiteral("repairHints")).isArray());
            };
            const QJsonObject yolo = profiles.value(QStringLiteral("yolo")).toObject();
            const QJsonObject ocr = profiles.value(QStringLiteral("ocr")).toObject();
            const QJsonObject tensorrt = profiles.value(QStringLiteral("tensorrt")).toObject();
            if (!yolo.isEmpty()) {
                sawYoloProfile = true;
                validateProfile(yolo);
            }
            if (!ocr.isEmpty()) {
                sawOcrProfile = true;
                validateProfile(ocr);
            }
            if (!tensorrt.isEmpty()) {
                sawTensorRtProfile = true;
                validateProfile(tensorrt);
            }
        }
        QVERIFY(sawPython);
        QVERIFY(sawUltralytics);
        QVERIFY(sawPaddleOcr);
        QVERIFY(sawPaddle);
        QVERIFY(sawProfiles);
        QVERIFY(sawYoloProfile);
        QVERIFY(sawOcrProfile);
        QVERIFY(sawTensorRtProfile);
        QVERIFY(sawProfileArtifact);
    }

    void workerRunsOcrRecognitionTrainingScaffoldEndToEnd()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);
        const QString outputPath = dir.filePath(QStringLiteral("worker-ocr"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("ocr-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_recognition");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("epochs"), 2);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageWidth"), 32);
        request.parameters.insert(QStringLiteral("imageHeight"), 16);
        request.parameters.insert(QStringLiteral("maxTextLength"), 8);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawCheckpoint = false;
        bool sawPreview = false;
        bool sawCtcLoss = false;
        bool sawAccuracy = false;
        bool sawEditDistance = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawCheckpoint = sawCheckpoint || kind == QStringLiteral("checkpoint");
                sawPreview = sawPreview || kind == QStringLiteral("preview");
            } else if (message.first == QStringLiteral("metric")) {
                const QString name = message.second.value(QStringLiteral("name")).toString();
                sawCtcLoss = sawCtcLoss || name == QStringLiteral("ctcLoss");
                sawAccuracy = sawAccuracy || name == QStringLiteral("accuracy");
                sawEditDistance = sawEditDistance || name == QStringLiteral("editDistance");
            }
        }
        QVERIFY(sawCheckpoint);
        QVERIFY(sawPreview);
        QVERIFY(sawCtcLoss);
        QVERIFY(sawAccuracy);
        QVERIFY(sawEditDistance);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("checkpoint_latest.aitrain"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("preview_latest.json"))));
        QCOMPARE(messages.last().first, QStringLiteral("completed"));
    }
};

QTEST_MAIN(OcrSegmentationWorkerTests)
#include "tst_ocr_segmentation_worker.moc"
