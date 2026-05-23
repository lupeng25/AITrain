#include "TestSupport.h"

class OcrSegmentationWorkerTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase()
    {
        qputenv("AITRAIN_ENABLE_DIAGNOSTIC_BACKENDS", "1");
    }

    void removedTrainingBackendIsRejected()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("removed-backend");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = datasetRoot;
        request.outputPath = dir.filePath(QStringLiteral("run"));
        request.parameters.insert(QStringLiteral("trainingBackend"),
            QStringLiteral("tiny") + QStringLiteral("_linear") + QStringLiteral("_detector"));
        const QString scriptPath = dir.filePath(QStringLiteral("removed_backend_fixture.py"));
        writeTextFile(scriptPath, QStringLiteral("print('this removed backend fixture must not run')\n"));
        request.parameters.insert(QStringLiteral("pythonTrainerScript"), scriptPath);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        bool ok = true;
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
        QTRY_VERIFY_WITH_TIMEOUT(finished, 60000);
        QVERIFY(!ok);
        QVERIFY(finishedMessage.contains(QStringLiteral("not supported")));

        bool sawUnsupported = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("failed")) {
                sawUnsupported = message.second.value(QStringLiteral("errorCode")).toString()
                    == QStringLiteral("unsupported_training_backend");
            }
        }
        QVERIFY(sawUnsupported);
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);
    }

    void pythonTrainerProtocolUsesTemporaryFixtureOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);

        const QString scriptPath = dir.filePath(QStringLiteral("fixture_trainer.py"));
        writeTextFile(scriptPath,
            QStringLiteral(
                "import argparse, json, pathlib, sys\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--request', required=True)\n"
                "args = parser.parse_args()\n"
                "request = json.loads(pathlib.Path(args.request).read_text(encoding='utf-8'))\n"
                "out = pathlib.Path(request['outputPath'])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "report = out / 'official_fixture_training_report.json'\n"
                "report.write_text(json.dumps({'ok': True, 'backend': request['backend']}), encoding='utf-8')\n"
                "print(json.dumps({'type': 'artifact', 'payload': {'taskId': request['taskId'], 'kind': 'training_report', 'path': str(report), 'backend': request['backend']}}), flush=True)\n"
                "print(json.dumps({'type': 'completed', 'payload': {'taskId': request['taskId'], 'message': 'temporary fixture completed', 'backend': request['backend'], 'reportPath': str(report)}}), flush=True)\n"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("temporary-fixture");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = datasetRoot;
        request.outputPath = dir.filePath(QStringLiteral("run"));
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("pythonTrainerScript"), scriptPath);

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
        QTRY_VERIFY2_WITH_TIMEOUT(finished, qPrintable(finishedMessage), 60000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawReportArtifact = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("artifact")) {
                sawReportArtifact = sawReportArtifact
                    || message.second.value(QStringLiteral("kind")).toString() == QStringLiteral("training_report");
            }
        }
        QVERIFY(sawReportArtifact);
        QVERIFY(QFileInfo::exists(QDir(request.outputPath).filePath(QStringLiteral("official_fixture_training_report.json"))));
    }

    void workerRunsPaddleOcrRecOfficialAdapterPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR Rec adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetPath = QDir(dir.path()).filePath(QStringLiteral("ocr-rec"));
        writeTinyOcrRecDataset(datasetPath);
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-rec-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-rec-official-task");
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
            sawBackend = sawBackend
                || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_rec_official");
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
            QSKIP("Python executable is not available for the official PaddleOCR Det adapter test.");
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
            sawBackend = sawBackend
                || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_det_official");
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

    void workerEnvironmentCheckReportsOfficialTrainerProfiles()
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
            const QJsonObject tensorRt = profiles.value(QStringLiteral("tensorrt")).toObject();
            if (!yolo.isEmpty()) {
                sawYoloProfile = true;
                validateProfile(yolo);
            }
            if (!ocr.isEmpty()) {
                sawOcrProfile = true;
                validateProfile(ocr);
            }
            if (!tensorRt.isEmpty()) {
                sawTensorRtProfile = true;
                validateProfile(tensorRt);
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
};

QTEST_MAIN(OcrSegmentationWorkerTests)
#include "tst_ocr_segmentation_worker.moc"
