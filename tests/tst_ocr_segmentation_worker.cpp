#include "TestSupport.h"

namespace {

QString writeFakePaddleOcrRepo(const QString& root)
{
    QDir repo(root);
    if (!repo.mkpath(QStringLiteral("tools/infer"))
        || !repo.mkpath(QStringLiteral("configs/rec/PP-OCRv5/multi_language"))
        || !repo.mkpath(QStringLiteral("configs/det/PP-OCRv5"))
        || !repo.mkpath(QStringLiteral("ppocr/utils/dict"))) {
        return {};
    }
    writeTextFile(repo.filePath(QStringLiteral("tools/train.py")), QStringLiteral("# fake train\n"));
    writeTextFile(repo.filePath(QStringLiteral("tools/export_model.py")), QStringLiteral("# fake export\n"));
    writeTextFile(repo.filePath(QStringLiteral("tools/infer/predict_system.py")), QStringLiteral("# fake system predict\n"));
    writeTextFile(repo.filePath(QStringLiteral("ppocr/utils/dict/ppocrv5_dict.txt")), QStringLiteral("a\nb\n1\n2\nz\n"));
    writeTextFile(repo.filePath(QStringLiteral("ppocr/utils/dict/ppocrv5_en_dict.txt")), QStringLiteral("a\nb\n1\n2\nz\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_mobile_rec\n"
            "Architecture:\n"
            "  model_type: rec\n"
            "  algorithm: SVTR_LCNet\n"
            "PostProcess:\n"
            "  name: CTCLabelDecode\n"
            "Metric:\n"
            "  name: RecMetric\n"
            "Train:\n"
            "  dataset:\n"
            "    transforms:\n"
            "    - RecConAug:\n"
            "        image_shape: [48, 320, 3]\n"
            "        max_text_length: 25\n"
            "  loader: {}\n"
            "  sampler: {}\n"
            "Eval:\n"
            "  dataset:\n"
            "    transforms:\n"
            "    - RecResizeImg:\n"
            "        image_shape: [3, 48, 320]\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_server_rec\n"
            "Architecture:\n"
            "  model_type: rec\n"
            "  algorithm: SVTR_HGNet\n"
            "PostProcess:\n"
            "  name: CTCLabelDecode\n"
            "Metric:\n"
            "  name: RecMetric\n"
            "Train:\n"
            "  dataset: {}\n"
            "  loader: {}\n"
            "  sampler: {}\n"
            "Eval:\n"
            "  dataset: {}\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/rec/PP-OCRv5/multi_language/en_PP-OCRv5_mobile_rec.yaml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: en_PP-OCRv5_mobile_rec\n"
            "Architecture:\n"
            "  model_type: rec\n"
            "  algorithm: SVTR_LCNet\n"
            "PostProcess:\n"
            "  name: CTCLabelDecode\n"
            "Metric:\n"
            "  name: RecMetric\n"
            "Train:\n"
            "  dataset: {}\n"
            "  loader: {}\n"
            "  sampler: {}\n"
            "Eval:\n"
            "  dataset: {}\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_mobile_det\n"
            "Architecture:\n"
            "  model_type: det\n"
            "  algorithm: DB\n"
            "PostProcess:\n"
            "  name: DBPostProcess\n"
            "Metric:\n"
            "  name: DetMetric\n"
            "Train:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"
            "Eval:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/det/PP-OCRv5/PP-OCRv5_server_det.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_server_det\n"
            "Architecture:\n"
            "  model_type: det\n"
            "  algorithm: DB\n"
            "PostProcess:\n"
            "  name: DBPostProcess\n"
            "Metric:\n"
            "  name: DetMetric\n"
            "Train:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"
            "Eval:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"));
    return repo.absolutePath();
}

bool pythonCanImportModule(const QString& python, const QString& module)
{
    QProcess process;
    process.start(python, QStringList() << QStringLiteral("-c") << QStringLiteral("import %1").arg(module));
    return process.waitForStarted(1000)
        && process.waitForFinished(5000)
        && process.exitStatus() == QProcess::NormalExit
        && process.exitCode() == 0;
}

void writeFakeUltralyticsOnnxExportPackage(const QString& root)
{
    writeTextFile(
        QDir(root).filePath(QStringLiteral("ultralytics/__init__.py")),
        QStringLiteral(
            "from pathlib import Path\n"
            "__version__ = 'test-fixture'\n"
            "\n"
            "class YOLO:\n"
            "    def __init__(self, model):\n"
            "        self.model = str(model)\n"
            "        self.task = 'detect'\n"
            "\n"
            "    def export(self, **kwargs):\n"
            "        import onnx\n"
            "        from onnx import TensorProto, helper\n"
            "        model_path = Path(self.model)\n"
            "        output_path = model_path.with_suffix('.onnx')\n"
            "        output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "        input_info = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 32, 32])\n"
            "        output_info = helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 5, 1])\n"
            "        values = helper.make_tensor('values', TensorProto.FLOAT, [1, 5, 1], [16.0, 16.0, 8.0, 8.0, 0.10])\n"
            "        node = helper.make_node('Constant', inputs=[], outputs=['output0'], value=values)\n"
            "        graph = helper.make_graph([node], 'aitrain_fake_yolo', [input_info], [output_info])\n"
            "        model = helper.make_model(graph, producer_name='aitrain-test', opset_imports=[helper.make_opsetid('', 13)])\n"
            "        model.ir_version = 8\n"
            "        onnx.save(model, output_path)\n"
            "        return str(output_path)\n"));
}

QString repoRelativeFilePath(const QString& relative)
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir::current().absoluteFilePath(relative),
        QDir(applicationDir).absoluteFilePath(relative),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../%1").arg(relative)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../%1").arg(relative)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../../%1").arg(relative)),
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return QDir::current().absoluteFilePath(relative);
}

} // namespace

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

    void mismatchedTrainingBackendIsRejected()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("mismatched-backend");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = datasetRoot;
        request.outputPath = dir.filePath(QStringLiteral("run"));
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_segment"));

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

        bool sawMismatch = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("failed")) {
                sawMismatch = message.second.value(QStringLiteral("errorCode")).toString()
                    == QStringLiteral("training_backend_task_mismatch");
            }
        }
        QVERIFY(sawMismatch);
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);
    }

    void trainingSnapshotMismatchIsRejected()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);

        const QString snapshotOutput = dir.filePath(QStringLiteral("snapshot"));
        const aitrain::WorkflowResult snapshot = aitrain::createDatasetSnapshotReport(
            datasetRoot,
            snapshotOutput,
            QStringLiteral("yolo_detection"),
            QJsonObject{{QStringLiteral("maxFiles"), 20000}});
        QVERIFY2(snapshot.ok, qPrintable(snapshot.error));
        const QString snapshotHash = snapshot.payload.value(QStringLiteral("contentHash")).toString();
        QVERIFY(!snapshotHash.isEmpty());

        writeTextFile(QDir(datasetRoot).filePath(QStringLiteral("labels/train/a.txt")),
            QStringLiteral("0 0.5 0.5 0.30 0.30\n"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("snapshot-mismatch");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = datasetRoot;
        request.outputPath = dir.filePath(QStringLiteral("run"));
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect"));
        request.parameters.insert(QStringLiteral("datasetFormat"), QStringLiteral("yolo_detection"));
        request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshotHash);
        request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.reportPath);

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

        bool sawSnapshotMismatch = false;
        QStringList receivedMessages;
        for (const auto& message : messages) {
            receivedMessages.append(QStringLiteral("%1 %2").arg(
                message.first,
                QString::fromUtf8(QJsonDocument(message.second).toJson(QJsonDocument::Compact))));
            if (message.first == QStringLiteral("failed")) {
                const QJsonObject details = message.second.value(QStringLiteral("details")).toObject();
                sawSnapshotMismatch = message.second.value(QStringLiteral("errorCode")).toString()
                    == QStringLiteral("dataset_snapshot_mismatch")
                    && !details.value(QStringLiteral("currentSnapshotHash")).toString().isEmpty();
            }
        }
        QVERIFY2(sawSnapshotMismatch, qPrintable(receivedMessages.join(QStringLiteral("\n"))));
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

    void pythonTrainerExtractsJsonAfterProgressRedraw()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);

        const QString scriptPath = dir.filePath(QStringLiteral("mixed_line_trainer.py"));
        writeTextFile(scriptPath,
            QStringLiteral(
                "import argparse, json, pathlib, sys\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--request', required=True)\n"
                "args = parser.parse_args()\n"
                "request = json.loads(pathlib.Path(args.request).read_text(encoding='utf-8'))\n"
                "out = pathlib.Path(request['outputPath'])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "progress = {'type': 'progress', 'payload': {'taskId': request['taskId'], 'phase': 'train', 'percent': 7, 'backend': request['backend'], 'liveMetrics': {'loss': 1.25}}}\n"
                "sys.stdout.write('\\x1b[K        1/1 0G 0% 0/2 1.0it/s' + json.dumps(progress) + '\\r\\n')\n"
                "sys.stdout.flush()\n"
                "print(json.dumps({'type': 'completed', 'payload': {'taskId': request['taskId'], 'message': 'mixed line completed', 'backend': request['backend']}}), flush=True)\n"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("mixed-line-fixture");
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
            60000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawMixedProgress = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("progress")
                && message.second.value(QStringLiteral("percent")).toInt() == 7
                && message.second.value(QStringLiteral("liveMetrics")).toObject().value(QStringLiteral("loss")).toDouble() == 1.25) {
                sawMixedProgress = true;
            }
        }
        QVERIFY2(sawMixedProgress, "Worker dropped the JSON progress event embedded after a progress redraw.");
    }

    void workerRunsOfficialYoloEvaluationAdapter()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);
        const QString fakePackageRoot = dir.filePath(QStringLiteral("fake_ultralytics"));
        QVERIFY(QDir().mkpath(fakePackageRoot));
        writeFakeUltralyticsPackage(fakePackageRoot);
        const QString modelPath = dir.filePath(QStringLiteral("best.pt"));
        writeTextFile(modelPath, QStringLiteral("fake checkpoint\n"));
        const QString outputPath = dir.filePath(QStringLiteral("evaluation"));

        QJsonObject options;
        options.insert(QStringLiteral("pythonExecutable"), python);
        options.insert(QStringLiteral("pythonPathPrepend"), fakePackageRoot);
        options.insert(QStringLiteral("ultralyticsValArgs"), QJsonObject{
            {QStringLiteral("split"), QStringLiteral("val")},
            {QStringLiteral("plots"), true},
            {QStringLiteral("save_json"), true}
        });

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
        QVERIFY2(client.requestModelEvaluation(
            workerExecutablePath(),
            modelPath,
            datasetRoot,
            outputPath,
            QStringLiteral("detection"),
            options,
            &error,
            QStringLiteral("official-yolo-eval-fixture")), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(finished, qPrintable(finishedMessage), 60000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        const QString reportPath = QDir(outputPath).filePath(QStringLiteral("evaluation_report.json"));
        const QJsonObject report = readJsonObject(reportPath);
        QCOMPARE(report.value(QStringLiteral("evaluationSource")).toString(), QStringLiteral("ultralytics_official_val"));
        QCOMPARE(report.value(QStringLiteral("runtime")).toString(), QStringLiteral("ultralytics_official_val"));
        QVERIFY(report.value(QStringLiteral("metrics")).toObject().contains(QStringLiteral("mAP50")));
        QVERIFY(!report.value(QStringLiteral("metrics")).toObject().contains(QStringLiteral("cocoMap50_95")));
        QVERIFY(QFileInfo::exists(report.value(QStringLiteral("officialMetricsPath")).toString()));

        bool sawOfficialMetrics = false;
        bool sawEvaluationReport = false;
        for (const auto& message : messages) {
            if (message.first != QStringLiteral("artifact")) {
                continue;
            }
            sawEvaluationReport = sawEvaluationReport
                || message.second.value(QStringLiteral("kind")).toString() == QStringLiteral("evaluation_report");
            sawOfficialMetrics = sawOfficialMetrics
                || message.second.value(QStringLiteral("kind")).toString() == QStringLiteral("official_metrics");
        }
        QVERIFY(sawEvaluationReport);
        QVERIFY(sawOfficialMetrics);
    }

    void pipelinePreExportsYoloPtForExportInferBenchmark()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        if (!pythonCanImportModule(python, QStringLiteral("onnx"))) {
            QSKIP("Python onnx package is not available for the fake YOLO export fixture.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);
        const QString fakePackageRoot = dir.filePath(QStringLiteral("fake_ultralytics_export"));
        QVERIFY(QDir().mkpath(fakePackageRoot));
        writeFakeUltralyticsOnnxExportPackage(fakePackageRoot);

        const QString checkpointPath = dir.filePath(QStringLiteral("weights/external.pt"));
        writeTextFile(checkpointPath, QStringLiteral("fake external yolo checkpoint\n"));
        const QString outputPath = dir.filePath(QStringLiteral("pipeline"));
        const QString sampleImagePath = QDir(datasetRoot).filePath(QStringLiteral("images/val/a.png"));

        QJsonObject options;
        options.insert(QStringLiteral("modelPath"), checkpointPath);
        options.insert(QStringLiteral("datasetPath"), datasetRoot);
        options.insert(QStringLiteral("datasetFormat"), QStringLiteral("yolo_detection"));
        options.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
        options.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect"));
        options.insert(QStringLiteral("exportFormat"), QStringLiteral("onnx"));
        options.insert(QStringLiteral("sampleImagePath"), sampleImagePath);
        options.insert(QStringLiteral("pythonExecutable"), python);
        options.insert(QStringLiteral("pythonPathPrepend"), fakePackageRoot);
        options.insert(QStringLiteral("ultralyticsExportArgs"), QJsonObject{
            {QStringLiteral("format"), QStringLiteral("onnx")},
            {QStringLiteral("imgsz"), 32},
            {QStringLiteral("batch"), 1},
            {QStringLiteral("device"), QStringLiteral("cpu")}
        });

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
        QVERIFY2(client.requestLocalPipeline(
            workerExecutablePath(),
            outputPath,
            QStringLiteral("export-infer-benchmark-report"),
            options,
            &error,
            QStringLiteral("pipeline-pt-preexport-fixture")), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            60000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        const QString officialOnnxPath = QDir(outputPath).filePath(QStringLiteral("official_yolo_export/model.onnx"));
        const QString pipelineReportPath = QDir(outputPath).filePath(QStringLiteral("local_pipeline_plan.json"));
        const QString finalExportPath = QDir(outputPath).filePath(QStringLiteral("export/model.onnx"));
        QVERIFY(QFileInfo::exists(officialOnnxPath));
        QVERIFY(QFileInfo::exists(pipelineReportPath));
        QVERIFY(QFileInfo::exists(finalExportPath));

        const QJsonObject report = readJsonObject(pipelineReportPath);
        QCOMPARE(report.value(QStringLiteral("state")).toString(), QStringLiteral("completed"));
        QCOMPARE(QFileInfo(report.value(QStringLiteral("pipelineOfficialExportPath")).toString()).absoluteFilePath(), QFileInfo(officialOnnxPath).absoluteFilePath());
        QCOMPARE(report.value(QStringLiteral("pipelineOfficialExportSourceCheckpointPath")).toString(), checkpointPath);
        QCOMPARE(QFileInfo(report.value(QStringLiteral("modelPath")).toString()).absoluteFilePath(), QFileInfo(officialOnnxPath).absoluteFilePath());
        QCOMPARE(QFileInfo(report.value(QStringLiteral("exportPath")).toString()).absoluteFilePath(), QFileInfo(finalExportPath).absoluteFilePath());

        bool sawOfficialExportStep = false;
        bool sawExportStep = false;
        bool sawBenchmarkStep = false;
        for (const QJsonValue& value : report.value(QStringLiteral("steps")).toArray()) {
            const QJsonObject step = value.toObject();
            const QString command = step.value(QStringLiteral("command")).toString();
            sawOfficialExportStep = sawOfficialExportStep || command == QStringLiteral("officialYoloExport");
            sawExportStep = sawExportStep || command == QStringLiteral("exportModel");
            sawBenchmarkStep = sawBenchmarkStep || command == QStringLiteral("benchmarkModel");
        }
        QVERIFY(sawOfficialExportStep);
        QVERIFY(sawExportStep);
        QVERIFY(sawBenchmarkStep);

        bool sawInternalModelExportEvent = false;
        for (const auto& message : messages) {
            sawInternalModelExportEvent = sawInternalModelExportEvent
                || message.first == QStringLiteral("modelExport")
                || (message.first == QStringLiteral("artifact")
                    && message.second.value(QStringLiteral("kind")).toString() == QStringLiteral("official_yolo_export"));
        }
        QVERIFY(sawInternalModelExportEvent);
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
        request.parameters.insert(QStringLiteral("modelPreset"), QStringLiteral("PP-OCRv4_mobile_rec"));
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
        request.parameters.insert(QStringLiteral("modelPreset"), QStringLiteral("PP-OCRv4_mobile_det"));
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

    void workerRunsPaddleOcrRecV5PresetPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR Rec adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString repoPath = writeFakePaddleOcrRepo(QDir(dir.path()).filePath(QStringLiteral("fake-paddleocr")));
        QVERIFY(!repoPath.isEmpty());
        const QString datasetPath = QDir(dir.path()).filePath(QStringLiteral("ocr-rec"));
        writeTinyOcrRecDataset(datasetPath);
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-rec-v5-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-rec-v5-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_recognition");
        request.datasetPath = datasetPath;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_rec_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("paddleOcrRepoPath"), repoPath);
        request.parameters.insert(QStringLiteral("modelPreset"), QStringLiteral("PP-OCRv5_mobile_rec"));
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);

        WorkerClient client;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(finished, qPrintable(finishedMessage), 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));

        const QString configPath = QDir(outputPath).filePath(QStringLiteral("aitrain_ppocrv5_mobile_rec.yml"));
        const QString reportPath = QDir(outputPath).filePath(QStringLiteral("paddleocr_official_rec_report.json"));
        QVERIFY(QFileInfo::exists(configPath));
        const QJsonObject report = readJsonObject(reportPath);
        QCOMPARE(report.value(QStringLiteral("ocrVersion")).toString(), QStringLiteral("PP-OCRv5"));
        QCOMPARE(report.value(QStringLiteral("modelPreset")).toString(), QStringLiteral("PP-OCRv5_mobile_rec"));
        QCOMPARE(report.value(QStringLiteral("configSource")).toString(), QStringLiteral("builtin_preset"));
        QCOMPARE(report.value(QStringLiteral("recAlgorithm")).toString(), QStringLiteral("SVTR_LCNet"));
        QVERIFY(report.value(QStringLiteral("presetDictionaryPath")).toString().contains(QStringLiteral("ppocrv5_dict.txt")));
    }

    void workerRunsPaddleOcrDetV5PresetPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR Det adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString repoPath = writeFakePaddleOcrRepo(QDir(dir.path()).filePath(QStringLiteral("fake-paddleocr")));
        QVERIFY(!repoPath.isEmpty());
        const QString datasetPath = QDir(dir.path()).filePath(QStringLiteral("ocr-det"));
        writeTinyOcrDetDataset(datasetPath);
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-det-v5-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-det-v5-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_detection");
        request.datasetPath = datasetPath;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_det_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("paddleOcrRepoPath"), repoPath);
        request.parameters.insert(QStringLiteral("modelPreset"), QStringLiteral("PP-OCRv5_server_det"));
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageSize"), 64);

        WorkerClient client;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(finished, qPrintable(finishedMessage), 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));

        const QString configPath = QDir(outputPath).filePath(QStringLiteral("aitrain_ppocrv5_server_det.yml"));
        const QString reportPath = QDir(outputPath).filePath(QStringLiteral("paddleocr_official_det_report.json"));
        QVERIFY(QFileInfo::exists(configPath));
        const QJsonObject report = readJsonObject(reportPath);
        QCOMPARE(report.value(QStringLiteral("ocrVersion")).toString(), QStringLiteral("PP-OCRv5"));
        QCOMPARE(report.value(QStringLiteral("modelPreset")).toString(), QStringLiteral("PP-OCRv5_server_det"));
        QCOMPARE(report.value(QStringLiteral("resolvedModelName")).toString(), QStringLiteral("PP-OCRv5_server_det"));
        QCOMPARE(report.value(QStringLiteral("configSource")).toString(), QStringLiteral("builtin_preset"));
    }

    void paddleOcrSystemV5ServerRecUsesHgNetAlgorithm()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR System adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QDir root(dir.path());
        const QString recModelDir = root.filePath(QStringLiteral("rec-model"));
        const QString detModelDir = root.filePath(QStringLiteral("det-model"));
        QDir().mkpath(recModelDir);
        QDir().mkpath(detModelDir);
        writeTextFile(QDir(recModelDir).filePath(QStringLiteral("inference.yml")),
            QStringLiteral("Global:\n  model_name: PP-OCRv5_server_rec\nArchitecture:\n  algorithm: SVTR_HGNet\n"));

        const QString requestPath = root.filePath(QStringLiteral("system-request.json"));
        const QString outputPath = root.filePath(QStringLiteral("system-output"));
        QJsonObject parameters;
        parameters.insert(QStringLiteral("prepareOnly"), true);
        parameters.insert(QStringLiteral("detModelDir"), detModelDir);
        parameters.insert(QStringLiteral("recModelDir"), recModelDir);
        parameters.insert(QStringLiteral("dictionaryFile"), root.filePath(QStringLiteral("dict.txt")));
        parameters.insert(QStringLiteral("inferenceImage"), root.filePath(QStringLiteral("sample.png")));
        parameters.insert(QStringLiteral("recModelPreset"), QStringLiteral("PP-OCRv5_server_rec"));
        QJsonObject request;
        request.insert(QStringLiteral("protocolVersion"), 1);
        request.insert(QStringLiteral("taskId"), QStringLiteral("paddleocr-system-v5-command"));
        request.insert(QStringLiteral("taskType"), QStringLiteral("ocr"));
        request.insert(QStringLiteral("datasetPath"), root.filePath(QStringLiteral("sample.png")));
        request.insert(QStringLiteral("outputPath"), outputPath);
        request.insert(QStringLiteral("backend"), QStringLiteral("paddleocr_system_official"));
        request.insert(QStringLiteral("parameters"), parameters);
        writeTextFile(requestPath, QString::fromUtf8(QJsonDocument(request).toJson(QJsonDocument::Indented)));

        QProcess process;
        const QString adapterPath = repoRelativeFilePath(QStringLiteral("python_trainers/ocr_system/paddleocr_system_official_adapter.py"));
        process.setWorkingDirectory(QFileInfo(adapterPath).absolutePath() + QStringLiteral("/../.."));
        process.start(python, QStringList()
                << adapterPath
                << QStringLiteral("--request")
                << requestPath);
        QVERIFY2(process.waitForFinished(15000), qPrintable(QString::fromUtf8(process.readAllStandardError())));
        QCOMPARE(process.exitCode(), 0);

        const QJsonObject report = readJsonObject(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_system_report.json")));
        QCOMPARE(report.value(QStringLiteral("recAlgorithm")).toString(), QStringLiteral("SVTR_HGNet"));
        const QJsonArray command = report.value(QStringLiteral("predictCommand")).toArray();
        bool sawHgNet = false;
        for (const QJsonValue& value : command) {
            sawHgNet = sawHgNet || value.toString() == QStringLiteral("--rec_algorithm=SVTR_HGNet");
        }
        QVERIFY(sawHgNet);
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
