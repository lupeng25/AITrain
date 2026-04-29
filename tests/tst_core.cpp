#include "WorkerClient.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionDataset.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/ProjectRepository.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>

namespace {

void writeTextFile(const QString& path, const QString& content)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
    file.write(content.toUtf8());
}

void writeTinyPng(const QString& path)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QImage image(8, 8, QImage::Format_RGB888);
    image.fill(Qt::white);
    QVERIFY(image.save(path));
}

void writeTinyDetectionDataset(const QString& root)
{
    writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
    writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
}

QString workerExecutablePath()
{
    const QString extension =
#ifdef Q_OS_WIN
        QStringLiteral(".exe");
#else
        QString();
#endif
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QString siblingBin = QDir(applicationDir).absoluteFilePath(QStringLiteral("../bin/aitrain_worker%1").arg(extension));
    if (QFileInfo::exists(siblingBin)) {
        return QDir::cleanPath(siblingBin);
    }
    return QDir(applicationDir).absoluteFilePath(QStringLiteral("aitrain_worker%1").arg(extension));
}

} // namespace

class CoreTests : public QObject {
    Q_OBJECT

private slots:
    void taskStateTransitions()
    {
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Running));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Failed));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Paused));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Paused, aitrain::TaskState::Running));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Paused, aitrain::TaskState::Canceled));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Completed));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Completed, aitrain::TaskState::Running));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Failed, aitrain::TaskState::Running));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Canceled, aitrain::TaskState::Running));
    }

    void protocolRoundTrip()
    {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), QStringLiteral("abc"));
        payload.insert(QStringLiteral("value"), 42);

        const QByteArray encoded = aitrain::protocol::encodeMessage(QStringLiteral("metric"), payload, QStringLiteral("req-1"));
        QVERIFY(encoded.endsWith('\n'));

        QString type;
        QJsonObject decodedPayload;
        QString requestId;
        QString error;
        QVERIFY(aitrain::protocol::decodeMessage(encoded, &type, &decodedPayload, &requestId, &error));
        QCOMPARE(type, QStringLiteral("metric"));
        QCOMPARE(requestId, QStringLiteral("req-1"));
        QCOMPARE(decodedPayload.value(QStringLiteral("taskId")).toString(), QStringLiteral("abc"));
        QCOMPARE(decodedPayload.value(QStringLiteral("value")).toInt(), 42);
        QVERIFY(error.isEmpty());
    }

    void repositoryStoresTasksAndMetrics()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        aitrain::ProjectRepository repository;
        QString error;
        QVERIFY2(repository.open(dir.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
        QVERIFY2(repository.upsertProject(QStringLiteral("demo"), dir.path(), &error), qPrintable(error));

        aitrain::TaskRecord task;
        task.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        task.projectName = QStringLiteral("demo");
        task.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        task.taskType = QStringLiteral("detection");
        task.kind = aitrain::TaskKind::Train;
        task.state = aitrain::TaskState::Queued;
        task.workDir = dir.filePath(QStringLiteral("runs/1"));
        task.createdAt = QDateTime::currentDateTimeUtc();
        task.updatedAt = task.createdAt;
        QVERIFY2(repository.insertTask(task, &error), qPrintable(error));
        QVERIFY2(repository.updateTaskState(task.id, aitrain::TaskState::Running, QStringLiteral("started"), &error), qPrintable(error));
        QVERIFY2(!repository.updateTaskState(task.id, aitrain::TaskState::Queued, QStringLiteral("invalid"), &error), qPrintable(error));

        aitrain::MetricPoint metric;
        metric.taskId = task.id;
        metric.name = QStringLiteral("loss");
        metric.value = 0.75;
        metric.step = 1;
        metric.epoch = 1;
        metric.createdAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.insertMetric(metric, &error), qPrintable(error));

        aitrain::ArtifactRecord artifact;
        artifact.taskId = task.id;
        artifact.kind = QStringLiteral("checkpoint");
        artifact.path = dir.filePath(QStringLiteral("runs/1/checkpoint_best.aitrain"));
        artifact.message = QStringLiteral("scaffold checkpoint");
        QVERIFY2(repository.insertArtifact(artifact, &error), qPrintable(error));

        QJsonObject exportConfig;
        exportConfig.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        exportConfig.insert(QStringLiteral("sourceCheckpoint"), artifact.path);
        exportConfig.insert(QStringLiteral("input"), QJsonObject{
            {QStringLiteral("name"), QStringLiteral("features")},
            {QStringLiteral("shape"), QJsonArray{16, 7}}
        });
        exportConfig.insert(QStringLiteral("outputs"), QJsonArray{
            QJsonObject{{QStringLiteral("name"), QStringLiteral("objectness")}, {QStringLiteral("shape"), QJsonArray{16, 1}}},
            QJsonObject{{QStringLiteral("name"), QStringLiteral("class_probabilities")}, {QStringLiteral("shape"), QJsonArray{16, 1}}},
            QJsonObject{{QStringLiteral("name"), QStringLiteral("boxes")}, {QStringLiteral("shape"), QJsonArray{16, 4}}}
        });
        aitrain::ExportRecord exportRecord;
        exportRecord.taskId = task.id;
        exportRecord.sourceCheckpointPath = artifact.path;
        exportRecord.format = QStringLiteral("onnx");
        exportRecord.path = dir.filePath(QStringLiteral("runs/1/model.onnx"));
        exportRecord.configJson = QString::fromUtf8(QJsonDocument(exportConfig).toJson(QJsonDocument::Compact));
        exportRecord.inputShapeJson = QString::fromUtf8(QJsonDocument(exportConfig.value(QStringLiteral("input")).toObject()).toJson(QJsonDocument::Compact));
        exportRecord.outputShapeJson = QString::fromUtf8(QJsonDocument(QJsonObject{{QStringLiteral("outputs"), exportConfig.value(QStringLiteral("outputs")).toArray()}}).toJson(QJsonDocument::Compact));
        exportRecord.createdAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.insertExport(exportRecord, &error), qPrintable(error));

        aitrain::EnvironmentCheckRecord check;
        check.name = QStringLiteral("Worker");
        check.status = QStringLiteral("ok");
        check.message = QStringLiteral("Worker executable found");
        QVERIFY2(repository.insertEnvironmentCheck(check, &error), qPrintable(error));

        const QVector<aitrain::TaskRecord> tasks = repository.recentTasks(10, &error);
        QCOMPARE(tasks.size(), 1);
        QCOMPARE(tasks.first().id, task.id);
        QCOMPARE(tasks.first().state, aitrain::TaskState::Running);
        QVERIFY(tasks.first().startedAt.isValid());
        QVERIFY(!tasks.first().finishedAt.isValid());

        const QVector<aitrain::EnvironmentCheckRecord> checks = repository.recentEnvironmentChecks(10, &error);
        QCOMPARE(checks.size(), 1);
        QCOMPARE(checks.first().name, QStringLiteral("Worker"));
        QCOMPARE(checks.first().status, QStringLiteral("ok"));

        const QVector<aitrain::ExportRecord> exports = repository.recentExports(10, &error);
        QCOMPARE(exports.size(), 1);
        QCOMPARE(exports.first().taskId, task.id);
        QCOMPARE(exports.first().format, QStringLiteral("onnx"));
        QVERIFY(exports.first().configJson.contains(QStringLiteral("sourceCheckpoint")));
        QVERIFY(exports.first().inputShapeJson.contains(QStringLiteral("features")));
        QVERIFY(exports.first().outputShapeJson.contains(QStringLiteral("boxes")));

        aitrain::DatasetRecord dataset;
        dataset.name = QStringLiteral("demo-dataset");
        dataset.format = QStringLiteral("yolo_detection");
        dataset.rootPath = dir.filePath(QStringLiteral("datasets/demo"));
        dataset.validationStatus = QStringLiteral("valid");
        dataset.sampleCount = 2;
        dataset.lastReportJson = QStringLiteral("{\"ok\":true}");
        dataset.lastValidatedAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.upsertDatasetValidation(dataset, &error), qPrintable(error));
        const QVector<aitrain::DatasetRecord> datasets = repository.recentDatasets(10, &error);
        QCOMPARE(datasets.size(), 1);
        QCOMPARE(datasets.first().validationStatus, QStringLiteral("valid"));
        QCOMPARE(datasets.first().sampleCount, 2);
        const aitrain::DatasetRecord loadedDataset = repository.datasetByRootPath(dataset.rootPath, &error);
        QCOMPARE(loadedDataset.format, QStringLiteral("yolo_detection"));

        QVERIFY2(repository.updateTaskState(task.id, aitrain::TaskState::Completed, QStringLiteral("done"), &error), qPrintable(error));
        const QVector<aitrain::TaskRecord> completedTasks = repository.recentTasks(10, &error);
        QCOMPARE(completedTasks.first().state, aitrain::TaskState::Completed);
        QVERIFY(completedTasks.first().finishedAt.isValid());
        QVERIFY(!repository.updateTaskState(task.id, aitrain::TaskState::Running, QStringLiteral("invalid"), &error));
    }

    void repositoryMarksInterruptedTasksFailed()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        aitrain::ProjectRepository repository;
        QString error;
        QVERIFY2(repository.open(dir.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

        aitrain::TaskRecord task;
        task.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        task.projectName = QStringLiteral("demo");
        task.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        task.taskType = QStringLiteral("detection");
        task.kind = aitrain::TaskKind::Train;
        task.state = aitrain::TaskState::Running;
        task.workDir = dir.filePath(QStringLiteral("runs/1"));
        task.createdAt = QDateTime::currentDateTimeUtc();
        task.updatedAt = task.createdAt;
        task.startedAt = task.createdAt;
        QVERIFY2(repository.insertTask(task, &error), qPrintable(error));

        QVERIFY2(repository.markInterruptedTasksFailed(QStringLiteral("Worker interrupted"), &error), qPrintable(error));
        const QVector<aitrain::TaskRecord> tasks = repository.recentTasks(10, &error);
        QCOMPARE(tasks.size(), 1);
        QCOMPARE(tasks.first().state, aitrain::TaskState::Failed);
        QCOMPARE(tasks.first().message, QStringLiteral("Worker interrupted"));
        QVERIFY(tasks.first().finishedAt.isValid());
    }

    void yoloDetectionDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("1 0.5 0.5 0.20 0.20\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);
        QVERIFY(!valid.previewSamples.isEmpty());
        QVERIFY(valid.previewSamples.first().contains(QStringLiteral("bbox=")));

        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("3 0.5 0.5 0.25 0.25\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY(!invalid.ok);
        QVERIFY(!invalid.issues.isEmpty());
        QCOMPARE(invalid.issues.first().line, 1);
    }

    void yoloDetectionDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("source"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        for (int index = 0; index < 4; ++index) {
            const QString split = index < 3 ? QStringLiteral("train") : QStringLiteral("val");
            const QString name = QStringLiteral("sample_%1").arg(index);
            writeTinyPng(QDir(root).filePath(QStringLiteral("images/%1/%2.jpg").arg(split, name)));
            writeTextFile(QDir(root).filePath(QStringLiteral("labels/%1/%2.txt").arg(split, name)), QStringLiteral("0 0.5 0.5 0.2 0.2\n"));
        }

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.25);
        options.insert(QStringLiteral("testRatio"), 0.25);
        options.insert(QStringLiteral("seed"), 7);
        const QString output = dir.filePath(QStringLiteral("normalized"));
        const aitrain::DatasetSplitResult result = aitrain::splitYoloDetectionDataset(root, output, options);
        QVERIFY2(result.ok, qPrintable(result.errors.join(QStringLiteral("\n"))));
        QCOMPARE(result.trainCount, 2);
        QCOMPARE(result.valCount, 1);
        QCOMPARE(result.testCount, 1);
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("data.yaml"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("split_report.json"))));
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/train"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 2);
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/val"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 1);
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/test"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 1);
        QVERIFY(QFileInfo::exists(QDir(root).filePath(QStringLiteral("images/train/sample_0.jpg"))));
    }

    void detectionDatasetLoadsSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/b.txt")), QStringLiteral("1 0.4 0.4 0.20 0.30\n"));

        QString error;
        const aitrain::DetectionDatasetInfo info = aitrain::readDetectionDatasetInfo(root, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(info.classCount, 2);
        QCOMPARE(info.classNames.size(), 2);
        QCOMPARE(info.classNames.at(1), QStringLiteral("dog"));

        aitrain::DetectionDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));
        QCOMPARE(dataset.size(), 2);
        QCOMPARE(dataset.info().classCount, 2);
        QCOMPARE(dataset.samples().first().boxes.size(), 1);
        QCOMPARE(dataset.samples().first().boxes.first().classId, 0);
        QCOMPARE(dataset.samples().at(1).boxes.first().classId, 1);
    }

    void detectionDatasetRejectsInvalidLabel()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("2 0.5 0.5 0.25 0.25\n"));

        QString error;
        aitrain::DetectionDataset dataset;
        QVERIFY(!dataset.load(root, QStringLiteral("train"), &error));
        QVERIFY(error.contains(QStringLiteral("class id")));
    }

    void detectionDataLoaderBuildsLetterboxBatch()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));

        QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train")));
        QImage wideImage(16, 8, QImage::Format_RGB888);
        wideImage.fill(Qt::white);
        QVERIFY(wideImage.save(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.5 0.5\n"));

        QString error;
        aitrain::DetectionDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));

        aitrain::DetectionDataLoader loader(dataset, 1, QSize(32, 32));
        QVERIFY(loader.hasNext());
        aitrain::DetectionBatch batch;
        QVERIFY2(loader.next(&batch, &error), qPrintable(error));
        QCOMPARE(batch.images.size(), 1);
        QCOMPARE(batch.images.first().size(), QSize(32, 32));
        QCOMPARE(batch.boxes.first().size(), 1);
        QCOMPARE(batch.boxes.first().first().classId, 0);
        QCOMPARE(batch.boxes.first().first().xCenter, 0.5);
        QCOMPARE(batch.boxes.first().first().yCenter, 0.5);
        QCOMPARE(batch.boxes.first().first().width, 0.5);
        QCOMPARE(batch.boxes.first().first().height, 0.25);
        QVERIFY(!loader.hasNext());
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
        QCOMPARE(result.steps, 4);
        QCOMPARE(callbackCount, 4);
        QVERIFY(result.finalLoss >= 0.0);
        QVERIFY(QFileInfo::exists(result.checkpointPath));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("type")).toString(), QStringLiteral("tiny_linear_detector"));
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
        QVERIFY(!json.value(QStringLiteral("objectnessWeights")).toArray().isEmpty());
        QVERIFY(!json.value(QStringLiteral("classLogits")).toArray().isEmpty());
        QVERIFY(json.value(QStringLiteral("priorBox")).isObject());

        QString error;
        aitrain::DetectionBaselineCheckpoint loadedExport;
        QVERIFY2(aitrain::loadDetectionBaselineCheckpoint(exported.exportPath, &loadedExport, &error), qPrintable(error));
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(
            loadedExport,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
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

    void detectionExportRejectsTensorRtUntilBackendExists()
    {
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            QStringLiteral("missing.aitrain"),
            QStringLiteral("model.engine"),
            QStringLiteral("tensorrt"));
        QVERIFY(!exported.ok);
        QCOMPARE(exported.format, QStringLiteral("tensorrt"));
        QVERIFY(exported.error.contains(QStringLiteral("TensorRT")));
        QVERIFY(exported.error.contains(QStringLiteral("not available")));
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
        QVERIFY2(client.requestInference(
            workerExecutablePath(),
            exported.exportPath,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            outputPath,
            &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
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

    void yoloSegmentationDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [part]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.1 0.1 0.8 0.1 0.8 0.8\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("0 0.1 0.1 0.8 0.1 0.8 0.8\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QVERIFY(!valid.previewSamples.isEmpty());
        QVERIFY(valid.previewSamples.first().contains(QStringLiteral("polygon=")));

        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("0 0.1 0.1 0.2\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY(!invalid.ok);
    }

    void paddleOcrDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/a.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\nb\nc\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.jpg\tabc\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validatePaddleOcrRecDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 1);
        QVERIFY(!valid.previewSamples.isEmpty());

        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/missing.jpg\taz\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validatePaddleOcrRecDataset(root);
        QVERIFY(!invalid.ok);
        QVERIFY(invalid.errors.join(QStringLiteral("\n")).contains(QStringLiteral("字典")));
    }
};

QTEST_MAIN(CoreTests)
#include "tst_core.moc"
