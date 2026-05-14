#include "TestSupport.h"

class RepositoryWorkflowTests : public QObject {
    Q_OBJECT

private slots:
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
        const QVector<aitrain::ArtifactRecord> taskArtifacts = repository.artifactsForTask(task.id, &error);
        QCOMPARE(taskArtifacts.size(), 1);
        QCOMPARE(taskArtifacts.first().kind, QStringLiteral("checkpoint"));
        QCOMPARE(taskArtifacts.first().path, artifact.path);
        const QVector<aitrain::MetricPoint> taskMetrics = repository.metricsForTask(task.id, &error);
        QCOMPARE(taskMetrics.size(), 1);
        QCOMPARE(taskMetrics.first().name, QStringLiteral("loss"));
        QCOMPARE(taskMetrics.first().step, 1);
        const QVector<aitrain::ExportRecord> taskExports = repository.exportsForTask(task.id, &error);
        QCOMPARE(taskExports.size(), 1);
        QCOMPARE(taskExports.first().path, exportRecord.path);

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
        const QVector<aitrain::DatasetVersionRecord> versions = repository.datasetVersions(loadedDataset.id, &error);
        QCOMPARE(versions.size(), 1);
        QCOMPARE(versions.first().rootPath, dataset.rootPath);
        QVERIFY(versions.first().metadataJson.contains(QStringLiteral("\"ok\":true")));

        aitrain::ExperimentRecord experiment;
        experiment.name = QStringLiteral("det-baseline");
        experiment.taskType = QStringLiteral("detection");
        experiment.datasetId = loadedDataset.id;
        experiment.notes = QStringLiteral("local product workflow smoke");
        const int experimentId = repository.upsertExperiment(experiment, &error);
        QVERIFY2(experimentId > 0, qPrintable(error));

        aitrain::DatasetSnapshotRecord snapshot;
        snapshot.datasetId = loadedDataset.id;
        snapshot.name = QStringLiteral("snapshot-a");
        snapshot.rootPath = dataset.rootPath;
        snapshot.manifestPath = QDir(dataset.rootPath).filePath(QStringLiteral("snapshot.json"));
        snapshot.contentHash = QStringLiteral("abc123");
        snapshot.fileCount = 4;
        snapshot.totalBytes = 128;
        const int snapshotId = repository.insertDatasetSnapshot(snapshot, &error);
        QVERIFY2(snapshotId > 0, qPrintable(error));

        aitrain::ExperimentRunRecord run;
        run.experimentId = experimentId;
        run.taskId = task.id;
        run.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
        run.modelPreset = QStringLiteral("yolov8n.yaml");
        run.datasetSnapshotId = snapshotId;
        run.requestJson = QStringLiteral("{}");
        const int runId = repository.insertExperimentRun(run, &error);
        QVERIFY2(runId > 0, qPrintable(error));
        QCOMPARE(repository.insertExperimentRun(run, &error), runId);
        QCOMPARE(repository.experimentRunForTask(task.id, &error).id, runId);
        QCOMPARE(repository.datasetSnapshotById(snapshotId, &error).contentHash, snapshot.contentHash);
        QCOMPARE(repository.latestDatasetSnapshot(loadedDataset.id, &error).id, snapshotId);
        QVERIFY2(repository.updateExperimentRunSummary(task.id, QStringLiteral("{\"mAP50\":0.5}"), QStringLiteral("[{\"kind\":\"checkpoint\"}]"), &error), qPrintable(error));
        QCOMPARE(repository.experimentRunForTask(task.id, &error).bestMetricsJson, QStringLiteral("{\"mAP50\":0.5}"));

        aitrain::EvaluationReportRecord report;
        report.taskId = task.id;
        report.modelPath = exportRecord.path;
        report.taskType = QStringLiteral("detection");
        report.datasetSnapshotId = snapshotId;
        report.reportPath = QDir(dataset.rootPath).filePath(QStringLiteral("evaluation_report.json"));
        report.summaryJson = QStringLiteral("{}");
        const int reportId = repository.insertEvaluationReport(report, &error);
        QVERIFY2(reportId > 0, qPrintable(error));

        aitrain::ModelVersionRecord model;
        model.modelName = QStringLiteral("detector");
        model.version = QStringLiteral("v1");
        model.sourceTaskId = task.id;
        model.experimentRunId = runId;
        model.datasetSnapshotId = snapshotId;
        model.checkpointPath = exportRecord.sourceCheckpointPath;
        model.onnxPath = exportRecord.path;
        model.evaluationReportId = reportId;
        model.status = QStringLiteral("draft");
        const int modelId = repository.upsertModelVersion(model, &error);
        QVERIFY2(modelId > 0, qPrintable(error));

        aitrain::PipelineRunRecord pipeline;
        pipeline.name = QStringLiteral("local-loop");
        pipeline.templateId = QStringLiteral("train-evaluate-export-register");
        pipeline.taskIdsJson = QStringLiteral("[\"%1\"]").arg(task.id);
        pipeline.state = QStringLiteral("completed");
        const int pipelineId = repository.insertPipelineRun(pipeline, &error);
        QVERIFY2(pipelineId > 0, qPrintable(error));

        QCOMPARE(repository.recentExperiments(10, &error).first().name, experiment.name);
        QCOMPARE(repository.experimentRuns(experimentId, &error).first().trainingBackend, run.trainingBackend);
        QCOMPARE(repository.datasetSnapshots(loadedDataset.id, &error).first().contentHash, snapshot.contentHash);
        QCOMPARE(repository.recentEvaluationReports(10, &error).first().reportPath, report.reportPath);
        QCOMPARE(repository.recentModelVersions(10, &error).first().modelName, model.modelName);
        QCOMPARE(repository.recentPipelineRuns(10, &error).first().templateId, pipeline.templateId);

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

    void productWorkflowWritesReports()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);
        const QString outputRoot = dir.filePath(QStringLiteral("reports"));

        const aitrain::WorkflowResult snapshot = aitrain::createDatasetSnapshotReport(
            datasetRoot,
            QDir(outputRoot).filePath(QStringLiteral("snapshot")),
            QStringLiteral("yolo_detection"));
        QVERIFY2(snapshot.ok, qPrintable(snapshot.error));
        QVERIFY(QFileInfo::exists(snapshot.reportPath));
        QVERIFY(snapshot.payload.value(QStringLiteral("contentHash")).toString().size() >= 32);
        QCOMPARE(snapshot.payload.value(QStringLiteral("fileCount")).toInt(), 5);
        const QString firstHash = snapshot.payload.value(QStringLiteral("contentHash")).toString();
        const QJsonArray files = snapshot.payload.value(QStringLiteral("files")).toArray();
        QCOMPARE(files.size(), 5);
        QString lastPath;
        bool sawImage = false;
        bool sawLabel = false;
        bool sawConfig = false;
        for (const QJsonValue& value : files) {
            const QJsonObject file = value.toObject();
            const QString path = file.value(QStringLiteral("path")).toString();
            QVERIFY(lastPath.isEmpty() || lastPath.compare(path, Qt::CaseInsensitive) <= 0);
            lastPath = path;
            sawImage = sawImage || file.value(QStringLiteral("role")).toString() == QStringLiteral("image");
            sawLabel = sawLabel || file.value(QStringLiteral("role")).toString() == QStringLiteral("label");
            sawConfig = sawConfig || file.value(QStringLiteral("role")).toString() == QStringLiteral("config");
        }
        QVERIFY(sawImage);
        QVERIFY(sawLabel);
        QVERIFY(sawConfig);
        QVERIFY(snapshot.payload.value(QStringLiteral("keyFiles")).toArray().size() >= 1);
        QVERIFY(snapshot.payload.value(QStringLiteral("roleCounts")).toObject().value(QStringLiteral("image")).toInt() >= 1);

        const aitrain::WorkflowResult sameSnapshot = aitrain::createDatasetSnapshotReport(
            datasetRoot,
            QDir(outputRoot).filePath(QStringLiteral("snapshot_same")),
            QStringLiteral("yolo_detection"));
        QVERIFY2(sameSnapshot.ok, qPrintable(sameSnapshot.error));
        QCOMPARE(sameSnapshot.payload.value(QStringLiteral("contentHash")).toString(), firstHash);
        writeTextFile(QDir(datasetRoot).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.30 0.30\n"));
        const aitrain::WorkflowResult changedSnapshot = aitrain::createDatasetSnapshotReport(
            datasetRoot,
            QDir(outputRoot).filePath(QStringLiteral("snapshot_changed")),
            QStringLiteral("yolo_detection"));
        QVERIFY2(changedSnapshot.ok, qPrintable(changedSnapshot.error));
        QVERIFY(changedSnapshot.payload.value(QStringLiteral("contentHash")).toString() != firstHash);

        const aitrain::WorkflowResult quality = aitrain::curateDatasetQualityReport(
            datasetRoot,
            QDir(outputRoot).filePath(QStringLiteral("quality")),
            QStringLiteral("yolo_detection"));
        QVERIFY2(quality.ok, qPrintable(quality.error));
        QVERIFY(QFileInfo::exists(quality.reportPath));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("classDistributionPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("problemSamplesPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("reworkSampleSetPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("prelabelCandidatesPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("trainingReadinessPath")).toString()));
        QVERIFY(quality.payload.value(QStringLiteral("ok")).toBool());
        const QJsonObject ready = quality.payload.value(QStringLiteral("trainingReadiness")).toObject();
        QVERIFY(ready.value(QStringLiteral("canTrain")).toBool());
        QCOMPARE(ready.value(QStringLiteral("status")).toString(), QStringLiteral("ready"));

        aitrain::DetectionTrainingOptions evaluationTrainingOptions;
        evaluationTrainingOptions.epochs = 1;
        evaluationTrainingOptions.batchSize = 1;
        evaluationTrainingOptions.imageSize = QSize(32, 32);
        evaluationTrainingOptions.outputPath = QDir(outputRoot).filePath(QStringLiteral("evaluation-model"));
        const aitrain::DetectionTrainingResult evaluationTraining = aitrain::trainDetectionBaseline(datasetRoot, evaluationTrainingOptions);
        QVERIFY2(evaluationTraining.ok, qPrintable(evaluationTraining.error));
        const QString modelPath = evaluationTraining.checkpointPath;
        const aitrain::WorkflowResult evaluation = aitrain::evaluateModelReport(
            modelPath,
            datasetRoot,
            QDir(outputRoot).filePath(QStringLiteral("evaluation")),
            QStringLiteral("detection"));
        QVERIFY2(evaluation.ok, qPrintable(evaluation.error));
        QVERIFY(QFileInfo::exists(evaluation.reportPath));
        QVERIFY(!evaluation.payload.value(QStringLiteral("scaffold")).toBool());
        QVERIFY(QFileInfo::exists(evaluation.payload.value(QStringLiteral("perClassMetricsPath")).toString()));
        QVERIFY(QFileInfo::exists(evaluation.payload.value(QStringLiteral("errorSamplesPath")).toString()));
        QVERIFY(QFileInfo::exists(evaluation.payload.value(QStringLiteral("confusionMatrixPath")).toString()));
        QVERIFY(QFileInfo::exists(evaluation.payload.value(QStringLiteral("evaluationSummaryPath")).toString()));
        QVERIFY(QFileInfo(evaluation.payload.value(QStringLiteral("overlayDir")).toString()).exists());
        QVERIFY(evaluation.payload.value(QStringLiteral("metrics")).toObject().contains(QStringLiteral("mAP50")));
        QVERIFY(evaluation.payload.value(QStringLiteral("metrics")).toObject().contains(QStringLiteral("mAP50_95")));
        QVERIFY(!evaluation.payload.value(QStringLiteral("cocoMapThresholds")).toArray().isEmpty());
        QVERIFY(evaluation.payload.value(QStringLiteral("decisionSummary")).toObject().contains(QStringLiteral("status")));
        QVERIFY(evaluation.payload.value(QStringLiteral("errorTaxonomy")).toObject().contains(QStringLiteral("reasonCounts")));

        const aitrain::WorkflowResult segmentationEvaluation = aitrain::evaluateModelReport(
            modelPath,
            datasetRoot,
            QDir(outputRoot).filePath(QStringLiteral("segmentation-evaluation")),
            QStringLiteral("segmentation"));
        QVERIFY(!segmentationEvaluation.ok);

        const aitrain::WorkflowResult benchmark = aitrain::benchmarkModelReport(
            modelPath,
            QDir(outputRoot).filePath(QStringLiteral("benchmark")),
            QJsonObject{
                {QStringLiteral("datasetPath"), datasetRoot},
                {QStringLiteral("warmupIterations"), 0},
                {QStringLiteral("iterations"), 2}
            });
        QVERIFY2(benchmark.ok, qPrintable(benchmark.error));
        QVERIFY(QFileInfo::exists(benchmark.reportPath));
        QVERIFY(benchmark.payload.contains(QStringLiteral("averageMs")));
        QVERIFY(benchmark.payload.value(QStringLiteral("timedInference")).toBool());
        QVERIFY(benchmark.payload.value(QStringLiteral("runtimeUsable")).toBool());
        QVERIFY(!benchmark.payload.value(QStringLiteral("modelSha256")).toString().isEmpty());
        QVERIFY(benchmark.payload.value(QStringLiteral("modelDigest")).toObject().contains(QStringLiteral("sha256")));
        QVERIFY(benchmark.payload.value(QStringLiteral("sampleDigest")).toObject().contains(QStringLiteral("sha256")));
        QVERIFY(benchmark.payload.value(QStringLiteral("latency")).toObject().contains(QStringLiteral("p95Ms")));
        QCOMPARE(benchmark.payload.value(QStringLiteral("deploymentConclusion")).toString(), QStringLiteral("local-runtime-available"));

        const aitrain::WorkflowResult deploymentValidation = aitrain::validateDeploymentArtifactReport(
            modelPath,
            QDir(outputRoot).filePath(QStringLiteral("deployment-validation")),
            QStringLiteral("tiny_detector_json"));
        QVERIFY2(deploymentValidation.ok, qPrintable(deploymentValidation.error));
        QVERIFY(QFileInfo::exists(deploymentValidation.reportPath));
        QCOMPARE(deploymentValidation.payload.value(QStringLiteral("status")).toString(), QStringLiteral("passed"));

        const QString ocrRoot = dir.filePath(QStringLiteral("customer-ocr"));
        const QString detDataset = QDir(ocrRoot).filePath(QStringLiteral("det"));
        const QString recDataset = QDir(ocrRoot).filePath(QStringLiteral("rec"));
        const QString systemImages = QDir(ocrRoot).filePath(QStringLiteral("system"));
        writeTinyPng(QDir(detDataset).filePath(QStringLiteral("images/a.png")));
        writeTinyPng(QDir(recDataset).filePath(QStringLiteral("images/a.png")));
        writeTinyPng(QDir(systemImages).filePath(QStringLiteral("a.png")));
        const QString detReportPath = QDir(ocrRoot).filePath(QStringLiteral("det_report.json"));
        const QString recReportPath = QDir(ocrRoot).filePath(QStringLiteral("rec_report.json"));
        const QString systemReportPath = QDir(ocrRoot).filePath(QStringLiteral("system_report.json"));
        writeTextFile(detReportPath, QStringLiteral("{\"status\":\"passed\"}"));
        writeTextFile(recReportPath, QStringLiteral("{\"metrics\":{\"accuracy\":0.82,\"cer\":0.12,\"samples\":5}}"));
        writeTextFile(systemReportPath, QStringLiteral("{\"status\":\"passed\"}"));
        const aitrain::WorkflowResult ocrAcceptance = aitrain::runCustomerOcrAcceptanceReport(
            QDir(outputRoot).filePath(QStringLiteral("ocr-acceptance")),
            QJsonObject{
                {QStringLiteral("detDatasetPath"), detDataset},
                {QStringLiteral("recDatasetPath"), recDataset},
                {QStringLiteral("systemImagesPath"), systemImages},
                {QStringLiteral("detReportPath"), detReportPath},
                {QStringLiteral("recReportPath"), recReportPath},
                {QStringLiteral("systemReportPath"), systemReportPath}
            });
        QVERIFY2(ocrAcceptance.ok, qPrintable(ocrAcceptance.error));
        QCOMPARE(ocrAcceptance.payload.value(QStringLiteral("status")).toString(), QStringLiteral("passed"));

        const QString publicOcrRoot = dir.filePath(QStringLiteral("generated-ocr-smoke"));
        writeTinyPng(QDir(publicOcrRoot).filePath(QStringLiteral("det/images/a.png")));
        writeTinyPng(QDir(publicOcrRoot).filePath(QStringLiteral("rec/images/a.png")));
        writeTinyPng(QDir(publicOcrRoot).filePath(QStringLiteral("system/a.png")));
        const aitrain::WorkflowResult blockedOcrAcceptance = aitrain::runCustomerOcrAcceptanceReport(
            QDir(outputRoot).filePath(QStringLiteral("ocr-acceptance-blocked")),
            QJsonObject{
                {QStringLiteral("detDatasetPath"), QDir(publicOcrRoot).filePath(QStringLiteral("det"))},
                {QStringLiteral("recDatasetPath"), QDir(publicOcrRoot).filePath(QStringLiteral("rec"))},
                {QStringLiteral("systemImagesPath"), QDir(publicOcrRoot).filePath(QStringLiteral("system"))},
                {QStringLiteral("detReportPath"), detReportPath},
                {QStringLiteral("recReportPath"), recReportPath},
                {QStringLiteral("systemReportPath"), systemReportPath}
            });
        QVERIFY2(blockedOcrAcceptance.ok, qPrintable(blockedOcrAcceptance.error));
        QCOMPARE(blockedOcrAcceptance.payload.value(QStringLiteral("status")).toString(), QStringLiteral("blocked"));

        const aitrain::WorkflowResult diagnostics = aitrain::collectDiagnosticsReport(
            QDir(outputRoot).filePath(QStringLiteral("diagnostics")),
            QJsonObject{
                {QStringLiteral("projectName"), QStringLiteral("demo")},
                {QStringLiteral("projectPath"), dir.path()},
                {QStringLiteral("workerExecutable"), QStringLiteral("aitrain_worker.exe")}
            });
        QVERIFY2(diagnostics.ok, qPrintable(diagnostics.error));
        QVERIFY(QFileInfo::exists(diagnostics.payload.value(QStringLiteral("bundlePath")).toString()));

        const aitrain::WorkflowResult delivery = aitrain::generateTrainingDeliveryReport(
            QDir(outputRoot).filePath(QStringLiteral("delivery")),
            QJsonObject{
                {QStringLiteral("projectName"), QStringLiteral("demo")},
                {QStringLiteral("modelPath"), modelPath},
                {QStringLiteral("taskType"), QStringLiteral("detection")},
                {QStringLiteral("trainingBackend"), QStringLiteral("tiny_linear_detector")},
                {QStringLiteral("evaluationReportPath"), evaluation.reportPath},
                {QStringLiteral("benchmarkReportPath"), benchmark.reportPath},
                {QStringLiteral("deploymentValidationReportPath"), deploymentValidation.reportPath}
            });
        QVERIFY2(delivery.ok, qPrintable(delivery.error));
        QVERIFY(QFileInfo::exists(delivery.reportPath));
        QVERIFY(delivery.reportPath.endsWith(QStringLiteral(".html")));
        QVERIFY(!delivery.payload.value(QStringLiteral("scaffold")).toBool());
        const QString modelCardPath = delivery.payload.value(QStringLiteral("modelCardPath")).toString();
        const QString inventoryPath = delivery.payload.value(QStringLiteral("artifactInventoryPath")).toString();
        const QString manifestPath = delivery.payload.value(QStringLiteral("deliveryManifestPath")).toString();
        QVERIFY(QFileInfo::exists(modelCardPath));
        QVERIFY(QFileInfo::exists(inventoryPath));
        QVERIFY(QFileInfo::exists(manifestPath));
        QCOMPARE(delivery.payload.value(QStringLiteral("packageStatus")).toString(), QStringLiteral("ready_for_handoff_review"));
        QVERIFY(delivery.payload.value(QStringLiteral("evaluationSummary")).toObject().contains(QStringLiteral("metrics")));
        QVERIFY(delivery.payload.value(QStringLiteral("benchmarkSummary")).toObject().contains(QStringLiteral("latency")));
        QVERIFY(!delivery.payload.value(QStringLiteral("limitations")).toArray().isEmpty());
        QFile modelCardFile(modelCardPath);
        QVERIFY(modelCardFile.open(QIODevice::ReadOnly));
        const QJsonObject modelCard = QJsonDocument::fromJson(modelCardFile.readAll()).object();
        QVERIFY(modelCard.value(QStringLiteral("evaluationSummary")).toObject().contains(QStringLiteral("metrics")));
        QVERIFY(modelCard.value(QStringLiteral("benchmarkSummary")).toObject().contains(QStringLiteral("latency")));
        QFile inventoryFile(inventoryPath);
        QVERIFY(inventoryFile.open(QIODevice::ReadOnly));
        const QJsonArray inventory = QJsonDocument::fromJson(inventoryFile.readAll()).object().value(QStringLiteral("items")).toArray();
        QVERIFY(!inventory.isEmpty());
        bool sawModelHash = false;
        for (const QJsonValue& value : inventory) {
            const QJsonObject item = value.toObject();
            sawModelHash = sawModelHash || (item.value(QStringLiteral("kind")).toString() == QStringLiteral("model")
                && !item.value(QStringLiteral("sha256")).toString().isEmpty());
        }
        QVERIFY(sawModelHash);
        QFile manifestFile(manifestPath);
        QVERIFY(manifestFile.open(QIODevice::ReadOnly));
        const QJsonObject manifest = QJsonDocument::fromJson(manifestFile.readAll()).object();
        QCOMPARE(manifest.value(QStringLiteral("packageStatus")).toString(), QStringLiteral("ready_for_handoff_review"));
        QVERIFY(manifest.value(QStringLiteral("checks")).toArray().size() >= 4);

        QJsonObject pipelineOptions;
        pipelineOptions.insert(QStringLiteral("datasetPath"), datasetRoot);
        pipelineOptions.insert(QStringLiteral("datasetFormat"), QStringLiteral("yolo_detection"));
        pipelineOptions.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
        pipelineOptions.insert(QStringLiteral("trainingBackend"), QStringLiteral("tiny_linear_detector"));
        pipelineOptions.insert(QStringLiteral("epochs"), 1);
        pipelineOptions.insert(QStringLiteral("batchSize"), 1);
        pipelineOptions.insert(QStringLiteral("imageSize"), 32);
        pipelineOptions.insert(QStringLiteral("exportFormat"), QStringLiteral("onnx"));
        pipelineOptions.insert(QStringLiteral("sampleImagePath"), QDir(datasetRoot).filePath(QStringLiteral("images/val/a.png")));
        const aitrain::WorkflowResult pipeline = aitrain::runLocalPipelinePlan(
            QDir(outputRoot).filePath(QStringLiteral("pipeline")),
            QStringLiteral("train-evaluate-export-register"),
            pipelineOptions);
        QVERIFY2(pipeline.ok, qPrintable(pipeline.error));
        QVERIFY(QFileInfo::exists(pipeline.reportPath));
        QVERIFY(!pipeline.payload.value(QStringLiteral("scaffold")).toBool());
        QCOMPARE(pipeline.payload.value(QStringLiteral("state")).toString(), QStringLiteral("completed"));
        QVERIFY(!pipeline.payload.value(QStringLiteral("steps")).toArray().isEmpty());
        QVERIFY(!pipeline.payload.value(QStringLiteral("taskIds")).toArray().isEmpty());
    }

    void datasetQualityCatchesProblemSamples()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/good.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/no_label.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/bad_bbox.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/good.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/bad_bbox.txt")), QStringLiteral("2 1.5 0.5 0.2 0.2\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/orphan.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/missing_label.txt")), QStringLiteral(""));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/zero.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        QFile zeroImage(QDir(root).filePath(QStringLiteral("images/train/zero.png")));
        QVERIFY(zeroImage.open(QIODevice::WriteOnly | QIODevice::Truncate));
        zeroImage.close();
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/duplicate.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/duplicate_copy.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/bad.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/bad.txt")), QStringLiteral("9 0.5 0.5 0.2 0.2\n"));

        const QString outputRoot = dir.filePath(QStringLiteral("quality"));
        const aitrain::WorkflowResult quality = aitrain::curateDatasetQualityReport(
            root,
            outputRoot,
            QStringLiteral("yolo_detection"),
            QJsonObject{{QStringLiteral("maxFiles"), 100}, {QStringLiteral("maxIssues"), 1000}, {QStringLiteral("maxProblemSamples"), 1000}});
        QVERIFY2(quality.ok, qPrintable(quality.error));
        QVERIFY(!quality.payload.value(QStringLiteral("ok")).toBool());
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("reportPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("classDistributionPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("problemSamplesPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("imageStatisticsPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("splitDistributionPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("xAnyLabelingFixListPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("xAnyLabelingFixManifestPath")).toString()));
        QVERIFY(QFileInfo::exists(quality.payload.value(QStringLiteral("trainingReadinessPath")).toString()));

        const QJsonObject report = readJsonObject(quality.payload.value(QStringLiteral("reportPath")).toString());
        QCOMPARE(report.value(QStringLiteral("schemaVersion")).toInt(), 2);
        QCOMPARE(report.value(QStringLiteral("trainingReadiness")).toObject().value(QStringLiteral("status")).toString(), QStringLiteral("blocked"));
        QVERIFY(!report.value(QStringLiteral("trainingReadiness")).toObject().value(QStringLiteral("canTrain")).toBool());
        QVERIFY(report.value(QStringLiteral("severityCounts")).toObject().value(QStringLiteral("error")).toInt() > 0);
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("issues")).toArray(), QStringLiteral("bbox_out_of_bounds")));
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("issues")).toArray(), QStringLiteral("class_id_out_of_range")));
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("issues")).toArray(), QStringLiteral("duplicate_label_row")));
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("issues")).toArray(), QStringLiteral("orphan_label")));
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("issues")).toArray(), QStringLiteral("zero_byte_image")));
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("issues")).toArray(), QStringLiteral("duplicate_image_hash")));
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("problemSamples")).toArray(), QStringLiteral("missing_label")));
        QVERIFY(jsonArrayContainsCode(report.value(QStringLiteral("problemSamples")).toArray(), QStringLiteral("bbox_out_of_bounds")));

        const QJsonObject problems = readJsonObject(quality.payload.value(QStringLiteral("problemSamplesPath")).toString());
        QVERIFY(jsonArrayContainsCode(problems.value(QStringLiteral("samples")).toArray(), QStringLiteral("missing_label")));
        QVERIFY(jsonArrayContainsCode(problems.value(QStringLiteral("samples")).toArray(), QStringLiteral("duplicate_image_hash")));

        const QString segRoot = dir.filePath(QStringLiteral("segmentation"));
        writeTinySegmentationDataset(segRoot);
        writeTextFile(QDir(segRoot).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.1 0.1 0.2 0.2\n"));
        const aitrain::WorkflowResult segQuality = aitrain::curateDatasetQualityReport(
            segRoot,
            dir.filePath(QStringLiteral("seg-quality")),
            QStringLiteral("yolo_segmentation"));
        QVERIFY2(segQuality.ok, qPrintable(segQuality.error));
        QVERIFY(jsonArrayContainsCode(segQuality.payload.value(QStringLiteral("problemSamples")).toArray(), QStringLiteral("polygon_points_too_few")));

        const QString recRoot = dir.filePath(QStringLiteral("ocr-rec"));
        writeTextFile(QDir(recRoot).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\n"));
        writeTinyPng(QDir(recRoot).filePath(QStringLiteral("images/a.png")));
        writeTextFile(QDir(recRoot).filePath(QStringLiteral("rec_gt.txt")),
            QStringLiteral("images/a.png\tab\nimages/a.png\ta\nimages/missing.png\ta\n"));
        const aitrain::WorkflowResult recQuality = aitrain::curateDatasetQualityReport(
            recRoot,
            dir.filePath(QStringLiteral("rec-quality")),
            QStringLiteral("paddleocr_rec"));
        QVERIFY2(recQuality.ok, qPrintable(recQuality.error));
        const QJsonArray recProblems = recQuality.payload.value(QStringLiteral("problemSamples")).toArray();
        QVERIFY(jsonArrayContainsCode(recProblems, QStringLiteral("char_not_in_dictionary")));
        QVERIFY(jsonArrayContainsCode(recProblems, QStringLiteral("duplicate_ocr_sample")));
        QVERIFY(jsonArrayContainsCode(recProblems, QStringLiteral("missing_image")));

        const QString detRoot = dir.filePath(QStringLiteral("ocr-det"));
        writeTinyPng(QDir(detRoot).filePath(QStringLiteral("images/a.png")));
        writeTinyPng(QDir(detRoot).filePath(QStringLiteral("images/b.png")));
        writeTinyPng(QDir(detRoot).filePath(QStringLiteral("images/c.png")));
        writeTextFile(QDir(detRoot).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\tbad-json\n"
                           "images/a.png\t[{\"points\":[[1,1],[2,2],[3,3],[4,4]]}]\n"
                           "images/b.png\t[{\"transcription\":\"x\",\"points\":[[1,1],[2,2]]}]\n"
                           "images/c.png\t[{\"transcription\":\"x\",\"points\":[[1,1],[100,1],[100,100],[1,100]]}]\n"));
        const aitrain::WorkflowResult detQuality = aitrain::curateDatasetQualityReport(
            detRoot,
            dir.filePath(QStringLiteral("det-quality")),
            QStringLiteral("paddleocr_det"));
        QVERIFY2(detQuality.ok, qPrintable(detQuality.error));
        const QJsonArray detProblems = detQuality.payload.value(QStringLiteral("problemSamples")).toArray();
        QVERIFY(jsonArrayContainsCode(detProblems, QStringLiteral("invalid_det_json")));
        QVERIFY(jsonArrayContainsCode(detProblems, QStringLiteral("missing_transcription")));
        QVERIFY(jsonArrayContainsCode(detProblems, QStringLiteral("det_points_too_few")));
        QVERIFY(jsonArrayContainsCode(detProblems, QStringLiteral("det_point_out_of_bounds")));
    }

    void evaluateModelReportRunsRealSegmentationAndOcrOnnxSmoke()
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
        QString segDataset = QDir(segRoot).filePath(QStringLiteral("dataset"));
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/aitrain-yolo-segment/weights/best.onnx"));
            segDataset = QDir(segRoot).filePath(QStringLiteral("yolo_segment"));
        }
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/acceptance-yolo-segment/weights/best.onnx"));
            segDataset = QDir(segRoot).filePath(QStringLiteral("generated/yolo_segment"));
        }
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/cpu-yolo-segment/weights/best.onnx"));
            segDataset = QDir(segRoot).filePath(QStringLiteral("yolo_segment"));
        }

        const QString ocrOnnx = QDir(ocrRoot).filePath(QStringLiteral("runs/paddleocr_rec/paddleocr_rec_ctc.onnx"));
        QString ocrDataset = QDir(ocrRoot).filePath(QStringLiteral("dataset"));
        if (!QFileInfo::exists(QDir(ocrDataset).filePath(QStringLiteral("rec_gt.txt")))) {
            ocrDataset = QDir(ocrRoot).filePath(QStringLiteral("paddleocr_rec"));
        }
        if (!QFileInfo::exists(segOnnx) || !QFileInfo::exists(ocrOnnx)
            || !QFileInfo::exists(QDir(segDataset).filePath(QStringLiteral("data.yaml")))
            || !QFileInfo::exists(QDir(ocrDataset).filePath(QStringLiteral("rec_gt.txt")))) {
            QSKIP("Segmentation/OCR ONNX smoke artifacts are not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const aitrain::WorkflowResult segEvaluation = aitrain::evaluateModelReport(
            segOnnx,
            segDataset,
            dir.filePath(QStringLiteral("segmentation-evaluation")),
            QStringLiteral("segmentation"));
        QVERIFY2(segEvaluation.ok, qPrintable(segEvaluation.error));
        QVERIFY(!segEvaluation.payload.value(QStringLiteral("scaffold")).toBool(true));
        QVERIFY(QFileInfo::exists(segEvaluation.reportPath));
        QVERIFY(QFileInfo::exists(segEvaluation.payload.value(QStringLiteral("perClassMetricsPath")).toString()));
        QVERIFY(QFileInfo::exists(segEvaluation.payload.value(QStringLiteral("errorSamplesPath")).toString()));
        QVERIFY(QFileInfo::exists(segEvaluation.payload.value(QStringLiteral("confusionMatrixPath")).toString()));
        QVERIFY(QFileInfo(segEvaluation.payload.value(QStringLiteral("overlayDir")).toString()).exists());
        const QJsonObject segMetrics = segEvaluation.payload.value(QStringLiteral("metrics")).toObject();
        QVERIFY(segMetrics.contains(QStringLiteral("maskIoU")));
        QVERIFY(segMetrics.contains(QStringLiteral("maskMap50")));
        QVERIFY(segMetrics.contains(QStringLiteral("maskMap50_95")));

        const aitrain::WorkflowResult ocrEvaluation = aitrain::evaluateModelReport(
            ocrOnnx,
            ocrDataset,
            dir.filePath(QStringLiteral("ocr-evaluation")),
            QStringLiteral("ocr_recognition"));
        QVERIFY2(ocrEvaluation.ok, qPrintable(ocrEvaluation.error));
        QVERIFY(!ocrEvaluation.payload.value(QStringLiteral("scaffold")).toBool(true));
        QVERIFY(QFileInfo::exists(ocrEvaluation.reportPath));
        QVERIFY(QFileInfo::exists(ocrEvaluation.payload.value(QStringLiteral("errorSamplesPath")).toString()));
        QVERIFY(QFileInfo(ocrEvaluation.payload.value(QStringLiteral("overlayDir")).toString()).exists());
        const QJsonObject ocrMetrics = ocrEvaluation.payload.value(QStringLiteral("metrics")).toObject();
        QVERIFY(ocrMetrics.contains(QStringLiteral("accuracy")));
        QVERIFY(ocrMetrics.contains(QStringLiteral("editDistance")));
        QVERIFY(ocrMetrics.contains(QStringLiteral("cer")));
        QVERIFY(ocrMetrics.contains(QStringLiteral("wer")));
    }

    void workerRunsDatasetConversion()
    {
        QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_worker_conversion_XXXXXX")));
        QVERIFY(temp.isValid());
        const QDir root(temp.path());
        const QDir source(root.filePath(QStringLiteral("source_yolo")));
        writeTinyPng(source.filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(source.filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(source.filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(source.filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(source.filePath(QStringLiteral("data.yaml")), QStringLiteral("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"));

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
        QVERIFY2(client.requestDatasetConversion(
                     workerExecutablePath(),
                     source.absolutePath(),
                     root.filePath(QStringLiteral("converted_coco")),
                     QStringLiteral("yolo_detection"),
                     QStringLiteral("coco_json"),
                     QJsonObject{{QStringLiteral("copyImages"), true}},
                     &error,
                     QStringLiteral("dataset-conversion-test")),
            qPrintable(error));

        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            30000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_coco/annotations/train.json"))));
        QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_coco/dataset_conversion_report.json"))));

        bool sawConversionPayload = false;
        bool sawReportArtifact = false;
        bool sawStartProgress = false;
        bool sawDoneProgress = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("datasetConversion")) {
                sawConversionPayload = true;
                QCOMPARE(message.second.value(QStringLiteral("taskId")).toString(), QStringLiteral("dataset-conversion-test"));
                QVERIFY(message.second.value(QStringLiteral("ok")).toBool());
            } else if (message.first == QStringLiteral("artifact")
                && message.second.value(QStringLiteral("kind")).toString() == QStringLiteral("dataset_conversion_report")) {
                sawReportArtifact = true;
                QCOMPARE(message.second.value(QStringLiteral("taskId")).toString(), QStringLiteral("dataset-conversion-test"));
            } else if (message.first == QStringLiteral("progress")) {
                const int percent = message.second.value(QStringLiteral("percent")).toInt();
                sawStartProgress = sawStartProgress || percent == 0;
                sawDoneProgress = sawDoneProgress || percent == 100;
            }
        }
        QVERIFY(sawConversionPayload);
        QVERIFY(sawReportArtifact);
        QVERIFY(sawStartProgress);
        QVERIFY(sawDoneProgress);
    }

    void workerRunsProductWorkflowCommands()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);
        aitrain::DetectionTrainingOptions evaluationTrainingOptions;
        evaluationTrainingOptions.epochs = 1;
        evaluationTrainingOptions.batchSize = 1;
        evaluationTrainingOptions.imageSize = QSize(32, 32);
        evaluationTrainingOptions.outputPath = dir.filePath(QStringLiteral("evaluation-model"));
        const aitrain::DetectionTrainingResult evaluationTraining = aitrain::trainDetectionBaseline(datasetRoot, evaluationTrainingOptions);
        QVERIFY2(evaluationTraining.ok, qPrintable(evaluationTraining.error));
        const QString modelPath = evaluationTraining.checkpointPath;

        auto runCommand = [this](const std::function<bool(WorkerClient&, QString*)>& start, const QString& expectedMessageType, const QStringList& expectedArtifactKinds = {}) {
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
            QVERIFY2(start(client, &error), qPrintable(error));
            QTRY_VERIFY2_WITH_TIMEOUT(
                finished || !client.isRunning(),
                qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
                60000);
            if (finished) {
                QVERIFY2(ok, qPrintable(finishedMessage));
            }
            if (client.isRunning()) {
                QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);
            }

            bool sawExpected = false;
            QSet<QString> artifactKinds;
            QStringList observedTypes;
            for (const auto& message : messages) {
                observedTypes.append(message.first);
                sawExpected = sawExpected || message.first == expectedMessageType;
                if (message.first == QStringLiteral("artifact")) {
                    artifactKinds.insert(message.second.value(QStringLiteral("kind")).toString());
                }
            }
            QVERIFY2(sawExpected, qPrintable(QStringLiteral("Missing expected message: %1. Observed: %2")
                .arg(expectedMessageType, observedTypes.join(QStringLiteral(", ")))));
            QVERIFY(!artifactKinds.isEmpty());
            for (const QString& kind : expectedArtifactKinds) {
                QVERIFY2(artifactKinds.contains(kind), qPrintable(QStringLiteral("Missing artifact kind: %1").arg(kind)));
            }
        };

        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestDatasetSnapshot(
                workerExecutablePath(),
                datasetRoot,
                dir.filePath(QStringLiteral("worker-snapshot")),
                QStringLiteral("yolo_detection"),
                {},
                error,
                QStringLiteral("snapshot-task"));
        }, QStringLiteral("datasetSnapshot"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-snapshot/dataset_snapshot_manifest.json"))));

        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestDatasetCuration(
                workerExecutablePath(),
                datasetRoot,
                dir.filePath(QStringLiteral("worker-quality")),
                QStringLiteral("yolo_detection"),
                {},
                error,
                QStringLiteral("quality-task"));
        }, QStringLiteral("datasetQuality"), QStringList()
            << QStringLiteral("dataset_quality_report")
            << QStringLiteral("class_distribution")
            << QStringLiteral("problem_samples")
            << QStringLiteral("image_statistics")
            << QStringLiteral("split_distribution")
            << QStringLiteral("xanylabeling_fix_list")
            << QStringLiteral("xanylabeling_fix_manifest")
            << QStringLiteral("dataset_training_readiness"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-quality/dataset_quality_report.json"))));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-quality/dataset_training_readiness.json"))));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-quality/image_statistics.csv"))));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-quality/split_distribution.csv"))));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-quality/xanylabeling_fix_list.txt"))));

        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestModelEvaluation(
                workerExecutablePath(),
                modelPath,
                datasetRoot,
                dir.filePath(QStringLiteral("worker-evaluation")),
                QStringLiteral("detection"),
                {},
                error,
                QStringLiteral("evaluation-task"));
        }, QStringLiteral("evaluationReport"), QStringList()
            << QStringLiteral("evaluation_report")
            << QStringLiteral("evaluation_summary"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-evaluation/evaluation_report.json"))));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-evaluation/evaluation_summary.md"))));

        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestModelBenchmark(
                workerExecutablePath(),
                modelPath,
                dir.filePath(QStringLiteral("worker-benchmark")),
                {},
                error,
                QStringLiteral("benchmark-task"));
        }, QStringLiteral("benchmarkReport"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-benchmark/benchmark_report.json"))));

        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestDeploymentValidation(
                workerExecutablePath(),
                modelPath,
                dir.filePath(QStringLiteral("worker-deployment-validation")),
                QStringLiteral("tiny_detector_json"),
                QString(),
                {},
                error,
                QStringLiteral("deployment-validation-task"));
        }, QStringLiteral("deploymentValidation"), QStringList()
            << QStringLiteral("deployment_validation_report")
            << QStringLiteral("deployment_validation_summary"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-deployment-validation/deployment_validation_report.json"))));

        const QString ocrRoot = dir.filePath(QStringLiteral("worker-customer-ocr"));
        const QString detDataset = QDir(ocrRoot).filePath(QStringLiteral("det"));
        const QString recDataset = QDir(ocrRoot).filePath(QStringLiteral("rec"));
        const QString systemImages = QDir(ocrRoot).filePath(QStringLiteral("system"));
        writeTinyPng(QDir(detDataset).filePath(QStringLiteral("images/a.png")));
        writeTinyPng(QDir(recDataset).filePath(QStringLiteral("images/a.png")));
        writeTinyPng(QDir(systemImages).filePath(QStringLiteral("a.png")));
        const QString detReportPath = QDir(ocrRoot).filePath(QStringLiteral("det_report.json"));
        const QString recReportPath = QDir(ocrRoot).filePath(QStringLiteral("rec_report.json"));
        const QString systemReportPath = QDir(ocrRoot).filePath(QStringLiteral("system_report.json"));
        writeTextFile(detReportPath, QStringLiteral("{\"status\":\"passed\"}"));
        writeTextFile(recReportPath, QStringLiteral("{\"metrics\":{\"accuracy\":0.8,\"cer\":0.1,\"samples\":3}}"));
        writeTextFile(systemReportPath, QStringLiteral("{\"status\":\"passed\"}"));
        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestCustomerOcrAcceptance(
                workerExecutablePath(),
                dir.filePath(QStringLiteral("worker-ocr-acceptance")),
                QJsonObject{
                    {QStringLiteral("detDatasetPath"), detDataset},
                    {QStringLiteral("recDatasetPath"), recDataset},
                    {QStringLiteral("systemImagesPath"), systemImages},
                    {QStringLiteral("detReportPath"), detReportPath},
                    {QStringLiteral("recReportPath"), recReportPath},
                    {QStringLiteral("systemReportPath"), systemReportPath}
                },
                error,
                QStringLiteral("ocr-acceptance-task"));
        }, QStringLiteral("customerOcrAcceptance"), QStringList()
            << QStringLiteral("customer_ocr_acceptance")
            << QStringLiteral("customer_ocr_acceptance_summary"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-ocr-acceptance/customer_ocr_validation_manifest.json"))));

        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestDiagnosticsBundle(
                workerExecutablePath(),
                dir.filePath(QStringLiteral("worker-diagnostics")),
                QJsonObject{{QStringLiteral("projectName"), QStringLiteral("demo")}},
                error,
                QStringLiteral("diagnostics-task"));
        }, QStringLiteral("diagnosticBundle"), QStringList()
            << QStringLiteral("diagnostic_manifest")
            << QStringLiteral("diagnostic_bundle")
            << QStringLiteral("diagnostic_summary"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-diagnostics/diagnostic_bundle.json"))));

        runCommand([&](WorkerClient& client, QString* error) {
            return client.requestDeliveryReport(
                workerExecutablePath(),
                dir.filePath(QStringLiteral("worker-delivery")),
                QJsonObject{{QStringLiteral("projectName"), QStringLiteral("demo")}},
                error,
                QStringLiteral("delivery-task"));
        }, QStringLiteral("deliveryReport"), QStringList()
            << QStringLiteral("training_delivery_report")
            << QStringLiteral("delivery_manifest"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-delivery/training_delivery_report.html"))));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-delivery/delivery_manifest.json"))));

        runCommand([&](WorkerClient& client, QString* error) {
            QJsonObject pipelineOptions;
            pipelineOptions.insert(QStringLiteral("datasetPath"), datasetRoot);
            pipelineOptions.insert(QStringLiteral("datasetFormat"), QStringLiteral("yolo_detection"));
            pipelineOptions.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
            pipelineOptions.insert(QStringLiteral("trainingBackend"), QStringLiteral("tiny_linear_detector"));
            pipelineOptions.insert(QStringLiteral("epochs"), 1);
            pipelineOptions.insert(QStringLiteral("batchSize"), 1);
            pipelineOptions.insert(QStringLiteral("imageSize"), 32);
            pipelineOptions.insert(QStringLiteral("exportFormat"), QStringLiteral("onnx"));
            pipelineOptions.insert(QStringLiteral("sampleImagePath"), QDir(datasetRoot).filePath(QStringLiteral("images/val/a.png")));
            return client.requestLocalPipeline(
                workerExecutablePath(),
                dir.filePath(QStringLiteral("worker-pipeline")),
                QStringLiteral("train-evaluate-export-register"),
                pipelineOptions,
                error,
                QStringLiteral("pipeline-task"));
        }, QStringLiteral("pipelinePlan"), QStringList()
            << QStringLiteral("local_pipeline_execution")
            << QStringLiteral("delivery_report"));
        QVERIFY(QFileInfo::exists(dir.filePath(QStringLiteral("worker-pipeline/local_pipeline_plan.json"))));
    }

    void workerRunsLocalPipelineWithOfficialPythonTraining()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        const QString trainerScript = mockPythonTrainerScriptPath();
        QVERIFY2(!trainerScript.isEmpty(), "Mock Python trainer script is not available.");

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetRoot = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);
        const QString outputPath = dir.filePath(QStringLiteral("worker-pipeline-official"));

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

        QJsonObject pipelineOptions;
        pipelineOptions.insert(QStringLiteral("datasetPath"), datasetRoot);
        pipelineOptions.insert(QStringLiteral("datasetFormat"), QStringLiteral("yolo_detection"));
        pipelineOptions.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
        pipelineOptions.insert(QStringLiteral("trainingBackend"), QStringLiteral("python_mock"));
        pipelineOptions.insert(QStringLiteral("pythonExecutable"), python);
        pipelineOptions.insert(QStringLiteral("pythonTrainerScript"), trainerScript);
        pipelineOptions.insert(QStringLiteral("epochs"), 1);
        pipelineOptions.insert(QStringLiteral("mockStepsPerEpoch"), 2);
        pipelineOptions.insert(QStringLiteral("mockSleepMs"), 0);
        pipelineOptions.insert(QStringLiteral("mockCheckpointMode"), QStringLiteral("tiny_linear_detector"));
        pipelineOptions.insert(QStringLiteral("exportFormat"), QStringLiteral("onnx"));
        pipelineOptions.insert(QStringLiteral("sampleImagePath"), QDir(datasetRoot).filePath(QStringLiteral("images/val/a.png")));

        QString error;
        QVERIFY2(client.requestLocalPipeline(
            workerExecutablePath(),
            outputPath,
            QStringLiteral("train-evaluate-export-register"),
            pipelineOptions,
            &error,
            QStringLiteral("pipeline-official-task")), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            120000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawPipelinePlan = false;
        QString pipelineReportPath;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("pipelinePlan")) {
                sawPipelinePlan = true;
                QCOMPARE(message.second.value(QStringLiteral("state")).toString(), QStringLiteral("completed"));
                pipelineReportPath = message.second.value(QStringLiteral("reportPath")).toString();
                QVERIFY(!message.second.value(QStringLiteral("modelPath")).toString().isEmpty());
                break;
            }
        }
        QVERIFY(sawPipelinePlan);
        QVERIFY(QFileInfo::exists(pipelineReportPath));

        const QString pipelineJsonPath = QDir(outputPath).filePath(QStringLiteral("local_pipeline_plan.json"));
        QVERIFY(QFileInfo::exists(pipelineJsonPath));
        QFile file(pipelineJsonPath);
        QVERIFY(file.open(QIODevice::ReadOnly));
        const QJsonObject pipeline = QJsonDocument::fromJson(file.readAll()).object();
        QCOMPARE(pipeline.value(QStringLiteral("state")).toString(), QStringLiteral("completed"));
        QVERIFY(!pipeline.value(QStringLiteral("modelPath")).toString().isEmpty());
        QVERIFY(QFileInfo::exists(pipeline.value(QStringLiteral("modelPath")).toString()));
        QVERIFY(QFileInfo::exists(pipeline.value(QStringLiteral("evaluationReportPath")).toString()));
        QVERIFY(QFileInfo::exists(pipeline.value(QStringLiteral("deliveryReportPath")).toString()));

        const QJsonArray steps = pipeline.value(QStringLiteral("steps")).toArray();
        bool sawCompletedTrainingStep = false;
        for (const QJsonValue& value : steps) {
            const QJsonObject step = value.toObject();
            if (step.value(QStringLiteral("command")).toString() == QStringLiteral("startTrain")) {
                QCOMPARE(step.value(QStringLiteral("state")).toString(), QStringLiteral("completed"));
                sawCompletedTrainingStep = true;
            }
        }
        QVERIFY(sawCompletedTrainingStep);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("training/python_trainer_request.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("training/python_trainer_request.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("training/python_mock_training_report.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("training/python_mock_detection_checkpoint.aitrain"))));
    }

};

QTEST_MAIN(RepositoryWorkflowTests)
#include "tst_repository_workflow.moc"
