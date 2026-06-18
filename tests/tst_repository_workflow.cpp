#include "TestSupport.h"

#include <QSqlDatabase>
#include <QSqlQuery>

class RepositoryWorkflowTests : public QObject {
    Q_OBJECT

private slots:
    void repositoryRecordsSchemaBaseline()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        aitrain::ProjectRepository repository;
        QString error;
        QVERIFY2(repository.open(dir.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
        QCOMPARE(aitrain::ProjectRepository::currentSchemaVersion(), 1);
        QCOMPARE(repository.schemaVersion(&error), 1);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        const QVector<int> migrations = repository.appliedSchemaMigrations(&error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(migrations.size(), 1);
        QCOMPARE(migrations.value(0), 1);
        repository.close();

        QVERIFY2(repository.open(dir.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
        const QVector<int> reopenedMigrations = repository.appliedSchemaMigrations(&error);
        QCOMPARE(reopenedMigrations.size(), 1);
        QCOMPARE(reopenedMigrations.value(0), 1);
        QVERIFY2(error.isEmpty(), qPrintable(error));
    }

    void repositoryBaselinesLegacyDatabaseWithoutMigrationTable()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString dbPath = dir.filePath(QStringLiteral("legacy.sqlite"));
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        const QString connectionName = QStringLiteral("legacy_%1").arg(QUuid::createUuid().toString(QUuid::Id128));

        QSqlDatabase legacyDb = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
        legacyDb.setDatabaseName(dbPath);
        QVERIFY(legacyDb.open());
        QSqlQuery query(legacyDb);
        QVERIFY(query.exec(QStringLiteral("create table tasks ("
                                          "id text primary key,"
                                          "project_name text not null,"
                                          "plugin_id text not null,"
                                          "task_type text not null,"
                                          "kind text not null,"
                                          "state text not null,"
                                          "work_dir text not null,"
                                          "message text,"
                                          "created_at text not null,"
                                          "updated_at text not null)")));
        query.prepare(QStringLiteral("insert into tasks(id, project_name, plugin_id, task_type, kind, state, work_dir, message, created_at, updated_at) "
                                     "values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
        query.addBindValue(taskId);
        query.addBindValue(QStringLiteral("legacy"));
        query.addBindValue(QStringLiteral("com.aitrain.plugins.yolo_native"));
        query.addBindValue(QStringLiteral("detection"));
        query.addBindValue(QStringLiteral("train"));
        query.addBindValue(QStringLiteral("queued"));
        query.addBindValue(dir.filePath(QStringLiteral("runs/legacy")));
        query.addBindValue(QStringLiteral("legacy queued task"));
        query.addBindValue(QStringLiteral("2026-05-01T00:00:00.000Z"));
        query.addBindValue(QStringLiteral("2026-05-01T00:00:00.000Z"));
        QVERIFY(query.exec());
        legacyDb.close();
        legacyDb = QSqlDatabase();
        QSqlDatabase::removeDatabase(connectionName);

        aitrain::ProjectRepository repository;
        QString error;
        QVERIFY2(repository.open(dbPath, &error), qPrintable(error));
        QCOMPARE(repository.schemaVersion(&error), 1);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        const QVector<int> legacyMigrations = repository.appliedSchemaMigrations(&error);
        QCOMPARE(legacyMigrations.size(), 1);
        QCOMPARE(legacyMigrations.value(0), 1);
        QVERIFY2(error.isEmpty(), qPrintable(error));

        const QVector<aitrain::TaskRecord> tasks = repository.recentTasks(10, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(tasks.size(), 1);
        QCOMPARE(tasks.first().id, taskId);
        QCOMPARE(tasks.first().message, QStringLiteral("legacy queued task"));
    }

    void repositoryStoresOfficialTrainingArtifacts()
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
        QVERIFY2(repository.updateTaskState(task.id, aitrain::TaskState::Completed, QStringLiteral("official training completed"), &error), qPrintable(error));

        aitrain::MetricPoint metric;
        metric.taskId = task.id;
        metric.name = QStringLiteral("mAP50");
        metric.value = 0.75;
        metric.step = 1;
        metric.epoch = 1;
        metric.createdAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.insertMetric(metric, &error), qPrintable(error));

        aitrain::ArtifactRecord artifact;
        artifact.taskId = task.id;
        artifact.kind = QStringLiteral("onnx");
        artifact.path = dir.filePath(QStringLiteral("runs/1/best.onnx"));
        artifact.message = QStringLiteral("Official Ultralytics ONNX export");
        QVERIFY2(repository.insertArtifact(artifact, &error), qPrintable(error));

        QJsonObject exportConfig;
        exportConfig.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        exportConfig.insert(QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_detect"));
        exportConfig.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_detection"));
        exportConfig.insert(QStringLiteral("scaffold"), false);
        exportConfig.insert(QStringLiteral("sourceCheckpoint"), artifact.path);

        aitrain::ExportRecord exportRecord;
        exportRecord.taskId = task.id;
        exportRecord.sourceCheckpointPath = artifact.path;
        exportRecord.format = QStringLiteral("onnx");
        exportRecord.path = dir.filePath(QStringLiteral("runs/1/model.onnx"));
        exportRecord.configJson = QString::fromUtf8(QJsonDocument(exportConfig).toJson(QJsonDocument::Compact));
        exportRecord.createdAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.insertExport(exportRecord, &error), qPrintable(error));

        const QVector<aitrain::TaskRecord> tasks = repository.recentTasks(10, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(tasks.size(), 1);
        QCOMPARE(tasks.first().state, aitrain::TaskState::Completed);

        const QVector<aitrain::ArtifactRecord> artifacts = repository.artifactsForTask(task.id, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(artifacts.size(), 1);
        QCOMPARE(artifacts.first().kind, QStringLiteral("onnx"));
    }

    void deliveryReportDoesNotMarkOfficialTrainingAsScaffold()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        QJsonObject context;
        context.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
        context.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect"));
        context.insert(QStringLiteral("modelPath"), dir.filePath(QStringLiteral("best.onnx")));
        context.insert(QStringLiteral("scaffold"), false);

        const aitrain::WorkflowResult result = aitrain::generateTrainingDeliveryReport(
            dir.filePath(QStringLiteral("delivery")),
            context);
        QVERIFY2(result.ok, qPrintable(result.error));
        QVERIFY(QFileInfo::exists(result.reportPath));

        const QJsonArray limitations = result.payload.value(QStringLiteral("limitations")).toArray();
        for (const QJsonValue& value : limitations) {
            QVERIFY(!value.toString().contains(QStringLiteral("diagnostic"), Qt::CaseInsensitive));
        }
    }

    void anomalyPipelineSkipsOnnxExportForSidecarArtifacts()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the fake Anomalib benchmark adapter.");
        }

        const QString modelDir = dir.filePath(QStringLiteral("model"));
        const QString checkpointPath = QDir(modelDir).filePath(QStringLiteral("model.ckpt"));
        const QString sidecarPath = QDir(modelDir).filePath(QStringLiteral("anomaly_sidecar.json"));
        const QString escapedCheckpointPath = QString(checkpointPath).replace(QLatin1Char('\\'), QStringLiteral("\\\\"));
        writeTextFile(checkpointPath, QStringLiteral("fake checkpoint\n"));
        writeTextFile(
            sidecarPath,
            QStringLiteral("{\n"
                           "  \"schemaVersion\": 1,\n"
                           "  \"kind\": \"anomaly_sidecar\",\n"
                           "  \"taskType\": \"anomaly_detection\",\n"
                           "  \"runtime\": \"anomalib_python\",\n"
                           "  \"trainingBackend\": \"anomalib_patchcore\",\n"
                           "  \"checkpointPath\": \"%1\"\n"
                           "}\n").arg(escapedCheckpointPath));

        const QString adapterScript = dir.filePath(QStringLiteral("fake_anomalib_adapter.py"));
        writeTextFile(
            adapterScript,
            QStringLiteral("import argparse, json\n"
                           "from pathlib import Path\n"
                           "parser = argparse.ArgumentParser()\n"
                           "parser.add_argument('--request', required=True)\n"
                           "parser.add_argument('--mode', default='benchmark')\n"
                           "args = parser.parse_args()\n"
                           "request = json.loads(Path(args.request).read_text(encoding='utf-8-sig'))\n"
                           "out = Path(request['outputPath'])\n"
                           "out.mkdir(parents=True, exist_ok=True)\n"
                           "report = {\n"
                           "  'schemaVersion': 1,\n"
                           "  'kind': 'benchmark_report',\n"
                           "  'ok': True,\n"
                           "  'status': 'completed',\n"
                           "  'runtime': 'anomalib_python',\n"
                           "  'modelFamily': 'anomaly_detection',\n"
                           "  'runtimeUsable': True,\n"
                           "  'timedInference': True,\n"
                           "  'averageMs': 1.0,\n"
                           "  'p50Ms': 1.0,\n"
                           "  'p95Ms': 1.0,\n"
                           "  'p99Ms': 1.0,\n"
                           "  'throughput': 1000.0\n"
                           "}\n"
                           "(out / 'benchmark_report.json').write_text(json.dumps(report), encoding='utf-8')\n"));

        const QString datasetPath = dir.filePath(QStringLiteral("empty_dataset"));
        QDir().mkpath(datasetPath);

        QJsonObject benchmarkOptions;
        benchmarkOptions.insert(QStringLiteral("runtime"), QStringLiteral("anomalib_python"));
        benchmarkOptions.insert(QStringLiteral("pythonExecutable"), python);
        benchmarkOptions.insert(QStringLiteral("anomalibAdapterScript"), adapterScript);

        QJsonObject options;
        options.insert(QStringLiteral("taskType"), QStringLiteral("anomaly_detection"));
        options.insert(QStringLiteral("datasetFormat"), QStringLiteral("anomaly_folder"));
        options.insert(QStringLiteral("datasetPath"), datasetPath);
        options.insert(QStringLiteral("trainingBackend"), QStringLiteral("anomalib_patchcore"));
        options.insert(QStringLiteral("runtime"), QStringLiteral("anomalib_python"));
        options.insert(QStringLiteral("modelPath"), sidecarPath);
        options.insert(QStringLiteral("exportFormat"), QStringLiteral("anomalib_python"));
        options.insert(QStringLiteral("benchmarkOptions"), benchmarkOptions);

        const aitrain::WorkflowResult result = aitrain::runLocalPipelinePlan(
            dir.filePath(QStringLiteral("pipeline")),
            QStringLiteral("export-infer-benchmark-report"),
            options);
        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.payload.value(QStringLiteral("state")).toString(), QStringLiteral("completed"));
        QCOMPARE(QFileInfo(result.payload.value(QStringLiteral("modelPath")).toString()).absoluteFilePath(), QFileInfo(sidecarPath).absoluteFilePath());
        QVERIFY(result.payload.value(QStringLiteral("exportPath")).toString().isEmpty());
        QVERIFY(!QFileInfo::exists(QDir(dir.filePath(QStringLiteral("pipeline"))).filePath(QStringLiteral("export/model.onnx"))));

        bool sawSkippedExport = false;
        bool sawSkippedInference = false;
        bool sawCompletedBenchmark = false;
        const QJsonArray steps = result.payload.value(QStringLiteral("steps")).toArray();
        for (const QJsonValue& value : steps) {
            const QJsonObject step = value.toObject();
            const QString command = step.value(QStringLiteral("command")).toString();
            const QString state = step.value(QStringLiteral("state")).toString();
            if (command == QStringLiteral("exportModel")) {
                sawSkippedExport = state == QStringLiteral("skipped");
            } else if (command == QStringLiteral("infer")) {
                sawSkippedInference = state == QStringLiteral("skipped");
            } else if (command == QStringLiteral("benchmarkModel")) {
                sawCompletedBenchmark = state == QStringLiteral("completed");
            }
        }
        QVERIFY(sawSkippedExport);
        QVERIFY(sawSkippedInference);
        QVERIFY(sawCompletedBenchmark);
    }
};

QTEST_MAIN(RepositoryWorkflowTests)
#include "tst_repository_workflow.moc"
