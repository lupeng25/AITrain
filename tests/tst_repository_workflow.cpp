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
        QCOMPARE(migrations, QVector<int>{1});
        repository.close();

        QVERIFY2(repository.open(dir.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
        QCOMPARE(repository.appliedSchemaMigrations(&error), QVector<int>{1});
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
        QCOMPARE(repository.appliedSchemaMigrations(&error), QVector<int>{1});
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
};

QTEST_MAIN(RepositoryWorkflowTests)
#include "tst_repository_workflow.moc"
