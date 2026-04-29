#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/ProjectRepository.h"

#include <QTemporaryDir>
#include <QTest>
#include <QUuid>

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
};

QTEST_MAIN(CoreTests)
#include "tst_core.moc"
