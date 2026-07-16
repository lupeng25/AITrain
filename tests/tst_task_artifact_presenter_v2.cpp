#include "TaskArtifactPresenterV2.h"

#include "aitrain/v2/ProjectWorkspaceV2.h"
#include "aitrain/v2/StorageV2.h"

#include <QDir>
#include <QHash>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

class TaskArtifactPresenterV2Tests : public QObject {
    Q_OBJECT

private slots:
    void readsPersistedTaskArtifactsMetricsAndWorkflowOnly();
    void invalidSelectionClearsReadModelWithoutPrivateAccess();
};

void TaskArtifactPresenterV2Tests::readsPersistedTaskArtifactsMetricsAndWorkflowOnly()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"),
        QStringLiteral("detection"), &task, &error), qPrintable(error));

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")),
        &error), qPrintable(error));
    const aitrain::v2::ArtifactId trainingReportId = aitrain::v2::ArtifactId::create();
    const aitrain::v2::ArtifactId runtimeReportId = aitrain::v2::ArtifactId::create();
    const aitrain::v2::ArtifactId evidenceId = aitrain::v2::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(trainingReportId, taskId,
        QStringLiteral("training_delivery_report_v2"),
        {{QStringLiteral("delivery_report.json"), QString(64, QLatin1Char('a')), 128}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(runtimeReportId, taskId,
        QStringLiteral("runtime_delivery_report_v2"),
        {{QStringLiteral("delivery_report.md"), QString(64, QLatin1Char('b')), 96}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(evidenceId, taskId,
        QStringLiteral("evidence_bundle_v2"),
        {{QStringLiteral("evidence.json"), QString(64, QLatin1Char('c')), 256}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordMetric(taskId, QStringLiteral("mAP50"), 0.75,
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::v2::WorkflowRunSnapshotV2 workflow;
    workflow.id = aitrain::v2::WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("training-delivery-v2");
    aitrain::v2::WorkflowStepSnapshotV2 step;
    step.id = aitrain::v2::WorkflowStepId::create();
    step.workflowRunId = workflow.id;
    step.ordinal = 0;
    step.kind = QStringLiteral("Evaluate");
    step.backend = QStringLiteral("ultralytics_official_val");
    QVERIFY2(storage.createWorkflowRun(workflow, {step}, &error), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    TaskArtifactPresenterV2 presenter(&query);
    // 夹具只打开 V2 Workspace，页面数据来自 V2 查询服务；
    // Presenter 刷新因此证明不依赖 project.sqlite 双事实源。
    QCOMPARE(presenter.objectName(), QStringLiteral("TaskArtifactPresenterV2"));
    QSignalSpy taskSpy(&presenter, &TaskArtifactPresenterV2::taskRowsChanged);
    QSignalSpy detailsSpy(&presenter, &TaskArtifactPresenterV2::detailsChanged);

    QVERIFY(presenter.refresh());
    QCOMPARE(taskSpy.count(), 1);
    QCOMPARE(presenter.property("taskCount").toInt(), 1);
    QCOMPARE(presenter.taskRows().constFirst().taskId, taskId.toString());
    QVERIFY(presenter.selectTask(taskId.toString()));
    QCOMPARE(detailsSpy.count(), 1);
    QCOMPARE(presenter.property("selectedTaskId").toString(), taskId.toString());
    QCOMPARE(presenter.property("artifactCount").toInt(), 3);
    QCOMPARE(presenter.property("metricCount").toInt(), 1);
    QCOMPARE(presenter.property("workflowStepCount").toInt(), 1);
    QHash<QString, ArtifactFileItemV2> deliveryArtifacts;
    for (const ArtifactFileItemV2& artifact : presenter.details().artifacts) {
        deliveryArtifacts.insert(artifact.kind, artifact);
        QVERIFY(!QDir::isAbsolutePath(artifact.relativePath));
    }
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("training_delivery_report_v2")).artifactId,
        trainingReportId.toString());
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("runtime_delivery_report_v2")).artifactId,
        runtimeReportId.toString());
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("evidence_bundle_v2")).artifactId,
        evidenceId.toString());
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("training_delivery_report_v2")).relativePath,
        QStringLiteral("delivery_report.json"));
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("runtime_delivery_report_v2")).sha256,
        QString(64, QLatin1Char('b')));
    QCOMPARE(presenter.details().workflowSteps.constFirst().kind, QStringLiteral("Evaluate"));
}

void TaskArtifactPresenterV2Tests::invalidSelectionClearsReadModelWithoutPrivateAccess()
{
    aitrain::v2::ProjectQueryServiceV2 query(nullptr);
    TaskArtifactPresenterV2 presenter(&query);
    QSignalSpy failureSpy(&presenter, &TaskArtifactPresenterV2::queryFailed);
    QVERIFY(!presenter.selectTask(QStringLiteral("not-a-task-id")));
    QCOMPARE(presenter.selectedTaskId(), QString());
    QCOMPARE(presenter.artifactCount(), 0);
    QCOMPARE(failureSpy.count(), 1);
    QVERIFY(!presenter.lastError().isEmpty());
}

QTEST_MAIN(TaskArtifactPresenterV2Tests)
#include "tst_task_artifact_presenter_v2.moc"
