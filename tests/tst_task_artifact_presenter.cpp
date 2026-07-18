#include "TaskArtifactPresenter.h"

#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/storage/ProjectStore.h"

#include <QDir>
#include <QCryptographicHash>
#include <QEventLoop>
#include <QFile>
#include <QHash>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QTest>

#include <utility>

class TaskArtifactPresenterTests : public QObject {
    Q_OBJECT

private slots:
    void readsPersistedTaskArtifactsMetricsAndWorkflowOnly();
    void readsCommittedArtifactPreviewByIdentity();
    void readsCommittedArtifactPreviewAsynchronouslyWithMetadataSnapshot();
    void invalidSelectionClearsReadModelWithoutPrivateAccess();
};

void TaskArtifactPresenterTests::readsPersistedTaskArtifactsMetricsAndWorkflowOnly()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"),
        QStringLiteral("detection"), &task, &error), qPrintable(error));
    aitrain::Failure failure;
    failure.code = aitrain::FailureCode::InvalidDataset;
    failure.message = QStringLiteral("snapshot mismatch");
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::TaskState::Failed, failure, &error), qPrintable(error));

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")),
        &error), qPrintable(error));
    const aitrain::ArtifactId trainingReportId = aitrain::ArtifactId::create();
    const aitrain::ArtifactId runtimeReportId = aitrain::ArtifactId::create();
    const aitrain::ArtifactId evidenceId = aitrain::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(trainingReportId, taskId,
        QStringLiteral("training_delivery_report"),
        {{QStringLiteral("delivery_report.json"), QString(64, QLatin1Char('a')), 128}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(runtimeReportId, taskId,
        QStringLiteral("runtime_delivery_report"),
        {{QStringLiteral("delivery_report.md"), QString(64, QLatin1Char('b')), 96}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(evidenceId, taskId,
        QStringLiteral("evidence_bundle"),
        {{QStringLiteral("evidence.json"), QString(64, QLatin1Char('c')), 256}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordMetric(taskId, QStringLiteral("mAP50"), 0.75,
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::WorkflowRunSnapshot workflow;
    workflow.id = aitrain::WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("training-delivery");
    aitrain::WorkflowStepSnapshot step;
    step.id = aitrain::WorkflowStepId::create();
    step.workflowRunId = workflow.id;
    step.ordinal = 0;
    step.kind = QStringLiteral("Evaluate");
    step.backend = QStringLiteral("ultralytics_official_val");
    QVERIFY2(storage.createWorkflowRun(workflow, {step}, &error), qPrintable(error));

    aitrain::ProjectQueryService query(&workspace);
    TaskArtifactPresenter presenter(&query);
    // 夹具只打开  Workspace，页面数据来自  查询服务；
    // Presenter 刷新因此证明不依赖 project.sqlite 双事实源。
    QCOMPARE(presenter.objectName(), QStringLiteral("TaskArtifactPresenter"));
    QSignalSpy taskSpy(&presenter, &TaskArtifactPresenter::taskRowsChanged);
    QSignalSpy detailsSpy(&presenter, &TaskArtifactPresenter::detailsChanged);

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
    QHash<QString, ArtifactFileItem> deliveryArtifacts;
    for (const ArtifactFileItem& artifact : presenter.details().artifacts) {
        deliveryArtifacts.insert(artifact.kind, artifact);
        QVERIFY(!QDir::isAbsolutePath(artifact.relativePath));
    }
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("training_delivery_report")).artifactId,
        trainingReportId.toString());
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("runtime_delivery_report")).artifactId,
        runtimeReportId.toString());
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("evidence_bundle")).artifactId,
        evidenceId.toString());
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("training_delivery_report")).relativePath,
        QStringLiteral("delivery_report.json"));
    QCOMPARE(deliveryArtifacts.value(QStringLiteral("runtime_delivery_report")).sha256,
        QString(64, QLatin1Char('b')));
    QCOMPARE(presenter.details().workflowSteps.constFirst().kind, QStringLiteral("Evaluate"));
    QCOMPARE(presenter.details().failureCode, QStringLiteral("invalid_dataset"));
    QVERIFY(presenter.details().failureAction.contains(QStringLiteral("数据集页")));
    QVERIFY(presenter.details().summary.contains(QStringLiteral("失败代码：invalid_dataset")));
}

void TaskArtifactPresenterTests::invalidSelectionClearsReadModelWithoutPrivateAccess()
{
    aitrain::ProjectQueryService query(nullptr);
    TaskArtifactPresenter presenter(&query);
    QSignalSpy failureSpy(&presenter, &TaskArtifactPresenter::queryFailed);
    QVERIFY(!presenter.selectTask(QStringLiteral("not-a-task-id")));
    QCOMPARE(presenter.selectedTaskId(), QString());
    QCOMPARE(presenter.artifactCount(), 0);
    QCOMPARE(failureSpy.count(), 1);
    QVERIFY(!presenter.lastError().isEmpty());
}

void TaskArtifactPresenterTests::readsCommittedArtifactPreviewByIdentity()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("preview"), QStringLiteral("report"), &task, &error),
        qPrintable(error));

    const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
    const QByteArray content = QByteArrayLiteral("{\"status\":\"ok\"}\n");
    const QString sha256 = QString::fromLatin1(QCryptographicHash::hash(content, QCryptographicHash::Sha256).toHex());
    const QString artifactRoot = QDir(workspace.workspacePath()).filePath(
        QStringLiteral("artifacts/committed/%1").arg(artifactId.toString()));
    QVERIFY(QDir().mkpath(artifactRoot));
    QFile file(QDir(artifactRoot).filePath(QStringLiteral("evaluation_report.json")));
    QVERIFY(file.open(QIODevice::WriteOnly));
    QCOMPARE(file.write(content), content.size());
    file.close();

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")), &error),
        qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, taskId, QStringLiteral("evaluation_report"),
        {{QStringLiteral("evaluation_report.json"), sha256, content.size()}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::ProjectQueryService query(&workspace);
    aitrain::ArtifactFilePreview preview;
    QVERIFY2(query.artifactFilePreview(artifactId, QStringLiteral("evaluation_report.json"), &preview, 1024, &error),
        qPrintable(error));
    QCOMPARE(preview.relativePath, QStringLiteral("evaluation_report.json"));
    QCOMPARE(preview.content, content);
    QVERIFY(!preview.truncated);
    QVERIFY(!query.artifactFilePreview(artifactId, QStringLiteral("../project.sqlite"), &preview, 1024, &error));
}

void TaskArtifactPresenterTests::readsCommittedArtifactPreviewAsynchronouslyWithMetadataSnapshot()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("preview_async"), QStringLiteral("report"),
        &task, &error), qPrintable(error));

    const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
    QByteArray content;
    content.resize(2 * 1024 * 1024);
    for (int i = 0; i < content.size(); ++i) content[i] = static_cast<char>(i % 251);
    const QString sha256 = QString::fromLatin1(QCryptographicHash::hash(
        content, QCryptographicHash::Sha256).toHex());
    const QString artifactRoot = QDir(workspace.workspacePath()).filePath(
        QStringLiteral("artifacts/committed/%1").arg(artifactId.toString()));
    QVERIFY(QDir().mkpath(artifactRoot));
    QFile file(QDir(artifactRoot).filePath(QStringLiteral("large.bin")));
    QVERIFY(file.open(QIODevice::WriteOnly));
    QCOMPARE(file.write(content), content.size());
    file.close();

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")),
        &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, taskId, QStringLiteral("report"),
        {{QStringLiteral("large.bin"), sha256, content.size()}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::ProjectQueryService query(&workspace);
    QObject receiver;
    QEventLoop loop;
    QTimer timeout;
    timeout.setSingleShot(true);
    timeout.setInterval(5000);
    connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
    bool callbackCalled = false;
    bool callbackSuccess = false;
    bool callbackOnCallingThread = false;
    aitrain::ArtifactFilePreview result;
    QString callbackError;
    QThread* callingThread = QThread::currentThread();
    QVERIFY2(query.artifactFilePreviewAsync(artifactId, QStringLiteral("large.bin"), &receiver,
        [&](bool success, aitrain::ArtifactFilePreview preview, QString readError) {
            callbackCalled = true;
            callbackSuccess = success;
            callbackOnCallingThread = QThread::currentThread() == callingThread;
            result = std::move(preview);
            callbackError = std::move(readError);
            loop.quit();
        }, 1024, &error), qPrintable(error));
    timeout.start();
    loop.exec();
    QVERIFY2(callbackCalled, "异步 Artifact 预览回调未返回");
    QVERIFY2(callbackSuccess, qPrintable(callbackError));
    QVERIFY(callbackOnCallingThread);
    QCOMPARE(result.relativePath, QStringLiteral("large.bin"));
    QCOMPARE(result.sha256, sha256);
    QCOMPARE(result.byteCount, qint64(content.size()));
    QCOMPARE(result.content, content.left(1024));
    QVERIFY(result.truncated);
}

QTEST_MAIN(TaskArtifactPresenterTests)
#include "tst_task_artifact_presenter.moc"
