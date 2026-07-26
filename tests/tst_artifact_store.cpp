#include "aitrain/artifact/ArtifactStore.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QTest>

namespace {

aitrain::TaskSnapshot createTask(aitrain::ProjectStore* storage)
{
    aitrain::TaskSnapshot task;
    task.id = aitrain::TaskId::create();
    task.requestId = aitrain::RequestId::create();
    task.capabilityId = QStringLiteral("yolo");
    task.taskType = QStringLiteral("detection");
    QString error;
    if (!storage->createTask(task, &error)) {
        return {};
    }
    return task;
}

bool writeFile(const QString& path, const QByteArray& content)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(content) == content.size();
}

aitrain::WorkflowRunSnapshot createSealedEvidenceWorkflow(
    aitrain::ProjectStore* storage,
    const aitrain::TaskSnapshot& task,
    QString* error)
{
    aitrain::WorkflowRunSnapshot workflow;
    workflow.id = aitrain::WorkflowRunId::create();
    workflow.taskId = task.id;
    workflow.templateId = QStringLiteral("artifact-recovery-evidence");
    workflow.terminalPolicy = aitrain::WorkflowTerminalPolicy::EvidenceRequired;
    aitrain::WorkflowStepSnapshot step;
    step.id = aitrain::WorkflowStepId::create();
    step.workflowRunId = workflow.id;
    step.ordinal = 0;
    step.kind = QStringLiteral("ValidateDataset");
    step.backend = QStringLiteral("fixture");
    const aitrain::Failure failure{aitrain::FailureCode::InvalidDataset, QStringLiteral("测试失败。"),
        QStringLiteral("修复测试输入。"), QDateTime::currentDateTimeUtc()};
    if (!storage->createWorkflowRun(workflow, {step}, error)
        || !storage->transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Pending,
            aitrain::WorkflowStepState::Running, {}, {}, error)
        || !storage->transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Running,
            aitrain::WorkflowStepState::Failed, {}, failure, error)
        || !storage->sealWorkflowTerminalization(workflow.id, aitrain::TaskState::Failed,
            failure, QDateTime::currentDateTimeUtc(), error)) return {};
    return workflow;
}

} // namespace

class ArtifactStoreTests : public QObject {
    Q_OBJECT

private slots:
    void commitsVerifiedArtifactAtomically();
    void cancellationDuringCommitKeepsOnlyAbortableStaging();
    void abortRemovesStagingAndEmptyStagingCannotCommit();
    void recoveryPreservesActiveAndCleansAbandonedStaging();
    void rejectsMismatchedStagingIdentity();
    void recoveryCompletesEvidenceCommitAfterDirectoryRename();
    void recoveryCleansJournalAfterDatabaseCommit();
    void discardUsesTrashAndRecoveryRestoresDatabaseOwnedArtifact();
};

void ArtifactStoreTests::commitsVerifiedArtifactAtomically()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = createTask(&storage);
    QVERIFY(task.id.isValid());
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(artifacts.begin(task.id, QStringLiteral("model"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(stagingPath).filePath(QStringLiteral("nested/model.onnx")), QByteArray("verified model")));

    QString artifactPath;
    QVERIFY2(artifacts.commit(artifactId, task.id, QStringLiteral("model"), stagingPath, &storage, &artifactPath, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(stagingPath));
    QVERIFY(QFileInfo::exists(QDir(artifactPath).filePath(QStringLiteral("nested/model.onnx"))));
    QVERIFY(QFileInfo::exists(QDir(artifactPath).filePath(QStringLiteral("manifest.json"))));
    QCOMPARE(storage.artifactCount(task.id, &error), 1);
    QCOMPARE(storage.artifactFileCount(artifactId, &error), 1);
}

void ArtifactStoreTests::cancellationDuringCommitKeepsOnlyAbortableStaging()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = createTask(&storage);
    QVERIFY(task.id.isValid());
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(artifacts.begin(task.id, QStringLiteral("model"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(stagingPath).filePath(QStringLiteral("model.onnx")), QByteArray(3 * 1024 * 1024, 'm')));

    int cancellationChecks = 0;
    bool canceled = false;
    QVERIFY(!artifacts.commit(artifactId, task.id, QStringLiteral("model"), stagingPath, &storage, nullptr, &error,
        [&cancellationChecks]() { return ++cancellationChecks >= 3; }, &canceled));
    QVERIFY(canceled);
    QVERIFY(error.contains(QStringLiteral("取消")));
    QVERIFY(QFileInfo::exists(stagingPath));
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
    QVERIFY2(artifacts.abort(stagingPath, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(stagingPath));
}

void ArtifactStoreTests::abortRemovesStagingAndEmptyStagingCannotCommit()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = createTask(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(artifacts.begin(task.id, QStringLiteral("report"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(!artifacts.commit(artifactId, task.id, QStringLiteral("report"), stagingPath, &storage, nullptr, &error));
    QVERIFY(error.contains(QStringLiteral("不能为空")));
    QVERIFY2(artifacts.abort(stagingPath, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(stagingPath));
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
}

void ArtifactStoreTests::recoveryPreservesActiveAndCleansAbandonedStaging()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot activeTask = createTask(&storage);
    const aitrain::TaskSnapshot failedTask = createTask(&storage);
    QVERIFY2(storage.transitionTask(failedTask.id, aitrain::TaskState::Created, aitrain::TaskState::Failed,
        {aitrain::FailureCode::ProcessCrashed, QStringLiteral("fixture"),
            aitrain::defaultFailureSuggestedAction(aitrain::FailureCode::ProcessCrashed),
            QDateTime::currentDateTimeUtc()}, &error), qPrintable(error));

    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ArtifactId activeArtifactId;
    aitrain::ArtifactId failedArtifactId;
    aitrain::ArtifactId orphanArtifactId;
    aitrain::ArtifactId metadataOnlyArtifactId;
    QString activeStaging;
    QString failedStaging;
    QString orphanStaging;
    QString metadataOnlyStaging;
    QVERIFY2(artifacts.begin(activeTask.id, QStringLiteral("report"), &activeArtifactId, &activeStaging, &error), qPrintable(error));
    QVERIFY2(artifacts.begin(failedTask.id, QStringLiteral("report"), &failedArtifactId, &failedStaging, &error), qPrintable(error));
    QVERIFY2(artifacts.begin(aitrain::TaskId::create(), QStringLiteral("report"), &orphanArtifactId, &orphanStaging, &error), qPrintable(error));
    QVERIFY2(artifacts.begin(activeTask.id, QStringLiteral("report"), &metadataOnlyArtifactId, &metadataOnlyStaging, &error), qPrintable(error));
    QVERIFY(QDir(metadataOnlyStaging).removeRecursively());
    const QString metadataOnlyPath = directory.filePath(QStringLiteral("store/.staging-meta/%1.json").arg(metadataOnlyArtifactId.toString()));
    QVERIFY(QFileInfo::exists(metadataOnlyPath));

    QStringList diagnostics;
    QVERIFY2(artifacts.recoverStaging(&storage, &diagnostics, &error), qPrintable(error));
    QVERIFY(QFileInfo::exists(activeStaging));
    QVERIFY(!QFileInfo::exists(failedStaging));
    QVERIFY(!QFileInfo::exists(orphanStaging));
    QVERIFY(!QFileInfo::exists(metadataOnlyPath));
    QVERIFY(!diagnostics.isEmpty());
    QVERIFY2(artifacts.abort(activeStaging, &error), qPrintable(error));
}

void ArtifactStoreTests::rejectsMismatchedStagingIdentity()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = createTask(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ArtifactId firstId;
    aitrain::ArtifactId secondId;
    QString firstStaging;
    QString secondStaging;
    QVERIFY2(artifacts.begin(task.id, QStringLiteral("model"), &firstId, &firstStaging, &error), qPrintable(error));
    QVERIFY2(artifacts.begin(task.id, QStringLiteral("model"), &secondId, &secondStaging, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(secondStaging).filePath(QStringLiteral("model.onnx")), QByteArray("fixture")));

    QVERIFY(!artifacts.commit(firstId, task.id, QStringLiteral("model"), secondStaging,
        &storage, nullptr, &error));
    QVERIFY(QFileInfo::exists(firstStaging));
    QVERIFY(QFileInfo::exists(secondStaging));
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
    QVERIFY2(artifacts.abort(firstStaging, &error), qPrintable(error));
    QVERIFY2(artifacts.abort(secondStaging, &error), qPrintable(error));
}

void ArtifactStoreTests::recoveryCompletesEvidenceCommitAfterDirectoryRename()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = createTask(&storage);
    const auto workflow = createSealedEvidenceWorkflow(&storage, task, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));
    const QString storeRoot = directory.filePath(QStringLiteral("store"));
    aitrain::ArtifactStore interrupted(storeRoot,
        [](aitrain::ArtifactCommitFailPoint point) {
            return point == aitrain::ArtifactCommitFailPoint::AfterDirectoryRenameBeforeDatabase;
        });
    aitrain::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(interrupted.begin(task.id, QStringLiteral("evidence_bundle"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(stagingPath).filePath(QStringLiteral("evidence.json")), QByteArray("{}")));
    const aitrain::ArtifactCommitResult pending = interrupted.commit(
        artifactId, task.id, QStringLiteral("evidence_bundle"), stagingPath,
        &storage, nullptr, &error, {}, nullptr, workflow.id);
    QCOMPARE(pending.status, aitrain::ArtifactCommitStatus::PendingRecovery);
    bool exists = true;
    QVERIFY2(storage.artifactExists(artifactId, &exists, &error), qPrintable(error));
    QVERIFY(!exists);
    const QString committedPath = QDir(storeRoot).filePath(QStringLiteral("committed/%1").arg(artifactId.toString()));
    const QString journalPath = QDir(storeRoot).filePath(QStringLiteral(".staging-meta/%1.json").arg(artifactId.toString()));
    QVERIFY(QFileInfo::exists(committedPath));
    QVERIFY(QFileInfo::exists(journalPath));
    QString abortError;
    QVERIFY(!interrupted.abort(stagingPath, &abortError));
    QVERIFY(QFileInfo::exists(committedPath));
    QVERIFY(QFileInfo::exists(journalPath));

    aitrain::ArtifactStore recovered(storeRoot);
    QStringList diagnostics;
    error.clear();
    QVERIFY2(recovered.recoverStaging(&storage, &diagnostics, &error), qPrintable(error));
    QVERIFY2(storage.artifactExists(artifactId, &exists, &error), qPrintable(error));
    QVERIFY(exists);
    aitrain::WorkflowTerminalizationSnapshot terminalization;
    QVERIFY2(storage.workflowTerminalization(workflow.id, &terminalization, &error), qPrintable(error));
    QCOMPARE(terminalization.state, aitrain::WorkflowTerminalizationState::EvidenceAttached);
    QCOMPARE(terminalization.evidenceArtifactId, artifactId);
    QVERIFY(!QFileInfo::exists(journalPath));
}

void ArtifactStoreTests::recoveryCleansJournalAfterDatabaseCommit()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = createTask(&storage);
    const auto workflow = createSealedEvidenceWorkflow(&storage, task, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));
    const QString storeRoot = directory.filePath(QStringLiteral("store"));
    aitrain::ArtifactStore interrupted(storeRoot,
        [](aitrain::ArtifactCommitFailPoint point) {
            return point == aitrain::ArtifactCommitFailPoint::AfterDatabaseBeforeJournalRemoval;
        });
    aitrain::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(interrupted.begin(task.id, QStringLiteral("evidence_bundle"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(stagingPath).filePath(QStringLiteral("evidence.json")), QByteArray("{}")));
    const aitrain::ArtifactCommitResult committed = interrupted.commit(
        artifactId, task.id, QStringLiteral("evidence_bundle"), stagingPath,
        &storage, nullptr, &error, {}, nullptr, workflow.id);
    QCOMPARE(committed.status, aitrain::ArtifactCommitStatus::Committed);
    QVERIFY(committed.cleanupPending);
    bool exists = false;
    QVERIFY2(storage.artifactExists(artifactId, &exists, &error), qPrintable(error));
    QVERIFY(exists);
    const QString journalPath = QDir(storeRoot).filePath(QStringLiteral(".staging-meta/%1.json").arg(artifactId.toString()));
    QVERIFY(QFileInfo::exists(journalPath));

    aitrain::ArtifactStore recovered(storeRoot);
    QStringList diagnostics;
    error.clear();
    QVERIFY2(recovered.recoverStaging(&storage, &diagnostics, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(journalPath));
    QCOMPARE(storage.artifactCount(task.id, &error), 1);
    aitrain::WorkflowTerminalizationSnapshot terminalization;
    QVERIFY2(storage.workflowTerminalization(workflow.id, &terminalization, &error), qPrintable(error));
    QCOMPARE(terminalization.state, aitrain::WorkflowTerminalizationState::EvidenceAttached);
}

void ArtifactStoreTests::discardUsesTrashAndRecoveryRestoresDatabaseOwnedArtifact()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = createTask(&storage);
    const QString storeRoot = directory.filePath(QStringLiteral("store"));
    aitrain::ArtifactStore artifacts(storeRoot);
    aitrain::ArtifactId artifactId;
    QString stagingPath;
    QString committedPath;
    QVERIFY2(artifacts.begin(task.id, QStringLiteral("report"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(stagingPath).filePath(QStringLiteral("report.json")), QByteArray("{}")));
    QVERIFY2(artifacts.commit(artifactId, task.id, QStringLiteral("report"), stagingPath,
        &storage, &committedPath, &error), qPrintable(error));

    const QString trashPath = QDir(storeRoot).filePath(
        QStringLiteral(".trash/%1").arg(artifactId.toString()));
    QVERIFY(QDir().mkpath(QFileInfo(trashPath).absolutePath()));
    QVERIFY(QDir().rename(committedPath, trashPath));
    QStringList diagnostics;
    QVERIFY2(artifacts.recoverStaging(&storage, &diagnostics, &error), qPrintable(error));
    QVERIFY(QDir(committedPath).exists());
    QVERIFY(!QFileInfo::exists(trashPath));

    const aitrain::ArtifactDiscardResult discarded =
        artifacts.discardCommitted(artifactId, &storage, &error);
    QCOMPARE(discarded.status, aitrain::ArtifactDiscardStatus::Discarded);
    bool exists = true;
    QVERIFY2(storage.artifactExists(artifactId, &exists, &error), qPrintable(error));
    QVERIFY(!exists);
    QVERIFY(!QFileInfo::exists(committedPath));
    QVERIFY(!QFileInfo::exists(trashPath));
}

QTEST_MAIN(ArtifactStoreTests)
#include "tst_artifact_store.moc"
