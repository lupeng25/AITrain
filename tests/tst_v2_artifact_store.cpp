#include "aitrain/v2/ArtifactStoreV2.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QTest>

namespace {

aitrain::v2::TaskSnapshot createTask(aitrain::v2::StorageV2* storage)
{
    aitrain::v2::TaskSnapshot task;
    task.id = aitrain::v2::TaskId::create();
    task.requestId = aitrain::v2::RequestId::create();
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

aitrain::v2::WorkflowRunSnapshotV2 createSealedEvidenceWorkflow(
    aitrain::v2::StorageV2* storage,
    const aitrain::v2::TaskSnapshot& task,
    QString* error)
{
    aitrain::v2::WorkflowRunSnapshotV2 workflow;
    workflow.id = aitrain::v2::WorkflowRunId::create();
    workflow.taskId = task.id;
    workflow.templateId = QStringLiteral("artifact-recovery-evidence-v2");
    workflow.terminalPolicy = aitrain::v2::WorkflowTerminalPolicyV2::EvidenceRequired;
    aitrain::v2::WorkflowStepSnapshotV2 step;
    step.id = aitrain::v2::WorkflowStepId::create();
    step.workflowRunId = workflow.id;
    step.ordinal = 0;
    step.kind = QStringLiteral("ValidateDataset");
    step.backend = QStringLiteral("fixture");
    const aitrain::v2::Failure failure{aitrain::v2::FailureCode::InvalidDataset, QStringLiteral("测试失败。"),
        QStringLiteral("修复测试输入。"), QDateTime::currentDateTimeUtc()};
    if (!storage->createWorkflowRun(workflow, {step}, error)
        || !storage->transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Pending,
            aitrain::v2::WorkflowStepState::Running, {}, {}, error)
        || !storage->transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Running,
            aitrain::v2::WorkflowStepState::Failed, {}, failure, error)
        || !storage->sealWorkflowTerminalization(workflow.id, aitrain::v2::TaskState::Failed,
            failure, QDateTime::currentDateTimeUtc(), error)) return {};
    return workflow;
}

} // namespace

class V2ArtifactStoreTests : public QObject {
    Q_OBJECT

private slots:
    void commitsVerifiedArtifactAtomically();
    void cancellationDuringCommitKeepsOnlyAbortableStaging();
    void abortRemovesStagingAndEmptyStagingCannotCommit();
    void recoveryPreservesActiveAndCleansAbandonedStaging();
    void recoveryCompletesEvidenceCommitAfterDirectoryRename();
    void recoveryCleansJournalAfterDatabaseCommit();
};

void V2ArtifactStoreTests::commitsVerifiedArtifactAtomically()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = createTask(&storage);
    QVERIFY(task.id.isValid());
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::ArtifactId artifactId;
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

void V2ArtifactStoreTests::cancellationDuringCommitKeepsOnlyAbortableStaging()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = createTask(&storage);
    QVERIFY(task.id.isValid());
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::ArtifactId artifactId;
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

void V2ArtifactStoreTests::abortRemovesStagingAndEmptyStagingCannotCommit()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = createTask(&storage);
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(artifacts.begin(task.id, QStringLiteral("report"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(!artifacts.commit(artifactId, task.id, QStringLiteral("report"), stagingPath, &storage, nullptr, &error));
    QVERIFY(error.contains(QStringLiteral("不能为空")));
    QVERIFY2(artifacts.abort(stagingPath, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(stagingPath));
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
}

void V2ArtifactStoreTests::recoveryPreservesActiveAndCleansAbandonedStaging()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot activeTask = createTask(&storage);
    const aitrain::v2::TaskSnapshot failedTask = createTask(&storage);
    QVERIFY2(storage.transitionTask(failedTask.id, aitrain::v2::TaskState::Created, aitrain::v2::TaskState::Failed,
        {aitrain::v2::FailureCode::ProcessCrashed, QStringLiteral("fixture")}, &error), qPrintable(error));

    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::ArtifactId activeArtifactId;
    aitrain::v2::ArtifactId failedArtifactId;
    aitrain::v2::ArtifactId orphanArtifactId;
    aitrain::v2::ArtifactId metadataOnlyArtifactId;
    QString activeStaging;
    QString failedStaging;
    QString orphanStaging;
    QString metadataOnlyStaging;
    QVERIFY2(artifacts.begin(activeTask.id, QStringLiteral("report"), &activeArtifactId, &activeStaging, &error), qPrintable(error));
    QVERIFY2(artifacts.begin(failedTask.id, QStringLiteral("report"), &failedArtifactId, &failedStaging, &error), qPrintable(error));
    QVERIFY2(artifacts.begin(aitrain::v2::TaskId::create(), QStringLiteral("report"), &orphanArtifactId, &orphanStaging, &error), qPrintable(error));
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

void V2ArtifactStoreTests::recoveryCompletesEvidenceCommitAfterDirectoryRename()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = createTask(&storage);
    const auto workflow = createSealedEvidenceWorkflow(&storage, task, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));
    const QString storeRoot = directory.filePath(QStringLiteral("store"));
    aitrain::v2::ArtifactStoreV2 interrupted(storeRoot,
        [](aitrain::v2::ArtifactCommitFailPointV2 point) {
            return point == aitrain::v2::ArtifactCommitFailPointV2::AfterDirectoryRenameBeforeDatabase;
        });
    aitrain::v2::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(interrupted.begin(task.id, QStringLiteral("evidence_bundle_v2"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(stagingPath).filePath(QStringLiteral("evidence.json")), QByteArray("{}")));
    QVERIFY(!interrupted.commit(artifactId, task.id, QStringLiteral("evidence_bundle_v2"), stagingPath,
        &storage, nullptr, &error, {}, nullptr, workflow.id));
    bool exists = true;
    QVERIFY2(storage.artifactExists(artifactId, &exists, &error), qPrintable(error));
    QVERIFY(!exists);
    QVERIFY(QFileInfo::exists(QDir(storeRoot).filePath(QStringLiteral("artifacts/%1").arg(artifactId.toString()))));

    aitrain::v2::ArtifactStoreV2 recovered(storeRoot);
    QStringList diagnostics;
    error.clear();
    QVERIFY2(recovered.recoverStaging(&storage, &diagnostics, &error), qPrintable(error));
    QVERIFY2(storage.artifactExists(artifactId, &exists, &error), qPrintable(error));
    QVERIFY(exists);
    aitrain::v2::WorkflowTerminalizationSnapshotV2 terminalization;
    QVERIFY2(storage.workflowTerminalization(workflow.id, &terminalization, &error), qPrintable(error));
    QCOMPARE(terminalization.state, aitrain::v2::WorkflowTerminalizationStateV2::EvidenceAttached);
    QCOMPARE(terminalization.evidenceArtifactId, artifactId);
    QVERIFY(!QFileInfo::exists(QDir(storeRoot).filePath(QStringLiteral(".staging-meta/%1.json").arg(artifactId.toString()))));
}

void V2ArtifactStoreTests::recoveryCleansJournalAfterDatabaseCommit()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = createTask(&storage);
    const auto workflow = createSealedEvidenceWorkflow(&storage, task, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));
    const QString storeRoot = directory.filePath(QStringLiteral("store"));
    aitrain::v2::ArtifactStoreV2 interrupted(storeRoot,
        [](aitrain::v2::ArtifactCommitFailPointV2 point) {
            return point == aitrain::v2::ArtifactCommitFailPointV2::AfterDatabaseBeforeJournalRemoval;
        });
    aitrain::v2::ArtifactId artifactId;
    QString stagingPath;
    QVERIFY2(interrupted.begin(task.id, QStringLiteral("evidence_bundle_v2"), &artifactId, &stagingPath, &error), qPrintable(error));
    QVERIFY(writeFile(QDir(stagingPath).filePath(QStringLiteral("evidence.json")), QByteArray("{}")));
    QVERIFY(!interrupted.commit(artifactId, task.id, QStringLiteral("evidence_bundle_v2"), stagingPath,
        &storage, nullptr, &error, {}, nullptr, workflow.id));
    bool exists = false;
    QVERIFY2(storage.artifactExists(artifactId, &exists, &error), qPrintable(error));
    QVERIFY(exists);
    const QString journalPath = QDir(storeRoot).filePath(QStringLiteral(".staging-meta/%1.json").arg(artifactId.toString()));
    QVERIFY(QFileInfo::exists(journalPath));

    aitrain::v2::ArtifactStoreV2 recovered(storeRoot);
    QStringList diagnostics;
    error.clear();
    QVERIFY2(recovered.recoverStaging(&storage, &diagnostics, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(journalPath));
    QCOMPARE(storage.artifactCount(task.id, &error), 1);
    aitrain::v2::WorkflowTerminalizationSnapshotV2 terminalization;
    QVERIFY2(storage.workflowTerminalization(workflow.id, &terminalization, &error), qPrintable(error));
    QCOMPARE(terminalization.state, aitrain::v2::WorkflowTerminalizationStateV2::EvidenceAttached);
}

QTEST_MAIN(V2ArtifactStoreTests)
#include "tst_v2_artifact_store.moc"
