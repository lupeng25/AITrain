#include "DatasetCatalogPresenter.h"

#include "aitrain/storage/ProjectStore.h"

#include <QDir>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

namespace {

aitrain::TaskSnapshot createTask(aitrain::ProjectStore* storage, QString* error)
{
    aitrain::TaskSnapshot task;
    task.id = aitrain::TaskId::create();
    task.requestId = aitrain::RequestId::create();
    task.capabilityId = QStringLiteral("dataset.catalog.presenter.test");
    task.taskType = QStringLiteral("dataset_snapshot");
    if (!storage || !storage->createTask(task, error)) return {};
    return task;
}

aitrain::DatasetSnapshotRecord registerSnapshot(
    aitrain::ProjectStore* storage,
    const QString& projectRoot,
    const aitrain::TaskSnapshot& task,
    const aitrain::DatasetId& datasetId,
    QChar hashChar,
    int fileCount,
    QString* error)
{
    const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
    const QString manifestSha256(64, hashChar);
    if (!storage->recordArtifactWithFiles(artifactId, task.id,
            QStringLiteral("dataset_snapshot"),
            {{QStringLiteral("dataset_snapshot.json"), manifestSha256, 64}},
            QDateTime::currentDateTimeUtc(), error)) {
        return {};
    }

    aitrain::DatasetSnapshotRecord snapshot;
    snapshot.datasetId = datasetId;
    snapshot.id = aitrain::SnapshotId::create();
    snapshot.taskId = task.id;
    snapshot.artifactId = artifactId;
    snapshot.rootPath = QDir(projectRoot).filePath(
        QStringLiteral(".aitrain/artifacts/committed/%1").arg(artifactId.toString()));
    snapshot.datasetFormat = QStringLiteral("yolo_detection");
    snapshot.driverId = QStringLiteral("yolo_detection");
    snapshot.driverVersion = QStringLiteral("2.0");
    snapshot.rootHash = QString(64, QChar(hashChar.unicode() + 1));
    snapshot.manifestSha256 = manifestSha256;
    snapshot.fileCount = fileCount;
    snapshot.totalBytes = fileCount * 10;
    snapshot.createdAt = QDateTime::currentDateTimeUtc().addSecs(fileCount);
    if (!storage->registerDatasetSnapshot(&snapshot, error)) return {};
    return snapshot;
}

} // namespace

class DatasetCatalogPresenterTests final : public QObject {
    Q_OBJECT

private slots:
    void searchesBeyondFirstPageAndBindsCursorToQuery();
    void exposesNewestSnapshotIdentityWithoutPaths();
    void missingQueryServiceFailsAndClearsRows();
    void versionAndArtifactCursorsStayBoundToSelectedObjects();
};

void DatasetCatalogPresenterTests::exposesNewestSnapshotIdentityWithoutPaths()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.createProject(projectRoot, &error), qPrintable(error));

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(projectRoot).filePath(QStringLiteral(".aitrain/project.sqlite")), &error),
        qPrintable(error));
    storage.setArtifactStoreRoot(QDir(projectRoot).filePath(QStringLiteral(".aitrain/artifacts")));
    const auto task = createTask(&storage, &error);
    QVERIFY2(task.id.isValid(), qPrintable(error));
    const auto datasetId = aitrain::DatasetId::create();
    const auto older = registerSnapshot(&storage, projectRoot, task, datasetId,
        QLatin1Char('1'), 1, &error);
    QVERIFY2(older.id.isValid(), qPrintable(error));
    const auto newer = registerSnapshot(&storage, projectRoot, task, datasetId,
        QLatin1Char('3'), 2, &error);
    QVERIFY2(newer.id.isValid(), qPrintable(error));

    aitrain::ProjectQueryService query(&workspace);
    DatasetCatalogPresenter presenter(&query);
    QSignalSpy changed(&presenter, &DatasetCatalogPresenter::datasetsChanged);
    QVERIFY2(presenter.refresh(), qPrintable(presenter.lastError()));
    QCOMPARE(presenter.objectName(), QStringLiteral("DatasetCatalogPresenter"));
    QCOMPARE(presenter.datasetCount(), 1);
    QCOMPARE(changed.count(), 1);
    const DatasetCatalogListItem& row = presenter.datasets().first();
    QCOMPARE(row.datasetId, datasetId.toString());
    QCOMPARE(row.datasetFormat, QStringLiteral("yolo_detection"));
    QCOMPARE(row.versionCount, qint64(2));
    QCOMPARE(row.snapshotCount, qint64(2));
    QCOMPARE(row.latestVersionId, newer.datasetVersionId.toString());
    QCOMPARE(row.latestSnapshotId, newer.id.toString());
    QCOMPARE(row.latestArtifactId, newer.artifactId.toString());
    QCOMPARE(row.latestFileCount, qsizetype(2));
    QVERIFY(!row.latestRootHash.isEmpty());
    const QString visible = QStringList{
        row.datasetId, row.datasetFormat, row.latestVersionId,
        row.latestSnapshotId, row.latestArtifactId, row.latestRootHash,
        row.latestCreatedAt}.join(QLatin1Char('\n'));
    QVERIFY(!visible.contains(projectRoot));
    QVERIFY(!visible.contains(QStringLiteral("rootPath")));
}

void DatasetCatalogPresenterTests::missingQueryServiceFailsAndClearsRows()
{
    DatasetCatalogPresenter presenter(nullptr);
    QSignalSpy failed(&presenter, &DatasetCatalogPresenter::queryFailed);
    QVERIFY(!presenter.refresh());
    QCOMPARE(presenter.datasetCount(), 0);
    QVERIFY(!presenter.lastError().isEmpty());
    QCOMPARE(failed.count(), 1);
}

void DatasetCatalogPresenterTests::versionAndArtifactCursorsStayBoundToSelectedObjects()
{
    QTemporaryDir directory; QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace; QString error;
    const QString root = directory.filePath(QStringLiteral("project"));
    QVERIFY2(workspace.createProject(root, &error), qPrintable(error));
    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(root).filePath(QStringLiteral(".aitrain/project.sqlite")), &error), qPrintable(error));
    storage.setArtifactStoreRoot(QDir(root).filePath(QStringLiteral(".aitrain/artifacts")));
    const auto task = createTask(&storage, &error); QVERIFY2(task.id.isValid(), qPrintable(error));
    const auto dataset = aitrain::DatasetId::create();
    const auto first = registerSnapshot(&storage, root, task, dataset, QLatin1Char('1'), 1, &error);
    const auto second = registerSnapshot(&storage, root, task, dataset, QLatin1Char('3'), 2, &error);
    QVERIFY2(first.id.isValid() && second.id.isValid(), qPrintable(error));
    aitrain::ProjectQueryService query(&workspace);
    const auto page = query.datasetSnapshots(dataset, {1, {}}, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error)); QVERIFY(page.hasMore); QCOMPARE(page.items.size(), 1);
    QCOMPARE(page.items.first().snapshotId, second.id);
    const auto next = query.datasetSnapshots(dataset, {1, page.nextCursor}, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error)); QVERIFY(!next.hasMore); QCOMPARE(next.items.first().snapshotId, first.id);
    query.datasetSnapshots(aitrain::DatasetId::create(), {1, page.nextCursor}, &error); QVERIFY(!error.isEmpty());
    for (int i = 0; i < 2; ++i) QVERIFY2(storage.recordArtifactWithFiles(aitrain::ArtifactId::create(), task.id, QStringLiteral("a|b"),
        {{QStringLiteral("report.json"), QString(64, QLatin1Char('a')), 2}}, QDateTime::currentDateTimeUtc().addSecs(i), &error), qPrintable(error));
    const auto artifacts = query.artifactCatalog({QStringLiteral("a|b")}, {1, {}}, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error)); QVERIFY(artifacts.hasMore); QCOMPARE(artifacts.items.size(), 1);
    query.artifactCatalog({QStringLiteral("a"), QStringLiteral("b")}, {1, artifacts.nextCursor}, &error);
    QVERIFY(!error.isEmpty());
}

void DatasetCatalogPresenterTests::searchesBeyondFirstPageAndBindsCursorToQuery()
{
    QTemporaryDir directory; aitrain::ProjectWorkspace workspace; QString error;
    const QString root = directory.filePath(QStringLiteral("search"));
    QVERIFY2(workspace.createProject(root, &error), qPrintable(error));
    aitrain::ProjectStore storage; QVERIFY(storage.open(QDir(root).filePath(QStringLiteral(".aitrain/project.sqlite")), &error));
    const QString needle = QStringLiteral("零件 100%_\"A\"");
    storage.setArtifactStoreRoot(QDir(root).filePath(QStringLiteral(".aitrain/artifacts")));
    for (int i = 0; i < 65; ++i) {
        const auto task = createTask(&storage, &error); QVERIFY(task.id.isValid());
        const auto snapshot = registerSnapshot(&storage, root, task, aitrain::DatasetId::create(), QLatin1Char('1'), i + 1, &error);
        QVERIFY2(snapshot.id.isValid(), qPrintable(error));
        aitrain::WorkflowRunSnapshot workflow; workflow.id = aitrain::WorkflowRunId::create(); workflow.taskId = task.id; workflow.templateId = QStringLiteral("dataset_snapshot_import");
        aitrain::WorkflowStepSnapshot step; step.id = aitrain::WorkflowStepId::create(); step.workflowRunId = workflow.id; step.ordinal = 0; step.kind = QStringLiteral("SnapshotImport"); step.backend = QStringLiteral("dataset_snapshot_import");
        step.parameterSummary = {{QStringLiteral("targetDatasetName"), i < 2 ? needle : QStringLiteral("背景数据-%1").arg(i)}};
        QVERIFY2(storage.createWorkflowRun(workflow, {step}, &error), qPrintable(error));
    }
    aitrain::ProjectQueryService query(&workspace);
    const auto initial = query.datasetCatalog({50, {}}, &error); QCOMPARE(initial.items.size(), 50);
    for (const auto& item : initial.items) QVERIFY(item.displayName != needle);
    const aitrain::CatalogFilter filter{needle, {}, {}};
    const auto first = query.datasetCatalog({1, {}}, &error, filter); QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(first.items.size(), 1); QCOMPARE(first.items.first().displayName, needle); QVERIFY(first.hasMore);
    const auto second = query.datasetCatalog({1, first.nextCursor}, &error, filter); QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(second.items.size(), 1); QVERIFY(second.items.first().datasetId != first.items.first().datasetId); QVERIFY(!second.hasMore);
    query.datasetCatalog({1, first.nextCursor}, &error, {QStringLiteral("背景"), {}, {}}); QVERIFY(error.contains(QStringLiteral("InvalidPageCursor")));
    QCOMPARE(query.datasetCatalog({50, {}}, &error, {QStringLiteral("100%_"), {}, {}}).items.size(), 2);
    QCOMPARE(query.datasetCatalog({50, {}}, &error, {QStringLiteral("不存在"), {}, {}}).items.size(), 0);
}

QTEST_GUILESS_MAIN(DatasetCatalogPresenterTests)

#include "tst_dataset_catalog_presenter.moc"
