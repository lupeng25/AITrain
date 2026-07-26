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
    void exposesNewestSnapshotIdentityWithoutPaths();
    void missingQueryServiceFailsAndClearsRows();
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

QTEST_GUILESS_MAIN(DatasetCatalogPresenterTests)

#include "tst_dataset_catalog_presenter.moc"
