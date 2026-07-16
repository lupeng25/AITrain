#include "ProjectSummaryPresenterV2.h"

#include "aitrain/v2/ProjectWorkspaceV2.h"
#include "aitrain/v2/StorageV2.h"

#include <QDir>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

class ProjectSummaryPresenterV2Tests : public QObject {
    Q_OBJECT

private slots:
    void mapsOnlyPersistedV2SummaryFacts();
    void unavailableWorkspaceClearsReadModel();
};

void ProjectSummaryPresenterV2Tests::mapsOnlyPersistedV2SummaryFacts()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"),
        QStringLiteral("detection"), &task, &error), qPrintable(error));

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(
        QStringLiteral("project-v2.sqlite")), &error), qPrintable(error));
    const aitrain::v2::ArtifactId artifactId = aitrain::v2::ArtifactId::create();
    const QVector<aitrain::v2::ArtifactFileSnapshot> files{
        {QStringLiteral("reports/summary.json"), QString(64, QLatin1Char('b')), 64}};
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, taskId,
        QStringLiteral("summary_report_v2"), files,
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    ProjectSummaryPresenterV2 presenter(&query);
    QCOMPARE(presenter.objectName(), QStringLiteral("ProjectSummaryPresenterV2"));
    QSignalSpy changedSpy(&presenter, &ProjectSummaryPresenterV2::summaryChanged);

    QVERIFY(presenter.refresh());
    QCOMPARE(changedSpy.count(), 1);
    QVERIFY(presenter.property("available").toBool());
    QCOMPARE(presenter.property("taskCount").toLongLong(), qint64(1));
    QCOMPARE(presenter.property("committedArtifactCount").toLongLong(), qint64(1));
    QCOMPARE(presenter.viewModel().activeTaskCount, qint64(1));
    QCOMPARE(presenter.viewModel().datasetCount, qint64(0));
    QCOMPARE(presenter.viewModel().modelPackageCount, qint64(0));
    QVERIFY(presenter.lastError().isEmpty());
}

void ProjectSummaryPresenterV2Tests::unavailableWorkspaceClearsReadModel()
{
    aitrain::v2::ProjectQueryServiceV2 query(nullptr);
    ProjectSummaryPresenterV2 presenter(&query);
    QSignalSpy failureSpy(&presenter, &ProjectSummaryPresenterV2::queryFailed);
    QVERIFY(!presenter.refresh());
    QVERIFY(!presenter.available());
    QCOMPARE(presenter.taskCount(), qint64(0));
    QCOMPARE(presenter.committedArtifactCount(), qint64(0));
    QCOMPARE(failureSpy.count(), 1);
    QVERIFY(!presenter.lastError().isEmpty());

    presenter.clear();
    QVERIFY(presenter.lastError().isEmpty());
}

QTEST_GUILESS_MAIN(ProjectSummaryPresenterV2Tests)
#include "tst_project_summary_presenter_v2.moc"
