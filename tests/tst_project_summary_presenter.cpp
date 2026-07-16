#include "ProjectSummaryPresenter.h"

#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/storage/ProjectStore.h"

#include <QDir>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

class ProjectSummaryPresenterTests : public QObject {
    Q_OBJECT

private slots:
    void mapsOnlyPersistedSummaryFacts();
    void unavailableWorkspaceClearsReadModel();
};

void ProjectSummaryPresenterTests::mapsOnlyPersistedSummaryFacts()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"),
        QStringLiteral("detection"), &task, &error), qPrintable(error));

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(
        QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
    const QVector<aitrain::ArtifactFileSnapshot> files{
        {QStringLiteral("reports/summary.json"), QString(64, QLatin1Char('b')), 64}};
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, taskId,
        QStringLiteral("summary_report"), files,
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::ProjectQueryService query(&workspace);
    ProjectSummaryPresenter presenter(&query);
    QCOMPARE(presenter.objectName(), QStringLiteral("ProjectSummaryPresenter"));
    QSignalSpy changedSpy(&presenter, &ProjectSummaryPresenter::summaryChanged);

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

void ProjectSummaryPresenterTests::unavailableWorkspaceClearsReadModel()
{
    aitrain::ProjectQueryService query(nullptr);
    ProjectSummaryPresenter presenter(&query);
    QSignalSpy failureSpy(&presenter, &ProjectSummaryPresenter::queryFailed);
    QVERIFY(!presenter.refresh());
    QVERIFY(!presenter.available());
    QCOMPARE(presenter.taskCount(), qint64(0));
    QCOMPARE(presenter.committedArtifactCount(), qint64(0));
    QCOMPARE(failureSpy.count(), 1);
    QVERIFY(!presenter.lastError().isEmpty());

    presenter.clear();
    QVERIFY(presenter.lastError().isEmpty());
}

QTEST_GUILESS_MAIN(ProjectSummaryPresenterTests)
#include "tst_project_summary_presenter.moc"
