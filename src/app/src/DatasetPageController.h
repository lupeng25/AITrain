#pragma once

#include "MainWindowState.h"
#include "aitrain/domain/DomainTypes.h"
#include "aitrain/workflow/ProjectQueryService.h"

#include <QObject>
#include <QStringList>
#include <QVector>

class DatasetWorkspacePage;
class TaskRuntimeController;
class DatasetCatalogPresenter;
struct TaskViewState;
namespace aitrain {
class ProjectQueryService;
}

class DatasetPageController final : public QObject
{
    Q_OBJECT

public:
    explicit DatasetPageController(
        const aitrain::ProjectQueryService* queryService,
        TaskRuntimeController* taskRuntime,
        QObject* parent = nullptr);

    void attach(DatasetWorkspacePage* page);
    DatasetWorkbenchState& state();
    const DatasetWorkbenchState& state() const;
    void reset();
    void invalidateAsyncPreviews();
    void setProjectContext(bool projectOpen, const QString& projectRoot);
    void setWorkerExecutable(const QString& executable);
    void applyTaskViewState(const TaskViewState& state);
    void openQualityReport(bool repairList);

public slots:
    void startConversion();
    void cancelConversion();
    void runDataQuality();
    void runSplit();
    void runSnapshotImport();
    void refreshCatalog();
    void browseDataset();
    void browseConversionInput();
    void updateConversionTargets();
    void refreshConversionDefaults();
    void setConversionRunning(bool running);
    void createAnnotationSession();
    void syncAnnotationSession();
    void browseSampleReview();
    void loadSampleReview();
    void refreshSampleReview();
    void openSelectedReviewSample();
    void loadSnapshots(bool append = false);
    void selectSnapshot(int index);
    void loadSamples(bool append = false);
    void previewSample(int row);

signals:
    void taskStarted(const QString& taskId, const QString& workflowKind);
    void statusChanged(const QString& text);
    void selectionChanged();
    void repairLoopChanged(
        const QString& summary, const QVector<QStringList>& rows);

private:
    void startFormatProbe(const QString& path, bool conversionSource);
    void applyFormatProbe(const QString& path, const QString& detectedFormat,
        bool conversionSource, quint64 generation);
    void loadSampleReviewCandidate(const aitrain::ArtifactId& artifactId,
        const QStringList& candidates, int index, quint64 generation,
        const QString& lastError = QString());
    QJsonArray filteredSampleReviewRows() const;
    void appendConversionLog(const QString& text);
    void setConversionError(const QString& text);
    void renderCatalog();
    void selectCatalogRow();
    void loadQualityReport(const QString& taskId);
    void renderQualityReport(const QJsonObject& report);

    TaskRuntimeController* taskRuntime_ = nullptr;
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    DatasetCatalogPresenter* catalogPresenter_ = nullptr;
    DatasetWorkspacePage* page_ = nullptr;
    DatasetWorkbenchState state_;
    bool projectOpen_ = false;
    QString projectRoot_;
    QString catalogSearch_;
    QString workerExecutable_;
    quint64 formatProbeGeneration_ = 0;
    quint64 sampleReviewGeneration_ = 0;
    quint64 previewGeneration_ = 0;
    quint64 qualityGeneration_ = 0;
    QVector<QString> catalogCursors_{QString()};
    QVector<aitrain::DatasetSnapshotReadModel> snapshots_;
    QString snapshotCursor_;
    QString sampleCursor_;
    QString activeTaskId_;
    QString activeKind_;
    QString activeSnapshotId_;
    QString pendingTargetId_;
    QString selectAfterRefresh_;
    QHash<QString, qint64> sampleCounts_;
};
