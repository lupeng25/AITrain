#pragma once

#include <QObject>
#include <QJsonObject>
#include <QMap>
#include <QVariant>
#include <QPointer>
class QTimer;

namespace aitrain { class ProjectQueryService; }
class TaskArtifactPresenter;

class TaskRuntimeController;
class TrainingWorkspacePage;
struct TaskViewState;

struct TrainingDatasetBinding final
{
    QString datasetId;
    QString datasetVersionId;
    QString snapshotId;
    QString snapshotArtifactId;
    QString datasetFormat;
    QString displayName;
    QString deploymentSampleRelativePath;
};

class TrainingPageController final : public QObject
{
    Q_OBJECT

public:
    explicit TrainingPageController(TaskRuntimeController* taskRuntime,
        QObject* parent = nullptr);

    void attach(TrainingWorkspacePage* page);
    void setQueryService(const aitrain::ProjectQueryService* query);
    void refreshHistory();
    void setProjectContext(bool projectOpen, const QString& projectRoot);
    void setWorkerExecutable(const QString& executable);
    void setDatasetBinding(const TrainingDatasetBinding& binding);
    void refreshCapabilities();
    void refreshDefaults();
    QString currentTaskType() const;
    void appendLog(const QString& text);
    void applyTaskViewState(const TaskViewState& state);

signals:
    void taskStarted(const QString& taskId, const QString& workflowKind);
    void runStarted(const QString& taskId);

private:
    void renderHistory();
    void captureParameterDefaults();
    QJsonObject historyParameters(const QString& taskId);
    bool historyBinding(const QJsonObject& configuration, TrainingDatasetBinding* binding, QString* versionLabel);
    void showHistoricalConfiguration();
    void copyHistoricalConfiguration();
    void refreshResultModel();
    void openHistory(const QString& taskId);
    void chooseDataset();
    void synchronizeBackend();
    bool validateParameters();
    void editAdvanced();
    void cancelAdvanced();
    void start();
    void refreshTaskTypes(const QString& preferredTask = QString());
    void refreshModelPresets();
    void refreshSummary();
    QJsonObject collectArguments(const QString& prefix) const;
    QJsonObject collectExportArguments() const;

    TaskRuntimeController* taskRuntime_ = nullptr;
    void initializeDraftPersistence();
    void scheduleDraftSave();
    void saveDraft();
    void restoreDraft();
    void discardDraft();
    void resetDraftControls();
    QJsonObject draftControls() const;
    void applyDraftControls(const QJsonObject& controls);
    QPointer<TrainingWorkspacePage> page_;
    QString draftProjectId_;
    QTimer* draftTimer_ = nullptr;
    bool restoringDraft_ = false;
    bool draftDirty_ = false;
    QJsonObject initialDraftControls_;
    TrainingDatasetBinding binding_;
    bool projectOpen_ = false;
    QString projectRoot_;
    QString workerExecutable_;
    QString modelPresetBackend_;
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    TaskArtifactPresenter* history_ = nullptr;
    QString historySearch_;
    QString selectedTaskId_;
    QString activeTaskId_;
    QString resultModelId_;
    QMap<QString, QVariant> advancedBackup_;
    QMap<QString, QVariant> defaultParameterControls_;
    QMap<QString, QJsonObject> historyConfigurations_;
    qint64 liveMetricSequence_ = 0;
    qint64 liveArtifactSequence_ = 0;
};
