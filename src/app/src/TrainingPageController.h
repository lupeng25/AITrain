#pragma once

#include <QObject>
#include <QJsonObject>

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
};

class TrainingPageController final : public QObject
{
    Q_OBJECT

public:
    explicit TrainingPageController(TaskRuntimeController* taskRuntime,
        QObject* parent = nullptr);

    void attach(TrainingWorkspacePage* page);
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
    void start();
    void refreshTaskTypes(const QString& preferredTask = QString());
    void refreshModelPresets();
    void refreshSummary();
    QJsonObject collectArguments(const QString& prefix) const;
    QJsonObject collectExportArguments() const;

    TaskRuntimeController* taskRuntime_ = nullptr;
    TrainingWorkspacePage* page_ = nullptr;
    TrainingDatasetBinding binding_;
    bool projectOpen_ = false;
    QString projectRoot_;
    QString workerExecutable_;
    qint64 liveMetricSequence_ = 0;
    qint64 liveArtifactSequence_ = 0;
};
