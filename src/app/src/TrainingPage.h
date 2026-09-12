#pragma once

#include <QJsonObject>
#include "WorkbenchWidgets.h"

class QLabel;
struct TaskArtifactView;
struct TaskMetricView;

struct TrainingFormData final
{
    QString capabilityId;
    QString taskType;
    QString backendId;
    QString modelPreset;
    int epochs = 0;
    int batchSize = 0;
    int imageSize = 0;
    int gridSize = 0;
    bool horizontalFlip = false;
    bool colorJitter = false;
    QJsonObject backendArguments;
    QJsonObject exportArguments;
};

class TrainingWorkspacePage final : public aitrain_app::WorkspaceViewHost
{
    Q_OBJECT

public:
    enum View { Catalog, Configuration, Monitor, Advanced, FullLog, Artifacts };
    explicit TrainingWorkspacePage(QWidget* parent = nullptr);
    void buildLayout();
    QTableWidget* historyTable = nullptr;
    QLabel* historyStatus = nullptr;
    QPushButton* moreHistory = nullptr;
    QPushButton* cancelTaskButton = nullptr;
    QPushButton* modelsButton = nullptr;

    TrainingFormData formData() const;
    void setDatasetSummary(const QString& text);
    void setBackendSummary(const QString& text);
    void setRunSummary(const QString& text);
    void setDatasetSummaryToolTip(const QString& text);
    void setRunSummaryToolTip(const QString& text);
    void setBackendPanels(const QString& backendId);
    void setProgress(int progress);
    void setPhase(const QString& text);
    void setLiveValue(const QString& objectName, const QString& value);
    void addMetric(const QString& name, double value);
    void updateArtifact(const QString& artifactId, const QString& kind,
        const QString& relativePath);
    void appendLog(const QString& text);
    void resetRuntimeProjection();

protected:
    void keyPressEvent(QKeyEvent* event) override;

signals:
    void selectDatasetRequested();
    void refreshHistoryRequested();
    void moreHistoryRequested();
    void historyRequested(const QString& taskId);
    void advancedRequested();
    void applyAdvancedRequested();
    void cancelAdvancedRequested();
    void openTaskRequested(const QString& taskId);
    void currentTaskRequested();
    void logRequested();
    void configurationRequested();
    void copyConfigurationRequested();
    void modelsRequested();
    void modelRequested(const QString& modelPackageId);
    void startRequested();
    void cancelRequested();
};
