#pragma once

#include "MetricsWidget.h"
#include "Sidebar.h"
#include "StatusPill.h"
#include "WorkerClient.h"
#include "aitrain/core/PluginManager.h"
#include "aitrain/core/ProjectRepository.h"

#include <QComboBox>
#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QVector>

class InfoPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(const QString& licenseOwner = QString(), const QString& licenseExpiry = QString(), QWidget* parent = nullptr);

private slots:
    void createProject();
    void browseDataset();
    void validateDataset();
    void splitDataset();
    void curateDataset();
    void createDatasetSnapshot();
    void startTraining();
    void startModelExport();
    void startInference();
    void cancelSelectedTask();
    void runEnvironmentCheck();
    void handleWorkerMessage(const QString& type, const QJsonObject& payload);
    void refreshPlugins();
    void showPage(int pageIndex, const QString& title);
    void updateSelectedTaskDetails();
    void updateArtifactPreviewFromSelection();
    void openSelectedArtifactDirectory();
    void copySelectedArtifactPath();
    void useSelectedArtifactForInference();
    void useSelectedArtifactForExport();
    void registerSelectedArtifactAsModelVersion();
    void evaluateSelectedArtifact();
    void benchmarkSelectedArtifact();
    void generateDeliveryReportFromSelectedArtifact();
    void runLocalPipelinePlanFromCurrentDataset();
    void reproduceSelectedTrainingTask();
    void refreshModelRegistry();

private:
    enum PageIndex {
        DashboardPage = 0,
        ProjectPage,
        DatasetPage,
        TrainingPage,
        TaskQueuePage,
        ModelRegistryPage,
        ConversionPage,
        InferencePage,
        PluginsPage,
        EnvironmentPage
    };

    QWidget* buildTopBar();
    QWidget* buildDashboardPage();
    QWidget* buildProjectPage();
    QWidget* buildDatasetPage();
    QWidget* buildTrainingPage();
    QWidget* buildTaskQueuePage();
    QWidget* buildModelRegistryPage();
    QWidget* buildConversionPage();
    QWidget* buildInferencePage();
    QWidget* buildPluginsPage();
    QWidget* buildEnvironmentPage();

    InfoPanel* createMetricCard(const QString& label, const QString& value, const QString& caption);
    QString pageCaption(int pageIndex) const;
    QString workerExecutablePath() const;
    QStringList pluginSearchPaths() const;
    void ensureProjectSubdirs(const QString& rootPath);
    void appendLog(const QString& text);
    void loadPluginCombos();
    QString currentDatasetFormat() const;
    QString currentTaskType() const;
    QString currentTaskKindFilter() const;
    QString currentTaskStateFilter() const;
    void updateRecentTasks();
    void updateDatasetList();
    void updateTaskTable(QTableWidget* table, const QVector<aitrain::TaskRecord>& tasks);
    void updateHeaderState();
    void updateEnvironmentTable(const QJsonObject& payload);
    void updateDatasetValidationResult(const QJsonObject& payload);
    void updateDatasetSplitResult(const QJsonObject& payload);
    struct PendingTrainingTask {
        QString taskId;
        aitrain::TrainingRequest request;
        bool needsSnapshot = false;
        int datasetId = 0;
        QString datasetFormat;
    };
    void startQueuedTraining(const QString& taskId, const aitrain::TrainingRequest& request);
    void startNextQueuedTask();
    void startSnapshotForQueuedTraining(const PendingTrainingTask& pending);
    void configureTable(QTableWidget* table) const;
    void updateDashboardSummary();
    void updateProjectSummary();
    void updatePluginSummary();
    void updateEnvironmentSummary();
    void updateTrainingSelectionSummary();
    void refreshTrainingDefaults();
    void updateAnnotationToolStatus();
    void refreshAfterAnnotation();
    void applyTaskFilters();
    void updateModelRegistry();
    bool attachLatestSnapshotToRequest(aitrain::TrainingRequest& request, int datasetId, QString* error);
    int recordExperimentRunForRequest(const aitrain::TrainingRequest& request, int datasetId, QString* error);
    void updateExperimentRunSummary(const QString& taskId);
    QString createRepositoryTask(aitrain::TaskKind kind, const QString& taskType, const QString& pluginId, const QString& workDir, const QString& message, const QString& requestedTaskId = {});
    QString selectedTaskId() const;
    QString selectedArtifactPath() const;
    void previewArtifactPath(const QString& path);

    aitrain::PluginManager pluginManager_;
    aitrain::ProjectRepository repository_;
    WorkerClient worker_;

    QString currentProjectPath_;
    QString currentProjectName_;
    QString currentTaskId_;
    QString currentDatasetPath_;
    QString currentDatasetFormat_;
    bool currentDatasetValid_ = false;

    QVector<PendingTrainingTask> pendingTrainingTasks_;
    PendingTrainingTask activeSnapshotTrainingTask_;
    bool hasActiveSnapshotTrainingTask_ = false;

    Sidebar* sidebar_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    QLabel* pageTitle_ = nullptr;
    QLabel* pageCaption_ = nullptr;
    QLabel* headerProjectLabel_ = nullptr;
    StatusPill* workerPill_ = nullptr;
    StatusPill* pluginPill_ = nullptr;
    StatusPill* gpuPill_ = nullptr;
    StatusPill* licensePill_ = nullptr;
    QString licenseOwner_;
    QString licenseExpiry_;

    QLabel* projectLabel_ = nullptr;
    QLabel* gpuLabel_ = nullptr;
    QLabel* dashboardProjectValue_ = nullptr;
    QLabel* dashboardTaskValue_ = nullptr;
    QLabel* dashboardPluginValue_ = nullptr;
    QLabel* dashboardDatasetValue_ = nullptr;
    QLabel* dashboardModelValue_ = nullptr;
    QLabel* dashboardEnvironmentValue_ = nullptr;
    QLabel* dashboardNextStepLabel_ = nullptr;
    QLabel* projectConsoleStatusLabel_ = nullptr;
    QLabel* projectPathSummaryLabel_ = nullptr;
    QLabel* projectSqliteSummaryLabel_ = nullptr;
    QLabel* projectDatasetSummaryLabel_ = nullptr;
    QLabel* projectTaskSummaryLabel_ = nullptr;
    QLabel* projectExportSummaryLabel_ = nullptr;
    QLabel* pluginConsoleStatusLabel_ = nullptr;
    QLabel* pluginSearchPathLabel_ = nullptr;
    QLabel* pluginCountSummaryLabel_ = nullptr;
    QLabel* pluginDatasetFormatSummaryLabel_ = nullptr;
    QLabel* pluginExportFormatSummaryLabel_ = nullptr;
    QLabel* pluginGpuSummaryLabel_ = nullptr;
    QLabel* environmentConsoleStatusLabel_ = nullptr;
    QLabel* environmentOkSummaryLabel_ = nullptr;
    QLabel* environmentWarningSummaryLabel_ = nullptr;
    QLabel* environmentMissingSummaryLabel_ = nullptr;
    QLabel* environmentUncheckedSummaryLabel_ = nullptr;
    QTableWidget* recentTasksTable_ = nullptr;
    QTableWidget* taskQueueTable_ = nullptr;
    QTableWidget* taskArtifactTable_ = nullptr;
    QTableWidget* taskMetricTable_ = nullptr;
    QTableWidget* taskExportTable_ = nullptr;
    QTableWidget* modelVersionTable_ = nullptr;
    QTableWidget* evaluationReportTable_ = nullptr;
    QTableWidget* pipelineRunTable_ = nullptr;
    QTableWidget* datasetListTable_ = nullptr;
    QTableWidget* pluginTable_ = nullptr;
    QTableWidget* environmentTable_ = nullptr;
    QComboBox* taskKindFilterCombo_ = nullptr;
    QComboBox* taskStateFilterCombo_ = nullptr;
    QLineEdit* projectNameEdit_ = nullptr;
    QLineEdit* projectRootEdit_ = nullptr;
    QLineEdit* taskSearchEdit_ = nullptr;
    QLineEdit* datasetPathEdit_ = nullptr;
    QLineEdit* splitOutputEdit_ = nullptr;
    QLineEdit* splitTrainRatioEdit_ = nullptr;
    QLineEdit* splitValRatioEdit_ = nullptr;
    QLineEdit* splitTestRatioEdit_ = nullptr;
    QLineEdit* splitSeedEdit_ = nullptr;
    QComboBox* datasetFormatCombo_ = nullptr;
    QComboBox* pluginCombo_ = nullptr;
    QComboBox* taskTypeCombo_ = nullptr;
    QComboBox* trainingBackendCombo_ = nullptr;
    QComboBox* modelPresetCombo_ = nullptr;
    QLabel* validationSummaryLabel_ = nullptr;
    QLabel* selectedTaskSummaryLabel_ = nullptr;
    QLabel* modelRegistrySummaryLabel_ = nullptr;
    QLabel* datasetDetailLabel_ = nullptr;
    QLabel* annotationToolStatusLabel_ = nullptr;
    QLabel* trainingDatasetSummaryLabel_ = nullptr;
    QLabel* trainingBackendHintLabel_ = nullptr;
    QLabel* trainingRunSummaryLabel_ = nullptr;
    QTableWidget* validationIssuesTable_ = nullptr;
    QTableWidget* datasetPreviewTable_ = nullptr;
    QPlainTextEdit* validationOutput_ = nullptr;
    QLineEdit* epochsEdit_ = nullptr;
    QLineEdit* batchEdit_ = nullptr;
    QLineEdit* imageSizeEdit_ = nullptr;
    QLineEdit* gridSizeEdit_ = nullptr;
    QLineEdit* resumeCheckpointEdit_ = nullptr;
    QLineEdit* conversionCheckpointEdit_ = nullptr;
    QComboBox* conversionFormatCombo_ = nullptr;
    QLineEdit* conversionOutputEdit_ = nullptr;
    QLabel* exportResultLabel_ = nullptr;
    QLineEdit* inferenceCheckpointEdit_ = nullptr;
    QLineEdit* inferenceImageEdit_ = nullptr;
    QLineEdit* inferenceOutputEdit_ = nullptr;
    QLabel* inferenceResultLabel_ = nullptr;
    QLabel* inferenceOverlayLabel_ = nullptr;
    QCheckBox* horizontalFlipCheck_ = nullptr;
    QCheckBox* colorJitterCheck_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel* latestCheckpointLabel_ = nullptr;
    QLabel* latestPreviewPathLabel_ = nullptr;
    QLabel* latestPreviewImageLabel_ = nullptr;
    QLabel* artifactImagePreviewLabel_ = nullptr;
    QPlainTextEdit* artifactPreviewText_ = nullptr;
    QTextEdit* logEdit_ = nullptr;
    MetricsWidget* metricsWidget_ = nullptr;
};
