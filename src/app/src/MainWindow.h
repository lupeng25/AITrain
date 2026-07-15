#pragma once

#include "MetricsWidget.h"
#include "MainWindowState.h"
#include "Sidebar.h"
#include "StatusPill.h"
#include "WorkerClient.h"
#include "aitrain/core/ProjectRepository.h"

#include <QComboBox>
#include <QCheckBox>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QStackedWidget>
#include <QStringList>
#include <QTableWidget>
#include <QTextEdit>
#include <QVector>

class InfoPanel;
class EvaluationReportView;
class TaskArtifactPanel;
class QPushButton;
class QTabWidget;
class QToolButton;
class QFrame;
class QResizeEvent;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(const QString& licenseOwner = QString(), const QString& licenseExpiry = QString(),
        QWidget* parent = nullptr);

protected:
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void createProject();
    void browseDataset();
    void browseDatasetConversionInput();
    void browseDatasetConversionOutput();
    void updateDatasetConversionTargetFormats();
    void startDatasetConversion();
    void cancelDatasetConversion();
    void validateDataset();
    void splitDataset();
    void curateDataset();
    void createDatasetSnapshot();
    void openDatasetQualityReport();
    void openDatasetQualityFixList();
    void launchXAnyLabelingForQualityFix();
    void prepareXAnyLabelingAnnotationSession();
    void syncXAnyLabelingAnnotationSession();
    void browseSampleReviewFile();
    void loadSampleReviewFile();
    void generateFilteredReviewList();
    void openSelectedReviewSample();
    void launchXAnyLabelingForReview();
    void startTraining();
    void startModelExport();
    void validateDeploymentArtifact();
    void startInference();
    void runCustomerOcrAcceptance();
    void collectDiagnosticsBundle();
    void importAcceptanceEvidence();
    void cancelSelectedTask();
    void runEnvironmentCheck();
    void handleWorkerMessage(const QString& type, const QJsonObject& payload);
    void refreshBuiltInCapabilities();
    void showPage(int pageIndex, const QString& title);
    void updateSelectedTaskDetails();
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
    void useSelectedComparisonForInference();
    void useSelectedComparisonForExport();
    void openSelectedComparisonReport();
    void updateSelectedEvaluationReportDetails();
    void openEvaluationReportsPage();

private:
    enum PageIndex {
        DashboardPage = 0,
        ProjectPage,
        DatasetPage,
        TrainingPage,
        TaskQueuePage,
        ModelRegistryPage,
        DeploymentPage,
        EnvironmentPage,
        SystemSettingsPage,
        PageCount
    };

    QWidget* buildTopBar();
    QWidget* buildPageHeading();
    QWidget* buildInspector();
    QWidget* buildDashboardPage();
    QWidget* buildProjectPage();
    QWidget* buildDatasetPage();
    QWidget* buildSampleReviewPanel();
    QWidget* buildTrainingPage();
    QWidget* buildTaskQueuePage();
    QWidget* buildModelRegistryPage();
    QWidget* buildEvaluationReportsPanel();
    QWidget* buildDeploymentPage();
    QWidget* buildModelExportPanel();
    QWidget* buildInferenceValidationPanel();
    QWidget* buildDeliveryEvidencePanel();
    QWidget* buildCapabilitiesPanel();
    QWidget* buildEnvironmentPage();
    QWidget* buildSystemSettingsPage();
    QWidget* buildApplicationSettingsPanel();

    InfoPanel* createMetricCard(const QString& label, const QString& value, const QString& caption);
    QString pageCaption(int pageIndex) const;
    void showDatasetTab(int tabIndex);
    void showModelWorkspaceTab(int tabIndex);
    void showDeploymentTab(int tabIndex);
    void showSystemSettingsTab(int tabIndex);
    QString workerExecutablePath() const;
    QString defaultProjectPath() const;
    QString configuredDefaultProjectPath() const;
    void ensureProjectSubdirs(const QString& rootPath);
    void appendLog(const QString& text);
    void loadCapabilityCombos();
    QString currentDatasetFormat() const;
    QString currentTaskType() const;
    QString currentTaskKindFilter() const;
    QString currentTaskStateFilter() const;
    void handleProgressMessage(const QJsonObject& payload);
    void handleMetricMessage(const QJsonObject& payload);
    void handleArtifactMessage(const QJsonObject& payload);
    void handleTaskStateMessage(const QString& type, const QJsonObject& payload);
    void handleDatasetQualityMessage(const QJsonObject& payload);
    void handleAnnotationSessionMessage(const QJsonObject& payload);
    void handleAnnotationSyncMessage(const QJsonObject& payload);
    void handleDatasetSnapshotMessage(const QJsonObject& payload);
    void handleEvaluationReportMessage(const QJsonObject& payload);
    void handlePipelinePlanMessage(const QJsonObject& payload);
    void handleModelExportMessage(const QJsonObject& payload);
    void handleDeploymentValidationMessage(const QJsonObject& payload);
    void handleInferenceResultMessage(const QJsonObject& payload);
    void handleCustomerOcrAcceptanceMessage(const QJsonObject& payload);
    void handleDiagnosticBundleMessage(const QJsonObject& payload);
    void updateRecentTasks();
    void updateDatasetList();
    void updateTaskTable(QTableWidget* table, const QVector<aitrain::TaskRecord>& tasks);
    void updateHeaderState();
    void updateResponsiveChrome();
    void ensureWorkspacePage(int pageIndex);
    void updateEnvironmentTable(const QJsonObject& payload);
    void updateDatasetValidationResult(const QJsonObject& payload);
    void updateDatasetSplitResult(const QJsonObject& payload);
    void updateDatasetConversionResult(const QJsonObject& payload);
    void setDatasetConversionFormRunning(bool running);
    void clearDatasetConversionErrors();
    void appendDatasetConversionLog(const QString& text);
    void refreshDatasetConversionDefaultsFromCurrentDataset();
    void updateDatasetRepairLoopFromQuality(const QJsonObject& payload);
    void updateDatasetRepairLoopFromValidation(const QJsonObject& payload);
    void setDatasetRepairLoopRows(const QString& summary, const QVector<QStringList>& rows);
    void startQueuedTraining(const QString& taskId, const aitrain::TrainingRequest& request);
    void startNextQueuedTask();
    void startSnapshotForQueuedTraining(const PendingTrainingTask& pending);
    void configureTable(QTableWidget* table) const;
    void updateDashboardSummary();
    void updateProjectSummary();
    void updateCapabilitySummary();
    void updateEnvironmentSummary();
    void updateSettingsSummary();
    void updateDeliveryAcceptanceSummary();
    void refreshSampleReviewTable();
    QJsonArray filteredSampleReviewRows() const;
    void updateTrainingSelectionSummary();
    void refreshModelExportFormatOptions();
    void refreshTrainingDefaults();
    void storeLanguagePreference(const QString& languageCode);
    void updateLanguageButtonState();
    void storeDefaultProjectPathPreference(const QString& path);
    void openLocalDirectory(const QString& path);
    void copyLocalPath(const QString& path, const QString& label);
    void updateAnnotationToolStatus();
    void refreshAfterAnnotation();
    void applyTaskFilters();
    void ensureVisibleTaskSelection();
    void clearSelectedTaskDetails();
    void updateModelRegistry();
    void updateModelComparison(
        const QVector<aitrain::ModelVersionRecord>& models,
        const QVector<aitrain::EvaluationReportRecord>& reports);
    bool attachLatestSnapshotToRequest(aitrain::TrainingRequest& request, int datasetId, QString* error);
    int recordExperimentRunForRequest(const aitrain::TrainingRequest& request, int datasetId, QString* error);
    void updateExperimentRunSummary(const QString& taskId);
    QLabel* trainingLiveValueLabel(const QString& objectName) const;
    void registerPipelineModelVersion(const QJsonObject& payload);
    QString createRepositoryTask(aitrain::TaskKind kind, const QString& taskType, const QString& capabilityId, const QString& workDir, const QString& message, const QString& requestedTaskId = {});
    QString selectedTaskId() const;
    QString selectedArtifactPath() const;
    QString selectedEvaluationReportPath() const;
    QString selectedComparisonModelPath() const;
    QString selectedComparisonReportPath() const;

    aitrain::ProjectRepository repository_;
    WorkerClient worker_;
    MainWindowState state_;

    QString currentProjectPath_;
    QString currentProjectName_;

    Sidebar* sidebar_ = nullptr;
    QFrame* inspector_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    QTabWidget* datasetTabs_ = nullptr;
    QTabWidget* modelWorkspaceTabs_ = nullptr;
    QTabWidget* deploymentTabs_ = nullptr;
    QTabWidget* systemSettingsTabs_ = nullptr;
    QLabel* pageTitle_ = nullptr;
    QLabel* pageCaption_ = nullptr;
    QLabel* headerProjectLabel_ = nullptr;
    StatusPill* workerPill_ = nullptr;
    StatusPill* capabilityPill_ = nullptr;
    StatusPill* gpuPill_ = nullptr;
    StatusPill* licensePill_ = nullptr;
    QToolButton* topBarZhLanguageButton_ = nullptr;
    QToolButton* topBarEnLanguageButton_ = nullptr;
    QToolButton* inspectorToggleButton_ = nullptr;
    bool inspectorUserOverride_ = false;
    bool applyingResponsiveChrome_ = false;
    StatusPill* pageContextPill_ = nullptr;
    QToolButton* settingsZhLanguageButton_ = nullptr;
    QToolButton* settingsEnLanguageButton_ = nullptr;
    QString licenseOwner_;
    QString licenseExpiry_;
    QLabel* inspectorProjectLabel_ = nullptr;
    QLabel* inspectorCapabilityLabel_ = nullptr;
    QLabel* inspectorWorkerLabel_ = nullptr;
    QLabel* inspectorGpuLabel_ = nullptr;

    QLabel* projectLabel_ = nullptr;
    QLabel* gpuLabel_ = nullptr;
    QLabel* dashboardProjectValue_ = nullptr;
    QLabel* dashboardTaskValue_ = nullptr;
    QLabel* dashboardCapabilityValue_ = nullptr;
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
    QLabel* capabilityConsoleStatusLabel_ = nullptr;
    QLabel* capabilitySourceLabel_ = nullptr;
    QLabel* capabilityCountSummaryLabel_ = nullptr;
    QLabel* capabilityDatasetFormatSummaryLabel_ = nullptr;
    QLabel* capabilityExportFormatSummaryLabel_ = nullptr;
    QLabel* capabilityGpuSummaryLabel_ = nullptr;
    QLabel* environmentConsoleStatusLabel_ = nullptr;
    QLabel* environmentOkSummaryLabel_ = nullptr;
    QLabel* environmentWarningSummaryLabel_ = nullptr;
    QLabel* environmentMissingSummaryLabel_ = nullptr;
    QLabel* environmentUncheckedSummaryLabel_ = nullptr;
    QLabel* settingsDefaultProjectPathStatusLabel_ = nullptr;
    QLabel* settingsCurrentProjectPathLabel_ = nullptr;
    QLineEdit* settingsDefaultProjectPathEdit_ = nullptr;
    QTableWidget* recentTasksTable_ = nullptr;
    QTableWidget* taskQueueTable_ = nullptr;
    TaskArtifactPanel* taskArtifactPanel_ = nullptr;
    QTableWidget* modelVersionTable_ = nullptr;
    QTableWidget* evaluationReportTable_ = nullptr;
    QTableWidget* pipelineRunTable_ = nullptr;
    QTableWidget* datasetListTable_ = nullptr;
    QTableWidget* capabilityTable_ = nullptr;
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
    QComboBox* datasetConversionSourceFormatCombo_ = nullptr;
    QComboBox* datasetConversionTargetFormatCombo_ = nullptr;
    QLineEdit* datasetConversionInputEdit_ = nullptr;
    QLineEdit* datasetConversionOutputEdit_ = nullptr;
    QLabel* datasetConversionStatusLabel_ = nullptr;
    QLabel* datasetConversionSourceErrorLabel_ = nullptr;
    QLabel* datasetConversionTargetErrorLabel_ = nullptr;
    QLabel* datasetConversionInputErrorLabel_ = nullptr;
    QLabel* datasetConversionOutputErrorLabel_ = nullptr;
    QLabel* datasetConversionResultLabel_ = nullptr;
    QPushButton* datasetConversionStartButton_ = nullptr;
    QPushButton* datasetConversionCancelButton_ = nullptr;
    QPushButton* datasetConversionBrowseInputButton_ = nullptr;
    QPushButton* datasetConversionBrowseOutputButton_ = nullptr;
    QProgressBar* datasetConversionProgressBar_ = nullptr;
    QPlainTextEdit* datasetConversionLog_ = nullptr;
    QComboBox* capabilityCombo_ = nullptr;
    QComboBox* taskTypeCombo_ = nullptr;
    QComboBox* trainingBackendCombo_ = nullptr;
    QComboBox* modelPresetCombo_ = nullptr;
    QLabel* validationSummaryLabel_ = nullptr;
    QLabel* datasetRepairLoopLabel_ = nullptr;
    QLabel* modelRegistrySummaryLabel_ = nullptr;
    QLabel* modelComparisonSummaryLabel_ = nullptr;
    QLabel* datasetDetailLabel_ = nullptr;
    QLabel* annotationToolStatusLabel_ = nullptr;
    QLabel* trainingDatasetSummaryLabel_ = nullptr;
    QLabel* trainingBackendHintLabel_ = nullptr;
    QLabel* trainingRunSummaryLabel_ = nullptr;
    QTableWidget* validationIssuesTable_ = nullptr;
    QTableWidget* datasetRepairLoopTable_ = nullptr;
    QTableWidget* modelComparisonTable_ = nullptr;
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
    QLineEdit* conversionValidationImageEdit_ = nullptr;
    QLabel* exportResultLabel_ = nullptr;
    QLabel* deploymentValidationResultLabel_ = nullptr;
    QLineEdit* inferenceCheckpointEdit_ = nullptr;
    QLineEdit* inferenceImageEdit_ = nullptr;
    QLineEdit* inferenceOutputEdit_ = nullptr;
    QLabel* inferenceResultLabel_ = nullptr;
    QLabel* inferenceOverlayLabel_ = nullptr;
    QCheckBox* horizontalFlipCheck_ = nullptr;
    QCheckBox* colorJitterCheck_ = nullptr;
    QLineEdit* reviewSamplePathEdit_ = nullptr;
    QComboBox* reviewSourceFilterCombo_ = nullptr;
    QComboBox* reviewReasonFilterCombo_ = nullptr;
    QLineEdit* reviewSearchEdit_ = nullptr;
    QTableWidget* sampleReviewTable_ = nullptr;
    QLabel* sampleReviewSummaryLabel_ = nullptr;
    QLineEdit* customerOcrDetDatasetEdit_ = nullptr;
    QLineEdit* customerOcrRecDatasetEdit_ = nullptr;
    QLineEdit* customerOcrSystemImagesEdit_ = nullptr;
    QLineEdit* customerOcrDetReportEdit_ = nullptr;
    QLineEdit* customerOcrRecReportEdit_ = nullptr;
    QLineEdit* customerOcrSystemReportEdit_ = nullptr;
    QLineEdit* customerOcrOutputEdit_ = nullptr;
    QLineEdit* customerOcrMinAccEdit_ = nullptr;
    QLineEdit* customerOcrMaxCerEdit_ = nullptr;
    QCheckBox* customerOcrAllowPublicCheck_ = nullptr;
    QLabel* customerOcrStatusLabel_ = nullptr;
    QLabel* diagnosticsStatusLabel_ = nullptr;
    QLabel* deliveryAcceptanceSummaryLabel_ = nullptr;
    QTableWidget* deliveryAcceptanceTable_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel* trainingPhaseLabel_ = nullptr;
    QLabel* trainingEpochValueLabel_ = nullptr;
    QLabel* trainingBatchValueLabel_ = nullptr;
    QLabel* trainingEtaValueLabel_ = nullptr;
    QLabel* trainingDeviceValueLabel_ = nullptr;
    QLabel* trainingLossValueLabel_ = nullptr;
    QLabel* trainingMapValueLabel_ = nullptr;
    QLabel* latestCheckpointLabel_ = nullptr;
    QLabel* latestOnnxLabel_ = nullptr;
    QLabel* latestReportLabel_ = nullptr;
    QLabel* latestPreviewPathLabel_ = nullptr;
    QLabel* latestPreviewImageLabel_ = nullptr;
    QTextEdit* logEdit_ = nullptr;
    MetricsWidget* metricsWidget_ = nullptr;
    EvaluationReportView* evaluationReportView_ = nullptr;
};
