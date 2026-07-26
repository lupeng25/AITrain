#pragma once

#include "MetricsWidget.h"
#include "MainWindowState.h"
#include "Sidebar.h"
#include "StatusPill.h"
#include "WorkerClient.h"
#include "MainWindowSupport.h"
#include "aitrain/workflow\ProjectWorkspace.h"
#include "aitrain/workflow\ProjectQueryService.h"

#include <QComboBox>
#include <QCheckBox>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPair>
#include <QProgressBar>
#include <QStackedWidget>
#include <QStringList>
#include <QTableWidget>
#include <QTextEdit>
#include <QVector>

class InfoPanel;
class TaskArtifactPanel;
class TaskArtifactPresenter;
class DatasetCatalogPresenter;
class DeliveryEvidencePresenter;
class ProjectSummaryPresenter;
class DiagnosticBundlePresenter;
class EnvironmentCheckPresenter;
class ModelRegistryPresenter;
class QPushButton;
class QCloseEvent;
class QTabWidget;
class QToolButton;
class QFrame;
class QResizeEvent;
class ApplicationEventRouter;
class TaskRuntimeController;
struct TaskViewState;
class WorkspaceRouter;
class WorkspaceReadModelCoordinator;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
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

    explicit MainWindow(const QString& licenseOwner = QString(), const QString& licenseExpiry = QString(),
        QWidget* parent = nullptr);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void createProject();
    void browseDataset();
    void browseDatasetConversionInput();
    void updateDatasetConversionTargetFormats();
    void startDatasetConversion();
    void cancelDatasetConversion();
    void runDataQualityWorkflow();
    void runDatasetSplitWorkflow();
    void runDatasetSnapshotImportWorkflow();
    void openDatasetQualityReport();
    void openDatasetQualityFixList();
    void createXAnyLabelingAnnotationSession();
    void syncXAnyLabelingAnnotationSession();
    void browseSampleReviewFile();
    void loadSampleReviewFile();
    void openSelectedReviewSample();
    void startTraining();
    void validateDeploymentModelPackage();
    void startInference();
    void importModelPackage();
    void importOcrOfficialReports();
    void runOcrAcceptanceWorkflow();
    void collectDiagnosticsBundle();
    void importAcceptanceEvidence();
    void cancelSelectedTask();
    void runEnvironmentCheck();
    void handleTaskViewStateChanged(const TaskViewState& state);
    void refreshBuiltInCapabilities();
    void showPage(int pageIndex, const QString& title);
    void updateSelectedTaskDetails();
    void refreshModelRegistry();

private:
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
    QWidget* buildDeploymentPage();
    QWidget* buildDeploymentValidationPanel();
    QWidget* buildInferenceValidationPanel();
    QWidget* buildDeliveryEvidencePanel();
    QWidget* buildCapabilitiesPanel();
    QWidget* buildEnvironmentPage();
    QWidget* buildSystemSettingsPage();
    QWidget* buildApplicationSettingsPanel();

    InfoPanel* createMetricCard(const QString& label, const QString& value, const QString& caption);
    QString pageCaption(int pageIndex) const;
    void showDatasetTab(int tabIndex);
    void showDeploymentTab(int tabIndex);
    void showSystemSettingsTab(int tabIndex);
    QString workerExecutablePath() const;
    QString defaultProjectPath() const;
    QString configuredDefaultProjectPath() const;
    void appendLog(const QString& text);
    void loadCapabilityCombos();
    QString currentDatasetFormat() const;
    QString currentTaskType() const;
    QString currentTaskKindFilter() const;
    QString currentTaskStateFilter() const;
    void updateRecentTasks();
    void updateTaskTable();
    void updateDatasetList();
    void updateHeaderState();
    void updateResponsiveChrome();
    void ensureWorkspacePage(int pageIndex);
    void setDatasetConversionFormRunning(bool running);
    void clearDatasetConversionErrors();
    void appendDatasetConversionLog(const QString& text);
    void refreshDatasetConversionDefaultsFromCurrentDataset();
    void setDatasetRepairLoopRows(const QString& summary, const QVector<QStringList>& rows);
    void configureTable(QTableWidget* table) const;
    void updateDashboardSummary();
    void updateProjectSummary();
    void updateCapabilitySummary();
    void updateEnvironmentSummary();
    void refreshEnvironmentReportView();
    void updateSettingsSummary();
    void updateDeliveryAcceptanceSummary();
    void renderDeliveryAcceptanceSummary();
    void updateTaskCancelButton();
    void refreshSampleReviewTable();
    QJsonArray filteredSampleReviewRows() const;
    void updateTrainingSelectionSummary();
    void refreshTrainingDefaults();
    void storeLanguagePreference(const QString& languageCode);
    void updateLanguageButtonState();
    void storeDefaultProjectPathPreference(const QString& path);
    void updateAnnotationToolStatus();
    void applyTaskFilters();
    void ensureVisibleTaskSelection();
    void clearSelectedTaskDetails();
    void updateModelRegistry();
    QLabel* trainingLiveValueLabel(const QString& objectName) const;
    QString selectedTaskId() const;
    void startDatasetFormatProbe(const QString& path, bool conversionSource);
    void applyDatasetFormatProbe(const QString& path, const QString& detectedFormat,
        bool conversionSource, quint64 generation);
    void loadSampleReviewArtifactCandidate(const aitrain::ArtifactId& artifactId,
        const QStringList& candidates, int index, quint64 generation,
        const QString& lastError = QString());
    void finishProjectOpen(const QString& projectName, const QString& projectPath,
        quint64 generation, const aitrain_app::ProjectOpenProbeResult& result);
    void setProjectOpenUiBusy(bool busy);

    aitrain::ProjectWorkspace workspace_;
    aitrain::ProjectQueryService queryService_;
    ProjectSummaryPresenter* projectSummaryPresenter_ = nullptr;
    TaskArtifactPresenter* taskArtifactPresenter_ = nullptr;
    DiagnosticBundlePresenter* diagnosticBundlePresenter_ = nullptr;
    EnvironmentCheckPresenter* environmentCheckPresenter_ = nullptr;
    ModelRegistryPresenter* modelRegistryPresenter_ = nullptr;
    DatasetCatalogPresenter* datasetCatalogPresenter_ = nullptr;
    DeliveryEvidencePresenter* deliveryEvidencePresenter_ = nullptr;
    QString activeTaskId_;
    QString activeWorkflowKind_;
    qint64 liveMetricSequence_ = 0;
    qint64 liveArtifactSequence_ = 0;
    ApplicationEventRouter* eventRouter_ = nullptr;
    TaskRuntimeController* taskController_ = nullptr;
    WorkspaceRouter* workspaceRouter_ = nullptr;
    WorkspaceReadModelCoordinator* readModelCoordinator_ = nullptr;
    WorkerClient& workerClient();
    MainWindowState state_;

    QString currentProjectPath_;
    QString currentProjectName_;
    QString pendingProjectPath_;
    QString pendingProjectName_;
    quint64 projectOpenGeneration_ = 0;
    bool projectOpenInProgress_ = false;
    quint64 datasetFormatProbeGeneration_ = 0;
    quint64 sampleReviewPreviewGeneration_ = 0;

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
    bool closePending_ = false;
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
    QLineEdit* settingsDefaultProjectPathEdit_ = nullptr;
    QTableWidget* recentTasksTable_ = nullptr;
    QTableWidget* taskQueueTable_ = nullptr;
    QPushButton* taskCancelButton_ = nullptr;
    TaskArtifactPanel* taskArtifactPanel_ = nullptr;
    QTableWidget* ModelPackageTable_ = nullptr;
    QLineEdit* modelImportSourceEdit_ = nullptr;
    QLineEdit* modelImportManifestEdit_ = nullptr;
    QLabel* modelImportResultLabel_ = nullptr;
    QTableWidget* datasetListTable_ = nullptr;
    QTableWidget* capabilityTable_ = nullptr;
    QTableWidget* environmentTable_ = nullptr;
    QComboBox* taskKindFilterCombo_ = nullptr;
    QComboBox* taskStateFilterCombo_ = nullptr;
    QLineEdit* projectNameEdit_ = nullptr;
    QLineEdit* projectRootEdit_ = nullptr;
    QPushButton* projectOpenButton_ = nullptr;
    QLineEdit* taskSearchEdit_ = nullptr;
    QLineEdit* datasetPathEdit_ = nullptr;
    QLabel* datasetProbeStatusLabel_ = nullptr;
    QLineEdit* splitSourceDatasetIdEdit_ = nullptr;
    QLineEdit* splitSourceDatasetVersionIdEdit_ = nullptr;
    QLineEdit* splitSourceSnapshotIdEdit_ = nullptr;
    QLineEdit* splitSourceSnapshotArtifactIdEdit_ = nullptr;
    QLineEdit* splitTargetDatasetIdEdit_ = nullptr;
    QLineEdit* splitTargetDatasetNameEdit_ = nullptr;
    QLineEdit* splitTrainRatioEdit_ = nullptr;
    QLineEdit* splitValRatioEdit_ = nullptr;
    QLineEdit* splitTestRatioEdit_ = nullptr;
    QLineEdit* splitSeedEdit_ = nullptr;
    QComboBox* datasetFormatCombo_ = nullptr;
    QLineEdit* dataQualityDatasetIdEdit_ = nullptr;
    QLineEdit* dataQualityDatasetVersionIdEdit_ = nullptr;
    QLineEdit* dataQualitySnapshotIdEdit_ = nullptr;
    QLineEdit* dataQualitySnapshotArtifactIdEdit_ = nullptr;
    QLineEdit* datasetSnapshotTargetDatasetIdEdit_ = nullptr;
    QLineEdit* datasetSnapshotTargetDatasetNameEdit_ = nullptr;
    QComboBox* datasetConversionSourceFormatCombo_ = nullptr;
    QComboBox* datasetConversionTargetFormatCombo_ = nullptr;
    QLineEdit* datasetConversionInputEdit_ = nullptr;
    QLabel* datasetConversionProbeStatusLabel_ = nullptr;
    QLineEdit* datasetConversionTargetDatasetIdEdit_ = nullptr;
    QLineEdit* datasetConversionTargetDatasetNameEdit_ = nullptr;
    QLabel* datasetConversionStatusLabel_ = nullptr;
    QLabel* datasetConversionSourceErrorLabel_ = nullptr;
    QLabel* datasetConversionTargetErrorLabel_ = nullptr;
    QLabel* datasetConversionInputErrorLabel_ = nullptr;
    QLabel* datasetConversionResultLabel_ = nullptr;
    QPushButton* datasetConversionStartButton_ = nullptr;
    QPushButton* datasetConversionCancelButton_ = nullptr;
    QPushButton* datasetConversionBrowseInputButton_ = nullptr;
    QProgressBar* datasetConversionProgressBar_ = nullptr;
    QPlainTextEdit* datasetConversionLog_ = nullptr;
    QComboBox* capabilityCombo_ = nullptr;
    QComboBox* taskTypeCombo_ = nullptr;
    QComboBox* trainingBackendCombo_ = nullptr;
    QComboBox* modelPresetCombo_ = nullptr;
    QLabel* validationSummaryLabel_ = nullptr;
    QLabel* datasetRepairLoopLabel_ = nullptr;
    QLabel* modelRegistrySummaryLabel_ = nullptr;
    QLabel* datasetDetailLabel_ = nullptr;
    QLabel* annotationToolStatusLabel_ = nullptr;
    QLabel* trainingDatasetSummaryLabel_ = nullptr;
    QLabel* trainingBackendHintLabel_ = nullptr;
    QLabel* trainingRunSummaryLabel_ = nullptr;
    QTableWidget* validationIssuesTable_ = nullptr;
    QTableWidget* datasetRepairLoopTable_ = nullptr;
    QTableWidget* datasetPreviewTable_ = nullptr;
    QPlainTextEdit* validationOutput_ = nullptr;
    QLineEdit* epochsEdit_ = nullptr;
    QLineEdit* batchEdit_ = nullptr;
    QLineEdit* imageSizeEdit_ = nullptr;
    QLineEdit* gridSizeEdit_ = nullptr;
    QLineEdit* deploymentSampleDatasetIdEdit_ = nullptr;
    QLineEdit* deploymentSampleDatasetVersionIdEdit_ = nullptr;
    QLineEdit* deploymentSampleSnapshotIdEdit_ = nullptr;
    QLineEdit* deploymentSampleSnapshotArtifactIdEdit_ = nullptr;
    QLineEdit* deploymentSampleRelativePathEdit_ = nullptr;
    QLabel* deploymentValidationResultLabel_ = nullptr;
    QComboBox* deploymentModelPackageCombo_ = nullptr;
    QComboBox* inferenceModelPackageCombo_ = nullptr;
    bool modelImportInProgress_ = false;
    QLineEdit* inferenceSampleDatasetIdEdit_ = nullptr;
    QLineEdit* inferenceSampleDatasetVersionIdEdit_ = nullptr;
    QLineEdit* inferenceSampleSnapshotIdEdit_ = nullptr;
    QLineEdit* inferenceSampleSnapshotArtifactIdEdit_ = nullptr;
    QLineEdit* inferenceSampleRelativePathEdit_ = nullptr;
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
    QLineEdit* customerOcrDetReportEdit_ = nullptr;
    QLineEdit* customerOcrRecReportEdit_ = nullptr;
    QLineEdit* customerOcrSystemReportEdit_ = nullptr;
    QLineEdit* customerOcrDetSnapshotIdEdit_ = nullptr;
    QLineEdit* customerOcrDetSnapshotArtifactIdEdit_ = nullptr;
    QLineEdit* customerOcrRecSnapshotIdEdit_ = nullptr;
    QLineEdit* customerOcrRecSnapshotArtifactIdEdit_ = nullptr;
    QLineEdit* customerOcrSystemSnapshotIdEdit_ = nullptr;
    QLineEdit* customerOcrSystemSnapshotArtifactIdEdit_ = nullptr;
    QLineEdit* customerOcrCohortIdEdit_ = nullptr;
    QLineEdit* customerOcrDomainIdEdit_ = nullptr;
    QComboBox* customerOcrEvidenceClassCombo_ = nullptr;
    QLineEdit* customerOcrDetReportArtifactIdEdit_ = nullptr;
    QLineEdit* customerOcrRecReportArtifactIdEdit_ = nullptr;
    QLineEdit* customerOcrSystemReportArtifactIdEdit_ = nullptr;
    QLineEdit* customerOcrMinDetHmeanEdit_ = nullptr;
    QLineEdit* customerOcrMinAccEdit_ = nullptr;
    QLineEdit* customerOcrMaxCerEdit_ = nullptr;
    QLineEdit* customerOcrMinSystemAccEdit_ = nullptr;
    QLabel* customerOcrStatusLabel_ = nullptr;
    QLabel* diagnosticsStatusLabel_ = nullptr;
    QPointer<QLabel> deliveryAcceptanceSummaryLabel_;
    QPointer<QTableWidget> deliveryAcceptanceTable_;
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
    QLabel* latestPreviewLabel_ = nullptr;
    QLabel* latestPreviewImageLabel_ = nullptr;
    QTextEdit* logEdit_ = nullptr;
    MetricsWidget* metricsWidget_ = nullptr;
};
