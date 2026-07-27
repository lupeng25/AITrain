#pragma once

#include "MetricsWidget.h"
#include "Sidebar.h"
#include "StatusPill.h"
#include "WorkerClient.h"
#include "MainWindowSupport.h"

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
class TaskArtifactPage;
class TaskArtifactPageController;
class DashboardWorkspacePage;
class DashboardPageController;
class ProjectWorkspacePage;
class ProjectPageController;
class SettingsWorkspacePage;
class SettingsPageController;
class RuntimeDeliveryWorkspacePage;
class RuntimeDeliveryPageController;
class TrainingWorkspacePage;
class DatasetWorkspacePage;
class DatasetPageController;
class DeliveryEvidenceWorkspacePage;
class DeliveryEvidencePageController;
class TrainingPageController;
class ProjectSessionController;
class EnvironmentCheckPresenter;
class EnvironmentWorkspacePage;
class EnvironmentPageController;
class ModelRegistryPresenter;
class ModelRegistryWorkspacePage;
class ModelRegistryPageController;
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
    void openDatasetQualityReport();
    void openDatasetQualityFixList();
    void handleTaskViewStateChanged(const TaskViewState& state);
    void showPage(int pageIndex, const QString& title);

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
    QWidget* buildDeliveryEvidencePanel();
    QWidget* buildEnvironmentPage();
    QWidget* buildSystemSettingsPage();

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
    QString currentProjectPath() const;
    QString currentProjectName() const;
    quint64 projectOpenGeneration() const;
    void updateRecentTasks();
    void updateDatasetList();
    void updateHeaderState();
    void updateResponsiveChrome();
    void ensureWorkspacePage(int pageIndex);
    void setDatasetRepairLoopRows(const QString& summary, const QVector<QStringList>& rows);
    void configureTable(QTableWidget* table) const;
    void updateDashboardSummary();
    void updateProjectSummary();
    void updateDeliveryAcceptanceSummary();
    void updateTaskCancelButton();
    void updateTrainingSelectionSummary();
    void refreshTrainingDefaults();
    void updateLanguageButtonState();
    void updateAnnotationToolStatus();
    void clearSelectedTaskDetails();
    void updateModelRegistry();
    void syncModelPackageCombos();
    QString selectedTaskId() const;
    void activateProjectUi(const QString& projectName,
        const QString& canonicalRoot, quint64 generation);

    TaskArtifactPageController* taskArtifactPageController_ = nullptr;
    DashboardPageController* dashboardPageController_ = nullptr;
    ProjectPageController* projectPageController_ = nullptr;
    SettingsPageController* settingsPageController_ = nullptr;
    EnvironmentPageController* environmentPageController_ = nullptr;
    ModelRegistryPageController* modelRegistryPageController_ = nullptr;
    RuntimeDeliveryPageController* runtimeDeliveryPageController_ = nullptr;
    TrainingPageController* trainingPageController_ = nullptr;
    DatasetPageController* datasetPageController_ = nullptr;
    DeliveryEvidencePageController* deliveryEvidencePageController_ = nullptr;
    ApplicationEventRouter* eventRouter_ = nullptr;
    TaskRuntimeController* taskController_ = nullptr;
    ProjectSessionController* projectSessionController_ = nullptr;
    WorkspaceRouter* workspaceRouter_ = nullptr;
    WorkspaceReadModelCoordinator* readModelCoordinator_ = nullptr;
    WorkerClient& workerClient();
    Sidebar* sidebar_ = nullptr;
    QFrame* inspector_ = nullptr;
    QStackedWidget* stack_ = nullptr;
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
    QString licenseOwner_;
    QString licenseExpiry_;
    QLabel* inspectorProjectLabel_ = nullptr;
    QLabel* inspectorCapabilityLabel_ = nullptr;
    QLabel* inspectorWorkerLabel_ = nullptr;
    QLabel* inspectorGpuLabel_ = nullptr;

    DashboardWorkspacePage* dashboardPage_ = nullptr;
    ProjectWorkspacePage* projectPage_ = nullptr;
    SettingsWorkspacePage* settingsPage_ = nullptr;
    TaskArtifactPage* taskArtifactPage_ = nullptr;
    ModelRegistryWorkspacePage* modelRegistryPage_ = nullptr;
    RuntimeDeliveryWorkspacePage* runtimeDeliveryPage_ = nullptr;
    TrainingWorkspacePage* trainingPage_ = nullptr;
    DatasetWorkspacePage* datasetPage_ = nullptr;
    DeliveryEvidenceWorkspacePage* deliveryEvidencePage_ = nullptr;
    EnvironmentWorkspacePage* environmentPage_ = nullptr;
};
