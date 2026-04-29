#pragma once

#include "MetricsWidget.h"
#include "Sidebar.h"
#include "StatusPill.h"
#include "WorkerClient.h"
#include "aitrain/core/PluginManager.h"
#include "aitrain/core/ProjectRepository.h"

#include <QComboBox>
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
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void createProject();
    void browseDataset();
    void validateDataset();
    void startTraining();
    void cancelSelectedTask();
    void runEnvironmentCheck();
    void handleWorkerMessage(const QString& type, const QJsonObject& payload);
    void refreshPlugins();
    void showPage(int pageIndex, const QString& title);

private:
    enum PageIndex {
        DashboardPage = 0,
        ProjectPage,
        DatasetPage,
        TrainingPage,
        TaskQueuePage,
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
    void updateRecentTasks();
    void updateTaskTable(QTableWidget* table, const QVector<aitrain::TaskRecord>& tasks);
    void updateHeaderState();
    void updateEnvironmentTable(const QJsonObject& payload);
    void startQueuedTraining(const QString& taskId, const aitrain::TrainingRequest& request);
    void startNextQueuedTask();
    void configureTable(QTableWidget* table) const;

    aitrain::PluginManager pluginManager_;
    aitrain::ProjectRepository repository_;
    WorkerClient worker_;

    QString currentProjectPath_;
    QString currentProjectName_;
    QString currentTaskId_;

    struct PendingTrainingTask {
        QString taskId;
        aitrain::TrainingRequest request;
    };
    QVector<PendingTrainingTask> pendingTrainingTasks_;

    Sidebar* sidebar_ = nullptr;
    QStackedWidget* stack_ = nullptr;
    QLabel* pageTitle_ = nullptr;
    QLabel* pageCaption_ = nullptr;
    QLabel* headerProjectLabel_ = nullptr;
    StatusPill* workerPill_ = nullptr;
    StatusPill* pluginPill_ = nullptr;
    StatusPill* gpuPill_ = nullptr;

    QLabel* projectLabel_ = nullptr;
    QLabel* gpuLabel_ = nullptr;
    QLabel* dashboardProjectValue_ = nullptr;
    QLabel* dashboardTaskValue_ = nullptr;
    QLabel* dashboardPluginValue_ = nullptr;
    QTableWidget* recentTasksTable_ = nullptr;
    QTableWidget* taskQueueTable_ = nullptr;
    QTableWidget* pluginTable_ = nullptr;
    QTableWidget* environmentTable_ = nullptr;
    QLineEdit* projectNameEdit_ = nullptr;
    QLineEdit* projectRootEdit_ = nullptr;
    QLineEdit* datasetPathEdit_ = nullptr;
    QComboBox* datasetFormatCombo_ = nullptr;
    QComboBox* pluginCombo_ = nullptr;
    QComboBox* taskTypeCombo_ = nullptr;
    QPlainTextEdit* validationOutput_ = nullptr;
    QLineEdit* epochsEdit_ = nullptr;
    QLineEdit* batchEdit_ = nullptr;
    QLineEdit* imageSizeEdit_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QTextEdit* logEdit_ = nullptr;
    MetricsWidget* metricsWidget_ = nullptr;
};
