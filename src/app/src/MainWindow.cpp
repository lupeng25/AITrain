#include "WorkbenchTranslation.h"
#include "MainWindow.h"

#include "ApplicationEventRouter.h"
#include "TaskRuntimeController.h"
#include "WorkspaceReadModelCoordinator.h"

#include "EvaluationReportView.h"
#include "DatasetPageController.h"
#include "DatasetPage.h"
#include "DashboardPageController.h"
#include "DeliveryEvidencePageController.h"
#include "EnvironmentCheckPresenter.h"
#include "EnvironmentPageController.h"
#include "ApplicationSettingsService.h"
#include "ModelRegistryPresenter.h"
#include "ModelRegistryPageController.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "ProjectPageController.h"
#include "ProjectSessionController.h"
#include "SettingsPage.h"
#include "SettingsPageController.h"
#include "RuntimeDeliveryPageController.h"
#include "TrainingPageController.h"
#include "WorkspaceRouter.h"
#include "TaskArtifactPageController.h"
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/VisionModelRuntime.h"

#include <QApplication>
#include <QCloseEvent>
#include <QCheckBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidgetItem>
#include <QTime>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUuid>

using namespace aitrain_app;

MainWindow::MainWindow(const QString& licenseOwner, const QString& licenseExpiry, QWidget* parent)
    : QMainWindow(parent)
    , licenseOwner_(licenseOwner)
    , licenseExpiry_(licenseExpiry)
{
    settingsPageController_ = new SettingsPageController(defaultProjectPath(), this);
    workspaceRouter_ = new WorkspaceRouter(PageCount, this);
    taskController_ = new TaskRuntimeController(this);
    projectSessionController_ = new ProjectSessionController(
        taskController_, this);
    const aitrain::ProjectQueryService* queryService =
        projectSessionController_->queryService();
    taskArtifactPageController_ = new TaskArtifactPageController(queryService, this);
    dashboardPageController_ = new DashboardPageController(queryService, this);
    setWindowTitle(QStringLiteral("AITrain Studio"));
    setMinimumSize(1024, 700);

    auto* central = new QWidget(this);
    auto* rootLayout = new QHBoxLayout(central);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    sidebar_ = new Sidebar;
    sidebar_->addSection(aitrain_app::workbenchText(QStringLiteral("工作区")));
    sidebar_->addItem(aitrain_app::workbenchText(QStringLiteral("数据集")), DatasetPage);
    sidebar_->addItem(aitrain_app::workbenchText(QStringLiteral("训练")), TrainingPage);
    sidebar_->addItem(aitrain_app::workbenchText(QStringLiteral("模型")), ModelRegistryPage);
    sidebar_->addToolItem(aitrain_app::workbenchText(QStringLiteral("环境与诊断")), EnvironmentPage);
    sidebar_->addToolItem(aitrain_app::workbenchText(QStringLiteral("设置")), SystemSettingsPage);
    rootLayout->addWidget(sidebar_);

    auto* content = new QWidget;
    auto* contentLayout = new QVBoxLayout(content);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(0);
    contentLayout->addWidget(buildTopBar());
    contentLayout->addWidget(buildPageHeading());

    stack_ = new QStackedWidget;
    stack_->setObjectName(QStringLiteral("WorkspaceStack"));
    QWidget* dashboardPage = buildDashboardPage();
    dashboardPage->setProperty("workspaceInitialized", true);
    stack_->addWidget(dashboardPage);
    for (int pageIndex = ProjectPage; pageIndex < PageCount; ++pageIndex) {
        auto* placeholder = new QWidget;
        placeholder->setProperty("workspaceInitialized", false);
        stack_->addWidget(placeholder);
    }
    contentLayout->addWidget(stack_, 1);

    rootLayout->addWidget(content, 1);
    setCentralWidget(central);

    statusBar()->setSizeGripEnabled(false);
    statusBar()->addPermanentWidget(workerPill_);
    auto* taskRecordButton = new QPushButton(aitrain_app::workbenchText(QStringLiteral("任务记录")));
    taskRecordButton->setObjectName(QStringLiteral("ActivityTaskButton"));
    connect(taskRecordButton, &QPushButton::clicked, this, [this]() {
        showPage(TaskQueuePage, aitrain_app::workbenchText(QStringLiteral("任务记录")));
        if (taskController_->taskId().isValid()) taskArtifactPageController_->openTask(taskController_->taskId().toString());
    });
    statusBar()->addPermanentWidget(taskRecordButton);
    auto* hideActivity = new QPushButton(aitrain_app::workbenchText(QStringLiteral("收起")));
    statusBar()->addPermanentWidget(hideActivity);
    connect(hideActivity, &QPushButton::clicked, this, [this]() { statusBar()->hide(); });
    connect(taskController_, &TaskRuntimeController::stateChanged, this, [this](TaskRuntimeController::State state) {
        if (state != TaskRuntimeController::State::Idle) statusBar()->show();
    });
    statusBar()->hide();

    datasetPageController_ = new DatasetPageController(
        queryService, taskController_, this);
    datasetPageController_->setWorkerExecutable(workerExecutablePath());
    connect(datasetPageController_, &DatasetPageController::statusChanged,
        this, [this](const QString& text) {
            workerPill_->setStatus(text, StatusPill::Tone::Info);
            statusBar()->showMessage(text, 3000);
        });
    connect(datasetPageController_, &DatasetPageController::selectionChanged,
        this, [this]() {
            updateTrainingSelectionSummary();
        });
    connect(datasetPageController_, &DatasetPageController::repairLoopChanged,
        this, &MainWindow::setDatasetRepairLoopRows);
    deliveryEvidencePageController_ = new DeliveryEvidencePageController(
        queryService, taskController_, this);
    deliveryEvidencePageController_->setWorkerExecutable(
        workerExecutablePath());
    connect(deliveryEvidencePageController_,
        &DeliveryEvidencePageController::statusChanged,
        this, [this](const QString& text) {
            workerPill_->setStatus(text, StatusPill::Tone::Info);
            statusBar()->showMessage(text, 5000);
        });
    modelRegistryPageController_ = new ModelRegistryPageController(
        queryService, taskController_, this);
    environmentPageController_ = new EnvironmentPageController(
        queryService, taskController_, this);
    runtimeDeliveryPageController_ = new RuntimeDeliveryPageController(
        taskController_, this);
    trainingPageController_ = new TrainingPageController(
        taskController_, this);
    trainingPageController_->setQueryService(queryService);
    trainingPageController_->setWorkerExecutable(workerExecutablePath());
    connect(trainingPageController_, &TrainingPageController::runStarted,
        this, [this](const QString& taskId) {
            workerPill_->setStatus(
                tr("训练运行中"), StatusPill::Tone::Info);
            appendLog(tr("任务已启动：%1").arg(taskId));
            updateRecentTasks();
        });
    runtimeDeliveryPageController_->setWorkerExecutable(workerExecutablePath());
    runtimeDeliveryPageController_->setQueryService(queryService);
    connect(runtimeDeliveryPageController_,
        &RuntimeDeliveryPageController::runStarted, this, [this]() {
            workerPill_->setStatus(
                tr("Runtime Delivery 运行中"), StatusPill::Tone::Info);
        });
    environmentPageController_->setWorkerExecutable(workerExecutablePath());
    connect(environmentPageController_, &EnvironmentPageController::runStarted,
        this, [this]() {
            workerPill_->setStatus(tr("环境自检中"), StatusPill::Tone::Info);
        });
    modelRegistryPageController_->setWorkerExecutable(workerExecutablePath());
    connect(modelRegistryPageController_, &ModelRegistryPageController::packagesChanged,
        this, &MainWindow::syncModelPackageCombos);
    connect(modelRegistryPageController_, &ModelRegistryPageController::importStarted,
        this, [this]() {
            workerPill_->setStatus(tr("模型导入中"), StatusPill::Tone::Info);
            updateTaskCancelButton();
        });
    connect(modelRegistryPageController_, &ModelRegistryPageController::runtimeModelRequested,
        this, [this](const QString& modelPackageId) {
            showPage(DeploymentPage, tr("Runtime Delivery"));
            runtimeDeliveryPageController_
                ->selectModelPackageForInference(modelPackageId);
        });
    projectPageController_ = new ProjectPageController(
        queryService, projectSessionController_, this);
    connect(settingsPageController_, &SettingsPageController::languageChanged,
        this, [this](const QString&) { updateLanguageButtonState(); });
    connect(settingsPageController_, &SettingsPageController::defaultProjectPathChanged,
        this, [this](const QString& path) {
            projectPageController_->setDefaultRoot(
                QDir::toNativeSeparators(path), currentProjectPath().isEmpty());
            statusBar()->showMessage(tr("默认项目目录已保存。"), 3000);
        });
    readModelCoordinator_ = new WorkspaceReadModelCoordinator(this);
    eventRouter_ = new ApplicationEventRouter(&taskController_->workerClient(), this);
    connect(projectPageController_, &ProjectPageController::sessionRequestAccepted,
        this, [this]() {
            // 项目切换会改变 Artifact 根目录；丢弃旧项目未返回的异步预览。
            datasetPageController_->invalidateAsyncPreviews();
        });
    connect(projectSessionController_, &ProjectSessionController::activated,
        this, &MainWindow::activateProjectUi);
    const auto currentGeneration = [this](quint64 generation) {
        return generation == projectOpenGeneration();
    };
    connect(readModelCoordinator_, &WorkspaceReadModelCoordinator::refreshTaskList,
        this, [this, currentGeneration](quint64 generation) {
            if (currentGeneration(generation)) updateRecentTasks();
        });
    connect(readModelCoordinator_, &WorkspaceReadModelCoordinator::refreshDatasetCatalog,
        this, [this, currentGeneration](quint64 generation) {
            if (currentGeneration(generation)) updateDatasetList();
        });
    connect(readModelCoordinator_, &WorkspaceReadModelCoordinator::refreshModelRegistry,
        this, [this, currentGeneration](quint64 generation) {
            if (currentGeneration(generation)) updateModelRegistry();
        });
    connect(readModelCoordinator_, &WorkspaceReadModelCoordinator::refreshSelectedTask,
        this, [this, currentGeneration](quint64 generation) {
            if (currentGeneration(generation)) taskArtifactPageController_->refreshSelected();
        });
    connect(readModelCoordinator_, &WorkspaceReadModelCoordinator::refreshProjectSummary,
        this, [this, currentGeneration](quint64 generation) {
            if (!currentGeneration(generation)) return;
            updateProjectSummary();
            updateDashboardSummary();
        });
    connect(readModelCoordinator_, &WorkspaceReadModelCoordinator::refreshEnvironmentReport,
        this, [this, currentGeneration](quint64 generation) {
            if (!currentGeneration(generation)) return;
            if (taskController_->taskId().isValid()) {
                environmentPageController_->selectTask(
                    taskController_->taskId().toString());
            }
        });
    connect(readModelCoordinator_, &WorkspaceReadModelCoordinator::refreshDeliveryEvidence,
        this, [this, currentGeneration](quint64 generation) {
            if (currentGeneration(generation)) updateDeliveryAcceptanceSummary();
        });
    connect(sidebar_, &Sidebar::pageRequested, workspaceRouter_, &WorkspaceRouter::navigate);
    connect(workspaceRouter_, &WorkspaceRouter::pageRequested, this, &MainWindow::showPage);
    connect(eventRouter_, &ApplicationEventRouter::taskViewStateChanged,
        this, &MainWindow::handleTaskViewStateChanged);
    connect(eventRouter_, &ApplicationEventRouter::taskFactsInvalidated, this,
        [this](const QString& taskId) {
            const QString workflowKind = taskController_->workflowKind();
            const bool isEnvironmentTask = workflowKind == QStringLiteral("environment_check")
                || environmentPageController_->selectedTaskId() == taskId;
            if (isEnvironmentTask && environmentPageController_->selectTask(taskId)) {
                updateDashboardSummary();
            }
            RefreshDomains domains = RefreshDomain::TaskList
                | RefreshDomain::SelectedTask | RefreshDomain::ProjectSummary;
            if (isEnvironmentTask) {
                domains |= RefreshDomain::EnvironmentReport
                    | RefreshDomain::DeliveryEvidence;
            }
            if (workflowKind.contains(QStringLiteral("dataset"))
                || workflowKind.contains(QStringLiteral("annotation"))) {
                domains |= RefreshDomain::DatasetCatalog
                    | RefreshDomain::DeliveryEvidence;
            }
            if (workflowKind.contains(QStringLiteral("training"))) {
                domains |= RefreshDomain::ModelRegistry
                    | RefreshDomain::DeliveryEvidence;
            }
            if (workflowKind.contains(QStringLiteral("model_import"))) {
                domains |= RefreshDomain::ModelRegistry;
            }
            if (workflowKind.contains(QStringLiteral("runtime"))
                || workflowKind.contains(QStringLiteral("diagnostics"))
                || workflowKind.contains(QStringLiteral("evidence"))
                || workflowKind.contains(QStringLiteral("ocr"))) {
                domains |= RefreshDomain::DeliveryEvidence;
            }
            readModelCoordinator_->invalidate(domains);
        });
    connect(environmentPageController_, &EnvironmentPageController::changed, this, [this]() {
        runtimeDeliveryPageController_->refreshEnvironment();
        updateDashboardSummary();
    });
    connect(&workerClient(), &WorkerClient::logLine, this, &MainWindow::appendLog);
    connect(&workerClient(), &WorkerClient::connected, this, [this]() {
        workerPill_->setStatus(tr("Worker 已连接"), StatusPill::Tone::Success);
        updateTaskCancelButton();
        updateHeaderState();
    });
    connect(&workerClient(), &WorkerClient::workerLost, this,
        [this](const aitrain::TaskId& taskId) {
        if (!taskId.isValid() || !projectSessionController_
            || !projectSessionController_->isOpen()) return;
        QString recoveryError;
        if (!projectSessionController_->workspace()
            || !projectSessionController_->workspace()->recoverAfterWorkerLoss(
                taskId, &recoveryError)) {
            appendLog(uiText("Worker 异常退出后任务恢复失败：%1").arg(recoveryError));
            statusBar()->showMessage(uiText("任务恢复失败，请重新打开项目重试。"), 8000);
            return;
        }
        // WorkerLost 事件已先清理瞬态投影；持久化恢复完成后立即刷新查询
        // 服务，避免任务列表继续显示 Running/CancelRequested。
        readModelCoordinator_->invalidate(RefreshDomain::TaskList
            | RefreshDomain::SelectedTask | RefreshDomain::ProjectSummary
            | RefreshDomain::DatasetCatalog | RefreshDomain::ModelRegistry
            | RefreshDomain::EnvironmentReport | RefreshDomain::DeliveryEvidence);
    });
    connect(&workerClient(), &WorkerClient::finished, this,
        [this](WorkerClient::WorkerTerminalStatus status, const QString& message) {
        const bool ok = status == WorkerClient::WorkerTerminalStatus::Succeeded;
        const bool canceled = status == WorkerClient::WorkerTerminalStatus::Canceled;
        workerPill_->setStatus(ok ? tr("任务完成") : (canceled ? tr("任务已取消") : tr("任务失败")),
            ok ? StatusPill::Tone::Success : (canceled ? StatusPill::Tone::Warning : StatusPill::Tone::Error));
        modelRegistryPageController_->finishImport(ok, message);
        updateHeaderState();
        updateTaskCancelButton();
        appendLog(ok ? tr("任务完成：%1").arg(message)
            : (canceled ? tr("任务已取消：%1").arg(message) : tr("任务失败：%1").arg(message)));
    });
    connect(&workerClient(), &WorkerClient::idle, this, [this]() {
        updateTaskCancelButton();
        if (!closePending_) {
            return;
        }
        closePending_ = false;
        QTimer::singleShot(0, this, [this]() { close(); });
    });

    updateHeaderState();
    showPage(ProjectPage, aitrain_app::workbenchText(QStringLiteral("项目")));
    updateHeaderState();
    updateResponsiveChrome();
    updateDashboardSummary();
    updateLanguageButtonState();
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (taskController_->isRunning()) {
        closePending_ = true;
        taskController_->cancel();
        statusBar()->showMessage(uiText("正在异步取消当前任务，任务结束后关闭窗口。"), 5000);
        event->ignore();
        return;
    }
    QMainWindow::closeEvent(event);
}

WorkerClient& MainWindow::workerClient()
{
    return taskController_->workerClient();
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    updateResponsiveChrome();
}

void MainWindow::updateResponsiveChrome()
{
    if (sidebar_) {
        sidebar_->setCompact(false);
        sidebar_->setFixedWidth(width() < 1180 ? 176 : 192);
    }
}

QString MainWindow::workerExecutablePath() const
{
    const QString name =
#if defined(Q_OS_WIN)
        QStringLiteral("aitrain_worker.exe");
#else
        QStringLiteral("aitrain_worker");
#endif
    const QString appDir = QApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).filePath(name),
        QDir(appDir).filePath(QStringLiteral("../worker/") + name),
        QDir(appDir).filePath(QStringLiteral("../bin/") + name)
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QDir::cleanPath(candidate);
        }
    }
    return QDir(appDir).filePath(name);
}

QString MainWindow::defaultProjectPath() const
{
    return QDir::home().filePath(QStringLiteral("AITrainProjects/local_project"));
}

QString MainWindow::configuredDefaultProjectPath() const
{
    return settingsPageController_
        ? settingsPageController_->configuredDefaultProjectPath()
        : defaultProjectPath();
}

void MainWindow::appendLog(const QString& text)
{
    if (trainingPageController_) {
        trainingPageController_->appendLog(text);
    }
}

void MainWindow::loadCapabilityCombos()
{
    QComboBox* datasetFormatCombo = datasetPage_
        ? datasetPage_->datasetFormatCombo : nullptr;
    const QString previousDatasetFormat = datasetFormatCombo
        ? comboCurrentDataOrText(datasetFormatCombo) : QString();

    QStringList formats;
    const QVector<aitrain::CapabilityDescriptor> capabilities =
        aitrain::BuiltinCapabilityRegistry::instance().capabilities();
    for (const aitrain::CapabilityDescriptor& capability : capabilities) {
        for (const QString& format : capability.datasetFormats) {
            if (!formats.contains(format)) {
                formats.append(format);
            }
        }
    }

    if (trainingPageController_) {
        trainingPageController_->refreshCapabilities();
    }
    if (datasetFormatCombo) {
        const QSignalBlocker blocker(datasetFormatCombo);
        if (datasetFormatCombo->count() > 0) {
            datasetFormatCombo->clear();
        }
        for (const QString& format : formats) {
            datasetFormatCombo->addItem(datasetFormatLabel(format), format);
        }
        const int restoredIndex = previousDatasetFormat.isEmpty()
            ? -1 : datasetFormatCombo->findData(previousDatasetFormat);
        if (restoredIndex >= 0) {
            datasetFormatCombo->setCurrentIndex(restoredIndex);
        } else if (datasetFormatCombo->count() > 0) {
            datasetFormatCombo->setCurrentIndex(0);
        }
        datasetPageController_->state().currentFormat = currentDatasetFormat();
    }
    if (stack_ && stack_->widget(TrainingPage)
        && stack_->widget(TrainingPage)->property("workspaceInitialized").toBool()) {
        trainingPageController_->refreshCapabilities();
    }
}

QString MainWindow::currentDatasetFormat() const
{
    if (datasetPage_ && datasetPage_->datasetFormatCombo) {
        return comboCurrentDataOrText(datasetPage_->datasetFormatCombo);
    }
    return datasetPageController_->state().currentFormat;
}

QString MainWindow::currentTaskType() const
{
    return trainingPageController_
        ? trainingPageController_->currentTaskType() : QString();
}

QString MainWindow::currentProjectPath() const
{
    return projectSessionController_
        ? projectSessionController_->currentRoot() : QString();
}

QString MainWindow::currentProjectName() const
{
    return projectSessionController_
        ? projectSessionController_->currentDisplayName() : QString();
}

quint64 MainWindow::projectOpenGeneration() const
{
    return projectSessionController_ ? projectSessionController_->generation() : 0;
}

void MainWindow::updateLanguageButtonState()
{
    const QString language = aitrain_app::configuredLanguageCode();
    const auto setChecked = [&language](QToolButton* button, const QString& buttonLanguage) {
        if (!button) {
            return;
        }
        const QSignalBlocker blocker(button);
        button->setChecked(language == buttonLanguage);
    };
    setChecked(topBarZhLanguageButton_, QStringLiteral("zh_CN"));
    setChecked(topBarEnLanguageButton_, QStringLiteral("en_US"));
    if (settingsPage_) {
        settingsPage_->setLanguageCode(language);
    }
}
