#include "MainWindow.h"

#include "ApplicationEventRouter.h"
#include "TaskExecutionController.h"

#include "EvaluationReportView.h"
#include "DiagnosticBundlePresenter.h"
#include "DatasetCatalogPresenter.h"
#include "DeliveryEvidencePresenter.h"
#include "EnvironmentCheckPresenter.h"
#include "ApplicationSettingsService.h"
#include "ModelRegistryPresenter.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "ProjectSummaryPresenter.h"
#include "WorkspaceRouter.h"
#include "TaskArtifactPresenter.h"
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
    , queryService_(&workspace_)
    , licenseOwner_(licenseOwner)
    , licenseExpiry_(licenseExpiry)
{
    projectSummaryPresenter_ = new ProjectSummaryPresenter(&queryService_, this);
    taskArtifactPresenter_ = new TaskArtifactPresenter(&queryService_, this);
    diagnosticBundlePresenter_ = new DiagnosticBundlePresenter(&queryService_, this);
    environmentCheckPresenter_ = new EnvironmentCheckPresenter(&queryService_, this);
    modelRegistryPresenter_ = new ModelRegistryPresenter(&queryService_, this);
    datasetCatalogPresenter_ = new DatasetCatalogPresenter(&queryService_, this);
    deliveryEvidencePresenter_ = new DeliveryEvidencePresenter(&queryService_, this);
    setWindowTitle(QStringLiteral("AITrain Studio"));
    setMinimumSize(1180, 760);

    auto* central = new QWidget(this);
    auto* rootLayout = new QHBoxLayout(central);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    sidebar_ = new Sidebar;
    sidebar_->setObjectName(QStringLiteral("WorkspaceSidebar"));
    sidebar_->addSection(tr("工作台"));
    sidebar_->addItem(tr("总览"), DashboardPage);
    sidebar_->addItem(tr("项目"), ProjectPage);
    sidebar_->addSection(tr("数据与训练"));
    sidebar_->addItem(tr("数据集"), DatasetPage);
    sidebar_->addItem(tr("训练实验"), TrainingPage);
    sidebar_->addItem(tr("任务与产物"), TaskQueuePage);
    sidebar_->addSection(uiText("模型与部署"));
    sidebar_->addItem(tr("模型库"), ModelRegistryPage);
    sidebar_->addItem(uiText("部署验证"), DeploymentPage);
    sidebar_->addSection(tr("系统"));
    sidebar_->addItem(tr("环境"), EnvironmentPage);
    sidebar_->addItem(uiText("系统设置"), SystemSettingsPage);
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
    inspector_ = qobject_cast<QFrame*>(buildInspector());
    rootLayout->addWidget(inspector_);
    setCentralWidget(central);

    statusBar()->showMessage(tr("就绪"));
    statusBar()->setVisible(false);

    workspaceRouter_ = new WorkspaceRouter(PageCount, this);
    eventRouter_ = new ApplicationEventRouter(&worker_, this);
    taskController_ = new TaskExecutionController(&worker_, this);
    connect(sidebar_, &Sidebar::pageRequested, workspaceRouter_, &WorkspaceRouter::navigate);
    connect(workspaceRouter_, &WorkspaceRouter::pageRequested, this, &MainWindow::showPage);
    connect(eventRouter_, &ApplicationEventRouter::taskViewStateChanged,
        this, &MainWindow::handleTaskViewStateChanged);
    connect(eventRouter_, &ApplicationEventRouter::taskFactsInvalidated, this,
        [this](const QString& taskId) {
            const bool isEnvironmentTask = activeWorkflowKind_ == QStringLiteral("environment_check")
                || (environmentCheckPresenter_
                    && environmentCheckPresenter_->viewModel().taskId == taskId);
            if (isEnvironmentTask && environmentCheckPresenter_
                && environmentCheckPresenter_->selectTask(taskId)) {
                refreshEnvironmentReportView();
                updateEnvironmentSummary();
                updateDashboardSummary();
            }
            updateRecentTasks();
            updateSelectedTaskDetails();
            updateProjectSummary();
            updateDashboardSummary();
            updateDeliveryAcceptanceSummary();
            updateModelRegistry();
        });
    connect(environmentCheckPresenter_, &EnvironmentCheckPresenter::changed, this, [this]() {
        refreshEnvironmentReportView();
        updateEnvironmentSummary();
        updateDashboardSummary();
    });
    connect(deliveryEvidencePresenter_, &DeliveryEvidencePresenter::changed, this,
        &MainWindow::renderDeliveryAcceptanceSummary);
    connect(deliveryEvidencePresenter_, &DeliveryEvidencePresenter::queryFailed, this,
        [this](const QString&) { renderDeliveryAcceptanceSummary(); });
    connect(&worker_, &WorkerClient::logLine, this, &MainWindow::appendLog);
    connect(&worker_, &WorkerClient::connected, this, [this]() {
        workerPill_->setStatus(tr("Worker 已连接"), StatusPill::Tone::Success);
        updateTaskCancelButton();
        updateHeaderState();
    });
    connect(&worker_, &WorkerClient::workerLost, this,
        [this](const aitrain::TaskId& taskId) {
        if (!taskId.isValid() || !workspace_.isOpen()) return;
        QString recoveryError;
        if (!workspace_.recoverAfterWorkerLoss(taskId, &recoveryError)) {
            appendLog(uiText("Worker 异常退出后任务恢复失败：%1").arg(recoveryError));
            statusBar()->showMessage(uiText("任务恢复失败，请重新打开项目重试。"), 8000);
            return;
        }
        // WorkerLost 事件已先清理瞬态投影；持久化恢复完成后立即刷新查询
        // 服务，避免任务列表继续显示 Running/CancelRequested。
        updateRecentTasks();
        updateSelectedTaskDetails();
        updateProjectSummary();
        updateDashboardSummary();
        updateDeliveryAcceptanceSummary();
        updateModelRegistry();
    });
    connect(&worker_, &WorkerClient::finished, this,
        [this](WorkerClient::WorkerTerminalStatus status, const QString& message) {
        const bool ok = status == WorkerClient::WorkerTerminalStatus::Succeeded;
        const bool canceled = status == WorkerClient::WorkerTerminalStatus::Canceled;
        workerPill_->setStatus(ok ? tr("任务完成") : (canceled ? tr("任务已取消") : tr("任务失败")),
            ok ? StatusPill::Tone::Success : (canceled ? StatusPill::Tone::Warning : StatusPill::Tone::Error));
        if (modelImportInProgress_) {
            modelImportInProgress_ = false;
            if (modelImportResultLabel_) {
                modelImportResultLabel_->setText(ok
                    ? uiText(" 模型导入完成。")
                    : uiText(" 模型导入失败：%1").arg(message));
            }
            updateModelRegistry();
        }
        updateHeaderState();
        updateTaskCancelButton();
        appendLog(ok ? tr("任务完成：%1").arg(message)
            : (canceled ? tr("任务已取消：%1").arg(message) : tr("任务失败：%1").arg(message)));
    });
    connect(&worker_, &WorkerClient::idle, this, [this]() {
        updateTaskCancelButton();
        if (!closePending_) {
            return;
        }
        closePending_ = false;
        QTimer::singleShot(0, this, [this]() { close(); });
    });

    refreshBuiltInCapabilities();
    showPage(TrainingPage, tr("训练实验"));
    updateHeaderState();
    updateResponsiveChrome();
    updateDashboardSummary();
    updateLanguageButtonState();
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (worker_.isRunning()) {
        closePending_ = true;
        worker_.cancel();
        statusBar()->showMessage(uiText("正在异步取消当前任务，任务结束后关闭窗口。"), 5000);
        event->ignore();
        return;
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    updateResponsiveChrome();
}

void MainWindow::updateResponsiveChrome()
{
    const int width = this->width();
    if (sidebar_) {
        const bool compact = width < 1366;
        sidebar_->setCompact(compact);
        sidebar_->setFixedWidth(compact ? 72 : (width >= 1600 ? 216 : 200));
    }
    if (inspectorToggleButton_ && !inspectorUserOverride_) {
        applyingResponsiveChrome_ = true;
        inspectorToggleButton_->setChecked(width >= 1366);
        applyingResponsiveChrome_ = false;
    }
    if (inspector_) {
        inspector_->setVisible(!inspectorToggleButton_ || inspectorToggleButton_->isChecked());
        inspector_->setFixedWidth(width >= 1600 ? 304 : 272);
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
    ApplicationSettingsService settings;
    const QString configured = settings.defaultProjectPath(defaultProjectPath()).trimmed();
    if (configured.isEmpty()) {
        return defaultProjectPath();
    }
    return QDir::cleanPath(QDir::fromNativeSeparators(configured));
}

void MainWindow::appendLog(const QString& text)
{
    if (logEdit_) {
        QString line = text;
        constexpr int maxLogLineChars = 8000;
        if (line.size() > maxLogLineChars) {
            line = line.left(maxLogLineChars) + QStringLiteral(" ... [log_truncated]");
        }
        logEdit_->append(QStringLiteral("[%1] %2").arg(QTime::currentTime().toString(QStringLiteral("HH:mm:ss")), line));
    }
}

void MainWindow::loadCapabilityCombos()
{
    const QString currentCapability = capabilityCombo_ ? capabilityCombo_->currentData().toString() : QString();
    const QString previousDatasetFormat = datasetFormatCombo_ ? comboCurrentDataOrText(datasetFormatCombo_) : QString();

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

    if (capabilityCombo_) {
        bool capabilityItemsMatch = capabilityCombo_->count() == capabilities.size();
        for (int index = 0; capabilityItemsMatch && index < capabilities.size(); ++index) {
            if (capabilityCombo_->itemData(index).toString() != capabilities.at(index).id
                || capabilityCombo_->itemText(index) != capabilities.at(index).displayName) {
                capabilityItemsMatch = false;
            }
        }

        if (!capabilityItemsMatch) {
            const QSignalBlocker blocker(capabilityCombo_);
            if (capabilityCombo_->count() > 0) {
                capabilityCombo_->clear();
            }
            for (const aitrain::CapabilityDescriptor& capability : capabilities) {
                capabilityCombo_->addItem(capability.displayName, capability.id);
            }
            if (!currentCapability.isEmpty()) {
                const int index = capabilityCombo_->findData(currentCapability);
                if (index >= 0) {
                    capabilityCombo_->setCurrentIndex(index);
                }
            }
        }
    }
    if (datasetFormatCombo_) {
        const QSignalBlocker blocker(datasetFormatCombo_);
        if (datasetFormatCombo_->count() > 0) {
            datasetFormatCombo_->clear();
        }
        for (const QString& format : formats) {
            datasetFormatCombo_->addItem(datasetFormatLabel(format), format);
        }
        const int restoredIndex = previousDatasetFormat.isEmpty() ? -1 : datasetFormatCombo_->findData(previousDatasetFormat);
        if (restoredIndex >= 0) {
            datasetFormatCombo_->setCurrentIndex(restoredIndex);
        } else if (datasetFormatCombo_->count() > 0) {
            datasetFormatCombo_->setCurrentIndex(0);
        }
        state_.dataset.currentFormat = currentDatasetFormat();
    }
    if (stack_ && stack_->widget(TrainingPage)
        && stack_->widget(TrainingPage)->property("workspaceInitialized").toBool()) {
        refreshTrainingDefaults();
    }
}

QString MainWindow::currentDatasetFormat() const
{
    return comboCurrentDataOrText(datasetFormatCombo_);
}

QString MainWindow::currentTaskType() const
{
    return comboCurrentDataOrText(taskTypeCombo_);
}

QString MainWindow::currentTaskKindFilter() const
{
    return taskKindFilterCombo_ ? taskKindFilterCombo_->currentData().toString() : QString();
}

QString MainWindow::currentTaskStateFilter() const
{
    return taskStateFilterCombo_ ? taskStateFilterCombo_->currentData().toString() : QString();
}

void MainWindow::storeLanguagePreference(const QString& languageCode)
{
    const QString previous = aitrain_app::configuredLanguageCode();
    aitrain_app::storeLanguageCode(languageCode);
    updateLanguageButtonState();
    if (previous != aitrain_app::configuredLanguageCode()) {
        QMessageBox::information(this, uiText("界面语言"), uiText("语言设置已保存，重启 AITrain Studio 后生效。"));
    }
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
    setChecked(settingsZhLanguageButton_, QStringLiteral("zh_CN"));
    setChecked(settingsEnLanguageButton_, QStringLiteral("en_US"));
}

void MainWindow::storeDefaultProjectPathPreference(const QString& path)
{
    const QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(path.trimmed()));
    if (normalized.isEmpty() || normalized == QStringLiteral(".")) {
        QMessageBox::warning(this, uiText("默认项目目录"), uiText("目录不能为空。"));
        return;
    }

    ApplicationSettingsService settings;
    settings.setDefaultProjectPath(normalized);
    const QString native = QDir::toNativeSeparators(normalized);
    if (settingsDefaultProjectPathEdit_) {
        settingsDefaultProjectPathEdit_->setText(native);
    }
    if (projectRootEdit_ && currentProjectPath_.isEmpty()) {
        projectRootEdit_->setText(native);
    }
    if (settingsDefaultProjectPathStatusLabel_) {
        settingsDefaultProjectPathStatusLabel_->setText(uiText("默认项目目录已保存。"));
    }
    statusBar()->showMessage(uiText("默认项目目录已保存。"), 3000);
}
