#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/DetectionTrainer.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
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
#include <QSettings>
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
#include <QUrl>
#include <QUuid>

using namespace aitrain_app;

MainWindow::MainWindow(const QString& licenseOwner, const QString& licenseExpiry, QWidget* parent)
    : QMainWindow(parent)
    , licenseOwner_(licenseOwner)
    , licenseExpiry_(licenseExpiry)
{
    setWindowTitle(QStringLiteral("AITrain Studio"));
    setMinimumSize(1180, 760);

    auto* central = new QWidget(this);
    auto* rootLayout = new QHBoxLayout(central);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    sidebar_ = new Sidebar;
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

    connect(sidebar_, &Sidebar::pageRequested, this, &MainWindow::showPage);
    connect(&worker_, &WorkerClient::messageReceived, this, &MainWindow::handleWorkerMessage);
    connect(&worker_, &WorkerClient::logLine, this, &MainWindow::appendLog);
    connect(&worker_, &WorkerClient::connected, this, [this]() {
        workerPill_->setStatus(tr("Worker 已连接"), StatusPill::Tone::Success);
        updateHeaderState();
    });
    connect(&worker_, &WorkerClient::idle, this, [this]() {
        QTimer::singleShot(0, this, &MainWindow::startNextQueuedTask);
    });
    connect(&worker_, &WorkerClient::finished, this, [this](bool ok, const QString& message) {
        progressBar_->setValue(ok ? 100 : progressBar_->value());
        if (trainingPhaseLabel_ && !state_.training.currentTaskId.isEmpty()) {
            trainingPhaseLabel_->setText(ok
                ? uiText("阶段：快照 -> 训练 -> 验证 -> 导出 -> 完成 | 当前：完成")
                : uiText("阶段：快照 -> 训练 -> 验证 -> 导出 -> 完成 | 当前：失败 | %1").arg(message));
        }
        if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingEtaValue")); label && ok) {
            label->setText(QStringLiteral("0s"));
        }
        workerPill_->setStatus(ok ? tr("任务完成") : tr("任务失败"),
            ok ? StatusPill::Tone::Success : StatusPill::Tone::Error);
        updateHeaderState();
        appendLog(ok ? tr("任务完成：%1").arg(message) : tr("任务失败：%1").arg(message));
        if (!state_.dataset.currentConversionTaskId.isEmpty()) {
            if (datasetConversionProgressBar_ && ok) {
                datasetConversionProgressBar_->setValue(100);
            }
            setDatasetConversionFormRunning(false);
            if (datasetConversionStatusLabel_) {
                datasetConversionStatusLabel_->setText(ok
                        ? uiText("数据集转换已完成。")
                        : uiText("数据集转换失败：%1").arg(message));
            }
            if (!ok) {
                appendDatasetConversionLog(uiText("数据集转换失败：%1").arg(message));
            }
            state_.dataset.currentConversionTaskId.clear();
        }
        const QString kind;
        const QString path;
        if (!state_.training.currentTaskId.isEmpty()) {
            QString error;
            repository_.updateTaskState(state_.training.currentTaskId, ok ? aitrain::TaskState::Completed : aitrain::TaskState::Failed, message, &error);
            if (ok) {
                updateExperimentRunSummary(state_.training.currentTaskId);
            }
            state_.training.currentTaskId.clear();
            updateRecentTasks();
            updateSelectedTaskDetails();
            updateModelRegistry();
        } else if (kind == QStringLiteral("export") && exportResultLabel_) {
            exportResultLabel_->setText(tr("导出完成：%1").arg(QDir::toNativeSeparators(path)));
        } else if (kind == QStringLiteral("inference_overlay") && inferenceOverlayLabel_) {
            loadInferenceOverlay(inferenceOverlayLabel_, path);
        } else if (kind == QStringLiteral("inference_predictions") && inferenceResultLabel_) {
            inferenceResultLabel_->setText(inferenceSummaryFromPredictions(path));
        }
    });

    refreshBuiltInCapabilities();
    aitrain_app::translateWidgetTree(this);
    showPage(TrainingPage, tr("训练实验"));
    updateHeaderState();
    updateResponsiveChrome();
    updateDashboardSummary();
    updateLanguageButtonState();
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
    QSettings settings;
    const QString configured = settings.value(defaultProjectPathSettingsKey(), defaultProjectPath()).toString().trimmed();
    if (configured.isEmpty()) {
        return defaultProjectPath();
    }
    return QDir::cleanPath(QDir::fromNativeSeparators(configured));
}

void MainWindow::ensureProjectSubdirs(const QString& rootPath)
{
    QDir root(rootPath);
    root.mkpath(QStringLiteral("."));
    root.mkpath(QStringLiteral("datasets"));
    root.mkpath(QStringLiteral("runs"));
    root.mkpath(QStringLiteral("models"));
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
    if (capabilityCombo_) {
        const QSignalBlocker blocker(capabilityCombo_);
        capabilityCombo_->clear();
        for (const aitrain::CapabilityDescriptor& capability : capabilities) {
            capabilityCombo_->addItem(capability.displayName, capability.id);
            for (const QString& format : capability.datasetFormats) {
                if (!formats.contains(format)) {
                    formats.append(format);
                }
            }
        }
        if (!currentCapability.isEmpty()) {
            const int index = capabilityCombo_->findData(currentCapability);
            if (index >= 0) {
                capabilityCombo_->setCurrentIndex(index);
            }
        }
    } else {
        for (const aitrain::CapabilityDescriptor& capability : capabilities) {
            for (const QString& format : capability.datasetFormats) {
                if (!formats.contains(format)) {
                    formats.append(format);
                }
            }
        }
    }
    if (datasetFormatCombo_) {
        const QSignalBlocker blocker(datasetFormatCombo_);
        datasetFormatCombo_->clear();
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

    QSettings settings;
    settings.setValue(defaultProjectPathSettingsKey(), normalized);
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

void MainWindow::openLocalDirectory(const QString& path)
{
    const QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(path.trimmed()));
    if (normalized.isEmpty() || normalized == QStringLiteral(".")) {
        statusBar()->showMessage(uiText("当前未打开项目。"), 3000);
        return;
    }

    const QFileInfo info(normalized);
    const QString directory = info.isDir() ? info.absoluteFilePath() : info.absolutePath();
    if (!QDir(directory).exists()) {
        statusBar()->showMessage(uiText("目录不存在：%1").arg(QDir::toNativeSeparators(directory)), 5000);
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(directory));
}

void MainWindow::copyLocalPath(const QString& path, const QString& label)
{
    const QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(path.trimmed()));
    if (normalized.isEmpty() || normalized == QStringLiteral(".")) {
        statusBar()->showMessage(uiText("当前未打开项目。"), 3000);
        return;
    }

    QApplication::clipboard()->setText(QDir::toNativeSeparators(normalized));
    statusBar()->showMessage(uiText("路径已复制：%1").arg(label), 3000);
}
