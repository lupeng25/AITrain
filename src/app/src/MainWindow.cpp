#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/PluginInterfaces.h"

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
    sidebar_->addItem(uiText("样本复核"), SampleReviewPage);
    sidebar_->addItem(tr("训练实验"), TrainingPage);
    sidebar_->addItem(tr("任务与产物"), TaskQueuePage);
    sidebar_->addSection(tr("模型交付"));
    sidebar_->addItem(tr("模型库"), ModelRegistryPage);
    sidebar_->addItem(tr("评估报告"), EvaluationReportsPage);
    sidebar_->addItem(tr("模型导出"), ConversionPage);
    sidebar_->addItem(tr("推理验证"), InferencePage);
    sidebar_->addItem(uiText("交付验收"), DeliveryAcceptancePage);
    sidebar_->addSection(tr("系统"));
    sidebar_->addItem(tr("插件"), PluginsPage);
    sidebar_->addItem(tr("环境"), EnvironmentPage);
    sidebar_->addItem(uiText("设置"), SettingsPage);
    rootLayout->addWidget(sidebar_);

    auto* content = new QWidget;
    auto* contentLayout = new QVBoxLayout(content);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(0);
    contentLayout->addWidget(buildTopBar());

    stack_ = new QStackedWidget;
    stack_->addWidget(buildDashboardPage());
    stack_->addWidget(buildProjectPage());
    stack_->addWidget(buildDatasetPage());
    stack_->addWidget(buildSampleReviewPage());
    stack_->addWidget(buildTrainingPage());
    stack_->addWidget(buildTaskQueuePage());
    stack_->addWidget(buildModelRegistryPage());
    stack_->addWidget(buildEvaluationReportsPage());
    stack_->addWidget(buildConversionPage());
    stack_->addWidget(buildInferencePage());
    stack_->addWidget(buildDeliveryAcceptancePage());
    stack_->addWidget(buildPluginsPage());
    stack_->addWidget(buildEnvironmentPage());
    stack_->addWidget(buildSettingsPage());
    contentLayout->addWidget(stack_, 1);

    rootLayout->addWidget(content, 1);
    setCentralWidget(central);

    statusBar()->showMessage(tr("就绪"));

    connect(sidebar_, &Sidebar::pageRequested, this, &MainWindow::showPage);
    connect(&worker_, &WorkerClient::messageReceived, this, &MainWindow::handleWorkerMessage);
    connect(&worker_, &WorkerClient::logLine, this, &MainWindow::appendLog);
    connect(&worker_, &WorkerClient::connected, this, [this]() {
        workerPill_->setStatus(tr("Worker 已连接"), StatusPill::Tone::Success);
    });
    connect(&worker_, &WorkerClient::idle, this, &MainWindow::startNextQueuedTask);
    connect(&worker_, &WorkerClient::finished, this, [this](bool ok, const QString& message) {
        progressBar_->setValue(ok ? 100 : progressBar_->value());
        workerPill_->setStatus(ok ? tr("任务完成") : tr("任务失败"),
            ok ? StatusPill::Tone::Success : StatusPill::Tone::Error);
        appendLog(ok ? tr("任务完成：%1").arg(message) : tr("任务失败：%1").arg(message));
        if (!currentDatasetConversionTaskId_.isEmpty()) {
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
            currentDatasetConversionTaskId_.clear();
        }
        const QString kind;
        const QString path;
        if (!currentTaskId_.isEmpty()) {
            QString error;
            repository_.updateTaskState(currentTaskId_, ok ? aitrain::TaskState::Completed : aitrain::TaskState::Failed, message, &error);
            if (ok) {
                updateExperimentRunSummary(currentTaskId_);
            }
            currentTaskId_.clear();
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
        startNextQueuedTask();
    });

    refreshPlugins();
    aitrain_app::translateWidgetTree(this);
    showPage(DashboardPage, tr("总览"));
    updateHeaderState();
    updateDashboardSummary();
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

QStringList MainWindow::pluginSearchPaths() const
{
    const QString appDir = QApplication::applicationDirPath();
    return {
        QDir(appDir).filePath(QStringLiteral("plugins/models")),
        QDir(appDir).filePath(QStringLiteral("../plugins/models")),
        QDir(appDir).filePath(QStringLiteral("../../plugins/models"))
    };
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
        logEdit_->append(QStringLiteral("[%1] %2").arg(QTime::currentTime().toString(QStringLiteral("HH:mm:ss")), text));
    }
}

void MainWindow::loadPluginCombos()
{
    const QString currentPlugin = pluginCombo_ ? pluginCombo_->currentData().toString() : QString();
    if (pluginCombo_) {
        pluginCombo_->clear();
    }
    if (datasetFormatCombo_) {
        datasetFormatCombo_->clear();
    }

    QStringList formats;
    for (auto* plugin : pluginManager_.plugins()) {
        const aitrain::PluginManifest manifest = plugin->manifest();
        if (pluginCombo_) {
            pluginCombo_->addItem(manifest.name, manifest.id);
        }
        for (const QString& format : manifest.datasetFormats) {
            if (!formats.contains(format)) {
                formats.append(format);
            }
        }
    }
    if (datasetFormatCombo_) {
        for (const QString& format : formats) {
            datasetFormatCombo_->addItem(datasetFormatLabel(format), format);
        }
    }
    if (pluginCombo_ && !currentPlugin.isEmpty()) {
        const int index = pluginCombo_->findData(currentPlugin);
        if (index >= 0) {
            pluginCombo_->setCurrentIndex(index);
        }
    }
    refreshTrainingDefaults();
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
