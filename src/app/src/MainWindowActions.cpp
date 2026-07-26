#include "MainWindow.h"
#include "TaskRuntimeController.h"
#include "WorkspaceReadModelCoordinator.h"
#include "EnvironmentCheckPresenter.h"

#include "DatasetConversionUiModel.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QApplication>
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
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUuid>

using namespace aitrain_app;

void MainWindow::createProject()
{
    if (taskController_->isRunning()) {
        QMessageBox::warning(this, uiText("项目"),
            uiText("Worker 正在执行任务，请等待任务终态后再切换项目。"));
        return;
    }
    if (projectOpenInProgress_) {
        return;
    }
    const QString projectName = projectNameEdit_->text().trimmed();
    const QString projectPath = QDir::fromNativeSeparators(projectRootEdit_->text().trimmed());
    if (projectName.isEmpty() || projectPath.isEmpty()) {
        QMessageBox::warning(this, uiText("项目"), uiText("项目名称和目录不能为空。"));
        return;
    }

    pendingProjectName_ = projectName;
    pendingProjectPath_ = projectPath;
    const quint64 generation = ++projectOpenGeneration_;
    // 项目切换会改变 Artifact 根目录；丢弃旧项目尚未返回的样本复核
    // 读取结果，防止其在新项目激活后污染 Dataset 工作区状态。
    ++sampleReviewPreviewGeneration_;
    if (sampleReviewPreviewGeneration_ == 0) ++sampleReviewPreviewGeneration_;
    projectOpenInProgress_ = true;
    setProjectOpenUiBusy(true);
    if (projectConsoleStatusLabel_) {
        projectConsoleStatusLabel_->setText(uiText("正在后台预检项目恢复，请稍候…"));
    }
    statusBar()->showMessage(uiText("正在后台打开项目：%1").arg(compactPathForStatus(projectPath)), 5000);

    probeProjectOpenAsync(this, projectPath,
        [this, projectName, projectPath, generation](
            const ProjectOpenProbeResult& result) {
            finishProjectOpen(projectName, projectPath, generation, result);
        });
}

void MainWindow::setProjectOpenUiBusy(bool busy)
{
    if (projectOpenButton_) projectOpenButton_->setEnabled(!busy);
    if (projectNameEdit_) projectNameEdit_->setEnabled(!busy);
    if (projectRootEdit_) projectRootEdit_->setEnabled(!busy);
}

void MainWindow::finishProjectOpen(const QString& projectName, const QString& projectPath,
    quint64 generation, const ProjectOpenProbeResult& result)
{
    if (!projectOpenInProgress_ || generation != projectOpenGeneration_
        || projectPath != pendingProjectPath_) {
        return;
    }
    projectOpenInProgress_ = false;
    setProjectOpenUiBusy(false);
    pendingProjectName_.clear();
    pendingProjectPath_.clear();

    if (!result.succeeded || !result.prepared.isValid()) {
        const QString message = result.error.isEmpty()
            ? uiText("候选项目工作区预检失败。") : result.error;
        if (projectConsoleStatusLabel_) {
            projectConsoleStatusLabel_->setText(uiText("项目预检失败：%1").arg(message));
        }
        statusBar()->showMessage(message, 8000);
        QMessageBox::critical(this, uiText("项目"), uiText("无法打开项目工作区：%1").arg(message));
        return;
    }

    // 预检线程已经完成完整恢复；这里仅在凭证仍匹配时把当前 GUI session
    // 绑定到同一路径。openPrepared 不会把后台线程的 QSqlDatabase 带回 GUI。
    QString error;
    if (!workspace_.openPrepared(result.prepared, &error)) {
        const QString message = error.isEmpty()
            ? uiText("项目打开凭证已失效，需要重新尝试。") : error;
        if (projectConsoleStatusLabel_) {
            projectConsoleStatusLabel_->setText(uiText("项目激活失败：%1").arg(message));
        }
        statusBar()->showMessage(message, 8000);
        QMessageBox::critical(this, uiText("项目"), uiText("无法激活项目工作区：%1").arg(message));
        return;
    }

    currentProjectName_ = projectName;
    currentProjectPath_ = result.prepared.canonicalRoot;
    readModelCoordinator_->setGeneration(generation);
    clearSelectedTaskDetails();
    state_.dataset = DatasetWorkbenchState();
    for (QLineEdit* field : {dataQualityDatasetIdEdit_, dataQualityDatasetVersionIdEdit_,
            dataQualitySnapshotIdEdit_, dataQualitySnapshotArtifactIdEdit_,
            splitSourceDatasetIdEdit_, splitSourceDatasetVersionIdEdit_,
            splitSourceSnapshotIdEdit_, splitSourceSnapshotArtifactIdEdit_}) {
        if (field) field->clear();
    }

    projectLabel_->setText(uiText("当前项目：%1（工作区已就绪）").arg(currentProjectName_));
    if (dashboardProjectValue_) {
        dashboardProjectValue_->setText(currentProjectName_);
    }
    updateHeaderState();
    readModelCoordinator_->invalidate(RefreshDomain::TaskList
        | RefreshDomain::DatasetCatalog | RefreshDomain::ModelRegistry
        | RefreshDomain::ProjectSummary | RefreshDomain::DeliveryEvidence);
    updateSettingsSummary();
    refreshTrainingDefaults();
    statusBar()->showMessage(uiText("项目已打开：%1").arg(currentProjectName_), 5000);
}

void MainWindow::runEnvironmentCheck()
{
    if (taskController_->isRunning()) {
        QMessageBox::warning(this, uiText("环境自检"), uiText("Worker 正在执行任务，稍后再运行环境自检。"));
        return;
    }

    if (!workspace_.isOpen() || currentProjectPath_.isEmpty()) {
        QMessageBox::warning(this, uiText("环境自检"), uiText("请先打开项目。"));
        return;
    }
    if (environmentCheckPresenter_) {
        environmentCheckPresenter_->clear();
    }
    if (environmentTable_) {
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            auto* statusItem = new QTableWidgetItem(uiText("检测中"));
            environmentTable_->setItem(row, 1, statusItem);
            environmentTable_->setItem(row, 2, new QTableWidgetItem(uiText("等待 Worker 返回结果。")));
        }
    }
    updateEnvironmentSummary();

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("environment_check");
    QString error;
    aitrain::worker_protocol::EnvironmentCheckCommand environmentCommand;
    environmentCommand.context.taskId = taskId;
    environmentCommand.context.projectRoot = currentProjectPath_;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{environmentCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        QMessageBox::critical(this, uiText("环境自检"), error);
        return;
    }
    workerPill_->setStatus(uiText("环境自检中"), StatusPill::Tone::Info);
}

void MainWindow::refreshBuiltInCapabilities()
{
    const QVector<aitrain::CapabilityDescriptor> capabilities =
        aitrain::BuiltinCapabilityRegistry::instance().capabilities();
    if (capabilityTable_) {
        capabilityTable_->setRowCount(0);
        if (capabilities.isEmpty()) {
            capabilityTable_->setRowCount(1);
            capabilityTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无内置能力")));
            for (int column = 1; column < capabilityTable_->columnCount(); ++column) {
                capabilityTable_->setItem(0, column, new QTableWidgetItem(uiText("内置能力注册表为空。")));
            }
        }
        for (const aitrain::CapabilityDescriptor& capability : capabilities) {
            const int row = capabilityTable_->rowCount();
            capabilityTable_->insertRow(row);
            capabilityTable_->setItem(row, 0, new QTableWidgetItem(capability.id));
            capabilityTable_->setItem(row, 1, new QTableWidgetItem(capability.displayName));
            capabilityTable_->setItem(row, 2, new QTableWidgetItem(uiText("内置")));
            capabilityTable_->setItem(row, 3, new QTableWidgetItem(compactListSummary(capability.taskTypes, 4)));
            capabilityTable_->setItem(row, 4, new QTableWidgetItem(compactListSummary(capability.datasetFormats, 4)));
            capabilityTable_->setItem(row, 5, new QTableWidgetItem(compactListSummary(capability.backendIds, 4)));
            capabilityTable_->setItem(row, 6, new QTableWidgetItem(uiText("内置")));
        }
    }
    updateHeaderState();
    updateCapabilitySummary();
    updateDashboardSummary();
}
