#include "MainWindow.h"
#include "TaskExecutionController.h"
#include "EnvironmentCheckPresenter.h"

#include "DatasetConversionUiModel.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/DetectionTrainer.h"
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
    currentProjectName_ = projectNameEdit_->text().trimmed();
    currentProjectPath_ = QDir::fromNativeSeparators(projectRootEdit_->text().trimmed());
    if (currentProjectName_.isEmpty() || currentProjectPath_.isEmpty()) {
        QMessageBox::warning(this, uiText("项目"), uiText("项目名称和目录不能为空。"));
        return;
    }

    QString error;
    if (!workspace_.open(currentProjectPath_, &error)) {
        QMessageBox::critical(this, uiText("项目"), uiText("无法打开项目工作区：%1").arg(error));
        return;
    }
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
    updateRecentTasks();
    updateDatasetList();
    updateModelRegistry();
    updateDashboardSummary();
    updateSettingsSummary();
    refreshTrainingDefaults();
    statusBar()->showMessage(uiText("项目已打开：%1").arg(currentProjectName_), 5000);
}

void MainWindow::runEnvironmentCheck()
{
    if (worker_.isRunning()) {
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
