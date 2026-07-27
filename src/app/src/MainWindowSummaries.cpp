#include "MainWindow.h"

#include "DashboardPageController.h"
#include "DatasetPageController.h"
#include "DatasetPage.h"
#include "ProjectPageController.h"
#include "SettingsPageController.h"
#include "RuntimeDeliveryPageController.h"
#include "TrainingPageController.h"
#include "DeliveryEvidencePageController.h"
#include "EnvironmentCheckPresenter.h"
#include "EnvironmentPageController.h"
#include "ProjectSessionController.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "WorkspaceRouter.h"
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/VisionModelRuntime.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
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
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

using namespace aitrain_app;

QString MainWindow::pageCaption(int pageIndex) const
{
    switch (pageIndex) {
    case DashboardPage: return tr("本机项目、数据、训练、模型交付状态总览");
    case ProjectPage: return tr("创建或打开本地训练项目，统一管理数据、运行和模型产物");
    case DatasetPage: return uiText("导入、校验、转换、快照，并处理质量复核样本");
    case TrainingPage: return tr("启动官方后端优先的训练实验，并监控指标、日志和产物");
    case TaskQueuePage: return tr("追踪历史任务、指标、导出记录和所有 Worker 产物");
    case ModelRegistryPage: return uiText("管理模型版本、评估报告、对比和流水线记录");
    case DeploymentPage: return uiText("导出模型，运行推理验证，并查看部署验证状态");
    case EnvironmentPage: return uiText("检查运行环境，并集中查看交付证据、诊断包和客户域 OCR 验收");
    case SystemSettingsPage: return uiText("管理内置能力、界面语言、默认目录和授权状态");
    default: return {};
    }
}

void MainWindow::showPage(int pageIndex, const QString& title)
{
    ensureWorkspacePage(pageIndex);
    stack_->setCurrentIndex(pageIndex);
    if (workspaceRouter_) {
        workspaceRouter_->synchronize(pageIndex, title);
    }
    if (pageIndex == TrainingPage) {
        loadCapabilityCombos();
    }
    pageTitle_->setText(title);
    pageCaption_->setText(QStringLiteral("%1 / %2")
        .arg(currentProjectName().isEmpty() ? uiText("本地工作台") : currentProjectName(), title));
    sidebar_->setCurrentIndex(pageIndex);
    if (pageIndex == TaskQueuePage) {
        updateRecentTasks();
    }
    if (pageIndex == ModelRegistryPage) {
        updateModelRegistry();
    }
    if (pageIndex == DatasetPage) {
        if (!datasetPageController_->state().sampleReviewSamples.isEmpty()) {
            datasetPageController_->refreshSampleReview();
        }
    }
    if (pageIndex == EnvironmentPage) {
        updateDeliveryAcceptanceSummary();
    }
    if (pageIndex == SystemSettingsPage) {
        settingsPageController_->refresh();
    }
}

void MainWindow::showDatasetTab(int tabIndex)
{
    showPage(DatasetPage, uiText("数据集"));
    if (datasetPage_ && datasetPage_->tabs) {
        datasetPage_->tabs->setCurrentIndex(tabIndex);
    }
}

void MainWindow::showDeploymentTab(int tabIndex)
{
    showPage(DeploymentPage, uiText("部署验证"));
    runtimeDeliveryPageController_->showTab(tabIndex);
}

void MainWindow::showSystemSettingsTab(int tabIndex)
{
    showPage(SystemSettingsPage, uiText("系统设置"));
    settingsPageController_->showTab(tabIndex);
}

void MainWindow::updateHeaderState()
{
    if (headerProjectLabel_) {
        headerProjectLabel_->setText(currentProjectPath().isEmpty()
            ? uiText("未打开项目")
            : currentProjectName());
    }
    if (pageContextPill_) {
        pageContextPill_->setStatus(currentProjectPath().isEmpty() ? uiText("项目未打开") : uiText("项目已就绪"),
            currentProjectPath().isEmpty() ? StatusPill::Tone::Neutral : StatusPill::Tone::Success);
    }
    const int capabilityCount = aitrain::BuiltinCapabilityRegistry::instance().capabilities().size();
    if (capabilityPill_) {
        capabilityPill_->setStatus(uiText("内置能力 %1").arg(capabilityCount),
            capabilityCount > 0 ? StatusPill::Tone::Success : StatusPill::Tone::Warning);
    }
    if (inspectorProjectLabel_) {
        inspectorProjectLabel_->setText(currentProjectPath().isEmpty()
                ? uiText("未打开项目")
                : currentProjectName());
    }
    if (inspectorCapabilityLabel_) {
        inspectorCapabilityLabel_->setText(uiText("内置能力 %1 项").arg(capabilityCount));
    }
    if (inspectorWorkerLabel_) {
        inspectorWorkerLabel_->setText(workerPill_ ? workerPill_->text() : uiText("Worker：等待连接"));
    }
    if (inspectorGpuLabel_) {
        inspectorGpuLabel_->setText(gpuPill_ ? gpuPill_->text() : uiText("GPU：等待环境检查"));
    }
    if (settingsPage_) {
        settingsPageController_->refreshCapabilities();
    }
}

void MainWindow::ensureWorkspacePage(int pageIndex)
{
    if (!stack_ || pageIndex < 0 || pageIndex >= PageCount) {
        return;
    }
    QWidget* placeholder = stack_->widget(pageIndex);
    if (!placeholder || placeholder->property("workspaceInitialized").toBool()) {
        return;
    }

    QWidget* page = nullptr;
    switch (pageIndex) {
    case ProjectPage: page = buildProjectPage(); break;
    case DatasetPage: page = buildDatasetPage(); break;
    case TrainingPage: page = buildTrainingPage(); break;
    case TaskQueuePage: page = buildTaskQueuePage(); break;
    case ModelRegistryPage: page = buildModelRegistryPage(); break;
    case DeploymentPage: page = buildDeploymentPage(); break;
    case EnvironmentPage: page = buildEnvironmentPage(); break;
    case SystemSettingsPage: page = buildSystemSettingsPage(); break;
    default: return;
    }

    page->setProperty("workspaceInitialized", true);
    stack_->removeWidget(placeholder);
    delete placeholder;
    stack_->insertWidget(pageIndex, page);

    if (pageIndex == TrainingPage) {
        refreshTrainingDefaults();
    }
    if (pageIndex == SystemSettingsPage) {
        settingsPageController_->refresh();
    }
}

void MainWindow::updateProjectSummary()
{
    projectPageController_->setContext(
        !currentProjectPath().isEmpty() && projectSessionController_
            && projectSessionController_->isOpen(),
        currentProjectName());
    projectPageController_->refresh();
}

void MainWindow::updateDeliveryAcceptanceSummary()
{
    deliveryEvidencePageController_->refresh();
}
void MainWindow::updateDashboardSummary()
{
    QString environmentText = uiText("待检测");
    bool hasMissing = false;
    bool hasWarning = false;
    bool hasChecked = false;
    const QJsonObject environmentReport = environmentPageController_->report();
    for (const QJsonValue& value
         : environmentReport.value(QStringLiteral("checks")).toArray()) {
        const QString status = value.toObject()
            .value(QStringLiteral("status")).toString();
        hasChecked = hasChecked || !status.isEmpty();
        hasMissing = hasMissing || status == QStringLiteral("missing");
        hasWarning = hasWarning || status == QStringLiteral("warning")
            || status == QStringLiteral("hardware-blocked");
    }
    if (hasChecked) {
        environmentText = hasMissing ? uiText("缺失")
            : (hasWarning ? uiText("警告") : uiText("通过"));
    }
    dashboardPageController_->setContext(
        !currentProjectPath().isEmpty() && projectSessionController_
            && projectSessionController_->isOpen(),
        currentProjectName(),
        aitrain::BuiltinCapabilityRegistry::instance().capabilities().size(),
        environmentText,
        gpuPill_ ? gpuPill_->text() : QString());
    dashboardPageController_->refresh();
    if (settingsPage_) {
        settingsPageController_->refreshCapabilities();
    }
}

void MainWindow::updateTrainingSelectionSummary()
{
    const QString datasetPath = datasetPageController_->state().currentPath;
    const QString datasetFormat = datasetPageController_->state().currentFormat;
    const bool hasCommittedIdentity = datasetPageController_->state().currentValid
        && !datasetPageController_->state().currentDatasetId.isEmpty()
        && !datasetPageController_->state().currentDatasetVersionId.isEmpty()
        && !datasetPageController_->state().currentSnapshotId.isEmpty()
        && !datasetPageController_->state().currentSnapshotArtifactId.isEmpty();
    const QString state = hasCommittedIdentity ? uiText("已提交快照")
        : (datasetPageController_->state().currentValid ? uiText("已校验") : uiText("待校验"));
    const QString snapshotId =
        datasetPageController_->state().currentSnapshotId;
    const QString snapshotArtifactId =
        datasetPageController_->state().currentSnapshotArtifactId;
    QString snapshotText = snapshotId.isEmpty()
        ? uiText("快照：尚未选择 committed Snapshot 身份")
        : uiText("快照：%1 | Artifact %2").arg(snapshotId.left(12), snapshotArtifactId.left(12));
    if (datasetPage_ && datasetPage_->datasetDetailLabel) {
        datasetPage_->datasetDetailLabel->setText(hasCommittedIdentity
            ? uiText("格式：%1 | 状态：%2 | Dataset：%3\n%4")
                .arg(datasetFormatLabel(datasetFormat), state,
                    datasetPageController_->state().currentDatasetId.left(12), snapshotText)
            : (datasetPath.isEmpty()
                ? uiText("选择已登记快照或导入数据集后显示格式、校验状态和最近报告。")
                : uiText("格式：%1 | 状态：%2\n外部数据仅停留在导入边界，请先创建并提交 Snapshot Artifact。\n%3")
                    .arg(datasetFormatLabel(datasetFormat), state, snapshotText)));
    }
}

void MainWindow::refreshTrainingDefaults()
{
    TrainingDatasetBinding binding;
    binding.datasetId = datasetPageController_->state().currentDatasetId;
    binding.datasetVersionId = datasetPageController_->state().currentDatasetVersionId;
    binding.snapshotId = datasetPageController_->state().currentSnapshotId;
    binding.snapshotArtifactId = datasetPageController_->state().currentSnapshotArtifactId;
    binding.datasetFormat = datasetPageController_->state().currentFormat;
    if (trainingPageController_) {
        trainingPageController_->setDatasetBinding(binding);
    }
    updateTrainingSelectionSummary();
}
