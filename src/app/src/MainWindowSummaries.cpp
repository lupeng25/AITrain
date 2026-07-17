#include "MainWindow.h"

#include "ProjectSummaryPresenter.h"
#include "DeliveryEvidencePresenter.h"
#include "EnvironmentCheckPresenter.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "WorkspaceRouter.h"
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
    case SystemSettingsPage: return uiText("管理内置能力、界面语言、默认目录、授权状态和本地路径");
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
        .arg(currentProjectName_.isEmpty() ? uiText("本地工作台") : currentProjectName_, title));
    sidebar_->setCurrentIndex(pageIndex);
    if (pageIndex == TaskQueuePage) {
        updateRecentTasks();
    }
    if (pageIndex == ModelRegistryPage) {
        updateModelRegistry();
    }
    if (pageIndex == DatasetPage) {
        if (!state_.dataset.sampleReviewSamples.isEmpty()) {
            refreshSampleReviewTable();
        }
    }
    if (pageIndex == EnvironmentPage) {
        updateEnvironmentSummary();
        updateDeliveryAcceptanceSummary();
    }
    if (pageIndex == SystemSettingsPage) {
        updateCapabilitySummary();
        updateSettingsSummary();
    }
}

void MainWindow::showDatasetTab(int tabIndex)
{
    showPage(DatasetPage, uiText("数据集"));
    if (datasetTabs_) {
        datasetTabs_->setCurrentIndex(tabIndex);
    }
}

void MainWindow::showDeploymentTab(int tabIndex)
{
    showPage(DeploymentPage, uiText("部署验证"));
    if (deploymentTabs_) {
        deploymentTabs_->setCurrentIndex(tabIndex);
    }
}

void MainWindow::showSystemSettingsTab(int tabIndex)
{
    showPage(SystemSettingsPage, uiText("系统设置"));
    if (systemSettingsTabs_) {
        systemSettingsTabs_->setCurrentIndex(tabIndex);
    }
}

void MainWindow::updateHeaderState()
{
    if (headerProjectLabel_) {
        headerProjectLabel_->setText(currentProjectPath_.isEmpty()
            ? uiText("未打开项目")
            : currentProjectName_);
    }
    if (pageContextPill_) {
        pageContextPill_->setStatus(currentProjectPath_.isEmpty() ? uiText("项目未打开") : uiText("项目已就绪"),
            currentProjectPath_.isEmpty() ? StatusPill::Tone::Neutral : StatusPill::Tone::Success);
    }
    const int capabilityCount = aitrain::BuiltinCapabilityRegistry::instance().capabilities().size();
    if (capabilityPill_) {
        capabilityPill_->setStatus(uiText("内置能力 %1").arg(capabilityCount),
            capabilityCount > 0 ? StatusPill::Tone::Success : StatusPill::Tone::Warning);
    }
    if (dashboardCapabilityValue_) {
        dashboardCapabilityValue_->setText(QString::number(capabilityCount));
    }
    if (inspectorProjectLabel_) {
        inspectorProjectLabel_->setText(currentProjectPath_.isEmpty()
                ? uiText("未打开项目")
                : currentProjectName_);
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
    updateCapabilitySummary();
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
        refreshBuiltInCapabilities();
    }
}

void MainWindow::updateProjectSummary()
{
    const bool workspaceOpen = !currentProjectPath_.isEmpty() && workspace_.isOpen();
    if (workspaceOpen) {
        projectSummaryPresenter_->refresh();
    } else {
        projectSummaryPresenter_->clear();
    }
    const ProjectSummaryViewModel& summary = projectSummaryPresenter_->viewModel();
    const bool hasProject = workspaceOpen && summary.available;
    if (projectConsoleStatusLabel_) {
        projectConsoleStatusLabel_->setText(hasProject
            ? uiText("已打开：%1").arg(currentProjectName_)
            : (workspaceOpen
                    ? uiText(" 项目汇总读取失败：%1").arg(projectSummaryPresenter_->lastError())
                    : uiText("未打开项目。")));
    }
    if (projectPathSummaryLabel_) {
        projectPathSummaryLabel_->setText(hasProject
            ? compactPathForStatus(currentProjectPath_, 74)
            : uiText("未打开"));
        projectPathSummaryLabel_->setToolTip(hasProject
            ? QDir::toNativeSeparators(currentProjectPath_)
            : QString());
    }
    if (projectSqliteSummaryLabel_) {
        projectSqliteSummaryLabel_->setText(hasProject ? uiText(" 已连接") : uiText("未连接"));
    }

    if (projectDatasetSummaryLabel_) {
        projectDatasetSummaryLabel_->setText(QString::number(summary.datasetCount));
        projectDatasetSummaryLabel_->setToolTip(uiText("版本 %1，快照 %2")
            .arg(summary.datasetVersionCount)
            .arg(summary.datasetSnapshotCount));
    }
    if (projectTaskSummaryLabel_) {
        projectTaskSummaryLabel_->setText(QString::number(summary.taskCount));
        projectTaskSummaryLabel_->setToolTip(uiText("活动 %1，成功 %2，失败 %3，取消 %4")
            .arg(summary.activeTaskCount)
            .arg(summary.succeededTaskCount)
            .arg(summary.failedTaskCount)
            .arg(summary.canceledTaskCount));
    }
    if (projectExportSummaryLabel_) {
        projectExportSummaryLabel_->setText(QString::number(summary.modelPackageCount));
        projectExportSummaryLabel_->setToolTip(uiText("已校验模型包 %1；已提交产物 %2")
            .arg(summary.verifiedModelPackageCount)
            .arg(summary.committedArtifactCount));
    }
}

void MainWindow::updateCapabilitySummary()
{
    const QVector<aitrain::CapabilityDescriptor> capabilities =
        aitrain::BuiltinCapabilityRegistry::instance().capabilities();
    QStringList datasetFormats;
    QStringList exportFormats;
    int gpuCapabilities = 0;
    for (const aitrain::CapabilityDescriptor& capability : capabilities) {
        datasetFormats.append(capability.datasetFormats);
        for (const QString& backendId : capability.backendIds) {
            const aitrain::BackendDescriptor backend =
                aitrain::BuiltinCapabilityRegistry::instance().backend(backendId);
            exportFormats.append(backend.exportFormats);
            if (backend.devicePolicy == QStringLiteral("gpu_required")
                || backend.devicePolicy == QStringLiteral("gpu_recommended")) {
                ++gpuCapabilities;
            }
        }
    }

    if (capabilityConsoleStatusLabel_) {
        capabilityConsoleStatusLabel_->setText(capabilities.isEmpty()
            ? uiText("内置能力注册表为空。")
            : uiText("已注册 %1 个内置能力。").arg(capabilities.size()));
    }
    if (capabilitySourceLabel_) {
        capabilitySourceLabel_->setText(uiText("能力来源：编译期内置注册表"));
        capabilitySourceLabel_->setToolTip(uiText("能力由编译期注册表提供。"));
    }
    if (capabilityCountSummaryLabel_) {
        capabilityCountSummaryLabel_->setText(QString::number(capabilities.size()));
    }
    if (capabilityDatasetFormatSummaryLabel_) {
        capabilityDatasetFormatSummaryLabel_->setText(QString::number(uniqueStringCount(datasetFormats)));
        capabilityDatasetFormatSummaryLabel_->setToolTip(compactListSummary(datasetFormats, 12));
    }
    if (capabilityExportFormatSummaryLabel_) {
        capabilityExportFormatSummaryLabel_->setText(QString::number(uniqueStringCount(exportFormats)));
        capabilityExportFormatSummaryLabel_->setToolTip(compactListSummary(exportFormats, 12));
    }
    if (capabilityGpuSummaryLabel_) {
        capabilityGpuSummaryLabel_->setText(QString::number(gpuCapabilities));
    }
}

void MainWindow::updateEnvironmentSummary()
{
    int ok = 0;
    int warning = 0;
    int missing = 0;
    int blocked = 0;
    int unchecked = 0;
    if (environmentTable_) {
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            const QString state = environmentTable_->item(row, 1) ? environmentTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
            if (state == QStringLiteral("ok")) {
                ++ok;
            } else if (state == QStringLiteral("hardware-blocked")) {
                ++warning;
                ++blocked;
            } else if (state == QStringLiteral("warning")) {
                ++warning;
            } else if (state == QStringLiteral("missing")) {
                ++missing;
            } else {
                ++unchecked;
            }
        }
    }
    if (environmentOkSummaryLabel_) {
        environmentOkSummaryLabel_->setText(QString::number(ok));
    }
    if (environmentWarningSummaryLabel_) {
        environmentWarningSummaryLabel_->setText(QString::number(warning));
    }
    if (environmentMissingSummaryLabel_) {
        environmentMissingSummaryLabel_->setText(QString::number(missing));
    }
    if (environmentUncheckedSummaryLabel_) {
        environmentUncheckedSummaryLabel_->setText(QString::number(unchecked));
    }
    if (environmentConsoleStatusLabel_) {
        if (missing > 0) {
            environmentConsoleStatusLabel_->setText(uiText("发现 %1 项缺失，相关能力会被阻塞。").arg(missing));
        } else if (warning > 0) {
            environmentConsoleStatusLabel_->setText(
                blocked > 0
                    ? uiText("发现 %1 项警告（其中 %2 项硬件受限），可继续但需要关注。").arg(warning).arg(blocked)
                    : uiText("发现 %1 项警告，可继续但需要关注。").arg(warning));
        } else if (unchecked > 0) {
            environmentConsoleStatusLabel_->setText(uiText("尚有 %1 项未检测。").arg(unchecked));
        } else {
            environmentConsoleStatusLabel_->setText(uiText("环境自检通过。"));
        }
    }
}

void MainWindow::refreshEnvironmentReportView()
{
    if (!environmentTable_) {
        return;
    }

    const QJsonObject report = environmentCheckPresenter_
        ? environmentCheckPresenter_->viewModel().report : QJsonObject();
    environmentTable_->setRowCount(0);

    const auto addRow = [this](const QString& name, const QString& status,
        const QString& message) {
        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        auto* nameItem = new QTableWidgetItem(name);
        auto* statusItem = new QTableWidgetItem(status == QStringLiteral("ok")
                ? uiText("通过")
                : status == QStringLiteral("missing")
                    ? uiText("缺失")
                    : status == QStringLiteral("hardware-blocked")
                        ? uiText("硬件受限")
                        : status == QStringLiteral("warning")
                            ? uiText("警告") : uiText("未检测"));
        statusItem->setData(Qt::UserRole, status);
        auto* messageItem = new QTableWidgetItem(message.isEmpty()
            ? uiText("未提供说明。") : message);
        environmentTable_->setItem(row, 0, nameItem);
        environmentTable_->setItem(row, 1, statusItem);
        environmentTable_->setItem(row, 2, messageItem);
    };

    if (report.isEmpty()) {
        const QStringList rows = {
            QStringLiteral("NVIDIA Driver"), QStringLiteral("CUDA Runtime"),
            QStringLiteral("cuDNN"), QStringLiteral("TensorRT"),
            QStringLiteral("ONNX Runtime"), QStringLiteral("Qt Runtime Modules"),
            QStringLiteral("内置能力"), QStringLiteral("Worker")};
        for (const QString& name : rows) {
            addRow(name, QStringLiteral("unchecked"), uiText("点击执行环境自检。"));
        }
        return;
    }

    for (const QJsonValue& value : report.value(QStringLiteral("checks")).toArray()) {
        const QJsonObject check = value.toObject();
        addRow(check.value(QStringLiteral("name")).toString(),
            check.value(QStringLiteral("status")).toString(),
            check.value(QStringLiteral("message")).toString());
    }
    const QJsonObject profiles = report.value(QStringLiteral("profiles")).toObject();
    for (auto it = profiles.constBegin(); it != profiles.constEnd(); ++it) {
        const QJsonObject profile = it.value().toObject();
        QString message = profile.value(QStringLiteral("message")).toString();
        if (message.isEmpty()) {
            QStringList hints;
            for (const QJsonValue& hint : profile.value(QStringLiteral("repairHints")).toArray()) {
                hints.append(hint.toString());
            }
            message = hints.join(QStringLiteral("；"));
        }
        addRow(profile.value(QStringLiteral("title")).toString(it.key()),
            profile.value(QStringLiteral("status")).toString(), message);
    }
}

void MainWindow::updateSettingsSummary()
{
    if (settingsDefaultProjectPathEdit_) {
        settingsDefaultProjectPathEdit_->setText(QDir::toNativeSeparators(configuredDefaultProjectPath()));
    }
    if (settingsCurrentProjectPathLabel_) {
        settingsCurrentProjectPathLabel_->setText(currentProjectPath_.isEmpty()
            ? uiText("未打开项目")
            : compactPathForStatus(currentProjectPath_, 92));
        settingsCurrentProjectPathLabel_->setToolTip(currentProjectPath_.isEmpty()
            ? QString()
            : QDir::toNativeSeparators(currentProjectPath_));
    }
    updateLanguageButtonState();
}

void MainWindow::updateDeliveryAcceptanceSummary()
{
    if (!deliveryAcceptanceTable_) {
        return;
    }
    if (deliveryAcceptanceTable_->rowCount() == 0) {
        const QStringList stages = {
            uiText("本机 RC"),
            uiText("Clean Windows"),
            uiText("TensorRT"),
            uiText("客户域 OCR"),
            uiText("包体完整性"),
            uiText("部署验证"),
            uiText("诊断包")
        };
        for (const QString& stage : stages) {
            const int row = deliveryAcceptanceTable_->rowCount();
            deliveryAcceptanceTable_->insertRow(row);
            deliveryAcceptanceTable_->setItem(row, 0, new QTableWidgetItem(stage));
            deliveryAcceptanceTable_->setItem(row, 1, new QTableWidgetItem(QStringLiteral("not_run")));
            deliveryAcceptanceTable_->setItem(row, 2, new QTableWidgetItem(QString()));
            deliveryAcceptanceTable_->setItem(row, 3, new QTableWidgetItem(uiText("等待导入外部结果或运行对应 Worker/脚本。")));
        }
    }

    if (deliveryEvidencePresenter_ && workspace_.isOpen()) {
        deliveryEvidencePresenter_->refresh();
        for (const auto& evidence : deliveryEvidencePresenter_->viewModel().records) {
            QString stage = evidence.evidenceKind;
            const QString normalized = evidence.evidenceKind.toLower();
            if (normalized.contains(QStringLiteral("clean"))) stage = uiText("Clean Windows");
            else if (normalized.contains(QStringLiteral("tensor"))) stage = uiText("TensorRT");
            else if (normalized.contains(QStringLiteral("ocr"))) stage = uiText("客户域 OCR");
            int row = -1;
            for (int index = 0; index < deliveryAcceptanceTable_->rowCount(); ++index) {
                if (deliveryAcceptanceTable_->item(index, 0)
                    && deliveryAcceptanceTable_->item(index, 0)->text() == stage) {
                    row = index;
                    break;
                }
            }
            if (row < 0) {
                row = deliveryAcceptanceTable_->rowCount();
                deliveryAcceptanceTable_->insertRow(row);
                deliveryAcceptanceTable_->setItem(row, 0, new QTableWidgetItem(stage));
            }
            deliveryAcceptanceTable_->setItem(row, 1,
                new QTableWidgetItem(evidence.verified ? QStringLiteral("passed") : QStringLiteral("collected")));
            deliveryAcceptanceTable_->setItem(row, 2,
                new QTableWidgetItem(evidence.evidenceArtifactId.toString()));
            deliveryAcceptanceTable_->setItem(row, 3,
                new QTableWidgetItem(evidence.limitations.join(QStringLiteral(" | "))));
        }
    }

    int passed = 0;
    int blocked = 0;
    int hardwareBlocked = 0;
    int notRun = 0;
    int collected = 0;
    for (int row = 0; row < deliveryAcceptanceTable_->rowCount(); ++row) {
        const QString status = deliveryAcceptanceTable_->item(row, 1)
            ? deliveryAcceptanceTable_->item(row, 1)->text()
            : QString();
        if (status == QStringLiteral("passed")) {
            ++passed;
        } else if (status == QStringLiteral("blocked") || status == QStringLiteral("failed")) {
            ++blocked;
        } else if (status == QStringLiteral("hardware-blocked")) {
            ++hardwareBlocked;
        } else if (status == QStringLiteral("collected") || status == QStringLiteral("imported")) {
            ++collected;
        } else {
            ++notRun;
        }
    }
    if (deliveryAcceptanceSummaryLabel_) {
        deliveryAcceptanceSummaryLabel_->setText(uiText("验收状态：passed %1 / blocked %2 / hardware-blocked %3 / collected %4 / not-run %5")
            .arg(passed)
            .arg(blocked)
            .arg(hardwareBlocked)
            .arg(collected)
            .arg(notRun));
    }
}

void MainWindow::updateDashboardSummary()
{
    updateProjectSummary();
    const ProjectSummaryViewModel& summary = projectSummaryPresenter_->viewModel();
    const bool hasProject = !currentProjectPath_.isEmpty()
        && workspace_.isOpen() && summary.available;
    if (dashboardProjectValue_) {
        dashboardProjectValue_->setText(hasProject ? currentProjectName_ : uiText("未打开"));
    }
    if (projectLabel_) {
        projectLabel_->setText(hasProject
            ? uiText("当前项目：%1").arg(QDir::toNativeSeparators(currentProjectPath_))
            : uiText("未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    }

    if (dashboardDatasetValue_) {
        dashboardDatasetValue_->setText(hasProject
            ? QStringLiteral("%1 / %2").arg(summary.datasetSnapshotCount).arg(summary.datasetCount)
            : QStringLiteral("0"));
        dashboardDatasetValue_->setToolTip(uiText("快照 / 数据集；版本 %1")
            .arg(summary.datasetVersionCount));
    }
    if (dashboardTaskValue_) {
        dashboardTaskValue_->setText(QString::number(summary.taskCount));
        dashboardTaskValue_->setToolTip(uiText("活动任务 %1；成功 %2；失败 %3；取消 %4")
            .arg(summary.activeTaskCount)
            .arg(summary.succeededTaskCount)
            .arg(summary.failedTaskCount)
            .arg(summary.canceledTaskCount));
    }
    if (dashboardModelValue_) {
        dashboardModelValue_->setText(hasProject
            ? QStringLiteral("%1 / %2").arg(summary.verifiedModelPackageCount).arg(summary.modelPackageCount)
            : QStringLiteral("0"));
    }
    if (dashboardCapabilityValue_) {
        dashboardCapabilityValue_->setText(QString::number(aitrain::BuiltinCapabilityRegistry::instance().capabilities().size()));
    }

    QString environmentText = uiText("待检测");
    if (environmentTable_ && environmentTable_->rowCount() > 0) {
        bool hasMissing = false;
        bool hasWarning = false;
        bool hasChecked = false;
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            const QString state = environmentTable_->item(row, 1) ? environmentTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
            hasChecked = hasChecked || !state.isEmpty();
            hasMissing = hasMissing || state == QStringLiteral("missing");
            hasWarning = hasWarning || state == QStringLiteral("warning") || state == QStringLiteral("hardware-blocked");
        }
        if (hasChecked) {
            environmentText = hasMissing ? uiText("缺失")
                : (hasWarning ? uiText("警告") : uiText("通过"));
        }
    }
    if (dashboardEnvironmentValue_) {
        dashboardEnvironmentValue_->setText(environmentText);
    }
    updateCapabilitySummary();
    updateEnvironmentSummary();

    if (dashboardNextStepLabel_) {
        QString nextStep;
        if (!hasProject) {
            nextStep = uiText("先创建或打开一个本地项目。项目目录会集中保存数据集索引、任务历史、训练报告和模型产物。");
        } else if (summary.datasetSnapshotCount == 0) {
            nextStep = uiText("下一步：导入数据并创建数据集快照。训练工作流只消费已登记的不可变快照。");
        } else if (summary.taskCount == 0) {
            nextStep = uiText("下一步：进入训练实验，选择已登记的数据集快照并启动官方后端工作流。");
        } else if (summary.modelPackageCount == 0) {
            nextStep = uiText("下一步：在任务与产物中检查工作流产物，并完成模型包登记后进入部署验证。");
        } else {
            nextStep = uiText("项目已记录数据集快照、任务与模型包。可继续进入部署验证或追加实验。");
        }
        dashboardNextStepLabel_->setText(nextStep);
    }
}

void MainWindow::updateTrainingSelectionSummary()
{
    const QString datasetPath = !state_.dataset.currentPath.isEmpty()
        ? state_.dataset.currentPath
        : QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    const QString datasetFormat = !state_.dataset.currentFormat.isEmpty()
        ? state_.dataset.currentFormat
        : currentDatasetFormat();
    const bool hasCommittedIdentity = state_.dataset.currentValid
        && !state_.dataset.currentDatasetId.isEmpty()
        && !state_.dataset.currentDatasetVersionId.isEmpty()
        && !state_.dataset.currentSnapshotId.isEmpty()
        && !state_.dataset.currentSnapshotArtifactId.isEmpty();
    const QString state = hasCommittedIdentity ? uiText("已提交快照")
        : (state_.dataset.currentValid ? uiText("已校验") : uiText("待校验"));
    const QString fullPathText = datasetPath.isEmpty() ? QString() : QDir::toNativeSeparators(datasetPath);
    const QString datasetName = datasetPath.isEmpty() ? QString() : QFileInfo(datasetPath).fileName();
    const QString headerPathText = datasetPath.isEmpty()
        ? uiText("未选择")
        : (datasetName.isEmpty() ? compactPathForStatus(datasetPath, 36) : datasetName);
    const QString detailPathText = datasetPath.isEmpty() ? uiText("未选择") : compactPathForStatus(datasetPath, 92);
    const QString snapshotId = dataQualitySnapshotIdEdit_ ? dataQualitySnapshotIdEdit_->text().trimmed() : QString();
    const QString snapshotArtifactId = dataQualitySnapshotArtifactIdEdit_ ? dataQualitySnapshotArtifactIdEdit_->text().trimmed() : QString();
    QString snapshotText = snapshotId.isEmpty()
        ? uiText("快照：尚未选择 committed Snapshot 身份")
        : uiText("快照：%1 | Artifact %2").arg(snapshotId.left(12), snapshotArtifactId.left(12));
    bool datasetReady = state_.dataset.currentValid
        && (datasetPath.isEmpty() || state_.dataset.currentPath == datasetPath)
        && state_.dataset.currentFormat == datasetFormat;
    datasetReady = datasetReady && !snapshotId.isEmpty() && !snapshotArtifactId.isEmpty();

    if (trainingDatasetSummaryLabel_) {
        trainingDatasetSummaryLabel_->setText(hasCommittedIdentity
            ? uiText("当前数据集：%1 | %2\nDataset %3 / Version %4\n%5")
                .arg(datasetFormatLabel(datasetFormat), state,
                    state_.dataset.currentDatasetId.left(12),
                    state_.dataset.currentDatasetVersionId.left(12), snapshotText)
            : (datasetPath.isEmpty()
                ? uiText("当前数据集：未选择。请先选择已登记快照或导入外部数据集。")
                : uiText("当前数据集：%1 | %2 | %3\n%4")
                    .arg(datasetFormatLabel(datasetFormat), state, headerPathText, snapshotText)));
        trainingDatasetSummaryLabel_->setToolTip(hasCommittedIdentity
            ? snapshotText
            : (datasetPath.isEmpty() ? QString() : uiText("数据集：%1\n%2").arg(fullPathText, snapshotText)));
    }
    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(hasCommittedIdentity
            ? uiText("格式：%1 | 状态：%2 | Dataset：%3\n%4")
                .arg(datasetFormatLabel(datasetFormat), state,
                    state_.dataset.currentDatasetId.left(12), snapshotText)
            : (datasetPath.isEmpty()
                ? uiText("选择已登记快照或导入数据集后显示格式、校验状态和最近报告。")
                : uiText("格式：%1 | 状态：%2 | 路径：%3\n%4")
                    .arg(datasetFormatLabel(datasetFormat), state, detailPathText, snapshotText)));
    }
    if (trainingBackendHintLabel_ && trainingBackendCombo_) {
        trainingBackendHintLabel_->setText(trainingBackendDescription(trainingBackendCombo_->currentData().toString()));
    }
    const QString visibleBackend = trainingBackendCombo_
        ? trainingBackendCombo_->currentData().toString().trimmed().toLower()
        : QString();
    if (auto* yoloPanel = findChild<QWidget*>(QStringLiteral("YoloOfficialArgsGroup"))) {
        yoloPanel->setVisible(yoloPanel->property("advancedExpanded").toBool()
            && visibleBackend.startsWith(QStringLiteral("ultralytics_yolo_")));
    }
    if (auto* smpPanel = findChild<QWidget*>(QStringLiteral("SmpSemanticArgsGroup"))) {
        smpPanel->setVisible(smpPanel->property("advancedExpanded").toBool()
            && visibleBackend == QStringLiteral("smp_semantic_segmentation"));
    }
    if (auto* anomalyPanel = findChild<QWidget*>(QStringLiteral("AnomalyDetectionArgsGroup"))) {
        anomalyPanel->setVisible(anomalyPanel->property("advancedExpanded").toBool()
            && (visibleBackend == QStringLiteral("anomalib_patchcore")
                || visibleBackend == QStringLiteral("anomalib_efficientad")));
    }
    if (auto* caption = findChild<QLabel*>(QStringLiteral("TrainingLiveCaption_TrainingMapValue"))) {
        caption->setText((visibleBackend == QStringLiteral("anomalib_patchcore") || visibleBackend == QStringLiteral("anomalib_efficientad"))
            ? QStringLiteral("Score/F1")
            : QStringLiteral("mAP"));
    }
    if (trainingRunSummaryLabel_) {
        const QString backend = trainingBackendCombo_
            ? trainingBackendCombo_->currentData().toString()
            : defaultBackendForTask(currentTaskType());
        const QString model = modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString();
        trainingRunSummaryLabel_->setText(uiText("运行摘要：%1 | 后端 %2 | 模型 %3 | epoch %4 / batch %5 / image %6")
            .arg(taskTypeLabel(currentTaskType()),
                backend.isEmpty() ? uiText("未选择") : backend,
                model.isEmpty() ? uiText("默认") : model,
                epochsEdit_ ? epochsEdit_->text() : QStringLiteral("-"),
                batchEdit_ ? batchEdit_->text() : QStringLiteral("-"),
                imageSizeEdit_ ? imageSizeEdit_->text() : QStringLiteral("-")));
            trainingRunSummaryLabel_->setToolTip(uiText("训练只消费四重身份：Dataset %1 / Version %2 / Snapshot %3 / Artifact %4")
            .arg(dataQualityDatasetIdEdit_ ? dataQualityDatasetIdEdit_->text().trimmed() : QString(),
                dataQualityDatasetVersionIdEdit_ ? dataQualityDatasetVersionIdEdit_->text().trimmed() : QString(),
                snapshotId,
                snapshotArtifactId));
    }
}

void MainWindow::refreshTrainingDefaults()
{
    if (!trainingBackendCombo_ || !modelPresetCombo_) {
        updateTrainingSelectionSummary();
        return;
    }

    const QString datasetFormat = !state_.dataset.currentFormat.isEmpty()
        ? state_.dataset.currentFormat
        : currentDatasetFormat();
    QString preferredCapability;
    QString preferredTask;
    QString preferredBackend;

    if (datasetFormat == QStringLiteral("yolo_detection")) {
        preferredCapability = QStringLiteral("yolo");
        preferredTask = QStringLiteral("detection");
        preferredBackend = QStringLiteral("ultralytics_yolo_detect");
    } else if (datasetFormat == QStringLiteral("yolo_segmentation")) {
        preferredCapability = QStringLiteral("yolo");
        preferredTask = QStringLiteral("segmentation");
        preferredBackend = QStringLiteral("ultralytics_yolo_segment");
    } else if (datasetFormat == QStringLiteral("yolo_obb")) {
        preferredCapability = QStringLiteral("yolo");
        preferredTask = QStringLiteral("obb_detection");
        preferredBackend = QStringLiteral("ultralytics_yolo_obb");
    } else if (datasetFormat == QStringLiteral("semantic_segmentation_mask")) {
        preferredCapability = QStringLiteral("semantic_segmentation");
        preferredTask = QStringLiteral("semantic_segmentation");
        preferredBackend = QStringLiteral("smp_semantic_segmentation");
    } else if (datasetFormat == QStringLiteral("anomaly_folder")) {
        preferredCapability = QStringLiteral("anomaly_detection");
        preferredTask = QStringLiteral("anomaly_detection");
        preferredBackend = QStringLiteral("anomalib_patchcore");
    } else if (datasetFormat == QStringLiteral("paddleocr_det")) {
        preferredCapability = QStringLiteral("paddleocr");
        preferredTask = QStringLiteral("ocr_detection");
        preferredBackend = QStringLiteral("paddleocr_det_official");
    } else if (datasetFormat == QStringLiteral("paddleocr_rec")) {
        preferredCapability = QStringLiteral("paddleocr");
        preferredTask = QStringLiteral("ocr_recognition");
        preferredBackend = QStringLiteral("paddleocr_rec_official");
    }

    if (!preferredCapability.isEmpty() && capabilityCombo_) {
        QSignalBlocker block(capabilityCombo_);
        setComboCurrentData(capabilityCombo_, preferredCapability);
    }

    if (taskTypeCombo_) {
        const QString currentTask = currentTaskType();
        QSignalBlocker block(taskTypeCombo_);
        taskTypeCombo_->clear();
        const aitrain::CapabilityDescriptor capability = capabilityCombo_
            ? aitrain::BuiltinCapabilityRegistry::instance().capability(capabilityCombo_->currentData().toString())
            : aitrain::CapabilityDescriptor();
        if (!capability.id.isEmpty()) {
            addTaskTypeItems(taskTypeCombo_, capability.taskTypes);
        }
        const QString targetTask = preferredTask.isEmpty() ? currentTask : preferredTask;
        const int taskIndex = taskTypeCombo_->findData(targetTask);
        if (taskIndex >= 0) {
            taskTypeCombo_->setCurrentIndex(taskIndex);
        } else if (taskTypeCombo_->count() > 0) {
            taskTypeCombo_->setCurrentIndex(0);
        }
    }

    if (preferredBackend.isEmpty()) {
        preferredBackend = defaultBackendForTask(currentTaskType());
    }
    {
        QSignalBlocker block(trainingBackendCombo_);
        setComboCurrentData(trainingBackendCombo_, preferredBackend);
    }
    const QString backend = trainingBackendCombo_->currentData().toString();
    {
        QSignalBlocker block(modelPresetCombo_);
        modelPresetCombo_->clear();
        modelPresetCombo_->addItems(modelPresetItemsForBackend(backend));
        modelPresetCombo_->setCurrentText(defaultModelForBackend(backend));
    }
    updateTrainingSelectionSummary();
}
