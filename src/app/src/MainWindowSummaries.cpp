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
    stack_->setCurrentIndex(pageIndex);
    pageTitle_->setText(title);
    pageCaption_->setText(pageCaption(pageIndex));
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

void MainWindow::showModelWorkspaceTab(int tabIndex)
{
    showPage(ModelRegistryPage, uiText("模型库"));
    if (modelWorkspaceTabs_) {
        modelWorkspaceTabs_->setCurrentIndex(tabIndex);
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
    headerProjectLabel_->setText(currentProjectPath_.isEmpty()
        ? tr("项目：未打开")
        : tr("项目：%1").arg(currentProjectName_));
    const int capabilityCount = aitrain::BuiltinCapabilityRegistry::instance().capabilities().size();
    capabilityPill_->setStatus(uiText("内置能力 %1").arg(capabilityCount), capabilityCount > 0 ? StatusPill::Tone::Success : StatusPill::Tone::Warning);
    if (dashboardCapabilityValue_) {
        dashboardCapabilityValue_->setText(QString::number(capabilityCount));
    }
    updateCapabilitySummary();
}

void MainWindow::updateEnvironmentTable(const QJsonObject& payload)
{
    const QJsonArray checks = payload.value(QStringLiteral("checks")).toArray();
    if (!environmentTable_) {
        return;
    }

    bool hasMissing = false;
    bool hasWarning = false;
    const auto markStatus = [&hasMissing, &hasWarning](const QString& status) {
        if (status == QStringLiteral("missing")) {
            hasMissing = true;
        } else if (status == QStringLiteral("warning") || status == QStringLiteral("hardware-blocked")) {
            hasWarning = true;
        }
    };
    const auto appendRow = [this, &payload, &markStatus](const QString& name, const QString& status, const QString& message, const QJsonObject& details = {}) {
        markStatus(status);
        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        environmentTable_->setItem(row, 0, new QTableWidgetItem(name));
        auto* statusItem = new QTableWidgetItem(environmentStatusLabel(status));
        statusItem->setData(Qt::UserRole, status);
        environmentTable_->setItem(row, 1, statusItem);
        environmentTable_->setItem(row, 2, new QTableWidgetItem(message));

        if (repository_.isOpen()) {
            aitrain::EnvironmentCheckRecord record;
            record.name = name;
            record.status = status;
            record.message = message;
            if (!details.isEmpty()) {
                record.detailsJson = QString::fromUtf8(QJsonDocument(details).toJson(QJsonDocument::Compact));
            }
            record.checkedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
            QString error;
            repository_.insertEnvironmentCheck(record, &error);
        }
    };

    environmentTable_->setRowCount(0);
    for (const QJsonValue& value : checks) {
        const QJsonObject check = value.toObject();
        const QString name = check.value(QStringLiteral("name")).toString();
        const QString status = check.value(QStringLiteral("status")).toString();
        const QString message = check.value(QStringLiteral("message")).toString();
        appendRow(name, status, message, check.value(QStringLiteral("details")).toObject());
    }

    const QJsonObject profiles = payload.value(QStringLiteral("profiles")).toObject();
    for (auto it = profiles.constBegin(); it != profiles.constEnd(); ++it) {
        const QJsonObject profile = it.value().toObject();
        const QString title = profile.value(QStringLiteral("title")).toString(it.key());
        const QString status = profile.value(QStringLiteral("status")).toString(QStringLiteral("warning"));
        const QJsonArray repairHints = profile.value(QStringLiteral("repairHints")).toArray();
        QStringList hintText;
        for (const QJsonValue& value : repairHints) {
            const QString text = value.toString().trimmed();
            if (!text.isEmpty()) {
                hintText.append(text);
            }
        }
        if (hintText.isEmpty()) {
            hintText.append(uiText("暂无修复建议。"));
        }
        appendRow(
            QStringLiteral("Profile / %1").arg(title),
            status,
            hintText.join(QStringLiteral(" | ")),
            profile);

        const QJsonArray profileChecks = profile.value(QStringLiteral("checks")).toArray();
        for (const QJsonValue& checkValue : profileChecks) {
            const QJsonObject check = checkValue.toObject();
            appendRow(
                QStringLiteral("%1 / %2").arg(title, check.value(QStringLiteral("name")).toString()),
                check.value(QStringLiteral("status")).toString(QStringLiteral("warning")),
                check.value(QStringLiteral("message")).toString(),
                check.value(QStringLiteral("details")).toObject());
        }
    }

    {
        const int capabilityCount = aitrain::BuiltinCapabilityRegistry::instance().capabilities().size();
        appendRow(uiText("内置能力"), capabilityCount > 0 ? QStringLiteral("ok") : QStringLiteral("warning"),
            capabilityCount > 0
                ? uiText("已注册 %1 个内置能力。").arg(capabilityCount)
                : uiText("内置能力注册表为空。"));
    }

    const StatusPill::Tone tone = hasMissing ? StatusPill::Tone::Error : (hasWarning ? StatusPill::Tone::Warning : StatusPill::Tone::Success);
    const QString text = hasMissing ? uiText("环境缺失") : (hasWarning ? uiText("环境警告") : uiText("环境通过"));
    gpuPill_->setStatus(text, tone);
    workerPill_->setStatus(uiText("Worker 空闲"), StatusPill::Tone::Neutral);
    gpuLabel_->setText(uiText("GPU / 运行时：%1").arg(text));
    if (dashboardEnvironmentValue_) {
        dashboardEnvironmentValue_->setText(hasMissing ? uiText("缺失") : (hasWarning ? uiText("警告") : uiText("通过")));
    }
    updateEnvironmentSummary();
    updateDashboardSummary();
    statusBar()->showMessage(uiText("环境自检完成"), 5000);
}

void MainWindow::updateProjectSummary()
{
    const bool hasProject = !currentProjectPath_.isEmpty() && repository_.isOpen();
    if (projectConsoleStatusLabel_) {
        projectConsoleStatusLabel_->setText(hasProject
            ? uiText("已打开：%1").arg(currentProjectName_)
            : uiText("未打开项目。"));
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
        projectSqliteSummaryLabel_->setText(hasProject ? uiText("已连接") : uiText("未连接"));
    }

    int datasetCount = 0;
    int taskCount = 0;
    int exportCount = 0;
    if (hasProject) {
        QString error;
        datasetCount = repository_.recentDatasets(200, &error).size();
        taskCount = repository_.recentTasks(200, &error).size();
        exportCount = repository_.recentExports(200, &error).size();
    }
    if (projectDatasetSummaryLabel_) {
        projectDatasetSummaryLabel_->setText(QString::number(datasetCount));
    }
    if (projectTaskSummaryLabel_) {
        projectTaskSummaryLabel_->setText(QString::number(taskCount));
    }
    if (projectExportSummaryLabel_) {
        projectExportSummaryLabel_->setText(QString::number(exportCount));
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
    const bool hasProject = !currentProjectPath_.isEmpty() && repository_.isOpen();
    if (dashboardProjectValue_) {
        dashboardProjectValue_->setText(hasProject ? currentProjectName_ : uiText("未打开"));
    }
    if (projectLabel_) {
        projectLabel_->setText(hasProject
            ? uiText("当前项目：%1").arg(QDir::toNativeSeparators(currentProjectPath_))
            : uiText("未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    }

    int datasetCount = 0;
    int validDatasetCount = 0;
    int taskCount = 0;
    int exportCount = 0;
    int modelVersionCount = 0;
    if (hasProject) {
        QString error;
        const QVector<aitrain::DatasetRecord> datasets = repository_.recentDatasets(200, &error);
        datasetCount = datasets.size();
        for (const aitrain::DatasetRecord& dataset : datasets) {
            if (dataset.validationStatus == QStringLiteral("valid")) {
                ++validDatasetCount;
            }
        }
        taskCount = repository_.recentTasks(200, &error).size();
        exportCount = repository_.recentExports(200, &error).size();
        modelVersionCount = repository_.recentModelVersions(200, &error).size();
    }

    if (dashboardDatasetValue_) {
        dashboardDatasetValue_->setText(hasProject
            ? QStringLiteral("%1 / %2").arg(validDatasetCount).arg(datasetCount)
            : QStringLiteral("0"));
    }
    if (dashboardTaskValue_) {
        dashboardTaskValue_->setText(QString::number(taskCount));
    }
    if (dashboardModelValue_) {
        dashboardModelValue_->setText(hasProject
            ? QStringLiteral("%1 / %2").arg(modelVersionCount).arg(exportCount)
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
    updateProjectSummary();
    updateCapabilitySummary();
    updateEnvironmentSummary();

    if (dashboardNextStepLabel_) {
        QString nextStep;
        if (!hasProject) {
            nextStep = uiText("先创建或打开一个本地项目。项目目录会集中保存数据集索引、任务历史、训练报告和模型产物。");
        } else if (validDatasetCount == 0) {
            nextStep = uiText("下一步：导入并校验 detection、segmentation 或 OCR Rec 数据集。只有通过校验的数据集会进入训练主流程。");
        } else if (taskCount == 0) {
            nextStep = uiText("下一步：进入训练实验，选择已校验数据集。平台会按任务类型优先选择官方 YOLO / OCR 后端。");
        } else if (exportCount == 0) {
            nextStep = uiText("下一步：在任务与产物中查看 checkpoint / report / ONNX，注册到模型库后再进入部署验证。");
        } else {
            nextStep = uiText("项目已具备可复验闭环：数据集、任务历史和模型导出均已记录。可继续进入部署验证或追加实验。");
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
    const QString state = state_.dataset.currentValid ? uiText("已校验") : uiText("待校验");
    const QString fullPathText = datasetPath.isEmpty() ? QString() : QDir::toNativeSeparators(datasetPath);
    const QString datasetName = datasetPath.isEmpty() ? QString() : QFileInfo(datasetPath).fileName();
    const QString headerPathText = datasetPath.isEmpty()
        ? uiText("未选择")
        : (datasetName.isEmpty() ? compactPathForStatus(datasetPath, 36) : datasetName);
    const QString detailPathText = datasetPath.isEmpty() ? uiText("未选择") : compactPathForStatus(datasetPath, 92);
    QString snapshotText = uiText("快照：未选择数据集");
    QString snapshotManifestPath;
    bool datasetReady = state_.dataset.currentValid && state_.dataset.currentPath == datasetPath && state_.dataset.currentFormat == datasetFormat;
    if (!datasetPath.isEmpty() && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &error);
        datasetReady = datasetReady
            || (dataset.rootPath == datasetPath
                && dataset.format == datasetFormat
                && dataset.validationStatus == QStringLiteral("valid"));
        const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(dataset.id, &error);
        snapshotManifestPath = snapshot.manifestPath;
        snapshotText = snapshot.id > 0
            ? uiText("快照：#%1 | %2 文件 | hash %3")
                .arg(snapshot.id)
                .arg(snapshot.fileCount)
                .arg(snapshot.contentHash.left(12))
            : uiText("快照：暂无，启动训练时将自动创建。");
    }

    if (trainingDatasetSummaryLabel_) {
        trainingDatasetSummaryLabel_->setText(datasetPath.isEmpty()
            ? uiText("当前数据集：未选择。请先在数据集页导入并通过校验。")
            : uiText("当前数据集：%1 | %2 | %3\n%4")
                .arg(datasetFormatLabel(datasetFormat), state, headerPathText, snapshotText));
        trainingDatasetSummaryLabel_->setToolTip(datasetPath.isEmpty()
            ? QString()
            : uiText("数据集：%1\n快照：%2")
                .arg(fullPathText, QDir::toNativeSeparators(snapshotManifestPath)));
    }
    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(datasetPath.isEmpty()
            ? uiText("选择或导入数据集后显示格式、样本数、校验状态和最近报告。")
            : uiText("格式：%1 | 状态：%2 | 路径：%3\n%4")
                .arg(datasetFormatLabel(datasetFormat), state, detailPathText, snapshotText));
    }
    if (trainingBackendHintLabel_ && trainingBackendCombo_) {
        trainingBackendHintLabel_->setText(trainingBackendDescription(trainingBackendCombo_->currentData().toString()));
    }
    const QString visibleBackend = trainingBackendCombo_
        ? trainingBackendCombo_->currentData().toString().trimmed().toLower()
        : QString();
    if (auto* yoloPanel = findChild<QWidget*>(QStringLiteral("YoloOfficialArgsGroup"))) {
        yoloPanel->setVisible(visibleBackend.startsWith(QStringLiteral("ultralytics_yolo")));
    }
    if (auto* smpPanel = findChild<QWidget*>(QStringLiteral("SmpSemanticArgsGroup"))) {
        smpPanel->setVisible(visibleBackend == QStringLiteral("smp_semantic_segmentation"));
    }
    if (auto* anomalyPanel = findChild<QWidget*>(QStringLiteral("AnomalyDetectionArgsGroup"))) {
        anomalyPanel->setVisible(visibleBackend == QStringLiteral("anomalib_patchcore")
            || visibleBackend == QStringLiteral("anomalib_efficientad"));
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
        const QJsonObject preflight = trainingPreflightReport(
            datasetPath,
            datasetFormat,
            datasetReady,
            snapshotManifestPath,
            currentTaskType(),
            backend,
            model,
            epochsEdit_ ? epochsEdit_->text().toInt() : 0,
            batchEdit_ ? batchEdit_->text().toInt() : 0,
            imageSizeEdit_ ? imageSizeEdit_->text().toInt() : 0);
        trainingRunSummaryLabel_->setText(uiText("运行摘要：%1 | 后端 %2 | 模型 %3 | epoch %4 / batch %5 / image %6")
            .arg(taskTypeLabel(currentTaskType()),
                backend.isEmpty() ? uiText("未选择") : backend,
                model.isEmpty() ? uiText("默认") : model,
                epochsEdit_ ? epochsEdit_->text() : QStringLiteral("-"),
                batchEdit_ ? batchEdit_->text() : QStringLiteral("-"),
                imageSizeEdit_ ? imageSizeEdit_->text() : QStringLiteral("-")));
        trainingRunSummaryLabel_->setToolTip(trainingPreflightSummaryText(preflight));
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

    if (datasetFormat == QStringLiteral("yolo_detection") || datasetFormat == QStringLiteral("yolo_txt")) {
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
