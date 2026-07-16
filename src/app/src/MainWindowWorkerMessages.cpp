#include "MainWindow.h"

#include "DiagnosticBundlePresenterV2.h"
#include "EnvironmentCheckPresenterV2.h"
#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/WorkerProtocol.h"

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
#include <QtMath>

using namespace aitrain_app;
namespace wp = aitrain::worker_protocol;

namespace {
void setAcceptanceTableRow(QTableWidget* table, const QString& stage, const QString& status, const QString& evidence, const QString& message)
{
    if (!table) {
        return;
    }
    int row = -1;
    for (int index = 0; index < table->rowCount(); ++index) {
        if (table->item(index, 0) && table->item(index, 0)->text() == stage) {
            row = index;
            break;
        }
    }
    if (row < 0) {
        row = table->rowCount();
        table->insertRow(row);
    }
    table->setItem(row, 0, new QTableWidgetItem(stage));
    table->setItem(row, 1, new QTableWidgetItem(status));
    table->setItem(row, 2, new QTableWidgetItem(QDir::toNativeSeparators(evidence)));
    table->setItem(row, 3, new QTableWidgetItem(message));
}

QString phaseText(const QString& phase)
{
    if (phase == QStringLiteral("snapshot")) return uiText("快照");
    if (phase == QStringLiteral("train")) return uiText("训练");
    if (phase == QStringLiteral("validate")) return uiText("验证");
    if (phase == QStringLiteral("export")) return uiText("导出");
    if (phase == QStringLiteral("completed")) return uiText("完成");
    if (phase == QStringLiteral("failed")) return uiText("失败");
    return phase.isEmpty() ? uiText("运行中") : phase;
}

QString etaText(int seconds)
{
    if (seconds <= 0) {
        return QStringLiteral("--");
    }
    const int hours = seconds / 3600;
    const int minutes = (seconds % 3600) / 60;
    const int secs = seconds % 60;
    if (hours > 0) {
        return QStringLiteral("%1h %2m").arg(hours).arg(minutes);
    }
    if (minutes > 0) {
        return QStringLiteral("%1m %2s").arg(minutes).arg(secs);
    }
    return QStringLiteral("%1s").arg(secs);
}

QString metricText(const QJsonObject& metrics, const QStringList& keys)
{
    for (const QString& key : keys) {
        if (metrics.contains(key)) {
            return QString::number(metrics.value(key).toDouble(), 'f', 4);
        }
    }
    return QStringLiteral("--");
}

QString artifactDisplayName(const QString& path)
{
    const QString fileName = QFileInfo(path).fileName();
    return fileName.isEmpty() ? compactPathForStatus(path, 44) : fileName;
}

QStringList stringListFromJsonArray(const QJsonArray& array)
{
    QStringList values;
    for (const QJsonValue& value : array) {
        values.append(value.toString());
    }
    return values;
}

QString runtimeDeliverySummary(const QJsonObject& payload)
{
    QStringList lines;
    lines.append(uiText("Runtime Delivery：%1 | Runtime %2")
        .arg(payload.value(QStringLiteral("state")).toString(),
            payload.value(QStringLiteral("runtimeStatus")).toString()));
    const QJsonArray steps = payload.value(QStringLiteral("steps")).toArray();
    for (const QJsonValue& value : steps) {
        const QJsonObject step = value.toObject();
        lines.append(QStringLiteral("%1. %2 — %3")
            .arg(step.value(QStringLiteral("ordinal")).toInt() + 1)
            .arg(step.value(QStringLiteral("kind")).toString(),
                step.value(QStringLiteral("state")).toString()));
    }
    const QString evidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    lines.append(evidenceArtifactId.isEmpty()
        ? uiText("Evidence：未生成")
        : uiText("Evidence Artifact：%1（可在“任务与产物”查看已提交文件）").arg(evidenceArtifactId));
    return lines.join(QLatin1Char('\n'));
}
} // namespace

void MainWindow::handleWorkerMessage(const QString& type, const QJsonObject& payload)
{
    const QString messageTaskId = payload.value(wp::field::taskId()).toString().trimmed();
    const bool isActiveV2Task = !activeV2TaskId_.isEmpty()
        && (messageTaskId.isEmpty() || messageTaskId == activeV2TaskId_);
    if (isActiveV2Task && type == wp::event::annotationSessionV2()) {
        handleAnnotationSessionMessage(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::annotationSyncV2()) {
        handleAnnotationSyncMessage(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::runtimeDeliveryWorkflowV2()) {
        const QString summary = runtimeDeliverySummary(payload);
        if (inferenceResultLabel_) inferenceResultLabel_->setText(summary);
        if (deploymentValidationResultLabel_) deploymentValidationResultLabel_->setText(summary);
        setAcceptanceTableRow(deliveryAcceptanceTable_, uiText("Runtime Delivery"),
            payload.value(QStringLiteral("state")).toString(),
            payload.value(QStringLiteral("evidenceArtifactId")).toString(),
            payload.value(QStringLiteral("runtimeStatus")).toString());
        appendLog(uiText("Runtime Delivery 六步状态与 Evidence 已返回。"));
        updateDeliveryAcceptanceSummary();
        updateRecentTasks();
        updateSelectedTaskDetails();
        updateModelRegistry();
        return;
    }
    if (isActiveV2Task && type == wp::event::dataQualityWorkflowV2()) {
        handleDataQualityWorkflowV2Message(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::diagnosticsWorkflowV2()) {
        handleDiagnosticsWorkflowV2(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::environmentCheckWorkflowV2()) {
        if (environmentCheckPresenter_->selectTask(activeV2TaskId_)) {
            updateEnvironmentTable(environmentCheckPresenter_->viewModel().report);
            appendLog(uiText("Environment Check V2 已通过 TaskId 加载已提交报告与 Evidence。"));
        } else {
            appendLog(uiText("Environment Check V2 报告读取失败：%1")
                .arg(environmentCheckPresenter_->lastError()));
        }
        return;
    }
    if (isActiveV2Task && type == wp::event::datasetConversionWorkflowV2()) {
        handleDatasetConversionWorkflowV2(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::datasetSplitWorkflowV2()) {
        handleDatasetSplitWorkflowV2(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::datasetSnapshotImportWorkflowV2()) {
        handleDatasetSnapshotImportWorkflowV2(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::ocrOfficialReportsImportedV2()) {
        handleOcrOfficialReportsImportedV2(payload);
        return;
    }
    if (isActiveV2Task && type == wp::event::ocrAcceptanceWorkflowV2()) {
        handleOcrAcceptanceWorkflowV2(payload);
        return;
    }
    if (isActiveV2Task && (type == wp::event::completed()
            || type == wp::event::failed() || type == wp::event::canceled())) {
        const QString message = payload.value(wp::field::message()).toString();
        if (activeV2WorkflowKind_ == QStringLiteral("dataset_conversion_v2")) {
            setDatasetConversionFormRunning(false);
        }
        QString workflowName = uiText("Runtime Delivery");
        if (activeV2WorkflowKind_.startsWith(QStringLiteral("annotation"))) {
            workflowName = uiText("Annotation Session V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("ocr_report_import_v2")) {
            workflowName = uiText("OCR 官方报告受控导入");
        } else if (activeV2WorkflowKind_ == QStringLiteral("ocr_acceptance_v2")) {
            workflowName = uiText("OCR Acceptance V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("data_quality_v2")) {
            workflowName = uiText("Data Quality V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("dataset_conversion_v2")) {
            workflowName = uiText("Dataset Conversion V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("dataset_split_v2")) {
            workflowName = uiText("Dataset Split V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("dataset_snapshot_import_v2")) {
            workflowName = uiText("Dataset Snapshot Import V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("diagnostics_v2")) {
            workflowName = uiText("Diagnostics Bundle V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("environment_check_v2")) {
            workflowName = uiText("Environment Check V2");
        } else if (activeV2WorkflowKind_ == QStringLiteral("training_v2")) {
            workflowName = uiText("Training Workflow V2");
        }
        appendLog(type == wp::event::completed()
            ? uiText("%1 已完成：%2").arg(workflowName, message)
            : (type == wp::event::canceled()
                ? uiText("%1 已取消：%2").arg(workflowName, message)
                : uiText("%1 失败：%2").arg(workflowName, message)));
        activeV2TaskId_.clear();
        activeV2WorkflowKind_.clear();
        updateRecentTasks();
        updateSelectedTaskDetails();
        updateProjectSummary();
        updateDashboardSummary();
        return;
    }
    if (type == wp::event::progress()) {
        handleProgressMessage(payload);
    } else if (type == wp::event::metric()) {
        handleMetricMessage(payload);
    } else if (type == wp::event::artifact()) {
        handleArtifactMessage(payload);
    } else if (wp::isTaskStateEvent(type) && type != wp::event::completed()) {
        handleTaskStateMessage(type, payload);
    } else if (type == wp::event::modelImportV2()) {
        v2ModelImportInProgress_ = false;
        if (v2ModelImportResultLabel_) {
            v2ModelImportResultLabel_->setText(uiText("V2 模型包已登记：%1").arg(payload.value(QStringLiteral("modelPackageId")).toString()));
        }
        updateModelRegistry();
    }
}

void MainWindow::handleProgressMessage(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    int percent = payload.value(QStringLiteral("percent")).toInt(-1);
    if (percent < 0 && payload.contains(QStringLiteral("value"))) {
        percent = qRound(payload.value(QStringLiteral("value")).toDouble() * 100.0);
    }
    percent = qBound(0, percent < 0 ? 0 : percent, 100);
    const QString message = payload.value(QStringLiteral("message")).toString();
    if (activeV2WorkflowKind_ == QStringLiteral("dataset_conversion_v2")
        && (taskId.isEmpty() || taskId == activeV2TaskId_)) {
        if (datasetConversionProgressBar_) {
            datasetConversionProgressBar_->setValue(percent);
        }
        if (!message.isEmpty()) {
            appendDatasetConversionLog(message);
        }
    }
    if (activeV2WorkflowKind_ == QStringLiteral("training_v2")) {
        progressBar_->setValue(percent);
    }
    const QString phase = payload.value(QStringLiteral("phase")).toString();
    if (trainingPhaseLabel_ && (!phase.isEmpty() || !message.isEmpty())) {
        const QString stageFlow = uiText("快照 -> 训练 -> 验证 -> 导出 -> 完成");
        const QString current = phaseText(phase);
        trainingPhaseLabel_->setText(message.isEmpty()
            ? uiText("阶段：%1 | 当前：%2").arg(stageFlow, current)
            : uiText("阶段：%1 | 当前：%2 | %3").arg(stageFlow, current, message));
    }
    const int epoch = payload.value(QStringLiteral("epoch")).toInt();
    const int epochs = payload.value(QStringLiteral("epochs")).toInt();
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingEpochValue"))) {
        label->setText(epochs > 0
            ? QStringLiteral("%1/%2").arg(qMax(0, epoch)).arg(epochs)
            : QStringLiteral("--"));
    }
    const int batch = payload.value(QStringLiteral("batch")).toInt();
    const int batches = payload.value(QStringLiteral("batches")).toInt();
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingBatchValue"))) {
        label->setText(batches > 0
            ? QStringLiteral("%1/%2").arg(qMax(0, batch)).arg(batches)
            : QStringLiteral("--"));
    }
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingEtaValue"))) {
        label->setText(etaText(payload.value(QStringLiteral("etaSeconds")).toInt()));
    }
    const QString device = payload.value(QStringLiteral("device")).toString();
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingDeviceValue")); label && !device.isEmpty()) {
        label->setText(device);
    }
    const QJsonObject liveMetrics = payload.value(QStringLiteral("liveMetrics")).toObject();
    if (!liveMetrics.isEmpty()) {
        if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingLossValue"))) {
            label->setText(metricText(liveMetrics, {
                QStringLiteral("loss"),
                QStringLiteral("boxLoss"),
                QStringLiteral("classLoss"),
                QStringLiteral("dflLoss")
            }));
        }
        if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingMapValue"))) {
            label->setText(metricText(liveMetrics, {
                QStringLiteral("mAP50"),
                QStringLiteral("maskMap50"),
                QStringLiteral("precision"),
                QStringLiteral("maskPrecision"),
                QStringLiteral("accuracy")
            }));
        }
    }
    if (!message.isEmpty()) {
        statusBar()->showMessage(message, 3000);
    }
}

void MainWindow::handleMetricMessage(const QJsonObject& payload)
{
    const QString name = payload.value(QStringLiteral("name")).toString();
    const double value = payload.value(QStringLiteral("value")).toDouble();
    metricsWidget_->addMetric(name, value);
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingLossValue"));
        label && (name.contains(QStringLiteral("Loss"), Qt::CaseInsensitive)
            || name.compare(QStringLiteral("loss"), Qt::CaseInsensitive) == 0)) {
        label->setText(QString::number(value, 'f', 4));
    }
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingMapValue"));
        label && (name == QStringLiteral("mAP50")
            || name == QStringLiteral("maskMap50")
            || name == QStringLiteral("precision")
            || name == QStringLiteral("accuracy"))) {
        label->setText(QString::number(value, 'f', 4));
    }

}

void MainWindow::handleArtifactMessage(const QJsonObject& payload)
{
    const QString path = payload.value(QStringLiteral("path")).toString();
    const QString kind = payload.value(QStringLiteral("kind")).toString();
    const QString artifactId = payload.value(QStringLiteral("artifactId")).toString();
    const QString relativePath = payload.value(QStringLiteral("relativePath")).toString();
    const QString display = !relativePath.isEmpty() ? relativePath
        : (!artifactId.isEmpty() ? artifactId : artifactDisplayName(path));
    appendLog(uiText("已提交产物：%1 / %2").arg(kind, display));
    if (kind == QStringLiteral("checkpoint") && latestCheckpointLabel_) {
        latestCheckpointLabel_->setText(uiText("最新 checkpoint：%1")
            .arg(display));
        latestCheckpointLabel_->setToolTip(QString());
    } else if (kind == QStringLiteral("onnx") && latestOnnxLabel_) {
        latestOnnxLabel_->setText(uiText("最新 ONNX：%1").arg(display));
        latestOnnxLabel_->setToolTip(QString());
    } else if ((kind == QStringLiteral("report") || kind == QStringLiteral("training_results_csv") || kind == QStringLiteral("training_args")) && latestReportLabel_) {
        latestReportLabel_->setText(uiText("训练报告：%1").arg(display));
        latestReportLabel_->setToolTip(QString());
    } else if ((kind == QStringLiteral("preview") || kind == QStringLiteral("training_plot")) && latestPreviewPathLabel_) {
        latestPreviewPathLabel_->setText(uiText("最新预览：%1")
            .arg(display));
        latestPreviewPathLabel_->setToolTip(QString());
        if (latestPreviewImageLabel_ && !path.isEmpty()) {
            QPixmap preview(path);
            if (!preview.isNull()) {
                latestPreviewImageLabel_->setPixmap(preview.scaled(
                    latestPreviewImageLabel_->size().boundedTo(QSize(320, 220)),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation));
            } else {
                latestPreviewImageLabel_->setText(uiText("预览图加载失败"));
            }
        }
    }
    if (!path.isEmpty() && kind == QStringLiteral("inference_overlay") && inferenceOverlayLabel_) {
        loadInferenceOverlay(inferenceOverlayLabel_, path);
    } else if (!path.isEmpty() && kind == QStringLiteral("inference_predictions") && inferenceResultLabel_) {
        inferenceResultLabel_->setText(inferenceSummaryFromPredictions(path));
    }
}

void MainWindow::handleTaskStateMessage(const QString& type, const QJsonObject& payload)
{
    if (type == wp::event::canceled()) {
        const QString canceledMessage = payload.value(wp::field::message()).toString();
        workerPill_->setStatus(uiText("任务已取消"), StatusPill::Tone::Warning);
        appendLog(uiText("任务已取消：%1").arg(canceledMessage));
        updateRecentTasks();
        return;
    }

    if (type == wp::event::failed()) {
        const QString failedMessage = payload.value(wp::field::message()).toString();
        appendLog(uiText("任务失败：%1").arg(failedMessage));
        updateRecentTasks();
        updateModelRegistry();
    }
}

void MainWindow::handleDataQualityWorkflowV2Message(const QJsonObject& payload)
{
    const QJsonObject summary = payload.value(QStringLiteral("summary")).toObject();
    const QString state = payload.value(QStringLiteral("state")).toString();
    const QString repairArtifactId = payload.value(QStringLiteral("repairManifestArtifactId")).toString();
    const QString reportArtifactId = payload.value(QStringLiteral("qualityReportArtifactId")).toString();
    const QString evidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(uiText("Data Quality V2：%1；问题 %2；Report Artifact %3；Evidence %4")
            .arg(state)
            .arg(summary.value(QStringLiteral("issueCount")).toInt())
            .arg(reportArtifactId, evidenceArtifactId));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        validationIssuesTable_->insertRow(0);
        validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(
            state == QStringLiteral("succeeded") ? uiText("完成") : uiText("失败")));
        validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("data_quality_v2")));
        validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(reportArtifactId));
        validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
        validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(
            uiText("问题明细与报告文件请在“任务与产物”按 ArtifactId 查看。")));
    }
    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(uiText("Repair ArtifactId：%1 | Report ArtifactId：%2")
            .arg(repairArtifactId, reportArtifactId));
    }
    setDatasetRepairLoopRows(uiText("修复闭环：Data Quality V2 已返回受控 Artifact。"),
        QVector<QStringList>{
            QStringList() << uiText("质量报告") << state << uiText("Report ArtifactId：%1").arg(reportArtifactId),
            QStringList() << uiText("外部修复") << uiText("等待") << uiText("Repair ArtifactId：%1").arg(repairArtifactId),
            QStringList() << uiText("复检") << uiText("等待") << uiText("修复同步后使用新 Snapshot 身份重新运行。")});
    updateRecentTasks();
    updateSelectedTaskDetails();
}

void MainWindow::handleDatasetSplitWorkflowV2(const QJsonObject& payload)
{
    const QString state = payload.value(QStringLiteral("state")).toString();
    const QString summary = state == QStringLiteral("succeeded")
        ? uiText("划分已登记：Dataset %1，Version %2，Snapshot %3。")
            .arg(payload.value(QStringLiteral("datasetId")).toString(),
                payload.value(QStringLiteral("datasetVersionId")).toString(),
                payload.value(QStringLiteral("snapshotId")).toString())
        : uiText("划分未登记目标快照：%1")
            .arg(payload.value(wp::field::message()).toString());
    if (validationSummaryLabel_) validationSummaryLabel_->setText(summary);
    if (validationOutput_) {
        validationOutput_->setPlainText(uiText(
            "Plan ArtifactId：%1\nSplit ArtifactId：%2\nSnapshot ArtifactId：%3\nEvidence ArtifactId：%4")
            .arg(payload.value(QStringLiteral("splitPlanArtifactId")).toString(),
                payload.value(QStringLiteral("splitArtifactId")).toString(),
                payload.value(QStringLiteral("snapshotArtifactId")).toString(),
                payload.value(QStringLiteral("evidenceArtifactId")).toString()));
    }
    appendLog(summary);
    updateRecentTasks();
    updateSelectedTaskDetails();
    updateProjectSummary();
    updateDashboardSummary();
}

void MainWindow::handleAnnotationSessionMessage(const QJsonObject& payload)
{
    state_.dataset.latestAnnotationSessionArtifactId = payload.value(QStringLiteral("sessionArtifactId")).toString();
    state_.dataset.latestAnnotationEvidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    const QString terminalState = payload.value(QStringLiteral("state")).toString();
    const QString status = payload.value(QStringLiteral("status")).toString();
    const QString executable = resolvedXAnyLabelingProgram();

    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(uiText("Annotation Session V2：%1 | Session ArtifactId %2 | Evidence %3")
            .arg(status.isEmpty() ? uiText("已准备") : status)
            .arg(state_.dataset.latestAnnotationSessionArtifactId)
            .arg(state_.dataset.latestAnnotationEvidenceArtifactId));
    }

    QVector<QStringList> rows;
    rows.append(QStringList()
        << uiText("会话准备")
        << (executable.isEmpty() ? uiText("缺少工具") : uiText("已完成"))
        << uiText("Session ArtifactId：%1").arg(state_.dataset.latestAnnotationSessionArtifactId));
    rows.append(QStringList()
        << uiText("外部修复")
        << (executable.isEmpty() ? uiText("阻塞") : uiText("已启动"))
        << (executable.isEmpty()
            ? uiText("配置 AITRAIN_XANYLABELING_EXE 或 .deps/tools/annotation-tools 后重新准备。")
            : uiText("按问题清单修复样本，保存后回到 AITrain。")));
    rows.append(QStringList()
        << uiText("同步复检")
        << uiText("等待")
        << uiText("标注完成后点击“同步标注会话”。"));
    setDatasetRepairLoopRows(
        terminalState != QStringLiteral("succeeded")
            ? uiText("修复闭环：会话未创建，请查看任务 Evidence。")
            : executable.isEmpty()
            ? uiText("修复闭环：会话已生成，但未检测到 X-AnyLabeling。")
            : uiText("修复闭环：X-AnyLabeling 会话已准备。"),
        rows);

    if (terminalState != QStringLiteral("succeeded")
        || state_.dataset.latestAnnotationSessionArtifactId.isEmpty()) {
        statusBar()->showMessage(uiText("Annotation Session V2 未创建正式会话；请查看任务 Evidence。"), 6000);
        updateRecentTasks();
        updateSelectedTaskDetails();
        return;
    }

    if (executable.isEmpty()) {
        updateAnnotationToolStatus();
        statusBar()->showMessage(uiText("X-AnyLabeling 会话已生成，但未找到本地工具。"), 6000);
        return;
    }

    if (QProcess::startDetached(executable,
            QStringList() << state_.dataset.annotationWorkingDirectory,
            state_.dataset.annotationWorkingDirectory)) {
        updateAnnotationToolStatus();
        statusBar()->showMessage(uiText("已启动 X-AnyLabeling 修复会话。"), 5000);
    } else {
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(executable)));
    }
    updateRecentTasks();
    updateSelectedTaskDetails();
}

void MainWindow::handleAnnotationSyncMessage(const QJsonObject& payload)
{
    state_.dataset.latestAnnotationSyncReportArtifactId = payload.value(QStringLiteral("syncReportArtifactId")).toString();
    state_.dataset.latestAnnotationEvidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    const QString terminalState = payload.value(QStringLiteral("state")).toString();
    const QString status = payload.value(QStringLiteral("status")).toString();
    const bool createdVersion = payload.value(QStringLiteral("newDatasetVersionCreated")).toBool();
    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(uiText("Annotation Sync V2：%1 | 新版本 %2 | Report ArtifactId %3")
            .arg(status)
            .arg(createdVersion ? uiText("已创建") : uiText("未创建"))
            .arg(state_.dataset.latestAnnotationSyncReportArtifactId));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    setDatasetRepairLoopRows(
        terminalState == QStringLiteral("succeeded")
            ? uiText("修复闭环：标注同步已收口。")
            : uiText("修复闭环：标注同步未成功，请查看 Evidence。"),
        QVector<QStringList>{
            QStringList() << uiText("外部修复") << uiText("已保存") << uiText("工作目录已由 Worker 重新校验。"),
            QStringList() << uiText("同步") << uiText("完成") << uiText("状态 %1；Report ArtifactId %2").arg(status, state_.dataset.latestAnnotationSyncReportArtifactId),
            QStringList() << uiText("复检") << (createdVersion ? uiText("已生成新版本") : uiText("未生成新版本")) << uiText("任务、Artifact 和 Dataset Version 由 V2 Query/Presenter 刷新。")
        });
    statusBar()->showMessage(terminalState == QStringLiteral("succeeded")
        ? uiText("Annotation Sync V2 已返回结构化状态")
        : uiText("Annotation Sync V2 未成功；未登记正式新版本"), 5000);
    updateRecentTasks();
    updateSelectedTaskDetails();
    updateProjectSummary();
    updateDashboardSummary();
}

void MainWindow::handleDatasetSnapshotImportWorkflowV2(const QJsonObject& payload)
{
    const QString state = payload.value(QStringLiteral("state")).toString();
    const QString datasetId = payload.value(QStringLiteral("datasetId")).toString();
    const QString versionId = payload.value(QStringLiteral("datasetVersionId")).toString();
    const QString snapshotId = payload.value(QStringLiteral("snapshotId")).toString();
    const QString snapshotArtifactId = payload.value(QStringLiteral("snapshotArtifactId")).toString();
    const QString evidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(
            uiText("Snapshot Import %1：Dataset %2 | Version %3 | Snapshot %4 | Evidence %5")
                .arg(state, datasetId, versionId, snapshotId, evidenceArtifactId));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (validationOutput_) validationOutput_->appendPlainText(
        uiText("Snapshot ArtifactId：%1").arg(snapshotArtifactId));
    updateTrainingSelectionSummary();
    updateRecentTasks();
    updateSelectedTaskDetails();
    updateProjectSummary();
    updateDashboardSummary();
}

void MainWindow::handleOcrOfficialReportsImportedV2(const QJsonObject& payload)
{
    const QString detArtifactId = payload.value(QStringLiteral("detReportArtifactId")).toString();
    const QString recArtifactId = payload.value(QStringLiteral("recReportArtifactId")).toString();
    const QString systemArtifactId = payload.value(QStringLiteral("systemReportArtifactId")).toString();
    const QString evidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    if (customerOcrDetReportArtifactIdEdit_) customerOcrDetReportArtifactIdEdit_->setText(detArtifactId);
    if (customerOcrRecReportArtifactIdEdit_) customerOcrRecReportArtifactIdEdit_->setText(recArtifactId);
    if (customerOcrSystemReportArtifactIdEdit_) customerOcrSystemReportArtifactIdEdit_->setText(systemArtifactId);
    if (customerOcrStatusLabel_) {
        customerOcrStatusLabel_->setText(uiText("官方报告导入：%1；Evidence Artifact：%2")
            .arg(payload.value(QStringLiteral("state")).toString(), evidenceArtifactId));
    }
    setAcceptanceTableRow(deliveryAcceptanceTable_, uiText("OCR 报告导入"),
        payload.value(QStringLiteral("state")).toString(), evidenceArtifactId,
        uiText("只展示 committed ArtifactId；原始报告路径未越过导入边界。"));
    updateDeliveryAcceptanceSummary();
    updateRecentTasks();
    updateSelectedTaskDetails();
}

void MainWindow::handleOcrAcceptanceWorkflowV2(const QJsonObject& payload)
{
    const QString state = payload.value(QStringLiteral("state")).toString();
    const QString evidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    const QString acceptanceArtifactId = payload.value(QStringLiteral("acceptanceReportArtifactId")).toString();
    if (customerOcrStatusLabel_) {
        customerOcrStatusLabel_->setText(uiText("OCR Acceptance V2：%1 | production accepted：%2 | Evidence：%3")
            .arg(state, payload.value(QStringLiteral("productionAccepted")).toBool()
                ? uiText("是") : uiText("否"), evidenceArtifactId));
    }
    setAcceptanceTableRow(deliveryAcceptanceTable_, uiText("客户域 OCR"), state,
        evidenceArtifactId, acceptanceArtifactId.isEmpty()
            ? payload.value(wp::field::message()).toString()
            : uiText("验收报告 Artifact：%1").arg(acceptanceArtifactId));
    updateDeliveryAcceptanceSummary();
    updateRecentTasks();
    updateSelectedTaskDetails();
}

void MainWindow::handleDiagnosticsWorkflowV2(const QJsonObject& payload)
{
    const QString taskId = payload.value(wp::field::taskId()).toString();
    const bool loaded = diagnosticBundlePresenter_
        && diagnosticBundlePresenter_->selectTask(taskId);
    const DiagnosticBundleViewModelV2 model = loaded
        ? diagnosticBundlePresenter_->viewModel() : DiagnosticBundleViewModelV2();
    if (diagnosticsStatusLabel_) {
        diagnosticsStatusLabel_->setText(loaded
            ? uiText("Diagnostics V2：%1；Bundle Artifact：%2；Evidence：%3")
                .arg(model.state, model.diagnosticsArtifactId, model.evidenceArtifactId)
            : uiText("Diagnostics V2 已返回，但无法从 V2 Query 读取任务事实。"));
    }
    setAcceptanceTableRow(
        deliveryAcceptanceTable_,
        uiText("诊断包"),
        loaded ? model.state : payload.value(QStringLiteral("state")).toString(),
        loaded ? model.evidenceArtifactId : QString(),
        loaded ? uiText("Bundle Artifact：%1").arg(model.diagnosticsArtifactId)
               : uiText("只读 Presenter 查询失败。"));
    updateDeliveryAcceptanceSummary();
    updateRecentTasks();
    updateSelectedTaskDetails();
}
