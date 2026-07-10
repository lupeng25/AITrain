#include "MainWindow.h"

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
} // namespace

void MainWindow::handleWorkerMessage(const QString& type, const QJsonObject& payload)
{
    if (type == wp::event::progress()) {
        handleProgressMessage(payload);
    } else if (type == wp::event::metric()) {
        handleMetricMessage(payload);
    } else if (type == wp::event::artifact()) {
        handleArtifactMessage(payload);
    } else if (wp::isTaskStateEvent(type) && type != wp::event::completed()) {
        handleTaskStateMessage(type, payload);
    } else if (type == wp::event::environmentCheck()) {
        updateEnvironmentTable(payload);
    } else if (type == wp::event::datasetValidation()) {
        updateDatasetValidationResult(payload);
    } else if (type == wp::event::datasetSplit()) {
        updateDatasetSplitResult(payload);
    } else if (type == wp::event::datasetConversion()) {
        updateDatasetConversionResult(payload);
    } else if (type == wp::event::datasetQuality()) {
        handleDatasetQualityMessage(payload);
    } else if (type == wp::event::annotationSession()) {
        handleAnnotationSessionMessage(payload);
    } else if (type == wp::event::annotationSync()) {
        handleAnnotationSyncMessage(payload);
    } else if (type == wp::event::datasetSnapshot()) {
        handleDatasetSnapshotMessage(payload);
    } else if (type == wp::event::evaluationReport()) {
        handleEvaluationReportMessage(payload);
    } else if (type == wp::event::benchmarkReport()) {
        updateModelRegistry();
    } else if (type == wp::event::pipelinePlan()) {
        handlePipelinePlanMessage(payload);
    } else if (type == wp::event::deliveryReport()) {
        updateModelRegistry();
    } else if (type == wp::event::modelExport()) {
        handleModelExportMessage(payload);
    } else if (type == wp::event::deploymentValidation()) {
        handleDeploymentValidationMessage(payload);
    } else if (type == wp::event::inferenceResult()) {
        handleInferenceResultMessage(payload);
    } else if (type == wp::event::customerOcrAcceptance()) {
        handleCustomerOcrAcceptanceMessage(payload);
    } else if (type == wp::event::diagnosticBundle()) {
        handleDiagnosticBundleMessage(payload);
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
    if (!state_.dataset.currentConversionTaskId.isEmpty() && taskId == state_.dataset.currentConversionTaskId) {
        if (datasetConversionProgressBar_) {
            datasetConversionProgressBar_->setValue(percent);
        }
        if (!message.isEmpty()) {
            appendDatasetConversionLog(message);
        }
    }
    if (!state_.training.currentTaskId.isEmpty()) {
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

    aitrain::MetricPoint point;
    point.taskId = state_.training.currentTaskId;
    point.name = name;
    point.value = value;
    point.step = payload.value(QStringLiteral("step")).toInt();
    point.epoch = payload.value(QStringLiteral("epoch")).toInt();
    point.createdAt = QDateTime::currentDateTimeUtc();
    QString error;
    repository_.insertMetric(point, &error);
}

void MainWindow::handleArtifactMessage(const QJsonObject& payload)
{
    const QString path = payload.value(QStringLiteral("path")).toString();
    const QString kind = payload.value(QStringLiteral("kind")).toString();
    appendLog(uiText("产物：%1").arg(path));
    if (kind == QStringLiteral("checkpoint") && latestCheckpointLabel_) {
        latestCheckpointLabel_->setText(uiText("最新 checkpoint：%1")
            .arg(artifactDisplayName(path)));
        latestCheckpointLabel_->setToolTip(QDir::toNativeSeparators(path));
    } else if (kind == QStringLiteral("onnx") && latestOnnxLabel_) {
        latestOnnxLabel_->setText(uiText("最新 ONNX：%1").arg(artifactDisplayName(path)));
        latestOnnxLabel_->setToolTip(QDir::toNativeSeparators(path));
    } else if ((kind == QStringLiteral("report") || kind == QStringLiteral("training_results_csv") || kind == QStringLiteral("training_args")) && latestReportLabel_) {
        latestReportLabel_->setText(uiText("训练报告：%1").arg(artifactDisplayName(path)));
        latestReportLabel_->setToolTip(QDir::toNativeSeparators(path));
    } else if ((kind == QStringLiteral("preview") || kind == QStringLiteral("training_plot")) && latestPreviewPathLabel_) {
        latestPreviewPathLabel_->setText(uiText("最新预览：%1")
            .arg(artifactDisplayName(path)));
        latestPreviewPathLabel_->setToolTip(QDir::toNativeSeparators(path));
        if (latestPreviewImageLabel_) {
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
    if (kind == QStringLiteral("export") && exportResultLabel_) {
        exportResultLabel_->setText(uiText("导出完成：%1").arg(QDir::toNativeSeparators(path)));
    } else if (kind == QStringLiteral("inference_overlay") && inferenceOverlayLabel_) {
        loadInferenceOverlay(inferenceOverlayLabel_, path);
    } else if (kind == QStringLiteral("inference_predictions") && inferenceResultLabel_) {
        inferenceResultLabel_->setText(inferenceSummaryFromPredictions(path));
    }
    if (repository_.isOpen()) {
        aitrain::ArtifactRecord artifact;
        artifact.taskId = payload.value(QStringLiteral("taskId")).toString(state_.training.currentTaskId);
        artifact.kind = kind;
        artifact.path = path;
        artifact.message = uiText("Worker 上报产物");
        artifact.createdAt = QDateTime::currentDateTimeUtc();
        QString error;
        repository_.insertArtifact(artifact, &error);
    }
}

void MainWindow::handleTaskStateMessage(const QString& type, const QJsonObject& payload)
{
    if (type == wp::event::canceled()) {
        QString error;
        const QString canceledTaskId = payload.value(wp::field::taskId()).toString(state_.training.currentTaskId);
        const QString canceledMessage = payload.value(wp::field::message()).toString();
        if (!state_.dataset.currentConversionTaskId.isEmpty() && canceledTaskId == state_.dataset.currentConversionTaskId) {
            setDatasetConversionFormRunning(false);
            if (datasetConversionStatusLabel_) {
                datasetConversionStatusLabel_->setText(uiText("数据集转换已取消。"));
            }
            state_.dataset.currentConversionTaskId.clear();
        }
        repository_.updateTaskState(canceledTaskId, wp::taskStateForEvent(type), canceledMessage, &error);
        if (state_.training.hasActiveSnapshotTrainingTask && canceledTaskId == state_.training.currentTaskId) {
            repository_.updateTaskState(
                state_.training.activeSnapshotTrainingTask.taskId,
                aitrain::TaskState::Canceled,
                uiText("自动数据快照已取消，训练未启动。"),
                &error);
            state_.training.hasActiveSnapshotTrainingTask = false;
            state_.training.activeSnapshotTrainingTask = PendingTrainingTask();
        }
        workerPill_->setStatus(uiText("任务已取消"), StatusPill::Tone::Warning);
        appendLog(uiText("任务已取消：%1").arg(canceledMessage));
        state_.training.currentTaskId.clear();
        updateRecentTasks();
        startNextQueuedTask();
        return;
    }

    if (type == wp::event::failed()) {
        const QString failedTaskId = payload.value(wp::field::taskId()).toString(state_.training.currentTaskId);
        const QString failedMessage = payload.value(wp::field::message()).toString();
        if (!state_.dataset.currentConversionTaskId.isEmpty() && failedTaskId == state_.dataset.currentConversionTaskId) {
            setDatasetConversionFormRunning(false);
            if (datasetConversionStatusLabel_) {
                datasetConversionStatusLabel_->setText(uiText("数据集转换失败：%1").arg(failedMessage));
            }
            state_.dataset.currentConversionTaskId.clear();
        }
        QString error;
        repository_.updateTaskState(
            failedTaskId,
            wp::taskStateForEvent(type),
            failedMessage,
            &error);
        if (state_.training.hasActiveSnapshotTrainingTask && failedTaskId == state_.training.currentTaskId) {
            repository_.updateTaskState(
                state_.training.activeSnapshotTrainingTask.taskId,
                aitrain::TaskState::Failed,
                uiText("自动数据快照失败：%1").arg(failedMessage),
                &error);
            state_.training.hasActiveSnapshotTrainingTask = false;
            state_.training.activeSnapshotTrainingTask = PendingTrainingTask();
        }
        updateRecentTasks();
        updateModelRegistry();
    }
}

void MainWindow::handleDatasetQualityMessage(const QJsonObject& payload)
{
    state_.dataset.latestQualityFixListPath = payload.value(QStringLiteral("xAnyLabelingFixListPath")).toString();
    state_.dataset.latestQualityFixManifestPath = payload.value(QStringLiteral("xAnyLabelingFixManifestPath")).toString();
    state_.dataset.latestQualityReportPath = payload.value(QStringLiteral("reportPath")).toString();
    if (validationSummaryLabel_) {
        const QJsonObject severityCounts = payload.value(QStringLiteral("severityCounts")).toObject();
        const QJsonObject summary = payload.value(QStringLiteral("summary")).toObject();
        const QJsonObject readiness = payload.value(QStringLiteral("trainingReadiness")).toObject();
        const QString readinessStatus = readiness.value(QStringLiteral("status")).toString(
            payload.value(QStringLiteral("ok")).toBool() ? QStringLiteral("ready") : QStringLiteral("blocked"));
        validationSummaryLabel_->setText(uiText("质量报告完成：error %1 / warning %2 / info %3，问题样本 %4，重复图片 %5。")
            .arg(severityCounts.value(QStringLiteral("error")).toInt())
            .arg(severityCounts.value(QStringLiteral("warning")).toInt())
            .arg(severityCounts.value(QStringLiteral("info")).toInt())
            .arg(summary.value(QStringLiteral("problemSampleCount")).toInt())
            .arg(summary.value(QStringLiteral("duplicateImageCount")).toInt()));
        validationSummaryLabel_->setToolTip(uiText("Training readiness: %1. Readiness JSON is attached to the dataset quality artifacts.")
            .arg(readinessStatus));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        const QJsonArray samples = !payload.value(QStringLiteral("problemSamples")).toArray().isEmpty()
            ? payload.value(QStringLiteral("problemSamples")).toArray()
            : payload.value(QStringLiteral("issues")).toArray();
        if (samples.isEmpty()) {
            validationIssuesTable_->insertRow(0);
            validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(uiText("通过")));
            validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("ok")));
            validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString()));
            validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
            validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(uiText("未发现需要修复的问题样本。")));
        } else {
            for (const QJsonValue& value : samples) {
                const QJsonObject issue = value.toObject();
                const int row = validationIssuesTable_->rowCount();
                validationIssuesTable_->insertRow(row);
                validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(issueSeverityLabel(issue.value(QStringLiteral("severity")).toString())));
                validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(issue.value(QStringLiteral("code")).toString()));
                const QString issuePath = !issue.value(QStringLiteral("imagePath")).toString().isEmpty()
                    ? issue.value(QStringLiteral("imagePath")).toString()
                    : (!issue.value(QStringLiteral("labelPath")).toString().isEmpty()
                        ? issue.value(QStringLiteral("labelPath")).toString()
                        : issue.value(QStringLiteral("filePath")).toString());
                validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(issuePath));
                const int line = issue.value(QStringLiteral("line")).toInt();
                validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(line > 0 ? QString::number(line) : QString()));
                validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(issue.value(QStringLiteral("message")).toString()));
            }
        }
    }
    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(uiText("修复清单：%1")
            .arg(state_.dataset.latestQualityFixListPath.isEmpty() ? uiText("暂无") : QDir::toNativeSeparators(state_.dataset.latestQualityFixListPath)));
    }
    updateDatasetRepairLoopFromQuality(payload);
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    if (!datasetPath.isEmpty()) {
        state_.dataset.currentPath = datasetPath;
        state_.dataset.currentFormat = format;
        state_.dataset.currentValid = payload.value(QStringLiteral("ok")).toBool();
    }
    if (repository_.isOpen() && !datasetPath.isEmpty()) {
        const QJsonObject summary = payload.value(QStringLiteral("summary")).toObject();
        aitrain::DatasetRecord dataset;
        dataset.name = QFileInfo(datasetPath).fileName();
        dataset.format = format;
        dataset.rootPath = datasetPath;
        dataset.validationStatus = payload.value(QStringLiteral("ok")).toBool() ? QStringLiteral("valid") : QStringLiteral("invalid");
        dataset.sampleCount = summary.value(QStringLiteral("sampleCount")).toInt();
        dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
        QString error;
        repository_.upsertDatasetValidation(dataset, &error);
        updateDatasetList();
    }
}

void MainWindow::handleAnnotationSessionMessage(const QJsonObject& payload)
{
    state_.dataset.latestAnnotationSessionManifestPath = payload.value(QStringLiteral("manifestPath")).toString(
        payload.value(QStringLiteral("reportPath")).toString());
    state_.dataset.latestAnnotationLaunchRequestPath = payload.value(QStringLiteral("launchRequestPath")).toString();
    const QString status = payload.value(QStringLiteral("status")).toString();
    const QString executable = payload.value(QStringLiteral("xAnyLabelingExecutable")).toString();
    const int reviewCount = payload.value(QStringLiteral("reviewSampleCount")).toInt();

    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(uiText("X-AnyLabeling 会话：%1 | 复核样本 %2 | manifest %3")
            .arg(status.isEmpty() ? uiText("已准备") : status)
            .arg(reviewCount)
            .arg(QDir::toNativeSeparators(state_.dataset.latestAnnotationSessionManifestPath)));
    }

    QVector<QStringList> rows;
    rows.append(QStringList()
        << uiText("会话准备")
        << (executable.isEmpty() ? uiText("缺少工具") : uiText("已完成"))
        << QDir::toNativeSeparators(state_.dataset.latestAnnotationSessionManifestPath));
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
        executable.isEmpty()
            ? uiText("修复闭环：会话已生成，但未检测到 X-AnyLabeling。")
            : uiText("修复闭环：X-AnyLabeling 会话已准备。"),
        rows);

    if (executable.isEmpty()) {
        updateAnnotationToolStatus();
        statusBar()->showMessage(uiText("X-AnyLabeling 会话已生成，但未找到本地工具。"), 6000);
        return;
    }

    QFile file(state_.dataset.latestAnnotationLaunchRequestPath);
    if (!file.open(QIODevice::ReadOnly)) {
        statusBar()->showMessage(uiText("X-AnyLabeling 会话已生成，但 launch_request 无法读取。"), 6000);
        return;
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    const QJsonObject launchRequest = document.object();
    const QString launchExecutable = launchRequest.value(QStringLiteral("executable")).toString(executable);
    const QStringList arguments = stringListFromJsonArray(launchRequest.value(QStringLiteral("arguments")).toArray());
    const QString workingDirectory = launchRequest.value(QStringLiteral("workingDirectory")).toString();
    if (QProcess::startDetached(launchExecutable, arguments, workingDirectory)) {
        updateAnnotationToolStatus();
        statusBar()->showMessage(uiText("已启动 X-AnyLabeling 修复会话。"), 5000);
    } else {
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(launchExecutable)));
    }
}

void MainWindow::handleAnnotationSyncMessage(const QJsonObject& payload)
{
    state_.dataset.latestAnnotationSyncReportPath = payload.value(QStringLiteral("reportPath")).toString();
    const int scannedCount = payload.value(QStringLiteral("scannedLabelCount")).toInt();
    const int modifiedCount = payload.value(QStringLiteral("modifiedLabelCount")).toInt();
    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(uiText("X-AnyLabeling 同步：扫描标签 %1，疑似修改 %2；报告 %3")
            .arg(scannedCount)
            .arg(modifiedCount)
            .arg(QDir::toNativeSeparators(state_.dataset.latestAnnotationSyncReportPath)));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    setDatasetRepairLoopRows(
        uiText("修复闭环：标注同步完成，请重新校验。"),
        QVector<QStringList>{
            QStringList() << uiText("外部修复") << uiText("已保存") << uiText("同步扫描了 %1 个标签文件。").arg(scannedCount),
            QStringList() << uiText("同步") << uiText("完成") << uiText("疑似修改 %1 个标签文件；报告 %2").arg(modifiedCount).arg(QDir::toNativeSeparators(state_.dataset.latestAnnotationSyncReportPath)),
            QStringList() << uiText("复检") << uiText("待执行") << uiText("点击“标注后刷新 / 重新校验”重新生成质量报告或校验报告。")
        });
    statusBar()->showMessage(uiText("X-AnyLabeling 标注同步完成"), 5000);
}

void MainWindow::handleDatasetSnapshotMessage(const QJsonObject& payload)
{
    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(uiText("数据集快照完成：%1 个文件，hash %2。")
            .arg(payload.value(QStringLiteral("fileCount")).toInt())
            .arg(payload.value(QStringLiteral("contentHash")).toString().left(12)));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (repository_.isOpen()) {
        QString error;
        const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
        aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &error);
        if (dataset.id <= 0 && !datasetPath.isEmpty()) {
            aitrain::DatasetRecord seed;
            seed.name = QFileInfo(datasetPath).fileName();
            seed.format = payload.value(QStringLiteral("format")).toString(state_.dataset.currentFormat);
            seed.rootPath = datasetPath;
            seed.validationStatus = state_.dataset.currentValid ? QStringLiteral("valid") : QStringLiteral("snapshot");
            seed.sampleCount = payload.value(QStringLiteral("fileCount")).toInt();
            seed.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
            seed.lastValidatedAt = QDateTime::currentDateTimeUtc();
            repository_.upsertDatasetValidation(seed, &error);
            dataset = repository_.datasetByRootPath(datasetPath, &error);
        }
        if (dataset.id > 0) {
            aitrain::DatasetSnapshotRecord snapshot;
            snapshot.datasetId = dataset.id;
            snapshot.name = QFileInfo(datasetPath).fileName();
            snapshot.rootPath = datasetPath;
            snapshot.manifestPath = payload.value(QStringLiteral("manifestPath")).toString();
            snapshot.contentHash = payload.value(QStringLiteral("contentHash")).toString();
            snapshot.fileCount = payload.value(QStringLiteral("fileCount")).toInt();
            snapshot.totalBytes = payload.value(QStringLiteral("totalBytes")).toVariant().toLongLong();
            snapshot.metadataJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
            snapshot.createdAt = QDateTime::currentDateTimeUtc();
            const int snapshotId = repository_.insertDatasetSnapshot(snapshot, &error);
            if (state_.training.hasActiveSnapshotTrainingTask
                && state_.training.activeSnapshotTrainingTask.request.datasetPath == datasetPath
                && snapshotId > 0) {
                snapshot.id = snapshotId;
                state_.training.activeSnapshotTrainingTask.request.parameters.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
                state_.training.activeSnapshotTrainingTask.request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
                state_.training.activeSnapshotTrainingTask.request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
                state_.training.activeSnapshotTrainingTask.needsSnapshot = false;
                recordExperimentRunForRequest(state_.training.activeSnapshotTrainingTask.request, state_.training.activeSnapshotTrainingTask.datasetId, &error);
                state_.training.pendingTrainingTasks.prepend(state_.training.activeSnapshotTrainingTask);
                state_.training.hasActiveSnapshotTrainingTask = false;
                state_.training.activeSnapshotTrainingTask = PendingTrainingTask();
            }
        }
    }
    updateTrainingSelectionSummary();
    updateDatasetList();
}

void MainWindow::handleEvaluationReportMessage(const QJsonObject& payload)
{
    if (repository_.isOpen()) {
        aitrain::EvaluationReportRecord report;
        report.taskId = payload.value(QStringLiteral("taskId")).toString(state_.training.currentTaskId);
        report.modelPath = payload.value(QStringLiteral("modelPath")).toString();
        report.taskType = payload.value(QStringLiteral("taskType")).toString();
        report.datasetSnapshotId = payload.value(QStringLiteral("datasetSnapshotId")).toInt();
        report.reportPath = payload.value(QStringLiteral("reportPath")).toString();
        report.summaryJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        report.createdAt = QDateTime::currentDateTimeUtc();
        QString error;
        repository_.insertEvaluationReport(report, &error);
        updateModelRegistry();
    }
}

void MainWindow::handlePipelinePlanMessage(const QJsonObject& payload)
{
    if (repository_.isOpen()) {
        aitrain::PipelineRunRecord pipeline;
        pipeline.name = uiText("本地闭环流水线");
        pipeline.templateId = payload.value(QStringLiteral("templateId")).toString();
        QJsonArray taskIds = payload.value(QStringLiteral("taskIds")).toArray();
        if (taskIds.isEmpty()) {
            const QString fallbackTaskId = payload.value(QStringLiteral("taskId")).toString(state_.training.currentTaskId);
            if (!fallbackTaskId.isEmpty()) {
                taskIds.append(fallbackTaskId);
            }
        }
        pipeline.taskIdsJson = QString::fromUtf8(QJsonDocument(taskIds).toJson(QJsonDocument::Compact));
        pipeline.state = payload.value(QStringLiteral("state")).toString(QStringLiteral("planned"));
        pipeline.summaryJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        pipeline.createdAt = QDateTime::currentDateTimeUtc();
        pipeline.updatedAt = pipeline.createdAt;
        QString error;
        repository_.insertPipelineRun(pipeline, &error);
        registerPipelineModelVersion(payload);
        updateModelRegistry();
    }
}

void MainWindow::handleModelExportMessage(const QJsonObject& payload)
{
    if (exportResultLabel_) {
        const QString exportPath = payload.value(QStringLiteral("exportPath")).toString();
        const QString reportPath = payload.value(QStringLiteral("reportPath")).toString();
        exportResultLabel_->setText(reportPath.isEmpty()
            ? uiText("导出完成：%1").arg(QDir::toNativeSeparators(exportPath))
            : uiText("导出完成：%1；报告：%2").arg(QDir::toNativeSeparators(exportPath), QDir::toNativeSeparators(reportPath)));
    }
    if (repository_.isOpen()) {
        const QJsonObject config = payload.value(QStringLiteral("config")).toObject();
        aitrain::ExportRecord exportRecord;
        exportRecord.taskId = payload.value(QStringLiteral("taskId")).toString(state_.training.currentTaskId);
        exportRecord.sourceCheckpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
        exportRecord.format = payload.value(QStringLiteral("format")).toString();
        exportRecord.path = payload.value(QStringLiteral("exportPath")).toString();
        exportRecord.configJson = QString::fromUtf8(QJsonDocument(config).toJson(QJsonDocument::Compact));
        exportRecord.inputShapeJson = QString::fromUtf8(QJsonDocument(config.value(QStringLiteral("input")).toObject()).toJson(QJsonDocument::Compact));
        exportRecord.outputShapeJson = QString::fromUtf8(QJsonDocument(QJsonObject{{QStringLiteral("outputs"), config.value(QStringLiteral("outputs")).toArray()}}).toJson(QJsonDocument::Compact));
        exportRecord.createdAt = QDateTime::currentDateTimeUtc();
        QString error;
        repository_.insertExport(exportRecord, &error);
    }
    updateModelRegistry();
}

void MainWindow::handleDeploymentValidationMessage(const QJsonObject& payload)
{
    state_.delivery.latestDeploymentValidationReportPath = payload.value(QStringLiteral("reportPath")).toString();
    const QString status = payload.value(QStringLiteral("status")).toString();
    const QString runtime = payload.value(QStringLiteral("runtime")).toString();
    const QString modelPath = payload.value(QStringLiteral("modelPath")).toString();
    if (deploymentValidationResultLabel_) {
        deploymentValidationResultLabel_->setText(uiText("部署验证：%1 | runtime %2 | %3")
            .arg(status, runtime, QDir::toNativeSeparators(state_.delivery.latestDeploymentValidationReportPath)));
    }
    setAcceptanceTableRow(
        deliveryAcceptanceTable_,
        uiText("部署验证"),
        status,
        state_.delivery.latestDeploymentValidationReportPath,
        uiText("模型：%1").arg(QDir::toNativeSeparators(modelPath)));
    updateDeliveryAcceptanceSummary();
    updateModelRegistry();
}

void MainWindow::handleInferenceResultMessage(const QJsonObject& payload)
{
    if (inferenceResultLabel_) {
        inferenceResultLabel_->setText(inferenceSummaryFromPredictions(
            payload.value(QStringLiteral("predictionsPath")).toString(),
            payload));
    }
    loadInferenceOverlay(inferenceOverlayLabel_, payload.value(QStringLiteral("overlayPath")).toString());
}

void MainWindow::handleCustomerOcrAcceptanceMessage(const QJsonObject& payload)
{
    state_.delivery.latestCustomerOcrAcceptanceReportPath = payload.value(QStringLiteral("reportPath")).toString();
    const QString status = payload.value(QStringLiteral("status")).toString();
    const QJsonObject metrics = payload.value(QStringLiteral("metrics")).toObject();
    if (customerOcrStatusLabel_) {
        customerOcrStatusLabel_->setText(uiText("客户域 OCR 验收：%1 | accuracy %2 | CER %3 | %4")
            .arg(status)
            .arg(metrics.value(QStringLiteral("recAccuracy")).toDouble(), 0, 'f', 4)
            .arg(metrics.value(QStringLiteral("recCer")).toDouble(), 0, 'f', 4)
            .arg(QDir::toNativeSeparators(state_.delivery.latestCustomerOcrAcceptanceReportPath)));
    }
    setAcceptanceTableRow(
        deliveryAcceptanceTable_,
        uiText("客户域 OCR"),
        status,
        state_.delivery.latestCustomerOcrAcceptanceReportPath,
        payload.value(QStringLiteral("publicDataBoundary")).toString());
    updateDeliveryAcceptanceSummary();
}

void MainWindow::handleDiagnosticBundleMessage(const QJsonObject& payload)
{
    state_.delivery.latestDiagnosticBundlePath = payload.value(QStringLiteral("bundlePath")).toString();
    const QString manifestPath = payload.value(QStringLiteral("manifestPath")).toString(payload.value(QStringLiteral("reportPath")).toString());
    if (diagnosticsStatusLabel_) {
        diagnosticsStatusLabel_->setText(uiText("诊断包已生成：%1").arg(QDir::toNativeSeparators(state_.delivery.latestDiagnosticBundlePath)));
    }
    setAcceptanceTableRow(
        deliveryAcceptanceTable_,
        uiText("诊断包"),
        QStringLiteral("collected"),
        manifestPath,
        uiText("包含环境、GPU、最近任务、产物和授权摘要。"));
    updateDeliveryAcceptanceSummary();
}
