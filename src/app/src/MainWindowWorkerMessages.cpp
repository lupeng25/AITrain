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

void MainWindow::handleWorkerMessage(const QString& type, const QJsonObject& payload)
{
    if (type == QStringLiteral("progress")) {
        handleProgressMessage(payload);
    } else if (type == QStringLiteral("metric")) {
        handleMetricMessage(payload);
    } else if (type == QStringLiteral("artifact")) {
        handleArtifactMessage(payload);
    } else if (type == QStringLiteral("paused")
        || type == QStringLiteral("resumed")
        || type == QStringLiteral("canceled")
        || type == QStringLiteral("failed")) {
        handleTaskStateMessage(type, payload);
    } else if (type == QStringLiteral("environmentCheck")) {
        updateEnvironmentTable(payload);
    } else if (type == QStringLiteral("datasetValidation")) {
        updateDatasetValidationResult(payload);
    } else if (type == QStringLiteral("datasetSplit")) {
        updateDatasetSplitResult(payload);
    } else if (type == QStringLiteral("datasetQuality")) {
        handleDatasetQualityMessage(payload);
    } else if (type == QStringLiteral("datasetSnapshot")) {
        handleDatasetSnapshotMessage(payload);
    } else if (type == QStringLiteral("evaluationReport")) {
        handleEvaluationReportMessage(payload);
    } else if (type == QStringLiteral("benchmarkReport")) {
        updateModelRegistry();
    } else if (type == QStringLiteral("pipelinePlan")) {
        handlePipelinePlanMessage(payload);
    } else if (type == QStringLiteral("deliveryReport")) {
        updateModelRegistry();
    } else if (type == QStringLiteral("modelExport")) {
        handleModelExportMessage(payload);
    } else if (type == QStringLiteral("inferenceResult")) {
        handleInferenceResultMessage(payload);
    }
}

void MainWindow::handleProgressMessage(const QJsonObject& payload)
{
    if (!currentTaskId_.isEmpty()) {
        progressBar_->setValue(payload.value(QStringLiteral("percent")).toInt());
    }
    const QString message = payload.value(QStringLiteral("message")).toString();
    if (!message.isEmpty()) {
        statusBar()->showMessage(message, 3000);
    }
}

void MainWindow::handleMetricMessage(const QJsonObject& payload)
{
    const QString name = payload.value(QStringLiteral("name")).toString();
    const double value = payload.value(QStringLiteral("value")).toDouble();
    metricsWidget_->addMetric(name, value);

    aitrain::MetricPoint point;
    point.taskId = currentTaskId_;
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
        const QString checkpointName = QFileInfo(path).fileName();
        latestCheckpointLabel_->setText(uiText("最新 checkpoint：%1")
            .arg(checkpointName.isEmpty() ? compactPathForStatus(path, 44) : checkpointName));
        latestCheckpointLabel_->setToolTip(QDir::toNativeSeparators(path));
    } else if (kind == QStringLiteral("preview") && latestPreviewPathLabel_) {
        const QString previewName = QFileInfo(path).fileName();
        latestPreviewPathLabel_->setText(uiText("最新预览：%1")
            .arg(previewName.isEmpty() ? compactPathForStatus(path, 44) : previewName));
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
        artifact.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
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
    if (type == QStringLiteral("paused")) {
        QString error;
        repository_.updateTaskState(currentTaskId_, aitrain::TaskState::Paused, payload.value(QStringLiteral("message")).toString(), &error);
        workerPill_->setStatus(uiText("任务已暂停"), StatusPill::Tone::Warning);
        updateRecentTasks();
        return;
    }

    if (type == QStringLiteral("resumed")) {
        QString error;
        repository_.updateTaskState(currentTaskId_, aitrain::TaskState::Running, payload.value(QStringLiteral("message")).toString(), &error);
        workerPill_->setStatus(uiText("训练运行中"), StatusPill::Tone::Info);
        updateRecentTasks();
        return;
    }

    if (type == QStringLiteral("canceled")) {
        QString error;
        const QString canceledTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
        const QString canceledMessage = payload.value(QStringLiteral("message")).toString();
        repository_.updateTaskState(canceledTaskId, aitrain::TaskState::Canceled, canceledMessage, &error);
        if (hasActiveSnapshotTrainingTask_ && canceledTaskId == currentTaskId_) {
            repository_.updateTaskState(
                activeSnapshotTrainingTask_.taskId,
                aitrain::TaskState::Canceled,
                uiText("自动数据快照已取消，训练未启动。"),
                &error);
            hasActiveSnapshotTrainingTask_ = false;
            activeSnapshotTrainingTask_ = PendingTrainingTask();
        }
        workerPill_->setStatus(uiText("任务已取消"), StatusPill::Tone::Warning);
        appendLog(uiText("任务已取消：%1").arg(canceledMessage));
        currentTaskId_.clear();
        updateRecentTasks();
        startNextQueuedTask();
        return;
    }

    if (type == QStringLiteral("failed")) {
        const QString failedTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
        const QString failedMessage = payload.value(QStringLiteral("message")).toString();
        QString error;
        repository_.updateTaskState(
            failedTaskId,
            aitrain::TaskState::Failed,
            failedMessage,
            &error);
        if (hasActiveSnapshotTrainingTask_ && failedTaskId == currentTaskId_) {
            repository_.updateTaskState(
                activeSnapshotTrainingTask_.taskId,
                aitrain::TaskState::Failed,
                uiText("自动数据快照失败：%1").arg(failedMessage),
                &error);
            hasActiveSnapshotTrainingTask_ = false;
            activeSnapshotTrainingTask_ = PendingTrainingTask();
        }
        updateRecentTasks();
        updateModelRegistry();
    }
}

void MainWindow::handleDatasetQualityMessage(const QJsonObject& payload)
{
    latestQualityFixListPath_ = payload.value(QStringLiteral("xAnyLabelingFixListPath")).toString();
    latestQualityFixManifestPath_ = payload.value(QStringLiteral("xAnyLabelingFixManifestPath")).toString();
    latestQualityReportPath_ = payload.value(QStringLiteral("reportPath")).toString();
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
            .arg(latestQualityFixListPath_.isEmpty() ? uiText("暂无") : QDir::toNativeSeparators(latestQualityFixListPath_)));
    }
    updateDatasetRepairLoopFromQuality(payload);
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    if (!datasetPath.isEmpty()) {
        currentDatasetPath_ = datasetPath;
        currentDatasetFormat_ = format;
        currentDatasetValid_ = payload.value(QStringLiteral("ok")).toBool();
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
            seed.format = payload.value(QStringLiteral("format")).toString(currentDatasetFormat_);
            seed.rootPath = datasetPath;
            seed.validationStatus = currentDatasetValid_ ? QStringLiteral("valid") : QStringLiteral("snapshot");
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
            if (hasActiveSnapshotTrainingTask_
                && activeSnapshotTrainingTask_.request.datasetPath == datasetPath
                && snapshotId > 0) {
                snapshot.id = snapshotId;
                activeSnapshotTrainingTask_.request.parameters.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
                activeSnapshotTrainingTask_.request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
                activeSnapshotTrainingTask_.request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
                activeSnapshotTrainingTask_.needsSnapshot = false;
                recordExperimentRunForRequest(activeSnapshotTrainingTask_.request, activeSnapshotTrainingTask_.datasetId, &error);
                pendingTrainingTasks_.prepend(activeSnapshotTrainingTask_);
                hasActiveSnapshotTrainingTask_ = false;
                activeSnapshotTrainingTask_ = PendingTrainingTask();
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
        report.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
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
            const QString fallbackTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
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
        exportRecord.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
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

void MainWindow::handleInferenceResultMessage(const QJsonObject& payload)
{
    if (inferenceResultLabel_) {
        inferenceResultLabel_->setText(inferenceSummaryFromPredictions(
            payload.value(QStringLiteral("predictionsPath")).toString(),
            payload));
    }
    loadInferenceOverlay(inferenceOverlayLabel_, payload.value(QStringLiteral("overlayPath")).toString());
}
