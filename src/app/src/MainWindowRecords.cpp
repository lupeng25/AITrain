#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"
#include "TaskArtifactPanel.h"
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

void MainWindow::updateRecentTasks()
{
    if (!repository_.isOpen()) {
        updateDashboardSummary();
        return;
    }
    QString error;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(20, &error);
    if (dashboardTaskValue_) {
        dashboardTaskValue_->setText(QString::number(tasks.size()));
    }
    if (recentTasksTable_) {
        updateTaskTable(recentTasksTable_, tasks);
    }
    if (taskQueueTable_) {
        updateTaskTable(taskQueueTable_, tasks);
        applyTaskFilters();
    }
    updateDashboardSummary();
}

void MainWindow::updateDatasetList()
{
    if (!repository_.isOpen()) {
        updateDashboardSummary();
        return;
    }

    QString error;
    const QVector<aitrain::DatasetRecord> datasets = repository_.recentDatasets(50, &error);
    if (datasetListTable_) {
        datasetListTable_->setRowCount(0);
    }
    if (datasetListTable_ && datasets.isEmpty()) {
        datasetListTable_->insertRow(0);
        datasetListTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无数据集记录")));
        for (int column = 1; column < datasetListTable_->columnCount(); ++column) {
            datasetListTable_->setItem(0, column, new QTableWidgetItem(QString()));
        }
        updateDashboardSummary();
        return;
    }

    if (datasetListTable_) {
        for (const aitrain::DatasetRecord& dataset : datasets) {
            const int row = datasetListTable_->rowCount();
            datasetListTable_->insertRow(row);
            auto* nameItem = new QTableWidgetItem(dataset.name);
            nameItem->setData(Qt::UserRole, dataset.id);
            datasetListTable_->setItem(row, 0, nameItem);
            auto* formatItem = new QTableWidgetItem(datasetFormatLabel(dataset.format));
            formatItem->setData(Qt::UserRole, dataset.format);
            datasetListTable_->setItem(row, 1, formatItem);
            auto* statusItem = new QTableWidgetItem(dataset.validationStatus == QStringLiteral("valid") ? uiText("通过") : uiText("未通过"));
            statusItem->setData(Qt::UserRole, dataset.validationStatus);
            datasetListTable_->setItem(row, 2, statusItem);
            datasetListTable_->setItem(row, 3, new QTableWidgetItem(QString::number(dataset.sampleCount)));
            auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(dataset.rootPath));
            pathItem->setData(Qt::UserRole, dataset.rootPath);
            datasetListTable_->setItem(row, 4, pathItem);
        }
    }
    updateDashboardSummary();
}

void MainWindow::updateAnnotationToolStatus()
{
    if (annotationToolStatusLabel_) {
        annotationToolStatusLabel_->setText(xAnyLabelingStatusText());
        annotationToolStatusLabel_->setToolTip(QDir::toNativeSeparators(resolvedXAnyLabelingProgram()));
    }
}

void MainWindow::refreshAfterAnnotation()
{
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (datasetPath.isEmpty()) {
        QMessageBox::information(this, uiText("标注后刷新"), uiText("请先选择数据集目录。"));
        return;
    }

    const QString detectedFormat = detectDatasetFormatFromPath(datasetPath);
    if (!detectedFormat.isEmpty() && datasetFormatCombo_) {
        const int index = datasetFormatCombo_->findData(detectedFormat);
        if (index >= 0) {
            datasetFormatCombo_->setCurrentIndex(index);
        }
    }
    currentDatasetPath_ = datasetPath;
    const QString selectedFormat = currentDatasetFormat();
    currentDatasetFormat_ = selectedFormat.isEmpty() ? detectedFormat : selectedFormat;
    currentDatasetValid_ = false;
    updateTrainingSelectionSummary();
    refreshTrainingDefaults();
    validateDataset();
}

void MainWindow::applyTaskFilters()
{
    if (!taskQueueTable_ || taskQueueTable_->columnCount() < 7) {
        return;
    }

    const QString kind = currentTaskKindFilter();
    const QString state = currentTaskStateFilter();
    const QString query = taskSearchEdit_ ? taskSearchEdit_->text().trimmed() : QString();

    for (int row = 0; row < taskQueueTable_->rowCount(); ++row) {
        const QString rowKind = taskQueueTable_->item(row, 0)
            ? taskQueueTable_->item(row, 0)->data(Qt::UserRole + 1).toString()
            : QString();
        if (rowKind == QStringLiteral("empty")) {
            taskQueueTable_->setRowHidden(row, false);
            continue;
        }

        bool visible = true;
        if (!kind.isEmpty()) {
            visible = visible && taskQueueTable_->item(row, 1)
                && taskQueueTable_->item(row, 1)->data(Qt::UserRole).toString() == kind;
        }
        if (!state.isEmpty()) {
            visible = visible && taskQueueTable_->item(row, 4)
                && taskQueueTable_->item(row, 4)->data(Qt::UserRole).toString() == state;
        }
        if (!query.isEmpty()) {
            bool matched = false;
            for (int column = 0; column < taskQueueTable_->columnCount(); ++column) {
                auto* item = taskQueueTable_->item(row, column);
                if (item && item->text().contains(query, Qt::CaseInsensitive)) {
                    matched = true;
                    break;
                }
            }
            visible = visible && matched;
        }
        taskQueueTable_->setRowHidden(row, !visible);
    }

    ensureVisibleTaskSelection();
}

void MainWindow::ensureVisibleTaskSelection()
{
    if (!taskQueueTable_) {
        return;
    }

    auto isSelectableRow = [this](int row) {
        if (row < 0 || row >= taskQueueTable_->rowCount() || taskQueueTable_->isRowHidden(row)) {
            return false;
        }
        auto* idItem = taskQueueTable_->item(row, 0);
        if (!idItem) {
            return false;
        }
        if (idItem->data(Qt::UserRole + 1).toString() == QStringLiteral("empty")) {
            return false;
        }
        return !idItem->data(Qt::UserRole).toString().isEmpty();
    };

    int selectedRow = -1;
    if (!taskQueueTable_->selectedItems().isEmpty()) {
        selectedRow = taskQueueTable_->selectedItems().first()->row();
    }
    if (isSelectableRow(selectedRow)) {
        return;
    }

    int preferredRow = -1;
    int fallbackRow = -1;
    for (int row = 0; row < taskQueueTable_->rowCount(); ++row) {
        if (!isSelectableRow(row)) {
            continue;
        }
        if (fallbackRow < 0) {
            fallbackRow = row;
        }
        auto* kindItem = taskQueueTable_->item(row, 1);
        if (kindItem && kindItem->data(Qt::UserRole).toString() == QStringLiteral("evaluate")) {
            preferredRow = row;
            break;
        }
    }

    const int rowToSelect = preferredRow >= 0 ? preferredRow : fallbackRow;
    if (rowToSelect >= 0) {
        const QSignalBlocker blocker(taskQueueTable_);
        taskQueueTable_->clearSelection();
        taskQueueTable_->selectRow(rowToSelect);
        taskQueueTable_->setCurrentCell(rowToSelect, 0);
        updateSelectedTaskDetails();
        return;
    }

    {
        const QSignalBlocker blocker(taskQueueTable_);
        taskQueueTable_->clearSelection();
    }
    clearSelectedTaskDetails();
}

void MainWindow::clearSelectedTaskDetails()
{
    if (taskArtifactPanel_) {
        taskArtifactPanel_->clear();
    }
}

void MainWindow::updateTaskTable(QTableWidget* table, const QVector<aitrain::TaskRecord>& tasks)
{
    if (!table) {
        return;
    }

    table->setRowCount(0);
    if (tasks.isEmpty()) {
        table->insertRow(0);
        auto* item = new QTableWidgetItem(uiText("暂无任务记录"));
        item->setData(Qt::UserRole + 1, QStringLiteral("empty"));
        table->setItem(0, 0, item);
        for (int column = 1; column < table->columnCount(); ++column) {
            table->setItem(0, column, new QTableWidgetItem(QString()));
        }
        return;
    }

    for (const aitrain::TaskRecord& task : tasks) {
        const int row = table->rowCount();
        table->insertRow(row);
        auto* idItem = new QTableWidgetItem(task.id.left(8));
        idItem->setData(Qt::UserRole, task.id);
        table->setItem(row, 0, idItem);
        if (table->columnCount() == 5) {
            table->setItem(row, 1, new QTableWidgetItem(task.pluginId));
            auto* typeItem = new QTableWidgetItem(taskTypeLabel(task.taskType));
            typeItem->setData(Qt::UserRole, task.taskType);
            table->setItem(row, 2, typeItem);
            auto* stateItem = new QTableWidgetItem(taskStateLabel(task.state));
            stateItem->setData(Qt::UserRole, static_cast<int>(task.state));
            table->setItem(row, 3, stateItem);
            table->setItem(row, 4, new QTableWidgetItem(task.message));
        } else {
            auto* kindItem = new QTableWidgetItem(taskKindLabel(task.kind));
            QString kindData;
            switch (task.kind) {
            case aitrain::TaskKind::Train: kindData = QStringLiteral("train"); break;
            case aitrain::TaskKind::Validate: kindData = QStringLiteral("validate"); break;
            case aitrain::TaskKind::Export: kindData = QStringLiteral("export"); break;
            case aitrain::TaskKind::Infer: kindData = QStringLiteral("infer"); break;
            case aitrain::TaskKind::Evaluate: kindData = QStringLiteral("evaluate"); break;
            case aitrain::TaskKind::Benchmark: kindData = QStringLiteral("benchmark"); break;
            case aitrain::TaskKind::Curate: kindData = QStringLiteral("curate"); break;
            case aitrain::TaskKind::Snapshot: kindData = QStringLiteral("snapshot"); break;
            case aitrain::TaskKind::Pipeline: kindData = QStringLiteral("pipeline"); break;
            case aitrain::TaskKind::Report: kindData = QStringLiteral("report"); break;
            }
            kindItem->setData(Qt::UserRole, kindData);
            table->setItem(row, 1, kindItem);
            table->setItem(row, 2, new QTableWidgetItem(task.pluginId));
            auto* typeItem = new QTableWidgetItem(taskTypeLabel(task.taskType));
            typeItem->setData(Qt::UserRole, task.taskType);
            table->setItem(row, 3, typeItem);
            auto* stateItem = new QTableWidgetItem(taskStateLabel(task.state));
            QString stateData;
            switch (task.state) {
            case aitrain::TaskState::Queued: stateData = QStringLiteral("queued"); break;
            case aitrain::TaskState::Running: stateData = QStringLiteral("running"); break;
            case aitrain::TaskState::Paused: stateData = QStringLiteral("paused"); break;
            case aitrain::TaskState::Completed: stateData = QStringLiteral("completed"); break;
            case aitrain::TaskState::Failed: stateData = QStringLiteral("failed"); break;
            case aitrain::TaskState::Canceled: stateData = QStringLiteral("canceled"); break;
            }
            stateItem->setData(Qt::UserRole, stateData);
            table->setItem(row, 4, stateItem);
            table->setItem(row, 5, new QTableWidgetItem(task.updatedAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            table->setItem(row, 6, new QTableWidgetItem(task.message));
        }
    }
}

QString MainWindow::createRepositoryTask(aitrain::TaskKind kind,
    const QString& taskType,
    const QString& pluginId,
    const QString& workDir,
    const QString& message,
    const QString& requestedTaskId)
{
    if (!repository_.isOpen()) {
        return QString();
    }

    const QString taskId = requestedTaskId.isEmpty()
        ? QUuid::createUuid().toString(QUuid::WithoutBraces)
        : requestedTaskId;
    aitrain::TaskRecord record;
    record.id = taskId;
    record.projectName = currentProjectName_.isEmpty() ? QStringLiteral("manual") : currentProjectName_;
    record.pluginId = pluginId;
    record.taskType = taskType;
    record.kind = kind;
    record.state = aitrain::TaskState::Queued;
    record.workDir = workDir;
    record.message = message;
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;

    QString error;
    if (!repository_.insertTask(record, &error)) {
        QMessageBox::critical(this, uiText("任务"), error);
        return QString();
    }
    if (!repository_.updateTaskState(taskId, aitrain::TaskState::Running, message, &error)) {
        QMessageBox::critical(this, uiText("任务"), error);
        return QString();
    }
    currentTaskId_ = taskId;
    updateRecentTasks();
    return taskId;
}

bool MainWindow::attachLatestSnapshotToRequest(aitrain::TrainingRequest& request, int datasetId, QString* error)
{
    if (!repository_.isOpen() || datasetId <= 0) {
        return false;
    }

    const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(datasetId, error);
    if (snapshot.id <= 0 || snapshot.manifestPath.isEmpty() || !QFileInfo::exists(snapshot.manifestPath)) {
        return false;
    }

    request.parameters.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
    request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
    request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
    return true;
}

int MainWindow::recordExperimentRunForRequest(const aitrain::TrainingRequest& request, int datasetId, QString* error)
{
    if (!repository_.isOpen() || request.taskId.isEmpty()) {
        return 0;
    }

    const int snapshotId = request.parameters.value(QStringLiteral("datasetSnapshotId")).toInt();
    if (snapshotId <= 0) {
        return 0;
    }

    aitrain::ExperimentRecord experiment;
    const QString datasetName = QFileInfo(request.datasetPath).fileName().isEmpty()
        ? QStringLiteral("dataset")
        : QFileInfo(request.datasetPath).fileName();
    experiment.name = QStringLiteral("%1 / %2").arg(datasetName, taskTypeLabel(request.taskType));
    experiment.taskType = request.taskType;
    experiment.datasetId = datasetId;
    experiment.notes = uiText("Phase 35 自动记录的本地复现实验。");
    experiment.tagsJson = QString::fromUtf8(QJsonDocument(QJsonArray{
        QStringLiteral("phase35"),
        QStringLiteral("local-reproducible")
    }).toJson(QJsonDocument::Compact));
    experiment.createdAt = QDateTime::currentDateTimeUtc();
    experiment.updatedAt = experiment.createdAt;
    const int experimentId = repository_.upsertExperiment(experiment, error);
    if (experimentId <= 0) {
        return 0;
    }

    aitrain::ExperimentRunRecord run;
    run.experimentId = experimentId;
    run.taskId = request.taskId;
    run.trainingBackend = request.parameters.value(QStringLiteral("trainingBackend")).toString();
    run.modelPreset = request.parameters.value(QStringLiteral("modelPreset")).toString(
        request.parameters.value(QStringLiteral("model")).toString());
    run.datasetSnapshotId = snapshotId;
    run.requestJson = QString::fromUtf8(QJsonDocument(request.toJson()).toJson(QJsonDocument::Compact));
    run.environmentJson = QStringLiteral("{}");
    run.bestMetricsJson = QStringLiteral("{}");
    run.artifactSummaryJson = QStringLiteral("[]");
    run.createdAt = QDateTime::currentDateTimeUtc();
    run.updatedAt = run.createdAt;
    return repository_.insertExperimentRun(run, error);
}

void MainWindow::updateExperimentRunSummary(const QString& taskId)
{
    if (!repository_.isOpen() || taskId.isEmpty()) {
        return;
    }

    QString error;
    const aitrain::ExperimentRunRecord run = repository_.experimentRunForTask(taskId, &error);
    if (run.id <= 0) {
        return;
    }

    QJsonObject bestMetrics;
    const QVector<aitrain::MetricPoint> metrics = repository_.metricsForTask(taskId, &error);
    for (const aitrain::MetricPoint& metric : metrics) {
        const QString key = metric.name;
        if (key.isEmpty()) {
            continue;
        }
        const QString lower = key.toLower();
        const bool minimize = lower.contains(QStringLiteral("loss"))
            || lower.contains(QStringLiteral("distance"))
            || lower == QStringLiteral("cer")
            || lower == QStringLiteral("wer");
        if (!bestMetrics.contains(key)
            || (minimize && metric.value < bestMetrics.value(key).toDouble())
            || (!minimize && metric.value > bestMetrics.value(key).toDouble())) {
            bestMetrics.insert(key, metric.value);
        }
    }

    QJsonArray artifactSummary;
    const QVector<aitrain::ArtifactRecord> artifacts = repository_.artifactsForTask(taskId, &error);
    for (const aitrain::ArtifactRecord& artifact : artifacts) {
        artifactSummary.append(QJsonObject{
            {QStringLiteral("kind"), artifact.kind},
            {QStringLiteral("path"), artifact.path},
            {QStringLiteral("message"), artifact.message}
        });
    }

    repository_.updateExperimentRunSummary(
        taskId,
        QString::fromUtf8(QJsonDocument(bestMetrics).toJson(QJsonDocument::Compact)),
        QString::fromUtf8(QJsonDocument(artifactSummary).toJson(QJsonDocument::Compact)),
        &error);
}

void MainWindow::registerPipelineModelVersion(const QJsonObject& payload)
{
    if (!repository_.isOpen()) {
        return;
    }
    if (payload.value(QStringLiteral("state")).toString() != QStringLiteral("completed")) {
        return;
    }

    const QString sourceTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
    const QString exportPath = payload.value(QStringLiteral("exportPath")).toString();
    const QString modelPath = exportPath.isEmpty()
        ? payload.value(QStringLiteral("modelPath")).toString()
        : exportPath;
    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        return;
    }

    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    QString error;
    const aitrain::ExperimentRunRecord run = repository_.experimentRunForTask(sourceTaskId, &error);

    QJsonObject metricSummary;
    const QString evaluationReportPath = payload.value(QStringLiteral("evaluationReportPath")).toString();
    const QString benchmarkReportPath = payload.value(QStringLiteral("benchmarkReportPath")).toString();
    const QString deliveryReportPath = payload.value(QStringLiteral("deliveryReportPath")).toString();
    const QJsonObject evaluationSummary = compactEvaluationSummary(evaluationReportPath);
    const QJsonObject benchmarkSummary = compactBenchmarkSummary(benchmarkReportPath);
    if (!evaluationSummary.isEmpty()) {
        metricSummary.insert(QStringLiteral("evaluation"), evaluationSummary);
    }
    if (!benchmarkSummary.isEmpty()) {
        metricSummary.insert(QStringLiteral("benchmark"), benchmarkSummary);
    }
    metricSummary.insert(QStringLiteral("lineage"), QJsonObject{
        {QStringLiteral("sourceTaskId"), sourceTaskId},
        {QStringLiteral("datasetSnapshotId"), payload.value(QStringLiteral("datasetSnapshotId")).toInt()},
        {QStringLiteral("trainingBackend"), options.value(QStringLiteral("trainingBackend")).toString()},
        {QStringLiteral("modelPreset"), options.value(QStringLiteral("model")).toString()}
    });
    metricSummary.insert(QStringLiteral("artifacts"), QJsonObject{
        {QStringLiteral("modelPath"), modelPath},
        {QStringLiteral("checkpointPath"), payload.value(QStringLiteral("modelPath")).toString()},
        {QStringLiteral("exportPath"), exportPath},
        {QStringLiteral("evaluationReportPath"), evaluationReportPath},
        {QStringLiteral("benchmarkReportPath"), benchmarkReportPath},
        {QStringLiteral("deliveryReportPath"), deliveryReportPath}
    });
    QJsonArray limitations;
    const QString backendForLimit = options.value(QStringLiteral("trainingBackend")).toString();
    if (backendForLimit == QStringLiteral("tiny_linear_detector")
        || backendForLimit == QStringLiteral("python_mock")) {
        limitations.append(QStringLiteral("scaffold-or-diagnostic-backend"));
    }
    if (benchmarkSummary.value(QStringLiteral("failureCategory")).toString() == QStringLiteral("hardware-blocked")
        || benchmarkSummary.value(QStringLiteral("deploymentConclusion")).toString() == QStringLiteral("hardware-blocked")) {
        limitations.append(QStringLiteral("tensorrt-hardware-blocked"));
    }
    if (!limitations.isEmpty()) {
        metricSummary.insert(QStringLiteral("limitations"), limitations);
    }

    const QString backend = options.value(QStringLiteral("trainingBackend")).toString();
    const QString datasetName = QFileInfo(options.value(QStringLiteral("datasetPath")).toString()).fileName();
    const QString modelName = QStringLiteral("%1_%2_%3")
        .arg(currentProjectName_.isEmpty() ? QStringLiteral("pipeline_model") : currentProjectName_)
        .arg(datasetName.isEmpty() ? QStringLiteral("dataset") : datasetName)
        .arg(backend.isEmpty() ? QStringLiteral("pipeline") : backend);
    const QString version = QStringLiteral("pipeline_%1")
        .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMddHHmmss")));

    aitrain::ModelVersionRecord record;
    record.modelName = modelName;
    record.version = version;
    record.sourceTaskId = sourceTaskId;
    record.experimentRunId = run.id;
    record.datasetSnapshotId = run.datasetSnapshotId > 0
        ? run.datasetSnapshotId
        : payload.value(QStringLiteral("datasetSnapshotId")).toInt();
    record.status = QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) == 0
        ? QStringLiteral("exported")
        : QStringLiteral("validated");
    record.notes = uiText("由本地流水线自动注册。");
    record.metricsJson = QString::fromUtf8(QJsonDocument(metricSummary).toJson(QJsonDocument::Compact));
    if (QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) == 0) {
        record.onnxPath = modelPath;
        record.checkpointPath = payload.value(QStringLiteral("modelPath")).toString();
    } else {
        record.checkpointPath = modelPath;
    }
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;
    repository_.upsertModelVersion(record, &error);
}

void MainWindow::updateModelRegistry()
{
    if (!repository_.isOpen()) {
        if (modelVersionTable_) {
            modelVersionTable_->setRowCount(0);
            modelVersionTable_->insertRow(0);
            modelVersionTable_->setItem(0, 0, new QTableWidgetItem(uiText("请先打开项目")));
        }
        if (evaluationReportTable_) {
            evaluationReportTable_->setRowCount(0);
        }
        if (pipelineRunTable_) {
            pipelineRunTable_->setRowCount(0);
        }
        return;
    }

    QString error;
    const QVector<aitrain::ModelVersionRecord> models = repository_.recentModelVersions(200, &error);
    const QVector<aitrain::EvaluationReportRecord> reports = repository_.recentEvaluationReports(100, &error);
    const QVector<aitrain::PipelineRunRecord> pipelines = repository_.recentPipelineRuns(100, &error);

    if (modelRegistrySummaryLabel_) {
        modelRegistrySummaryLabel_->setText(uiText("模型版本：%1；评估报告：%2；流水线记录：%3。评估、基准和报告 v1 通过 Worker 生成 artifact，完整质量分析仍按 scaffold 标注。")
            .arg(models.size())
            .arg(reports.size())
            .arg(pipelines.size()));
    }

    if (modelVersionTable_) {
        modelVersionTable_->setRowCount(0);
        if (models.isEmpty()) {
            modelVersionTable_->insertRow(0);
            modelVersionTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无模型版本")));
            for (int column = 1; column < modelVersionTable_->columnCount(); ++column) {
                modelVersionTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::ModelVersionRecord& model : models) {
                const int row = modelVersionTable_->rowCount();
                modelVersionTable_->insertRow(row);
                modelVersionTable_->setItem(row, 0, new QTableWidgetItem(model.modelName));
                modelVersionTable_->setItem(row, 1, new QTableWidgetItem(model.version));
                modelVersionTable_->setItem(row, 2, new QTableWidgetItem(model.status));
                auto* checkpointItem = new QTableWidgetItem(QDir::toNativeSeparators(model.checkpointPath));
                checkpointItem->setData(Qt::UserRole, model.checkpointPath);
                modelVersionTable_->setItem(row, 3, checkpointItem);
                auto* onnxItem = new QTableWidgetItem(QDir::toNativeSeparators(model.onnxPath));
                onnxItem->setData(Qt::UserRole, model.onnxPath);
                modelVersionTable_->setItem(row, 4, onnxItem);
                modelVersionTable_->setItem(row, 5, new QTableWidgetItem(model.sourceTaskId.left(8)));

                const QJsonObject metrics = QJsonDocument::fromJson(model.metricsJson.toUtf8()).object();
                const QString metricsText = metrics.isEmpty()
                    ? model.metricsJson
                    : modelSummaryText(metrics);
                modelVersionTable_->setItem(row, 6, new QTableWidgetItem(metricsText));
                modelVersionTable_->setItem(row, 7, new QTableWidgetItem(model.updatedAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }

    if (evaluationReportTable_) {
        evaluationReportTable_->setRowCount(0);
        if (reports.isEmpty()) {
            evaluationReportTable_->insertRow(0);
            evaluationReportTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无评估报告")));
            for (int column = 1; column < evaluationReportTable_->columnCount(); ++column) {
                evaluationReportTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::EvaluationReportRecord& report : reports) {
                const int row = evaluationReportTable_->rowCount();
                evaluationReportTable_->insertRow(row);
                evaluationReportTable_->setItem(row, 0, new QTableWidgetItem(report.taskId.left(8)));
                evaluationReportTable_->setItem(row, 1, new QTableWidgetItem(taskTypeLabel(report.taskType)));
                evaluationReportTable_->setItem(row, 2, new QTableWidgetItem(QDir::toNativeSeparators(report.modelPath)));
                auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(report.reportPath));
                pathItem->setData(Qt::UserRole, report.reportPath);
                evaluationReportTable_->setItem(row, 3, pathItem);
                evaluationReportTable_->setItem(row, 4, new QTableWidgetItem(report.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
            if (evaluationReportTable_->rowCount() > 0) {
                evaluationReportTable_->selectRow(0);
            }
        }
    }
    updateSelectedEvaluationReportDetails();

    if (pipelineRunTable_) {
        pipelineRunTable_->setRowCount(0);
        if (pipelines.isEmpty()) {
            pipelineRunTable_->insertRow(0);
            pipelineRunTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无流水线记录")));
            for (int column = 1; column < pipelineRunTable_->columnCount(); ++column) {
                pipelineRunTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::PipelineRunRecord& pipeline : pipelines) {
                const int row = pipelineRunTable_->rowCount();
                pipelineRunTable_->insertRow(row);
                pipelineRunTable_->setItem(row, 0, new QTableWidgetItem(pipeline.name));
                pipelineRunTable_->setItem(row, 1, new QTableWidgetItem(pipeline.templateId));
                pipelineRunTable_->setItem(row, 2, new QTableWidgetItem(pipeline.state));
                pipelineRunTable_->setItem(row, 3, new QTableWidgetItem(pipeline.summaryJson.left(160)));
                pipelineRunTable_->setItem(row, 4, new QTableWidgetItem(pipeline.updatedAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }
}

void MainWindow::refreshModelRegistry()
{
    updateModelRegistry();
}

void MainWindow::updateDatasetValidationResult(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const int sampleCount = payload.value(QStringLiteral("sampleCount")).toInt();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    const QJsonArray issues = payload.value(QStringLiteral("issues")).toArray();

    currentDatasetPath_ = datasetPath;
    currentDatasetFormat_ = format;
    currentDatasetValid_ = ok;

    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(ok
            ? uiText("校验通过：%1 个样本。").arg(sampleCount)
            : uiText("校验失败：发现 %1 个问题，训练已被阻止。").arg(issues.size()));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (datasetPreviewTable_) {
        datasetPreviewTable_->setRowCount(0);
        const QJsonArray previewSamples = payload.value(QStringLiteral("previewSamples")).toArray();
        if (previewSamples.isEmpty()) {
            datasetPreviewTable_->insertRow(0);
            datasetPreviewTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无样本预览")));
            datasetPreviewTable_->setItem(0, 1, new QTableWidgetItem(QString()));
        } else {
            for (const QJsonValue& sampleValue : previewSamples) {
                const QString sample = sampleValue.toString();
                const QStringList parts = sample.split(QLatin1Char('\t'));
                const int row = datasetPreviewTable_->rowCount();
                datasetPreviewTable_->insertRow(row);
                datasetPreviewTable_->setItem(row, 0, new QTableWidgetItem(parts.value(0)));
                datasetPreviewTable_->setItem(row, 1, new QTableWidgetItem(parts.size() > 1 ? parts.mid(1).join(QStringLiteral("\t")) : uiText("标注文件已匹配")));
            }
        }
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        if (issues.isEmpty()) {
            validationIssuesTable_->insertRow(0);
            validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(uiText("通过")));
            validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("ok")));
            validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(datasetPath));
            validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
            validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(uiText("未发现数据集问题。")));
        } else {
            for (const QJsonValue& value : issues) {
                const QJsonObject issue = value.toObject();
                const int row = validationIssuesTable_->rowCount();
                validationIssuesTable_->insertRow(row);
                validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(issueSeverityLabel(issue.value(QStringLiteral("severity")).toString())));
                validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(issue.value(QStringLiteral("code")).toString()));
                validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(issue.value(QStringLiteral("filePath")).toString()));
                const int line = issue.value(QStringLiteral("line")).toInt();
                validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(line > 0 ? QString::number(line) : QString()));
                validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(issue.value(QStringLiteral("message")).toString()));
            }
        }
    }

    if (repository_.isOpen()) {
        aitrain::DatasetRecord dataset;
        dataset.name = QFileInfo(datasetPath).fileName();
        dataset.format = format;
        dataset.rootPath = datasetPath;
        dataset.validationStatus = ok ? QStringLiteral("valid") : QStringLiteral("invalid");
        dataset.sampleCount = sampleCount;
        dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
        QString error;
        repository_.upsertDatasetValidation(dataset, &error);
        updateDatasetList();
    }

    updateTrainingSelectionSummary();
    refreshTrainingDefaults();
    updateDashboardSummary();
    workerPill_->setStatus(uiText("Worker 空闲"), StatusPill::Tone::Neutral);
    statusBar()->showMessage(ok ? uiText("数据集校验通过") : uiText("数据集校验失败"), 5000);
}

void MainWindow::updateDatasetSplitResult(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString(QStringLiteral("yolo_detection"));
    const int trainCount = payload.value(QStringLiteral("trainCount")).toInt();
    const int valCount = payload.value(QStringLiteral("valCount")).toInt();
    const int testCount = payload.value(QStringLiteral("testCount")).toInt();
    const QJsonArray errors = payload.value(QStringLiteral("errors")).toArray();

    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(ok
            ? uiText("划分完成：train %1 / val %2 / test %3。").arg(trainCount).arg(valCount).arg(testCount)
            : uiText("划分失败：%1 个错误。").arg(errors.size()));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        if (ok) {
            validationIssuesTable_->insertRow(0);
            validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(uiText("通过")));
            validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("split_ok")));
            validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(outputPath));
            validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
            validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(uiText("数据集已复制到标准划分目录。")));
        } else {
            for (const QJsonValue& value : errors) {
                const int row = validationIssuesTable_->rowCount();
                validationIssuesTable_->insertRow(row);
                validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(uiText("错误")));
                validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(QStringLiteral("split_error")));
                validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(outputPath));
                validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(QString()));
                validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(value.toString()));
            }
        }
    }

    if (ok && repository_.isOpen()) {
        aitrain::DatasetRecord dataset;
        dataset.name = QFileInfo(outputPath).fileName();
        dataset.format = format;
        dataset.rootPath = outputPath;
        dataset.validationStatus = QStringLiteral("valid");
        dataset.sampleCount = trainCount + valCount + testCount;
        dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
        QString error;
        repository_.upsertDatasetValidation(dataset, &error);
        updateDatasetList();
        datasetPathEdit_->setText(QDir::toNativeSeparators(outputPath));
        setComboCurrentData(datasetFormatCombo_, format);
        currentDatasetPath_ = outputPath;
        currentDatasetFormat_ = format;
        currentDatasetValid_ = true;
    }

    updateTrainingSelectionSummary();
    refreshTrainingDefaults();
    updateDashboardSummary();
    workerPill_->setStatus(uiText("Worker 空闲"), StatusPill::Tone::Neutral);
    statusBar()->showMessage(ok ? uiText("数据集划分完成") : uiText("数据集划分失败"), 5000);
}
