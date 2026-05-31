#include "MainWindow.h"

#include "DatasetConversionUiModel.h"
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
#include <QTextStream>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

using namespace aitrain_app;

void MainWindow::cancelSelectedTask()
{
    if (!taskQueueTable_ || !repository_.isOpen()) {
        return;
    }

    const int row = taskQueueTable_->currentRow();
    if (row < 0 || !taskQueueTable_->item(row, 0)) {
        QMessageBox::information(this, uiText("任务队列"), uiText("请先选择一个任务。"));
        return;
    }

    const QString taskId = taskQueueTable_->item(row, 0)->data(Qt::UserRole).toString();
    if (taskId.isEmpty()) {
        return;
    }

    QString error;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(200, &error);
    for (const aitrain::TaskRecord& task : tasks) {
        if (task.id != taskId) {
            continue;
        }

        if (task.state == aitrain::TaskState::Queued) {
            for (int index = 0; index < state_.training.pendingTrainingTasks.size(); ++index) {
                if (state_.training.pendingTrainingTasks.at(index).taskId == taskId) {
                    state_.training.pendingTrainingTasks.remove(index);
                    break;
                }
            }
            if (!repository_.updateTaskState(taskId, aitrain::TaskState::Canceled, uiText("用户取消排队任务。"), &error)) {
                QMessageBox::warning(this, uiText("任务队列"), error);
            }
            updateRecentTasks();
            return;
        }

        if ((task.state == aitrain::TaskState::Running || task.state == aitrain::TaskState::Paused) && taskId == state_.training.currentTaskId) {
            worker_.cancel();
            return;
        }

        QMessageBox::information(this, uiText("任务队列"), uiText("只能取消排队任务或当前 Worker 正在运行的任务。"));
        return;
    }
}

void MainWindow::reproduceSelectedTrainingTask()
{
    if (!repository_.isOpen()) {
        QMessageBox::information(this, uiText("复现实验"), uiText("请先打开项目。"));
        return;
    }
    if (currentProjectPath_.isEmpty()) {
        QMessageBox::information(this, uiText("复现实验"), uiText("请先打开或创建项目。"));
        return;
    }

    const QString sourceTaskId = selectedTaskId();
    if (sourceTaskId.isEmpty()) {
        QMessageBox::information(this, uiText("复现实验"), uiText("请先选择一个训练任务。"));
        return;
    }

    QString error;
    aitrain::TaskRecord sourceTask;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(500, &error);
    for (const aitrain::TaskRecord& task : tasks) {
        if (task.id == sourceTaskId) {
            sourceTask = task;
            break;
        }
    }
    if (sourceTask.id.isEmpty() || sourceTask.kind != aitrain::TaskKind::Train) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("只能复现历史训练任务。"));
        return;
    }

    const aitrain::ExperimentRunRecord sourceRun = repository_.experimentRunForTask(sourceTaskId, &error);
    if (sourceRun.id <= 0 || sourceRun.requestJson.trimmed().isEmpty()) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("该训练任务没有可复现的 request 记录。"));
        return;
    }

    QJsonParseError parseError;
    const QJsonDocument requestDoc = QJsonDocument::fromJson(sourceRun.requestJson.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !requestDoc.isObject()) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("原训练 request JSON 无法解析：%1").arg(parseError.errorString()));
        return;
    }

    aitrain::TrainingRequest request = aitrain::TrainingRequest::fromJson(requestDoc.object());
    const int snapshotId = sourceRun.datasetSnapshotId > 0
        ? sourceRun.datasetSnapshotId
        : request.parameters.value(QStringLiteral("datasetSnapshotId")).toInt();
    const aitrain::DatasetSnapshotRecord snapshot = repository_.datasetSnapshotById(snapshotId, &error);
    if (snapshot.id <= 0 || snapshot.manifestPath.isEmpty() || !QFileInfo::exists(snapshot.manifestPath)) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("原实验的数据快照 manifest 缺失，无法按同一快照复现。请重新创建快照或选择其他训练任务。"));
        return;
    }

    const QString newTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString runDir = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(newTaskId));
    QDir().mkpath(runDir);

    request.taskId = newTaskId;
    request.projectPath = currentProjectPath_;
    request.outputPath = runDir;
    if (request.pluginId.isEmpty()) {
        request.pluginId = sourceTask.pluginId;
    }
    if (request.taskType.isEmpty()) {
        request.taskType = sourceTask.taskType;
    }
    request.parameters.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
    request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
    request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
    if (!request.parameters.contains(QStringLiteral("seed"))) {
        request.parameters.insert(QStringLiteral("seed"), 42);
    }
    request.parameters.insert(QStringLiteral("reproducedFromTaskId"), sourceTaskId);
    request.parameters.insert(QStringLiteral("reproducedFromExperimentRunId"), sourceRun.id);
    request.parameters.insert(QStringLiteral("reproduceMode"), QStringLiteral("same_snapshot_same_params"));

    aitrain::TaskRecord record;
    record.id = newTaskId;
    record.projectName = currentProjectName_;
    record.pluginId = request.pluginId;
    record.taskType = request.taskType;
    record.kind = aitrain::TaskKind::Train;
    record.state = aitrain::TaskState::Queued;
    record.workDir = runDir;
    record.message = worker_.isRunning() ? uiText("复现实验已排队。") : uiText("复现实验等待 Worker 启动。");
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;
    if (!repository_.insertTask(record, &error)) {
        QMessageBox::critical(this, uiText("复现实验"), error);
        return;
    }
    recordExperimentRunForRequest(request, snapshot.datasetId, &error);

    PendingTrainingTask pending{newTaskId, request, false, snapshot.datasetId, QString()};
    if (worker_.isRunning()) {
        state_.training.pendingTrainingTasks.append(pending);
        workerPill_->setStatus(uiText("复现实验已排队"), StatusPill::Tone::Info);
        updateRecentTasks();
        return;
    }

    updateRecentTasks();
    startQueuedTraining(newTaskId, request);
}

void MainWindow::startQueuedTraining(const QString& taskId, const aitrain::TrainingRequest& request)
{
    metricsWidget_->clear();
    logEdit_->clear();
    progressBar_->setValue(0);
    state_.training.currentTaskId = taskId;

    QString error;
    if (!worker_.startTraining(workerExecutablePath(), request, &error)) {
        repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, nullptr);
        state_.training.currentTaskId.clear();
        updateRecentTasks();
        QMessageBox::critical(this, QStringLiteral("Worker"), error);
        startNextQueuedTask();
        return;
    }

    repository_.updateTaskState(taskId, aitrain::TaskState::Running, QStringLiteral("started"), &error);
    workerPill_->setStatus(uiText("训练运行中"), StatusPill::Tone::Info);
    appendLog(uiText("任务已启动：%1").arg(taskId));
    updateRecentTasks();
}

void MainWindow::startNextQueuedTask()
{
    if (worker_.isRunning() || state_.training.pendingTrainingTasks.isEmpty()) {
        return;
    }

    const PendingTrainingTask next = state_.training.pendingTrainingTasks.takeFirst();
    if (next.needsSnapshot) {
        startSnapshotForQueuedTraining(next);
        return;
    }

    QString error;
    recordExperimentRunForRequest(next.request, next.datasetId, &error);
    startQueuedTraining(next.taskId, next.request);
}

void MainWindow::startSnapshotForQueuedTraining(const PendingTrainingTask& pending)
{
    if (!repository_.isOpen()) {
        return;
    }

    if (pending.datasetId <= 0 || pending.request.datasetPath.isEmpty()) {
        QString error;
        repository_.updateTaskState(pending.taskId, aitrain::TaskState::Failed, uiText("无法为训练创建数据快照：数据集记录缺失。"), &error);
        updateRecentTasks();
        return;
    }

    const QString snapshotTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(snapshotTaskId));
    const QString createdTaskId = createRepositoryTask(
        aitrain::TaskKind::Snapshot,
        QStringLiteral("dataset_snapshot"),
        QStringLiteral("com.aitrain.plugins.dataset_interop"),
        outputPath,
        uiText("为训练自动创建数据快照。"),
        snapshotTaskId);
    if (createdTaskId.isEmpty()) {
        return;
    }

    state_.training.hasActiveSnapshotTrainingTask = true;
    state_.training.activeSnapshotTrainingTask = pending;

    QJsonObject options;
    options.insert(QStringLiteral("maxFiles"), 20000);

    QString error;
    if (!worker_.requestDatasetSnapshot(workerExecutablePath(), pending.request.datasetPath, outputPath, pending.datasetFormat, options, &error, createdTaskId)) {
        repository_.updateTaskState(createdTaskId, aitrain::TaskState::Failed, error, nullptr);
        repository_.updateTaskState(pending.taskId, aitrain::TaskState::Failed, uiText("自动数据快照失败：%1").arg(error), nullptr);
        state_.training.hasActiveSnapshotTrainingTask = false;
        state_.training.activeSnapshotTrainingTask = PendingTrainingTask();
        state_.training.currentTaskId.clear();
        updateRecentTasks();
        QMessageBox::critical(this, uiText("数据快照"), error);
        return;
    }

    workerPill_->setStatus(uiText("自动数据快照创建中"), StatusPill::Tone::Info);
    appendLog(uiText("训练任务 %1 正在等待自动数据快照 %2。").arg(pending.taskId.left(8), createdTaskId.left(8)));
    updateRecentTasks();
}

void MainWindow::runLocalPipelinePlanFromCurrentDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("本地流水线"), uiText("Worker 正在执行任务，稍后再执行流水线。"));
        return;
    }

    const QStringList templateLabels = {
        uiText("训练->评估->导出->注册->报告"),
        uiText("导出->推理->基准->报告")
    };
    bool ok = false;
    const QString selectedTemplate = QInputDialog::getItem(
        this,
        uiText("本地流水线"),
        uiText("选择流水线模板"),
        templateLabels,
        0,
        false,
        &ok);
    if (!ok || selectedTemplate.isEmpty()) {
        return;
    }
    const QString templateId = selectedTemplate == templateLabels.at(1)
        ? QStringLiteral("export-infer-benchmark-report")
        : QStringLiteral("train-evaluate-export-register");

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Pipeline,
            QStringLiteral("local_pipeline"),
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("本地流水线执行中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    int datasetId = 0;
    if (repository_.isOpen()) {
        QString repositoryError;
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &repositoryError);
        datasetId = dataset.id;
    }

    QJsonObject options;
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    options.insert(QStringLiteral("datasetId"), datasetId);
    options.insert(QStringLiteral("datasetPath"), datasetPath);
    options.insert(QStringLiteral("datasetFormat"), currentDatasetFormat());
    options.insert(QStringLiteral("taskType"), currentTaskType());
    options.insert(QStringLiteral("trainingBackend"), trainingBackendCombo_ ? trainingBackendCombo_->currentData().toString() : QString());
    options.insert(QStringLiteral("modelPreset"), modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString());
    options.insert(QStringLiteral("epochs"), epochsEdit_ ? epochsEdit_->text().toInt() : 1);
    options.insert(QStringLiteral("batchSize"), batchEdit_ ? batchEdit_->text().toInt() : 1);
    options.insert(QStringLiteral("imageSize"), imageSizeEdit_ ? imageSizeEdit_->text().toInt() : 640);
    options.insert(QStringLiteral("exportFormat"), QStringLiteral("onnx"));
    options.insert(QStringLiteral("sourceTaskId"), selectedTaskId());
    options.insert(QStringLiteral("modelPath"), selectedArtifactPath());
    options.insert(QStringLiteral("sampleImagePath"), inferenceImageEdit_ ? QDir::fromNativeSeparators(inferenceImageEdit_->text().trimmed()) : QString());

    QString error;
    if (!worker_.requestLocalPipeline(workerExecutablePath(), outputPath, templateId, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            state_.training.currentTaskId.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("本地流水线"), error);
        return;
    }
    workerPill_->setStatus(uiText("本地流水线执行中"), StatusPill::Tone::Info);
}
