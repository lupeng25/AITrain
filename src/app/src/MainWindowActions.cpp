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

void MainWindow::openEvaluationReportsPage()
{
    showPage(EvaluationReportsPage, uiText("评估报告"));
}

void MainWindow::createProject()
{
    currentProjectName_ = projectNameEdit_->text().trimmed();
    currentProjectPath_ = QDir::fromNativeSeparators(projectRootEdit_->text().trimmed());
    if (currentProjectName_.isEmpty() || currentProjectPath_.isEmpty()) {
        QMessageBox::warning(this, uiText("项目"), uiText("项目名称和目录不能为空。"));
        return;
    }

    ensureProjectSubdirs(currentProjectPath_);
    QString error;
    if (!repository_.open(QDir(currentProjectPath_).filePath(QStringLiteral("project.sqlite")), &error)
        || !repository_.upsertProject(currentProjectName_, currentProjectPath_, &error)) {
        QMessageBox::critical(this, uiText("项目"), error);
        return;
    }
    repository_.markInterruptedTasksFailed(uiText("上次会话结束时任务未正常完成，已标记为失败。"), &error);

    projectLabel_->setText(uiText("当前项目：%1").arg(currentProjectPath_));
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

void MainWindow::browseDataset()
{
    const QString directory = QFileDialog::getExistingDirectory(this, uiText("选择数据集目录"));
    if (!directory.isEmpty()) {
        datasetPathEdit_->setText(QDir::toNativeSeparators(directory));
        const QString detectedFormat = detectDatasetFormatFromPath(directory);
        if (!detectedFormat.isEmpty() && datasetFormatCombo_) {
            const int index = datasetFormatCombo_->findData(detectedFormat);
            if (index >= 0) {
                datasetFormatCombo_->setCurrentIndex(index);
            }
        }
        if (splitOutputEdit_ && currentProjectPath_.isEmpty()) {
            splitOutputEdit_->setText(QDir::toNativeSeparators(QDir(directory).absoluteFilePath(QStringLiteral("../normalized"))));
        }
        currentDatasetPath_ = directory;
        const QString selectedFormat = currentDatasetFormat();
        currentDatasetFormat_ = selectedFormat.isEmpty() ? detectedFormat : selectedFormat;
        currentDatasetValid_ = false;
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
    }
}

void MainWindow::validateDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据集校验"), uiText("Worker 正在执行任务，稍后再校验数据集。"));
        return;
    }

    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_->text());
    if (format.isEmpty() || path.isEmpty()) {
        validationSummaryLabel_->setText(uiText("请选择数据集目录和格式。"));
        return;
    }

    currentDatasetValid_ = false;
    currentDatasetPath_ = path;
    currentDatasetFormat_ = format;
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
    }
    validationSummaryLabel_->setText(uiText("正在通过 Worker 校验数据集。"));
    validationOutput_->setPlainText(uiText("等待校验结果。"));

    QJsonObject options;
    options.insert(QStringLiteral("maxIssues"), 200);
    options.insert(QStringLiteral("maxFiles"), 5000);
    options.insert(QStringLiteral("allowEmptyLabels"), false);
    options.insert(QStringLiteral("maxTextLength"), 25);

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Validate,
            QStringLiteral("dataset_validation"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集校验中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestDatasetValidation(workerExecutablePath(), path, format, options, &error, taskId, outputPath)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        validationSummaryLabel_->setText(uiText("无法启动数据集校验：%1").arg(error));
        QMessageBox::critical(this, uiText("数据集校验"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集校验中"), StatusPill::Tone::Info);
}

void MainWindow::splitDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("Worker 正在执行任务，稍后再划分数据集。"));
        return;
    }

    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_->text());
    if (path.isEmpty() || format.isEmpty()) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("请先选择数据集目录和格式。"));
        return;
    }
    if (format != QStringLiteral("yolo_detection") && format != QStringLiteral("yolo_txt")
        && format != QStringLiteral("yolo_segmentation")
        && format != QStringLiteral("paddleocr_det")
        && format != QStringLiteral("paddleocr_rec")) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("当前划分支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec 格式。"));
        return;
    }

    bool datasetReady = currentDatasetValid_ && currentDatasetPath_ == path && currentDatasetFormat_ == format;
    if (!datasetReady && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(path, &error);
        datasetReady = dataset.rootPath == path
            && dataset.format == format
            && dataset.validationStatus == QStringLiteral("valid");
    }
    if (!datasetReady) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("请先通过当前格式的数据集校验。"));
        return;
    }

    QString outputPath = QDir::fromNativeSeparators(splitOutputEdit_->text().trimmed());
    if (outputPath.isEmpty()) {
        const QString datasetName = QFileInfo(path).fileName();
        const QString basePath = currentProjectPath_.isEmpty()
            ? QDir(path).absoluteFilePath(QStringLiteral("../normalized"))
            : QDir(currentProjectPath_).filePath(QStringLiteral("datasets/normalized/%1").arg(datasetName));
        outputPath = QDir::cleanPath(basePath);
        splitOutputEdit_->setText(QDir::toNativeSeparators(outputPath));
    }

    QJsonObject options;
    options.insert(QStringLiteral("trainRatio"), splitTrainRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("valRatio"), splitValRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("testRatio"), splitTestRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("seed"), splitSeedEdit_->text().toInt());
    options.insert(QStringLiteral("maxIssues"), 200);
    options.insert(QStringLiteral("allowEmptyLabels"), false);

    QString taskId;
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Validate,
            QStringLiteral("dataset_split"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集划分中。"));
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestDatasetSplit(workerExecutablePath(), path, outputPath, format, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据集划分"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集划分中"), StatusPill::Tone::Info);
    statusBar()->showMessage(uiText("正在划分数据集"), 3000);
}

void MainWindow::curateDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据质量报告"), uiText("Worker 正在执行任务，稍后再生成数据质量报告。"));
        return;
    }
    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (path.isEmpty() || format.isEmpty()) {
        QMessageBox::warning(this, uiText("数据质量报告"), uiText("请先选择数据集目录和格式。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Curate,
            QStringLiteral("dataset_quality"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据质量报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("maxIssues"), 500);
    options.insert(QStringLiteral("maxProblemSamples"), 500);
    options.insert(QStringLiteral("maxFiles"), 20000);
    options.insert(QStringLiteral("duplicateHashLimit"), 20000);
    options.insert(QStringLiteral("distributionWarningThreshold"), 0.25);
    options.insert(QStringLiteral("exportXAnyLabelingFixList"), true);

    QString error;
    if (!worker_.requestDatasetCuration(workerExecutablePath(), path, outputPath, format, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据质量报告"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据质量报告生成中"), StatusPill::Tone::Info);
}

void MainWindow::openDatasetQualityFixList()
{
    if (latestQualityFixListPath_.isEmpty() || !QFileInfo::exists(latestQualityFixListPath_)) {
        QMessageBox::information(this, uiText("问题清单"), uiText("请先生成数据质量报告。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(latestQualityFixListPath_));
}

void MainWindow::launchXAnyLabelingForQualityFix()
{
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (datasetPath.isEmpty()) {
        QMessageBox::information(this, uiText("X-AnyLabeling 修复"), uiText("请先选择数据集目录。"));
        return;
    }
    if (!latestQualityFixListPath_.isEmpty()) {
        statusBar()->showMessage(uiText("问题清单：%1").arg(QDir::toNativeSeparators(latestQualityFixListPath_)), 6000);
    }
    const QString program = resolvedXAnyLabelingProgram();
    if (program.isEmpty()) {
        updateAnnotationToolStatus();
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或将 X-AnyLabeling.exe 放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(datasetPath));
    if (QProcess::startDetached(program, QStringList() << datasetPath)) {
        statusBar()->showMessage(uiText("已启动 X-AnyLabeling，请按问题清单修复样本。"), 5000);
    } else {
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(program)));
    }
}

void MainWindow::createDatasetSnapshot()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据集快照"), uiText("Worker 正在执行任务，稍后再创建数据集快照。"));
        return;
    }
    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (path.isEmpty() || format.isEmpty()) {
        QMessageBox::warning(this, uiText("数据集快照"), uiText("请先选择数据集目录和格式。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Snapshot,
            QStringLiteral("dataset_snapshot"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集快照创建中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("maxFiles"), 20000);

    QString error;
    if (!worker_.requestDatasetSnapshot(workerExecutablePath(), path, outputPath, format, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据集快照"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集快照创建中"), StatusPill::Tone::Info);
}

void MainWindow::startModelExport()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("模型导出"), uiText("Worker 正在执行任务，稍后再导出模型。"));
        return;
    }
    const QString checkpointPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
    if (checkpointPath.isEmpty()) {
        QMessageBox::warning(this, uiText("模型导出"), uiText("请选择模型输入。"));
        return;
    }
    const QString format = conversionFormatCombo_
        ? conversionFormatCombo_->currentData().toString()
        : QStringLiteral("tiny_detector_json");
    QString outputPath = QDir::fromNativeSeparators(conversionOutputEdit_ ? conversionOutputEdit_->text().trimmed() : QString());
    if (outputPath.isEmpty()) {
        const QString outputDir = !currentProjectPath_.isEmpty()
            ? QDir(currentProjectPath_).filePath(QStringLiteral("models/exported"))
            : QFileInfo(checkpointPath).absoluteDir().absolutePath();
        QDir().mkpath(outputDir);
        outputPath = QDir(outputDir).filePath(defaultExportFileName(format));
    }

    QString taskId;
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Export,
            QStringLiteral("model_export"),
            QStringLiteral("com.aitrain.plugins.yolo_native"),
            QFileInfo(outputPath).absolutePath(),
            uiText("模型导出中。"));
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestModelExport(workerExecutablePath(), checkpointPath, outputPath, format, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("模型导出"), error);
        return;
    }
    if (exportResultLabel_) {
        exportResultLabel_->setText(uiText("正在导出：%1").arg(QDir::toNativeSeparators(outputPath)));
    }
    workerPill_->setStatus(uiText("模型导出中"), StatusPill::Tone::Info);
}

void MainWindow::startInference()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("推理"), uiText("Worker 正在执行任务，稍后再推理。"));
        return;
    }
    const QString checkpointPath = QDir::fromNativeSeparators(inferenceCheckpointEdit_ ? inferenceCheckpointEdit_->text().trimmed() : QString());
    const QString imagePath = QDir::fromNativeSeparators(inferenceImageEdit_ ? inferenceImageEdit_->text().trimmed() : QString());
    QString outputPath = QDir::fromNativeSeparators(inferenceOutputEdit_ ? inferenceOutputEdit_->text().trimmed() : QString());
    if (checkpointPath.isEmpty() || imagePath.isEmpty()) {
        QMessageBox::warning(this, uiText("推理"), uiText("请选择模型文件和图片。"));
        return;
    }
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("inference"));
    }

    QString taskId;
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Infer,
            QStringLiteral("inference"),
            QStringLiteral("com.aitrain.plugins.yolo_native"),
            outputPath,
            uiText("推理中。"));
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestInference(workerExecutablePath(), checkpointPath, imagePath, outputPath, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("推理"), error);
        return;
    }
    if (inferenceResultLabel_) {
        inferenceResultLabel_->setText(uiText("正在推理：%1").arg(QDir::toNativeSeparators(imagePath)));
    }
    setInferenceOverlayText(inferenceOverlayLabel_, uiText("推理运行中\n等待 Worker 写入 overlay 产物。"));
    workerPill_->setStatus(uiText("推理中"), StatusPill::Tone::Info);
}

void MainWindow::startTraining()
{
    if (currentProjectPath_.isEmpty()) {
        createProject();
        if (currentProjectPath_.isEmpty()) {
            return;
        }
    }
    if (pluginCombo_->currentData().toString().isEmpty() || currentTaskType().isEmpty()) {
        QMessageBox::warning(this, uiText("训练"), uiText("请选择可用插件和任务类型。"));
        return;
    }
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_->text());
    const QString datasetFormat = currentDatasetFormat();
    if (datasetPath.isEmpty() || datasetFormat.isEmpty()) {
        QMessageBox::warning(this, uiText("训练"), uiText("请先选择并校验数据集。"));
        return;
    }
    auto* selectedPlugin = pluginManager_.pluginById(pluginCombo_->currentData().toString());
    if (!selectedPlugin || !selectedPlugin->datasetAdapter(datasetFormat)) {
        QMessageBox::warning(this, uiText("训练"), uiText("当前训练插件不支持所选数据集格式。"));
        return;
    }
    bool datasetReady = currentDatasetValid_ && currentDatasetPath_ == datasetPath && currentDatasetFormat_ == datasetFormat;
    if (!datasetReady && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &error);
        datasetReady = dataset.rootPath == datasetPath
            && dataset.format == datasetFormat
            && dataset.validationStatus == QStringLiteral("valid");
    }
    if (!datasetReady) {
        QMessageBox::warning(this, uiText("训练"), uiText("数据集未通过当前格式校验，不能启动训练。"));
        return;
    }

    const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString runDir = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
    QDir().mkpath(runDir);

    QJsonObject parameters;
    parameters.insert(QStringLiteral("epochs"), epochsEdit_->text().toInt());
    parameters.insert(QStringLiteral("batchSize"), batchEdit_->text().toInt());
    parameters.insert(QStringLiteral("imageSize"), imageSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("gridSize"), gridSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("seed"), 42);
    parameters.insert(QStringLiteral("resumeCheckpointPath"), QDir::fromNativeSeparators(resumeCheckpointEdit_->text().trimmed()));
    parameters.insert(QStringLiteral("horizontalFlip"), horizontalFlipCheck_ && horizontalFlipCheck_->isChecked());
    parameters.insert(QStringLiteral("colorJitter"), colorJitterCheck_ && colorJitterCheck_->isChecked());
    const QString trainingBackend = trainingBackendCombo_
        ? trainingBackendCombo_->currentData().toString().trimmed()
        : defaultBackendForTask(currentTaskType());
    const QString backendForRequest = trainingBackend.isEmpty() ? defaultBackendForTask(currentTaskType()) : trainingBackend;
    const QString modelPreset = modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString();
    parameters.insert(QStringLiteral("trainingBackend"), backendForRequest);
    if (!modelPreset.isEmpty()) {
        parameters.insert(QStringLiteral("modelPreset"), modelPreset);
        if (backendForRequest.startsWith(QStringLiteral("ultralytics_yolo"))) {
            parameters.insert(QStringLiteral("model"), modelPreset);
        }
    }
    aitrain::TrainingRequest request;
    request.taskId = taskId;
    request.projectPath = currentProjectPath_;
    request.pluginId = pluginCombo_->currentData().toString();
    request.taskType = currentTaskType();
    request.datasetPath = datasetPath;
    request.outputPath = runDir;
    request.parameters = parameters;

    int datasetId = 0;
    bool needsSnapshot = true;
    if (repository_.isOpen()) {
        QString snapshotError;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &snapshotError);
        datasetId = dataset.id;
        needsSnapshot = !attachLatestSnapshotToRequest(request, datasetId, &snapshotError);
    }

    aitrain::TaskRecord record;
    record.id = taskId;
    record.projectName = currentProjectName_;
    record.pluginId = request.pluginId;
    record.taskType = request.taskType;
    record.kind = aitrain::TaskKind::Train;
    record.state = aitrain::TaskState::Queued;
    record.workDir = runDir;
    record.message = needsSnapshot
        ? uiText("等待自动创建数据快照。")
        : (worker_.isRunning() ? uiText("等待当前任务完成。") : uiText("等待 Worker 启动。"));
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;
    QString error;
    if (!repository_.insertTask(record, &error)) {
        QMessageBox::critical(this, uiText("任务"), error);
        return;
    }

    if (!needsSnapshot) {
        recordExperimentRunForRequest(request, datasetId, &error);
    }

    PendingTrainingTask pending{taskId, request, needsSnapshot, datasetId, datasetFormat};
    if (worker_.isRunning() || needsSnapshot) {
        pendingTrainingTasks_.append(pending);
        workerPill_->setStatus(uiText("任务已排队"), StatusPill::Tone::Info);
        appendLog(uiText("任务已加入队列：%1").arg(taskId));
        updateRecentTasks();
        startNextQueuedTask();
        return;
    }

    startQueuedTraining(taskId, request);
}

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
            for (int index = 0; index < pendingTrainingTasks_.size(); ++index) {
                if (pendingTrainingTasks_.at(index).taskId == taskId) {
                    pendingTrainingTasks_.remove(index);
                    break;
                }
            }
            if (!repository_.updateTaskState(taskId, aitrain::TaskState::Canceled, uiText("用户取消排队任务。"), &error)) {
                QMessageBox::warning(this, uiText("任务队列"), error);
            }
            updateRecentTasks();
            return;
        }

        if ((task.state == aitrain::TaskState::Running || task.state == aitrain::TaskState::Paused) && taskId == currentTaskId_) {
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
        pendingTrainingTasks_.append(pending);
        workerPill_->setStatus(uiText("复现实验已排队"), StatusPill::Tone::Info);
        updateRecentTasks();
        return;
    }

    updateRecentTasks();
    startQueuedTraining(newTaskId, request);
}

void MainWindow::runEnvironmentCheck()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("环境自检"), uiText("Worker 正在执行任务，稍后再运行环境自检。"));
        return;
    }

    if (environmentTable_) {
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            auto* statusItem = new QTableWidgetItem(uiText("检测中"));
            environmentTable_->setItem(row, 1, statusItem);
            environmentTable_->setItem(row, 2, new QTableWidgetItem(uiText("等待 Worker 返回结果。")));
        }
    }
    updateEnvironmentSummary();

    QString error;
    if (!worker_.requestEnvironmentCheck(workerExecutablePath(), &error)) {
        QMessageBox::critical(this, uiText("环境自检"), error);
        return;
    }
    workerPill_->setStatus(uiText("环境自检中"), StatusPill::Tone::Info);
}

void MainWindow::refreshPlugins()
{
    pluginManager_.scan(pluginSearchPaths());
    if (pluginTable_) {
        pluginTable_->setRowCount(0);
        const QVector<aitrain::IModelPlugin*> plugins = pluginManager_.plugins();
        if (plugins.isEmpty()) {
            pluginTable_->setRowCount(1);
            pluginTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无插件")));
            for (int column = 1; column < pluginTable_->columnCount(); ++column) {
                pluginTable_->setItem(0, column, new QTableWidgetItem(uiText("重新扫描或检查 plugins/models 目录。")));
            }
        }
        for (auto* plugin : plugins) {
            const aitrain::PluginManifest manifest = plugin->manifest();
            const int row = pluginTable_->rowCount();
            pluginTable_->insertRow(row);
            pluginTable_->setItem(row, 0, new QTableWidgetItem(manifest.id));
            pluginTable_->setItem(row, 1, new QTableWidgetItem(manifest.name));
            pluginTable_->setItem(row, 2, new QTableWidgetItem(manifest.version));
            pluginTable_->setItem(row, 3, new QTableWidgetItem(compactListSummary(manifest.taskTypes, 4)));
            pluginTable_->setItem(row, 4, new QTableWidgetItem(compactListSummary(manifest.datasetFormats, 4)));
            pluginTable_->setItem(row, 5, new QTableWidgetItem(compactListSummary(manifest.exportFormats, 4)));
            pluginTable_->setItem(row, 6, new QTableWidgetItem(manifest.requiresGpu ? uiText("需要") : uiText("否")));
        }
    }
    loadPluginCombos();
    updateHeaderState();
    updatePluginSummary();
    updateDashboardSummary();
    refreshTrainingDefaults();
}

void MainWindow::startQueuedTraining(const QString& taskId, const aitrain::TrainingRequest& request)
{
    metricsWidget_->clear();
    logEdit_->clear();
    progressBar_->setValue(0);
    currentTaskId_ = taskId;

    QString error;
    if (!worker_.startTraining(workerExecutablePath(), request, &error)) {
        repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, nullptr);
        currentTaskId_.clear();
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
    if (worker_.isRunning() || pendingTrainingTasks_.isEmpty()) {
        return;
    }

    const PendingTrainingTask next = pendingTrainingTasks_.takeFirst();
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

    hasActiveSnapshotTrainingTask_ = true;
    activeSnapshotTrainingTask_ = pending;

    QJsonObject options;
    options.insert(QStringLiteral("maxFiles"), 20000);

    QString error;
    if (!worker_.requestDatasetSnapshot(workerExecutablePath(), pending.request.datasetPath, outputPath, pending.datasetFormat, options, &error, createdTaskId)) {
        repository_.updateTaskState(createdTaskId, aitrain::TaskState::Failed, error, nullptr);
        repository_.updateTaskState(pending.taskId, aitrain::TaskState::Failed, uiText("自动数据快照失败：%1").arg(error), nullptr);
        hasActiveSnapshotTrainingTask_ = false;
        activeSnapshotTrainingTask_ = PendingTrainingTask();
        currentTaskId_.clear();
        updateRecentTasks();
        QMessageBox::critical(this, uiText("数据快照"), error);
        return;
    }

    workerPill_->setStatus(uiText("自动数据快照创建中"), StatusPill::Tone::Info);
    appendLog(uiText("训练任务 %1 正在等待自动数据快照 %2。").arg(pending.taskId.left(8), createdTaskId.left(8)));
    updateRecentTasks();
}

void MainWindow::evaluateSelectedArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("模型评估"), uiText("Worker 正在执行任务，稍后再评估模型。"));
        return;
    }
    const QString modelPath = selectedArtifactPath();
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (modelPath.isEmpty() || datasetPath.isEmpty()) {
        QMessageBox::warning(this, uiText("模型评估"), uiText("请先选择模型产物，并在数据集页选择评估数据集。"));
        return;
    }

    const QString taskType = currentTaskType().isEmpty() ? QStringLiteral("detection") : currentTaskType();
    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Evaluate,
            taskType,
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("模型评估报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("scaffoldAcknowledged"), true);
    if (repository_.isOpen()) {
        QString snapshotError;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &snapshotError);
        const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(dataset.id, &snapshotError);
        if (snapshot.id > 0) {
            options.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
            options.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
            options.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
        }
    }
    QString error;
    if (!worker_.requestModelEvaluation(workerExecutablePath(), modelPath, datasetPath, outputPath, taskType, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("模型评估"), error);
        return;
    }
    workerPill_->setStatus(uiText("模型评估中"), StatusPill::Tone::Info);
}

void MainWindow::benchmarkSelectedArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("部署基准"), uiText("Worker 正在执行任务，稍后再运行部署基准。"));
        return;
    }
    const QString modelPath = selectedArtifactPath();
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, uiText("部署基准"), uiText("请先选择一个模型产物。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Benchmark,
            QStringLiteral("model_benchmark"),
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("部署基准报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("device"), QStringLiteral("cpu"));
    options.insert(QStringLiteral("batch"), 1);
    QString error;
    if (!worker_.requestModelBenchmark(workerExecutablePath(), modelPath, outputPath, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("部署基准"), error);
        return;
    }
    workerPill_->setStatus(uiText("部署基准运行中"), StatusPill::Tone::Info);
}

void MainWindow::generateDeliveryReportFromSelectedArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("交付报告"), uiText("Worker 正在执行任务，稍后再生成交付报告。"));
        return;
    }
    const QString modelPath = selectedArtifactPath();
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, uiText("交付报告"), uiText("请先选择一个模型或报告产物。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Report,
            QStringLiteral("delivery_report"),
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("训练交付报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject context;
    context.insert(QStringLiteral("projectName"), currentProjectName_);
    context.insert(QStringLiteral("projectPath"), currentProjectPath_);
    context.insert(QStringLiteral("modelPath"), modelPath);
    context.insert(QStringLiteral("datasetPath"), currentDatasetPath_);
    context.insert(QStringLiteral("datasetFormat"), currentDatasetFormat_);
    context.insert(QStringLiteral("sourceTaskId"), selectedTaskId());
    QString error;
    if (!worker_.requestDeliveryReport(workerExecutablePath(), outputPath, context, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("交付报告"), error);
        return;
    }
    workerPill_->setStatus(uiText("交付报告生成中"), StatusPill::Tone::Info);
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
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("本地流水线"), error);
        return;
    }
    workerPill_->setStatus(uiText("本地流水线执行中"), StatusPill::Tone::Info);
}
