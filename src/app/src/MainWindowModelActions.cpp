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
#include <QComboBox>
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
#include <QSet>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QTime>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

using namespace aitrain_app;

namespace {
QString yoloArgObjectName(const QString& key)
{
    return QStringLiteral("YoloTrainArg_%1").arg(key);
}

QString yoloTrainArgText(const QWidget* root, const QString& key)
{
    if (!root) {
        return {};
    }
    if (const auto* edit = root->findChild<QLineEdit*>(yoloArgObjectName(key))) {
        return edit->text().trimmed();
    }
    if (const auto* combo = root->findChild<QComboBox*>(yoloArgObjectName(key))) {
        const QString value = combo->currentData().toString().trimmed();
        return value.isEmpty() ? combo->currentText().trimmed() : value;
    }
    return {};
}

bool isIntegerLike(const QString& text)
{
    bool ok = false;
    text.toInt(&ok);
    return ok;
}

bool isNumberLike(const QString& text)
{
    bool ok = false;
    text.toDouble(&ok);
    return ok;
}

QJsonValue yoloTrainArgJsonValue(const QString& key, const QString& text)
{
    const QString normalized = text.trimmed();
    if (normalized.isEmpty() || normalized == QStringLiteral("默认")) {
        return QJsonValue();
    }
    const QSet<QString> boolKeys = {
        QStringLiteral("cos_lr"), QStringLiteral("amp"), QStringLiteral("deterministic"),
        QStringLiteral("resume"), QStringLiteral("rect"), QStringLiteral("single_cls"),
        QStringLiteral("val"), QStringLiteral("plots"), QStringLiteral("overlap_mask")
    };
    if (boolKeys.contains(key)) {
        const QString lower = normalized.toLower();
        if (lower == QStringLiteral("true") || lower == QStringLiteral("1") || lower == QStringLiteral("yes")) {
            return true;
        }
        if (lower == QStringLiteral("false") || lower == QStringLiteral("0") || lower == QStringLiteral("no")) {
            return false;
        }
    }
    if (key == QStringLiteral("classes")) {
        QJsonArray values;
        for (const QString& part : normalized.split(QLatin1Char(','), QString::SkipEmptyParts)) {
            bool ok = false;
            const int value = part.trimmed().toInt(&ok);
            if (ok) {
                values.append(value);
            }
        }
        return values;
    }
    if (key == QStringLiteral("freeze") && normalized.contains(QLatin1Char(','))) {
        QJsonArray values;
        for (const QString& part : normalized.split(QLatin1Char(','), QString::SkipEmptyParts)) {
            bool ok = false;
            const int value = part.trimmed().toInt(&ok);
            if (ok) {
                values.append(value);
            }
        }
        return values;
    }
    const QSet<QString> intKeys = {
        QStringLiteral("workers"), QStringLiteral("patience"), QStringLiteral("save_period"),
        QStringLiteral("freeze"), QStringLiteral("nbs"), QStringLiteral("max_det"),
        QStringLiteral("close_mosaic"), QStringLiteral("mask_ratio")
    };
    if (intKeys.contains(key) && isIntegerLike(normalized)) {
        return normalized.toInt();
    }
    const QSet<QString> numericKeys = {
        QStringLiteral("lr0"), QStringLiteral("lrf"), QStringLiteral("momentum"),
        QStringLiteral("weight_decay"), QStringLiteral("warmup_epochs"), QStringLiteral("warmup_momentum"),
        QStringLiteral("warmup_bias_lr"), QStringLiteral("fraction"), QStringLiteral("multi_scale"),
        QStringLiteral("box"), QStringLiteral("cls"), QStringLiteral("dfl"), QStringLiteral("hsv_h"),
        QStringLiteral("hsv_s"), QStringLiteral("hsv_v"), QStringLiteral("degrees"), QStringLiteral("translate"),
        QStringLiteral("scale"), QStringLiteral("shear"), QStringLiteral("perspective"), QStringLiteral("flipud"),
        QStringLiteral("fliplr"), QStringLiteral("mosaic"), QStringLiteral("mixup"), QStringLiteral("cutmix"),
        QStringLiteral("copy_paste")
    };
    if (numericKeys.contains(key) && isNumberLike(normalized)) {
        return normalized.toDouble();
    }
    if (normalized == QStringLiteral("true")) {
        return true;
    }
    if (normalized == QStringLiteral("false")) {
        return false;
    }
    return normalized;
}

QJsonObject yoloTrainArgsFromUi(const QWidget* root)
{
    const QStringList keys = {
        QStringLiteral("device"), QStringLiteral("workers"), QStringLiteral("patience"), QStringLiteral("optimizer"),
        QStringLiteral("lr0"), QStringLiteral("lrf"), QStringLiteral("momentum"), QStringLiteral("weight_decay"),
        QStringLiteral("warmup_epochs"), QStringLiteral("cos_lr"), QStringLiteral("amp"), QStringLiteral("deterministic"),
        QStringLiteral("cache"), QStringLiteral("pretrained"), QStringLiteral("resume"), QStringLiteral("save_period"),
        QStringLiteral("fraction"), QStringLiteral("rect"), QStringLiteral("multi_scale"), QStringLiteral("single_cls"),
        QStringLiteral("classes"), QStringLiteral("freeze"), QStringLiteral("box"), QStringLiteral("cls"),
        QStringLiteral("dfl"), QStringLiteral("nbs"), QStringLiteral("val"), QStringLiteral("plots"),
        QStringLiteral("max_det"), QStringLiteral("hsv_h"), QStringLiteral("hsv_s"), QStringLiteral("hsv_v"),
        QStringLiteral("degrees"), QStringLiteral("translate"), QStringLiteral("scale"), QStringLiteral("shear"),
        QStringLiteral("perspective"), QStringLiteral("flipud"), QStringLiteral("fliplr"), QStringLiteral("mosaic"),
        QStringLiteral("mixup"), QStringLiteral("cutmix"), QStringLiteral("copy_paste"), QStringLiteral("copy_paste_mode"),
        QStringLiteral("close_mosaic"), QStringLiteral("overlap_mask"), QStringLiteral("mask_ratio")
    };
    QJsonObject args;
    for (const QString& key : keys) {
        const QString text = yoloTrainArgText(root, key);
        const QJsonValue value = yoloTrainArgJsonValue(key, text);
        if (!value.isUndefined() && !value.isNull()) {
            args.insert(key, value);
        }
    }
    return args;
}
} // namespace

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
        : QStringLiteral("onnx");
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
            state_.training.currentTaskId.clear();
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
            state_.training.currentTaskId.clear();
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
    bool datasetReady = state_.dataset.currentValid && state_.dataset.currentPath == datasetPath && state_.dataset.currentFormat == datasetFormat;
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
    const QString yoloSeedText = yoloTrainArgText(this, QStringLiteral("seed"));
    parameters.insert(QStringLiteral("seed"), yoloSeedText.isEmpty() ? 42 : yoloSeedText.toInt());
    parameters.insert(QStringLiteral("resumeCheckpointPath"), QDir::fromNativeSeparators(resumeCheckpointEdit_->text().trimmed()));
    parameters.insert(QStringLiteral("horizontalFlip"), horizontalFlipCheck_ && horizontalFlipCheck_->isChecked());
    parameters.insert(QStringLiteral("colorJitter"), colorJitterCheck_ && colorJitterCheck_->isChecked());
    const QString trainingBackend = trainingBackendCombo_
        ? trainingBackendCombo_->currentData().toString().trimmed()
        : defaultBackendForTask(currentTaskType());
    const QString backendForRequest = trainingBackend.isEmpty() ? defaultBackendForTask(currentTaskType()) : trainingBackend;
    const QString modelPreset = modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString();
    QString latestSnapshotManifest;
    if (repository_.isOpen()) {
        QString snapshotError;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &snapshotError);
        const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(dataset.id, &snapshotError);
        latestSnapshotManifest = snapshot.manifestPath;
    }
    const QJsonObject preflight = trainingPreflightReport(
        datasetPath,
        datasetFormat,
        datasetReady,
        latestSnapshotManifest,
        currentTaskType(),
        backendForRequest,
        modelPreset,
        epochsEdit_ ? epochsEdit_->text().toInt() : 0,
        batchEdit_ ? batchEdit_->text().toInt() : 0,
        imageSizeEdit_ ? imageSizeEdit_->text().toInt() : 0);
    if (!preflight.value(QStringLiteral("canStart")).toBool()) {
        QStringList blockers;
        const QJsonArray blockerArray = preflight.value(QStringLiteral("blockers")).toArray();
        for (const QJsonValue& value : blockerArray) {
            blockers.append(value.toString());
        }
        QMessageBox::warning(
            this,
            uiText("训练"),
            QStringLiteral("Training preflight blocked:\n%1").arg(blockers.join(QStringLiteral("\n"))));
        return;
    }
    parameters.insert(QStringLiteral("trainingBackend"), backendForRequest);
    if (backendForRequest.startsWith(QStringLiteral("ultralytics_yolo"))) {
        QJsonObject yoloArgs = yoloTrainArgsFromUi(this);
        if ((horizontalFlipCheck_ && horizontalFlipCheck_->isChecked()) && !yoloArgs.contains(QStringLiteral("fliplr"))) {
            yoloArgs.insert(QStringLiteral("fliplr"), 0.5);
        }
        if (!yoloArgs.isEmpty()) {
            parameters.insert(QStringLiteral("ultralyticsTrainArgs"), yoloArgs);
        }
    }
    if (backendForRequest == QStringLiteral("paddleocr_det_official")
        || backendForRequest == QStringLiteral("paddleocr_rec_official")
        || backendForRequest == QStringLiteral("paddleocr_ppocrv4_rec")) {
        parameters.insert(QStringLiteral("runOfficial"), true);
        parameters.insert(QStringLiteral("prepareOnly"), false);
    }
    parameters.insert(QStringLiteral("trainingPreflight"), preflight);
    parameters.insert(QStringLiteral("trainingTemplate"), QStringLiteral("manual_worker_training_v1"));
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
        state_.training.pendingTrainingTasks.append(pending);
        workerPill_->setStatus(uiText("任务已排队"), StatusPill::Tone::Info);
        appendLog(uiText("任务已加入队列：%1").arg(taskId));
        updateRecentTasks();
        startNextQueuedTask();
        return;
    }

    QTimer::singleShot(0, this, [this, taskId, request]() {
        startQueuedTraining(taskId, request);
    });
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
            state_.training.currentTaskId.clear();
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
            state_.training.currentTaskId.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("部署基准"), error);
        return;
    }
    workerPill_->setStatus(uiText("部署基准运行中"), StatusPill::Tone::Info);
}

void MainWindow::useSelectedComparisonForInference()
{
    const QString modelPath = selectedComparisonModelPath();
    if (modelPath.isEmpty()) {
        QMessageBox::information(this, uiText("模型对比"), uiText("请先选择一个对比候选。"));
        return;
    }
    if (inferenceCheckpointEdit_) {
        inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(modelPath));
    }
    showPage(InferencePage, uiText("推理验证"));
}

void MainWindow::useSelectedComparisonForExport()
{
    const QString modelPath = selectedComparisonModelPath();
    if (modelPath.isEmpty()) {
        QMessageBox::information(this, uiText("模型对比"), uiText("请先选择一个对比候选。"));
        return;
    }
    if (conversionCheckpointEdit_) {
        conversionCheckpointEdit_->setText(QDir::toNativeSeparators(modelPath));
    }
    showPage(ConversionPage, uiText("模型导出"));
}

void MainWindow::openSelectedComparisonReport()
{
    const QString reportPath = selectedComparisonReportPath();
    if (reportPath.isEmpty() || !QFileInfo::exists(reportPath)) {
        QMessageBox::information(this, uiText("模型对比"), uiText("选中候选没有可打开的评估报告。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(reportPath));
}
