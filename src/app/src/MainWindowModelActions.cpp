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

QString smpArgObjectName(const QString& key)
{
    return QStringLiteral("SmpTrainArg_%1").arg(key);
}

QString yoloExportArgObjectName(const QString& prefix, const QString& key)
{
    return QStringLiteral("%1_%2").arg(prefix, key);
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

QString smpTrainArgText(const QWidget* root, const QString& key)
{
    if (!root) {
        return {};
    }
    if (const auto* edit = root->findChild<QLineEdit*>(smpArgObjectName(key))) {
        return edit->text().trimmed();
    }
    if (const auto* combo = root->findChild<QComboBox*>(smpArgObjectName(key))) {
        const QString value = combo->currentData().toString().trimmed();
        return value.isEmpty() ? combo->currentText().trimmed() : value;
    }
    return {};
}

QString yoloExportArgText(const QWidget* root, const QString& prefix, const QString& key)
{
    if (!root) {
        return {};
    }
    const QString objectName = yoloExportArgObjectName(prefix, key);
    if (const auto* edit = root->findChild<QLineEdit*>(objectName)) {
        return edit->text().trimmed();
    }
    if (const auto* combo = root->findChild<QComboBox*>(objectName)) {
        const QString value = combo->currentData().toString().trimmed();
        return value.isEmpty() ? combo->currentText().trimmed() : value;
    }
    if (const auto* check = root->findChild<QCheckBox*>(objectName)) {
        return check->isChecked() ? QStringLiteral("true") : QStringLiteral("false");
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

QJsonValue smpTrainArgJsonValue(const QString& key, const QString& text)
{
    const QString normalized = text.trimmed();
    if (normalized.isEmpty()) {
        return QJsonValue();
    }
    const QSet<QString> intKeys = {
        QStringLiteral("seed"),
        QStringLiteral("workers"),
        QStringLiteral("ignoreIndex")
    };
    if (intKeys.contains(key) && isIntegerLike(normalized)) {
        return normalized.toInt();
    }
    if (key == QStringLiteral("learningRate") && isNumberLike(normalized)) {
        return normalized.toDouble();
    }
    return normalized;
}

QJsonObject smpTrainArgsFromUi(const QWidget* root)
{
    const QStringList keys = {
        QStringLiteral("seed"),
        QStringLiteral("device"),
        QStringLiteral("workers"),
        QStringLiteral("learningRate"),
        QStringLiteral("optimizer"),
        QStringLiteral("loss"),
        QStringLiteral("encoderWeights"),
        QStringLiteral("ignoreIndex")
    };
    QJsonObject args;
    for (const QString& key : keys) {
        const QJsonValue value = smpTrainArgJsonValue(key, smpTrainArgText(root, key));
        if (!value.isUndefined() && !value.isNull()) {
            args.insert(key, value);
        }
    }
    return args;
}

QJsonObject yoloTrainingExportArgsFromUi(const QWidget* root, int imageSize)
{
    const bool dynamic = yoloExportArgText(root, QStringLiteral("YoloTrainExportArg"), QStringLiteral("dynamic")) == QStringLiteral("true");
    const bool half = yoloExportArgText(root, QStringLiteral("YoloTrainExportArg"), QStringLiteral("half")) == QStringLiteral("true");
    const bool int8 = yoloExportArgText(root, QStringLiteral("YoloTrainExportArg"), QStringLiteral("int8")) == QStringLiteral("true");
    QString device = yoloTrainArgText(root, QStringLiteral("device"));
    if (device.isEmpty() || device == QStringLiteral("默认")) {
        device = QStringLiteral("cpu");
    }

    QJsonObject args;
    args.insert(QStringLiteral("format"), int8 ? QStringLiteral("tensorrt") : QStringLiteral("onnx"));
    args.insert(QStringLiteral("dynamic"), dynamic);
    args.insert(QStringLiteral("half"), half);
    args.insert(QStringLiteral("int8"), int8);
    args.insert(QStringLiteral("imgsz"), imageSize);
    args.insert(QStringLiteral("batch"), 1);
    args.insert(QStringLiteral("device"), device);
    const QString endToEnd = yoloExportArgText(root, QStringLiteral("YoloTrainExportArg"), QStringLiteral("end2end"));
    if (!endToEnd.isEmpty()) {
        args.insert(QStringLiteral("end2end"), endToEnd);
    }
    return args;
}

QJsonObject yoloModelExportArgsFromUi(const QWidget* root, const QString& format)
{
    QJsonObject args;
    args.insert(QStringLiteral("format"), format);
    const auto insertBoolIfTrue = [&](const QString& key) {
        if (yoloExportArgText(root, QStringLiteral("YoloModelExportArg"), key) == QStringLiteral("true")) {
            args.insert(key, true);
        }
    };
    insertBoolIfTrue(QStringLiteral("dynamic"));
    insertBoolIfTrue(QStringLiteral("half"));
    insertBoolIfTrue(QStringLiteral("int8"));
    const QString endToEnd = yoloExportArgText(root, QStringLiteral("YoloModelExportArg"), QStringLiteral("end2end"));
    if (!endToEnd.isEmpty()) {
        args.insert(QStringLiteral("end2end"), endToEnd);
    }
    for (const QString& key : {QStringLiteral("imgsz"), QStringLiteral("batch")}) {
        const QString text = yoloExportArgText(root, QStringLiteral("YoloModelExportArg"), key);
        if (!text.isEmpty() && isIntegerLike(text)) {
            args.insert(key, text.toInt());
        }
    }
    const QString device = yoloExportArgText(root, QStringLiteral("YoloModelExportArg"), QStringLiteral("device"));
    if (!device.isEmpty()) {
        args.insert(QStringLiteral("device"), device);
    }
    const QString data = QDir::fromNativeSeparators(yoloExportArgText(root, QStringLiteral("YoloModelExportArg"), QStringLiteral("data")));
    if (!data.isEmpty()) {
        args.insert(QStringLiteral("data"), data);
    }
    return args;
}

bool jsonLooksYolo26(const QJsonObject& object)
{
    const QStringList keys = {
        QStringLiteral("modelSeries"),
        QStringLiteral("model"),
        QStringLiteral("modelName"),
        QStringLiteral("sourceCheckpoint"),
        QStringLiteral("sourceOnnx")
    };
    for (const QString& key : keys) {
        const QString value = object.value(key).toString().trimmed().toLower();
        if (value.contains(QStringLiteral("yolo26"))) {
            return true;
        }
    }
    const QJsonObject trainingReport = object.value(QStringLiteral("trainingReport")).toObject();
    if (!trainingReport.isEmpty() && jsonLooksYolo26(trainingReport)) {
        return true;
    }
    const QJsonObject ncnn = object.value(QStringLiteral("ncnn")).toObject();
    return !ncnn.isEmpty() && jsonLooksYolo26(ncnn);
}

QJsonObject readJsonObjectFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    return document.isObject() ? document.object() : QJsonObject();
}

bool modelExportSourceLooksYolo26(const QString& path)
{
    const QString normalized = QDir::fromNativeSeparators(path.trimmed());
    if (normalized.isEmpty()) {
        return false;
    }
    if (normalized.toLower().contains(QStringLiteral("yolo26"))) {
        return true;
    }

    const QFileInfo info(normalized);
    const QString suffix = info.suffix().toLower();
    if ((suffix == QStringLiteral("json") || suffix == QStringLiteral("aitrain"))
        && jsonLooksYolo26(readJsonObjectFile(normalized))) {
        return true;
    }

    const QString sidecarPath = info.dir().filePath(info.completeBaseName() + QStringLiteral(".aitrain-export.json"));
    if (QFileInfo::exists(sidecarPath) && jsonLooksYolo26(readJsonObjectFile(sidecarPath))) {
        return true;
    }

    const QString siblingReport = info.dir().filePath(QStringLiteral("ultralytics_training_report.json"));
    if (QFileInfo::exists(siblingReport) && jsonLooksYolo26(readJsonObjectFile(siblingReport))) {
        return true;
    }

    const QString parentReport = QFileInfo(info.dir().absolutePath()).dir().filePath(QStringLiteral("ultralytics_training_report.json"));
    return QFileInfo::exists(parentReport) && jsonLooksYolo26(readJsonObjectFile(parentReport));
}
} // namespace

void MainWindow::refreshModelExportFormatOptions()
{
    if (!conversionFormatCombo_) {
        return;
    }

    const QString inputPath = conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString();
    const bool yolo26 = modelExportSourceLooksYolo26(inputPath);
    const QString currentFormat = conversionFormatCombo_->currentData().toString();
    QSignalBlocker blocker(conversionFormatCombo_);

    const int ncnnIndex = conversionFormatCombo_->findData(QStringLiteral("ncnn"));
    if (yolo26 && ncnnIndex >= 0) {
        conversionFormatCombo_->removeItem(ncnnIndex);
    } else if (!yolo26 && ncnnIndex < 0) {
        const int insertAt = qMin(1, conversionFormatCombo_->count());
        conversionFormatCombo_->insertItem(insertAt, exportComboLabel(QStringLiteral("ncnn")), QStringLiteral("ncnn"));
    }

    if (yolo26 && currentFormat == QStringLiteral("ncnn")) {
        setComboCurrentData(conversionFormatCombo_, QStringLiteral("onnx"));
        if (conversionOutputEdit_) {
            const QString outputSuffix = QFileInfo(conversionOutputEdit_->text().trimmed()).suffix().toLower();
            if (outputSuffix == QStringLiteral("param") || outputSuffix == QStringLiteral("bin")) {
                conversionOutputEdit_->clear();
            }
        }
    } else if (!currentFormat.isEmpty()) {
        setComboCurrentData(conversionFormatCombo_, currentFormat);
    }
    conversionFormatCombo_->setToolTip(yolo26
        ? uiText("YOLO26 不支持导出为 NCNN；请使用 ONNX 或 TensorRT。")
        : QString());
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
        : QStringLiteral("onnx");
    if (format == QStringLiteral("ncnn") && modelExportSourceLooksYolo26(checkpointPath)) {
        QMessageBox::warning(
            this,
            uiText("模型导出"),
            uiText("YOLO26 不支持导出为 NCNN；请使用 ONNX 或 TensorRT。"));
        refreshModelExportFormatOptions();
        return;
    }
    QString outputPath = QDir::fromNativeSeparators(conversionOutputEdit_ ? conversionOutputEdit_->text().trimmed() : QString());
    if (outputPath.isEmpty()) {
        const QString outputDir = !currentProjectPath_.isEmpty()
            ? QDir(currentProjectPath_).filePath(QStringLiteral("models/exported"))
            : QFileInfo(checkpointPath).absoluteDir().absolutePath();
        QDir().mkpath(outputDir);
        outputPath = QDir(outputDir).filePath(defaultExportFileName(format));
    }
    const QJsonObject yoloExportArgs = yoloModelExportArgsFromUi(this, format);
    if (QFileInfo(checkpointPath).suffix().compare(QStringLiteral("pt"), Qt::CaseInsensitive) == 0
        && format.startsWith(QStringLiteral("tensorrt"))
        && yoloExportArgs.value(QStringLiteral("int8")).toBool()
        && yoloExportArgs.value(QStringLiteral("data")).toString().trimmed().isEmpty()) {
        QMessageBox::warning(
            this,
            uiText("模型导出"),
            uiText("TensorRT INT8 官方导出需要 calibration data.yaml。请在“官方参数”中选择本次训练使用的 YOLO data.yaml。"));
        return;
    }
    QJsonObject exportOptions;
    exportOptions.insert(QStringLiteral("ultralyticsExportArgs"), yoloExportArgs);

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
    if (!worker_.requestModelExport(workerExecutablePath(), checkpointPath, outputPath, format, exportOptions, &error, taskId)) {
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
    parameters.insert(QStringLiteral("datasetFormat"), datasetFormat);
    const QString trainingBackend = trainingBackendCombo_
        ? trainingBackendCombo_->currentData().toString().trimmed()
        : defaultBackendForTask(currentTaskType());
    const QString backendForRequest = trainingBackend.isEmpty() ? defaultBackendForTask(currentTaskType()) : trainingBackend;
    const QString modelPreset = modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString();
    const QString seedText = backendForRequest == QStringLiteral("smp_semantic_segmentation")
        ? smpTrainArgText(this, QStringLiteral("seed"))
        : yoloTrainArgText(this, QStringLiteral("seed"));
    bool seedOk = false;
    const int seed = seedText.toInt(&seedOk);
    parameters.insert(QStringLiteral("seed"), seedOk ? seed : 42);
    parameters.insert(QStringLiteral("resumeCheckpointPath"), QDir::fromNativeSeparators(resumeCheckpointEdit_->text().trimmed()));
    parameters.insert(QStringLiteral("horizontalFlip"), horizontalFlipCheck_ && horizontalFlipCheck_->isChecked());
    parameters.insert(QStringLiteral("colorJitter"), colorJitterCheck_ && colorJitterCheck_->isChecked());
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
    const QString yolo26PythonExecutable = QDir::fromNativeSeparators(
        preflight.value(QStringLiteral("yolo26PythonExecutable")).toString().trimmed());
    if (!yolo26PythonExecutable.isEmpty()) {
        parameters.insert(QStringLiteral("pythonExecutable"), yolo26PythonExecutable);
    }
    if (backendForRequest.startsWith(QStringLiteral("ultralytics_yolo"))) {
        QJsonObject yoloArgs = yoloTrainArgsFromUi(this);
        if ((horizontalFlipCheck_ && horizontalFlipCheck_->isChecked()) && !yoloArgs.contains(QStringLiteral("fliplr"))) {
            yoloArgs.insert(QStringLiteral("fliplr"), 0.5);
        }
        if (!yoloArgs.isEmpty()) {
            parameters.insert(QStringLiteral("ultralyticsTrainArgs"), yoloArgs);
        }
        parameters.insert(QStringLiteral("ultralyticsExportArgs"), yoloTrainingExportArgsFromUi(
            this,
            imageSizeEdit_ ? imageSizeEdit_->text().toInt() : 640));
    }
    if (backendForRequest == QStringLiteral("smp_semantic_segmentation")) {
        const QJsonObject smpArgs = smpTrainArgsFromUi(this);
        for (auto it = smpArgs.constBegin(); it != smpArgs.constEnd(); ++it) {
            parameters.insert(it.key(), it.value());
        }
        parameters.insert(QStringLiteral("modelFamily"), QStringLiteral("semantic_segmentation"));
        parameters.insert(QStringLiteral("taskType"), QStringLiteral("semantic_segmentation"));
        parameters.insert(QStringLiteral("exportOnnx"), true);
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
    showDeploymentTab(1);
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
    showDeploymentTab(0);
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
