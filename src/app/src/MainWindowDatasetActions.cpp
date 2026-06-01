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

namespace {
QString sampleTextField(const QJsonObject& sample, const QStringList& keys)
{
    for (const QString& key : keys) {
        const QJsonValue value = sample.value(key);
        if (value.isString() && !value.toString().trimmed().isEmpty()) {
            return value.toString().trimmed();
        }
        if (value.isDouble()) {
            return QString::number(value.toDouble(), 'g', 8);
        }
        if (value.isObject() || value.isArray()) {
            const QString compact = QString::fromUtf8(QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact));
            return compact.mid(1, qMax(0, compact.size() - 2));
        }
    }
    return QString();
}

QString samplePathField(const QJsonObject& sample, const QStringList& keys)
{
    return QDir::fromNativeSeparators(sampleTextField(sample, keys));
}

QJsonObject normalizedReviewSample(QJsonObject sample, const QString& source)
{
    if (sample.value(QStringLiteral("source")).toString().isEmpty()) {
        sample.insert(QStringLiteral("source"), source);
    }
    if (sample.value(QStringLiteral("reason")).toString().isEmpty()) {
        const QString reason = sampleTextField(sample, QStringList()
            << QStringLiteral("code")
            << QStringLiteral("errorType")
            << QStringLiteral("type")
            << QStringLiteral("category"));
        if (!reason.isEmpty()) {
            sample.insert(QStringLiteral("reason"), reason);
        }
    }
    return sample;
}

void appendReviewSamplesFromArray(QJsonArray* target, const QJsonArray& sourceArray, const QString& source)
{
    if (!target) {
        return;
    }
    for (const QJsonValue& value : sourceArray) {
        if (value.isObject()) {
            target->append(normalizedReviewSample(value.toObject(), source));
        }
    }
}

QJsonArray extractReviewSamples(const QJsonDocument& document)
{
    QJsonArray samples;
    if (document.isArray()) {
        appendReviewSamplesFromArray(&samples, document.array(), QStringLiteral("array"));
        return samples;
    }
    const QJsonObject root = document.object();
    const QList<QPair<QString, QString>> keys = {
        qMakePair(QStringLiteral("problemSamples"), QStringLiteral("problem_samples")),
        qMakePair(QStringLiteral("errorSamples"), QStringLiteral("error_samples")),
        qMakePair(QStringLiteral("lowConfidenceSamples"), QStringLiteral("low_confidence")),
        qMakePair(QStringLiteral("samples"), QStringLiteral("samples")),
        qMakePair(QStringLiteral("reworkSamples"), QStringLiteral("rework_samples")),
        qMakePair(QStringLiteral("items"), QStringLiteral("items"))
    };
    for (const auto& item : keys) {
        appendReviewSamplesFromArray(&samples, root.value(item.first).toArray(), item.second);
    }
    const QJsonObject nested = root.value(QStringLiteral("payload")).toObject();
    if (!nested.isEmpty()) {
        for (const auto& item : keys) {
            appendReviewSamplesFromArray(&samples, nested.value(item.first).toArray(), item.second);
        }
    }
    return samples;
}

QString reviewMetricText(const QJsonObject& sample)
{
    QStringList parts;
    for (const QString& key : {
             QStringLiteral("confidence"),
             QStringLiteral("matchedIou"),
             QStringLiteral("matchedMaskIoU"),
             QStringLiteral("editDistance"),
             QStringLiteral("cer")}) {
        if (sample.contains(key)) {
            parts.append(QStringLiteral("%1=%2").arg(key, sampleTextField(sample, QStringList() << key)));
        }
    }
    const QJsonObject prediction = sample.value(QStringLiteral("prediction")).toObject();
    if (!prediction.isEmpty()) {
        const double confidence = prediction.value(QStringLiteral("confidence")).toDouble(-1.0);
        if (confidence >= 0.0) {
            parts.append(QStringLiteral("pred_conf=%1").arg(confidence, 0, 'f', 4));
        }
    }
    return parts.join(QStringLiteral(" | "));
}

QString reviewClassText(const QJsonObject& sample)
{
    const QString direct = sampleTextField(sample, QStringList()
        << QStringLiteral("className")
        << QStringLiteral("class")
        << QStringLiteral("category")
        << QStringLiteral("label"));
    if (!direct.isEmpty()) {
        return direct;
    }
    const QJsonObject prediction = sample.value(QStringLiteral("prediction")).toObject();
    const QJsonObject box = prediction.value(QStringLiteral("box")).toObject();
    if (box.contains(QStringLiteral("classId"))) {
        return QStringLiteral("class_%1").arg(box.value(QStringLiteral("classId")).toInt());
    }
    const QJsonObject groundTruth = sample.value(QStringLiteral("groundTruth")).toObject();
    if (groundTruth.contains(QStringLiteral("classId"))) {
        return QStringLiteral("class_%1").arg(groundTruth.value(QStringLiteral("classId")).toInt());
    }
    return QString();
}

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

QString defaultDatasetConversionOutputPath(const QString& sourcePath, const QString& projectPath, const QString& targetFormat)
{
    const QString normalizedSourcePath = QDir::fromNativeSeparators(sourcePath.trimmed());
    if (normalizedSourcePath.isEmpty()) {
        return QString();
    }

    const QFileInfo sourceInfo(normalizedSourcePath);
    QString datasetName = sourceInfo.isFile() ? sourceInfo.completeBaseName() : sourceInfo.fileName();
    if (datasetName.isEmpty()) {
        datasetName = QStringLiteral("dataset");
    }
    const QString suffix = targetFormat.trimmed().isEmpty() ? QStringLiteral("converted") : targetFormat.trimmed();
    const QString directoryName = QStringLiteral("%1-%2").arg(datasetName, suffix);
    const QString normalizedProjectPath = QDir::fromNativeSeparators(projectPath.trimmed());
    if (!normalizedProjectPath.isEmpty()) {
        const QString conversionRoot = QDir(normalizedProjectPath).filePath(QStringLiteral("datasets/converted"));
        QDir().mkpath(conversionRoot);
        return QDir::cleanPath(QDir(conversionRoot).filePath(directoryName));
    }
    const QDir sourceDir(sourceInfo.isFile() ? sourceInfo.absolutePath() : normalizedSourcePath);
    const QString outputPath = sourceDir.absoluteFilePath(QStringLiteral("../converted/%1").arg(directoryName));
    return QDir::cleanPath(outputPath);
}

void setFieldErrorLabel(QLabel* label, const QString& text)
{
    if (!label) {
        return;
    }
    label->setText(text);
    label->setVisible(!text.isEmpty());
}
} // namespace

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
        state_.dataset.currentPath = directory;
        const QString selectedFormat = currentDatasetFormat();
        state_.dataset.currentFormat = selectedFormat.isEmpty() ? detectedFormat : selectedFormat;
        state_.dataset.currentValid = false;
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
        refreshDatasetConversionDefaultsFromCurrentDataset();
    }
}

void MainWindow::updateDatasetConversionTargetFormats()
{
    if (!datasetConversionSourceFormatCombo_ || !datasetConversionTargetFormatCombo_) {
        return;
    }

    const QString sourceFormat = comboCurrentDataOrText(datasetConversionSourceFormatCombo_);
    const QString previousTarget = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
    const QStringList targets = supportedDatasetConversionTargets(sourceFormat);

    QSignalBlocker blocker(datasetConversionTargetFormatCombo_);
    datasetConversionTargetFormatCombo_->clear();
    for (const QString& target : targets) {
        addComboItem(datasetConversionTargetFormatCombo_, datasetConversionFormatLabel(target), target);
    }
    const int previousIndex = previousTarget.isEmpty() ? -1 : datasetConversionTargetFormatCombo_->findData(previousTarget);
    if (previousIndex >= 0) {
        datasetConversionTargetFormatCombo_->setCurrentIndex(previousIndex);
    } else if (datasetConversionTargetFormatCombo_->count() > 0) {
        datasetConversionTargetFormatCombo_->setCurrentIndex(0);
    }

    if (datasetConversionOutputEdit_ && datasetConversionOutputEdit_->text().trimmed().isEmpty()) {
        const QString inputPath = QDir::fromNativeSeparators(datasetConversionInputEdit_ ? datasetConversionInputEdit_->text().trimmed() : QString());
        const QString targetFormat = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
        const QString outputPath = defaultDatasetConversionOutputPath(inputPath, currentProjectPath_, targetFormat);
        if (!outputPath.isEmpty()) {
            datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(outputPath));
        }
    }
}

void MainWindow::refreshDatasetConversionDefaultsFromCurrentDataset()
{
    if (!datasetConversionInputEdit_ || !datasetConversionSourceFormatCombo_) {
        return;
    }

    QString inputPath = datasetPathEdit_ ? QDir::fromNativeSeparators(datasetPathEdit_->text().trimmed()) : QString();
    if (inputPath.isEmpty()) {
        inputPath = state_.dataset.currentPath;
    }
    if (!inputPath.isEmpty()) {
        datasetConversionInputEdit_->setText(QDir::toNativeSeparators(inputPath));
    }

    QString sourceFormat = state_.dataset.currentFormat;
    if (sourceFormat.isEmpty()) {
        sourceFormat = currentDatasetFormat();
    }
    if (!sourceFormat.isEmpty() && datasetConversionSourceFormatCombo_->findData(sourceFormat) >= 0) {
        setComboCurrentData(datasetConversionSourceFormatCombo_, sourceFormat);
    } else {
        updateDatasetConversionTargetFormats();
    }

    if (datasetConversionOutputEdit_ && datasetConversionOutputEdit_->text().trimmed().isEmpty()) {
        const QString targetFormat = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
        const QString outputPath = defaultDatasetConversionOutputPath(inputPath, currentProjectPath_, targetFormat);
        if (!outputPath.isEmpty()) {
            datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(outputPath));
        }
    }
}

void MainWindow::browseDatasetConversionInput()
{
    const QString sourceFormat = comboCurrentDataOrText(datasetConversionSourceFormatCombo_);
    const bool expectsCocoJsonFile = sourceFormat == QStringLiteral("coco_json");
    const QString selectedPath = expectsCocoJsonFile
        ? QFileDialog::getOpenFileName(this, QStringLiteral("选择 COCO JSON 标注文件"), QString(), QStringLiteral("JSON 文件 (*.json);;所有文件 (*)"))
        : QFileDialog::getExistingDirectory(this, QStringLiteral("选择待转换数据集目录"));
    if (selectedPath.isEmpty()) {
        return;
    }

    const QString normalizedInputPath = QDir::fromNativeSeparators(selectedPath);
    if (datasetConversionInputEdit_) {
        datasetConversionInputEdit_->setText(QDir::toNativeSeparators(normalizedInputPath));
    }

    if (!expectsCocoJsonFile) {
        const QString detectedFormat = detectDatasetFormatFromPath(normalizedInputPath);
        if (!detectedFormat.isEmpty()
            && supportedDatasetConversionSourceFormats().contains(detectedFormat)
            && datasetConversionSourceFormatCombo_) {
            setComboCurrentData(datasetConversionSourceFormatCombo_, detectedFormat);
        } else {
            updateDatasetConversionTargetFormats();
        }
    }

    if (datasetConversionOutputEdit_) {
        const QString targetFormat = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
        const QString outputPath = defaultDatasetConversionOutputPath(normalizedInputPath, currentProjectPath_, targetFormat);
        if (!outputPath.isEmpty()) {
            datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(outputPath));
        }
    }
}

void MainWindow::browseDatasetConversionOutput()
{
    const QString directory = QFileDialog::getExistingDirectory(this, QStringLiteral("选择转换输出目录"));
    if (!directory.isEmpty() && datasetConversionOutputEdit_) {
        datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(directory));
    }
}

void MainWindow::clearDatasetConversionErrors()
{
    setFieldErrorLabel(datasetConversionSourceErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionTargetErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionInputErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionOutputErrorLabel_, QString());
}

void MainWindow::appendDatasetConversionLog(const QString& text)
{
    if (!datasetConversionLog_ || text.isEmpty()) {
        return;
    }
    if (datasetConversionLog_->toPlainText().trimmed() == QStringLiteral("等待转换。")) {
        datasetConversionLog_->clear();
    }
    datasetConversionLog_->appendPlainText(text);
}

void MainWindow::setDatasetConversionFormRunning(bool running)
{
    if (datasetConversionSourceFormatCombo_) {
        datasetConversionSourceFormatCombo_->setEnabled(!running);
    }
    if (datasetConversionTargetFormatCombo_) {
        datasetConversionTargetFormatCombo_->setEnabled(!running);
    }
    if (datasetConversionInputEdit_) {
        datasetConversionInputEdit_->setEnabled(!running);
    }
    if (datasetConversionOutputEdit_) {
        datasetConversionOutputEdit_->setEnabled(!running);
    }
    if (datasetConversionBrowseInputButton_) {
        datasetConversionBrowseInputButton_->setEnabled(!running);
    }
    if (datasetConversionBrowseOutputButton_) {
        datasetConversionBrowseOutputButton_->setEnabled(!running);
    }
    if (datasetConversionStartButton_) {
        datasetConversionStartButton_->setEnabled(!running);
    }
    if (datasetConversionCancelButton_) {
        datasetConversionCancelButton_->setEnabled(running);
    }
}

void MainWindow::cancelDatasetConversion()
{
    if (!worker_.isRunning()) {
        return;
    }
    worker_.cancel();
    if (datasetConversionStatusLabel_) {
        datasetConversionStatusLabel_->setText(QStringLiteral("正在取消数据集转换。"));
    }
    appendDatasetConversionLog(QStringLiteral("正在取消数据集转换。"));
}

void MainWindow::startDatasetConversion()
{
    clearDatasetConversionErrors();

    DatasetConversionForm form;
    form.sourceFormat = comboCurrentDataOrText(datasetConversionSourceFormatCombo_);
    form.targetFormat = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
    form.inputPath = datasetConversionInputEdit_ ? datasetConversionInputEdit_->text() : QString();
    form.outputPath = datasetConversionOutputEdit_ ? datasetConversionOutputEdit_->text() : QString();
    form.workerRunning = worker_.isRunning();

    const DatasetConversionValidation validation = validateDatasetConversionForm(form);
    if (!validation.ok) {
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(validation.summary);
        }
        setFieldErrorLabel(datasetConversionSourceErrorLabel_, validation.sourceFormatError);
        setFieldErrorLabel(datasetConversionTargetErrorLabel_, validation.targetFormatError);
        setFieldErrorLabel(datasetConversionInputErrorLabel_, validation.inputPathError);
        setFieldErrorLabel(datasetConversionOutputErrorLabel_, validation.outputPathError);
        return;
    }

    const QString sourcePath = normalizedDatasetConversionPath(form.inputPath);
    const QString outputPath = normalizedDatasetConversionPath(form.outputPath);
    if (!QDir().mkpath(outputPath)) {
        const QString message = QStringLiteral("无法创建输出目录：%1").arg(QDir::toNativeSeparators(outputPath));
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(message);
        }
        setFieldErrorLabel(datasetConversionOutputErrorLabel_, message);
        appendDatasetConversionLog(message);
        return;
    }

    QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Validate,
            QStringLiteral("dataset_conversion"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集格式转换中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    } else {
        state_.training.currentTaskId = taskId;
    }

    QJsonObject options;
    options.insert(QStringLiteral("copyImages"), true);
    options.insert(QStringLiteral("maxIssues"), 200);

    if (datasetConversionProgressBar_) {
        datasetConversionProgressBar_->setValue(0);
    }
    if (datasetConversionLog_) {
        datasetConversionLog_->clear();
    }
    if (datasetConversionResultLabel_) {
        datasetConversionResultLabel_->setText(QStringLiteral("等待转换结果。"));
    }
    if (datasetConversionStatusLabel_) {
        datasetConversionStatusLabel_->setText(QStringLiteral("正在通过 Worker 转换数据集。"));
    }
    appendDatasetConversionLog(QStringLiteral("开始转换数据集。"));
    state_.dataset.currentConversionTaskId = taskId;
    setDatasetConversionFormRunning(true);

    QString error;
    if (!worker_.requestDatasetConversion(
            workerExecutablePath(),
            sourcePath,
            outputPath,
            form.sourceFormat,
            form.targetFormat,
            options,
            &error,
            taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            updateRecentTasks();
        }
        state_.training.currentTaskId.clear();
        state_.dataset.currentConversionTaskId.clear();
        setDatasetConversionFormRunning(false);
        const QString message = QStringLiteral("无法启动数据集转换：%1").arg(error);
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(message);
        }
        appendDatasetConversionLog(message);
        QMessageBox::critical(this, QStringLiteral("数据集转换"), message);
        return;
    }

    workerPill_->setStatus(QStringLiteral("数据集转换中"), StatusPill::Tone::Info);
    statusBar()->showMessage(QStringLiteral("正在转换数据集"), 3000);
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

    state_.dataset.currentValid = false;
    state_.dataset.currentPath = path;
    state_.dataset.currentFormat = format;
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
            state_.training.currentTaskId.clear();
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

    bool datasetReady = state_.dataset.currentValid && state_.dataset.currentPath == path && state_.dataset.currentFormat == format;
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
            state_.training.currentTaskId.clear();
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
            state_.training.currentTaskId.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据质量报告"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据质量报告生成中"), StatusPill::Tone::Info);
    setDatasetRepairLoopRows(
        uiText("修复闭环：质量报告生成中。"),
        QVector<QStringList>{
            QStringList() << uiText("质量报告") << uiText("运行中") << uiText("Worker 正在扫描数据、问题样本和修复清单。"),
            QStringList() << uiText("外部修复") << uiText("等待") << uiText("报告完成后再打开问题清单。"),
            QStringList() << uiText("复检") << uiText("等待") << uiText("修复完成后重新生成质量报告。")
        });
}

void MainWindow::openDatasetQualityFixList()
{
    if (state_.dataset.latestQualityFixListPath.isEmpty() || !QFileInfo::exists(state_.dataset.latestQualityFixListPath)) {
        QMessageBox::information(this, uiText("问题清单"), uiText("请先生成数据质量报告。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(state_.dataset.latestQualityFixListPath));
}

void MainWindow::openDatasetQualityReport()
{
    if (state_.dataset.latestQualityReportPath.isEmpty() || !QFileInfo::exists(state_.dataset.latestQualityReportPath)) {
        QMessageBox::information(this, uiText("质量报告"), uiText("请先生成数据质量报告。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(state_.dataset.latestQualityReportPath));
}

void MainWindow::launchXAnyLabelingForQualityFix()
{
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (datasetPath.isEmpty()) {
        QMessageBox::information(this, uiText("X-AnyLabeling 修复"), uiText("请先选择数据集目录。"));
        return;
    }
    if (!state_.dataset.latestQualityFixListPath.isEmpty()) {
        statusBar()->showMessage(uiText("问题清单：%1").arg(QDir::toNativeSeparators(state_.dataset.latestQualityFixListPath)), 6000);
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

void MainWindow::browseSampleReviewFile()
{
    const QString file = QFileDialog::getOpenFileName(
        this,
        uiText("选择复核样本文件"),
        currentProjectPath_,
        QStringLiteral("Review samples (*.json);;All files (*.*)"));
    if (!file.isEmpty() && reviewSamplePathEdit_) {
        reviewSamplePathEdit_->setText(QDir::toNativeSeparators(file));
        loadSampleReviewFile();
    }
}

void MainWindow::loadSampleReviewFile()
{
    const QString path = QDir::fromNativeSeparators(reviewSamplePathEdit_ ? reviewSamplePathEdit_->text().trimmed() : QString());
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        QMessageBox::warning(this, uiText("样本复核"), uiText("请选择存在的复核样本 JSON 文件。"));
        return;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, uiText("样本复核"), uiText("无法读取复核样本文件：%1").arg(QDir::toNativeSeparators(path)));
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || (!document.isObject() && !document.isArray())) {
        QMessageBox::critical(this, uiText("样本复核"), uiText("复核样本 JSON 解析失败：%1").arg(parseError.errorString()));
        return;
    }

    state_.dataset.sampleReviewSamples = extractReviewSamples(document);
    if (reviewSourceFilterCombo_) {
        reviewSourceFilterCombo_->clear();
        reviewSourceFilterCombo_->addItem(uiText("全部来源"), QString());
    }
    if (reviewReasonFilterCombo_) {
        reviewReasonFilterCombo_->clear();
        reviewReasonFilterCombo_->addItem(uiText("全部问题"), QString());
    }
    QStringList sources;
    QStringList reasons;
    for (const QJsonValue& value : state_.dataset.sampleReviewSamples) {
        const QJsonObject sample = value.toObject();
        const QString source = sample.value(QStringLiteral("source")).toString();
        const QString reason = sample.value(QStringLiteral("reason")).toString();
        if (!source.isEmpty() && !sources.contains(source)) {
            sources.append(source);
        }
        if (!reason.isEmpty() && !reasons.contains(reason)) {
            reasons.append(reason);
        }
    }
    sources.sort(Qt::CaseInsensitive);
    reasons.sort(Qt::CaseInsensitive);
    for (const QString& source : sources) {
        reviewSourceFilterCombo_->addItem(source, source);
    }
    for (const QString& reason : reasons) {
        reviewReasonFilterCombo_->addItem(reason, reason);
    }
    state_.dataset.latestReviewListPath.clear();
    refreshSampleReviewTable();
    statusBar()->showMessage(uiText("已加载复核样本：%1 条").arg(state_.dataset.sampleReviewSamples.size()), 4000);
}

QJsonArray MainWindow::filteredSampleReviewRows() const
{
    QJsonArray rows;
    const QString sourceFilter = reviewSourceFilterCombo_ ? reviewSourceFilterCombo_->currentData().toString() : QString();
    const QString reasonFilter = reviewReasonFilterCombo_ ? reviewReasonFilterCombo_->currentData().toString() : QString();
    const QString query = reviewSearchEdit_ ? reviewSearchEdit_->text().trimmed().toLower() : QString();
    for (const QJsonValue& value : state_.dataset.sampleReviewSamples) {
        const QJsonObject sample = value.toObject();
        const QString source = sample.value(QStringLiteral("source")).toString();
        const QString reason = sample.value(QStringLiteral("reason")).toString();
        if (!sourceFilter.isEmpty() && source != sourceFilter) {
            continue;
        }
        if (!reasonFilter.isEmpty() && reason != reasonFilter) {
            continue;
        }
        if (!query.isEmpty()) {
            const QString combined = QString::fromUtf8(QJsonDocument(sample).toJson(QJsonDocument::Compact)).toLower();
            if (!combined.contains(query)) {
                continue;
            }
        }
        rows.append(sample);
    }
    return rows;
}

void MainWindow::refreshSampleReviewTable()
{
    if (!sampleReviewTable_) {
        return;
    }
    const QJsonArray rows = filteredSampleReviewRows();
    sampleReviewTable_->setRowCount(0);
    for (const QJsonValue& value : rows) {
        const QJsonObject sample = value.toObject();
        const int row = sampleReviewTable_->rowCount();
        sampleReviewTable_->insertRow(row);
        const QString imagePath = samplePathField(sample, QStringList()
            << QStringLiteral("imagePath")
            << QStringLiteral("path")
            << QStringLiteral("filePath"));
        const QString labelPath = samplePathField(sample, QStringList()
            << QStringLiteral("labelPath")
            << QStringLiteral("annotationPath")
            << QStringLiteral("gtPath"));
        sampleReviewTable_->setItem(row, 0, new QTableWidgetItem(sample.value(QStringLiteral("source")).toString()));
        sampleReviewTable_->setItem(row, 1, new QTableWidgetItem(sample.value(QStringLiteral("reason")).toString()));
        sampleReviewTable_->setItem(row, 2, new QTableWidgetItem(reviewClassText(sample)));
        sampleReviewTable_->setItem(row, 3, new QTableWidgetItem(reviewMetricText(sample)));
        auto* imageItem = new QTableWidgetItem(compactPathForStatus(imagePath, 80));
        imageItem->setData(Qt::UserRole, imagePath);
        imageItem->setToolTip(QDir::toNativeSeparators(imagePath));
        sampleReviewTable_->setItem(row, 4, imageItem);
        auto* labelItem = new QTableWidgetItem(compactPathForStatus(labelPath, 80));
        labelItem->setData(Qt::UserRole, labelPath);
        labelItem->setToolTip(QDir::toNativeSeparators(labelPath));
        sampleReviewTable_->setItem(row, 5, labelItem);
        sampleReviewTable_->setItem(row, 6, new QTableWidgetItem(sampleTextField(sample, QStringList()
            << QStringLiteral("message")
            << QStringLiteral("note")
            << QStringLiteral("description")
            << QStringLiteral("groundTruth")
            << QStringLiteral("prediction"))));
    }
    if (sampleReviewSummaryLabel_) {
        sampleReviewSummaryLabel_->setText(uiText("复核样本：显示 %1 / 总计 %2；清单 %3")
            .arg(rows.size())
            .arg(state_.dataset.sampleReviewSamples.size())
            .arg(state_.dataset.latestReviewListPath.isEmpty() ? uiText("尚未生成") : QDir::toNativeSeparators(state_.dataset.latestReviewListPath)));
    }
}

void MainWindow::generateFilteredReviewList()
{
    const QJsonArray rows = filteredSampleReviewRows();
    if (rows.isEmpty()) {
        QMessageBox::information(this, uiText("样本复核"), uiText("当前过滤条件下没有样本。"));
        return;
    }
    const QString sourcePath = QDir::fromNativeSeparators(reviewSamplePathEdit_ ? reviewSamplePathEdit_->text().trimmed() : QString());
    const QString outputDir = !currentProjectPath_.isEmpty()
        ? QDir(currentProjectPath_).filePath(QStringLiteral("datasets/review"))
        : QFileInfo(sourcePath).absolutePath();
    QDir().mkpath(outputDir);
    const QString listPath = QDir(outputDir).filePath(QStringLiteral("xanylabeling_review_list.txt"));
    const QString manifestPath = QDir(outputDir).filePath(QStringLiteral("rework_sample_set.json"));

    QFile listFile(listPath);
    if (!listFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        QMessageBox::critical(this, uiText("样本复核"), uiText("无法写入复核清单：%1").arg(QDir::toNativeSeparators(listPath)));
        return;
    }
    QTextStream listStream(&listFile);
    listStream.setCodec("UTF-8");
    QJsonArray manifestSamples;
    for (const QJsonValue& value : rows) {
        const QJsonObject sample = value.toObject();
        const QString imagePath = samplePathField(sample, QStringList()
            << QStringLiteral("imagePath")
            << QStringLiteral("path")
            << QStringLiteral("filePath"));
        if (!imagePath.isEmpty()) {
            listStream << QDir::toNativeSeparators(imagePath) << QLatin1Char('\n');
        }
        manifestSamples.append(sample);
    }
    listFile.close();

    QFile manifestFile(manifestPath);
    if (!manifestFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QMessageBox::critical(this, uiText("样本复核"), uiText("无法写入复核 manifest：%1").arg(QDir::toNativeSeparators(manifestPath)));
        return;
    }
    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 1);
    manifest.insert(QStringLiteral("kind"), QStringLiteral("rework_sample_set"));
    manifest.insert(QStringLiteral("sourcePath"), sourcePath);
    manifest.insert(QStringLiteral("listPath"), listPath);
    manifest.insert(QStringLiteral("sampleCount"), rows.size());
    manifest.insert(QStringLiteral("samples"), manifestSamples);
    manifestFile.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented));
    manifestFile.close();

    state_.dataset.latestReviewListPath = listPath;
    refreshSampleReviewTable();
    statusBar()->showMessage(uiText("复核清单已生成：%1").arg(QDir::toNativeSeparators(listPath)), 5000);
}

void MainWindow::openSelectedReviewSample()
{
    if (!sampleReviewTable_ || sampleReviewTable_->currentRow() < 0) {
        QMessageBox::information(this, uiText("样本复核"), uiText("请先选择一条复核样本。"));
        return;
    }
    QTableWidgetItem* item = sampleReviewTable_->item(sampleReviewTable_->currentRow(), 4);
    const QString path = item ? item->data(Qt::UserRole).toString() : QString();
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        QMessageBox::warning(this, uiText("样本复核"), uiText("样本图片不存在。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

void MainWindow::launchXAnyLabelingForReview()
{
    if (state_.dataset.latestReviewListPath.isEmpty()) {
        generateFilteredReviewList();
    }
    const QString program = resolvedXAnyLabelingProgram();
    if (program.isEmpty()) {
        updateAnnotationToolStatus();
        QMessageBox::warning(this, QStringLiteral("X-AnyLabeling"), uiText("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"));
        return;
    }
    QString targetDir = state_.dataset.currentPath;
    if (targetDir.isEmpty() && !state_.dataset.latestReviewListPath.isEmpty()) {
        targetDir = QFileInfo(state_.dataset.latestReviewListPath).absolutePath();
    }
    if (targetDir.isEmpty()) {
        targetDir = currentProjectPath_;
    }
    if (QProcess::startDetached(program, QStringList() << targetDir)) {
        statusBar()->showMessage(uiText("已启动 X-AnyLabeling；复核清单：%1").arg(QDir::toNativeSeparators(state_.dataset.latestReviewListPath)), 6000);
    } else {
        QMessageBox::warning(this, QStringLiteral("X-AnyLabeling"), uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(program)));
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
            state_.training.currentTaskId.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据集快照"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集快照创建中"), StatusPill::Tone::Info);
}
