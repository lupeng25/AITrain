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

    QString datasetName = QFileInfo(normalizedSourcePath).fileName();
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
    const QString outputPath = QDir(normalizedSourcePath).absoluteFilePath(QStringLiteral("../converted/%1").arg(directoryName));
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
        inputPath = currentDatasetPath_;
    }
    if (!inputPath.isEmpty()) {
        datasetConversionInputEdit_->setText(QDir::toNativeSeparators(inputPath));
    }

    QString sourceFormat = currentDatasetFormat_;
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
    const QString directory = QFileDialog::getExistingDirectory(this, QStringLiteral("选择待转换数据集目录"));
    if (directory.isEmpty()) {
        return;
    }

    const QString normalizedDirectory = QDir::fromNativeSeparators(directory);
    if (datasetConversionInputEdit_) {
        datasetConversionInputEdit_->setText(QDir::toNativeSeparators(normalizedDirectory));
    }

    const QString detectedFormat = detectDatasetFormatFromPath(normalizedDirectory);
    if (!detectedFormat.isEmpty()
        && supportedDatasetConversionSourceFormats().contains(detectedFormat)
        && datasetConversionSourceFormatCombo_) {
        setComboCurrentData(datasetConversionSourceFormatCombo_, detectedFormat);
    } else {
        updateDatasetConversionTargetFormats();
    }

    if (datasetConversionOutputEdit_) {
        const QString targetFormat = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
        const QString outputPath = defaultDatasetConversionOutputPath(normalizedDirectory, currentProjectPath_, targetFormat);
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
        currentTaskId_ = taskId;
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
    currentDatasetConversionTaskId_ = taskId;
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
        currentTaskId_.clear();
        currentDatasetConversionTaskId_.clear();
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
    if (latestQualityFixListPath_.isEmpty() || !QFileInfo::exists(latestQualityFixListPath_)) {
        QMessageBox::information(this, uiText("问题清单"), uiText("请先生成数据质量报告。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(latestQualityFixListPath_));
}

void MainWindow::openDatasetQualityReport()
{
    if (latestQualityReportPath_.isEmpty() || !QFileInfo::exists(latestQualityReportPath_)) {
        QMessageBox::information(this, uiText("质量报告"), uiText("请先生成数据质量报告。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(latestQualityReportPath_));
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

    sampleReviewSamples_ = extractReviewSamples(document);
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
    for (const QJsonValue& value : sampleReviewSamples_) {
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
    latestReviewListPath_.clear();
    refreshSampleReviewTable();
    statusBar()->showMessage(uiText("已加载复核样本：%1 条").arg(sampleReviewSamples_.size()), 4000);
}

QJsonArray MainWindow::filteredSampleReviewRows() const
{
    QJsonArray rows;
    const QString sourceFilter = reviewSourceFilterCombo_ ? reviewSourceFilterCombo_->currentData().toString() : QString();
    const QString reasonFilter = reviewReasonFilterCombo_ ? reviewReasonFilterCombo_->currentData().toString() : QString();
    const QString query = reviewSearchEdit_ ? reviewSearchEdit_->text().trimmed().toLower() : QString();
    for (const QJsonValue& value : sampleReviewSamples_) {
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
            .arg(sampleReviewSamples_.size())
            .arg(latestReviewListPath_.isEmpty() ? uiText("尚未生成") : QDir::toNativeSeparators(latestReviewListPath_)));
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

    latestReviewListPath_ = listPath;
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
    if (latestReviewListPath_.isEmpty()) {
        generateFilteredReviewList();
    }
    const QString program = resolvedXAnyLabelingProgram();
    if (program.isEmpty()) {
        updateAnnotationToolStatus();
        QMessageBox::warning(this, QStringLiteral("X-AnyLabeling"), uiText("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"));
        return;
    }
    QString targetDir = currentDatasetPath_;
    if (targetDir.isEmpty() && !latestReviewListPath_.isEmpty()) {
        targetDir = QFileInfo(latestReviewListPath_).absolutePath();
    }
    if (targetDir.isEmpty()) {
        targetDir = currentProjectPath_;
    }
    if (QProcess::startDetached(program, QStringList() << targetDir)) {
        statusBar()->showMessage(uiText("已启动 X-AnyLabeling；复核清单：%1").arg(QDir::toNativeSeparators(latestReviewListPath_)), 6000);
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

void MainWindow::validateDeploymentArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("部署验证"), uiText("Worker 正在执行任务，稍后再验证部署产物。"));
        return;
    }
    QString modelPath = QDir::fromNativeSeparators(conversionOutputEdit_ ? conversionOutputEdit_->text().trimmed() : QString());
    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        modelPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
    }
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, uiText("部署验证"), uiText("请选择已导出的模型产物或模型输入。"));
        return;
    }
    const QString sampleImagePath = QDir::fromNativeSeparators(conversionValidationImageEdit_ ? conversionValidationImageEdit_->text().trimmed() : QString());
    const QString format = conversionFormatCombo_ ? conversionFormatCombo_->currentData().toString() : QString();
    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Benchmark,
            QStringLiteral("deployment_validation"),
            QStringLiteral("com.aitrain.plugins.yolo_native"),
            outputPath,
            uiText("部署产物验证中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    } else {
        outputPath = QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("deployment_validation"));
    }

    QJsonObject options;
    QString error;
    if (!worker_.requestDeploymentValidation(workerExecutablePath(), modelPath, outputPath, format, sampleImagePath, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("部署验证"), error);
        return;
    }
    if (deploymentValidationResultLabel_) {
        deploymentValidationResultLabel_->setText(uiText("正在验证部署产物：%1").arg(QDir::toNativeSeparators(modelPath)));
    }
    workerPill_->setStatus(uiText("部署验证中"), StatusPill::Tone::Info);
}

void MainWindow::runCustomerOcrAcceptance()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("OCR 验收"), uiText("Worker 正在执行任务，稍后再运行 OCR 验收。"));
        return;
    }
    QJsonObject options;
    options.insert(QStringLiteral("detDatasetPath"), QDir::fromNativeSeparators(customerOcrDetDatasetEdit_ ? customerOcrDetDatasetEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("recDatasetPath"), QDir::fromNativeSeparators(customerOcrRecDatasetEdit_ ? customerOcrRecDatasetEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("systemImagesPath"), QDir::fromNativeSeparators(customerOcrSystemImagesEdit_ ? customerOcrSystemImagesEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("detReportPath"), QDir::fromNativeSeparators(customerOcrDetReportEdit_ ? customerOcrDetReportEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("recReportPath"), QDir::fromNativeSeparators(customerOcrRecReportEdit_ ? customerOcrRecReportEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("systemReportPath"), QDir::fromNativeSeparators(customerOcrSystemReportEdit_ ? customerOcrSystemReportEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("detOnnxEvidencePath"), QDir::fromNativeSeparators(customerOcrDetOnnxEvidenceEdit_ ? customerOcrDetOnnxEvidenceEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("minRecAccuracy"), customerOcrMinAccEdit_ ? customerOcrMinAccEdit_->text().toDouble() : 0.70);
    options.insert(QStringLiteral("maxRecCer"), customerOcrMaxCerEdit_ ? customerOcrMaxCerEdit_->text().toDouble() : 0.30);
    options.insert(QStringLiteral("requireFullDomainEvidence"), true);
    options.insert(QStringLiteral("allowPublicLikeData"), customerOcrAllowPublicCheck_ && customerOcrAllowPublicCheck_->isChecked());
    options.insert(QStringLiteral("requireDetOnnxEvidence"), customerOcrRequireDetOnnxCheck_ && customerOcrRequireDetOnnxCheck_->isChecked());
    options.insert(QStringLiteral("minDetSamples"), 1);
    options.insert(QStringLiteral("minRecSamples"), 1);
    options.insert(QStringLiteral("minSystemImages"), 1);

    QString taskId;
    QString outputPath = QDir::fromNativeSeparators(customerOcrOutputEdit_ ? customerOcrOutputEdit_->text().trimmed() : QString());
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        if (outputPath.isEmpty()) {
            outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        }
        taskId = createRepositoryTask(
            aitrain::TaskKind::Report,
            QStringLiteral("customer_ocr_acceptance"),
            QStringLiteral("com.aitrain.plugins.ocr_rec_native"),
            outputPath,
            uiText("客户域 OCR 验收中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestCustomerOcrAcceptance(workerExecutablePath(), outputPath, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("OCR 验收"), error);
        return;
    }
    if (customerOcrStatusLabel_) {
        customerOcrStatusLabel_->setText(uiText("客户域 OCR 验收运行中。"));
    }
    workerPill_->setStatus(uiText("OCR 验收中"), StatusPill::Tone::Info);
}

void MainWindow::collectDiagnosticsBundle()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("诊断包"), uiText("Worker 正在执行任务，稍后再生成诊断包。"));
        return;
    }

    QJsonArray recentTasks;
    QJsonArray recentFailures;
    QJsonArray artifactIndex;
    if (repository_.isOpen()) {
        QString error;
        const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(20, &error);
        for (const aitrain::TaskRecord& task : tasks) {
            QJsonObject taskObject;
            taskObject.insert(QStringLiteral("id"), task.id);
            taskObject.insert(QStringLiteral("kind"), aitrain::taskKindToString(task.kind));
            taskObject.insert(QStringLiteral("state"), aitrain::taskStateToString(task.state));
            taskObject.insert(QStringLiteral("taskType"), task.taskType);
            taskObject.insert(QStringLiteral("workDir"), task.workDir);
            taskObject.insert(QStringLiteral("message"), task.message);
            taskObject.insert(QStringLiteral("updatedAt"), task.updatedAt.toUTC().toString(Qt::ISODateWithMs));
            recentTasks.append(taskObject);
            if (task.state == aitrain::TaskState::Failed) {
                recentFailures.append(taskObject);
            }
            const QVector<aitrain::ArtifactRecord> artifacts = repository_.artifactsForTask(task.id, &error);
            for (const aitrain::ArtifactRecord& artifact : artifacts) {
                artifactIndex.append(QJsonObject{
                    {QStringLiteral("taskId"), artifact.taskId},
                    {QStringLiteral("kind"), artifact.kind},
                    {QStringLiteral("path"), artifact.path},
                    {QStringLiteral("message"), artifact.message}
                });
            }
        }
    }

    QJsonObject context;
    context.insert(QStringLiteral("projectName"), currentProjectName_);
    context.insert(QStringLiteral("projectPath"), currentProjectPath_);
    context.insert(QStringLiteral("workerExecutable"), workerExecutablePath());
    context.insert(QStringLiteral("recentTasks"), recentTasks);
    context.insert(QStringLiteral("recentFailures"), recentFailures);
    context.insert(QStringLiteral("artifactIndex"), artifactIndex);
    context.insert(QStringLiteral("licenseSummary"), QJsonObject{
        {QStringLiteral("status"), licenseOwner_.isEmpty() ? QStringLiteral("unknown") : QStringLiteral("registered")},
        {QStringLiteral("owner"), licenseOwner_},
        {QStringLiteral("expiry"), licenseExpiry_}
    });
    context.insert(QStringLiteral("pluginSummary"), QJsonObject{
        {QStringLiteral("count"), pluginManager_.plugins().size()},
        {QStringLiteral("searchPaths"), QJsonArray::fromStringList(pluginSearchPaths())}
    });

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Report,
            QStringLiteral("diagnostic_bundle"),
            QStringLiteral("com.aitrain.system"),
            outputPath,
            uiText("诊断包生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    } else {
        outputPath = QDir(QDir::tempPath()).filePath(QStringLiteral("aitrain-diagnostics"));
    }

    QString error;
    if (!worker_.requestDiagnosticsBundle(workerExecutablePath(), outputPath, context, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("诊断包"), error);
        return;
    }
    if (diagnosticsStatusLabel_) {
        diagnosticsStatusLabel_->setText(uiText("诊断包生成中。"));
    }
    workerPill_->setStatus(uiText("诊断包生成中"), StatusPill::Tone::Info);
}

void MainWindow::importAcceptanceEvidence()
{
    const QString file = QFileDialog::getOpenFileName(
        this,
        uiText("导入验收结果"),
        currentProjectPath_,
        QStringLiteral("Acceptance evidence (*.json *.md *.txt);;All files (*.*)"));
    if (file.isEmpty()) {
        return;
    }
    QString status = QStringLiteral("imported");
    QString stage = QFileInfo(file).completeBaseName();
    QString message = uiText("已导入外部验收结果。");
    if (QFileInfo(file).suffix().compare(QStringLiteral("json"), Qt::CaseInsensitive) == 0) {
        QFile jsonFile(file);
        if (jsonFile.open(QIODevice::ReadOnly)) {
            const QJsonDocument document = QJsonDocument::fromJson(jsonFile.readAll());
            const QJsonObject object = document.object();
            status = object.value(QStringLiteral("status")).toString(object.value(QStringLiteral("ok")).toBool(false) ? QStringLiteral("passed") : QStringLiteral("blocked"));
            stage = object.value(QStringLiteral("kind")).toString(stage);
            message = object.value(QStringLiteral("message")).toString(object.value(QStringLiteral("note")).toString(message));
        }
    } else {
        QFile textFile(file);
        if (textFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            const QString text = QString::fromUtf8(textFile.readAll()).toLower();
            if (text.contains(QStringLiteral("hardware-blocked"))) {
                status = QStringLiteral("hardware-blocked");
            } else if (text.contains(QStringLiteral("blocked")) || text.contains(QStringLiteral("failed"))) {
                status = QStringLiteral("blocked");
            } else if (text.contains(QStringLiteral("passed"))) {
                status = QStringLiteral("passed");
            }
        }
    }
    setAcceptanceTableRow(deliveryAcceptanceTable_, stage, status, file, message);
    updateDeliveryAcceptanceSummary();
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
            uiText("璁粌"),
            QStringLiteral("Training preflight blocked:\n%1").arg(blockers.join(QStringLiteral("\n"))));
        return;
    }
    parameters.insert(QStringLiteral("trainingBackend"), backendForRequest);
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
    context.insert(QStringLiteral("taskType"), currentTaskType());
    context.insert(QStringLiteral("trainingBackend"), trainingBackendCombo_ ? trainingBackendCombo_->currentData().toString() : QString());
    context.insert(QStringLiteral("modelPreset"), modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString());
    context.insert(QStringLiteral("sourceTaskId"), selectedTaskId());
    if (repository_.isOpen() && !currentDatasetPath_.isEmpty()) {
        QString snapshotError;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(currentDatasetPath_, &snapshotError);
        const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(dataset.id, &snapshotError);
        if (snapshot.id > 0) {
            context.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
            context.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
            context.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
        }
    }
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
