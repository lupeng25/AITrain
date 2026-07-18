#include "MainWindow.h"
#include "TaskExecutionController.h"

#include "DatasetConversionUiModel.h"
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
#include <QDir>
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
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
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

QString translatedText(const QString& source)
{
    const QString fixCountTemplate = QStringLiteral("请修正 %1 个字段后再转换。");
    const int markerIndex = fixCountTemplate.indexOf(QStringLiteral("%1"));
    const QString prefix = fixCountTemplate.left(markerIndex);
    const QString suffix = fixCountTemplate.mid(markerIndex + 2);
    if (markerIndex >= 0 && source.startsWith(prefix) && source.endsWith(suffix)) {
        const QString count = source.mid(prefix.size(), source.size() - prefix.size() - suffix.size());
        return uiText("请修正 %1 个字段后再转换。").arg(count);
    }
    return translateText("MainWindow", source);
}

DatasetConversionValidation translatedValidation(DatasetConversionValidation validation)
{
    validation.summary = translatedText(validation.summary);
    validation.sourceFormatError = translatedText(validation.sourceFormatError);
    validation.targetFormatError = translatedText(validation.targetFormatError);
    validation.inputPathError = translatedText(validation.inputPathError);
    validation.outputPathError = translatedText(validation.outputPathError);
    QStringList messages;
    for (const QString& message : validation.messages) {
        messages.append(translatedText(message));
    }
    validation.messages = messages;
    return validation;
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
        qMakePair(QStringLiteral("issues"), QStringLiteral("quality_issues")),
        qMakePair(QStringLiteral("actions"), QStringLiteral("repair_actions")),
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
        state_.dataset.currentPath = directory;
        const QString selectedFormat = currentDatasetFormat();
        state_.dataset.currentFormat = selectedFormat.isEmpty() ? detectedFormat : selectedFormat;
        state_.dataset.currentDatasetId.clear();
        state_.dataset.currentDatasetVersionId.clear();
        state_.dataset.currentSnapshotId.clear();
        state_.dataset.currentSnapshotArtifactId.clear();
        state_.dataset.currentValid = false;
        if (dataQualityDatasetIdEdit_) dataQualityDatasetIdEdit_->clear();
        if (dataQualityDatasetVersionIdEdit_) dataQualityDatasetVersionIdEdit_->clear();
        if (dataQualitySnapshotIdEdit_) dataQualitySnapshotIdEdit_->clear();
        if (dataQualitySnapshotArtifactIdEdit_) dataQualitySnapshotArtifactIdEdit_->clear();
        if (splitSourceDatasetIdEdit_) splitSourceDatasetIdEdit_->clear();
        if (splitSourceDatasetVersionIdEdit_) splitSourceDatasetVersionIdEdit_->clear();
        if (splitSourceSnapshotIdEdit_) splitSourceSnapshotIdEdit_->clear();
        if (splitSourceSnapshotArtifactIdEdit_) splitSourceSnapshotArtifactIdEdit_->clear();
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

    if (datasetConversionTargetDatasetNameEdit_
        && datasetConversionTargetDatasetNameEdit_->text().trimmed().isEmpty()
        && !inputPath.isEmpty()) {
        datasetConversionTargetDatasetNameEdit_->setText(
            QStringLiteral("%1-%2").arg(QFileInfo(inputPath).completeBaseName(),
                comboCurrentDataOrText(datasetConversionTargetFormatCombo_)));
    }
}

void MainWindow::browseDatasetConversionInput()
{
    const QString sourceFormat = comboCurrentDataOrText(datasetConversionSourceFormatCombo_);
    const bool expectsCocoJsonFile = sourceFormat == QStringLiteral("coco_json");
    const QString selectedPath = expectsCocoJsonFile
        ? QFileDialog::getOpenFileName(this, uiText("选择 COCO JSON 标注文件"), QString(), uiText("JSON 文件 (*.json);;所有文件 (*)"))
        : QFileDialog::getExistingDirectory(this, uiText("选择待转换数据集目录"));
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

    if (datasetConversionTargetDatasetNameEdit_
        && datasetConversionTargetDatasetNameEdit_->text().trimmed().isEmpty()) {
        datasetConversionTargetDatasetNameEdit_->setText(
            QStringLiteral("%1-%2").arg(QFileInfo(normalizedInputPath).completeBaseName(),
                comboCurrentDataOrText(datasetConversionTargetFormatCombo_)));
    }
}

void MainWindow::clearDatasetConversionErrors()
{
    setFieldErrorLabel(datasetConversionSourceErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionTargetErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionInputErrorLabel_, QString());
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
    if (datasetConversionTargetDatasetIdEdit_) datasetConversionTargetDatasetIdEdit_->setEnabled(!running);
    if (datasetConversionTargetDatasetNameEdit_) datasetConversionTargetDatasetNameEdit_->setEnabled(!running);
    if (datasetConversionBrowseInputButton_) {
        datasetConversionBrowseInputButton_->setEnabled(!running);
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
        datasetConversionStatusLabel_->setText(uiText("正在取消数据集转换。"));
    }
    appendDatasetConversionLog(uiText("正在取消数据集转换。"));
}

void MainWindow::startDatasetConversion()
{
    clearDatasetConversionErrors();
    if (worker_.isRunning() || currentProjectPath_.isEmpty() || !workspace_.isOpen()) {
        if (datasetConversionStatusLabel_) datasetConversionStatusLabel_->setText(
            uiText("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    const QString sourceFormat = comboCurrentDataOrText(datasetConversionSourceFormatCombo_);
    const QString targetFormat = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
    const QString sourcePath = normalizedDatasetConversionPath(
        datasetConversionInputEdit_ ? datasetConversionInputEdit_->text() : QString());
    const QString targetDatasetId = datasetConversionTargetDatasetIdEdit_
        ? datasetConversionTargetDatasetIdEdit_->text().trimmed() : QString();
    const QString targetDatasetName = datasetConversionTargetDatasetNameEdit_
        ? datasetConversionTargetDatasetNameEdit_->text().trimmed() : QString();
    aitrain::DatasetId parsedDatasetId;
    QString error;
    if (sourceFormat.isEmpty() || targetFormat.isEmpty() || sourcePath.isEmpty()
        || !QFileInfo::exists(sourcePath) || targetDatasetName.isEmpty()
        || !aitrain::DatasetId::parse(targetDatasetId, &parsedDatasetId, &error)) {
        if (datasetConversionStatusLabel_) datasetConversionStatusLabel_->setText(
            uiText("请填写存在的外部源、源/目标格式、有效目标 DatasetId 和审计名称。"));
        setFieldErrorLabel(datasetConversionInputErrorLabel_,
            QFileInfo::exists(sourcePath) ? QString() : uiText("外部源路径不存在。"));
        return;
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
        datasetConversionResultLabel_->setText(uiText("等待转换结果。"));
    }
    if (datasetConversionStatusLabel_) {
        datasetConversionStatusLabel_->setText(uiText("正在通过 Worker 转换数据集。"));
    }
    appendDatasetConversionLog(uiText("开始转换数据集。"));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("dataset_conversion");
    setDatasetConversionFormRunning(true);

    aitrain::worker_protocol::DatasetConversionCommand conversionCommand;
    conversionCommand.context.taskId = taskId;
    conversionCommand.context.projectRoot = currentProjectPath_;
    conversionCommand.sourcePath = sourcePath;
    conversionCommand.sourceFormat = sourceFormat;
    conversionCommand.targetFormat = targetFormat;
    conversionCommand.targetDatasetId = targetDatasetId;
    conversionCommand.targetDatasetName = targetDatasetName;
    conversionCommand.options = options;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{conversionCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        setDatasetConversionFormRunning(false);
        const QString message = uiText("无法启动数据集转换：%1").arg(error);
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(message);
        }
        appendDatasetConversionLog(message);
        QMessageBox::critical(this, uiText("数据集转换"), message);
        return;
    }

    workerPill_->setStatus(uiText("数据集转换中"), StatusPill::Tone::Info);
    statusBar()->showMessage(uiText("正在转换数据集"), 3000);
}

void MainWindow::runDataQualityWorkflow()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据质量报告"), uiText("Worker 正在执行任务，稍后再生成数据质量报告。"));
        return;
    }
    if (currentProjectPath_.isEmpty() || !workspace_.isOpen()) {
        QMessageBox::warning(this, uiText("数据质量报告"), uiText("请先创建或打开项目。"));
        return;
    }
    const QString datasetId = dataQualityDatasetIdEdit_ ? dataQualityDatasetIdEdit_->text().trimmed() : QString();
    const QString datasetVersionId = dataQualityDatasetVersionIdEdit_
        ? dataQualityDatasetVersionIdEdit_->text().trimmed() : QString();
    const QString snapshotId = dataQualitySnapshotIdEdit_ ? dataQualitySnapshotIdEdit_->text().trimmed() : QString();
    const QString snapshotArtifactId = dataQualitySnapshotArtifactIdEdit_
        ? dataQualitySnapshotArtifactIdEdit_->text().trimmed() : QString();
    aitrain::DatasetId parsedDatasetId;
    aitrain::DatasetVersionId parsedVersionId;
    aitrain::SnapshotId parsedSnapshotId;
    aitrain::ArtifactId parsedArtifactId;
    QString error;
    if (!aitrain::DatasetId::parse(datasetId, &parsedDatasetId, &error)
        || !aitrain::DatasetVersionId::parse(datasetVersionId, &parsedVersionId, &error)
        || !aitrain::SnapshotId::parse(snapshotId, &parsedSnapshotId, &error)
        || !aitrain::ArtifactId::parse(snapshotArtifactId, &parsedArtifactId, &error)) {
        QMessageBox::warning(this, uiText("数据质量报告"),
            uiText("请填写同一条快照记录的 DatasetId、DatasetVersionId、SnapshotId 和 ArtifactId。"));
        return;
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
    }
        validationSummaryLabel_->setText(uiText("Data Quality 正在校验已登记快照。"));
    validationOutput_->setPlainText(uiText("结果将以 ArtifactId 返回，文件请在“任务与产物”查看。"));

    QJsonObject options;
    options.insert(QStringLiteral("maxIssues"), 500);
    options.insert(QStringLiteral("minimumNormalizedArea"), 0.01);
    options.insert(QStringLiteral("minimumPolygonAreaPixels"), 16.0);
    options.insert(QStringLiteral("maxTextLength"), 25);
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("data_quality");
    aitrain::worker_protocol::DataQualityCommand qualityCommand;
    qualityCommand.context.taskId = taskId;
    qualityCommand.context.projectRoot = currentProjectPath_;
    qualityCommand.datasetId = datasetId;
    qualityCommand.datasetVersionId = datasetVersionId;
    qualityCommand.snapshotId = snapshotId;
    qualityCommand.snapshotArtifactId = snapshotArtifactId;
    qualityCommand.options = options;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{qualityCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        validationSummaryLabel_->setText(uiText("无法启动 Data Quality ：%1").arg(error));
        QMessageBox::critical(this, uiText("数据质量报告"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据质量报告生成中"), StatusPill::Tone::Info);
}

void MainWindow::runDatasetSplitWorkflow()
{
    if (worker_.isRunning() || currentProjectPath_.isEmpty() || !workspace_.isOpen()) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("Worker 正在执行任务，稍后再划分数据集。"));
        return;
    }
    const QString sourceDatasetId = splitSourceDatasetIdEdit_->text().trimmed();
    const QString sourceDatasetVersionId = splitSourceDatasetVersionIdEdit_->text().trimmed();
    const QString sourceSnapshotId = splitSourceSnapshotIdEdit_->text().trimmed();
    const QString sourceSnapshotArtifactId = splitSourceSnapshotArtifactIdEdit_->text().trimmed();
    const QString targetDatasetId = splitTargetDatasetIdEdit_->text().trimmed();
    const QString targetDatasetName = splitTargetDatasetNameEdit_->text().trimmed();
    aitrain::DatasetId parsedSourceDatasetId;
    aitrain::DatasetVersionId parsedSourceVersionId;
    aitrain::SnapshotId parsedSourceSnapshotId;
    aitrain::ArtifactId parsedSourceArtifactId;
    aitrain::DatasetId parsedTargetDatasetId;
    QString error;
    if (!aitrain::DatasetId::parse(sourceDatasetId, &parsedSourceDatasetId, &error)
        || !aitrain::DatasetVersionId::parse(sourceDatasetVersionId, &parsedSourceVersionId, &error)
        || !aitrain::SnapshotId::parse(sourceSnapshotId, &parsedSourceSnapshotId, &error)
        || !aitrain::ArtifactId::parse(sourceSnapshotArtifactId, &parsedSourceArtifactId, &error)
        || !aitrain::DatasetId::parse(targetDatasetId, &parsedTargetDatasetId, &error)
        || targetDatasetName.isEmpty()) {
        QMessageBox::warning(this, uiText("数据集划分"),
            uiText("请填写同一源快照的四重 ID、有效目标 DatasetId 和审计名称。"));
        return;
    }

    QJsonObject options;
    options.insert(QStringLiteral("trainRatio"), splitTrainRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("valRatio"), splitValRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("testRatio"), splitTestRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("seed"), splitSeedEdit_->text().toInt());
    options.insert(QStringLiteral("maxIssues"), 200);
    options.insert(QStringLiteral("allowEmptyLabels"), false);

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("dataset_split");
    aitrain::worker_protocol::DatasetSplitCommand splitCommand;
    splitCommand.context.taskId = taskId;
    splitCommand.context.projectRoot = currentProjectPath_;
    splitCommand.sourceDatasetId = sourceDatasetId;
    splitCommand.sourceDatasetVersionId = sourceDatasetVersionId;
    splitCommand.sourceSnapshotId = sourceSnapshotId;
    splitCommand.sourceSnapshotArtifactId = sourceSnapshotArtifactId;
    splitCommand.targetDatasetId = targetDatasetId;
    splitCommand.targetDatasetName = targetDatasetName;
    splitCommand.options = options;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{splitCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        QMessageBox::critical(this, uiText("数据集划分"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集划分中"), StatusPill::Tone::Info);
    statusBar()->showMessage(uiText("正在划分数据集"), 3000);
}

void MainWindow::openDatasetQualityFixList()
{
    showPage(TaskQueuePage, uiText("任务与产物"));
    statusBar()->showMessage(uiText("请按 Repair ArtifactId 查看受控问题清单。"), 5000);
}

void MainWindow::openDatasetQualityReport()
{
    showPage(TaskQueuePage, uiText("任务与产物"));
    statusBar()->showMessage(uiText("请按 Quality Report ArtifactId 查看受控报告。"), 5000);
}

void MainWindow::createXAnyLabelingAnnotationSession()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("X-AnyLabeling 会话"), uiText("Worker 正在执行任务，稍后再准备标注会话。"));
        return;
    }

    if (!workspace_.isOpen() || currentProjectPath_.isEmpty()) {
        QMessageBox::information(this, uiText("X-AnyLabeling 会话"), uiText("请先打开项目。"));
        return;
    }
    bool accepted = false;
    const QString repairArtifactText = QInputDialog::getText(this, uiText(" 修复清单"),
        uiText("输入 Data Quality 产生的 Repair ArtifactId："), QLineEdit::Normal,
        QString(), &accepted).trimmed();
    aitrain::ArtifactId repairArtifactId;
    QString error;
    if (!accepted || !aitrain::ArtifactId::parse(repairArtifactText, &repairArtifactId, &error)) {
        if (accepted) QMessageBox::warning(this, uiText(" 修复清单"), uiText("Repair ArtifactId 无效。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    const QString defaultWorkingDirectory = QDir(currentProjectPath_).filePath(
        QStringLiteral("annotation-workspaces/%1").arg(taskId.toString()));
    const QString workingDirectory = QDir::fromNativeSeparators(QInputDialog::getText(this,
        uiText("标注工作目录"), uiText("输入新的空工作目录（不会写入 Artifact 或数据库）："),
        QLineEdit::Normal, QDir::toNativeSeparators(defaultWorkingDirectory), &accepted).trimmed());
    if (!accepted || workingDirectory.isEmpty()) return;

    QJsonObject toolSummary{{QStringLiteral("tool"), QStringLiteral("X-AnyLabeling")},
        {QStringLiteral("integration"), QStringLiteral("external_process")},
        {QStringLiteral("mode"), QStringLiteral("quality_fix")}};
    QJsonObject options{{QStringLiteral("launchAfterCreate"), true}};
    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("annotation_create");
    state_.dataset.annotationWorkingDirectory = workingDirectory;
    aitrain::worker_protocol::AnnotationSessionCreateCommand createCommand;
    createCommand.context.taskId = taskId;
    createCommand.context.projectRoot = currentProjectPath_;
    createCommand.repairManifestArtifactId = repairArtifactId.toString();
    createCommand.workingDirectory = workingDirectory;
    createCommand.toolSummary = toolSummary;
    createCommand.options = options;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{createCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        QMessageBox::critical(this, uiText("X-AnyLabeling 会话"), error);
        return;
    }

    workerPill_->setStatus(uiText("标注会话准备中"), StatusPill::Tone::Info);
    setDatasetRepairLoopRows(
        uiText("修复闭环：正在准备 X-AnyLabeling 会话。"),
        QVector<QStringList>{
            QStringList() << uiText("会话准备") << uiText("运行中") << uiText("Worker 正在校验 Repair Artifact 并准备受控副本。"),
            QStringList() << uiText("外部修复") << uiText("等待") << uiText("会话 Artifact 提交后会启动本地 X-AnyLabeling。"),
            QStringList() << uiText("同步复检") << uiText("等待") << uiText("标注完成后点击“同步标注会话”。")
        });
}

void MainWindow::syncXAnyLabelingAnnotationSession()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("X-AnyLabeling 同步"), uiText("Worker 正在执行任务，稍后再同步标注会话。"));
        return;
    }

    if (!workspace_.isOpen() || currentProjectPath_.isEmpty()) {
        QMessageBox::information(this, uiText("X-AnyLabeling 同步"), uiText("请先打开项目。"));
        return;
    }
    bool accepted = false;
    const QString sessionArtifactText = QInputDialog::getText(this, uiText(" 标注会话"),
        uiText("输入 Session ArtifactId："), QLineEdit::Normal,
        state_.dataset.latestAnnotationSessionArtifactId, &accepted).trimmed();
    aitrain::ArtifactId sessionArtifactId;
    QString error;
    if (!accepted || !aitrain::ArtifactId::parse(sessionArtifactText, &sessionArtifactId, &error)) {
        if (accepted) QMessageBox::warning(this, uiText(" 标注会话"), uiText("Session ArtifactId 无效。"));
        return;
    }
    const QString workingDirectory = QDir::fromNativeSeparators(QInputDialog::getText(this,
        uiText("标注工作目录"), uiText("输入该会话使用的外部工作目录："), QLineEdit::Normal,
        QDir::toNativeSeparators(state_.dataset.annotationWorkingDirectory), &accepted).trimmed());
    if (!accepted || workingDirectory.isEmpty()) return;

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    QJsonObject options{{QStringLiteral("validateInventory"), true},
        {QStringLiteral("validateHashes"), true}};
    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("annotation_sync");
    state_.dataset.latestAnnotationSessionArtifactId = sessionArtifactId.toString();
    state_.dataset.annotationWorkingDirectory = workingDirectory;
    aitrain::worker_protocol::AnnotationSessionSyncCommand syncCommand;
    syncCommand.context.taskId = taskId;
    syncCommand.context.projectRoot = currentProjectPath_;
    syncCommand.sessionArtifactId = sessionArtifactId.toString();
    syncCommand.workingDirectory = workingDirectory;
    syncCommand.options = options;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{syncCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        QMessageBox::critical(this, uiText("X-AnyLabeling 同步"), error);
        return;
    }

    workerPill_->setStatus(uiText("标注同步中"), StatusPill::Tone::Info);
    setDatasetRepairLoopRows(
        uiText("修复闭环：正在同步 X-AnyLabeling 标注会话。"),
        QVector<QStringList>{
            QStringList() << uiText("外部修复") << uiText("已返回") << uiText("Session ArtifactId：%1").arg(sessionArtifactId.toString()),
            QStringList() << uiText("同步") << uiText("运行中") << uiText("Worker 正在重验基线、编辑白名单、文件集合和哈希。"),
            QStringList() << uiText("复检") << uiText("等待") << uiText("同步成功后由 Query/Presenter 刷新新 Dataset Version。")
        });
}

void MainWindow::browseSampleReviewFile()
{
    bool accepted = false;
    const QString artifactText = QInputDialog::getText(
        this, uiText("选择复核 Artifact"), uiText("输入已提交的质量/复核 ArtifactId："),
        QLineEdit::Normal, state_.dataset.sampleReviewArtifactId, &accepted).trimmed();
    if (accepted && reviewSamplePathEdit_) {
        reviewSamplePathEdit_->setText(artifactText);
        loadSampleReviewFile();
    }
}

void MainWindow::loadSampleReviewFile()
{
    const QString artifactText = reviewSamplePathEdit_ ? reviewSamplePathEdit_->text().trimmed() : QString();
    aitrain::ArtifactId artifactId;
    QString error;
    if (!aitrain::ArtifactId::parse(artifactText, &artifactId, &error)) {
        QMessageBox::warning(this, uiText("样本复核"), uiText("请输入有效的已提交复核 ArtifactId。"));
        return;
    }
    if (!workspace_.isOpen()) {
        QMessageBox::warning(this, uiText("样本复核"), uiText("请先打开项目。"));
        return;
    }
    const QStringList candidates = {
        QStringLiteral("problem_samples.json"),
        QStringLiteral("quality_analysis.json"),
        QStringLiteral("xanylabeling_review_manifest.json"),
        QStringLiteral("repair_manifest.json"),
        QStringLiteral("quality_report.json")};
    aitrain::ArtifactFilePreview preview;
    QString selectedFile;
    for (const QString& candidate : candidates) {
        if (queryService_.artifactFilePreview(artifactId, candidate, &preview, 4 * 1024 * 1024, &error)) {
            selectedFile = candidate;
            break;
        }
    }
    if (selectedFile.isEmpty()) {
        QMessageBox::warning(this, uiText("样本复核"),
            uiText("Artifact 内没有可读取的质量/复核 JSON：%1").arg(error));
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(preview.content, &parseError);
    if (parseError.error != QJsonParseError::NoError || (!document.isObject() && !document.isArray())) {
        QMessageBox::critical(this, uiText("样本复核"), uiText("复核样本 JSON 解析失败：%1").arg(parseError.errorString()));
        return;
    }

    state_.dataset.sampleReviewArtifactId = artifactId.toString();
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
    refreshSampleReviewTable();
    statusBar()->showMessage(uiText("已加载 Artifact %1 的复核样本：%2 条")
        .arg(artifactId.toString(), QString::number(state_.dataset.sampleReviewSamples.size())), 4000);
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
        const ReviewSamplePathView paths = reviewSamplePathView(sample);
        sampleReviewTable_->setItem(row, 0, new QTableWidgetItem(sample.value(QStringLiteral("source")).toString()));
        sampleReviewTable_->setItem(row, 1, new QTableWidgetItem(sample.value(QStringLiteral("reason")).toString()));
        sampleReviewTable_->setItem(row, 2, new QTableWidgetItem(reviewClassText(sample)));
        sampleReviewTable_->setItem(row, 3, new QTableWidgetItem(reviewMetricText(sample)));
        auto* imageItem = new QTableWidgetItem(paths.imageRelativePath);
        imageItem->setToolTip(paths.imageRelativePath.isEmpty()
            ? uiText("未提供有效的 Snapshot 相对路径；外部或越界路径已隐藏。")
            : uiText("Snapshot 内相对路径：%1").arg(paths.imageRelativePath));
        sampleReviewTable_->setItem(row, 4, imageItem);
        auto* labelItem = new QTableWidgetItem(paths.labelRelativePath);
        labelItem->setToolTip(paths.labelRelativePath.isEmpty()
            ? uiText("未提供有效的 Snapshot 相对路径；外部或越界路径已隐藏。")
            : uiText("Snapshot 内相对路径：%1").arg(paths.labelRelativePath));
        sampleReviewTable_->setItem(row, 5, labelItem);
        sampleReviewTable_->setItem(row, 6, new QTableWidgetItem(sampleTextField(sample, QStringList()
            << QStringLiteral("message")
            << QStringLiteral("note")
            << QStringLiteral("description")
            << QStringLiteral("groundTruth")
            << QStringLiteral("prediction"))));
    }
    if (sampleReviewSummaryLabel_) {
        sampleReviewSummaryLabel_->setText(uiText("复核样本：显示 %1 / 总计 %2；页面只读")
            .arg(rows.size())
            .arg(state_.dataset.sampleReviewSamples.size()));
    }
}

void MainWindow::openSelectedReviewSample()
{
    if (!sampleReviewTable_ || sampleReviewTable_->currentRow() < 0) {
        QMessageBox::information(this, uiText("样本复核"), uiText("请先选择一条复核样本。"));
        return;
    }
    const int row = sampleReviewTable_->currentRow();
    const QString imagePath = sampleReviewTable_->item(row, 4)
        ? sampleReviewTable_->item(row, 4)->text() : QString();
    const QString labelPath = sampleReviewTable_->item(row, 5)
        ? sampleReviewTable_->item(row, 5)->text() : QString();
    if (state_.dataset.sampleReviewArtifactId.isEmpty()) {
        QMessageBox::warning(this, uiText("样本复核"), uiText("当前没有已加载的复核 Artifact。"));
        return;
    }
    QMessageBox::information(this, uiText("样本复核"),
        uiText("当前复核记录属于 Artifact %1。\n图片相对路径：%2\n标签相对路径：%3\n\n原始文件不会由 GUI 直接打开；请在任务与产物页按 ArtifactId 进行受控预览。")
            .arg(state_.dataset.sampleReviewArtifactId, imagePath, labelPath));
}

void MainWindow::runDatasetSnapshotImportWorkflow()
{
    if (worker_.isRunning() || currentProjectPath_.isEmpty() || !workspace_.isOpen()) {
        QMessageBox::warning(this, uiText("数据集快照"), uiText("Worker 正在执行任务，稍后再创建数据集快照。"));
        return;
    }
    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    const QString targetDatasetId = datasetSnapshotTargetDatasetIdEdit_
        ? datasetSnapshotTargetDatasetIdEdit_->text().trimmed() : QString();
    const QString targetDatasetName = datasetSnapshotTargetDatasetNameEdit_
        ? datasetSnapshotTargetDatasetNameEdit_->text().trimmed() : QString();
    aitrain::DatasetId parsedDatasetId;
    QString error;
    if (path.isEmpty() || format.isEmpty() || !QFileInfo::exists(path)
        || targetDatasetName.isEmpty()
        || !aitrain::DatasetId::parse(targetDatasetId, &parsedDatasetId, &error)) {
        QMessageBox::warning(this, uiText("数据集快照"),
            uiText("请填写存在的外部源、格式、有效目标 DatasetId 和审计名称。"));
        return;
    }

    QJsonObject options;
    options.insert(QStringLiteral("maxFiles"), 20000);
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("dataset_snapshot_import");
    aitrain::worker_protocol::DatasetSnapshotImportCommand snapshotCommand;
    snapshotCommand.context.taskId = taskId;
    snapshotCommand.context.projectRoot = currentProjectPath_;
    snapshotCommand.sourcePath = path;
    snapshotCommand.sourceFormat = format;
    snapshotCommand.targetDatasetId = targetDatasetId;
    snapshotCommand.targetDatasetName = targetDatasetName;
    snapshotCommand.options = options;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{snapshotCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        QMessageBox::critical(this, uiText("数据集快照"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集快照创建中"), StatusPill::Tone::Info);
}
