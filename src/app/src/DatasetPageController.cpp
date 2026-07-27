#include "DatasetPageController.h"

#include "DatasetPage.h"
#include "DatasetConversionUiModel.h"
#include "DatasetCatalogPresenter.h"
#include "MainWindowSupport.h"
#include "TaskRuntimeController.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/workflow/ProjectQueryService.h"

#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <QFileDialog>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPointer>
#include <QPushButton>
#include <QSignalBlocker>
#include <QTableWidget>
#include <QTableWidgetItem>

using namespace aitrain_app;

namespace {

QString sampleTextField(
    const QJsonObject& sample, const QStringList& keys)
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
            const QString compact = QString::fromUtf8(
                QJsonDocument(QJsonArray{value})
                    .toJson(QJsonDocument::Compact));
            return compact.mid(1, qMax(0, compact.size() - 2));
        }
    }
    return {};
}

QJsonObject normalizedReviewSample(
    QJsonObject sample, const QString& source)
{
    if (sample.value(QStringLiteral("source")).toString().isEmpty()) {
        sample.insert(QStringLiteral("source"), source);
    }
    if (sample.value(QStringLiteral("reason")).toString().isEmpty()) {
        const QString reason = sampleTextField(sample, {
            QStringLiteral("code"), QStringLiteral("errorType"),
            QStringLiteral("type"), QStringLiteral("category")});
        if (!reason.isEmpty()) {
            sample.insert(QStringLiteral("reason"), reason);
        }
    }
    return sample;
}

void appendReviewSamples(
    QJsonArray* target, const QJsonArray& source, const QString& sourceName)
{
    for (const QJsonValue& value : source) {
        if (value.isObject()) {
            target->append(
                normalizedReviewSample(value.toObject(), sourceName));
        }
    }
}

QJsonArray extractReviewSamples(const QJsonDocument& document)
{
    QJsonArray samples;
    if (document.isArray()) {
        appendReviewSamples(&samples, document.array(),
            QStringLiteral("array"));
        return samples;
    }
    const QJsonObject root = document.object();
    const QList<QPair<QString, QString>> keys = {
        {QStringLiteral("problemSamples"), QStringLiteral("problem_samples")},
        {QStringLiteral("errorSamples"), QStringLiteral("error_samples")},
        {QStringLiteral("lowConfidenceSamples"), QStringLiteral("low_confidence")},
        {QStringLiteral("samples"), QStringLiteral("samples")},
        {QStringLiteral("issues"), QStringLiteral("quality_issues")},
        {QStringLiteral("actions"), QStringLiteral("repair_actions")},
        {QStringLiteral("reworkSamples"), QStringLiteral("rework_samples")},
        {QStringLiteral("items"), QStringLiteral("items")}};
    for (const auto& key : keys) {
        appendReviewSamples(
            &samples, root.value(key.first).toArray(), key.second);
    }
    const QJsonObject payload = root.value(QStringLiteral("payload")).toObject();
    for (const auto& key : keys) {
        appendReviewSamples(
            &samples, payload.value(key.first).toArray(), key.second);
    }
    return samples;
}

QString reviewMetricText(const QJsonObject& sample)
{
    QStringList parts;
    for (const QString& key : {
             QStringLiteral("confidence"), QStringLiteral("matchedIou"),
             QStringLiteral("matchedMaskIoU"),
             QStringLiteral("editDistance"), QStringLiteral("cer")}) {
        if (sample.contains(key)) {
            parts.append(QStringLiteral("%1=%2")
                .arg(key, sampleTextField(sample, {key})));
        }
    }
    const double confidence = sample.value(QStringLiteral("prediction"))
        .toObject().value(QStringLiteral("confidence")).toDouble(-1.0);
    if (confidence >= 0.0) {
        parts.append(
            QStringLiteral("pred_conf=%1").arg(confidence, 0, 'f', 4));
    }
    return parts.join(QStringLiteral(" | "));
}

QString reviewClassText(const QJsonObject& sample)
{
    const QString direct = sampleTextField(sample, {
        QStringLiteral("className"), QStringLiteral("class"),
        QStringLiteral("category"), QStringLiteral("label")});
    if (!direct.isEmpty()) return direct;
    for (const QString& key : {
             QStringLiteral("prediction"), QStringLiteral("groundTruth")}) {
        const QJsonObject object = sample.value(key).toObject();
        const QJsonObject box = object.value(QStringLiteral("box")).toObject();
        if (box.contains(QStringLiteral("classId"))) {
            return QStringLiteral("class_%1")
                .arg(box.value(QStringLiteral("classId")).toInt());
        }
        if (object.contains(QStringLiteral("classId"))) {
            return QStringLiteral("class_%1")
                .arg(object.value(QStringLiteral("classId")).toInt());
        }
    }
    return {};
}

} // namespace

DatasetPageController::DatasetPageController(
    const aitrain::ProjectQueryService* queryService,
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , taskRuntime_(taskRuntime)
    , queryService_(queryService)
{
    catalogPresenter_ = new DatasetCatalogPresenter(queryService, this);
}

void DatasetPageController::attach(DatasetWorkspacePage* page)
{
    page_ = page;
    connect(page_->datasetListTable, &QTableWidget::itemSelectionChanged,
        this, [this]() {
            if (!page_ || page_->datasetListTable->selectedItems().isEmpty()) {
                return;
            }
            const int row =
                page_->datasetListTable->selectedItems().first()->row();
            const auto value = [this, row](int column, int role) {
                QTableWidgetItem* item =
                    page_->datasetListTable->item(row, column);
                return item ? item->data(role).toString() : QString();
            };
            const QString datasetId = value(0, Qt::UserRole);
            if (datasetId.isEmpty()) return;
            const QString format = value(1, Qt::UserRole);
            const QString snapshotId = value(2, Qt::UserRole);
            const QString artifactId = value(4, Qt::UserRole);
            const QString versionId = value(4, Qt::UserRole + 1);
            page_->datasetPathEdit->clear();
            const int formatIndex =
                page_->datasetFormatCombo->findData(format);
            if (formatIndex >= 0) {
                page_->datasetFormatCombo->setCurrentIndex(formatIndex);
            }
            state_.currentPath.clear();
            state_.currentFormat = format;
            state_.currentDatasetId = datasetId;
            state_.currentDatasetVersionId = versionId;
            state_.currentSnapshotId = snapshotId;
            state_.currentSnapshotArtifactId = artifactId;
            state_.currentValid = !versionId.isEmpty()
                && !snapshotId.isEmpty() && !artifactId.isEmpty();
            for (QLineEdit* edit : {
                     page_->dataQualityDatasetIdEdit,
                     page_->splitSourceDatasetIdEdit}) {
                edit->setText(datasetId);
            }
            for (QLineEdit* edit : {
                     page_->dataQualityDatasetVersionIdEdit,
                     page_->splitSourceDatasetVersionIdEdit}) {
                edit->setText(versionId);
            }
            for (QLineEdit* edit : {
                     page_->dataQualitySnapshotIdEdit,
                     page_->splitSourceSnapshotIdEdit}) {
                edit->setText(snapshotId);
            }
            for (QLineEdit* edit : {
                     page_->dataQualitySnapshotArtifactIdEdit,
                     page_->splitSourceSnapshotArtifactIdEdit}) {
                edit->setText(artifactId);
            }
            emit selectionChanged();
        });
    connect(page_->datasetFormatCombo,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, [this]() {
            state_.currentFormat =
                comboCurrentDataOrText(page_->datasetFormatCombo);
            state_.currentDatasetId.clear();
            state_.currentDatasetVersionId.clear();
            state_.currentSnapshotId.clear();
            state_.currentSnapshotArtifactId.clear();
            state_.currentValid = false;
            refreshConversionDefaults();
            emit selectionChanged();
        });
}

DatasetWorkbenchState& DatasetPageController::state()
{
    return state_;
}

const DatasetWorkbenchState& DatasetPageController::state() const
{
    return state_;
}

void DatasetPageController::reset()
{
    state_ = DatasetWorkbenchState();
    invalidateAsyncPreviews();
}

void DatasetPageController::invalidateAsyncPreviews()
{
    ++sampleReviewGeneration_;
    if (sampleReviewGeneration_ == 0) ++sampleReviewGeneration_;
    ++formatProbeGeneration_;
    if (formatProbeGeneration_ == 0) ++formatProbeGeneration_;
}

void DatasetPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
}

void DatasetPageController::setWorkerExecutable(const QString& executable)
{
    workerExecutable_ = executable;
}

void DatasetPageController::setConversionRunning(bool running)
{
    if (!page_) return;
    for (QWidget* control : {
             static_cast<QWidget*>(page_->datasetConversionSourceFormatCombo),
             static_cast<QWidget*>(page_->datasetConversionTargetFormatCombo),
             static_cast<QWidget*>(page_->datasetConversionInputEdit),
             static_cast<QWidget*>(page_->datasetConversionTargetDatasetIdEdit),
             static_cast<QWidget*>(page_->datasetConversionTargetDatasetNameEdit),
             static_cast<QWidget*>(page_->datasetConversionBrowseInputButton)}) {
        if (control) control->setEnabled(!running);
    }
    if (page_->datasetConversionStartButton) {
        page_->datasetConversionStartButton->setEnabled(!running);
    }
    if (page_->datasetConversionCancelButton) {
        page_->datasetConversionCancelButton->setEnabled(running);
    }
}

void DatasetPageController::appendConversionLog(const QString& text)
{
    if (!page_ || !page_->datasetConversionLog || text.isEmpty()) return;
    if (page_->datasetConversionLog->toPlainText().trimmed()
        == QStringLiteral("等待转换。")) {
        page_->datasetConversionLog->clear();
    }
    page_->datasetConversionLog->appendPlainText(text);
}

void DatasetPageController::setConversionError(const QString& text)
{
    if (!page_ || !page_->datasetConversionInputErrorLabel) return;
    page_->datasetConversionInputErrorLabel->setText(text);
    page_->datasetConversionInputErrorLabel->setVisible(!text.isEmpty());
}

void DatasetPageController::cancelConversion()
{
    if (!taskRuntime_->isRunning()) return;
    taskRuntime_->cancel();
    if (page_ && page_->datasetConversionStatusLabel) {
        page_->datasetConversionStatusLabel->setText(
            tr("正在取消数据集转换。"));
    }
    appendConversionLog(tr("正在取消数据集转换。"));
}

void DatasetPageController::startConversion()
{
    if (!page_) return;
    setConversionError({});
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        page_->datasetConversionStatusLabel->setText(
            tr("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    const QString sourceFormat =
        comboCurrentDataOrText(page_->datasetConversionSourceFormatCombo);
    const QString targetFormat =
        comboCurrentDataOrText(page_->datasetConversionTargetFormatCombo);
    const QString sourcePath = normalizedDatasetConversionPath(
        page_->datasetConversionInputEdit
            ? page_->datasetConversionInputEdit->text() : QString());
    const QString targetDatasetId =
        page_->datasetConversionTargetDatasetIdEdit
            ? page_->datasetConversionTargetDatasetIdEdit->text().trimmed()
            : QString();
    const QString targetDatasetName =
        page_->datasetConversionTargetDatasetNameEdit
            ? page_->datasetConversionTargetDatasetNameEdit->text().trimmed()
            : QString();
    aitrain::DatasetId parsedDatasetId;
    QString error;
    if (sourceFormat.isEmpty() || targetFormat.isEmpty()
        || sourcePath.isEmpty() || !QFileInfo::exists(sourcePath)
        || targetDatasetName.isEmpty()
        || !aitrain::DatasetId::parse(
            targetDatasetId, &parsedDatasetId, &error)) {
        page_->datasetConversionStatusLabel->setText(
            tr("请填写存在的外部源、源/目标格式、有效目标 DatasetId 和审计名称。"));
        setConversionError(QFileInfo::exists(sourcePath)
            ? QString() : tr("外部源路径不存在。"));
        return;
    }
    page_->datasetConversionProgressBar->setValue(0);
    page_->datasetConversionLog->clear();
    page_->datasetConversionResultLabel->setText(tr("等待转换结果。"));
    page_->datasetConversionStatusLabel->setText(
        tr("正在通过 Worker 转换数据集。"));
    appendConversionLog(tr("开始转换数据集。"));
    setConversionRunning(true);

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::DatasetConversionCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.sourcePath = sourcePath;
    command.sourceFormat = sourceFormat;
    command.targetFormat = targetFormat;
    command.targetDatasetId = targetDatasetId;
    command.targetDatasetName = targetDatasetName;
    command.options = QJsonObject{
        {QStringLiteral("copyImages"), true},
        {QStringLiteral("maxIssues"), 200}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        setConversionRunning(false);
        const QString message =
            tr("无法启动数据集转换：%1").arg(error);
        page_->datasetConversionStatusLabel->setText(message);
        appendConversionLog(message);
        QMessageBox::critical(page_, tr("数据集转换"), message);
        return;
    }
    emit taskStarted(taskId.toString(), QStringLiteral("dataset_conversion"));
    emit statusChanged(tr("数据集转换中"));
}

void DatasetPageController::runDataQuality()
{
    if (!page_) return;
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(page_, tr("数据质量报告"),
            tr("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    const QString datasetId = page_->dataQualityDatasetIdEdit->text().trimmed();
    const QString versionId =
        page_->dataQualityDatasetVersionIdEdit->text().trimmed();
    const QString snapshotId =
        page_->dataQualitySnapshotIdEdit->text().trimmed();
    const QString artifactId =
        page_->dataQualitySnapshotArtifactIdEdit->text().trimmed();
    aitrain::DatasetId parsedDatasetId;
    aitrain::DatasetVersionId parsedVersionId;
    aitrain::SnapshotId parsedSnapshotId;
    aitrain::ArtifactId parsedArtifactId;
    QString error;
    if (!aitrain::DatasetId::parse(datasetId, &parsedDatasetId, &error)
        || !aitrain::DatasetVersionId::parse(
            versionId, &parsedVersionId, &error)
        || !aitrain::SnapshotId::parse(
            snapshotId, &parsedSnapshotId, &error)
        || !aitrain::ArtifactId::parse(
            artifactId, &parsedArtifactId, &error)) {
        QMessageBox::warning(page_, tr("数据质量报告"),
            tr("请填写同一条快照记录的 DatasetId、DatasetVersionId、SnapshotId 和 ArtifactId。"));
        return;
    }
    page_->validationIssuesTable->setRowCount(0);
    page_->validationSummaryLabel->setText(
        tr("Data Quality 正在校验已登记快照。"));
    page_->validationOutput->setPlainText(
        tr("结果将以 ArtifactId 返回，文件请在“任务与产物”查看。"));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::DataQualityCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.datasetId = datasetId;
    command.datasetVersionId = versionId;
    command.snapshotId = snapshotId;
    command.snapshotArtifactId = artifactId;
    command.options = QJsonObject{
        {QStringLiteral("maxIssues"), 500},
        {QStringLiteral("minimumNormalizedArea"), 0.01},
        {QStringLiteral("minimumPolygonAreaPixels"), 16.0},
        {QStringLiteral("maxTextLength"), 25}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        page_->validationSummaryLabel->setText(
            tr("无法启动 Data Quality：%1").arg(error));
        QMessageBox::critical(page_, tr("数据质量报告"), error);
        return;
    }
    emit taskStarted(taskId.toString(), QStringLiteral("data_quality"));
    emit statusChanged(tr("数据质量报告生成中"));
}

void DatasetPageController::runSplit()
{
    if (!page_) return;
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(page_, tr("数据集划分"),
            tr("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    const QString datasetId =
        page_->splitSourceDatasetIdEdit->text().trimmed();
    const QString versionId =
        page_->splitSourceDatasetVersionIdEdit->text().trimmed();
    const QString snapshotId =
        page_->splitSourceSnapshotIdEdit->text().trimmed();
    const QString artifactId =
        page_->splitSourceSnapshotArtifactIdEdit->text().trimmed();
    const QString targetId =
        page_->splitTargetDatasetIdEdit->text().trimmed();
    const QString targetName =
        page_->splitTargetDatasetNameEdit->text().trimmed();
    aitrain::DatasetId parsedDatasetId;
    aitrain::DatasetVersionId parsedVersionId;
    aitrain::SnapshotId parsedSnapshotId;
    aitrain::ArtifactId parsedArtifactId;
    aitrain::DatasetId parsedTargetId;
    QString error;
    if (!aitrain::DatasetId::parse(datasetId, &parsedDatasetId, &error)
        || !aitrain::DatasetVersionId::parse(
            versionId, &parsedVersionId, &error)
        || !aitrain::SnapshotId::parse(
            snapshotId, &parsedSnapshotId, &error)
        || !aitrain::ArtifactId::parse(
            artifactId, &parsedArtifactId, &error)
        || !aitrain::DatasetId::parse(targetId, &parsedTargetId, &error)
        || targetName.isEmpty()) {
        QMessageBox::warning(page_, tr("数据集划分"),
            tr("请填写同一源快照的四重 ID、有效目标 DatasetId 和审计名称。"));
        return;
    }
    aitrain::worker_protocol::DatasetSplitCommand command;
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.sourceDatasetId = datasetId;
    command.sourceDatasetVersionId = versionId;
    command.sourceSnapshotId = snapshotId;
    command.sourceSnapshotArtifactId = artifactId;
    command.targetDatasetId = targetId;
    command.targetDatasetName = targetName;
    command.options = QJsonObject{
        {QStringLiteral("trainRatio"),
            page_->splitTrainRatioEdit->text().toDouble()},
        {QStringLiteral("valRatio"),
            page_->splitValRatioEdit->text().toDouble()},
        {QStringLiteral("testRatio"),
            page_->splitTestRatioEdit->text().toDouble()},
        {QStringLiteral("seed"), page_->splitSeedEdit->text().toInt()},
        {QStringLiteral("maxIssues"), 200},
        {QStringLiteral("allowEmptyLabels"), false}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("数据集划分"), error);
        return;
    }
    emit taskStarted(taskId.toString(), QStringLiteral("dataset_split"));
    emit statusChanged(tr("数据集划分中"));
}

void DatasetPageController::runSnapshotImport()
{
    if (!page_) return;
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(page_, tr("数据集快照"),
            tr("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    const QString format =
        comboCurrentDataOrText(page_->datasetFormatCombo);
    const QString path = QDir::fromNativeSeparators(
        page_->datasetPathEdit->text().trimmed());
    const QString targetId =
        page_->datasetSnapshotTargetDatasetIdEdit->text().trimmed();
    const QString targetName =
        page_->datasetSnapshotTargetDatasetNameEdit->text().trimmed();
    aitrain::DatasetId parsedTargetId;
    QString error;
    if (path.isEmpty() || format.isEmpty() || !QFileInfo::exists(path)
        || targetName.isEmpty()
        || !aitrain::DatasetId::parse(
            targetId, &parsedTargetId, &error)) {
        QMessageBox::warning(page_, tr("数据集快照"),
            tr("请填写存在的外部源、格式、有效目标 DatasetId 和审计名称。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::DatasetSnapshotImportCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.sourcePath = path;
    command.sourceFormat = format;
    command.targetDatasetId = targetId;
    command.targetDatasetName = targetName;
    command.options = QJsonObject{{QStringLiteral("maxFiles"), 20000}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("数据集快照"), error);
        return;
    }
    emit taskStarted(
        taskId.toString(), QStringLiteral("dataset_snapshot_import"));
    emit statusChanged(tr("数据集快照创建中"));
}

void DatasetPageController::refreshCatalog()
{
    catalogPresenter_->refresh({50, {}});
    if (!page_ || !page_->datasetListTable) return;
    QTableWidget* table = page_->datasetListTable;
    table->setRowCount(0);
    const QVector<DatasetCatalogListItem>& datasets =
        catalogPresenter_->datasets();
    if (datasets.isEmpty()) {
        table->insertRow(0);
        table->setItem(
            0, 0, new QTableWidgetItem(tr("暂无数据集记录")));
        for (int column = 1; column < table->columnCount(); ++column) {
            table->setItem(0, column, new QTableWidgetItem(QString()));
        }
        return;
    }
    for (const DatasetCatalogListItem& dataset : datasets) {
        const int row = table->rowCount();
        table->insertRow(row);
        auto* name = new QTableWidgetItem(dataset.datasetId.left(12));
        name->setData(Qt::UserRole, dataset.datasetId);
        table->setItem(row, 0, name);
        auto* format =
            new QTableWidgetItem(datasetFormatLabel(dataset.datasetFormat));
        format->setData(Qt::UserRole, dataset.datasetFormat);
        table->setItem(row, 1, format);
        auto* status = new QTableWidgetItem(
            dataset.latestSnapshotId.isEmpty()
                ? tr("尚无快照") : tr("已提交快照"));
        status->setData(Qt::UserRole, dataset.latestSnapshotId);
        table->setItem(row, 2, status);
        table->setItem(row, 3,
            new QTableWidgetItem(QString::number(dataset.latestFileCount)));
        auto* identity =
            new QTableWidgetItem(dataset.latestSnapshotId);
        identity->setData(Qt::UserRole, dataset.latestArtifactId);
        identity->setData(Qt::UserRole + 1, dataset.latestVersionId);
        identity->setToolTip(
            tr("Version %1\nArtifact %2\nRoot hash %3")
                .arg(dataset.latestVersionId, dataset.latestArtifactId,
                    dataset.latestRootHash));
        table->setItem(row, 4, identity);
    }
}

void DatasetPageController::browseDataset()
{
    if (!page_) return;
    const QString directory = QFileDialog::getExistingDirectory(
        page_, tr("选择数据集目录"));
    if (directory.isEmpty()) return;
    page_->datasetPathEdit->setText(QDir::toNativeSeparators(directory));
    state_.currentPath = directory;
    state_.currentFormat =
        comboCurrentDataOrText(page_->datasetFormatCombo);
    state_.currentDatasetId.clear();
    state_.currentDatasetVersionId.clear();
    state_.currentSnapshotId.clear();
    state_.currentSnapshotArtifactId.clear();
    state_.currentValid = false;
    for (QLineEdit* edit : {
             page_->dataQualityDatasetIdEdit,
             page_->dataQualityDatasetVersionIdEdit,
             page_->dataQualitySnapshotIdEdit,
             page_->dataQualitySnapshotArtifactIdEdit,
             page_->splitSourceDatasetIdEdit,
             page_->splitSourceDatasetVersionIdEdit,
             page_->splitSourceSnapshotIdEdit,
             page_->splitSourceSnapshotArtifactIdEdit}) {
        edit->clear();
    }
    startFormatProbe(directory, false);
    refreshConversionDefaults();
    emit selectionChanged();
}

void DatasetPageController::updateConversionTargets()
{
    if (!page_) return;
    const QString source =
        comboCurrentDataOrText(page_->datasetConversionSourceFormatCombo);
    const QString previous =
        comboCurrentDataOrText(page_->datasetConversionTargetFormatCombo);
    const QSignalBlocker blocker(
        page_->datasetConversionTargetFormatCombo);
    page_->datasetConversionTargetFormatCombo->clear();
    for (const QString& target : supportedDatasetConversionTargets(source)) {
        addComboItem(page_->datasetConversionTargetFormatCombo,
            datasetConversionFormatLabel(target), target);
    }
    const int index =
        page_->datasetConversionTargetFormatCombo->findData(previous);
    if (index >= 0) {
        page_->datasetConversionTargetFormatCombo->setCurrentIndex(index);
    } else if (page_->datasetConversionTargetFormatCombo->count() > 0) {
        page_->datasetConversionTargetFormatCombo->setCurrentIndex(0);
    }
}

void DatasetPageController::refreshConversionDefaults()
{
    if (!page_ || !page_->datasetConversionInputEdit) return;
    QString inputPath =
        QDir::fromNativeSeparators(page_->datasetPathEdit->text().trimmed());
    if (inputPath.isEmpty()) inputPath = state_.currentPath;
    if (!inputPath.isEmpty()) {
        page_->datasetConversionInputEdit->setText(
            QDir::toNativeSeparators(inputPath));
    }
    if (!state_.currentFormat.isEmpty()
        && page_->datasetConversionSourceFormatCombo->findData(
               state_.currentFormat) >= 0) {
        setComboCurrentData(
            page_->datasetConversionSourceFormatCombo, state_.currentFormat);
    } else {
        updateConversionTargets();
    }
    if (page_->datasetConversionTargetDatasetNameEdit->text()
            .trimmed().isEmpty() && !inputPath.isEmpty()) {
        page_->datasetConversionTargetDatasetNameEdit->setText(
            QStringLiteral("%1-%2")
                .arg(QFileInfo(inputPath).completeBaseName(),
                    comboCurrentDataOrText(
                        page_->datasetConversionTargetFormatCombo)));
    }
}

void DatasetPageController::browseConversionInput()
{
    if (!page_) return;
    const QString source =
        comboCurrentDataOrText(page_->datasetConversionSourceFormatCombo);
    const bool coco = source == QStringLiteral("coco_json");
    const QString selected = coco
        ? QFileDialog::getOpenFileName(page_, tr("选择 COCO JSON 标注文件"),
              QString(), tr("JSON 文件 (*.json);;所有文件 (*)"))
        : QFileDialog::getExistingDirectory(
              page_, tr("选择待转换数据集目录"));
    if (selected.isEmpty()) return;
    const QString path = QDir::fromNativeSeparators(selected);
    page_->datasetConversionInputEdit->setText(
        QDir::toNativeSeparators(path));
    if (coco) {
        page_->datasetConversionProbeStatusLabel->setText(
            tr("COCO JSON 输入无需目录格式探测。"));
    } else {
        startFormatProbe(path, true);
    }
    if (page_->datasetConversionTargetDatasetNameEdit->text()
            .trimmed().isEmpty()) {
        page_->datasetConversionTargetDatasetNameEdit->setText(
            QStringLiteral("%1-%2")
                .arg(QFileInfo(path).completeBaseName(),
                    comboCurrentDataOrText(
                        page_->datasetConversionTargetFormatCombo)));
    }
}

void DatasetPageController::startFormatProbe(
    const QString& path, bool conversionSource)
{
    const QString normalized = QDir::fromNativeSeparators(path.trimmed());
    if (normalized.isEmpty()) return;
    const quint64 generation = ++formatProbeGeneration_;
    QLabel* status = conversionSource
        ? page_->datasetConversionProbeStatusLabel
        : page_->datasetProbeStatusLabel;
    status->setText(
        tr("正在后台探测数据集格式，完整校验仍由 Worker 执行。"));
    status->setVisible(true);
    detectDatasetFormatAsync(this, normalized,
        [this, normalized, conversionSource, generation](
            const QString& format) {
            applyFormatProbe(
                normalized, format, conversionSource, generation);
        });
}

void DatasetPageController::applyFormatProbe(
    const QString& path, const QString& format,
    bool conversionSource, quint64 generation)
{
    if (!page_ || generation != formatProbeGeneration_) return;
    QLabel* status = conversionSource
        ? page_->datasetConversionProbeStatusLabel
        : page_->datasetProbeStatusLabel;
    const QString current = conversionSource
        ? page_->datasetConversionInputEdit->text().trimmed()
        : page_->datasetPathEdit->text().trimmed();
    if (QDir::cleanPath(QDir::fromNativeSeparators(current))
        != QDir::cleanPath(QDir::fromNativeSeparators(path))) {
        return;
    }
    if (conversionSource) {
        const bool supported = !format.isEmpty()
            && supportedDatasetConversionSourceFormats().contains(format);
        if (supported) {
            setComboCurrentData(
                page_->datasetConversionSourceFormatCombo, format);
            status->setText(
                tr("后台探测完成：%1。Worker 将在转换前重新校验。")
                    .arg(datasetConversionFormatLabel(format)));
        } else {
            status->setText(
                tr("后台探测未识别格式；请手动选择，Worker 将执行完整校验。"));
        }
        return;
    }
    const int index = page_->datasetFormatCombo->findData(format);
    if (index >= 0) {
        page_->datasetFormatCombo->setCurrentIndex(index);
        state_.currentFormat = format;
        status->setText(
            tr("后台探测完成：%1。Worker 将在校验前重新验证。")
                .arg(datasetFormatLabel(format)));
        emit selectionChanged();
    } else {
        status->setText(
            tr("后台探测未识别格式；请手动选择，Worker 将执行完整校验。"));
    }
}

void DatasetPageController::createAnnotationSession()
{
    if (!page_) return;
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(page_, tr("X-AnyLabeling 会话"),
            tr("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    bool accepted = false;
    const QString repairText = QInputDialog::getText(page_,
        tr("修复清单"), tr("输入 Data Quality 产生的 Repair ArtifactId："),
        QLineEdit::Normal, QString(), &accepted).trimmed();
    aitrain::ArtifactId repairId;
    QString error;
    if (!accepted
        || !aitrain::ArtifactId::parse(repairText, &repairId, &error)) {
        if (accepted) {
            QMessageBox::warning(
                page_, tr("修复清单"), tr("Repair ArtifactId 无效。"));
        }
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    const QString defaultDirectory = QDir(projectRoot_).filePath(
        QStringLiteral("annotation-workspaces/%1").arg(taskId.toString()));
    const QString directory = QDir::fromNativeSeparators(
        QInputDialog::getText(page_, tr("标注工作目录"),
            tr("输入新的空工作目录（不会写入 Artifact 或数据库）："),
            QLineEdit::Normal, QDir::toNativeSeparators(defaultDirectory),
            &accepted).trimmed());
    if (!accepted || directory.isEmpty()) return;
    aitrain::worker_protocol::AnnotationSessionCreateCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.repairManifestArtifactId = repairId.toString();
    command.workingDirectory = directory;
    command.toolSummary = QJsonObject{
        {QStringLiteral("tool"), QStringLiteral("X-AnyLabeling")},
        {QStringLiteral("integration"), QStringLiteral("external_process")},
        {QStringLiteral("mode"), QStringLiteral("quality_fix")}};
    command.options =
        QJsonObject{{QStringLiteral("launchAfterCreate"), true}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("X-AnyLabeling 会话"), error);
        return;
    }
    state_.annotationWorkingDirectory = directory;
    emit taskStarted(taskId.toString(), QStringLiteral("annotation_create"));
    emit statusChanged(tr("标注会话准备中"));
    emit repairLoopChanged(
        tr("修复闭环：正在准备 X-AnyLabeling 会话。"),
        QVector<QStringList>{
            {tr("会话准备"), tr("运行中"),
                tr("Worker 正在校验 Repair Artifact 并准备受控副本。")},
            {tr("外部修复"), tr("等待"),
                tr("会话 Artifact 提交后会启动本地 X-AnyLabeling。")},
            {tr("同步复检"), tr("等待"),
                tr("标注完成后点击“同步标注会话”。")}});
}

void DatasetPageController::syncAnnotationSession()
{
    if (!page_) return;
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(page_, tr("X-AnyLabeling 同步"),
            tr("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    bool accepted = false;
    const QString sessionText = QInputDialog::getText(page_,
        tr("标注会话"), tr("输入 Session ArtifactId："),
        QLineEdit::Normal, state_.latestAnnotationSessionArtifactId,
        &accepted).trimmed();
    aitrain::ArtifactId sessionId;
    QString error;
    if (!accepted
        || !aitrain::ArtifactId::parse(sessionText, &sessionId, &error)) {
        if (accepted) {
            QMessageBox::warning(
                page_, tr("标注会话"), tr("Session ArtifactId 无效。"));
        }
        return;
    }
    const QString directory = QDir::fromNativeSeparators(
        QInputDialog::getText(page_, tr("标注工作目录"),
            tr("输入该会话使用的外部工作目录："), QLineEdit::Normal,
            QDir::toNativeSeparators(state_.annotationWorkingDirectory),
            &accepted).trimmed());
    if (!accepted || directory.isEmpty()) return;
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::AnnotationSessionSyncCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.sessionArtifactId = sessionId.toString();
    command.workingDirectory = directory;
    command.options = QJsonObject{
        {QStringLiteral("validateInventory"), true},
        {QStringLiteral("validateHashes"), true}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("X-AnyLabeling 同步"), error);
        return;
    }
    state_.latestAnnotationSessionArtifactId = sessionId.toString();
    state_.annotationWorkingDirectory = directory;
    emit taskStarted(taskId.toString(), QStringLiteral("annotation_sync"));
    emit statusChanged(tr("标注同步中"));
    emit repairLoopChanged(
        tr("修复闭环：正在同步 X-AnyLabeling 标注会话。"),
        QVector<QStringList>{
            {tr("外部修复"), tr("已返回"),
                tr("Session ArtifactId：%1").arg(sessionId.toString())},
            {tr("同步"), tr("运行中"),
                tr("Worker 正在重验基线、编辑白名单、文件集合和哈希。")},
            {tr("复检"), tr("等待"),
                tr("同步成功后由 Query/Presenter 刷新新 Dataset Version。")}});
}

void DatasetPageController::browseSampleReview()
{
    if (!page_) return;
    bool accepted = false;
    const QString artifactText = QInputDialog::getText(page_,
        tr("选择复核 Artifact"), tr("输入已提交的质量/复核 ArtifactId："),
        QLineEdit::Normal, state_.sampleReviewArtifactId,
        &accepted).trimmed();
    if (!accepted) return;
    page_->reviewSamplePathEdit->setText(artifactText);
    loadSampleReview();
}

void DatasetPageController::loadSampleReview()
{
    if (!page_) return;
    aitrain::ArtifactId artifactId;
    QString error;
    if (!aitrain::ArtifactId::parse(
            page_->reviewSamplePathEdit->text().trimmed(),
            &artifactId, &error)) {
        QMessageBox::warning(page_, tr("样本复核"),
            tr("请输入有效的已提交复核 ArtifactId。"));
        return;
    }
    if (!projectOpen_) {
        QMessageBox::warning(
            page_, tr("样本复核"), tr("请先打开项目。"));
        return;
    }
    const QStringList candidates = {
        QStringLiteral("problem_samples.json"),
        QStringLiteral("quality_analysis.json"),
        QStringLiteral("xanylabeling_review_manifest.json"),
        QStringLiteral("repair_manifest.json"),
        QStringLiteral("quality_report.json")};
    ++sampleReviewGeneration_;
    if (sampleReviewGeneration_ == 0) ++sampleReviewGeneration_;
    page_->reviewSourceFilterCombo->setEnabled(false);
    page_->reviewReasonFilterCombo->setEnabled(false);
    loadSampleReviewCandidate(
        artifactId, candidates, 0, sampleReviewGeneration_);
}

void DatasetPageController::loadSampleReviewCandidate(
    const aitrain::ArtifactId& artifactId, const QStringList& candidates,
    int index, quint64 generation, const QString& lastError)
{
    if (!page_ || generation != sampleReviewGeneration_) return;
    if (index >= candidates.size()) {
        QMessageBox::warning(page_, tr("样本复核"),
            tr("Artifact 内没有可读取的质量/复核 JSON：%1")
                .arg(lastError));
        page_->reviewSourceFilterCombo->setEnabled(true);
        page_->reviewReasonFilterCombo->setEnabled(true);
        return;
    }
    const QString candidate = candidates.at(index);
    QString requestError;
    QPointer<DatasetPageController> self(this);
    if (!queryService_->artifactFilePreviewAsync(
            artifactId, candidate, this,
            [self, artifactId, candidates, index, generation](
                bool success, aitrain::ArtifactFilePreview preview,
                QString error) {
                if (!self
                    || generation != self->sampleReviewGeneration_) {
                    return;
                }
                if (!success) {
                    self->loadSampleReviewCandidate(artifactId, candidates,
                        index + 1, generation, error);
                    return;
                }
                QJsonParseError parseError;
                const QJsonDocument document =
                    QJsonDocument::fromJson(preview.content, &parseError);
                if (parseError.error != QJsonParseError::NoError
                    || (!document.isObject() && !document.isArray())) {
                    self->loadSampleReviewCandidate(artifactId, candidates,
                        index + 1, generation, parseError.errorString());
                    return;
                }
                self->state_.sampleReviewArtifactId = artifactId.toString();
                self->state_.sampleReviewSamples =
                    extractReviewSamples(document);
                self->page_->reviewSourceFilterCombo->clear();
                self->page_->reviewSourceFilterCombo->addItem(
                    self->tr("全部来源"), QString());
                self->page_->reviewReasonFilterCombo->clear();
                self->page_->reviewReasonFilterCombo->addItem(
                    self->tr("全部问题"), QString());
                QStringList sources;
                QStringList reasons;
                for (const QJsonValue& value
                    : self->state_.sampleReviewSamples) {
                    const QJsonObject sample = value.toObject();
                    const QString source =
                        sample.value(QStringLiteral("source")).toString();
                    const QString reason =
                        sample.value(QStringLiteral("reason")).toString();
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
                    self->page_->reviewSourceFilterCombo->addItem(
                        source, source);
                }
                for (const QString& reason : reasons) {
                    self->page_->reviewReasonFilterCombo->addItem(
                        reason, reason);
                }
                self->page_->reviewSourceFilterCombo->setEnabled(true);
                self->page_->reviewReasonFilterCombo->setEnabled(true);
                self->refreshSampleReview();
                emit self->statusChanged(
                    self->tr("已加载复核样本：%1 条")
                        .arg(self->state_.sampleReviewSamples.size()));
            },
            4 * 1024 * 1024, &requestError)) {
        loadSampleReviewCandidate(
            artifactId, candidates, index + 1, generation, requestError);
    }
}

QJsonArray DatasetPageController::filteredSampleReviewRows() const
{
    QJsonArray rows;
    if (!page_) return rows;
    const QString sourceFilter =
        page_->reviewSourceFilterCombo->currentData().toString();
    const QString reasonFilter =
        page_->reviewReasonFilterCombo->currentData().toString();
    const QString query =
        page_->reviewSearchEdit->text().trimmed().toLower();
    for (const QJsonValue& value : state_.sampleReviewSamples) {
        const QJsonObject sample = value.toObject();
        if (!sourceFilter.isEmpty()
            && sample.value(QStringLiteral("source")).toString()
                != sourceFilter) {
            continue;
        }
        if (!reasonFilter.isEmpty()
            && sample.value(QStringLiteral("reason")).toString()
                != reasonFilter) {
            continue;
        }
        if (!query.isEmpty()
            && !QString::fromUtf8(
                    QJsonDocument(sample).toJson(QJsonDocument::Compact))
                    .toLower().contains(query)) {
            continue;
        }
        rows.append(sample);
    }
    return rows;
}

void DatasetPageController::refreshSampleReview()
{
    if (!page_ || !page_->sampleReviewTable) return;
    const QJsonArray rows = filteredSampleReviewRows();
    QTableWidget* table = page_->sampleReviewTable;
    table->setRowCount(0);
    for (const QJsonValue& value : rows) {
        const QJsonObject sample = value.toObject();
        const int row = table->rowCount();
        table->insertRow(row);
        const ReviewSamplePathView paths = reviewSamplePathView(sample);
        table->setItem(row, 0,
            new QTableWidgetItem(
                sample.value(QStringLiteral("source")).toString()));
        table->setItem(row, 1,
            new QTableWidgetItem(
                sample.value(QStringLiteral("reason")).toString()));
        table->setItem(row, 2,
            new QTableWidgetItem(reviewClassText(sample)));
        table->setItem(row, 3,
            new QTableWidgetItem(reviewMetricText(sample)));
        auto* image = new QTableWidgetItem(paths.imageRelativePath);
        image->setToolTip(paths.imageRelativePath.isEmpty()
            ? tr("未提供有效的 Snapshot 相对路径；外部或越界路径已隐藏。")
            : tr("Snapshot 内相对路径：%1")
                  .arg(paths.imageRelativePath));
        table->setItem(row, 4, image);
        auto* label = new QTableWidgetItem(paths.labelRelativePath);
        label->setToolTip(paths.labelRelativePath.isEmpty()
            ? tr("未提供有效的 Snapshot 相对路径；外部或越界路径已隐藏。")
            : tr("Snapshot 内相对路径：%1")
                  .arg(paths.labelRelativePath));
        table->setItem(row, 5, label);
        table->setItem(row, 6, new QTableWidgetItem(
            sampleTextField(sample, {
                QStringLiteral("message"), QStringLiteral("note"),
                QStringLiteral("description"),
                QStringLiteral("groundTruth"),
                QStringLiteral("prediction")})));
    }
    page_->sampleReviewSummaryLabel->setText(
        tr("复核样本：显示 %1 / 总计 %2；页面只读")
            .arg(rows.size()).arg(state_.sampleReviewSamples.size()));
}

void DatasetPageController::openSelectedReviewSample()
{
    if (!page_ || page_->sampleReviewTable->currentRow() < 0) {
        QMessageBox::information(
            page_, tr("样本复核"), tr("请先选择一条复核样本。"));
        return;
    }
    if (state_.sampleReviewArtifactId.isEmpty()) {
        QMessageBox::warning(
            page_, tr("样本复核"), tr("当前没有已加载的复核 Artifact。"));
        return;
    }
    const int row = page_->sampleReviewTable->currentRow();
    const auto text = [this, row](int column) {
        QTableWidgetItem* item =
            page_->sampleReviewTable->item(row, column);
        return item ? item->text() : QString();
    };
    QMessageBox::information(page_, tr("样本复核"),
        tr("当前复核记录属于 Artifact %1。\n图片相对路径：%2\n标签相对路径：%3\n\n原始文件不会由 GUI 直接打开；请在任务与产物页按 ArtifactId 进行受控预览。")
            .arg(state_.sampleReviewArtifactId, text(4), text(5)));
}
