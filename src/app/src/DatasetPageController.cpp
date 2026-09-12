#include "WorkbenchTranslation.h"
#include "DatasetPageController.h"

#include "DatasetPage.h"
#include "DatasetConversionUiModel.h"
#include "DatasetCatalogPresenter.h"
#include "MainWindowSupport.h"
#include "TaskRuntimeController.h"
#include "ApplicationEventRouter.h"
#include "ProjectObjectSelectors.h"
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
#include <QBuffer>
#include <QImageReader>
#include <QPixmap>

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
    connect(this, &DatasetPageController::taskStarted, this,
        [this](const QString& taskId, const QString& kind) {
            activeTaskId_ = taskId;
            activeKind_ = kind;
            activeSnapshotId_ = state_.currentSnapshotId;
        });
}

void DatasetPageController::attach(DatasetWorkspacePage* page)
{
    if (page_ == page) return;
    if (page_) disconnect(page_, nullptr, this, nullptr);
    page_ = page;
    connect(page_->datasetListTable, &QTableWidget::itemSelectionChanged,
        this, &DatasetPageController::selectCatalogRow);
    connect(page_, &DatasetWorkspacePage::importRequested, this, &DatasetPageController::runSnapshotImport);
    connect(page_, &DatasetWorkspacePage::qualityRequested, this, &DatasetPageController::runDataQuality);
    connect(page_, &DatasetWorkspacePage::splitRequested, this, &DatasetPageController::runSplit);
    connect(page_, &DatasetWorkspacePage::conversionRequested, this, &DatasetPageController::startConversion);
    connect(page_, &DatasetWorkspacePage::cancelRequested, this, &DatasetPageController::cancelConversion);
    connect(page_, &DatasetWorkspacePage::browseDatasetRequested, this, &DatasetPageController::browseDataset);
    connect(page_, &DatasetWorkspacePage::browseConversionRequested, this, &DatasetPageController::browseConversionInput);
    connect(page_, &DatasetWorkspacePage::conversionSourceChanged, this, &DatasetPageController::updateConversionTargets);
    connect(page_, &DatasetWorkspacePage::createAnnotationRequested, this, &DatasetPageController::createAnnotationSession);
    connect(page_, &DatasetWorkspacePage::syncAnnotationRequested, this, &DatasetPageController::syncAnnotationSession);
    connect(page_, &DatasetWorkspacePage::chooseReviewRequested, this, &DatasetPageController::browseSampleReview);
    connect(page_, &DatasetWorkspacePage::loadReviewRequested, this, &DatasetPageController::loadSampleReview);
    connect(page_, &DatasetWorkspacePage::reviewFilterChanged, this, &DatasetPageController::refreshSampleReview);
    connect(page_, &DatasetWorkspacePage::openReviewSampleRequested, this, &DatasetPageController::openSelectedReviewSample);
    connect(page_, &DatasetWorkspacePage::sampleSelected, this, &DatasetPageController::previewSample);
    connect(page_->views, &QStackedWidget::currentChanged, this, [this](int mode) {
        if (mode == DatasetWorkspacePage::Detail) previewSample(page_->datasetPreviewTable->currentRow());
    });
    connect(page_, &DatasetWorkspacePage::snapshotChanged, this, &DatasetPageController::selectSnapshot);
    connect(page_, &DatasetWorkspacePage::moreSnapshotsRequested, this, [this]() { loadSnapshots(true); });
    connect(page_, &DatasetWorkspacePage::moreSamplesRequested, this, [this]() { loadSamples(true); });
    bindCatalogSearch(page_->findChild<QLineEdit*>(QStringLiteral("DatasetCatalogSearch")), this, [this](const QString& text) {
        catalogSearch_ = text; catalogPresenter_->setCatalogFilter({text, {}, {}});
        catalogCursors_ = {QString()}; refreshCatalog();
    });
    connect(page_, &DatasetWorkspacePage::refreshRequested, this, [this]() { catalogCursors_ = {QString()}; refreshCatalog(); });
    connect(page_, &DatasetWorkspacePage::nextPageRequested, this, [this]() {
        if (catalogPresenter_->hasMore()) { catalogCursors_.append(catalogPresenter_->nextCursor()); refreshCatalog(); }
    });
    connect(page_, &DatasetWorkspacePage::previousPageRequested, this, [this]() {
        if (catalogCursors_.size() > 1) { catalogCursors_.removeLast(); refreshCatalog(); }
    });
    connect(page_, &DatasetWorkspacePage::importVersionRequested, this, [this]() {
        if (!state_.currentValid) return;
        page_->datasetSnapshotTargetDatasetIdEdit->setText(state_.currentDatasetId);
        page_->datasetSnapshotTargetDatasetNameEdit->setText(state_.currentDisplayName);
        page_->datasetSnapshotTargetDatasetNameEdit->setReadOnly(true);
        page_->datasetFormatCombo->setEnabled(false);
        const QSignalBlocker blocker(page_->datasetFormatCombo);
        setComboCurrentData(page_->datasetFormatCombo, state_.currentFormat);
        page_->showView(DatasetWorkspacePage::Import);
    });
    connect(page_->control<QPushButton>(QStringLiteral("OpenDatasetImportButton")), &QPushButton::clicked, this, [this]() {
        page_->datasetSnapshotTargetDatasetIdEdit->clear();
        page_->datasetSnapshotTargetDatasetNameEdit->setReadOnly(false);
        page_->datasetFormatCombo->setEnabled(true);
        page_->operationStatusLabel->clear();
    });
    connect(page_->datasetFormatCombo,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, [this]() {
            // 导入表单的格式属于草稿，不改写正在浏览或用于训练的快照身份。
            refreshConversionDefaults();
        });
    updateConversionTargets();
    refreshCatalog();
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
    catalogCursors_ = {QString()};
    snapshots_.clear();
    snapshotCursor_.clear();
    sampleCursor_.clear();
    activeTaskId_.clear();
    activeKind_.clear();
    pendingTargetId_.clear();
    selectAfterRefresh_.clear();
    sampleCounts_.clear();
    catalogPresenter_->clear();
    invalidateAsyncPreviews();
    if (page_) {
        page_->showView(DatasetWorkspacePage::Catalog);
        page_->setSelectionAvailable(false);
        page_->snapshotCombo->clear();
        page_->datasetPreviewTable->setRowCount(0);
        page_->validationIssuesTable->setRowCount(0);
        page_->validationSummaryLabel->setText(aitrain_app::workbenchText(QStringLiteral("尚未检查所选快照。")));
    }
}

void DatasetPageController::invalidateAsyncPreviews()
{
    ++sampleReviewGeneration_;
    if (sampleReviewGeneration_ == 0) ++sampleReviewGeneration_;
    ++formatProbeGeneration_;
    if (formatProbeGeneration_ == 0) ++formatProbeGeneration_;
    ++previewGeneration_;
    ++qualityGeneration_;
}

void DatasetPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    if (projectRoot_ != projectRoot) {
        catalogSearch_.clear(); catalogPresenter_->setCatalogFilter({}); catalogCursors_ = {QString()};
        if (page_) { const QSignalBlocker blocker(page_->findChild<QLineEdit*>(QStringLiteral("DatasetCatalogSearch"))); page_->findChild<QLineEdit*>(QStringLiteral("DatasetCatalogSearch"))->clear(); }
    }
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
        == aitrain_app::workbenchText(QStringLiteral("等待转换。"))) {
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
    page_->datasetConversionTargetDatasetIdEdit->setText(aitrain::DatasetId::create().toString());
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
            aitrain_app::workbenchText(QStringLiteral("请选择存在的来源、受支持的目标格式，并填写新数据集名称。")));
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
    pendingTargetId_ = targetDatasetId;
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
    const QString datasetId = state_.currentDatasetId;
    const QString versionId = state_.currentDatasetVersionId;
    const QString snapshotId = state_.currentSnapshotId;
    const QString artifactId = state_.currentSnapshotArtifactId;
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
            aitrain_app::workbenchText(QStringLiteral("请先选择一份已提交的数据版本。")));
        return;
    }
    page_->validationIssuesTable->setRowCount(0);
    page_->validationSummaryLabel->setText(
        aitrain_app::workbenchText(QStringLiteral("正在检查所选数据版本的质量…")));
    page_->showView(DatasetWorkspacePage::Quality);
    page_->validationOutput->setPlainText(
        aitrain_app::workbenchText(QStringLiteral("检查完成后将在此显示结果与报告。")));
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
    const QString datasetId = state_.currentDatasetId;
    const QString versionId = state_.currentDatasetVersionId;
    const QString snapshotId = state_.currentSnapshotId;
    const QString artifactId = state_.currentSnapshotArtifactId;
    page_->splitTargetDatasetIdEdit->setText(aitrain::DatasetId::create().toString());
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
            aitrain_app::workbenchText(QStringLiteral("请选择已提交的数据版本，并填写新数据集名称。")));
        return;
    }
    bool trainOk = false, valOk = false, testOk = false, seedOk = false;
    const double trainRatio = page_->splitTrainRatioEdit->text().toDouble(&trainOk);
    const double valRatio = page_->splitValRatioEdit->text().toDouble(&valOk);
    const double testRatio = page_->splitTestRatioEdit->text().toDouble(&testOk);
    page_->splitSeedEdit->text().toInt(&seedOk);
    if (!trainOk || !valOk || !testOk || !seedOk || trainRatio <= 0.0 || valRatio < 0.0 || testRatio < 0.0
        || qAbs(trainRatio + valRatio + testRatio - 1.0) > 0.000001) {
        QMessageBox::warning(page_, aitrain_app::workbenchText(QStringLiteral("划分比例")), aitrain_app::workbenchText(QStringLiteral("训练比例必须大于 0，验证/测试比例不得为负，三项之和必须为 1；随机种子应为整数。")));
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
    pendingTargetId_ = targetId;
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
    if (page_->datasetSnapshotTargetDatasetIdEdit->text().isEmpty())
        page_->datasetSnapshotTargetDatasetIdEdit->setText(aitrain::DatasetId::create().toString());
    const QString targetId = page_->datasetSnapshotTargetDatasetIdEdit->text().trimmed();
    const QString targetName =
        page_->datasetSnapshotTargetDatasetNameEdit->text().trimmed();
    aitrain::DatasetId parsedTargetId;
    QString error;
    if (path.isEmpty() || format.isEmpty() || !QFileInfo::exists(path)
        || targetName.isEmpty()
        || !aitrain::DatasetId::parse(
            targetId, &parsedTargetId, &error)) {
        QMessageBox::warning(page_, tr("数据集快照"),
            aitrain_app::workbenchText(QStringLiteral("请选择存在的来源目录和格式，并填写数据集名称。")));
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
    pendingTargetId_ = targetId;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("数据集快照"), error);
        return;
    }
    emit taskStarted(
        taskId.toString(), QStringLiteral("dataset_snapshot_import"));
    page_->operationStatusLabel->setText(aitrain_app::workbenchText(QStringLiteral("正在导入，完成后会选中新数据版本。")));
    emit statusChanged(tr("数据集快照创建中"));
}

void DatasetPageController::refreshCatalog()
{
    catalogPresenter_->clear();
    if (projectOpen_) catalogPresenter_->refresh({50, catalogCursors_.constLast()});
    renderCatalog();
}

void DatasetPageController::browseDataset()
{
    if (!page_) return;
    const QString directory = QFileDialog::getExistingDirectory(
        page_, tr("选择数据集目录"));
    if (directory.isEmpty()) return;
    page_->datasetPathEdit->setText(QDir::toNativeSeparators(directory));
    if (page_->datasetSnapshotTargetDatasetNameEdit->text().trimmed().isEmpty())
        page_->datasetSnapshotTargetDatasetNameEdit->setText(QFileInfo(directory).fileName());
    startFormatProbe(directory, false);
    refreshConversionDefaults();
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
        status->setText(
            tr("后台探测完成：%1。Worker 将在校验前重新验证。")
                .arg(datasetFormatLabel(format)));
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
    const QString repairText = state_.latestRepairArtifactId.isEmpty()
        ? selectProjectArtifact(page_, queryService_, {QStringLiteral("dataset_repair_manifest")}, aitrain_app::workbenchText(QStringLiteral("选择修复清单")))
        : state_.latestRepairArtifactId;
    accepted = !repairText.isEmpty();
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
    const QString sessionText = selectProjectArtifact(page_, queryService_,
        {QStringLiteral("annotation_session")}, aitrain_app::workbenchText(QStringLiteral("选择标注会话")));
    accepted = !sessionText.isEmpty();
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
    const QString artifactText = selectProjectArtifact(page_, queryService_,
        {QStringLiteral("dataset_quality_report"), QStringLiteral("dataset_quality_analysis"),
            QStringLiteral("dataset_repair_manifest")}, aitrain_app::workbenchText(QStringLiteral("选择复核报告")));
    if (artifactText.isEmpty()) return;
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
