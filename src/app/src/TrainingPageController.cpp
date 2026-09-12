#include "WorkbenchTranslation.h"
#include "TrainingPageController.h"

#include "MainWindowSupport.h"
#include "ApplicationEventRouter.h"
#include "TaskRuntimeController.h"
#include "TrainingPage.h"
#include "TaskArtifactPresenter.h"
#include "aitrain/product/ProductCapabilityContract.h"
#include <QLabel>
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/workflow/TrainingWorkflowProfile.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileInfo>
#include <QJsonArray>
#include <QLineEdit>
#include <QMessageBox>
#include <QSet>
#include <QSignalBlocker>
#include <QTimer>

using namespace aitrain_app;

namespace {

QJsonValue typedArgument(const QString& key, const QString& text)
{
    const QString value = text.trimmed();
    if (value.isEmpty() || value == aitrain_app::workbenchText(QStringLiteral("默认"))) {
        return {};
    }
    const QString lower = value.toLower();
    if (lower == QStringLiteral("true") || lower == QStringLiteral("false")) {
        return lower == QStringLiteral("true");
    }
    const QSet<QString> integerKeys = {
        QStringLiteral("seed"), QStringLiteral("workers"),
        QStringLiteral("patience"), QStringLiteral("save_period"),
        QStringLiteral("freeze"), QStringLiteral("nbs"),
        QStringLiteral("max_det"), QStringLiteral("close_mosaic"),
        QStringLiteral("mask_ratio"), QStringLiteral("ignoreIndex"),
        QStringLiteral("numNeighbors")};
    bool ok = false;
    if (integerKeys.contains(key)) {
        const int integer = value.toInt(&ok);
        if (ok) return integer;
    }
    const QSet<QString> numericKeys = {
        QStringLiteral("lr0"), QStringLiteral("lrf"),
        QStringLiteral("momentum"), QStringLiteral("weight_decay"),
        QStringLiteral("warmup_epochs"), QStringLiteral("warmup_momentum"),
        QStringLiteral("warmup_bias_lr"), QStringLiteral("fraction"),
        QStringLiteral("multi_scale"), QStringLiteral("box"),
        QStringLiteral("cls"), QStringLiteral("dfl"),
        QStringLiteral("hsv_h"), QStringLiteral("hsv_s"),
        QStringLiteral("hsv_v"), QStringLiteral("degrees"),
        QStringLiteral("translate"), QStringLiteral("scale"),
        QStringLiteral("shear"), QStringLiteral("perspective"),
        QStringLiteral("flipud"), QStringLiteral("fliplr"),
        QStringLiteral("mosaic"), QStringLiteral("mixup"),
        QStringLiteral("cutmix"), QStringLiteral("copy_paste"),
        QStringLiteral("learningRate"), QStringLiteral("quantile"),
        QStringLiteral("coresetSamplingRatio"), QStringLiteral("lr"),
        QStringLiteral("weightDecay")};
    if (numericKeys.contains(key)) {
        const double number = value.toDouble(&ok);
        if (ok) return number;
    }
    if (key == QStringLiteral("classes") || key == QStringLiteral("layers")
        || (key == QStringLiteral("freeze")
            && value.contains(QLatin1Char(',')))) {
        QJsonArray values;
        for (const QString& part :
            value.split(QLatin1Char(','), QString::SkipEmptyParts)) {
            if (key == QStringLiteral("classes")
                || key == QStringLiteral("freeze")) {
                const int integer = part.trimmed().toInt(&ok);
                if (ok) values.append(integer);
            } else {
                values.append(part.trimmed());
            }
        }
        return values;
    }
    return value;
}

QString controlValue(const QObject* object)
{
    if (const auto* edit = qobject_cast<const QLineEdit*>(object)) {
        return edit->text();
    }
    if (const auto* combo = qobject_cast<const QComboBox*>(object)) {
        const QString data = combo->currentData().toString();
        return data.isEmpty() ? combo->currentText() : data;
    }
    if (const auto* check = qobject_cast<const QCheckBox*>(object)) {
        return check->isChecked()
            ? QStringLiteral("true") : QStringLiteral("false");
    }
    return {};
}

} // namespace

TrainingPageController::TrainingPageController(
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , taskRuntime_(taskRuntime)
{
}

void TrainingPageController::attach(TrainingWorkspacePage* page)
{
    page_ = page;
    connect(page_, &TrainingWorkspacePage::startRequested,
        this, &TrainingPageController::start);
    connect(page_, &TrainingWorkspacePage::cancelRequested,
        taskRuntime_, &TaskRuntimeController::cancel);
    connect(taskRuntime_, &TaskRuntimeController::stateChanged, this, [this](TaskRuntimeController::State state) {
        if (state == TaskRuntimeController::State::CancelRequested && selectedTaskId_ == activeTaskId_) {
            page_->cancelTaskButton->setEnabled(false); page_->cancelTaskButton->setText(aitrain_app::workbenchText(QStringLiteral("正在取消")));
            page_->setPhase(aitrain_app::workbenchText(QStringLiteral("已请求取消，等待 Worker 返回终态。")));
        }
    });
    connect(page_, &TrainingWorkspacePage::logRequested, this, [this]() {
        if (selectedTaskId_ == activeTaskId_) page_->setMode(TrainingWorkspacePage::FullLog, TrainingWorkspacePage::Monitor);
        else if (!selectedTaskId_.isEmpty()) emit page_->openTaskRequested(selectedTaskId_);
    });
    auto* capability = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingCapability"));
    auto* taskType = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingTaskType"));
    auto* backend = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingBackend"));
    auto* model = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingModelPreset"));
    connect(backend, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &TrainingPageController::synchronizeBackend);
    connect(page_, &TrainingWorkspacePage::selectDatasetRequested, this, &TrainingPageController::chooseDataset);
    bindCatalogSearch(page_->findChild<QLineEdit*>(QStringLiteral("TrainingHistorySearch")), this, [this](const QString& text) {
        historySearch_ = text; refreshHistory();
    });
    connect(page_, &TrainingWorkspacePage::refreshHistoryRequested, this, &TrainingPageController::refreshHistory);
    connect(page_, &TrainingWorkspacePage::historyRequested, this, &TrainingPageController::openHistory);
    connect(page_, &TrainingWorkspacePage::moreHistoryRequested, this, [this]() {
        if (history_) { history_->loadMore(); renderHistory(); }
    });
    connect(page_, &TrainingWorkspacePage::currentTaskRequested, this, [this]() {
        if (!selectedTaskId_.isEmpty()) emit page_->openTaskRequested(selectedTaskId_);
    });
    connect(page_, &TrainingWorkspacePage::advancedRequested, this, &TrainingPageController::editAdvanced);
    connect(page_, &TrainingWorkspacePage::cancelAdvancedRequested, this, &TrainingPageController::cancelAdvanced);
    connect(page_, &TrainingWorkspacePage::applyAdvancedRequested, this, [this]() {
        if (validateParameters()) { advancedBackup_.clear(); page_->setMode(TrainingWorkspacePage::Configuration); scheduleDraftSave(); }
    });
    connect(model, &QComboBox::currentTextChanged,
        this, &TrainingPageController::refreshSummary);
    for (const QString& name : {
             QStringLiteral("TrainingEpochs"),
             QStringLiteral("TrainingBatchSize"),
             QStringLiteral("TrainingImageSize")}) {
        connect(page_->findChild<QLineEdit*>(name), &QLineEdit::textChanged,
            this, &TrainingPageController::refreshSummary);
    }
    connect(page_, &TrainingWorkspacePage::configurationRequested, this, &TrainingPageController::showHistoricalConfiguration);
    connect(page_, &TrainingWorkspacePage::copyConfigurationRequested, this, &TrainingPageController::copyHistoricalConfiguration);
    refreshCapabilities();
    connect(page_, &TrainingWorkspacePage::modelsRequested, this, [this]() { if (!resultModelId_.isEmpty()) emit page_->modelRequested(resultModelId_); });
    captureParameterDefaults();
    initializeDraftPersistence();
    if (!draftProjectId_.isEmpty()) {
        restoringDraft_ = true;
        restoreDraft();
        restoringDraft_ = false;
    }
    refreshHistory();
}

void TrainingPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    const QString nextId = projectOpen && queryService_ ? queryService_->projectIdentity() : QString();
    const bool changed = projectRoot_ != projectRoot || draftProjectId_ != nextId;
    if (changed) { saveDraft(); restoringDraft_ = true; if (draftTimer_) draftTimer_->stop(); }
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
    if (page_) {
        page_->findChild<QPushButton*>(QStringLiteral("TrainingSaveDraft"))->setEnabled(projectOpen && !nextId.isEmpty());
        page_->findChild<QPushButton*>(QStringLiteral("TrainingDiscardDraft"))->setEnabled(projectOpen && !nextId.isEmpty());
    }
    if (changed) {
        binding_ = {}; selectedTaskId_.clear(); activeTaskId_.clear(); resultModelId_.clear(); historyConfigurations_.clear();
        if (page_) { page_->modelsButton->setEnabled(false); page_->resetRuntimeProjection(); page_->setMode(TrainingWorkspacePage::Catalog); }
        draftProjectId_ = nextId; advancedBackup_.clear(); historySearch_.clear();
        if (page_) { const QSignalBlocker blocker(page_->findChild<QLineEdit*>(QStringLiteral("TrainingHistorySearch"))); page_->findChild<QLineEdit*>(QStringLiteral("TrainingHistorySearch"))->clear(); }
        resetDraftControls(); refreshDefaults(); refreshHistory(); restoreDraft();
        restoringDraft_ = false; draftDirty_ = false;
    }
}

void TrainingPageController::setWorkerExecutable(const QString& executable)
{
    workerExecutable_ = executable;
}

void TrainingPageController::setDatasetBinding(
    const TrainingDatasetBinding& binding)
{
    if (!advancedBackup_.isEmpty()) cancelAdvanced();
    const bool formatChanged = binding_.datasetFormat != binding.datasetFormat;
    binding_ = binding;
    if (formatChanged || modelPresetBackend_.isEmpty()) refreshDefaults();
    else refreshSummary();
    scheduleDraftSave();
}

void TrainingPageController::refreshCapabilities()
{
    if (!page_) {
        return;
    }
    auto* combo = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingCapability"));
    if (!combo) {
        return;
    }
    const QString previous = combo->currentData().toString();
    const QVector<aitrain::CapabilityDescriptor> capabilities =
        aitrain::BuiltinCapabilityRegistry::instance().capabilities();
    const QSignalBlocker blocker(combo);
    combo->clear();
    for (const auto& capability : capabilities) {
        combo->addItem(aitrain_app::workbenchText(capability.displayName), capability.id);
    }
    const int previousIndex = combo->findData(previous);
    if (previousIndex >= 0) {
        combo->setCurrentIndex(previousIndex);
    }
    refreshTaskTypes();
    refreshDefaults();
}

QString TrainingPageController::currentTaskType() const
{
    return page_ ? page_->formData().taskType : QString();
}

void TrainingPageController::refreshTaskTypes(const QString& preferredTask)
{
    if (!page_) {
        return;
    }
    auto* capability = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingCapability"));
    auto* taskType = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingTaskType"));
    if (!capability || !taskType) {
        return;
    }
    const QString current = preferredTask.isEmpty()
        ? taskType->currentData().toString() : preferredTask;
    const aitrain::CapabilityDescriptor descriptor =
        aitrain::BuiltinCapabilityRegistry::instance().capability(
            capability->currentData().toString());
    const QSignalBlocker blocker(taskType);
    taskType->clear();
    addTaskTypeItems(taskType, descriptor.taskTypes);
    const int index = taskType->findData(current);
    if (index >= 0) {
        taskType->setCurrentIndex(index);
    } else if (taskType->count() > 0) {
        taskType->setCurrentIndex(0);
    }
}

void TrainingPageController::refreshModelPresets()
{
    if (!page_) {
        return;
    }
    auto* backend = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingBackend"));
    auto* model = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingModelPreset"));
    if (!backend || !model) {
        return;
    }
    const QString backendId = backend->currentData().toString();
    if (modelPresetBackend_ == backendId && model->count() > 0) return;
    modelPresetBackend_ = backendId;
    const QSignalBlocker blocker(model);
    model->clear();
    model->addItems(modelPresetItemsForBackend(backendId));
    model->setCurrentText(defaultModelForBackend(backendId));
}

void TrainingPageController::refreshDefaults()
{
    if (!page_) return;
    auto* backend = page_->findChild<QComboBox*>(QStringLiteral("TrainingBackend"));
    const QString previous = backend->currentData().toString();
    {
        const QSignalBlocker blocker(backend);
        backend->clear();
        for (const auto& item : aitrain::ProductCapabilityContract::instance().trainingBackends()) {
            if (binding_.datasetFormat.isEmpty() || item.datasetFormat == binding_.datasetFormat)
                backend->addItem(aitrain_app::workbenchText(item.displayName), item.id);
        }
        const int index = backend->findData(previous);
        if (index >= 0) backend->setCurrentIndex(index);
    }
    synchronizeBackend();
}

void TrainingPageController::synchronizeBackend()
{
    if (!page_) return;
    const QString backendId = page_->formData().backendId;
    aitrain::TrainingBackendContract contract;
    if (aitrain::ProductCapabilityContract::instance().resolveTrainingBackend(backendId, &contract)) {
        auto* capability = page_->findChild<QComboBox*>(QStringLiteral("TrainingCapability"));
        const QSignalBlocker blocker(capability);
        setComboCurrentData(capability, contract.capabilityId);
        refreshTaskTypes(contract.taskType);
    }
    refreshModelPresets();
    auto* batch = page_->findChild<QLineEdit*>(QStringLiteral("TrainingBatchSize"));
    const bool fixed = backendId == QStringLiteral("anomalib_efficientad");
    if (fixed) batch->setText(QStringLiteral("1"));
    batch->setEnabled(!fixed);
    batch->setToolTip(fixed ? aitrain_app::workbenchText(QStringLiteral("EfficientAD 官方后端批次固定为 1。")) : QString());
    refreshSummary();
}

void TrainingPageController::refreshSummary()
{
    if (!page_) {
        return;
    }
    const TrainingFormData form = page_->formData();
    const bool ready = !binding_.datasetId.isEmpty()
        && !binding_.datasetVersionId.isEmpty()
        && !binding_.snapshotId.isEmpty()
        && !binding_.snapshotArtifactId.isEmpty();
    page_->setDatasetSummary(ready
        ? aitrain_app::workbenchText(QStringLiteral("%1 · %2 · 已提交快照")).arg(binding_.displayName.isEmpty()
                ? aitrain_app::workbenchText(QStringLiteral("所选数据集")) : binding_.displayName, datasetFormatLabel(binding_.datasetFormat))
        : aitrain_app::workbenchText(QStringLiteral("尚未选择数据版本。")));
    page_->setDatasetSummaryToolTip(ready
        ? tr("训练只消费持久化身份：Dataset %1 / Version %2 / Snapshot %3 / Artifact %4")
              .arg(binding_.datasetId, binding_.datasetVersionId,
                  binding_.snapshotId, binding_.snapshotArtifactId)
        : QString());
    page_->setBackendSummary(trainingBackendDescription(form.backendId));
    if (auto* sample = page_->findChild<QLabel*>(QStringLiteral("TrainingSampleNote")))
        sample->setText(binding_.deploymentSampleRelativePath.isEmpty()
            ? aitrain_app::workbenchText(QStringLiteral("交付检查样本：尚未选择（YOLO / SMP 训练必须选择）"))
            : aitrain_app::workbenchText(QStringLiteral("交付检查样本：%1")).arg(binding_.deploymentSampleRelativePath));
    page_->setBackendPanels(form.backendId);
    page_->setRunSummary(
        tr("运行摘要：%1 | 后端 %2 | 模型 %3 | epoch %4 / batch %5 / image %6")
            .arg(taskTypeLabel(form.taskType),
                form.backendId.isEmpty() ? tr("未选择") : form.backendId,
                form.modelPreset.isEmpty() ? tr("默认") : form.modelPreset)
            .arg(form.epochs).arg(form.batchSize).arg(form.imageSize));
    page_->setRunSummaryToolTip(
        tr("训练只消费四重身份：Dataset %1 / Version %2 / Snapshot %3 / Artifact %4")
            .arg(binding_.datasetId, binding_.datasetVersionId,
                binding_.snapshotId, binding_.snapshotArtifactId));
}

void TrainingPageController::appendLog(const QString& text)
{
    if (page_ && !activeTaskId_.isEmpty() && selectedTaskId_ == activeTaskId_
        && taskRuntime_->taskId().toString() == activeTaskId_) {
        page_->appendLog(text);
    }
}

void TrainingPageController::applyTaskViewState(const TaskViewState& state)
{
    if (!page_) {
        return;
    }
    if (state.terminal) refreshHistory();
    if (state.taskId != selectedTaskId_) return;
    page_->cancelTaskButton->setEnabled(!state.terminal && state.status != QStringLiteral("cancel_requested"));
    page_->cancelTaskButton->setText(state.status == QStringLiteral("cancel_requested") ? aitrain_app::workbenchText(QStringLiteral("正在取消")) : aitrain_app::workbenchText(QStringLiteral("取消任务")));
    page_->setProgress(state.progress);
    if (!state.terminal) {
        page_->setPhase(state.status == QStringLiteral("cancel_requested")
            ? aitrain_app::workbenchText(QStringLiteral("已请求取消，等待 Worker 返回终态。"))
            : aitrain_app::workbenchText(QStringLiteral("训练运行中 · %1% · 阶段明细见任务记录")).arg(state.progress));
    }
    const qint64 metricCount =
        qMax<qint64>(0, state.metricSequence - liveMetricSequence_);
    const int firstMetric = qMax(0,
        state.metrics.size() - static_cast<int>(
            qMin<qint64>(metricCount, state.metrics.size())));
    for (int index = firstMetric; index < state.metrics.size(); ++index) {
        const TaskMetricView& metric = state.metrics.at(index);
        if (!metric.name.isEmpty()) {
            page_->addMetric(metric.name, metric.value);
        }
        if (metric.details.contains(QStringLiteral("epoch"))) {
            page_->setLiveValue(QStringLiteral("TrainingEpochValue"),
                QString::number(metric.details.value(
                    QStringLiteral("epoch")).toInt()));
        }
        if (metric.details.contains(QStringLiteral("step"))) {
            page_->setLiveValue(QStringLiteral("TrainingBatchValue"),
                QString::number(metric.details.value(
                    QStringLiteral("step")).toInt()));
        }
        if (metric.details.value(QStringLiteral("device")).isString()) {
            page_->setLiveValue(QStringLiteral("TrainingDeviceValue"),
                metric.details.value(QStringLiteral("device")).toString());
        }
        const QString value = QStringLiteral("%1: %2")
            .arg(metric.name).arg(metric.value, 0, 'g', 6);
        const QString name = metric.name.toLower();
        page_->setLiveValue(name.contains(QStringLiteral("loss"))
                || name == QStringLiteral("cer")
                || name == QStringLiteral("wer")
                ? QStringLiteral("TrainingLossValue")
                : QStringLiteral("TrainingMapValue"), value);
    }
    liveMetricSequence_ = state.metricSequence;
    const qint64 artifactCount =
        qMax<qint64>(0, state.artifactSequence - liveArtifactSequence_);
    const int firstArtifact = qMax(0,
        state.artifacts.size() - static_cast<int>(
            qMin<qint64>(artifactCount, state.artifacts.size())));
    for (int index = firstArtifact; index < state.artifacts.size(); ++index) {
        const TaskArtifactView& artifact = state.artifacts.at(index);
        page_->updateArtifact(
            artifact.artifactId, artifact.kind, artifact.relativePath);
    }
    liveArtifactSequence_ = state.artifactSequence;
    if (state.terminal) {
        refreshResultModel();
        page_->setPhase(state.status == QStringLiteral("succeeded")
            ? tr("训练 Workflow 已完成；持久化事实已刷新。")
            : tr("训练 Workflow 已终止：%1").arg(state.status));
        if (state.status == QStringLiteral("succeeded")) {
            page_->setLiveValue(
                QStringLiteral("TrainingEtaValue"), QStringLiteral("0s"));
        }
    }
}

QJsonObject TrainingPageController::collectArguments(
    const QString& prefix) const
{
    QJsonObject result;
    const QList<QObject*> controls = page_->findChildren<QObject*>();
    for (QObject* control : controls) {
        const QString name = control->objectName();
        if (!name.startsWith(prefix)) {
            continue;
        }
        const QString key = name.mid(prefix.size());
        const QJsonValue value = typedArgument(key, controlValue(control));
        if (!value.isUndefined() && !value.isNull()) {
            result.insert(key, value);
        }
    }
    return result;
}

QJsonObject TrainingPageController::collectExportArguments() const
{
    QJsonObject result = collectArguments(
        QStringLiteral("YoloTrainExportArg_"));
    result.insert(QStringLiteral("imgsz"), page_->formData().imageSize);
    result.insert(QStringLiteral("batch"), 1);
    const bool int8 = result.value(QStringLiteral("int8")).toBool();
    result.insert(QStringLiteral("format"), int8
        ? QStringLiteral("tensorrt") : QStringLiteral("onnx"));
    const auto* deviceControl = page_->findChild<QLineEdit*>(
        QStringLiteral("YoloTrainArg_device"));
    QString device = deviceControl
        ? controlValue(deviceControl).trimmed() : QString();
    if (device.isEmpty() || device == aitrain_app::workbenchText(QStringLiteral("默认"))) {
        device = QStringLiteral("cpu");
    }
    result.insert(QStringLiteral("device"), device);
    return result;
}

void TrainingPageController::start()
{
    if (taskRuntime_->isRunning()) {
        QMessageBox::warning(page_, tr("训练"),
            tr("Worker 正在执行任务，请先等待或取消当前任务。"));
        return;
    }
    if (!validateParameters()) return;
    const TrainingFormData form = page_->formData();
    aitrain::DatasetId datasetId;
    aitrain::DatasetVersionId versionId;
    aitrain::SnapshotId snapshotId;
    aitrain::ArtifactId artifactId;
    QString error;
    if (!projectOpen_ || projectRoot_.isEmpty()
        || !aitrain::DatasetId::parse(binding_.datasetId, &datasetId, &error)
        || !aitrain::DatasetVersionId::parse(
            binding_.datasetVersionId, &versionId, &error)
        || !aitrain::SnapshotId::parse(
            binding_.snapshotId, &snapshotId, &error)
        || !aitrain::ArtifactId::parse(
            binding_.snapshotArtifactId, &artifactId, &error)) {
        QMessageBox::warning(page_, tr("训练"),
            tr("训练只消费已登记的 Snapshot。请先在数据集页选择完整的 Dataset、Version、Snapshot 和 Artifact 身份。"));
        return;
    }
    if (!aitrain::BuiltinCapabilityRegistry::instance().supports(
            form.capabilityId, form.taskType, binding_.datasetFormat,
            form.backendId, &error)) {
        QMessageBox::warning(page_, tr("训练"), error);
        return;
    }
    if ((form.backendId.startsWith(QStringLiteral("ultralytics_")) || form.backendId.startsWith(QStringLiteral("smp_"))) && binding_.deploymentSampleRelativePath.isEmpty()) {
        page_->findChild<QLabel*>(QStringLiteral("TrainingFormError"))->setText(aitrain_app::workbenchText(QStringLiteral("请先选择同一数据版本中的交付检查样本。")));
        return;
    }
    aitrain::TrainingWorkflowProfile workflow;
    if (!aitrain::resolveTrainingWorkflowProfile(
            form.backendId, &workflow, &error)) {
        QMessageBox::critical(page_, tr("训练配置"),
            tr("所选训练后端没有训练工作流，已阻止启动：%1").arg(error));
        return;
    }

    QJsonObject parameters{
        {QStringLiteral("epochs"), form.epochs},
        {QStringLiteral("batchSize"),
            form.backendId == QStringLiteral("anomalib_efficientad")
                ? 1 : form.batchSize},
        {QStringLiteral("imageSize"), form.imageSize},
        {QStringLiteral("gridSize"), form.gridSize},
        {QStringLiteral("horizontalFlip"), form.horizontalFlip},
        {QStringLiteral("colorJitter"), form.colorJitter},
        {QStringLiteral("seed"), 42},
        {QStringLiteral("trainingBackend"), form.backendId},
        {QStringLiteral("trainingTemplate"), workflow.templateId}};
    if (!form.modelPreset.isEmpty()) {
        parameters.insert(QStringLiteral("modelPreset"), form.modelPreset);
    }
    if (form.backendId.startsWith(QStringLiteral("ultralytics_yolo_"))) {
        QJsonObject args =
            collectArguments(QStringLiteral("YoloTrainArg_"));
        if (form.backendId != QStringLiteral("ultralytics_yolo_segment")) {
            args.remove(QStringLiteral("copy_paste_mode")); args.remove(QStringLiteral("overlap_mask")); args.remove(QStringLiteral("mask_ratio"));
        }
        parameters.insert(QStringLiteral("seed"),
            args.value(QStringLiteral("seed")).toInt(42));
        if (form.horizontalFlip && !args.contains(QStringLiteral("fliplr"))) {
            args.insert(QStringLiteral("fliplr"), 0.5);
        }
        parameters.insert(QStringLiteral("ultralyticsTrainArgs"), args);
        parameters.insert(QStringLiteral("ultralyticsExportArgs"),
            collectExportArguments());
        parameters.insert(QStringLiteral("model"), form.modelPreset);
    } else if (form.backendId
        == QStringLiteral("smp_semantic_segmentation")) {
        const QJsonObject args =
            collectArguments(QStringLiteral("SmpTrainArg_"));
        for (auto it = args.constBegin(); it != args.constEnd(); ++it) {
            parameters.insert(it.key(), it.value());
        }
        parameters.insert(QStringLiteral("modelFamily"),
            QStringLiteral("semantic_segmentation"));
        parameters.insert(QStringLiteral("taskType"),
            QStringLiteral("semantic_segmentation"));
        parameters.insert(QStringLiteral("exportOnnx"), true);
    } else if (form.backendId.startsWith(QStringLiteral("anomalib_"))) {
        const QJsonObject args =
            collectArguments(QStringLiteral("AnomalyTrainArg_"));
        for (auto it = args.constBegin(); it != args.constEnd(); ++it) {
            parameters.insert(it.key(), it.value());
        }
        parameters.insert(QStringLiteral("modelFamily"),
            QStringLiteral("anomaly_detection"));
        parameters.insert(QStringLiteral("taskType"),
            QStringLiteral("anomaly_detection"));
        parameters.insert(QStringLiteral("runtime"),
            QStringLiteral("anomalib_python"));
        parameters.insert(QStringLiteral("exportFormats"), QJsonArray{});
    } else if (form.backendId.startsWith(QStringLiteral("paddleocr_"))) {
        parameters.insert(QStringLiteral("runOfficial"), true);
        parameters.insert(QStringLiteral("prepareOnly"), false);
    }

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::TrainingCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.datasetId = datasetId.toString();
    command.datasetVersionId = versionId.toString();
    command.snapshotId = snapshotId.toString();
    command.snapshotArtifactId = artifactId.toString();
    command.capabilityId = form.capabilityId;
    command.taskType = form.taskType;
    command.trainingBackend = form.backendId;
    command.deploymentSampleRelativePath = binding_.deploymentSampleRelativePath;
    command.parameters = parameters;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, QStringLiteral("Worker"), error);
        return;
    }
    activeTaskId_ = taskId.toString();
    selectedTaskId_ = activeTaskId_;
    page_->resetRuntimeProjection();
    page_->setMode(TrainingWorkspacePage::Monitor);
    resultModelId_.clear(); page_->modelsButton->setEnabled(false);
    page_->setLiveValue(QStringLiteral("TrainingMonitorContext"), QStringLiteral("%1 · %2").arg(binding_.displayName, form.modelPreset));
    page_->cancelTaskButton->setEnabled(true);
    page_->setPhase(aitrain_app::workbenchText(QStringLiteral("任务已提交，等待 Worker。")));
    refreshHistory();
    liveMetricSequence_ = 0;
    liveArtifactSequence_ = 0;
    emit taskStarted(taskId.toString(), QStringLiteral("training"));
    emit runStarted(taskId.toString());
}
