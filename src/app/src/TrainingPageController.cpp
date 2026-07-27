#include "TrainingPageController.h"

#include "MainWindowSupport.h"
#include "ApplicationEventRouter.h"
#include "TaskRuntimeController.h"
#include "TrainingPage.h"
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

using namespace aitrain_app;

namespace {

QJsonValue typedArgument(const QString& key, const QString& text)
{
    const QString value = text.trimmed();
    if (value.isEmpty() || value == QStringLiteral("默认")) {
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
    auto* capability = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingCapability"));
    auto* taskType = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingTaskType"));
    auto* backend = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingBackend"));
    auto* model = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingModelPreset"));
    connect(capability, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, [this]() {
            refreshTaskTypes();
            refreshDefaults();
        });
    connect(taskType, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &TrainingPageController::refreshDefaults);
    connect(backend, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, [this]() {
            refreshModelPresets();
            const QString backendId = page_->formData().backendId;
            if (auto* batch = page_->findChild<QLineEdit*>(
                    QStringLiteral("TrainingBatchSize"))) {
                const bool fixedBatch = backendId
                    == QStringLiteral("anomalib_efficientad");
                if (fixedBatch) {
                    const QSignalBlocker blocker(batch);
                    batch->setText(QStringLiteral("1"));
                }
                batch->setEnabled(!fixedBatch);
                batch->setToolTip(fixedBatch
                    ? tr("Anomalib 2.5 EfficientAD 训练 batchSize 固定为 1。")
                    : QString());
            }
            refreshSummary();
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
    refreshCapabilities();
}

void TrainingPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
}

void TrainingPageController::setWorkerExecutable(const QString& executable)
{
    workerExecutable_ = executable;
}

void TrainingPageController::setDatasetBinding(
    const TrainingDatasetBinding& binding)
{
    binding_ = binding;
    refreshDefaults();
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
        combo->addItem(capability.displayName, capability.id);
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
    const QSignalBlocker blocker(model);
    model->clear();
    model->addItems(modelPresetItemsForBackend(backendId));
    model->setCurrentText(defaultModelForBackend(backendId));
}

void TrainingPageController::refreshDefaults()
{
    if (!page_) {
        return;
    }
    QString capabilityId;
    QString taskTypeId;
    QString backendId;
    const QString format = binding_.datasetFormat;
    if (format == QStringLiteral("yolo_detection")) {
        capabilityId = QStringLiteral("yolo");
        taskTypeId = QStringLiteral("detection");
        backendId = QStringLiteral("ultralytics_yolo_detect");
    } else if (format == QStringLiteral("yolo_segmentation")) {
        capabilityId = QStringLiteral("yolo");
        taskTypeId = QStringLiteral("segmentation");
        backendId = QStringLiteral("ultralytics_yolo_segment");
    } else if (format == QStringLiteral("yolo_obb")) {
        capabilityId = QStringLiteral("yolo");
        taskTypeId = QStringLiteral("obb_detection");
        backendId = QStringLiteral("ultralytics_yolo_obb");
    } else if (format == QStringLiteral("semantic_segmentation_mask")) {
        capabilityId = QStringLiteral("semantic_segmentation");
        taskTypeId = QStringLiteral("semantic_segmentation");
        backendId = QStringLiteral("smp_semantic_segmentation");
    } else if (format == QStringLiteral("anomaly_folder")) {
        capabilityId = QStringLiteral("anomaly_detection");
        taskTypeId = QStringLiteral("anomaly_detection");
        backendId = QStringLiteral("anomalib_patchcore");
    } else if (format == QStringLiteral("paddleocr_det")) {
        capabilityId = QStringLiteral("paddleocr");
        taskTypeId = QStringLiteral("ocr_detection");
        backendId = QStringLiteral("paddleocr_det_official");
    } else if (format == QStringLiteral("paddleocr_rec")) {
        capabilityId = QStringLiteral("paddleocr");
        taskTypeId = QStringLiteral("ocr_recognition");
        backendId = QStringLiteral("paddleocr_rec_official");
    }
    auto* capability = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingCapability"));
    auto* backend = page_->findChild<QComboBox*>(
        QStringLiteral("TrainingBackend"));
    if (!capability || !backend) {
        refreshSummary();
        return;
    }
    if (!capabilityId.isEmpty()) {
        const QSignalBlocker blocker(capability);
        setComboCurrentData(capability, capabilityId);
    }
    refreshTaskTypes(taskTypeId);
    if (backendId.isEmpty()) {
        backendId = defaultBackendForTask(currentTaskType());
    }
    {
        const QSignalBlocker blocker(backend);
        setComboCurrentData(backend, backendId);
    }
    refreshModelPresets();
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
        ? tr("当前数据集：%1 | 已提交快照\nDataset %2 / Version %3\n快照：%4 | Artifact %5")
              .arg(datasetFormatLabel(binding_.datasetFormat),
                  binding_.datasetId.left(12),
                  binding_.datasetVersionId.left(12),
                  binding_.snapshotId.left(12),
                  binding_.snapshotArtifactId.left(12))
        : tr("当前数据集：未选择。请先在数据集页选择完整的已提交快照身份。"));
    page_->setDatasetSummaryToolTip(ready
        ? tr("训练只消费持久化身份：Dataset %1 / Version %2 / Snapshot %3 / Artifact %4")
              .arg(binding_.datasetId, binding_.datasetVersionId,
                  binding_.snapshotId, binding_.snapshotArtifactId)
        : QString());
    page_->setBackendSummary(trainingBackendDescription(form.backendId));
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
    if (page_) {
        page_->appendLog(text);
    }
}

void TrainingPageController::applyTaskViewState(const TaskViewState& state)
{
    if (!page_) {
        return;
    }
    page_->setProgress(state.progress);
    if (!state.terminal) {
        page_->setPhase(tr(
            "阶段：校验快照 -> 训练 -> 评估 -> 导出 -> 部署验证 -> 登记模型 -> 交付报告 | 当前：Worker 运行中（%1%）")
            .arg(state.progress));
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
    const auto* deviceControl = page_->findChild<QComboBox*>(
        QStringLiteral("YoloTrainArg_device"));
    QString device = deviceControl
        ? controlValue(deviceControl).trimmed() : QString();
    if (device.isEmpty() || device == QStringLiteral("默认")) {
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
    command.parameters = parameters;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, QStringLiteral("Worker"), error);
        return;
    }
    page_->resetRuntimeProjection();
    liveMetricSequence_ = 0;
    liveArtifactSequence_ = 0;
    emit taskStarted(taskId.toString(), QStringLiteral("training"));
    emit runStarted(taskId.toString());
}
