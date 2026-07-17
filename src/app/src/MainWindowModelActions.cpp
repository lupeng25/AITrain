#include "MainWindow.h"
#include "TaskExecutionController.h"

#include "DatasetConversionUiModel.h"
#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/workflow/TrainingWorkflowProfile.h"

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
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>

using namespace aitrain_app;
namespace wp = aitrain::worker_protocol;

namespace {
QString yoloArgObjectName(const QString& key)
{
    return QStringLiteral("YoloTrainArg_%1").arg(key);
}

QString smpArgObjectName(const QString& key)
{
    return QStringLiteral("SmpTrainArg_%1").arg(key);
}

QString anomalyArgObjectName(const QString& key)
{
    return QStringLiteral("AnomalyTrainArg_%1").arg(key);
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

QString anomalyTrainArgText(const QWidget* root, const QString& key)
{
    if (!root) {
        return {};
    }
    if (const auto* edit = root->findChild<QLineEdit*>(anomalyArgObjectName(key))) {
        return edit->text().trimmed();
    }
    if (const auto* combo = root->findChild<QComboBox*>(anomalyArgObjectName(key))) {
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

QJsonValue anomalyTrainArgJsonValue(const QString& key, const QString& text)
{
    const QString normalized = text.trimmed();
    if (normalized.isEmpty()) {
        return QJsonValue();
    }
    const QSet<QString> intKeys = {
        QStringLiteral("seed"),
        QStringLiteral("workers"),
        QStringLiteral("numNeighbors")
    };
    if (intKeys.contains(key) && isIntegerLike(normalized)) {
        return normalized.toInt();
    }
    const QSet<QString> numericKeys = {
        QStringLiteral("quantile"),
        QStringLiteral("coresetSamplingRatio"),
        QStringLiteral("lr"),
        QStringLiteral("weightDecay")
    };
    if (numericKeys.contains(key) && isNumberLike(normalized)) {
        return normalized.toDouble();
    }
    if (key == QStringLiteral("layers")) {
        QJsonArray layers;
        for (const QString& part : normalized.split(QLatin1Char(','), QString::SkipEmptyParts)) {
            const QString layer = part.trimmed();
            if (!layer.isEmpty()) {
                layers.append(layer);
            }
        }
        return layers;
    }
    return normalized;
}

QJsonObject anomalyTrainArgsFromUi(const QWidget* root)
{
    const QStringList keys = {
        QStringLiteral("seed"),
        QStringLiteral("device"),
        QStringLiteral("workers"),
        QStringLiteral("thresholdStrategy"),
        QStringLiteral("quantile"),
        QStringLiteral("backbone"),
        QStringLiteral("layers"),
        QStringLiteral("coresetSamplingRatio"),
        QStringLiteral("numNeighbors"),
        QStringLiteral("modelSize"),
        QStringLiteral("lr"),
        QStringLiteral("weightDecay")
    };
    QJsonObject args;
    for (const QString& key : keys) {
        const QJsonValue value = anomalyTrainArgJsonValue(key, anomalyTrainArgText(root, key));
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

} // namespace

void MainWindow::startInference()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("推理"), uiText("Worker 正在执行任务，稍后再运行交付工作流。"));
        return;
    }
    const QString modelPackageText = inferenceModelPackageCombo_
        ? inferenceModelPackageCombo_->currentData().toString().trimmed()
        : QString();
    const QString sampleDatasetIdText = inferenceSampleDatasetIdEdit_
        ? inferenceSampleDatasetIdEdit_->text().trimmed() : QString();
    const QString sampleDatasetVersionIdText = inferenceSampleDatasetVersionIdEdit_
        ? inferenceSampleDatasetVersionIdEdit_->text().trimmed() : QString();
    const QString sampleSnapshotIdText = inferenceSampleSnapshotIdEdit_
        ? inferenceSampleSnapshotIdEdit_->text().trimmed() : QString();
    const QString sampleSnapshotArtifactIdText = inferenceSampleSnapshotArtifactIdEdit_
        ? inferenceSampleSnapshotArtifactIdEdit_->text().trimmed() : QString();
    const QString sampleRelativePath = QDir::fromNativeSeparators(
        inferenceSampleRelativePathEdit_ ? inferenceSampleRelativePathEdit_->text().trimmed() : QString());
    aitrain::ModelPackageId modelPackageId;
    aitrain::DatasetId sampleDatasetId;
    aitrain::DatasetVersionId sampleDatasetVersionId;
    aitrain::SnapshotId sampleSnapshotId;
    aitrain::ArtifactId sampleSnapshotArtifactId;
    QString error;
    if (!workspace_.isOpen() || currentProjectPath_.isEmpty()
        || !aitrain::ModelPackageId::parse(modelPackageText, &modelPackageId, &error)
        || !aitrain::DatasetId::parse(sampleDatasetIdText, &sampleDatasetId, &error)
        || !aitrain::DatasetVersionId::parse(sampleDatasetVersionIdText, &sampleDatasetVersionId, &error)
        || !aitrain::SnapshotId::parse(sampleSnapshotIdText, &sampleSnapshotId, &error)
        || !aitrain::ArtifactId::parse(sampleSnapshotArtifactIdText, &sampleSnapshotArtifactId, &error)
        || sampleRelativePath.isEmpty()) {
        QMessageBox::warning(this, uiText("推理"), uiText("请选择已验证模型包，并填写样本 Snapshot 身份和 Artifact 内相对路径。"));
        return;
    }

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    QJsonObject options;
    options.insert(QStringLiteral("benchmarkWarmup"), 1);
    options.insert(QStringLiteral("benchmarkIterations"), 3);
    activeTaskId_ = taskId.toString();
    aitrain::worker_protocol::RuntimeDeliveryCommand runtimeCommand;
    runtimeCommand.context.taskId = taskId;
    runtimeCommand.context.projectRoot = currentProjectPath_;
    runtimeCommand.modelPackageId = modelPackageId.toString();
    runtimeCommand.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    runtimeCommand.sampleDatasetId = sampleDatasetId.toString();
    runtimeCommand.sampleDatasetVersionId = sampleDatasetVersionId.toString();
    runtimeCommand.sampleSnapshotId = sampleSnapshotId.toString();
    runtimeCommand.sampleSnapshotArtifactId = sampleSnapshotArtifactId.toString();
    runtimeCommand.sampleRelativePath = sampleRelativePath;
    runtimeCommand.options = options;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{runtimeCommand}, &error)) {
        activeTaskId_.clear();
        QMessageBox::critical(this, uiText("推理"), error);
        return;
    }
    if (inferenceResultLabel_) {
        inferenceResultLabel_->setText(uiText(
            "Runtime Delivery 已派发：等待六步状态与最终 Evidence。\n"
            "注意：底层 ONNX Runtime 单次同步 infer 进入后不可中断，取消会在该次调用返回后收口。"));
    }
    setInferenceOverlayText(inferenceOverlayLabel_, uiText(
        "Runtime Delivery 运行中\n最终预测与 overlay 请在“任务与产物”中查看已提交 Artifact。"));
    workerPill_->setStatus(uiText("Runtime Delivery 运行中"), StatusPill::Tone::Info);
}

void MainWindow::startTraining()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("训练"), uiText("Worker 正在执行任务；当前版本只允许一个活动任务。"));
        return;
    }
    if (currentProjectPath_.isEmpty()) {
        createProject();
        if (currentProjectPath_.isEmpty()) {
            return;
        }
    }
    if (capabilityCombo_->currentData().toString().isEmpty() || currentTaskType().isEmpty()) {
        QMessageBox::warning(this, uiText("训练"), uiText("请选择可用内置能力和任务类型。"));
        return;
    }
    const QString datasetFormat = currentDatasetFormat();
    if (datasetFormat.isEmpty()) {
        QMessageBox::warning(this, uiText("训练"), uiText("请选择与已登记 Snapshot 一致的数据集格式。"));
        return;
    }

    const QString datasetId = dataQualityDatasetIdEdit_ ? dataQualityDatasetIdEdit_->text().trimmed() : QString();
    const QString datasetVersionId = dataQualityDatasetVersionIdEdit_
        ? dataQualityDatasetVersionIdEdit_->text().trimmed() : QString();
    const QString snapshotId = dataQualitySnapshotIdEdit_ ? dataQualitySnapshotIdEdit_->text().trimmed() : QString();
    const QString snapshotArtifactId = dataQualitySnapshotArtifactIdEdit_
        ? dataQualitySnapshotArtifactIdEdit_->text().trimmed() : QString();
    aitrain::DatasetId parsedDatasetId;
    aitrain::DatasetVersionId parsedDatasetVersionId;
    aitrain::SnapshotId parsedSnapshotId;
    aitrain::ArtifactId parsedSnapshotArtifactId;
    QString snapshotIdentityError;
    if (!aitrain::DatasetId::parse(datasetId, &parsedDatasetId, &snapshotIdentityError)
        || !aitrain::DatasetVersionId::parse(datasetVersionId, &parsedDatasetVersionId, &snapshotIdentityError)
        || !aitrain::SnapshotId::parse(snapshotId, &parsedSnapshotId, &snapshotIdentityError)
        || !aitrain::ArtifactId::parse(snapshotArtifactId, &parsedSnapshotArtifactId, &snapshotIdentityError)) {
        QMessageBox::warning(this, uiText("训练"),
            uiText("训练只消费已登记的 Snapshot。请在“数据集”页填写同一条记录的 DatasetId、DatasetVersionId、SnapshotId 和 Snapshot ArtifactId。\n%1")
                .arg(snapshotIdentityError));
        return;
    }

    const aitrain::TaskId taskId = aitrain::TaskId::create();

    const QString trainingBackend = trainingBackendCombo_
        ? trainingBackendCombo_->currentData().toString().trimmed()
        : defaultBackendForTask(currentTaskType());
    const QString backendForRequest = trainingBackend.isEmpty() ? defaultBackendForTask(currentTaskType()) : trainingBackend;
    QString capabilityError;
    if (!aitrain::BuiltinCapabilityRegistry::instance().supports(
            capabilityCombo_->currentData().toString(),
            currentTaskType(),
            datasetFormat,
            backendForRequest,
            &capabilityError)) {
        QMessageBox::warning(this, uiText("训练"), capabilityError);
        return;
    }
    const int requestedBatchSize = batchEdit_ ? batchEdit_->text().toInt() : 1;
    const int effectiveBatchSize = backendForRequest == QStringLiteral("anomalib_efficientad") ? 1 : requestedBatchSize;
    QJsonObject parameters;
    parameters.insert(QStringLiteral("epochs"), epochsEdit_->text().toInt());
    parameters.insert(QStringLiteral("batchSize"), effectiveBatchSize);
    parameters.insert(QStringLiteral("imageSize"), imageSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("gridSize"), gridSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("datasetId"), datasetId);
    parameters.insert(QStringLiteral("datasetVersionId"), datasetVersionId);
    parameters.insert(QStringLiteral("snapshotId"), snapshotId);
    parameters.insert(QStringLiteral("snapshotArtifactId"), snapshotArtifactId);
    const QString modelPreset = modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString();
    const QString seedText = backendForRequest == QStringLiteral("smp_semantic_segmentation")
        ? smpTrainArgText(this, QStringLiteral("seed"))
        : ((backendForRequest == QStringLiteral("anomalib_patchcore") || backendForRequest == QStringLiteral("anomalib_efficientad"))
            ? anomalyTrainArgText(this, QStringLiteral("seed"))
            : yoloTrainArgText(this, QStringLiteral("seed")));
    bool seedOk = false;
    const int seed = seedText.toInt(&seedOk);
    parameters.insert(QStringLiteral("seed"), seedOk ? seed : 42);
    parameters.insert(QStringLiteral("horizontalFlip"), horizontalFlipCheck_ && horizontalFlipCheck_->isChecked());
    parameters.insert(QStringLiteral("colorJitter"), colorJitterCheck_ && colorJitterCheck_->isChecked());
    parameters.insert(QStringLiteral("trainingBackend"), backendForRequest);
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
    if (backendForRequest == QStringLiteral("anomalib_patchcore")
        || backendForRequest == QStringLiteral("anomalib_efficientad")) {
        const QJsonObject anomalyArgs = anomalyTrainArgsFromUi(this);
        for (auto it = anomalyArgs.constBegin(); it != anomalyArgs.constEnd(); ++it) {
            parameters.insert(it.key(), it.value());
        }
        parameters.insert(QStringLiteral("modelFamily"), QStringLiteral("anomaly_detection"));
        parameters.insert(QStringLiteral("taskType"), QStringLiteral("anomaly_detection"));
        parameters.insert(QStringLiteral("runtime"), QStringLiteral("anomalib_python"));
        parameters.insert(QStringLiteral("exportFormats"), QJsonArray{});
    }
    if (backendForRequest == QStringLiteral("paddleocr_det_official")
        || backendForRequest == QStringLiteral("paddleocr_rec_official")
        || backendForRequest == QStringLiteral("paddleocr_ppocrv4_rec")) {
        parameters.insert(QStringLiteral("runOfficial"), true);
        parameters.insert(QStringLiteral("prepareOnly"), false);
    }
    aitrain::TrainingWorkflowProfile workflowProfile;
    QString workflowProfileError;
    if (!aitrain::resolveTrainingWorkflowProfile(
            backendForRequest, &workflowProfile, &workflowProfileError)) {
        QMessageBox::critical(
            this,
            uiText("训练配置"),
            uiText("所选训练后端没有训练工作流，已阻止启动：%1").arg(workflowProfileError));
        return;
    }
    parameters.insert(QStringLiteral("trainingTemplate"), workflowProfile.templateId);
    if (!modelPreset.isEmpty()) {
        parameters.insert(QStringLiteral("modelPreset"), modelPreset);
        if (backendForRequest.startsWith(QStringLiteral("ultralytics_yolo"))) {
            parameters.insert(QStringLiteral("model"), modelPreset);
        }
    }
    QJsonObject adapterParameters = parameters;
    adapterParameters.remove(QStringLiteral("datasetId"));
    adapterParameters.remove(QStringLiteral("datasetVersionId"));
    adapterParameters.remove(QStringLiteral("snapshotId"));
    adapterParameters.remove(QStringLiteral("snapshotArtifactId"));

    metricsWidget_->clear();
    logEdit_->clear();
    progressBar_->setValue(0);
    if (trainingPhaseLabel_) {
        trainingPhaseLabel_->setText(uiText("阶段：校验快照 -> 训练 -> 评估 -> 导出 -> 部署验证 -> 登记模型 -> 交付报告 | 当前：等待 Worker 启动"));
    }
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingEpochValue"))) label->setText(QStringLiteral("--"));
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingBatchValue"))) label->setText(QStringLiteral("--"));
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingEtaValue"))) label->setText(QStringLiteral("--"));
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingDeviceValue"))) label->setText(QStringLiteral("--"));
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingLossValue"))) label->setText(QStringLiteral("--"));
    if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingMapValue"))) label->setText(QStringLiteral("--"));
    if (latestCheckpointLabel_) latestCheckpointLabel_->setText(uiText("最新 checkpoint：暂无"));
    if (latestOnnxLabel_) latestOnnxLabel_->setText(uiText("最新 ONNX：暂无"));
    if (latestReportLabel_) latestReportLabel_->setText(uiText("训练报告：暂无"));
    if (latestPreviewLabel_) latestPreviewLabel_->setText(uiText("最新预览：暂无"));
    if (latestPreviewImageLabel_) {
        latestPreviewImageLabel_->clear();
        latestPreviewImageLabel_->setText(uiText("暂无预览图"));
    }

    activeTaskId_ = taskId.toString();
    activeWorkflowKind_ = QStringLiteral("training");
    aitrain::worker_protocol::TrainingCommand trainingCommand;
    trainingCommand.context.taskId = taskId;
    trainingCommand.context.projectRoot = currentProjectPath_;
    trainingCommand.datasetId = datasetId;
    trainingCommand.datasetVersionId = datasetVersionId;
    trainingCommand.snapshotId = snapshotId;
    trainingCommand.snapshotArtifactId = snapshotArtifactId;
    trainingCommand.capabilityId = capabilityCombo_->currentData().toString();
    trainingCommand.taskType = currentTaskType();
    trainingCommand.trainingBackend = backendForRequest;
    trainingCommand.parameters = adapterParameters;
    QString error;
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{trainingCommand}, &error)) {
        activeTaskId_.clear();
        activeWorkflowKind_.clear();
        updateRecentTasks();
        QMessageBox::critical(this, QStringLiteral("Worker"), error);
        return;
    }
    workerPill_->setStatus(uiText("训练运行中"), StatusPill::Tone::Info);
    appendLog(uiText("任务已启动：%1").arg(taskId.toString()));
    updateRecentTasks();
}

void MainWindow::importModelPackage()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("模型导入"), uiText("Worker 正在执行任务，稍后再导入模型。"));
        return;
    }
    const QString sourceFilePath = QDir::fromNativeSeparators(modelImportSourceEdit_ ? modelImportSourceEdit_->text().trimmed() : QString());
    const QString manifestPath = QDir::fromNativeSeparators(modelImportManifestEdit_ ? modelImportManifestEdit_->text().trimmed() : QString());
    if (!workspace_.isOpen() || !QFileInfo(sourceFilePath).isFile() || !QFileInfo(manifestPath).isFile()) {
        QMessageBox::warning(this, uiText("模型导入"), uiText("请先打开项目，并选择常规模型文件和用户确认的 Manifest 草稿 JSON。"));
        return;
    }
    QFile manifestFile(manifestPath);
    if (!manifestFile.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, uiText("模型导入"), uiText("无法读取 Manifest 草稿：%1").arg(manifestPath));
        return;
    }
    const QJsonDocument document = QJsonDocument::fromJson(manifestFile.readAll());
    if (!document.isObject()) {
        QMessageBox::warning(this, uiText("模型导入"), uiText("Manifest 草稿必须是 JSON 对象。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    QString error;
    aitrain::worker_protocol::ModelImportCommand importCommand;
    importCommand.context.taskId = taskId;
    importCommand.context.projectRoot = currentProjectPath_;
    importCommand.sourceFilePath = sourceFilePath;
    importCommand.manifestDraft = document.object();
    if (!taskController_->start(workerExecutablePath(),
            aitrain::worker_protocol::TaskCommand{importCommand}, &error)) {
        QMessageBox::critical(this, uiText("模型导入"), error);
        return;
    }
    modelImportInProgress_ = true;
    if (modelImportResultLabel_) {
        modelImportResultLabel_->setText(uiText("正在导入模型并计算 SHA-256：%1").arg(QDir::toNativeSeparators(sourceFilePath)));
    }
    workerPill_->setStatus(uiText("模型导入中"), StatusPill::Tone::Info);
}
