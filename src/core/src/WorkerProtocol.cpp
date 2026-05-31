#include "aitrain/core/WorkerProtocol.h"

namespace aitrain {
namespace worker_protocol {

namespace command {
QString startTrain() { return QStringLiteral("startTrain"); }
QString pause() { return QStringLiteral("pause"); }
QString resume() { return QStringLiteral("resume"); }
QString heartbeat() { return QStringLiteral("heartbeat"); }
QString environmentCheck() { return QStringLiteral("environmentCheck"); }
QString validateDataset() { return QStringLiteral("validateDataset"); }
QString splitDataset() { return QStringLiteral("splitDataset"); }
QString convertDataset() { return QStringLiteral("convertDataset"); }
QString curateDataset() { return QStringLiteral("curateDataset"); }
QString createDatasetSnapshot() { return QStringLiteral("createDatasetSnapshot"); }
QString evaluateModel() { return QStringLiteral("evaluateModel"); }
QString benchmarkModel() { return QStringLiteral("benchmarkModel"); }
QString runLocalPipeline() { return QStringLiteral("runLocalPipeline"); }
QString generateDeliveryReport() { return QStringLiteral("generateDeliveryReport"); }
QString runCustomerOcrAcceptance() { return QStringLiteral("runCustomerOcrAcceptance"); }
QString collectDiagnostics() { return QStringLiteral("collectDiagnostics"); }
QString validateDeploymentArtifact() { return QStringLiteral("validateDeploymentArtifact"); }
QString exportModel() { return QStringLiteral("exportModel"); }
QString infer() { return QStringLiteral("infer"); }
QString cancel() { return QStringLiteral("cancel"); }
} // namespace command

namespace event {
QString ready() { return QStringLiteral("ready"); }
QString log() { return QStringLiteral("log"); }
QString heartbeat() { return QStringLiteral("heartbeat"); }
QString progress() { return QStringLiteral("progress"); }
QString metric() { return QStringLiteral("metric"); }
QString artifact() { return QStringLiteral("artifact"); }
QString completed() { return QStringLiteral("completed"); }
QString paused() { return QStringLiteral("paused"); }
QString resumed() { return QStringLiteral("resumed"); }
QString canceled() { return QStringLiteral("canceled"); }
QString failed() { return QStringLiteral("failed"); }
QString environmentCheck() { return QStringLiteral("environmentCheck"); }
QString datasetValidation() { return QStringLiteral("datasetValidation"); }
QString datasetSplit() { return QStringLiteral("datasetSplit"); }
QString datasetConversion() { return QStringLiteral("datasetConversion"); }
QString datasetQuality() { return QStringLiteral("datasetQuality"); }
QString datasetSnapshot() { return QStringLiteral("datasetSnapshot"); }
QString evaluationReport() { return QStringLiteral("evaluationReport"); }
QString benchmarkReport() { return QStringLiteral("benchmarkReport"); }
QString pipelinePlan() { return QStringLiteral("pipelinePlan"); }
QString deliveryReport() { return QStringLiteral("deliveryReport"); }
QString modelExport() { return QStringLiteral("modelExport"); }
QString deploymentValidation() { return QStringLiteral("deploymentValidation"); }
QString inferenceResult() { return QStringLiteral("inferenceResult"); }
QString customerOcrAcceptance() { return QStringLiteral("customerOcrAcceptance"); }
QString diagnosticBundle() { return QStringLiteral("diagnosticBundle"); }
} // namespace event

namespace field {
QString taskId() { return QStringLiteral("taskId"); }
QString command() { return QStringLiteral("command"); }
QString status() { return QStringLiteral("status"); }
QString errorCode() { return QStringLiteral("errorCode"); }
QString message() { return QStringLiteral("message"); }
QString datasetPath() { return QStringLiteral("datasetPath"); }
QString sourcePath() { return QStringLiteral("sourcePath"); }
QString outputPath() { return QStringLiteral("outputPath"); }
QString modelPath() { return QStringLiteral("modelPath"); }
QString checkpointPath() { return QStringLiteral("checkpointPath"); }
QString imagePath() { return QStringLiteral("imagePath"); }
QString format() { return QStringLiteral("format"); }
QString sourceFormat() { return QStringLiteral("sourceFormat"); }
QString targetFormat() { return QStringLiteral("targetFormat"); }
QString taskType() { return QStringLiteral("taskType"); }
QString templateId() { return QStringLiteral("templateId"); }
QString sampleImagePath() { return QStringLiteral("sampleImagePath"); }
QString options() { return QStringLiteral("options"); }
QString context() { return QStringLiteral("context"); }
QString reportPath() { return QStringLiteral("reportPath"); }
QString bundlePath() { return QStringLiteral("bundlePath"); }
} // namespace field

bool isControlCommand(const QString& type)
{
    return type == command::heartbeat()
        || type == command::cancel()
        || type == command::pause()
        || type == command::resume();
}

bool isTerminalEvent(const QString& type)
{
    return type == event::completed()
        || type == event::failed()
        || type == event::canceled();
}

bool isTaskStateEvent(const QString& type)
{
    return isTerminalEvent(type)
        || type == event::paused()
        || type == event::resumed();
}

TaskState taskStateForEvent(const QString& type, TaskState fallback)
{
    if (type == event::completed()) {
        return TaskState::Completed;
    }
    if (type == event::failed()) {
        return TaskState::Failed;
    }
    if (type == event::canceled()) {
        return TaskState::Canceled;
    }
    if (type == event::paused()) {
        return TaskState::Paused;
    }
    if (type == event::resumed()) {
        return TaskState::Running;
    }
    return fallback;
}

QJsonObject datasetValidationRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& format,
    const QJsonObject& options,
    const QString& outputPath)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::datasetPath(), datasetPath);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::format(), format);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject datasetSplitRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::datasetPath(), datasetPath);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::format(), format);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject datasetConversionRequest(
    const QString& taskId,
    const QString& sourcePath,
    const QString& outputPath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::sourcePath(), sourcePath);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::sourceFormat(), sourceFormat);
    payload.insert(field::targetFormat(), targetFormat);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject datasetCurationRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options)
{
    return datasetSplitRequest(taskId, datasetPath, outputPath, format, options);
}

QJsonObject datasetSnapshotRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options)
{
    return datasetSplitRequest(taskId, datasetPath, outputPath, format, options);
}

QJsonObject modelEvaluationRequest(
    const QString& taskId,
    const QString& modelPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& taskType,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::modelPath(), modelPath);
    payload.insert(field::datasetPath(), datasetPath);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::taskType(), taskType);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject modelBenchmarkRequest(
    const QString& taskId,
    const QString& modelPath,
    const QString& outputPath,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::modelPath(), modelPath);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject localPipelineRequest(
    const QString& taskId,
    const QString& outputPath,
    const QString& templateId,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::templateId(), templateId);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject deliveryReportRequest(
    const QString& taskId,
    const QString& outputPath,
    const QJsonObject& context)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::context(), context);
    return payload;
}

QJsonObject customerOcrAcceptanceRequest(
    const QString& taskId,
    const QString& outputPath,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject diagnosticsBundleRequest(
    const QString& taskId,
    const QString& outputPath,
    const QJsonObject& context)
{
    return deliveryReportRequest(taskId, outputPath, context);
}

QJsonObject deploymentValidationRequest(
    const QString& taskId,
    const QString& modelPath,
    const QString& outputPath,
    const QString& format,
    const QString& sampleImagePath,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::modelPath(), modelPath);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::format(), format);
    payload.insert(field::sampleImagePath(), sampleImagePath);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject modelExportRequest(
    const QString& taskId,
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::checkpointPath(), checkpointPath);
    payload.insert(field::outputPath(), outputPath);
    payload.insert(field::format(), format);
    return payload;
}

QJsonObject inferenceRequest(
    const QString& taskId,
    const QString& checkpointPath,
    const QString& imagePath,
    const QString& outputPath)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(field::checkpointPath(), checkpointPath);
    payload.insert(field::imagePath(), imagePath);
    payload.insert(field::outputPath(), outputPath);
    return payload;
}

} // namespace worker_protocol
} // namespace aitrain
