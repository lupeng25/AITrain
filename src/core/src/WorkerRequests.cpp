#include "aitrain/core/WorkerRequests.h"

#include "aitrain/core/WorkerProtocol.h"

namespace aitrain {
namespace worker_requests {

namespace {

namespace wp = aitrain::worker_protocol;

DatasetPathRequest parseDatasetPathRequest(const QJsonObject& object)
{
    DatasetPathRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.datasetPath = object.value(wp::field::datasetPath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.format = object.value(wp::field::format()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

} // namespace

DatasetValidationRequest parseDatasetValidationRequest(const QJsonObject& object)
{
    return parseDatasetPathRequest(object);
}

DatasetSplitRequest parseDatasetSplitRequest(const QJsonObject& object)
{
    return parseDatasetPathRequest(object);
}

DatasetConversionRequest parseDatasetConversionRequest(const QJsonObject& object)
{
    DatasetConversionRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.sourcePath = object.value(wp::field::sourcePath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.sourceFormat = object.value(wp::field::sourceFormat()).toString();
    request.targetFormat = object.value(wp::field::targetFormat()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

DatasetCurationRequest parseDatasetCurationRequest(const QJsonObject& object)
{
    return parseDatasetPathRequest(object);
}

AnnotationSessionRequest parseAnnotationSessionRequest(const QJsonObject& object)
{
    return parseDatasetPathRequest(object);
}

AnnotationSyncRequest parseAnnotationSyncRequest(const QJsonObject& object)
{
    AnnotationSyncRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.sessionManifestPath = object.value(wp::field::sessionManifestPath()).toString();
    request.datasetPath = object.value(wp::field::datasetPath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.format = object.value(wp::field::format()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

DatasetSnapshotRequest parseDatasetSnapshotRequest(const QJsonObject& object)
{
    return parseDatasetPathRequest(object);
}

ModelEvaluationRequest parseModelEvaluationRequest(const QJsonObject& object)
{
    ModelEvaluationRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.modelPath = object.value(wp::field::modelPath()).toString();
    request.datasetPath = object.value(wp::field::datasetPath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.taskType = object.value(wp::field::taskType()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

ModelBenchmarkRequest parseModelBenchmarkRequest(const QJsonObject& object)
{
    ModelBenchmarkRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.modelPath = object.value(wp::field::modelPath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

LocalPipelineRequest parseLocalPipelineRequest(const QJsonObject& object)
{
    LocalPipelineRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.templateId = object.value(wp::field::templateId()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

DeliveryReportRequest parseDeliveryReportRequest(const QJsonObject& object)
{
    DeliveryReportRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.context = object.value(wp::field::context()).toObject();
    return request;
}

CustomerOcrAcceptanceRequest parseCustomerOcrAcceptanceRequest(const QJsonObject& object)
{
    CustomerOcrAcceptanceRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

DiagnosticsBundleRequest parseDiagnosticsBundleRequest(const QJsonObject& object)
{
    DiagnosticsBundleRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.context = object.value(wp::field::context()).toObject();
    return request;
}

DeploymentValidationRequest parseDeploymentValidationRequest(const QJsonObject& object)
{
    DeploymentValidationRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.modelPath = object.value(wp::field::modelPath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.format = object.value(wp::field::format()).toString();
    request.sampleImagePath = object.value(wp::field::sampleImagePath()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

ModelExportRequest parseModelExportRequest(const QJsonObject& object)
{
    ModelExportRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.checkpointPath = object.value(wp::field::checkpointPath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    request.format = object.value(wp::field::format()).toString();
    request.options = object.value(wp::field::options()).toObject();
    return request;
}

InferenceRequest parseInferenceRequest(const QJsonObject& object)
{
    InferenceRequest request;
    request.taskId = object.value(wp::field::taskId()).toString();
    request.checkpointPath = object.value(wp::field::checkpointPath()).toString();
    request.imagePath = object.value(wp::field::imagePath()).toString();
    request.outputPath = object.value(wp::field::outputPath()).toString();
    return request;
}

TrainingRequest parseTrainingRequest(const QJsonObject& object)
{
    return TrainingRequest::fromJson(object);
}

} // namespace worker_requests
} // namespace aitrain
