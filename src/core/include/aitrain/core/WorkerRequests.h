#pragma once

#include "aitrain/core/TaskModels.h"

#include <QJsonObject>
#include <QString>

namespace aitrain {
namespace worker_requests {

struct DatasetPathRequest {
    QString taskId;
    QString datasetPath;
    QString outputPath;
    QString format;
    QJsonObject options;
};

struct DatasetConversionRequest {
    QString taskId;
    QString sourcePath;
    QString outputPath;
    QString sourceFormat;
    QString targetFormat;
    QJsonObject options;
};

struct AnnotationSyncRequest {
    QString taskId;
    QString sessionManifestPath;
    QString datasetPath;
    QString outputPath;
    QString format;
    QJsonObject options;
};

struct ModelEvaluationRequest {
    QString taskId;
    QString modelPath;
    QString datasetPath;
    QString outputPath;
    QString taskType;
    QJsonObject options;
};

struct ModelBenchmarkRequest {
    QString taskId;
    QString modelPath;
    QString outputPath;
    QJsonObject options;
};

struct LocalPipelineRequest {
    QString taskId;
    QString outputPath;
    QString templateId;
    QJsonObject options;
};

struct ContextReportRequest {
    QString taskId;
    QString outputPath;
    QJsonObject context;
};

struct OptionsReportRequest {
    QString taskId;
    QString outputPath;
    QJsonObject options;
};

struct DeploymentValidationRequest {
    QString taskId;
    QString modelPath;
    QString outputPath;
    QString format;
    QString sampleImagePath;
    QJsonObject options;
};

struct ModelExportRequest {
    QString taskId;
    QString checkpointPath;
    QString outputPath;
    QString format;
    QJsonObject options;
};

struct InferenceRequest {
    QString taskId;
    QString checkpointPath;
    QString imagePath;
    QString outputPath;
};

using DatasetValidationRequest = DatasetPathRequest;
using DatasetSplitRequest = DatasetPathRequest;
using DatasetCurationRequest = DatasetPathRequest;
using AnnotationSessionRequest = DatasetPathRequest;
using DatasetSnapshotRequest = DatasetPathRequest;
using DeliveryReportRequest = ContextReportRequest;
using DiagnosticsBundleRequest = ContextReportRequest;
using CustomerOcrAcceptanceRequest = OptionsReportRequest;

DatasetValidationRequest parseDatasetValidationRequest(const QJsonObject& object);
DatasetSplitRequest parseDatasetSplitRequest(const QJsonObject& object);
DatasetConversionRequest parseDatasetConversionRequest(const QJsonObject& object);
DatasetCurationRequest parseDatasetCurationRequest(const QJsonObject& object);
AnnotationSessionRequest parseAnnotationSessionRequest(const QJsonObject& object);
AnnotationSyncRequest parseAnnotationSyncRequest(const QJsonObject& object);
DatasetSnapshotRequest parseDatasetSnapshotRequest(const QJsonObject& object);
ModelEvaluationRequest parseModelEvaluationRequest(const QJsonObject& object);
ModelBenchmarkRequest parseModelBenchmarkRequest(const QJsonObject& object);
LocalPipelineRequest parseLocalPipelineRequest(const QJsonObject& object);
DeliveryReportRequest parseDeliveryReportRequest(const QJsonObject& object);
CustomerOcrAcceptanceRequest parseCustomerOcrAcceptanceRequest(const QJsonObject& object);
DiagnosticsBundleRequest parseDiagnosticsBundleRequest(const QJsonObject& object);
DeploymentValidationRequest parseDeploymentValidationRequest(const QJsonObject& object);
ModelExportRequest parseModelExportRequest(const QJsonObject& object);
InferenceRequest parseInferenceRequest(const QJsonObject& object);
TrainingRequest parseTrainingRequest(const QJsonObject& object);

} // namespace worker_requests
} // namespace aitrain
