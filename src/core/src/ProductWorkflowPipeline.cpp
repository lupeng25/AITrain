#include "aitrain/core/ProductWorkflow.h"

#include "ProductWorkflowSupport.h"
#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/OcrRecDataset.h"
#include "aitrain/core/SegmentationDataset.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMap>
#include <QRegularExpression>
#include <QSet>
#include <QTextStream>
#include <QThread>

#include <algorithm>
namespace aitrain {
using namespace workflow_detail;

namespace {
QString officialTrainingBackendForPipelineTask(const QString& taskType)
{
    const QString normalized = taskType.trimmed().toLower();
    if (normalized == QStringLiteral("detection")) {
        return QStringLiteral("ultralytics_yolo_detect");
    }
    if (normalized == QStringLiteral("segmentation")) {
        return QStringLiteral("ultralytics_yolo_segment");
    }
    if (normalized == QStringLiteral("obb_detection") || normalized == QStringLiteral("obb")) {
        return QStringLiteral("ultralytics_yolo_obb");
    }
    if (normalized == QStringLiteral("anomaly_detection")) {
        return QStringLiteral("anomalib_patchcore");
    }
    if (normalized == QStringLiteral("ocr_detection")) {
        return QStringLiteral("paddleocr_det_official");
    }
    if (normalized == QStringLiteral("ocr_recognition")) {
        return QStringLiteral("paddleocr_rec_official");
    }
    return {};
}

bool isAnomalyPipelineTask(const QString& taskType, const QString& trainingBackend, const QJsonObject& options)
{
    const QString normalizedTask = taskType.trimmed().toLower();
    const QString normalizedBackend = trainingBackend.trimmed().toLower();
    return normalizedTask == QStringLiteral("anomaly_detection")
        || normalizedBackend == QStringLiteral("anomalib_patchcore")
        || normalizedBackend == QStringLiteral("anomalib_efficientad")
        || options.value(QStringLiteral("runtime")).toString().trimmed().toLower() == QStringLiteral("anomalib_python")
        || options.value(QStringLiteral("modelFamily")).toString().trimmed().toLower() == QStringLiteral("anomaly_detection");
}

QString anomalySidecarPathForModel(const QString& path)
{
    if (path.trimmed().isEmpty()) {
        return {};
    }
    const QFileInfo info(path);
    if (info.fileName().compare(QStringLiteral("anomaly_sidecar.json"), Qt::CaseInsensitive) == 0
        && info.exists()) {
        return info.absoluteFilePath();
    }
    const QString siblingSidecar = info.absoluteDir().filePath(QStringLiteral("anomaly_sidecar.json"));
    if (QFileInfo::exists(siblingSidecar)) {
        return QFileInfo(siblingSidecar).absoluteFilePath();
    }
    return {};
}

QString resolvedAnomalyBackend(const QString& trainingBackend)
{
    const QString normalized = trainingBackend.trimmed().toLower();
    if (normalized == QStringLiteral("anomalib_efficientad")) {
        return QStringLiteral("anomalib_efficientad");
    }
    return QStringLiteral("anomalib_patchcore");
}
} // namespace

WorkflowResult runLocalPipelinePlan(const QString& outputPath, const QString& templateId, const QJsonObject& options)
{
    const QString resolvedTemplate = templateId.isEmpty()
        ? QStringLiteral("train-evaluate-export-register")
        : templateId.trimmed().toLower();
    if (resolvedTemplate != QStringLiteral("train-evaluate-export-register")
        && resolvedTemplate != QStringLiteral("export-infer-benchmark-report")) {
        return failedResult(QStringLiteral("Unsupported local pipeline template: %1").arg(resolvedTemplate));
    }

    QDir().mkpath(outputPath);
    QJsonObject pipeline;
    pipeline.insert(QStringLiteral("kind"), QStringLiteral("local_pipeline_execution"));
    pipeline.insert(QStringLiteral("createdAt"), nowIso());
    pipeline.insert(QStringLiteral("templateId"), resolvedTemplate);
    pipeline.insert(QStringLiteral("state"), QStringLiteral("running"));
    pipeline.insert(QStringLiteral("scaffold"), false);
    pipeline.insert(QStringLiteral("options"), options);

    const QString datasetPath = options.value(QStringLiteral("datasetPath")).toString();
    const QString datasetFormat = options.value(QStringLiteral("datasetFormat")).toString();
    const QString taskType = options.value(QStringLiteral("taskType")).toString(QStringLiteral("detection"));
    const QString trainingBackend = options.value(QStringLiteral("trainingBackend")).toString();
    const QString modelPreset = options.value(QStringLiteral("modelPreset")).toString();
    const int epochs = qMax(1, options.value(QStringLiteral("epochs")).toInt(1));
    const QString exportFormat = options.value(QStringLiteral("exportFormat")).toString(QStringLiteral("onnx"));
    const QString preferredSampleImage = options.value(QStringLiteral("sampleImagePath")).toString(options.value(QStringLiteral("imagePath")).toString());
    const bool anomalyPipeline = isAnomalyPipelineTask(taskType, trainingBackend, options);

    QString modelPath = options.value(QStringLiteral("modelPath")).toString(options.value(QStringLiteral("checkpointPath")).toString());
    QString exportPath;
    QString exportReportPath;
    QString inferencePredictionsPath;
    QString inferenceOverlayPath;
    QString evaluationReportPath;
    QString benchmarkReportPath;
    QString deliveryReportPath;
    QString datasetSnapshotManifestPath;
    int datasetSnapshotId = options.value(QStringLiteral("datasetSnapshotId")).toInt();
    QString datasetSnapshotHash = options.value(QStringLiteral("datasetSnapshotHash")).toString();
    QString datasetSnapshotManifest = options.value(QStringLiteral("datasetSnapshotManifest")).toString();

    QJsonArray stepArray;
    QJsonArray stepTaskIds;
    QJsonArray artifactArray;
    QString failureReason;
    bool officialExportStepAppended = false;

    const auto appendStep = [&](const QString& command, const QString& state, const QString& message, const QString& reportPath, const QJsonArray& artifacts = QJsonArray()) {
        const QString stepTaskId = QStringLiteral("pipeline_step_%1").arg(stepArray.size() + 1);
        QJsonObject step;
        step.insert(QStringLiteral("taskId"), stepTaskId);
        step.insert(QStringLiteral("command"), command);
        step.insert(QStringLiteral("state"), state);
        step.insert(QStringLiteral("message"), message);
        step.insert(QStringLiteral("finishedAt"), nowIso());
        if (!reportPath.isEmpty()) {
            step.insert(QStringLiteral("reportPath"), reportPath);
        }
        if (!artifacts.isEmpty()) {
            step.insert(QStringLiteral("artifacts"), artifacts);
        }
        stepArray.append(step);
        stepTaskIds.append(stepTaskId);
    };

    const auto appendArtifactsFromPayload = [&](const QJsonObject& payload) {
        for (const QString& key : payload.keys()) {
            if (!key.endsWith(QStringLiteral("Path"))) {
                continue;
            }
            const QString path = payload.value(key).toString();
            if (path.isEmpty()) {
                continue;
            }
            artifactArray.append(pathArtifact(QStringLiteral("workflow_artifact"), path, key));
        }
    };

    const auto appendOfficialExportStep = [&]() {
        if (officialExportStepAppended
            || !options.value(QStringLiteral("pipelineOfficialExportCompleted")).toBool(false)) {
            return;
        }
        officialExportStepAppended = true;
        const QJsonObject exportPayload = options.value(QStringLiteral("pipelineOfficialExportPayload")).toObject();
        const QString officialExportPath = options.value(QStringLiteral("pipelineOfficialExportPath")).toString(
            exportPayload.value(QStringLiteral("exportPath")).toString());
        const QString officialReportPath = options.value(QStringLiteral("pipelineOfficialExportReportPath")).toString(
            exportPayload.value(QStringLiteral("reportPath")).toString());
        QJsonArray officialArtifacts;
        if (!officialExportPath.isEmpty()) {
            const QJsonObject artifact = pathArtifact(QStringLiteral("official_yolo_export"), officialExportPath, QStringLiteral("Official YOLO ONNX pre-export"));
            officialArtifacts.append(artifact);
            artifactArray.append(artifact);
        }
        if (!officialReportPath.isEmpty()) {
            const QJsonObject artifact = pathArtifact(QStringLiteral("official_yolo_export_report"), officialReportPath, QStringLiteral("Official YOLO export sidecar"));
            officialArtifacts.append(artifact);
            artifactArray.append(artifact);
        }
        appendStep(
            QStringLiteral("officialYoloExport"),
            QStringLiteral("completed"),
            QStringLiteral("Official YOLO .pt pre-export completed."),
            officialReportPath.isEmpty() ? officialExportPath : officialReportPath,
            officialArtifacts);
    };

    const auto mergedStepOptions = [&](const QString& nestedKey) {
        QJsonObject merged = options;
        const QJsonObject nested = options.value(nestedKey).toObject();
        for (auto it = nested.constBegin(); it != nested.constEnd(); ++it) {
            merged.insert(it.key(), it.value());
        }
        if (anomalyPipeline) {
            merged.insert(QStringLiteral("runtime"), QStringLiteral("anomalib_python"));
            merged.insert(QStringLiteral("trainingBackend"), resolvedAnomalyBackend(trainingBackend));
            merged.insert(QStringLiteral("modelFamily"), QStringLiteral("anomaly_detection"));
            merged.insert(QStringLiteral("taskType"), QStringLiteral("anomaly_detection"));
        }
        return merged;
    };

    const auto failPipeline = [&](const QString& stepName, const QString& message) -> WorkflowResult {
        failureReason = message;
        appendStep(stepName, QStringLiteral("failed"), message, QString());
        pipeline.insert(QStringLiteral("state"), QStringLiteral("failed"));
        pipeline.insert(QStringLiteral("failureReason"), failureReason);
        pipeline.insert(QStringLiteral("finishedAt"), nowIso());
        pipeline.insert(QStringLiteral("steps"), stepArray);
        pipeline.insert(QStringLiteral("taskIds"), stepTaskIds);
        pipeline.insert(QStringLiteral("artifacts"), artifactArray);
        QString error;
        const QString reportPath = QDir(outputPath).filePath(QStringLiteral("local_pipeline_plan.json"));
        if (!writeJsonFile(reportPath, pipeline, &error)) {
            return failedResult(error);
        }
        pipeline.insert(QStringLiteral("reportPath"), reportPath);
        return resultFromReport(reportPath, pipeline);
    };

    auto runValidateStep = [&]() -> bool {
        if (datasetPath.isEmpty() || datasetFormat.isEmpty()) {
            failureReason = QStringLiteral("Dataset path and dataset format are required for validation.");
            return false;
        }
        const DatasetValidationResult validation = validateByFormat(datasetPath, datasetFormat, options.value(QStringLiteral("validationOptions")).toObject());
        const QString reportPath = QDir(outputPath).filePath(QStringLiteral("dataset_validation_report.json"));
        QString error;
        QJsonObject payload = validation.toJson();
        payload.insert(QStringLiteral("createdAt"), nowIso());
        payload.insert(QStringLiteral("datasetPath"), datasetPath);
        payload.insert(QStringLiteral("format"), datasetFormat);
        if (!writeJsonFile(reportPath, payload, &error)) {
            failureReason = error;
            return false;
        }
        artifactArray.append(pathArtifact(QStringLiteral("dataset_validation_report"), reportPath, QStringLiteral("Validation report")));
        appendStep(QStringLiteral("validateDataset"),
            validation.ok ? QStringLiteral("completed") : QStringLiteral("failed"),
            validation.ok ? QStringLiteral("Dataset validation completed.") : QStringLiteral("Dataset validation failed."),
            reportPath,
            QJsonArray{pathArtifact(QStringLiteral("dataset_validation_report"), reportPath)});
        if (!validation.ok) {
            failureReason = validation.errors.isEmpty() ? QStringLiteral("Dataset validation failed.") : validation.errors.first();
            return false;
        }
        return true;
    };

    auto runSnapshotStep = [&]() -> bool {
        const WorkflowResult snapshot = createDatasetSnapshotReport(
            datasetPath,
            QDir(outputPath).filePath(QStringLiteral("dataset_snapshot")),
            datasetFormat,
            options.value(QStringLiteral("snapshotOptions")).toObject());
        if (!snapshot.ok) {
            failureReason = snapshot.error;
            return false;
        }
        datasetSnapshotManifestPath = snapshot.reportPath;
        datasetSnapshotHash = snapshot.payload.value(QStringLiteral("contentHash")).toString();
        datasetSnapshotManifest = snapshot.payload.value(QStringLiteral("manifestPath")).toString(snapshot.reportPath);
        artifactArray.append(pathArtifact(QStringLiteral("dataset_snapshot_manifest"), snapshot.reportPath, QStringLiteral("Dataset snapshot manifest")));
        appendArtifactsFromPayload(snapshot.payload);
        appendStep(QStringLiteral("createDatasetSnapshot"),
            QStringLiteral("completed"),
            QStringLiteral("Dataset snapshot created."),
            snapshot.reportPath,
            QJsonArray{pathArtifact(QStringLiteral("dataset_snapshot_manifest"), snapshot.reportPath)});
        return true;
    };

    auto runTrainStep = [&]() -> bool {
        const QString backend = trainingBackend.trimmed().toLower();
        const QString resolvedTrainingBackend = backend.isEmpty()
            ? officialTrainingBackendForPipelineTask(taskType)
            : backend;
        if (options.value(QStringLiteral("pipelineOfficialTrainingCompleted")).toBool(false)) {
            const QJsonObject completedPayload = options.value(QStringLiteral("pipelineOfficialTrainingPayload")).toObject();
            const QJsonArray trainingArtifacts = options.value(QStringLiteral("pipelineOfficialTrainingArtifacts")).toArray();
            const QJsonArray trainingMetrics = options.value(QStringLiteral("pipelineOfficialTrainingMetrics")).toArray();
            const QString trainingCheckpointPath = options.value(QStringLiteral("pipelineOfficialTrainingCheckpointPath")).toString();
            QString trainingAnomalySidecarPath = options.value(QStringLiteral("pipelineOfficialTrainingAnomalySidecarPath")).toString();
            const QString trainingOnnxPath = options.value(QStringLiteral("pipelineOfficialTrainingOnnxPath")).toString();
            const QString trainingReportPath = options.value(QStringLiteral("pipelineOfficialTrainingReportPath")).toString();

            if (trainingAnomalySidecarPath.isEmpty()) {
                for (const QJsonValue& value : trainingArtifacts) {
                    const QJsonObject artifact = value.toObject();
                    if (artifact.value(QStringLiteral("kind")).toString() == QStringLiteral("anomaly_sidecar")) {
                        trainingAnomalySidecarPath = artifact.value(QStringLiteral("path")).toString();
                        break;
                    }
                }
            }

            if (anomalyPipeline && !trainingAnomalySidecarPath.isEmpty()) {
                modelPath = trainingAnomalySidecarPath;
            } else if (!trainingOnnxPath.isEmpty()) {
                modelPath = trainingOnnxPath;
            } else if (!trainingCheckpointPath.isEmpty()) {
                modelPath = trainingCheckpointPath;
            } else {
                modelPath = completedPayload.value(QStringLiteral("onnxPath")).toString(
                    completedPayload.value(QStringLiteral("checkpointPath")).toString());
            }
            if (modelPath.isEmpty()) {
                failureReason = anomalyPipeline
                    ? QStringLiteral("Pipeline Anomalib training finished without anomaly_sidecar.json.")
                    : QStringLiteral("Pipeline official training finished without a checkpointPath or onnxPath.");
                return false;
            }

            for (const QJsonValue& value : trainingArtifacts) {
                const QJsonObject artifact = value.toObject();
                const QString path = artifact.value(QStringLiteral("path")).toString();
                if (path.isEmpty()) {
                    continue;
                }
                artifactArray.append(pathArtifact(
                    artifact.value(QStringLiteral("kind")).toString(QStringLiteral("pipeline_training_artifact")),
                    path,
                    artifact.value(QStringLiteral("name")).toString(
                        artifact.value(QStringLiteral("message")).toString(QStringLiteral("Pipeline training artifact")))));
            }
            if (!trainingReportPath.isEmpty()) {
                artifactArray.append(pathArtifact(QStringLiteral("training_report"), trainingReportPath, QStringLiteral("Official training report")));
            }

            QJsonObject stepPayload;
            stepPayload.insert(QStringLiteral("artifacts"), trainingArtifacts);
            stepPayload.insert(QStringLiteral("metrics"), trainingMetrics);
            stepPayload.insert(QStringLiteral("completedPayload"), completedPayload);
            const QString stepReportPath = !trainingReportPath.isEmpty() ? trainingReportPath : modelPath;
            appendStep(QStringLiteral("startTrain"),
                QStringLiteral("completed"),
                QStringLiteral("Official backend training completed through Worker-managed Python trainer."),
                stepReportPath,
                QJsonArray{
                    pathArtifact(QStringLiteral("anomaly_sidecar"), trainingAnomalySidecarPath),
                    pathArtifact(QStringLiteral("checkpoint"), trainingCheckpointPath),
                    pathArtifact(QStringLiteral("onnx"), trainingOnnxPath),
                    pathArtifact(QStringLiteral("training_report"), trainingReportPath)});
            Q_UNUSED(stepPayload)
            return true;
        }

        if (resolvedTrainingBackend.isEmpty()) {
            failureReason = QStringLiteral("Pipeline task type '%1' does not have an official production training backend.").arg(taskType);
            return false;
        }

        const QString requestPath = QDir(outputPath).filePath(QStringLiteral("training_request.json"));
        QJsonObject request;
        request.insert(QStringLiteral("taskType"), taskType);
        request.insert(QStringLiteral("datasetPath"), datasetPath);
        request.insert(QStringLiteral("trainingBackend"), resolvedTrainingBackend);
        request.insert(QStringLiteral("modelPreset"), modelPreset);
        request.insert(QStringLiteral("epochs"), epochs);
        request.insert(QStringLiteral("note"), QStringLiteral("Pipeline recorded a reproducible training request. Execute through Worker for official backend training."));
        QString error;
        if (!writeJsonFile(requestPath, request, &error)) {
            failureReason = error;
            return false;
        }
        artifactArray.append(pathArtifact(QStringLiteral("training_request"), requestPath, QStringLiteral("External training request")));
        appendStep(QStringLiteral("startTrain"),
            QStringLiteral("queued_external"),
            QStringLiteral("Training backend is external/offical; request recorded for Worker execution."),
            requestPath,
            QJsonArray{pathArtifact(QStringLiteral("training_request"), requestPath)});
        return true;
    };

    auto runEvaluateStep = [&]() -> bool {
        if (modelPath.isEmpty() || datasetPath.isEmpty()) {
            appendStep(QStringLiteral("evaluateModel"),
                QStringLiteral("skipped"),
                QStringLiteral("Evaluation skipped because modelPath or datasetPath is missing."),
                QString());
            return true;
        }
        QJsonObject evalOptions = anomalyPipeline
            ? mergedStepOptions(QStringLiteral("evaluationOptions"))
            : options.value(QStringLiteral("evaluationOptions")).toObject();
        if (!datasetSnapshotHash.isEmpty()) {
            evalOptions.insert(QStringLiteral("datasetSnapshotHash"), datasetSnapshotHash);
        }
        if (!datasetSnapshotManifest.isEmpty()) {
            evalOptions.insert(QStringLiteral("datasetSnapshotManifest"), datasetSnapshotManifest);
        }
        if (datasetSnapshotId > 0) {
            evalOptions.insert(QStringLiteral("datasetSnapshotId"), datasetSnapshotId);
        }
        const WorkflowResult evaluation = evaluateModelReport(
            modelPath,
            datasetPath,
            QDir(outputPath).filePath(QStringLiteral("evaluation")),
            taskType,
            evalOptions);
        if (!evaluation.ok) {
            failureReason = evaluation.error;
            return false;
        }
        evaluationReportPath = evaluation.reportPath;
        const QJsonObject evaluationArtifact = pathArtifact(QStringLiteral("evaluation_report"), evaluation.reportPath, QStringLiteral("Evaluation report"));
        artifactArray.append(evaluationArtifact);
        appendArtifactsFromPayload(evaluation.payload);
        if (!evaluation.payload.value(QStringLiteral("ok")).toBool(true)) {
            const QString failureCategory = evaluation.payload.value(QStringLiteral("failureCategory")).toString();
            const QString message = evaluation.payload.value(QStringLiteral("message")).toString(
                QStringLiteral("Model evaluation report did not pass."));
            if (failureCategory == QStringLiteral("official-only")) {
                appendStep(QStringLiteral("evaluateModel"),
                    QStringLiteral("skipped"),
                    message,
                    evaluation.reportPath,
                    QJsonArray{evaluationArtifact});
                return true;
            }
            failureReason = message;
            return false;
        }
        appendStep(QStringLiteral("evaluateModel"),
            QStringLiteral("completed"),
            QStringLiteral("Model evaluation completed."),
            evaluation.reportPath,
            QJsonArray{evaluationArtifact});
        return true;
    };

    auto runExportStep = [&]() -> bool {
        appendOfficialExportStep();
        if (modelPath.isEmpty()) {
            failureReason = QStringLiteral("Model export requires modelPath/checkpointPath.");
            return false;
        }
        if (anomalyPipeline) {
            const QString sidecarPath = anomalySidecarPathForModel(modelPath);
            if (sidecarPath.isEmpty()) {
                failureReason = QStringLiteral("Anomaly detection pipeline expects anomaly_sidecar.json; ONNX/TensorRT/NCNN export is not supported for Anomalib v1 artifacts.");
                return false;
            }
            modelPath = sidecarPath;
            const QJsonObject sidecarArtifact = pathArtifact(
                QStringLiteral("anomaly_sidecar"),
                sidecarPath,
                QStringLiteral("Anomalib Python runtime sidecar"));
            artifactArray.append(sidecarArtifact);
            appendStep(QStringLiteral("exportModel"),
                QStringLiteral("skipped"),
                QStringLiteral("Anomaly detection v1 uses Worker-managed Python/Anomalib sidecar artifacts; ONNX/TensorRT/NCNN export is not supported."),
                sidecarPath,
                QJsonArray{sidecarArtifact});
            return true;
        }
        const QString exportDir = QDir(outputPath).filePath(QStringLiteral("export"));
        const QString suffix = QFileInfo(modelPath).suffix().toLower();
        const QString outputModelPath = QDir(exportDir).filePath(
            exportFormat == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (exportFormat == QStringLiteral("ncnn") ? QStringLiteral("model.param")
                : (exportFormat.startsWith(QStringLiteral("tensorrt")) ? QStringLiteral("model.engine") : QStringLiteral("model.export.json"))));
        const DetectionExportResult exportResult = exportDetectionCheckpoint(
            modelPath,
            outputModelPath,
            exportFormat);
        if (!exportResult.ok) {
            failureReason = exportResult.error;
            return false;
        }
        exportPath = exportResult.exportPath;
        exportReportPath = exportResult.reportPath;
        artifactArray.append(pathArtifact(QStringLiteral("model_export"), exportPath, QStringLiteral("Model export")));
        if (!exportReportPath.isEmpty()) {
            artifactArray.append(pathArtifact(QStringLiteral("model_export_report"), exportReportPath, QStringLiteral("Model export report")));
        }
        appendStep(QStringLiteral("exportModel"),
            QStringLiteral("completed"),
            QStringLiteral("Model export completed."),
            exportReportPath.isEmpty() ? exportPath : exportReportPath,
            QJsonArray{
                pathArtifact(QStringLiteral("model_export"), exportPath),
                pathArtifact(QStringLiteral("model_export_report"), exportReportPath)});
        Q_UNUSED(suffix)
        return true;
    };

    auto runInferenceSmokeStep = [&]() -> bool {
        const QString candidateModel = exportPath.isEmpty() ? modelPath : exportPath;
        const QString imagePath = !preferredSampleImage.isEmpty()
            ? preferredSampleImage
            : firstImageFileUnder(datasetPath);
        if (candidateModel.isEmpty() || imagePath.isEmpty() || !QFileInfo::exists(imagePath)) {
            appendStep(QStringLiteral("infer"),
                QStringLiteral("skipped"),
                QStringLiteral("Inference smoke skipped because model/image input is missing."),
                QString());
            return true;
        }

        if (anomalyPipeline) {
            const QString sidecarPath = anomalySidecarPathForModel(candidateModel);
            if (sidecarPath.isEmpty()) {
                failureReason = QStringLiteral("Anomalib inference smoke requires anomaly_sidecar.json.");
                return false;
            }
            modelPath = sidecarPath;
            QJsonObject inferenceOptions = mergedStepOptions(QStringLiteral("inferenceOptions"));
            inferenceOptions.insert(QStringLiteral("sampleImagePath"), imagePath);
            inferenceOptions.insert(QStringLiteral("imagePath"), imagePath);
            const WorkflowResult inference = validateDeploymentArtifactReport(
                sidecarPath,
                QDir(outputPath).filePath(QStringLiteral("inference")),
                QStringLiteral("anomalib_python"),
                inferenceOptions);
            if (!inference.ok) {
                failureReason = inference.error;
                return false;
            }
            const QJsonObject inferenceArtifact = pathArtifact(
                QStringLiteral("deployment_validation_report"),
                inference.reportPath,
                QStringLiteral("Anomalib inference smoke report"));
            artifactArray.append(inferenceArtifact);
            inferencePredictionsPath = inference.payload.value(QStringLiteral("predictionsPath")).toString();
            inferenceOverlayPath = inference.payload.value(QStringLiteral("overlayPath")).toString();
            const QString heatmapPath = inference.payload.value(QStringLiteral("heatmapPath")).toString();
            const QString maskPath = inference.payload.value(QStringLiteral("maskPath")).toString();
            if (!inferencePredictionsPath.isEmpty()) {
                artifactArray.append(pathArtifact(QStringLiteral("inference_predictions"), inferencePredictionsPath, QStringLiteral("Anomaly inference predictions")));
            }
            if (!inferenceOverlayPath.isEmpty()) {
                artifactArray.append(pathArtifact(QStringLiteral("inference_overlay"), inferenceOverlayPath, QStringLiteral("Anomaly inference overlay")));
            }
            if (!heatmapPath.isEmpty()) {
                artifactArray.append(pathArtifact(QStringLiteral("inference_heatmap"), heatmapPath, QStringLiteral("Anomaly inference heatmap")));
            }
            if (!maskPath.isEmpty()) {
                artifactArray.append(pathArtifact(QStringLiteral("inference_mask"), maskPath, QStringLiteral("Anomaly inference mask")));
            }
            if (!inference.payload.value(QStringLiteral("ok")).toBool(true)) {
                failureReason = inference.payload.value(QStringLiteral("message")).toString(
                    QStringLiteral("Anomalib inference smoke did not pass."));
                return false;
            }
            appendStep(QStringLiteral("infer"),
                QStringLiteral("completed"),
                QStringLiteral("Anomalib Python inference smoke completed."),
                inference.reportPath,
                QJsonArray{inferenceArtifact});
            return true;
        }

        const QString suffix = QFileInfo(candidateModel).suffix().toLower();
        const QString inferenceDir = QDir(outputPath).filePath(QStringLiteral("inference"));
        QDir().mkpath(inferenceDir);
        QJsonArray predictions;
        QImage overlay;
        QString error;
        QString inferenceTaskType = QStringLiteral("detection");
        if (suffix == QStringLiteral("onnx")) {
            const QString family = inferOnnxModelFamily(candidateModel);
            if (family == QStringLiteral("semantic_segmentation")) {
                inferenceTaskType = QStringLiteral("semantic_segmentation");
                const SemanticSegmentationPrediction prediction = predictSemanticSegmentationOnnxRuntime(candidateModel, imagePath, &error);
                if (error.isEmpty()) {
                    predictions.append(semanticSegmentationPredictionToJson(prediction));
                    overlay = renderSemanticSegmentationPrediction(imagePath, prediction, &error);
                }
            } else if (family == QStringLiteral("yolo_segmentation")) {
                inferenceTaskType = QStringLiteral("segmentation");
                DetectionInferenceOptions inferenceOptions;
                const QVector<SegmentationPrediction> segPredictions = predictSegmentationOnnxRuntime(candidateModel, imagePath, inferenceOptions, &error);
                for (const SegmentationPrediction& prediction : segPredictions) {
                    predictions.append(segmentationPredictionToJson(prediction));
                }
                overlay = renderSegmentationPredictions(imagePath, segPredictions, &error);
            } else if (family == QStringLiteral("yolo_obb")) {
                inferenceTaskType = QStringLiteral("obb_detection");
                DetectionInferenceOptions inferenceOptions;
                const QVector<ObbPrediction> obbPredictions = predictObbOnnxRuntime(candidateModel, imagePath, inferenceOptions, &error);
                for (const ObbPrediction& prediction : obbPredictions) {
                    predictions.append(obbPredictionToJson(prediction));
                }
                overlay = renderObbPredictions(imagePath, obbPredictions, &error);
            } else if (family == QStringLiteral("ocr_recognition")) {
                appendStep(QStringLiteral("infer"),
                    QStringLiteral("skipped"),
                    QStringLiteral("OCR inference smoke is official-only; use PaddleOCR official Rec/System task artifacts instead of AITrain C++ ONNX OCR postprocess."),
                    candidateModel);
                return true;
            } else if (family == QStringLiteral("ocr_detection")) {
                appendStep(QStringLiteral("infer"),
                    QStringLiteral("skipped"),
                    QStringLiteral("OCR inference smoke is official-only; use PaddleOCR official Det/System task artifacts instead of AITrain C++ ONNX OCR postprocess."),
                    candidateModel);
                return true;
            } else {
                DetectionInferenceOptions inferenceOptions;
                const QVector<DetectionPrediction> detPredictions = predictDetectionOnnxRuntime(candidateModel, imagePath, inferenceOptions, &error);
                for (const DetectionPrediction& prediction : detPredictions) {
                    predictions.append(detectionPredictionToJson(prediction));
                }
                overlay = renderDetectionPredictions(imagePath, detPredictions, &error);
            }
        } else {
            failureReason = QStringLiteral("Inference smoke supports official ONNX model artifacts only. Unsupported model format: %1").arg(candidateModel);
            return false;
        }
        if (!error.isEmpty()) {
            failureReason = error;
            return false;
        }

        inferencePredictionsPath = QDir(inferenceDir).filePath(QStringLiteral("inference_predictions.json"));
        QJsonObject predictionRoot;
        predictionRoot.insert(QStringLiteral("createdAt"), nowIso());
        predictionRoot.insert(QStringLiteral("modelPath"), candidateModel);
        predictionRoot.insert(QStringLiteral("imagePath"), imagePath);
        predictionRoot.insert(QStringLiteral("taskType"), inferenceTaskType);
        predictionRoot.insert(QStringLiteral("predictions"), predictions);
        if (!writeJsonFile(inferencePredictionsPath, predictionRoot, &error)) {
            failureReason = error;
            return false;
        }
        artifactArray.append(pathArtifact(QStringLiteral("inference_predictions"), inferencePredictionsPath, QStringLiteral("Inference predictions")));
        if (!overlay.isNull()) {
            inferenceOverlayPath = QDir(inferenceDir).filePath(QStringLiteral("inference_overlay.png"));
            overlay.save(inferenceOverlayPath);
            artifactArray.append(pathArtifact(QStringLiteral("inference_overlay"), inferenceOverlayPath, QStringLiteral("Inference overlay")));
        }
        appendStep(QStringLiteral("infer"),
            QStringLiteral("completed"),
            QStringLiteral("Inference smoke completed."),
            inferencePredictionsPath,
            QJsonArray{
                pathArtifact(QStringLiteral("inference_predictions"), inferencePredictionsPath),
                pathArtifact(QStringLiteral("inference_overlay"), inferenceOverlayPath)});
        return true;
    };

    auto runBenchmarkStep = [&]() -> bool {
        const QString benchmarkModelPath = exportPath.isEmpty() ? modelPath : exportPath;
        if (benchmarkModelPath.isEmpty()) {
            failureReason = QStringLiteral("Benchmark step requires a model artifact.");
            return false;
        }
        QJsonObject benchmarkOptions = anomalyPipeline
            ? mergedStepOptions(QStringLiteral("benchmarkOptions"))
            : options.value(QStringLiteral("benchmarkOptions")).toObject();
        benchmarkOptions.insert(QStringLiteral("datasetPath"), datasetPath);
        if (!preferredSampleImage.isEmpty()) {
            benchmarkOptions.insert(QStringLiteral("sampleImagePath"), preferredSampleImage);
        } else if (anomalyPipeline) {
            const QString firstImage = firstImageFileUnder(datasetPath);
            if (!firstImage.isEmpty()) {
                benchmarkOptions.insert(QStringLiteral("sampleImagePath"), firstImage);
                benchmarkOptions.insert(QStringLiteral("imagePath"), firstImage);
            }
        }
        const WorkflowResult benchmark = benchmarkModelReport(
            benchmarkModelPath,
            QDir(outputPath).filePath(QStringLiteral("benchmark")),
            benchmarkOptions);
        if (!benchmark.ok) {
            failureReason = benchmark.error;
            return false;
        }
        benchmarkReportPath = benchmark.reportPath;
        const QJsonObject benchmarkArtifact = pathArtifact(QStringLiteral("benchmark_report"), benchmark.reportPath, QStringLiteral("Benchmark report"));
        artifactArray.append(benchmarkArtifact);
        if (!benchmark.payload.value(QStringLiteral("ok")).toBool(true)) {
            const QString failureCategory = benchmark.payload.value(QStringLiteral("failureCategory")).toString();
            const QString message = benchmark.payload.value(QStringLiteral("message")).toString(
                QStringLiteral("Benchmark report did not pass."));
            if (failureCategory == QStringLiteral("official-only")) {
                appendStep(QStringLiteral("benchmarkModel"),
                    QStringLiteral("skipped"),
                    message,
                    benchmark.reportPath,
                    QJsonArray{benchmarkArtifact});
                return true;
            }
            failureReason = message;
            return false;
        }
        appendStep(QStringLiteral("benchmarkModel"),
            QStringLiteral("completed"),
            QStringLiteral("Benchmark completed."),
            benchmark.reportPath,
            QJsonArray{benchmarkArtifact});
        return true;
    };

    auto runRegisterStep = [&]() -> bool {
        const QString registerPath = QDir(outputPath).filePath(QStringLiteral("model_registration_candidate.json"));
        QJsonObject candidate;
        candidate.insert(QStringLiteral("createdAt"), nowIso());
        candidate.insert(QStringLiteral("modelPath"), exportPath.isEmpty() ? modelPath : exportPath);
        candidate.insert(QStringLiteral("sourceModelPath"), modelPath);
        candidate.insert(QStringLiteral("evaluationReportPath"), evaluationReportPath);
        candidate.insert(QStringLiteral("benchmarkReportPath"), benchmarkReportPath);
        candidate.insert(QStringLiteral("datasetPath"), datasetPath);
        candidate.insert(QStringLiteral("datasetFormat"), datasetFormat);
        candidate.insert(QStringLiteral("datasetSnapshotId"), datasetSnapshotId);
        candidate.insert(QStringLiteral("datasetSnapshotHash"), datasetSnapshotHash);
        candidate.insert(QStringLiteral("datasetSnapshotManifest"), datasetSnapshotManifest);
        candidate.insert(QStringLiteral("trainingBackend"), trainingBackend);
        candidate.insert(QStringLiteral("modelPreset"), modelPreset);
        candidate.insert(QStringLiteral("taskType"), taskType);
        candidate.insert(QStringLiteral("state"), QStringLiteral("candidate"));
        candidate.insert(QStringLiteral("note"), QStringLiteral("Pipeline-generated model registration candidate for GUI/model registry ingestion."));
        QString error;
        if (!writeJsonFile(registerPath, candidate, &error)) {
            failureReason = error;
            return false;
        }
        artifactArray.append(pathArtifact(QStringLiteral("model_registration_candidate"), registerPath, QStringLiteral("Model registration candidate")));
        appendStep(QStringLiteral("registerModel"),
            QStringLiteral("completed"),
            QStringLiteral("Model registration candidate generated."),
            registerPath,
            QJsonArray{pathArtifact(QStringLiteral("model_registration_candidate"), registerPath)});
        return true;
    };

    auto runDeliveryStep = [&]() -> bool {
        QJsonObject deliveryContext = options.value(QStringLiteral("deliveryContext")).toObject();
        deliveryContext.insert(QStringLiteral("templateId"), resolvedTemplate);
        deliveryContext.insert(QStringLiteral("taskType"), taskType);
        deliveryContext.insert(QStringLiteral("trainingBackend"), trainingBackend);
        if (anomalyPipeline) {
            deliveryContext.insert(QStringLiteral("runtime"), QStringLiteral("anomalib_python"));
            deliveryContext.insert(QStringLiteral("runtimeBoundary"), QStringLiteral("Worker-managed Python/Anomalib artifact runtime; no AITrain C++ ONNX/TensorRT/NCNN anomaly export."));
        }
        deliveryContext.insert(QStringLiteral("modelPreset"), modelPreset);
        deliveryContext.insert(QStringLiteral("modelPath"), exportPath.isEmpty() ? modelPath : exportPath);
        deliveryContext.insert(QStringLiteral("datasetPath"), datasetPath);
        deliveryContext.insert(QStringLiteral("datasetFormat"), datasetFormat);
        deliveryContext.insert(QStringLiteral("datasetSnapshotId"), datasetSnapshotId);
        deliveryContext.insert(QStringLiteral("datasetSnapshotHash"), datasetSnapshotHash);
        deliveryContext.insert(QStringLiteral("datasetSnapshotManifest"), datasetSnapshotManifest);
        deliveryContext.insert(QStringLiteral("evaluationReportPath"), evaluationReportPath);
        deliveryContext.insert(QStringLiteral("benchmarkReportPath"), benchmarkReportPath);
        deliveryContext.insert(QStringLiteral("exportPath"), exportPath);
        deliveryContext.insert(QStringLiteral("exportReportPath"), exportReportPath);
        deliveryContext.insert(QStringLiteral("inferencePredictionsPath"), inferencePredictionsPath);
        deliveryContext.insert(QStringLiteral("inferenceOverlayPath"), inferenceOverlayPath);
        deliveryContext.insert(QStringLiteral("artifacts"), artifactArray);
        const WorkflowResult delivery = generateTrainingDeliveryReport(
            QDir(outputPath).filePath(QStringLiteral("delivery")),
            deliveryContext);
        if (!delivery.ok) {
            failureReason = delivery.error;
            return false;
        }
        deliveryReportPath = delivery.reportPath;
        appendArtifactsFromPayload(delivery.payload);
        artifactArray.append(pathArtifact(QStringLiteral("delivery_report"), delivery.reportPath, QStringLiteral("Delivery report")));
        appendStep(QStringLiteral("generateTrainingDeliveryReport"),
            QStringLiteral("completed"),
            QStringLiteral("Delivery report generated."),
            delivery.reportPath,
            QJsonArray{pathArtifact(QStringLiteral("delivery_report"), delivery.reportPath)});
        return true;
    };

    if (resolvedTemplate == QStringLiteral("train-evaluate-export-register")) {
        if (!runValidateStep()) {
            return failPipeline(QStringLiteral("validateDataset"), failureReason);
        }
        if (!runSnapshotStep()) {
            return failPipeline(QStringLiteral("createDatasetSnapshot"), failureReason);
        }
        if (!runTrainStep()) {
            return failPipeline(QStringLiteral("startTrain"), failureReason);
        }
        if (!runEvaluateStep()) {
            return failPipeline(QStringLiteral("evaluateModel"), failureReason);
        }
        if (!runExportStep()) {
            return failPipeline(QStringLiteral("exportModel"), failureReason);
        }
        if (!runRegisterStep()) {
            return failPipeline(QStringLiteral("registerModel"), failureReason);
        }
        if (!runDeliveryStep()) {
            return failPipeline(QStringLiteral("generateTrainingDeliveryReport"), failureReason);
        }
    } else {
        if (!runExportStep()) {
            return failPipeline(QStringLiteral("exportModel"), failureReason);
        }
        if (!runInferenceSmokeStep()) {
            return failPipeline(QStringLiteral("infer"), failureReason);
        }
        if (!runBenchmarkStep()) {
            return failPipeline(QStringLiteral("benchmarkModel"), failureReason);
        }
        if (!runDeliveryStep()) {
            return failPipeline(QStringLiteral("generateTrainingDeliveryReport"), failureReason);
        }
    }

    pipeline.insert(QStringLiteral("state"), QStringLiteral("completed"));
    pipeline.insert(QStringLiteral("steps"), stepArray);
    pipeline.insert(QStringLiteral("taskIds"), stepTaskIds);
    pipeline.insert(QStringLiteral("artifacts"), artifactArray);
    pipeline.insert(QStringLiteral("modelPath"), modelPath);
    pipeline.insert(QStringLiteral("exportPath"), exportPath);
    pipeline.insert(QStringLiteral("pipelineOfficialExportPath"), options.value(QStringLiteral("pipelineOfficialExportPath")).toString());
    pipeline.insert(QStringLiteral("pipelineOfficialExportReportPath"), options.value(QStringLiteral("pipelineOfficialExportReportPath")).toString());
    pipeline.insert(QStringLiteral("pipelineOfficialExportSourceCheckpointPath"), options.value(QStringLiteral("pipelineOfficialExportSourceCheckpointPath")).toString());
    pipeline.insert(QStringLiteral("evaluationReportPath"), evaluationReportPath);
    pipeline.insert(QStringLiteral("benchmarkReportPath"), benchmarkReportPath);
    pipeline.insert(QStringLiteral("deliveryReportPath"), deliveryReportPath);
    pipeline.insert(QStringLiteral("datasetSnapshotManifestPath"), datasetSnapshotManifestPath);
    pipeline.insert(QStringLiteral("finishedAt"), nowIso());

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("local_pipeline_plan.json"));
    if (!writeJsonFile(reportPath, pipeline, &error)) {
        return failedResult(error);
    }
    pipeline.insert(QStringLiteral("reportPath"), reportPath);
    return resultFromReport(reportPath, pipeline);
}
} // namespace aitrain
