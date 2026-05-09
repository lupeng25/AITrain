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
        if (taskType == QStringLiteral("detection")
            && (backend.isEmpty()
                || backend == QStringLiteral("tiny_linear_detector")
                || backend == QStringLiteral("tiny_detector")
                || backend == QStringLiteral("tiny_linear"))) {
            DetectionTrainingOptions trainOptions;
            trainOptions.epochs = epochs;
            trainOptions.batchSize = qMax(1, options.value(QStringLiteral("batchSize")).toInt(1));
            const int imageSize = qMax(32, options.value(QStringLiteral("imageSize")).toInt(320));
            trainOptions.imageSize = QSize(imageSize, imageSize);
            trainOptions.outputPath = QDir(outputPath).filePath(QStringLiteral("training"));
            trainOptions.trainingBackend = backend.isEmpty() ? QStringLiteral("tiny_linear_detector") : backend;
            const DetectionTrainingResult trainingResult = trainDetectionBaseline(datasetPath, trainOptions);
            if (!trainingResult.ok) {
                failureReason = trainingResult.error;
                return false;
            }
            modelPath = trainingResult.checkpointPath;
            artifactArray.append(pathArtifact(QStringLiteral("checkpoint"), modelPath, QStringLiteral("Trained checkpoint")));
            appendStep(QStringLiteral("startTrain"),
                QStringLiteral("completed"),
                QStringLiteral("Detection training completed with local tiny backend."),
                modelPath,
                QJsonArray{pathArtifact(QStringLiteral("checkpoint"), modelPath)});
            return true;
        }

        if (options.value(QStringLiteral("pipelineOfficialTrainingCompleted")).toBool(false)) {
            const QJsonObject completedPayload = options.value(QStringLiteral("pipelineOfficialTrainingPayload")).toObject();
            const QJsonArray trainingArtifacts = options.value(QStringLiteral("pipelineOfficialTrainingArtifacts")).toArray();
            const QJsonArray trainingMetrics = options.value(QStringLiteral("pipelineOfficialTrainingMetrics")).toArray();
            const QString trainingCheckpointPath = options.value(QStringLiteral("pipelineOfficialTrainingCheckpointPath")).toString();
            const QString trainingOnnxPath = options.value(QStringLiteral("pipelineOfficialTrainingOnnxPath")).toString();
            const QString trainingReportPath = options.value(QStringLiteral("pipelineOfficialTrainingReportPath")).toString();

            if (!trainingOnnxPath.isEmpty()) {
                modelPath = trainingOnnxPath;
            } else if (!trainingCheckpointPath.isEmpty()) {
                modelPath = trainingCheckpointPath;
            } else {
                modelPath = completedPayload.value(QStringLiteral("onnxPath")).toString(
                    completedPayload.value(QStringLiteral("checkpointPath")).toString());
            }
            if (modelPath.isEmpty()) {
                failureReason = QStringLiteral("Pipeline official training finished without a checkpointPath or onnxPath.");
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
                    pathArtifact(QStringLiteral("checkpoint"), trainingCheckpointPath),
                    pathArtifact(QStringLiteral("onnx"), trainingOnnxPath),
                    pathArtifact(QStringLiteral("training_report"), trainingReportPath)});
            Q_UNUSED(stepPayload)
            return true;
        }

        const QString requestPath = QDir(outputPath).filePath(QStringLiteral("training_request.json"));
        QJsonObject request;
        request.insert(QStringLiteral("taskType"), taskType);
        request.insert(QStringLiteral("datasetPath"), datasetPath);
        request.insert(QStringLiteral("trainingBackend"), trainingBackend);
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
        QJsonObject evalOptions = options.value(QStringLiteral("evaluationOptions")).toObject();
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
        artifactArray.append(pathArtifact(QStringLiteral("evaluation_report"), evaluation.reportPath, QStringLiteral("Evaluation report")));
        appendArtifactsFromPayload(evaluation.payload);
        appendStep(QStringLiteral("evaluateModel"),
            QStringLiteral("completed"),
            QStringLiteral("Model evaluation completed."),
            evaluation.reportPath,
            QJsonArray{pathArtifact(QStringLiteral("evaluation_report"), evaluation.reportPath)});
        return true;
    };

    auto runExportStep = [&]() -> bool {
        if (modelPath.isEmpty()) {
            failureReason = QStringLiteral("Model export requires modelPath/checkpointPath.");
            return false;
        }
        const QString exportDir = QDir(outputPath).filePath(QStringLiteral("export"));
        const QString suffix = QFileInfo(modelPath).suffix().toLower();
        const QString outputModelPath = QDir(exportDir).filePath(
            exportFormat == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (exportFormat.startsWith(QStringLiteral("tensorrt")) ? QStringLiteral("model.engine") : QStringLiteral("model.export.json")));
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

        const QString suffix = QFileInfo(candidateModel).suffix().toLower();
        const QString inferenceDir = QDir(outputPath).filePath(QStringLiteral("inference"));
        QDir().mkpath(inferenceDir);
        QJsonArray predictions;
        QImage overlay;
        QString error;
        QString inferenceTaskType = QStringLiteral("detection");
        if (suffix == QStringLiteral("onnx")) {
            const QString family = inferOnnxModelFamily(candidateModel);
            if (family == QStringLiteral("yolo_segmentation")) {
                inferenceTaskType = QStringLiteral("segmentation");
                DetectionInferenceOptions inferenceOptions;
                const QVector<SegmentationPrediction> segPredictions = predictSegmentationOnnxRuntime(candidateModel, imagePath, inferenceOptions, &error);
                for (const SegmentationPrediction& prediction : segPredictions) {
                    predictions.append(segmentationPredictionToJson(prediction));
                }
                overlay = renderSegmentationPredictions(imagePath, segPredictions, &error);
            } else if (family == QStringLiteral("ocr_recognition")) {
                inferenceTaskType = QStringLiteral("ocr_recognition");
                const OcrRecPrediction prediction = predictOcrRecOnnxRuntime(candidateModel, imagePath, &error);
                predictions.append(ocrRecPredictionToJson(prediction));
                overlay = renderOcrRecPrediction(imagePath, prediction, &error);
            } else if (family == QStringLiteral("ocr_detection")) {
                inferenceTaskType = QStringLiteral("ocr_detection");
                OcrDetPostprocessOptions detOptions;
                const QVector<OcrDetPrediction> detPredictions = predictOcrDetOnnxRuntime(candidateModel, imagePath, detOptions, &error);
                for (const OcrDetPrediction& prediction : detPredictions) {
                    predictions.append(ocrDetPredictionToJson(prediction));
                }
                overlay = renderOcrDetPredictions(imagePath, detPredictions, &error);
            } else {
                DetectionInferenceOptions inferenceOptions;
                const QVector<DetectionPrediction> detPredictions = predictDetectionOnnxRuntime(candidateModel, imagePath, inferenceOptions, &error);
                for (const DetectionPrediction& prediction : detPredictions) {
                    predictions.append(detectionPredictionToJson(prediction));
                }
                overlay = renderDetectionPredictions(imagePath, detPredictions, &error);
            }
        } else {
            DetectionBaselineCheckpoint checkpoint;
            if (!loadDetectionBaselineCheckpoint(candidateModel, &checkpoint, &error)) {
                failureReason = error;
                return false;
            }
            DetectionInferenceOptions inferenceOptions;
            const QVector<DetectionPrediction> detPredictions = predictDetectionBaseline(checkpoint, imagePath, inferenceOptions, &error);
            for (const DetectionPrediction& prediction : detPredictions) {
                predictions.append(detectionPredictionToJson(prediction));
            }
            overlay = renderDetectionPredictions(imagePath, detPredictions, &error);
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
        QJsonObject benchmarkOptions = options.value(QStringLiteral("benchmarkOptions")).toObject();
        benchmarkOptions.insert(QStringLiteral("datasetPath"), datasetPath);
        if (!preferredSampleImage.isEmpty()) {
            benchmarkOptions.insert(QStringLiteral("sampleImagePath"), preferredSampleImage);
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
        artifactArray.append(pathArtifact(QStringLiteral("benchmark_report"), benchmark.reportPath, QStringLiteral("Benchmark report")));
        appendStep(QStringLiteral("benchmarkModel"),
            QStringLiteral("completed"),
            QStringLiteral("Benchmark completed."),
            benchmark.reportPath,
            QJsonArray{pathArtifact(QStringLiteral("benchmark_report"), benchmark.reportPath)});
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