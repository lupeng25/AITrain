#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/OcrRecTrainer.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/SegmentationTrainer.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QStandardPaths>
#include <QThread>

using namespace worker_support;
void WorkerSession::evaluateModel(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString modelPath = payload.value(QStringLiteral("modelPath")).toString();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString taskType = payload.value(QStringLiteral("taskType")).toString(QStringLiteral("detection"));
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(modelPath).absoluteDir().absolutePath(), taskId);
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始评估模型。"));
    send(QStringLiteral("progress"), progress);

    const aitrain::WorkflowResult result = aitrain::evaluateModelReport(modelPath, datasetPath, outputPath, taskType, options);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Model evaluation report"));
    send(QStringLiteral("artifact"), artifact);
    for (const auto& item : {
             qMakePair(QStringLiteral("per_class_metrics"), QStringLiteral("perClassMetricsPath")),
             qMakePair(QStringLiteral("error_samples"), QStringLiteral("errorSamplesPath")),
             qMakePair(QStringLiteral("confusion_matrix"), QStringLiteral("confusionMatrixPath")),
             qMakePair(QStringLiteral("evaluation_summary"), QStringLiteral("evaluationSummaryPath")),
             qMakePair(QStringLiteral("evaluation_overlays"), QStringLiteral("overlayDir"))}) {
        const QString path = result.payload.value(item.second).toString();
        if (!path.isEmpty()) {
            QJsonObject extraArtifact;
            extraArtifact.insert(QStringLiteral("taskId"), taskId);
            extraArtifact.insert(QStringLiteral("kind"), item.first);
            extraArtifact.insert(QStringLiteral("path"), path);
            extraArtifact.insert(QStringLiteral("message"), QStringLiteral("Model evaluation artifact"));
            send(QStringLiteral("artifact"), extraArtifact);
        }
    }
    const QString legacyOverlaysPath = result.payload.value(QStringLiteral("overlaysPath")).toString();
    if (!legacyOverlaysPath.isEmpty()) {
        QJsonObject extraArtifact;
        extraArtifact.insert(QStringLiteral("taskId"), taskId);
        extraArtifact.insert(QStringLiteral("kind"), QStringLiteral("evaluation_overlays"));
        extraArtifact.insert(QStringLiteral("path"), legacyOverlaysPath);
        extraArtifact.insert(QStringLiteral("message"), QStringLiteral("Model evaluation artifact"));
        send(QStringLiteral("artifact"), extraArtifact);
    }
    QJsonObject progressDone;
    progressDone.insert(QStringLiteral("taskId"), taskId);
    progressDone.insert(QStringLiteral("percent"), 100);
    progressDone.insert(QStringLiteral("message"), QStringLiteral("模型评估完成。"));
    send(QStringLiteral("progress"), progressDone);
    send(QStringLiteral("evaluationReport"), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Model evaluation completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::benchmarkModel(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString modelPath = payload.value(QStringLiteral("modelPath")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(modelPath).absoluteDir().absolutePath(), taskId);
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始部署基准测试。"));
    send(QStringLiteral("progress"), progress);

    const aitrain::WorkflowResult result = aitrain::benchmarkModelReport(modelPath, outputPath, options);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("benchmark_report"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Model benchmark report"));
    send(QStringLiteral("artifact"), artifact);
    QJsonObject progressDone;
    progressDone.insert(QStringLiteral("taskId"), taskId);
    progressDone.insert(QStringLiteral("percent"), 100);
    progressDone.insert(QStringLiteral("message"), QStringLiteral("部署基准测试完成。"));
    send(QStringLiteral("progress"), progressDone);
    send(QStringLiteral("benchmarkReport"), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Model benchmark completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::generateDeliveryReport(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QDir::currentPath(), taskId);
    }
    QJsonObject context = payload.value(QStringLiteral("context")).toObject();
    context.insert(QStringLiteral("taskId"), taskId);

    const aitrain::WorkflowResult result = aitrain::generateTrainingDeliveryReport(outputPath, context);
    if (!result.ok) {
        fail(result.error);
        return;
    }
    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("training_delivery_report"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Training delivery HTML report"));
    send(QStringLiteral("artifact"), artifact);
    const QString jsonPath = result.payload.value(QStringLiteral("jsonPath")).toString();
    if (!jsonPath.isEmpty()) {
        QJsonObject jsonArtifact;
        jsonArtifact.insert(QStringLiteral("taskId"), taskId);
        jsonArtifact.insert(QStringLiteral("kind"), QStringLiteral("training_delivery_report_json"));
        jsonArtifact.insert(QStringLiteral("path"), jsonPath);
        jsonArtifact.insert(QStringLiteral("message"), QStringLiteral("Training delivery report JSON context"));
        send(QStringLiteral("artifact"), jsonArtifact);
    }
    const QString modelCardPath = result.payload.value(QStringLiteral("modelCardPath")).toString();
    if (!modelCardPath.isEmpty()) {
        QJsonObject modelCardArtifact;
        modelCardArtifact.insert(QStringLiteral("taskId"), taskId);
        modelCardArtifact.insert(QStringLiteral("kind"), QStringLiteral("model_card"));
        modelCardArtifact.insert(QStringLiteral("path"), modelCardPath);
        modelCardArtifact.insert(QStringLiteral("message"), QStringLiteral("Model card JSON"));
        send(QStringLiteral("artifact"), modelCardArtifact);
    }
    const QString inventoryPath = result.payload.value(QStringLiteral("artifactInventoryPath")).toString();
    if (!inventoryPath.isEmpty()) {
        QJsonObject inventoryArtifact;
        inventoryArtifact.insert(QStringLiteral("taskId"), taskId);
        inventoryArtifact.insert(QStringLiteral("kind"), QStringLiteral("delivery_artifact_inventory"));
        inventoryArtifact.insert(QStringLiteral("path"), inventoryPath);
        inventoryArtifact.insert(QStringLiteral("message"), QStringLiteral("Delivery artifact inventory"));
        send(QStringLiteral("artifact"), inventoryArtifact);
    }
    const QString manifestPath = result.payload.value(QStringLiteral("deliveryManifestPath")).toString();
    if (!manifestPath.isEmpty()) {
        QJsonObject manifestArtifact;
        manifestArtifact.insert(QStringLiteral("taskId"), taskId);
        manifestArtifact.insert(QStringLiteral("kind"), QStringLiteral("delivery_manifest"));
        manifestArtifact.insert(QStringLiteral("path"), manifestPath);
        manifestArtifact.insert(QStringLiteral("message"), QStringLiteral("Delivery manifest"));
        send(QStringLiteral("artifact"), manifestArtifact);
    }
    send(QStringLiteral("deliveryReport"), result.payload);
    socket_.waitForBytesWritten(1000);
    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Training delivery report generated"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::runCustomerOcrAcceptance(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QDir::currentPath(), taskId);
    }
    QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    options.insert(QStringLiteral("taskId"), taskId);

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始客户域 OCR 验收。"));
    send(QStringLiteral("progress"), progress);

    const aitrain::WorkflowResult result = aitrain::runCustomerOcrAcceptanceReport(outputPath, options);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("customer_ocr_acceptance"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Customer OCR acceptance report"));
    send(QStringLiteral("artifact"), artifact);
    const QString summaryPath = result.payload.value(QStringLiteral("summaryPath")).toString();
    if (!summaryPath.isEmpty()) {
        QJsonObject summaryArtifact;
        summaryArtifact.insert(QStringLiteral("taskId"), taskId);
        summaryArtifact.insert(QStringLiteral("kind"), QStringLiteral("customer_ocr_acceptance_summary"));
        summaryArtifact.insert(QStringLiteral("path"), summaryPath);
        summaryArtifact.insert(QStringLiteral("message"), QStringLiteral("Customer OCR acceptance summary"));
        send(QStringLiteral("artifact"), summaryArtifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("客户域 OCR 验收报告已生成。"));
    send(QStringLiteral("progress"), doneProgress);
    send(QStringLiteral("customerOcrAcceptance"), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Customer OCR acceptance completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::collectDiagnostics(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QDir::currentPath(), taskId);
    }
    QJsonObject context = payload.value(QStringLiteral("context")).toObject();
    context.insert(QStringLiteral("taskId"), taskId);
    if (context.value(QStringLiteral("workerExecutable")).toString().isEmpty()) {
        context.insert(QStringLiteral("workerExecutable"), QCoreApplication::applicationFilePath());
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始收集诊断包。"));
    send(QStringLiteral("progress"), progress);

    const aitrain::WorkflowResult result = aitrain::collectDiagnosticsReport(outputPath, context);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    for (const auto& item : {
             qMakePair(QStringLiteral("diagnostic_manifest"), QStringLiteral("manifestPath")),
             qMakePair(QStringLiteral("diagnostic_bundle"), QStringLiteral("bundlePath")),
             qMakePair(QStringLiteral("diagnostic_summary"), QStringLiteral("summaryPath"))}) {
        const QString path = result.payload.value(item.second).toString();
        if (path.isEmpty()) {
            continue;
        }
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), item.first);
        artifact.insert(QStringLiteral("path"), path);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Diagnostic bundle artifact"));
        send(QStringLiteral("artifact"), artifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("诊断包已生成。"));
    send(QStringLiteral("progress"), doneProgress);
    send(QStringLiteral("diagnosticBundle"), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Diagnostics collected"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::validateDeploymentArtifact(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString modelPath = payload.value(QStringLiteral("modelPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(modelPath).absoluteDir().absolutePath(), taskId);
    }
    QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    const QString sampleImagePath = payload.value(QStringLiteral("sampleImagePath")).toString();
    if (!sampleImagePath.isEmpty()) {
        options.insert(QStringLiteral("sampleImagePath"), sampleImagePath);
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始验证部署产物。"));
    send(QStringLiteral("progress"), progress);

    const aitrain::WorkflowResult result = aitrain::validateDeploymentArtifactReport(modelPath, outputPath, format, options);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    for (const auto& item : {
             qMakePair(QStringLiteral("deployment_validation_report"), QStringLiteral("reportPath")),
             qMakePair(QStringLiteral("deployment_validation_summary"), QStringLiteral("summaryPath")),
             qMakePair(QStringLiteral("deployment_predictions"), QStringLiteral("predictionsPath")),
             qMakePair(QStringLiteral("deployment_overlay"), QStringLiteral("overlayPath"))}) {
        const QString path = result.payload.value(item.second).toString();
        if (path.isEmpty()) {
            continue;
        }
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), item.first);
        artifact.insert(QStringLiteral("path"), path);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Deployment validation artifact"));
        send(QStringLiteral("artifact"), artifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("部署产物验证已完成。"));
    send(QStringLiteral("progress"), doneProgress);
    send(QStringLiteral("deploymentValidation"), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Deployment validation completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::exportModel(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString checkpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString(QStringLiteral("tiny_detector_json"));

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始导出模型。"));
    send(QStringLiteral("progress"), startProgress);

    const aitrain::DetectionExportResult result = aitrain::exportDetectionCheckpoint(checkpointPath, outputPath, format);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("模型导出完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("export"));
    artifact.insert(QStringLiteral("path"), result.exportPath);
    QString artifactMessage = QStringLiteral("Tiny detector JSON scaffold export");
    if (result.format == QStringLiteral("onnx")) {
        artifactMessage = QStringLiteral("ONNX model export");
    } else if (result.format == QStringLiteral("ncnn")) {
        artifactMessage = QStringLiteral("NCNN param export");
    } else if (result.format.startsWith(QStringLiteral("tensorrt"))) {
        artifactMessage = QStringLiteral("TensorRT engine export");
    }
    artifact.insert(QStringLiteral("message"), artifactMessage);
    send(QStringLiteral("artifact"), artifact);

    const QJsonObject ncnnConfig = result.config.value(QStringLiteral("ncnn")).toObject();
    const QString ncnnBinPath = ncnnConfig.value(QStringLiteral("binPath")).toString();
    if (result.format == QStringLiteral("ncnn") && !ncnnBinPath.isEmpty()) {
        QJsonObject binArtifact;
        binArtifact.insert(QStringLiteral("taskId"), taskId);
        binArtifact.insert(QStringLiteral("kind"), QStringLiteral("export_sidecar"));
        binArtifact.insert(QStringLiteral("path"), ncnnBinPath);
        binArtifact.insert(QStringLiteral("message"), QStringLiteral("NCNN binary weights"));
        send(QStringLiteral("artifact"), binArtifact);
    }

    QJsonObject response;
    response.insert(QStringLiteral("ok"), true);
    response.insert(QStringLiteral("format"), result.format);
    response.insert(QStringLiteral("taskId"), taskId);
    response.insert(QStringLiteral("checkpointPath"), checkpointPath);
    response.insert(QStringLiteral("exportPath"), result.exportPath);
    response.insert(QStringLiteral("reportPath"), result.reportPath);
    response.insert(QStringLiteral("config"), result.config);
    response.insert(QStringLiteral("exportedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("modelExport"), response);

    QJsonObject completed;
    completed.insert(QStringLiteral("message"), QStringLiteral("Model export completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::runInference(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString checkpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
    const QString imagePath = payload.value(QStringLiteral("imagePath")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    aitrain::DetectionInferenceOptions options;
    options.confidenceThreshold = payload.value(QStringLiteral("confidenceThreshold")).toDouble(options.confidenceThreshold);
    options.iouThreshold = payload.value(QStringLiteral("iouThreshold")).toDouble(options.iouThreshold);
    options.maxDetections = payload.value(QStringLiteral("maxDetections")).toInt(options.maxDetections);
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("inference"));
    }
    if (!QDir().mkpath(outputPath)) {
        fail(QStringLiteral("Cannot create inference output directory: %1").arg(outputPath));
        return;
    }

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始推理。"));
    send(QStringLiteral("progress"), startProgress);

    QElapsedTimer elapsed;
    elapsed.start();
    QString error;
    QJsonArray predictionArray;
    QImage overlay;
    QString taskType = QStringLiteral("detection");
    int predictionCount = 0;
    const QString modelSuffix = QFileInfo(checkpointPath).suffix().toLower();
    const bool onnxModel = modelSuffix == QStringLiteral("onnx");
    const bool tensorRtModel = modelSuffix == QStringLiteral("engine") || modelSuffix == QStringLiteral("plan");
    if (onnxModel) {
        const QString modelFamily = aitrain::inferOnnxModelFamily(checkpointPath);
        if (modelFamily == QStringLiteral("yolo_segmentation")) {
            taskType = QStringLiteral("segmentation");
            const QVector<aitrain::SegmentationPrediction> predictions = aitrain::predictSegmentationOnnxRuntime(checkpointPath, imagePath, options, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            for (const aitrain::SegmentationPrediction& prediction : predictions) {
                predictionArray.append(aitrain::segmentationPredictionToJson(prediction));
            }
            overlay = aitrain::renderSegmentationPredictions(imagePath, predictions, &error);
            predictionCount = predictions.size();
        } else if (modelFamily == QStringLiteral("ocr_recognition")) {
            taskType = QStringLiteral("ocr_recognition");
            const aitrain::OcrRecPrediction prediction = aitrain::predictOcrRecOnnxRuntime(checkpointPath, imagePath, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            predictionArray.append(aitrain::ocrRecPredictionToJson(prediction));
            overlay = aitrain::renderOcrRecPrediction(imagePath, prediction, &error);
            predictionCount = 1;
        } else if (modelFamily == QStringLiteral("ocr_detection")) {
            taskType = QStringLiteral("ocr_detection");
            aitrain::OcrDetPostprocessOptions detOptions;
            detOptions.binaryThreshold = payload.value(QStringLiteral("binaryThreshold")).toDouble(detOptions.binaryThreshold);
            detOptions.boxThreshold = payload.value(QStringLiteral("boxThreshold")).toDouble(detOptions.boxThreshold);
            detOptions.minArea = payload.value(QStringLiteral("minArea")).toInt(detOptions.minArea);
            detOptions.maxDetections = payload.value(QStringLiteral("maxDetections")).toInt(detOptions.maxDetections);
            const QVector<aitrain::OcrDetPrediction> predictions = aitrain::predictOcrDetOnnxRuntime(checkpointPath, imagePath, detOptions, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            for (const aitrain::OcrDetPrediction& prediction : predictions) {
                predictionArray.append(aitrain::ocrDetPredictionToJson(prediction));
            }
            overlay = aitrain::renderOcrDetPredictions(imagePath, predictions, &error);
            predictionCount = predictions.size();
        } else {
            const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(checkpointPath, imagePath, options, &error);
            if (!error.isEmpty()) {
                fail(error);
                return;
            }
            for (const aitrain::DetectionPrediction& prediction : predictions) {
                predictionArray.append(aitrain::detectionPredictionToJson(prediction));
            }
            overlay = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
            predictionCount = predictions.size();
        }
    } else if (tensorRtModel) {
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionTensorRt(checkpointPath, imagePath, options, &error);
        if (!error.isEmpty()) {
            fail(error);
            return;
        }
        for (const aitrain::DetectionPrediction& prediction : predictions) {
            predictionArray.append(aitrain::detectionPredictionToJson(prediction));
        }
        overlay = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
        predictionCount = predictions.size();
    } else {
        aitrain::DetectionBaselineCheckpoint checkpoint;
        if (!aitrain::loadDetectionBaselineCheckpoint(checkpointPath, &checkpoint, &error)) {
            fail(error);
            return;
        }
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(checkpoint, imagePath, options, &error);
        if (!error.isEmpty()) {
            fail(error);
            return;
        }
        for (const aitrain::DetectionPrediction& prediction : predictions) {
            predictionArray.append(aitrain::detectionPredictionToJson(prediction));
        }
        overlay = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
        predictionCount = predictions.size();
    }
    if (overlay.isNull()) {
        fail(error);
        return;
    }

    const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
    QFile predictionsFile(predictionsPath);
    if (!predictionsFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        fail(QStringLiteral("Cannot write inference predictions: %1").arg(predictionsPath));
        return;
    }
    QJsonObject predictionsDocument;
    predictionsDocument.insert(QStringLiteral("taskId"), taskId);
    predictionsDocument.insert(QStringLiteral("checkpointPath"), checkpointPath);
    predictionsDocument.insert(QStringLiteral("imagePath"), imagePath);
    predictionsDocument.insert(QStringLiteral("taskType"), taskType);
    predictionsDocument.insert(QStringLiteral("runtime"), onnxModel
        ? QStringLiteral("onnxruntime")
        : (tensorRtModel ? QStringLiteral("tensorrt") : QStringLiteral("tiny_detector")));
    predictionsDocument.insert(QStringLiteral("elapsedMs"), static_cast<int>(elapsed.elapsed()));
    predictionsDocument.insert(QStringLiteral("postprocess"), QJsonObject{
        {QStringLiteral("confidenceThreshold"), options.confidenceThreshold},
        {QStringLiteral("iouThreshold"), options.iouThreshold},
        {QStringLiteral("maxDetections"), options.maxDetections}
    });
    predictionsDocument.insert(QStringLiteral("predictions"), predictionArray);
    predictionsFile.write(QJsonDocument(predictionsDocument).toJson(QJsonDocument::Indented));
    predictionsFile.close();

    QJsonObject renderLog;
    renderLog.insert(QStringLiteral("message"), QStringLiteral("Rendering inference overlay."));
    send(QStringLiteral("log"), renderLog);
    QJsonObject saveLog;
    saveLog.insert(QStringLiteral("message"), QStringLiteral("Saving inference overlay."));
    send(QStringLiteral("log"), saveLog);
    const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
    if (!overlay.save(overlayPath)) {
        fail(QStringLiteral("Cannot write inference overlay: %1").arg(overlayPath));
        return;
    }
    const int elapsedMs = static_cast<int>(elapsed.elapsed());

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("推理完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject predictionsArtifact;
    predictionsArtifact.insert(QStringLiteral("taskId"), taskId);
    predictionsArtifact.insert(QStringLiteral("kind"), QStringLiteral("inference_predictions"));
    predictionsArtifact.insert(QStringLiteral("path"), predictionsPath);
    predictionsArtifact.insert(QStringLiteral("message"), QStringLiteral("Inference predictions"));
    send(QStringLiteral("artifact"), predictionsArtifact);

    QJsonObject overlayArtifact;
    overlayArtifact.insert(QStringLiteral("taskId"), taskId);
    overlayArtifact.insert(QStringLiteral("kind"), QStringLiteral("inference_overlay"));
    overlayArtifact.insert(QStringLiteral("path"), overlayPath);
    overlayArtifact.insert(QStringLiteral("message"), QStringLiteral("Inference overlay"));
    send(QStringLiteral("artifact"), overlayArtifact);

    QJsonObject response;
    response.insert(QStringLiteral("ok"), true);
    response.insert(QStringLiteral("taskId"), taskId);
    response.insert(QStringLiteral("checkpointPath"), checkpointPath);
    response.insert(QStringLiteral("imagePath"), imagePath);
    response.insert(QStringLiteral("taskType"), taskType);
    response.insert(QStringLiteral("predictionsPath"), predictionsPath);
    response.insert(QStringLiteral("overlayPath"), overlayPath);
    response.insert(QStringLiteral("elapsedMs"), elapsedMs);
    response.insert(QStringLiteral("predictionCount"), predictionCount);
    response.insert(QStringLiteral("finishedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("inferenceResult"), response);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Inference completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

