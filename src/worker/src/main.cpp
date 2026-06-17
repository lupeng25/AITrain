#include "WorkerSession.h"

#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/PluginManager.h"
#include "aitrain/core/PluginMarketplace.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/core/VisionPostprocess.h"

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>

namespace {

void writeJsonLine(const QJsonObject& object)
{
    QTextStream stream(stdout);
    stream << QString::fromUtf8(QJsonDocument(object).toJson(QJsonDocument::Compact)) << QLatin1Char('\n');
    stream.flush();
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error)
{
    const QFileInfo info(path);
    if (!QDir().mkpath(info.absolutePath())) {
        if (error) {
            *error = QStringLiteral("Cannot create output directory: %1").arg(info.absolutePath());
        }
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write JSON file: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

int runSelfCheck()
{
    QJsonArray checks;
    const QVector<aitrain::RuntimeDependencyCheck> runtimeChecks =
        aitrain::defaultRuntimeDependencyChecks(QCoreApplication::applicationDirPath());
    bool hasMissing = false;
    bool hasWarning = false;
    for (const aitrain::RuntimeDependencyCheck& check : runtimeChecks) {
        checks.append(check.toJson());
        hasMissing = hasMissing || check.status == QStringLiteral("missing");
        hasWarning = hasWarning || check.status == QStringLiteral("warning");
    }

    QJsonObject result;
    result.insert(QStringLiteral("ok"), true);
    result.insert(QStringLiteral("status"), hasMissing
        ? QStringLiteral("missing")
        : (hasWarning ? QStringLiteral("warning") : QStringLiteral("ok")));
    result.insert(QStringLiteral("applicationDir"), QCoreApplication::applicationDirPath());
    result.insert(QStringLiteral("ncnnBackend"), aitrain::ncnnBackendStatus().toJson());
    result.insert(QStringLiteral("tensorRtBackend"), aitrain::tensorRtBackendStatus().toJson());
    result.insert(QStringLiteral("checks"), checks);
    writeJsonLine(result);
    return 0;
}

int runPluginSmoke(const QString& pluginDirectory)
{
    aitrain::PluginManager manager;
    manager.scan(QStringList() << pluginDirectory);

    QJsonArray pluginArray;
    QStringList pluginIds;
    for (aitrain::IModelPlugin* plugin : manager.plugins()) {
        if (!plugin) {
            continue;
        }
        const aitrain::PluginManifest manifest = plugin->manifest();
        pluginIds.append(manifest.id);
        pluginArray.append(manifest.toJson());
    }

    const QStringList requiredIds = {
        QStringLiteral("com.aitrain.plugins.dataset_interop"),
        QStringLiteral("com.aitrain.plugins.yolo_native"),
        QStringLiteral("com.aitrain.plugins.semantic_segmentation"),
        QStringLiteral("com.aitrain.plugins.ocr_rec_native")
    };
    QStringList missingIds;
    for (const QString& requiredId : requiredIds) {
        if (!pluginIds.contains(requiredId)) {
            missingIds.append(requiredId);
        }
    }

    QJsonObject result;
    result.insert(QStringLiteral("ok"), missingIds.isEmpty());
    result.insert(QStringLiteral("pluginDirectory"), QFileInfo(pluginDirectory).absoluteFilePath());
    result.insert(QStringLiteral("pluginCount"), pluginArray.size());
    result.insert(QStringLiteral("plugins"), pluginArray);
    result.insert(QStringLiteral("errors"), QJsonArray::fromStringList(manager.errors()));
    result.insert(QStringLiteral("missingRequiredPlugins"), QJsonArray::fromStringList(missingIds));

    const QString marketplaceRoot = QDir(QFileInfo(pluginDirectory).absolutePath()).filePath(QStringLiteral("marketplace"));
    aitrain::PluginMarketplace marketplace(marketplaceRoot, QFileInfo(pluginDirectory).absoluteFilePath());
    aitrain::PluginMarketplaceReport marketplaceReport;
    const QVector<aitrain::InstalledPluginRecord> installed = marketplace.installedPlugins(&marketplaceReport);
    QJsonArray installedArray;
    for (const aitrain::InstalledPluginRecord& record : installed) {
        installedArray.append(record.toJson());
    }
    result.insert(QStringLiteral("marketplaceRoot"), QFileInfo(marketplaceRoot).absoluteFilePath());
    result.insert(QStringLiteral("marketplaceStatePath"), marketplace.statePath());
    result.insert(QStringLiteral("marketplaceInstalledPlugins"), installedArray);
    result.insert(QStringLiteral("marketplaceState"), marketplaceReport.toJson());
    writeJsonLine(result);
    return missingIds.isEmpty() ? 0 : 4;
}

int runTensorRtSmoke(const QString& onnxPath)
{
    const QString modelPath = QFileInfo(onnxPath).absoluteFilePath();
    if (!QFileInfo::exists(modelPath) || QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) != 0) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("TensorRT smoke requires an existing official ONNX model artifact: %1").arg(modelPath)}
        });
        return 5;
    }

    const QString outputRoot = QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("aitrain_tensorrt_smoke"));
    const QString exportPath = QDir(outputRoot).filePath(QStringLiteral("export/model.engine"));
    const aitrain::DetectionExportResult exported =
        aitrain::exportDetectionCheckpoint(modelPath, exportPath, QStringLiteral("tensorrt"));
    if (!exported.ok) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("export")},
            {QStringLiteral("onnxPath"), modelPath},
            {QStringLiteral("error"), exported.error},
            {QStringLiteral("tensorRtBackend"), aitrain::tensorRtBackendStatus().toJson()}
        });
        return 7;
    }

    writeJsonLine(QJsonObject{
        {QStringLiteral("ok"), true},
        {QStringLiteral("stage"), QStringLiteral("export_completed")},
        {QStringLiteral("onnxPath"), modelPath},
        {QStringLiteral("enginePath"), exported.exportPath},
        {QStringLiteral("reportPath"), exported.reportPath},
        {QStringLiteral("tensorRtBackend"), aitrain::tensorRtBackendStatus().toJson()}
    });
    return 0;
}

int runNcnnSmoke(
    const QString& onnxPath,
    const QString& imagePath,
    const QString& outputDirectory,
    const QString& taskType)
{
    const QString modelPath = QFileInfo(onnxPath).absoluteFilePath();
    const QString samplePath = QFileInfo(imagePath).absoluteFilePath();
    const QString outputPath = QFileInfo(outputDirectory.isEmpty()
        ? QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("aitrain_ncnn_smoke"))
        : outputDirectory).absoluteFilePath();
    const QString exportPath = QDir(outputPath).filePath(QStringLiteral("export/model.param"));
    const QString validationPath = QDir(outputPath).filePath(QStringLiteral("deployment-validation"));

    if (!QFileInfo::exists(modelPath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("NCNN smoke requires an existing ONNX model: %1").arg(modelPath)}
        });
        return 9;
    }
    if (samplePath.isEmpty() || !QFileInfo::exists(samplePath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("NCNN smoke requires an existing sample image: %1").arg(samplePath)}
        });
        return 9;
    }

    const aitrain::DetectionExportResult exported =
        aitrain::exportDetectionCheckpoint(modelPath, exportPath, QStringLiteral("ncnn"));
    if (!exported.ok) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("export")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("onnxPath"), modelPath},
            {QStringLiteral("error"), exported.error},
            {QStringLiteral("ncnnBackend"), aitrain::ncnnBackendStatus().toJson()}
        });
        return 10;
    }

    QJsonObject options;
    options.insert(QStringLiteral("sampleImagePath"), samplePath);
    if (taskType == QStringLiteral("segmentation")) {
        options.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_segmentation"));
    } else {
        options.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_detection"));
    }
    const aitrain::WorkflowResult validation =
        aitrain::validateDeploymentArtifactReport(exported.exportPath, validationPath, QStringLiteral("ncnn"), options);
    const QString status = validation.payload.value(QStringLiteral("status")).toString(QStringLiteral("failed"));
    const bool ok = validation.ok && status == QStringLiteral("passed");
    writeJsonLine(QJsonObject{
        {QStringLiteral("ok"), ok},
        {QStringLiteral("stage"), ok ? QStringLiteral("completed") : QStringLiteral("validation")},
        {QStringLiteral("status"), status},
        {QStringLiteral("onnxPath"), modelPath},
        {QStringLiteral("paramPath"), exported.exportPath},
        {QStringLiteral("exportReportPath"), exported.reportPath},
        {QStringLiteral("validationReportPath"), validation.reportPath},
        {QStringLiteral("validation"), validation.payload},
        {QStringLiteral("ncnnBackend"), aitrain::ncnnBackendStatus().toJson()}
    });
    return ok ? 0 : 11;
}

int runNcnnParamSmoke(
    const QString& paramPath,
    const QString& imagePath,
    const QString& outputDirectory,
    const QString& taskType)
{
    const QString modelPath = QFileInfo(paramPath).absoluteFilePath();
    const QString samplePath = QFileInfo(imagePath).absoluteFilePath();
    const QString outputPath = QFileInfo(outputDirectory.isEmpty()
        ? QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("aitrain_ncnn_param_smoke"))
        : outputDirectory).absoluteFilePath();
    const QString validationPath = QDir(outputPath).filePath(QStringLiteral("deployment-validation"));

    if (!QFileInfo::exists(modelPath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("NCNN param smoke requires an existing param model: %1").arg(modelPath)}
        });
        return 9;
    }
    if (samplePath.isEmpty() || !QFileInfo::exists(samplePath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("NCNN param smoke requires an existing sample image: %1").arg(samplePath)}
        });
        return 9;
    }

    QJsonObject options;
    options.insert(QStringLiteral("sampleImagePath"), samplePath);
    if (taskType == QStringLiteral("segmentation")) {
        options.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_segmentation"));
    } else {
        options.insert(QStringLiteral("modelFamily"), QStringLiteral("yolo_detection"));
    }

    const aitrain::WorkflowResult validation =
        aitrain::validateDeploymentArtifactReport(modelPath, validationPath, QStringLiteral("ncnn"), options);
    const QString status = validation.payload.value(QStringLiteral("status")).toString(QStringLiteral("failed"));
    const bool ok = validation.ok && status == QStringLiteral("passed");
    writeJsonLine(QJsonObject{
        {QStringLiteral("ok"), ok},
        {QStringLiteral("stage"), ok ? QStringLiteral("completed") : QStringLiteral("validation")},
        {QStringLiteral("status"), status},
        {QStringLiteral("paramPath"), modelPath},
        {QStringLiteral("validationReportPath"), validation.reportPath},
        {QStringLiteral("validation"), validation.payload},
        {QStringLiteral("ncnnBackend"), aitrain::ncnnBackendStatus().toJson()}
    });
    return ok ? 0 : 11;
}

int runSemanticOnnxSmoke(
    const QString& onnxPath,
    const QString& imagePath,
    const QString& outputDirectory)
{
    const QString modelPath = QFileInfo(onnxPath).absoluteFilePath();
    const QString samplePath = QFileInfo(imagePath).absoluteFilePath();
    const QString outputPath = QFileInfo(outputDirectory.isEmpty()
        ? QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("aitrain_semantic_onnx_smoke"))
        : outputDirectory).absoluteFilePath();
    const QString inferencePath = QDir(outputPath).filePath(QStringLiteral("inference"));
    const QString benchmarkPath = QDir(outputPath).filePath(QStringLiteral("benchmark"));
    const QString deploymentPath = QDir(outputPath).filePath(QStringLiteral("deployment-validation"));

    if (!QFileInfo::exists(modelPath) || QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) != 0) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("Semantic ONNX smoke requires an existing .onnx model: %1").arg(modelPath)}
        });
        return 12;
    }
    if (samplePath.isEmpty() || !QFileInfo::exists(samplePath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("Semantic ONNX smoke requires an existing sample image: %1").arg(samplePath)}
        });
        return 12;
    }

    QString familyWarning;
    const QString modelFamily = aitrain::inferOnnxModelFamily(modelPath, &familyWarning);
    if (modelFamily != QStringLiteral("semantic_segmentation")) {
        QJsonObject result{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("model-family")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("onnxPath"), modelPath},
            {QStringLiteral("modelFamily"), modelFamily},
            {QStringLiteral("error"), QStringLiteral("Semantic ONNX smoke requires modelFamily=semantic_segmentation.")}
        };
        if (!familyWarning.isEmpty()) {
            result.insert(QStringLiteral("modelFamilyWarning"), familyWarning);
        }
        writeJsonLine(result);
        return 13;
    }

    if (!QDir().mkpath(inferencePath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("output")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("error"), QStringLiteral("Cannot create semantic ONNX smoke output directory: %1").arg(inferencePath)}
        });
        return 14;
    }

    QElapsedTimer timer;
    timer.start();
    QString error;
    const aitrain::SemanticSegmentationPrediction prediction =
        aitrain::predictSemanticSegmentationOnnxRuntime(modelPath, samplePath, &error);
    QImage overlay;
    if (error.isEmpty()) {
        overlay = aitrain::renderSemanticSegmentationPrediction(samplePath, prediction, &error);
    }
    const int elapsedMs = static_cast<int>(timer.elapsed());
    if (!error.isEmpty()) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("inference")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("onnxPath"), modelPath},
            {QStringLiteral("imagePath"), samplePath},
            {QStringLiteral("error"), error}
        });
        return 15;
    }

    const QString predictionsPath = QDir(inferencePath).filePath(QStringLiteral("inference_predictions.json"));
    const QString overlayPath = QDir(inferencePath).filePath(QStringLiteral("inference_overlay.png"));
    QJsonArray predictions;
    predictions.append(aitrain::semanticSegmentationPredictionToJson(prediction));
    QJsonObject predictionsDocument{
        {QStringLiteral("ok"), true},
        {QStringLiteral("checkpointPath"), modelPath},
        {QStringLiteral("imagePath"), samplePath},
        {QStringLiteral("taskType"), QStringLiteral("semantic_segmentation")},
        {QStringLiteral("runtime"), QStringLiteral("onnxruntime")},
        {QStringLiteral("elapsedMs"), elapsedMs},
        {QStringLiteral("predictions"), predictions}
    };
    if (!familyWarning.isEmpty()) {
        predictionsDocument.insert(QStringLiteral("modelFamilyWarning"), familyWarning);
    }
    if (!writeJsonFile(predictionsPath, predictionsDocument, &error)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("inference-output")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("error"), error}
        });
        return 16;
    }
    if (overlay.isNull() || !overlay.save(overlayPath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("overlay")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("error"), QStringLiteral("Cannot write semantic ONNX overlay: %1").arg(overlayPath)}
        });
        return 17;
    }

    QJsonObject benchmarkOptions{
        {QStringLiteral("runtime"), QStringLiteral("onnxruntime")},
        {QStringLiteral("sampleImagePath"), samplePath},
        {QStringLiteral("warmupIterations"), 2},
        {QStringLiteral("iterations"), 10},
        {QStringLiteral("device"), QStringLiteral("cpu")}
    };
    const aitrain::WorkflowResult benchmark =
        aitrain::benchmarkModelReport(modelPath, benchmarkPath, benchmarkOptions);

    QJsonObject deploymentOptions{
        {QStringLiteral("sampleImagePath"), samplePath},
        {QStringLiteral("modelFamily"), QStringLiteral("semantic_segmentation")}
    };
    const aitrain::WorkflowResult deployment =
        aitrain::validateDeploymentArtifactReport(modelPath, deploymentPath, QStringLiteral("onnx"), deploymentOptions);

    const bool benchmarkUsable = benchmark.ok
        && benchmark.payload.value(QStringLiteral("runtimeUsable")).toBool(false)
        && benchmark.payload.value(QStringLiteral("timedInference")).toBool(false);
    const QString deploymentStatus = deployment.payload.value(QStringLiteral("status")).toString(QStringLiteral("failed"));
    const bool deploymentOk = deployment.ok && deploymentStatus == QStringLiteral("passed");
    const bool ok = benchmarkUsable && deploymentOk;
    const QString summaryPath = QDir(outputPath).filePath(QStringLiteral("smp_semantic_onnx_smoke_summary.json"));
    QJsonObject summary{
        {QStringLiteral("ok"), ok},
        {QStringLiteral("stage"), ok ? QStringLiteral("completed") : QStringLiteral("validation")},
        {QStringLiteral("status"), ok ? QStringLiteral("passed") : QStringLiteral("failed")},
        {QStringLiteral("onnxPath"), modelPath},
        {QStringLiteral("imagePath"), samplePath},
        {QStringLiteral("modelFamily"), modelFamily},
        {QStringLiteral("predictionsPath"), predictionsPath},
        {QStringLiteral("overlayPath"), overlayPath},
        {QStringLiteral("benchmarkReportPath"), benchmark.reportPath},
        {QStringLiteral("deploymentReportPath"), deployment.reportPath},
        {QStringLiteral("benchmark"), benchmark.payload},
        {QStringLiteral("deployment"), deployment.payload}
    };
    if (!familyWarning.isEmpty()) {
        summary.insert(QStringLiteral("modelFamilyWarning"), familyWarning);
    }
    QString summaryError;
    if (!writeJsonFile(summaryPath, summary, &summaryError)) {
        summary.insert(QStringLiteral("summaryWriteError"), summaryError);
    } else {
        summary.insert(QStringLiteral("summaryPath"), summaryPath);
    }
    writeJsonLine(summary);
    return ok ? 0 : 18;
}

int runObbOnnxSmoke(
    const QString& onnxPath,
    const QString& imagePath,
    const QString& outputDirectory)
{
    const QString modelPath = QFileInfo(onnxPath).absoluteFilePath();
    const QString samplePath = QFileInfo(imagePath).absoluteFilePath();
    const QString outputPath = QFileInfo(outputDirectory.isEmpty()
        ? QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("aitrain_obb_onnx_smoke"))
        : outputDirectory).absoluteFilePath();
    const QString inferencePath = QDir(outputPath).filePath(QStringLiteral("inference"));
    const QString benchmarkPath = QDir(outputPath).filePath(QStringLiteral("benchmark"));
    const QString deploymentPath = QDir(outputPath).filePath(QStringLiteral("deployment-validation"));

    if (!QFileInfo::exists(modelPath) || QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) != 0) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("OBB ONNX smoke requires an existing .onnx model: %1").arg(modelPath)}
        });
        return 19;
    }
    if (samplePath.isEmpty() || !QFileInfo::exists(samplePath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("input")},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("error"), QStringLiteral("OBB ONNX smoke requires an existing sample image: %1").arg(samplePath)}
        });
        return 19;
    }

    QString familyWarning;
    const QString modelFamily = aitrain::inferOnnxModelFamily(modelPath, &familyWarning);
    if (modelFamily != QStringLiteral("yolo_obb")) {
        QJsonObject result{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("model-family")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("onnxPath"), modelPath},
            {QStringLiteral("modelFamily"), modelFamily},
            {QStringLiteral("error"), QStringLiteral("OBB ONNX smoke requires modelFamily=yolo_obb from sidecar or training report.")}
        };
        if (!familyWarning.isEmpty()) {
            result.insert(QStringLiteral("modelFamilyWarning"), familyWarning);
        }
        writeJsonLine(result);
        return 20;
    }

    if (!QDir().mkpath(inferencePath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("output")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("error"), QStringLiteral("Cannot create OBB ONNX smoke output directory: %1").arg(inferencePath)}
        });
        return 21;
    }

    QElapsedTimer timer;
    timer.start();
    QString error;
    aitrain::DetectionInferenceOptions options;
    const QVector<aitrain::ObbPrediction> predictions =
        aitrain::predictObbOnnxRuntime(modelPath, samplePath, options, &error);
    QImage overlay;
    if (error.isEmpty()) {
        overlay = aitrain::renderObbPredictions(samplePath, predictions, &error);
    }
    const int elapsedMs = static_cast<int>(timer.elapsed());
    if (!error.isEmpty()) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("inference")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("onnxPath"), modelPath},
            {QStringLiteral("imagePath"), samplePath},
            {QStringLiteral("error"), error}
        });
        return 22;
    }

    const QString predictionsPath = QDir(inferencePath).filePath(QStringLiteral("inference_predictions.json"));
    const QString overlayPath = QDir(inferencePath).filePath(QStringLiteral("inference_overlay.png"));
    QJsonArray predictionArray;
    for (const aitrain::ObbPrediction& prediction : predictions) {
        predictionArray.append(aitrain::obbPredictionToJson(prediction));
    }
    QJsonObject predictionsDocument{
        {QStringLiteral("ok"), true},
        {QStringLiteral("checkpointPath"), modelPath},
        {QStringLiteral("imagePath"), samplePath},
        {QStringLiteral("taskType"), QStringLiteral("obb_detection")},
        {QStringLiteral("runtime"), QStringLiteral("onnxruntime")},
        {QStringLiteral("elapsedMs"), elapsedMs},
        {QStringLiteral("predictions"), predictionArray}
    };
    if (!familyWarning.isEmpty()) {
        predictionsDocument.insert(QStringLiteral("modelFamilyWarning"), familyWarning);
    }
    if (!writeJsonFile(predictionsPath, predictionsDocument, &error)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("inference-output")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("error"), error}
        });
        return 23;
    }
    if (overlay.isNull() || !overlay.save(overlayPath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("overlay")},
            {QStringLiteral("status"), QStringLiteral("failed")},
            {QStringLiteral("error"), QStringLiteral("Cannot write OBB ONNX overlay: %1").arg(overlayPath)}
        });
        return 24;
    }

    QJsonObject benchmarkOptions{
        {QStringLiteral("runtime"), QStringLiteral("onnxruntime")},
        {QStringLiteral("sampleImagePath"), samplePath},
        {QStringLiteral("warmupIterations"), 2},
        {QStringLiteral("iterations"), 10},
        {QStringLiteral("device"), QStringLiteral("cpu")}
    };
    const aitrain::WorkflowResult benchmark =
        aitrain::benchmarkModelReport(modelPath, benchmarkPath, benchmarkOptions);

    QJsonObject deploymentOptions{
        {QStringLiteral("sampleImagePath"), samplePath},
        {QStringLiteral("modelFamily"), QStringLiteral("yolo_obb")}
    };
    const aitrain::WorkflowResult deployment =
        aitrain::validateDeploymentArtifactReport(modelPath, deploymentPath, QStringLiteral("onnx"), deploymentOptions);

    const bool benchmarkUsable = benchmark.ok
        && benchmark.payload.value(QStringLiteral("runtimeUsable")).toBool(false)
        && benchmark.payload.value(QStringLiteral("timedInference")).toBool(false);
    const QString deploymentStatus = deployment.payload.value(QStringLiteral("status")).toString(QStringLiteral("failed"));
    const bool deploymentOk = deployment.ok && deploymentStatus == QStringLiteral("passed");
    const bool ok = benchmarkUsable && deploymentOk;
    const QString summaryPath = QDir(outputPath).filePath(QStringLiteral("obb_onnx_smoke_summary.json"));
    QJsonObject summary{
        {QStringLiteral("ok"), ok},
        {QStringLiteral("stage"), ok ? QStringLiteral("completed") : QStringLiteral("validation")},
        {QStringLiteral("status"), ok ? QStringLiteral("passed") : QStringLiteral("failed")},
        {QStringLiteral("onnxPath"), modelPath},
        {QStringLiteral("imagePath"), samplePath},
        {QStringLiteral("modelFamily"), modelFamily},
        {QStringLiteral("predictionsPath"), predictionsPath},
        {QStringLiteral("overlayPath"), overlayPath},
        {QStringLiteral("predictionCount"), predictionArray.size()},
        {QStringLiteral("benchmarkReportPath"), benchmark.reportPath},
        {QStringLiteral("deploymentReportPath"), deployment.reportPath},
        {QStringLiteral("benchmark"), benchmark.payload},
        {QStringLiteral("deployment"), deployment.payload}
    };
    if (!familyWarning.isEmpty()) {
        summary.insert(QStringLiteral("modelFamilyWarning"), familyWarning);
    }
    QString summaryError;
    if (!writeJsonFile(summaryPath, summary, &summaryError)) {
        summary.insert(QStringLiteral("summaryWriteError"), summaryError);
    } else {
        summary.insert(QStringLiteral("summaryPath"), summaryPath);
    }
    writeJsonLine(summary);
    return ok ? 0 : 25;
}

} // namespace

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("aitrain_worker"));

    QCommandLineParser parser;
    parser.addHelpOption();
    QCommandLineOption serverOption(QStringLiteral("server"), QStringLiteral("QLocalServer name."), QStringLiteral("name"));
    QCommandLineOption selfCheckOption(QStringLiteral("self-check"), QStringLiteral("Run package/runtime self-check and print JSON."));
    QCommandLineOption pluginSmokeOption(QStringLiteral("plugin-smoke"), QStringLiteral("Scan model plugin directory and print JSON."), QStringLiteral("directory"));
    QCommandLineOption tensorRtSmokeOption(QStringLiteral("tensorrt-smoke"), QStringLiteral("Run TensorRT export smoke for an official ONNX model and print JSON."), QStringLiteral("onnx"));
    QCommandLineOption ncnnSmokeOption(QStringLiteral("ncnn-smoke"), QStringLiteral("Run NCNN export and deployment validation smoke and print JSON."), QStringLiteral("onnx"));
    QCommandLineOption ncnnParamSmokeOption(QStringLiteral("ncnn-param-smoke"), QStringLiteral("Run NCNN deployment validation smoke for an existing .param/.bin artifact and print JSON."), QStringLiteral("param"));
    QCommandLineOption semanticOnnxSmokeOption(QStringLiteral("semantic-onnx-smoke"), QStringLiteral("Run semantic segmentation ONNX Runtime inference, benchmark, and deployment validation smoke and print JSON."), QStringLiteral("onnx"));
    QCommandLineOption obbOnnxSmokeOption(QStringLiteral("obb-onnx-smoke"), QStringLiteral("Run OBB ONNX Runtime inference, overlay, benchmark, and deployment validation smoke and print JSON."), QStringLiteral("onnx"));
    QCommandLineOption ocrDetOnnxSmokeOption(QStringLiteral("ocr-det-onnx-smoke"), QStringLiteral("Deprecated: OCR is official-only; use PaddleOCR official Det/Rec/System reports instead."), QStringLiteral("onnx"));
    QCommandLineOption imageOption(QStringLiteral("image"), QStringLiteral("Image path for smoke checks."), QStringLiteral("path"));
    QCommandLineOption outputOption(QStringLiteral("output"), QStringLiteral("Output directory for smoke artifacts."), QStringLiteral("directory"));
    QCommandLineOption taskTypeOption(QStringLiteral("task-type"), QStringLiteral("Task type for model smoke checks."), QStringLiteral("type"), QStringLiteral("detection"));
    QCommandLineOption binaryThresholdOption(QStringLiteral("binary-threshold"), QStringLiteral("OCR Det binary threshold."), QStringLiteral("value"), QStringLiteral("0.05"));
    QCommandLineOption boxThresholdOption(QStringLiteral("box-threshold"), QStringLiteral("OCR Det box confidence threshold."), QStringLiteral("value"), QStringLiteral("0.0"));
    QCommandLineOption minAreaOption(QStringLiteral("min-area"), QStringLiteral("OCR Det minimum connected component area."), QStringLiteral("pixels"), QStringLiteral("1"));
    QCommandLineOption maxDetectionsOption(QStringLiteral("max-detections"), QStringLiteral("OCR Det maximum detections."), QStringLiteral("count"), QStringLiteral("100"));
    parser.addOption(serverOption);
    parser.addOption(selfCheckOption);
    parser.addOption(pluginSmokeOption);
    parser.addOption(tensorRtSmokeOption);
    parser.addOption(ncnnSmokeOption);
    parser.addOption(ncnnParamSmokeOption);
    parser.addOption(semanticOnnxSmokeOption);
    parser.addOption(obbOnnxSmokeOption);
    parser.addOption(ocrDetOnnxSmokeOption);
    parser.addOption(imageOption);
    parser.addOption(outputOption);
    parser.addOption(taskTypeOption);
    parser.addOption(binaryThresholdOption);
    parser.addOption(boxThresholdOption);
    parser.addOption(minAreaOption);
    parser.addOption(maxDetectionsOption);
    parser.process(app);

    if (parser.isSet(selfCheckOption)) {
        return runSelfCheck();
    }
    if (parser.isSet(pluginSmokeOption)) {
        return runPluginSmoke(parser.value(pluginSmokeOption));
    }
    if (parser.isSet(tensorRtSmokeOption)) {
        return runTensorRtSmoke(parser.value(tensorRtSmokeOption));
    }
    if (parser.isSet(ncnnSmokeOption)) {
        return runNcnnSmoke(
            parser.value(ncnnSmokeOption),
            parser.value(imageOption),
            parser.value(outputOption),
            parser.value(taskTypeOption).trimmed().toLower());
    }
    if (parser.isSet(ncnnParamSmokeOption)) {
        return runNcnnParamSmoke(
            parser.value(ncnnParamSmokeOption),
            parser.value(imageOption),
            parser.value(outputOption),
            parser.value(taskTypeOption).trimmed().toLower());
    }
    if (parser.isSet(semanticOnnxSmokeOption)) {
        return runSemanticOnnxSmoke(
            parser.value(semanticOnnxSmokeOption),
            parser.value(imageOption),
            parser.value(outputOption));
    }
    if (parser.isSet(obbOnnxSmokeOption)) {
        return runObbOnnxSmoke(
            parser.value(obbOnnxSmokeOption),
            parser.value(imageOption),
            parser.value(outputOption));
    }
    if (parser.isSet(ocrDetOnnxSmokeOption)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("status"), QStringLiteral("blocked")},
            {QStringLiteral("stage"), QStringLiteral("official-only")},
            {QStringLiteral("modelPath"), QFileInfo(parser.value(ocrDetOnnxSmokeOption)).absoluteFilePath()},
            {QStringLiteral("error"), QStringLiteral("OCR Det ONNX smoke is deprecated. AITrain OCR product routes are official-only; use PaddleOCR official Det/Rec/System reports and predict_system.py artifacts.")}
        });
        return 11;
    }

    const QString serverName = parser.value(serverOption);
    if (serverName.isEmpty()) {
        qCritical("Missing --server argument.");
        return 2;
    }

    WorkerSession session;
    if (!session.connectToServer(serverName)) {
        qCritical("Failed to connect to controller.");
        return 3;
    }

    return app.exec();
}
