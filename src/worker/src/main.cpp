#include "WorkerSession.h"

#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/PluginManager.h"

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
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
    writeJsonLine(result);
    return missingIds.isEmpty() ? 0 : 4;
}

bool writeTextFile(const QString& path, const QString& content, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Cannot create directory for %1").arg(path);
        }
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot write %1").arg(path);
        }
        return false;
    }
    file.write(content.toUtf8());
    return true;
}

bool writeSmokeImage(const QString& path, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Cannot create image directory for %1").arg(path);
        }
        return false;
    }
    QImage image(48, 48, QImage::Format_RGB32);
    image.fill(QColor(24, 36, 48));
    for (int y = 12; y < 36; ++y) {
        for (int x = 14; x < 34; ++x) {
            image.setPixelColor(x, y, QColor(220, 80, 64));
        }
    }
    if (!image.save(path)) {
        if (error) {
            *error = QStringLiteral("Cannot write smoke image %1").arg(path);
        }
        return false;
    }
    return true;
}

int runTensorRtSmoke(const QString& workDirectory)
{
    QString error;
    const QString root = QFileInfo(workDirectory).absoluteFilePath();
    const QString datasetPath = QDir(root).filePath(QStringLiteral("dataset"));
    const QString outputPath = QDir(root).filePath(QStringLiteral("run"));
    const QString exportPath = QDir(root).filePath(QStringLiteral("export/model.engine"));

    QDir().mkpath(root);
    writeTextFile(QDir(datasetPath).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"), &error);
    writeSmokeImage(QDir(datasetPath).filePath(QStringLiteral("images/train/a.png")), &error);
    writeSmokeImage(QDir(datasetPath).filePath(QStringLiteral("images/val/a.png")), &error);
    writeTextFile(QDir(datasetPath).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.40 0.50\n"), &error);
    writeTextFile(QDir(datasetPath).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.40 0.50\n"), &error);
    if (!error.isEmpty()) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("fixture")},
            {QStringLiteral("error"), error}
        });
        return 5;
    }

    aitrain::DetectionTrainingOptions trainingOptions;
    trainingOptions.epochs = 1;
    trainingOptions.batchSize = 1;
    trainingOptions.outputPath = outputPath;
    const aitrain::DetectionTrainingResult training =
        aitrain::trainDetectionBaseline(datasetPath, trainingOptions);
    if (!training.ok) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("train")},
            {QStringLiteral("error"), training.error}
        });
        return 6;
    }

    const aitrain::DetectionExportResult exported =
        aitrain::exportDetectionCheckpoint(training.checkpointPath, exportPath, QStringLiteral("tensorrt"));
    if (!exported.ok) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("export")},
            {QStringLiteral("checkpointPath"), training.checkpointPath},
            {QStringLiteral("error"), exported.error},
            {QStringLiteral("tensorRtBackend"), aitrain::tensorRtBackendStatus().toJson()}
        });
        return 7;
    }

    aitrain::DetectionInferenceOptions inferenceOptions;
    inferenceOptions.confidenceThreshold = 0.0;
    const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionTensorRt(
        exported.exportPath,
        QDir(datasetPath).filePath(QStringLiteral("images/val/a.png")),
        inferenceOptions,
        &error);
    if (!error.isEmpty()) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("infer")},
            {QStringLiteral("enginePath"), exported.exportPath},
            {QStringLiteral("error"), error}
        });
        return 8;
    }

    writeJsonLine(QJsonObject{
        {QStringLiteral("ok"), true},
        {QStringLiteral("stage"), QStringLiteral("completed")},
        {QStringLiteral("enginePath"), exported.exportPath},
        {QStringLiteral("reportPath"), exported.reportPath},
        {QStringLiteral("predictionCount"), predictions.size()}
    });
    return 0;
}

int runOcrDetOnnxSmoke(
    const QString& onnxPath,
    const QString& imagePath,
    const QString& outputDirectory,
    double binaryThreshold,
    double boxThreshold,
    int minArea,
    int maxDetections)
{
    const QString modelPath = QFileInfo(onnxPath).absoluteFilePath();
    const QString samplePath = QFileInfo(imagePath).absoluteFilePath();
    const QString outputPath = QFileInfo(outputDirectory.isEmpty()
        ? QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("aitrain_ocr_det_onnx_smoke"))
        : outputDirectory).absoluteFilePath();

    if (!QFileInfo::exists(modelPath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("model")},
            {QStringLiteral("error"), QStringLiteral("OCR Det ONNX model does not exist: %1").arg(modelPath)}
        });
        return 5;
    }
    if (!QFileInfo::exists(samplePath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("image")},
            {QStringLiteral("error"), QStringLiteral("OCR Det smoke image does not exist: %1").arg(samplePath)}
        });
        return 6;
    }
    if (!QDir().mkpath(outputPath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("output")},
            {QStringLiteral("error"), QStringLiteral("Cannot create OCR Det smoke output directory: %1").arg(outputPath)}
        });
        return 7;
    }

    const QString family = aitrain::inferOnnxModelFamily(modelPath);
    if (family != QStringLiteral("ocr_detection")) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("model-family")},
            {QStringLiteral("modelPath"), modelPath},
            {QStringLiteral("inferredModelFamily"), family},
            {QStringLiteral("error"), QStringLiteral("Expected OCR Det ONNX model family.")}
        });
        return 8;
    }

    aitrain::OcrDetPostprocessOptions options;
    options.binaryThreshold = binaryThreshold;
    options.boxThreshold = boxThreshold;
    options.minArea = minArea;
    options.maxDetections = maxDetections;

    QString error;
    const QVector<aitrain::OcrDetPrediction> predictions =
        aitrain::predictOcrDetOnnxRuntime(modelPath, samplePath, options, &error);
    if (!error.isEmpty()) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("inference")},
            {QStringLiteral("modelPath"), modelPath},
            {QStringLiteral("imagePath"), samplePath},
            {QStringLiteral("error"), error}
        });
        return 9;
    }

    QJsonArray predictionArray;
    for (const aitrain::OcrDetPrediction& prediction : predictions) {
        predictionArray.append(aitrain::ocrDetPredictionToJson(prediction));
    }

    const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("ocr_det_onnx_predictions.json"));
    QFile predictionsFile(predictionsPath);
    if (!predictionsFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("predictions")},
            {QStringLiteral("error"), QStringLiteral("Cannot write OCR Det predictions: %1").arg(predictionsPath)}
        });
        return 10;
    }
    QJsonObject predictionsDocument;
    predictionsDocument.insert(QStringLiteral("taskType"), QStringLiteral("ocr_detection"));
    predictionsDocument.insert(QStringLiteral("runtime"), QStringLiteral("onnxruntime"));
    predictionsDocument.insert(QStringLiteral("modelPath"), modelPath);
    predictionsDocument.insert(QStringLiteral("imagePath"), samplePath);
    predictionsDocument.insert(QStringLiteral("postprocess"), QJsonObject{
        {QStringLiteral("binaryThreshold"), options.binaryThreshold},
        {QStringLiteral("boxThreshold"), options.boxThreshold},
        {QStringLiteral("minArea"), options.minArea},
        {QStringLiteral("maxDetections"), options.maxDetections}
    });
    predictionsDocument.insert(QStringLiteral("predictions"), predictionArray);
    predictionsFile.write(QJsonDocument(predictionsDocument).toJson(QJsonDocument::Indented));
    predictionsFile.close();

    const QImage overlay = aitrain::renderOcrDetPredictions(samplePath, predictions, &error);
    if (overlay.isNull()) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("overlay")},
            {QStringLiteral("error"), error}
        });
        return 11;
    }
    const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("ocr_det_onnx_overlay.png"));
    if (!overlay.save(overlayPath)) {
        writeJsonLine(QJsonObject{
            {QStringLiteral("ok"), false},
            {QStringLiteral("stage"), QStringLiteral("overlay")},
            {QStringLiteral("error"), QStringLiteral("Cannot write OCR Det overlay: %1").arg(overlayPath)}
        });
        return 12;
    }

    const QJsonObject result{
        {QStringLiteral("ok"), true},
        {QStringLiteral("stage"), QStringLiteral("completed")},
        {QStringLiteral("taskType"), QStringLiteral("ocr_detection")},
        {QStringLiteral("runtime"), QStringLiteral("onnxruntime")},
        {QStringLiteral("modelPath"), modelPath},
        {QStringLiteral("imagePath"), samplePath},
        {QStringLiteral("outputPath"), outputPath},
        {QStringLiteral("predictionsPath"), predictionsPath},
        {QStringLiteral("overlayPath"), overlayPath},
        {QStringLiteral("predictionCount"), predictions.size()},
        {QStringLiteral("postprocess"), predictionsDocument.value(QStringLiteral("postprocess")).toObject()}
    };
    writeJsonLine(result);
    return 0;
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
    QCommandLineOption tensorRtSmokeOption(QStringLiteral("tensorrt-smoke"), QStringLiteral("Run a tiny TensorRT export/inference smoke check and print JSON."), QStringLiteral("directory"));
    QCommandLineOption ocrDetOnnxSmokeOption(QStringLiteral("ocr-det-onnx-smoke"), QStringLiteral("Run OCR Det ONNX Runtime DB postprocess smoke and print JSON."), QStringLiteral("onnx"));
    QCommandLineOption imageOption(QStringLiteral("image"), QStringLiteral("Image path for smoke checks."), QStringLiteral("path"));
    QCommandLineOption outputOption(QStringLiteral("output"), QStringLiteral("Output directory for smoke artifacts."), QStringLiteral("directory"));
    QCommandLineOption binaryThresholdOption(QStringLiteral("binary-threshold"), QStringLiteral("OCR Det binary threshold."), QStringLiteral("value"), QStringLiteral("0.05"));
    QCommandLineOption boxThresholdOption(QStringLiteral("box-threshold"), QStringLiteral("OCR Det box confidence threshold."), QStringLiteral("value"), QStringLiteral("0.0"));
    QCommandLineOption minAreaOption(QStringLiteral("min-area"), QStringLiteral("OCR Det minimum connected component area."), QStringLiteral("pixels"), QStringLiteral("1"));
    QCommandLineOption maxDetectionsOption(QStringLiteral("max-detections"), QStringLiteral("OCR Det maximum detections."), QStringLiteral("count"), QStringLiteral("100"));
    parser.addOption(serverOption);
    parser.addOption(selfCheckOption);
    parser.addOption(pluginSmokeOption);
    parser.addOption(tensorRtSmokeOption);
    parser.addOption(ocrDetOnnxSmokeOption);
    parser.addOption(imageOption);
    parser.addOption(outputOption);
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
    if (parser.isSet(ocrDetOnnxSmokeOption)) {
        return runOcrDetOnnxSmoke(
            parser.value(ocrDetOnnxSmokeOption),
            parser.value(imageOption),
            parser.value(outputOption),
            parser.value(binaryThresholdOption).toDouble(),
            parser.value(boxThresholdOption).toDouble(),
            parser.value(minAreaOption).toInt(),
            parser.value(maxDetectionsOption).toInt());
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
