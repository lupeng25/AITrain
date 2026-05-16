#include "DetectionTrainerInternal.h"

#include "aitrain/core/Deployment.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPainter>
#include <QProcess>
#include <QQueue>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QSet>
#include <QtEndian>
#include <QtMath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
namespace aitrain {

using namespace detection_detail;

QJsonObject yoloOnnxExportConfig(const QString& sourceOnnxPath, const QString& exportPath, const QString& format);

QString ncnnConverterExecutableName()
{
#ifdef Q_OS_WIN
    return QStringLiteral("onnx2ncnn.exe");
#else
    return QStringLiteral("onnx2ncnn");
#endif
}

QString unquotedPath(QString value)
{
    value = value.trimmed();
    if (value.size() >= 2
        && ((value.startsWith(QLatin1Char('"')) && value.endsWith(QLatin1Char('"')))
            || (value.startsWith(QLatin1Char('\'')) && value.endsWith(QLatin1Char('\''))))) {
        value = value.mid(1, value.size() - 2);
    }
    return QDir::fromNativeSeparators(value);
}

QString existingExecutablePath(const QString& candidate)
{
    if (candidate.trimmed().isEmpty()) {
        return {};
    }
    const QFileInfo info(unquotedPath(candidate));
    return info.exists() && info.isFile() ? info.absoluteFilePath() : QString();
}

QString ncnnParamPathForOutput(const QString& outputPath, const QString& sourcePath)
{
    QString finalOutputPath = outputPath;
    if (finalOutputPath.isEmpty()) {
        finalOutputPath = QFileInfo(sourcePath).absoluteDir().filePath(QStringLiteral("model.param"));
    }
    if (QFileInfo(finalOutputPath).isDir()) {
        finalOutputPath = QDir(finalOutputPath).filePath(QStringLiteral("model.param"));
    }

    QFileInfo outputInfo(finalOutputPath);
    const QString suffix = outputInfo.suffix().toLower();
    if (suffix == QStringLiteral("bin")) {
        finalOutputPath = outputInfo.absoluteDir().filePath(QStringLiteral("%1.param").arg(outputInfo.completeBaseName()));
    } else if (suffix != QStringLiteral("param")) {
        finalOutputPath = outputInfo.absoluteDir().filePath(QStringLiteral("%1.param").arg(outputInfo.fileName()));
    }
    return QDir::cleanPath(finalOutputPath);
}

QString ncnnBinPathForParam(const QString& paramPath)
{
    const QFileInfo info(paramPath);
    return info.absoluteDir().filePath(QStringLiteral("%1.bin").arg(info.completeBaseName()));
}

struct NcnnExportParamMetadata {
    QString inputBlob;
    QStringList outputBlobs;
    QSize inputSize;
};

NcnnExportParamMetadata parseNcnnParamForExportSidecar(const QString& paramPath)
{
    NcnnExportParamMetadata metadata;
    QFile file(paramPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return metadata;
    }

    QSet<QString> produced;
    QSet<QString> consumed;
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#')) || line == QStringLiteral("7767517")) {
            continue;
        }
        const QStringList tokens = line.split(QRegularExpression(QStringLiteral("\\s+")), QString::SkipEmptyParts);
        if (tokens.size() < 4) {
            continue;
        }
        bool okBottom = false;
        bool okTop = false;
        const int bottomCount = tokens.at(2).toInt(&okBottom);
        const int topCount = tokens.at(3).toInt(&okTop);
        if (!okBottom || !okTop || bottomCount < 0 || topCount < 0 || tokens.size() < 4 + bottomCount + topCount) {
            continue;
        }

        for (int index = 0; index < bottomCount; ++index) {
            consumed.insert(tokens.at(4 + index));
        }
        QStringList topBlobs;
        for (int index = 0; index < topCount; ++index) {
            const QString blob = tokens.at(4 + bottomCount + index);
            topBlobs.append(blob);
            produced.insert(blob);
        }

        if (tokens.at(0) == QStringLiteral("Input") && !topBlobs.isEmpty()) {
            metadata.inputBlob = topBlobs.first();
            int width = 0;
            int height = 0;
            for (int index = 4 + bottomCount + topCount; index < tokens.size(); ++index) {
                const QString token = tokens.at(index);
                const int equalIndex = token.indexOf(QLatin1Char('='));
                if (equalIndex <= 0) {
                    continue;
                }
                const int key = token.left(equalIndex).toInt();
                const int value = token.mid(equalIndex + 1).toInt();
                if (key == 0) width = value;
                if (key == 1) height = value;
            }
            if (width > 0 && height > 0) {
                metadata.inputSize = QSize(width, height);
            }
        }
    }

    QStringList outputBlobs;
    for (const QString& blob : produced) {
        if (!consumed.contains(blob)) {
            outputBlobs.append(blob);
        }
    }
    outputBlobs.sort();
    metadata.outputBlobs = outputBlobs;
    return metadata;
}

struct NcnnConverterResolution {
    QString executablePath;
    QString message;
};

NcnnConverterResolution resolveNcnnOnnx2Ncnn()
{
    const QString configured = unquotedPath(QString::fromLocal8Bit(qgetenv("AITRAIN_NCNN_ONNX2NCNN")));
    if (!configured.isEmpty()) {
        const QString executable = existingExecutablePath(configured);
        if (!executable.isEmpty()) {
            return {executable, QString()};
        }
        return {{}, QStringLiteral("Configured NCNN converter was not found: %1").arg(configured)};
    }

    const QString executableName = ncnnConverterExecutableName();
    QStringList candidates;
    const auto appendRootCandidates = [&candidates, &executableName](const QString& rootValue) {
        const QString root = unquotedPath(rootValue);
        if (root.isEmpty()) {
            return;
        }
        const QDir rootDir(root);
        candidates << rootDir.filePath(QStringLiteral("bin/%1").arg(executableName))
                   << rootDir.filePath(QStringLiteral("tools/onnx/%1").arg(executableName))
                   << rootDir.filePath(QStringLiteral("x64/bin/%1").arg(executableName))
                   << rootDir.filePath(QStringLiteral("x64/tools/onnx/%1").arg(executableName))
                   << rootDir.filePath(executableName);
    };
    appendRootCandidates(QString::fromLocal8Bit(qgetenv("AITRAIN_NCNN_ROOT")));
    appendRootCandidates(QString::fromLocal8Bit(qgetenv("NCNN_ROOT")));

    const QDir appDir(QCoreApplication::applicationDirPath());
    candidates << appDir.filePath(QStringLiteral("runtimes/ncnn/%1").arg(executableName))
               << appDir.filePath(QStringLiteral("../runtimes/ncnn/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/tools/onnx/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/x64/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/x64/tools/onnx/%1").arg(executableName));

    for (const QString& candidate : candidates) {
        const QString executable = existingExecutablePath(candidate);
        if (!executable.isEmpty()) {
            return {executable, QString()};
        }
    }

    const QString pathExecutable = QStandardPaths::findExecutable(QStringLiteral("onnx2ncnn"));
    if (!pathExecutable.isEmpty()) {
        return {pathExecutable, QString()};
    }

    return {{}, QStringLiteral("NCNN export requires onnx2ncnn. Set AITRAIN_NCNN_ONNX2NCNN to onnx2ncnn.exe or AITRAIN_NCNN_ROOT to an NCNN install root.")};
}

bool isWindowsCommandScript(const QString& path)
{
#ifdef Q_OS_WIN
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == QStringLiteral("bat") || suffix == QStringLiteral("cmd");
#else
    Q_UNUSED(path);
    return false;
#endif
}

bool runOnnx2Ncnn(
    const QString& sourceOnnxPath,
    const QString& paramPath,
    const QString& binPath,
    const CancellationCallback& shouldCancel,
    QString* converterPath,
    QString* error)
{
    if (isCancellationRequested(shouldCancel)) {
        if (error) {
            *error = QStringLiteral("Canceled by user");
        }
        return false;
    }
    const NcnnConverterResolution converter = resolveNcnnOnnx2Ncnn();
    if (converter.executablePath.isEmpty()) {
        if (error) {
            *error = converter.message;
        }
        return false;
    }

    if (!QFileInfo::exists(sourceOnnxPath)) {
        if (error) {
            *error = QStringLiteral("Cannot read source ONNX model for NCNN export: %1").arg(sourceOnnxPath);
        }
        return false;
    }
    if (!QDir().mkpath(QFileInfo(paramPath).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Cannot create NCNN export directory: %1").arg(QFileInfo(paramPath).absolutePath());
        }
        return false;
    }

    QFile::remove(paramPath);
    QFile::remove(binPath);

    QProcess process;
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.setWorkingDirectory(QFileInfo(paramPath).absolutePath());
#ifdef Q_OS_WIN
    if (isWindowsCommandScript(converter.executablePath)) {
        process.start(QStringLiteral("cmd.exe"), QStringList()
            << QStringLiteral("/D")
            << QStringLiteral("/C")
            << QDir::toNativeSeparators(converter.executablePath)
            << QDir::toNativeSeparators(sourceOnnxPath)
            << QDir::toNativeSeparators(paramPath)
            << QDir::toNativeSeparators(binPath));
    } else
#endif
    {
        process.start(converter.executablePath, QStringList() << sourceOnnxPath << paramPath << binPath);
    }

    if (!process.waitForStarted(5000)) {
        if (error) {
            *error = QStringLiteral("Could not start NCNN converter %1: %2").arg(converter.executablePath, process.errorString());
        }
        return false;
    }
    while (!process.waitForFinished(100)) {
        if (isCancellationRequested(shouldCancel)) {
            process.terminate();
            if (!process.waitForFinished(1500)) {
                process.kill();
                process.waitForFinished(1500);
            }
            QFile::remove(paramPath);
            QFile::remove(binPath);
            if (error) {
                *error = QStringLiteral("Canceled by user");
            }
            return false;
        }
        if (process.state() == QProcess::NotRunning) {
            break;
        }
    }
    if (process.state() != QProcess::NotRunning) {
        if (error) {
            *error = QStringLiteral("NCNN converter did not finish: %1").arg(process.errorString());
        }
        return false;
    }

    const QString converterOutput = QString::fromLocal8Bit(process.readAll()).trimmed();
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        if (error) {
            *error = QStringLiteral("NCNN converter failed with exit code %1: %2").arg(process.exitCode()).arg(converterOutput);
        }
        return false;
    }
    if (!QFileInfo::exists(paramPath) || !QFileInfo::exists(binPath)) {
        if (error) {
            *error = QStringLiteral("NCNN converter finished but did not produce expected .param/.bin files. Output: %1").arg(converterOutput);
        }
        return false;
    }

    if (converterPath) {
        *converterPath = converter.executablePath;
    }
    return true;
}

QJsonObject ncnnMetadata(
    const QString& paramPath,
    const QString& binPath,
    const QString& converterPath,
    const QString& sourceOnnxPath)
{
    const QString modelFamily = inferOnnxModelFamily(sourceOnnxPath);
    const NcnnExportParamMetadata paramMetadata = parseNcnnParamForExportSidecar(paramPath);
    QJsonObject metadata{
        {QStringLiteral("paramPath"), paramPath},
        {QStringLiteral("binPath"), binPath},
        {QStringLiteral("converter"), converterPath},
        {QStringLiteral("sourceOnnx"), sourceOnnxPath},
        {QStringLiteral("runtime"), QStringLiteral("ncnn")},
        {QStringLiteral("runtimeValidation"), QStringLiteral("runtime-inference")},
        {QStringLiteral("note"), QStringLiteral("NCNN runtime validation is available when this build is configured with an NCNN SDK/runtime.")}
    };
    if (!paramMetadata.inputBlob.isEmpty()) {
        metadata.insert(QStringLiteral("inputBlob"), paramMetadata.inputBlob);
    }
    if (!paramMetadata.outputBlobs.isEmpty()) {
        metadata.insert(QStringLiteral("outputBlobs"), QJsonArray::fromStringList(paramMetadata.outputBlobs));
    }
    metadata.insert(QStringLiteral("inputSize"),
        paramMetadata.inputSize.isValid() && !paramMetadata.inputSize.isEmpty() ? paramMetadata.inputSize.width() : 640);
    if (modelFamily == QStringLiteral("yolo_detection") || modelFamily == QStringLiteral("yolo_segmentation")) {
        metadata.insert(QStringLiteral("decoder"), QStringLiteral("auto"));
        metadata.insert(QStringLiteral("strides"), QJsonArray{8, 16, 32});
        metadata.insert(QStringLiteral("regMax"), 16);
    }
    return metadata;
}

QJsonObject ncnnOnnxExportConfig(
    const QString& sourceOnnxPath,
    const QString& paramPath,
    const QString& binPath,
    const QString& converterPath)
{
    const QString modelFamily = inferOnnxModelFamily(sourceOnnxPath);
    QJsonObject config;
    if (modelFamily == QStringLiteral("yolo_detection") || modelFamily == QStringLiteral("yolo_segmentation")) {
        config = yoloOnnxExportConfig(sourceOnnxPath, paramPath, QStringLiteral("ncnn"));
    } else if (modelFamily == QStringLiteral("ocr_recognition")) {
        config = QJsonObject{
            {QStringLiteral("format"), QStringLiteral("ncnn")},
            {QStringLiteral("backend"), QStringLiteral("paddleocr_rec")},
            {QStringLiteral("modelFamily"), QStringLiteral("ocr_recognition")},
            {QStringLiteral("scaffold"), false},
            {QStringLiteral("sourceCheckpoint"), sourceOnnxPath},
            {QStringLiteral("sourceOnnx"), sourceOnnxPath},
            {QStringLiteral("exportPath"), paramPath},
            {QStringLiteral("trainingReport"), loadOcrRecReport(sourceOnnxPath)}
        };
    } else {
        config = QJsonObject{
            {QStringLiteral("format"), QStringLiteral("ncnn")},
            {QStringLiteral("backend"), QStringLiteral("onnx2ncnn")},
            {QStringLiteral("modelFamily"), modelFamily.isEmpty() ? QStringLiteral("unknown_onnx") : modelFamily},
            {QStringLiteral("scaffold"), false},
            {QStringLiteral("sourceCheckpoint"), sourceOnnxPath},
            {QStringLiteral("sourceOnnx"), sourceOnnxPath},
            {QStringLiteral("exportPath"), paramPath}
        };
    }
    config.insert(QStringLiteral("ncnn"), ncnnMetadata(paramPath, binPath, converterPath, sourceOnnxPath));
    return config;
}

QJsonObject yoloOnnxExportConfig(const QString& sourceOnnxPath, const QString& exportPath, const QString& format)
{
    const QStringList classNames = ultralyticsClassNames(sourceOnnxPath);
    QJsonObject report = loadUltralyticsTrainingReport(sourceOnnxPath);
    const bool segmentation = report.value(QStringLiteral("backend")).toString() == QStringLiteral("ultralytics_yolo_segment");
    return QJsonObject{
        {QStringLiteral("format"), format},
        {QStringLiteral("backend"), segmentation ? QStringLiteral("ultralytics_yolo_segment") : QStringLiteral("ultralytics_yolo_detect")},
        {QStringLiteral("modelFamily"), segmentation ? QStringLiteral("yolo_segmentation") : QStringLiteral("yolo_detection")},
        {QStringLiteral("scaffold"), false},
        {QStringLiteral("sourceCheckpoint"), sourceOnnxPath},
        {QStringLiteral("sourceOnnx"), sourceOnnxPath},
        {QStringLiteral("exportPath"), exportPath},
        {QStringLiteral("classNames"), QJsonArray::fromStringList(classNames)},
        {QStringLiteral("trainingReport"), report},
        {QStringLiteral("postprocess"), QJsonObject{
            {QStringLiteral("decoder"), segmentation ? QStringLiteral("yolo_v8_segmentation") : QStringLiteral("yolo_v8_detection")},
            {QStringLiteral("nms"), QStringLiteral("AITrain runtime")},
            {QStringLiteral("coordinates"), QStringLiteral("letterbox_to_original_image")}
        }}
    };
}

DetectionExportResult exportDetectionCheckpoint(
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format)
{
    return exportDetectionCheckpoint(checkpointPath, outputPath, format, CancellationCallback());
}

DetectionExportResult exportDetectionCheckpoint(
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format,
    const CancellationCallback& shouldCancel)
{
    DetectionExportResult result;
    const QString normalizedFormat = format.isEmpty() ? QStringLiteral("tiny_detector_json") : format.toLower();
    const bool tensorRtFormat = normalizedFormat == QStringLiteral("tensorrt")
        || normalizedFormat == QStringLiteral("tensorrt_fp16");
    const bool ncnnFormat = normalizedFormat == QStringLiteral("ncnn");
    result.format = normalizedFormat;
    result.sourceCheckpointPath = checkpointPath;
    if (isCancellationRequested(shouldCancel)) {
        result.error = QStringLiteral("Canceled by user");
        return result;
    }
    if (normalizedFormat != QStringLiteral("tiny_detector_json")
        && normalizedFormat != QStringLiteral("onnx")
        && !ncnnFormat
        && !tensorRtFormat) {
        if (normalizedFormat.startsWith(QStringLiteral("tensorrt"))) {
            result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
        } else {
            result.error = QStringLiteral("Unsupported detection export format: %1").arg(normalizedFormat);
        }
        return result;
    }

    const bool sourceIsOnnx = QFileInfo(checkpointPath).suffix().toLower() == QStringLiteral("onnx");
    if (sourceIsOnnx) {
        QString finalOutputPath = outputPath;
        if (finalOutputPath.isEmpty()) {
            finalOutputPath = ncnnFormat
                ? QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("model.param"))
                : QFileInfo(checkpointPath).absoluteDir().filePath(
                    tensorRtFormat ? QStringLiteral("model.engine") : QFileInfo(checkpointPath).fileName());
        }
        if (QFileInfo(finalOutputPath).isDir()) {
            finalOutputPath = QDir(finalOutputPath).filePath(
                ncnnFormat ? QStringLiteral("model.param") : (tensorRtFormat ? QStringLiteral("model.engine") : QFileInfo(checkpointPath).fileName()));
        }
        if (ncnnFormat) {
            finalOutputPath = ncnnParamPathForOutput(finalOutputPath, checkpointPath);
        }
        if (!QDir().mkpath(QFileInfo(finalOutputPath).absolutePath())) {
            result.error = QStringLiteral("Cannot create export directory: %1").arg(QFileInfo(finalOutputPath).absolutePath());
            return result;
        }

        if (normalizedFormat == QStringLiteral("onnx")) {
            if (isCancellationRequested(shouldCancel)) {
                result.error = QStringLiteral("Canceled by user");
                return result;
            }
            if (QFileInfo(checkpointPath).absoluteFilePath() != QFileInfo(finalOutputPath).absoluteFilePath()) {
                QFile::remove(finalOutputPath);
                if (!QFile::copy(checkpointPath, finalOutputPath)) {
                    result.error = QStringLiteral("Cannot copy ONNX model to export path: %1").arg(finalOutputPath);
                    return result;
                }
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            const QJsonObject config = yoloOnnxExportConfig(checkpointPath, finalOutputPath, normalizedFormat);
            if (isCancellationRequested(shouldCancel)) {
                result.error = QStringLiteral("Canceled by user");
                return result;
            }
            if (!writeJsonObject(reportPath, config, &result.error)) {
                return result;
            }
            result.ok = true;
            result.exportPath = finalOutputPath;
            result.reportPath = reportPath;
            result.config = config;
            return result;
        }

        if (ncnnFormat) {
            const QString binPath = ncnnBinPathForParam(finalOutputPath);
            QString converterPath;
            if (!runOnnx2Ncnn(checkpointPath, finalOutputPath, binPath, shouldCancel, &converterPath, &result.error)) {
                return result;
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            const QJsonObject config = ncnnOnnxExportConfig(checkpointPath, finalOutputPath, binPath, converterPath);
            if (isCancellationRequested(shouldCancel)) {
                result.error = QStringLiteral("Canceled by user");
                return result;
            }
            if (!writeJsonObject(reportPath, config, &result.error)) {
                return result;
            }
            result.ok = true;
            result.exportPath = finalOutputPath;
            result.reportPath = reportPath;
            result.config = config;
            return result;
        }

        if (tensorRtFormat) {
#ifndef AITRAIN_WITH_TENSORRT_SDK
            result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
            return result;
#else
            QFile onnxFile(checkpointPath);
            if (!onnxFile.open(QIODevice::ReadOnly)) {
                result.error = QStringLiteral("Cannot read source ONNX model for TensorRT export: %1").arg(checkpointPath);
                return result;
            }
            const QByteArray onnxModel = onnxFile.readAll();
            if (isCancellationRequested(shouldCancel)) {
                result.error = QStringLiteral("Canceled by user");
                return result;
            }
            const bool fp16 = normalizedFormat == QStringLiteral("tensorrt_fp16");
            if (!writeTensorRtEngineFromOnnx(onnxModel, finalOutputPath, fp16, &result.error)) {
                return result;
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            QJsonObject config = yoloOnnxExportConfig(checkpointPath, finalOutputPath, normalizedFormat);
            config.insert(QStringLiteral("backend"), QStringLiteral("tensorrt_ultralytics_yolo_detect"));
            config.insert(QStringLiteral("tensorRt"), QJsonObject{
                {QStringLiteral("precision"), fp16 ? QStringLiteral("fp16") : QStringLiteral("fp32")},
                {QStringLiteral("workspaceBytes"), static_cast<double>(size_t{1} << 30)},
                {QStringLiteral("sourceOnnx"), checkpointPath}
            });
            if (isCancellationRequested(shouldCancel)) {
                result.error = QStringLiteral("Canceled by user");
                return result;
            }
            if (!writeJsonObject(reportPath, config, &result.error)) {
                return result;
            }
            result.ok = true;
            result.exportPath = finalOutputPath;
            result.reportPath = reportPath;
            result.config = config;
            return result;
#endif
        }
    }

    QString error;
    DetectionBaselineCheckpoint checkpoint;
    if (!loadDetectionBaselineCheckpoint(checkpointPath, &checkpoint, &error)) {
        result.error = error;
        return result;
    }
    if (checkpoint.type != QStringLiteral("tiny_linear_detector")) {
        result.error = QStringLiteral("Only tiny_linear_detector checkpoints can be exported by this scaffold exporter");
        return result;
    }

    QString finalOutputPath = outputPath;
    if (finalOutputPath.isEmpty()) {
        finalOutputPath = QFileInfo(checkpointPath).absoluteDir().filePath(
            normalizedFormat == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (ncnnFormat ? QStringLiteral("model.param") : (tensorRtFormat ? QStringLiteral("model.engine") : QStringLiteral("model.aitrain-export.json"))));
    }
    if (QFileInfo(finalOutputPath).isDir()) {
        finalOutputPath = QDir(finalOutputPath).filePath(
            normalizedFormat == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (ncnnFormat ? QStringLiteral("model.param") : (tensorRtFormat ? QStringLiteral("model.engine") : QStringLiteral("model.aitrain-export.json"))));
    }
    if (ncnnFormat) {
        finalOutputPath = ncnnParamPathForOutput(finalOutputPath, checkpointPath);
    }
    if (!QDir().mkpath(QFileInfo(finalOutputPath).absolutePath())) {
        result.error = QStringLiteral("Cannot create export directory: %1").arg(QFileInfo(finalOutputPath).absolutePath());
        return result;
    }

    if (normalizedFormat == QStringLiteral("onnx")) {
        if (isCancellationRequested(shouldCancel)) {
            result.error = QStringLiteral("Canceled by user");
            return result;
        }
        if (!writeTinyDetectorOnnxModel(checkpoint, finalOutputPath, &result.error)) {
            return result;
        }
        const QString reportPath = onnxExportReportPath(finalOutputPath);
        const QJsonObject config = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
        if (isCancellationRequested(shouldCancel)) {
            result.error = QStringLiteral("Canceled by user");
            return result;
        }
        if (!writeJsonObject(reportPath, config, &result.error)) {
            return result;
        }
        result.ok = true;
        result.exportPath = finalOutputPath;
        result.reportPath = reportPath;
        result.config = config;
        return result;
    }

    if (ncnnFormat) {
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            result.error = QStringLiteral("Cannot create temporary directory for NCNN export.");
            return result;
        }
        const QString tempOnnxPath = tempDir.filePath(QStringLiteral("tiny_detector.onnx"));
        if (isCancellationRequested(shouldCancel)) {
            result.error = QStringLiteral("Canceled by user");
            return result;
        }
        if (!writeTinyDetectorOnnxModel(checkpoint, tempOnnxPath, &result.error)) {
            return result;
        }

        const QString binPath = ncnnBinPathForParam(finalOutputPath);
        QString converterPath;
        if (!runOnnx2Ncnn(tempOnnxPath, finalOutputPath, binPath, shouldCancel, &converterPath, &result.error)) {
            return result;
        }

        const QString reportPath = onnxExportReportPath(finalOutputPath);
        QJsonObject config = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
        config.insert(QStringLiteral("ncnn"), ncnnMetadata(finalOutputPath, binPath, converterPath, tempOnnxPath));
        if (isCancellationRequested(shouldCancel)) {
            result.error = QStringLiteral("Canceled by user");
            return result;
        }
        if (!writeJsonObject(reportPath, config, &result.error)) {
            return result;
        }
        result.ok = true;
        result.exportPath = finalOutputPath;
        result.reportPath = reportPath;
        result.config = config;
        return result;
    }

    if (tensorRtFormat) {
#ifndef AITRAIN_WITH_TENSORRT_SDK
        result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
        return result;
#else
        if (isCancellationRequested(shouldCancel)) {
            result.error = QStringLiteral("Canceled by user");
            return result;
        }
        const QByteArray onnxModel = tinyDetectorOnnxModel(checkpoint, &result.error);
        if (onnxModel.isEmpty()) {
            return result;
        }
        const bool fp16 = normalizedFormat == QStringLiteral("tensorrt_fp16");
        if (!writeTensorRtEngineFromOnnx(onnxModel, finalOutputPath, fp16, &result.error)) {
            return result;
        }

        const QString reportPath = onnxExportReportPath(finalOutputPath);
        QJsonObject config = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
        config.insert(QStringLiteral("backend"), QStringLiteral("tensorrt_tiny_detector"));
        config.insert(QStringLiteral("tensorRt"), QJsonObject{
            {QStringLiteral("precision"), fp16 ? QStringLiteral("fp16") : QStringLiteral("fp32")},
            {QStringLiteral("workspaceBytes"), static_cast<double>(size_t{1} << 30)},
            {QStringLiteral("dynamicShape"), false},
            {QStringLiteral("engineCache"), true}
        });
        if (isCancellationRequested(shouldCancel)) {
            result.error = QStringLiteral("Canceled by user");
            return result;
        }
        if (!writeJsonObject(reportPath, config, &result.error)) {
            return result;
        }
        result.ok = true;
        result.exportPath = finalOutputPath;
        result.reportPath = reportPath;
        result.config = config;
        return result;
#endif
    }

    QJsonObject exportObject = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
    exportObject.insert(QStringLiteral("sourceCheckpoint"), checkpointPath);
    exportObject.insert(QStringLiteral("note"), QStringLiteral("Scaffold export for AITrain tiny detector. This is not ONNX."));
    exportObject.insert(QStringLiteral("type"), checkpoint.type);
    exportObject.insert(QStringLiteral("datasetPath"), checkpoint.datasetPath);
    exportObject.insert(QStringLiteral("imageWidth"), checkpoint.imageSize.width());
    exportObject.insert(QStringLiteral("imageHeight"), checkpoint.imageSize.height());
    exportObject.insert(QStringLiteral("gridSize"), checkpoint.gridSize);
    exportObject.insert(QStringLiteral("featureCount"), checkpoint.featureCount);
    exportObject.insert(QStringLiteral("classNames"), QJsonArray::fromStringList(checkpoint.classNames));
    exportObject.insert(QStringLiteral("classLogits"), doubleArray(checkpoint.classLogits));
    exportObject.insert(QStringLiteral("objectnessWeights"), doubleArray(checkpoint.objectnessWeights));
    exportObject.insert(QStringLiteral("classWeights"), doubleArray(checkpoint.classWeights));
    exportObject.insert(QStringLiteral("boxWeights"), doubleArray(checkpoint.boxWeights));
    exportObject.insert(QStringLiteral("priorBox"), boxObject(checkpoint.priorBox));
    exportObject.insert(QStringLiteral("metrics"), QJsonObject{
        {QStringLiteral("finalLoss"), checkpoint.finalLoss},
        {QStringLiteral("precision"), checkpoint.precision},
        {QStringLiteral("recall"), checkpoint.recall},
        {QStringLiteral("mAP50"), checkpoint.map50}
    });

    if (isCancellationRequested(shouldCancel)) {
        result.error = QStringLiteral("Canceled by user");
        return result;
    }
    QFile file(finalOutputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        result.error = QStringLiteral("Cannot write export artifact: %1").arg(finalOutputPath);
        return result;
    }
    file.write(QJsonDocument(exportObject).toJson(QJsonDocument::Indented));
    file.close();

    result.ok = true;
    result.exportPath = finalOutputPath;
    result.reportPath = finalOutputPath;
    result.config = exportObject;
    return result;
}

} // namespace aitrain
