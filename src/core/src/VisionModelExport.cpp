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

QString pnnxConverterExecutableName()
{
#ifdef Q_OS_WIN
    return QStringLiteral("pnnx.exe");
#else
    return QStringLiteral("pnnx");
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

QString officialOnnxSiblingForCheckpoint(const QString& checkpointPath)
{
    const QFileInfo checkpointInfo(checkpointPath);
    if (checkpointInfo.suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) == 0) {
        return {};
    }

    const QString siblingPath = checkpointInfo.absoluteDir().filePath(
        QStringLiteral("%1.onnx").arg(checkpointInfo.completeBaseName()));
    const QFileInfo siblingInfo(siblingPath);
    return siblingInfo.exists() && siblingInfo.isFile() ? siblingInfo.absoluteFilePath() : QString();
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
               << QDir::current().filePath(QStringLiteral(".deps/sdks/ncnn/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/sdks/ncnn/tools/onnx/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/sdks/ncnn/x64/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/sdks/ncnn/x64/tools/onnx/%1").arg(executableName))
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

NcnnConverterResolution resolveNcnnPnnx()
{
    const QString configured = unquotedPath(QString::fromLocal8Bit(qgetenv("AITRAIN_NCNN_PNNX")));
    if (!configured.isEmpty()) {
        const QString executable = existingExecutablePath(configured);
        if (!executable.isEmpty()) {
            return {executable, QString()};
        }
        return {{}, QStringLiteral("Configured NCNN pnnx converter was not found: %1").arg(configured)};
    }

    const QString alternateConfigured = unquotedPath(QString::fromLocal8Bit(qgetenv("AITRAIN_PNNX")));
    if (!alternateConfigured.isEmpty()) {
        const QString executable = existingExecutablePath(alternateConfigured);
        if (!executable.isEmpty()) {
            return {executable, QString()};
        }
        return {{}, QStringLiteral("Configured pnnx converter was not found: %1").arg(alternateConfigured)};
    }

    const QString executableName = pnnxConverterExecutableName();
    QStringList candidates;
    const auto appendRootCandidates = [&candidates, &executableName](const QString& rootValue) {
        const QString root = unquotedPath(rootValue);
        if (root.isEmpty()) {
            return;
        }
        const QDir rootDir(root);
        candidates << rootDir.filePath(QStringLiteral("bin/%1").arg(executableName))
                   << rootDir.filePath(QStringLiteral("x64/bin/%1").arg(executableName))
                   << rootDir.filePath(executableName);
    };
    appendRootCandidates(QString::fromLocal8Bit(qgetenv("AITRAIN_NCNN_ROOT")));
    appendRootCandidates(QString::fromLocal8Bit(qgetenv("NCNN_ROOT")));

    const QDir appDir(QCoreApplication::applicationDirPath());
    candidates << appDir.filePath(QStringLiteral("runtimes/ncnn/%1").arg(executableName))
               << appDir.filePath(QStringLiteral("../runtimes/ncnn/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/sdks/ncnn/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/sdks/ncnn/x64/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/sdks/pnnx/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/sdks/pnnx/pnnx/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/x64/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/pnnx/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/pnnx/pnnx/%1").arg(executableName));

    for (const QString& candidate : candidates) {
        const QString executable = existingExecutablePath(candidate);
        if (!executable.isEmpty()) {
            return {executable, QString()};
        }
    }

    const QString pathExecutable = QStandardPaths::findExecutable(QStringLiteral("pnnx"));
    if (!pathExecutable.isEmpty()) {
        return {pathExecutable, QString()};
    }

    return {{}, QStringLiteral("NCNN export requires pnnx for modern YOLO ONNX conversion. Set AITRAIN_NCNN_PNNX to pnnx.exe or place pnnx under AITRAIN_NCNN_ROOT.")};
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

int positiveIntFromJsonValue(const QJsonValue& value)
{
    if (value.isDouble()) {
        const int number = value.toInt();
        return number > 0 ? number : 0;
    }
    if (value.isString()) {
        bool ok = false;
        const int number = value.toString().trimmed().toInt(&ok);
        return ok && number > 0 ? number : 0;
    }
    return 0;
}

int imageSizeFromJsonValue(const QJsonValue& value)
{
    const int scalar = positiveIntFromJsonValue(value);
    if (scalar > 0) {
        return scalar;
    }
    if (!value.isObject()) {
        return 0;
    }
    const QJsonObject object = value.toObject();
    const int width = positiveIntFromJsonValue(object.value(QStringLiteral("width")));
    const int height = positiveIntFromJsonValue(object.value(QStringLiteral("height")));
    if (width > 0 && height > 0 && width == height) {
        return width;
    }
    const int w = positiveIntFromJsonValue(object.value(QStringLiteral("w")));
    const int h = positiveIntFromJsonValue(object.value(QStringLiteral("h")));
    return w > 0 && h > 0 && w == h ? w : 0;
}

int yoloNcnnInputSize(const QString& sourceOnnxPath, const NcnnExportParamMetadata& paramMetadata = NcnnExportParamMetadata())
{
    if (paramMetadata.inputSize.isValid() && !paramMetadata.inputSize.isEmpty()
        && paramMetadata.inputSize.width() == paramMetadata.inputSize.height()) {
        return paramMetadata.inputSize.width();
    }

    const QJsonObject exportSidecar = loadOnnxExportConfig(sourceOnnxPath);
    int size = imageSizeFromJsonValue(exportSidecar.value(QStringLiteral("inputSize")));
    if (size > 0) {
        return size;
    }
    size = imageSizeFromJsonValue(exportSidecar.value(QStringLiteral("ncnn")).toObject().value(QStringLiteral("inputSize")));
    if (size > 0) {
        return size;
    }

    QJsonObject report = exportSidecar.value(QStringLiteral("trainingReport")).toObject();
    if (report.isEmpty()) {
        report = loadUltralyticsTrainingReport(sourceOnnxPath);
    }
    size = imageSizeFromJsonValue(report.value(QStringLiteral("ultralyticsExportArgs")).toObject().value(QStringLiteral("imgsz")));
    if (size > 0) {
        return size;
    }
    size = imageSizeFromJsonValue(report.value(QStringLiteral("ultralyticsTrainArgs")).toObject().value(QStringLiteral("imgsz")));
    return size > 0 ? size : 640;
}

QString yoloModelSeriesForOnnx(const QString& sourceOnnxPath)
{
    const QJsonObject exportSidecar = loadOnnxExportConfig(sourceOnnxPath);
    QJsonObject report = exportSidecar.value(QStringLiteral("trainingReport")).toObject();
    if (report.isEmpty()) {
        report = loadUltralyticsTrainingReport(sourceOnnxPath);
    }

    QString modelSeries = exportSidecar.value(QStringLiteral("modelSeries")).toString().trimmed().toLower();
    if (modelSeries.isEmpty()) {
        modelSeries = report.value(QStringLiteral("modelSeries")).toString().trimmed().toLower();
    }
    if (modelSeries.isEmpty()) {
        const QString modelName = report.value(QStringLiteral("model")).toString().trimmed().toLower();
        if (modelName.contains(QStringLiteral("yolo26"))) {
            modelSeries = QStringLiteral("yolo26");
        }
    }
    if (modelSeries.isEmpty() && QFileInfo(sourceOnnxPath).absoluteFilePath().toLower().contains(QStringLiteral("yolo26"))) {
        modelSeries = QStringLiteral("yolo26");
    }
    return modelSeries;
}

bool isYolo26OnnxSource(const QString& sourceOnnxPath)
{
    return yoloModelSeriesForOnnx(sourceOnnxPath) == QStringLiteral("yolo26");
}

bool runProcessWithCancellation(
    QProcess* process,
    const CancellationCallback& shouldCancel,
    const QString& timeoutMessage,
    QString* error)
{
    if (!process->waitForStarted(5000)) {
        if (error) {
            *error = timeoutMessage.arg(process->errorString());
        }
        return false;
    }
    while (!process->waitForFinished(100)) {
        if (isCancellationRequested(shouldCancel)) {
            process->terminate();
            if (!process->waitForFinished(1500)) {
                process->kill();
                process->waitForFinished(1500);
            }
            if (error) {
                *error = QStringLiteral("Canceled by user");
            }
            return false;
        }
        if (process->state() == QProcess::NotRunning) {
            break;
        }
    }
    if (process->state() != QProcess::NotRunning) {
        if (error) {
            *error = QStringLiteral("NCNN converter did not finish: %1").arg(process->errorString());
        }
        return false;
    }
    return true;
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

    if (!runProcessWithCancellation(
            &process,
            shouldCancel,
            QStringLiteral("Could not start NCNN converter %1: %%1").arg(converter.executablePath),
            error)) {
        QFile::remove(paramPath);
        QFile::remove(binPath);
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

bool runPnnxNcnn(
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
    const NcnnConverterResolution converter = resolveNcnnPnnx();
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

    const QFileInfo paramInfo(paramPath);
    const QString outputDir = paramInfo.absolutePath();
    const QString baseName = paramInfo.completeBaseName();
    const int inputSize = yoloNcnnInputSize(sourceOnnxPath);
    const QString inputShape = QStringLiteral("inputshape=[1,3,%1,%1]").arg(inputSize);
    const QStringList converterArgs{
        sourceOnnxPath,
        inputShape,
        QStringLiteral("ncnnparam=%1").arg(paramPath),
        QStringLiteral("ncnnbin=%1").arg(binPath),
        QStringLiteral("pnnxparam=%1").arg(QDir(outputDir).filePath(baseName + QStringLiteral(".pnnx.param"))),
        QStringLiteral("pnnxbin=%1").arg(QDir(outputDir).filePath(baseName + QStringLiteral(".pnnx.bin"))),
        QStringLiteral("pnnxpy=%1").arg(QDir(outputDir).filePath(baseName + QStringLiteral("_pnnx.py"))),
        QStringLiteral("pnnxonnx=%1").arg(QDir(outputDir).filePath(baseName + QStringLiteral(".pnnx.onnx"))),
        QStringLiteral("ncnnpy=%1").arg(QDir(outputDir).filePath(baseName + QStringLiteral("_ncnn.py"))),
        QStringLiteral("fp16=0")
    };

    QProcess process;
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.setWorkingDirectory(outputDir);
#ifdef Q_OS_WIN
    if (isWindowsCommandScript(converter.executablePath)) {
        process.start(QStringLiteral("cmd.exe"),
            QStringList() << QStringLiteral("/D") << QStringLiteral("/C") << QDir::toNativeSeparators(converter.executablePath) << converterArgs);
    } else
#endif
    {
        process.start(converter.executablePath, converterArgs);
    }

    if (!runProcessWithCancellation(
            &process,
            shouldCancel,
            QStringLiteral("Could not start NCNN pnnx converter %1: %%1").arg(converter.executablePath),
            error)) {
        QFile::remove(paramPath);
        QFile::remove(binPath);
        return false;
    }

    const QString converterOutput = QString::fromLocal8Bit(process.readAll()).trimmed();
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        if (error) {
            *error = QStringLiteral("NCNN pnnx converter failed with exit code %1: %2").arg(process.exitCode()).arg(converterOutput);
        }
        return false;
    }
    if (!QFileInfo::exists(paramPath) || !QFileInfo::exists(binPath)) {
        if (error) {
            *error = QStringLiteral("NCNN pnnx converter finished but did not produce expected .param/.bin files. Output: %1").arg(converterOutput);
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
        modelFamily == QStringLiteral("yolo_detection") || modelFamily == QStringLiteral("yolo_segmentation")
            ? yoloNcnnInputSize(sourceOnnxPath, paramMetadata)
            : (paramMetadata.inputSize.isValid() && !paramMetadata.inputSize.isEmpty() ? paramMetadata.inputSize.width() : 640));
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
            {QStringLiteral("backend"), QStringLiteral("paddleocr_rec_official")},
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
    const QJsonObject exportSidecar = loadOnnxExportConfig(sourceOnnxPath);
    QJsonObject report = loadUltralyticsTrainingReport(sourceOnnxPath);
    const QString configuredFamily = exportSidecar.value(QStringLiteral("modelFamily")).toString();
    const QString configuredBackend = exportSidecar.value(QStringLiteral("backend")).toString();
    const QString reportBackend = report.value(QStringLiteral("backend")).toString();
    const bool segmentation = configuredFamily == QStringLiteral("yolo_segmentation")
        || configuredBackend == QStringLiteral("ultralytics_yolo_segment")
        || reportBackend == QStringLiteral("ultralytics_yolo_segment");
    QJsonObject config{
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
    QJsonObject exportArgs = exportSidecar.value(QStringLiteral("ultralyticsExportArgs")).toObject();
    if (exportArgs.isEmpty()) {
        exportArgs = report.value(QStringLiteral("ultralyticsExportArgs")).toObject();
    }
    if (!exportArgs.isEmpty()) {
        config.insert(QStringLiteral("ultralyticsExportArgs"), exportArgs);
    }

    const QString modelSeries = yoloModelSeriesForOnnx(sourceOnnxPath);
    if (!modelSeries.isEmpty()) {
        config.insert(QStringLiteral("modelSeries"), modelSeries);
    }

    QString task = exportSidecar.value(QStringLiteral("task")).toString();
    if (task.isEmpty()) {
        task = report.value(QStringLiteral("task")).toString();
    }
    if (task.isEmpty()) {
        task = segmentation ? QStringLiteral("segmentation") : QStringLiteral("detection");
    }
    config.insert(QStringLiteral("task"), task);

    QJsonObject outputShapes = exportSidecar.value(QStringLiteral("outputShapes")).toObject();
    if (outputShapes.isEmpty()) {
        outputShapes = report.value(QStringLiteral("outputShapes")).toObject();
    }
    if (!outputShapes.isEmpty()) {
        config.insert(QStringLiteral("outputShapes"), outputShapes);
    }

    const QString sourceTrainingReport = exportSidecar.value(QStringLiteral("sourceTrainingReport")).toString();
    if (!sourceTrainingReport.isEmpty()) {
        config.insert(QStringLiteral("sourceTrainingReport"), sourceTrainingReport);
    }
    return config;
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
    const QString normalizedFormat = format.isEmpty() ? QStringLiteral("onnx") : format.toLower();
    const bool tensorRtFormat = normalizedFormat == QStringLiteral("tensorrt")
        || normalizedFormat == QStringLiteral("tensorrt_fp16");
    const bool ncnnFormat = normalizedFormat == QStringLiteral("ncnn");
    result.format = normalizedFormat;
    result.sourceCheckpointPath = checkpointPath;
    if (isCancellationRequested(shouldCancel)) {
        result.error = QStringLiteral("Canceled by user");
        return result;
    }
    if (normalizedFormat != QStringLiteral("onnx")
        && !ncnnFormat
        && !tensorRtFormat) {
        if (normalizedFormat.startsWith(QStringLiteral("tensorrt"))) {
            result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
        } else {
            result.error = QStringLiteral("Unsupported detection export format: %1").arg(normalizedFormat);
        }
        return result;
    }

    const QString siblingOnnxPath = officialOnnxSiblingForCheckpoint(checkpointPath);
    const QString effectiveCheckpointPath = siblingOnnxPath.isEmpty() ? checkpointPath : siblingOnnxPath;
    const bool sourceIsOnnx = QFileInfo(effectiveCheckpointPath).suffix().toLower() == QStringLiteral("onnx");
    if (sourceIsOnnx) {
        const QString sourceModelFamily = inferOnnxModelFamily(effectiveCheckpointPath);
        QString finalOutputPath = outputPath;
        if (finalOutputPath.isEmpty()) {
            finalOutputPath = ncnnFormat
                ? QFileInfo(effectiveCheckpointPath).absoluteDir().filePath(QStringLiteral("model.param"))
                : QFileInfo(effectiveCheckpointPath).absoluteDir().filePath(
                    tensorRtFormat ? QStringLiteral("model.engine") : QFileInfo(effectiveCheckpointPath).fileName());
        }
        if (QFileInfo(finalOutputPath).isDir()) {
            finalOutputPath = QDir(finalOutputPath).filePath(
                ncnnFormat ? QStringLiteral("model.param") : (tensorRtFormat ? QStringLiteral("model.engine") : QFileInfo(effectiveCheckpointPath).fileName()));
        }
        if (ncnnFormat) {
            finalOutputPath = ncnnParamPathForOutput(finalOutputPath, effectiveCheckpointPath);
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
            if (QFileInfo(effectiveCheckpointPath).absoluteFilePath() != QFileInfo(finalOutputPath).absoluteFilePath()) {
                QFile::remove(finalOutputPath);
                if (!QFile::copy(effectiveCheckpointPath, finalOutputPath)) {
                    result.error = QStringLiteral("Cannot copy ONNX model to export path: %1").arg(finalOutputPath);
                    return result;
                }
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            QJsonObject config;
            if (sourceModelFamily == QStringLiteral("semantic_segmentation")) {
                config = loadOnnxExportConfig(effectiveCheckpointPath);
                config.insert(QStringLiteral("format"), QStringLiteral("onnx"));
                config.insert(QStringLiteral("backend"), QStringLiteral("smp_semantic_segmentation"));
                config.insert(QStringLiteral("modelFamily"), QStringLiteral("semantic_segmentation"));
                config.insert(QStringLiteral("taskType"), QStringLiteral("semantic_segmentation"));
                config.insert(QStringLiteral("datasetFormat"), QStringLiteral("semantic_segmentation_mask"));
                config.insert(QStringLiteral("sourceOnnx"), effectiveCheckpointPath);
                config.insert(QStringLiteral("exportPath"), finalOutputPath);
            } else {
                config = yoloOnnxExportConfig(effectiveCheckpointPath, finalOutputPath, normalizedFormat);
            }
            if (!siblingOnnxPath.isEmpty()) {
                config.insert(QStringLiteral("sourceCheckpoint"), checkpointPath);
                config.insert(QStringLiteral("sourceOnnx"), effectiveCheckpointPath);
            }
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
            if (sourceModelFamily == QStringLiteral("semantic_segmentation")) {
                result.error = QStringLiteral("SMP semantic segmentation uses ONNX Runtime deployment; NCNN export is not part of the SMP capability scope.");
                return result;
            }
            if (isYolo26OnnxSource(effectiveCheckpointPath)) {
                result.error = QStringLiteral("YOLO26 NCNN export is not supported by AITrain; use ONNX or TensorRT for YOLO26 deployment.");
                return result;
            }
            const QString binPath = ncnnBinPathForParam(finalOutputPath);
            QString converterPath;
            const QString modelFamily = sourceModelFamily;
            QString pnnxError;
            const bool preferPnnx = modelFamily == QStringLiteral("yolo_detection") || modelFamily == QStringLiteral("yolo_segmentation");
            bool converted = false;
            if (preferPnnx) {
                converted = runPnnxNcnn(effectiveCheckpointPath, finalOutputPath, binPath, shouldCancel, &converterPath, &pnnxError);
            }
            if (!converted) {
                QString onnx2NcnnError;
                converted = runOnnx2Ncnn(effectiveCheckpointPath, finalOutputPath, binPath, shouldCancel, &converterPath, &onnx2NcnnError);
                if (!converted) {
                    result.error = !pnnxError.isEmpty()
                        ? QStringLiteral("NCNN pnnx conversion failed: %1; onnx2ncnn conversion failed: %2").arg(pnnxError, onnx2NcnnError)
                        : onnx2NcnnError;
                    return result;
                }
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            QJsonObject config = ncnnOnnxExportConfig(effectiveCheckpointPath, finalOutputPath, binPath, converterPath);
            if (!siblingOnnxPath.isEmpty()) {
                config.insert(QStringLiteral("sourceCheckpoint"), checkpointPath);
                config.insert(QStringLiteral("sourceOnnx"), effectiveCheckpointPath);
            }
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
            if (sourceModelFamily == QStringLiteral("semantic_segmentation")) {
                result.error = QStringLiteral("SMP semantic segmentation uses ONNX Runtime deployment; TensorRT export is not part of the SMP capability scope.");
                return result;
            }
#ifndef AITRAIN_WITH_TENSORRT_SDK
            result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
            return result;
#else
            QFile onnxFile(effectiveCheckpointPath);
            if (!onnxFile.open(QIODevice::ReadOnly)) {
                result.error = QStringLiteral("Cannot read source ONNX model for TensorRT export: %1").arg(effectiveCheckpointPath);
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
            QJsonObject config = yoloOnnxExportConfig(effectiveCheckpointPath, finalOutputPath, normalizedFormat);
            const bool segmentation = config.value(QStringLiteral("modelFamily")).toString() == QStringLiteral("yolo_segmentation");
            config.insert(
                QStringLiteral("backend"),
                segmentation ? QStringLiteral("tensorrt_ultralytics_yolo_segment") : QStringLiteral("tensorrt_ultralytics_yolo_detect"));
            config.insert(QStringLiteral("tensorRt"), QJsonObject{
                {QStringLiteral("precision"), fp16 ? QStringLiteral("fp16") : QStringLiteral("fp32")},
                {QStringLiteral("workspaceBytes"), static_cast<double>(size_t{1} << 30)},
                {QStringLiteral("sourceOnnx"), effectiveCheckpointPath}
            });
            if (!siblingOnnxPath.isEmpty()) {
                config.insert(QStringLiteral("sourceCheckpoint"), checkpointPath);
                config.insert(QStringLiteral("sourceOnnx"), effectiveCheckpointPath);
            }
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

    result.error = QStringLiteral("Unsupported model export source: production export requires an official ONNX model artifact. Legacy AITrain diagnostic checkpoints are no longer supported.");
    return result;
}

} // namespace aitrain
