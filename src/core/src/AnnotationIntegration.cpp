#include "aitrain/core/AnnotationIntegration.h"

#include "YoloDatasetLayout.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QRegularExpression>
#include <QSet>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTextStream>
#include <QVector>

namespace aitrain {
namespace {

bool canceled(const CancellationCallback& shouldCancel)
{
    return shouldCancel && shouldCancel();
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
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

bool readJsonFile(const QString& path, QJsonObject* object, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("Cannot read JSON file: %1").arg(path);
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Invalid JSON object: %1").arg(path);
        }
        return false;
    }
    if (object) {
        *object = document.object();
    }
    return true;
}

bool writeTextFile(const QString& path, const QString& text, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot write text file: %1").arg(path);
        }
        return false;
    }
    file.write(text.toUtf8());
    return true;
}

QString normalizedPath(const QString& path)
{
    if (path.trimmed().isEmpty()) {
        return {};
    }
    return QDir::cleanPath(QDir::fromNativeSeparators(path.trimmed()));
}

QString nowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
}

QString resolveExecutableCandidate(const QString& candidate)
{
    const QString trimmed = candidate.trimmed();
    if (trimmed.isEmpty()) {
        return {};
    }

    const QFileInfo info(trimmed);
    if (info.isAbsolute() || trimmed.contains(QLatin1Char('/')) || trimmed.contains(QLatin1Char('\\'))) {
        return info.exists() && info.isFile() ? info.absoluteFilePath() : QString();
    }

    const QString pathExecutable = QStandardPaths::findExecutable(trimmed);
    if (!pathExecutable.isEmpty()) {
        return pathExecutable;
    }
    return info.exists() && info.isFile() ? info.absoluteFilePath() : QString();
}

QJsonObject processProbe(
    const QString& executable,
    const QStringList& arguments,
    int timeoutMs,
    const CancellationCallback& shouldCancel)
{
    QJsonObject object;
    object.insert(QStringLiteral("executable"), executable);
    object.insert(QStringLiteral("arguments"), QJsonArray::fromStringList(arguments));
    if (executable.isEmpty()) {
        object.insert(QStringLiteral("status"), QStringLiteral("missing"));
        object.insert(QStringLiteral("message"), QStringLiteral("X-AnyLabeling executable was not found."));
        return object;
    }

    QProcess process;
    process.start(executable, arguments);
    if (!process.waitForStarted(3000)) {
        object.insert(QStringLiteral("status"), QStringLiteral("failed"));
        object.insert(QStringLiteral("message"), process.errorString());
        return object;
    }

    QElapsedTimer timer;
    timer.start();
    while (process.state() != QProcess::NotRunning && timer.elapsed() < timeoutMs) {
        if (canceled(shouldCancel)) {
            process.kill();
            process.waitForFinished(1000);
            object.insert(QStringLiteral("status"), QStringLiteral("canceled"));
            object.insert(QStringLiteral("message"), QStringLiteral("Canceled by user"));
            return object;
        }
        process.waitForFinished(100);
    }
    if (process.state() != QProcess::NotRunning) {
        process.kill();
        process.waitForFinished(1000);
        object.insert(QStringLiteral("status"), QStringLiteral("timeout"));
        object.insert(QStringLiteral("message"), QStringLiteral("X-AnyLabeling command timed out."));
        return object;
    }

    const QString stdoutText = QString::fromLocal8Bit(process.readAllStandardOutput()).trimmed();
    const QString stderrText = QString::fromLocal8Bit(process.readAllStandardError()).trimmed();
    object.insert(QStringLiteral("exitCode"), process.exitCode());
    object.insert(QStringLiteral("stdout"), stdoutText);
    object.insert(QStringLiteral("stderr"), stderrText);
    object.insert(QStringLiteral("status"),
        process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0
            ? QStringLiteral("ok")
            : QStringLiteral("failed"));
    object.insert(QStringLiteral("message"), stdoutText.isEmpty() ? stderrText : stdoutText.left(500));
    return object;
}

QStringList readClassesTxt(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    QStringList classes;
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (!line.isEmpty()) {
            classes.append(line);
        }
    }
    return classes;
}

QStringList classNamesForDataset(const QString& datasetPath, const QString& format)
{
    if (format == QStringLiteral("yolo_detection")
        || format == QStringLiteral("yolo_segmentation")
        || format == QStringLiteral("yolo_obb")
        || format == QStringLiteral("yolo_txt")) {
        QString error;
        const YoloDataYaml layout = parseYoloDataYaml(datasetPath, &error);
        if (!layout.classNames.isEmpty()) {
            return layout.classNames;
        }
        if (layout.classCount > 0) {
            QStringList generated;
            for (int index = 0; index < layout.classCount; ++index) {
                generated.append(QStringLiteral("class_%1").arg(index));
            }
            return generated;
        }
    }
    const QString rootClasses = QDir(datasetPath).filePath(QStringLiteral("classes.txt"));
    const QStringList rootClassNames = readClassesTxt(rootClasses);
    if (!rootClassNames.isEmpty()) {
        return rootClassNames;
    }
    if (format == QStringLiteral("anomaly_folder")) {
        return {QStringLiteral("ok"), QStringLiteral("ng")};
    }
    return {};
}

QString existingClassesPathNear(const QString& sourcePath)
{
    const QFileInfo sourceInfo(sourcePath);
    const QString baseDir = sourceInfo.isDir() ? sourceInfo.absoluteFilePath() : sourceInfo.absolutePath();
    const QStringList candidates = {
        QDir(baseDir).filePath(QStringLiteral("classes.txt")),
        QDir(sourceInfo.absolutePath()).filePath(QStringLiteral("classes.txt"))
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return {};
}

QJsonArray stringArray(const QStringList& values)
{
    QJsonArray array;
    for (const QString& value : values) {
        array.append(value);
    }
    return array;
}

QJsonArray extractReviewSamples(const QJsonObject& root)
{
    const QStringList arrayKeys = {
        QStringLiteral("samples"),
        QStringLiteral("issues"),
        QStringLiteral("errorSamples"),
        QStringLiteral("candidates")
    };
    for (const QString& key : arrayKeys) {
        const QJsonArray array = root.value(key).toArray();
        if (!array.isEmpty()) {
            return array;
        }
    }
    return {};
}

QJsonArray reviewSamplesFromListFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }

    QJsonArray samples;
    while (!file.atEnd()) {
        const QString imagePath = QString::fromUtf8(file.readLine()).trimmed();
        if (imagePath.isEmpty()) {
            continue;
        }
        samples.append(QJsonObject{
            {QStringLiteral("source"), QStringLiteral("review_list")},
            {QStringLiteral("imagePath"), imagePath}
        });
    }
    return samples;
}

QJsonArray sourceReviewSamples(const QJsonObject& options)
{
    const QStringList jsonSourceKeys = {
        QStringLiteral("reviewSourcePath"),
        QStringLiteral("reworkSampleSetPath"),
        QStringLiteral("qualityFixManifestPath"),
        QStringLiteral("xAnyLabelingFixManifestPath")
    };
    for (const QString& key : jsonSourceKeys) {
        const QString sourcePath = options.value(key).toString();
        if (sourcePath.isEmpty() || !QFileInfo::exists(sourcePath)) {
            continue;
        }
        QJsonObject source;
        QString error;
        if (!readJsonFile(sourcePath, &source, &error)) {
            continue;
        }
        const QJsonArray samples = extractReviewSamples(source);
        if (!samples.isEmpty()) {
            return samples;
        }
    }

    const QString reviewListPath = options.value(QStringLiteral("reviewListPath")).toString();
    if (!reviewListPath.isEmpty() && QFileInfo::exists(reviewListPath)) {
        return reviewSamplesFromListFile(reviewListPath);
    }
    return {};
}

QStringList yoloSplitLabelDirs(const QString& datasetPath, const QString& format)
{
    if (format != QStringLiteral("yolo_detection")
        && format != QStringLiteral("yolo_segmentation")
        && format != QStringLiteral("yolo_obb")
        && format != QStringLiteral("yolo_txt")) {
        return {};
    }
    QStringList dirs;
    QString error;
    const YoloDataYaml layout = parseYoloDataYaml(datasetPath, &error);
    if (layout.exists) {
        for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
            const YoloSplitPaths paths = yoloSplitPaths(layout, split);
            if (!paths.labelDir.isEmpty() && QFileInfo::exists(paths.labelDir)) {
                dirs.append(QFileInfo(paths.labelDir).absoluteFilePath());
            }
        }
    }
    const QString labelsRoot = QDir(datasetPath).filePath(QStringLiteral("labels"));
    if (QFileInfo::exists(labelsRoot)) {
        dirs.append(QFileInfo(labelsRoot).absoluteFilePath());
    }
    dirs.removeDuplicates();
    return dirs;
}

QJsonArray modifiedLabelFiles(
    const QString& datasetPath,
    const QString& format,
    const QDateTime& since,
    int* scannedCount)
{
    QJsonArray changed;
    int scanned = 0;
    const QStringList labelDirs = yoloSplitLabelDirs(datasetPath, format);
    for (const QString& labelDir : labelDirs) {
        QDirIterator it(labelDir, {QStringLiteral("*.txt")}, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            const QString filePath = it.next();
            ++scanned;
            const QFileInfo info(filePath);
            if (since.isValid() && info.lastModified().toUTC() < since) {
                continue;
            }
            QJsonObject item;
            item.insert(QStringLiteral("path"), info.absoluteFilePath());
            item.insert(QStringLiteral("modifiedAt"), info.lastModified().toUTC().toString(Qt::ISODateWithMs));
            changed.append(item);
        }
    }
    if (scannedCount) {
        *scannedCount = scanned;
    }
    return changed;
}

bool isXLabelDocument(const QJsonObject& object)
{
    return object.contains(QStringLiteral("imagePath"))
        && object.value(QStringLiteral("shapes")).isArray();
}

QJsonArray modifiedXLabelFiles(
    const QString& labelsPath,
    const QDateTime& since,
    int* scannedCount)
{
    QJsonArray changed;
    int scanned = 0;
    if (labelsPath.isEmpty() || !QFileInfo::exists(labelsPath)) {
        if (scannedCount) {
            *scannedCount = scanned;
        }
        return changed;
    }

    QDirIterator it(labelsPath, {QStringLiteral("*.json")}, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString filePath = it.next();
        QJsonObject label;
        QString readError;
        if (!readJsonFile(filePath, &label, &readError) || !isXLabelDocument(label)) {
            continue;
        }
        ++scanned;
        const QFileInfo info(filePath);
        if (since.isValid() && info.lastModified().toUTC() < since) {
            continue;
        }
        QJsonObject item;
        item.insert(QStringLiteral("source"), QStringLiteral("session_output"));
        item.insert(QStringLiteral("path"), info.absoluteFilePath());
        item.insert(QStringLiteral("modifiedAt"), info.lastModified().toUTC().toString(Qt::ISODateWithMs));
        item.insert(QStringLiteral("imagePath"), label.value(QStringLiteral("imagePath")).toString());
        item.insert(QStringLiteral("shapeCount"), label.value(QStringLiteral("shapes")).toArray().size());
        changed.append(item);
    }
    if (scannedCount) {
        *scannedCount = scanned;
    }
    return changed;
}

QJsonArray combinedLabelChanges(const QJsonArray& datasetLabels, const QJsonArray& sessionLabels)
{
    QJsonArray combined;
    for (const QJsonValue& value : datasetLabels) {
        combined.append(value);
    }
    for (const QJsonValue& value : sessionLabels) {
        combined.append(value);
    }
    return combined;
}

QStringList xLabelImageFilters()
{
    return {
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.png"),
        QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff"),
        QStringLiteral("*.webp")
    };
}

bool directoryHasImages(const QString& path)
{
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        return false;
    }
    QDirIterator it(path, xLabelImageFilters(), QDir::Files);
    return it.hasNext();
}

QString firstXLabelImagePath(const QString& labelsPath)
{
    QDirIterator it(labelsPath, {QStringLiteral("*.json")}, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QJsonObject label;
        QString error;
        if (readJsonFile(it.next(), &label, &error) && isXLabelDocument(label)) {
            const QString imagePath = label.value(QStringLiteral("imagePath")).toString().trimmed();
            if (!imagePath.isEmpty()) {
                return imagePath;
            }
        }
    }
    return {};
}

QString yoloModeForFormat(const QString& format)
{
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        return QStringLiteral("detect");
    }
    if (format == QStringLiteral("yolo_segmentation")) {
        return QStringLiteral("segment");
    }
    if (format == QStringLiteral("yolo_obb")) {
        return QStringLiteral("obb");
    }
    return {};
}

bool isYoloFormat(const QString& format)
{
    return format == QStringLiteral("yolo_detection")
        || format == QStringLiteral("yolo_segmentation")
        || format == QStringLiteral("yolo_obb")
        || format == QStringLiteral("yolo_txt");
}

QString comparablePath(QString path)
{
    path = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    path.replace(QLatin1Char('\\'), QLatin1Char('/'));
#ifdef Q_OS_WIN
    path = path.toLower();
#endif
    return path;
}

struct YoloCliPaths {
    QString imagesDir;
    QString labelsDir;
    QString yamlPath;
    QString yamlError;
    QString stagingRoot;
    int stagedImageCount = 0;
    int stagedLabelCount = 0;
    int emptyLabelCount = 0;
    bool staged = false;
    QString errorCode;
    QString errorMessage;
};

struct YoloCliSplitDirs {
    QString split;
    QString imageDir;
    QString labelDir;
};

struct XLabelCliPaths {
    QString imagesDir;
    QString labelsDir;
    QString errorCode;
    QString errorMessage;
};

QStringList cliImageFilters()
{
    return {
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.png"),
        QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff"),
        QStringLiteral("*.webp")
    };
}

QVector<YoloCliSplitDirs> yoloSplitDirsForCli(const QString& sourcePath, const YoloDataYaml& layout)
{
    QVector<YoloCliSplitDirs> dirs;
    if (layout.exists) {
        QStringList splits = {QStringLiteral("train"), QStringLiteral("val")};
        if (layout.splitImagePaths.contains(QStringLiteral("test"))) {
            splits.append(QStringLiteral("test"));
        }
        for (const QString& split : splits) {
            const YoloSplitPaths paths = yoloSplitPaths(layout, split);
            if (!paths.imageDir.isEmpty()) {
                dirs.append({split, paths.imageDir, paths.labelDir});
            }
        }
        return dirs;
    }

    const QString imagesRoot = QDir(sourcePath).filePath(QStringLiteral("images"));
    const QString labelsRoot = QDir(sourcePath).filePath(QStringLiteral("labels"));
    if (QFileInfo::exists(QDir(imagesRoot).filePath(QStringLiteral("train")))
        || QFileInfo::exists(QDir(imagesRoot).filePath(QStringLiteral("val")))) {
        dirs.append({QStringLiteral("train"),
            QDir(imagesRoot).filePath(QStringLiteral("train")),
            QDir(labelsRoot).filePath(QStringLiteral("train"))});
        dirs.append({QStringLiteral("val"),
            QDir(imagesRoot).filePath(QStringLiteral("val")),
            QDir(labelsRoot).filePath(QStringLiteral("val"))});
        if (QFileInfo::exists(QDir(imagesRoot).filePath(QStringLiteral("test")))) {
            dirs.append({QStringLiteral("test"),
                QDir(imagesRoot).filePath(QStringLiteral("test")),
                QDir(labelsRoot).filePath(QStringLiteral("test"))});
        }
    } else {
        dirs.append({QStringLiteral("images"), imagesRoot, labelsRoot});
    }
    return dirs;
}

QString sanitizedStagingFileName(const QString& split, QString relativePath)
{
    relativePath = QDir::cleanPath(relativePath);
    relativePath.replace(QLatin1Char('\\'), QLatin1Char('/'));
    QString name = split.isEmpty() ? relativePath : QStringLiteral("%1__%2").arg(split, relativePath);
    for (int index = 0; index < name.size(); ++index) {
        const QChar ch = name.at(index);
        if (ch.isLetterOrNumber() || ch == QLatin1Char('.') || ch == QLatin1Char('_') || ch == QLatin1Char('-')) {
            continue;
        }
        name[index] = QLatin1Char('_');
    }
    return name;
}

QString uniqueStagingImageName(const QString& proposedName, QSet<QString>* usedNames)
{
    const QFileInfo info(proposedName);
    const QString suffix = info.suffix();
    const QString base = suffix.isEmpty()
        ? proposedName
        : proposedName.left(proposedName.size() - suffix.size() - 1);
    QString candidate = proposedName;
    int index = 1;
    while (usedNames->contains(comparablePath(candidate))) {
        candidate = suffix.isEmpty()
            ? QStringLiteral("%1_%2").arg(base).arg(index)
            : QStringLiteral("%1_%2.%3").arg(base).arg(index).arg(suffix);
        ++index;
    }
    usedNames->insert(comparablePath(candidate));
    return candidate;
}

bool copyFileForStaging(const QString& sourcePath, const QString& targetPath, QString* error)
{
    QDir().mkpath(QFileInfo(targetPath).absolutePath());
    QFile::remove(targetPath);
    if (!QFileInfo::exists(sourcePath)) {
        return writeTextFile(targetPath, QString(), error);
    }
    if (!QFile::copy(sourcePath, targetPath)) {
        if (error) {
            *error = QStringLiteral("Cannot copy staging file from %1 to %2.").arg(sourcePath, targetPath);
        }
        return false;
    }
    return true;
}

QSet<QString> persistXLabelImages(
    const QString& stagedImagesDir,
    const QString& outputPath,
    int* copiedCount,
    QString* error)
{
    QSet<QString> imageNames;
    if (copiedCount) {
        *copiedCount = 0;
    }
    const QString persistentImagesDir = QDir(outputPath).filePath(QStringLiteral("images"));
    if (!QDir().mkpath(persistentImagesDir)) {
        if (error) {
            *error = QStringLiteral("Cannot create persistent XLABEL image directory: %1").arg(persistentImagesDir);
        }
        return imageNames;
    }

    QDirIterator it(stagedImagesDir, cliImageFilters(), QDir::Files);
    while (it.hasNext()) {
        const QString stagedImagePath = it.next();
        const QString imageName = QFileInfo(stagedImagePath).fileName();
        const QString targetPath = QDir(persistentImagesDir).filePath(imageName);
        QString copyError;
        if (!copyFileForStaging(stagedImagePath, targetPath, &copyError)) {
            if (error) {
                *error = copyError;
            }
            return {};
        }
        imageNames.insert(imageName);
        if (copiedCount) {
            ++(*copiedCount);
        }
    }
    return imageNames;
}

int rewriteXLabelImagePaths(
    const QString& outputPath,
    const QSet<QString>& imageNames,
    QString* error)
{
    int rewrittenCount = 0;
    const QDir outputDir(outputPath);
    for (const QString& imageName : imageNames) {
        const QString labelPath = outputDir.filePath(QFileInfo(imageName).completeBaseName() + QStringLiteral(".json"));
        if (!QFileInfo::exists(labelPath)) {
            continue;
        }
        QJsonObject label;
        QString readError;
        if (!readJsonFile(labelPath, &label, &readError)) {
            if (error) {
                *error = readError;
            }
            return -1;
        }
        label.insert(QStringLiteral("imagePath"), QStringLiteral("images/%1").arg(imageName));
        QString writeError;
        if (!writeJsonFile(labelPath, label, &writeError)) {
            if (error) {
                *error = writeError;
            }
            return -1;
        }
        ++rewrittenCount;
    }
    return rewrittenCount;
}

YoloCliPaths stageYoloInputForCli(
    const QString& sourcePath,
    const YoloDataYaml& layout,
    const QString& stagingRoot)
{
    YoloCliPaths paths;
    paths.staged = true;
    paths.stagingRoot = stagingRoot;
    paths.imagesDir = QDir(stagingRoot).filePath(QStringLiteral("images"));
    paths.labelsDir = QDir(stagingRoot).filePath(QStringLiteral("labels"));
    QDir().mkpath(paths.imagesDir);
    QDir().mkpath(paths.labelsDir);

    QSet<QString> usedSourceImages;
    QSet<QString> usedStagingNames;
    const QVector<YoloCliSplitDirs> splitDirs = yoloSplitDirsForCli(sourcePath, layout);
    for (const YoloCliSplitDirs& splitDirsForCli : splitDirs) {
        if (!QFileInfo::exists(splitDirsForCli.imageDir)) {
            continue;
        }
        QDirIterator it(splitDirsForCli.imageDir, cliImageFilters(), QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            const QString imagePath = it.next();
            const QString sourceKey = comparablePath(imagePath);
            if (usedSourceImages.contains(sourceKey)) {
                continue;
            }
            usedSourceImages.insert(sourceKey);

            const QString relativeImagePath = QDir(splitDirsForCli.imageDir).relativeFilePath(imagePath);
            const QString stagedImageName = uniqueStagingImageName(
                sanitizedStagingFileName(splitDirsForCli.split, relativeImagePath),
                &usedStagingNames);
            const QString stagedImagePath = QDir(paths.imagesDir).filePath(stagedImageName);
            QString error;
            if (!copyFileForStaging(imagePath, stagedImagePath, &error)) {
                paths.errorCode = QStringLiteral("xanylabeling_staging_failed");
                paths.errorMessage = error;
                return paths;
            }

            const QFileInfo relativeInfo(relativeImagePath);
            const QString relativeLabelPath = QDir(relativeInfo.path()).filePath(relativeInfo.completeBaseName() + QStringLiteral(".txt"));
            const QString sourceLabelPath = QDir(splitDirsForCli.labelDir).filePath(relativeLabelPath);
            const QString stagedLabelPath = QDir(paths.labelsDir).filePath(QFileInfo(stagedImageName).completeBaseName() + QStringLiteral(".txt"));
            if (!copyFileForStaging(sourceLabelPath, stagedLabelPath, &error)) {
                paths.errorCode = QStringLiteral("xanylabeling_staging_failed");
                paths.errorMessage = error;
                return paths;
            }
            ++paths.stagedImageCount;
            if (QFileInfo::exists(sourceLabelPath)) {
                ++paths.stagedLabelCount;
            } else {
                ++paths.emptyLabelCount;
            }
        }
    }

    if (paths.stagedImageCount <= 0) {
        paths.errorCode = QStringLiteral("yolo_images_missing");
        paths.errorMessage = QStringLiteral("No YOLO images were found for X-AnyLabeling CLI staging.");
    }
    return paths;
}

YoloCliPaths yoloCliPathsForSource(
    const QString& sourcePath,
    const QString& format,
    const QJsonObject& options,
    const QString& stagingRoot)
{
    YoloCliPaths paths;
    const QString optionImages = options.value(QStringLiteral("imagesPath")).toString().trimmed();
    const QString optionLabels = options.value(QStringLiteral("labelsPath")).toString().trimmed();
    if (!isYoloFormat(format)) {
        paths.imagesDir = optionImages.isEmpty()
            ? QDir(sourcePath).filePath(QStringLiteral("images"))
            : normalizedPath(optionImages);
        paths.labelsDir = optionLabels.isEmpty()
            ? QDir(sourcePath).filePath(QStringLiteral("labels"))
            : normalizedPath(optionLabels);
        return paths;
    }

    QString yamlError;
    const YoloDataYaml layout = parseYoloDataYaml(sourcePath, &yamlError);
    paths.yamlPath = layout.yamlPath;
    paths.yamlError = yamlError;
    if (!optionImages.isEmpty() || !optionLabels.isEmpty()) {
        paths.imagesDir = optionImages.isEmpty()
            ? QDir(sourcePath).filePath(QStringLiteral("images"))
            : normalizedPath(optionImages);
        paths.labelsDir = optionLabels.isEmpty()
            ? QDir(sourcePath).filePath(QStringLiteral("labels"))
            : normalizedPath(optionLabels);
        return paths;
    }

    if (stagingRoot.isEmpty()) {
        paths.errorCode = QStringLiteral("xanylabeling_staging_failed");
        paths.errorMessage = QStringLiteral("Cannot create temporary staging directory for X-AnyLabeling CLI input.");
        return paths;
    }

    paths = stageYoloInputForCli(sourcePath, layout, stagingRoot);
    paths.yamlPath = layout.yamlPath;
    paths.yamlError = yamlError;
    if (!optionImages.isEmpty()) {
        paths.imagesDir = normalizedPath(optionImages);
    }
    if (!optionLabels.isEmpty()) {
        paths.labelsDir = normalizedPath(optionLabels);
    }
    return paths;
}

QString inferXLabelImagesDir(const QString& labelsDir)
{
    const QString imagePath = firstXLabelImagePath(labelsDir);
    if (!imagePath.isEmpty()) {
        const QFileInfo imageInfo(imagePath);
        if (imageInfo.isAbsolute()) {
            if (imageInfo.exists() || directoryHasImages(imageInfo.absolutePath())) {
                return imageInfo.absolutePath();
            }
        }
        const QString resolvedImagePath = QDir(labelsDir).filePath(imagePath);
        const QFileInfo resolvedImageInfo(resolvedImagePath);
        if (resolvedImageInfo.exists()) {
            return resolvedImageInfo.absolutePath();
        }
        const QString resolvedImageDir = resolvedImageInfo.absolutePath();
        if (directoryHasImages(resolvedImageDir)) {
            return QFileInfo(resolvedImageDir).absoluteFilePath();
        }
    }

    const QString imagesDir = QDir(labelsDir).filePath(QStringLiteral("images"));
    if (directoryHasImages(imagesDir)) {
        return QFileInfo(imagesDir).absoluteFilePath();
    }
    if (directoryHasImages(labelsDir)) {
        return QFileInfo(labelsDir).absoluteFilePath();
    }
    return {};
}

XLabelCliPaths xLabelCliPathsForSource(const QString& sourcePath, const QJsonObject& options)
{
    XLabelCliPaths paths;
    const QString optionImages = options.value(QStringLiteral("imagesPath")).toString().trimmed();
    const QString optionLabels = options.value(QStringLiteral("labelsPath")).toString().trimmed();
    paths.labelsDir = optionLabels.isEmpty() ? normalizedPath(sourcePath) : normalizedPath(optionLabels);
    if (paths.labelsDir.isEmpty() || !QFileInfo::exists(paths.labelsDir)) {
        paths.errorCode = QStringLiteral("xlabel_labels_missing");
        paths.errorMessage = QStringLiteral("XLABEL label directory was not found.");
        return paths;
    }

    paths.imagesDir = optionImages.isEmpty()
        ? inferXLabelImagesDir(paths.labelsDir)
        : normalizedPath(optionImages);
    if (paths.imagesDir.isEmpty() || !QFileInfo::exists(paths.imagesDir)) {
        paths.errorCode = QStringLiteral("xlabel_images_missing");
        paths.errorMessage = QStringLiteral("XLABEL to YOLO conversion requires an image directory. Set imagesPath or place images beside the XLABEL files.");
        return paths;
    }
    return paths;
}

QJsonObject conversionReportBase(const DatasetConversionRequest& request, const QString& engine)
{
    QJsonObject report;
    report.insert(QStringLiteral("ok"), false);
    report.insert(QStringLiteral("conversionEngine"), engine);
    report.insert(QStringLiteral("sourceFormat"), request.sourceFormat);
    report.insert(QStringLiteral("targetFormat"), request.targetFormat);
    report.insert(QStringLiteral("sourcePath"), normalizedPath(request.sourcePath));
    report.insert(QStringLiteral("outputPath"), normalizedPath(request.outputPath));
    report.insert(QStringLiteral("checkedAt"), nowIso());
    return report;
}

bool writeConversionReport(DatasetConversionResult* result, const QJsonObject& report)
{
    const QString reportPath = QDir(result->outputPath).filePath(QStringLiteral("dataset_conversion_report.json"));
    QString error;
    const bool wrote = writeJsonFile(reportPath, report, &error);
    if (wrote) {
        result->reportPath = reportPath;
    } else {
        result->errorCode = QStringLiteral("report_write_failed");
        result->errorMessage = error;
    }
    return wrote;
}

DatasetConversionResult failConversionWithReport(
    DatasetConversionResult result,
    QJsonObject report,
    const QString& errorCode,
    const QString& errorMessage)
{
    result.ok = false;
    result.errorCode = errorCode;
    result.errorMessage = errorMessage;
    report.insert(QStringLiteral("errorCode"), errorCode);
    report.insert(QStringLiteral("errorMessage"), errorMessage);
    writeConversionReport(&result, report);
    return result;
}

} // namespace

QStringList xAnyLabelingExecutableCandidates(const QJsonObject& options)
{
    QStringList candidates;
    const QString requested = options.value(QStringLiteral("xAnyLabelingExecutable")).toString().trimmed();
    if (!requested.isEmpty()) {
        candidates.append(requested);
    }
    const QString envProgram = QString::fromLocal8Bit(qgetenv("AITRAIN_XANYLABELING_EXE")).trimmed();
    if (!envProgram.isEmpty()) {
        candidates.append(envProgram);
    }
    if (options.value(QStringLiteral("disableXAnyLabelingAutoDiscovery")).toBool(false)) {
        candidates.removeDuplicates();
        return candidates;
    }
    const QString appDir = QCoreApplication::applicationDirPath();
    candidates << QDir(appDir).filePath(QStringLiteral("X-AnyLabeling.exe"))
               << QDir(appDir).filePath(QStringLiteral("xanylabeling.exe"))
               << QDir(appDir).filePath(QStringLiteral("tools/x-anylabeling/X-AnyLabeling.exe"))
               << QDir::current().absoluteFilePath(QStringLiteral(".deps/tools/annotation-tools/X-AnyLabeling/X-AnyLabeling.exe"))
               << QDir::current().absoluteFilePath(QStringLiteral(".deps/annotation-tools/X-AnyLabeling/X-AnyLabeling.exe"))
               << QStringLiteral("xanylabeling")
               << QStringLiteral("X-AnyLabeling.exe");
    candidates.removeDuplicates();
    return candidates;
}

QString resolveXAnyLabelingExecutable(const QJsonObject& options)
{
    for (const QString& candidate : xAnyLabelingExecutableCandidates(options)) {
        const QString resolved = resolveExecutableCandidate(candidate);
        if (!resolved.isEmpty()) {
            return resolved;
        }
    }
    return {};
}

WorkflowResult inspectXAnyLabelingEnvironment(
    const QString& outputPath,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel)
{
    WorkflowResult result;
    const QString outDir = outputPath.isEmpty() ? QDir::tempPath() : outputPath;
    const QString executable = resolveXAnyLabelingExecutable(options);
    const QString reportPath = QDir(outDir).filePath(QStringLiteral("xanylabeling_environment_report.json"));

    QJsonObject report;
    report.insert(QStringLiteral("schemaVersion"), 1);
    report.insert(QStringLiteral("kind"), QStringLiteral("xanylabeling_environment_report"));
    report.insert(QStringLiteral("checkedAt"), nowIso());
    report.insert(QStringLiteral("executable"), executable);
    report.insert(QStringLiteral("candidates"), stringArray(xAnyLabelingExecutableCandidates(options)));
    report.insert(QStringLiteral("licenseBoundary"), QStringLiteral("local_external_dependency"));
    report.insert(QStringLiteral("redistributionReviewRequired"), true);

    if (executable.isEmpty()) {
        report.insert(QStringLiteral("status"), QStringLiteral("missing"));
        report.insert(QStringLiteral("message"), QStringLiteral("X-AnyLabeling was not found. Configure AITRAIN_XANYLABELING_EXE or install it under .deps/tools/annotation-tools/X-AnyLabeling."));
    } else {
        const QJsonObject version = processProbe(executable, {QStringLiteral("version")}, 10000, shouldCancel);
        const bool versionCanceled = version.value(QStringLiteral("status")).toString() == QStringLiteral("canceled");
        const QJsonObject checks = versionCanceled
            ? QJsonObject{}
            : processProbe(executable, {QStringLiteral("checks")}, 20000, shouldCancel);
        const bool checksCanceled = checks.value(QStringLiteral("status")).toString() == QStringLiteral("canceled");
        const QJsonObject convert = (versionCanceled || checksCanceled)
            ? QJsonObject{}
            : processProbe(executable, {QStringLiteral("convert")}, 20000, shouldCancel);
        report.insert(QStringLiteral("versionProbe"), version);
        report.insert(QStringLiteral("checksProbe"), checks);
        report.insert(QStringLiteral("convertProbe"), convert);
        const bool convertCanceled = convert.value(QStringLiteral("status")).toString() == QStringLiteral("canceled");
        const bool wasCanceled = versionCanceled || checksCanceled || convertCanceled;
        const bool ok = version.value(QStringLiteral("status")).toString() == QStringLiteral("ok")
            || checks.value(QStringLiteral("status")).toString() == QStringLiteral("ok")
            || convert.value(QStringLiteral("status")).toString() == QStringLiteral("ok");
        report.insert(QStringLiteral("status"), wasCanceled ? QStringLiteral("canceled") : (ok ? QStringLiteral("ok") : QStringLiteral("failed")));
        report.insert(QStringLiteral("message"), wasCanceled
            ? QStringLiteral("Canceled by user")
            : (ok
            ? QStringLiteral("X-AnyLabeling executable is reachable.")
            : QStringLiteral("X-AnyLabeling executable was found but probes failed.")));
    }

    QString error;
    if (!writeJsonFile(reportPath, report, &error)) {
        result.ok = false;
        result.error = error;
        return result;
    }
    result.ok = report.value(QStringLiteral("status")).toString() != QStringLiteral("failed")
        && report.value(QStringLiteral("status")).toString() != QStringLiteral("canceled");
    result.error = result.ok ? QString() : report.value(QStringLiteral("message")).toString();
    result.reportPath = reportPath;
    result.payload = report;
    result.payload.insert(QStringLiteral("reportPath"), reportPath);
    return result;
}

WorkflowResult prepareAnnotationSession(
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel)
{
    WorkflowResult result;
    if (canceled(shouldCancel)) {
        result.error = QStringLiteral("Canceled by user");
        return result;
    }

    const QString cleanDatasetPath = normalizedPath(datasetPath);
    const QString cleanOutputPath = normalizedPath(outputPath);
    const QString mode = options.value(QStringLiteral("mode")).toString(QStringLiteral("quality_fix"));
    const QString executable = resolveXAnyLabelingExecutable(options);
    const QString manifestPath = QDir(cleanOutputPath).filePath(QStringLiteral("xany_session_manifest.json"));
    const QString classesPath = QDir(cleanOutputPath).filePath(QStringLiteral("classes.txt"));
    const QString reviewSamplesPath = QDir(cleanOutputPath).filePath(QStringLiteral("review_samples.json"));
    const QString launchRequestPath = QDir(cleanOutputPath).filePath(QStringLiteral("launch_request.json"));

    const QStringList classNames = classNamesForDataset(cleanDatasetPath, format);
    QString error;
    if (!classNames.isEmpty() && !writeTextFile(classesPath, classNames.join(QLatin1Char('\n')) + QLatin1Char('\n'), &error)) {
        result.error = error;
        return result;
    }

    const QJsonArray reviewSamples = sourceReviewSamples(options);
    QJsonObject reviewDocument;
    reviewDocument.insert(QStringLiteral("schemaVersion"), 1);
    reviewDocument.insert(QStringLiteral("kind"), QStringLiteral("xanylabeling_review_samples"));
    reviewDocument.insert(QStringLiteral("createdAt"), nowIso());
    reviewDocument.insert(QStringLiteral("datasetPath"), cleanDatasetPath);
    reviewDocument.insert(QStringLiteral("format"), format);
    reviewDocument.insert(QStringLiteral("mode"), mode);
    reviewDocument.insert(QStringLiteral("samples"), reviewSamples);
    if (!writeJsonFile(reviewSamplesPath, reviewDocument, &error)) {
        result.error = error;
        return result;
    }

    QStringList arguments;
    arguments << QStringLiteral("--filename") << cleanDatasetPath
              << QStringLiteral("--output") << cleanOutputPath
              << QStringLiteral("--autosave")
              << QStringLiteral("--no-auto-update-check");
    if (!classNames.isEmpty()) {
        arguments << QStringLiteral("--labels") << classesPath
                  << QStringLiteral("--validatelabel") << QStringLiteral("exact");
    }

    QJsonObject launchRequest;
    launchRequest.insert(QStringLiteral("executable"), executable);
    launchRequest.insert(QStringLiteral("arguments"), stringArray(arguments));
    launchRequest.insert(QStringLiteral("workingDirectory"), cleanDatasetPath);
    launchRequest.insert(QStringLiteral("launchMode"), QStringLiteral("external_process"));
    if (!writeJsonFile(launchRequestPath, launchRequest, &error)) {
        result.error = error;
        return result;
    }

    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 1);
    manifest.insert(QStringLiteral("kind"), QStringLiteral("xanylabeling_annotation_session"));
    manifest.insert(QStringLiteral("createdAt"), nowIso());
    manifest.insert(QStringLiteral("datasetPath"), cleanDatasetPath);
    manifest.insert(QStringLiteral("format"), format);
    manifest.insert(QStringLiteral("mode"), mode);
    manifest.insert(QStringLiteral("outputPath"), cleanOutputPath);
    manifest.insert(QStringLiteral("classCount"), classNames.size());
    manifest.insert(QStringLiteral("classesPath"), classNames.isEmpty() ? QString() : classesPath);
    manifest.insert(QStringLiteral("reviewSamplesPath"), reviewSamplesPath);
    manifest.insert(QStringLiteral("reviewSampleCount"), reviewSamples.size());
    manifest.insert(QStringLiteral("launchRequestPath"), launchRequestPath);
    manifest.insert(QStringLiteral("xAnyLabelingExecutable"), executable);
    manifest.insert(QStringLiteral("toolStatus"), executable.isEmpty() ? QStringLiteral("missing") : QStringLiteral("ready"));
    manifest.insert(QStringLiteral("note"), QStringLiteral("AITrain launches X-AnyLabeling as a local external tool and does not overwrite labels automatically."));
    if (!writeJsonFile(manifestPath, manifest, &error)) {
        result.error = error;
        return result;
    }

    result.ok = true;
    result.reportPath = manifestPath;
    result.payload = manifest;
    result.payload.insert(QStringLiteral("status"), executable.isEmpty() ? QStringLiteral("blocked") : QStringLiteral("ready"));
    result.payload.insert(QStringLiteral("manifestPath"), manifestPath);
    return result;
}

WorkflowResult syncAnnotationSession(
    const QString& sessionManifestPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel)
{
    WorkflowResult result;
    if (canceled(shouldCancel)) {
        result.error = QStringLiteral("Canceled by user");
        return result;
    }

    const QString manifestPath = normalizedPath(sessionManifestPath.isEmpty()
        ? options.value(QStringLiteral("sessionManifestPath")).toString()
        : sessionManifestPath);
    const QString cleanDatasetPath = normalizedPath(datasetPath);
    const QString cleanOutputPath = normalizedPath(outputPath);
    const QString reportPath = QDir(cleanOutputPath).filePath(QStringLiteral("annotation_sync_report.json"));

    QJsonObject manifest;
    QString readError;
    if (!manifestPath.isEmpty() && QFileInfo::exists(manifestPath)) {
        readJsonFile(manifestPath, &manifest, &readError);
    }
    const QDateTime createdAt = QDateTime::fromString(manifest.value(QStringLiteral("createdAt")).toString(), Qt::ISODateWithMs);
    const QString sessionOutputPath = normalizedPath(manifest.value(QStringLiteral("outputPath")).toString());
    int scannedDatasetLabelCount = 0;
    const QJsonArray changedDatasetLabels = modifiedLabelFiles(cleanDatasetPath, format, createdAt, &scannedDatasetLabelCount);
    int scannedSessionOutputLabelCount = 0;
    const QJsonArray changedSessionOutputLabels = modifiedXLabelFiles(sessionOutputPath, createdAt, &scannedSessionOutputLabelCount);
    const QJsonArray changedLabels = combinedLabelChanges(changedDatasetLabels, changedSessionOutputLabels);

    QJsonObject report;
    report.insert(QStringLiteral("schemaVersion"), 1);
    report.insert(QStringLiteral("kind"), QStringLiteral("annotation_sync_report"));
    report.insert(QStringLiteral("checkedAt"), nowIso());
    report.insert(QStringLiteral("datasetPath"), cleanDatasetPath);
    report.insert(QStringLiteral("format"), format);
    report.insert(QStringLiteral("sessionManifestPath"), manifestPath);
    report.insert(QStringLiteral("sessionOutputPath"), sessionOutputPath);
    report.insert(QStringLiteral("sessionLoaded"), !manifest.isEmpty());
    report.insert(QStringLiteral("sessionReadError"), readError);
    report.insert(QStringLiteral("scannedDatasetLabelCount"), scannedDatasetLabelCount);
    report.insert(QStringLiteral("modifiedDatasetLabelCount"), changedDatasetLabels.size());
    report.insert(QStringLiteral("modifiedDatasetLabels"), changedDatasetLabels);
    report.insert(QStringLiteral("scannedSessionOutputLabelCount"), scannedSessionOutputLabelCount);
    report.insert(QStringLiteral("modifiedSessionOutputLabelCount"), changedSessionOutputLabels.size());
    report.insert(QStringLiteral("modifiedSessionOutputLabels"), changedSessionOutputLabels);
    report.insert(QStringLiteral("scannedLabelCount"), scannedDatasetLabelCount + scannedSessionOutputLabelCount);
    report.insert(QStringLiteral("modifiedLabelCount"), changedLabels.size());
    report.insert(QStringLiteral("modifiedLabels"), changedLabels);
    report.insert(QStringLiteral("nextAction"), QStringLiteral("run_dataset_validation_or_quality_report"));
    report.insert(QStringLiteral("status"), QStringLiteral("synced"));
    report.insert(QStringLiteral("message"), QStringLiteral("Annotation sync inspected dataset labels and session XLABEL output timestamps. AITrain did not modify user labels."));

    QString error;
    if (!writeJsonFile(reportPath, report, &error)) {
        result.error = error;
        return result;
    }
    result.ok = true;
    result.reportPath = reportPath;
    result.payload = report;
    result.payload.insert(QStringLiteral("reportPath"), reportPath);
    return result;
}

DatasetConversionResult convertDatasetWithXAnyLabelingCli(
    const DatasetConversionRequest& request,
    const CancellationCallback& shouldCancel)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = normalizedPath(request.sourcePath);
    result.outputPath = normalizedPath(request.outputPath);
    result.conversionMatrixVersion = 2;
    result.imagePolicy = QStringLiteral("external_xanylabeling_cli");

    if (canceled(shouldCancel)) {
        result.errorCode = QStringLiteral("canceled");
        result.errorMessage = QStringLiteral("Canceled by user");
        return result;
    }

    const QString executable = resolveXAnyLabelingExecutable(request.options);
    QJsonObject report = conversionReportBase(request, QStringLiteral("xanylabeling_cli"));
    report.insert(QStringLiteral("xAnyLabelingExecutable"), executable);
    if (executable.isEmpty()) {
        result.errorCode = QStringLiteral("xanylabeling_missing");
        result.errorMessage = QStringLiteral("X-AnyLabeling executable was not found.");
        report.insert(QStringLiteral("errorCode"), result.errorCode);
        report.insert(QStringLiteral("errorMessage"), result.errorMessage);
        writeConversionReport(&result, report);
        return result;
    }

    const QString sourceFormat = request.sourceFormat.trimmed();
    const QString targetFormat = request.targetFormat.trimmed();
    QString task;
    QString mode;
    QStringList arguments;
    QString classesPath = request.options.value(QStringLiteral("classesPath")).toString();
    QTemporaryDir yoloStagingDir(QDir::temp().filePath(QStringLiteral("aitrain_xanylabeling_cli_XXXXXX")));
    YoloCliPaths yoloCliPaths;
    bool hasYoloCliPaths = false;
    if (targetFormat == QStringLiteral("xanylabeling_xlabel")) {
        mode = yoloModeForFormat(sourceFormat);
        if (!mode.isEmpty()) {
            task = QStringLiteral("yolo2xlabel");
            if (classesPath.isEmpty()) {
                const QStringList classNames = classNamesForDataset(result.sourcePath, sourceFormat);
                if (!classNames.isEmpty()) {
                    classesPath = QDir(result.outputPath).filePath(QStringLiteral("classes.txt"));
                    QString writeError;
                    if (!writeTextFile(classesPath, classNames.join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
                        return failConversionWithReport(
                            result,
                            report,
                            QStringLiteral("classes_write_failed"),
                            writeError);
                    }
                }
            }
            if (classesPath.isEmpty()) {
                return failConversionWithReport(
                    result,
                    report,
                    QStringLiteral("classes_missing"),
                    QStringLiteral("YOLO to XLABEL conversion requires classes.txt or class names in data.yaml."));
            }
            yoloCliPaths = yoloCliPathsForSource(
                result.sourcePath,
                sourceFormat,
                request.options,
                yoloStagingDir.isValid() ? yoloStagingDir.path() : QString());
            hasYoloCliPaths = true;
            if (!yoloCliPaths.errorCode.isEmpty()) {
                return failConversionWithReport(
                    result,
                    report,
                    yoloCliPaths.errorCode,
                    yoloCliPaths.errorMessage);
            }
            report.insert(QStringLiteral("resolvedImagesPath"), yoloCliPaths.imagesDir);
            report.insert(QStringLiteral("resolvedLabelsPath"), yoloCliPaths.labelsDir);
            report.insert(QStringLiteral("stagedInput"), yoloCliPaths.staged);
            report.insert(QStringLiteral("stagedImageCount"), yoloCliPaths.stagedImageCount);
            report.insert(QStringLiteral("stagedLabelCount"), yoloCliPaths.stagedLabelCount);
            report.insert(QStringLiteral("emptyLabelCount"), yoloCliPaths.emptyLabelCount);
            if (!yoloCliPaths.yamlPath.isEmpty()) {
                report.insert(QStringLiteral("dataYamlPath"), yoloCliPaths.yamlPath);
            }
            if (!yoloCliPaths.yamlError.isEmpty()) {
                report.insert(QStringLiteral("dataYamlWarning"), yoloCliPaths.yamlError);
            }
            arguments << QStringLiteral("convert") << QStringLiteral("--task") << task
                      << QStringLiteral("--mode") << mode
                      << QStringLiteral("--images") << yoloCliPaths.imagesDir
                      << QStringLiteral("--labels") << yoloCliPaths.labelsDir
                      << QStringLiteral("--output") << result.outputPath;
            if (!classesPath.isEmpty()) {
                arguments << QStringLiteral("--classes") << classesPath;
            }
        }
    } else if (sourceFormat == QStringLiteral("xanylabeling_xlabel")) {
        mode = yoloModeForFormat(targetFormat);
        if (!mode.isEmpty()) {
            task = QStringLiteral("xlabel2yolo");
            if (classesPath.isEmpty()) {
                classesPath = existingClassesPathNear(result.sourcePath);
            }
            if (classesPath.isEmpty()) {
                return failConversionWithReport(
                    result,
                    report,
                    QStringLiteral("classes_missing"),
                    QStringLiteral("XLABEL to YOLO conversion requires classes.txt in the XLABEL source directory or a classesPath option."));
            }
            const XLabelCliPaths xLabelCliPaths = xLabelCliPathsForSource(result.sourcePath, request.options);
            if (!xLabelCliPaths.errorCode.isEmpty()) {
                return failConversionWithReport(
                    result,
                    report,
                    xLabelCliPaths.errorCode,
                    xLabelCliPaths.errorMessage);
            }
            report.insert(QStringLiteral("resolvedImagesPath"), xLabelCliPaths.imagesDir);
            report.insert(QStringLiteral("resolvedLabelsPath"), xLabelCliPaths.labelsDir);
            arguments << QStringLiteral("convert") << QStringLiteral("--task") << task
                      << QStringLiteral("--mode") << mode
                      << QStringLiteral("--images") << xLabelCliPaths.imagesDir
                      << QStringLiteral("--labels") << xLabelCliPaths.labelsDir
                      << QStringLiteral("--output") << result.outputPath;
            arguments << QStringLiteral("--classes") << classesPath;
        }
    }

    if (task.isEmpty()) {
        result.errorCode = QStringLiteral("unsupported_xanylabeling_cli_pair");
        result.errorMessage = QStringLiteral("X-AnyLabeling CLI conversion is only enabled for YOLO detect/segment/OBB <-> XLABEL in this release.");
        report.insert(QStringLiteral("errorCode"), result.errorCode);
        report.insert(QStringLiteral("errorMessage"), result.errorMessage);
        writeConversionReport(&result, report);
        return result;
    }

    report.insert(QStringLiteral("command"), QJsonObject{
        {QStringLiteral("task"), task},
        {QStringLiteral("mode"), mode},
        {QStringLiteral("arguments"), stringArray(arguments)}
    });
    const QJsonObject probe = processProbe(executable, arguments, request.options.value(QStringLiteral("timeoutMs")).toInt(120000), shouldCancel);
    report.insert(QStringLiteral("process"), probe);
    const QString probeStatus = probe.value(QStringLiteral("status")).toString();
    if (probeStatus == QStringLiteral("canceled")) {
        result.errorCode = QStringLiteral("canceled");
        result.errorMessage = QStringLiteral("Canceled by user");
        report.insert(QStringLiteral("errorCode"), result.errorCode);
        report.insert(QStringLiteral("errorMessage"), result.errorMessage);
        writeConversionReport(&result, report);
        return result;
    }
    result.ok = probeStatus == QStringLiteral("ok");
    if (!result.ok) {
        result.errorCode = QStringLiteral("xanylabeling_cli_failed");
        result.errorMessage = probe.value(QStringLiteral("message")).toString(QStringLiteral("X-AnyLabeling CLI conversion failed."));
        report.insert(QStringLiteral("errorCode"), result.errorCode);
        report.insert(QStringLiteral("errorMessage"), result.errorMessage);
    } else {
        if (hasYoloCliPaths && yoloCliPaths.staged) {
            QString persistError;
            int persistedImageCount = 0;
            const QSet<QString> persistedImageNames = persistXLabelImages(
                yoloCliPaths.imagesDir,
                result.outputPath,
                &persistedImageCount,
                &persistError);
            if (!persistError.isEmpty()) {
                return failConversionWithReport(
                    result,
                    report,
                    QStringLiteral("xanylabeling_output_persist_failed"),
                    persistError);
            }
            if (persistedImageCount != yoloCliPaths.stagedImageCount) {
                return failConversionWithReport(
                    result,
                    report,
                    QStringLiteral("xanylabeling_output_persist_incomplete"),
                    QStringLiteral("Persisted %1 of %2 staged XLABEL images.")
                        .arg(persistedImageCount)
                        .arg(yoloCliPaths.stagedImageCount));
            }
            QString rewriteError;
            const int rewrittenXLabelCount = rewriteXLabelImagePaths(
                result.outputPath,
                persistedImageNames,
                &rewriteError);
            if (rewrittenXLabelCount < 0) {
                return failConversionWithReport(
                    result,
                    report,
                    QStringLiteral("xanylabeling_image_path_rewrite_failed"),
                    rewriteError);
            }
            const QString persistentImagesPath = QDir(result.outputPath).filePath(QStringLiteral("images"));
            report.insert(QStringLiteral("persistentImagesPath"), persistentImagesPath);
            report.insert(QStringLiteral("persistedImageCount"), persistedImageCount);
            report.insert(QStringLiteral("rewrittenXLabelCount"), rewrittenXLabelCount);
            result.outputFiles.insert(QStringLiteral("imagesRoot"), persistentImagesPath);
        }
        report.insert(QStringLiteral("ok"), true);
        result.convertedSampleCount = -1;
        result.convertedAnnotationCount = -1;
        result.outputFiles.insert(QStringLiteral("xanylabeling_output"), result.outputPath);
    }
    writeConversionReport(&result, report);
    return result;
}

} // namespace aitrain
