#include "aitrain/dataset/DatasetConversionService.h"

#include "aitrain/core/DatasetConversion.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QDomDocument>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSet>

#include <algorithm>
#include <utility>

namespace aitrain {
namespace {

constexpr auto kArtifactKind = "dataset_conversion";
constexpr int kMaxPortableRelativePathBytes = 768;
constexpr int kMaxPortablePathComponentBytes = 160;

struct SourceEntry final {
    QString relativePath;
    QString absolutePath;
    QString sha256;
    qint64 bytes = 0;
};

bool canceled(const aitrain::CancellationCallback& cancellation)
{
    return aitrain::isCancellationRequested(cancellation);
}

bool safeRelativePath(const QString& path)
{
    const QString clean = QDir::cleanPath(path);
    return !clean.isEmpty() && !QDir::isAbsolutePath(clean)
        && clean != QStringLiteral("..") && !clean.startsWith(QStringLiteral("../"));
}

bool portableRelativePath(const QString& path)
{
    const QString clean = QDir::fromNativeSeparators(QDir::cleanPath(path));
    if (!safeRelativePath(clean) || clean.toUtf8().size() > kMaxPortableRelativePathBytes) return false;
    const QStringList components = clean.split(QLatin1Char('/'),
#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
        QString::SkipEmptyParts
#else
        Qt::SkipEmptyParts
#endif
    );
    for (const QString& component : components) {
        if (component.toUtf8().size() > kMaxPortablePathComponentBytes) return false;
    }
    return true;
}

QString normalizedAbsolutePath(const QString& path)
{
    const QFileInfo info(path);
    const QString canonical = info.canonicalFilePath();
    return QDir::cleanPath(canonical.isEmpty() ? info.absoluteFilePath() : canonical);
}

bool pathContains(const QString& root, const QString& candidate)
{
    const QString relative = QDir::cleanPath(QDir(root).relativeFilePath(candidate));
    return relative == QStringLiteral(".")
        || (relative != QStringLiteral("..") && !relative.startsWith(QStringLiteral("../")));
}

bool pathsOverlap(const QString& left, const QString& right)
{
    const QString normalizedLeft = normalizedAbsolutePath(left);
    const QString normalizedRight = normalizedAbsolutePath(right);
    return pathContains(normalizedLeft, normalizedRight) || pathContains(normalizedRight, normalizedLeft);
}

QString frozenSourcePath(const DatasetArtifactConversionRequest& request)
{
    const QFileInfo source(request.sourcePath);
    if (request.sourceFormat.trimmed().compare(QStringLiteral("voc_xml"), Qt::CaseInsensitive) == 0) {
        const QFileInfo directory = source.isDir() ? source : QFileInfo(source.absolutePath());
        if (directory.fileName().compare(QStringLiteral("Annotations"), Qt::CaseInsensitive) == 0) {
            return directory.absoluteDir().absolutePath();
        }
    }
    return request.sourcePath;
}

bool hashFile(const QString& path,
    QString* sha256,
    const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("dataset_conversion_source_unreadable:%1").arg(path);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("dataset_conversion_canceled");
            return false;
        }
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("dataset_conversion_source_read_failed:%1").arg(path);
            return false;
        }
        hash.addData(block);
    }
    *sha256 = QString::fromLatin1(hash.result().toHex());
    return true;
}

bool collectSourceEntries(const QString& sourcePath,
    QVector<SourceEntry>* entries,
    QString* rootPath,
    const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    const QFileInfo sourceInfo(sourcePath);
    if (!sourceInfo.exists()) {
        if (error) *error = QStringLiteral("dataset_conversion_source_missing:%1").arg(sourcePath);
        return false;
    }
    const QString root = sourceInfo.isDir() ? sourceInfo.absoluteFilePath() : sourceInfo.absolutePath();
    QStringList paths;
    if (sourceInfo.isDir()) {
        QDirIterator iterator(root, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
        while (iterator.hasNext()) paths.append(iterator.next());
    } else {
        paths.append(sourceInfo.absoluteFilePath());
        // COCO/VOC 的图像通常与标注同目录或其子目录。转换计划冻结整个输入根，
        // 防止转换期间只替换图像而标注文件哈希未变化。
        QDirIterator iterator(root, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
        while (iterator.hasNext()) {
            const QString path = iterator.next();
            if (QDir::cleanPath(path) != QDir::cleanPath(sourceInfo.absoluteFilePath())) paths.append(path);
        }
    }
    std::sort(paths.begin(), paths.end());
    for (const QString& path : paths) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("dataset_conversion_canceled");
            return false;
        }
        const QString relative = QDir::cleanPath(QDir(root).relativeFilePath(path));
        QString sha256;
        if (!safeRelativePath(relative)) {
            if (error) *error = QStringLiteral("dataset_conversion_source_path_unsafe:%1").arg(relative);
            return false;
        }
        if (!portableRelativePath(relative)) {
            if (error) *error = QStringLiteral("dataset_conversion_relative_path_too_long:%1").arg(relative);
            return false;
        }
        if (!hashFile(path, &sha256, cancellation, error)) return false;
        entries->append({relative, path, sha256, QFileInfo(path).size()});
    }
    if (entries->isEmpty()) {
        if (error) *error = QStringLiteral("dataset_conversion_source_empty");
        return false;
    }
    *rootPath = QDir::cleanPath(root);
    return true;
}

QString sourceRootHash(const QVector<SourceEntry>& entries)
{
    QCryptographicHash hash(QCryptographicHash::Sha256);
    for (const SourceEntry& entry : entries) {
        hash.addData(entry.relativePath.toUtf8());
        hash.addData("\0", 1);
        hash.addData(QByteArray::number(entry.bytes));
        hash.addData("\0", 1);
        hash.addData(entry.sha256.toLatin1());
        hash.addData("\n", 1);
    }
    return QString::fromLatin1(hash.result().toHex());
}

bool writeJson(const QString& path, const QJsonObject& object, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) *error = QStringLiteral("dataset_conversion_staging_unwritable:%1").arg(path);
        return false;
    }
    QSaveFile file(path);
    const QByteArray bytes = QJsonDocument(object).toJson(QJsonDocument::Indented);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("dataset_conversion_plan_write_failed:%1").arg(file.errorString());
        return false;
    }
    return true;
}

bool addTarget(QJsonArray* outputs, QSet<QString>* normalizedTargets, const QString& path, QString* error)
{
    const QString clean = QDir::cleanPath(path);
    const QString key = clean.toCaseFolded();
    if (!safeRelativePath(clean)) {
        if (error) *error = QStringLiteral("dataset_conversion_target_conflict:%1").arg(clean);
        return false;
    }
    if (!portableRelativePath(clean)) {
        if (error) *error = QStringLiteral("dataset_conversion_relative_path_too_long:%1").arg(clean);
        return false;
    }
    if (normalizedTargets->contains(key)) {
        if (error) *error = QStringLiteral("dataset_conversion_target_conflict:%1").arg(clean);
        return false;
    }
    normalizedTargets->insert(key);
    outputs->append(clean);
    return true;
}

bool addCommonOutputs(QJsonArray* outputs, QSet<QString>* targets, QString* error)
{
    return addTarget(outputs, targets, QStringLiteral("data.yaml"), error)
        && addTarget(outputs, targets, QStringLiteral("conversion_plan.json"), error)
        && addTarget(outputs, targets, QStringLiteral("dataset_conversion_report.json"), error)
        && addTarget(outputs, targets, QStringLiteral("conversion_commit_report.json"), error);
}

bool addYoloSampleOutputs(const QString& imagePath,
    QJsonArray* outputs,
    QSet<QString>* targets,
    QString* error)
{
    const QFileInfo image(imagePath);
    if (image.fileName().isEmpty() || image.completeBaseName().isEmpty()) {
        if (error) *error = QStringLiteral("dataset_conversion_source_image_path_invalid:%1").arg(imagePath);
        return false;
    }
    return addTarget(outputs, targets, QStringLiteral("images/train/%1").arg(image.fileName()), error)
        && addTarget(outputs, targets, QStringLiteral("images/val/%1").arg(image.fileName()), error)
        && addTarget(outputs, targets, QStringLiteral("labels/train/%1.txt").arg(image.completeBaseName()), error)
        && addTarget(outputs, targets, QStringLiteral("labels/val/%1.txt").arg(image.completeBaseName()), error);
}

bool validCocoPolygon(const QJsonValue& value)
{
    if (value.isObject()) return false; // RLE 不能无损转换为当前 YOLO polygon 合同。
    const QJsonArray segments = value.toArray();
    if (segments.isEmpty()) return false;
    if (!segments.first().isArray()) return segments.size() >= 6 && segments.size() % 2 == 0;
    for (const QJsonValue& segment : segments) {
        const QJsonArray polygon = segment.toArray();
        if (polygon.size() >= 6 && polygon.size() % 2 == 0) return true;
    }
    return false;
}

bool planCocoToYolo(const DatasetArtifactConversionRequest& request,
    const QString& targetFormat,
    const QSet<QString>& frozenSourcePaths,
    QJsonArray* outputs,
    QString* error)
{
    QFile file(request.sourcePath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("dataset_conversion_source_unreadable:%1").arg(request.sourcePath);
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) *error = QStringLiteral("dataset_conversion_coco_json_invalid:%1").arg(parseError.errorString());
        return false;
    }
    const QJsonObject root = document.object();
    QHash<int, QJsonObject> images;
    for (const QJsonValue& value : root.value(QStringLiteral("images")).toArray()) {
        const QJsonObject image = value.toObject();
        images.insert(image.value(QStringLiteral("id")).toInt(), image);
    }
    QSet<int> namedCategories;
    for (const QJsonValue& value : root.value(QStringLiteral("categories")).toArray()) {
        const QJsonObject category = value.toObject();
        if (!category.value(QStringLiteral("name")).toString().trimmed().isEmpty()) {
            namedCategories.insert(category.value(QStringLiteral("id")).toInt());
        }
    }
    QSet<int> convertibleImages;
    for (const QJsonValue& value : root.value(QStringLiteral("annotations")).toArray()) {
        const QJsonObject annotation = value.toObject();
        const int imageId = annotation.value(QStringLiteral("image_id")).toInt();
        const int categoryId = annotation.value(QStringLiteral("category_id")).toInt();
        const QJsonObject image = images.value(imageId);
        if (image.isEmpty() || !namedCategories.contains(categoryId)
            || image.value(QStringLiteral("width")).toDouble() <= 0.0
            || image.value(QStringLiteral("height")).toDouble() <= 0.0) {
            continue;
        }
        if (targetFormat == QStringLiteral("yolo_detection")) {
            const QJsonArray bbox = annotation.value(QStringLiteral("bbox")).toArray();
            if (bbox.size() == 4 && bbox.at(2).toDouble() > 0.0 && bbox.at(3).toDouble() > 0.0) {
                convertibleImages.insert(imageId);
            }
        } else if (validCocoPolygon(annotation.value(QStringLiteral("segmentation")))) {
            convertibleImages.insert(imageId);
        }
    }
    if (convertibleImages.isEmpty()) {
        if (error) *error = QStringLiteral("dataset_conversion_no_convertible_samples");
        return false;
    }
    QSet<QString> targets;
    if (!addCommonOutputs(outputs, &targets, error)) return false;
    QList<int> imageIds = convertibleImages.values();
    std::sort(imageIds.begin(), imageIds.end());
    for (int imageId : imageIds) {
        const QString source = images.value(imageId).value(QStringLiteral("file_name")).toString();
        if (!QFileInfo(source).isAbsolute() && !portableRelativePath(source)) {
            if (error) *error = QStringLiteral("dataset_conversion_relative_path_too_long:%1").arg(source);
            return false;
        }
        const QString absoluteSource = QFileInfo(source).isAbsolute()
            ? source : QFileInfo(request.sourcePath).absoluteDir().filePath(source);
        const QString sourceKey = normalizedAbsolutePath(absoluteSource).toCaseFolded();
        if (!QFileInfo(absoluteSource).isFile() || QFileInfo(absoluteSource).isSymLink()
            || !frozenSourcePaths.contains(sourceKey)) {
            if (error) *error = QStringLiteral("dataset_conversion_source_image_not_frozen:%1").arg(source);
            return false;
        }
        if (!addYoloSampleOutputs(absoluteSource, outputs, &targets, error)) return false;
    }
    return true;
}

QString resolveVocImagePath(const QString& sourceRoot, const QString& declaredPath, const QString& fileName)
{
    const QStringList candidates = {declaredPath, fileName,
        declaredPath.isEmpty() ? QString() : QDir(sourceRoot).filePath(declaredPath),
        QDir(sourceRoot).filePath(fileName),
        QDir(sourceRoot).filePath(QStringLiteral("../JPEGImages/%1").arg(fileName)),
        QDir(sourceRoot).filePath(QStringLiteral("../images/%1").arg(fileName))};
    for (const QString& candidate : candidates) {
        const QFileInfo info(QDir::cleanPath(candidate));
        if (!candidate.isEmpty() && info.exists() && info.isFile()) return info.absoluteFilePath();
    }
    return {};
}

bool planVocToYolo(const DatasetArtifactConversionRequest& request,
    const QSet<QString>& frozenSourcePaths,
    QJsonArray* outputs,
    QString* error)
{
    const QFileInfo sourceInfo(request.sourcePath);
    QString sourceRoot = sourceInfo.isDir() ? sourceInfo.absoluteFilePath() : sourceInfo.absolutePath();
    QStringList xmlPaths;
    if (sourceInfo.isDir()) {
        QDirIterator iterator(sourceRoot, QStringList{QStringLiteral("*.xml")}, QDir::Files);
        while (iterator.hasNext()) xmlPaths.append(iterator.next());
    } else if (sourceInfo.isFile() && sourceInfo.suffix().compare(QStringLiteral("xml"), Qt::CaseInsensitive) == 0) {
        xmlPaths.append(sourceInfo.absoluteFilePath());
    } else {
        if (error) *error = QStringLiteral("dataset_conversion_voc_source_invalid:%1").arg(request.sourcePath);
        return false;
    }
    std::sort(xmlPaths.begin(), xmlPaths.end());
    QSet<QString> targets;
    if (!addCommonOutputs(outputs, &targets, error)) return false;
    int sampleCount = 0;
    for (const QString& xmlPath : xmlPaths) {
        QFile file(xmlPath);
        QDomDocument document;
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text) || !document.setContent(&file)) continue;
        const QDomElement annotation = document.documentElement();
        if (annotation.tagName() != QLatin1String("annotation")) continue;
        const QString fileName = annotation.firstChildElement(QStringLiteral("filename")).text().trimmed();
        const QString declaredPath = annotation.firstChildElement(QStringLiteral("path")).text().trimmed();
        const QDomElement size = annotation.firstChildElement(QStringLiteral("size"));
        if (fileName.isEmpty() || size.firstChildElement(QStringLiteral("width")).text().toInt() <= 0
            || size.firstChildElement(QStringLiteral("height")).text().toInt() <= 0) continue;
        bool hasValidObject = false;
        const QDomNodeList objects = annotation.elementsByTagName(QStringLiteral("object"));
        for (int index = 0; index < objects.size(); ++index) {
            const QDomElement object = objects.at(index).toElement();
            const QDomElement bbox = object.firstChildElement(QStringLiteral("bndbox"));
            if (!object.firstChildElement(QStringLiteral("name")).text().trimmed().isEmpty()
                && bbox.firstChildElement(QStringLiteral("xmax")).text().toDouble() > bbox.firstChildElement(QStringLiteral("xmin")).text().toDouble()
                && bbox.firstChildElement(QStringLiteral("ymax")).text().toDouble() > bbox.firstChildElement(QStringLiteral("ymin")).text().toDouble()) {
                hasValidObject = true;
                break;
            }
        }
        const QString imagePath = resolveVocImagePath(sourceRoot, declaredPath, fileName);
        if (!hasValidObject || imagePath.isEmpty()) continue;
        if (QFileInfo(imagePath).isSymLink()
            || !frozenSourcePaths.contains(normalizedAbsolutePath(imagePath).toCaseFolded())) {
            if (error) *error = QStringLiteral("dataset_conversion_source_image_not_frozen:%1").arg(imagePath);
            return false;
        }
        if (!addYoloSampleOutputs(imagePath, outputs, &targets, error)) return false;
        ++sampleCount;
    }
    if (sampleCount == 0) {
        if (error) *error = QStringLiteral("dataset_conversion_no_convertible_samples");
        return false;
    }
    return true;
}

bool validateConversionRoute(const DatasetArtifactConversionRequest& request, QString* error)
{
    const QString source = request.sourceFormat.trimmed().toLower();
    const QString target = request.targetFormat.trimmed().toLower();
    if (!request.options.value(QStringLiteral("copyImages")).toBool(true)) {
        if (error) *error = QStringLiteral("dataset_conversion_backend_unsupported:%1->%2:non_self_contained_reference_output")
            .arg(source, target);
        return false;
    }
    if ((source == QStringLiteral("coco_json")
            && (target == QStringLiteral("yolo_detection") || target == QStringLiteral("yolo_segmentation")))
        || (source == QStringLiteral("voc_xml") && target == QStringLiteral("yolo_detection"))) {
        return true;
    }
    QString reason = QStringLiteral("no_target_driver");
    if (source == QStringLiteral("xanylabeling_xlabel") || target == QStringLiteral("xanylabeling_xlabel")) {
        reason = QStringLiteral("external_cli_output_cannot_be_frozen_before_materialize");
    } else if (source == QStringLiteral("voc_xml") && target == QStringLiteral("yolo_segmentation")) {
        reason = QStringLiteral("voc_has_no_polygon_semantics");
    } else if (source.startsWith(QStringLiteral("yolo_"))) {
        reason = QStringLiteral("coco_or_voc_target_has_no_driver");
    }
    if (error) *error = QStringLiteral("dataset_conversion_backend_unsupported:%1->%2:%3")
        .arg(source, target, reason);
    return false;
}

bool buildPlan(const DatasetArtifactConversionRequest& request,
    const QVector<SourceEntry>& entries,
    const QString& sourceRoot,
    QJsonObject* plan,
    QString* error)
{
    QJsonArray sourceFiles;
    for (const SourceEntry& entry : entries) {
        sourceFiles.append(QJsonObject{{QStringLiteral("relativePath"), entry.relativePath},
            {QStringLiteral("sha256"), entry.sha256},
            {QStringLiteral("bytes"), QString::number(entry.bytes)}});
    }
    const QString sourceFormat = request.sourceFormat.trimmed().toLower();
    const QString targetFormat = request.targetFormat.trimmed().toLower();
    QSet<QString> frozenSourcePaths;
    for (const SourceEntry& entry : entries) {
        frozenSourcePaths.insert(normalizedAbsolutePath(entry.absolutePath).toCaseFolded());
    }
    QJsonArray outputs;
    const bool planned = sourceFormat == QStringLiteral("coco_json")
        ? planCocoToYolo(request, targetFormat, frozenSourcePaths, &outputs, error)
        : planVocToYolo(request, frozenSourcePaths, &outputs, error);
    if (!planned) {
        return false;
    }
    QJsonObject value{{QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("sourcePath"), QFileInfo(request.sourcePath).absoluteFilePath()},
        {QStringLiteral("sourceRoot"), sourceRoot},
        {QStringLiteral("sourceFormat"), sourceFormat},
        {QStringLiteral("targetFormat"), targetFormat},
        {QStringLiteral("sourceRootHash"), sourceRootHash(entries)},
        {QStringLiteral("sourceFiles"), sourceFiles},
        {QStringLiteral("plannedOutputs"), outputs},
        {QStringLiteral("deferredOutputs"), QJsonArray{QStringLiteral("conversion_commit_report.json")}},
        {QStringLiteral("conflicts"), QJsonArray()},
        {QStringLiteral("overwritePolicy"), QStringLiteral("reject")}};
    QJsonObject hashInput = value;
    value.insert(QStringLiteral("planHash"), QString::fromLatin1(QCryptographicHash::hash(
        QJsonDocument(hashInput).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex()));
    *plan = value;
    return true;
}

bool verifyStagingAgainstPlan(const QString& stagingPath,
    const QJsonObject& plan,
    bool includeDeferred,
    QString* error)
{
    QSet<QString> expected;
    for (const QJsonValue& value : plan.value(QStringLiteral("plannedOutputs")).toArray()) {
        expected.insert(QDir::cleanPath(value.toString()).toCaseFolded());
    }
    if (!includeDeferred) {
        for (const QJsonValue& value : plan.value(QStringLiteral("deferredOutputs")).toArray()) {
            expected.remove(QDir::cleanPath(value.toString()).toCaseFolded());
        }
    }
    QSet<QString> actual;
    QDirIterator iterator(stagingPath, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        const QString relative = QDir::cleanPath(QDir(stagingPath).relativeFilePath(iterator.next()));
        if (!safeRelativePath(relative)) {
            if (error) *error = QStringLiteral("dataset_conversion_unplanned_output:%1").arg(relative);
            return false;
        }
        actual.insert(relative.toCaseFolded());
    }
    if (actual != expected) {
        QStringList missing;
        QStringList unexpected;
        for (const QString& path : expected) if (!actual.contains(path)) missing.append(path);
        for (const QString& path : actual) if (!expected.contains(path)) unexpected.append(path);
        std::sort(missing.begin(), missing.end());
        std::sort(unexpected.begin(), unexpected.end());
        if (error) *error = QStringLiteral("dataset_conversion_output_plan_mismatch:missing=[%1];unexpected=[%2]")
            .arg(missing.join(QLatin1Char(',')), unexpected.join(QLatin1Char(',')));
        return false;
    }
    return true;
}

bool sourceStillMatches(const DatasetArtifactConversionRequest& request,
    const QJsonObject& plan,
    const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    QVector<SourceEntry> current;
    QString root;
    if (!collectSourceEntries(frozenSourcePath(request), &current, &root, cancellation, error)) return false;
    if (QDir::cleanPath(root) != QDir::cleanPath(plan.value(QStringLiteral("sourceRoot")).toString())
        || sourceRootHash(current) != plan.value(QStringLiteral("sourceRootHash")).toString()) {
        if (error) *error = QStringLiteral("dataset_conversion_source_changed_after_plan");
        return false;
    }
    return true;
}

QJsonArray validationIssues(const DatasetDriverValidationResult& validation)
{
    return validation.issues;
}

bool failInjected(const DatasetConversionFailureInjector& injector,
    DatasetConversionFailPoint point,
    const QString& stagingPath,
    QString* error)
{
    if (injector && injector(point, stagingPath)) {
        if (error) *error = QStringLiteral("dataset_conversion_injected_failure");
        return true;
    }
    return false;
}

bool ioFailureInjected(const DatasetConversionIoFailureInjector& injector,
    DatasetConversionIoOperation operation,
    const QString& path,
    QString* error)
{
    if (!injector) return false;
    const QString failure = injector(operation, path).trimmed();
    if (failure.isEmpty()) return false;
    if (error) *error = failure;
    return true;
}

} // namespace

DatasetConversionService::DatasetConversionService(ArtifactStore* artifactStore,
    ProjectStore* storage,
    const DatasetDriverRegistry* drivers,
    DatasetConversionFailureInjector failureInjector,
    DatasetConversionIoFailureInjector ioFailureInjector)
    : artifactStore_(artifactStore)
    , storage_(storage)
    , drivers_(drivers)
    , failureInjector_(std::move(failureInjector))
    , ioFailureInjector_(std::move(ioFailureInjector))
{
}

bool DatasetConversionService::convert(const TaskId& taskId,
    const DatasetArtifactConversionRequest& request,
    DatasetArtifactConversion* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation,
    const std::function<void(int, const QString&)>& progress) const
{
    if (!artifactStore_ || !storage_ || !storage_->isOpen() || !drivers_ || !result
        || !taskId.isValid() || request.sourcePath.trimmed().isEmpty()
        || request.sourceFormat.trimmed().isEmpty() || request.targetFormat.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("dataset_conversion_invalid_arguments");
        return false;
    }
    if (!validateConversionRoute(request, error)) return false;
    bool taskExists = false;
    if (!storage_->taskExists(taskId, &taskExists, error) || !taskExists) {
        if (error && error->isEmpty()) *error = QStringLiteral("dataset_conversion_task_missing");
        return false;
    }
    const DatasetDriver* targetDriver = drivers_->driverForFormat(request.targetFormat);
    if (!targetDriver) {
        if (error) *error = QStringLiteral("dataset_conversion_target_driver_missing:%1").arg(request.targetFormat);
        return false;
    }
    const QFileInfo sourceInfo(frozenSourcePath(request));
    const QString sourceRootForOverlap = sourceInfo.isDir() ? sourceInfo.absoluteFilePath() : sourceInfo.absolutePath();
    if (pathsOverlap(sourceRootForOverlap, artifactStore_->rootPath())) {
        if (error) *error = QStringLiteral("dataset_conversion_source_artifact_root_overlap");
        return false;
    }

    ArtifactId artifactId;
    QString stagingPath;
    if (!artifactStore_->begin(taskId, QString::fromLatin1(kArtifactKind), &artifactId, &stagingPath, error)) return false;
    const auto abort = [&]() {
        QString cleanupError;
        artifactStore_->abort(stagingPath, &cleanupError);
    };

    QVector<SourceEntry> sourceEntries;
    QString sourceRoot;
    QJsonObject plan;
    if (progress) progress(5, QStringLiteral("正在冻结转换输入并生成输出计划。"));
    if (!collectSourceEntries(frozenSourcePath(request), &sourceEntries, &sourceRoot, cancellation, error)
        || !buildPlan(request, sourceEntries, sourceRoot, &plan, error)
        || !writeJson(QDir(stagingPath).filePath(QStringLiteral("conversion_plan.json")), plan, error)
        || failInjected(failureInjector_, DatasetConversionFailPoint::AfterPlan, stagingPath, error)
        || !sourceStillMatches(request, plan, cancellation, error)) {
        abort();
        return false;
    }

    if (progress) progress(25, QStringLiteral("转换计划已确认，正在写入 Artifact staging。"));
    aitrain::DatasetConversionRequest legacyRequest;
    legacyRequest.sourcePath = request.sourcePath;
    legacyRequest.sourceFormat = request.sourceFormat.trimmed().toLower();
    legacyRequest.targetFormat = request.targetFormat.trimmed().toLower();
    legacyRequest.outputPath = stagingPath;
    legacyRequest.options = request.options;
    if (ioFailureInjected(ioFailureInjector_, DatasetConversionIoOperation::MaterializeWrite,
            stagingPath, error)) {
        abort();
        return false;
    }
    const aitrain::DatasetConversionResult converted = aitrain::convertDataset(legacyRequest, cancellation);
    if (!converted.ok || canceled(cancellation)) {
        if (error) *error = converted.errorCode.isEmpty()
            ? QStringLiteral("dataset_conversion_canceled")
            : QStringLiteral("%1:%2").arg(converted.errorCode, converted.errorMessage);
        abort();
        return false;
    }
    if (failInjected(failureInjector_, DatasetConversionFailPoint::AfterMaterialize, stagingPath, error)
        || !sourceStillMatches(request, plan, cancellation, error)
        || !verifyStagingAgainstPlan(stagingPath, plan, false, error)) {
        abort();
        return false;
    }

    if (progress) progress(75, QStringLiteral("转换输出已生成，正在执行目标 Driver 校验。"));
    DatasetOperationContext context;
    context.isCancellationRequested = cancellation;
    context.reportProgress = [progress](int percent, const QString& message) {
        if (progress) progress(qMin(89, 75 + qMax(0, percent) / 8), message);
    };
    DatasetInspection inspection;
    DatasetDriverValidationResult validation;
    if (!targetDriver->inspect(stagingPath, request.targetFormat, &inspection, context, error)
        || !targetDriver->validate(inspection, &validation, context, error)
        || !validation.valid) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("dataset_conversion_target_validation_failed:%1")
                .arg(QString::fromUtf8(QJsonDocument(validationIssues(validation)).toJson(QJsonDocument::Compact)));
        }
        abort();
        return false;
    }
    if (failInjected(failureInjector_, DatasetConversionFailPoint::AfterTargetValidation, stagingPath, error)) {
        abort();
        return false;
    }
    QJsonObject report = converted.toJson();
    report.remove(QStringLiteral("outputPath"));
    report.remove(QStringLiteral("reportPath"));
    report.remove(QStringLiteral("validationReportPath"));
    report.insert(QStringLiteral("artifactRelativeRoot"), QStringLiteral("."));
    report.insert(QStringLiteral("outputFiles"), QJsonObject{
        {QStringLiteral("dataYaml"), QStringLiteral("data.yaml")},
        {QStringLiteral("imagesRoot"), QStringLiteral("images")},
        {QStringLiteral("labelsRoot"), QStringLiteral("labels")}});
    report.insert(QStringLiteral("schemaVersion"), 2);
    report.insert(QStringLiteral("planHash"), plan.value(QStringLiteral("planHash")));
    report.insert(QStringLiteral("sourceRootHash"), plan.value(QStringLiteral("sourceRootHash")));
    report.insert(QStringLiteral("targetDriver"), QJsonObject{{QStringLiteral("id"), targetDriver->id()},
        {QStringLiteral("version"), targetDriver->version()}});
    report.insert(QStringLiteral("targetDriverValidation"), QJsonObject{{QStringLiteral("valid"), true},
        {QStringLiteral("issues"), validation.issues}, {QStringLiteral("details"), validation.details}});
    const QString reportPath = QDir(stagingPath).filePath(QStringLiteral("conversion_commit_report.json"));
    if (ioFailureInjected(ioFailureInjector_, DatasetConversionIoOperation::ReportWrite,
            reportPath, error)
        || !writeJson(QDir(stagingPath).filePath(QStringLiteral("dataset_conversion_report.json")), report, error)
        || !writeJson(reportPath, report, error)
        || !verifyStagingAgainstPlan(stagingPath, plan, true, error)) {
        abort();
        return false;
    }

    if (progress) progress(90, QStringLiteral("目标校验通过，正在原子提交转换 Artifact。"));
    QString artifactPath;
    bool commitCanceled = false;
    if (ioFailureInjected(ioFailureInjector_, DatasetConversionIoOperation::Commit,
            artifactStore_->rootPath(), error)) {
        abort();
        return false;
    }
    if (!artifactStore_->commit(artifactId, taskId, QString::fromLatin1(kArtifactKind), stagingPath,
            storage_, &artifactPath, error, cancellation, &commitCanceled)) {
        // commit 在目录 rename 前失败时 staging 仍可直接清理；rename 后的中断由
        // ArtifactStore journal 在下次 recoverStaging 时恢复，不能删除半提交事实。
        if (QFileInfo::exists(stagingPath)) abort();
        return false;
    }
    result->artifactId = artifactId;
    result->artifactPath = artifactPath;
    result->planPath = QDir(artifactPath).filePath(QStringLiteral("conversion_plan.json"));
    result->conversionReportPath = QDir(artifactPath).filePath(QStringLiteral("conversion_commit_report.json"));
    result->plan = plan;
    result->conversionReport = report;
    result->targetValidation = validation;
    if (progress) progress(100, QStringLiteral("数据集转换 Artifact 已提交。"));
    return true;
}

} // namespace aitrain
