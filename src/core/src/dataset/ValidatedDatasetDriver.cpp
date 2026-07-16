#include "aitrain/dataset/ValidatedDatasetDriver.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QDirIterator>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QSet>
#include <QTemporaryDir>

#include <utility>

namespace aitrain {
namespace {

bool canceled(const DatasetOperationContext& context)
{
    return context.isCancellationRequested && context.isCancellationRequested();
}

QString canonicalRoot(const QString& path)
{
    const QFileInfo info(path);
    const QString canonical = info.canonicalFilePath();
    return canonical.isEmpty() ? QDir::cleanPath(info.absoluteFilePath()) : QDir::cleanPath(canonical);
}

bool pathInside(const QString& rootPath, const QString& candidatePath)
{
    const QDir root(rootPath);
    const QString relative = QDir::cleanPath(root.relativeFilePath(candidatePath));
    return relative == QStringLiteral(".")
        || (relative != QStringLiteral("..") && !relative.startsWith(QStringLiteral("../")));
}

QString planHash(QJsonObject manifest)
{
    manifest.remove(QStringLiteral("planHash"));
    return QString::fromLatin1(QCryptographicHash::hash(
        QJsonDocument(manifest).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
}

bool safeRelativePath(const QString& path)
{
    const QString clean = QDir::cleanPath(path);
    return !clean.isEmpty() && !QDir::isAbsolutePath(clean) && clean != QStringLiteral("..")
        && !clean.startsWith(QStringLiteral("../"));
}

bool readAndHash(const QString& path, QByteArray* content, QString* sha256, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("dataset_split_file_unreadable:%1").arg(path);
        }
        return false;
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    QByteArray collected;
    while (!file.atEnd()) {
        const QByteArray chunk = file.read(1024 * 1024);
        if (chunk.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) {
                *error = QStringLiteral("dataset_split_file_read_failed:%1").arg(path);
            }
            return false;
        }
        digest.addData(chunk);
        if (content) {
            collected.append(chunk);
        }
    }
    if (content) {
        *content = collected;
    }
    if (sha256) {
        *sha256 = QString::fromLatin1(digest.result().toHex());
    }
    return true;
}

QString splitFromTarget(const QString& target)
{
    const QStringList parts = QDir::cleanPath(target).split(QLatin1Char('/'));
    for (const QString& part : parts) {
        if (part == QStringLiteral("train") || part == QStringLiteral("val") || part == QStringLiteral("test")) {
            return part;
        }
    }
    return QStringLiteral("metadata");
}

QString normalizedTargetBaseName(const QString& target)
{
    QString name = QFileInfo(target).fileName();
    const int underscore = name.indexOf(QLatin1Char('_'));
    if (underscore > 0) {
        bool numericPrefix = false;
        name.left(underscore).toInt(&numericPrefix);
        if (numericPrefix) {
            name = name.mid(underscore + 1);
        }
    }
    return name;
}

QString sampleKeyForTarget(const QString& target)
{
    const QString split = splitFromTarget(target);
    return split + QLatin1Char('/') + QFileInfo(normalizedTargetBaseName(target)).completeBaseName();
}

bool isCopiedDatasetFile(const QString& format, const QString& target)
{
    const QString clean = QDir::cleanPath(target);
    if (clean.startsWith(QStringLiteral("images/")) || clean.startsWith(QStringLiteral("labels/"))
        || clean.startsWith(QStringLiteral("masks/"))) {
        return true;
    }
    if (format == QStringLiteral("anomaly_folder")) {
        return clean.startsWith(QStringLiteral("train/")) || clean.startsWith(QStringLiteral("val/"))
            || clean.startsWith(QStringLiteral("test/"));
    }
    return format == QStringLiteral("paddleocr_rec") && clean == QStringLiteral("dict.txt");
}

struct SourceFile final {
    QString relativePath;
    QString fileName;
    QString sha256;
    qint64 bytes = 0;
    bool used = false;
};

bool sourceRoleMatchesTarget(const QString& target, const QString& source)
{
    const QString cleanTarget = QDir::cleanPath(target);
    const QString cleanSource = QDir::cleanPath(source);
    const bool sourceIsMask = cleanSource.startsWith(QStringLiteral("masks/"))
        || cleanSource.startsWith(QStringLiteral("ground_truth/"));
    if (cleanTarget.startsWith(QStringLiteral("masks/"))) {
        return sourceIsMask;
    }
    if (cleanTarget.startsWith(QStringLiteral("images/"))) {
        return !sourceIsMask && !cleanSource.startsWith(QStringLiteral("labels/"));
    }
    if (cleanTarget.startsWith(QStringLiteral("labels/"))) {
        return cleanSource.startsWith(QStringLiteral("labels/"));
    }
    if (cleanTarget == QStringLiteral("dict.txt")) {
        return QFileInfo(cleanSource).fileName() == QStringLiteral("dict.txt");
    }
    if (cleanTarget.startsWith(QStringLiteral("train/")) || cleanTarget.startsWith(QStringLiteral("val/"))
        || cleanTarget.startsWith(QStringLiteral("test/"))) {
        return !sourceIsMask;
    }
    return true;
}

bool buildMaterializationEntries(const QString& sourceRoot,
    const QString& plannedRoot,
    const QString& format,
    QJsonArray* entries,
    QString* error)
{
    QVector<SourceFile> sources;
    QDirIterator sourceIterator(sourceRoot, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
    while (sourceIterator.hasNext()) {
        const QString path = sourceIterator.next();
        const QString relative = QDir::cleanPath(QDir(sourceRoot).relativeFilePath(path));
        QString sha256;
        if (!safeRelativePath(relative) || !readAndHash(path, nullptr, &sha256, error)) {
            return false;
        }
        const QFileInfo info(path);
        sources.append({relative, info.fileName(), sha256, info.size(), false});
    }

    QStringList targets;
    QDirIterator targetIterator(plannedRoot, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
    while (targetIterator.hasNext()) {
        const QString path = targetIterator.next();
        const QString relative = QDir::cleanPath(QDir(plannedRoot).relativeFilePath(path));
        if (relative != QStringLiteral("split_report.json")) {
            targets.append(relative);
        }
    }
    std::sort(targets.begin(), targets.end());
    QSet<QString> targetSet;
    QSet<QString> sourceSet;
    for (const QString& target : targets) {
        if (!safeRelativePath(target) || targetSet.contains(target)) {
            if (error) {
                *error = QStringLiteral("dataset_split_target_conflict:%1").arg(target);
            }
            return false;
        }
        targetSet.insert(target);
        const QString plannedPath = QDir(plannedRoot).filePath(target);
        QByteArray content;
        QString sha256;
        const bool copiedFile = isCopiedDatasetFile(format, target);
        if (!readAndHash(plannedPath, copiedFile ? nullptr : &content, &sha256, error)) {
            return false;
        }
        const qint64 byteCount = QFileInfo(plannedPath).size();
        if (!copiedFile && byteCount > 64LL * 1024LL * 1024LL) {
            if (error) {
                *error = QStringLiteral("dataset_split_inline_file_too_large:%1").arg(target);
            }
            return false;
        }
        QJsonObject entry{
            {QStringLiteral("targetRelativePath"), target},
            {QStringLiteral("split"), splitFromTarget(target)},
            {QStringLiteral("sampleKey"), sampleKeyForTarget(target)},
            {QStringLiteral("bytes"), QString::number(byteCount)},
            {QStringLiteral("sha256"), sha256}};
        if (copiedFile) {
            const QString expectedName = normalizedTargetBaseName(target);
            int matchedIndex = -1;
            for (int index = 0; index < sources.size(); ++index) {
                const SourceFile& source = sources.at(index);
                if (!source.used && sourceRoleMatchesTarget(target, source.relativePath)
                    && source.fileName == expectedName && source.sha256 == sha256
                    && source.bytes == byteCount) {
                    matchedIndex = index;
                    break;
                }
            }
            if (matchedIndex < 0) {
                for (int index = 0; index < sources.size(); ++index) {
                    const SourceFile& source = sources.at(index);
                    if (!source.used && sourceRoleMatchesTarget(target, source.relativePath)
                        && source.sha256 == sha256 && source.bytes == byteCount) {
                        if (matchedIndex >= 0) {
                            matchedIndex = -2;
                            break;
                        }
                        matchedIndex = index;
                    }
                }
            }
            if (matchedIndex < 0) {
                if (error) {
                    *error = QStringLiteral("dataset_split_source_pair_unresolved:%1").arg(target);
                }
                return false;
            }
            SourceFile& source = sources[matchedIndex];
            if (sourceSet.contains(source.relativePath)) {
                if (error) {
                    *error = QStringLiteral("dataset_split_source_leakage:%1").arg(source.relativePath);
                }
                return false;
            }
            source.used = true;
            sourceSet.insert(source.relativePath);
            entry.insert(QStringLiteral("sourceRelativePath"), source.relativePath);
        } else {
            entry.insert(QStringLiteral("inlineBase64"), QString::fromLatin1(content.toBase64()));
        }
        entries->append(entry);
    }
    return !entries->isEmpty();
}

QJsonArray issueArray(const aitrain::DatasetValidationResult& result)
{
    QJsonArray issues;
    for (const aitrain::DatasetValidationResult::Issue& issue : result.issues) {
        issues.append(QJsonObject{
            {QStringLiteral("severity"), issue.severity},
            {QStringLiteral("code"), issue.code},
            {QStringLiteral("path"), issue.filePath},
            {QStringLiteral("line"), issue.line},
            {QStringLiteral("message"), issue.message}});
    }
    return issues;
}

bool calculateRootHash(const QString& root,
    const QString& format,
    const QString& driverId,
    const QString& driverVersion,
    const DatasetOperationContext& context,
    QString* rootHash,
    QString* error)
{
    QTemporaryDir temporary;
    if (!temporary.isValid()) {
        if (error) {
            *error = QStringLiteral("dataset_snapshot_temp_unavailable");
        }
        return false;
    }
    DatasetSnapshotOptions options;
    options.isCancellationRequested = context.isCancellationRequested;
    DatasetSnapshotResult result;
    if (!createDatasetSnapshot(root, temporary.filePath(QStringLiteral("manifest.json")),
            format, driverId, driverVersion, options, &result, error)) {
        return false;
    }
    *rootHash = result.rootHash;
    return true;
}

} // namespace

ValidatedDatasetDriver::ValidatedDatasetDriver(QString driverId,
    QString format,
    Validator validator,
    Splitter splitter,
    LayoutDetector layoutDetector)
    : driverId_(std::move(driverId))
    , format_(std::move(format))
    , validator_(std::move(validator))
    , splitter_(std::move(splitter))
    , layoutDetector_(std::move(layoutDetector))
{
}

QString ValidatedDatasetDriver::id() const
{
    return driverId_;
}

QString ValidatedDatasetDriver::version() const
{
    return QStringLiteral("2.0");
}

QStringList ValidatedDatasetDriver::supportedFormats() const
{
    return {format_};
}

bool ValidatedDatasetDriver::detect(const QString& sourcePath,
    DatasetInspection* inspection,
    const DatasetOperationContext& context,
    QString* error) const
{
    if (!layoutDetector_ || !layoutDetector_(sourcePath)) {
        return false;
    }
    DatasetInspection candidate;
    if (!inspect(sourcePath, format_, &candidate, context, error)) {
        return false;
    }
    if (!inspection) {
        if (error) {
            *error = QStringLiteral("dataset_inspection_output_missing");
        }
        return false;
    }
    *inspection = candidate;
    return true;
}

bool ValidatedDatasetDriver::inspect(const QString& sourcePath,
    const QString& format,
    DatasetInspection* inspection,
    const DatasetOperationContext& context,
    QString* error) const
{
    if (!inspection || format.trimmed().toLower() != format_) {
        if (error) {
            *error = QStringLiteral("dataset_driver_format_mismatch:%1").arg(format_);
        }
        return false;
    }
    const QString root = canonicalRoot(sourcePath);
    if (!QDir(root).exists()) {
        if (error) {
            *error = QStringLiteral("dataset_root_missing:%1").arg(sourcePath);
        }
        return false;
    }
    if (canceled(context)) {
        if (error) {
            *error = QStringLiteral("dataset_inspection_canceled");
        }
        return false;
    }
    const aitrain::DatasetValidationResult result = validator_(root, QJsonObject{});
    if (canceled(context)) {
        if (error) {
            *error = QStringLiteral("dataset_inspection_canceled");
        }
        return false;
    }
    DatasetInspection value;
    value.sourcePath = root;
    value.format = format_;
    value.sampleCount = result.sampleCount;
    QJsonArray preview;
    for (const QString& sample : result.previewSamples) {
        preview.append(sample);
    }
    value.details = QJsonObject{
        {QStringLiteral("valid"), result.ok},
        {QStringLiteral("issues"), issueArray(result)},
        {QStringLiteral("previewSamples"), preview},
        {QStringLiteral("validation"), result.toJson()}};
    *inspection = value;
    return true;
}

bool ValidatedDatasetDriver::validate(const DatasetInspection& inspection,
    DatasetDriverValidationResult* validation,
    const DatasetOperationContext& context,
    QString* error) const
{
    if (!validation || inspection.format != format_ || canonicalRoot(inspection.sourcePath) != inspection.sourcePath) {
        if (error) {
            *error = QStringLiteral("dataset_validation_input_invalid:%1").arg(format_);
        }
        return false;
    }
    if (canceled(context)) {
        if (error) {
            *error = QStringLiteral("dataset_validation_canceled");
        }
        return false;
    }
    const aitrain::DatasetValidationResult current = validator_(inspection.sourcePath, QJsonObject{});
    validation->valid = current.ok;
    validation->issues = issueArray(current);
    validation->details = QJsonObject{{QStringLiteral("validation"), current.toJson()}};
    if (context.reportDiagnostic) {
        for (const QJsonValue& value : validation->issues) {
            const QJsonObject issue = value.toObject();
            context.reportDiagnostic(issue.value(QStringLiteral("code")).toString(),
                issue.value(QStringLiteral("message")).toString());
        }
    }
    return true;
}

bool ValidatedDatasetDriver::planSplit(const DatasetInspection& inspection,
    const QJsonObject& options,
    DatasetSplitPlan* plan,
    const DatasetOperationContext& context,
    QString* error) const
{
    if (!plan || inspection.format != format_) {
        if (error) {
            *error = QStringLiteral("dataset_split_plan_input_invalid:%1").arg(format_);
        }
        return false;
    }
    DatasetDriverValidationResult validation;
    if (!validate(inspection, &validation, context, error)) {
        return false;
    }
    if (!validation.valid) {
        if (error) {
            *error = QStringLiteral("dataset_split_requires_valid_dataset:%1").arg(format_);
        }
        return false;
    }
    const double trainRatio = options.value(QStringLiteral("trainRatio")).toDouble(0.8);
    const double valRatio = options.value(QStringLiteral("valRatio")).toDouble(0.2);
    const double testRatio = options.value(QStringLiteral("testRatio")).toDouble(0.0);
    if (trainRatio <= 0.0 || valRatio < 0.0 || testRatio < 0.0
        || qAbs(trainRatio + valRatio + testRatio - 1.0) > 0.000001) {
        if (error) {
            *error = QStringLiteral("dataset_split_ratio_invalid");
        }
        return false;
    }
    QString sourceRootHash;
    if (!calculateRootHash(inspection.sourcePath, format_, id(), version(), context, &sourceRootHash, error)) {
        return false;
    }
    QJsonObject normalizedOptions = options;
    //  Driver 的输入边界固定为 sourceRoot；禁止旧校验器通过可选绝对路径读取根目录外的标签或字典。
    normalizedOptions.remove(QStringLiteral("labelFile"));
    normalizedOptions.remove(QStringLiteral("dictionaryFile"));
    normalizedOptions.insert(QStringLiteral("trainRatio"), trainRatio);
    normalizedOptions.insert(QStringLiteral("valRatio"), valRatio);
    normalizedOptions.insert(QStringLiteral("testRatio"), testRatio);
    if (!normalizedOptions.contains(QStringLiteral("seed"))) {
        normalizedOptions.insert(QStringLiteral("seed"), 42);
    }
    QTemporaryDir plannedOutput;
    if (!plannedOutput.isValid()) {
        if (error) {
            *error = QStringLiteral("dataset_split_plan_temp_unavailable");
        }
        return false;
    }
    const aitrain::DatasetSplitResult split = splitter_(inspection.sourcePath, plannedOutput.path(), normalizedOptions);
    if (!split.ok) {
        if (error) {
            *error = QStringLiteral("dataset_split_plan_failed:%1").arg(split.errors.join(QStringLiteral(" | ")));
        }
        return false;
    }
    if (canceled(context)) {
        if (error) {
            *error = QStringLiteral("dataset_split_plan_canceled");
        }
        return false;
    }
    QJsonArray entries;
    if (!buildMaterializationEntries(inspection.sourcePath, plannedOutput.path(), format_, &entries, error)) {
        return false;
    }
    if (format_.startsWith(QStringLiteral("yolo_"))) {
        QHash<QString, int> pairRoles;
        for (const QJsonValue& value : entries) {
            const QJsonObject entry = value.toObject();
            const QString target = entry.value(QStringLiteral("targetRelativePath")).toString();
            if (!entry.contains(QStringLiteral("sourceRelativePath"))) {
                continue;
            }
            const QString key = entry.value(QStringLiteral("sampleKey")).toString();
            if (target.startsWith(QStringLiteral("images/"))) {
                pairRoles[key] |= 1;
            } else if (target.startsWith(QStringLiteral("labels/"))) {
                pairRoles[key] |= 2;
            }
        }
        for (auto iterator = pairRoles.cbegin(); iterator != pairRoles.cend(); ++iterator) {
            if (iterator.value() != 3) {
                if (error) {
                    *error = QStringLiteral("dataset_split_pair_incomplete:%1").arg(iterator.key());
                }
                return false;
            }
        }
    }
    QJsonObject manifest{
        {QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("driverId"), id()},
        {QStringLiteral("driverVersion"), version()},
        {QStringLiteral("format"), format_},
        {QStringLiteral("sourceRootHash"), sourceRootHash},
        {QStringLiteral("sampleCount"), QString::number(inspection.sampleCount)},
        {QStringLiteral("options"), normalizedOptions},
        {QStringLiteral("entries"), entries}};
    manifest.insert(QStringLiteral("planHash"), planHash(manifest));
    plan->format = format_;
    plan->sourceRoot = inspection.sourcePath;
    plan->planHash = manifest.value(QStringLiteral("planHash")).toString();
    plan->manifest = manifest;
    return true;
}

bool ValidatedDatasetDriver::materializeSplit(const DatasetSplitPlan& plan,
    const QString& stagingPath,
    const DatasetOperationContext& context,
    QString* error) const
{
    if (plan.format != format_ || plan.sourceRoot.trimmed().isEmpty()
        || plan.manifest.contains(QStringLiteral("sourceRoot"))
        || plan.planHash.isEmpty() || plan.planHash != planHash(plan.manifest)) {
        if (error) {
            *error = QStringLiteral("dataset_split_plan_tampered:%1").arg(format_);
        }
        return false;
    }
    const QString sourceRoot = canonicalRoot(plan.sourceRoot);
    const QString stagingRoot = QDir::cleanPath(QFileInfo(stagingPath).absoluteFilePath());
    const QDir staging(stagingRoot);
    if (stagingPath.trimmed().isEmpty() || pathInside(sourceRoot, stagingRoot) || pathInside(stagingRoot, sourceRoot)
        || (staging.exists() && !staging.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty())) {
        if (error) {
            *error = QStringLiteral("dataset_split_staging_unsafe_or_not_empty");
        }
        return false;
    }
    QString currentRootHash;
    if (!calculateRootHash(sourceRoot, format_, id(), version(), context, &currentRootHash, error)) {
        return false;
    }
    if (currentRootHash != plan.manifest.value(QStringLiteral("sourceRootHash")).toString()) {
        if (error) {
            *error = QStringLiteral("dataset_source_changed_after_plan");
        }
        return false;
    }
    if (canceled(context)) {
        if (error) {
            *error = QStringLiteral("dataset_split_materialize_canceled");
        }
        return false;
    }
    if (!QDir().mkpath(stagingRoot)) {
        if (error) {
            *error = QStringLiteral("dataset_split_staging_create_failed:%1").arg(stagingRoot);
        }
        return false;
    }
    const QJsonArray entries = plan.manifest.value(QStringLiteral("entries")).toArray();
    if (entries.isEmpty()) {
        if (error) {
            *error = QStringLiteral("dataset_split_entries_missing");
        }
        return false;
    }
    QSet<QString> targets;
    QSet<QString> sources;
    for (int index = 0; index < entries.size(); ++index) {
        if (canceled(context)) {
            if (error) {
                *error = QStringLiteral("dataset_split_materialize_canceled");
            }
            return false;
        }
        const QJsonObject entry = entries.at(index).toObject();
        const QString target = entry.value(QStringLiteral("targetRelativePath")).toString();
        const QString source = entry.value(QStringLiteral("sourceRelativePath")).toString();
        const bool hasInline = entry.contains(QStringLiteral("inlineBase64"));
        if (!safeRelativePath(target) || targets.contains(target)
            || (source.isEmpty() == !hasInline)
            || entry.value(QStringLiteral("sha256")).toString().size() != 64
            || (!source.isEmpty() && (!safeRelativePath(source) || sources.contains(source)))) {
            if (error) {
                *error = QStringLiteral("dataset_split_entry_path_or_leakage_invalid:%1").arg(target);
            }
            return false;
        }
        targets.insert(target);
        if (!source.isEmpty()) {
            sources.insert(source);
        }
        const QString destination = QDir(stagingRoot).filePath(target);
        if (!QDir().mkpath(QFileInfo(destination).absolutePath()) || QFileInfo::exists(destination)) {
            if (error) {
                *error = QStringLiteral("dataset_split_target_prepare_failed:%1").arg(target);
            }
            return false;
        }
        if (!source.isEmpty()) {
            const QString sourcePath = QDir(sourceRoot).filePath(source);
            if (!pathInside(sourceRoot, canonicalRoot(sourcePath)) || !QFile::copy(sourcePath, destination)) {
                if (error) {
                    *error = QStringLiteral("dataset_split_copy_failed:%1").arg(source);
                }
                return false;
            }
        } else {
            const QByteArray bytes = QByteArray::fromBase64(entry.value(QStringLiteral("inlineBase64")).toString().toLatin1());
            QSaveFile output(destination);
            if (!output.open(QIODevice::WriteOnly) || output.write(bytes) != bytes.size() || !output.commit()) {
                if (error) {
                    *error = QStringLiteral("dataset_split_generated_write_failed:%1").arg(target);
                }
                return false;
            }
        }
        QString actualHash;
        if (!readAndHash(destination, nullptr, &actualHash, error)
            || actualHash != entry.value(QStringLiteral("sha256")).toString()
            || QFileInfo(destination).size() != entry.value(QStringLiteral("bytes")).toString().toLongLong()) {
            if (error) {
                *error = QStringLiteral("dataset_split_entry_integrity_failed:%1").arg(target);
            }
            return false;
        }
        if (context.reportProgress) {
            context.reportProgress((index + 1) * 100 / qMax(1, entries.size()), QStringLiteral("正在按不可变清单物化数据集拆分。"));
        }
    }
    QSaveFile manifestFile(QDir(stagingRoot).filePath(QStringLiteral("split_plan.json")));
    const QByteArray bytes = QJsonDocument(plan.manifest).toJson(QJsonDocument::Indented);
    if (!manifestFile.open(QIODevice::WriteOnly) || manifestFile.write(bytes) != bytes.size() || !manifestFile.commit()) {
        if (error) {
            *error = QStringLiteral("dataset_split_manifest_commit_failed:%1").arg(manifestFile.errorString());
        }
        return false;
    }
    if (context.reportProgress) {
        context.reportProgress(100, QStringLiteral("数据集拆分已物化到 staging。"));
    }
    return true;
}

bool ValidatedDatasetDriver::snapshot(const DatasetInspection& inspection,
    const QString& manifestPath,
    const DatasetSnapshotOptions& options,
    DatasetSnapshotResult* result,
    QString* error) const
{
    if (inspection.format != format_ || inspection.sourcePath.isEmpty()) {
        if (error) {
            *error = QStringLiteral("dataset_snapshot_input_invalid:%1").arg(format_);
        }
        return false;
    }
    return createDatasetSnapshot(inspection.sourcePath, manifestPath, format_, id(), version(), options, result, error);
}

} // namespace aitrain
