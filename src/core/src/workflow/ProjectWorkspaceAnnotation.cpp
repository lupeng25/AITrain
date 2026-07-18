#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/dataset/BuiltinDatasetDrivers.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSet>

#include <algorithm>

namespace aitrain {
namespace {

struct BaselineSnapshot final {
    DatasetSnapshotRecord record;
    QString rootPath;
    QJsonObject manifest;
    QHash<QString, QJsonObject> filesByKey;
};

struct SessionContract final {
    ArtifactId artifactId;
    ArtifactId repairArtifactId;
    SnapshotId snapshotId;
    DatasetVersionId datasetVersionId;
    QString datasetFormat;
    QDateTime createdAt;
    QJsonObject manifest;
    QHash<QString, QString> baselineHashes;
    QSet<QString> editableKeys;
};

struct SyncInspection final {
    AnnotationSyncStatus status = AnnotationSyncStatus::InvalidSession;
    QString code;
    QString message;
    SessionContract session;
    BaselineSnapshot baseline;
    QStringList changedFiles;
};

bool canceled(const aitrain::CancellationCallback& cancellation)
{
    return aitrain::isCancellationRequested(cancellation);
}

QString statusText(AnnotationSyncStatus status)
{
    switch (status) {
    case AnnotationSyncStatus::Inspected: return QStringLiteral("Inspected");
    case AnnotationSyncStatus::ChangesDetected: return QStringLiteral("ChangesDetected");
    case AnnotationSyncStatus::NoChanges: return QStringLiteral("NoChanges");
    case AnnotationSyncStatus::InvalidSession: return QStringLiteral("InvalidSession");
    case AnnotationSyncStatus::Conflict: return QStringLiteral("Conflict");
    case AnnotationSyncStatus::Canceled: return QStringLiteral("Canceled");
    }
    return QStringLiteral("InvalidSession");
}

Failure failure(FailureCode code, const QString& message, const QString& action)
{
    return {code, message, action, QDateTime::currentDateTimeUtc()};
}

bool safeRelative(const QString& value)
{
    const QString path = QDir::cleanPath(value);
    return !path.isEmpty() && !QDir::isAbsolutePath(path) && path != QStringLiteral("..")
        && !path.startsWith(QStringLiteral("../"));
}

QString pathKey(const QString& value)
{
    return QDir::cleanPath(value).toCaseFolded();
}

bool readBytesAndHash(const QString& path, QByteArray* bytes, QString* sha256,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    const QFileInfo info(path);
    if (info.isSymLink() || !info.isFile()) {
        if (error) *error = QStringLiteral("annotation.file_not_regular:%1").arg(path);
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("annotation.file_unreadable:%1").arg(path);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    QByteArray content;
    while (!file.atEnd()) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("annotation.canceled");
            return false;
        }
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("annotation.file_read_failed:%1").arg(path);
            return false;
        }
        hash.addData(block);
        if (bytes) content.append(block);
    }
    if (bytes) *bytes = content;
    if (sha256) *sha256 = QString::fromLatin1(hash.result().toHex());
    return true;
}

bool writeBytes(const QString& path, const QByteArray& bytes, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) *error = QStringLiteral("annotation.output_directory_failed:%1").arg(path);
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("annotation.output_write_failed:%1").arg(path);
        return false;
    }
    return true;
}

QByteArray jsonBytes(const QJsonObject& object)
{
    return QJsonDocument(object).toJson(QJsonDocument::Indented);
}

bool commitFiles(ArtifactStore* store, ProjectStore* storage, const TaskId& taskId,
    const QString& kind, const QVector<QPair<QString, QByteArray>>& files, ArtifactId* result,
    QString* error, const aitrain::CancellationCallback& cancellation)
{
    ArtifactId id;
    QString staging;
    if (!store->begin(taskId, kind, &id, &staging, error)) return false;
    const auto abort = [&]() { QString ignored; store->abort(staging, &ignored); };
    for (const auto& file : files) {
        if (!safeRelative(file.first) || canceled(cancellation)
            || !writeBytes(QDir(staging).filePath(file.first), file.second, error)) {
            if (canceled(cancellation) && error) *error = QStringLiteral("annotation.canceled");
            abort();
            return false;
        }
    }
    QString committedPath;
    if (!store->commit(id, taskId, kind, staging, storage, &committedPath, error, cancellation)) {
        abort();
        return false;
    }
    *result = id;
    return true;
}

QString artifactPath(ArtifactStore* store, const ArtifactId& id)
{
    return store->artifactPath(id);
}

bool readVerifiedArtifactJson(ProjectStore* storage, ArtifactStore* store,
    const ArtifactSnapshot& artifact, const QString& relativePath, QJsonObject* result,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    const auto declared = std::find_if(artifact.files.cbegin(), artifact.files.cend(), [&](const ArtifactFileSnapshot& file) {
        return file.relativePath == relativePath;
    });
    QByteArray bytes;
    QString hash;
    const QString path = QDir(artifactPath(store, artifact.id)).filePath(relativePath);
    if (declared == artifact.files.cend()
        || !readBytesAndHash(path, &bytes, &hash, cancellation, error)
        || hash != declared->sha256 || bytes.size() != declared->byteCount) {
        if (error && error->isEmpty()) *error = QStringLiteral("annotation.artifact_file_integrity_failed:%1").arg(relativePath);
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) *error = QStringLiteral("annotation.artifact_json_invalid:%1").arg(relativePath);
        return false;
    }
    *result = document.object();
    Q_UNUSED(storage);
    return true;
}

bool loadBaseline(ProjectStore* storage, ArtifactStore* store, const SnapshotId& id,
    BaselineSnapshot* result, bool verifySource, const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    DatasetSnapshotRecord record;
    ArtifactSnapshot artifact;
    QJsonObject manifest;
    if (!storage->datasetSnapshot(id, &record, error)
        || !storage->artifact(record.artifactId, &artifact, error)
        || artifact.kind != QStringLiteral("dataset_snapshot")
        || !readVerifiedArtifactJson(storage, store, artifact, QStringLiteral("dataset_snapshot.json"),
            &manifest, cancellation, error)) return false;
    if (manifest.value(QStringLiteral("schemaVersion")).toInt() != 2
        || !manifest.value(QStringLiteral("complete")).toBool()
        || manifest.value(QStringLiteral("snapshotId")).toString() != id.toString()
        || manifest.value(QStringLiteral("datasetFormat")).toString() != record.datasetFormat
        || manifest.value(QStringLiteral("rootHash")).toString() != record.rootHash) {
        if (error) *error = QStringLiteral("annotation.snapshot_manifest_invalid");
        return false;
    }
    QHash<QString, QJsonObject> files;
    for (const QJsonValue& value : manifest.value(QStringLiteral("files")).toArray()) {
        const QJsonObject item = value.toObject();
        const QString relative = QDir::cleanPath(item.value(QStringLiteral("relativePath")).toString());
        const QString key = pathKey(relative);
        if (!safeRelative(relative) || files.contains(key)
            || !isSha256Hex(item.value(QStringLiteral("sha256")).toString())
            || item.value(QStringLiteral("bytes")).toString().toLongLong() < 0) {
            if (error) *error = QStringLiteral("annotation.snapshot_inventory_invalid:%1").arg(relative);
            return false;
        }
        files.insert(key, item);
    }
    if (files.size() != record.fileCount) {
        if (error) *error = QStringLiteral("annotation.snapshot_file_count_mismatch");
        return false;
    }
    const QString rootPath = store->artifactPath(record.artifactId);
    if (rootPath.isEmpty()) {
        if (error) *error = QStringLiteral("annotation.snapshot_artifact_path_invalid");
        return false;
    }
    if (verifySource) {
        const QDir root(rootPath);
        if (!root.exists()) {
            if (error) *error = QStringLiteral("annotation.baseline_unavailable");
            return false;
        }
        for (auto it = files.cbegin(); it != files.cend(); ++it) {
            const QJsonObject item = it.value();
            const QString relative = item.value(QStringLiteral("relativePath")).toString();
            QString actual;
            const QString absolute = root.filePath(relative);
            if (!readBytesAndHash(absolute, nullptr, &actual, cancellation, error)
                || QFileInfo(absolute).size() != item.value(QStringLiteral("bytes")).toString().toLongLong()
                || actual != item.value(QStringLiteral("sha256")).toString()) {
                if (error && error->isEmpty()) *error = QStringLiteral("annotation.baseline_changed:%1").arg(relative);
                return false;
            }
        }
    }
    result->record = record;
    result->rootPath = rootPath;
    result->manifest = manifest;
    result->filesByKey = files;
    return true;
}

QString resolvedPathThroughExistingParent(const QString& path, QString* error)
{
    const QString absolute = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    QFileInfo cursor(absolute);
    QStringList missing;
    while (!cursor.exists()) {
        const QString name = cursor.fileName();
        if (name.isEmpty()) {
            if (error) *error = QStringLiteral("annotation.path_has_no_existing_parent:%1").arg(path);
            return {};
        }
        missing.prepend(name);
        const QString parent = cursor.absolutePath();
        if (parent == cursor.absoluteFilePath()) {
            if (error) *error = QStringLiteral("annotation.path_has_no_existing_parent:%1").arg(path);
            return {};
        }
        cursor.setFile(parent);
    }
    const QString canonicalParent = cursor.canonicalFilePath();
    if (canonicalParent.isEmpty()) {
        if (error) *error = QStringLiteral("annotation.path_canonicalization_failed:%1").arg(path);
        return {};
    }
    QString resolved = canonicalParent;
    for (const QString& segment : missing) resolved = QDir(resolved).filePath(segment);
    return QDir::cleanPath(resolved);
}

bool pathsOverlap(const QString& left, const QString& right)
{
    QString ignored;
    const QString a = resolvedPathThroughExistingParent(left, &ignored).toCaseFolded();
    const QString b = resolvedPathThroughExistingParent(right, &ignored).toCaseFolded();
    if (a.isEmpty() || b.isEmpty()) return true;
    return a == b || a.startsWith(b + QLatin1Char('/')) || b.startsWith(a + QLatin1Char('/'));
}

bool prepareEmptyWorkingDirectory(const QString& path, const QString& baselineRoot,
    const QString& artifactRoot, QString* resolvedRoot, bool* rootCreated, QString* error)
{
    if (!resolvedRoot || !rootCreated || path.trimmed().isEmpty() || QFileInfo(path).isSymLink()) {
        if (error) *error = QStringLiteral("annotation.working_directory_symlink_or_invalid");
        return false;
    }
    QString resolutionError;
    const QString resolved = resolvedPathThroughExistingParent(path, &resolutionError);
    if (resolved.isEmpty() || pathsOverlap(resolved, baselineRoot) || pathsOverlap(resolved, artifactRoot)) {
        if (error) *error = QStringLiteral("annotation.working_directory_not_independent");
        return false;
    }
    QDir directory(path);
    if (directory.exists() && !directory.entryList(QDir::AllEntries | QDir::NoDotAndDotDot).isEmpty()) {
        if (error) *error = QStringLiteral("annotation.working_directory_not_empty");
        return false;
    }
    const bool created = !directory.exists();
    if (created && !QDir().mkpath(directory.absolutePath())) {
        if (error) *error = QStringLiteral("annotation.working_directory_create_failed");
        return false;
    }
    const QFileInfo createdInfo(path);
    if (createdInfo.isSymLink() || !createdInfo.isDir()
        || QDir::cleanPath(createdInfo.canonicalFilePath()).compare(resolved, Qt::CaseInsensitive) != 0) {
        if (error) *error = QStringLiteral("annotation.working_directory_identity_changed");
        return false;
    }
    *resolvedRoot = resolved;
    *rootCreated = created;
    return true;
}

void cleanupPreparedWorkingCopy(const QString& resolvedRoot, bool rootCreated,
    const BaselineSnapshot& baseline)
{
    const QFileInfo rootInfo(resolvedRoot);
    if (resolvedRoot.isEmpty() || rootInfo.isSymLink() || !rootInfo.isDir()
        || QDir::cleanPath(rootInfo.canonicalFilePath()).compare(
            QDir::cleanPath(resolvedRoot), Qt::CaseInsensitive) != 0) return;
    QSet<QString> directories;
    const QDir root(resolvedRoot);
    for (auto it = baseline.filesByKey.cbegin(); it != baseline.filesByKey.cend(); ++it) {
        const QString relative = QDir::cleanPath(it.value().value(QStringLiteral("relativePath")).toString());
        if (!safeRelative(relative)) continue;
        const QString target = QDir::cleanPath(root.filePath(relative));
        if (!target.startsWith(QDir::cleanPath(resolvedRoot) + QLatin1Char('/'), Qt::CaseInsensitive)) continue;
        const QFileInfo targetInfo(target);
        if (targetInfo.isFile() || targetInfo.isSymLink()) QFile::remove(target);
        QString parent = QFileInfo(relative).path();
        while (parent != QStringLiteral(".") && safeRelative(parent)) {
            directories.insert(QDir::cleanPath(parent));
            parent = QFileInfo(parent).path();
        }
    }
    QStringList orderedDirectories = directories.values();
    std::sort(orderedDirectories.begin(), orderedDirectories.end(), [](const QString& left, const QString& right) {
        return left.size() > right.size();
    });
    for (const QString& relative : orderedDirectories) root.rmdir(relative);
    if (rootCreated && root.entryList(QDir::AllEntries | QDir::NoDotAndDotDot).isEmpty()) {
        QDir().rmdir(resolvedRoot);
    }
}

bool resolveReadableWorkingDirectory(const QString& path, const QString& baselineRoot,
    const QString& artifactRoot, QString* resolvedRoot, QString* error)
{
    if (!resolvedRoot || path.trimmed().isEmpty() || QFileInfo(path).isSymLink()) {
        if (error) *error = QStringLiteral("annotation.working_directory_symlink_or_invalid");
        return false;
    }
    const QFileInfo info(path);
    const QString resolved = resolvedPathThroughExistingParent(path, error);
    if (!info.isDir() || resolved.isEmpty() || pathsOverlap(resolved, baselineRoot)
        || pathsOverlap(resolved, artifactRoot)
        || QDir::cleanPath(info.canonicalFilePath()).compare(resolved, Qt::CaseInsensitive) != 0) {
        if (error && error->isEmpty()) *error = QStringLiteral("annotation.working_directory_identity_changed");
        return false;
    }
    *resolvedRoot = resolved;
    return true;
}

bool copyBaseline(const BaselineSnapshot& baseline, const QString& destination,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    const QDir source(baseline.rootPath);
    const QDir target(destination);
    for (auto it = baseline.filesByKey.cbegin(); it != baseline.filesByKey.cend(); ++it) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("annotation.canceled");
            return false;
        }
        const QJsonObject item = it.value();
        const QString relative = item.value(QStringLiteral("relativePath")).toString();
        QByteArray bytes;
        QString hash;
        if (!readBytesAndHash(source.filePath(relative), &bytes, &hash, cancellation, error)
            || hash != item.value(QStringLiteral("sha256")).toString()
            || !writeBytes(target.filePath(relative), bytes, error)) return false;
    }
    return true;
}

QJsonObject toolSummary(const QJsonObject& parameters)
{
    const QByteArray compact = QJsonDocument(parameters).toJson(QJsonDocument::Compact);
    return {{QStringLiteral("toolId"), parameters.value(QStringLiteral("toolId")).toString(QStringLiteral("x-anylabeling"))},
        {QStringLiteral("toolVersion"), parameters.value(QStringLiteral("toolVersion")).toString()},
        {QStringLiteral("mode"), parameters.value(QStringLiteral("mode")).toString(QStringLiteral("review"))},
        {QStringLiteral("parametersSha256"), QString::fromLatin1(QCryptographicHash::hash(compact, QCryptographicHash::Sha256).toHex())}};
}

bool annotationFormatSupported(const QString& format)
{
    return format == QStringLiteral("yolo_detection")
        || format == QStringLiteral("yolo_segmentation")
        || format == QStringLiteral("yolo_obb")
        || format == QStringLiteral("semantic_segmentation_mask")
        || format == QStringLiteral("paddleocr_det")
        || format == QStringLiteral("paddleocr_rec");
}

bool validateRepairAction(const QString& format, const QJsonObject& action,
    const BaselineSnapshot& baseline, QString* error)
{
    const QString source = QDir::cleanPath(action.value(QStringLiteral("sourceRelativePath")).toString());
    const QString sample = QDir::cleanPath(action.value(QStringLiteral("sampleRelativePath")).toString());
    const QString repairAction = action.value(QStringLiteral("action")).toString();
    const QString fileName = QFileInfo(source).fileName();
    bool pathValid = false;
    QSet<QString> allowedActions;
    if (format == QStringLiteral("yolo_detection")) {
        pathValid = source.startsWith(QStringLiteral("labels/"))
            && source.endsWith(QStringLiteral(".txt"), Qt::CaseInsensitive);
        allowedActions = {QStringLiteral("review_bbox"), QStringLiteral("review_empty_label")};
    } else if (format == QStringLiteral("yolo_segmentation")) {
        pathValid = source.startsWith(QStringLiteral("labels/"))
            && source.endsWith(QStringLiteral(".txt"), Qt::CaseInsensitive);
        allowedActions = {QStringLiteral("review_polygon")};
    } else if (format == QStringLiteral("yolo_obb")) {
        pathValid = source.startsWith(QStringLiteral("labels/"))
            && source.endsWith(QStringLiteral(".txt"), Qt::CaseInsensitive);
        allowedActions = {QStringLiteral("review_obb_quad")};
    } else if (format == QStringLiteral("semantic_segmentation_mask")) {
        pathValid = source.startsWith(QStringLiteral("masks/"))
            && source.endsWith(QStringLiteral(".png"), Qt::CaseInsensitive);
        allowedActions = {QStringLiteral("review_background_only_mask")};
    } else if (format == QStringLiteral("paddleocr_det")) {
        pathValid = fileName.startsWith(QStringLiteral("det_gt"))
            && fileName.endsWith(QStringLiteral(".txt"), Qt::CaseInsensitive);
        allowedActions = {QStringLiteral("review_text_polygon")};
    } else if (format == QStringLiteral("paddleocr_rec")) {
        pathValid = fileName.startsWith(QStringLiteral("rec_gt"))
            && fileName.endsWith(QStringLiteral(".txt"), Qt::CaseInsensitive);
        allowedActions = {QStringLiteral("review_duplicate_reference"), QStringLiteral("review_transcription")};
    }
    if (!safeRelative(source) || !safeRelative(sample) || !pathValid
        || !allowedActions.contains(repairAction) || !baseline.filesByKey.contains(pathKey(source))) {
        if (error) *error = QStringLiteral("annotation.repair_action_out_of_scope:%1:%2")
            .arg(source, repairAction);
        return false;
    }
    return true;
}

bool loadRepairContract(ProjectStore* storage, ArtifactStore* store, const ArtifactId& artifactId,
    BaselineSnapshot* baseline, QJsonObject* repair, QJsonObject* xany,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    ArtifactSnapshot artifact;
    if (!storage->artifact(artifactId, &artifact, error)
        || artifact.kind != QStringLiteral("dataset_repair_manifest")
        || !readVerifiedArtifactJson(storage, store, artifact, QStringLiteral("repair_manifest.json"), repair, cancellation, error)
        || !readVerifiedArtifactJson(storage, store, artifact, QStringLiteral("xanylabeling_review_manifest.json"), xany, cancellation, error)) return false;
    SnapshotId snapshotId;
    if (repair->value(QStringLiteral("schemaVersion")).toInt() != 2
        || repair->value(QStringLiteral("kind")).toString() != QStringLiteral("dataset_repair_manifest")
        || repair->value(QStringLiteral("mutatesSource")).toBool(true)
        || xany->value(QStringLiteral("schemaVersion")).toInt() != 2
        || xany->value(QStringLiteral("kind")).toString() != QStringLiteral("xanylabeling_review_manifest")
        || xany->value(QStringLiteral("datasetSnapshotId")).toString()
            != repair->value(QStringLiteral("datasetSnapshotId")).toString()
        || !SnapshotId::parse(repair->value(QStringLiteral("datasetSnapshotId")).toString(), &snapshotId, error)
        || !loadBaseline(storage, store, snapshotId, baseline, true, cancellation, error)
        || repair->value(QStringLiteral("datasetFormat")).toString() != baseline->record.datasetFormat) {
        if (error && error->isEmpty()) *error = QStringLiteral("annotation.repair_manifest_contract_invalid");
        return false;
    }
    const QString format = baseline->record.datasetFormat;
    // anomaly_folder 的质量动作是“收集新的异常评估样本”，不是对现有基线文件的
    // 白名单编辑。当前会话合同禁止增删文件，因此必须精确拒绝，不能伪装为可同步格式。
    if (format == QStringLiteral("anomaly_folder")) {
        if (error) *error = QStringLiteral(
            "annotation.format_not_implemented:anomaly_folder:review_only_add_samples");
        return false;
    }
    if (!annotationFormatSupported(format)) {
        if (error) *error = QStringLiteral("annotation.format_not_implemented:%1").arg(format);
        return false;
    }
    const QJsonArray actions = repair->value(QStringLiteral("actions")).toArray();
    if (actions != xany->value(QStringLiteral("samples")).toArray()) {
        if (error) *error = QStringLiteral("annotation.review_manifest_lineage_mismatch");
        return false;
    }
    for (const QJsonValue& value : actions) {
        if (!validateRepairAction(format, value.toObject(), *baseline, error)) return false;
    }
    return true;
}

bool loadSessionContract(ProjectStore* storage, ArtifactStore* store, const ArtifactId& artifactId,
    SessionContract* result, const aitrain::CancellationCallback& cancellation, QString* error)
{
    ArtifactSnapshot artifact;
    QJsonObject manifest;
    if (!storage->artifact(artifactId, &artifact, error)
        || artifact.kind != QStringLiteral("annotation_session")
        || !readVerifiedArtifactJson(storage, store, artifact, QStringLiteral("annotation_session.json"),
            &manifest, cancellation, error)) return false;
    SessionContract parsed;
    parsed.artifactId = artifactId;
    parsed.createdAt = QDateTime::fromString(manifest.value(QStringLiteral("createdAt")).toString(), Qt::ISODateWithMs);
    if (manifest.value(QStringLiteral("schemaVersion")).toInt() != 2
        || manifest.value(QStringLiteral("kind")).toString() != QStringLiteral("annotation_session")
        || manifest.value(QStringLiteral("outputArtifactId")).toString() != artifactId.toString()
        || !parsed.createdAt.isValid()
        || !ArtifactId::parse(manifest.value(QStringLiteral("sourceRepairManifestArtifactId")).toString(), &parsed.repairArtifactId, error)
        || !SnapshotId::parse(manifest.value(QStringLiteral("sourceSnapshotId")).toString(), &parsed.snapshotId, error)
        || !DatasetVersionId::parse(manifest.value(QStringLiteral("sourceDatasetVersionId")).toString(), &parsed.datasetVersionId, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("annotation.session_manifest_contract_invalid");
        return false;
    }
    parsed.datasetFormat = manifest.value(QStringLiteral("datasetFormat")).toString();
    if (!annotationFormatSupported(parsed.datasetFormat)) {
        if (error) *error = QStringLiteral("annotation.session_format_not_implemented:%1").arg(parsed.datasetFormat);
        return false;
    }
    for (const QJsonValue& value : manifest.value(QStringLiteral("baselineFiles")).toArray()) {
        const QJsonObject item = value.toObject();
        const QString relative = QDir::cleanPath(item.value(QStringLiteral("relativePath")).toString());
        const QString hash = item.value(QStringLiteral("sha256")).toString();
        if (!safeRelative(relative) || !isSha256Hex(hash) || parsed.baselineHashes.contains(pathKey(relative))) {
            if (error) *error = QStringLiteral("annotation.session_baseline_inventory_invalid:%1").arg(relative);
            return false;
        }
        parsed.baselineHashes.insert(pathKey(relative), hash);
    }
    for (const QJsonValue& value : manifest.value(QStringLiteral("editableFiles")).toArray()) {
        const QString relative = QDir::cleanPath(value.toString());
        if (!safeRelative(relative) || !parsed.baselineHashes.contains(pathKey(relative))) {
            if (error) *error = QStringLiteral("annotation.session_editable_inventory_invalid:%1").arg(relative);
            return false;
        }
        parsed.editableKeys.insert(pathKey(relative));
    }
    if (parsed.baselineHashes.isEmpty()) {
        if (error) *error = QStringLiteral("annotation.session_baseline_inventory_empty");
        return false;
    }
    parsed.manifest = manifest;
    *result = parsed;
    return true;
}

bool validateSessionLineage(ProjectStore* storage, ArtifactStore* store,
    const SessionContract& session, const BaselineSnapshot& baseline,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    if (baseline.record.id != session.snapshotId
        || baseline.record.datasetVersionId != session.datasetVersionId
        || baseline.record.datasetFormat != session.datasetFormat
        || baseline.filesByKey.size() != session.baselineHashes.size()) {
        if (error) *error = QStringLiteral("annotation.session_baseline_lineage_mismatch");
        return false;
    }
    for (auto it = baseline.filesByKey.cbegin(); it != baseline.filesByKey.cend(); ++it) {
        if (!session.baselineHashes.contains(it.key())
            || session.baselineHashes.value(it.key()) != it.value().value(QStringLiteral("sha256")).toString()) {
            if (error) *error = QStringLiteral("annotation.session_baseline_hash_mismatch:%1")
                .arg(it.value().value(QStringLiteral("relativePath")).toString());
            return false;
        }
    }
    ArtifactSnapshot repairArtifact;
    QJsonObject repair;
    QJsonObject xany;
    if (!storage->artifact(session.repairArtifactId, &repairArtifact, error)
        || repairArtifact.kind != QStringLiteral("dataset_repair_manifest")
        || !readVerifiedArtifactJson(storage, store, repairArtifact, QStringLiteral("repair_manifest.json"),
            &repair, cancellation, error)
        || !readVerifiedArtifactJson(storage, store, repairArtifact, QStringLiteral("xanylabeling_review_manifest.json"),
            &xany, cancellation, error)
        || repair.value(QStringLiteral("schemaVersion")).toInt() != 2
        || repair.value(QStringLiteral("kind")).toString() != QStringLiteral("dataset_repair_manifest")
        || repair.value(QStringLiteral("datasetSnapshotId")).toString() != session.snapshotId.toString()
        || repair.value(QStringLiteral("datasetFormat")).toString() != session.datasetFormat
        || repair.value(QStringLiteral("mutatesSource")).toBool(true)
        || xany.value(QStringLiteral("schemaVersion")).toInt() != 2
        || xany.value(QStringLiteral("kind")).toString() != QStringLiteral("xanylabeling_review_manifest")
        || xany.value(QStringLiteral("datasetSnapshotId")).toString() != session.snapshotId.toString()
        || repair.value(QStringLiteral("actions")).toArray() != xany.value(QStringLiteral("samples")).toArray()
        || repair.value(QStringLiteral("actions")) != session.manifest.value(QStringLiteral("problemLineage"))) {
        if (error && error->isEmpty()) *error = QStringLiteral("annotation.session_repair_lineage_invalid");
        return false;
    }
    QSet<QString> expectedEditable;
    for (const QJsonValue& value : repair.value(QStringLiteral("actions")).toArray()) {
        const QJsonObject action = value.toObject();
        if (!validateRepairAction(session.datasetFormat, action, baseline, error)) return false;
        expectedEditable.insert(pathKey(action.value(QStringLiteral("sourceRelativePath")).toString()));
    }
    if (expectedEditable != session.editableKeys) {
        if (error) *error = QStringLiteral("annotation.session_editable_lineage_mismatch");
        return false;
    }
    return true;
}

bool collectWorkingFiles(const QString& root, QHash<QString, QString>* relativeByKey, QString* error)
{
    QDir directory(root);
    if (!directory.exists()) {
        if (error) *error = QStringLiteral("annotation.working_directory_missing");
        return false;
    }
    QDirIterator iterator(directory.absolutePath(), QDir::Files | QDir::System, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        const QString absolute = iterator.next();
        const QFileInfo info(absolute);
        const QString relative = QDir::cleanPath(directory.relativeFilePath(absolute));
        const QString key = pathKey(relative);
        if (info.isSymLink() || !safeRelative(relative) || relativeByKey->contains(key)) {
            if (error) *error = QStringLiteral("annotation.working_inventory_invalid:%1").arg(relative);
            return false;
        }
        relativeByKey->insert(key, relative);
    }
    return true;
}

bool validateWithDriver(const QString& root, const QString& format,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    DatasetDriverRegistry registry;
    if (!registerBuiltinDatasetDrivers(&registry, error)) return false;
    const DatasetDriver* driver = registry.driverForFormat(format);
    DatasetInspection inspection;
    DatasetDriverValidationResult validation;
    DatasetOperationContext context;
    context.isCancellationRequested = cancellation;
    if (!driver || !driver->inspect(root, format, &inspection, context, error)
        || !driver->validate(inspection, &validation, context, error) || !validation.valid) {
        if (error && error->isEmpty()) *error = QStringLiteral("annotation.driver_validation_failed");
        return false;
    }
    return true;
}

TaskState terminalTaskState(const WorkflowRunExecutionResult& run)
{
    if (run.state == WorkflowStepState::Succeeded) return TaskState::Succeeded;
    if (run.state == WorkflowStepState::Canceled) return TaskState::Canceled;
    return TaskState::Failed;
}

Failure terminalFailure(const WorkflowRunExecutionResult& run, const QString& retryAction)
{
    if (run.state == WorkflowStepState::Succeeded) return {};
    if (run.failure.isFailure()) {
        Failure value = run.failure;
        if (value.suggestedAction.trimmed().isEmpty()) value.suggestedAction = retryAction;
        if (!value.occurredAt.isValid()) value.occurredAt = QDateTime::currentDateTimeUtc();
        return value;
    }
    return failure(run.state == WorkflowStepState::Canceled ? FailureCode::Canceled : FailureCode::InternalError,
        QStringLiteral("标注会话 Workflow 未返回完整终态。"), retryAction);
}

bool finishWorkflow(ProjectWorkspace* workspace, ProjectStore* storage, const TaskId& taskId,
    const WorkflowRunId& workflowId, const WorkflowRunExecutionResult& run,
    const QString& retryAction, ArtifactId* evidenceArtifactId, QString* error)
{
    const TaskState state = terminalTaskState(run);
    if (state == TaskState::Canceled) {
        TaskSnapshot task;
        if (!storage->task(taskId, &task, error)) return false;
        if (task.state == TaskState::Running && !workspace->requestTaskCancellation(taskId, error)) return false;
    }
    if (!storage->sealWorkflowTerminalization(workflowId, state, terminalFailure(run, retryAction),
            QDateTime::currentDateTimeUtc(), error)) return false;
    EvidenceBundle evidence;
    EvidenceArtifactBundle artifact;
    if (!workspace->buildWorkflowEvidenceBundle(workflowId, &evidence, error)
        || !workspace->commitEvidenceBundle(evidence, &artifact, error)
        || !workspace->closeWorkflowTerminalization(workflowId, error)) return false;
    *evidenceArtifactId = artifact.artifactId;
    return true;
}

} // namespace

bool ProjectWorkspace::createAnnotationSession(const TaskId& taskId,
    const AnnotationSessionCreateRequest& request, AnnotationSessionCreateResult* result,
    QString* error, const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !request.repairManifestArtifactId.isValid()
        || request.workingDirectory.trimmed().isEmpty() || !result) {
        if (error) *error = QStringLiteral("创建标注会话需要运行中任务、修复清单 Artifact、独立工作目录和输出对象。");
        return false;
    }
    TaskSnapshot task;
    ArtifactSnapshot repairArtifact;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running
        || !storage_.artifact(request.repairManifestArtifactId, &repairArtifact, error)) return false;

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("annotation_session_create");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    const QStringList kinds{QStringLiteral("ValidateRepairManifest"), QStringLiteral("PrepareWorkingCopy"),
        QStringLiteral("CommitAnnotationSession")};
    QVector<WorkflowStepSnapshot> steps;
    const QJsonObject parameters{{QStringLiteral("repairManifestArtifactId"), request.repairManifestArtifactId.toString()},
        {QStringLiteral("toolParameters"), toolSummary(request.toolParameters)}};
    for (int index = 0; index < kinds.size(); ++index) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = index;
        step.kind = kinds.at(index);
        step.backend = QStringLiteral("builtin_annotation_session");
        step.parameterSummary = parameters;
        if (index == 0) step.inputArtifactId = request.repairManifestArtifactId;
        steps.append(step);
    }
    WorkflowInputBinding input;
    input.workflowRunId = workflow.id;
    input.role = QStringLiteral("dataset_repair_manifest");
    input.sourceArtifactId = repairArtifact.id;
    input.sourceTaskId = repairArtifact.taskId;
    input.sourceArtifactKind = repairArtifact.kind;
    if (!storage_.createWorkflowRunWithInput(workflow, steps, input, error)) return false;

    BaselineSnapshot baseline;
    QJsonObject repair;
    QJsonObject xany;
    QSet<QString> editable;
    ArtifactId sessionArtifactId;
    QString resolvedWorkingRoot;
    bool workingRootCreated = false;
    bool workingCopyPrepared = false;
    QString executionError;
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult run;
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) -> WorkflowStepExecutionResult {
        if (canceled(stepCancellation)) return {WorkflowStepState::Canceled, {},
            failure(FailureCode::Canceled, QStringLiteral("标注会话创建已取消。"),
                QStringLiteral("可从同一修复清单 Artifact 重新创建会话。"))};
        ArtifactId output;
        if (step.kind == QStringLiteral("ValidateRepairManifest")) {
            if (!loadRepairContract(&storage_, artifactStore_.get(), request.repairManifestArtifactId,
                    &baseline, &repair, &xany, stepCancellation, &executionError)) {
                const bool wasCanceled = executionError == QStringLiteral("annotation.canceled");
                const FailureCode code = wasCanceled ? FailureCode::Canceled
                    : (executionError.startsWith(QStringLiteral("annotation.format_not_implemented"))
                        ? FailureCode::BackendUnsupported : FailureCode::ArtifactIncompatible);
                return {wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    failure(code, executionError, QStringLiteral("重新生成完整的 Data Quality 修复清单 Artifact。"))};
            }
            const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("annotation_repair_validation")},
                {QStringLiteral("valid"), true},
                {QStringLiteral("sourceRepairManifestArtifactId"), request.repairManifestArtifactId.toString()},
                {QStringLiteral("sourceSnapshotId"), baseline.record.id.toString()},
                {QStringLiteral("sourceDatasetVersionId"), baseline.record.datasetVersionId.toString()},
                {QStringLiteral("datasetFormat"), baseline.record.datasetFormat}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("annotation_repair_validation"),
                    {{QStringLiteral("repair_validation.json"), jsonBytes(report)}}, &output, &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                    QStringLiteral("清理工作区 staging 后重新创建标注会话。"))};
            }
        } else if (step.kind == QStringLiteral("PrepareWorkingCopy")) {
            if (!prepareEmptyWorkingDirectory(request.workingDirectory, baseline.rootPath,
                    artifactStore_->rootPath(), &resolvedWorkingRoot, &workingRootCreated, &executionError)) {
                return {WorkflowStepState::Failed, {}, failure(FailureCode::InvalidRequest, executionError,
                    QStringLiteral("选择空的非符号链接独立工作目录后重新创建会话。"))};
            }
            workingCopyPrepared = true;
            if (!copyBaseline(baseline, resolvedWorkingRoot, stepCancellation, &executionError)) {
                cleanupPreparedWorkingCopy(resolvedWorkingRoot, workingRootCreated, baseline);
                workingCopyPrepared = false;
                const bool wasCanceled = executionError == QStringLiteral("annotation.canceled");
                return {wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    failure(wasCanceled ? FailureCode::Canceled : FailureCode::InvalidRequest, executionError,
                        QStringLiteral("选择空的独立工作目录后重新创建会话。"))};
            }
            QJsonArray editableFiles;
            for (const QJsonValue& value : repair.value(QStringLiteral("actions")).toArray()) {
                const QString relative = QDir::cleanPath(value.toObject().value(QStringLiteral("sourceRelativePath")).toString());
                if (!editable.contains(pathKey(relative))) {
                    editable.insert(pathKey(relative));
                    editableFiles.append(relative);
                }
            }
            const QJsonObject plan{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("annotation_working_copy_plan")},
                {QStringLiteral("sourceSnapshotId"), baseline.record.id.toString()},
                {QStringLiteral("fileCount"), QString::number(baseline.filesByKey.size())},
                {QStringLiteral("editableFiles"), editableFiles},
                {QStringLiteral("workingDirectoryPersisted"), false}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("annotation_working_copy_plan"),
                    {{QStringLiteral("working_copy_plan.json"), jsonBytes(plan)}}, &output, &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                    QStringLiteral("清理工作区 staging 后重新创建标注会话。"))};
            }
        } else if (step.kind == QStringLiteral("CommitAnnotationSession")) {
            QString staging;
            if (!artifactStore_->begin(taskId, QStringLiteral("annotation_session"),
                    &sessionArtifactId, &staging, &executionError)) {
                cleanupPreparedWorkingCopy(resolvedWorkingRoot, workingRootCreated, baseline);
                workingCopyPrepared = false;
                return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                    QStringLiteral("清理工作区 staging 后重新创建标注会话。"))};
            }
            QJsonArray baselineFiles;
            QStringList orderedKeys = baseline.filesByKey.keys();
            std::sort(orderedKeys.begin(), orderedKeys.end());
            for (const QString& key : orderedKeys) baselineFiles.append(baseline.filesByKey.value(key));
            QJsonArray editableFiles;
            QStringList orderedEditable = editable.values();
            std::sort(orderedEditable.begin(), orderedEditable.end());
            for (const QString& key : orderedEditable) {
                editableFiles.append(baseline.filesByKey.value(key).value(QStringLiteral("relativePath")).toString());
            }
            const QJsonObject manifest{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("annotation_session")},
                {QStringLiteral("sourceDatasetId"), baseline.record.datasetId.toString()},
                {QStringLiteral("sourceDatasetVersionId"), baseline.record.datasetVersionId.toString()},
                {QStringLiteral("sourceSnapshotId"), baseline.record.id.toString()},
                {QStringLiteral("sourceSnapshotArtifactId"), baseline.record.artifactId.toString()},
                {QStringLiteral("sourceRepairManifestArtifactId"), request.repairManifestArtifactId.toString()},
                {QStringLiteral("datasetFormat"), baseline.record.datasetFormat},
                {QStringLiteral("outputArtifactId"), sessionArtifactId.toString()},
                {QStringLiteral("createdAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
                {QStringLiteral("toolParameters"), toolSummary(request.toolParameters)},
                {QStringLiteral("problemLineage"), repair.value(QStringLiteral("actions"))},
                {QStringLiteral("baselineFiles"), baselineFiles},
                {QStringLiteral("editableFiles"), editableFiles},
                {QStringLiteral("workingDirectoryPersisted"), false}};
            const auto abort = [&]() { QString ignored; artifactStore_->abort(staging, &ignored); };
            if (!writeBytes(QDir(staging).filePath(QStringLiteral("annotation_session.json")), jsonBytes(manifest), &executionError)) {
                abort();
                cleanupPreparedWorkingCopy(resolvedWorkingRoot, workingRootCreated, baseline);
                workingCopyPrepared = false;
                return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                    QStringLiteral("清理工作区 staging 后重新创建标注会话。"))};
            }
            QString committedPath;
            if (!artifactStore_->commit(sessionArtifactId, taskId, QStringLiteral("annotation_session"), staging,
                    &storage_, &committedPath, &executionError, stepCancellation)) {
                abort();
                cleanupPreparedWorkingCopy(resolvedWorkingRoot, workingRootCreated, baseline);
                workingCopyPrepared = false;
                const bool wasCanceled = executionError.contains(QStringLiteral("取消"));
                return {wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    failure(wasCanceled ? FailureCode::Canceled : FailureCode::ArtifactIncomplete, executionError,
                        QStringLiteral("清理工作区 staging 后重新创建标注会话。"))};
            }
            output = sessionArtifactId;
        } else {
            return {WorkflowStepState::Failed, {}, failure(FailureCode::InternalError,
                QStringLiteral("annotation.create.unknown_step:%1").arg(step.kind),
                QStringLiteral("检查 Annotation Session Workflow 模板。"))};
        }
        return {WorkflowStepState::Succeeded, output, {}};
    };
    if (!runner.run(workflow.id, executor, &run, error, cancellation)) return false;
    if (run.state != WorkflowStepState::Succeeded && workingCopyPrepared) {
        cleanupPreparedWorkingCopy(resolvedWorkingRoot, workingRootCreated, baseline);
        workingCopyPrepared = false;
    }
    ArtifactId evidence;
    if (!finishWorkflow(this, &storage_, taskId, workflow.id, run,
            QStringLiteral("修复输入 Artifact 或工作目录后重新创建标注会话。"), &evidence, error)) return false;
    result->workflowRunId = workflow.id;
    result->terminalState = terminalTaskState(run);
    result->sessionArtifactId = sessionArtifactId;
    result->evidenceArtifactId = evidence;
    result->status = run.state == WorkflowStepState::Succeeded ? AnnotationSyncStatus::Inspected
        : (run.state == WorkflowStepState::Canceled ? AnnotationSyncStatus::Canceled : AnnotationSyncStatus::InvalidSession);
    return true;
}

bool ProjectWorkspace::syncAnnotationSession(const TaskId& taskId,
    const AnnotationSessionSyncRequest& request, AnnotationSessionSyncResult* result,
    QString* error, const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !request.sessionArtifactId.isValid()
        || request.workingDirectory.trimmed().isEmpty() || !result) {
        if (error) *error = QStringLiteral("同步标注会话需要运行中任务、会话 Artifact、工作目录和输出对象。");
        return false;
    }
    TaskSnapshot task;
    ArtifactSnapshot inputArtifact;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running
        || !storage_.artifact(request.sessionArtifactId, &inputArtifact, error)) return false;

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("annotation_session_sync");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    const QStringList kinds{QStringLiteral("InspectSession"), QStringLiteral("DetectChanges"),
        QStringLiteral("CommitDatasetSnapshot"), QStringLiteral("RenderSyncReport")};
    QVector<WorkflowStepSnapshot> steps;
    const QJsonObject parameters{{QStringLiteral("annotationSessionArtifactId"), request.sessionArtifactId.toString()}};
    for (int index = 0; index < kinds.size(); ++index) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = index;
        step.kind = kinds.at(index);
        step.backend = QStringLiteral("builtin_annotation_sync");
        step.parameterSummary = parameters;
        if (index == 0) step.inputArtifactId = request.sessionArtifactId;
        steps.append(step);
    }
    WorkflowInputBinding input;
    input.workflowRunId = workflow.id;
    input.role = QStringLiteral("annotation_session");
    input.sourceArtifactId = inputArtifact.id;
    input.sourceTaskId = inputArtifact.taskId;
    input.sourceArtifactKind = inputArtifact.kind;
    if (!storage_.createWorkflowRunWithInput(workflow, steps, input, error)) return false;

    SyncInspection inspection;
    ArtifactId inspectionArtifactId;
    ArtifactId changesArtifactId;
    ArtifactId syncReportArtifactId;
    DatasetSnapshotRecord newSnapshot;
    QString resolvedSyncWorkingRoot;
    QString executionError;
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult run;
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) -> WorkflowStepExecutionResult {
        if (canceled(stepCancellation)) return {WorkflowStepState::Canceled, {},
            failure(FailureCode::Canceled, QStringLiteral("标注会话同步已取消。"),
                QStringLiteral("确认工作目录未继续写入后重新同步会话。"))};
        ArtifactId output;
        if (step.kind == QStringLiteral("InspectSession")) {
            if (!loadSessionContract(&storage_, artifactStore_.get(), request.sessionArtifactId,
                    &inspection.session, stepCancellation, &executionError)) {
                inspection.status = AnnotationSyncStatus::InvalidSession;
                inspection.code = QStringLiteral("annotation.session.invalid_manifest");
                inspection.message = executionError;
            } else if (!loadBaseline(&storage_, artifactStore_.get(), inspection.session.snapshotId,
                    &inspection.baseline, true, stepCancellation, &executionError)
                || inspection.baseline.record.datasetVersionId != inspection.session.datasetVersionId
                || inspection.baseline.record.datasetFormat != inspection.session.datasetFormat) {
                inspection.status = AnnotationSyncStatus::Conflict;
                inspection.code = QStringLiteral("annotation.session.baseline_changed");
                inspection.message = executionError.isEmpty()
                    ? QStringLiteral("会话基线 Dataset Version/Snapshot 已不一致。") : executionError;
            } else if (!validateSessionLineage(&storage_, artifactStore_.get(), inspection.session,
                    inspection.baseline, stepCancellation, &executionError)) {
                inspection.status = AnnotationSyncStatus::InvalidSession;
                inspection.code = QStringLiteral("annotation.session.lineage_invalid");
                inspection.message = executionError;
            } else {
                inspection.status = AnnotationSyncStatus::Inspected;
                inspection.code = QStringLiteral("annotation.session.inspected");
                inspection.message = QStringLiteral("会话 Manifest 和基线 Snapshot 已重新校验。");
            }
            const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("annotation_session_inspection")},
                {QStringLiteral("status"), statusText(inspection.status)},
                {QStringLiteral("code"), inspection.code}, {QStringLiteral("message"), inspection.message},
                {QStringLiteral("annotationSessionArtifactId"), request.sessionArtifactId.toString()},
                {QStringLiteral("sourceSnapshotId"), inspection.session.snapshotId.toString()},
                {QStringLiteral("sourceDatasetVersionId"), inspection.session.datasetVersionId.toString()}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("annotation_session_inspection"),
                    {{QStringLiteral("session_inspection.json"), jsonBytes(report)}}, &inspectionArtifactId,
                    &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                    QStringLiteral("清理 staging 后重新同步会话。"))};
            }
            output = inspectionArtifactId;
        } else if (step.kind == QStringLiteral("DetectChanges")) {
            QJsonArray changed;
            if (inspection.status == AnnotationSyncStatus::Inspected) {
                QHash<QString, QString> actualFiles;
                if (!resolveReadableWorkingDirectory(request.workingDirectory,
                        inspection.baseline.rootPath, artifactStore_->rootPath(),
                        &resolvedSyncWorkingRoot, &executionError)
                    || !collectWorkingFiles(resolvedSyncWorkingRoot, &actualFiles, &executionError)
                    || actualFiles.size() != inspection.session.baselineHashes.size()) {
                    inspection.status = AnnotationSyncStatus::InvalidSession;
                    inspection.code = QStringLiteral("annotation.session.inventory_changed");
                    if (executionError.isEmpty()) executionError = QStringLiteral("工作目录文件集合与会话基线不一致。");
                    inspection.message = executionError;
                } else {
                    QStringList keys = inspection.session.baselineHashes.keys();
                    std::sort(keys.begin(), keys.end());
                    for (const QString& key : keys) {
                        if (canceled(stepCancellation)) return {WorkflowStepState::Canceled, {},
                            failure(FailureCode::Canceled, QStringLiteral("标注会话同步已取消。"),
                                QStringLiteral("确认工作目录未继续写入后重新同步会话。"))};
                        if (!actualFiles.contains(key)) {
                            inspection.status = AnnotationSyncStatus::InvalidSession;
                            inspection.code = QStringLiteral("annotation.session.inventory_changed");
                            inspection.message = QStringLiteral("工作目录缺少基线文件。");
                            break;
                        }
                        QString hash;
                        if (!readBytesAndHash(QDir(resolvedSyncWorkingRoot).filePath(actualFiles.value(key)),
                                nullptr, &hash, stepCancellation, &executionError)) {
                            inspection.status = AnnotationSyncStatus::InvalidSession;
                            inspection.code = QStringLiteral("annotation.session.file_unreadable");
                            inspection.message = executionError;
                            break;
                        }
                        if (hash != inspection.session.baselineHashes.value(key)) {
                            inspection.changedFiles.append(actualFiles.value(key));
                            changed.append(QJsonObject{{QStringLiteral("relativePath"), actualFiles.value(key)},
                                {QStringLiteral("baselineSha256"), inspection.session.baselineHashes.value(key)},
                                {QStringLiteral("actualSha256"), hash}});
                            if (!inspection.session.editableKeys.contains(key)) {
                                inspection.status = AnnotationSyncStatus::InvalidSession;
                                inspection.code = QStringLiteral("annotation.session.change_out_of_scope");
                                inspection.message = QStringLiteral("检测到白名单外文件变更：%1").arg(actualFiles.value(key));
                                break;
                            }
                        }
                    }
                    if (inspection.status == AnnotationSyncStatus::Inspected) {
                        if (inspection.changedFiles.isEmpty()) {
                            inspection.status = AnnotationSyncStatus::NoChanges;
                            inspection.code = QStringLiteral("annotation.session.no_changes");
                            inspection.message = QStringLiteral("工作目录与会话基线完全一致。");
                        } else if (!validateWithDriver(resolvedSyncWorkingRoot, inspection.session.datasetFormat,
                                stepCancellation, &executionError)) {
                            inspection.status = AnnotationSyncStatus::InvalidSession;
                            inspection.code = QStringLiteral("annotation.session.driver_validation_failed");
                            inspection.message = executionError;
                        } else {
                            inspection.status = AnnotationSyncStatus::ChangesDetected;
                            inspection.code = QStringLiteral("annotation.session.changes_detected");
                            inspection.message = QStringLiteral("检测到白名单内有效标注变更。");
                        }
                    }
                }
            }
            const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("annotation_change_report")},
                {QStringLiteral("status"), statusText(inspection.status)},
                {QStringLiteral("code"), inspection.code}, {QStringLiteral("message"), inspection.message},
                {QStringLiteral("changedFiles"), changed},
                {QStringLiteral("sourceSnapshotId"), inspection.session.snapshotId.toString()},
                {QStringLiteral("sourceDatasetVersionId"), inspection.session.datasetVersionId.toString()}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("annotation_change_report"),
                    {{QStringLiteral("change_report.json"), jsonBytes(report)}}, &changesArtifactId,
                    &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                    QStringLiteral("清理 staging 后重新同步会话。"))};
            }
            output = changesArtifactId;
        } else if (step.kind == QStringLiteral("CommitDatasetSnapshot")) {
            if (inspection.status == AnnotationSyncStatus::InvalidSession
                || inspection.status == AnnotationSyncStatus::Conflict) {
                return {WorkflowStepState::Failed, {}, failure(
                    inspection.status == AnnotationSyncStatus::Conflict
                        ? FailureCode::ArtifactIncompatible : FailureCode::InvalidDataset,
                    inspection.code + QLatin1Char(':') + inspection.message,
                    QStringLiteral("不要覆盖原始数据；解决冲突或越界变更后创建新会话。"))};
            }
            if (inspection.status == AnnotationSyncStatus::NoChanges) {
                const QJsonObject decision{{QStringLiteral("schemaVersion"), 2},
                    {QStringLiteral("kind"), QStringLiteral("annotation_sync_decision")},
                    {QStringLiteral("status"), statusText(inspection.status)},
                    {QStringLiteral("newDatasetVersionCreated"), false}};
                if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("annotation_sync_decision"),
                        {{QStringLiteral("sync_decision.json"), jsonBytes(decision)}}, &output,
                        &executionError, stepCancellation)) {
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                        QStringLiteral("清理 staging 后重新同步会话。"))};
                }
            } else if (inspection.status == AnnotationSyncStatus::ChangesDetected) {
                ArtifactId snapshotArtifactId;
                QString staging;
                QString verifiedRoot;
                if (!resolveReadableWorkingDirectory(resolvedSyncWorkingRoot,
                        inspection.baseline.rootPath, artifactStore_->rootPath(),
                        &verifiedRoot, &executionError)
                    || verifiedRoot.compare(resolvedSyncWorkingRoot, Qt::CaseInsensitive) != 0) {
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::InvalidRequest,
                        executionError.isEmpty() ? QStringLiteral("annotation.working_directory_identity_changed") : executionError,
                        QStringLiteral("不要替换会话工作目录；重新创建受控会话。"))};
                }
                if (!artifactStore_->begin(taskId, QStringLiteral("dataset_snapshot"),
                        &snapshotArtifactId, &staging, &executionError)) {
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                        QStringLiteral("清理 staging 后重新同步会话。"))};
                }
                const auto abort = [&]() { QString ignored; artifactStore_->abort(staging, &ignored); };
                QHash<QString, QString> workingFiles;
                if (!collectWorkingFiles(resolvedSyncWorkingRoot, &workingFiles, &executionError)) {
                    abort();
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::InvalidDataset, executionError,
                        QStringLiteral("恢复会话工作目录的完整基线文件集合。"))};
                }
                QStringList keys = workingFiles.keys();
                std::sort(keys.begin(), keys.end());
                for (const QString& key : keys) {
                    QByteArray bytes;
                    if (!readBytesAndHash(QDir(resolvedSyncWorkingRoot).filePath(workingFiles.value(key)),
                            &bytes, nullptr, stepCancellation, &executionError)
                        || !writeBytes(QDir(staging).filePath(workingFiles.value(key)), bytes, &executionError)) {
                        abort();
                        const bool wasCanceled = executionError == QStringLiteral("annotation.canceled");
                        return {wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                            failure(wasCanceled ? FailureCode::Canceled : FailureCode::ArtifactIncomplete,
                                executionError, QStringLiteral("确认工作目录稳定后重新同步会话。"))};
                    }
                }
                DatasetDriverRegistry registry;
                if (!registerBuiltinDatasetDrivers(&registry, &executionError)) {
                    abort();
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::InternalError, executionError,
                        QStringLiteral("检查内置 Dataset Driver 注册。"))};
                }
                const DatasetDriver* driver = registry.driverForFormat(inspection.session.datasetFormat);
                DatasetSnapshotOptions options;
                options.classDefinitions = inspection.baseline.manifest.value(QStringLiteral("classDefinitions")).toArray();
                options.isCancellationRequested = stepCancellation;
                DatasetSnapshotResult snapshotResult;
                const QString snapshotManifestPath = QDir(staging).filePath(QStringLiteral("dataset_snapshot.json"));
                if (!driver || !createDatasetSnapshot(staging, snapshotManifestPath,
                        inspection.session.datasetFormat, driver->id(), driver->version(), options,
                        &snapshotResult, &executionError)) {
                    abort();
                    const bool wasCanceled = executionError == QStringLiteral("snapshot_canceled");
                    return {wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                        failure(wasCanceled ? FailureCode::Canceled : FailureCode::InvalidDataset,
                            executionError, QStringLiteral("修复工作目录后重新同步会话。"))};
                }
                QJsonObject enriched = snapshotResult.manifest;
                enriched.insert(QStringLiteral("sourceSnapshotId"), inspection.session.snapshotId.toString());
                enriched.insert(QStringLiteral("sourceDatasetVersionId"), inspection.session.datasetVersionId.toString());
                enriched.insert(QStringLiteral("annotationSessionArtifactId"), request.sessionArtifactId.toString());
                enriched.insert(QStringLiteral("changedFiles"), QJsonArray::fromStringList(inspection.changedFiles));
                QFile::remove(snapshotManifestPath);
                if (!writeBytes(snapshotManifestPath, jsonBytes(enriched), &executionError)) {
                    abort();
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                        QStringLiteral("清理 staging 后重新同步会话。"))};
                }
                QString committedPath;
                if (!artifactStore_->commit(snapshotArtifactId, taskId, QStringLiteral("dataset_snapshot"),
                        staging, &storage_, &committedPath, &executionError, stepCancellation)) {
                    abort();
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                        QStringLiteral("清理 staging 后重新同步会话。"))};
                }
                ArtifactSnapshot committed;
                if (!storage_.artifact(snapshotArtifactId, &committed, &executionError)) {
                    QString ignored; artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                        QStringLiteral("重新同步会话并生成完整 Snapshot Artifact。"))};
                }
                QString manifestHash;
                for (const ArtifactFileSnapshot& file : committed.files) {
                    if (file.relativePath == QStringLiteral("dataset_snapshot.json")) manifestHash = file.sha256;
                }
                newSnapshot.id = snapshotResult.snapshotId;
                newSnapshot.taskId = taskId;
                newSnapshot.artifactId = snapshotArtifactId;
                newSnapshot.rootPath = committedPath;
                newSnapshot.datasetFormat = inspection.session.datasetFormat;
                newSnapshot.driverId = driver->id();
                newSnapshot.driverVersion = driver->version();
                newSnapshot.rootHash = snapshotResult.rootHash;
                newSnapshot.manifestSha256 = manifestHash;
                newSnapshot.fileCount = snapshotResult.fileCount;
                newSnapshot.totalBytes = snapshotResult.totalBytes;
                if (manifestHash.isEmpty() || !storage_.registerDatasetSnapshot(&newSnapshot, &executionError)) {
                    QString ignored; artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                    newSnapshot = {};
                    return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                        QStringLiteral("重新同步会话并登记新 Dataset Version/Snapshot。"))};
                }
                output = snapshotArtifactId;
            } else {
                return {WorkflowStepState::Failed, {}, failure(FailureCode::InternalError,
                    QStringLiteral("annotation.sync.unexpected_status:%1").arg(statusText(inspection.status)),
                    QStringLiteral("检查 Annotation Sync Workflow 状态机。"))};
            }
        } else if (step.kind == QStringLiteral("RenderSyncReport")) {
            const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("annotation_sync_report")},
                {QStringLiteral("status"), statusText(inspection.status)},
                {QStringLiteral("code"), inspection.code}, {QStringLiteral("message"), inspection.message},
                {QStringLiteral("annotationSessionArtifactId"), request.sessionArtifactId.toString()},
                {QStringLiteral("sourceSnapshotId"), inspection.session.snapshotId.toString()},
                {QStringLiteral("sourceDatasetVersionId"), inspection.session.datasetVersionId.toString()},
                {QStringLiteral("newSnapshotId"), newSnapshot.id.toString()},
                {QStringLiteral("newDatasetVersionId"), newSnapshot.datasetVersionId.toString()},
                {QStringLiteral("newSnapshotArtifactId"), newSnapshot.artifactId.toString()},
                {QStringLiteral("changedFiles"), QJsonArray::fromStringList(inspection.changedFiles)},
                {QStringLiteral("mutatedSource"), false}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("annotation_sync_report"),
                    {{QStringLiteral("annotation_sync_report.json"), jsonBytes(report)}}, &syncReportArtifactId,
                    &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, failure(FailureCode::ArtifactIncomplete, executionError,
                    QStringLiteral("清理 staging 后重新同步会话。"))};
            }
            output = syncReportArtifactId;
        } else {
            return {WorkflowStepState::Failed, {}, failure(FailureCode::InternalError,
                QStringLiteral("annotation.sync.unknown_step:%1").arg(step.kind),
                QStringLiteral("检查 Annotation Sync Workflow 模板。"))};
        }
        return {WorkflowStepState::Succeeded, output, {}};
    };
    if (!runner.run(workflow.id, executor, &run, error, cancellation)) return false;
    ArtifactId evidence;
    if (!finishWorkflow(this, &storage_, taskId, workflow.id, run,
            QStringLiteral("解决会话冲突、越界变更或工作目录问题后重新同步。"), &evidence, error)) return false;
    result->workflowRunId = workflow.id;
    result->terminalState = terminalTaskState(run);
    result->status = run.state == WorkflowStepState::Canceled ? AnnotationSyncStatus::Canceled : inspection.status;
    result->inspectionArtifactId = inspectionArtifactId;
    result->changesArtifactId = changesArtifactId;
    result->syncReportArtifactId = syncReportArtifactId.isValid() ? syncReportArtifactId : changesArtifactId;
    result->evidenceArtifactId = evidence;
    result->datasetSnapshot = newSnapshot;
    return true;
}

} // namespace aitrain
