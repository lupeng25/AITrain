#include "aitrain/artifact/ArtifactStore.h"
#include "aitrain/domain/ArtifactMemberPath.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSaveFile>
#include <QVector>

#include <limits>
#include <utility>

namespace aitrain {
namespace {

constexpr auto kStagingDirectoryName = ".staging";
constexpr auto kStagingMetadataDirectoryName = ".staging-meta";
constexpr auto kCommittedDirectoryName = "committed";
constexpr auto kTrashDirectoryName = ".trash";
constexpr auto kJournalSchemaVersion = 2;
constexpr auto kPhaseBegun = "begun";
constexpr auto kPhasePrepared = "prepared";
constexpr auto kPhaseFilesCommitted = "files_committed";
constexpr auto kPhaseCatalogCommitted = "catalog_committed";

struct FileEntry final {
    QString relativePath;
    QString sha256;
    qint64 bytes = 0;
};

struct CommitJournal final {
    ArtifactId artifactId;
    TaskId taskId;
    QString kind;
    QString phase;
    QDateTime createdAt;
    WorkflowRunId workflowRunId;
    QVector<FileEntry> files;
    QString completionAction;
};

ArtifactCommitPhase commitPhase(const QString& value)
{
    if (value == QLatin1String(kPhaseBegun)) return ArtifactCommitPhase::Begun;
    if (value == QLatin1String(kPhasePrepared)) return ArtifactCommitPhase::Prepared;
    if (value == QLatin1String(kPhaseFilesCommitted)) return ArtifactCommitPhase::FilesCommitted;
    if (value == QLatin1String(kPhaseCatalogCommitted)) return ArtifactCommitPhase::CatalogCommitted;
    return ArtifactCommitPhase::None;
}

QString utcText(const QDateTime& value)
{
    return value.toUTC().toString(Qt::ISODateWithMs);
}

bool sameFiles(const QVector<FileEntry>& actual, const QVector<ArtifactFileSnapshot>& stored)
{
    if (actual.size() != stored.size()) return false;
    QHash<QString, QPair<QString, qint64>> expected;
    for (const FileEntry& entry : actual) {
        expected.insert(entry.relativePath, {entry.sha256, entry.bytes});
    }
    for (const ArtifactFileSnapshot& file : stored) {
        if (!expected.contains(file.relativePath)
            || expected.value(file.relativePath).first != file.sha256
            || expected.value(file.relativePath).second != file.byteCount) {
            return false;
        }
    }
    return true;
}

bool collectFiles(const QString& stagingPath,
    QVector<FileEntry>* entries,
    QString* error,
    const aitrain::CancellationCallback& cancellation,
    bool* canceled)
{
    if (canceled) {
        *canceled = false;
    }
    // Artifact staging 只允许收集普通文件。QDir::Files 默认可能跟随文件
    // 符号链接，若把链接目标的内容写入 inventory，提交后的 Artifact 就不再
    // 是不可变快照；同时也可能把工作区外的文件带入交付包。
    QDirIterator iterator(stagingPath, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        if (aitrain::isCancellationRequested(cancellation)) {
            if (canceled) *canceled = true;
            if (error) *error = QStringLiteral("Artifact 提交已取消。");
            return false;
        }
        const QString absolutePath = iterator.next();
        const QFileInfo info(absolutePath);
        QString relativePath;
        const QString rawRelativePath = QDir(stagingPath).relativeFilePath(absolutePath);
        if (!normalizeArtifactMemberPath(rawRelativePath, &relativePath, error)) {
            return false;
        }
        if (info.isSymLink() || !info.isFile()
            || relativePath.isEmpty()
            || relativePath == QStringLiteral("..")
            || relativePath.startsWith(QStringLiteral("../"))) {
            if (error) {
                *error = QStringLiteral("Artifact staging 只能包含位于根目录内的普通文件：%1").arg(absolutePath);
            }
            return false;
        }
        if (relativePath == QStringLiteral("manifest.json")) {
            continue;
        }
        QFile file(absolutePath);
        if (!file.open(QIODevice::ReadOnly)) {
            if (error) {
                *error = QStringLiteral("无法读取 staging 文件：%1").arg(absolutePath);
            }
            return false;
        }
        QCryptographicHash hash(QCryptographicHash::Sha256);
        while (!file.atEnd()) {
            if (aitrain::isCancellationRequested(cancellation)) {
                if (canceled) *canceled = true;
                if (error) *error = QStringLiteral("Artifact 提交已取消。");
                return false;
            }
            const QByteArray block = file.read(1024 * 1024);
            if (block.isEmpty() && file.error() != QFileDevice::NoError) {
                if (error) {
                    *error = QStringLiteral("读取 staging 文件失败：%1").arg(absolutePath);
                }
                return false;
            }
            hash.addData(block);
        }
        entries->append({relativePath, QString::fromLatin1(hash.result().toHex()), info.size()});
    }
    if (entries->isEmpty()) {
        if (error) {
            *error = QStringLiteral("Artifact staging 不能为空。");
        }
        return false;
    }
    return true;
}

bool writeManifest(const QString& stagingPath,
    const CommitJournal& journal,
    const QVector<FileEntry>& entries,
    QString* error)
{
    QJsonArray files;
    for (const FileEntry& entry : entries) {
        files.append(QJsonObject{{QStringLiteral("relativePath"), entry.relativePath},
            {QStringLiteral("sha256"), entry.sha256},
            {QStringLiteral("bytes"), QString::number(entry.bytes)}});
    }
    const QJsonObject manifest{{QStringLiteral("schemaVersion"), kJournalSchemaVersion},
        {QStringLiteral("artifactId"), journal.artifactId.toString()},
        {QStringLiteral("taskId"), journal.taskId.toString()},
        {QStringLiteral("kind"), journal.kind},
        {QStringLiteral("commitPhase"), QString::fromLatin1(kPhasePrepared)},
        {QStringLiteral("createdAt"), utcText(journal.createdAt)},
        {QStringLiteral("workflowRunId"), journal.workflowRunId.toString()},
        {QStringLiteral("completionAction"), journal.completionAction},
        {QStringLiteral("files"), files}};
    QSaveFile file(QDir(stagingPath).filePath(QStringLiteral("manifest.json")));
    if (!file.open(QIODevice::WriteOnly)
        || file.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented)) < 0
        || !file.commit()) {
        if (error) {
            *error = QStringLiteral("无法写入 Artifact manifest：%1").arg(file.errorString());
        }
        return false;
    }
    return true;
}

QString stagingMetadataPath(const QString& rootPath, const ArtifactId& artifactId)
{
    return QDir(rootPath).filePath(QStringLiteral("%1/%2.json").arg(QString::fromLatin1(kStagingMetadataDirectoryName), artifactId.toString()));
}

bool writeStagingMetadata(const QString& rootPath, const CommitJournal& journal, QString* error)
{
    const QString path = stagingMetadataPath(rootPath, journal.artifactId);
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) {
            *error = QStringLiteral("无法创建 Artifact staging 元数据目录：%1").arg(path);
        }
        return false;
    }
    QSaveFile file(path);
    QJsonArray files;
    for (const FileEntry& entry : journal.files) {
        files.append(QJsonObject{
            {QStringLiteral("relativePath"), entry.relativePath},
            {QStringLiteral("sha256"), entry.sha256},
            {QStringLiteral("bytes"), QString::number(entry.bytes)}});
    }
    const QJsonObject metadata{{QStringLiteral("schemaVersion"), kJournalSchemaVersion},
        {QStringLiteral("artifactId"), journal.artifactId.toString()},
        {QStringLiteral("taskId"), journal.taskId.toString()},
        {QStringLiteral("kind"), journal.kind},
        {QStringLiteral("phase"), journal.phase},
        {QStringLiteral("createdAt"), utcText(journal.createdAt)},
        {QStringLiteral("workflowRunId"), journal.workflowRunId.toString()},
        {QStringLiteral("completionAction"), journal.completionAction},
        {QStringLiteral("files"), files}};
    if (!file.open(QIODevice::WriteOnly)
        || file.write(QJsonDocument(metadata).toJson(QJsonDocument::Compact)) < 0
        || !file.commit()) {
        if (error) {
            *error = QStringLiteral("无法写入 Artifact staging 元数据：%1").arg(file.errorString());
        }
        return false;
    }
    return true;
}

bool readCommitJournal(const QString& metadataPath, CommitJournal* journal, QString* error)
{
    if (!journal) {
        if (error) *error = QStringLiteral("读取 Artifact 提交日志需要输出对象。");
        return false;
    }
    QFile file(metadataPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("无法读取 Artifact staging 元数据：%1").arg(metadataPath);
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Artifact staging 元数据不是有效 JSON：%1").arg(metadataPath);
        }
        return false;
    }
    const QJsonObject object = document.object();
    CommitJournal parsed;
    if (object.value(QStringLiteral("schemaVersion")).toInt(-1) != kJournalSchemaVersion
        || !ArtifactId::parse(object.value(QStringLiteral("artifactId")).toString(), &parsed.artifactId, error)
        || !TaskId::parse(object.value(QStringLiteral("taskId")).toString(), &parsed.taskId, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("Artifact 提交日志版本或标识无效：%1").arg(metadataPath);
        return false;
    }
    parsed.kind = object.value(QStringLiteral("kind")).toString().trimmed();
    parsed.phase = object.value(QStringLiteral("phase")).toString();
    parsed.createdAt = QDateTime::fromString(object.value(QStringLiteral("createdAt")).toString(), Qt::ISODateWithMs);
    const QString workflowRunId = object.value(QStringLiteral("workflowRunId")).toString();
    if (!workflowRunId.isEmpty() && !WorkflowRunId::parse(workflowRunId, &parsed.workflowRunId, error)) return false;
    parsed.completionAction = object.value(QStringLiteral("completionAction")).toString();
    for (const QJsonValue& value : object.value(QStringLiteral("files")).toArray()) {
        const QJsonObject fileObject = value.toObject();
        bool byteCountOk = false;
        FileEntry entry;
        entry.relativePath = fileObject.value(QStringLiteral("relativePath")).toString();
        entry.sha256 = fileObject.value(QStringLiteral("sha256")).toString();
        entry.bytes = fileObject.value(QStringLiteral("bytes")).toString().toLongLong(&byteCountOk);
        if (!byteCountOk || entry.relativePath.isEmpty()
            || entry.sha256.size() != 64 || entry.bytes < 0) {
            if (error) *error = QStringLiteral("Artifact journal v2 inventory 无效：%1").arg(metadataPath);
            return false;
        }
        parsed.files.append(entry);
    }
    if (parsed.kind.isEmpty() || !parsed.createdAt.isValid()
        || (parsed.phase != QLatin1String(kPhaseBegun)
            && parsed.phase != QLatin1String(kPhasePrepared)
            && parsed.phase != QLatin1String(kPhaseFilesCommitted)
            && parsed.phase != QLatin1String(kPhaseCatalogCommitted))
        || (parsed.phase != QLatin1String(kPhaseBegun)
            && (parsed.files.isEmpty() || parsed.completionAction.isEmpty()))) {
        if (error) *error = QStringLiteral("Artifact 提交日志字段无效：%1").arg(metadataPath);
        return false;
    }
    *journal = parsed;
    return true;
}

bool readAndVerifyManifest(const QString& artifactPath,
    const CommitJournal& journal,
    QVector<FileEntry>* entries,
    QString* error)
{
    QFile file(QDir(artifactPath).filePath(QStringLiteral("manifest.json")));
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取 Artifact manifest：%1").arg(file.fileName());
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    const QJsonObject object = document.object();
    if (parseError.error != QJsonParseError::NoError || !document.isObject()
        || object.value(QStringLiteral("schemaVersion")).toInt(-1) != kJournalSchemaVersion
        || object.value(QStringLiteral("artifactId")).toString() != journal.artifactId.toString()
        || object.value(QStringLiteral("taskId")).toString() != journal.taskId.toString()
        || object.value(QStringLiteral("kind")).toString() != journal.kind
        || object.value(QStringLiteral("createdAt")).toString() != utcText(journal.createdAt)
        || object.value(QStringLiteral("workflowRunId")).toString() != journal.workflowRunId.toString()
        || object.value(QStringLiteral("completionAction")).toString() != journal.completionAction) {
        if (error) *error = QStringLiteral("Artifact manifest 与提交日志不一致：%1").arg(file.fileName());
        return false;
    }
    QVector<FileEntry> actual;
    if (!collectFiles(artifactPath, &actual, error, {}, nullptr)) return false;
    const QJsonArray files = object.value(QStringLiteral("files")).toArray();
    if (files.size() != actual.size()) {
        if (error) *error = QStringLiteral("Artifact manifest 文件数量与磁盘不一致：%1").arg(artifactPath);
        return false;
    }
    QHash<QString, QPair<QString, qint64>> declared;
    for (const QJsonValue& value : files) {
        const QJsonObject item = value.toObject();
        const QJsonValue byteValue = item.value(QStringLiteral("bytes"));
        bool bytesOk = false;
        const qint64 bytes = byteValue.isString()
            ? byteValue.toString().toLongLong(&bytesOk)
            : static_cast<qint64>(byteValue.toDouble(-1));
        if (!bytesOk && byteValue.isDouble()) {
            bytesOk = byteValue.toDouble() >= 0
                && byteValue.toDouble() <= static_cast<double>(std::numeric_limits<qint64>::max());
        }
        if (!bytesOk || bytes < 0) {
            if (error) *error = QStringLiteral("Artifact manifest 文件大小无效：%1").arg(file.fileName());
            return false;
        }
        declared.insert(item.value(QStringLiteral("relativePath")).toString(),
            {item.value(QStringLiteral("sha256")).toString(), bytes});
    }
    if (declared.size() != files.size()) {
        if (error) *error = QStringLiteral("Artifact manifest 包含重复文件路径：%1").arg(file.fileName());
        return false;
    }
    for (const FileEntry& entry : actual) {
        if (!declared.contains(entry.relativePath)
            || declared.value(entry.relativePath).first != entry.sha256
            || declared.value(entry.relativePath).second != entry.bytes) {
            if (error) *error = QStringLiteral("Artifact 文件与 manifest 校验失败：%1").arg(entry.relativePath);
            return false;
        }
    }
    if (actual.size() != journal.files.size()) {
        if (error) *error = QStringLiteral("Artifact 文件与 journal v2 固化清单数量不一致：%1").arg(artifactPath);
        return false;
    }
    QHash<QString, QPair<QString, qint64>> journalInventory;
    for (const FileEntry& entry : journal.files) {
        journalInventory.insert(entry.relativePath, {entry.sha256, entry.bytes});
    }
    for (const FileEntry& entry : actual) {
        if (!journalInventory.contains(entry.relativePath)
            || journalInventory.value(entry.relativePath).first != entry.sha256
            || journalInventory.value(entry.relativePath).second != entry.bytes) {
            if (error) *error = QStringLiteral("Artifact 文件与 journal v2 固化清单不一致：%1")
                .arg(entry.relativePath);
            return false;
        }
    }
    *entries = actual;
    return true;
}

QVector<ArtifactFileSnapshot> snapshots(const QVector<FileEntry>& entries)
{
    QVector<ArtifactFileSnapshot> result;
    result.reserve(entries.size());
    for (const FileEntry& entry : entries) result.append({entry.relativePath, entry.sha256, entry.bytes});
    return result;
}

bool persistArtifactRecord(ProjectStore* storage,
    const CommitJournal& journal,
    const QVector<FileEntry>& entries,
    QString* error)
{
    const QVector<ArtifactFileSnapshot> files = snapshots(entries);
    if (journal.workflowRunId.isValid()) {
        if (journal.kind != QStringLiteral("evidence_bundle")) {
            if (error) *error = QStringLiteral("只有 Evidence Artifact 可以关联工作流终态。");
            return false;
        }
        return storage->recordEvidenceArtifactWithFilesAndAttachTerminalization(
            journal.artifactId, journal.taskId, journal.workflowRunId, files, journal.createdAt, error);
    }
    return storage->recordArtifactWithFiles(
        journal.artifactId, journal.taskId, journal.kind, files, journal.createdAt, error);
}

bool verifyStoredArtifact(ProjectStore* storage,
    const CommitJournal& journal,
    const QVector<FileEntry>& entries,
    QString* error)
{
    ArtifactSnapshot stored;
    if (!storage->artifact(journal.artifactId, &stored, error)) return false;
    if (stored.taskId != journal.taskId || stored.kind != journal.kind
        || stored.createdAt.toUTC() != journal.createdAt.toUTC()
        || !sameFiles(entries, stored.files)) {
        if (error) *error = QStringLiteral("Artifact 数据库记录与提交日志或磁盘文件不一致：%1")
            .arg(journal.artifactId.toString());
        return false;
    }
    return true;
}

void appendDiagnostic(QStringList* diagnostics, const QString& message)
{
    if (diagnostics) {
        diagnostics->append(message);
    }
}

} // namespace

ArtifactStore::ArtifactStore(QString rootPath,
    ArtifactCommitFailureInjector failureInjector)
    : rootPath_(QDir::cleanPath(std::move(rootPath)))
    , failureInjector_(std::move(failureInjector))
{
}

bool ArtifactStore::begin(const TaskId& taskId, const QString& kind, ArtifactId* artifactId, QString* stagingPath, QString* error)
{
    if (!taskId.isValid() || kind.trimmed().isEmpty() || !artifactId || !stagingPath) {
        if (error) {
            *error = QStringLiteral("创建 Artifact staging 的参数无效。");
        }
        return false;
    }
    const ArtifactId id = ArtifactId::create();
    const QString path = QDir(rootPath_).filePath(QStringLiteral("%1/%2").arg(QString::fromLatin1(kStagingDirectoryName), id.toString()));
    if (!QDir().mkpath(path)) {
        if (error) {
            *error = QStringLiteral("无法创建 Artifact staging：%1").arg(path);
        }
        return false;
    }
    const CommitJournal journal{id, taskId, kind.trimmed(), QString::fromLatin1(kPhaseBegun),
        QDateTime::currentDateTimeUtc(), {}, {}, {}};
    if (!writeStagingMetadata(rootPath_, journal, error)) {
        QDir(path).removeRecursively();
        return false;
    }
    *artifactId = id;
    *stagingPath = path;
    return true;
}

ArtifactCommitResult ArtifactStore::commit(const ArtifactId& artifactId,
    const TaskId& taskId,
    const QString& kind,
    const QString& stagingPath,
    ProjectStore* storage,
    QString* artifactPath,
    QString* error,
    const aitrain::CancellationCallback& cancellation,
    bool* canceled,
    const WorkflowRunId& workflowRunId,
    ArtifactCommitPhase* phase)
{
    if (phase) *phase = ArtifactCommitPhase::None;
    if (canceled) {
        *canceled = false;
    }
    if (!artifactId.isValid() || !taskId.isValid() || kind.trimmed().isEmpty() || !storage || !storage->isOpen()) {
        if (error) {
            *error = QStringLiteral("提交 Artifact 的参数无效。");
        }
        return false;
    }
    const QDir staging(stagingPath);
    const QString stagingRoot = QDir(rootPath_).absoluteFilePath(QString::fromLatin1(kStagingDirectoryName));
    const QString normalizedStagingPath = QDir::cleanPath(staging.absolutePath());
    const QString stagingParent = QDir::cleanPath(QFileInfo(normalizedStagingPath).dir().absolutePath());
    if (!staging.exists() || stagingParent != QDir::cleanPath(stagingRoot)
        || QFileInfo(normalizedStagingPath).fileName() != artifactId.toString()) {
        if (error) {
            *error = QStringLiteral("Artifact staging 路径无效。");
        }
        return false;
    }
    CommitJournal journal;
    if (!readCommitJournal(stagingMetadataPath(rootPath_, artifactId), &journal, error)
        || journal.artifactId != artifactId || journal.taskId != taskId
        || journal.kind != kind.trimmed()) {
        if (error && error->isEmpty()) *error = QStringLiteral("Artifact 提交参数与 staging 日志不一致。");
        return false;
    }
    if (phase) *phase = commitPhase(journal.phase);
    if (workflowRunId.isValid()) journal.workflowRunId = workflowRunId;
    if (journal.workflowRunId.isValid() && journal.kind != QStringLiteral("evidence_bundle")) {
        if (error) *error = QStringLiteral("只有 Evidence Artifact 可以关联工作流终态。");
        return false;
    }
    QVector<FileEntry> entries;
    bool collectCanceled = false;
    if (!collectFiles(staging.absolutePath(), &entries, error, cancellation, &collectCanceled)) {
        if (collectCanceled && canceled) {
            *canceled = true;
        }
        return false;
    }
    journal.files = entries;
    journal.completionAction = journal.workflowRunId.isValid()
        ? QStringLiteral("attach_evidence") : QStringLiteral("catalog_only");
    if (!writeManifest(staging.absolutePath(), journal, entries, error)) return false;
    journal.phase = QString::fromLatin1(kPhasePrepared);
    if (!writeStagingMetadata(rootPath_, journal, error)) return false;
    if (phase) *phase = ArtifactCommitPhase::Prepared;
    if (aitrain::isCancellationRequested(cancellation)) {
        if (canceled) *canceled = true;
        if (error) *error = QStringLiteral("Artifact 提交已取消。");
        return false;
    }
    const QString finalPath = QDir(rootPath_).filePath(QStringLiteral("%1/%2")
        .arg(QString::fromLatin1(kCommittedDirectoryName), artifactId.toString()));
    if (QFileInfo::exists(finalPath) || !QDir().mkpath(QFileInfo(finalPath).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Artifact 目标路径不可用：%1").arg(finalPath);
        }
        return false;
    }
    if (!QDir().rename(staging.absolutePath(), finalPath)) {
        if (error) {
            *error = QStringLiteral("Artifact 原子提交失败：%1").arg(finalPath);
        }
        return false;
    }
    journal.phase = QString::fromLatin1(kPhaseFilesCommitted);
    if (!writeStagingMetadata(rootPath_, journal, error)) {
        if (artifactPath) *artifactPath = finalPath;
        return {ArtifactCommitStatus::PendingRecovery};
    }
    if (phase) *phase = ArtifactCommitPhase::FilesCommitted;
    if (failureInjector_
        && failureInjector_(ArtifactCommitFailPoint::AfterDirectoryRenameBeforeDatabase)) {
        if (error) *error = QStringLiteral("故障注入：Artifact 目录已提交但数据库尚未登记。");
        if (artifactPath) *artifactPath = finalPath;
        return {ArtifactCommitStatus::PendingRecovery};
    }
    if (!persistArtifactRecord(storage, journal, entries, error)) {
        if (artifactPath) *artifactPath = finalPath;
        return {ArtifactCommitStatus::PendingRecovery};
    }
    journal.phase = QString::fromLatin1(kPhaseCatalogCommitted);
    QString journalError;
    writeStagingMetadata(rootPath_, journal, &journalError);
    if (phase) *phase = ArtifactCommitPhase::CatalogCommitted;
    if (failureInjector_
        && failureInjector_(ArtifactCommitFailPoint::AfterDatabaseBeforeJournalRemoval)) {
        if (error) *error = QStringLiteral("故障注入：Artifact 数据库已登记但提交日志尚未清理。");
        if (artifactPath) *artifactPath = finalPath;
        return {ArtifactCommitStatus::Committed, true};
    }
    if (!QFile::remove(stagingMetadataPath(rootPath_, artifactId))
        && QFileInfo::exists(stagingMetadataPath(rootPath_, artifactId))) {
        if (error) *error = QStringLiteral("Artifact 已提交，但无法清理提交日志：%1")
            .arg(stagingMetadataPath(rootPath_, artifactId));
        if (artifactPath) *artifactPath = finalPath;
        return {ArtifactCommitStatus::Committed, true};
    }
    if (artifactPath) {
        *artifactPath = finalPath;
    }
    return true;
}

bool ArtifactStore::abort(const QString& stagingPath, QString* error)
{
    const QString stagingRoot = QDir(rootPath_).absoluteFilePath(QString::fromLatin1(kStagingDirectoryName));
    const QString normalizedPath = QDir::cleanPath(QDir(stagingPath).absolutePath());
    const QString stagingParent = QDir::cleanPath(QFileInfo(normalizedPath).dir().absolutePath());
    if (stagingParent != QDir::cleanPath(stagingRoot)) {
        if (error) {
            *error = QStringLiteral("无法清理 Artifact staging：%1").arg(stagingPath);
        }
        return false;
    }
    ArtifactId artifactId;
    if (!ArtifactId::parse(QFileInfo(normalizedPath).fileName(), &artifactId, error)) {
        return false;
    }
    const QString metadataPath = stagingMetadataPath(rootPath_, artifactId);
    const QString finalPath = QDir(rootPath_).filePath(QStringLiteral("%1/%2")
        .arg(QString::fromLatin1(kCommittedDirectoryName), artifactId.toString()));
    // rename 与 phase journal 更新之间也存在崩溃窗口。只要 committed 目录
    // 已出现，abort 就不能再依据旧 phase 删除日志。
    if (QFileInfo::exists(finalPath)) {
        if (error) *error = QStringLiteral("Artifact 文件已进入 committed，必须保留日志交由恢复流程处理：%1")
            .arg(artifactId.toString());
        return false;
    }
    if (QFileInfo::exists(metadataPath)) {
        CommitJournal journal;
        if (!readCommitJournal(metadataPath, &journal, error)) return false;
        if (journal.artifactId != artifactId
            || journal.phase == QLatin1String(kPhaseFilesCommitted)
            || journal.phase == QLatin1String(kPhaseCatalogCommitted)) {
            if (error) *error = QStringLiteral("Artifact 已进入文件提交阶段，必须保留日志交由恢复流程处理：%1")
                .arg(artifactId.toString());
            return false;
        }
    }
    const bool stagingExists = QDir(normalizedPath).exists();
    if (stagingExists && !QDir(normalizedPath).removeRecursively()) {
        if (error) *error = QStringLiteral("无法清理 Artifact staging：%1").arg(stagingPath);
        return false;
    }
    if (QFileInfo::exists(metadataPath) && !QFile::remove(metadataPath)) {
        if (error) *error = QStringLiteral("无法清理 Artifact staging 元数据：%1").arg(metadataPath);
        return false;
    }
    return true;
}

ArtifactDiscardResult ArtifactStore::discardCommitted(
    const ArtifactId& artifactId, ProjectStore* storage, QString* error)
{
    if (!artifactId.isValid() || !storage || !storage->isOpen()) {
        if (error) *error = QStringLiteral("清理已提交 Artifact 需要有效 ID 和已打开存储。");
        return {ArtifactDiscardStatus::Failed};
    }
    bool discardable = false;
    if (!storage->artifactDiscardable(artifactId, &discardable, error)) {
        return {ArtifactDiscardStatus::Failed};
    }
    if (!discardable) {
        if (error) *error = QStringLiteral("Artifact 已被引用，不能删除。");
        return {ArtifactDiscardStatus::NotDiscardable};
    }
    const QString finalPath = QDir(rootPath_).filePath(QStringLiteral("%1/%2")
        .arg(QString::fromLatin1(kCommittedDirectoryName), artifactId.toString()));
    const QString trashPath = QDir(rootPath_).filePath(QStringLiteral("%1/%2")
        .arg(QString::fromLatin1(kTrashDirectoryName), artifactId.toString()));
    if (!QDir(finalPath).exists() || QFileInfo::exists(trashPath)
        || !QDir().mkpath(QFileInfo(trashPath).absolutePath())) {
        if (error) *error = QStringLiteral("Artifact discard 文件状态无效：%1").arg(artifactId.toString());
        return {ArtifactDiscardStatus::Failed};
    }
    if (!QDir().rename(finalPath, trashPath)) {
        if (error) *error = QStringLiteral("无法把 Artifact 原子移动到 trash：%1").arg(finalPath);
        return {ArtifactDiscardStatus::Failed};
    }
    if (!storage->removeUnreferencedArtifact(artifactId, error)) {
        if (!QDir().rename(trashPath, finalPath) && error) {
            *error += QStringLiteral("；同时无法从 trash 恢复 committed 目录。");
        }
        return {ArtifactDiscardStatus::NotDiscardable};
    }
    if (!QDir(trashPath).removeRecursively()) {
        if (error) *error = QStringLiteral("Artifact 已从目录删除，但 trash 清理待恢复：%1").arg(trashPath);
        return {ArtifactDiscardStatus::CleanupPending};
    }
    return {ArtifactDiscardStatus::Discarded};
}

bool ArtifactStore::recoverStaging(ProjectStore* storage, QStringList* diagnostics, QString* error)
{
    if (!storage || !storage->isOpen()) {
        if (error) {
            *error = QStringLiteral("恢复 Artifact staging 需要已打开的  存储。");
        }
        return false;
    }
    const QDir trashRoot(QDir(rootPath_).filePath(QString::fromLatin1(kTrashDirectoryName)));
    const QDir committedRoot(QDir(rootPath_).filePath(QString::fromLatin1(kCommittedDirectoryName)));
    if (trashRoot.exists()) {
        const QFileInfoList trashDirectories =
            trashRoot.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        for (const QFileInfo& trashInfo : trashDirectories) {
            ArtifactId artifactId;
            if (!ArtifactId::parse(trashInfo.fileName(), &artifactId)) {
                appendDiagnostic(diagnostics,
                    QStringLiteral("保留未知名称的 Artifact trash：%1").arg(trashInfo.absoluteFilePath()));
                continue;
            }
            const QString finalPath = committedRoot.filePath(artifactId.toString());
            if (QDir(finalPath).exists()) {
                if (error) *error = QStringLiteral("Artifact committed 与 trash 同时存在：%1")
                    .arg(artifactId.toString());
                return false;
            }
            bool databaseExists = false;
            if (!storage->artifactExists(artifactId, &databaseExists, error)) return false;
            if (databaseExists) {
                if (!QDir().rename(trashInfo.absoluteFilePath(), finalPath)) {
                    if (error) *error = QStringLiteral("无法从 trash 恢复已登记 Artifact：%1")
                        .arg(artifactId.toString());
                    return false;
                }
                appendDiagnostic(diagnostics,
                    QStringLiteral("已从 trash 恢复数据库仍引用的 Artifact：%1").arg(finalPath));
            } else if (!QDir(trashInfo.absoluteFilePath()).removeRecursively()) {
                appendDiagnostic(diagnostics,
                    QStringLiteral("Artifact 数据库记录已删除，但 trash 清理仍待完成：%1")
                        .arg(trashInfo.absoluteFilePath()));
            } else {
                appendDiagnostic(diagnostics,
                    QStringLiteral("已清理无数据库记录的 Artifact trash：%1")
                        .arg(trashInfo.absoluteFilePath()));
            }
        }
    }
    const QDir stagingRoot(QDir(rootPath_).filePath(QString::fromLatin1(kStagingDirectoryName)));
    const QDir finalRoot(QDir(rootPath_).filePath(QString::fromLatin1(kCommittedDirectoryName)));
    const QDir metadataRoot(QDir(rootPath_).filePath(QString::fromLatin1(kStagingMetadataDirectoryName)));
    if (metadataRoot.exists()) {
        const QFileInfoList metadataFiles = metadataRoot.entryInfoList(QStringList() << QStringLiteral("*.json"), QDir::Files, QDir::Name);
        for (const QFileInfo& metadataInfo : metadataFiles) {
            ArtifactId artifactId;
            if (!ArtifactId::parse(metadataInfo.completeBaseName(), &artifactId)) {
                appendDiagnostic(diagnostics, QStringLiteral("保留未知名称的 Artifact staging 元数据：%1").arg(metadataInfo.absoluteFilePath()));
                continue;
            }
            CommitJournal journal;
            QString journalError;
            if (!readCommitJournal(metadataInfo.absoluteFilePath(), &journal, &journalError)
                || journal.artifactId != artifactId) {
                appendDiagnostic(diagnostics, QStringLiteral("保留损坏的 Artifact 提交日志：%1（%2）")
                    .arg(metadataInfo.absoluteFilePath(), journalError));
                continue;
            }
            const QString stagingPath = stagingRoot.filePath(artifactId.toString());
            const QString finalPath = finalRoot.filePath(artifactId.toString());
            const bool stagingExists = QDir(stagingPath).exists();
            const bool finalExists = QDir(finalPath).exists();
            bool databaseExists = false;
            if (!storage->artifactExists(artifactId, &databaseExists, error)) return false;

            if (databaseExists && !finalExists) {
                if (error) *error = QStringLiteral("Artifact 数据库记录存在但最终目录缺失：%1").arg(finalPath);
                appendDiagnostic(diagnostics, *error);
                return false;
            }
            if (stagingExists && finalExists) {
                if (error) *error = QStringLiteral("Artifact staging 与最终目录同时存在，无法安全恢复：%1").arg(artifactId.toString());
                appendDiagnostic(diagnostics, *error);
                return false;
            }
            if (finalExists) {
                QVector<FileEntry> entries;
                if (!readAndVerifyManifest(finalPath, journal, &entries, error)) return false;
                if (journal.workflowRunId.isValid()) {
                    if (!persistArtifactRecord(storage, journal, entries, error)) return false;
                } else if (databaseExists) {
                    if (!verifyStoredArtifact(storage, journal, entries, error)) return false;
                } else if (!persistArtifactRecord(storage, journal, entries, error)) {
                    return false;
                }
                if (!QFile::remove(metadataInfo.absoluteFilePath())
                    && QFileInfo::exists(metadataInfo.absoluteFilePath())) {
                    if (error) *error = QStringLiteral("已恢复 Artifact，但无法清理提交日志：%1")
                        .arg(metadataInfo.absoluteFilePath());
                    return false;
                }
                appendDiagnostic(diagnostics, databaseExists
                    ? QStringLiteral("已核对 Artifact 并清理遗留提交日志：%1").arg(finalPath)
                    : QStringLiteral("已从提交日志恢复 Artifact 数据库关联：%1").arg(finalPath));
                continue;
            }
            if (!stagingExists) {
                if (!QFile::remove(metadataInfo.absoluteFilePath())) {
                    appendDiagnostic(diagnostics, QStringLiteral("无法清理孤儿 Artifact 提交日志：%1").arg(metadataInfo.absoluteFilePath()));
                } else {
                    appendDiagnostic(diagnostics, QStringLiteral("已清理孤儿 Artifact 提交日志：%1").arg(metadataInfo.absoluteFilePath()));
                }
                continue;
            }

            bool taskExists = false;
            if (!storage->taskExists(journal.taskId, &taskExists, error)) return false;
            bool shouldRemove = !taskExists;
            QString removalReason = QStringLiteral("关联任务不存在");
            if (taskExists) {
                TaskSnapshot task;
                if (!storage->task(journal.taskId, &task, error)) return false;
                shouldRemove = task.state == TaskState::Failed || task.state == TaskState::Canceled;
                removalReason = QStringLiteral("关联任务已终态失败或取消");
                if (!shouldRemove) {
                    appendDiagnostic(diagnostics, QStringLiteral("保留 Artifact staging：%1（任务状态：%2）")
                        .arg(stagingPath, taskStateToString(task.state)));
                    continue;
                }
            }
            if (!QDir(stagingPath).removeRecursively()) {
                appendDiagnostic(diagnostics, QStringLiteral("无法清理 Artifact staging：%1").arg(stagingPath));
                continue;
            }
            QFile::remove(metadataInfo.absoluteFilePath());
            appendDiagnostic(diagnostics, QStringLiteral("已清理 Artifact staging：%1（%2）")
                .arg(stagingPath, removalReason));
        }
    }

    if (stagingRoot.exists()) {
        const QFileInfoList stagingDirectories = stagingRoot.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        for (const QFileInfo& stagingInfo : stagingDirectories) {
            ArtifactId artifactId;
            if (!ArtifactId::parse(stagingInfo.fileName(), &artifactId)
                || !QFileInfo::exists(stagingMetadataPath(rootPath_, artifactId))) {
                appendDiagnostic(diagnostics, QStringLiteral("保留缺少提交日志的 Artifact staging：%1")
                    .arg(stagingInfo.absoluteFilePath()));
            }
        }
    }
    return true;
}

QString ArtifactStore::rootPath() const
{
    return rootPath_;
}

bool ArtifactStore::openVerified(const ArtifactSnapshot& artifact,
    VerifiedArtifactDirectory* result, ArtifactReadError* readError,
    QString* error) const
{
    return VerifiedArtifactReader(artifactPath(artifact.id))
        .verifyInventory(artifact, result, readError, error);
}

QString ArtifactStore::artifactPath(const ArtifactId& artifactId) const
{
    if (!artifactId.isValid()) return {};
    return QDir(rootPath_).filePath(QStringLiteral("%1/%2")
        .arg(QString::fromLatin1(kCommittedDirectoryName), artifactId.toString()));
}

} // namespace aitrain
