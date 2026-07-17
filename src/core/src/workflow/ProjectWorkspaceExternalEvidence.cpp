#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/artifact/ArtifactStore.h"

#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonArray>
#include <QSaveFile>
#include <QSet>

namespace aitrain {
namespace {

bool hasOnlyKeys(const QJsonObject& object, const QSet<QString>& allowed, QString* error)
{
    for (const QString& key : object.keys()) {
        if (!allowed.contains(key)) {
            if (error) *error = QStringLiteral("外部验收证据包含未知字段：%1").arg(key);
            return false;
        }
    }
    return true;
}

bool requiredString(const QJsonObject& object, const QString& key, QString* value, QString* error)
{
    const QJsonValue candidate = object.value(key);
    if (!candidate.isString() || candidate.toString().trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("外部验收证据字段 %1 必须是非空字符串。").arg(key);
        return false;
    }
    if (value) *value = candidate.toString().trimmed();
    return true;
}

bool parseEvidence(const QByteArray& bytes,
    QString* kind,
    QString* status,
    QString* producer,
    QDateTime* observedAt,
    QStringList* limitations,
    QJsonObject* summary,
    QString* error)
{
    if (bytes.size() > 1024 * 1024) {
        if (error) *error = QStringLiteral("外部验收证据超过 1 MiB 大小上限。");
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) *error = QStringLiteral("外部验收证据必须是合法 JSON 对象：%1").arg(parseError.errorString());
        return false;
    }
    const QJsonObject object = document.object();
    if (!hasOnlyKeys(object, {
            QStringLiteral("schemaVersion"), QStringLiteral("kind"), QStringLiteral("evidenceKind"),
            QStringLiteral("status"), QStringLiteral("producer"), QStringLiteral("observedAt"),
            QStringLiteral("message"), QStringLiteral("limitations")}, error)
        || object.value(QStringLiteral("schemaVersion")).toInt(-1) != 1
        || object.value(QStringLiteral("kind")).toString() != QStringLiteral("aitrain_external_acceptance_evidence")) {
        if (error && error->isEmpty()) *error = QStringLiteral("外部验收证据 schemaVersion/kind 无效。");
        return false;
    }
    QString observedAtText;
    if (!requiredString(object, QStringLiteral("evidenceKind"), kind, error)
        || !requiredString(object, QStringLiteral("status"), status, error)
        || !requiredString(object, QStringLiteral("producer"), producer, error)
        || !requiredString(object, QStringLiteral("observedAt"), &observedAtText, error)) {
        return false;
    }
    static const QSet<QString> statuses = {
        QStringLiteral("passed"), QStringLiteral("failed"), QStringLiteral("blocked"),
        QStringLiteral("collected"), QStringLiteral("imported")};
    if (!statuses.contains(*status)) {
        if (error) *error = QStringLiteral("外部验收证据 status 不受支持：%1").arg(*status);
        return false;
    }
    const QDateTime parsedAt = QDateTime::fromString(observedAtText, Qt::ISODate);
    if (!parsedAt.isValid()) {
        if (error) *error = QStringLiteral("外部验收证据 observedAt 必须是 ISO-8601 时间。");
        return false;
    }
    if (object.contains(QStringLiteral("message")) && !object.value(QStringLiteral("message")).isString()) {
        if (error) *error = QStringLiteral("外部验收证据 message 必须是字符串。");
        return false;
    }
    QStringList parsedLimitations;
    if (object.contains(QStringLiteral("limitations"))) {
        if (!object.value(QStringLiteral("limitations")).isArray()) {
            if (error) *error = QStringLiteral("外部验收证据 limitations 必须是字符串数组。");
            return false;
        }
        const QJsonArray values = object.value(QStringLiteral("limitations")).toArray();
        if (values.size() > 32) {
            if (error) *error = QStringLiteral("外部验收证据 limitations 数量超过上限。");
            return false;
        }
        for (const QJsonValue& value : values) {
            if (!value.isString() || value.toString().trimmed().isEmpty()) {
                if (error) *error = QStringLiteral("外部验收证据 limitations 必须只包含非空字符串。");
                return false;
            }
            parsedLimitations.append(value.toString().trimmed());
        }
    }
    if (parsedLimitations.isEmpty()) {
        parsedLimitations.append(QStringLiteral("外部证据已导入，但未经过 AITrain 内部生产验收证明；status 不会自动变为 verified。"));
    }
    if (observedAt) *observedAt = parsedAt.toUTC();
    if (limitations) *limitations = parsedLimitations;
    if (summary) {
        *summary = {
            {QStringLiteral("evidenceKind"), *kind},
            {QStringLiteral("status"), *status},
            {QStringLiteral("producer"), *producer},
            {QStringLiteral("observedAt"), parsedAt.toUTC().toString(Qt::ISODateWithMs)},
            {QStringLiteral("message"), object.value(QStringLiteral("message")).toString()},
            {QStringLiteral("limitations"), QJsonArray::fromStringList(parsedLimitations)},
            {QStringLiteral("verified"), false}
        };
    }
    return true;
}

bool writeFile(const QString& path, const QByteArray& bytes, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入外部验收证据 Artifact：%1").arg(file.errorString());
        return false;
    }
    return true;
}

} // namespace

bool ProjectWorkspace::importExternalAcceptanceEvidence(const TaskId& taskId,
    const ExternalAcceptanceEvidenceImportRequest& request,
    ExternalAcceptanceEvidenceImportResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !result || request.sourcePath.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("外部验收证据导入参数无效。");
        return false;
    }
    if (isCancellationRequested(cancellation)) {
        if (error) *error = QStringLiteral("external_acceptance_evidence_canceled");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("仅运行中的任务可以导入外部验收证据。");
        return false;
    }
    const QFileInfo source(request.sourcePath);
    if (!source.exists() || !source.isFile() || source.isSymLink()
        || source.suffix().compare(QStringLiteral("json"), Qt::CaseInsensitive) != 0) {
        if (error) *error = QStringLiteral("外部验收证据只能是普通 JSON 文件。");
        return false;
    }
    QFile file(source.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取外部验收证据：%1").arg(file.errorString());
        return false;
    }
    const QByteArray bytes = file.read(1024 * 1024 + 1);
    QString kind;
    QString status;
    QString producer;
    QDateTime observedAt;
    QStringList limitations;
    QJsonObject summary;
    if (!parseEvidence(bytes, &kind, &status, &producer, &observedAt,
            &limitations, &summary, error)) {
        return false;
    }
    ArtifactId artifactId;
    QString stagingPath;
    if (!artifactStore_->begin(taskId, QStringLiteral("external_acceptance_evidence"),
            &artifactId, &stagingPath, error)) {
        return false;
    }
    const auto abort = [&]() {
        QString ignored;
        artifactStore_->abort(stagingPath, &ignored);
    };
    if (!writeFile(QDir(stagingPath).filePath(QStringLiteral("acceptance.json")), bytes, error)) {
        abort();
        return false;
    }
    QString committedPath;
    if (!artifactStore_->commit(artifactId, taskId, QStringLiteral("external_acceptance_evidence"),
            stagingPath, &storage_, &committedPath, error)) {
        abort();
        return false;
    }
    result->evidenceArtifactId = artifactId;
    result->evidenceKind = kind;
    result->status = status;
    result->producer = producer;
    result->observedAt = observedAt;
    result->limitations = limitations;
    result->summary = summary;
    return true;
}

} // namespace aitrain
