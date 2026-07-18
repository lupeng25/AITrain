#include "aitrain/protocol/Protocol.h"

#include <QJsonDocument>
#include <QJsonParseError>
#include <QRegularExpression>

#include <cmath>
#include <limits>

namespace aitrain {
namespace {

const QSet<QString>& knownKinds()
{
    static const QSet<QString> values = {
        QStringLiteral("command.start_task"),
        QStringLiteral("command.cancel_task"),
        QStringLiteral("event.ready"),
        QStringLiteral("event.progress"),
        QStringLiteral("event.metric"),
        QStringLiteral("event.artifact_candidate"),
        QStringLiteral("event.artifact"),
        QStringLiteral("event.result"),
        QStringLiteral("event.log"),
        QStringLiteral("event.succeeded"),
        QStringLiteral("event.failed"),
        QStringLiteral("event.canceled")
    };
    return values;
}

bool readTaskId(const QJsonObject& object, TaskId* result, QString* error)
{
    return TaskId::parse(object.value(QStringLiteral("taskId")).toString(), result, error);
}

bool readRequestId(const QJsonObject& object, RequestId* result, QString* error)
{
    return RequestId::parse(object.value(QStringLiteral("requestId")).toString(), result, error);
}

bool readMessageId(const QJsonObject& object, MessageId* result, QString* error)
{
    return MessageId::parse(object.value(QStringLiteral("messageId")).toString(), result, error);
}

bool readPositiveSequence(const QJsonObject& object, quint64* sequence, QString* error)
{
    const QJsonValue value = object.value(QStringLiteral("sequence"));
    bool ok = false;
    const quint64 numericValue = value.isString()
        ? value.toString().toULongLong(&ok)
        : 0;
    if (!value.isString()
        || !ok
        || numericValue == 0
        || numericValue > static_cast<quint64>(std::numeric_limits<qint64>::max())) {
        if (error) {
            *error = QStringLiteral("Protocol sequence must be a positive decimal string.");
        }
        return false;
    }
    if (sequence) {
        *sequence = static_cast<quint64>(numericValue);
    }
    return true;
}

} // namespace

bool isKnownProtocolKind(const QString& kind)
{
    return knownKinds().contains(kind);
}

bool isProtocolLogKind(const QString& kind)
{
    return kind == QStringLiteral("event.log");
}

bool validateProtocolEnvelope(const ProtocolEnvelope& envelope, QString* error)
{
    if (!envelope.messageId.isValid() || !envelope.requestId.isValid() || !envelope.taskId.isValid()) {
        if (error) {
            *error = QStringLiteral("Protocol envelope requires valid messageId, requestId, and taskId.");
        }
        return false;
    }
    if (envelope.sequence == 0) {
        if (error) {
            *error = QStringLiteral("Protocol envelope sequence must be positive.");
        }
        return false;
    }
    if (!isKnownProtocolKind(envelope.kind)) {
        if (error) {
            *error = QStringLiteral("Unsupported Protocol kind: %1").arg(envelope.kind);
        }
        return false;
    }
    if (!envelope.timestamp.isValid()) {
        if (error) {
            *error = QStringLiteral("Protocol envelope timestamp is invalid.");
        }
        return false;
    }
    return true;
}

QByteArray encodeProtocolMessage(const ProtocolEnvelope& envelope, QString* error)
{
    if (!validateProtocolEnvelope(envelope, error)) {
        return {};
    }

    QJsonObject object;
    object.insert(QStringLiteral("protocol"), kProtocolVersion);
    object.insert(QStringLiteral("messageId"), envelope.messageId.toString());
    object.insert(QStringLiteral("requestId"), envelope.requestId.toString());
    object.insert(QStringLiteral("taskId"), envelope.taskId.toString());
    if (!envelope.controlToken.isEmpty()) {
        object.insert(QStringLiteral("controlToken"), envelope.controlToken);
    }
    object.insert(QStringLiteral("sequence"), QString::number(envelope.sequence));
    object.insert(QStringLiteral("kind"), envelope.kind);
    object.insert(QStringLiteral("timestamp"), envelope.timestamp.toUTC().toString(Qt::ISODateWithMs));
    object.insert(QStringLiteral("payload"), envelope.payload);
    QByteArray bytes = QJsonDocument(object).toJson(QJsonDocument::Compact);
    bytes.append('\n');

    const qsizetype maxBytes = isProtocolLogKind(envelope.kind)
        ? kProtocolMaxLogMessageBytes
        : kProtocolMaxControlMessageBytes;
    if (bytes.size() > maxBytes) {
        if (error) {
            *error = QStringLiteral("Protocol message exceeds maximum size for kind %1.").arg(envelope.kind);
        }
        return {};
    }
    return bytes;
}

bool decodeProtocolMessage(const QByteArray& bytes, ProtocolEnvelope* envelope, QString* error)
{
    const QByteArray line = bytes.endsWith('\n') ? bytes.left(bytes.size() - 1) : bytes;
    if (line.isEmpty() || line.size() > kProtocolMaxControlMessageBytes) {
        if (error) {
            *error = QStringLiteral("Protocol message size is invalid.");
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Protocol JSON parse failed: %1").arg(parseError.errorString());
        }
        return false;
    }

    const QJsonObject object = document.object();
    if (object.value(QStringLiteral("protocol")).toInt(-1) != kProtocolVersion) {
        if (error) {
            *error = QStringLiteral("Unsupported Protocol version.");
        }
        return false;
    }

    ProtocolEnvelope parsed;
    if (!readMessageId(object, &parsed.messageId, error)
        || !readRequestId(object, &parsed.requestId, error)
        || !readTaskId(object, &parsed.taskId, error)
        || !readPositiveSequence(object, &parsed.sequence, error)) {
        return false;
    }
    if (object.contains(QStringLiteral("controlToken"))
        && !object.value(QStringLiteral("controlToken")).isString()) {
        if (error) *error = QStringLiteral("Protocol controlToken must be a string when present.");
        return false;
    }
    parsed.controlToken = object.value(QStringLiteral("controlToken")).toString();
    parsed.kind = object.value(QStringLiteral("kind")).toString();
    parsed.timestamp = QDateTime::fromString(object.value(QStringLiteral("timestamp")).toString(), Qt::ISODateWithMs);
    parsed.payload = object.value(QStringLiteral("payload")).toObject();
    if (!object.value(QStringLiteral("payload")).isObject() || !validateProtocolEnvelope(parsed, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("Protocol payload must be an object.");
        }
        return false;
    }

    const qsizetype maxBytes = isProtocolLogKind(parsed.kind)
        ? kProtocolMaxLogMessageBytes
        : kProtocolMaxControlMessageBytes;
    if (bytes.size() > maxBytes) {
        if (error) {
            *error = QStringLiteral("Protocol message exceeds maximum size for kind %1.").arg(parsed.kind);
        }
        return false;
    }
    if (envelope) {
        *envelope = parsed;
    }
    return true;
}

bool ProtocolSequenceTracker::observe(const ProtocolEnvelope& envelope,
    const RequestId& expectedRequestId,
    const TaskId& expectedTaskId,
    QString* error)
{
    if (!validateProtocolEnvelope(envelope, error)) {
        return false;
    }
    if (envelope.requestId != expectedRequestId || envelope.taskId != expectedTaskId) {
        if (error) {
            *error = QStringLiteral("Protocol message does not belong to the active request/task.");
        }
        return false;
    }
    const QString messageKey = envelope.messageId.toString();
    if (observedMessageIds_.contains(messageKey)) {
        if (error) {
            *error = QStringLiteral("Protocol duplicate messageId: %1").arg(messageKey);
        }
        return false;
    }
    const QString requestKey = envelope.requestId.toString();
    const quint64 lastSequence = lastSequenceByRequest_.value(requestKey, 0);
    if (envelope.sequence <= lastSequence) {
        if (error) {
            *error = QStringLiteral("Protocol sequence is not strictly increasing.");
        }
        return false;
    }
    constexpr qsizetype kRetainedMessageIds = 4096;
    observedMessageIds_.insert(messageKey);
    observedMessageOrder_.enqueue(messageKey);
    while (observedMessageOrder_.size() > kRetainedMessageIds) {
        observedMessageIds_.remove(observedMessageOrder_.dequeue());
    }
    lastSequenceByRequest_.insert(requestKey, envelope.sequence);
    return true;
}

void ProtocolSequenceTracker::reset(const RequestId& requestId)
{
    lastSequenceByRequest_.remove(requestId.toString());
}

void ProtocolSequenceTracker::clear()
{
    lastSequenceByRequest_.clear();
    observedMessageIds_.clear();
    observedMessageOrder_.clear();
}

} // namespace aitrain
