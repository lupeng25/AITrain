#include "aitrain/v2/ProtocolV2.h"

#include <QJsonDocument>
#include <QJsonParseError>
#include <QRegularExpression>

#include <cmath>
#include <limits>

namespace aitrain::v2 {
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
    const double numericValue = value.toDouble(-1.0);
    if (!value.isDouble()
        || !std::isfinite(numericValue)
        || numericValue < 1.0
        || std::floor(numericValue) != numericValue
        || numericValue > static_cast<double>(std::numeric_limits<qint64>::max())) {
        if (error) {
            *error = QStringLiteral("Protocol V2 sequence must be a positive integer.");
        }
        return false;
    }
    if (sequence) {
        *sequence = static_cast<quint64>(numericValue);
    }
    return true;
}

} // namespace

bool isKnownProtocolV2Kind(const QString& kind)
{
    return knownKinds().contains(kind);
}

bool isProtocolV2LogKind(const QString& kind)
{
    return kind == QStringLiteral("event.log");
}

bool validateProtocolV2Envelope(const ProtocolEnvelope& envelope, QString* error)
{
    if (!envelope.messageId.isValid() || !envelope.requestId.isValid() || !envelope.taskId.isValid()) {
        if (error) {
            *error = QStringLiteral("Protocol V2 envelope requires valid messageId, requestId, and taskId.");
        }
        return false;
    }
    if (envelope.sequence == 0) {
        if (error) {
            *error = QStringLiteral("Protocol V2 envelope sequence must be positive.");
        }
        return false;
    }
    if (!isKnownProtocolV2Kind(envelope.kind)) {
        if (error) {
            *error = QStringLiteral("Unsupported Protocol V2 kind: %1").arg(envelope.kind);
        }
        return false;
    }
    if (!envelope.timestamp.isValid()) {
        if (error) {
            *error = QStringLiteral("Protocol V2 envelope timestamp is invalid.");
        }
        return false;
    }
    return true;
}

QByteArray encodeProtocolV2Message(const ProtocolEnvelope& envelope, QString* error)
{
    if (!validateProtocolV2Envelope(envelope, error)) {
        return {};
    }

    QJsonObject object;
    object.insert(QStringLiteral("protocol"), kProtocolV2Version);
    object.insert(QStringLiteral("messageId"), envelope.messageId.toString());
    object.insert(QStringLiteral("requestId"), envelope.requestId.toString());
    object.insert(QStringLiteral("taskId"), envelope.taskId.toString());
    object.insert(QStringLiteral("sequence"), static_cast<qint64>(envelope.sequence));
    object.insert(QStringLiteral("kind"), envelope.kind);
    object.insert(QStringLiteral("timestamp"), envelope.timestamp.toUTC().toString(Qt::ISODateWithMs));
    object.insert(QStringLiteral("payload"), envelope.payload);
    QByteArray bytes = QJsonDocument(object).toJson(QJsonDocument::Compact);
    bytes.append('\n');

    const qsizetype maxBytes = isProtocolV2LogKind(envelope.kind)
        ? kProtocolV2MaxLogMessageBytes
        : kProtocolV2MaxControlMessageBytes;
    if (bytes.size() > maxBytes) {
        if (error) {
            *error = QStringLiteral("Protocol V2 message exceeds maximum size for kind %1.").arg(envelope.kind);
        }
        return {};
    }
    return bytes;
}

bool decodeProtocolV2Message(const QByteArray& bytes, ProtocolEnvelope* envelope, QString* error)
{
    const QByteArray line = bytes.endsWith('\n') ? bytes.left(bytes.size() - 1) : bytes;
    if (line.isEmpty() || line.size() > kProtocolV2MaxControlMessageBytes) {
        if (error) {
            *error = QStringLiteral("Protocol V2 message size is invalid.");
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Protocol V2 JSON parse failed: %1").arg(parseError.errorString());
        }
        return false;
    }

    const QJsonObject object = document.object();
    if (object.value(QStringLiteral("protocol")).toInt(-1) != kProtocolV2Version) {
        if (error) {
            *error = QStringLiteral("Unsupported Protocol V2 version.");
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
    parsed.kind = object.value(QStringLiteral("kind")).toString();
    parsed.timestamp = QDateTime::fromString(object.value(QStringLiteral("timestamp")).toString(), Qt::ISODateWithMs);
    parsed.payload = object.value(QStringLiteral("payload")).toObject();
    if (!object.value(QStringLiteral("payload")).isObject() || !validateProtocolV2Envelope(parsed, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("Protocol V2 payload must be an object.");
        }
        return false;
    }

    const qsizetype maxBytes = isProtocolV2LogKind(parsed.kind)
        ? kProtocolV2MaxLogMessageBytes
        : kProtocolV2MaxControlMessageBytes;
    if (bytes.size() > maxBytes) {
        if (error) {
            *error = QStringLiteral("Protocol V2 message exceeds maximum size for kind %1.").arg(parsed.kind);
        }
        return false;
    }
    if (envelope) {
        *envelope = parsed;
    }
    return true;
}

bool ProtocolV2SequenceTracker::observe(const ProtocolEnvelope& envelope,
    const RequestId& expectedRequestId,
    const TaskId& expectedTaskId,
    QString* error)
{
    if (!validateProtocolV2Envelope(envelope, error)) {
        return false;
    }
    if (envelope.requestId != expectedRequestId || envelope.taskId != expectedTaskId) {
        if (error) {
            *error = QStringLiteral("Protocol V2 message does not belong to the active request/task.");
        }
        return false;
    }
    const QString messageKey = envelope.messageId.toString();
    if (observedMessageIds_.contains(messageKey)) {
        if (error) {
            *error = QStringLiteral("Protocol V2 duplicate messageId: %1").arg(messageKey);
        }
        return false;
    }
    const QString requestKey = envelope.requestId.toString();
    const quint64 lastSequence = lastSequenceByRequest_.value(requestKey, 0);
    if (envelope.sequence <= lastSequence) {
        if (error) {
            *error = QStringLiteral("Protocol V2 sequence is not strictly increasing.");
        }
        return false;
    }
    observedMessageIds_.insert(messageKey);
    lastSequenceByRequest_.insert(requestKey, envelope.sequence);
    return true;
}

void ProtocolV2SequenceTracker::reset(const RequestId& requestId)
{
    lastSequenceByRequest_.remove(requestId.toString());
}

void ProtocolV2SequenceTracker::clear()
{
    lastSequenceByRequest_.clear();
    observedMessageIds_.clear();
}

} // namespace aitrain::v2
