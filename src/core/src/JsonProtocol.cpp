#include "aitrain/core/JsonProtocol.h"

#include <QJsonDocument>
#include <QJsonParseError>

namespace aitrain::protocol {

QByteArray encodeMessage(const QString& type, const QJsonObject& payload, const QString& requestId)
{
    QJsonObject message;
    message.insert(QString::fromLatin1(kType), type);
    message.insert(QString::fromLatin1(kPayload), payload);
    if (!requestId.isEmpty()) {
        message.insert(QString::fromLatin1(kRequestId), requestId);
    }

    QByteArray encoded = QJsonDocument(message).toJson(QJsonDocument::Compact);
    encoded.append('\n');
    return encoded;
}

bool decodeMessage(const QByteArray& line, QString* type, QJsonObject* payload, QString* requestId, QString* error)
{
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(line.trimmed(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = parseError.errorString();
        }
        return false;
    }

    const QJsonObject object = document.object();
    const QString messageType = object.value(QString::fromLatin1(kType)).toString();
    if (messageType.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Message has no type");
        }
        return false;
    }

    if (type) {
        *type = messageType;
    }
    if (payload) {
        *payload = object.value(QString::fromLatin1(kPayload)).toObject();
    }
    if (requestId) {
        *requestId = object.value(QString::fromLatin1(kRequestId)).toString();
    }
    if (error) {
        error->clear();
    }
    return true;
}

QJsonObject errorPayload(const QString& message, const QString& code)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("message"), message);
    if (!code.isEmpty()) {
        payload.insert(QStringLiteral("code"), code);
    }
    return payload;
}

} // namespace aitrain::protocol

