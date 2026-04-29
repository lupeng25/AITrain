#pragma once

#include <QByteArray>
#include <QJsonObject>
#include <QString>

namespace aitrain {

namespace protocol {

constexpr const char* kType = "type";
constexpr const char* kPayload = "payload";
constexpr const char* kRequestId = "requestId";

QByteArray encodeMessage(const QString& type, const QJsonObject& payload, const QString& requestId = {});
bool decodeMessage(const QByteArray& line, QString* type, QJsonObject* payload, QString* requestId, QString* error);

QJsonObject errorPayload(const QString& message, const QString& code = {});

} // namespace protocol

} // namespace aitrain

