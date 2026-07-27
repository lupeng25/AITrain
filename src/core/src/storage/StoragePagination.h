#pragma once

#include "aitrain/domain/Pagination.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

namespace aitrain::storage_internal {

constexpr int kPageCursorVersion = 1;

struct PageCursor final {
    QString timestamp;
    QString id;
};

inline QString encodePageCursor(
    const QString& queryType, const PageCursor& cursor)
{
    const QJsonObject json{
        {QStringLiteral("v"), kPageCursorVersion},
        {QStringLiteral("q"), queryType},
        {QStringLiteral("t"), cursor.timestamp},
        {QStringLiteral("id"), cursor.id}};
    return QString::fromLatin1(
        QJsonDocument(json).toJson(QJsonDocument::Compact)
            .toBase64(QByteArray::Base64UrlEncoding
                | QByteArray::OmitTrailingEquals));
}

inline bool decodePageCursor(const QString& encoded,
    const QString& queryType, PageCursor* result, QString* error)
{
    if (encoded.isEmpty()) {
        if (result) *result = {};
        return true;
    }
    const QByteArray decoded = QByteArray::fromBase64(
        encoded.toLatin1(), QByteArray::Base64UrlEncoding);
    QJsonParseError parseError;
    const QJsonDocument document =
        QJsonDocument::fromJson(decoded, &parseError);
    const QJsonObject json = document.object();
    if (parseError.error != QJsonParseError::NoError
        || !document.isObject()
        || json.value(QStringLiteral("v")).toInt() != kPageCursorVersion
        || json.value(QStringLiteral("q")).toString() != queryType
        || json.value(QStringLiteral("t")).toString().isEmpty()
        || json.value(QStringLiteral("id")).toString().isEmpty()) {
        if (error) {
            *error = QStringLiteral(
                "InvalidPageCursor：分页游标版本、查询类型或排序键无效。");
        }
        return false;
    }
    if (result) {
        result->timestamp = json.value(QStringLiteral("t")).toString();
        result->id = json.value(QStringLiteral("id")).toString();
    }
    return true;
}

inline bool validatePageRequest(const PageRequest& request,
    const QString& queryType, PageCursor* cursor, QString* error)
{
    if (request.pageSize < 1 || request.pageSize > 200) {
        if (error) {
            *error = QStringLiteral(
                "InvalidPageCursor：pageSize 必须在 1–200 之间。");
        }
        return false;
    }
    return decodePageCursor(request.after, queryType, cursor, error);
}

} // namespace aitrain::storage_internal
