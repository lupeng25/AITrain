#pragma once

#include "aitrain/domain/Pagination.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QSqlQuery>
#include <QVariant>
#include <QString>

namespace aitrain::storage_internal {

constexpr int kPageCursorVersion = 1;

struct PageCursor final {
    QString timestamp;
    QString id;
};

inline QString catalogQueryType(const QString& type, const CatalogFilter& filter)
{
    if (filter.text.trimmed().isEmpty() && filter.kinds.isEmpty() && filter.state.isEmpty()) return type;
    auto kinds = filter.kinds; kinds.removeDuplicates(); kinds.sort();
    return type + QLatin1Char('/') + QString::fromUtf8(QJsonDocument(QJsonObject{
        {QStringLiteral("text"), filter.text.trimmed()},
        {QStringLiteral("kinds"), QJsonArray::fromStringList(kinds)},
        {QStringLiteral("state"), filter.state}}).toJson(QJsonDocument::Compact));
}

inline QString catalogKindClause(const CatalogFilter& filter, const QString& column)
{
    if (filter.kinds.isEmpty()) return {};
    QStringList parameters;
    for (int i = 0; i < filter.kinds.size(); ++i) parameters.append(QStringLiteral(":kind_%1").arg(i));
    return QStringLiteral(" and %1 in (%2) ").arg(column, parameters.join(QLatin1Char(',')));
}

inline void bindCatalogFilter(QSqlQuery& query, const CatalogFilter& filter)
{
    if (!filter.text.trimmed().isEmpty()) {
        query.bindValue(QStringLiteral(":search"), filter.text.trimmed().toLower());
    }
    for (int i = 0; i < filter.kinds.size(); ++i) query.bindValue(QStringLiteral(":kind_%1").arg(i), filter.kinds.at(i));
    if (!filter.state.isEmpty()) query.bindValue(QStringLiteral(":state"), filter.state);
}

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
