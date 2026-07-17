#include "aitrain/protocol/ProtocolSanitizer.h"

#include <QJsonArray>

namespace aitrain {
namespace protocol {
namespace {

bool isRelativeArtifactPathKey(const QString& key)
{
    const QString normalized = key.trimmed().toLower().remove(QLatin1Char('_')).remove(QLatin1Char('-'));
    return normalized == QStringLiteral("relativepath")
        || normalized.endsWith(QStringLiteral("relativepath"));
}

bool isPhysicalPathKey(const QString& key)
{
    const QString normalized = key.trimmed().toLower().remove(QLatin1Char('_')).remove(QLatin1Char('-'));
    if (isRelativeArtifactPathKey(key)) {
        return false;
    }
    return normalized == QStringLiteral("path")
        || normalized.endsWith(QStringLiteral("path"))
        || normalized == QStringLiteral("dir")
        || normalized.endsWith(QStringLiteral("dir"))
        || normalized == QStringLiteral("directory")
        || normalized.endsWith(QStringLiteral("directory"))
        || normalized == QStringLiteral("root")
        || normalized.endsWith(QStringLiteral("root"));
}

QJsonValue redactValue(const QJsonValue& value)
{
    if (value.isObject()) {
        QJsonObject redacted;
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it) {
            if (isPhysicalPathKey(it.key())) {
                continue;
            }
            redacted.insert(it.key(), redactValue(it.value()));
        }
        return redacted;
    }
    if (value.isArray()) {
        QJsonArray redacted;
        const QJsonArray array = value.toArray();
        for (const QJsonValue& item : array) {
            redacted.append(redactValue(item));
        }
        return redacted;
    }
    return value;
}

} // namespace

QJsonObject redactPhysicalPathFields(const QJsonObject& payload)
{
    return redactValue(payload).toObject();
}

} // namespace protocol
} // namespace aitrain
