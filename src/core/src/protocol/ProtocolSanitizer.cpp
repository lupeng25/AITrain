#include "aitrain/protocol/ProtocolSanitizer.h"

#include <QJsonArray>
#include <QRegularExpression>

namespace aitrain {
namespace protocol {
namespace {

bool isRelativeArtifactPathKey(const QString& key)
{
    const QString normalized = key.trimmed().toLower().remove(QLatin1Char('_')).remove(QLatin1Char('-'));
    return normalized == QStringLiteral("relativepath")
        || normalized.endsWith(QStringLiteral("relativepath"))
        || normalized == QStringLiteral("relativepaths")
        || normalized.endsWith(QStringLiteral("relativepaths"));
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
        || normalized.endsWith(QStringLiteral("root"))
        || normalized == QStringLiteral("paths")
        || normalized.endsWith(QStringLiteral("paths"))
        || normalized == QStringLiteral("directories")
        || normalized.endsWith(QStringLiteral("directories"))
        || normalized == QStringLiteral("filepaths")
        || normalized.endsWith(QStringLiteral("filepaths"));
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
    if (value.isString()) {
        // 字段名黑名单无法覆盖 message/details/traceback 等自由文本。跨进程
        // 协议在落盘和 GUI 转发前统一遮蔽 Windows 盘符路径及 UNC 路径；
        // Artifact 的相对成员仍由字段名规则保留。
        static const QRegularExpression absolutePath(
            // 允许路径中包含空格；为避免把路径前缀截断，匹配到消息/JSON
            // 分隔符或字符串结尾。该字段属于内部诊断文本，过度遮蔽优先于
            // 把部分物理路径暴露给 Worker/GUI。
            QStringLiteral("(?i)(?:[A-Z]:[\\\\/][^\\\"'<>|\\r\\n]*|\\\\\\\\[^\\\"'<>|\\r\\n]*)"));
        QString redacted = value.toString();
        redacted.replace(absolutePath, QStringLiteral("<physical-path>"));
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
