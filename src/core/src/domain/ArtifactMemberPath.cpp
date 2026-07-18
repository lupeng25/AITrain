#include "aitrain/domain/ArtifactMemberPath.h"

#include <QDir>
#include <QRegularExpression>

namespace aitrain {

bool normalizeArtifactMemberPath(const QString& value, QString* normalized, QString* error)
{
    const QString portable = QDir::fromNativeSeparators(value.trimmed());
    const QString clean = QDir::cleanPath(portable);
    static const QRegularExpression driveRelative(QStringLiteral("^[A-Za-z]:"));
    if (!normalized || portable.isEmpty() || clean.isEmpty() || clean == QStringLiteral(".")
        || QDir::isAbsolutePath(clean) || driveRelative.match(clean).hasMatch()
        || clean == QStringLiteral("..") || clean.startsWith(QStringLiteral("../"))
        || clean.contains(QStringLiteral("/../"))) {
        if (error) *error = QStringLiteral("Artifact 成员必须是安全的规范相对路径。");
        return false;
    }
    *normalized = clean;
    return true;
}

} // namespace aitrain
