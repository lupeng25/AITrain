#include "aitrain/core/DatasetValidation.h"

#include <QJsonArray>
#include <QJsonDocument>

namespace aitrain {

QJsonObject DatasetValidationResult::Issue::toJson() const
{
    return QJsonObject{
        {QStringLiteral("severity"), severity},
        {QStringLiteral("code"), code},
        {QStringLiteral("filePath"), filePath},
        {QStringLiteral("line"), line},
        {QStringLiteral("message"), message}};
}

QJsonObject DatasetValidationResult::toJson() const
{
    const QString metadataWarningPrefix = QStringLiteral("__aitrain_validation_metadata__=");
    QJsonObject metadata;
    QStringList visibleWarnings;
    for (const QString& warning : warnings) {
        if (warning.startsWith(metadataWarningPrefix)) {
            const QJsonDocument document = QJsonDocument::fromJson(warning.mid(metadataWarningPrefix.size()).toUtf8());
            if (document.isObject()) {
                metadata = document.object();
            }
        } else {
            visibleWarnings.append(warning);
        }
    }
    QJsonArray issueArray;
    for (const Issue& issue : issues) {
        issueArray.append(issue.toJson());
    }
    QJsonObject result{
        {QStringLiteral("ok"), ok},
        {QStringLiteral("sampleCount"), sampleCount},
        {QStringLiteral("errors"), QJsonArray::fromStringList(errors)},
        {QStringLiteral("warnings"), QJsonArray::fromStringList(visibleWarnings)},
        {QStringLiteral("previewSamples"), QJsonArray::fromStringList(previewSamples)},
        {QStringLiteral("issues"), issueArray}};
    if (!metadata.isEmpty()) {
        result.insert(QStringLiteral("metadata"), metadata);
    }
    return result;
}

} // namespace aitrain
