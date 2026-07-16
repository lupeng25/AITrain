#include "aitrain/v2/AnomalyFolderDatasetDriverV2.h"

#include <QDir>

namespace aitrain::v2 {
AnomalyFolderDatasetDriverV2::AnomalyFolderDatasetDriverV2()
    : ValidatedDatasetDriverV2(QStringLiteral("anomaly_folder"), QStringLiteral("anomaly_folder"),
        [](const QString& path, const QJsonObject& options) { return validateAnomalyFolderDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitAnomalyFolderDataset(source, output, options); },
        [](const QString& path) { return QDir(QDir(path).filePath(QStringLiteral("train/good"))).exists(); })
{
}
} // namespace aitrain::v2
