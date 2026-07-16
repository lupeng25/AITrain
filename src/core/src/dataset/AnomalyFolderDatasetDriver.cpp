#include "aitrain/dataset/AnomalyFolderDatasetDriver.h"

#include <QDir>

namespace aitrain {
AnomalyFolderDatasetDriver::AnomalyFolderDatasetDriver()
    : ValidatedDatasetDriver(QStringLiteral("anomaly_folder"), QStringLiteral("anomaly_folder"),
        [](const QString& path, const QJsonObject& options) { return validateAnomalyFolderDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitAnomalyFolderDataset(source, output, options); },
        [](const QString& path) { return QDir(QDir(path).filePath(QStringLiteral("train/good"))).exists(); })
{
}
} // namespace aitrain
