#include "aitrain/dataset/YoloDetectionDatasetDriver.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain {
YoloDetectionDatasetDriver::YoloDetectionDatasetDriver()
    : ValidatedDatasetDriver(QStringLiteral("yolo_detection"), QStringLiteral("yolo_detection"),
        [](const QString& path, const QJsonObject& options) { return validateYoloDetectionDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitYoloDetectionDataset(source, output, options); },
        [](const QString& path) {
            const QDir root(path);
            return root.exists() && QFileInfo::exists(root.filePath(QStringLiteral("data.yaml")))
                && QDir(root.filePath(QStringLiteral("images"))).exists()
                && QDir(root.filePath(QStringLiteral("labels"))).exists();
        })
{
}
} // namespace aitrain
