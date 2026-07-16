#include "aitrain/dataset/YoloSegmentationDatasetDriver.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain {
YoloSegmentationDatasetDriver::YoloSegmentationDatasetDriver()
    : ValidatedDatasetDriver(QStringLiteral("yolo_segmentation"), QStringLiteral("yolo_segmentation"),
        [](const QString& path, const QJsonObject& options) { return validateYoloSegmentationDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitYoloSegmentationDataset(source, output, options); },
        [](const QString& path) {
            const QDir root(path);
            return root.exists() && QFileInfo::exists(root.filePath(QStringLiteral("data.yaml")))
                && QDir(root.filePath(QStringLiteral("images"))).exists()
                && QDir(root.filePath(QStringLiteral("labels"))).exists();
        })
{
}
} // namespace aitrain
