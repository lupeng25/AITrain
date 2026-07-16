#include "aitrain/v2/YoloObbDatasetDriverV2.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain::v2 {
YoloObbDatasetDriverV2::YoloObbDatasetDriverV2()
    : ValidatedDatasetDriverV2(QStringLiteral("yolo_obb"), QStringLiteral("yolo_obb"),
        [](const QString& path, const QJsonObject& options) { return validateYoloObbDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitYoloObbDataset(source, output, options); },
        [](const QString& path) {
            const QDir root(path);
            return root.exists() && QFileInfo::exists(root.filePath(QStringLiteral("data.yaml")))
                && QDir(root.filePath(QStringLiteral("images"))).exists()
                && QDir(root.filePath(QStringLiteral("labels"))).exists();
        })
{
}
} // namespace aitrain::v2
