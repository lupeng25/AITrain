#include "aitrain/dataset/PaddleOcrDetDatasetDriver.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain {
PaddleOcrDetDatasetDriver::PaddleOcrDetDatasetDriver()
    : ValidatedDatasetDriver(QStringLiteral("paddleocr_det"), QStringLiteral("paddleocr_det"),
        [](const QString& path, const QJsonObject& options) { return validatePaddleOcrDetDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitPaddleOcrDetDataset(source, output, options); },
        [](const QString& path) {
            const QDir root(path);
            return QFileInfo::exists(root.filePath(QStringLiteral("det_gt.txt")))
                || QFileInfo::exists(root.filePath(QStringLiteral("det_gt_train.txt")));
        })
{
}
} // namespace aitrain
