#include "aitrain/v2/PaddleOcrDetDatasetDriverV2.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain::v2 {
PaddleOcrDetDatasetDriverV2::PaddleOcrDetDatasetDriverV2()
    : ValidatedDatasetDriverV2(QStringLiteral("paddleocr_det"), QStringLiteral("paddleocr_det"),
        [](const QString& path, const QJsonObject& options) { return validatePaddleOcrDetDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitPaddleOcrDetDataset(source, output, options); },
        [](const QString& path) {
            const QDir root(path);
            return QFileInfo::exists(root.filePath(QStringLiteral("det_gt.txt")))
                || QFileInfo::exists(root.filePath(QStringLiteral("det_gt_train.txt")));
        })
{
}
} // namespace aitrain::v2
