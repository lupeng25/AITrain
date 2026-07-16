#include "aitrain/v2/PaddleOcrRecDatasetDriverV2.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain::v2 {
PaddleOcrRecDatasetDriverV2::PaddleOcrRecDatasetDriverV2()
    : ValidatedDatasetDriverV2(QStringLiteral("paddleocr_rec"), QStringLiteral("paddleocr_rec"),
        [](const QString& path, const QJsonObject& options) { return validatePaddleOcrRecDataset(path, options); },
        [](const QString& source, const QString& output, const QJsonObject& options) { return splitPaddleOcrRecDataset(source, output, options); },
        [](const QString& path) {
            const QDir root(path);
            return QFileInfo::exists(root.filePath(QStringLiteral("rec_gt.txt")))
                || QFileInfo::exists(root.filePath(QStringLiteral("rec_gt_train.txt")));
        })
{
}
} // namespace aitrain::v2
