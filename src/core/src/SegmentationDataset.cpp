#include "aitrain/core/SegmentationDataset.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QPainter>
#include <QPolygonF>
#include <QRegularExpression>

namespace aitrain {
namespace {

QStringList splitFields(const QString& line)
{
    return line.split(QRegularExpression(QStringLiteral("\\s+")),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
        QString::SkipEmptyParts
#else
        Qt::SkipEmptyParts
#endif
    );
}

QStringList imageNameFilters()
{
    return {
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.png"),
        QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff")
    };
}

QFileInfoList imageFiles(const QDir& directory)
{
    QFileInfoList files;
    for (const QString& filter : imageNameFilters()) {
        files.append(directory.entryInfoList({filter}, QDir::Files, QDir::Name));
    }
    return files;
}

bool parseSegmentationLabelFile(
    const QString& labelPath,
    int classCount,
    QVector<SegmentationPolygon>* polygons,
    QString* error)
{
    QFile file(labelPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot open segmentation label file: %1").arg(labelPath);
        }
        return false;
    }

    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }

        const QStringList parts = splitFields(line);
        if (parts.size() < 7 || parts.size() % 2 == 0) {
            if (error) {
                *error = QStringLiteral("%1:%2 expected class id followed by at least 3 polygon points").arg(labelPath).arg(lineNumber);
            }
            return false;
        }

        bool classOk = false;
        const int classId = parts.at(0).toInt(&classOk);
        if (!classOk || classId < 0 || (classCount > 0 && classId >= classCount)) {
            if (error) {
                *error = QStringLiteral("%1:%2 invalid class id").arg(labelPath).arg(lineNumber);
            }
            return false;
        }

        SegmentationPolygon polygon;
        polygon.classId = classId;
        for (int index = 1; index < parts.size(); index += 2) {
            bool xOk = false;
            bool yOk = false;
            const double x = parts.at(index).toDouble(&xOk);
            const double y = parts.at(index + 1).toDouble(&yOk);
            if (!xOk || !yOk || x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0) {
                if (error) {
                    *error = QStringLiteral("%1:%2 polygon coordinates must be in [0,1]").arg(labelPath).arg(lineNumber);
                }
                return false;
            }
            polygon.points.append(QPointF(x, y));
        }
        polygons->append(polygon);
    }
    return true;
}

QPolygonF toPixelPolygon(const QVector<QPointF>& normalizedPolygon, const QSize& size)
{
    QPolygonF polygon;
    if (!size.isValid() || size.isEmpty()) {
        return polygon;
    }

    const double maxX = static_cast<double>(size.width() - 1);
    const double maxY = static_cast<double>(size.height() - 1);
    for (const QPointF& point : normalizedPolygon) {
        const double x = qBound(0.0, point.x(), 1.0) * maxX;
        const double y = qBound(0.0, point.y(), 1.0) * maxY;
        polygon.append(QPointF(x, y));
    }
    return polygon;
}

QColor classColor(int classId, int alpha)
{
    static const QVector<QColor> colors = {
        QColor(46, 204, 113),
        QColor(52, 152, 219),
        QColor(241, 196, 15),
        QColor(231, 76, 60),
        QColor(155, 89, 182),
        QColor(26, 188, 156)
    };
    QColor color = colors.at(qAbs(classId) % colors.size());
    color.setAlpha(alpha);
    return color;
}

} // namespace

bool SegmentationDataset::load(const QString& datasetPath, const QString& split, QString* error)
{
    rootPath_ = QDir::cleanPath(datasetPath);
    split_ = split;
    info_ = {};
    samples_.clear();

    info_ = readDetectionDatasetInfo(rootPath_, error);
    if (info_.classCount <= 0) {
        return false;
    }

    const QDir imageDir(QDir(rootPath_).filePath(QStringLiteral("images/%1").arg(split_)));
    const QDir labelDir(QDir(rootPath_).filePath(QStringLiteral("labels/%1").arg(split_)));
    if (!imageDir.exists()) {
        if (error) {
            *error = QStringLiteral("Missing image split directory: %1").arg(imageDir.path());
        }
        return false;
    }
    if (!labelDir.exists()) {
        if (error) {
            *error = QStringLiteral("Missing label split directory: %1").arg(labelDir.path());
        }
        return false;
    }

    const QFileInfoList images = imageFiles(imageDir);
    if (images.isEmpty()) {
        if (error) {
            *error = QStringLiteral("No segmentation images found in split: %1").arg(split_);
        }
        return false;
    }

    for (const QFileInfo& imageInfo : images) {
        const QFileInfo labelInfo(labelDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".txt")));
        if (!labelInfo.exists()) {
            if (error) {
                *error = QStringLiteral("Missing segmentation label for image: %1").arg(imageInfo.absoluteFilePath());
            }
            return false;
        }

        SegmentationSample sample;
        sample.imagePath = imageInfo.absoluteFilePath();
        sample.labelPath = labelInfo.absoluteFilePath();
        sample.imageSize = QImageReader(imageInfo.absoluteFilePath()).size();
        if (!parseSegmentationLabelFile(sample.labelPath, info_.classCount, &sample.polygons, error)) {
            return false;
        }
        samples_.append(sample);
    }
    return true;
}

QString SegmentationDataset::rootPath() const
{
    return rootPath_;
}

QString SegmentationDataset::split() const
{
    return split_;
}

DetectionDatasetInfo SegmentationDataset::info() const
{
    return info_;
}

QVector<SegmentationSample> SegmentationDataset::samples() const
{
    return samples_;
}

int SegmentationDataset::size() const
{
    return samples_.size();
}

bool SegmentationDataset::isEmpty() const
{
    return samples_.isEmpty();
}

QImage polygonToMask(const QVector<QPointF>& normalizedPolygon, const QSize& targetSize)
{
    if (normalizedPolygon.size() < 3 || !targetSize.isValid() || targetSize.isEmpty()) {
        return {};
    }

    QImage mask(targetSize, QImage::Format_ARGB32);
    mask.fill(Qt::transparent);
    QPainter painter(&mask);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::white);
    painter.drawPolygon(toPixelPolygon(normalizedPolygon, targetSize));
    painter.end();
    return mask;
}

QImage renderSegmentationOverlay(
    const QString& imagePath,
    const QVector<SegmentationPolygon>& polygons,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot load image for segmentation overlay: %1").arg(imagePath);
        }
        return {};
    }

    QImage output = image.convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&output);
    painter.setRenderHint(QPainter::Antialiasing, true);
    for (const SegmentationPolygon& polygon : polygons) {
        const QPolygonF pixelPolygon = toPixelPolygon(polygon.points, output.size());
        if (pixelPolygon.size() < 3) {
            continue;
        }
        painter.setPen(QPen(classColor(polygon.classId, 230), 2.0));
        painter.setBrush(classColor(polygon.classId, 80));
        painter.drawPolygon(pixelPolygon);
    }
    painter.end();
    return output;
}

} // namespace aitrain
