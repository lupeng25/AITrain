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

double polygonArea(const QVector<QPointF>& polygon)
{
    if (polygon.size() < 3) {
        return 0.0;
    }

    double area = 0.0;
    for (int index = 0; index < polygon.size(); ++index) {
        const QPointF& a = polygon.at(index);
        const QPointF& b = polygon.at((index + 1) % polygon.size());
        area += a.x() * b.y() - b.x() * a.y();
    }
    return qAbs(area) * 0.5;
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

QImage segmentationPolygonsToMask(const QVector<SegmentationPolygon>& polygons, const QSize& targetSize)
{
    if (!targetSize.isValid() || targetSize.isEmpty()) {
        return {};
    }

    QImage mask(targetSize, QImage::Format_ARGB32);
    mask.fill(Qt::transparent);
    QPainter painter(&mask);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setPen(Qt::NoPen);
    for (const SegmentationPolygon& polygon : polygons) {
        if (polygon.points.size() < 3 || polygonArea(polygon.points) <= 0.0) {
            continue;
        }
        const int value = qBound(1, polygon.classId + 1, 255);
        painter.setBrush(QColor(value, value, value, 255));
        painter.drawPolygon(toPixelPolygon(polygon.points, targetSize));
    }
    painter.end();
    return mask;
}

SegmentationPolygon mapPolygonToLetterbox(
    const SegmentationPolygon& polygon,
    const QSize& sourceSize,
    const LetterboxTransform& transform)
{
    SegmentationPolygon mapped;
    mapped.classId = polygon.classId;
    if (!sourceSize.isValid() || sourceSize.isEmpty() || !transform.targetSize.isValid() || transform.targetSize.isEmpty()) {
        mapped.points = polygon.points;
        return mapped;
    }

    const double sourceWidth = static_cast<double>(sourceSize.width());
    const double sourceHeight = static_cast<double>(sourceSize.height());
    const double targetWidth = static_cast<double>(transform.targetSize.width());
    const double targetHeight = static_cast<double>(transform.targetSize.height());
    for (const QPointF& point : polygon.points) {
        const double x = (point.x() * sourceWidth * transform.scale + transform.padX) / targetWidth;
        const double y = (point.y() * sourceHeight * transform.scale + transform.padY) / targetHeight;
        mapped.points.append(QPointF(qBound(0.0, x, 1.0), qBound(0.0, y, 1.0)));
    }
    return mapped;
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

SegmentationDataLoader::SegmentationDataLoader() = default;

SegmentationDataLoader::SegmentationDataLoader(const SegmentationDataset& dataset, int batchSize, QSize imageSize)
    : dataset_(dataset)
    , batchSize_(qMax(1, batchSize))
    , imageSize_(imageSize)
{
}

void SegmentationDataLoader::reset()
{
    cursor_ = 0;
}

bool SegmentationDataLoader::hasNext() const
{
    return cursor_ < dataset_.samples().size();
}

bool SegmentationDataLoader::next(SegmentationBatch* batch, QString* error)
{
    if (!batch) {
        if (error) {
            *error = QStringLiteral("SegmentationBatch output is null");
        }
        return false;
    }
    batch->images.clear();
    batch->masks.clear();
    batch->polygons.clear();
    batch->imagePaths.clear();

    const QVector<SegmentationSample> samples = dataset_.samples();
    if (cursor_ >= samples.size()) {
        return true;
    }

    const int end = qMin(samples.size(), cursor_ + batchSize_);
    for (; cursor_ < end; ++cursor_) {
        const SegmentationSample& sample = samples.at(cursor_);
        QImage image(sample.imagePath);
        if (image.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot load segmentation image: %1").arg(sample.imagePath);
            }
            return false;
        }

        LetterboxTransform transform;
        QImage processed = letterboxImage(image, imageSize_, &transform);
        if (processed.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot preprocess segmentation image: %1").arg(sample.imagePath);
            }
            return false;
        }

        QVector<SegmentationPolygon> mappedPolygons;
        for (const SegmentationPolygon& polygon : sample.polygons) {
            const SegmentationPolygon mapped = mapPolygonToLetterbox(polygon, image.size(), transform);
            if (mapped.points.size() < 3 || polygonArea(mapped.points) <= 0.0) {
                if (error) {
                    *error = QStringLiteral("Invalid segmentation polygon in label: %1").arg(sample.labelPath);
                }
                return false;
            }
            mappedPolygons.append(mapped);
        }

        const QImage mask = segmentationPolygonsToMask(mappedPolygons, imageSize_);
        if (mask.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot build segmentation mask: %1").arg(sample.labelPath);
            }
            return false;
        }

        batch->images.append(processed);
        batch->masks.append(mask);
        batch->polygons.append(mappedPolygons);
        batch->imagePaths.append(sample.imagePath);
    }
    return true;
}

} // namespace aitrain
