#include "aitrain/core/DetectionDataset.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QPainter>
#include <QImageReader>
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

QString stripComment(QString line)
{
    const int commentIndex = line.indexOf(QLatin1Char('#'));
    if (commentIndex >= 0) {
        line = line.left(commentIndex);
    }
    return line.trimmed();
}

QStringList parseInlineNames(const QString& value)
{
    QString trimmed = value.trimmed();
    if (!trimmed.startsWith(QLatin1Char('[')) || !trimmed.endsWith(QLatin1Char(']'))) {
        return {};
    }
    trimmed = trimmed.mid(1, trimmed.size() - 2);
    QStringList names;
    for (QString name : trimmed.split(QLatin1Char(','),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
             QString::SkipEmptyParts
#else
             Qt::SkipEmptyParts
#endif
             )) {
        name = name.trimmed();
        if ((name.startsWith(QLatin1Char('\'')) && name.endsWith(QLatin1Char('\'')))
            || (name.startsWith(QLatin1Char('"')) && name.endsWith(QLatin1Char('"')))) {
            name = name.mid(1, name.size() - 2);
        }
        if (!name.isEmpty()) {
            names.append(name);
        }
    }
    return names;
}

bool parseDetectionLabelFile(const QString& labelPath, int classCount, QVector<DetectionBox>* boxes, QString* error)
{
    QFile file(labelPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot open label file: %1").arg(labelPath);
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
        if (parts.size() != 5) {
            if (error) {
                *error = QStringLiteral("%1:%2 expected 5 YOLO detection columns").arg(labelPath).arg(lineNumber);
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

        double values[4] = {};
        for (int index = 0; index < 4; ++index) {
            bool ok = false;
            values[index] = parts.at(index + 1).toDouble(&ok);
            if (!ok || values[index] < 0.0 || values[index] > 1.0) {
                if (error) {
                    *error = QStringLiteral("%1:%2 bbox values must be in [0,1]").arg(labelPath).arg(lineNumber);
                }
                return false;
            }
        }
        if (values[2] <= 0.0 || values[3] <= 0.0) {
            if (error) {
                *error = QStringLiteral("%1:%2 bbox width and height must be greater than 0").arg(labelPath).arg(lineNumber);
            }
            return false;
        }

        DetectionBox box;
        box.classId = classId;
        box.xCenter = values[0];
        box.yCenter = values[1];
        box.width = values[2];
        box.height = values[3];
        boxes->append(box);
    }
    return true;
}

} // namespace

DetectionDatasetInfo readDetectionDatasetInfo(const QString& datasetPath, QString* error)
{
    DetectionDatasetInfo info;
    const QString yamlPath = QDir(datasetPath).filePath(QStringLiteral("data.yaml"));
    QFile file(yamlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot open data.yaml: %1").arg(yamlPath);
        }
        return info;
    }

    bool hasClassCount = false;
    while (!file.atEnd()) {
        const QString line = stripComment(QString::fromUtf8(file.readLine()));
        if (line.startsWith(QStringLiteral("nc:"))) {
            bool ok = false;
            const int classCount = line.mid(3).trimmed().toInt(&ok);
            if (!ok || classCount <= 0) {
                if (error) {
                    *error = QStringLiteral("data.yaml nc must be a positive integer");
                }
                return {};
            }
            info.classCount = classCount;
            hasClassCount = true;
        } else if (line.startsWith(QStringLiteral("names:"))) {
            info.classNames = parseInlineNames(line.mid(6));
        }
    }

    if (!hasClassCount && !info.classNames.isEmpty()) {
        info.classCount = info.classNames.size();
        hasClassCount = true;
    }
    if (!hasClassCount) {
        if (error) {
            *error = QStringLiteral("data.yaml must define nc or inline names");
        }
        return {};
    }
    if (!info.classNames.isEmpty() && info.classNames.size() != info.classCount) {
        if (error) {
            *error = QStringLiteral("data.yaml names count does not match nc");
        }
        return {};
    }
    return info;
}

bool DetectionDataset::load(const QString& datasetPath, const QString& split, QString* error)
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
            *error = QStringLiteral("No images found in split: %1").arg(split_);
        }
        return false;
    }

    for (const QFileInfo& imageInfo : images) {
        const QFileInfo labelInfo(labelDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".txt")));
        if (!labelInfo.exists()) {
            if (error) {
                *error = QStringLiteral("Missing label for image: %1").arg(imageInfo.absoluteFilePath());
            }
            return false;
        }

        DetectionSample sample;
        sample.imagePath = imageInfo.absoluteFilePath();
        sample.labelPath = labelInfo.absoluteFilePath();
        sample.imageSize = QImageReader(imageInfo.absoluteFilePath()).size();
        if (!parseDetectionLabelFile(sample.labelPath, info_.classCount, &sample.boxes, error)) {
            return false;
        }
        samples_.append(sample);
    }
    return true;
}

QString DetectionDataset::rootPath() const
{
    return rootPath_;
}

QString DetectionDataset::split() const
{
    return split_;
}

DetectionDatasetInfo DetectionDataset::info() const
{
    return info_;
}

QVector<DetectionSample> DetectionDataset::samples() const
{
    return samples_;
}

int DetectionDataset::size() const
{
    return samples_.size();
}

bool DetectionDataset::isEmpty() const
{
    return samples_.isEmpty();
}

QImage letterboxImage(const QImage& image, const QSize& targetSize, LetterboxTransform* transform)
{
    if (image.isNull() || !targetSize.isValid() || targetSize.isEmpty()) {
        if (transform) {
            *transform = {};
        }
        return {};
    }

    const double scale = qMin(static_cast<double>(targetSize.width()) / static_cast<double>(image.width()),
        static_cast<double>(targetSize.height()) / static_cast<double>(image.height()));
    const QSize resizedSize(qMax(1, qRound(image.width() * scale)), qMax(1, qRound(image.height() * scale)));
    const int padX = (targetSize.width() - resizedSize.width()) / 2;
    const int padY = (targetSize.height() - resizedSize.height()) / 2;

    QImage output(targetSize, QImage::Format_RGB888);
    output.fill(QColor(114, 114, 114));
    QPainter painter(&output);
    painter.drawImage(QRect(QPoint(padX, padY), resizedSize), image.convertToFormat(QImage::Format_RGB888));
    painter.end();

    if (transform) {
        transform->sourceSize = image.size();
        transform->targetSize = targetSize;
        transform->scale = scale;
        transform->padX = padX;
        transform->padY = padY;
    }
    return output;
}

DetectionBox mapBoxToLetterbox(const DetectionBox& box, const QSize& sourceSize, const LetterboxTransform& transform)
{
    DetectionBox mapped = box;
    if (!sourceSize.isValid() || sourceSize.isEmpty() || !transform.targetSize.isValid() || transform.targetSize.isEmpty()) {
        return mapped;
    }

    const double sourceWidth = static_cast<double>(sourceSize.width());
    const double sourceHeight = static_cast<double>(sourceSize.height());
    const double targetWidth = static_cast<double>(transform.targetSize.width());
    const double targetHeight = static_cast<double>(transform.targetSize.height());

    mapped.xCenter = (box.xCenter * sourceWidth * transform.scale + transform.padX) / targetWidth;
    mapped.yCenter = (box.yCenter * sourceHeight * transform.scale + transform.padY) / targetHeight;
    mapped.width = (box.width * sourceWidth * transform.scale) / targetWidth;
    mapped.height = (box.height * sourceHeight * transform.scale) / targetHeight;
    return mapped;
}

DetectionDataLoader::DetectionDataLoader() = default;

DetectionDataLoader::DetectionDataLoader(const DetectionDataset& dataset, int batchSize, QSize imageSize)
    : dataset_(dataset)
    , batchSize_(qMax(1, batchSize))
    , imageSize_(imageSize)
{
}

void DetectionDataLoader::reset()
{
    cursor_ = 0;
}

bool DetectionDataLoader::hasNext() const
{
    return cursor_ < dataset_.samples().size();
}

bool DetectionDataLoader::next(DetectionBatch* batch, QString* error)
{
    if (!batch) {
        if (error) {
            *error = QStringLiteral("DetectionBatch output is null");
        }
        return false;
    }
    batch->images.clear();
    batch->boxes.clear();
    batch->imagePaths.clear();

    const QVector<DetectionSample> samples = dataset_.samples();
    if (cursor_ >= samples.size()) {
        return true;
    }

    const int end = qMin(samples.size(), cursor_ + batchSize_);
    for (; cursor_ < end; ++cursor_) {
        const DetectionSample& sample = samples.at(cursor_);
        QImage image(sample.imagePath);
        if (image.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot load image: %1").arg(sample.imagePath);
            }
            return false;
        }

        LetterboxTransform transform;
        QImage processed = letterboxImage(image, imageSize_, &transform);
        if (processed.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot preprocess image: %1").arg(sample.imagePath);
            }
            return false;
        }

        QVector<DetectionBox> mappedBoxes;
        for (const DetectionBox& box : sample.boxes) {
            mappedBoxes.append(mapBoxToLetterbox(box, image.size(), transform));
        }
        batch->images.append(processed);
        batch->boxes.append(mappedBoxes);
        batch->imagePaths.append(sample.imagePath);
    }
    return true;
}

} // namespace aitrain
