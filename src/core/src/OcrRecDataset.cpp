#include "aitrain/core/OcrRecDataset.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QPainter>

namespace aitrain {
namespace {

QHash<QChar, int> dictionaryIndex(const OcrRecDictionary& dictionary)
{
    QHash<QChar, int> index;
    for (int i = 0; i < dictionary.characters.size(); ++i) {
        const QString character = dictionary.characters.at(i);
        if (!character.isEmpty()) {
            index.insert(character.at(0), i + 1);
        }
    }
    return index;
}

QString defaultLabelFilePath(const QString& rootPath)
{
    return QDir(rootPath).filePath(QStringLiteral("rec_gt.txt"));
}

QString defaultDictionaryFilePath(const QString& rootPath)
{
    return QDir(rootPath).filePath(QStringLiteral("dict.txt"));
}

} // namespace

bool readOcrRecDictionary(const QString& dictionaryFilePath, OcrRecDictionary* dictionary, QString* error)
{
    if (!dictionary) {
        if (error) {
            *error = QStringLiteral("OCR dictionary output is null");
        }
        return false;
    }
    dictionary->path = QDir::cleanPath(dictionaryFilePath);
    dictionary->characters.clear();

    QFile file(dictionaryFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot open OCR dictionary: %1").arg(dictionaryFilePath);
        }
        return false;
    }

    QHash<QChar, int> seen;
    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const QChar character = line.at(0);
        if (seen.contains(character)) {
            if (error) {
                *error = QStringLiteral("%1:%2 duplicate OCR dictionary character").arg(dictionaryFilePath).arg(lineNumber);
            }
            return false;
        }
        seen.insert(character, dictionary->characters.size() + 1);
        dictionary->characters.append(QString(character));
    }

    if (dictionary->characters.isEmpty()) {
        if (error) {
            *error = QStringLiteral("OCR dictionary is empty: %1").arg(dictionaryFilePath);
        }
        return false;
    }
    return true;
}

QVector<int> encodeOcrText(const QString& text, const OcrRecDictionary& dictionary, QString* error)
{
    QVector<int> encoded;
    const QHash<QChar, int> index = dictionaryIndex(dictionary);
    if (index.isEmpty()) {
        if (error) {
            *error = QStringLiteral("OCR dictionary is empty");
        }
        return encoded;
    }

    for (const QChar character : text) {
        if (!index.contains(character)) {
            if (error) {
                *error = QStringLiteral("OCR label character is not in dictionary: %1").arg(character);
            }
            encoded.clear();
            return encoded;
        }
        encoded.append(index.value(character));
    }
    return encoded;
}

QString decodeOcrText(const QVector<int>& encoded, const OcrRecDictionary& dictionary, bool collapseRepeats)
{
    QString text;
    int previous = -1;
    for (const int token : encoded) {
        if (token <= 0 || token > dictionary.characters.size()) {
            previous = token;
            continue;
        }
        if (collapseRepeats && token == previous) {
            continue;
        }
        text.append(dictionary.characters.at(token - 1));
        previous = token;
    }
    return text;
}

QImage resizePadOcrImage(const QImage& image, const QSize& targetSize)
{
    if (image.isNull() || !targetSize.isValid() || targetSize.isEmpty()) {
        return {};
    }

    const double scale = qMin(
        static_cast<double>(targetSize.width()) / static_cast<double>(image.width()),
        static_cast<double>(targetSize.height()) / static_cast<double>(image.height()));
    const QSize resizedSize(qMax(1, qRound(image.width() * scale)), qMax(1, qRound(image.height() * scale)));

    QImage output(targetSize, QImage::Format_RGB888);
    output.fill(Qt::white);
    QPainter painter(&output);
    painter.drawImage(QRect(QPoint(0, 0), resizedSize), image.convertToFormat(QImage::Format_RGB888));
    painter.end();
    return output;
}

bool OcrRecDataset::load(
    const QString& datasetPath,
    const QString& labelFilePath,
    const QString& dictionaryFilePath,
    int maxTextLength,
    QString* error)
{
    rootPath_ = QDir::cleanPath(datasetPath);
    labelFilePath_ = labelFilePath.isEmpty() ? defaultLabelFilePath(rootPath_) : QDir::cleanPath(labelFilePath);
    dictionary_ = {};
    samples_.clear();

    const QString dictPath = dictionaryFilePath.isEmpty() ? defaultDictionaryFilePath(rootPath_) : QDir::cleanPath(dictionaryFilePath);
    if (!readOcrRecDictionary(dictPath, &dictionary_, error)) {
        return false;
    }

    QFile file(labelFilePath_);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot open OCR label file: %1").arg(labelFilePath_);
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

        const int split = line.indexOf(QLatin1Char('\t'));
        if (split <= 0) {
            if (error) {
                *error = QStringLiteral("%1:%2 expected '<image path>\\t<label>'").arg(labelFilePath_).arg(lineNumber);
            }
            return false;
        }

        OcrRecSample sample;
        sample.relativeImagePath = line.left(split).trimmed();
        sample.label = line.mid(split + 1);
        if (sample.label.isEmpty()) {
            if (error) {
                *error = QStringLiteral("%1:%2 OCR label must not be empty").arg(labelFilePath_).arg(lineNumber);
            }
            return false;
        }
        if (maxTextLength > 0 && sample.label.size() > maxTextLength) {
            if (error) {
                *error = QStringLiteral("%1:%2 OCR label exceeds maxTextLength").arg(labelFilePath_).arg(lineNumber);
            }
            return false;
        }

        sample.imagePath = QDir(rootPath_).filePath(sample.relativeImagePath);
        if (!QFileInfo::exists(sample.imagePath)) {
            if (error) {
                *error = QStringLiteral("%1:%2 OCR image does not exist: %3").arg(labelFilePath_).arg(lineNumber).arg(sample.imagePath);
            }
            return false;
        }

        sample.imageSize = QImageReader(sample.imagePath).size();
        sample.encodedLabel = encodeOcrText(sample.label, dictionary_, error);
        if (sample.encodedLabel.isEmpty()) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("%1:%2 OCR label could not be encoded").arg(labelFilePath_).arg(lineNumber);
            }
            return false;
        }
        samples_.append(sample);
    }

    if (samples_.isEmpty()) {
        if (error) {
            *error = QStringLiteral("No OCR recognition samples found: %1").arg(labelFilePath_);
        }
        return false;
    }
    return true;
}

QString OcrRecDataset::rootPath() const
{
    return rootPath_;
}

QString OcrRecDataset::labelFilePath() const
{
    return labelFilePath_;
}

OcrRecDictionary OcrRecDataset::dictionary() const
{
    return dictionary_;
}

QVector<OcrRecSample> OcrRecDataset::samples() const
{
    return samples_;
}

int OcrRecDataset::size() const
{
    return samples_.size();
}

bool OcrRecDataset::isEmpty() const
{
    return samples_.isEmpty();
}

OcrRecDataLoader::OcrRecDataLoader() = default;

OcrRecDataLoader::OcrRecDataLoader(const OcrRecDataset& dataset, int batchSize, QSize imageSize)
    : dataset_(dataset)
    , batchSize_(qMax(1, batchSize))
    , imageSize_(imageSize)
{
}

void OcrRecDataLoader::reset()
{
    cursor_ = 0;
}

bool OcrRecDataLoader::hasNext() const
{
    return cursor_ < dataset_.samples().size();
}

bool OcrRecDataLoader::next(OcrRecBatch* batch, QString* error)
{
    if (!batch) {
        if (error) {
            *error = QStringLiteral("OcrRecBatch output is null");
        }
        return false;
    }

    batch->images.clear();
    batch->labels.clear();
    batch->labelLengths.clear();
    batch->texts.clear();
    batch->imagePaths.clear();

    const QVector<OcrRecSample> samples = dataset_.samples();
    if (cursor_ >= samples.size()) {
        return true;
    }

    const int end = qMin(samples.size(), cursor_ + batchSize_);
    for (; cursor_ < end; ++cursor_) {
        const OcrRecSample& sample = samples.at(cursor_);
        QImage image(sample.imagePath);
        if (image.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot load OCR image: %1").arg(sample.imagePath);
            }
            return false;
        }

        const QImage processed = resizePadOcrImage(image, imageSize_);
        if (processed.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot preprocess OCR image: %1").arg(sample.imagePath);
            }
            return false;
        }

        batch->images.append(processed);
        batch->labels.append(sample.encodedLabel);
        batch->labelLengths.append(sample.encodedLabel.size());
        batch->texts.append(sample.label);
        batch->imagePaths.append(sample.imagePath);
    }
    return true;
}

} // namespace aitrain
