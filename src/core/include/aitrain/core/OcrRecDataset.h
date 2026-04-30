#pragma once

#include <QImage>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

struct OcrRecDictionary {
    QString path;
    QStringList characters;
};

struct OcrRecSample {
    QString imagePath;
    QString relativeImagePath;
    QString label;
    QVector<int> encodedLabel;
    QSize imageSize;
};

struct OcrRecBatch {
    QVector<QImage> images;
    QVector<QVector<int>> labels;
    QVector<int> labelLengths;
    QVector<QString> texts;
    QVector<QString> imagePaths;
};

class OcrRecDataset {
public:
    bool load(
        const QString& datasetPath,
        const QString& labelFilePath = QString(),
        const QString& dictionaryFilePath = QString(),
        int maxTextLength = 25,
        QString* error = nullptr);

    QString rootPath() const;
    QString labelFilePath() const;
    OcrRecDictionary dictionary() const;
    QVector<OcrRecSample> samples() const;
    int size() const;
    bool isEmpty() const;

private:
    QString rootPath_;
    QString labelFilePath_;
    OcrRecDictionary dictionary_;
    QVector<OcrRecSample> samples_;
};

bool readOcrRecDictionary(const QString& dictionaryFilePath, OcrRecDictionary* dictionary, QString* error = nullptr);
QVector<int> encodeOcrText(const QString& text, const OcrRecDictionary& dictionary, QString* error = nullptr);
QString decodeOcrText(const QVector<int>& encoded, const OcrRecDictionary& dictionary, bool collapseRepeats = true);
QImage resizePadOcrImage(const QImage& image, const QSize& targetSize);

class OcrRecDataLoader {
public:
    OcrRecDataLoader();
    OcrRecDataLoader(const OcrRecDataset& dataset, int batchSize, QSize imageSize);

    void reset();
    bool hasNext() const;
    bool next(OcrRecBatch* batch, QString* error = nullptr);

private:
    OcrRecDataset dataset_;
    int batchSize_ = 1;
    QSize imageSize_;
    int cursor_ = 0;
};

} // namespace aitrain
