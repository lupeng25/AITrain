#pragma once

#include <QImage>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

struct DetectionBox {
    int classId = -1;
    double xCenter = 0.0;
    double yCenter = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct DetectionSample {
    QString imagePath;
    QString labelPath;
    QSize imageSize;
    QVector<DetectionBox> boxes;
};

struct DetectionDatasetInfo {
    int classCount = 0;
    QStringList classNames;
};

struct LetterboxTransform {
    QSize sourceSize;
    QSize targetSize;
    double scale = 1.0;
    int padX = 0;
    int padY = 0;
};

struct DetectionBatch {
    QVector<QImage> images;
    QVector<QVector<DetectionBox>> boxes;
    QVector<QString> imagePaths;
};

class DetectionDataset {
public:
    bool load(const QString& datasetPath, const QString& split, QString* error = nullptr);

    QString rootPath() const;
    QString split() const;
    DetectionDatasetInfo info() const;
    QVector<DetectionSample> samples() const;
    int size() const;
    bool isEmpty() const;

private:
    QString rootPath_;
    QString split_;
    DetectionDatasetInfo info_;
    QVector<DetectionSample> samples_;
};

DetectionDatasetInfo readDetectionDatasetInfo(const QString& datasetPath, QString* error = nullptr);
QImage letterboxImage(const QImage& image, const QSize& targetSize, LetterboxTransform* transform = nullptr);
DetectionBox mapBoxToLetterbox(const DetectionBox& box, const QSize& sourceSize, const LetterboxTransform& transform);

class DetectionDataLoader {
public:
    DetectionDataLoader();
    DetectionDataLoader(const DetectionDataset& dataset, int batchSize, QSize imageSize);

    void reset();
    bool hasNext() const;
    bool next(DetectionBatch* batch, QString* error = nullptr);

private:
    DetectionDataset dataset_;
    int batchSize_ = 1;
    QSize imageSize_;
    int cursor_ = 0;
};

} // namespace aitrain
