#pragma once

#include "aitrain/core/DetectionDataset.h"

#include <QImage>
#include <QPointF>
#include <QSize>
#include <QString>
#include <QVector>

namespace aitrain {

struct SegmentationPolygon {
    int classId = -1;
    QVector<QPointF> points;
};

struct SegmentationSample {
    QString imagePath;
    QString labelPath;
    QSize imageSize;
    QVector<SegmentationPolygon> polygons;
};

struct SegmentationBatch {
    QVector<QImage> images;
    QVector<QImage> masks;
    QVector<QVector<SegmentationPolygon>> polygons;
    QVector<QString> imagePaths;
};

class SegmentationDataset {
public:
    bool load(const QString& datasetPath, const QString& split, QString* error = nullptr);

    QString rootPath() const;
    QString split() const;
    DetectionDatasetInfo info() const;
    QVector<SegmentationSample> samples() const;
    int size() const;
    bool isEmpty() const;

private:
    QString rootPath_;
    QString split_;
    DetectionDatasetInfo info_;
    QVector<SegmentationSample> samples_;
};

QImage polygonToMask(const QVector<QPointF>& normalizedPolygon, const QSize& targetSize);
QImage segmentationPolygonsToMask(const QVector<SegmentationPolygon>& polygons, const QSize& targetSize);
SegmentationPolygon mapPolygonToLetterbox(
    const SegmentationPolygon& polygon,
    const QSize& sourceSize,
    const LetterboxTransform& transform);

QImage renderSegmentationOverlay(
    const QString& imagePath,
    const QVector<SegmentationPolygon>& polygons,
    QString* error = nullptr);

class SegmentationDataLoader {
public:
    SegmentationDataLoader();
    SegmentationDataLoader(const SegmentationDataset& dataset, int batchSize, QSize imageSize);

    void reset();
    bool hasNext() const;
    bool next(SegmentationBatch* batch, QString* error = nullptr);

private:
    SegmentationDataset dataset_;
    int batchSize_ = 1;
    QSize imageSize_;
    int cursor_ = 0;
};

} // namespace aitrain
