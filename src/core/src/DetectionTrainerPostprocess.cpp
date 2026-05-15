#include "DetectionTrainerInternal.h"

#include "aitrain/core/Deployment.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPainter>
#include <QProcess>
#include <QQueue>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QtEndian>
#include <QtMath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
namespace aitrain {

using namespace detection_detail;

QVector<DetectionPrediction> postProcessDetectionPredictions(
    const QVector<DetectionPrediction>& predictions,
    const DetectionInferenceOptions& options)
{
    if (options.maxDetections <= 0 || predictions.isEmpty()) {
        return {};
    }

    QVector<DetectionPrediction> candidates;
    candidates.reserve(predictions.size());
    for (const DetectionPrediction& prediction : predictions) {
        if (prediction.confidence >= options.confidenceThreshold) {
            candidates.append(prediction);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const DetectionPrediction& left, const DetectionPrediction& right) {
        return left.confidence > right.confidence;
    });

    const double iouThreshold = qBound(0.0, options.iouThreshold, 1.0);
    QVector<DetectionPrediction> selected;
    selected.reserve(qMin(options.maxDetections, candidates.size()));
    for (const DetectionPrediction& candidate : candidates) {
        bool suppressed = false;
        for (const DetectionPrediction& accepted : selected) {
            if (candidate.box.classId == accepted.box.classId && boxIou(candidate.box, accepted.box) > iouThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            selected.append(candidate);
            if (selected.size() >= options.maxDetections) {
                break;
            }
        }
    }
    return selected;
}

QVector<OcrDetPrediction> postProcessPaddleOcrDetDbMap(
    const QVector<float>& probabilityMap,
    const QSize& mapSize,
    const QSize& sourceSize,
    const OcrDetPostprocessOptions& options,
    QString* error)
{
    if (mapSize.isEmpty() || sourceSize.isEmpty() || probabilityMap.size() != mapSize.width() * mapSize.height()) {
        if (error) {
            *error = QStringLiteral("OCR Det DB probability map size is invalid");
        }
        return {};
    }

    const double binaryThreshold = qBound(0.0, options.binaryThreshold, 1.0);
    const double boxThreshold = qBound(0.0, options.boxThreshold, 1.0);
    const int minArea = qMax(1, options.minArea);
    const int maxDetections = qMax(0, options.maxDetections);
    if (maxDetections == 0) {
        return {};
    }

    const int width = mapSize.width();
    const int height = mapSize.height();
    QVector<uchar> visited(width * height, 0);
    QVector<OcrDetPrediction> detections;

    auto indexOf = [width](int x, int y) {
        return y * width + x;
    };

    for (int startY = 0; startY < height; ++startY) {
        for (int startX = 0; startX < width; ++startX) {
            const int startIndex = indexOf(startX, startY);
            if (visited.at(startIndex) || probabilityMap.at(startIndex) < binaryThreshold) {
                continue;
            }

            QQueue<QPoint> queue;
            queue.enqueue(QPoint(startX, startY));
            visited[startIndex] = 1;
            int minX = startX;
            int maxX = startX;
            int minY = startY;
            int maxY = startY;
            int area = 0;
            double confidenceSum = 0.0;

            while (!queue.isEmpty()) {
                const QPoint point = queue.dequeue();
                const int pointIndex = indexOf(point.x(), point.y());
                ++area;
                confidenceSum += static_cast<double>(probabilityMap.at(pointIndex));
                minX = qMin(minX, point.x());
                maxX = qMax(maxX, point.x());
                minY = qMin(minY, point.y());
                maxY = qMax(maxY, point.y());

                const QPoint neighbors[] = {
                    QPoint(point.x() + 1, point.y()),
                    QPoint(point.x() - 1, point.y()),
                    QPoint(point.x(), point.y() + 1),
                    QPoint(point.x(), point.y() - 1)
                };
                for (const QPoint& neighbor : neighbors) {
                    if (neighbor.x() < 0 || neighbor.x() >= width || neighbor.y() < 0 || neighbor.y() >= height) {
                        continue;
                    }
                    const int neighborIndex = indexOf(neighbor.x(), neighbor.y());
                    if (visited.at(neighborIndex) || probabilityMap.at(neighborIndex) < binaryThreshold) {
                        continue;
                    }
                    visited[neighborIndex] = 1;
                    queue.enqueue(neighbor);
                }
            }

            const double confidence = area > 0 ? confidenceSum / static_cast<double>(area) : 0.0;
            if (area < minArea || confidence < boxThreshold) {
                continue;
            }

            const double sx = static_cast<double>(sourceSize.width()) / static_cast<double>(qMax(1, width));
            const double sy = static_cast<double>(sourceSize.height()) / static_cast<double>(qMax(1, height));
            const double left = qBound(0.0, static_cast<double>(minX) * sx, static_cast<double>(sourceSize.width()));
            const double top = qBound(0.0, static_cast<double>(minY) * sy, static_cast<double>(sourceSize.height()));
            const double right = qBound(0.0, static_cast<double>(maxX + 1) * sx, static_cast<double>(sourceSize.width()));
            const double bottom = qBound(0.0, static_cast<double>(maxY + 1) * sy, static_cast<double>(sourceSize.height()));

            OcrDetPrediction prediction;
            prediction.confidence = confidence;
            prediction.pixelArea = area;
            prediction.polygon = {
                QPointF(left, top),
                QPointF(right, top),
                QPointF(right, bottom),
                QPointF(left, bottom)
            };
            const double sourceWidth = qMax(1, sourceSize.width());
            const double sourceHeight = qMax(1, sourceSize.height());
            prediction.box.classId = 0;
            prediction.box.xCenter = clamp01((left + right) / 2.0 / sourceWidth);
            prediction.box.yCenter = clamp01((top + bottom) / 2.0 / sourceHeight);
            prediction.box.width = qBound(1.0e-6, (right - left) / sourceWidth, 1.0);
            prediction.box.height = qBound(1.0e-6, (bottom - top) / sourceHeight, 1.0);
            detections.append(prediction);
        }
    }

    std::sort(detections.begin(), detections.end(), [](const OcrDetPrediction& left, const OcrDetPrediction& right) {
        return left.confidence > right.confidence;
    });
    if (detections.size() > maxDetections) {
        detections.resize(maxDetections);
    }
    return detections;
}

} // namespace aitrain
