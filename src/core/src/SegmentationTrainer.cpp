#include "aitrain/core/SegmentationTrainer.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace aitrain {
namespace {

double maskCoverage(const QImage& mask)
{
    if (mask.isNull() || mask.width() <= 0 || mask.height() <= 0) {
        return 0.0;
    }

    int filled = 0;
    for (int y = 0; y < mask.height(); ++y) {
        for (int x = 0; x < mask.width(); ++x) {
            if (qAlpha(mask.pixel(x, y)) > 0) {
                ++filled;
            }
        }
    }
    return static_cast<double>(filled) / static_cast<double>(mask.width() * mask.height());
}

double sampleCoverage(const SegmentationSample& sample, const QSize& imageSize)
{
    if (sample.polygons.isEmpty()) {
        return 0.0;
    }

    double coverage = 0.0;
    for (const SegmentationPolygon& polygon : sample.polygons) {
        coverage += maskCoverage(polygonToMask(polygon.points, imageSize));
    }
    return coverage / static_cast<double>(sample.polygons.size());
}

QJsonArray pointArray(const QVector<QPointF>& points)
{
    QJsonArray array;
    for (const QPointF& point : points) {
        array.append(QJsonArray{point.x(), point.y()});
    }
    return array;
}

QJsonArray samplePolygonArray(const SegmentationSample& sample)
{
    QJsonArray array;
    for (const SegmentationPolygon& polygon : sample.polygons) {
        QJsonObject object;
        object.insert(QStringLiteral("classId"), polygon.classId);
        object.insert(QStringLiteral("points"), pointArray(polygon.points));
        array.append(object);
    }
    return array;
}

bool writeCheckpoint(
    const QString& path,
    const QString& datasetPath,
    const SegmentationDataset& trainDataset,
    const SegmentationTrainingOptions& options,
    const SegmentationTrainingResult& result,
    QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write segmentation checkpoint: %1").arg(path);
        }
        return false;
    }

    QJsonArray previewPolygons;
    if (!trainDataset.samples().isEmpty()) {
        previewPolygons = samplePolygonArray(trainDataset.samples().first());
    }

    QJsonObject checkpoint;
    checkpoint.insert(QStringLiteral("type"), QStringLiteral("tiny_mask_segmentation_scaffold"));
    checkpoint.insert(QStringLiteral("datasetPath"), QDir::cleanPath(datasetPath));
    checkpoint.insert(QStringLiteral("createdAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    checkpoint.insert(QStringLiteral("note"), QStringLiteral("Scaffold checkpoint for YOLO segmentation admission. Replace with native mask head weights before claiming real training."));
    checkpoint.insert(QStringLiteral("epochs"), options.epochs);
    checkpoint.insert(QStringLiteral("batchSize"), options.batchSize);
    checkpoint.insert(QStringLiteral("imageWidth"), options.imageSize.width());
    checkpoint.insert(QStringLiteral("imageHeight"), options.imageSize.height());
    checkpoint.insert(QStringLiteral("steps"), result.steps);
    checkpoint.insert(QStringLiteral("finalLoss"), result.finalLoss);
    checkpoint.insert(QStringLiteral("maskLoss"), result.finalLoss);
    checkpoint.insert(QStringLiteral("maskCoverage"), result.maskCoverage);
    checkpoint.insert(QStringLiteral("classNames"), QJsonArray::fromStringList(trainDataset.info().classNames));
    checkpoint.insert(QStringLiteral("previewPolygons"), previewPolygons);

    file.write(QJsonDocument(checkpoint).toJson(QJsonDocument::Indented));
    return true;
}

} // namespace

SegmentationTrainingResult trainSegmentationBaseline(
    const QString& datasetPath,
    const SegmentationTrainingOptions& options,
    const SegmentationTrainingCallback& callback)
{
    SegmentationTrainingResult result;

    QString error;
    SegmentationDataset trainDataset;
    if (!trainDataset.load(datasetPath, QStringLiteral("train"), &error)) {
        result.error = error;
        return result;
    }

    const int epochs = qMax(1, options.epochs);
    const int batchSize = qMax(1, options.batchSize);
    const QSize imageSize = options.imageSize.isValid() && !options.imageSize.isEmpty()
        ? options.imageSize
        : QSize(320, 320);
    const int batchesPerEpoch = qMax(1, (trainDataset.size() + batchSize - 1) / batchSize);
    const int totalSteps = epochs * batchesPerEpoch;
    const double learningRate = qMax(0.001, options.learningRate);

    double finalLoss = 0.0;
    double finalCoverage = 0.0;
    int step = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        const QVector<SegmentationSample> samples = trainDataset.samples();
        for (int batchStart = 0; batchStart < samples.size(); batchStart += batchSize) {
            ++step;

            double batchCoverage = 0.0;
            int batchSamples = 0;
            const int batchEnd = qMin(samples.size(), batchStart + batchSize);
            for (int index = batchStart; index < batchEnd; ++index) {
                batchCoverage += sampleCoverage(samples.at(index), imageSize);
                ++batchSamples;
            }
            if (batchSamples > 0) {
                batchCoverage /= static_cast<double>(batchSamples);
            }

            const double progress = static_cast<double>(step) / static_cast<double>(qMax(1, totalSteps));
            const double maskLoss = qMax(0.01, (1.0 - qMin(0.95, batchCoverage)) / (1.0 + learningRate * 10.0 * progress * static_cast<double>(epochs)));
            finalLoss = maskLoss;
            finalCoverage = batchCoverage;

            if (callback) {
                SegmentationTrainingMetrics metrics;
                metrics.epoch = epoch;
                metrics.step = step;
                metrics.totalSteps = totalSteps;
                metrics.loss = maskLoss;
                metrics.maskLoss = maskLoss;
                metrics.maskCoverage = batchCoverage;
                if (!callback(metrics)) {
                    result.error = QStringLiteral("Segmentation scaffold training canceled");
                    return result;
                }
            }
        }
    }

    QString outputPath = options.outputPath;
    if (outputPath.isEmpty()) {
        outputPath = QDir(datasetPath).filePath(QStringLiteral("runs/segmentation_scaffold"));
    }
    if (!QDir().mkpath(outputPath)) {
        result.error = QStringLiteral("Cannot create segmentation output directory: %1").arg(outputPath);
        return result;
    }

    result.ok = true;
    result.steps = step;
    result.finalLoss = finalLoss;
    result.maskCoverage = finalCoverage;
    result.checkpointPath = QDir(outputPath).filePath(QStringLiteral("checkpoint_latest.aitrain"));
    if (!writeCheckpoint(result.checkpointPath, datasetPath, trainDataset, options, result, &result.error)) {
        result.ok = false;
        return result;
    }

    SegmentationDataset previewDataset;
    if (!previewDataset.load(datasetPath, QStringLiteral("val"), &error)) {
        previewDataset = trainDataset;
    }
    if (!previewDataset.samples().isEmpty()) {
        const SegmentationSample sample = previewDataset.samples().first();
        const QImage preview = renderSegmentationOverlay(sample.imagePath, sample.polygons, &error);
        if (!preview.isNull()) {
            result.previewPath = QDir(outputPath).filePath(QStringLiteral("preview_latest.png"));
            preview.save(result.previewPath);
        }
    }

    return result;
}

} // namespace aitrain
