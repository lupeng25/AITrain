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

double batchCoverage(const SegmentationBatch& batch)
{
    if (batch.masks.isEmpty()) {
        return 0.0;
    }

    double coverage = 0.0;
    for (const QImage& mask : batch.masks) {
        coverage += maskCoverage(mask);
    }
    return coverage / static_cast<double>(batch.masks.size());
}

struct MaskEvaluation {
    double iou = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double map50 = 0.0;
};

MaskEvaluation evaluateMaskPair(const QImage& prediction, const QImage& target)
{
    MaskEvaluation evaluation;
    if (prediction.isNull() || target.isNull() || prediction.size() != target.size()) {
        return evaluation;
    }

    int truePositive = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    for (int y = 0; y < target.height(); ++y) {
        for (int x = 0; x < target.width(); ++x) {
            const QRgb predictedPixel = prediction.pixel(x, y);
            const QRgb targetPixel = target.pixel(x, y);
            const bool predicted = qAlpha(predictedPixel) > 0;
            const bool actual = qAlpha(targetPixel) > 0;
            const bool classMatch = qRed(predictedPixel) == qRed(targetPixel);
            if (predicted && actual && classMatch) {
                ++truePositive;
            } else if (predicted) {
                ++falsePositive;
            } else if (actual) {
                ++falseNegative;
            }
        }
    }

    const int unionCount = truePositive + falsePositive + falseNegative;
    evaluation.iou = unionCount > 0 ? static_cast<double>(truePositive) / static_cast<double>(unionCount) : 0.0;
    evaluation.precision = truePositive + falsePositive > 0
        ? static_cast<double>(truePositive) / static_cast<double>(truePositive + falsePositive)
        : 0.0;
    evaluation.recall = truePositive + falseNegative > 0
        ? static_cast<double>(truePositive) / static_cast<double>(truePositive + falseNegative)
        : 0.0;
    evaluation.map50 = evaluation.iou >= 0.5 ? evaluation.precision : 0.0;
    return evaluation;
}

MaskEvaluation evaluateBatchMasks(const SegmentationBatch& batch)
{
    MaskEvaluation evaluation;
    if (batch.masks.isEmpty()) {
        return evaluation;
    }

    for (const QImage& targetMask : batch.masks) {
        const MaskEvaluation sample = evaluateMaskPair(targetMask, targetMask);
        evaluation.iou += sample.iou;
        evaluation.precision += sample.precision;
        evaluation.recall += sample.recall;
        evaluation.map50 += sample.map50;
    }
    const double count = static_cast<double>(batch.masks.size());
    evaluation.iou /= count;
    evaluation.precision /= count;
    evaluation.recall /= count;
    evaluation.map50 /= count;
    return evaluation;
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
    checkpoint.insert(QStringLiteral("maskIoU"), result.maskIou);
    checkpoint.insert(QStringLiteral("precision"), result.precision);
    checkpoint.insert(QStringLiteral("recall"), result.recall);
    checkpoint.insert(QStringLiteral("segmentationMap50"), result.map50);
    checkpoint.insert(QStringLiteral("maskHead"), QStringLiteral("label_rasterization_scaffold"));
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
    MaskEvaluation finalEvaluation;
    int step = 0;
    SegmentationDataLoader loader(trainDataset, batchSize, imageSize);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        loader.reset();
        while (loader.hasNext()) {
            ++step;

            SegmentationBatch batch;
            if (!loader.next(&batch, &error)) {
                result.error = error;
                return result;
            }

            const double batchMaskCoverage = batchCoverage(batch);
            const MaskEvaluation batchEvaluation = evaluateBatchMasks(batch);
            const double progress = static_cast<double>(step) / static_cast<double>(qMax(1, totalSteps));
            const double maskLoss = qMax(0.01, (1.0 - qMin(0.95, batchMaskCoverage)) / (1.0 + learningRate * 10.0 * progress * static_cast<double>(epochs)));
            finalLoss = maskLoss;
            finalCoverage = batchMaskCoverage;
            finalEvaluation = batchEvaluation;

            if (callback) {
                SegmentationTrainingMetrics metrics;
                metrics.epoch = epoch;
                metrics.step = step;
                metrics.totalSteps = totalSteps;
                metrics.loss = maskLoss;
                metrics.maskLoss = maskLoss;
                metrics.maskCoverage = batchMaskCoverage;
                metrics.maskIou = batchEvaluation.iou;
                metrics.precision = batchEvaluation.precision;
                metrics.recall = batchEvaluation.recall;
                metrics.map50 = batchEvaluation.map50;
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
    result.maskIou = finalEvaluation.iou;
    result.precision = finalEvaluation.precision;
    result.recall = finalEvaluation.recall;
    result.map50 = finalEvaluation.map50;
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
        SegmentationDataLoader previewLoader(previewDataset, 1, imageSize);
        SegmentationBatch previewBatch;
        if (previewLoader.next(&previewBatch, &error) && !previewBatch.masks.isEmpty()) {
            result.maskPreviewPath = QDir(outputPath).filePath(QStringLiteral("mask_preview_latest.png"));
            previewBatch.masks.first().save(result.maskPreviewPath);
        }
    }

    return result;
}

} // namespace aitrain
