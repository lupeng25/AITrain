#include "aitrain/core/ProductWorkflow.h"

#include "ProductWorkflowSupport.h"
#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/OcrRecDataset.h"
#include "aitrain/core/SegmentationDataset.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMap>
#include <QRegularExpression>
#include <QSet>
#include <QTextStream>
#include <QThread>

#include <algorithm>
namespace aitrain {
using namespace workflow_detail;
namespace {
double boxArea(const DetectionBox& box)
{
    return qMax(0.0, box.width) * qMax(0.0, box.height);
}

double boxIou(const DetectionBox& left, const DetectionBox& right)
{
    const double leftX1 = left.xCenter - left.width / 2.0;
    const double leftY1 = left.yCenter - left.height / 2.0;
    const double leftX2 = left.xCenter + left.width / 2.0;
    const double leftY2 = left.yCenter + left.height / 2.0;
    const double rightX1 = right.xCenter - right.width / 2.0;
    const double rightY1 = right.yCenter - right.height / 2.0;
    const double rightX2 = right.xCenter + right.width / 2.0;
    const double rightY2 = right.yCenter + right.height / 2.0;
    const double intersectionWidth = qMax(0.0, qMin(leftX2, rightX2) - qMax(leftX1, rightX1));
    const double intersectionHeight = qMax(0.0, qMin(leftY2, rightY2) - qMax(leftY1, rightY1));
    const double intersection = intersectionWidth * intersectionHeight;
    const double areaSum = boxArea(left) + boxArea(right) - intersection;
    return areaSum > 0.0 ? intersection / areaSum : 0.0;
}

QJsonObject detectionBoxToJson(const DetectionBox& box)
{
    return QJsonObject{
        {QStringLiteral("classId"), box.classId},
        {QStringLiteral("xCenter"), box.xCenter},
        {QStringLiteral("yCenter"), box.yCenter},
        {QStringLiteral("width"), box.width},
        {QStringLiteral("height"), box.height}
    };
}

QJsonObject evaluationDecisionSummary(
    const QString& taskType,
    const QJsonObject& metrics,
    const QJsonArray& errorSamples,
    int sampleCount)
{
    QString primaryMetricName;
    if (taskType == QStringLiteral("segmentation")) {
        primaryMetricName = QStringLiteral("maskMap50");
    } else if (taskType == QStringLiteral("ocr_recognition") || taskType == QStringLiteral("ocr")) {
        primaryMetricName = QStringLiteral("accuracy");
    } else {
        primaryMetricName = QStringLiteral("mAP50");
    }
    const double primaryMetric = metrics.value(primaryMetricName).toDouble();
    const bool hasSamples = sampleCount > 0;
    const bool hasErrors = !errorSamples.isEmpty();

    QJsonArray actions;
    QString status;
    if (!hasSamples) {
        status = QStringLiteral("blocked");
        actions.append(QStringLiteral("provide_evaluation_samples"));
    } else if (hasErrors) {
        status = QStringLiteral("needs_error_review");
        actions.append(QStringLiteral("review_error_samples"));
        actions.append(QStringLiteral("feed_errors_back_to_dataset_quality"));
    } else {
        status = QStringLiteral("accepted_for_local_smoke");
        actions.append(QStringLiteral("run_benchmark"));
        actions.append(QStringLiteral("generate_delivery_report"));
    }

    QJsonObject decision;
    decision.insert(QStringLiteral("schemaVersion"), 1);
    decision.insert(QStringLiteral("taskType"), taskType);
    decision.insert(QStringLiteral("status"), status);
    decision.insert(QStringLiteral("primaryMetric"), primaryMetricName);
    decision.insert(QStringLiteral("primaryMetricValue"), primaryMetric);
    decision.insert(QStringLiteral("sampleCount"), sampleCount);
    decision.insert(QStringLiteral("errorSampleCount"), errorSamples.size());
    decision.insert(QStringLiteral("recommendedActions"), actions);
    return decision;
}

QJsonObject errorTaxonomyObject(
    const QString& taskType,
    const QJsonObject& metrics,
    const QJsonArray& errorSamples,
    const QJsonArray& lowConfidenceSamples = {})
{
    QJsonObject reasonCounts;
    for (const QJsonValue& value : errorSamples) {
        const QJsonObject sample = value.toObject();
        const QString reason = sample.value(QStringLiteral("reason")).toString(
            taskType == QStringLiteral("ocr_recognition") ? QStringLiteral("ocr_mismatch") : QStringLiteral("unknown"));
        reasonCounts.insert(reason, reasonCounts.value(reason).toInt() + 1);
    }

    QJsonObject taxonomy;
    taxonomy.insert(QStringLiteral("schemaVersion"), 1);
    taxonomy.insert(QStringLiteral("taskType"), taskType);
    taxonomy.insert(QStringLiteral("sampleErrorCount"), errorSamples.size());
    taxonomy.insert(QStringLiteral("lowConfidenceCount"), lowConfidenceSamples.size());
    taxonomy.insert(QStringLiteral("reasonCounts"), reasonCounts);
    taxonomy.insert(QStringLiteral("falsePositiveCount"), metrics.value(QStringLiteral("fp")).toInt());
    taxonomy.insert(QStringLiteral("falseNegativeCount"), metrics.value(QStringLiteral("fn")).toInt());
    taxonomy.insert(QStringLiteral("truePositiveCount"), metrics.value(QStringLiteral("tp")).toInt());
    return taxonomy;
}

QString jsonValueText(const QJsonValue& value)
{
    if (value.isDouble()) {
        return QString::number(value.toDouble(), 'g', 12);
    }
    if (value.isBool()) {
        return value.toBool() ? QStringLiteral("true") : QStringLiteral("false");
    }
    if (value.isString()) {
        return value.toString();
    }
    const QString compact = QString::fromUtf8(QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact));
    return compact.mid(1, qMax(0, compact.size() - 2));
}

QString evaluationSummaryMarkdown(const QJsonObject& report)
{
    const QJsonObject decision = report.value(QStringLiteral("decisionSummary")).toObject();
    const QJsonObject metrics = report.value(QStringLiteral("metrics")).toObject();
    const QJsonObject taxonomy = report.value(QStringLiteral("errorTaxonomy")).toObject();

    QString markdown;
    markdown += QStringLiteral("# Evaluation Summary\n\n");
    markdown += QStringLiteral("- Task type: %1\n").arg(report.value(QStringLiteral("taskType")).toString());
    markdown += QStringLiteral("- Status: %1\n").arg(decision.value(QStringLiteral("status")).toString());
    markdown += QStringLiteral("- Primary metric: %1=%2\n")
        .arg(decision.value(QStringLiteral("primaryMetric")).toString())
        .arg(decision.value(QStringLiteral("primaryMetricValue")).toDouble(), 0, 'f', 6);
    markdown += QStringLiteral("- Samples: %1\n").arg(report.value(QStringLiteral("sampleCount")).toInt());
    markdown += QStringLiteral("- Error samples: %1\n\n").arg(decision.value(QStringLiteral("errorSampleCount")).toInt());
    markdown += QStringLiteral("## Metrics\n\n");
    for (auto it = metrics.constBegin(); it != metrics.constEnd(); ++it) {
        markdown += QStringLiteral("- %1: %2\n").arg(it.key(), jsonValueText(it.value()));
    }
    markdown += QStringLiteral("\n## Error Taxonomy\n\n");
    markdown += QStringLiteral("- False positives: %1\n").arg(taxonomy.value(QStringLiteral("falsePositiveCount")).toInt());
    markdown += QStringLiteral("- False negatives: %1\n").arg(taxonomy.value(QStringLiteral("falseNegativeCount")).toInt());
    markdown += QStringLiteral("- Low confidence samples: %1\n").arg(taxonomy.value(QStringLiteral("lowConfidenceCount")).toInt());
    markdown += QStringLiteral("\nThis summary is generated from local evaluation artifacts. Inspect the JSON report for full per-sample details.\n");
    return markdown;
}

bool detectionSplitExists(const QString& datasetPath, const QString& split)
{
    const QDir root(datasetPath);
    return QDir(root.filePath(QStringLiteral("images/%1").arg(split))).exists()
        && QDir(root.filePath(QStringLiteral("labels/%1").arg(split))).exists();
}

QString selectDetectionSplit(const QString& datasetPath)
{
    for (const QString& split : {QStringLiteral("val"), QStringLiteral("test"), QStringLiteral("train")}) {
        if (detectionSplitExists(datasetPath, split)) {
            return split;
        }
    }
    return QString();
}

QVector<DetectionPrediction> runDetectionPredictions(
    const QString& modelPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* runtime,
    QString* error)
{
    const QString suffix = QFileInfo(modelPath).suffix().toLower();
    if (suffix == QStringLiteral("onnx")) {
        if (runtime) {
            *runtime = QStringLiteral("onnxruntime");
        }
        return predictDetectionOnnxRuntime(modelPath, imagePath, options, error);
    }
    if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan")) {
        if (runtime) {
            *runtime = QStringLiteral("tensorrt");
        }
        if (!isTensorRtInferenceAvailable()) {
            if (error) {
                *error = QStringLiteral("hardware-blocked: TensorRT evaluation requires a compatible RTX / SM 75+ acceptance machine and runtime.");
            }
            return {};
        }
        return predictDetectionTensorRt(modelPath, imagePath, options, error);
    }

    DetectionBaselineCheckpoint checkpoint;
    if (!loadDetectionBaselineCheckpoint(modelPath, &checkpoint, error)) {
        return {};
    }
    if (runtime) {
        *runtime = QStringLiteral("tiny_detector");
    }
    return predictDetectionBaseline(checkpoint, imagePath, options, error);
}

struct DetectionEvaluationItem {
    int classId = 0;
    double confidence = 0.0;
    bool truePositive = false;
};

struct DetectionClassStats {
    int gt = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double precision = 0.0;
    double recall = 0.0;
    double ap50 = 0.0;
    double map5095 = 0.0;
    QVector<DetectionEvaluationItem> items;
};

struct DetectionMapSample {
    QVector<DetectionBox> groundTruth;
    QVector<DetectionPrediction> predictions;
};

struct SegmentationMapGroundTruth {
    int classId = 0;
    QImage mask;
};

struct SegmentationMapSample {
    QVector<SegmentationMapGroundTruth> groundTruth;
    QVector<SegmentationPrediction> predictions;
};

struct CocoMapMetrics {
    double map50 = 0.0;
    double map5095 = 0.0;
    QVector<double> perClassMap50;
    QVector<double> perClassMap5095;
    QJsonArray thresholds;
};

double ap50FromItems(QVector<DetectionEvaluationItem> items, int gtCount)
{
    if (gtCount <= 0) {
        return 0.0;
    }
    std::sort(items.begin(), items.end(), [](const DetectionEvaluationItem& left, const DetectionEvaluationItem& right) {
        return left.confidence > right.confidence;
    });

    QVector<double> recalls;
    QVector<double> precisions;
    int tp = 0;
    int fp = 0;
    for (const DetectionEvaluationItem& item : items) {
        if (item.truePositive) {
            ++tp;
        } else {
            ++fp;
        }
        recalls.append(static_cast<double>(tp) / static_cast<double>(gtCount));
        precisions.append(static_cast<double>(tp) / static_cast<double>(qMax(1, tp + fp)));
    }

    double ap = 0.0;
    for (int threshold = 0; threshold <= 100; ++threshold) {
        const double recallThreshold = static_cast<double>(threshold) / 100.0;
        double precisionAtRecall = 0.0;
        for (int index = 0; index < recalls.size(); ++index) {
            if (recalls.at(index) >= recallThreshold) {
                precisionAtRecall = qMax(precisionAtRecall, precisions.at(index));
            }
        }
        ap += precisionAtRecall;
    }
    return ap / 101.0;
}

QVector<double> cocoMapThresholds()
{
    QVector<double> thresholds;
    for (int step = 50; step <= 95; step += 5) {
        thresholds.append(static_cast<double>(step) / 100.0);
    }
    return thresholds;
}

CocoMapMetrics detectionCocoMapMetrics(const QVector<DetectionMapSample>& samples, int classCount)
{
    CocoMapMetrics metrics;
    metrics.perClassMap50 = QVector<double>(classCount, 0.0);
    metrics.perClassMap5095 = QVector<double>(classCount, 0.0);
    const QVector<double> thresholds = cocoMapThresholds();
    QJsonArray thresholdArray;
    QVector<int> classGtCounts(classCount, 0);
    for (const DetectionMapSample& sample : samples) {
        for (const DetectionBox& gt : sample.groundTruth) {
            if (gt.classId >= 0 && gt.classId < classCount) {
                ++classGtCounts[gt.classId];
            }
        }
    }

    for (const double threshold : thresholds) {
        QJsonArray perClassThresholdArray;
        double thresholdMap = 0.0;
        int thresholdClassCount = 0;
        for (int classId = 0; classId < classCount; ++classId) {
            QVector<DetectionEvaluationItem> items;
            for (const DetectionMapSample& sample : samples) {
                QVector<int> gtIndexes;
                for (int index = 0; index < sample.groundTruth.size(); ++index) {
                    if (sample.groundTruth.at(index).classId == classId) {
                        gtIndexes.append(index);
                    }
                }
                QVector<bool> matched(gtIndexes.size(), false);
                QVector<DetectionPrediction> predictions;
                for (const DetectionPrediction& prediction : sample.predictions) {
                    if (prediction.box.classId == classId) {
                        predictions.append(prediction);
                    }
                }
                std::sort(predictions.begin(), predictions.end(), [](const DetectionPrediction& left, const DetectionPrediction& right) {
                    return left.confidence > right.confidence;
                });
                for (const DetectionPrediction& prediction : predictions) {
                    int bestMatch = -1;
                    double bestIou = 0.0;
                    for (int localIndex = 0; localIndex < gtIndexes.size(); ++localIndex) {
                        if (matched.at(localIndex)) {
                            continue;
                        }
                        const double iou = boxIou(prediction.box, sample.groundTruth.at(gtIndexes.at(localIndex)));
                        if (iou > bestIou) {
                            bestIou = iou;
                            bestMatch = localIndex;
                        }
                    }
                    const bool truePositive = bestMatch >= 0 && bestIou >= threshold;
                    if (truePositive) {
                        matched[bestMatch] = true;
                    }
                    items.append(DetectionEvaluationItem{classId, prediction.confidence, truePositive});
                }
            }
            const double ap = ap50FromItems(items, classGtCounts.value(classId));
            if (qFuzzyCompare(threshold, 0.5)) {
                metrics.perClassMap50[classId] = ap;
            }
            metrics.perClassMap5095[classId] += ap;
            if (classGtCounts.value(classId) > 0) {
                thresholdMap += ap;
                ++thresholdClassCount;
            }
            perClassThresholdArray.append(QJsonObject{
                {QStringLiteral("classId"), classId},
                {QStringLiteral("gt"), classGtCounts.value(classId)},
                {QStringLiteral("ap"), ap}
            });
        }
        const double thresholdAverage = thresholdClassCount > 0
            ? thresholdMap / static_cast<double>(thresholdClassCount)
            : 0.0;
        thresholdArray.append(QJsonObject{
            {QStringLiteral("iouThreshold"), threshold},
            {QStringLiteral("mAP"), thresholdAverage},
            {QStringLiteral("perClass"), perClassThresholdArray}
        });
        if (qFuzzyCompare(threshold, 0.5)) {
            metrics.map50 = thresholdAverage;
        }
        metrics.map5095 += thresholdAverage;
    }

    for (int classId = 0; classId < metrics.perClassMap5095.size(); ++classId) {
        metrics.perClassMap5095[classId] = thresholds.isEmpty()
            ? 0.0
            : metrics.perClassMap5095.at(classId) / static_cast<double>(thresholds.size());
    }
    metrics.map5095 = thresholds.isEmpty() ? 0.0 : metrics.map5095 / static_cast<double>(thresholds.size());
    metrics.thresholds = thresholdArray;
    return metrics;
}

double maskIou(const QImage& leftMaskImage, const QImage& rightMaskImage);

CocoMapMetrics segmentationCocoMapMetrics(const QVector<SegmentationMapSample>& samples, int classCount)
{
    CocoMapMetrics metrics;
    metrics.perClassMap50 = QVector<double>(classCount, 0.0);
    metrics.perClassMap5095 = QVector<double>(classCount, 0.0);
    const QVector<double> thresholds = cocoMapThresholds();
    QJsonArray thresholdArray;
    QVector<int> classGtCounts(classCount, 0);
    for (const SegmentationMapSample& sample : samples) {
        for (const SegmentationMapGroundTruth& gt : sample.groundTruth) {
            if (gt.classId >= 0 && gt.classId < classCount) {
                ++classGtCounts[gt.classId];
            }
        }
    }

    for (const double threshold : thresholds) {
        QJsonArray perClassThresholdArray;
        double thresholdMap = 0.0;
        int thresholdClassCount = 0;
        for (int classId = 0; classId < classCount; ++classId) {
            QVector<DetectionEvaluationItem> items;
            for (const SegmentationMapSample& sample : samples) {
                QVector<int> gtIndexes;
                for (int index = 0; index < sample.groundTruth.size(); ++index) {
                    if (sample.groundTruth.at(index).classId == classId) {
                        gtIndexes.append(index);
                    }
                }
                QVector<bool> matched(gtIndexes.size(), false);
                QVector<SegmentationPrediction> predictions;
                for (const SegmentationPrediction& prediction : sample.predictions) {
                    if (prediction.detection.box.classId == classId) {
                        predictions.append(prediction);
                    }
                }
                std::sort(predictions.begin(), predictions.end(), [](const SegmentationPrediction& left, const SegmentationPrediction& right) {
                    return left.detection.confidence > right.detection.confidence;
                });
                for (const SegmentationPrediction& prediction : predictions) {
                    int bestMatch = -1;
                    double bestIou = 0.0;
                    for (int localIndex = 0; localIndex < gtIndexes.size(); ++localIndex) {
                        if (matched.at(localIndex)) {
                            continue;
                        }
                        const double iou = maskIou(prediction.mask, sample.groundTruth.at(gtIndexes.at(localIndex)).mask);
                        if (iou > bestIou) {
                            bestIou = iou;
                            bestMatch = localIndex;
                        }
                    }
                    const bool truePositive = bestMatch >= 0 && bestIou >= threshold;
                    if (truePositive) {
                        matched[bestMatch] = true;
                    }
                    items.append(DetectionEvaluationItem{classId, prediction.detection.confidence, truePositive});
                }
            }
            const double ap = ap50FromItems(items, classGtCounts.value(classId));
            if (qFuzzyCompare(threshold, 0.5)) {
                metrics.perClassMap50[classId] = ap;
            }
            metrics.perClassMap5095[classId] += ap;
            if (classGtCounts.value(classId) > 0) {
                thresholdMap += ap;
                ++thresholdClassCount;
            }
            perClassThresholdArray.append(QJsonObject{
                {QStringLiteral("classId"), classId},
                {QStringLiteral("gt"), classGtCounts.value(classId)},
                {QStringLiteral("ap"), ap}
            });
        }
        const double thresholdAverage = thresholdClassCount > 0
            ? thresholdMap / static_cast<double>(thresholdClassCount)
            : 0.0;
        thresholdArray.append(QJsonObject{
            {QStringLiteral("iouThreshold"), threshold},
            {QStringLiteral("mAP"), thresholdAverage},
            {QStringLiteral("perClass"), perClassThresholdArray}
        });
        if (qFuzzyCompare(threshold, 0.5)) {
            metrics.map50 = thresholdAverage;
        }
        metrics.map5095 += thresholdAverage;
    }

    for (int classId = 0; classId < metrics.perClassMap5095.size(); ++classId) {
        metrics.perClassMap5095[classId] = thresholds.isEmpty()
            ? 0.0
            : metrics.perClassMap5095.at(classId) / static_cast<double>(thresholds.size());
    }
    metrics.map5095 = thresholds.isEmpty() ? 0.0 : metrics.map5095 / static_cast<double>(thresholds.size());
    metrics.thresholds = thresholdArray;
    return metrics;
}

QString perClassMetricsCsv(const QStringList& classNames, const QVector<DetectionClassStats>& stats)
{
    QString csv = QStringLiteral("classId,className,gt,tp,fp,fn,precision,recall,ap50,map50_95\n");
    for (int classId = 0; classId < stats.size(); ++classId) {
        const DetectionClassStats& item = stats.at(classId);
        const QString className = classId < classNames.size() && !classNames.at(classId).isEmpty()
            ? classNames.at(classId)
            : QStringLiteral("class_%1").arg(classId);
        csv += QStringLiteral("%1,%2,%3,%4,%5,%6,%7,%8,%9\n")
            .arg(classId)
            .arg(csvEscape(className))
            .arg(item.gt)
            .arg(item.tp)
            .arg(item.fp)
            .arg(item.fn)
            .arg(item.precision, 0, 'f', 6)
            .arg(item.recall, 0, 'f', 6)
            .arg(item.ap50, 0, 'f', 6)
            .arg(item.map5095, 0, 'f', 6);
    }
    return csv;
}

QString confusionMatrixCsv(const QStringList& classNames, const QVector<QVector<int>>& matrix)
{
    QString csv = QStringLiteral("actual\\predicted");
    for (int classId = 0; classId < classNames.size(); ++classId) {
        csv += QStringLiteral(",%1").arg(csvEscape(classNames.at(classId)));
    }
    csv += QStringLiteral(",background\n");
    for (int row = 0; row < matrix.size(); ++row) {
        const QString rowName = row < classNames.size()
            ? classNames.at(row)
            : QStringLiteral("background");
        csv += csvEscape(rowName);
        for (int column = 0; column < matrix.at(row).size(); ++column) {
            csv += QStringLiteral(",%1").arg(matrix.at(row).at(column));
        }
        csv += QLatin1Char('\n');
    }
    return csv;
}

QJsonArray segmentationPolygonPointsToJson(const QVector<QPointF>& points)
{
    QJsonArray array;
    for (const QPointF& point : points) {
        array.append(QJsonObject{
            {QStringLiteral("x"), point.x()},
            {QStringLiteral("y"), point.y()}
        });
    }
    return array;
}

QJsonObject segmentationPolygonToJson(const SegmentationPolygon& polygon)
{
    return QJsonObject{
        {QStringLiteral("classId"), polygon.classId},
        {QStringLiteral("points"), segmentationPolygonPointsToJson(polygon.points)}
    };
}

double maskIou(const QImage& leftMaskImage, const QImage& rightMaskImage)
{
    if (leftMaskImage.isNull() || rightMaskImage.isNull()) {
        return 0.0;
    }
    QImage left = leftMaskImage.convertToFormat(QImage::Format_ARGB32);
    QImage right = rightMaskImage.convertToFormat(QImage::Format_ARGB32);
    if (left.size() != right.size()) {
        right = right.scaled(left.size(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
    }

    int intersection = 0;
    int unionPixels = 0;
    for (int y = 0; y < left.height(); ++y) {
        const QRgb* leftLine = reinterpret_cast<const QRgb*>(left.constScanLine(y));
        const QRgb* rightLine = reinterpret_cast<const QRgb*>(right.constScanLine(y));
        for (int x = 0; x < left.width(); ++x) {
            const bool leftActive = qAlpha(leftLine[x]) > 0;
            const bool rightActive = qAlpha(rightLine[x]) > 0;
            if (leftActive || rightActive) {
                ++unionPixels;
                if (leftActive && rightActive) {
                    ++intersection;
                }
            }
        }
    }
    return unionPixels > 0 ? static_cast<double>(intersection) / static_cast<double>(unionPixels) : 0.0;
}

QString segmentationPerClassMetricsCsv(
    const QStringList& classNames,
    const QVector<DetectionClassStats>& stats,
    const QVector<double>& maskIouSums,
    const QVector<int>& maskIouCounts)
{
    QString csv = QStringLiteral("classId,className,gt,tp,fp,fn,precision,recall,maskIoU,maskAP50,maskMap50_95\n");
    for (int classId = 0; classId < stats.size(); ++classId) {
        const DetectionClassStats& item = stats.at(classId);
        const QString className = classId < classNames.size() && !classNames.at(classId).isEmpty()
            ? classNames.at(classId)
            : QStringLiteral("class_%1").arg(classId);
        const double classMaskIou = classId < maskIouCounts.size() && maskIouCounts.at(classId) > 0
            ? maskIouSums.at(classId) / static_cast<double>(maskIouCounts.at(classId))
            : 0.0;
        csv += QStringLiteral("%1,%2,%3,%4,%5,%6,%7,%8,%9,%10\n")
            .arg(classId)
            .arg(csvEscape(className))
            .arg(item.gt)
            .arg(item.tp)
            .arg(item.fp)
            .arg(item.fn)
            .arg(item.precision, 0, 'f', 6)
            .arg(item.recall, 0, 'f', 6)
            .arg(classMaskIou, 0, 'f', 6)
            .arg(item.ap50, 0, 'f', 6)
            .arg(item.map5095, 0, 'f', 6);
    }
    return csv;
}

int stringEditDistance(const QString& left, const QString& right)
{
    const int leftLength = left.size();
    const int rightLength = right.size();
    QVector<int> previous(rightLength + 1, 0);
    QVector<int> current(rightLength + 1, 0);
    for (int j = 0; j <= rightLength; ++j) {
        previous[j] = j;
    }
    for (int i = 1; i <= leftLength; ++i) {
        current[0] = i;
        for (int j = 1; j <= rightLength; ++j) {
            const int substitutionCost = left.at(i - 1) == right.at(j - 1) ? 0 : 1;
            current[j] = qMin(
                qMin(previous[j] + 1, current[j - 1] + 1),
                previous[j - 1] + substitutionCost);
        }
        previous.swap(current);
    }
    return previous.at(rightLength);
}

QStringList splitOcrWords(const QString& text)
{
    const QString normalized = text.simplified();
    if (normalized.isEmpty()) {
        return {};
    }
    return normalized.split(QLatin1Char(' '), QString::SkipEmptyParts);
}

int wordEditDistance(const QStringList& left, const QStringList& right)
{
    const int leftLength = left.size();
    const int rightLength = right.size();
    QVector<int> previous(rightLength + 1, 0);
    QVector<int> current(rightLength + 1, 0);
    for (int j = 0; j <= rightLength; ++j) {
        previous[j] = j;
    }
    for (int i = 1; i <= leftLength; ++i) {
        current[0] = i;
        for (int j = 1; j <= rightLength; ++j) {
            const int substitutionCost = left.at(i - 1) == right.at(j - 1) ? 0 : 1;
            current[j] = qMin(
                qMin(previous[j] + 1, current[j - 1] + 1),
                previous[j - 1] + substitutionCost);
        }
        previous.swap(current);
    }
    return previous.at(rightLength);
}

QString resolveOcrPathOption(const QString& datasetPath, const QString& configuredPath)
{
    if (configuredPath.isEmpty()) {
        return QString();
    }
    const QFileInfo info(configuredPath);
    return info.isAbsolute()
        ? QDir::cleanPath(configuredPath)
        : QDir(datasetPath).filePath(configuredPath);
}

QString resolveOcrLabelFilePath(const QString& datasetPath, const QJsonObject& options)
{
    const QString configured = resolveOcrPathOption(datasetPath, options.value(QStringLiteral("labelFile")).toString());
    if (!configured.isEmpty()) {
        return configured;
    }
    const QDir root(datasetPath);
    const QString requestedSplit = options.value(QStringLiteral("split")).toString().trimmed();
    if (!requestedSplit.isEmpty()) {
        const QString splitPath = root.filePath(QStringLiteral("rec_gt_%1.txt").arg(requestedSplit));
        if (QFileInfo::exists(splitPath)) {
            return splitPath;
        }
    }
    for (const QString& candidate : {
             QStringLiteral("rec_gt_val.txt"),
             QStringLiteral("rec_gt_test.txt"),
             QStringLiteral("rec_gt_train.txt"),
             QStringLiteral("rec_gt.txt")}) {
        const QString path = root.filePath(candidate);
        if (QFileInfo::exists(path)) {
            return path;
        }
    }
    return root.filePath(QStringLiteral("rec_gt.txt"));
}

QString resolveOcrDictionaryPath(const QString& datasetPath, const QJsonObject& options)
{
    const QString configured = resolveOcrPathOption(datasetPath, options.value(QStringLiteral("dictionaryFile")).toString());
    if (!configured.isEmpty()) {
        return configured;
    }
    return QDir(datasetPath).filePath(QStringLiteral("dict.txt"));
}
} // namespace
WorkflowResult evaluateModelReport(const QString& modelPath, const QString& datasetPath, const QString& outputPath, const QString& taskType, const QJsonObject& options)
{
    const QFileInfo modelInfo(modelPath);
    if (!modelInfo.exists()) {
        return failedResult(QStringLiteral("Model file does not exist: %1").arg(modelPath));
    }
    const QDir datasetRoot(datasetPath);
    if (!datasetRoot.exists()) {
        return failedResult(QStringLiteral("Dataset directory does not exist: %1").arg(datasetPath));
    }

    if (taskType == QStringLiteral("detection") || taskType == QStringLiteral("yolo_detection")) {
        const QString split = options.value(QStringLiteral("split")).toString(selectDetectionSplit(datasetPath));
        if (split.isEmpty()) {
            return failedResult(QStringLiteral("No detection split found. Expected images/val, images/test, or images/train with matching labels."));
        }

        DetectionDataset dataset;
        QString error;
        if (!dataset.load(datasetPath, split, &error)) {
            return failedResult(error);
        }

        DetectionInferenceOptions inferenceOptions;
        inferenceOptions.iouThreshold = options.value(QStringLiteral("nmsIouThreshold")).toDouble(0.45);
        inferenceOptions.confidenceThreshold = options.value(QStringLiteral("confidenceThreshold")).toDouble(0.001);
        inferenceOptions.maxDetections = options.value(QStringLiteral("maxDetections")).toInt(100);
        const double matchIouThreshold = options.value(QStringLiteral("iouThreshold")).toDouble(0.5);
        const double lowConfidenceThreshold = options.value(QStringLiteral("lowConfidenceThreshold")).toDouble(0.25);
        const int maxErrorSamples = options.value(QStringLiteral("maxErrorSamples")).toInt(200);
        const int maxOverlaySamples = options.value(QStringLiteral("maxOverlaySamples")).toInt(50);

        QStringList classNames = dataset.info().classNames;
        const int classCount = qMax(dataset.info().classCount, classNames.size());
        while (classNames.size() < classCount) {
            classNames.append(QStringLiteral("class_%1").arg(classNames.size()));
        }
        QVector<DetectionClassStats> classStats(classCount);
        QVector<QVector<int>> confusion(classCount + 1, QVector<int>(classCount + 1, 0));
        QJsonArray sampleSummaries;
        QJsonArray errorSamples;
        QJsonArray lowConfidenceSamples;
        QVector<DetectionMapSample> mapSamples;
        QString runtime = QStringLiteral("unknown");
        int totalGt = 0;
        int totalPredictions = 0;
        int totalTp = 0;
        int totalFp = 0;
        int totalFn = 0;
        int overlayCount = 0;
        const QDir outputDir(outputPath);
        QDir().mkpath(outputDir.filePath(QStringLiteral("overlays")));

        for (const DetectionSample& sample : dataset.samples()) {
            QString predictionError;
            const QVector<DetectionPrediction> predictions = runDetectionPredictions(modelPath, sample.imagePath, inferenceOptions, &runtime, &predictionError);
            if (!predictionError.isEmpty()) {
                if (predictionError.startsWith(QStringLiteral("hardware-blocked"))) {
                    QJsonObject blocked;
                    blocked.insert(QStringLiteral("ok"), false);
                    blocked.insert(QStringLiteral("status"), QStringLiteral("hardware-blocked"));
                    blocked.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
                    blocked.insert(QStringLiteral("createdAt"), nowIso());
                    blocked.insert(QStringLiteral("modelPath"), modelPath);
                    blocked.insert(QStringLiteral("datasetPath"), datasetPath);
                    blocked.insert(QStringLiteral("taskType"), taskType);
                    blocked.insert(QStringLiteral("runtime"), runtime);
                    blocked.insert(QStringLiteral("scaffold"), false);
                    blocked.insert(QStringLiteral("message"), predictionError);
                    const QString reportPath = outputDir.filePath(QStringLiteral("evaluation_report.json"));
                    if (!writeJsonFile(reportPath, blocked, &error)) {
                        return failedResult(error);
                    }
                    blocked.insert(QStringLiteral("reportPath"), reportPath);
                    return resultFromReport(reportPath, blocked);
                }
                return failedResult(predictionError);
            }

            totalGt += sample.boxes.size();
            totalPredictions += predictions.size();
            mapSamples.append(DetectionMapSample{sample.boxes, predictions});
            for (const DetectionBox& gt : sample.boxes) {
                if (gt.classId >= 0 && gt.classId < classStats.size()) {
                    classStats[gt.classId].gt += 1;
                }
            }

            QVector<bool> gtMatched(sample.boxes.size(), false);
            QVector<DetectionPrediction> sortedPredictions = predictions;
            std::sort(sortedPredictions.begin(), sortedPredictions.end(), [](const DetectionPrediction& left, const DetectionPrediction& right) {
                return left.confidence > right.confidence;
            });

            QJsonArray samplePredictions;
            bool sampleHasError = false;
            for (const DetectionPrediction& prediction : sortedPredictions) {
                int bestMatch = -1;
                double bestIou = 0.0;
                int bestAny = -1;
                double bestAnyIou = 0.0;
                for (int index = 0; index < sample.boxes.size(); ++index) {
                    if (gtMatched.at(index)) {
                        continue;
                    }
                    const double iou = boxIou(prediction.box, sample.boxes.at(index));
                    if (iou > bestAnyIou) {
                        bestAnyIou = iou;
                        bestAny = index;
                    }
                    if (prediction.box.classId == sample.boxes.at(index).classId && iou > bestIou) {
                        bestIou = iou;
                        bestMatch = index;
                    }
                }

                const bool matched = bestMatch >= 0 && bestIou >= matchIouThreshold;
                const int predClass = prediction.box.classId >= 0 && prediction.box.classId < classStats.size()
                    ? prediction.box.classId
                    : classStats.size() - 1;
                if (matched) {
                    gtMatched[bestMatch] = true;
                    classStats[predClass].tp += 1;
                    classStats[predClass].items.append(DetectionEvaluationItem{predClass, prediction.confidence, true});
                    confusion[predClass][predClass] += 1;
                    ++totalTp;
                } else {
                    if (predClass >= 0 && predClass < classStats.size()) {
                        classStats[predClass].fp += 1;
                        classStats[predClass].items.append(DetectionEvaluationItem{predClass, prediction.confidence, false});
                    }
                    if (bestAny >= 0 && bestAnyIou >= matchIouThreshold) {
                        const int gtClass = sample.boxes.at(bestAny).classId;
                        if (gtClass >= 0 && gtClass < classCount && predClass >= 0 && predClass < classCount) {
                            confusion[gtClass][predClass] += 1;
                        }
                    } else if (predClass >= 0 && predClass < classCount) {
                        confusion[classCount][predClass] += 1;
                    }
                    ++totalFp;
                    sampleHasError = true;
                    if (errorSamples.size() < maxErrorSamples) {
                        errorSamples.append(QJsonObject{
                            {QStringLiteral("reason"), QStringLiteral("false_positive")},
                            {QStringLiteral("imagePath"), sample.imagePath},
                            {QStringLiteral("labelPath"), sample.labelPath},
                            {QStringLiteral("matchedIou"), bestAnyIou},
                            {QStringLiteral("prediction"), detectionPredictionToJson(prediction)}
                        });
                    }
                }
                if (prediction.confidence < lowConfidenceThreshold && lowConfidenceSamples.size() < maxErrorSamples) {
                    lowConfidenceSamples.append(QJsonObject{
                        {QStringLiteral("imagePath"), sample.imagePath},
                        {QStringLiteral("labelPath"), sample.labelPath},
                        {QStringLiteral("prediction"), detectionPredictionToJson(prediction)}
                    });
                }
                samplePredictions.append(detectionPredictionToJson(prediction));
            }

            for (int index = 0; index < sample.boxes.size(); ++index) {
                if (!gtMatched.at(index)) {
                    const DetectionBox& gt = sample.boxes.at(index);
                    if (gt.classId >= 0 && gt.classId < classStats.size()) {
                        classStats[gt.classId].fn += 1;
                        confusion[gt.classId][classCount] += 1;
                    }
                    ++totalFn;
                    sampleHasError = true;
                    if (errorSamples.size() < maxErrorSamples) {
                        errorSamples.append(QJsonObject{
                            {QStringLiteral("reason"), QStringLiteral("false_negative")},
                            {QStringLiteral("imagePath"), sample.imagePath},
                            {QStringLiteral("labelPath"), sample.labelPath},
                            {QStringLiteral("groundTruth"), detectionBoxToJson(gt)}
                        });
                    }
                }
            }

            QString overlayPath;
            if (sampleHasError && overlayCount < maxOverlaySamples) {
                QString overlayError;
                QImage overlay = renderDetectionPredictions(sample.imagePath, predictions, &overlayError);
                if (!overlay.isNull()) {
                    overlayPath = outputDir.filePath(QStringLiteral("overlays/%1_%2.png")
                        .arg(overlayCount, 4, 10, QLatin1Char('0'))
                        .arg(QFileInfo(sample.imagePath).completeBaseName()));
                    if (overlay.save(overlayPath)) {
                        ++overlayCount;
                    } else {
                        overlayPath.clear();
                    }
                }
            }

            QJsonObject sampleSummary;
            sampleSummary.insert(QStringLiteral("imagePath"), sample.imagePath);
            sampleSummary.insert(QStringLiteral("labelPath"), sample.labelPath);
            sampleSummary.insert(QStringLiteral("groundTruthCount"), sample.boxes.size());
            sampleSummary.insert(QStringLiteral("predictionCount"), predictions.size());
            sampleSummary.insert(QStringLiteral("hasError"), sampleHasError);
            if (!overlayPath.isEmpty()) {
                sampleSummary.insert(QStringLiteral("overlayPath"), overlayPath);
            }
            sampleSummary.insert(QStringLiteral("predictions"), samplePredictions);
            sampleSummaries.append(sampleSummary);
        }

        QJsonArray perClassArray;
        const CocoMapMetrics cocoMetrics = detectionCocoMapMetrics(mapSamples, classCount);
        double map50 = 0.0;
        int apClassCount = 0;
        for (int classId = 0; classId < classStats.size(); ++classId) {
            DetectionClassStats& stats = classStats[classId];
            stats.precision = stats.tp + stats.fp > 0 ? static_cast<double>(stats.tp) / static_cast<double>(stats.tp + stats.fp) : 0.0;
            stats.recall = stats.gt > 0 ? static_cast<double>(stats.tp) / static_cast<double>(stats.gt) : 0.0;
            stats.ap50 = ap50FromItems(stats.items, stats.gt);
            stats.map5095 = classId < cocoMetrics.perClassMap5095.size() ? cocoMetrics.perClassMap5095.at(classId) : 0.0;
            if (stats.gt > 0) {
                map50 += stats.ap50;
                ++apClassCount;
            }
            const QString className = classId < classNames.size() && !classNames.at(classId).isEmpty()
                ? classNames.at(classId)
                : QStringLiteral("class_%1").arg(classId);
            perClassArray.append(QJsonObject{
                {QStringLiteral("classId"), classId},
                {QStringLiteral("className"), className},
                {QStringLiteral("gt"), stats.gt},
                {QStringLiteral("tp"), stats.tp},
                {QStringLiteral("fp"), stats.fp},
                {QStringLiteral("fn"), stats.fn},
                {QStringLiteral("precision"), stats.precision},
                {QStringLiteral("recall"), stats.recall},
                {QStringLiteral("ap50"), stats.ap50},
                {QStringLiteral("map50_95"), stats.map5095}
            });
        }
        map50 = apClassCount > 0 ? map50 / static_cast<double>(apClassCount) : 0.0;
        const double map5095 = cocoMetrics.map5095;
        const double precision = totalTp + totalFp > 0 ? static_cast<double>(totalTp) / static_cast<double>(totalTp + totalFp) : 0.0;
        const double recall = totalGt > 0 ? static_cast<double>(totalTp) / static_cast<double>(totalGt) : 0.0;

        QJsonObject metrics;
        metrics.insert(QStringLiteral("precision"), precision);
        metrics.insert(QStringLiteral("recall"), recall);
        metrics.insert(QStringLiteral("mAP50"), map50);
        metrics.insert(QStringLiteral("mAP50_95"), map5095);
        metrics.insert(QStringLiteral("cocoMap50"), cocoMetrics.map50);
        metrics.insert(QStringLiteral("cocoMap50_95"), cocoMetrics.map5095);
        metrics.insert(QStringLiteral("tp"), totalTp);
        metrics.insert(QStringLiteral("fp"), totalFp);
        metrics.insert(QStringLiteral("fn"), totalFn);
        metrics.insert(QStringLiteral("gt"), totalGt);
        metrics.insert(QStringLiteral("predictions"), totalPredictions);

        QJsonObject report;
        report.insert(QStringLiteral("ok"), true);
        report.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
        report.insert(QStringLiteral("createdAt"), nowIso());
        report.insert(QStringLiteral("modelPath"), modelPath);
        report.insert(QStringLiteral("datasetPath"), datasetPath);
        report.insert(QStringLiteral("taskType"), QStringLiteral("detection"));
        report.insert(QStringLiteral("split"), split);
        report.insert(QStringLiteral("runtime"), runtime);
        report.insert(QStringLiteral("datasetSnapshotId"), options.value(QStringLiteral("datasetSnapshotId")).toInt());
        report.insert(QStringLiteral("datasetSnapshotHash"), options.value(QStringLiteral("datasetSnapshotHash")).toString());
        report.insert(QStringLiteral("datasetSnapshotManifest"), options.value(QStringLiteral("datasetSnapshotManifest")).toString());
        report.insert(QStringLiteral("scaffold"), false);
        report.insert(QStringLiteral("metrics"), metrics);
        report.insert(QStringLiteral("cocoMapThresholds"), cocoMetrics.thresholds);
        report.insert(QStringLiteral("perClass"), perClassArray);
        report.insert(QStringLiteral("samples"), sampleSummaries);
        report.insert(QStringLiteral("errorSamples"), errorSamples);
        report.insert(QStringLiteral("lowConfidenceSamples"), lowConfidenceSamples);
        report.insert(QStringLiteral("sampleCount"), dataset.size());
        report.insert(QStringLiteral("decisionSummary"), evaluationDecisionSummary(QStringLiteral("detection"), metrics, errorSamples, dataset.size()));
        report.insert(QStringLiteral("errorTaxonomy"), errorTaxonomyObject(QStringLiteral("detection"), metrics, errorSamples, lowConfidenceSamples));
        report.insert(QStringLiteral("parameters"), QJsonObject{
            {QStringLiteral("iouThreshold"), matchIouThreshold},
            {QStringLiteral("confidenceThreshold"), inferenceOptions.confidenceThreshold},
            {QStringLiteral("nmsIouThreshold"), inferenceOptions.iouThreshold},
            {QStringLiteral("maxDetections"), inferenceOptions.maxDetections},
            {QStringLiteral("lowConfidenceThreshold"), lowConfidenceThreshold}
        });
        report.insert(QStringLiteral("limitations"), QStringLiteral("Detection evaluation includes local COCO-style mAP50-95 over IoU thresholds 0.50:0.95. Use customer/domain acceptance gates before production claims."));

        const QString reportPath = outputDir.filePath(QStringLiteral("evaluation_report.json"));
        const QString perClassPath = outputDir.filePath(QStringLiteral("per_class_metrics.csv"));
        const QString errorPath = outputDir.filePath(QStringLiteral("error_samples.json"));
        const QString confusionPath = outputDir.filePath(QStringLiteral("confusion_matrix.csv"));
        const QString summaryPath = outputDir.filePath(QStringLiteral("evaluation_summary.md"));
        if (!writeTextFile(perClassPath, perClassMetricsCsv(classNames, classStats), &error)) {
            return failedResult(error);
        }
        if (!writeJsonFile(errorPath, QJsonObject{{QStringLiteral("samples"), errorSamples}, {QStringLiteral("lowConfidenceSamples"), lowConfidenceSamples}}, &error)) {
            return failedResult(error);
        }
        if (!writeTextFile(confusionPath, confusionMatrixCsv(classNames, confusion), &error)) {
            return failedResult(error);
        }
        report.insert(QStringLiteral("reportPath"), reportPath);
        report.insert(QStringLiteral("perClassMetricsPath"), perClassPath);
        report.insert(QStringLiteral("errorSamplesPath"), errorPath);
        report.insert(QStringLiteral("confusionMatrixPath"), confusionPath);
        report.insert(QStringLiteral("overlayDir"), outputDir.filePath(QStringLiteral("overlays")));
        report.insert(QStringLiteral("evaluationSummaryPath"), summaryPath);
        if (!writeTextFile(summaryPath, evaluationSummaryMarkdown(report), &error)) {
            return failedResult(error);
        }
        if (!writeJsonFile(reportPath, report, &error)) {
            return failedResult(error);
        }
        return resultFromReport(reportPath, report);
    }

    if (taskType == QStringLiteral("segmentation") || taskType == QStringLiteral("yolo_segmentation")) {
        const QString split = options.value(QStringLiteral("split")).toString(selectDetectionSplit(datasetPath));
        if (split.isEmpty()) {
            return failedResult(QStringLiteral("No segmentation split found. Expected images/val, images/test, or images/train with matching labels."));
        }
        if (QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) != 0) {
            return failedResult(QStringLiteral("Segmentation evaluation currently requires an ONNX model."));
        }
        const QString modelFamily = inferOnnxModelFamily(modelPath);
        if (modelFamily != QStringLiteral("yolo_segmentation")) {
            return failedResult(QStringLiteral("Segmentation evaluation expects a YOLO segmentation ONNX model. Inferred model family: %1").arg(modelFamily));
        }

        SegmentationDataset dataset;
        QString error;
        if (!dataset.load(datasetPath, split, &error)) {
            return failedResult(error);
        }

        DetectionInferenceOptions inferenceOptions;
        inferenceOptions.iouThreshold = options.value(QStringLiteral("nmsIouThreshold")).toDouble(0.45);
        inferenceOptions.confidenceThreshold = options.value(QStringLiteral("confidenceThreshold")).toDouble(0.001);
        inferenceOptions.maxDetections = options.value(QStringLiteral("maxDetections")).toInt(100);
        const double matchIouThreshold = options.value(QStringLiteral("iouThreshold")).toDouble(0.5);
        const int maxErrorSamples = options.value(QStringLiteral("maxErrorSamples")).toInt(200);
        const int maxOverlaySamples = options.value(QStringLiteral("maxOverlaySamples")).toInt(50);

        QStringList classNames = dataset.info().classNames;
        const int classCount = qMax(dataset.info().classCount, classNames.size());
        while (classNames.size() < classCount) {
            classNames.append(QStringLiteral("class_%1").arg(classNames.size()));
        }
        QVector<DetectionClassStats> classStats(classCount);
        QVector<double> classMaskIouSums(classCount, 0.0);
        QVector<int> classMaskIouCounts(classCount, 0);
        QVector<QVector<int>> confusion(classCount + 1, QVector<int>(classCount + 1, 0));
        QJsonArray sampleSummaries;
        QJsonArray errorSamples;
        QVector<SegmentationMapSample> mapSamples;
        QString runtime = QStringLiteral("onnxruntime");
        int totalGt = 0;
        int totalPredictions = 0;
        int totalTp = 0;
        int totalFp = 0;
        int totalFn = 0;
        double totalMaskIou = 0.0;
        int totalMatchedMasks = 0;
        int overlayCount = 0;
        const QDir outputDir(outputPath);
        QDir().mkpath(outputDir.filePath(QStringLiteral("overlays")));

        for (const SegmentationSample& sample : dataset.samples()) {
            QString predictionError;
            const QVector<SegmentationPrediction> predictions = predictSegmentationOnnxRuntime(modelPath, sample.imagePath, inferenceOptions, &predictionError);
            if (!predictionError.isEmpty()) {
                return failedResult(predictionError);
            }

            totalGt += sample.polygons.size();
            totalPredictions += predictions.size();
            for (const SegmentationPolygon& gt : sample.polygons) {
                if (gt.classId >= 0 && gt.classId < classStats.size()) {
                    classStats[gt.classId].gt += 1;
                }
            }

            QVector<QImage> gtMasks;
            gtMasks.reserve(sample.polygons.size());
            for (const SegmentationPolygon& gt : sample.polygons) {
                gtMasks.append(polygonToMask(gt.points, sample.imageSize));
            }
            QVector<SegmentationMapGroundTruth> mapGroundTruth;
            mapGroundTruth.reserve(sample.polygons.size());
            for (int index = 0; index < sample.polygons.size(); ++index) {
                mapGroundTruth.append(SegmentationMapGroundTruth{sample.polygons.at(index).classId, gtMasks.at(index)});
            }
            mapSamples.append(SegmentationMapSample{mapGroundTruth, predictions});
            QVector<bool> gtMatched(sample.polygons.size(), false);
            QVector<SegmentationPrediction> sortedPredictions = predictions;
            std::sort(sortedPredictions.begin(), sortedPredictions.end(), [](const SegmentationPrediction& left, const SegmentationPrediction& right) {
                return left.detection.confidence > right.detection.confidence;
            });

            QJsonArray samplePredictions;
            bool sampleHasError = false;
            for (const SegmentationPrediction& prediction : sortedPredictions) {
                int bestMatch = -1;
                double bestIou = 0.0;
                int bestAny = -1;
                double bestAnyIou = 0.0;
                for (int index = 0; index < sample.polygons.size(); ++index) {
                    if (gtMatched.at(index)) {
                        continue;
                    }
                    const double iou = maskIou(prediction.mask, gtMasks.at(index));
                    if (iou > bestAnyIou) {
                        bestAnyIou = iou;
                        bestAny = index;
                    }
                    if (prediction.detection.box.classId == sample.polygons.at(index).classId && iou > bestIou) {
                        bestIou = iou;
                        bestMatch = index;
                    }
                }

                const bool matched = bestMatch >= 0 && bestIou >= matchIouThreshold;
                const int predClass = prediction.detection.box.classId >= 0 && prediction.detection.box.classId < classStats.size()
                    ? prediction.detection.box.classId
                    : classStats.size() - 1;
                if (matched) {
                    gtMatched[bestMatch] = true;
                    classStats[predClass].tp += 1;
                    classStats[predClass].items.append(DetectionEvaluationItem{predClass, prediction.detection.confidence, true});
                    confusion[predClass][predClass] += 1;
                    ++totalTp;
                    totalMaskIou += bestIou;
                    ++totalMatchedMasks;
                    if (predClass >= 0 && predClass < classMaskIouSums.size()) {
                        classMaskIouSums[predClass] += bestIou;
                        classMaskIouCounts[predClass] += 1;
                    }
                } else {
                    if (predClass >= 0 && predClass < classStats.size()) {
                        classStats[predClass].fp += 1;
                        classStats[predClass].items.append(DetectionEvaluationItem{predClass, prediction.detection.confidence, false});
                    }
                    if (bestAny >= 0 && bestAnyIou >= matchIouThreshold) {
                        const int gtClass = sample.polygons.at(bestAny).classId;
                        if (gtClass >= 0 && gtClass < classCount && predClass >= 0 && predClass < classCount) {
                            confusion[gtClass][predClass] += 1;
                        }
                    } else if (predClass >= 0 && predClass < classCount) {
                        confusion[classCount][predClass] += 1;
                    }
                    ++totalFp;
                    sampleHasError = true;
                    if (errorSamples.size() < maxErrorSamples) {
                        errorSamples.append(QJsonObject{
                            {QStringLiteral("reason"), QStringLiteral("false_positive")},
                            {QStringLiteral("imagePath"), sample.imagePath},
                            {QStringLiteral("labelPath"), sample.labelPath},
                            {QStringLiteral("matchedMaskIoU"), bestAnyIou},
                            {QStringLiteral("prediction"), segmentationPredictionToJson(prediction)}
                        });
                    }
                }
                samplePredictions.append(segmentationPredictionToJson(prediction));
            }

            for (int index = 0; index < sample.polygons.size(); ++index) {
                if (!gtMatched.at(index)) {
                    const SegmentationPolygon& gt = sample.polygons.at(index);
                    if (gt.classId >= 0 && gt.classId < classStats.size()) {
                        classStats[gt.classId].fn += 1;
                        confusion[gt.classId][classCount] += 1;
                    }
                    ++totalFn;
                    sampleHasError = true;
                    if (errorSamples.size() < maxErrorSamples) {
                        errorSamples.append(QJsonObject{
                            {QStringLiteral("reason"), QStringLiteral("false_negative")},
                            {QStringLiteral("imagePath"), sample.imagePath},
                            {QStringLiteral("labelPath"), sample.labelPath},
                            {QStringLiteral("groundTruth"), segmentationPolygonToJson(gt)}
                        });
                    }
                }
            }

            QString overlayPath;
            if (sampleHasError && overlayCount < maxOverlaySamples) {
                QString overlayError;
                QImage overlay = renderSegmentationPredictions(sample.imagePath, predictions, &overlayError);
                if (!overlay.isNull()) {
                    overlayPath = outputDir.filePath(QStringLiteral("overlays/%1_%2.png")
                        .arg(overlayCount, 4, 10, QLatin1Char('0'))
                        .arg(QFileInfo(sample.imagePath).completeBaseName()));
                    if (overlay.save(overlayPath)) {
                        ++overlayCount;
                    } else {
                        overlayPath.clear();
                    }
                }
            }

            QJsonObject sampleSummary;
            sampleSummary.insert(QStringLiteral("imagePath"), sample.imagePath);
            sampleSummary.insert(QStringLiteral("labelPath"), sample.labelPath);
            sampleSummary.insert(QStringLiteral("groundTruthCount"), sample.polygons.size());
            sampleSummary.insert(QStringLiteral("predictionCount"), predictions.size());
            sampleSummary.insert(QStringLiteral("hasError"), sampleHasError);
            if (!overlayPath.isEmpty()) {
                sampleSummary.insert(QStringLiteral("overlayPath"), overlayPath);
            }
            sampleSummary.insert(QStringLiteral("predictions"), samplePredictions);
            sampleSummaries.append(sampleSummary);
        }

        QJsonArray perClassArray;
        const CocoMapMetrics cocoMetrics = segmentationCocoMapMetrics(mapSamples, classCount);
        double maskMap50 = 0.0;
        int apClassCount = 0;
        for (int classId = 0; classId < classStats.size(); ++classId) {
            DetectionClassStats& stats = classStats[classId];
            stats.precision = stats.tp + stats.fp > 0 ? static_cast<double>(stats.tp) / static_cast<double>(stats.tp + stats.fp) : 0.0;
            stats.recall = stats.gt > 0 ? static_cast<double>(stats.tp) / static_cast<double>(stats.gt) : 0.0;
            stats.ap50 = ap50FromItems(stats.items, stats.gt);
            stats.map5095 = classId < cocoMetrics.perClassMap5095.size() ? cocoMetrics.perClassMap5095.at(classId) : 0.0;
            const double classMaskIou = classMaskIouCounts.at(classId) > 0
                ? classMaskIouSums.at(classId) / static_cast<double>(classMaskIouCounts.at(classId))
                : 0.0;
            if (stats.gt > 0) {
                maskMap50 += stats.ap50;
                ++apClassCount;
            }
            const QString className = classId < classNames.size() && !classNames.at(classId).isEmpty()
                ? classNames.at(classId)
                : QStringLiteral("class_%1").arg(classId);
            perClassArray.append(QJsonObject{
                {QStringLiteral("classId"), classId},
                {QStringLiteral("className"), className},
                {QStringLiteral("gt"), stats.gt},
                {QStringLiteral("tp"), stats.tp},
                {QStringLiteral("fp"), stats.fp},
                {QStringLiteral("fn"), stats.fn},
                {QStringLiteral("precision"), stats.precision},
                {QStringLiteral("recall"), stats.recall},
                {QStringLiteral("maskIoU"), classMaskIou},
                {QStringLiteral("maskAP50"), stats.ap50},
                {QStringLiteral("maskMap50_95"), stats.map5095}
            });
        }
        maskMap50 = apClassCount > 0 ? maskMap50 / static_cast<double>(apClassCount) : 0.0;
        const double maskMap5095 = cocoMetrics.map5095;
        const double precision = totalTp + totalFp > 0 ? static_cast<double>(totalTp) / static_cast<double>(totalTp + totalFp) : 0.0;
        const double recall = totalGt > 0 ? static_cast<double>(totalTp) / static_cast<double>(totalGt) : 0.0;
        const double meanMaskIou = totalMatchedMasks > 0 ? totalMaskIou / static_cast<double>(totalMatchedMasks) : 0.0;

        QJsonObject metrics;
        metrics.insert(QStringLiteral("precision"), precision);
        metrics.insert(QStringLiteral("recall"), recall);
        metrics.insert(QStringLiteral("maskIoU"), meanMaskIou);
        metrics.insert(QStringLiteral("maskMap50"), maskMap50);
        metrics.insert(QStringLiteral("maskMap50_95"), maskMap5095);
        metrics.insert(QStringLiteral("cocoMaskMap50"), cocoMetrics.map50);
        metrics.insert(QStringLiteral("cocoMaskMap50_95"), cocoMetrics.map5095);
        metrics.insert(QStringLiteral("tp"), totalTp);
        metrics.insert(QStringLiteral("fp"), totalFp);
        metrics.insert(QStringLiteral("fn"), totalFn);
        metrics.insert(QStringLiteral("gt"), totalGt);
        metrics.insert(QStringLiteral("predictions"), totalPredictions);
        metrics.insert(QStringLiteral("matchedMasks"), totalMatchedMasks);

        QJsonObject report;
        report.insert(QStringLiteral("ok"), true);
        report.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
        report.insert(QStringLiteral("createdAt"), nowIso());
        report.insert(QStringLiteral("modelPath"), modelPath);
        report.insert(QStringLiteral("datasetPath"), datasetPath);
        report.insert(QStringLiteral("taskType"), QStringLiteral("segmentation"));
        report.insert(QStringLiteral("split"), split);
        report.insert(QStringLiteral("runtime"), runtime);
        report.insert(QStringLiteral("datasetSnapshotId"), options.value(QStringLiteral("datasetSnapshotId")).toInt());
        report.insert(QStringLiteral("datasetSnapshotHash"), options.value(QStringLiteral("datasetSnapshotHash")).toString());
        report.insert(QStringLiteral("datasetSnapshotManifest"), options.value(QStringLiteral("datasetSnapshotManifest")).toString());
        report.insert(QStringLiteral("scaffold"), false);
        report.insert(QStringLiteral("metrics"), metrics);
        report.insert(QStringLiteral("cocoMapThresholds"), cocoMetrics.thresholds);
        report.insert(QStringLiteral("perClass"), perClassArray);
        report.insert(QStringLiteral("samples"), sampleSummaries);
        report.insert(QStringLiteral("errorSamples"), errorSamples);
        report.insert(QStringLiteral("sampleCount"), dataset.size());
        report.insert(QStringLiteral("decisionSummary"), evaluationDecisionSummary(QStringLiteral("segmentation"), metrics, errorSamples, dataset.size()));
        report.insert(QStringLiteral("errorTaxonomy"), errorTaxonomyObject(QStringLiteral("segmentation"), metrics, errorSamples));
        report.insert(QStringLiteral("parameters"), QJsonObject{
            {QStringLiteral("iouThreshold"), matchIouThreshold},
            {QStringLiteral("confidenceThreshold"), inferenceOptions.confidenceThreshold},
            {QStringLiteral("nmsIouThreshold"), inferenceOptions.iouThreshold},
            {QStringLiteral("maxDetections"), inferenceOptions.maxDetections}
        });
        report.insert(QStringLiteral("limitations"), QStringLiteral("Segmentation evaluation includes local COCO-style mask mAP50-95 over IoU thresholds 0.50:0.95. Use customer/domain acceptance gates before production claims."));

        const QString reportPath = outputDir.filePath(QStringLiteral("evaluation_report.json"));
        const QString perClassPath = outputDir.filePath(QStringLiteral("per_class_metrics.csv"));
        const QString errorPath = outputDir.filePath(QStringLiteral("error_samples.json"));
        const QString confusionPath = outputDir.filePath(QStringLiteral("confusion_matrix.csv"));
        const QString summaryPath = outputDir.filePath(QStringLiteral("evaluation_summary.md"));
        if (!writeTextFile(perClassPath, segmentationPerClassMetricsCsv(classNames, classStats, classMaskIouSums, classMaskIouCounts), &error)) {
            return failedResult(error);
        }
        if (!writeJsonFile(errorPath, QJsonObject{{QStringLiteral("samples"), errorSamples}}, &error)) {
            return failedResult(error);
        }
        if (!writeTextFile(confusionPath, confusionMatrixCsv(classNames, confusion), &error)) {
            return failedResult(error);
        }
        report.insert(QStringLiteral("reportPath"), reportPath);
        report.insert(QStringLiteral("perClassMetricsPath"), perClassPath);
        report.insert(QStringLiteral("errorSamplesPath"), errorPath);
        report.insert(QStringLiteral("confusionMatrixPath"), confusionPath);
        report.insert(QStringLiteral("overlayDir"), outputDir.filePath(QStringLiteral("overlays")));
        report.insert(QStringLiteral("evaluationSummaryPath"), summaryPath);
        if (!writeTextFile(summaryPath, evaluationSummaryMarkdown(report), &error)) {
            return failedResult(error);
        }
        if (!writeJsonFile(reportPath, report, &error)) {
            return failedResult(error);
        }
        return resultFromReport(reportPath, report);
    }

    if (taskType == QStringLiteral("ocr_recognition") || taskType == QStringLiteral("ocr")) {
        if (QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) != 0) {
            return failedResult(QStringLiteral("OCR recognition evaluation currently requires an ONNX model."));
        }
        const QString modelFamily = inferOnnxModelFamily(modelPath);
        if (modelFamily != QStringLiteral("ocr_recognition")) {
            return failedResult(QStringLiteral("OCR recognition evaluation expects an OCR Rec ONNX model. Inferred model family: %1").arg(modelFamily));
        }

        const QString labelFilePath = resolveOcrLabelFilePath(datasetPath, options);
        const QString dictionaryPath = resolveOcrDictionaryPath(datasetPath, options);
        const int maxTextLength = options.value(QStringLiteral("maxTextLength")).toInt(64);
        const int maxErrorSamples = options.value(QStringLiteral("maxErrorSamples")).toInt(200);
        const int maxOverlaySamples = options.value(QStringLiteral("maxOverlaySamples")).toInt(50);

        OcrRecDataset dataset;
        QString error;
        if (!dataset.load(datasetPath, labelFilePath, dictionaryPath, maxTextLength, &error)) {
            return failedResult(error);
        }

        int correctCount = 0;
        int totalCount = 0;
        int totalEditDistance = 0;
        int totalCharCount = 0;
        int totalWordEditDistance = 0;
        int totalWordCount = 0;
        double confidenceSum = 0.0;
        int overlayCount = 0;
        QJsonArray sampleSummaries;
        QJsonArray errorSamples;
        const QDir outputDir(outputPath);
        QDir().mkpath(outputDir.filePath(QStringLiteral("overlays")));

        for (const OcrRecSample& sample : dataset.samples()) {
            QString predictionError;
            const OcrRecPrediction prediction = predictOcrRecOnnxRuntime(modelPath, sample.imagePath, &predictionError);
            if (!predictionError.isEmpty()) {
                return failedResult(predictionError);
            }

            const QString expected = sample.label;
            const QString predicted = prediction.text;
            const bool matched = expected == predicted;
            const int editDistance = stringEditDistance(expected, predicted);
            const QStringList expectedWords = splitOcrWords(expected);
            const QStringList predictedWords = splitOcrWords(predicted);
            const int wordDistance = wordEditDistance(expectedWords, predictedWords);

            ++totalCount;
            if (matched) {
                ++correctCount;
            }
            totalEditDistance += editDistance;
            totalCharCount += expected.size();
            totalWordEditDistance += wordDistance;
            totalWordCount += expectedWords.size();
            confidenceSum += prediction.confidence;

            QString overlayPath;
            if (!matched && overlayCount < maxOverlaySamples) {
                QString overlayError;
                const QImage overlay = renderOcrRecPrediction(sample.imagePath, prediction, &overlayError);
                if (!overlay.isNull()) {
                    overlayPath = outputDir.filePath(QStringLiteral("overlays/%1_%2.png")
                        .arg(overlayCount, 4, 10, QLatin1Char('0'))
                        .arg(QFileInfo(sample.imagePath).completeBaseName()));
                    if (overlay.save(overlayPath)) {
                        ++overlayCount;
                    } else {
                        overlayPath.clear();
                    }
                }
            }

            QJsonObject sampleSummary;
            sampleSummary.insert(QStringLiteral("imagePath"), sample.imagePath);
            sampleSummary.insert(QStringLiteral("labelPath"), dataset.labelFilePath());
            sampleSummary.insert(QStringLiteral("groundTruth"), expected);
            sampleSummary.insert(QStringLiteral("prediction"), predicted);
            sampleSummary.insert(QStringLiteral("confidence"), prediction.confidence);
            sampleSummary.insert(QStringLiteral("editDistance"), editDistance);
            sampleSummary.insert(QStringLiteral("wordEditDistance"), wordDistance);
            sampleSummary.insert(QStringLiteral("matched"), matched);
            if (!overlayPath.isEmpty()) {
                sampleSummary.insert(QStringLiteral("overlayPath"), overlayPath);
            }
            sampleSummaries.append(sampleSummary);

            if (!matched && errorSamples.size() < maxErrorSamples) {
                errorSamples.append(sampleSummary);
            }
        }

        const double accuracy = totalCount > 0 ? static_cast<double>(correctCount) / static_cast<double>(totalCount) : 0.0;
        const double cer = totalCharCount > 0 ? static_cast<double>(totalEditDistance) / static_cast<double>(totalCharCount) : 0.0;
        const double wer = totalWordCount > 0 ? static_cast<double>(totalWordEditDistance) / static_cast<double>(totalWordCount) : 0.0;
        const double averageEditDistance = totalCount > 0 ? static_cast<double>(totalEditDistance) / static_cast<double>(totalCount) : 0.0;
        const double averageConfidence = totalCount > 0 ? confidenceSum / static_cast<double>(totalCount) : 0.0;

        QJsonObject metrics;
        metrics.insert(QStringLiteral("accuracy"), accuracy);
        metrics.insert(QStringLiteral("editDistance"), averageEditDistance);
        metrics.insert(QStringLiteral("cer"), cer);
        metrics.insert(QStringLiteral("wer"), wer);
        metrics.insert(QStringLiteral("correct"), correctCount);
        metrics.insert(QStringLiteral("samples"), totalCount);
        metrics.insert(QStringLiteral("averageConfidence"), averageConfidence);

        QJsonObject report;
        report.insert(QStringLiteral("ok"), true);
        report.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
        report.insert(QStringLiteral("createdAt"), nowIso());
        report.insert(QStringLiteral("modelPath"), modelPath);
        report.insert(QStringLiteral("datasetPath"), datasetPath);
        report.insert(QStringLiteral("taskType"), QStringLiteral("ocr_recognition"));
        report.insert(QStringLiteral("runtime"), QStringLiteral("onnxruntime"));
        report.insert(QStringLiteral("datasetSnapshotId"), options.value(QStringLiteral("datasetSnapshotId")).toInt());
        report.insert(QStringLiteral("datasetSnapshotHash"), options.value(QStringLiteral("datasetSnapshotHash")).toString());
        report.insert(QStringLiteral("datasetSnapshotManifest"), options.value(QStringLiteral("datasetSnapshotManifest")).toString());
        report.insert(QStringLiteral("labelFilePath"), dataset.labelFilePath());
        report.insert(QStringLiteral("dictionaryPath"), dataset.dictionary().path);
        report.insert(QStringLiteral("scaffold"), false);
        report.insert(QStringLiteral("metrics"), metrics);
        report.insert(QStringLiteral("samples"), sampleSummaries);
        report.insert(QStringLiteral("errorSamples"), errorSamples);
        report.insert(QStringLiteral("sampleCount"), dataset.size());
        report.insert(QStringLiteral("decisionSummary"), evaluationDecisionSummary(QStringLiteral("ocr_recognition"), metrics, errorSamples, dataset.size()));
        report.insert(QStringLiteral("errorTaxonomy"), errorTaxonomyObject(QStringLiteral("ocr_recognition"), metrics, errorSamples));
        report.insert(QStringLiteral("limitations"), QStringLiteral("Phase 39A OCR evaluation uses exact-match accuracy, CER, and WER from OCR Rec ONNX predictions."));

        const QString reportPath = outputDir.filePath(QStringLiteral("evaluation_report.json"));
        const QString errorPath = outputDir.filePath(QStringLiteral("error_samples.json"));
        const QString summaryPath = outputDir.filePath(QStringLiteral("evaluation_summary.md"));
        if (!writeJsonFile(errorPath, QJsonObject{{QStringLiteral("samples"), errorSamples}}, &error)) {
            return failedResult(error);
        }
        report.insert(QStringLiteral("reportPath"), reportPath);
        report.insert(QStringLiteral("errorSamplesPath"), errorPath);
        report.insert(QStringLiteral("overlayDir"), outputDir.filePath(QStringLiteral("overlays")));
        report.insert(QStringLiteral("evaluationSummaryPath"), summaryPath);
        if (!writeTextFile(summaryPath, evaluationSummaryMarkdown(report), &error)) {
            return failedResult(error);
        }
        if (!writeJsonFile(reportPath, report, &error)) {
            return failedResult(error);
        }
        return resultFromReport(reportPath, report);
    }

    QJsonObject summary;
    summary.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
    summary.insert(QStringLiteral("createdAt"), nowIso());
    summary.insert(QStringLiteral("modelPath"), modelPath);
    summary.insert(QStringLiteral("datasetPath"), datasetPath);
    summary.insert(QStringLiteral("taskType"), taskType);
    summary.insert(QStringLiteral("datasetSnapshotId"), options.value(QStringLiteral("datasetSnapshotId")).toInt());
    summary.insert(QStringLiteral("datasetSnapshotHash"), options.value(QStringLiteral("datasetSnapshotHash")).toString());
    summary.insert(QStringLiteral("datasetSnapshotManifest"), options.value(QStringLiteral("datasetSnapshotManifest")).toString());
    summary.insert(QStringLiteral("scaffold"), true);
    summary.insert(QStringLiteral("note"), QStringLiteral("Real evaluation is implemented for detection, segmentation (ONNX), and OCR recognition (ONNX). Unsupported task types still produce scaffold summaries."));

    QJsonObject metrics;
    if (taskType == QStringLiteral("ocr_recognition") || taskType == QStringLiteral("ocr")) {
        metrics.insert(QStringLiteral("accuracy"), 0.0);
        metrics.insert(QStringLiteral("editDistance"), 0.0);
        metrics.insert(QStringLiteral("cer"), 0.0);
        metrics.insert(QStringLiteral("wer"), 0.0);
    } else if (taskType == QStringLiteral("segmentation")) {
        metrics.insert(QStringLiteral("maskIoU"), 0.0);
        metrics.insert(QStringLiteral("maskMap50"), 0.0);
        metrics.insert(QStringLiteral("maskMap50_95"), 0.0);
        metrics.insert(QStringLiteral("precision"), 0.0);
        metrics.insert(QStringLiteral("recall"), 0.0);
    } else {
        metrics.insert(QStringLiteral("precision"), 0.0);
        metrics.insert(QStringLiteral("recall"), 0.0);
        metrics.insert(QStringLiteral("mAP50"), 0.0);
        metrics.insert(QStringLiteral("mAP50_95"), 0.0);
    }
    summary.insert(QStringLiteral("metrics"), metrics);
    summary.insert(QStringLiteral("errorSamples"), QJsonArray());
    summary.insert(QStringLiteral("lowConfidenceSamples"), QJsonArray());
    summary.insert(QStringLiteral("sampleCount"), 0);
    summary.insert(QStringLiteral("decisionSummary"), evaluationDecisionSummary(taskType, metrics, QJsonArray(), 0));
    summary.insert(QStringLiteral("errorTaxonomy"), errorTaxonomyObject(taskType, metrics, QJsonArray()));

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("evaluation_report.json"));
    const QString summaryPath = QDir(outputPath).filePath(QStringLiteral("evaluation_summary.md"));
    summary.insert(QStringLiteral("reportPath"), reportPath);
    summary.insert(QStringLiteral("evaluationSummaryPath"), summaryPath);
    if (!writeTextFile(summaryPath, evaluationSummaryMarkdown(summary), &error)) {
        return failedResult(error);
    }
    if (!writeJsonFile(reportPath, summary, &error)) {
        return failedResult(error);
    }
    return resultFromReport(reportPath, summary);
}
} // namespace aitrain
