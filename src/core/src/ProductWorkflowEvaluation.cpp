#include "aitrain/core/ProductWorkflow.h"

#include "ProductWorkflowSupport.h"
#include "YoloDatasetLayout.h"
#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/OcrRecDataset.h"
#include "aitrain/core/SegmentationDataset.h"

#include <QCryptographicHash>
#include <QCoreApplication>
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
#include <QProcess>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QSet>
#include <QStandardPaths>
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
    QString yamlError;
    const YoloDataYaml layout = parseYoloDataYaml(datasetPath, &yamlError);
    if (!yamlError.isEmpty()) {
        return false;
    }
    const YoloSplitPaths splitPaths = yoloSplitPaths(layout, split);
    return QDir(splitPaths.imageDir).exists()
        && QDir(splitPaths.labelDir).exists();
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

bool pythonExecutableUsable(const QString& executable)
{
    if (executable.trimmed().isEmpty()) {
        return false;
    }
    QProcess process;
    process.start(executable, QStringList() << QStringLiteral("--version"));
    return process.waitForStarted(2000)
        && process.waitForFinished(5000)
        && process.exitStatus() == QProcess::NormalExit
        && process.exitCode() == 0;
}

QString officialYoloEvaluationPython(const QJsonObject& options)
{
    const QString requested = options.value(QStringLiteral("pythonExecutable")).toString().trimmed();
    if (pythonExecutableUsable(requested)) {
        return requested;
    }
    const QString envRequested = QString::fromLocal8Bit(qgetenv("AITRAIN_PYTHON_EXECUTABLE")).trimmed();
    if (pythonExecutableUsable(envRequested)) {
        return envRequested;
    }

    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir::current().absoluteFilePath(QStringLiteral(".deps/python-3.13.13-embed-amd64/python.exe")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../.deps/python-3.13.13-embed-amd64/python.exe")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../../.deps/python-3.13.13-embed-amd64/python.exe")),
        QStandardPaths::findExecutable(QStringLiteral("python")),
        QStandardPaths::findExecutable(QStringLiteral("python3"))
    };
    for (const QString& candidate : candidates) {
        if (pythonExecutableUsable(candidate)) {
            return candidate;
        }
    }
    return {};
}

QString officialYoloEvaluatorScriptPath(const QJsonObject& options)
{
    const QString requested = options.value(QStringLiteral("ultralyticsEvaluatorScript")).toString().trimmed();
    if (!requested.isEmpty() && QFileInfo::exists(requested)) {
        return QFileInfo(requested).absoluteFilePath();
    }
    const QString envRequested = QString::fromLocal8Bit(qgetenv("AITRAIN_YOLO_EVALUATOR_SCRIPT")).trimmed();
    if (!envRequested.isEmpty() && QFileInfo::exists(envRequested)) {
        return QFileInfo(envRequested).absoluteFilePath();
    }

    const QString script = QStringLiteral("python_trainers/yolo/ultralytics_evaluator.py");
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).absoluteFilePath(script),
        QDir(appDir).absoluteFilePath(QStringLiteral("../%1").arg(script)),
        QDir(appDir).absoluteFilePath(QStringLiteral("../../%1").arg(script)),
        QDir::current().absoluteFilePath(script)
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return candidates.first();
}

WorkflowResult officialYoloEvaluationFailure(
    const QString& outputPath,
    const QString& modelPath,
    const QString& datasetPath,
    const QString& taskType,
    const QString& message,
    const QString& errorCode)
{
    QJsonObject report;
    report.insert(QStringLiteral("ok"), false);
    report.insert(QStringLiteral("status"), QStringLiteral("failed"));
    report.insert(QStringLiteral("failureCategory"), QStringLiteral("official-evaluation"));
    report.insert(QStringLiteral("errorCode"), errorCode);
    report.insert(QStringLiteral("message"), message);
    report.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("modelPath"), modelPath);
    report.insert(QStringLiteral("datasetPath"), datasetPath);
    report.insert(QStringLiteral("taskType"), taskType);
    report.insert(QStringLiteral("runtime"), QStringLiteral("ultralytics_official_val"));
    report.insert(QStringLiteral("evaluationSource"), QStringLiteral("ultralytics_official_val"));
    report.insert(QStringLiteral("scaffold"), false);
    report.insert(QStringLiteral("metrics"), QJsonObject{});
    report.insert(QStringLiteral("perClass"), QJsonArray{});
    report.insert(QStringLiteral("errorSamples"), QJsonArray{});
    report.insert(QStringLiteral("lowConfidenceSamples"), QJsonArray{});
    report.insert(QStringLiteral("limitations"), QStringLiteral("YOLO detection/segmentation evaluation is official-only. AITrain local AP/mAP fallback is disabled."));

    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("evaluation_report.json"));
    QString writeError;
    writeJsonFile(reportPath, report, &writeError);

    WorkflowResult result;
    result.ok = false;
    result.error = message;
    result.reportPath = reportPath;
    result.payload = report;
    return result;
}

WorkflowResult runOfficialYoloEvaluation(
    const QString& modelPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& taskType,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel)
{
    QDir().mkpath(outputPath);
    const QString python = officialYoloEvaluationPython(options);
    if (python.isEmpty()) {
        return officialYoloEvaluationFailure(
            outputPath,
            modelPath,
            datasetPath,
            taskType,
            QStringLiteral("Python executable is required for Ultralytics official YOLO evaluation. Set pythonExecutable or AITRAIN_PYTHON_EXECUTABLE."),
            QStringLiteral("python_missing"));
    }
    const QString evaluatorScript = officialYoloEvaluatorScriptPath(options);
    if (!QFileInfo::exists(evaluatorScript)) {
        return officialYoloEvaluationFailure(
            outputPath,
            modelPath,
            datasetPath,
            taskType,
            QStringLiteral("Ultralytics evaluator script not found: %1").arg(evaluatorScript),
            QStringLiteral("ultralytics_evaluator_script_missing"));
    }

    QJsonObject request;
    request.insert(QStringLiteral("protocolVersion"), 1);
    request.insert(QStringLiteral("modelPath"), modelPath);
    request.insert(QStringLiteral("datasetPath"), datasetPath);
    request.insert(QStringLiteral("outputPath"), outputPath);
    request.insert(QStringLiteral("taskType"), taskType == QStringLiteral("yolo_segmentation") ? QStringLiteral("segmentation") : taskType);
    request.insert(QStringLiteral("options"), options);

    const QString requestPath = QDir(outputPath).filePath(QStringLiteral("ultralytics_evaluation_request.json"));
    QString error;
    if (!writeJsonFile(requestPath, request, &error)) {
        return failedResult(error);
    }

    QProcess process;
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    process.setProcessEnvironment(environment);
    process.setProgram(python);
    process.setArguments(QStringList() << QStringLiteral("-u") << evaluatorScript << QStringLiteral("--request") << requestPath);
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.start();
    if (!process.waitForStarted(5000)) {
        return officialYoloEvaluationFailure(
            outputPath,
            modelPath,
            datasetPath,
            taskType,
            QStringLiteral("Cannot start Ultralytics official evaluator: %1").arg(process.errorString()),
            QStringLiteral("ultralytics_evaluator_start_failed"));
    }

    QByteArray output;
    while (!process.waitForFinished(100)) {
        output.append(process.readAll());
        if (isCancellationRequested(shouldCancel)) {
            process.kill();
            process.waitForFinished(1500);
            return canceledResult();
        }
    }
    output.append(process.readAll());
    const QString logPath = QDir(outputPath).filePath(QStringLiteral("ultralytics_official_val.log"));
    writeTextFile(logPath, QString::fromUtf8(output), &error);

    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("evaluation_report.json"));
    QJsonObject report;
    if (!readJsonFile(reportPath, &report, &error)) {
        const QString processError = process.exitStatus() == QProcess::NormalExit
            ? QStringLiteral("Ultralytics evaluator exited with code %1.").arg(process.exitCode())
            : QStringLiteral("Ultralytics evaluator crashed.");
        return officialYoloEvaluationFailure(
            outputPath,
            modelPath,
            datasetPath,
            taskType,
            QStringLiteral("%1 Report was not readable: %2").arg(processError, error),
            QStringLiteral("ultralytics_evaluation_report_missing"));
    }

    report.insert(QStringLiteral("officialLogPath"), logPath);
    if (!writeJsonFile(reportPath, report, &error)) {
        return failedResult(error);
    }

    WorkflowResult result;
    result.ok = process.exitStatus() == QProcess::NormalExit
        && process.exitCode() == 0
        && report.value(QStringLiteral("ok")).toBool(false);
    result.reportPath = reportPath;
    result.payload = report;
    if (!result.ok) {
        result.error = report.value(QStringLiteral("message")).toString(
            QStringLiteral("Ultralytics official evaluation failed."));
    }
    return result;
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

    if (error) {
        *error = QStringLiteral("Unsupported detection model format: %1. Production evaluation requires official ONNX, NCNN, or TensorRT artifacts.").arg(modelPath);
    }
    return {};
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
    return evaluateModelReport(modelPath, datasetPath, outputPath, taskType, options, CancellationCallback());
}

WorkflowResult evaluateModelReport(
    const QString& modelPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& taskType,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel)
{
    if (isCancellationRequested(shouldCancel)) {
        return canceledResult();
    }
    const QFileInfo modelInfo(modelPath);
    if (!modelInfo.exists()) {
        return failedResult(QStringLiteral("Model file does not exist: %1").arg(modelPath));
    }
    const QDir datasetRoot(datasetPath);
    if (!datasetRoot.exists()) {
        return failedResult(QStringLiteral("Dataset directory does not exist: %1").arg(datasetPath));
    }

    if (taskType == QStringLiteral("detection")
        || taskType == QStringLiteral("yolo_detection")
        || taskType == QStringLiteral("segmentation")
        || taskType == QStringLiteral("yolo_segmentation")) {
        return runOfficialYoloEvaluation(modelPath, datasetPath, outputPath, taskType, options, shouldCancel);
    }

    if (taskType == QStringLiteral("ocr_recognition") || taskType == QStringLiteral("ocr")) {
        const QDir outputDir(outputPath);
        QString error;
        if (!QDir().mkpath(outputDir.absolutePath())) {
            return failedResult(QStringLiteral("Cannot create OCR evaluation output directory: %1").arg(outputPath));
        }
        if (isCancellationRequested(shouldCancel)) {
            return canceledResult();
        }

        QJsonObject report;
        report.insert(QStringLiteral("ok"), false);
        report.insert(QStringLiteral("kind"), QStringLiteral("evaluation_report"));
        report.insert(QStringLiteral("createdAt"), nowIso());
        report.insert(QStringLiteral("modelPath"), modelPath);
        report.insert(QStringLiteral("datasetPath"), datasetPath);
        report.insert(QStringLiteral("taskType"), QStringLiteral("ocr_recognition"));
        report.insert(QStringLiteral("runtime"), QStringLiteral("paddleocr_official"));
        report.insert(QStringLiteral("status"), QStringLiteral("blocked"));
        report.insert(QStringLiteral("failureCategory"), QStringLiteral("official-only"));
        report.insert(QStringLiteral("datasetSnapshotId"), options.value(QStringLiteral("datasetSnapshotId")).toInt());
        report.insert(QStringLiteral("datasetSnapshotHash"), options.value(QStringLiteral("datasetSnapshotHash")).toString());
        report.insert(QStringLiteral("datasetSnapshotManifest"), options.value(QStringLiteral("datasetSnapshotManifest")).toString());
        report.insert(QStringLiteral("scaffold"), false);
        report.insert(QStringLiteral("metrics"), QJsonObject{});
        report.insert(QStringLiteral("message"),
            QStringLiteral("OCR evaluation is official-only. Use PaddleOCR official Rec evaluation/predict reports or the customer OCR acceptance gate instead of AITrain C++ ONNX OCR postprocess."));
        report.insert(QStringLiteral("officialRoute"), QStringLiteral("paddleocr_rec_official / paddleocr_system_official"));
        report.insert(QStringLiteral("limitations"), QJsonArray{
            QStringLiteral("AITrain does not compute product OCR metrics from C++ ONNX OCR postprocess."),
            QStringLiteral("Production OCR evidence must come from PaddleOCR official reports on representative customer-domain data.")
        });

        const QString reportPath = outputDir.filePath(QStringLiteral("evaluation_report.json"));
        const QString summaryPath = outputDir.filePath(QStringLiteral("evaluation_summary.md"));
        report.insert(QStringLiteral("reportPath"), reportPath);
        report.insert(QStringLiteral("evaluationSummaryPath"), summaryPath);
        if (!writeTextFile(summaryPath,
                QStringLiteral("# OCR Evaluation\n\nStatus: blocked\n\nOCR evaluation is official-only. Use PaddleOCR official Rec/System reports and customer-domain acceptance evidence.\n"),
                &error)) {
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
    summary.insert(QStringLiteral("note"), QStringLiteral("Real evaluation is implemented for detection and segmentation ONNX artifacts. OCR evaluation is official-only and must use PaddleOCR official reports."));

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
    if (isCancellationRequested(shouldCancel)) {
        return canceledResult();
    }
    if (!writeTextFile(summaryPath, evaluationSummaryMarkdown(summary), &error)) {
        return failedResult(error);
    }
    if (!writeJsonFile(reportPath, summary, &error)) {
        return failedResult(error);
    }
    return resultFromReport(reportPath, summary);
}
} // namespace aitrain
