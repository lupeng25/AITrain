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

double percentileMs(QVector<double> values, double percentile)
{
    if (values.isEmpty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = qBound(0.0, percentile, 100.0);
    const int index = qBound(0, static_cast<int>((clamped / 100.0) * static_cast<double>(values.size() - 1) + 0.5), values.size() - 1);
    return values.at(index);
}

double averageMs(const QVector<double>& values)
{
    if (values.isEmpty()) {
        return 0.0;
    }
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

} // namespace

WorkflowResult benchmarkModelReport(const QString& modelPath, const QString& outputPath, const QJsonObject& options)
{
    return benchmarkModelReport(modelPath, outputPath, options, CancellationCallback());
}

WorkflowResult benchmarkModelReport(
    const QString& modelPath,
    const QString& outputPath,
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

    const QString suffix = modelInfo.suffix().toLower();
    const QString runtime = options.value(QStringLiteral("runtime")).toString(
        suffix == QStringLiteral("onnx") ? QStringLiteral("onnxruntime") :
        (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan") ? QStringLiteral("tensorrt") : QStringLiteral("file")));
    if (runtime == QStringLiteral("tensorrt") && !isTensorRtInferenceAvailable()) {
        QString digestError;
        const QJsonObject modelDigest = fileDigestObject(modelPath, &digestError);
        QJsonObject blocked;
        blocked.insert(QStringLiteral("ok"), false);
        blocked.insert(QStringLiteral("status"), QStringLiteral("hardware-blocked"));
        blocked.insert(QStringLiteral("failureCategory"), QStringLiteral("hardware-blocked"));
        blocked.insert(QStringLiteral("modelPath"), modelPath);
        blocked.insert(QStringLiteral("modelDigest"), modelDigest);
        blocked.insert(QStringLiteral("runtime"), runtime);
        blocked.insert(QStringLiteral("runtimeUsable"), false);
        blocked.insert(QStringLiteral("timedInference"), false);
        blocked.insert(QStringLiteral("deploymentConclusion"), QStringLiteral("hardware-blocked"));
        blocked.insert(QStringLiteral("message"), QStringLiteral("TensorRT benchmark requires a compatible RTX / SM 75+ acceptance machine and runtime."));
        if (!digestError.isEmpty()) {
            blocked.insert(QStringLiteral("digestError"), digestError);
        }
        QString error;
        const QString reportPath = QDir(outputPath).filePath(QStringLiteral("benchmark_report.json"));
        if (!writeJsonFile(reportPath, blocked, &error)) {
            return failedResult(error);
        }
        blocked.insert(QStringLiteral("reportPath"), reportPath);
        return resultFromReport(reportPath, blocked);
    }

    const QString datasetPath = options.value(QStringLiteral("datasetPath")).toString();
    const QString sampleImagePath = options.value(QStringLiteral("sampleImagePath")).toString(
        options.value(QStringLiteral("imagePath")).toString(firstImageFileUnder(datasetPath)));
    const int warmupIterations = qMax(0, options.value(QStringLiteral("warmupIterations")).toInt(2));
    const int iterations = qMax(1, options.value(QStringLiteral("iterations")).toInt(10));
    const QString provider = runtime == QStringLiteral("onnxruntime")
        ? QStringLiteral("CPUExecutionProvider")
        : (runtime == QStringLiteral("tensorrt") ? QStringLiteral("TensorRT") : QStringLiteral("local_file"));

    QVector<double> timings;
    QString inferenceError;
    QString modelFamily = suffix == QStringLiteral("onnx") ? inferOnnxModelFamily(modelPath) : QStringLiteral("unsupported_model_format");
    int outputCount = 0;
    bool timedInference = false;
    QString failureCategory;

    const auto runTimedInference = [&]() -> bool {
        if (isCancellationRequested(shouldCancel)) {
            inferenceError = QStringLiteral("Canceled by user");
            failureCategory = QStringLiteral("canceled");
            return false;
        }
        if (runtime == QStringLiteral("onnxruntime")) {
            if (!isOnnxRuntimeInferenceAvailable()) {
                inferenceError = QStringLiteral("ONNX Runtime is not available in this build.");
                failureCategory = QStringLiteral("runtime-missing");
                return false;
            }
            if (sampleImagePath.isEmpty() || !QFileInfo::exists(sampleImagePath)) {
                inferenceError = QStringLiteral("A sample image is required for timed ONNX Runtime benchmark.");
                failureCategory = QStringLiteral("sample-required");
                return false;
            }
            DetectionInferenceOptions inferenceOptions;
            inferenceOptions.confidenceThreshold = options.value(QStringLiteral("confidenceThreshold")).toDouble(0.25);
            inferenceOptions.iouThreshold = options.value(QStringLiteral("iouThreshold")).toDouble(0.45);
            inferenceOptions.maxDetections = options.value(QStringLiteral("maxDetections")).toInt(100);
            const int totalRuns = warmupIterations + iterations;
            for (int index = 0; index < totalRuns; ++index) {
                if (isCancellationRequested(shouldCancel)) {
                    inferenceError = QStringLiteral("Canceled by user");
                    failureCategory = QStringLiteral("canceled");
                    return false;
                }
                QElapsedTimer timer;
                timer.start();
                if (modelFamily == QStringLiteral("yolo_segmentation")) {
                    const QVector<SegmentationPrediction> predictions = predictSegmentationOnnxRuntime(modelPath, sampleImagePath, inferenceOptions, &inferenceError);
                    outputCount = predictions.size();
                } else if (modelFamily == QStringLiteral("ocr_recognition")) {
                    const OcrRecPrediction prediction = predictOcrRecOnnxRuntime(modelPath, sampleImagePath, &inferenceError);
                    outputCount = prediction.text.isEmpty() ? 0 : 1;
                } else if (modelFamily == QStringLiteral("ocr_detection")) {
                    OcrDetPostprocessOptions detOptions;
                    detOptions.maxDetections = inferenceOptions.maxDetections;
                    const QVector<OcrDetPrediction> predictions = predictOcrDetOnnxRuntime(modelPath, sampleImagePath, detOptions, &inferenceError);
                    outputCount = predictions.size();
                } else {
                    const QVector<DetectionPrediction> predictions = predictDetectionOnnxRuntime(modelPath, sampleImagePath, inferenceOptions, &inferenceError);
                    outputCount = predictions.size();
                    modelFamily = QStringLiteral("yolo_detection");
                }
                if (!inferenceError.isEmpty()) {
                    failureCategory = QStringLiteral("unsupported-model");
                    return false;
                }
                const double elapsedMs = static_cast<double>(timer.nsecsElapsed()) / 1000000.0;
                if (index >= warmupIterations) {
                    timings.append(elapsedMs);
                }
            }
            timedInference = true;
            return true;
        }

        if (runtime == QStringLiteral("file") && sampleImagePath.isEmpty()) {
            failureCategory = QStringLiteral("sample-required");
            return false;
        }

        if (sampleImagePath.isEmpty() || !QFileInfo::exists(sampleImagePath)) {
            failureCategory = QStringLiteral("sample-required");
            return false;
        }
        inferenceError = QStringLiteral("Unsupported model benchmark format: %1. Production benchmark requires official ONNX, NCNN, or TensorRT artifacts.").arg(modelPath);
        failureCategory = QStringLiteral("unsupported-model");
        return false;
    };

    runTimedInference();
    if (failureCategory == QStringLiteral("canceled") || isCancellationRequested(shouldCancel)) {
        return canceledResult();
    }

    QString modelDigestError;
    const QJsonObject modelDigest = fileDigestObject(modelPath, &modelDigestError);
    if (!modelDigestError.isEmpty()) {
        return failedResult(modelDigestError);
    }
    QString sampleDigestError;
    const QJsonObject sampleDigest = fileDigestObject(sampleImagePath, &sampleDigestError);
    if (!sampleDigestError.isEmpty()) {
        return failedResult(sampleDigestError);
    }

    if (!timedInference) {
        if (isCancellationRequested(shouldCancel)) {
            return canceledResult();
        }
        QElapsedTimer timer;
        timer.start();
        QFile file(modelPath);
        if (!file.open(QIODevice::ReadOnly)) {
            return failedResult(QStringLiteral("Cannot read model for benchmark: %1").arg(modelPath));
        }
        file.read(1024 * 1024);
        timings.append(static_cast<double>(qMax<qint64>(1, timer.elapsed())));
    }

    const double avg = averageMs(timings);
    const double p50 = percentileMs(timings, 50.0);
    const double p95 = percentileMs(timings, 95.0);
    const double p99 = percentileMs(timings, 99.0);
    const bool runtimeUsable = timedInference && inferenceError.isEmpty();
    if (failureCategory.isEmpty() && !runtimeUsable) {
        failureCategory = runtime == QStringLiteral("file")
            ? QStringLiteral("metadata-only")
            : QStringLiteral("unsupported-model");
    }

    QJsonObject latency;
    latency.insert(QStringLiteral("averageMs"), avg);
    latency.insert(QStringLiteral("p50Ms"), p50);
    latency.insert(QStringLiteral("p95Ms"), p95);
    latency.insert(QStringLiteral("p99Ms"), p99);
    latency.insert(QStringLiteral("throughput"), avg > 0.0 ? 1000.0 / avg : 0.0);

    QJsonObject runtimeObject;
    runtimeObject.insert(QStringLiteral("name"), runtime);
    runtimeObject.insert(QStringLiteral("provider"), provider);
    runtimeObject.insert(QStringLiteral("device"), options.value(QStringLiteral("device")).toString(QStringLiteral("cpu")));
    runtimeObject.insert(QStringLiteral("usable"), runtimeUsable);
    runtimeObject.insert(QStringLiteral("onnxRuntimeAvailable"), isOnnxRuntimeInferenceAvailable());
    runtimeObject.insert(QStringLiteral("tensorRt"), tensorRtBackendStatus().toJson());

    QJsonObject report;
    report.insert(QStringLiteral("ok"), runtimeUsable || runtime == QStringLiteral("file"));
    report.insert(QStringLiteral("kind"), QStringLiteral("benchmark_report"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("modelPath"), modelPath);
    report.insert(QStringLiteral("modelFamily"), modelFamily);
    report.insert(QStringLiteral("runtime"), runtime);
    report.insert(QStringLiteral("device"), options.value(QStringLiteral("device")).toString(QStringLiteral("cpu")));
    report.insert(QStringLiteral("provider"), provider);
    report.insert(QStringLiteral("batch"), options.value(QStringLiteral("batch")).toInt(1));
    report.insert(QStringLiteral("inputShape"), options.value(QStringLiteral("inputShape")).toString(QStringLiteral("auto")));
    report.insert(QStringLiteral("runtimeUsable"), runtimeUsable);
    report.insert(QStringLiteral("runtimeStatus"), runtimeUsable ? QStringLiteral("available") : failureCategory);
    report.insert(QStringLiteral("failureCategory"), runtimeUsable ? QStringLiteral("") : failureCategory);
    report.insert(QStringLiteral("warmupIterations"), warmupIterations);
    report.insert(QStringLiteral("iterations"), iterations);
    report.insert(QStringLiteral("averageMs"), avg);
    report.insert(QStringLiteral("p50Ms"), p50);
    report.insert(QStringLiteral("p95Ms"), p95);
    report.insert(QStringLiteral("p99Ms"), p99);
    report.insert(QStringLiteral("throughput"), avg > 0.0 ? 1000.0 / avg : 0.0);
    report.insert(QStringLiteral("latency"), latency);
    report.insert(QStringLiteral("outputCount"), outputCount);
    report.insert(QStringLiteral("sampleImagePath"), sampleImagePath);
    report.insert(QStringLiteral("modelDigest"), modelDigest);
    report.insert(QStringLiteral("sampleDigest"), sampleDigest);
    report.insert(QStringLiteral("modelBytes"), modelDigest.value(QStringLiteral("bytes")).toString());
    report.insert(QStringLiteral("modelSha256"), modelDigest.value(QStringLiteral("sha256")).toString());
    report.insert(QStringLiteral("sampleSha256"), sampleDigest.value(QStringLiteral("sha256")).toString());
    report.insert(QStringLiteral("timedInference"), timedInference);
    report.insert(QStringLiteral("scaffold"), !timedInference);
    report.insert(QStringLiteral("status"), timedInference ? QStringLiteral("completed") : QStringLiteral("limited"));
    report.insert(QStringLiteral("deploymentConclusion"), timedInference
        ? QStringLiteral("local-runtime-available")
        : QStringLiteral("input-required-or-runtime-limited"));
    report.insert(QStringLiteral("runtimeDetails"), runtimeObject);
    report.insert(QStringLiteral("cpuThreads"), QThread::idealThreadCount());
    report.insert(QStringLiteral("onnxRuntimeAvailable"), isOnnxRuntimeInferenceAvailable());
    report.insert(QStringLiteral("tensorRt"), tensorRtBackendStatus().toJson());
    if (!inferenceError.isEmpty()) {
        report.insert(QStringLiteral("message"), inferenceError);
    }
    report.insert(QStringLiteral("note"), timedInference
        ? QStringLiteral("Timed local inference benchmark completed with warmup and repeated measurements.")
        : QStringLiteral("No timed inference was run. Provide an official ONNX model and sampleImagePath or datasetPath for full local benchmark."));

    QString error;
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("benchmark_report.json"));
    if (isCancellationRequested(shouldCancel)) {
        return canceledResult();
    }
    if (!writeJsonFile(reportPath, report, &error)) {
        return failedResult(error);
    }
    report.insert(QStringLiteral("reportPath"), reportPath);
    return resultFromReport(reportPath, report);
}
} // namespace aitrain
