#include "aitrain/core/ProductWorkflow.h"

#include "ProductWorkflowSupport.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>
namespace aitrain {
using namespace workflow_detail;
namespace {
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
    markdown += QStringLiteral("\nThis summary is generated from evaluation artifacts. Inspect the JSON report for full details.\n");
    return markdown;
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
    summary.insert(QStringLiteral("note"), QStringLiteral("YOLO detection/segmentation evaluation is implemented through Ultralytics official val(). OCR evaluation is official-only and must use PaddleOCR official reports."));

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
