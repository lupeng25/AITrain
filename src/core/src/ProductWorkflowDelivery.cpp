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
QString htmlReport(const QJsonObject& context)
{
    const QJsonObject evaluation = context.value(QStringLiteral("evaluationSummary")).toObject();
    const QJsonObject benchmark = context.value(QStringLiteral("benchmarkSummary")).toObject();
    const QJsonObject deployment = context.value(QStringLiteral("deploymentValidationSummary")).toObject();
    const QJsonObject customerOcr = context.value(QStringLiteral("customerOcrAcceptanceSummary")).toObject();
    const QJsonArray inventory = context.value(QStringLiteral("inventory")).toArray();
    const QJsonArray limitations = context.value(QStringLiteral("limitations")).toArray();
    const QJsonObject evaluationMetrics = evaluation.value(QStringLiteral("metrics")).toObject();
    const QJsonObject latency = benchmark.value(QStringLiteral("latency")).toObject();

    QString inventoryRows;
    for (const QJsonValue& value : inventory) {
        const QJsonObject item = value.toObject();
        inventoryRows += QStringLiteral("<tr><td>%1</td><td>%2</td><td>%3</td><td>%4</td></tr>")
            .arg(htmlEscape(item.value(QStringLiteral("kind")).toString()))
            .arg(htmlEscape(QDir::toNativeSeparators(item.value(QStringLiteral("path")).toString())))
            .arg(htmlEscape(item.value(QStringLiteral("bytes")).toString()))
            .arg(htmlEscape(item.value(QStringLiteral("sha256")).toString().left(12)));
    }
    if (inventoryRows.isEmpty()) {
        inventoryRows = QStringLiteral("<tr><td colspan=\"4\">No delivery artifacts were attached.</td></tr>");
    }

    QString limitationItems;
    for (const QJsonValue& value : limitations) {
        limitationItems += QStringLiteral("<li>%1</li>").arg(htmlEscape(value.toString()));
    }

    QString html;
    html += QStringLiteral("<!doctype html><html><head><meta charset=\"utf-8\"><title>AITrain Delivery Report</title>");
    html += QStringLiteral("<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#111827;background:#f4f6f8}"
                           "section{background:white;border:1px solid #d8dee6;border-radius:8px;padding:18px;margin:14px 0}"
                           "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d8dee6;padding:8px;text-align:left}"
                           "th{background:#f4f6f8}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}"
                           ".metric{background:#f9fafb;border:1px solid #d8dee6;border-radius:6px;padding:10px}"
                           "pre{white-space:pre-wrap;background:#111827;color:#f9fafb;padding:14px;border-radius:6px}"
                           "h1,h2{margin:0 0 12px}</style></head><body>");
    html += QStringLiteral("<h1>AITrain Studio Training Delivery Report</h1>");
    html += QStringLiteral("<section><h2>Summary</h2><div class=\"grid\">"
                           "<div class=\"metric\"><b>Project</b><br>%1</div>"
                           "<div class=\"metric\"><b>Task type</b><br>%2</div>"
                           "<div class=\"metric\"><b>Backend</b><br>%3</div>"
                           "<div class=\"metric\"><b>Model</b><br>%4</div>"
                           "</div><p>Generated at %5. This is an offline local delivery artifact.</p></section>")
        .arg(htmlEscape(context.value(QStringLiteral("projectName")).toString(QStringLiteral("AITrain Studio"))))
        .arg(htmlEscape(context.value(QStringLiteral("taskType")).toString()))
        .arg(htmlEscape(context.value(QStringLiteral("trainingBackend")).toString()))
        .arg(htmlEscape(QDir::toNativeSeparators(context.value(QStringLiteral("modelPath")).toString())))
        .arg(htmlEscape(context.value(QStringLiteral("generatedAt")).toString(nowIso())));
    html += QStringLiteral("<section><h2>Evaluation</h2><pre>%1</pre></section>")
        .arg(htmlEscape(QString::fromUtf8(QJsonDocument(evaluationMetrics).toJson(QJsonDocument::Indented))));
    html += QStringLiteral("<section><h2>Benchmark</h2><div class=\"grid\">"
                           "<div class=\"metric\"><b>Runtime</b><br>%1</div>"
                           "<div class=\"metric\"><b>Status</b><br>%2</div>"
                           "<div class=\"metric\"><b>P95</b><br>%3 ms</div>"
                           "<div class=\"metric\"><b>Throughput</b><br>%4 /s</div>"
                           "</div></section>")
        .arg(htmlEscape(benchmark.value(QStringLiteral("runtime")).toString()))
        .arg(htmlEscape(benchmark.value(QStringLiteral("runtimeStatus")).toString()))
        .arg(latency.value(QStringLiteral("p95Ms")).toDouble(), 0, 'f', 2)
        .arg(latency.value(QStringLiteral("throughput")).toDouble(), 0, 'f', 2);
    html += QStringLiteral("<section><h2>Deployment Validation</h2><div class=\"grid\">"
                           "<div class=\"metric\"><b>Status</b><br>%1</div>"
                           "<div class=\"metric\"><b>Runtime</b><br>%2</div>"
                           "<div class=\"metric\"><b>Format</b><br>%3</div>"
                           "<div class=\"metric\"><b>Report</b><br>%4</div>"
                           "</div></section>")
        .arg(htmlEscape(deployment.value(QStringLiteral("status")).toString(QStringLiteral("not_provided"))))
        .arg(htmlEscape(deployment.value(QStringLiteral("runtime")).toString()))
        .arg(htmlEscape(deployment.value(QStringLiteral("format")).toString()))
        .arg(htmlEscape(QDir::toNativeSeparators(deployment.value(QStringLiteral("reportPath")).toString())));
    html += QStringLiteral("<section><h2>Customer OCR Acceptance</h2><div class=\"grid\">"
                           "<div class=\"metric\"><b>Status</b><br>%1</div>"
                           "<div class=\"metric\"><b>Rec Accuracy</b><br>%2</div>"
                           "<div class=\"metric\"><b>Rec CER</b><br>%3</div>"
                           "<div class=\"metric\"><b>Boundary</b><br>Customer-domain evidence only</div>"
                           "</div></section>")
        .arg(htmlEscape(customerOcr.value(QStringLiteral("status")).toString(QStringLiteral("not_provided"))))
        .arg(customerOcr.value(QStringLiteral("metrics")).toObject().value(QStringLiteral("recAccuracy")).toDouble(), 0, 'f', 6)
        .arg(customerOcr.value(QStringLiteral("metrics")).toObject().value(QStringLiteral("recCer")).toDouble(), 0, 'f', 6);
    html += QStringLiteral("<section><h2>Artifact Inventory</h2><table><thead><tr><th>Kind</th><th>Path</th><th>Bytes</th><th>SHA256</th></tr></thead><tbody>%1</tbody></table></section>")
        .arg(inventoryRows);
    html += QStringLiteral("<section><h2>Limitations</h2><ul>%1</ul></section>")
        .arg(limitationItems);
    html += QStringLiteral("<section><h2>Machine-readable Context</h2><pre>%1</pre></section>")
        .arg(htmlEscape(QString::fromUtf8(QJsonDocument(context).toJson(QJsonDocument::Indented))));
    html += QStringLiteral("</body></html>");
    return html;
}

QJsonObject deliveryManifestObject(const QJsonObject& context, const QJsonArray& inventory)
{
    QJsonArray checks;
    QJsonArray failures;
    const auto addCheck = [&checks, &failures](const QString& name, bool passed, const QString& message) {
        QJsonObject check;
        check.insert(QStringLiteral("name"), name);
        check.insert(QStringLiteral("passed"), passed);
        check.insert(QStringLiteral("message"), message);
        checks.append(check);
        if (!passed) {
            failures.append(check);
        }
    };

    bool hasModel = false;
    bool hasEvaluation = false;
    bool hasBenchmark = false;
    bool hasDeploymentValidation = false;
    bool hasForbiddenPath = false;
    QString forbiddenPath;
    for (const QJsonValue& value : inventory) {
        const QJsonObject item = value.toObject();
        const QString kind = item.value(QStringLiteral("kind")).toString();
        const QString path = item.value(QStringLiteral("path")).toString();
        hasModel = hasModel || kind == QStringLiteral("model");
        hasEvaluation = hasEvaluation || kind == QStringLiteral("evaluation_report");
        hasBenchmark = hasBenchmark || kind == QStringLiteral("benchmark_report");
        hasDeploymentValidation = hasDeploymentValidation || kind == QStringLiteral("deployment_validation_report");
        const QString lowerPath = QDir::fromNativeSeparators(path).toLower();
        if (lowerPath.contains(QStringLiteral("/.deps/"))
            || lowerPath.contains(QStringLiteral("private-key"))
            || lowerPath.contains(QStringLiteral("license-private-key"))) {
            hasForbiddenPath = true;
            forbiddenPath = path;
        }
    }

    addCheck(QStringLiteral("model_artifact_present"), hasModel, hasModel
        ? QStringLiteral("Selected model artifact is present.")
        : QStringLiteral("No model artifact was attached."));
    addCheck(QStringLiteral("evaluation_report_present"), hasEvaluation, hasEvaluation
        ? QStringLiteral("Evaluation report is attached.")
        : QStringLiteral("Evaluation report is missing; delivery remains review-only."));
    addCheck(QStringLiteral("benchmark_report_present"), hasBenchmark, hasBenchmark
        ? QStringLiteral("Benchmark report is attached.")
        : QStringLiteral("Benchmark report is missing; runtime acceptance remains unproven."));
    addCheck(QStringLiteral("deployment_validation_present"), hasDeploymentValidation, hasDeploymentValidation
        ? QStringLiteral("Deployment validation report is attached.")
        : QStringLiteral("Deployment validation is missing; exported artifact acceptance remains unproven."));
    addCheck(QStringLiteral("forbidden_content_absent"), !hasForbiddenPath, hasForbiddenPath
        ? QStringLiteral("Forbidden path appears in inventory: %1").arg(forbiddenPath)
        : QStringLiteral("No .deps or private-key paths were found in inventory."));

    QJsonObject reproducibility;
    reproducibility.insert(QStringLiteral("sourceTaskId"), context.value(QStringLiteral("sourceTaskId")).toString());
    reproducibility.insert(QStringLiteral("datasetSnapshotId"), context.value(QStringLiteral("datasetSnapshotId")).toInt());
    reproducibility.insert(QStringLiteral("datasetSnapshotHash"), context.value(QStringLiteral("datasetSnapshotHash")).toString());
    reproducibility.insert(QStringLiteral("datasetSnapshotManifest"), context.value(QStringLiteral("datasetSnapshotManifest")).toString());
    reproducibility.insert(QStringLiteral("trainingBackend"), context.value(QStringLiteral("trainingBackend")).toString());
    reproducibility.insert(QStringLiteral("modelPreset"), context.value(QStringLiteral("modelPreset")).toString());

    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 1);
    manifest.insert(QStringLiteral("kind"), QStringLiteral("delivery_manifest"));
    manifest.insert(QStringLiteral("createdAt"), nowIso());
    manifest.insert(QStringLiteral("projectName"), context.value(QStringLiteral("projectName")).toString());
    manifest.insert(QStringLiteral("modelPath"), context.value(QStringLiteral("modelPath")).toString());
    manifest.insert(QStringLiteral("datasetPath"), context.value(QStringLiteral("datasetPath")).toString());
    manifest.insert(QStringLiteral("datasetFormat"), context.value(QStringLiteral("datasetFormat")).toString());
    manifest.insert(QStringLiteral("taskType"), context.value(QStringLiteral("taskType")).toString());
    manifest.insert(QStringLiteral("artifactCount"), inventory.size());
    manifest.insert(QStringLiteral("inventory"), inventory);
    manifest.insert(QStringLiteral("reproducibility"), reproducibility);
    manifest.insert(QStringLiteral("checks"), checks);
    manifest.insert(QStringLiteral("failureCount"), failures.size());
    manifest.insert(QStringLiteral("failures"), failures);
    manifest.insert(QStringLiteral("packageStatus"), failures.isEmpty() ? QStringLiteral("ready_for_handoff_review") : QStringLiteral("blocked"));
    manifest.insert(QStringLiteral("note"), QStringLiteral("Manifest is an offline handoff index. It does not copy artifacts or certify customer acceptance."));
    return manifest;
}
} // namespace
WorkflowResult generateTrainingDeliveryReport(const QString& outputPath, const QJsonObject& context)
{
    QJsonObject reportContext = context;
    reportContext.insert(QStringLiteral("generatedAt"), nowIso());
    reportContext.insert(QStringLiteral("kind"), QStringLiteral("training_delivery_report"));
    reportContext.insert(QStringLiteral("scaffold"), false);
    reportContext.insert(QStringLiteral("note"), QStringLiteral("HTML delivery report summarizes lineage, evaluation, benchmark, and export artifacts for local handoff."));

    QJsonArray inventory;
    const auto addInventoryPath = [&inventory](const QString& kind, const QString& path, const QString& message) {
        if (path.isEmpty()) {
            return;
        }
        QFileInfo info(path);
        if (!info.exists()) {
            return;
        }
        QJsonObject item = pathArtifact(kind, path, message);
        item.insert(QStringLiteral("isDir"), info.isDir());
        item.insert(QStringLiteral("bytes"), QString::number(info.isDir() ? 0 : info.size()));
        item.insert(QStringLiteral("updatedAt"), info.lastModified().toUTC().toString(Qt::ISODateWithMs));
        if (!info.isDir()) {
            QString digestError;
            const QJsonObject digest = fileDigestObject(path, &digestError);
            item.insert(QStringLiteral("sha256"), digest.value(QStringLiteral("sha256")).toString());
            if (!digestError.isEmpty()) {
                item.insert(QStringLiteral("digestError"), digestError);
            }
        }
        inventory.append(item);
    };
    addInventoryPath(QStringLiteral("model"), reportContext.value(QStringLiteral("modelPath")).toString(), QStringLiteral("Selected model artifact"));
    addInventoryPath(QStringLiteral("dataset_snapshot"), reportContext.value(QStringLiteral("datasetSnapshotManifest")).toString(), QStringLiteral("Dataset snapshot manifest"));
    addInventoryPath(QStringLiteral("evaluation_report"), reportContext.value(QStringLiteral("evaluationReportPath")).toString(), QStringLiteral("Evaluation report"));
    addInventoryPath(QStringLiteral("benchmark_report"), reportContext.value(QStringLiteral("benchmarkReportPath")).toString(), QStringLiteral("Benchmark report"));
    addInventoryPath(QStringLiteral("deployment_validation_report"), reportContext.value(QStringLiteral("deploymentValidationReportPath")).toString(), QStringLiteral("Deployment validation report"));
    addInventoryPath(QStringLiteral("customer_ocr_acceptance"), reportContext.value(QStringLiteral("customerOcrAcceptanceReportPath")).toString(), QStringLiteral("Customer-domain OCR acceptance report"));
    addInventoryPath(QStringLiteral("export"), reportContext.value(QStringLiteral("exportPath")).toString(), QStringLiteral("Exported model"));
    addInventoryPath(QStringLiteral("export_report"), reportContext.value(QStringLiteral("exportReportPath")).toString(), QStringLiteral("Export report"));
    addInventoryPath(QStringLiteral("inference_predictions"), reportContext.value(QStringLiteral("inferencePredictionsPath")).toString(), QStringLiteral("Inference predictions"));
    addInventoryPath(QStringLiteral("inference_overlay"), reportContext.value(QStringLiteral("inferenceOverlayPath")).toString(), QStringLiteral("Inference overlay"));

    const QJsonObject evaluationSummary = evaluationDeliverySummary(reportContext.value(QStringLiteral("evaluationReportPath")).toString());
    const QJsonObject benchmarkSummary = benchmarkDeliverySummary(reportContext.value(QStringLiteral("benchmarkReportPath")).toString());
    const QJsonObject deploymentValidationSummary = readJsonObjectIfExists(reportContext.value(QStringLiteral("deploymentValidationReportPath")).toString());
    const QJsonObject customerOcrAcceptanceSummary = readJsonObjectIfExists(reportContext.value(QStringLiteral("customerOcrAcceptanceReportPath")).toString());
    const QJsonArray limitations = deliveryLimitations(reportContext, evaluationSummary, benchmarkSummary);
    reportContext.insert(QStringLiteral("evaluationSummary"), evaluationSummary);
    reportContext.insert(QStringLiteral("benchmarkSummary"), benchmarkSummary);
    reportContext.insert(QStringLiteral("deploymentValidationSummary"), deploymentValidationSummary);
    reportContext.insert(QStringLiteral("customerOcrAcceptanceSummary"), customerOcrAcceptanceSummary);
    reportContext.insert(QStringLiteral("limitations"), limitations);
    reportContext.insert(QStringLiteral("inventory"), inventory);
    QJsonObject deliveryManifest = deliveryManifestObject(reportContext, inventory);
    reportContext.insert(QStringLiteral("packageStatus"), deliveryManifest.value(QStringLiteral("packageStatus")).toString());

    QJsonObject modelCard;
    modelCard.insert(QStringLiteral("schemaVersion"), 1);
    modelCard.insert(QStringLiteral("createdAt"), nowIso());
    modelCard.insert(QStringLiteral("modelPath"), reportContext.value(QStringLiteral("modelPath")).toString());
    modelCard.insert(QStringLiteral("modelFamily"), reportContext.value(QStringLiteral("modelFamily")).toString());
    modelCard.insert(QStringLiteral("taskType"), reportContext.value(QStringLiteral("taskType")).toString());
    modelCard.insert(QStringLiteral("trainingBackend"), reportContext.value(QStringLiteral("trainingBackend")).toString());
    modelCard.insert(QStringLiteral("modelPreset"), reportContext.value(QStringLiteral("modelPreset")).toString());
    modelCard.insert(QStringLiteral("datasetPath"), reportContext.value(QStringLiteral("datasetPath")).toString());
    modelCard.insert(QStringLiteral("datasetFormat"), reportContext.value(QStringLiteral("datasetFormat")).toString());
    modelCard.insert(QStringLiteral("datasetSnapshotId"), reportContext.value(QStringLiteral("datasetSnapshotId")).toInt());
    modelCard.insert(QStringLiteral("datasetSnapshotHash"), reportContext.value(QStringLiteral("datasetSnapshotHash")).toString());
    modelCard.insert(QStringLiteral("datasetSnapshotManifest"), reportContext.value(QStringLiteral("datasetSnapshotManifest")).toString());
    modelCard.insert(QStringLiteral("sourceTaskId"), reportContext.value(QStringLiteral("sourceTaskId")).toString());
    modelCard.insert(QStringLiteral("evaluationReportPath"), reportContext.value(QStringLiteral("evaluationReportPath")).toString());
    modelCard.insert(QStringLiteral("benchmarkReportPath"), reportContext.value(QStringLiteral("benchmarkReportPath")).toString());
    modelCard.insert(QStringLiteral("exportPath"), reportContext.value(QStringLiteral("exportPath")).toString());
    modelCard.insert(QStringLiteral("evaluationSummary"), evaluationSummary);
    modelCard.insert(QStringLiteral("benchmarkSummary"), benchmarkSummary);
    modelCard.insert(QStringLiteral("deploymentValidationSummary"), deploymentValidationSummary);
    modelCard.insert(QStringLiteral("customerOcrAcceptanceSummary"), customerOcrAcceptanceSummary);
    modelCard.insert(QStringLiteral("inventory"), inventory);
    modelCard.insert(QStringLiteral("limitations"), limitations);
    modelCard.insert(QStringLiteral("packageStatus"), reportContext.value(QStringLiteral("packageStatus")).toString());

    QString error;
    const QString modelCardPath = QDir(outputPath).filePath(QStringLiteral("model_card.json"));
    const QString inventoryPath = QDir(outputPath).filePath(QStringLiteral("delivery_artifact_inventory.json"));
    const QString manifestPath = QDir(outputPath).filePath(QStringLiteral("delivery_manifest.json"));
    deliveryManifest.insert(QStringLiteral("manifestPath"), manifestPath);
    modelCard.insert(QStringLiteral("deliveryManifestPath"), manifestPath);
    if (!writeJsonFile(modelCardPath, modelCard, &error)) {
        return failedResult(error);
    }
    if (!writeJsonFile(inventoryPath, QJsonObject{
            {QStringLiteral("schemaVersion"), 1},
            {QStringLiteral("createdAt"), nowIso()},
            {QStringLiteral("artifactCount"), inventory.size()},
            {QStringLiteral("items"), inventory}}, &error)) {
        return failedResult(error);
    }
    if (!writeJsonFile(manifestPath, deliveryManifest, &error)) {
        return failedResult(error);
    }
    reportContext.insert(QStringLiteral("modelCardPath"), modelCardPath);
    reportContext.insert(QStringLiteral("artifactInventoryPath"), inventoryPath);
    reportContext.insert(QStringLiteral("deliveryManifestPath"), manifestPath);
    reportContext.insert(QStringLiteral("deliveryManifest"), deliveryManifest);
    reportContext.insert(QStringLiteral("artifactCount"), inventory.size());

    const QString htmlPath = QDir(outputPath).filePath(QStringLiteral("training_delivery_report.html"));
    if (!writeTextFile(htmlPath, htmlReport(reportContext), &error)) {
        return failedResult(error);
    }
    const QString jsonPath = QDir(outputPath).filePath(QStringLiteral("training_delivery_report.json"));
    if (!writeJsonFile(jsonPath, reportContext, &error)) {
        return failedResult(error);
    }
    reportContext.insert(QStringLiteral("reportPath"), htmlPath);
    reportContext.insert(QStringLiteral("jsonPath"), jsonPath);
    return resultFromReport(htmlPath, reportContext);
}
} // namespace aitrain
