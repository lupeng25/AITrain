#include "aitrain/core/ProductWorkflow.h"

#include "ProductWorkflowSupport.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>

namespace aitrain {
using namespace workflow_detail;
namespace {
QString normalizedPathForScan(const QString& path)
{
    return QDir::fromNativeSeparators(QDir::cleanPath(path)).toLower();
}

bool pathLooksLikePublicSmokeData(const QString& path)
{
    const QString normalized = normalizedPathForScan(path);
    if (normalized.isEmpty()) {
        return false;
    }
    return normalized.contains(QStringLiteral("/.deps/production-ocr-data"))
        || normalized.contains(QStringLiteral("total-text"))
        || normalized.contains(QStringLiteral("/public"))
        || normalized.contains(QStringLiteral("/generated"))
        || normalized.contains(QStringLiteral("/synthetic"))
        || normalized.contains(QStringLiteral("/smoke"))
        || normalized.contains(QStringLiteral("/examples/"))
        || normalized.contains(QStringLiteral("/example/"));
}

QStringList stringListFromPaths(const QJsonObject& object, const QStringList& keys)
{
    QStringList paths;
    for (const QString& key : keys) {
        const QJsonValue value = object.value(key);
        if (value.isArray()) {
            const QJsonArray array = value.toArray();
            for (const QJsonValue& item : array) {
                const QString path = QDir::fromNativeSeparators(item.toString().trimmed());
                if (!path.isEmpty() && !paths.contains(path)) {
                    paths.append(path);
                }
            }
        } else {
            const QString path = QDir::fromNativeSeparators(value.toString().trimmed());
            if (!path.isEmpty() && !paths.contains(path)) {
                paths.append(path);
            }
        }
    }
    return paths;
}

QJsonArray jsonArrayFromStringList(const QStringList& values)
{
    QJsonArray array;
    for (const QString& value : values) {
        array.append(value);
    }
    return array;
}

QJsonObject checkObjectWithDetails(
    const QString& name,
    const QString& status,
    bool passed,
    const QString& message,
    const QJsonObject& details = {})
{
    QJsonObject check;
    check.insert(QStringLiteral("name"), name);
    check.insert(QStringLiteral("status"), status);
    check.insert(QStringLiteral("passed"), passed);
    check.insert(QStringLiteral("message"), message);
    if (!details.isEmpty()) {
        check.insert(QStringLiteral("details"), details);
    }
    return check;
}

void appendGateCheck(QJsonArray* checks, bool* blocked, const QJsonObject& check)
{
    if (!checks || !blocked) {
        return;
    }
    checks->append(check);
    const QString status = check.value(QStringLiteral("status")).toString();
    if (!check.value(QStringLiteral("passed")).toBool()
        && status != QStringLiteral("warning")
        && status != QStringLiteral("not_applicable")) {
        *blocked = true;
    }
}

bool pathExists(const QString& path)
{
    return !path.trimmed().isEmpty() && QFileInfo::exists(path);
}

int countImagesUnder(const QString& path, int limit)
{
    if (path.isEmpty()) {
        return 0;
    }
    const QFileInfo info(path);
    if (!info.exists()) {
        return 0;
    }
    if (info.isFile()) {
        return isImageFile(info.suffix()) ? 1 : 0;
    }

    int count = 0;
    QDirIterator iterator(info.absoluteFilePath(), imageNameFilters(), QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        iterator.next();
        ++count;
        if (limit > 0 && count >= limit) {
            break;
        }
    }
    return count;
}

QJsonObject pathEvidenceObject(const QString& path)
{
    QJsonObject evidence;
    evidence.insert(QStringLiteral("path"), path);
    const QFileInfo info(path);
    evidence.insert(QStringLiteral("exists"), info.exists());
    evidence.insert(QStringLiteral("isDir"), info.exists() && info.isDir());
    if (info.exists() && !info.isDir()) {
        QString digestError;
        const QJsonObject digest = fileDigestObject(path, &digestError);
        evidence.insert(QStringLiteral("digest"), digest);
        if (!digestError.isEmpty()) {
            evidence.insert(QStringLiteral("digestError"), digestError);
        }
    }
    return evidence;
}

QJsonArray pathEvidenceArray(const QStringList& paths)
{
    QJsonArray array;
    for (const QString& path : paths) {
        array.append(pathEvidenceObject(path));
    }
    return array;
}

bool numericValueAtPath(const QJsonObject& object, const QString& path, double* value)
{
    QJsonValue current(object);
    const QStringList parts = path.split(QLatin1Char('.'), QString::SkipEmptyParts);
    for (const QString& part : parts) {
        if (!current.isObject()) {
            return false;
        }
        current = current.toObject().value(part);
    }
    if (current.isDouble()) {
        if (value) {
            *value = current.toDouble();
        }
        return true;
    }
    if (current.isString()) {
        bool ok = false;
        const double parsed = current.toString().toDouble(&ok);
        if (ok && value) {
            *value = parsed;
        }
        return ok;
    }
    return false;
}

double firstNumericValue(const QJsonObject& object, const QStringList& paths, double fallback, bool* found = nullptr)
{
    for (const QString& path : paths) {
        double value = 0.0;
        if (numericValueAtPath(object, path, &value)) {
            if (found) {
                *found = true;
            }
            return value;
        }
    }
    if (found) {
        *found = false;
    }
    return fallback;
}

QJsonObject readOptionalJson(const QString& path)
{
    QString error;
    QJsonObject object;
    if (!path.isEmpty() && QFileInfo::exists(path) && readJsonFile(path, &object, &error)) {
        return object;
    }
    return {};
}

QString customerOcrSummaryMarkdown(const QJsonObject& report)
{
    const QJsonObject thresholds = report.value(QStringLiteral("thresholds")).toObject();
    const QJsonObject metrics = report.value(QStringLiteral("metrics")).toObject();
    QString markdown;
    markdown += QStringLiteral("# Customer OCR Acceptance\n\n");
    markdown += QStringLiteral("- Status: %1\n").arg(report.value(QStringLiteral("status")).toString());
    markdown += QStringLiteral("- Created at: %1\n").arg(report.value(QStringLiteral("createdAt")).toString());
    markdown += QStringLiteral("- Rec accuracy: %1 (min %2)\n")
        .arg(metrics.value(QStringLiteral("recAccuracy")).toDouble(), 0, 'f', 6)
        .arg(thresholds.value(QStringLiteral("minRecAccuracy")).toDouble(), 0, 'f', 6);
    markdown += QStringLiteral("- Rec CER: %1 (max %2)\n")
        .arg(metrics.value(QStringLiteral("recCer")).toDouble(), 0, 'f', 6)
        .arg(thresholds.value(QStringLiteral("maxRecCer")).toDouble(), 0, 'f', 6);
    markdown += QStringLiteral("- Rec samples: %1\n\n").arg(metrics.value(QStringLiteral("recSamples")).toInt());
    markdown += QStringLiteral("## Checks\n\n");
    const QJsonArray checks = report.value(QStringLiteral("checks")).toArray();
    for (const QJsonValue& value : checks) {
        const QJsonObject check = value.toObject();
        markdown += QStringLiteral("- [%1] %2: %3\n")
            .arg(check.value(QStringLiteral("status")).toString())
            .arg(check.value(QStringLiteral("name")).toString())
            .arg(check.value(QStringLiteral("message")).toString());
    }
    markdown += QStringLiteral("\nPublic datasets such as Total-Text, generated smoke data, or .deps examples can prove workflow smoke only. Production OCR claims require customer-domain evidence.\n");
    markdown += QStringLiteral("\nSensitive data reminder: keep customer images and reports in the customer project/output folder and review them before sharing outside the delivery team.\n");
    return markdown;
}

QJsonObject processSnapshot(const QString& program, const QStringList& arguments, int timeoutMs)
{
    QJsonObject result;
    result.insert(QStringLiteral("program"), program);
    result.insert(QStringLiteral("arguments"), jsonArrayFromStringList(arguments));
    QProcess process;
    process.start(program, arguments);
    if (!process.waitForStarted(3000)) {
        result.insert(QStringLiteral("status"), QStringLiteral("missing"));
        result.insert(QStringLiteral("message"), process.errorString());
        return result;
    }
    const bool finished = process.waitForFinished(timeoutMs);
    if (!finished) {
        process.kill();
        process.waitForFinished(1000);
        result.insert(QStringLiteral("status"), QStringLiteral("timeout"));
        result.insert(QStringLiteral("message"), QStringLiteral("Command timed out."));
        return result;
    }
    result.insert(QStringLiteral("status"), process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0
        ? QStringLiteral("ok")
        : QStringLiteral("failed"));
    result.insert(QStringLiteral("exitCode"), process.exitCode());
    result.insert(QStringLiteral("stdout"), QString::fromLocal8Bit(process.readAllStandardOutput()).trimmed());
    result.insert(QStringLiteral("stderr"), QString::fromLocal8Bit(process.readAllStandardError()).trimmed());
    return result;
}

QString diagnosticsSummaryMarkdown(const QJsonObject& report)
{
    QString markdown;
    markdown += QStringLiteral("# AITrain Diagnostic Bundle\n\n");
    markdown += QStringLiteral("- Created at: %1\n").arg(report.value(QStringLiteral("createdAt")).toString());
    markdown += QStringLiteral("- Project: %1\n").arg(report.value(QStringLiteral("projectPath")).toString());
    markdown += QStringLiteral("- Worker: %1\n").arg(report.value(QStringLiteral("workerExecutable")).toString());
    markdown += QStringLiteral("- License: %1\n\n").arg(report.value(QStringLiteral("licenseSummary")).toObject().value(QStringLiteral("status")).toString());
    markdown += QStringLiteral("## Key Status\n\n");
    markdown += QStringLiteral("- TensorRT: %1\n").arg(report.value(QStringLiteral("tensorRtBackend")).toObject().value(QStringLiteral("status")).toString());
    markdown += QStringLiteral("- ONNX Runtime inference: %1\n").arg(report.value(QStringLiteral("onnxRuntimeInferenceAvailable")).toBool() ? QStringLiteral("available") : QStringLiteral("unavailable"));
    markdown += QStringLiteral("- Runtime dependency checks: %1\n").arg(report.value(QStringLiteral("runtimeDependencyChecks")).toArray().size());
    markdown += QStringLiteral("- Recent tasks: %1\n").arg(report.value(QStringLiteral("recentTasks")).toArray().size());
    markdown += QStringLiteral("- Artifact index items: %1\n\n").arg(report.value(QStringLiteral("artifactIndex")).toArray().size());
    markdown += QStringLiteral("This bundle is diagnostic evidence. It does not modify Python, CUDA, or global environment settings.\n");
    return markdown;
}

QString detectDeploymentFormat(const QString& modelPath, const QString& requestedFormat)
{
    const QString explicitFormat = requestedFormat.trimmed().toLower();
    if (!explicitFormat.isEmpty()) {
        return explicitFormat;
    }
    const QString suffix = QFileInfo(modelPath).suffix().toLower();
    if (suffix == QStringLiteral("onnx")) {
        return QStringLiteral("onnx");
    }
    if (suffix == QStringLiteral("param") || suffix == QStringLiteral("bin")) {
        return QStringLiteral("ncnn");
    }
    if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan")) {
        return QStringLiteral("tensorrt");
    }
    if (suffix == QStringLiteral("json") && QFileInfo(modelPath).fileName().compare(QStringLiteral("anomaly_sidecar.json"), Qt::CaseInsensitive) == 0) {
        return QStringLiteral("anomalib_python");
    }
    return QStringLiteral("unknown");
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

QString anomalibPythonExecutable(const QJsonObject& options)
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
        QDir(appDir).absoluteFilePath(QStringLiteral("python_env/Scripts/python.exe")),
        QDir(appDir).absoluteFilePath(QStringLiteral("python_env/python.exe")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../python_env/Scripts/python.exe")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../python_env/python.exe")),
        QDir::current().absoluteFilePath(QStringLiteral("python_env/Scripts/python.exe")),
        QDir::current().absoluteFilePath(QStringLiteral("python_env/python.exe")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/python-3.13.13-embed-amd64/python.exe")),
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

QString packagedPythonEnvRoot()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).absoluteFilePath(QStringLiteral("python_env")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../python_env")),
        QDir::current().absoluteFilePath(QStringLiteral("python_env"))
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("Scripts/python.exe")))
            || QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("python.exe")))) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return {};
}

void configurePackagedPythonEnvironment(QProcessEnvironment* environment)
{
    if (!environment) {
        return;
    }
    const QString pythonRoot = packagedPythonEnvRoot();
    if (pythonRoot.isEmpty()) {
        return;
    }
    QStringList pathEntries;
    pathEntries << QDir(pythonRoot).filePath(QStringLiteral("Scripts"));
    pathEntries << pythonRoot;
    const QString existingPath = environment->value(QStringLiteral("PATH"));
    if (!existingPath.isEmpty()) {
        pathEntries << existingPath;
    }
    environment->insert(QStringLiteral("PATH"), pathEntries.join(QDir::listSeparator()));
}

QString anomalibAdapterScriptPath(const QJsonObject& options)
{
    const QString requested = options.value(QStringLiteral("anomalibAdapterScript")).toString().trimmed();
    if (!requested.isEmpty() && QFileInfo::exists(requested)) {
        return QFileInfo(requested).absoluteFilePath();
    }
    const QString script = QStringLiteral("python_trainers/anomaly/anomalib_adapter.py");
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

bool pythonModuleAvailable(const QString& python, const QString& moduleName)
{
    if (python.isEmpty()) {
        return false;
    }
    QProcess process;
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    configurePackagedPythonEnvironment(&environment);
    process.setProcessEnvironment(environment);
    process.start(python, QStringList()
        << QStringLiteral("-c")
        << QStringLiteral("import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('%1') else 3)").arg(moduleName));
    return process.waitForStarted(2000)
        && process.waitForFinished(5000)
        && process.exitStatus() == QProcess::NormalExit
        && process.exitCode() == 0;
}

QString ncnnBinPathForParam(const QString& modelPath)
{
    const QFileInfo info(modelPath);
    if (info.suffix().compare(QStringLiteral("bin"), Qt::CaseInsensitive) == 0) {
        return info.absoluteFilePath();
    }
    return QDir(info.absolutePath()).filePath(info.completeBaseName() + QStringLiteral(".bin"));
}

QString ncnnFailureCategory(const QString& checkName, const QString& message)
{
    const QString lower = message.toLower();
    if (checkName == QStringLiteral("ncnn_runtime_available")
        || lower.contains(QStringLiteral("configure aitrain_ncnn_root"))
        || lower.contains(QStringLiteral("ncnn inference is not enabled"))) {
        return QStringLiteral("sdk_missing");
    }
    if (checkName == QStringLiteral("ncnn_sample_image_present")) {
        return QStringLiteral("sample_missing");
    }
    if (checkName == QStringLiteral("ncnn_sidecar_bin_exists")
        || lower.contains(QStringLiteral("sidecar"))
        || lower.contains(QStringLiteral("explicit ncnn runtime config"))
        || lower.contains(QStringLiteral("missing inputblob"))
        || lower.contains(QStringLiteral("missing outputblobs"))
        || lower.contains(QStringLiteral("inferred model family: <empty>"))) {
        return QStringLiteral("sidecar_missing");
    }
    if (lower.contains(QStringLiteral("unsupported layer type"))
        || lower.contains(QStringLiteral(" shape "))
        || lower.contains(QStringLiteral("'shape'"))) {
        return QStringLiteral("unsupported_layer");
    }
    return QStringLiteral("runtime_failed");
}

QString ncnnNextAction(const QString& failureCategory)
{
    if (failureCategory == QStringLiteral("sdk_missing")) {
        return QStringLiteral("Configure AITRAIN_NCNN_ROOT with an NCNN SDK/runtime, rebuild, then rerun deployment validation.");
    }
    if (failureCategory == QStringLiteral("sample_missing")) {
        return QStringLiteral("Provide sampleImagePath so the NCNN artifact can be loaded and executed.");
    }
    if (failureCategory == QStringLiteral("sidecar_missing")) {
        return QStringLiteral("Provide the AITrain export sidecar or explicit modelFamily, classNames, inputBlob, outputBlobs, decoder, inputSize, strides, and regMax.");
    }
    if (failureCategory == QStringLiteral("unsupported_layer")) {
        return QStringLiteral("Re-export a static-shape NCNN artifact or use a pnnx-compatible preconverted model; unsupported layers such as Shape are not accepted as passed smoke evidence.");
    }
    return QStringLiteral("Inspect deployment_validation_report.json, Worker logs, blob names, decoder config, and NCNN runtime DLL layout.");
}

QJsonArray ncnnDiagnosticHints(const QString& failureCategory)
{
    QJsonArray hints;
    if (failureCategory == QStringLiteral("sdk_missing")) {
        hints.append(QStringLiteral("Set CMake -DAITRAIN_NCNN_ROOT=<ncnn-sdk-root> and keep ncnn.dll next to the Worker or under runtimes/ncnn."));
        hints.append(QStringLiteral("Run tools/phase-ncnn-runtime-smoke.ps1 after rebuilding with NCNN enabled."));
    } else if (failureCategory == QStringLiteral("sample_missing")) {
        hints.append(QStringLiteral("Use a readable jpg/png/bmp/tif sample image representative of the exported model input domain."));
    } else if (failureCategory == QStringLiteral("sidecar_missing")) {
        hints.append(QStringLiteral("AITrain-exported NCNN artifacts should keep the matching .aitrain-export.json sidecar next to the .param/.bin files."));
        hints.append(QStringLiteral("External nihui/Tencent/pnnx artifacts must pass explicit blob and decoder config; blind auto-detection is intentionally blocked."));
    } else if (failureCategory == QStringLiteral("unsupported_layer")) {
        hints.append(QStringLiteral("YOLOv8-seg ONNX converted by onnx2ncnn may retain Shape layers; this is recorded as failed compatibility evidence."));
        hints.append(QStringLiteral("Use preconverted pnnx/DFL NCNN artifacts with an explicit AITrain sidecar for segmentation smoke."));
    } else {
        hints.append(QStringLiteral("Check input/output blob names, class count, decoder type, and mask prototype outputs."));
        hints.append(QStringLiteral("Keep CPU NCNN as the default acceptance path; Vulkan is not required for the RC gate."));
    }
    return hints;
}

QJsonObject withNcnnFailureDetails(
    const QString& checkName,
    const QString& message,
    QJsonObject details = {})
{
    const QString category = ncnnFailureCategory(checkName, message);
    details.insert(QStringLiteral("errorCode"), category);
    details.insert(QStringLiteral("failureCategory"), category);
    details.insert(QStringLiteral("nextAction"), ncnnNextAction(category));
    details.insert(QStringLiteral("diagnosticHints"), ncnnDiagnosticHints(category));
    return details;
}

void applyFailureSummaryFromChecks(QJsonObject* report, const QJsonArray& checks)
{
    if (!report || report->value(QStringLiteral("ok")).toBool()) {
        return;
    }
    if (report->contains(QStringLiteral("failureCategory"))) {
        return;
    }
    for (const QJsonValue& value : checks) {
        const QJsonObject check = value.toObject();
        if (check.value(QStringLiteral("passed")).toBool()) {
            continue;
        }
        const QJsonObject details = check.value(QStringLiteral("details")).toObject();
        const QString category = details.value(QStringLiteral("failureCategory")).toString();
        if (category.isEmpty()) {
            continue;
        }
        report->insert(QStringLiteral("errorCode"), details.value(QStringLiteral("errorCode")).toString(category));
        report->insert(QStringLiteral("failureCategory"), category);
        report->insert(QStringLiteral("nextAction"), details.value(QStringLiteral("nextAction")).toString());
        report->insert(QStringLiteral("diagnosticHints"), details.value(QStringLiteral("diagnosticHints")).toArray());
        return;
    }
}

QString deploymentSummaryMarkdown(const QJsonObject& report)
{
    QString markdown;
    markdown += QStringLiteral("# Deployment Artifact Validation\n\n");
    markdown += QStringLiteral("- Status: %1\n").arg(report.value(QStringLiteral("status")).toString());
    markdown += QStringLiteral("- Format: %1\n").arg(report.value(QStringLiteral("format")).toString());
    markdown += QStringLiteral("- Runtime: %1\n").arg(report.value(QStringLiteral("runtime")).toString());
    markdown += QStringLiteral("- Model: %1\n").arg(QDir::toNativeSeparators(report.value(QStringLiteral("modelPath")).toString()));
    markdown += QStringLiteral("- Sample image: %1\n\n").arg(QDir::toNativeSeparators(report.value(QStringLiteral("sampleImagePath")).toString()));
    const QString failureCategory = report.value(QStringLiteral("failureCategory")).toString();
    if (!failureCategory.isEmpty()) {
        markdown += QStringLiteral("- Failure category: %1\n").arg(failureCategory);
        markdown += QStringLiteral("- Next action: %1\n\n").arg(report.value(QStringLiteral("nextAction")).toString());
    }
    markdown += QStringLiteral("## Checks\n\n");
    const QJsonArray checks = report.value(QStringLiteral("checks")).toArray();
    for (const QJsonValue& value : checks) {
        const QJsonObject check = value.toObject();
        markdown += QStringLiteral("- [%1] %2: %3\n")
            .arg(check.value(QStringLiteral("status")).toString())
            .arg(check.value(QStringLiteral("name")).toString())
            .arg(check.value(QStringLiteral("message")).toString());
    }
    markdown += QStringLiteral("\nNCNN validation requires a configured NCNN SDK/runtime and a sample image for runtime inference.\n");
    return markdown;
}

QJsonObject predictionsDocument(
    const QString& modelPath,
    const QString& sampleImagePath,
    const QString& taskType,
    const QString& runtime,
    int elapsedMs,
    const QJsonArray& predictions)
{
    QJsonObject document;
    document.insert(QStringLiteral("createdAt"), nowIso());
    document.insert(QStringLiteral("modelPath"), modelPath);
    document.insert(QStringLiteral("imagePath"), sampleImagePath);
    document.insert(QStringLiteral("taskType"), taskType);
    document.insert(QStringLiteral("runtime"), runtime);
    document.insert(QStringLiteral("elapsedMs"), elapsedMs);
    document.insert(QStringLiteral("predictionCount"), predictions.size());
    document.insert(QStringLiteral("predictions"), predictions);
    return document;
}

void setDeploymentStatusFromChecks(QJsonObject* report, const QJsonArray& checks, const QString& preferredStatus = QString())
{
    if (!report) {
        return;
    }
    QString status = preferredStatus;
    if (status.isEmpty()) {
        status = QStringLiteral("passed");
        for (const QJsonValue& value : checks) {
            const QJsonObject check = value.toObject();
            const QString checkStatus = check.value(QStringLiteral("status")).toString();
            if (checkStatus == QStringLiteral("hardware-blocked")) {
                status = QStringLiteral("hardware-blocked");
                break;
            }
            if (!check.value(QStringLiteral("passed")).toBool()
                && checkStatus != QStringLiteral("not_applicable")) {
                status = checkStatus == QStringLiteral("blocked") ? QStringLiteral("blocked") : QStringLiteral("failed");
            }
        }
    }
    report->insert(QStringLiteral("status"), status);
    report->insert(QStringLiteral("ok"), status == QStringLiteral("passed"));
    report->insert(QStringLiteral("deploymentConclusion"), status);
}
} // namespace

WorkflowResult runCustomerOcrAcceptanceReport(const QString& outputPath, const QJsonObject& options)
{
    const QDir outputDir(outputPath);
    if (!QDir().mkpath(outputDir.absolutePath())) {
        return failedResult(QStringLiteral("Cannot create OCR acceptance output directory: %1").arg(outputPath));
    }

    const QString detDatasetPath = QDir::fromNativeSeparators(options.value(QStringLiteral("detDatasetPath")).toString().trimmed());
    const QString recDatasetPath = QDir::fromNativeSeparators(options.value(QStringLiteral("recDatasetPath")).toString().trimmed());
    const QString systemImagesPath = QDir::fromNativeSeparators(options.value(QStringLiteral("systemImagesPath")).toString().trimmed());
    const QString customerDatasetPath = QDir::fromNativeSeparators(options.value(QStringLiteral("customerDataset")).toString().trimmed());
    const QString detReportPath = QDir::fromNativeSeparators(options.value(QStringLiteral("detReportPath")).toString().trimmed());
    const QString recReportPath = QDir::fromNativeSeparators(options.value(QStringLiteral("recReportPath")).toString().trimmed());
    const QString systemReportPath = QDir::fromNativeSeparators(options.value(QStringLiteral("systemReportPath")).toString().trimmed());
    const bool allowPublicLikeData = options.value(QStringLiteral("allowPublicLikeData")).toBool(false);
    const bool requireFullDomainEvidence = options.value(QStringLiteral("requireFullDomainEvidence")).toBool(true);
    const double minRecAccuracy = options.value(QStringLiteral("minRecAccuracy")).toDouble(0.70);
    const double maxRecCer = options.value(QStringLiteral("maxRecCer")).toDouble(0.30);
    const int minDetSamples = options.value(QStringLiteral("minDetSamples")).toInt(1);
    const int minRecSamples = options.value(QStringLiteral("minRecSamples")).toInt(1);
    const int minSystemImages = options.value(QStringLiteral("minSystemImages")).toInt(1);

    QStringList dataPaths;
    dataPaths << detDatasetPath << recDatasetPath << systemImagesPath << customerDatasetPath;
    dataPaths.removeAll(QString());
    const QStringList extraDataPaths = stringListFromPaths(options, QStringList()
        << QStringLiteral("dataPaths")
        << QStringLiteral("customerDataPaths"));
    for (const QString& path : extraDataPaths) {
        if (!dataPaths.contains(path)) {
            dataPaths.append(path);
        }
    }

    bool blocked = false;
    QJsonArray checks;
    appendGateCheck(&checks, &blocked, checkObjectWithDetails(
        QStringLiteral("customer_domain_data_present"),
        dataPaths.isEmpty() ? QStringLiteral("blocked") : QStringLiteral("passed"),
        !dataPaths.isEmpty(),
        dataPaths.isEmpty()
            ? QStringLiteral("No customer-domain OCR dataset or image root was provided.")
            : QStringLiteral("Customer-domain data roots were provided."),
        QJsonObject{{QStringLiteral("paths"), jsonArrayFromStringList(dataPaths)}}));

    if (requireFullDomainEvidence) {
        const int detImageCount = countImagesUnder(detDatasetPath, minDetSamples);
        const int recImageCount = countImagesUnder(recDatasetPath, minRecSamples);
        const int systemImageCount = countImagesUnder(systemImagesPath, minSystemImages);
        appendGateCheck(&checks, &blocked, checkObjectWithDetails(
            QStringLiteral("det_dataset_samples"),
            detImageCount >= minDetSamples ? QStringLiteral("passed") : QStringLiteral("blocked"),
            detImageCount >= minDetSamples,
            QStringLiteral("Det dataset images: %1 / required %2.").arg(detImageCount).arg(minDetSamples),
            QJsonObject{{QStringLiteral("path"), detDatasetPath}, {QStringLiteral("imageCount"), detImageCount}}));
        appendGateCheck(&checks, &blocked, checkObjectWithDetails(
            QStringLiteral("rec_dataset_samples"),
            recImageCount >= minRecSamples ? QStringLiteral("passed") : QStringLiteral("blocked"),
            recImageCount >= minRecSamples,
            QStringLiteral("Rec dataset images: %1 / required %2.").arg(recImageCount).arg(minRecSamples),
            QJsonObject{{QStringLiteral("path"), recDatasetPath}, {QStringLiteral("imageCount"), recImageCount}}));
        appendGateCheck(&checks, &blocked, checkObjectWithDetails(
            QStringLiteral("system_image_samples"),
            systemImageCount >= minSystemImages ? QStringLiteral("passed") : QStringLiteral("blocked"),
            systemImageCount >= minSystemImages,
            QStringLiteral("System validation images: %1 / required %2.").arg(systemImageCount).arg(minSystemImages),
            QJsonObject{{QStringLiteral("path"), systemImagesPath}, {QStringLiteral("imageCount"), systemImageCount}}));
    }

    bool publicLike = false;
    QString publicLikePath;
    for (const QString& path : dataPaths) {
        if (pathLooksLikePublicSmokeData(path)) {
            publicLike = true;
            publicLikePath = path;
            break;
        }
    }
    appendGateCheck(&checks, &blocked, checkObjectWithDetails(
        QStringLiteral("customer_domain_not_public_smoke"),
        (!publicLike || allowPublicLikeData) ? QStringLiteral("passed") : QStringLiteral("blocked"),
        !publicLike || allowPublicLikeData,
        publicLike
            ? QStringLiteral("Public/generated/smoke-like OCR data detected: %1.").arg(publicLikePath)
            : QStringLiteral("No public/generated/smoke-like OCR data marker was detected."),
        QJsonObject{{QStringLiteral("allowPublicLikeData"), allowPublicLikeData}}));

    const bool recReportExists = pathExists(recReportPath);
    appendGateCheck(&checks, &blocked, checkObjectWithDetails(
        QStringLiteral("rec_official_report_present"),
        recReportExists ? QStringLiteral("passed") : QStringLiteral("blocked"),
        recReportExists,
        recReportExists
            ? QStringLiteral("OCR Rec official/evaluation report is present.")
            : QStringLiteral("OCR Rec official/evaluation report is required."),
        pathEvidenceObject(recReportPath)));
    if (requireFullDomainEvidence) {
        appendGateCheck(&checks, &blocked, checkObjectWithDetails(
            QStringLiteral("det_official_report_present"),
            pathExists(detReportPath) ? QStringLiteral("passed") : QStringLiteral("blocked"),
            pathExists(detReportPath),
            pathExists(detReportPath)
                ? QStringLiteral("OCR Det official/evaluation report is present.")
                : QStringLiteral("OCR Det official/evaluation report is required for full customer acceptance."),
            pathEvidenceObject(detReportPath)));
        appendGateCheck(&checks, &blocked, checkObjectWithDetails(
            QStringLiteral("system_official_report_present"),
            pathExists(systemReportPath) ? QStringLiteral("passed") : QStringLiteral("blocked"),
            pathExists(systemReportPath),
            pathExists(systemReportPath)
                ? QStringLiteral("OCR System official/evaluation report is present.")
                : QStringLiteral("OCR System official/evaluation report is required for full customer acceptance."),
            pathEvidenceObject(systemReportPath)));
    }

    const QJsonObject recReport = readOptionalJson(recReportPath);
    bool hasRecAccuracy = false;
    bool hasRecCer = false;
    const double recAccuracy = firstNumericValue(recReport, QStringList()
        << QStringLiteral("metrics.accuracy")
        << QStringLiteral("accuracy")
        << QStringLiteral("recAccuracy")
        << QStringLiteral("recognition.accuracy")
        << QStringLiteral("eval.accuracy"), 0.0, &hasRecAccuracy);
    const double recCer = firstNumericValue(recReport, QStringList()
        << QStringLiteral("metrics.cer")
        << QStringLiteral("cer")
        << QStringLiteral("recCer")
        << QStringLiteral("recognition.cer")
        << QStringLiteral("eval.cer"), 1.0, &hasRecCer);
    const int recSamplesFromReport = static_cast<int>(firstNumericValue(recReport, QStringList()
        << QStringLiteral("metrics.samples")
        << QStringLiteral("sampleCount")
        << QStringLiteral("samples")
        << QStringLiteral("summary.sampleCount"), 0.0));
    const int recSamples = qMax(recSamplesFromReport, countImagesUnder(recDatasetPath, minRecSamples));

    appendGateCheck(&checks, &blocked, checkObjectWithDetails(
        QStringLiteral("rec_accuracy_gate"),
        hasRecAccuracy && recAccuracy >= minRecAccuracy ? QStringLiteral("passed") : QStringLiteral("blocked"),
        hasRecAccuracy && recAccuracy >= minRecAccuracy,
        hasRecAccuracy
            ? QStringLiteral("Rec accuracy %1 / required >= %2.").arg(recAccuracy, 0, 'f', 6).arg(minRecAccuracy, 0, 'f', 6)
            : QStringLiteral("Rec accuracy was not found in the Rec report."),
        QJsonObject{{QStringLiteral("value"), recAccuracy}, {QStringLiteral("threshold"), minRecAccuracy}}));
    appendGateCheck(&checks, &blocked, checkObjectWithDetails(
        QStringLiteral("rec_cer_gate"),
        hasRecCer && recCer <= maxRecCer ? QStringLiteral("passed") : QStringLiteral("blocked"),
        hasRecCer && recCer <= maxRecCer,
        hasRecCer
            ? QStringLiteral("Rec CER %1 / required <= %2.").arg(recCer, 0, 'f', 6).arg(maxRecCer, 0, 'f', 6)
            : QStringLiteral("Rec CER was not found in the Rec report."),
        QJsonObject{{QStringLiteral("value"), recCer}, {QStringLiteral("threshold"), maxRecCer}}));
    appendGateCheck(&checks, &blocked, checkObjectWithDetails(
        QStringLiteral("rec_sample_count_gate"),
        recSamples >= minRecSamples ? QStringLiteral("passed") : QStringLiteral("blocked"),
        recSamples >= minRecSamples,
        QStringLiteral("Rec sample count %1 / required >= %2.").arg(recSamples).arg(minRecSamples),
        QJsonObject{{QStringLiteral("value"), recSamples}, {QStringLiteral("threshold"), minRecSamples}}));

    QJsonObject report;
    report.insert(QStringLiteral("schemaVersion"), 1);
    report.insert(QStringLiteral("kind"), QStringLiteral("customer_ocr_acceptance"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("status"), blocked ? QStringLiteral("blocked") : QStringLiteral("passed"));
    report.insert(QStringLiteral("ok"), !blocked);
    report.insert(QStringLiteral("checks"), checks);
    report.insert(QStringLiteral("dataLineage"), QJsonObject{
        {QStringLiteral("detDatasetPath"), detDatasetPath},
        {QStringLiteral("recDatasetPath"), recDatasetPath},
        {QStringLiteral("systemImagesPath"), systemImagesPath},
        {QStringLiteral("customerDataset"), customerDatasetPath},
        {QStringLiteral("dataPaths"), jsonArrayFromStringList(dataPaths)}
    });
    report.insert(QStringLiteral("reports"), QJsonObject{
        {QStringLiteral("detReport"), pathEvidenceObject(detReportPath)},
        {QStringLiteral("recReport"), pathEvidenceObject(recReportPath)},
        {QStringLiteral("systemReport"), pathEvidenceObject(systemReportPath)}
    });
    report.insert(QStringLiteral("metrics"), QJsonObject{
        {QStringLiteral("recAccuracy"), recAccuracy},
        {QStringLiteral("recCer"), recCer},
        {QStringLiteral("recSamples"), recSamples}
    });
    report.insert(QStringLiteral("thresholds"), QJsonObject{
        {QStringLiteral("minRecAccuracy"), minRecAccuracy},
        {QStringLiteral("maxRecCer"), maxRecCer},
        {QStringLiteral("minDetSamples"), minDetSamples},
        {QStringLiteral("minRecSamples"), minRecSamples},
        {QStringLiteral("minSystemImages"), minSystemImages},
        {QStringLiteral("requireFullDomainEvidence"), requireFullDomainEvidence},
        {QStringLiteral("allowPublicLikeData"), allowPublicLikeData}
    });
    report.insert(QStringLiteral("ocrImplementationPolicy"),
        QStringLiteral("official-only: customer OCR acceptance uses PaddleOCR official Det, Rec, and System reports; AITrain C++ OCR ONNX postprocess is not acceptance evidence."));
    report.insert(QStringLiteral("sensitiveDataNote"),
        QStringLiteral("Customer-domain OCR evidence may contain sensitive images and text. Review reports before external sharing."));
    report.insert(QStringLiteral("publicDataBoundary"),
        QStringLiteral("Public Total-Text/generated smoke data can validate workflow execution only; it cannot certify customer-domain production OCR accuracy."));

    QString error;
    const QString manifestPath = outputDir.filePath(QStringLiteral("customer_ocr_validation_manifest.json"));
    const QString summaryPath = outputDir.filePath(QStringLiteral("customer_ocr_validation_summary.md"));
    report.insert(QStringLiteral("reportPath"), manifestPath);
    report.insert(QStringLiteral("summaryPath"), summaryPath);
    if (!writeTextFile(summaryPath, customerOcrSummaryMarkdown(report), &error)) {
        return failedResult(error);
    }
    if (!writeJsonFile(manifestPath, report, &error)) {
        return failedResult(error);
    }
    return resultFromReport(manifestPath, report);
}

WorkflowResult collectDiagnosticsReport(const QString& outputPath, const QJsonObject& context)
{
    const QDir outputDir(outputPath);
    if (!QDir().mkpath(outputDir.absolutePath())) {
        return failedResult(QStringLiteral("Cannot create diagnostics output directory: %1").arg(outputPath));
    }

    QJsonArray runtimeChecks;
    const QVector<RuntimeDependencyCheck> dependencyChecks =
        defaultRuntimeDependencyChecks(QCoreApplication::applicationDirPath());
    for (const RuntimeDependencyCheck& check : dependencyChecks) {
        runtimeChecks.append(check.toJson());
    }

    QJsonArray artifactIndex = context.value(QStringLiteral("artifactIndex")).toArray();
    const QStringList artifactPaths = stringListFromPaths(context, QStringList()
        << QStringLiteral("artifactPaths")
        << QStringLiteral("recentArtifactPaths"));
    for (const QString& artifactPath : artifactPaths) {
        artifactIndex.append(pathEvidenceObject(artifactPath));
    }

    QJsonObject report;
    report.insert(QStringLiteral("schemaVersion"), 1);
    report.insert(QStringLiteral("kind"), QStringLiteral("diagnostic_bundle"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("projectPath"), context.value(QStringLiteral("projectPath")).toString());
    report.insert(QStringLiteral("projectName"), context.value(QStringLiteral("projectName")).toString());
    report.insert(QStringLiteral("workerExecutable"), context.value(QStringLiteral("workerExecutable")).toString());
    report.insert(QStringLiteral("applicationDir"), QCoreApplication::applicationDirPath());
    report.insert(QStringLiteral("licenseSummary"), context.value(QStringLiteral("licenseSummary")).toObject());
    report.insert(QStringLiteral("pluginSummary"), context.value(QStringLiteral("pluginSummary")).toObject());
    report.insert(QStringLiteral("recentTasks"), context.value(QStringLiteral("recentTasks")).toArray());
    report.insert(QStringLiteral("recentFailures"), context.value(QStringLiteral("recentFailures")).toArray());
    report.insert(QStringLiteral("failedRequests"), context.value(QStringLiteral("failedRequests")).toArray());
    report.insert(QStringLiteral("artifactIndex"), artifactIndex);
    report.insert(QStringLiteral("runtimeDependencyChecks"), runtimeChecks);
    report.insert(QStringLiteral("detectionTrainingBackend"), detectionTrainingBackendStatus());
    report.insert(QStringLiteral("tensorRtBackend"), tensorRtBackendStatus().toJson());
    report.insert(QStringLiteral("onnxRuntimeInferenceAvailable"), isOnnxRuntimeInferenceAvailable());
    report.insert(QStringLiteral("nvidiaSmi"), processSnapshot(QStringLiteral("nvidia-smi"), {}, 5000));
    report.insert(QStringLiteral("pythonVersion"), processSnapshot(QStringLiteral("python"), QStringList() << QStringLiteral("--version"), 5000));
    report.insert(QStringLiteral("globalEnvironmentPolicy"),
        QStringLiteral("Diagnostics are read-only. AITrain Studio does not modify global Python, CUDA, or driver configuration."));

    QString error;
    const QString bundlePath = outputDir.filePath(QStringLiteral("diagnostic_bundle.json"));
    const QString summaryPath = outputDir.filePath(QStringLiteral("diagnostic_summary.md"));
    const QString manifestPath = outputDir.filePath(QStringLiteral("diagnostic_manifest.json"));
    report.insert(QStringLiteral("bundlePath"), bundlePath);
    report.insert(QStringLiteral("summaryPath"), summaryPath);
    report.insert(QStringLiteral("manifestPath"), manifestPath);
    if (!writeJsonFile(bundlePath, report, &error)) {
        return failedResult(error);
    }
    if (!writeTextFile(summaryPath, diagnosticsSummaryMarkdown(report), &error)) {
        return failedResult(error);
    }
    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 1);
    manifest.insert(QStringLiteral("kind"), QStringLiteral("diagnostic_manifest"));
    manifest.insert(QStringLiteral("createdAt"), nowIso());
    manifest.insert(QStringLiteral("status"), QStringLiteral("collected"));
    manifest.insert(QStringLiteral("bundlePath"), bundlePath);
    manifest.insert(QStringLiteral("summaryPath"), summaryPath);
    manifest.insert(QStringLiteral("tensorRtStatus"), report.value(QStringLiteral("tensorRtBackend")).toObject().value(QStringLiteral("status")).toString());
    manifest.insert(QStringLiteral("onnxRuntimeInferenceAvailable"), report.value(QStringLiteral("onnxRuntimeInferenceAvailable")).toBool());
    manifest.insert(QStringLiteral("runtimeDependencyCheckCount"), runtimeChecks.size());
    manifest.insert(QStringLiteral("artifactCount"), artifactIndex.size());
    if (!writeJsonFile(manifestPath, manifest, &error)) {
        return failedResult(error);
    }
    report.insert(QStringLiteral("reportPath"), manifestPath);
    report.insert(QStringLiteral("manifest"), manifest);
    return resultFromReport(manifestPath, report);
}

WorkflowResult validateDeploymentArtifactReport(
    const QString& modelPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options)
{
    const QDir outputDir(outputPath);
    if (!QDir().mkpath(outputDir.absolutePath())) {
        return failedResult(QStringLiteral("Cannot create deployment validation output directory: %1").arg(outputPath));
    }

    const QString normalizedModelPath = QDir::fromNativeSeparators(modelPath.trimmed());
    const QString sampleImagePath = QDir::fromNativeSeparators(options.value(QStringLiteral("sampleImagePath")).toString().trimmed());
    const QString detectedFormat = detectDeploymentFormat(normalizedModelPath, format);
    QJsonArray checks;
    const bool modelExists = pathExists(normalizedModelPath);
    checks.append(checkObjectWithDetails(
        QStringLiteral("artifact_exists"),
        modelExists ? QStringLiteral("passed") : QStringLiteral("failed"),
        modelExists,
        modelExists ? QStringLiteral("Deployment artifact exists.") : QStringLiteral("Deployment artifact is missing."),
        pathEvidenceObject(normalizedModelPath)));

    QJsonObject report;
    report.insert(QStringLiteral("schemaVersion"), 1);
    report.insert(QStringLiteral("kind"), QStringLiteral("deployment_validation_report"));
    report.insert(QStringLiteral("createdAt"), nowIso());
    report.insert(QStringLiteral("modelPath"), normalizedModelPath);
    report.insert(QStringLiteral("sampleImagePath"), sampleImagePath);
    report.insert(QStringLiteral("format"), detectedFormat);
    report.insert(QStringLiteral("runtime"), QStringLiteral("artifact"));
    report.insert(QStringLiteral("checks"), checks);

    if (!modelExists) {
        setDeploymentStatusFromChecks(&report, checks, QStringLiteral("failed"));
    } else if (detectedFormat == QStringLiteral("anomalib_python")) {
        report.insert(QStringLiteral("runtime"), QStringLiteral("anomalib_python"));
        report.insert(QStringLiteral("modelFamily"), QStringLiteral("anomaly_detection"));
        report.insert(QStringLiteral("taskType"), QStringLiteral("anomaly_detection"));
        report.insert(QStringLiteral("runtimeValidation"), QStringLiteral("worker-managed-python"));
        report.insert(QStringLiteral("runtimeBoundary"), QStringLiteral("Anomaly detection v1 validates Worker-managed Python/Anomalib artifacts; this is not AITrain C++ packaged ONNX/TensorRT/NCNN runtime."));

        QJsonObject sidecar;
        QString error;
        const bool sidecarReadable = QFileInfo(normalizedModelPath).fileName().compare(QStringLiteral("anomaly_sidecar.json"), Qt::CaseInsensitive) == 0
            && readJsonFile(normalizedModelPath, &sidecar, &error);
        checks.append(checkObjectWithDetails(
            QStringLiteral("anomaly_sidecar_readable"),
            sidecarReadable ? QStringLiteral("passed") : QStringLiteral("blocked"),
            sidecarReadable,
            sidecarReadable
                ? QStringLiteral("Anomaly sidecar is readable.")
                : QStringLiteral("anomalib_python validation expects anomaly_sidecar.json as the deployment artifact."),
            sidecarReadable ? QJsonObject{{QStringLiteral("sidecarPath"), normalizedModelPath}} : pathEvidenceObject(normalizedModelPath)));

        QString checkpointPath = sidecar.value(QStringLiteral("checkpointPath")).toString();
        if (checkpointPath.isEmpty()) {
            checkpointPath = normalizedModelPath;
        }
        if (!checkpointPath.isEmpty() && QFileInfo(checkpointPath).isRelative()) {
            checkpointPath = QFileInfo(normalizedModelPath).absoluteDir().filePath(checkpointPath);
        }
        const bool checkpointExists = !checkpointPath.isEmpty() && QFileInfo::exists(checkpointPath);
        checks.append(checkObjectWithDetails(
            QStringLiteral("anomaly_checkpoint_exists"),
            checkpointExists ? QStringLiteral("passed") : QStringLiteral("blocked"),
            checkpointExists,
            checkpointExists
                ? QStringLiteral("Anomalib checkpoint referenced by sidecar exists.")
                : QStringLiteral("Anomalib sidecar must reference an existing checkpointPath."),
            pathEvidenceObject(checkpointPath)));
        report.insert(QStringLiteral("checkpointPath"), checkpointPath);

        const QString python = anomalibPythonExecutable(options);
        const bool pythonOk = !python.isEmpty();
        checks.append(checkObjectWithDetails(
            QStringLiteral("python_available"),
            pythonOk ? QStringLiteral("passed") : QStringLiteral("blocked"),
            pythonOk,
            pythonOk
                ? QStringLiteral("Python executable is available for Anomalib validation.")
                : QStringLiteral("Python executable is required for anomalib_python deployment validation."),
            QJsonObject{{QStringLiteral("pythonExecutable"), python}}));

        const bool anomalibOk = pythonModuleAvailable(python, QStringLiteral("anomalib"));
        checks.append(checkObjectWithDetails(
            QStringLiteral("anomalib_available"),
            anomalibOk ? QStringLiteral("passed") : QStringLiteral("blocked"),
            anomalibOk,
            anomalibOk
                ? QStringLiteral("Anomalib module is available.")
                : QStringLiteral("Anomalib is missing; install python_trainers/requirements-anomaly.txt in the selected Python."),
            QJsonObject{{QStringLiteral("pythonExecutable"), python}}));

        const QString adapterScript = anomalibAdapterScriptPath(options);
        const bool adapterExists = QFileInfo::exists(adapterScript);
        checks.append(checkObjectWithDetails(
            QStringLiteral("anomalib_adapter_exists"),
            adapterExists ? QStringLiteral("passed") : QStringLiteral("blocked"),
            adapterExists,
            adapterExists
                ? QStringLiteral("Anomalib adapter script is packaged.")
                : QStringLiteral("Anomalib adapter script is missing."),
            pathEvidenceObject(adapterScript)));

        const bool sampleImageExists = !sampleImagePath.isEmpty() && QFileInfo::exists(sampleImagePath);
        checks.append(checkObjectWithDetails(
            QStringLiteral("sample_image_present"),
            sampleImageExists ? QStringLiteral("passed") : QStringLiteral("blocked"),
            sampleImageExists,
            sampleImageExists
                ? QStringLiteral("Sample image is available for Anomalib inference validation.")
                : QStringLiteral("anomalib_python deployment validation requires a sample image."),
            pathEvidenceObject(sampleImagePath)));

        if (!sidecarReadable || !checkpointExists || !pythonOk || !anomalibOk || !adapterExists || !sampleImageExists) {
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks, QStringLiteral("blocked"));
        } else {
            QJsonObject request;
            request.insert(QStringLiteral("protocolVersion"), 1);
            request.insert(QStringLiteral("mode"), QStringLiteral("infer"));
            request.insert(QStringLiteral("modelPath"), normalizedModelPath);
            request.insert(QStringLiteral("imagePath"), sampleImagePath);
            request.insert(QStringLiteral("outputPath"), outputDir.filePath(QStringLiteral("anomalib-inference")));
            request.insert(QStringLiteral("backend"), sidecar.value(QStringLiteral("trainingBackend")).toString(QStringLiteral("anomalib_patchcore")));
            request.insert(QStringLiteral("options"), options);
            const QString requestPath = outputDir.filePath(QStringLiteral("anomalib_deployment_infer_request.json"));
            if (!writeJsonFile(requestPath, request, &error)) {
                checks.append(checkObjectWithDetails(
                    QStringLiteral("anomalib_runtime_inference"),
                    QStringLiteral("failed"),
                    false,
                    error));
            } else {
                QProcess process;
                QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
                environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
                environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
                configurePackagedPythonEnvironment(&environment);
                process.setProcessEnvironment(environment);
                process.setProgram(python);
                process.setArguments(QStringList() << QStringLiteral("-u") << adapterScript << QStringLiteral("--request") << requestPath << QStringLiteral("--mode") << QStringLiteral("infer"));
                process.setProcessChannelMode(QProcess::MergedChannels);
                QElapsedTimer timer;
                timer.start();
                process.start();
                QByteArray output;
                bool inferOk = process.waitForStarted(5000);
                if (inferOk) {
                    while (!process.waitForFinished(100)) {
                        output.append(process.readAll());
                    }
                    output.append(process.readAll());
                    inferOk = process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0;
                }
                const int elapsedMs = static_cast<int>(timer.elapsed());
                const QString logPath = outputDir.filePath(QStringLiteral("anomalib_deployment_infer.log"));
                writeTextFile(logPath, QString::fromUtf8(output), &error);
                const QString predictionsPath = QDir(request.value(QStringLiteral("outputPath")).toString()).filePath(QStringLiteral("inference_predictions.json"));
                QJsonObject predictionsReport;
                const bool predictionsReadable = readJsonFile(predictionsPath, &predictionsReport, &error);
                if (inferOk && predictionsReadable && predictionsReport.value(QStringLiteral("ok")).toBool(false)) {
                    const QJsonObject firstPrediction = predictionsReport.value(QStringLiteral("predictions")).toArray().isEmpty()
                        ? QJsonObject{}
                        : predictionsReport.value(QStringLiteral("predictions")).toArray().first().toObject();
                    report.insert(QStringLiteral("predictionsPath"), predictionsPath);
                    report.insert(QStringLiteral("overlayPath"), firstPrediction.value(QStringLiteral("overlayPath")).toString());
                    report.insert(QStringLiteral("heatmapPath"), firstPrediction.value(QStringLiteral("heatmapPath")).toString());
                    report.insert(QStringLiteral("maskPath"), firstPrediction.value(QStringLiteral("maskPath")).toString());
                    report.insert(QStringLiteral("predictionCount"), predictionsReport.value(QStringLiteral("predictions")).toArray().size());
                    report.insert(QStringLiteral("elapsedMs"), elapsedMs);
                }
                checks.append(checkObjectWithDetails(
                    QStringLiteral("anomalib_runtime_inference"),
                    inferOk && predictionsReadable && predictionsReport.value(QStringLiteral("ok")).toBool(false)
                        ? QStringLiteral("passed")
                        : QStringLiteral("failed"),
                    inferOk && predictionsReadable && predictionsReport.value(QStringLiteral("ok")).toBool(false),
                    inferOk && predictionsReadable && predictionsReport.value(QStringLiteral("ok")).toBool(false)
                        ? QStringLiteral("Anomalib Python artifact ran sample inference successfully.")
                        : QStringLiteral("Anomalib Python sample inference failed; inspect anomalib_deployment_infer.log and inference_predictions.json."),
                    QJsonObject{
                        {QStringLiteral("logPath"), logPath},
                        {QStringLiteral("predictionsPath"), predictionsPath},
                        {QStringLiteral("elapsedMs"), elapsedMs}}));
            }
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks);
        }
    } else if (detectedFormat == QStringLiteral("ncnn")) {
        const QString binPath = ncnnBinPathForParam(normalizedModelPath);
        const bool binExists = pathExists(binPath);
        checks.append(checkObjectWithDetails(
            QStringLiteral("ncnn_sidecar_bin_exists"),
            binExists ? QStringLiteral("passed") : QStringLiteral("failed"),
            binExists,
            binExists
                ? QStringLiteral("NCNN .bin sidecar exists.")
                : QStringLiteral("NCNN .bin sidecar is missing."),
            binExists
                ? pathEvidenceObject(binPath)
                : withNcnnFailureDetails(
                    QStringLiteral("ncnn_sidecar_bin_exists"),
                    QStringLiteral("NCNN .bin sidecar is missing."),
                    pathEvidenceObject(binPath))));
        report.insert(QStringLiteral("runtime"), QStringLiteral("ncnn"));
        report.insert(QStringLiteral("ncnnBinPath"), binPath);
        report.insert(QStringLiteral("runtimeValidation"), QStringLiteral("runtime-inference"));
        if (!binExists) {
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks, QStringLiteral("failed"));
        } else {
            const NcnnBackendStatus backendStatus = ncnnBackendStatus();
            checks.append(checkObjectWithDetails(
                QStringLiteral("ncnn_runtime_available"),
                backendStatus.inferenceAvailable ? QStringLiteral("passed") : QStringLiteral("failed"),
                backendStatus.inferenceAvailable,
                backendStatus.inferenceAvailable
                    ? QStringLiteral("NCNN runtime is available for deployment validation.")
                    : (backendStatus.message.isEmpty()
                        ? QStringLiteral("NCNN runtime is not available in this build or runtime layout.")
                        : backendStatus.message),
                backendStatus.inferenceAvailable
                    ? backendStatus.toJson()
                    : withNcnnFailureDetails(
                        QStringLiteral("ncnn_runtime_available"),
                        backendStatus.message,
                        backendStatus.toJson())));
            if (!backendStatus.inferenceAvailable) {
                report.insert(QStringLiteral("checks"), checks);
                setDeploymentStatusFromChecks(&report, checks, QStringLiteral("failed"));
            } else {
                const bool sampleImageExists = !sampleImagePath.isEmpty() && QFileInfo::exists(sampleImagePath);
                checks.append(checkObjectWithDetails(
                    QStringLiteral("ncnn_sample_image_present"),
                    sampleImageExists ? QStringLiteral("passed") : QStringLiteral("blocked"),
                    sampleImageExists,
                    sampleImageExists
                        ? QStringLiteral("NCNN sample image is available for runtime inference.")
                        : QStringLiteral("NCNN deployment validation requires a sample image to prove runtime inference."),
                    sampleImageExists
                        ? pathEvidenceObject(sampleImagePath)
                        : withNcnnFailureDetails(
                            QStringLiteral("ncnn_sample_image_present"),
                            QStringLiteral("NCNN deployment validation requires a sample image to prove runtime inference."),
                            pathEvidenceObject(sampleImagePath))));
                if (!sampleImageExists) {
                    report.insert(QStringLiteral("checks"), checks);
                    setDeploymentStatusFromChecks(&report, checks, QStringLiteral("blocked"));
                } else {
                    QString error;
                    QImage overlay;
                    QJsonArray predictionArray;
                    QString taskType = QStringLiteral("detection");
                    QString family = inferNcnnModelFamily(normalizedModelPath);
                    if (family.isEmpty()) {
                        family = options.value(QStringLiteral("modelFamily")).toString();
                    }
                    DetectionInferenceOptions inferenceOptions;
                    const QJsonObject ncnnRuntimeOptions = options;
                    QElapsedTimer timer;
                    timer.start();
                    if (family == QStringLiteral("yolo_segmentation")) {
                        taskType = QStringLiteral("segmentation");
                        const QVector<SegmentationPrediction> predictions =
                            predictSegmentationNcnnRuntime(normalizedModelPath, sampleImagePath, inferenceOptions, ncnnRuntimeOptions, &error);
                        for (const SegmentationPrediction& prediction : predictions) {
                            predictionArray.append(segmentationPredictionToJson(prediction));
                        }
                        if (error.isEmpty()) {
                            overlay = renderSegmentationPredictions(sampleImagePath, predictions, &error);
                        }
                    } else if (family == QStringLiteral("yolo_detection")) {
                        const QVector<DetectionPrediction> predictions =
                            predictDetectionNcnnRuntime(normalizedModelPath, sampleImagePath, inferenceOptions, ncnnRuntimeOptions, &error);
                        for (const DetectionPrediction& prediction : predictions) {
                            predictionArray.append(detectionPredictionToJson(prediction));
                        }
                        if (error.isEmpty()) {
                            overlay = renderDetectionPredictions(sampleImagePath, predictions, &error);
                        }
                    } else {
                        error = QStringLiteral("NCNN deployment validation supports YOLO detection/segmentation sidecars only. Inferred model family: %1")
                            .arg(family.isEmpty() ? QStringLiteral("<empty>") : family);
                    }
                    const int elapsedMs = static_cast<int>(timer.elapsed());
                    const QString predictionsPath = outputDir.filePath(QStringLiteral("deployment_predictions.json"));
                    const QString overlayPath = outputDir.filePath(QStringLiteral("deployment_overlay.png"));
                    if (error.isEmpty() && !writeJsonFile(predictionsPath, predictionsDocument(
                            normalizedModelPath,
                            sampleImagePath,
                            taskType,
                            QStringLiteral("ncnn"),
                            elapsedMs,
                            predictionArray), &error)) {
                        // writeJsonFile sets error.
                    }
                    if (error.isEmpty() && (overlay.isNull() || !overlay.save(overlayPath))) {
                        error = QStringLiteral("Cannot write deployment overlay: %1").arg(overlayPath);
                    }
                    checks.append(checkObjectWithDetails(
                        QStringLiteral("ncnn_runtime_inference"),
                        error.isEmpty() ? QStringLiteral("passed") : QStringLiteral("failed"),
                        error.isEmpty(),
                        error.isEmpty()
                            ? QStringLiteral("NCNN artifact ran inference successfully.")
                            : error,
                        error.isEmpty()
                            ? QJsonObject{{QStringLiteral("modelFamily"), family}, {QStringLiteral("taskType"), taskType}}
                            : withNcnnFailureDetails(
                                QStringLiteral("ncnn_runtime_inference"),
                                error,
                                QJsonObject{
                                    {QStringLiteral("modelFamily"), family},
                                    {QStringLiteral("taskType"), taskType}})));
                    report.insert(QStringLiteral("modelFamily"), family);
                    report.insert(QStringLiteral("taskType"), taskType);
                    report.insert(QStringLiteral("elapsedMs"), elapsedMs);
                    if (error.isEmpty()) {
                        report.insert(QStringLiteral("predictionsPath"), predictionsPath);
                        report.insert(QStringLiteral("overlayPath"), overlayPath);
                        report.insert(QStringLiteral("predictionCount"), predictionArray.size());
                    }
                    report.insert(QStringLiteral("checks"), checks);
                    setDeploymentStatusFromChecks(&report, checks);
                }
            }
        }
    } else if (detectedFormat == QStringLiteral("onnx")) {
        QString modelFamilyWarning;
        const QString onnxFamily = inferOnnxModelFamily(normalizedModelPath, &modelFamilyWarning);
        if (onnxFamily == QStringLiteral("ocr_recognition")
            || onnxFamily == QStringLiteral("ocr_detection")) {
            QJsonObject details{
                {QStringLiteral("modelFamily"), onnxFamily},
                {QStringLiteral("taskType"), onnxFamily},
                {QStringLiteral("officialRoute"), QStringLiteral("paddleocr_system_official")}
            };
            if (!modelFamilyWarning.isEmpty()) {
                details.insert(QStringLiteral("modelFamilyWarning"), modelFamilyWarning);
            }
            checks.append(checkObjectWithDetails(
                QStringLiteral("ocr_official_only_route"),
                QStringLiteral("blocked"),
                false,
                QStringLiteral("OCR deployment validation is official-only. Use PaddleOCR official Det/Rec/System reports and predict_system.py artifacts instead of AITrain C++ ONNX OCR postprocess."),
                details));
            report.insert(QStringLiteral("runtime"), QStringLiteral("paddleocr_official"));
            report.insert(QStringLiteral("modelFamily"), onnxFamily);
            report.insert(QStringLiteral("taskType"), onnxFamily);
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks, QStringLiteral("blocked"));
        } else if (sampleImagePath.isEmpty() || !QFileInfo::exists(sampleImagePath)) {
            checks.append(checkObjectWithDetails(
                QStringLiteral("onnx_sample_image_present"),
                QStringLiteral("blocked"),
                false,
                QStringLiteral("ONNX deployment validation requires a sample image to prove runtime inference."),
                pathEvidenceObject(sampleImagePath)));
            report.insert(QStringLiteral("runtime"), QStringLiteral("onnxruntime"));
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks, QStringLiteral("blocked"));
        } else if (!isOnnxRuntimeInferenceAvailable()) {
            checks.append(checkObjectWithDetails(
                QStringLiteral("onnxruntime_available"),
                QStringLiteral("failed"),
                false,
                QStringLiteral("ONNX Runtime inference is not available in this build or runtime layout.")));
            report.insert(QStringLiteral("runtime"), QStringLiteral("onnxruntime"));
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks, QStringLiteral("failed"));
        } else {
            QString error;
            QImage overlay;
            QJsonArray predictionArray;
            QString taskType = QStringLiteral("detection");
            const QString family = onnxFamily;
            QElapsedTimer timer;
            timer.start();
            if (family == QStringLiteral("semantic_segmentation")) {
                taskType = QStringLiteral("semantic_segmentation");
                const SemanticSegmentationPrediction prediction =
                    predictSemanticSegmentationOnnxRuntime(normalizedModelPath, sampleImagePath, &error);
                if (error.isEmpty()) {
                    predictionArray.append(semanticSegmentationPredictionToJson(prediction));
                    overlay = renderSemanticSegmentationPrediction(sampleImagePath, prediction, &error);
                }
            } else if (family == QStringLiteral("yolo_obb")) {
                taskType = QStringLiteral("obb_detection");
                DetectionInferenceOptions inferenceOptions;
                const QVector<ObbPrediction> predictions =
                    predictObbOnnxRuntime(normalizedModelPath, sampleImagePath, inferenceOptions, &error);
                for (const ObbPrediction& prediction : predictions) {
                    predictionArray.append(obbPredictionToJson(prediction));
                }
                if (error.isEmpty()) {
                    overlay = renderObbPredictions(sampleImagePath, predictions, &error);
                }
            } else if (family == QStringLiteral("yolo_segmentation")) {
                taskType = QStringLiteral("segmentation");
                DetectionInferenceOptions inferenceOptions;
                const QVector<SegmentationPrediction> predictions =
                    predictSegmentationOnnxRuntime(normalizedModelPath, sampleImagePath, inferenceOptions, &error);
                for (const SegmentationPrediction& prediction : predictions) {
                    predictionArray.append(segmentationPredictionToJson(prediction));
                }
                if (error.isEmpty()) {
                    overlay = renderSegmentationPredictions(sampleImagePath, predictions, &error);
                }
            } else {
                DetectionInferenceOptions inferenceOptions;
                const QVector<DetectionPrediction> predictions =
                    predictDetectionOnnxRuntime(normalizedModelPath, sampleImagePath, inferenceOptions, &error);
                for (const DetectionPrediction& prediction : predictions) {
                    predictionArray.append(detectionPredictionToJson(prediction));
                }
                if (error.isEmpty()) {
                    overlay = renderDetectionPredictions(sampleImagePath, predictions, &error);
                }
            }
            const int elapsedMs = static_cast<int>(timer.elapsed());
            const QString predictionsPath = outputDir.filePath(QStringLiteral("deployment_predictions.json"));
            const QString overlayPath = outputDir.filePath(QStringLiteral("deployment_overlay.png"));
            if (error.isEmpty() && !writeJsonFile(predictionsPath, predictionsDocument(
                    normalizedModelPath,
                    sampleImagePath,
                    taskType,
                    QStringLiteral("onnxruntime"),
                    elapsedMs,
                    predictionArray), &error)) {
                // writeJsonFile sets error.
            }
            if (error.isEmpty() && (overlay.isNull() || !overlay.save(overlayPath))) {
                error = QStringLiteral("Cannot write deployment overlay: %1").arg(overlayPath);
            }
            QJsonObject inferenceDetails{{QStringLiteral("modelFamily"), family}, {QStringLiteral("taskType"), taskType}};
            if (!modelFamilyWarning.isEmpty()) {
                inferenceDetails.insert(QStringLiteral("modelFamilyWarning"), modelFamilyWarning);
            }
            checks.append(checkObjectWithDetails(
                QStringLiteral("onnx_runtime_inference"),
                error.isEmpty() ? QStringLiteral("passed") : QStringLiteral("failed"),
                error.isEmpty(),
                error.isEmpty()
                    ? QStringLiteral("ONNX artifact ran inference successfully.")
                    : error,
                inferenceDetails));
            report.insert(QStringLiteral("runtime"), QStringLiteral("onnxruntime"));
            report.insert(QStringLiteral("modelFamily"), family);
            if (!modelFamilyWarning.isEmpty()) {
                report.insert(QStringLiteral("modelFamilyWarning"), modelFamilyWarning);
            }
            report.insert(QStringLiteral("taskType"), taskType);
            report.insert(QStringLiteral("elapsedMs"), elapsedMs);
            if (error.isEmpty()) {
                report.insert(QStringLiteral("predictionsPath"), predictionsPath);
                report.insert(QStringLiteral("overlayPath"), overlayPath);
                report.insert(QStringLiteral("predictionCount"), predictionArray.size());
            }
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks);
        }
    } else if (detectedFormat == QStringLiteral("tensorrt") || detectedFormat == QStringLiteral("tensorrt_engine")) {
        const TensorRtBackendStatus backendStatus = tensorRtBackendStatus();
        if (!isTensorRtInferenceAvailable()) {
            checks.append(checkObjectWithDetails(
                QStringLiteral("tensorrt_runtime_available"),
                QStringLiteral("hardware-blocked"),
                false,
                backendStatus.message.isEmpty()
                    ? QStringLiteral("TensorRT runtime is hardware-blocked or unavailable on this machine.")
                    : backendStatus.message,
                backendStatus.toJson()));
            report.insert(QStringLiteral("runtime"), QStringLiteral("tensorrt"));
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks, QStringLiteral("hardware-blocked"));
        } else if (sampleImagePath.isEmpty() || !QFileInfo::exists(sampleImagePath)) {
            checks.append(checkObjectWithDetails(
                QStringLiteral("tensorrt_sample_image_present"),
                QStringLiteral("blocked"),
                false,
                QStringLiteral("TensorRT validation requires a sample image."),
                pathEvidenceObject(sampleImagePath)));
            report.insert(QStringLiteral("runtime"), QStringLiteral("tensorrt"));
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks, QStringLiteral("blocked"));
        } else {
            DetectionInferenceOptions inferenceOptions;
            QString error;
            QElapsedTimer timer;
            timer.start();
            const QVector<DetectionPrediction> predictions =
                predictDetectionTensorRt(normalizedModelPath, sampleImagePath, inferenceOptions, &error);
            const int elapsedMs = static_cast<int>(timer.elapsed());
            QJsonArray predictionArray;
            for (const DetectionPrediction& prediction : predictions) {
                predictionArray.append(detectionPredictionToJson(prediction));
            }
            QImage overlay;
            if (error.isEmpty()) {
                overlay = renderDetectionPredictions(sampleImagePath, predictions, &error);
            }
            const QString predictionsPath = outputDir.filePath(QStringLiteral("deployment_predictions.json"));
            const QString overlayPath = outputDir.filePath(QStringLiteral("deployment_overlay.png"));
            if (error.isEmpty() && !writeJsonFile(predictionsPath, predictionsDocument(
                    normalizedModelPath,
                    sampleImagePath,
                    QStringLiteral("detection"),
                    QStringLiteral("tensorrt"),
                    elapsedMs,
                    predictionArray), &error)) {
                // writeJsonFile sets error.
            }
            if (error.isEmpty() && (overlay.isNull() || !overlay.save(overlayPath))) {
                error = QStringLiteral("Cannot write deployment overlay: %1").arg(overlayPath);
            }
            checks.append(checkObjectWithDetails(
                QStringLiteral("tensorrt_runtime_inference"),
                error.isEmpty() ? QStringLiteral("passed") : QStringLiteral("failed"),
                error.isEmpty(),
                error.isEmpty() ? QStringLiteral("TensorRT artifact ran inference successfully.") : error));
            report.insert(QStringLiteral("runtime"), QStringLiteral("tensorrt"));
            report.insert(QStringLiteral("elapsedMs"), elapsedMs);
            if (error.isEmpty()) {
                report.insert(QStringLiteral("predictionsPath"), predictionsPath);
                report.insert(QStringLiteral("overlayPath"), overlayPath);
                report.insert(QStringLiteral("predictionCount"), predictionArray.size());
            }
            report.insert(QStringLiteral("checks"), checks);
            setDeploymentStatusFromChecks(&report, checks);
        }
    } else {
        checks.append(checkObjectWithDetails(
            QStringLiteral("deployment_format_supported"),
            QStringLiteral("failed"),
            false,
            QStringLiteral("Unsupported deployment validation format: %1.").arg(detectedFormat)));
        report.insert(QStringLiteral("checks"), checks);
        setDeploymentStatusFromChecks(&report, checks, QStringLiteral("failed"));
    }

    QString error;
    const QString reportPath = outputDir.filePath(QStringLiteral("deployment_validation_report.json"));
    const QString summaryPath = outputDir.filePath(QStringLiteral("deployment_validation_summary.md"));
    report.insert(QStringLiteral("reportPath"), reportPath);
    report.insert(QStringLiteral("summaryPath"), summaryPath);
    applyFailureSummaryFromChecks(&report, report.value(QStringLiteral("checks")).toArray());
    if (!writeTextFile(summaryPath, deploymentSummaryMarkdown(report), &error)) {
        return failedResult(error);
    }
    if (!writeJsonFile(reportPath, report, &error)) {
        return failedResult(error);
    }
    return resultFromReport(reportPath, report);
}
} // namespace aitrain
