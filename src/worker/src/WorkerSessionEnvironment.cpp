#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/AnnotationIntegration.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QStandardPaths>
#include <QThread>

using namespace worker_support;
namespace wp = aitrain::worker_protocol;

void WorkerSession::runEnvironmentCheck(const QJsonObject& payload)
{
    Q_UNUSED(payload)

    const QString configuredReportDir = QString::fromLocal8Bit(qgetenv("AITRAIN_ENVIRONMENT_REPORT_DIR")).trimmed();
    const QString reportDir = configuredReportDir.isEmpty() ? QDir::tempPath() : configuredReportDir;

    QJsonArray checks;
    checks.append(nvidiaSmiCheck());
    const QVector<aitrain::RuntimeDependencyCheck> runtimeChecks =
        aitrain::defaultRuntimeDependencyChecks(QCoreApplication::applicationDirPath());
    for (const aitrain::RuntimeDependencyCheck& check : runtimeChecks) {
        checks.append(check.toJson());
    }
    const QString pythonExecutable = firstUsablePythonExecutable();
    checks.append(runPythonCommandCheck(
        QStringLiteral("Python"),
        pythonExecutable,
        QStringList() << QStringLiteral("--version"),
        5000,
        QStringLiteral("Python executable is not available. Configure pythonExecutable or install Python before using official Python trainers.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("Ultralytics YOLO"),
        QStringLiteral("ultralytics"),
        QStringLiteral("Ultralytics is not installed. The official YOLO detection/segmentation/OBB trainer backends will be unavailable.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("Segmentation Models PyTorch"),
        QStringLiteral("segmentation_models_pytorch"),
        QStringLiteral("segmentation-models-pytorch is not installed. The SMP semantic segmentation trainer backend will be unavailable.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("Anomalib"),
        QStringLiteral("anomalib"),
        QStringLiteral("Anomalib is not installed. PatchCore and EfficientAD anomaly detection backends will be unavailable.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("PaddleOCR"),
        QStringLiteral("paddleocr"),
        QStringLiteral("PaddleOCR is not installed. Official OCR detection, recognition, and system adapters will be unavailable.")));
    checks.append(pythonModuleCheck(
        pythonExecutable,
        QStringLiteral("PaddlePaddle"),
        QStringLiteral("paddle"),
        QStringLiteral("PaddlePaddle is not installed. Official PaddleOCR Det/Rec/System workflows will be unavailable.")));

    const aitrain::WorkflowResult xAnyEnvironment =
        aitrain::inspectXAnyLabelingEnvironment(reportDir, {}, pollingCancellationCallback(20));
    if (canceled_) {
        sendCanceledAndFinish(QStringLiteral("environment-check"), xAnyEnvironment.error);
        return;
    }
    const QJsonObject xAnyPayload = xAnyEnvironment.payload;
    const QString xAnyStatus = xAnyPayload.value(QStringLiteral("status")).toString(
        xAnyEnvironment.ok ? QStringLiteral("ok") : QStringLiteral("missing"));
    const QString xAnyMessage = xAnyPayload.value(QStringLiteral("message")).toString(xAnyEnvironment.error);
    checks.append(checkObject(
        QStringLiteral("X-AnyLabeling"),
        xAnyStatus,
        xAnyMessage,
        xAnyPayload));
    if (!xAnyEnvironment.reportPath.isEmpty()) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), QStringLiteral("environment-check"));
        artifact.insert(QStringLiteral("kind"), QStringLiteral("xanylabeling_environment_report"));
        artifact.insert(QStringLiteral("path"), xAnyEnvironment.reportPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("X-AnyLabeling environment report"));
        send(wp::event::artifact(), artifact);
    }
    checks.append(checkObject(QStringLiteral("Worker"), QStringLiteral("ok"), QStringLiteral("Worker 环境自检命令可用。")));

    QJsonObject profiles;
    profiles.insert(QStringLiteral("yolo"), yoloEnvironmentProfile(pythonExecutable));
    profiles.insert(QStringLiteral("smp_semantic_segmentation"), smpEnvironmentProfile(pythonExecutable));
    profiles.insert(QStringLiteral("anomaly_detection"), anomalibEnvironmentProfile(pythonExecutable));
    profiles.insert(QStringLiteral("ocr"), ocrEnvironmentProfile(pythonExecutable));
    profiles.insert(QStringLiteral("tensorrt"), tensorRtEnvironmentProfile(checks));
    {
        QJsonArray profileChecks;
        QJsonObject details;
        details.insert(QStringLiteral("executable"), xAnyPayload.value(QStringLiteral("executable")).toString());
        details.insert(QStringLiteral("reportPath"), xAnyEnvironment.reportPath);
        details.insert(QStringLiteral("licenseBoundary"), xAnyPayload.value(QStringLiteral("licenseBoundary")).toString());
        details.insert(QStringLiteral("redistributionReviewRequired"), xAnyPayload.value(QStringLiteral("redistributionReviewRequired")).toBool(true));
        profileChecks.append(profileCheck(
            QStringLiteral("xanylabelingExecutable"),
            xAnyStatus,
            xAnyMessage,
            details));
        QJsonArray repairHints;
        repairHints.append(QStringLiteral("Set `AITRAIN_XANYLABELING_EXE` to the local X-AnyLabeling executable."));
        repairHints.append(QStringLiteral("Or place X-AnyLabeling under `.deps/tools/annotation-tools/X-AnyLabeling`."));
        repairHints.append(QStringLiteral("Keep X-AnyLabeling as a local external dependency unless redistribution has a separate license/package review."));
        profiles.insert(
            QStringLiteral("xanylabeling"),
            makeProfile(QStringLiteral("xanylabeling"), QStringLiteral("X-AnyLabeling Profile"), profileChecks, repairHints));
    }

    QJsonObject result;
    result.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    result.insert(QStringLiteral("checks"), checks);
    result.insert(QStringLiteral("profiles"), profiles);

    const QString reportPath = QDir(reportDir).filePath(QStringLiteral("environment_profiles_report.json"));
    QString reportError;
    if (writeJsonFile(reportPath, result, &reportError)) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), QStringLiteral("environment-check"));
        artifact.insert(QStringLiteral("kind"), QStringLiteral("environment_profiles_report"));
        artifact.insert(QStringLiteral("path"), reportPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Environment profile report"));
        send(wp::event::artifact(), artifact);
        result.insert(QStringLiteral("reportPath"), reportPath);
    } else {
        QJsonObject logPayload;
        logPayload.insert(QStringLiteral("message"), QStringLiteral("Failed to write environment profile report: %1").arg(reportError));
        send(wp::event::log(), logPayload);
    }

    send(wp::event::environmentCheck(), result);
    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), QStringLiteral("environment-check"));
    completed.insert(QStringLiteral("message"), QStringLiteral("Environment check completed"));
    send(wp::event::completed(), completed);
    finishSession();
}
