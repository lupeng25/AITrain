#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/OcrRecTrainer.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/SegmentationTrainer.h"

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
void WorkerSession::runEnvironmentCheck(const QJsonObject& payload)
{
    Q_UNUSED(payload)

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
        QStringLiteral("Ultralytics is not installed. The official YOLO detection/segmentation trainer backend will be unavailable.")));
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
    checks.append(checkObject(QStringLiteral("Worker"), QStringLiteral("ok"), QStringLiteral("Worker 环境自检命令可用。")));

    QJsonObject profiles;
    profiles.insert(QStringLiteral("yolo"), yoloEnvironmentProfile(pythonExecutable));
    profiles.insert(QStringLiteral("ocr"), ocrEnvironmentProfile(pythonExecutable));
    profiles.insert(QStringLiteral("tensorrt"), tensorRtEnvironmentProfile(checks));

    QJsonObject result;
    result.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    result.insert(QStringLiteral("checks"), checks);
    result.insert(QStringLiteral("profiles"), profiles);

    const QString reportPath = QDir::current().filePath(QStringLiteral("environment_profiles_report.json"));
    QString reportError;
    if (writeJsonFile(reportPath, result, &reportError)) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), QStringLiteral("environment-check"));
        artifact.insert(QStringLiteral("kind"), QStringLiteral("environment_profiles_report"));
        artifact.insert(QStringLiteral("path"), reportPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Environment profile report"));
        send(QStringLiteral("artifact"), artifact);
        result.insert(QStringLiteral("reportPath"), reportPath);
    } else {
        QJsonObject logPayload;
        logPayload.insert(QStringLiteral("message"), QStringLiteral("Failed to write environment profile report: %1").arg(reportError));
        send(QStringLiteral("log"), logPayload);
    }

    send(QStringLiteral("environmentCheck"), result);
    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), QStringLiteral("environment-check"));
    completed.insert(QStringLiteral("message"), QStringLiteral("Environment check completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

