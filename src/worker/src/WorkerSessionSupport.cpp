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

namespace worker_support {

QJsonObject checkObject(const QString& name, const QString& status, const QString& message, const QJsonObject& details)
{
    QJsonObject object;
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("message"), message);
    object.insert(QStringLiteral("details"), details);
    return object;
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write JSON report: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

QString defaultTaskOutputPath(const QString& basePath, const QString& taskId)
{
    const QString id = taskId.isEmpty()
        ? QStringLiteral("manual-%1").arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMddHHmmsszzz")))
        : taskId;
    return QDir(basePath).filePath(QStringLiteral("runs/%1").arg(id));
}

QJsonObject nvidiaSmiCheck()
{
    QProcess process;
    process.start(QStringLiteral("nvidia-smi"),
        QStringList() << QStringLiteral("--query-gpu=name,memory.total")
                      << QStringLiteral("--format=csv,noheader"));
    if (!process.waitForStarted(1500)) {
        return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("missing"), QStringLiteral("未找到 nvidia-smi，可能未安装 NVIDIA 驱动。"));
    }
    if (!process.waitForFinished(2500)) {
        process.kill();
        process.waitForFinished();
        return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("warning"), QStringLiteral("nvidia-smi 执行超时。"));
    }

    const QString output = QString::fromLocal8Bit(process.readAllStandardOutput()).trimmed();
    const QString errorOutput = QString::fromLocal8Bit(process.readAllStandardError()).trimmed();
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0 || output.isEmpty()) {
        return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("missing"),
            errorOutput.isEmpty() ? QStringLiteral("nvidia-smi 未返回 GPU 信息。") : errorOutput);
    }

    QJsonObject details;
    details.insert(QStringLiteral("raw"), output);
    return checkObject(QStringLiteral("NVIDIA Driver"), QStringLiteral("ok"), QStringLiteral("检测到 NVIDIA GPU：%1").arg(output.split(QLatin1Char('\n')).first()), details);
}

QString firstUsablePythonExecutable(const QJsonObject& parameters)
{
    QStringList candidates;
    const QString requested = parameters.value(QStringLiteral("pythonExecutable")).toString().trimmed();
    if (!requested.isEmpty()) {
        candidates.append(requested);
    }
    const QString envRequested = QString::fromLocal8Bit(qgetenv("AITRAIN_PYTHON_EXECUTABLE")).trimmed();
    if (!envRequested.isEmpty()) {
        candidates.append(envRequested);
    }

    const QString applicationDir = QCoreApplication::applicationDirPath();
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/python-3.13.13-embed-amd64/python.exe")));
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/python-3.13.13-embed-amd64/python.exe")));
    candidates.append(QDir::current().absoluteFilePath(QStringLiteral(".deps/python-3.13.13-embed-amd64/python.exe")));
    candidates.append(QStandardPaths::findExecutable(QStringLiteral("python")));
    candidates.append(QStandardPaths::findExecutable(QStringLiteral("python3")));

    QStringList seen;
    for (const QString& candidate : candidates) {
        if (candidate.trimmed().isEmpty() || seen.contains(candidate)) {
            continue;
        }
        seen.append(candidate);
        QProcess process;
        process.start(candidate, QStringList() << QStringLiteral("--version"));
        if (!process.waitForStarted(1500)) {
            continue;
        }
        if (!process.waitForFinished(2500)) {
            process.kill();
            process.waitForFinished();
            continue;
        }
        if (process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
            return candidate;
        }
    }
    return {};
}

QString pythonTrainerScriptFileForBackend(const QString& backend)
{
    const QString normalized = backend.trimmed().toLower();
    if (normalized == QStringLiteral("ultralytics_yolo") || normalized == QStringLiteral("ultralytics_yolo_detect")) {
        return QStringLiteral("python_trainers/detection/ultralytics_trainer.py");
    }
    if (normalized == QStringLiteral("ultralytics_yolo_segment")) {
        return QStringLiteral("python_trainers/segmentation/ultralytics_trainer.py");
    }
    if (normalized == QStringLiteral("paddleocr_rec") && diagnosticTrainingBackendsEnabled()) {
        return QStringLiteral("python_trainers/ocr_rec/paddleocr_trainer.py");
    }
    if (normalized == QStringLiteral("paddleocr_rec_official") || normalized == QStringLiteral("paddleocr_ppocrv4_rec")) {
        return QStringLiteral("python_trainers/ocr_rec/paddleocr_official_adapter.py");
    }
    if (normalized == QStringLiteral("paddleocr_det_official")) {
        return QStringLiteral("python_trainers/ocr_det/paddleocr_det_official_adapter.py");
    }
    if (normalized == QStringLiteral("paddleocr_system_official")) {
        return QStringLiteral("python_trainers/ocr_system/paddleocr_system_official_adapter.py");
    }
    if (diagnosticTrainingBackendsEnabled()
        && (normalized == QStringLiteral("python_mock") || normalized == QStringLiteral("python_trainer_mock"))) {
        return QStringLiteral("python_trainers/mock_trainer.py");
    }
    return {};
}

QString pythonTrainerScriptPath(const QJsonObject& parameters, const QString& backend)
{
    if (diagnosticTrainingBackendsEnabled()) {
        const QString requested = parameters.value(QStringLiteral("pythonTrainerScript")).toString().trimmed();
        if (!requested.isEmpty()) {
            return QFileInfo(requested).absoluteFilePath();
        }
        const QString envRequested = QString::fromLocal8Bit(qgetenv("AITRAIN_PYTHON_TRAINER_SCRIPT")).trimmed();
        if (!envRequested.isEmpty()) {
            return QFileInfo(envRequested).absoluteFilePath();
        }
    }

    const QString trainerFile = pythonTrainerScriptFileForBackend(backend);
    if (trainerFile.isEmpty()) {
        return {};
    }
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(applicationDir).absoluteFilePath(trainerFile),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../%1").arg(trainerFile)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../%1").arg(trainerFile)),
        QDir::current().absoluteFilePath(trainerFile)
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return candidates.first();
}

QString requestedTrainingBackend(const aitrain::TrainingRequest& request)
{
    const QString backend = request.parameters.value(QStringLiteral("trainingBackend")).toString().trimmed();
    return backend.isEmpty() ? officialTrainingBackendForTask(request.taskType) : backend;
}

bool diagnosticTrainingBackendsEnabled()
{
    return QString::fromLocal8Bit(qgetenv("AITRAIN_ENABLE_DIAGNOSTIC_BACKENDS")).trimmed() == QStringLiteral("1");
}

QString officialTrainingBackendForTask(const QString& taskType)
{
    const QString normalized = taskType.trimmed().toLower();
    if (normalized == QStringLiteral("detection")) {
        return QStringLiteral("ultralytics_yolo_detect");
    }
    if (normalized == QStringLiteral("segmentation")) {
        return QStringLiteral("ultralytics_yolo_segment");
    }
    if (normalized == QStringLiteral("ocr_detection")) {
        return QStringLiteral("paddleocr_det_official");
    }
    if (normalized == QStringLiteral("ocr_recognition")) {
        return QStringLiteral("paddleocr_rec_official");
    }
    return {};
}

namespace {
bool isOfficialWorkerBackendId(const QString& normalized)
{
    return normalized == QStringLiteral("ultralytics_yolo")
        || normalized == QStringLiteral("ultralytics_yolo_detect")
        || normalized == QStringLiteral("ultralytics_yolo_segment")
        || normalized == QStringLiteral("paddleocr_det_official")
        || normalized == QStringLiteral("paddleocr_rec_official")
        || normalized == QStringLiteral("paddleocr_ppocrv4_rec")
        || normalized == QStringLiteral("paddleocr_system_official");
}

bool isDiagnosticBackendId(const QString& normalized, const QJsonObject& parameters)
{
    return normalized == QStringLiteral("python_mock")
        || normalized == QStringLiteral("python_trainer_mock")
        || normalized == QStringLiteral("tiny_linear_detector")
        || normalized == QStringLiteral("tiny_detector")
        || normalized == QStringLiteral("tiny_linear")
        || normalized == QStringLiteral("paddleocr_rec")
        || parameters.contains(QStringLiteral("pythonTrainerScript"));
}
} // namespace

bool isSupportedTrainingBackendId(const QString& backend, const QJsonObject& parameters)
{
    const QString normalized = backend.trimmed().toLower();
    const bool hasTrainerScriptOverride =
        parameters.contains(QStringLiteral("pythonTrainerScript"))
        || !QString::fromLocal8Bit(qgetenv("AITRAIN_PYTHON_TRAINER_SCRIPT")).trimmed().isEmpty();
    if (hasTrainerScriptOverride && !diagnosticTrainingBackendsEnabled()) {
        return false;
    }
    if (normalized.isEmpty()) {
        return false;
    }
    if (isOfficialWorkerBackendId(normalized)) {
        return true;
    }
    return diagnosticTrainingBackendsEnabled() && isDiagnosticBackendId(normalized, parameters);
}

bool isPythonTrainingBackendId(const QString& backend, const QJsonObject& parameters)
{
    const QString normalized = backend.trimmed().toLower();
    if (!isSupportedTrainingBackendId(backend, parameters)) {
        return false;
    }
    if (isOfficialWorkerBackendId(normalized)) {
        return true;
    }
    return diagnosticTrainingBackendsEnabled()
        && (parameters.contains(QStringLiteral("pythonTrainerScript"))
            || normalized == QStringLiteral("python_mock")
            || normalized == QStringLiteral("python_trainer_mock")
            || normalized == QStringLiteral("paddleocr_rec"));
}

QJsonObject runPythonCommandCheck(
    const QString& name,
    const QString& executable,
    const QStringList& arguments,
    int timeoutMs,
    const QString& missingMessage)
{
    if (executable.isEmpty()) {
        return checkObject(name, QStringLiteral("missing"), missingMessage);
    }

    QProcess process;
    process.start(executable, arguments);
    if (!process.waitForStarted(1500)) {
        return checkObject(name, QStringLiteral("missing"), QStringLiteral("%1: %2").arg(missingMessage, process.errorString()));
    }
    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        process.waitForFinished();
        return checkObject(name, QStringLiteral("warning"), QStringLiteral("%1 check timed out.").arg(name));
    }

    const QString stdoutText = QString::fromUtf8(process.readAllStandardOutput()).trimmed();
    const QString stderrText = QString::fromUtf8(process.readAllStandardError()).trimmed();
    QJsonObject details;
    details.insert(QStringLiteral("stdout"), stdoutText);
    details.insert(QStringLiteral("stderr"), stderrText);
    details.insert(QStringLiteral("executable"), executable);
    if (process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
        return checkObject(name, QStringLiteral("ok"), stdoutText.isEmpty() ? QStringLiteral("%1 is available.").arg(name) : stdoutText, details);
    }
    return checkObject(name, QStringLiteral("missing"), stderrText.isEmpty() ? missingMessage : stderrText, details);
}

QJsonObject pythonModuleCheck(const QString& executable, const QString& displayName, const QString& moduleName, const QString& missingMessage)
{
    return runPythonCommandCheck(
        displayName,
        executable,
        QStringList()
            << QStringLiteral("-c")
            << QStringLiteral("import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('%1') else 3)").arg(moduleName),
        5000,
        missingMessage);
}

QJsonObject profileCheck(const QString& name, const QString& status, const QString& message, const QJsonObject& details)
{
    QJsonObject object;
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("message"), message);
    object.insert(QStringLiteral("details"), details);
    return object;
}

QJsonObject makeProfile(const QString& id, const QString& title, const QJsonArray& checks, const QJsonArray& repairHints)
{
    bool hasMissing = false;
    bool hasWarning = false;
    bool hasBlocked = false;
    for (const QJsonValue& value : checks) {
        const QString status = value.toObject().value(QStringLiteral("status")).toString();
        hasMissing = hasMissing || status == QStringLiteral("missing");
        hasWarning = hasWarning || status == QStringLiteral("warning");
        hasBlocked = hasBlocked || status == QStringLiteral("hardware-blocked");
    }

    QString status = QStringLiteral("ok");
    if (hasBlocked) {
        status = QStringLiteral("hardware-blocked");
    } else if (hasMissing) {
        status = QStringLiteral("missing");
    } else if (hasWarning) {
        status = QStringLiteral("warning");
    }

    QJsonObject profile;
    profile.insert(QStringLiteral("id"), id);
    profile.insert(QStringLiteral("title"), title);
    profile.insert(QStringLiteral("status"), status);
    profile.insert(QStringLiteral("checks"), checks);
    profile.insert(QStringLiteral("repairHints"), repairHints);
    return profile;
}

QJsonObject runModuleProbe(const QString& pythonExecutable, const QString& checkName, const QString& moduleName, const QString& hint)
{
    if (pythonExecutable.isEmpty()) {
        return profileCheck(checkName, QStringLiteral("missing"), QStringLiteral("Python executable is unavailable."));
    }

    QProcess process;
    process.start(pythonExecutable, QStringList()
        << QStringLiteral("-c")
        << QStringLiteral("import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('%1') else 3)").arg(moduleName));
    if (!process.waitForStarted(2000) || !process.waitForFinished(5000)) {
        return profileCheck(checkName, QStringLiteral("warning"), QStringLiteral("Unable to probe module availability quickly."));
    }
    if (process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
        return profileCheck(checkName, QStringLiteral("ok"), QStringLiteral("Available."));
    }
    return profileCheck(checkName, QStringLiteral("missing"), hint);
}

QJsonObject yoloEnvironmentProfile(const QString& pythonExecutable)
{
    QJsonArray checks;
    QJsonArray repairHints;

    if (pythonExecutable.isEmpty()) {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("missing"),
            QStringLiteral("No usable Python executable was found for YOLO official backends.")));
        repairHints.append(QStringLiteral("Set training parameter `pythonExecutable` or environment variable `AITRAIN_PYTHON_EXECUTABLE` to a valid Python path."));
        repairHints.append(QStringLiteral("Use local embed Python under `.deps/python-3.13.13-embed-amd64/python.exe` when available."));
    } else {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("ok"),
            QStringLiteral("Python executable is available."),
            QJsonObject{{QStringLiteral("path"), pythonExecutable}}));
    }

    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("ultralytics"),
        QStringLiteral("ultralytics"),
        QStringLiteral("Ultralytics is missing; official YOLO detection/segmentation training will be unavailable.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("torch"),
        QStringLiteral("torch"),
        QStringLiteral("PyTorch is missing; official Ultralytics YOLO training requires torch.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("onnx"),
        QStringLiteral("onnx"),
        QStringLiteral("onnx package is missing; ONNX export validation may fail.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("onnxruntime"),
        QStringLiteral("onnxruntime"),
        QStringLiteral("onnxruntime package is missing; Python-side ONNX runtime checks may fail.")));

    repairHints.append(QStringLiteral("Install YOLO profile packages in selected Python: `pip install ultralytics onnx onnxruntime`."));
    repairHints.append(QStringLiteral("Keep official YOLO training in Worker-managed Python subprocesses; avoid GUI-embedded Python."));

    return makeProfile(QStringLiteral("yolo"), QStringLiteral("YOLO Profile"), checks, repairHints);
}

QJsonObject ocrEnvironmentProfile(const QString& pythonExecutable)
{
    QJsonArray checks;
    QJsonArray repairHints;

    const QString isolatedOcrPython = QString::fromLocal8Bit(qgetenv("AITRAIN_OCR_PYTHON_EXECUTABLE")).trimmed();
    if (isolatedOcrPython.isEmpty()) {
        checks.append(profileCheck(
            QStringLiteral("isolatedOcrPython"),
            QStringLiteral("warning"),
            QStringLiteral("No isolated OCR Python is configured via AITRAIN_OCR_PYTHON_EXECUTABLE.")));
        repairHints.append(QStringLiteral("Set `AITRAIN_OCR_PYTHON_EXECUTABLE` to isolated OCR Python for PaddleOCR official workflows."));
    } else {
        checks.append(profileCheck(
            QStringLiteral("isolatedOcrPython"),
            QStringLiteral("ok"),
            QStringLiteral("Isolated OCR Python is configured."),
            QJsonObject{{QStringLiteral("path"), isolatedOcrPython}}));
    }

    const QString activePython = !isolatedOcrPython.isEmpty() ? isolatedOcrPython : pythonExecutable;
    if (activePython.isEmpty()) {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("missing"),
            QStringLiteral("No usable Python executable is available for OCR checks.")));
    } else {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("ok"),
            QStringLiteral("Python executable is available for OCR profile checks."),
            QJsonObject{{QStringLiteral("path"), activePython}}));
    }

    checks.append(runModuleProbe(
        activePython,
        QStringLiteral("paddle"),
        QStringLiteral("paddle"),
        QStringLiteral("PaddlePaddle is missing; official OCR adapters will be unavailable.")));
    checks.append(runModuleProbe(
        activePython,
        QStringLiteral("paddleocr"),
        QStringLiteral("paddleocr"),
        QStringLiteral("PaddleOCR is missing; official OCR adapters will be unavailable.")));

    const QString repoRoot = QString::fromLocal8Bit(qgetenv("AITRAIN_PADDLEOCR_REPO")).trimmed();
    const QString sourceRoot = !repoRoot.isEmpty()
        ? repoRoot
        : QString::fromLocal8Bit(qgetenv("AITRAIN_PADDLEOCR_SOURCE_ROOT")).trimmed();
    if (sourceRoot.isEmpty()) {
        checks.append(profileCheck(
            QStringLiteral("paddleOcrSourceCheckout"),
            QStringLiteral("warning"),
            QStringLiteral("PaddleOCR source checkout path is not configured (AITRAIN_PADDLEOCR_REPO).")));
    } else {
        const bool trainScriptExists = QFileInfo::exists(QDir(sourceRoot).filePath(QStringLiteral("tools/train.py")));
        checks.append(profileCheck(
            QStringLiteral("paddleOcrSourceCheckout"),
            trainScriptExists ? QStringLiteral("ok") : QStringLiteral("warning"),
            trainScriptExists
                ? QStringLiteral("PaddleOCR source checkout is ready.")
                : QStringLiteral("PaddleOCR source checkout is configured but tools/train.py was not found."),
            QJsonObject{{QStringLiteral("path"), sourceRoot}}));
    }

    const bool smokeScriptReady = QFileInfo::exists(QDir::current().filePath(QStringLiteral("tools/phase16-ocr-official-smoke.ps1")));
    checks.append(profileCheck(
        QStringLiteral("officialSmokeScript"),
        smokeScriptReady ? QStringLiteral("ok") : QStringLiteral("missing"),
        smokeScriptReady
            ? QStringLiteral("Official OCR smoke script is available.")
            : QStringLiteral("Official OCR smoke script is missing.")));

    checks.append(profileCheck(
        QStringLiteral("torchPaddleConflictRisk"),
        QStringLiteral("warning"),
        QStringLiteral("Mixed Torch/Paddle environments may have DLL conflicts. Prefer isolated OCR Python for official OCR workflows.")));

    repairHints.append(QStringLiteral("Use isolated OCR Python and run `tools/phase16-ocr-official-smoke.ps1` to verify official OCR chain."));
    repairHints.append(QStringLiteral("If Torch/Paddle DLL conflicts appear, separate YOLO and OCR environments."));

    return makeProfile(QStringLiteral("ocr"), QStringLiteral("OCR Profile"), checks, repairHints);
}

QJsonObject tensorRtEnvironmentProfile(const QJsonArray& baseChecks)
{
    QJsonArray checks;
    QJsonArray repairHints;

    const auto findBaseCheck = [&baseChecks](const QString& name) {
        for (const QJsonValue& value : baseChecks) {
            const QJsonObject check = value.toObject();
            if (check.value(QStringLiteral("name")).toString() == name) {
                return check;
            }
        }
        return QJsonObject{};
    };

    const QJsonObject nvidia = findBaseCheck(QStringLiteral("NVIDIA Driver"));
    const QJsonObject cudaRuntime = findBaseCheck(QStringLiteral("CUDA Runtime"));
    const QJsonObject cudnn = findBaseCheck(QStringLiteral("cuDNN"));
    const QJsonObject tensorRt = findBaseCheck(QStringLiteral("TensorRT"));
    checks.append(profileCheck(
        QStringLiteral("nvidiaDriver"),
        nvidia.value(QStringLiteral("status")).toString(QStringLiteral("warning")),
        nvidia.value(QStringLiteral("message")).toString()));
    checks.append(profileCheck(
        QStringLiteral("cudaRuntime"),
        cudaRuntime.value(QStringLiteral("status")).toString(QStringLiteral("warning")),
        cudaRuntime.value(QStringLiteral("message")).toString()));
    checks.append(profileCheck(
        QStringLiteral("cuDnn"),
        cudnn.value(QStringLiteral("status")).toString(QStringLiteral("warning")),
        cudnn.value(QStringLiteral("message")).toString()));
    checks.append(profileCheck(
        QStringLiteral("tensorRtDll"),
        tensorRt.value(QStringLiteral("status")).toString(QStringLiteral("warning")),
        tensorRt.value(QStringLiteral("message")).toString()));

    const aitrain::TensorRtBackendStatus backend = aitrain::tensorRtBackendStatus();
    QJsonObject backendDetails = backend.toJson();
    QString hardwareStatus = backend.inferenceAvailable ? QStringLiteral("ok") : QStringLiteral("warning");
    QString hardwareMessage = backend.message;
    if (hardwareMessage.contains(QStringLiteral("SM 61"), Qt::CaseInsensitive)
        || hardwareMessage.contains(QStringLiteral("not supported"), Qt::CaseInsensitive)) {
        hardwareStatus = QStringLiteral("hardware-blocked");
        hardwareMessage = QStringLiteral("TensorRT is hardware-blocked on the current GPU. Use RTX / SM 75+ for acceptance.");
    }
    checks.append(profileCheck(
        QStringLiteral("sm75Acceptance"),
        hardwareStatus,
        hardwareMessage,
        backendDetails));

    repairHints.append(QStringLiteral("Keep TensorRT DLLs under runtimes/tensorrt or PATH and rerun environment check."));
    repairHints.append(QStringLiteral("Use RTX / SM 75+ machine for TensorRT acceptance; GTX 1060 / SM 61 should remain hardware-blocked."));

    return makeProfile(QStringLiteral("tensorrt"), QStringLiteral("TensorRT Profile"), checks, repairHints);
}



} // namespace worker_support
