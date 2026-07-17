#include "WorkerSessionSupport.h"

#include "aitrain/core/Deployment.h"
#include "aitrain/core/VisionModelRuntime.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>

namespace worker_support {

namespace {

constexpr int kProbeKillWaitMs = 500;

void killAndReapBounded(QProcess* process)
{
    if (!process || process->state() == QProcess::NotRunning) {
        return;
    }
    process->kill();
    // Never use the unbounded waitForFinished() overload for environment
    // probes. A broken executable or inherited pipe can otherwise pin the
    // Worker thread indefinitely after the timeout has already fired.
    process->waitForFinished(kProbeKillWaitMs);
    if (process->state() != QProcess::NotRunning) {
        process->kill();
        process->waitForFinished(kProbeKillWaitMs);
    }
}

} // namespace

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
        killAndReapBounded(&process);
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

QString packagedPythonEnvRoot()
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(applicationDir).absoluteFilePath(QStringLiteral("python_env")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../python_env")),
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

QString packagedPaddleOcrRepoPath()
{
    const QString pythonRoot = packagedPythonEnvRoot();
    if (pythonRoot.isEmpty()) {
        return {};
    }
    const QString repo = QDir(pythonRoot).filePath(QStringLiteral("PaddleOCR"));
    if (QFileInfo::exists(QDir(repo).filePath(QStringLiteral("tools/train.py")))) {
        return QFileInfo(repo).absoluteFilePath();
    }
    return {};
}

void configurePackagedPythonEnvironment(QProcessEnvironment* environment)
{
    if (!environment) {
        return;
    }
    const QString pythonRoot = packagedPythonEnvRoot();
    if (!pythonRoot.isEmpty()) {
        QStringList pathEntries;
        pathEntries << QDir(pythonRoot).filePath(QStringLiteral("Scripts"));
        pathEntries << pythonRoot;
        const QString existingPath = environment->value(QStringLiteral("PATH"));
        if (!existingPath.isEmpty()) {
            pathEntries << existingPath;
        }
        environment->insert(QStringLiteral("PATH"), pathEntries.join(QDir::listSeparator()));
    }

    if (!environment->contains(QStringLiteral("AITRAIN_PADDLEOCR_REPO"))) {
        const QString repo = packagedPaddleOcrRepoPath();
        if (!repo.isEmpty()) {
            environment->insert(QStringLiteral("AITRAIN_PADDLEOCR_REPO"), repo);
        }
    }
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
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("python_env/Scripts/python.exe")));
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("python_env/python.exe")));
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("../python_env/Scripts/python.exe")));
    candidates.append(QDir(applicationDir).absoluteFilePath(QStringLiteral("../python_env/python.exe")));
    candidates.append(QDir::current().absoluteFilePath(QStringLiteral("python_env/Scripts/python.exe")));
    candidates.append(QDir::current().absoluteFilePath(QStringLiteral("python_env/python.exe")));
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
            killAndReapBounded(&process);
            continue;
        }
        if (process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0) {
            return candidate;
        }
    }
    return {};
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
        killAndReapBounded(&process);
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
    if (!process.waitForStarted(2000)) {
        return profileCheck(checkName, QStringLiteral("warning"), QStringLiteral("Unable to probe module availability quickly."));
    }
    if (!process.waitForFinished(5000)) {
        killAndReapBounded(&process);
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
        repairHints.append(QStringLiteral("Install the AITrain Python AI Environment package so `python_env` is available beside the application."));
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
        QStringLiteral("Ultralytics is missing; official YOLO detection/segmentation/OBB training will be unavailable.")));
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

QJsonObject smpEnvironmentProfile(const QString& pythonExecutable)
{
    QJsonArray checks;
    QJsonArray repairHints;

    if (pythonExecutable.isEmpty()) {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("missing"),
            QStringLiteral("No usable Python executable was found for SMP semantic segmentation.")));
        repairHints.append(QStringLiteral("Set training parameter `pythonExecutable` or AITRAIN_PYTHON_EXECUTABLE to a valid Python path."));
    } else {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("ok"),
            QStringLiteral("Python executable is available."),
            QJsonObject{{QStringLiteral("path"), pythonExecutable}}));
    }

    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("segmentation_models_pytorch"),
        QStringLiteral("segmentation_models_pytorch"),
        QStringLiteral("segmentation-models-pytorch is missing; SMP semantic segmentation training will be unavailable.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("torch"),
        QStringLiteral("torch"),
        QStringLiteral("PyTorch is missing; SMP training requires torch.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("torchvision"),
        QStringLiteral("torchvision"),
        QStringLiteral("torchvision is missing; SMP training uses torchvision transforms/runtime helpers.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("timm"),
        QStringLiteral("timm"),
        QStringLiteral("timm is missing; SMP SegFormer/timm encoders require timm.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("onnx"),
        QStringLiteral("onnx"),
        QStringLiteral("onnx package is missing; SMP ONNX export validation may fail.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("onnxruntime"),
        QStringLiteral("onnxruntime"),
        QStringLiteral("onnxruntime package is missing; SMP evaluator/runtime smoke checks may fail.")));

    repairHints.append(QStringLiteral("Install SMP profile packages: `pip install -r python_trainers/requirements-smp.txt`."));
    repairHints.append(QStringLiteral("SMP supports ONNX Runtime inference/deployment validation only; NCNN/TensorRT export is not part of the SMP capability scope."));

    return makeProfile(QStringLiteral("smp_semantic_segmentation"), QStringLiteral("SMP Semantic Segmentation Profile"), checks, repairHints);
}

QJsonObject anomalibEnvironmentProfile(const QString& pythonExecutable)
{
    QJsonArray checks;
    QJsonArray repairHints;

    if (pythonExecutable.isEmpty()) {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("missing"),
            QStringLiteral("No usable Python executable was found for Anomalib anomaly detection.")));
        repairHints.append(QStringLiteral("Set training parameter `pythonExecutable` or AITRAIN_PYTHON_EXECUTABLE to a valid Python path."));
    } else {
        checks.append(profileCheck(
            QStringLiteral("pythonExecutable"),
            QStringLiteral("ok"),
            QStringLiteral("Python executable is available."),
            QJsonObject{{QStringLiteral("path"), pythonExecutable}}));
    }

    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("anomalib"),
        QStringLiteral("anomalib"),
        QStringLiteral("Anomalib is missing; PatchCore/EfficientAD anomaly detection will be unavailable.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("torch"),
        QStringLiteral("torch"),
        QStringLiteral("PyTorch is missing; Anomalib requires torch.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("torchvision"),
        QStringLiteral("torchvision"),
        QStringLiteral("torchvision is missing; Anomalib dataset transforms require torchvision.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("lightning"),
        QStringLiteral("lightning"),
        QStringLiteral("Lightning is missing; Anomalib >=2 uses Lightning for training orchestration.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("timm"),
        QStringLiteral("timm"),
        QStringLiteral("timm is missing; EfficientAD/PatchCore dependencies may require timm.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("Pillow"),
        QStringLiteral("PIL"),
        QStringLiteral("Pillow is missing; anomaly image loading will be unavailable.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("numpy"),
        QStringLiteral("numpy"),
        QStringLiteral("numpy is missing; anomaly reports and heatmaps require numpy.")));
    checks.append(runModuleProbe(
        pythonExecutable,
        QStringLiteral("opencv"),
        QStringLiteral("c"),
        QStringLiteral("opencv-python is missing; anomaly overlays and masks require c.")));

    repairHints.append(QStringLiteral("Install Anomalib profile packages: `pip install -r python_trainers/requirements-anomaly.txt`."));
    repairHints.append(QStringLiteral("EfficientAD requires imagenetDir from parameters, AITRAIN_ANOMALIB_IMAGENET_DIR, or `.deps/anomalib/imagenette`; AITrain will not auto-download external data."));
    repairHints.append(QStringLiteral("Anomaly v1 deployment boundary is Worker-managed Python/Anomalib artifacts, not AITrain C++ ONNX/TensorRT/NCNN runtime."));

    return makeProfile(QStringLiteral("anomaly_detection"), QStringLiteral("Anomalib Profile"), checks, repairHints);
}

QJsonObject ocrEnvironmentProfile(const QString& pythonExecutable)
{
    QJsonArray checks;
    QJsonArray repairHints;

    const QString isolatedOcrPython = QString::fromLocal8Bit(qgetenv("AITRAIN_OCR_PYTHON_EXECUTABLE")).trimmed();
    const QString packagedPython = firstUsablePythonExecutable();
    const QString normalizedPackagedPython = QFileInfo(packagedPython).absoluteFilePath().replace(QLatin1Char('\\'), QLatin1Char('/'));
    const bool packagedPythonDetected = !packagedPython.isEmpty()
        && normalizedPackagedPython.contains(QStringLiteral("/python_env/"), Qt::CaseInsensitive);
    if (isolatedOcrPython.isEmpty() && !packagedPythonDetected) {
        checks.append(profileCheck(
            QStringLiteral("isolatedOcrPython"),
            QStringLiteral("warning"),
            QStringLiteral("No isolated OCR Python is configured via AITRAIN_OCR_PYTHON_EXECUTABLE or packaged python_env.")));
        repairHints.append(QStringLiteral("Set `AITRAIN_OCR_PYTHON_EXECUTABLE` to isolated OCR Python for PaddleOCR official workflows."));
        repairHints.append(QStringLiteral("Install the AITrain Python AI Environment package into the application directory."));
    } else {
        checks.append(profileCheck(
            QStringLiteral("isolatedOcrPython"),
            QStringLiteral("ok"),
            isolatedOcrPython.isEmpty()
                ? QStringLiteral("Packaged OCR Python environment was found.")
                : QStringLiteral("Isolated OCR Python is configured."),
            QJsonObject{{QStringLiteral("path"), isolatedOcrPython.isEmpty() ? packagedPython : isolatedOcrPython}}));
    }

    const QString activePython = !isolatedOcrPython.isEmpty()
        ? isolatedOcrPython
        : (!packagedPython.isEmpty() ? packagedPython : pythonExecutable);
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
    const QString sourceRoot = !repoRoot.isEmpty() ? repoRoot : packagedPaddleOcrRepoPath();
    if (sourceRoot.isEmpty()) {
        checks.append(profileCheck(
            QStringLiteral("paddleOcrSourceCheckout"),
            QStringLiteral("warning"),
            QStringLiteral("PaddleOCR source checkout path is not configured and no packaged python_env/PaddleOCR checkout was found.")));
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
