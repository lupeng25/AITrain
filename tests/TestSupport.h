#include "WorkerClient.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionDataset.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/LicenseManager.h"
#include "aitrain/core/OcrRecDataset.h"
#include "aitrain/core/OcrRecTrainer.h"
#include "aitrain/core/PluginManager.h"
#include "aitrain/core/PluginMarketplace.h"
#include "aitrain/core/ProjectRepository.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/SegmentationDataset.h"
#include "aitrain/core/SegmentationTrainer.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QSet>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>
#include <functional>

namespace {

void writeTextFile(const QString& path, const QString& content)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
    file.write(content.toUtf8());
}

QJsonObject readJsonObject(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QJsonDocument::fromJson(file.readAll()).object();
}

bool jsonArrayContainsCode(const QJsonArray& array, const QString& code)
{
    for (const QJsonValue& value : array) {
        if (value.toObject().value(QStringLiteral("code")).toString() == code) {
            return true;
        }
    }
    return false;
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const QByteArray& name, const QByteArray& value)
        : name_(name)
        , hadValue_(qEnvironmentVariableIsSet(name.constData()))
        , oldValue_(qgetenv(name.constData()))
    {
        qputenv(name_.constData(), value);
    }

    ~ScopedEnvVar()
    {
        if (hadValue_) {
            qputenv(name_.constData(), oldValue_);
        } else {
            qunsetenv(name_.constData());
        }
    }

private:
    QByteArray name_;
    bool hadValue_ = false;
    QByteArray oldValue_;
};

void writeTinyPng(const QString& path)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QImage image(8, 8, QImage::Format_RGB888);
    image.fill(Qt::white);
    QVERIFY(image.save(path));
}

void writeTinyDetectionDataset(const QString& root)
{
    writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
    writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
}

void writeTinySegmentationDataset(const QString& root)
{
    writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [part]\n"));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
    writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.125 0.125 0.875 0.125 0.875 0.875 0.125 0.875\n"));
    writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.125 0.125 0.875 0.125 0.875 0.875 0.125 0.875\n"));
}

void writeTinyOcrRecDataset(const QString& root)
{
    writeTextFile(QDir(root).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\nb\n1\n2\n"));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/a.png")));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/b.png")));
    writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.png\tab12\nimages/b.png\tba\n"));
}

void writeTinyOcrDetDataset(const QString& root)
{
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/a.png")));
    writeTinyPng(QDir(root).filePath(QStringLiteral("images/b.png")));
    const QString boxA = QStringLiteral("[{\"transcription\":\"ab12\",\"points\":[[8,8],[42,8],[42,24],[8,24]]}]");
    const QString boxB = QStringLiteral("[{\"transcription\":\"###\",\"points\":[[52,8],[88,8],[88,24],[52,24]]}]");
    writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
        QStringLiteral("images/a.png\t%1\nimages/b.png\t%2\n").arg(boxA, boxB));
}

QString workerExecutablePath()
{
    const QString extension =
#ifdef Q_OS_WIN
        QStringLiteral(".exe");
#else
        QString();
#endif
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QString siblingBin = QDir(applicationDir).absoluteFilePath(QStringLiteral("../bin/aitrain_worker%1").arg(extension));
    if (QFileInfo::exists(siblingBin)) {
        return QDir::cleanPath(siblingBin);
    }
    return QDir(applicationDir).absoluteFilePath(QStringLiteral("aitrain_worker%1").arg(extension));
}

QByteArray testBase64UrlEncode(const QByteArray& input)
{
    return input.toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals);
}

QByteArray testBase64UrlDecode(QByteArray input)
{
    while (input.size() % 4 != 0) {
        input.append('=');
    }
    return QByteArray::fromBase64(input, QByteArray::Base64UrlEncoding);
}

QString tokenWithCustomer(const QString& token, const QString& customer)
{
    QList<QByteArray> parts = token.toLatin1().split('.');
    if (parts.size() != 3) {
        return token;
    }
    QJsonObject payload = QJsonDocument::fromJson(testBase64UrlDecode(parts.at(1))).object();
    payload.insert(QStringLiteral("customer"), customer);
    parts[1] = testBase64UrlEncode(QJsonDocument(payload).toJson(QJsonDocument::Compact));
    return QString::fromLatin1(parts.at(0) + QByteArrayLiteral(".") + parts.at(1) + QByteArrayLiteral(".") + parts.at(2));
}

QString builtPluginPath(const QString& fileName)
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../plugins/models/%1").arg(fileName)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("plugins/models/%1").arg(fileName)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../build-vscode/plugins/models/%1").arg(fileName)),
        QDir::current().absoluteFilePath(QStringLiteral("build-vscode/plugins/models/%1").arg(fileName))
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return {};
}

QString sha256FileForTest(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        hash.addData(file.read(1024 * 1024));
    }
    return QString::fromLatin1(hash.result().toHex());
}

QString powershellExecutablePath()
{
    const QStringList candidates = {
        QStandardPaths::findExecutable(QStringLiteral("powershell")),
        QStandardPaths::findExecutable(QStringLiteral("powershell.exe")),
        QStringLiteral("powershell")
    };
    for (const QString& candidate : candidates) {
        if (!candidate.isEmpty()) {
            return candidate;
        }
    }
    return {};
}

bool zipDirectoryForTest(const QString& sourceDir, const QString& zipPath, QString* error)
{
    const QString powershell = powershellExecutablePath();
    if (powershell.isEmpty()) {
        if (error) {
            *error = QStringLiteral("PowerShell is not available.");
        }
        return false;
    }
    QDir().mkpath(QFileInfo(zipPath).absolutePath());
    QFile::remove(zipPath);

    QTemporaryDir scriptDir;
    if (!scriptDir.isValid()) {
        if (error) {
            *error = QStringLiteral("Cannot create temporary directory for zip helper.");
        }
        return false;
    }
    const QString scriptPath = QDir(scriptDir.path()).filePath(QStringLiteral("zip_fixture.ps1"));
    writeTextFile(scriptPath,
        QStringLiteral("param([Parameter(Mandatory=$true)][string]$SourceDir,"
                       "[Parameter(Mandatory=$true)][string]$ZipPath)\n"
                       "Compress-Archive -Path (Join-Path $SourceDir '*') -DestinationPath $ZipPath -Force\n"));

    QProcess process;
    process.start(powershell,
        QStringList()
            << QStringLiteral("-NoProfile")
            << QStringLiteral("-ExecutionPolicy")
            << QStringLiteral("Bypass")
            << QStringLiteral("-File")
            << scriptPath
            << QFileInfo(sourceDir).absoluteFilePath()
            << QFileInfo(zipPath).absoluteFilePath());
    if (!process.waitForStarted(5000) || !process.waitForFinished(30000)
        || process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        if (error) {
            const QString stderrText = QString::fromLocal8Bit(process.readAllStandardError()).trimmed();
            *error = stderrText.isEmpty()
                ? QStringLiteral("Compress-Archive failed.")
                : stderrText;
        }
        return false;
    }
    return QFileInfo::exists(zipPath);
}

QString writeMarketplacePackageFixture(const QString& root, const QString& pluginDllPath, const QString& pluginId, const QString& version)
{
    const QString packageRoot = QDir(root).filePath(QStringLiteral("package"));
    const QString payloadPlugin = QDir(packageRoot).filePath(QStringLiteral("payload/plugins/models/%1").arg(QFileInfo(pluginDllPath).fileName()));
    if (!QDir().mkpath(QFileInfo(payloadPlugin).absolutePath())) {
        return {};
    }
    QFile::remove(payloadPlugin);
    if (!QFile::copy(pluginDllPath, payloadPlugin)) {
        return {};
    }
    writeTextFile(QDir(packageRoot).filePath(QStringLiteral("LICENSE")), QStringLiteral("Test license\n"));
    const QString relativeDll = QStringLiteral("payload/plugins/models/%1").arg(QFileInfo(pluginDllPath).fileName());
    const QString digest = sha256FileForTest(payloadPlugin);
    if (digest.isEmpty()) {
        return {};
    }
    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 1);
    manifest.insert(QStringLiteral("id"), pluginId);
    manifest.insert(QStringLiteral("name"), QStringLiteral("Marketplace Fixture"));
    manifest.insert(QStringLiteral("version"), version);
    manifest.insert(QStringLiteral("description"), QStringLiteral("Fixture plugin package."));
    manifest.insert(QStringLiteral("publisher"), QStringLiteral("AITrain Tests"));
    manifest.insert(QStringLiteral("license"), QStringLiteral("Test"));
    manifest.insert(QStringLiteral("category"), QStringLiteral("dataset_interop"));
    manifest.insert(QStringLiteral("capabilities"), QJsonArray{QStringLiteral("dataset_interop")});
    manifest.insert(QStringLiteral("entrypoints"), QJsonObject{{QStringLiteral("qtModelPlugin"), relativeDll}});
    manifest.insert(QStringLiteral("compatibility"), QJsonObject{
        {QStringLiteral("minAitrainVersion"), QStringLiteral("0.1.0")},
        {QStringLiteral("qtAbi"), aitrain::PluginMarketplace::currentQtAbi()},
        {QStringLiteral("requiresGpu"), false}
    });
    manifest.insert(QStringLiteral("files"), QJsonArray{relativeDll});
    manifest.insert(QStringLiteral("hashes"), QJsonObject{{relativeDll, digest}});
    QFile manifestFile(QDir(packageRoot).filePath(QStringLiteral("plugin.json")));
    if (!manifestFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return {};
    }
    manifestFile.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented));
    return packageRoot;
}

QString pythonExecutablePath()
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/python-3.13.13-embed-amd64/python.exe")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/python-3.13.13-embed-amd64/python.exe")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/python-3.13.13-embed-amd64/python.exe")),
        QStandardPaths::findExecutable(QStringLiteral("python")),
        QStandardPaths::findExecutable(QStringLiteral("python3"))
    };
    for (const QString& candidate : candidates) {
        if (candidate.isEmpty()) {
            continue;
        }
        QProcess process;
        process.start(candidate, QStringList() << QStringLiteral("--version"));
        if (process.waitForStarted(1000)
            && process.waitForFinished(2500)
            && process.exitStatus() == QProcess::NormalExit
            && process.exitCode() == 0) {
            return candidate;
        }
    }
    return {};
}

QString mockPythonTrainerScriptPath()
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../bin/python_trainers/mock_trainer.py")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("python_trainers/mock_trainer.py")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../python_trainers/mock_trainer.py")),
        QDir::current().absoluteFilePath(QStringLiteral("python_trainers/mock_trainer.py"))
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return {};
}

QString phase9RealSmokeRoot()
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    QStringList candidates;
    const QString acceptanceSmokeRoot = QString::fromLocal8Bit(qgetenv("AITRAIN_ACCEPTANCE_SMOKE_ROOT")).trimmed();
    if (!acceptanceSmokeRoot.isEmpty()) {
        candidates.append(QDir::cleanPath(acceptanceSmokeRoot));
        candidates.append(QDir(acceptanceSmokeRoot).filePath(QStringLiteral("generated")));
    }
    candidates.append({
        QDir::current().absoluteFilePath(QStringLiteral(".deps/phase9-real-smoke")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke/generated")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke/cpu-training-smoke")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke/cpu-training-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/phase9-real-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke/cpu-training-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke/cpu-training-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/phase9-real-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke/cpu-training-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke/cpu-training-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral(".deps/phase9-real-smoke"))
    });
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("request.json")))
            || QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("out/ultralytics_runs/phase9-real-smoke/weights/best.onnx")))
            || QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("runs/yolo_detect/ultralytics_runs/acceptance-yolo-detect/weights/best.onnx")))
            || QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("runs/yolo_detect/ultralytics_runs/cpu-yolo-detect/weights/best.onnx")))) {
            return QDir::cleanPath(candidate);
        }
    }
    return {};
}

QString phase11SegSmokeRoot()
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    QStringList candidates;
    const QString acceptanceSmokeRoot = QString::fromLocal8Bit(qgetenv("AITRAIN_ACCEPTANCE_SMOKE_ROOT")).trimmed();
    if (!acceptanceSmokeRoot.isEmpty()) {
        candidates.append(QDir::cleanPath(acceptanceSmokeRoot));
    }
    candidates.append({
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke/generated")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke/cpu-training-smoke")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke/cpu-training-smoke/generated")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/phase11-seg-smoke")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/phase13-examples")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke/cpu-training-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke/cpu-training-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/phase11-seg-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/phase13-examples")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke/cpu-training-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke/cpu-training-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/phase11-seg-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/phase13-examples"))
    });
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("out/ultralytics_runs/phase11-seg-smoke/weights/best.onnx")))
            || QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/aitrain-yolo-segment/weights/best.onnx")))
            || QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/acceptance-yolo-segment/weights/best.onnx")))
            || QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/cpu-yolo-segment/weights/best.onnx")))) {
            return QDir::cleanPath(candidate);
        }
    }
    return {};
}

QString phase14OcrSmokeRoot()
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    QStringList candidates;
    const QString acceptanceSmokeRoot = QString::fromLocal8Bit(qgetenv("AITRAIN_ACCEPTANCE_SMOKE_ROOT")).trimmed();
    if (!acceptanceSmokeRoot.isEmpty()) {
        candidates.append(QDir(acceptanceSmokeRoot).filePath(QStringLiteral("generated")));
    }
    candidates.append({
        QDir::current().absoluteFilePath(QStringLiteral(".deps/acceptance-smoke/generated")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/phase14-ocr-smoke")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/phase13-examples")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/acceptance-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/phase14-ocr-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../.deps/phase13-examples")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/acceptance-smoke/generated")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/phase14-ocr-smoke")),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../.deps/phase13-examples"))
    });
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("runs/paddleocr_rec/paddleocr_rec_ctc.onnx")))) {
            return QDir::cleanPath(candidate);
        }
    }
    return {};
}

void writeFakeUltralyticsPackage(const QString& root)
{
    writeTextFile(
        QDir(root).filePath(QStringLiteral("ultralytics/__init__.py")),
        QStringLiteral(
            "from pathlib import Path\n"
            "from types import SimpleNamespace\n"
            "\n"
            "class YOLO:\n"
            "    def __init__(self, model):\n"
            "        self.model = str(model)\n"
            "\n"
            "    def train(self, data, epochs, imgsz, batch, device, workers, project, name, exist_ok, verbose):\n"
            "        save_dir = Path(project) / name\n"
            "        weights_dir = save_dir / 'weights'\n"
            "        weights_dir.mkdir(parents=True, exist_ok=True)\n"
            "        (weights_dir / 'best.pt').write_text('fake best checkpoint\\n', encoding='utf-8')\n"
            "        (weights_dir / 'last.pt').write_text('fake last checkpoint\\n', encoding='utf-8')\n"
            "        (save_dir / 'results.csv').write_text(\n"
            "            'epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\\n'\n"
            "            '1,0.2,0.3,0.4,0.81,0.72,0.63,0.54\\n',\n"
            "            encoding='utf-8')\n"
            "        (save_dir / 'args.yaml').write_text('model: fake\\n', encoding='utf-8')\n"
            "        return SimpleNamespace(save_dir=str(save_dir))\n"
            "\n"
            "    def export(self, format, imgsz, device):\n"
            "        model_path = Path(self.model)\n"
            "        output_path = model_path.with_suffix('.onnx') if model_path.suffix else Path('model.onnx')\n"
            "        output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "        output_path.write_text('fake onnx\\n', encoding='utf-8')\n"
            "        return str(output_path)\n"));
}

} // namespace
