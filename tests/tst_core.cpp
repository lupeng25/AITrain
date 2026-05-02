#include "WorkerClient.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionDataset.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/LicenseManager.h"
#include "aitrain/core/OcrRecDataset.h"
#include "aitrain/core/OcrRecTrainer.h"
#include "aitrain/core/ProjectRepository.h"
#include "aitrain/core/SegmentationDataset.h"
#include "aitrain/core/SegmentationTrainer.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>

namespace {

void writeTextFile(const QString& path, const QString& content)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
    file.write(content.toUtf8());
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

class CoreTests : public QObject {
    Q_OBJECT

private slots:
    void taskStateTransitions()
    {
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Running));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Failed));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Paused));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Paused, aitrain::TaskState::Running));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Paused, aitrain::TaskState::Canceled));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Completed));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Completed, aitrain::TaskState::Running));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Failed, aitrain::TaskState::Running));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Canceled, aitrain::TaskState::Running));
    }

    void protocolRoundTrip()
    {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), QStringLiteral("abc"));
        payload.insert(QStringLiteral("value"), 42);

        const QByteArray encoded = aitrain::protocol::encodeMessage(QStringLiteral("metric"), payload, QStringLiteral("req-1"));
        QVERIFY(encoded.endsWith('\n'));

        QString type;
        QJsonObject decodedPayload;
        QString requestId;
        QString error;
        QVERIFY(aitrain::protocol::decodeMessage(encoded, &type, &decodedPayload, &requestId, &error));
        QCOMPARE(type, QStringLiteral("metric"));
        QCOMPARE(requestId, QStringLiteral("req-1"));
        QCOMPARE(decodedPayload.value(QStringLiteral("taskId")).toString(), QStringLiteral("abc"));
        QCOMPARE(decodedPayload.value(QStringLiteral("value")).toInt(), 42);
        QVERIFY(error.isEmpty());
    }

    void offlineLicenseTokensValidate()
    {
        if (!aitrain::licenseCryptoAvailable()) {
            QSKIP("Offline ECDSA license tests require the Windows CNG implementation");
        }

        QString error;
        aitrain::LicenseKeyPair keyPair;
        QVERIFY2(aitrain::generateLicenseKeyPair(&keyPair, &error), qPrintable(error));
        QVERIFY(!keyPair.publicKeyBase64.isEmpty());
        QVERIFY(!keyPair.privateKeyBase64.isEmpty());
        QCOMPARE(aitrain::publicKeyFromPrivateKey(keyPair.privateKeyBase64, &error), keyPair.publicKeyBase64);

        const QDateTime now = QDateTime::fromString(QStringLiteral("2026-05-01T00:00:00Z"), Qt::ISODate);
        aitrain::LicensePayload payload;
        payload.product = aitrain::licenseProductName();
        payload.customer = QStringLiteral("Smoke Customer");
        payload.machineCode = QStringLiteral("ABCD-EF12-3456-7890-CAFE");
        payload.licenseId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        payload.issuedAt = now.addDays(-1);
        payload.expiresAt = now.addDays(30);

        const QString token = aitrain::createLicenseToken(payload, keyPair.privateKeyBase64, &error);
        QVERIFY2(!token.isEmpty(), qPrintable(error));

        const aitrain::LicenseValidationResult valid =
            aitrain::validateLicenseToken(token, keyPair.publicKeyBase64, payload.machineCode, now);
        QVERIFY2(valid.isValid(), qPrintable(valid.message));
        QCOMPARE(valid.payload.customer, payload.customer);

        const aitrain::LicenseValidationResult mismatch =
            aitrain::validateLicenseToken(token, keyPair.publicKeyBase64, QStringLiteral("0000-0000-0000-0000-0000"), now);
        QCOMPARE(static_cast<int>(mismatch.status), static_cast<int>(aitrain::LicenseStatus::MachineMismatch));

        const aitrain::LicenseValidationResult tampered =
            aitrain::validateLicenseToken(tokenWithCustomer(token, QStringLiteral("Mallory")), keyPair.publicKeyBase64, payload.machineCode, now);
        QCOMPARE(static_cast<int>(tampered.status), static_cast<int>(aitrain::LicenseStatus::SignatureInvalid));

        aitrain::LicensePayload expiredPayload = payload;
        expiredPayload.licenseId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        expiredPayload.expiresAt = now.addSecs(-1);
        const QString expiredToken = aitrain::createLicenseToken(expiredPayload, keyPair.privateKeyBase64, &error);
        QVERIFY2(!expiredToken.isEmpty(), qPrintable(error));
        const aitrain::LicenseValidationResult expired =
            aitrain::validateLicenseToken(expiredToken, keyPair.publicKeyBase64, payload.machineCode, now);
        QCOMPARE(static_cast<int>(expired.status), static_cast<int>(aitrain::LicenseStatus::Expired));

        QCOMPARE(static_cast<int>(aitrain::validateLicenseToken(QString(), keyPair.publicKeyBase64, payload.machineCode, now).status),
            static_cast<int>(aitrain::LicenseStatus::MissingToken));
        QCOMPARE(static_cast<int>(aitrain::validateLicenseToken(QStringLiteral("bad-token"), keyPair.publicKeyBase64, payload.machineCode, now).status),
            static_cast<int>(aitrain::LicenseStatus::MalformedToken));
    }

    void packagingLayoutUsesPhaseSevenInstallShape()
    {
        const QString root = QDir::cleanPath(QDir(QDir::tempPath()).filePath(QStringLiteral("AITrainStudioPackage")));
        const aitrain::PackagingLayout layout = aitrain::packagingLayoutForRoot(root);
        const QString appExecutableName =
#ifdef Q_OS_WIN
            QStringLiteral("AITrainStudio.exe");
#else
            QStringLiteral("AITrainStudio");
#endif
        const QString workerExecutableName =
#ifdef Q_OS_WIN
            QStringLiteral("aitrain_worker.exe");
#else
            QStringLiteral("aitrain_worker");
#endif

        QCOMPARE(layout.rootPath, root);
        QCOMPARE(layout.appExecutablePath, QDir(root).filePath(appExecutableName));
        QCOMPARE(layout.workerExecutablePath, QDir(root).filePath(workerExecutableName));
        QCOMPARE(layout.pluginModelsDirectory, QDir(root).filePath(QStringLiteral("plugins/models")));
        QCOMPARE(layout.onnxRuntimeDirectory, QDir(root).filePath(QStringLiteral("runtimes/onnxruntime")));
        QCOMPARE(layout.tensorRtRuntimeDirectory, QDir(root).filePath(QStringLiteral("runtimes/tensorrt")));
        QCOMPARE(layout.examplesDirectory, QDir(root).filePath(QStringLiteral("examples")));
        QCOMPARE(layout.docsDirectory, QDir(root).filePath(QStringLiteral("docs")));
        QCOMPARE(layout.toJson().value(QStringLiteral("appExecutablePath")).toString(), layout.appExecutablePath);
    }

    void runtimeDependencyCheckReportsSearchPaths()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QStringList paths = aitrain::runtimeSearchPaths(dir.path());
        QVERIFY(paths.contains(QDir(dir.path()).filePath(QStringLiteral("runtimes/onnxruntime"))));
        QVERIFY(paths.contains(QDir(dir.path()).filePath(QStringLiteral("runtimes/tensorrt"))));

        const aitrain::RuntimeDependencyCheck check = aitrain::checkRuntimeDependency(
            QStringLiteral("Missing Runtime"),
            QStringList() << QStringLiteral("aitrain_definitely_missing_runtime"),
            QStringLiteral("expected missing runtime message."),
            dir.path());
        QCOMPARE(check.status, QStringLiteral("missing"));
        QVERIFY(check.message.contains(QStringLiteral("Missing Runtime")));
        QVERIFY(check.message.contains(QStringLiteral("expected missing runtime message")));

        const QString json = QString::fromUtf8(QJsonDocument(check.toJson()).toJson(QJsonDocument::Compact));
        QVERIFY(json.contains(QStringLiteral("runtimes/onnxruntime")));
        QVERIFY(json.contains(QStringLiteral("runtimes/tensorrt")));
        QVERIFY(json.contains(QStringLiteral("aitrain_definitely_missing_runtime")));
    }

    void repositoryStoresTasksAndMetrics()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        aitrain::ProjectRepository repository;
        QString error;
        QVERIFY2(repository.open(dir.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
        QVERIFY2(repository.upsertProject(QStringLiteral("demo"), dir.path(), &error), qPrintable(error));

        aitrain::TaskRecord task;
        task.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        task.projectName = QStringLiteral("demo");
        task.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        task.taskType = QStringLiteral("detection");
        task.kind = aitrain::TaskKind::Train;
        task.state = aitrain::TaskState::Queued;
        task.workDir = dir.filePath(QStringLiteral("runs/1"));
        task.createdAt = QDateTime::currentDateTimeUtc();
        task.updatedAt = task.createdAt;
        QVERIFY2(repository.insertTask(task, &error), qPrintable(error));
        QVERIFY2(repository.updateTaskState(task.id, aitrain::TaskState::Running, QStringLiteral("started"), &error), qPrintable(error));
        QVERIFY2(!repository.updateTaskState(task.id, aitrain::TaskState::Queued, QStringLiteral("invalid"), &error), qPrintable(error));

        aitrain::MetricPoint metric;
        metric.taskId = task.id;
        metric.name = QStringLiteral("loss");
        metric.value = 0.75;
        metric.step = 1;
        metric.epoch = 1;
        metric.createdAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.insertMetric(metric, &error), qPrintable(error));

        aitrain::ArtifactRecord artifact;
        artifact.taskId = task.id;
        artifact.kind = QStringLiteral("checkpoint");
        artifact.path = dir.filePath(QStringLiteral("runs/1/checkpoint_best.aitrain"));
        artifact.message = QStringLiteral("scaffold checkpoint");
        QVERIFY2(repository.insertArtifact(artifact, &error), qPrintable(error));

        QJsonObject exportConfig;
        exportConfig.insert(QStringLiteral("format"), QStringLiteral("onnx"));
        exportConfig.insert(QStringLiteral("sourceCheckpoint"), artifact.path);
        exportConfig.insert(QStringLiteral("input"), QJsonObject{
            {QStringLiteral("name"), QStringLiteral("features")},
            {QStringLiteral("shape"), QJsonArray{16, 7}}
        });
        exportConfig.insert(QStringLiteral("outputs"), QJsonArray{
            QJsonObject{{QStringLiteral("name"), QStringLiteral("objectness")}, {QStringLiteral("shape"), QJsonArray{16, 1}}},
            QJsonObject{{QStringLiteral("name"), QStringLiteral("class_probabilities")}, {QStringLiteral("shape"), QJsonArray{16, 1}}},
            QJsonObject{{QStringLiteral("name"), QStringLiteral("boxes")}, {QStringLiteral("shape"), QJsonArray{16, 4}}}
        });
        aitrain::ExportRecord exportRecord;
        exportRecord.taskId = task.id;
        exportRecord.sourceCheckpointPath = artifact.path;
        exportRecord.format = QStringLiteral("onnx");
        exportRecord.path = dir.filePath(QStringLiteral("runs/1/model.onnx"));
        exportRecord.configJson = QString::fromUtf8(QJsonDocument(exportConfig).toJson(QJsonDocument::Compact));
        exportRecord.inputShapeJson = QString::fromUtf8(QJsonDocument(exportConfig.value(QStringLiteral("input")).toObject()).toJson(QJsonDocument::Compact));
        exportRecord.outputShapeJson = QString::fromUtf8(QJsonDocument(QJsonObject{{QStringLiteral("outputs"), exportConfig.value(QStringLiteral("outputs")).toArray()}}).toJson(QJsonDocument::Compact));
        exportRecord.createdAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.insertExport(exportRecord, &error), qPrintable(error));

        aitrain::EnvironmentCheckRecord check;
        check.name = QStringLiteral("Worker");
        check.status = QStringLiteral("ok");
        check.message = QStringLiteral("Worker executable found");
        QVERIFY2(repository.insertEnvironmentCheck(check, &error), qPrintable(error));

        const QVector<aitrain::TaskRecord> tasks = repository.recentTasks(10, &error);
        QCOMPARE(tasks.size(), 1);
        QCOMPARE(tasks.first().id, task.id);
        QCOMPARE(tasks.first().state, aitrain::TaskState::Running);
        QVERIFY(tasks.first().startedAt.isValid());
        QVERIFY(!tasks.first().finishedAt.isValid());

        const QVector<aitrain::EnvironmentCheckRecord> checks = repository.recentEnvironmentChecks(10, &error);
        QCOMPARE(checks.size(), 1);
        QCOMPARE(checks.first().name, QStringLiteral("Worker"));
        QCOMPARE(checks.first().status, QStringLiteral("ok"));

        const QVector<aitrain::ExportRecord> exports = repository.recentExports(10, &error);
        QCOMPARE(exports.size(), 1);
        QCOMPARE(exports.first().taskId, task.id);
        QCOMPARE(exports.first().format, QStringLiteral("onnx"));
        QVERIFY(exports.first().configJson.contains(QStringLiteral("sourceCheckpoint")));
        QVERIFY(exports.first().inputShapeJson.contains(QStringLiteral("features")));
        QVERIFY(exports.first().outputShapeJson.contains(QStringLiteral("boxes")));
        const QVector<aitrain::ArtifactRecord> taskArtifacts = repository.artifactsForTask(task.id, &error);
        QCOMPARE(taskArtifacts.size(), 1);
        QCOMPARE(taskArtifacts.first().kind, QStringLiteral("checkpoint"));
        QCOMPARE(taskArtifacts.first().path, artifact.path);
        const QVector<aitrain::MetricPoint> taskMetrics = repository.metricsForTask(task.id, &error);
        QCOMPARE(taskMetrics.size(), 1);
        QCOMPARE(taskMetrics.first().name, QStringLiteral("loss"));
        QCOMPARE(taskMetrics.first().step, 1);
        const QVector<aitrain::ExportRecord> taskExports = repository.exportsForTask(task.id, &error);
        QCOMPARE(taskExports.size(), 1);
        QCOMPARE(taskExports.first().path, exportRecord.path);

        aitrain::DatasetRecord dataset;
        dataset.name = QStringLiteral("demo-dataset");
        dataset.format = QStringLiteral("yolo_detection");
        dataset.rootPath = dir.filePath(QStringLiteral("datasets/demo"));
        dataset.validationStatus = QStringLiteral("valid");
        dataset.sampleCount = 2;
        dataset.lastReportJson = QStringLiteral("{\"ok\":true}");
        dataset.lastValidatedAt = QDateTime::currentDateTimeUtc();
        QVERIFY2(repository.upsertDatasetValidation(dataset, &error), qPrintable(error));
        const QVector<aitrain::DatasetRecord> datasets = repository.recentDatasets(10, &error);
        QCOMPARE(datasets.size(), 1);
        QCOMPARE(datasets.first().validationStatus, QStringLiteral("valid"));
        QCOMPARE(datasets.first().sampleCount, 2);
        const aitrain::DatasetRecord loadedDataset = repository.datasetByRootPath(dataset.rootPath, &error);
        QCOMPARE(loadedDataset.format, QStringLiteral("yolo_detection"));
        const QVector<aitrain::DatasetVersionRecord> versions = repository.datasetVersions(loadedDataset.id, &error);
        QCOMPARE(versions.size(), 1);
        QCOMPARE(versions.first().rootPath, dataset.rootPath);
        QVERIFY(versions.first().metadataJson.contains(QStringLiteral("\"ok\":true")));

        QVERIFY2(repository.updateTaskState(task.id, aitrain::TaskState::Completed, QStringLiteral("done"), &error), qPrintable(error));
        const QVector<aitrain::TaskRecord> completedTasks = repository.recentTasks(10, &error);
        QCOMPARE(completedTasks.first().state, aitrain::TaskState::Completed);
        QVERIFY(completedTasks.first().finishedAt.isValid());
        QVERIFY(!repository.updateTaskState(task.id, aitrain::TaskState::Running, QStringLiteral("invalid"), &error));
    }

    void repositoryMarksInterruptedTasksFailed()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        aitrain::ProjectRepository repository;
        QString error;
        QVERIFY2(repository.open(dir.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

        aitrain::TaskRecord task;
        task.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        task.projectName = QStringLiteral("demo");
        task.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        task.taskType = QStringLiteral("detection");
        task.kind = aitrain::TaskKind::Train;
        task.state = aitrain::TaskState::Running;
        task.workDir = dir.filePath(QStringLiteral("runs/1"));
        task.createdAt = QDateTime::currentDateTimeUtc();
        task.updatedAt = task.createdAt;
        task.startedAt = task.createdAt;
        QVERIFY2(repository.insertTask(task, &error), qPrintable(error));

        QVERIFY2(repository.markInterruptedTasksFailed(QStringLiteral("Worker interrupted"), &error), qPrintable(error));
        const QVector<aitrain::TaskRecord> tasks = repository.recentTasks(10, &error);
        QCOMPARE(tasks.size(), 1);
        QCOMPARE(tasks.first().state, aitrain::TaskState::Failed);
        QCOMPARE(tasks.first().message, QStringLiteral("Worker interrupted"));
        QVERIFY(tasks.first().finishedAt.isValid());
    }

    void yoloDetectionDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("1 0.5 0.5 0.20 0.20\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);
        QVERIFY(!valid.previewSamples.isEmpty());
        QVERIFY(valid.previewSamples.first().contains(QStringLiteral("bbox=")));

        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("3 0.5 0.5 0.25 0.25\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY(!invalid.ok);
        QVERIFY(!invalid.issues.isEmpty());
        QCOMPARE(invalid.issues.first().line, 1);
    }

    void yoloDetectionDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("source"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        for (int index = 0; index < 4; ++index) {
            const QString split = index < 3 ? QStringLiteral("train") : QStringLiteral("val");
            const QString name = QStringLiteral("sample_%1").arg(index);
            writeTinyPng(QDir(root).filePath(QStringLiteral("images/%1/%2.jpg").arg(split, name)));
            writeTextFile(QDir(root).filePath(QStringLiteral("labels/%1/%2.txt").arg(split, name)), QStringLiteral("0 0.5 0.5 0.2 0.2\n"));
        }

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.25);
        options.insert(QStringLiteral("testRatio"), 0.25);
        options.insert(QStringLiteral("seed"), 7);
        const QString output = dir.filePath(QStringLiteral("normalized"));
        const aitrain::DatasetSplitResult result = aitrain::splitYoloDetectionDataset(root, output, options);
        QVERIFY2(result.ok, qPrintable(result.errors.join(QStringLiteral("\n"))));
        QCOMPARE(result.trainCount, 2);
        QCOMPARE(result.valCount, 1);
        QCOMPARE(result.testCount, 1);
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("data.yaml"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("split_report.json"))));
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/train"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 2);
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/val"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 1);
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/test"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 1);
        QVERIFY(QFileInfo::exists(QDir(root).filePath(QStringLiteral("images/train/sample_0.jpg"))));
    }

    void yoloSegmentationAndOcrDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString segRoot = dir.filePath(QStringLiteral("seg-source"));
        writeTinySegmentationDataset(segRoot);
        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        const QString segOutput = dir.filePath(QStringLiteral("seg-normalized"));
        const aitrain::DatasetSplitResult segResult = aitrain::splitYoloSegmentationDataset(segRoot, segOutput, options);
        QVERIFY2(segResult.ok, qPrintable(segResult.errors.join(QStringLiteral("\n"))));
        QVERIFY(QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("data.yaml"))));
        QVERIFY(QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("labels/train/a.txt")))
            || QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("labels/val/a.txt"))));
        QVERIFY(QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("split_report.json"))));

        const QString ocrRoot = dir.filePath(QStringLiteral("ocr-source"));
        writeTinyOcrRecDataset(ocrRoot);
        const QString ocrOutput = dir.filePath(QStringLiteral("ocr-normalized"));
        const aitrain::DatasetSplitResult ocrResult = aitrain::splitPaddleOcrRecDataset(ocrRoot, ocrOutput, options);
        QVERIFY2(ocrResult.ok, qPrintable(ocrResult.errors.join(QStringLiteral("\n"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("dict.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("rec_gt.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("rec_gt_train.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("rec_gt_val.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("split_report.json"))));
        const aitrain::DatasetValidationResult validation = aitrain::validatePaddleOcrRecDataset(ocrOutput);
        QVERIFY2(validation.ok, qPrintable(validation.errors.join(QStringLiteral("\n"))));
    }

    void detectionDatasetLoadsSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/b.txt")), QStringLiteral("1 0.4 0.4 0.20 0.30\n"));

        QString error;
        const aitrain::DetectionDatasetInfo info = aitrain::readDetectionDatasetInfo(root, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(info.classCount, 2);
        QCOMPARE(info.classNames.size(), 2);
        QCOMPARE(info.classNames.at(1), QStringLiteral("dog"));

        aitrain::DetectionDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));
        QCOMPARE(dataset.size(), 2);
        QCOMPARE(dataset.info().classCount, 2);
        QCOMPARE(dataset.samples().first().boxes.size(), 1);
        QCOMPARE(dataset.samples().first().boxes.first().classId, 0);
        QCOMPARE(dataset.samples().at(1).boxes.first().classId, 1);
    }

    void detectionDatasetRejectsInvalidLabel()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("2 0.5 0.5 0.25 0.25\n"));

        QString error;
        aitrain::DetectionDataset dataset;
        QVERIFY(!dataset.load(root, QStringLiteral("train"), &error));
        QVERIFY(error.contains(QStringLiteral("class id")));
    }

    void detectionDataLoaderBuildsLetterboxBatch()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));

        QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train")));
        QImage wideImage(16, 8, QImage::Format_RGB888);
        wideImage.fill(Qt::white);
        QVERIFY(wideImage.save(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.5 0.5\n"));

        QString error;
        aitrain::DetectionDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));

        aitrain::DetectionDataLoader loader(dataset, 1, QSize(32, 32));
        QVERIFY(loader.hasNext());
        aitrain::DetectionBatch batch;
        QVERIFY2(loader.next(&batch, &error), qPrintable(error));
        QCOMPARE(batch.images.size(), 1);
        QCOMPARE(batch.images.first().size(), QSize(32, 32));
        QCOMPARE(batch.boxes.first().size(), 1);
        QCOMPARE(batch.boxes.first().first().classId, 0);
        QCOMPARE(batch.boxes.first().first().xCenter, 0.5);
        QCOMPARE(batch.boxes.first().first().yCenter, 0.5);
        QCOMPARE(batch.boxes.first().first().width, 0.5);
        QCOMPARE(batch.boxes.first().first().height, 0.25);
        QVERIFY(!loader.hasNext());
    }

    void detectionTrainingBackendStatusMarksPhase8Scaffold()
    {
        const QJsonObject status = aitrain::detectionTrainingBackendStatus();
        QCOMPARE(status.value(QStringLiteral("phase")).toInt(), 8);
        QCOMPARE(status.value(QStringLiteral("activeBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QVERIFY(status.value(QStringLiteral("activeBackendScaffold")).toBool());
        QVERIFY(!status.value(QStringLiteral("realYoloStyleTrainingAvailable")).toBool(true));
        QVERIFY(!status.value(QStringLiteral("availableBackends")).toArray().isEmpty());
    }

    void detectionTrainerWritesTinyDetectorCheckpoint()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/b.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/c.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/b.txt")), QStringLiteral("1 0.4 0.4 0.20 0.30\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/c.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 2;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));

        int callbackCount = 0;
        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(
            root,
            options,
            [&callbackCount](const aitrain::DetectionTrainingMetrics& metrics) {
                ++callbackCount;
                return metrics.epoch >= 1 && metrics.step >= 1 && metrics.loss >= 0.0;
            });

        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.trainingBackend, QStringLiteral("tiny_linear_detector"));
        QCOMPARE(result.modelFamily, QStringLiteral("yolo_style_detection_scaffold"));
        QVERIFY(result.scaffold);
        QCOMPARE(result.modelArchitecture.value(QStringLiteral("family")).toString(), QStringLiteral("tiny_linear_detector_scaffold"));
        QCOMPARE(result.steps, 4);
        QCOMPARE(callbackCount, 4);
        QVERIFY(result.finalLoss >= 0.0);
        QVERIFY(QFileInfo::exists(result.checkpointPath));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("type")).toString(), QStringLiteral("tiny_linear_detector"));
        QCOMPARE(json.value(QStringLiteral("checkpointSchemaVersion")).toInt(), 2);
        QCOMPARE(json.value(QStringLiteral("trainingBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QCOMPARE(json.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_style_detection_scaffold"));
        QVERIFY(json.value(QStringLiteral("scaffold")).toBool());
        QCOMPARE(json.value(QStringLiteral("modelArchitecture")).toObject().value(QStringLiteral("family")).toString(), QStringLiteral("tiny_linear_detector_scaffold"));
        QVERIFY(!json.value(QStringLiteral("phase8")).toObject().value(QStringLiteral("realYoloStyleTraining")).toBool(true));
        QCOMPARE(json.value(QStringLiteral("steps")).toInt(), 4);
        QCOMPARE(json.value(QStringLiteral("gridSize")).toInt(), 4);
        QVERIFY(json.value(QStringLiteral("featureCount")).toInt() > 0);
        QVERIFY(!json.value(QStringLiteral("objectnessWeights")).toArray().isEmpty());
        QVERIFY(!json.value(QStringLiteral("classWeights")).toArray().isEmpty());
        QVERIFY(!json.value(QStringLiteral("boxWeights")).toArray().isEmpty());
        QVERIFY(json.contains(QStringLiteral("mAP50")));
        QVERIFY(json.value(QStringLiteral("precision")).toDouble() >= 0.0);
        QVERIFY(json.value(QStringLiteral("recall")).toDouble() >= 0.0);
        QVERIFY(json.value(QStringLiteral("mAP50")).toDouble() >= 0.0);
        QVERIFY(json.value(QStringLiteral("mAP50")).toDouble() <= 1.0);

        QString error;
        aitrain::DetectionBaselineCheckpoint loaded;
        QVERIFY2(aitrain::loadDetectionBaselineCheckpoint(result.checkpointPath, &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.type, QStringLiteral("tiny_linear_detector"));
        QCOMPARE(loaded.checkpointSchemaVersion, 2);
        QCOMPARE(loaded.trainingBackend, QStringLiteral("tiny_linear_detector"));
        QCOMPARE(loaded.modelFamily, QStringLiteral("yolo_style_detection_scaffold"));
        QVERIFY(loaded.scaffold);
        QCOMPARE(loaded.modelArchitecture.value(QStringLiteral("family")).toString(), QStringLiteral("tiny_linear_detector_scaffold"));
        QCOMPARE(loaded.steps, 4);
        QCOMPARE(loaded.gridSize, 4);
        QVERIFY(loaded.featureCount > 0);
        QVERIFY(!loaded.objectnessWeights.isEmpty());
        QVERIFY(!loaded.classWeights.isEmpty());
        QVERIFY(!loaded.boxWeights.isEmpty());
        QCOMPARE(loaded.classNames.size(), 2);
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(
            loaded,
            QDir(root).filePath(QStringLiteral("images/val/c.png")),
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
        QVERIFY(predictions.first().objectness >= 0.0);
        QVERIFY(predictions.first().confidence >= 0.0);
        QVERIFY(predictions.first().box.width > 0.0);
        const QJsonObject predictionJson = aitrain::detectionPredictionToJson(predictions.first());
        QCOMPARE(predictionJson.value(QStringLiteral("className")).toString(), predictions.first().className);
        const QImage rendered = aitrain::renderDetectionPredictions(
            QDir(root).filePath(QStringLiteral("images/val/c.png")),
            predictions,
            &error);
        QVERIFY2(!rendered.isNull(), qPrintable(error));
        QCOMPARE(rendered.size(), QSize(8, 8));
    }

    void detectionTrainerRejectsUnavailableYoloStyleBackend()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::DetectionTrainingOptions options;
        options.trainingBackend = QStringLiteral("yolo_style_libtorch");
        options.outputPath = dir.filePath(QStringLiteral("run"));

        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(root, options);
        QVERIFY(!result.ok);
        QCOMPARE(result.trainingBackend, QStringLiteral("yolo_style_libtorch"));
        QVERIFY(result.error.contains(QStringLiteral("not available")));
        QVERIFY(result.error.contains(QStringLiteral("YOLO-style")));
    }

    void detectionPostProcessAppliesThresholdNmsAndLimit()
    {
        QVector<aitrain::DetectionPrediction> predictions;

        aitrain::DetectionPrediction high;
        high.box.classId = 0;
        high.box.xCenter = 0.5;
        high.box.yCenter = 0.5;
        high.box.width = 0.4;
        high.box.height = 0.4;
        high.confidence = 0.9;
        high.objectness = 0.9;
        high.className = QStringLiteral("item");
        predictions.append(high);

        aitrain::DetectionPrediction overlap = high;
        overlap.box.xCenter = 0.52;
        overlap.box.yCenter = 0.52;
        overlap.confidence = 0.8;
        predictions.append(overlap);

        aitrain::DetectionPrediction otherClass = overlap;
        otherClass.box.classId = 1;
        otherClass.className = QStringLiteral("other");
        otherClass.confidence = 0.7;
        predictions.append(otherClass);

        aitrain::DetectionPrediction low = high;
        low.box.xCenter = 0.1;
        low.box.yCenter = 0.1;
        low.confidence = 0.05;
        predictions.append(low);

        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.1;
        options.iouThreshold = 0.5;
        options.maxDetections = 10;
        const QVector<aitrain::DetectionPrediction> filtered = aitrain::postProcessDetectionPredictions(predictions, options);
        QCOMPARE(filtered.size(), 2);
        QCOMPARE(filtered.at(0).confidence, 0.9);
        QCOMPARE(filtered.at(0).box.classId, 0);
        QCOMPARE(filtered.at(1).box.classId, 1);

        options.maxDetections = 1;
        const QVector<aitrain::DetectionPrediction> limited = aitrain::postProcessDetectionPredictions(predictions, options);
        QCOMPARE(limited.size(), 1);
        QCOMPARE(limited.first().confidence, 0.9);
    }

    void detectionTrainerLossDecreasesOnSimpleDataset()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.70 0.30 0.40 0.20\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.70 0.30 0.40 0.20\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 8;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.learningRate = 0.2;
        options.outputPath = dir.filePath(QStringLiteral("run"));

        double firstLoss = -1.0;
        double lastLoss = -1.0;
        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(
            root,
            options,
            [&firstLoss, &lastLoss](const aitrain::DetectionTrainingMetrics& metrics) {
                if (firstLoss < 0.0) {
                    firstLoss = metrics.loss;
                }
                lastLoss = metrics.loss;
                return true;
            });

        QVERIFY2(result.ok, qPrintable(result.error));
        QVERIFY(firstLoss >= 0.0);
        QVERIFY(lastLoss >= 0.0);
        QVERIFY2(lastLoss < firstLoss, qPrintable(QStringLiteral("first=%1 last=%2").arg(firstLoss).arg(lastLoss)));
    }

    void detectionTrainerSupportsAugmentationOptions()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.25 0.50 0.30 0.30\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.25 0.50 0.30 0.30\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.gridSize = 2;
        options.horizontalFlip = true;
        options.colorJitter = true;
        options.outputPath = dir.filePath(QStringLiteral("run"));

        const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(result.ok, qPrintable(result.error));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("gridSize")).toInt(), 2);
        QVERIFY(json.value(QStringLiteral("horizontalFlip")).toBool());
        QVERIFY(json.value(QStringLiteral("colorJitter")).toBool());
    }

    void detectionTrainerResumesTinyDetectorCheckpoint()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.60 0.40 0.30 0.20\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.60 0.40 0.30 0.20\n"));

        aitrain::DetectionTrainingOptions firstOptions;
        firstOptions.epochs = 1;
        firstOptions.batchSize = 1;
        firstOptions.imageSize = QSize(32, 32);
        firstOptions.outputPath = dir.filePath(QStringLiteral("run1"));
        const aitrain::DetectionTrainingResult first = aitrain::trainDetectionBaseline(root, firstOptions);
        QVERIFY2(first.ok, qPrintable(first.error));
        QCOMPARE(first.steps, 1);

        aitrain::DetectionTrainingOptions resumeOptions = firstOptions;
        resumeOptions.outputPath = dir.filePath(QStringLiteral("run2"));
        resumeOptions.resumeCheckpointPath = first.checkpointPath;
        const aitrain::DetectionTrainingResult resumed = aitrain::trainDetectionBaseline(root, resumeOptions);
        QVERIFY2(resumed.ok, qPrintable(resumed.error));
        QCOMPARE(resumed.steps, 2);

        QFile checkpoint(resumed.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("resumeFrom")).toString(), first.checkpointPath);
    }

    void detectionExportWritesTinyDetectorJson()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPath = dir.filePath(QStringLiteral("exports/model.aitrain-export.json"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(training.checkpointPath, exportPath);
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("tiny_detector_json"));
        QVERIFY(QFileInfo::exists(exported.exportPath));

        QFile file(exported.exportPath);
        QVERIFY(file.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(file.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("format")).toString(), QStringLiteral("tiny_detector_json"));
        QCOMPARE(json.value(QStringLiteral("note")).toString(), QStringLiteral("Scaffold export for AITrain tiny detector. This is not ONNX."));
        QCOMPARE(json.value(QStringLiteral("trainingBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QVERIFY(json.value(QStringLiteral("scaffold")).toBool());
        QVERIFY(!json.value(QStringLiteral("objectnessWeights")).toArray().isEmpty());
        QVERIFY(!json.value(QStringLiteral("classLogits")).toArray().isEmpty());
        QVERIFY(json.value(QStringLiteral("priorBox")).isObject());

        QString error;
        aitrain::DetectionBaselineCheckpoint loadedExport;
        QVERIFY2(aitrain::loadDetectionBaselineCheckpoint(exported.exportPath, &loadedExport, &error), qPrintable(error));
        QCOMPARE(loadedExport.trainingBackend, QStringLiteral("tiny_linear_detector"));
        QVERIFY(loadedExport.scaffold);
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(
            loadedExport,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(predictions.size(), 1);
    }

    void detectionExportWritesTinyDetectorOnnx()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPath = dir.filePath(QStringLiteral("exports/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            exportPath,
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("onnx"));
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));
        QCOMPARE(exported.config.value(QStringLiteral("format")).toString(), QStringLiteral("onnx"));
        QCOMPARE(exported.config.value(QStringLiteral("trainingBackend")).toString(), QStringLiteral("tiny_linear_detector"));
        QVERIFY(exported.config.value(QStringLiteral("scaffold")).toBool());
        QCOMPARE(exported.config.value(QStringLiteral("classNames")).toArray().first().toString(), QStringLiteral("item"));
        QCOMPARE(exported.config.value(QStringLiteral("input")).toObject().value(QStringLiteral("name")).toString(), QStringLiteral("features"));
        QCOMPARE(exported.config.value(QStringLiteral("outputs")).toArray().size(), 3);

        QFile file(exported.exportPath);
        QVERIFY(file.open(QIODevice::ReadOnly));
        const QByteArray bytes = file.readAll();
        QVERIFY(bytes.size() > 128);
        QVERIFY(bytes.contains("AITrain Studio"));
        QVERIFY(bytes.contains("Gemm"));
        QVERIFY(bytes.contains("Softmax"));
        QVERIFY(bytes.contains("objectness"));
        QVERIFY(bytes.contains("class_probabilities"));
        QVERIFY(bytes.contains("boxes"));
    }

    void detectionExportWritesNcnnParamBinWithConfiguredConverter()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString converterPath = dir.filePath(
#ifdef Q_OS_WIN
            QStringLiteral("onnx2ncnn.cmd")
#else
            QStringLiteral("onnx2ncnn")
#endif
        );
#ifdef Q_OS_WIN
        writeTextFile(converterPath,
            QStringLiteral("@echo off\r\n"
                           "echo fake ncnn param from %~1> \"%~2\"\r\n"
                           "echo fake ncnn bin from %~1> \"%~3\"\r\n"
                           "exit /b 0\r\n"));
#else
        writeTextFile(converterPath,
            QStringLiteral("#!/bin/sh\n"
                           "printf 'fake ncnn param from %s\\n' \"$1\" > \"$2\"\n"
                           "printf 'fake ncnn bin from %s\\n' \"$1\" > \"$3\"\n"));
        QFile::setPermissions(converterPath, QFile::permissions(converterPath)
            | QFileDevice::ExeOwner | QFileDevice::ExeUser | QFileDevice::ExeGroup | QFileDevice::ExeOther);
#endif
        ScopedEnvVar converterEnv("AITRAIN_NCNN_ONNX2NCNN", QFile::encodeName(converterPath));

        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));

        aitrain::DetectionTrainingOptions options;
        options.epochs = 1;
        options.batchSize = 1;
        options.imageSize = QSize(32, 32);
        options.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, options);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPrefix = dir.filePath(QStringLiteral("exports/mobile-model"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            exportPrefix,
            QStringLiteral("ncnn"));
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("ncnn"));
        QCOMPARE(exported.exportPath, QDir(dir.filePath(QStringLiteral("exports"))).filePath(QStringLiteral("mobile-model.param")));
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));

        const QJsonObject ncnn = exported.config.value(QStringLiteral("ncnn")).toObject();
        const QString binPath = ncnn.value(QStringLiteral("binPath")).toString();
        QCOMPARE(exported.config.value(QStringLiteral("format")).toString(), QStringLiteral("ncnn"));
        QCOMPARE(ncnn.value(QStringLiteral("paramPath")).toString(), exported.exportPath);
        QVERIFY(QFileInfo::exists(binPath));
        QCOMPARE(QFileInfo(binPath).fileName(), QStringLiteral("mobile-model.bin"));

        QFile paramFile(exported.exportPath);
        QVERIFY(paramFile.open(QIODevice::ReadOnly));
        QVERIFY(paramFile.readAll().contains("fake ncnn param"));
    }

    void detectionNcnnExportReportsMissingConverter()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const ScopedEnvVar converterEnv("AITRAIN_NCNN_ONNX2NCNN", QFile::encodeName(dir.filePath(QStringLiteral("missing-onnx2ncnn.exe"))));
        const QString sourceOnnx = dir.filePath(QStringLiteral("source.onnx"));
        writeTextFile(sourceOnnx, QStringLiteral("fake onnx\n"));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            sourceOnnx,
            dir.filePath(QStringLiteral("exports/model.param")),
            QStringLiteral("ncnn"));
        QVERIFY(!exported.ok);
        QCOMPARE(exported.format, QStringLiteral("ncnn"));
        QVERIFY(exported.error.contains(QStringLiteral("NCNN")));
        QVERIFY(exported.error.contains(QStringLiteral("not found")));
    }

    void detectionTensorRtBackendStatusIsExplicit()
    {
        const aitrain::TensorRtBackendStatus status = aitrain::tensorRtBackendStatus();
        QVERIFY(!status.message.isEmpty());
        QVERIFY(status.status == QStringLiteral("sdk_missing")
            || status.status == QStringLiteral("backend_not_implemented")
            || status.status == QStringLiteral("backend_available"));
        QCOMPARE(status.toJson().value(QStringLiteral("inferenceAvailable")).toBool(), status.inferenceAvailable);

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            QStringLiteral("missing.aitrain"),
            QStringLiteral("model.engine"),
            QStringLiteral("tensorrt"));
        QVERIFY(!exported.ok);
        QCOMPARE(exported.format, QStringLiteral("tensorrt"));
        if (status.exportAvailable) {
            QVERIFY(exported.error.contains(QStringLiteral("checkpoint"))
                || exported.error.contains(QStringLiteral("Checkpoint")));
        } else {
            QVERIFY(exported.error.contains(QStringLiteral("TensorRT")));
            QVERIFY(exported.error.contains(QStringLiteral("not available")));
            QVERIFY(exported.error.contains(status.message));
        }
    }

    void detectionTensorRtInferenceReportsClearMissingInput()
    {
        QString error;
        aitrain::DetectionInferenceOptions options;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionTensorRt(
            QStringLiteral("model.engine"),
            QStringLiteral("image.png"),
            options,
            &error);
        QVERIFY(predictions.isEmpty());
        QVERIFY(error.contains(QStringLiteral("TensorRT")));
        if (aitrain::isTensorRtInferenceAvailable()) {
            QVERIFY(error.contains(QStringLiteral("engine")));
        } else {
            QVERIFY(error.contains(QStringLiteral("not available")));
            QVERIFY(error.contains(aitrain::tensorRtBackendStatus().message));
        }
    }

    void detectionOnnxRuntimeReportsUnavailableWhenSdkMissing()
    {
        if (aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is enabled in this build.");
        }

        QString error;
        aitrain::DetectionInferenceOptions options;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(
            QStringLiteral("model.onnx"),
            QStringLiteral("image.png"),
            options,
            &error);
        QVERIFY(predictions.isEmpty());
        QVERIFY(error.contains(QStringLiteral("ONNX Runtime")));
        QVERIFY(error.contains(QStringLiteral("not enabled")));
    }

    void detectionOnnxRuntimeRunsExportedTinyDetector()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::DetectionTrainingOptions trainingOptions;
        trainingOptions.epochs = 1;
        trainingOptions.batchSize = 1;
        trainingOptions.imageSize = QSize(32, 32);
        trainingOptions.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, trainingOptions);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString exportPath = dir.filePath(QStringLiteral("exports/model.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            exportPath,
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));

        QString error;
        aitrain::DetectionInferenceOptions inferenceOptions;
        inferenceOptions.confidenceThreshold = 0.0;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(
            exported.exportPath,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QVERIFY(!predictions.isEmpty());
        QVERIFY(predictions.first().confidence >= 0.0);
        QVERIFY(predictions.first().objectness >= 0.0);
        QCOMPARE(predictions.first().className, QStringLiteral("item"));
        QVERIFY(predictions.first().box.width > 0.0);
        QVERIFY(predictions.first().box.height > 0.0);
    }

    void detectionOnnxRuntimeMatchesCheckpointPrediction()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::DetectionTrainingOptions trainingOptions;
        trainingOptions.epochs = 2;
        trainingOptions.batchSize = 1;
        trainingOptions.imageSize = QSize(32, 32);
        trainingOptions.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, trainingOptions);
        QVERIFY2(training.ok, qPrintable(training.error));

        const QString imagePath = QDir(root).filePath(QStringLiteral("images/val/a.png"));
        QString error;
        aitrain::DetectionBaselineCheckpoint checkpoint;
        QVERIFY2(aitrain::loadDetectionBaselineCheckpoint(training.checkpointPath, &checkpoint, &error), qPrintable(error));

        aitrain::DetectionInferenceOptions inferenceOptions;
        inferenceOptions.confidenceThreshold = 0.0;
        inferenceOptions.maxDetections = 1;
        const QVector<aitrain::DetectionPrediction> checkpointPredictions = aitrain::predictDetectionBaseline(
            checkpoint,
            imagePath,
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(checkpointPredictions.size(), 1);

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            dir.filePath(QStringLiteral("exports/model.onnx")),
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));

        const QVector<aitrain::DetectionPrediction> onnxPredictions = aitrain::predictDetectionOnnxRuntime(
            exported.exportPath,
            imagePath,
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(onnxPredictions.size(), 1);

        const aitrain::DetectionPrediction& checkpointPrediction = checkpointPredictions.first();
        const aitrain::DetectionPrediction& onnxPrediction = onnxPredictions.first();
        QCOMPARE(onnxPrediction.box.classId, checkpointPrediction.box.classId);
        QCOMPARE(onnxPrediction.className, checkpointPrediction.className);
        QVERIFY(qAbs(onnxPrediction.objectness - checkpointPrediction.objectness) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.confidence - checkpointPrediction.confidence) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.xCenter - checkpointPrediction.box.xCenter) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.yCenter - checkpointPrediction.box.yCenter) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.width - checkpointPrediction.box.width) <= 1.0e-5);
        QVERIFY(qAbs(onnxPrediction.box.height - checkpointPrediction.box.height) <= 1.0e-5);
    }

    void detectionOnnxRuntimeRunsUltralyticsYoloDetectionSmokeModel()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString smokeRoot = phase9RealSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("Phase 9 real Ultralytics smoke artifacts are not available.");
        }
        QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("out/ultralytics_runs/phase9-real-smoke/weights/best.onnx"));
        QString imagePath = QDir(smokeRoot).filePath(QStringLiteral("dataset/images/val/a.png"));
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_detect/ultralytics_runs/acceptance-yolo-detect/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("generated/yolo_detect/images/val/b.png"));
        }
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_detect/ultralytics_runs/cpu-yolo-detect/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("yolo_detect/images/val/val_00.png"));
        }
        if (!QFileInfo::exists(onnxPath) || !QFileInfo::exists(imagePath)) {
            QSKIP("Phase 9 real Ultralytics smoke artifacts are not available.");
        }

        QString error;
        aitrain::DetectionInferenceOptions inferenceOptions;
        inferenceOptions.confidenceThreshold = 0.0;
        inferenceOptions.maxDetections = 10;
        const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionOnnxRuntime(
            onnxPath,
            imagePath,
            inferenceOptions,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QVERIFY(!predictions.isEmpty());
        for (const aitrain::DetectionPrediction& prediction : predictions) {
            QVERIFY(prediction.confidence >= 0.0);
            QVERIFY(prediction.box.classId >= 0);
            QVERIFY(prediction.box.xCenter >= 0.0 && prediction.box.xCenter <= 1.0);
            QVERIFY(prediction.box.yCenter >= 0.0 && prediction.box.yCenter <= 1.0);
            QVERIFY(prediction.box.width > 0.0 && prediction.box.width <= 1.0);
            QVERIFY(prediction.box.height > 0.0 && prediction.box.height <= 1.0);
        }
    }

    void detectionExportCopiesUltralyticsYoloOnnxWithSidecar()
    {
        const QString smokeRoot = phase9RealSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("Phase 9 real Ultralytics smoke ONNX artifact is not available.");
        }
        const QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("out/ultralytics_runs/phase9-real-smoke/weights/best.onnx"));
        if (!QFileInfo::exists(onnxPath)) {
            QSKIP("Phase 9 real Ultralytics smoke ONNX artifact is not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString exportPath = dir.filePath(QStringLiteral("exports/copied-yolo.onnx"));
        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            onnxPath,
            exportPath,
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));
        QCOMPARE(exported.format, QStringLiteral("onnx"));
        QVERIFY(QFileInfo::exists(exported.exportPath));
        QVERIFY(QFileInfo::exists(exported.reportPath));
        QCOMPARE(exported.config.value(QStringLiteral("backend")).toString(), QStringLiteral("ultralytics_yolo_detect"));
        QCOMPARE(exported.config.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_detection"));
        QVERIFY(!exported.config.value(QStringLiteral("scaffold")).toBool(true));
        QCOMPARE(exported.config.value(QStringLiteral("classNames")).toArray().first().toString(), QStringLiteral("item"));
        QCOMPARE(exported.config.value(QStringLiteral("postprocess")).toObject().value(QStringLiteral("decoder")).toString(), QStringLiteral("yolo_v8_detection"));
    }

    void segmentationOnnxRuntimeRunsUltralyticsYoloSegmentationSmokeModel()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString smokeRoot = phase11SegSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("YOLO segmentation smoke ONNX artifact is not available.");
        }
        QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("out/ultralytics_runs/phase11-seg-smoke/weights/best.onnx"));
        QString imagePath = QDir(smokeRoot).filePath(QStringLiteral("dataset/images/val/a.png"));
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/aitrain-yolo-segment/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/acceptance-yolo-segment/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("generated/yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(onnxPath)) {
            onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/cpu-yolo-segment/weights/best.onnx"));
            imagePath = QDir(smokeRoot).filePath(QStringLiteral("yolo_segment/images/val/val_00.png"));
        }
        if (!QFileInfo::exists(onnxPath) || !QFileInfo::exists(imagePath)) {
            QSKIP("YOLO segmentation smoke ONNX artifact is not available.");
        }

        QCOMPARE(aitrain::inferOnnxModelFamily(onnxPath), QStringLiteral("yolo_segmentation"));
        QString error;
        aitrain::DetectionInferenceOptions options;
        options.confidenceThreshold = 0.0;
        options.maxDetections = 5;
        const QVector<aitrain::SegmentationPrediction> predictions = aitrain::predictSegmentationOnnxRuntime(
            onnxPath,
            imagePath,
            options,
            &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QVERIFY(!predictions.isEmpty());
        const QJsonObject predictionJson = aitrain::segmentationPredictionToJson(predictions.first());
        QCOMPARE(predictionJson.value(QStringLiteral("taskType")).toString(), QStringLiteral("segmentation"));
        QVERIFY(predictionJson.value(QStringLiteral("box")).toObject().value(QStringLiteral("xCenter")).toDouble() >= 0.0);
        QVERIFY(predictionJson.value(QStringLiteral("maskArea")).toDouble() >= 0.0);
        const QImage overlay = aitrain::renderSegmentationPredictions(imagePath, predictions, &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
    }

    void ocrRecOnnxRuntimeDecodesSmokeModel()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString smokeRoot = phase14OcrSmokeRoot();
        if (smokeRoot.isEmpty()) {
            QSKIP("OCR Rec smoke ONNX artifact is not available.");
        }
        const QString onnxPath = QDir(smokeRoot).filePath(QStringLiteral("runs/paddleocr_rec/paddleocr_rec_ctc.onnx"));
        const QString imagePath = QFileInfo::exists(QDir(smokeRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png")))
            ? QDir(smokeRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png"))
            : QDir(smokeRoot).filePath(QStringLiteral("dataset/images/a.png"));
        if (!QFileInfo::exists(onnxPath) || !QFileInfo::exists(imagePath)) {
            QSKIP("OCR Rec smoke ONNX artifact is not available.");
        }

        QCOMPARE(aitrain::inferOnnxModelFamily(onnxPath), QStringLiteral("ocr_recognition"));
        QString error;
        const aitrain::OcrRecPrediction prediction = aitrain::predictOcrRecOnnxRuntime(onnxPath, imagePath, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        const QJsonObject predictionJson = aitrain::ocrRecPredictionToJson(prediction);
        QCOMPARE(predictionJson.value(QStringLiteral("taskType")).toString(), QStringLiteral("ocr_recognition"));
        QVERIFY(predictionJson.value(QStringLiteral("tokens")).toArray().size() > 0);
        QVERIFY(prediction.confidence >= 0.0);
        const QImage overlay = aitrain::renderOcrRecPrediction(imagePath, prediction, &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
    }

    void workerRunsOnnxInferenceEndToEnd()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::DetectionTrainingOptions trainingOptions;
        trainingOptions.epochs = 1;
        trainingOptions.batchSize = 1;
        trainingOptions.imageSize = QSize(32, 32);
        trainingOptions.outputPath = dir.filePath(QStringLiteral("run"));
        const aitrain::DetectionTrainingResult training = aitrain::trainDetectionBaseline(root, trainingOptions);
        QVERIFY2(training.ok, qPrintable(training.error));

        const aitrain::DetectionExportResult exported = aitrain::exportDetectionCheckpoint(
            training.checkpointPath,
            dir.filePath(QStringLiteral("exports/model.onnx")),
            QStringLiteral("onnx"));
        QVERIFY2(exported.ok, qPrintable(exported.error));

        const QString outputPath = dir.filePath(QStringLiteral("worker-inference"));
        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.requestInference(
            workerExecutablePath(),
            exported.exportPath,
            QDir(root).filePath(QStringLiteral("images/val/a.png")),
            outputPath,
            &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringLiteral("%1 messages=%2 outputExists=%3")
            .arg(finishedMessage)
            .arg(messages.size())
            .arg(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"))))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        QVERIFY(!messages.isEmpty());
        QCOMPARE(messages.last().first, QStringLiteral("completed"));

        bool sawInferenceResult = false;
        bool sawPredictionsArtifact = false;
        bool sawOverlayArtifact = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("inferenceResult")) {
                sawInferenceResult = true;
                QCOMPARE(message.second.value(QStringLiteral("predictionCount")).toInt(), 1);
                QVERIFY(message.second.value(QStringLiteral("elapsedMs")).toInt() >= 0);
            } else if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawPredictionsArtifact = sawPredictionsArtifact || kind == QStringLiteral("inference_predictions");
                sawOverlayArtifact = sawOverlayArtifact || kind == QStringLiteral("inference_overlay");
            }
        }
        QVERIFY(sawInferenceResult);
        QVERIFY(sawPredictionsArtifact);
        QVERIFY(sawOverlayArtifact);

        const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
        const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
        QVERIFY(QFileInfo::exists(predictionsPath));
        QVERIFY(QFileInfo::exists(overlayPath));
        QFile predictionsFile(predictionsPath);
        QVERIFY(predictionsFile.open(QIODevice::ReadOnly));
        const QJsonObject predictionsJson = QJsonDocument::fromJson(predictionsFile.readAll()).object();
        QCOMPARE(predictionsJson.value(QStringLiteral("runtime")).toString(), QStringLiteral("onnxruntime"));
        QCOMPARE(predictionsJson.value(QStringLiteral("predictions")).toArray().size(), 1);
        QCOMPARE(predictionsJson.value(QStringLiteral("predictions")).toArray().first().toObject().value(QStringLiteral("className")).toString(), QStringLiteral("item"));
    }

    void workerRoutesSegmentationAndOcrOnnxInferenceEndToEnd()
    {
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            QSKIP("ONNX Runtime SDK is not enabled in this build.");
        }

        const QString segRoot = phase11SegSmokeRoot();
        const QString ocrRoot = phase14OcrSmokeRoot();
        if (segRoot.isEmpty() || ocrRoot.isEmpty()) {
            QSKIP("Segmentation/OCR ONNX smoke artifacts are not available.");
        }

        QString segOnnx = QDir(segRoot).filePath(QStringLiteral("out/ultralytics_runs/phase11-seg-smoke/weights/best.onnx"));
        QString segImage = QDir(segRoot).filePath(QStringLiteral("dataset/images/val/a.png"));
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/aitrain-yolo-segment/weights/best.onnx"));
            segImage = QDir(segRoot).filePath(QStringLiteral("yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/acceptance-yolo-segment/weights/best.onnx"));
            segImage = QDir(segRoot).filePath(QStringLiteral("generated/yolo_segment/images/val/b.png"));
        }
        if (!QFileInfo::exists(segOnnx)) {
            segOnnx = QDir(segRoot).filePath(QStringLiteral("runs/yolo_segment/ultralytics_runs/cpu-yolo-segment/weights/best.onnx"));
            segImage = QDir(segRoot).filePath(QStringLiteral("yolo_segment/images/val/val_00.png"));
        }
        const QString ocrOnnx = QDir(ocrRoot).filePath(QStringLiteral("runs/paddleocr_rec/paddleocr_rec_ctc.onnx"));
        const QString ocrImage = QFileInfo::exists(QDir(ocrRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png")))
            ? QDir(ocrRoot).filePath(QStringLiteral("paddleocr_rec/images/a.png"))
            : QDir(ocrRoot).filePath(QStringLiteral("dataset/images/a.png"));
        if (!QFileInfo::exists(segOnnx) || !QFileInfo::exists(segImage)
            || !QFileInfo::exists(ocrOnnx) || !QFileInfo::exists(ocrImage)) {
            QSKIP("Segmentation/OCR ONNX smoke artifacts are not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        auto runInference = [this, &dir](const QString& modelPath, const QString& imagePath, const QString& outputName, const QString& expectedTaskType) {
            const QString outputPath = dir.filePath(outputName);
            WorkerClient client;
            QVector<QPair<QString, QJsonObject>> messages;
            QStringList logs;
            bool finished = false;
            bool ok = false;
            QString finishedMessage;
            connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
                messages.append(qMakePair(type, payload));
            });
            connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
                logs.append(line);
            });
            connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
                finished = true;
                ok = result;
                finishedMessage = message;
            });

            QString error;
            QVERIFY2(client.requestInference(workerExecutablePath(), modelPath, imagePath, outputPath, &error), qPrintable(error));
            QTRY_VERIFY2_WITH_TIMEOUT(
                finished,
                qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
                20000);
            QVERIFY2(ok, qPrintable(finishedMessage));
            QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

            bool sawInferenceResult = false;
            bool sawPredictionsArtifact = false;
            bool sawOverlayArtifact = false;
            for (const auto& message : messages) {
                if (message.first == QStringLiteral("inferenceResult")) {
                    sawInferenceResult = true;
                    QCOMPARE(message.second.value(QStringLiteral("taskType")).toString(), expectedTaskType);
                    QVERIFY(message.second.value(QStringLiteral("predictionCount")).toInt() > 0);
                } else if (message.first == QStringLiteral("artifact")) {
                    const QString kind = message.second.value(QStringLiteral("kind")).toString();
                    sawPredictionsArtifact = sawPredictionsArtifact || kind == QStringLiteral("inference_predictions");
                    sawOverlayArtifact = sawOverlayArtifact || kind == QStringLiteral("inference_overlay");
                }
            }
            QVERIFY(sawInferenceResult);
            QVERIFY(sawPredictionsArtifact);
            QVERIFY(sawOverlayArtifact);

            const QString predictionsPath = QDir(outputPath).filePath(QStringLiteral("inference_predictions.json"));
            const QString overlayPath = QDir(outputPath).filePath(QStringLiteral("inference_overlay.png"));
            QVERIFY(QFileInfo::exists(predictionsPath));
            QVERIFY(QFileInfo::exists(overlayPath));
            QFile predictionsFile(predictionsPath);
            QVERIFY(predictionsFile.open(QIODevice::ReadOnly));
            const QJsonObject predictionsJson = QJsonDocument::fromJson(predictionsFile.readAll()).object();
            QCOMPARE(predictionsJson.value(QStringLiteral("taskType")).toString(), expectedTaskType);
            QVERIFY(predictionsJson.value(QStringLiteral("predictions")).toArray().size() > 0);
        };

        runInference(segOnnx, segImage, QStringLiteral("worker-seg-onnx-inference"), QStringLiteral("segmentation"));
        runInference(ocrOnnx, ocrImage, QStringLiteral("worker-ocr-onnx-inference"), QStringLiteral("ocr_recognition"));
    }

    void yoloSegmentationDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [part]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.1 0.1 0.8 0.1 0.8 0.8\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("0 0.1 0.1 0.8 0.1 0.8 0.8\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QVERIFY(!valid.previewSamples.isEmpty());
        QVERIFY(valid.previewSamples.first().contains(QStringLiteral("polygon=")));

        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("0 0.1 0.1 0.2\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY(!invalid.ok);
    }

    void ocrRecDatasetLoadsDictionaryAndLabels()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);

        QString error;
        aitrain::OcrRecDataset dataset;
        QVERIFY2(dataset.load(root, QString(), QString(), 8, &error), qPrintable(error));
        QCOMPARE(dataset.size(), 2);
        QCOMPARE(dataset.dictionary().characters.size(), 4);
        QCOMPARE(dataset.samples().first().label, QStringLiteral("ab12"));
        QCOMPARE(dataset.samples().first().encodedLabel.size(), 4);
        QCOMPARE(dataset.samples().first().encodedLabel.at(0), 1);
        QCOMPARE(dataset.samples().first().encodedLabel.at(1), 2);
        QCOMPARE(dataset.samples().first().encodedLabel.at(2), 3);
        QCOMPARE(dataset.samples().first().encodedLabel.at(3), 4);
        QCOMPARE(aitrain::decodeOcrText(dataset.samples().first().encodedLabel, dataset.dictionary()), QStringLiteral("ab12"));
    }

    void ocrRecDatasetRejectsUnknownDictionaryCharacter()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.png\taz\n"));

        QString error;
        aitrain::OcrRecDataset dataset;
        QVERIFY(!dataset.load(root, QString(), QString(), 8, &error));
        QVERIFY(error.contains(QStringLiteral("dictionary")));
    }

    void ocrRecDataLoaderBuildsPaddedBatch()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\nb\n"));

        QDir().mkpath(QDir(root).filePath(QStringLiteral("images")));
        QImage wideImage(16, 8, QImage::Format_RGB888);
        wideImage.fill(Qt::black);
        QVERIFY(wideImage.save(QDir(root).filePath(QStringLiteral("images/a.png"))));
        QImage tallImage(8, 16, QImage::Format_RGB888);
        tallImage.fill(Qt::black);
        QVERIFY(tallImage.save(QDir(root).filePath(QStringLiteral("images/b.png"))));
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.png\tab\nimages/b.png\tba\n"));

        QString error;
        aitrain::OcrRecDataset dataset;
        QVERIFY2(dataset.load(root, QString(), QString(), 8, &error), qPrintable(error));

        aitrain::OcrRecDataLoader loader(dataset, 2, QSize(32, 16));
        aitrain::OcrRecBatch batch;
        QVERIFY2(loader.next(&batch, &error), qPrintable(error));
        QCOMPARE(batch.images.size(), 2);
        QCOMPARE(batch.labels.size(), 2);
        QCOMPARE(batch.labelLengths.size(), 2);
        QCOMPARE(batch.labelLengths.at(0), 2);
        QCOMPARE(batch.labelLengths.at(1), 2);
        QCOMPARE(batch.texts.first(), QStringLiteral("ab"));
        QCOMPARE(batch.images.first().size(), QSize(32, 16));
        QCOMPARE(qRed(batch.images.first().pixel(31, 15)), 0);
        QCOMPARE(qRed(batch.images.at(1).pixel(0, 15)), 0);
        QCOMPARE(qRed(batch.images.at(1).pixel(31, 15)), 255);
        QVERIFY(!loader.hasNext());
    }

    void ocrRecTrainerWritesScaffoldCheckpointAndPreview()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);

        aitrain::OcrRecTrainingOptions options;
        options.epochs = 3;
        options.batchSize = 1;
        options.imageSize = QSize(32, 16);
        options.learningRate = 0.1;
        options.maxTextLength = 8;
        options.outputPath = dir.filePath(QStringLiteral("run"));

        double firstLoss = -1.0;
        double lastLoss = -1.0;
        int callbackCount = 0;
        const aitrain::OcrRecTrainingResult result = aitrain::trainOcrRecBaseline(
            root,
            options,
            [&firstLoss, &lastLoss, &callbackCount](const aitrain::OcrRecTrainingMetrics& metrics) {
                ++callbackCount;
                if (firstLoss < 0.0) {
                    firstLoss = metrics.loss;
                }
                lastLoss = metrics.loss;
                return metrics.accuracy > 0.0;
            });

        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.steps, 6);
        QCOMPARE(callbackCount, 6);
        QVERIFY(firstLoss >= 0.0);
        QVERIFY(lastLoss >= 0.0);
        QVERIFY(lastLoss < firstLoss);
        QCOMPARE(result.accuracy, 1.0);
        QCOMPARE(result.editDistance, 0.0);
        QVERIFY(QFileInfo::exists(result.checkpointPath));
        QVERIFY(QFileInfo::exists(result.previewPath));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("type")).toString(), QStringLiteral("tiny_ocr_recognition_scaffold"));
        QVERIFY(json.value(QStringLiteral("note")).toString().contains(QStringLiteral("Scaffold")));
        QCOMPARE(json.value(QStringLiteral("steps")).toInt(), 6);
        QCOMPARE(json.value(QStringLiteral("accuracy")).toDouble(), 1.0);
        QCOMPARE(json.value(QStringLiteral("editDistance")).toDouble(), 0.0);
        QCOMPARE(json.value(QStringLiteral("modelHead")).toString(), QStringLiteral("label_echo_ctc_scaffold"));

        QFile preview(result.previewPath);
        QVERIFY(preview.open(QIODevice::ReadOnly));
        const QJsonObject previewJson = QJsonDocument::fromJson(preview.readAll()).object();
        QCOMPARE(previewJson.value(QStringLiteral("label")).toString(), QStringLiteral("ab12"));
        QCOMPARE(previewJson.value(QStringLiteral("prediction")).toString(), QStringLiteral("ab12"));
    }

    void segmentationDatasetLoadsMasksAndOverlay()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinySegmentationDataset(root);

        QString error;
        aitrain::SegmentationDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));
        QCOMPARE(dataset.size(), 1);
        QCOMPARE(dataset.info().classCount, 1);
        QCOMPARE(dataset.samples().first().polygons.size(), 1);
        QCOMPARE(dataset.samples().first().polygons.first().points.size(), 4);

        const QImage mask = aitrain::polygonToMask(dataset.samples().first().polygons.first().points, QSize(8, 8));
        QVERIFY(!mask.isNull());
        QCOMPARE(mask.size(), QSize(8, 8));
        QVERIFY(qAlpha(mask.pixel(4, 4)) > 0);
        QCOMPARE(qAlpha(mask.pixel(0, 0)), 0);

        const QImage overlay = aitrain::renderSegmentationOverlay(
            dataset.samples().first().imagePath,
            dataset.samples().first().polygons,
            &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
        QCOMPARE(overlay.size(), QSize(8, 8));
    }

    void segmentationDataLoaderBuildsAlignedBatchMasks()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [part, edge]\n"));

        QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train")));
        QImage wideImage(16, 8, QImage::Format_RGB888);
        wideImage.fill(Qt::white);
        QVERIFY(wideImage.save(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
        QImage tallImage(8, 16, QImage::Format_RGB888);
        tallImage.fill(Qt::white);
        QVERIFY(tallImage.save(QDir(root).filePath(QStringLiteral("images/train/b.png"))));

        writeTextFile(
            QDir(root).filePath(QStringLiteral("labels/train/a.txt")),
            QStringLiteral("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
                           "1 0.05 0.05 0.15 0.05 0.15 0.20 0.05 0.20\n"));
        writeTextFile(
            QDir(root).filePath(QStringLiteral("labels/train/b.txt")),
            QStringLiteral("1 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"));

        QString error;
        aitrain::SegmentationDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));

        aitrain::SegmentationDataLoader loader(dataset, 2, QSize(32, 32));
        QVERIFY(loader.hasNext());
        aitrain::SegmentationBatch batch;
        QVERIFY2(loader.next(&batch, &error), qPrintable(error));
        QCOMPARE(batch.images.size(), 2);
        QCOMPARE(batch.masks.size(), 2);
        QCOMPARE(batch.polygons.size(), 2);
        QCOMPARE(batch.imagePaths.size(), 2);
        QCOMPARE(batch.images.first().size(), QSize(32, 32));
        QCOMPARE(batch.masks.first().size(), QSize(32, 32));

        QCOMPARE(batch.polygons.first().size(), 2);
        QCOMPARE(batch.polygons.first().first().classId, 0);
        QVERIFY(qAbs(batch.polygons.first().first().points.first().x() - 0.25) < 0.001);
        QVERIFY(qAbs(batch.polygons.first().first().points.first().y() - 0.375) < 0.001);
        QCOMPARE(qAlpha(batch.masks.first().pixel(16, 4)), 0);
        QCOMPARE(qRed(batch.masks.first().pixel(16, 16)), 1);
        QVERIFY(qAlpha(batch.masks.first().pixel(16, 16)) > 0);
        QCOMPARE(qRed(batch.masks.first().pixel(3, 10)), 2);

        QCOMPARE(batch.polygons.at(1).first().classId, 1);
        QCOMPARE(qAlpha(batch.masks.at(1).pixel(4, 16)), 0);
        QCOMPARE(qRed(batch.masks.at(1).pixel(16, 16)), 2);
        QVERIFY(!loader.hasNext());
    }

    void segmentationDataLoaderRejectsInvalidPolygons()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [part]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.1 0.1 0.2 0.2 0.3 0.3\n"));

        QString error;
        aitrain::SegmentationDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));

        aitrain::SegmentationDataLoader loader(dataset, 1, QSize(16, 16));
        aitrain::SegmentationBatch batch;
        QVERIFY(!loader.next(&batch, &error));
        QVERIFY(error.contains(QStringLiteral("Invalid segmentation polygon")));
    }

    void segmentationTrainerWritesScaffoldCheckpointAndPreview()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinySegmentationDataset(root);

        aitrain::SegmentationTrainingOptions options;
        options.epochs = 3;
        options.batchSize = 1;
        options.imageSize = QSize(16, 16);
        options.learningRate = 0.1;
        options.outputPath = dir.filePath(QStringLiteral("run"));

        double firstLoss = -1.0;
        double lastLoss = -1.0;
        int callbackCount = 0;
        double lastMaskIou = 0.0;
        double lastMap50 = 0.0;
        const aitrain::SegmentationTrainingResult result = aitrain::trainSegmentationBaseline(
            root,
            options,
            [&firstLoss, &lastLoss, &callbackCount, &lastMaskIou, &lastMap50](const aitrain::SegmentationTrainingMetrics& metrics) {
                ++callbackCount;
                if (firstLoss < 0.0) {
                    firstLoss = metrics.loss;
                }
                lastLoss = metrics.loss;
                lastMaskIou = metrics.maskIou;
                lastMap50 = metrics.map50;
                return metrics.maskCoverage > 0.0;
            });

        QVERIFY2(result.ok, qPrintable(result.error));
        QCOMPARE(result.steps, 3);
        QCOMPARE(callbackCount, 3);
        QVERIFY(firstLoss >= 0.0);
        QVERIFY(lastLoss >= 0.0);
        QVERIFY(lastLoss < firstLoss);
        QVERIFY(lastMaskIou > 0.0);
        QVERIFY(lastMap50 > 0.0);
        QVERIFY(result.maskCoverage > 0.0);
        QVERIFY(result.maskIou > 0.0);
        QVERIFY(result.precision > 0.0);
        QVERIFY(result.recall > 0.0);
        QVERIFY(result.map50 > 0.0);
        QVERIFY(QFileInfo::exists(result.checkpointPath));
        QVERIFY(QFileInfo::exists(result.previewPath));
        QVERIFY(QFileInfo::exists(result.maskPreviewPath));

        QFile checkpoint(result.checkpointPath);
        QVERIFY(checkpoint.open(QIODevice::ReadOnly));
        const QJsonObject json = QJsonDocument::fromJson(checkpoint.readAll()).object();
        QCOMPARE(json.value(QStringLiteral("type")).toString(), QStringLiteral("tiny_mask_segmentation_scaffold"));
        QVERIFY(json.value(QStringLiteral("note")).toString().contains(QStringLiteral("Scaffold")));
        QCOMPARE(json.value(QStringLiteral("steps")).toInt(), 3);
        QVERIFY(json.value(QStringLiteral("maskCoverage")).toDouble() > 0.0);
        QVERIFY(json.value(QStringLiteral("maskIoU")).toDouble() > 0.0);
        QVERIFY(json.value(QStringLiteral("precision")).toDouble() > 0.0);
        QVERIFY(json.value(QStringLiteral("recall")).toDouble() > 0.0);
        QVERIFY(json.value(QStringLiteral("segmentationMap50")).toDouble() > 0.0);
        QCOMPARE(json.value(QStringLiteral("maskHead")).toString(), QStringLiteral("label_rasterization_scaffold"));
        QVERIFY(!json.value(QStringLiteral("previewPolygons")).toArray().isEmpty());
    }

    void workerRunsSegmentationTrainingScaffoldEndToEnd()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinySegmentationDataset(root);
        const QString outputPath = dir.filePath(QStringLiteral("worker-segmentation"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("seg-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("segmentation");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("epochs"), 2);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageSize"), 16);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawCheckpoint = false;
        bool sawPreview = false;
        bool sawMaskPreview = false;
        bool sawMaskLoss = false;
        bool sawMaskCoverage = false;
        bool sawMaskIou = false;
        bool sawPrecision = false;
        bool sawRecall = false;
        bool sawSegmentationMap50 = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawCheckpoint = sawCheckpoint || kind == QStringLiteral("checkpoint");
                sawPreview = sawPreview || kind == QStringLiteral("preview");
                sawMaskPreview = sawMaskPreview || kind == QStringLiteral("mask_preview");
            } else if (message.first == QStringLiteral("metric")) {
                const QString name = message.second.value(QStringLiteral("name")).toString();
                sawMaskLoss = sawMaskLoss || name == QStringLiteral("maskLoss");
                sawMaskCoverage = sawMaskCoverage || name == QStringLiteral("maskCoverage");
                sawMaskIou = sawMaskIou || name == QStringLiteral("maskIoU");
                sawPrecision = sawPrecision || name == QStringLiteral("precision");
                sawRecall = sawRecall || name == QStringLiteral("recall");
                sawSegmentationMap50 = sawSegmentationMap50 || name == QStringLiteral("segmentationMap50");
            }
        }
        QVERIFY(sawCheckpoint);
        QVERIFY(sawPreview);
        QVERIFY(sawMaskPreview);
        QVERIFY(sawMaskLoss);
        QVERIFY(sawMaskCoverage);
        QVERIFY(sawMaskIou);
        QVERIFY(sawPrecision);
        QVERIFY(sawRecall);
        QVERIFY(sawSegmentationMap50);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("checkpoint_latest.aitrain"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("preview_latest.png"))));
        QCOMPARE(messages.last().first, QStringLiteral("completed"));
    }

    void workerRunsPythonTrainerMockEndToEnd()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        const QString trainerScript = mockPythonTrainerScriptPath();
        QVERIFY2(!trainerScript.isEmpty(), "Mock Python trainer script is not available.");

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);
        const QString outputPath = dir.filePath(QStringLiteral("worker-python-mock"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("python-mock-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("python_mock"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("pythonTrainerScript"), trainerScript);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("mockStepsPerEpoch"), 2);
        request.parameters.insert(QStringLiteral("mockSleepMs"), 0);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool success, const QString& message) {
            finished = true;
            ok = success;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawLoss = false;
        bool sawCheckpoint = false;
        bool sawReport = false;
        bool sawPythonBackend = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("metric")) {
                sawLoss = sawLoss || message.second.value(QStringLiteral("name")).toString() == QStringLiteral("loss");
            } else if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawCheckpoint = sawCheckpoint || kind == QStringLiteral("checkpoint");
                sawReport = sawReport || kind == QStringLiteral("training_report");
                sawPythonBackend = sawPythonBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("python_mock");
            } else if (message.first == QStringLiteral("completed")) {
                sawPythonBackend = sawPythonBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("python_mock");
            }
        }
        QVERIFY(sawLoss);
        QVERIFY(sawCheckpoint);
        QVERIFY(sawReport);
        QVERIFY(sawPythonBackend);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("python_trainer_request.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("python_mock_checkpoint.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("python_mock_training_report.json"))));
    }

    void workerPropagatesPythonTrainerMockFailure()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        const QString trainerScript = mockPythonTrainerScriptPath();
        QVERIFY2(!trainerScript.isEmpty(), "Mock Python trainer script is not available.");

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("python-mock-fail-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = root;
        request.outputPath = dir.filePath(QStringLiteral("worker-python-mock-fail"));
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("python_mock"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("pythonTrainerScript"), trainerScript);
        request.parameters.insert(QStringLiteral("mockMode"), QStringLiteral("fail"));

        WorkerClient client;
        bool finished = false;
        bool ok = true;
        QString finishedMessage;
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool success, const QString& message) {
            finished = true;
            ok = success;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY(!ok);
        QVERIFY(finishedMessage.contains(QStringLiteral("Mock Python trainer failure requested")));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);
    }

    void workerRunsUltralyticsYoloDetectionAdapterWithFakeOfficialPackage()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(root);
        const QString fakePackageRoot = dir.filePath(QStringLiteral("fake-pythonpath"));
        writeFakeUltralyticsPackage(fakePackageRoot);
        const QString outputPath = dir.filePath(QStringLiteral("worker-ultralytics-yolo"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("ultralytics-yolo-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        request.taskType = QStringLiteral("detection");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("pythonPathPrepend"), fakePackageRoot);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageSize"), 64);
        request.parameters.insert(QStringLiteral("device"), QStringLiteral("cpu"));
        request.parameters.insert(QStringLiteral("model"), QStringLiteral("fake-yolo.pt"));
        request.parameters.insert(QStringLiteral("runName"), QStringLiteral("fake-run"));
        request.parameters.insert(QStringLiteral("exportOnnx"), true);
        request.parameters.insert(QStringLiteral("compactEvents"), true);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawCompletedOnnx = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("ultralytics_yolo_detect");
            if (message.first == QStringLiteral("completed")) {
                sawCompletedOnnx = !message.second.value(QStringLiteral("onnxPath")).toString().isEmpty();
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawCompletedOnnx);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("aitrain_yolo_data.yaml"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("ultralytics_training_report.json"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("ultralytics_runs/fake-run/weights/best.pt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("ultralytics_runs/fake-run/weights/best.onnx"))));
    }

    void workerRunsPaddleOcrOfficialAdapterPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetPath = QDir(dir.path()).filePath(QStringLiteral("ocr-rec"));
        writeTinyOcrRecDataset(datasetPath);
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-official-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_recognition");
        request.datasetPath = datasetPath;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_rec_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageWidth"), 96);
        request.parameters.insert(QStringLiteral("imageHeight"), 32);
        request.parameters.insert(QStringLiteral("maxTextLength"), 8);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawPrepareOnly = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_rec_official");
            if (message.first == QStringLiteral("completed")) {
                sawPrepareOnly = message.second.value(QStringLiteral("mode")).toString() == QStringLiteral("prepareOnly");
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawPrepareOnly);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("aitrain_ppocrv4_rec.yml"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/train_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/val_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_rec_report.json"))));
    }

    void workerRunsPaddleOcrDetOfficialAdapterPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR det adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString datasetPath = QDir(dir.path()).filePath(QStringLiteral("ocr-det"));
        writeTinyOcrDetDataset(datasetPath);
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-det-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-det-official-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_detection");
        request.datasetPath = datasetPath;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_det_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("epochs"), 1);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageSize"), 64);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawPrepareOnly = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_det_official");
            if (message.first == QStringLiteral("completed")) {
                sawPrepareOnly = message.second.value(QStringLiteral("mode")).toString() == QStringLiteral("prepareOnly");
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawPrepareOnly);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("aitrain_ppocrv4_det.yml"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/train_det_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("official_data/val_det_list.txt"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_det_report.json"))));
    }

    void workerRunsPaddleOcrSystemOfficialAdapterPrepareOnly()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR system adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString outputPath = QDir(dir.path()).filePath(QStringLiteral("official-system-output"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("paddleocr-system-official-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr");
        request.datasetPath = dir.filePath(QStringLiteral("image.png"));
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("trainingBackend"), QStringLiteral("paddleocr_system_official"));
        request.parameters.insert(QStringLiteral("pythonExecutable"), python);
        request.parameters.insert(QStringLiteral("prepareOnly"), true);
        request.parameters.insert(QStringLiteral("detModelDir"), dir.filePath(QStringLiteral("det_model")));
        request.parameters.insert(QStringLiteral("recModelDir"), dir.filePath(QStringLiteral("rec_model")));
        request.parameters.insert(QStringLiteral("dictionaryFile"), dir.filePath(QStringLiteral("dict.txt")));
        request.parameters.insert(QStringLiteral("inferenceImage"), dir.filePath(QStringLiteral("image.png")));

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(
            finished,
            qPrintable(QStringLiteral("Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))),
            15000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawBackend = false;
        bool sawPrepareOnly = false;
        for (const auto& message : messages) {
            sawBackend = sawBackend || message.second.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_system_official");
            if (message.first == QStringLiteral("completed")) {
                sawPrepareOnly = message.second.value(QStringLiteral("mode")).toString() == QStringLiteral("prepareOnly");
            }
        }
        QVERIFY(sawBackend);
        QVERIFY(sawPrepareOnly);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("run_official_system_predict.ps1"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_system_report.json"))));
    }

    void workerEnvironmentCheckReportsPythonTrainerBackends()
    {
        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::idle, this, [&finished]() {
            finished = true;
        });

        QString error;
        QVERIFY2(client.requestEnvironmentCheck(workerExecutablePath(), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);

        bool sawPython = false;
        bool sawUltralytics = false;
        bool sawPaddleOcr = false;
        bool sawPaddle = false;
        for (const auto& message : messages) {
            if (message.first != QStringLiteral("environmentCheck")) {
                continue;
            }
            const QJsonArray checks = message.second.value(QStringLiteral("checks")).toArray();
            for (const QJsonValue& value : checks) {
                const QJsonObject check = value.toObject();
                const QString name = check.value(QStringLiteral("name")).toString();
                sawPython = sawPython || name == QStringLiteral("Python");
                sawUltralytics = sawUltralytics || name == QStringLiteral("Ultralytics YOLO");
                sawPaddleOcr = sawPaddleOcr || name == QStringLiteral("PaddleOCR");
                sawPaddle = sawPaddle || name == QStringLiteral("PaddlePaddle");
            }
        }
        QVERIFY(sawPython);
        QVERIFY(sawUltralytics);
        QVERIFY(sawPaddleOcr);
        QVERIFY(sawPaddle);
    }

    void paddleOcrDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/a.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\nb\nc\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.jpg\tabc\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validatePaddleOcrRecDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 1);
        QVERIFY(!valid.previewSamples.isEmpty());

        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/missing.jpg\taz\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validatePaddleOcrRecDataset(root);
        QVERIFY(!invalid.ok);
        QVERIFY(invalid.errors.join(QStringLiteral("\n")).contains(QStringLiteral("字典")));
    }

    void paddleOcrDetDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyOcrDetDataset(root);

        const aitrain::DatasetValidationResult valid = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);
        QVERIFY(!valid.previewSamples.isEmpty());

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/missing.png\t[{\"transcription\":\"x\",\"points\":[[0,0],[1,0],[1,1],[0,1]]}]\n"));
        const aitrain::DatasetValidationResult missing = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!missing.ok);
        QVERIFY(missing.errors.join(QStringLiteral("\n")).contains(QStringLiteral("不存在")));

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\tbad-json\n"));
        const aitrain::DatasetValidationResult badJson = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!badJson.ok);
        QVERIFY(badJson.errors.join(QStringLiteral("\n")).contains(QStringLiteral("JSON")));

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\t[{\"transcription\":\"x\",\"points\":[[0,0],[1,0],[1,1]]}]\n"));
        const aitrain::DatasetValidationResult tooFew = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!tooFew.ok);
        QVERIFY(tooFew.errors.join(QStringLiteral("\n")).contains(QStringLiteral("4")));

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\t[{\"transcription\":\"x\",\"points\":[[0,0],[1,0],[1,1],[0,1]]}]\n"
                           "images/a.png\t[{\"transcription\":\"y\",\"points\":[[0,0],[1,0],[1,1],[0,1]]}]\n"));
        const aitrain::DatasetValidationResult duplicate = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!duplicate.ok);
        QVERIFY(duplicate.errors.join(QStringLiteral("\n")).contains(QStringLiteral("重复")));
    }

    void paddleOcrDetDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("source"));
        writeTinyOcrDetDataset(root);
        const QString output = dir.filePath(QStringLiteral("split"));

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        options.insert(QStringLiteral("seed"), 7);
        const aitrain::DatasetSplitResult result = aitrain::splitPaddleOcrDetDataset(root, output, options);
        QVERIFY2(result.ok, qPrintable(result.errors.join(QStringLiteral("\n"))));
        QCOMPARE(result.trainCount, 1);
        QCOMPARE(result.valCount, 1);
        QCOMPARE(result.testCount, 0);
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt_train.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt_val.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt_test.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("split_report.json"))));

        const aitrain::DatasetValidationResult validation = aitrain::validatePaddleOcrDetDataset(output);
        QVERIFY2(validation.ok, qPrintable(validation.errors.join(QStringLiteral("\n"))));
        QCOMPARE(validation.sampleCount, 2);
    }

    void workerRunsOcrRecognitionTrainingScaffoldEndToEnd()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);
        const QString outputPath = dir.filePath(QStringLiteral("worker-ocr"));

        aitrain::TrainingRequest request;
        request.taskId = QStringLiteral("ocr-task");
        request.projectPath = dir.path();
        request.pluginId = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        request.taskType = QStringLiteral("ocr_recognition");
        request.datasetPath = root;
        request.outputPath = outputPath;
        request.parameters.insert(QStringLiteral("epochs"), 2);
        request.parameters.insert(QStringLiteral("batchSize"), 1);
        request.parameters.insert(QStringLiteral("imageWidth"), 32);
        request.parameters.insert(QStringLiteral("imageHeight"), 16);
        request.parameters.insert(QStringLiteral("maxTextLength"), 8);

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool result, const QString& message) {
            finished = true;
            ok = result;
            finishedMessage = message;
        });

        QString error;
        QVERIFY2(client.startTraining(workerExecutablePath(), request, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 15000);
        QVERIFY2(ok, qPrintable(finishedMessage));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        bool sawCheckpoint = false;
        bool sawPreview = false;
        bool sawCtcLoss = false;
        bool sawAccuracy = false;
        bool sawEditDistance = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                sawCheckpoint = sawCheckpoint || kind == QStringLiteral("checkpoint");
                sawPreview = sawPreview || kind == QStringLiteral("preview");
            } else if (message.first == QStringLiteral("metric")) {
                const QString name = message.second.value(QStringLiteral("name")).toString();
                sawCtcLoss = sawCtcLoss || name == QStringLiteral("ctcLoss");
                sawAccuracy = sawAccuracy || name == QStringLiteral("accuracy");
                sawEditDistance = sawEditDistance || name == QStringLiteral("editDistance");
            }
        }
        QVERIFY(sawCheckpoint);
        QVERIFY(sawPreview);
        QVERIFY(sawCtcLoss);
        QVERIFY(sawAccuracy);
        QVERIFY(sawEditDistance);
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("checkpoint_latest.aitrain"))));
        QVERIFY(QFileInfo::exists(QDir(outputPath).filePath(QStringLiteral("preview_latest.json"))));
        QCOMPARE(messages.last().first, QStringLiteral("completed"));
    }
};

QTEST_MAIN(CoreTests)
#include "tst_core.moc"
