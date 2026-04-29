#include "WorkerSession.h"

#include "aitrain/core/JsonProtocol.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QLibrary>
#include <QProcess>
#include <QRandomGenerator>

namespace {

QJsonObject checkObject(const QString& name, const QString& status, const QString& message, const QJsonObject& details = {})
{
    QJsonObject object;
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("message"), message);
    object.insert(QStringLiteral("details"), details);
    return object;
}

bool canLoadLibrary(const QString& libraryName)
{
    QLibrary library(libraryName);
    const bool loaded = library.load();
    if (loaded) {
        library.unload();
    }
    return loaded;
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

QJsonObject dllCheck(const QString& name, const QString& libraryName, const QString& missingMessage)
{
    return canLoadLibrary(libraryName)
        ? checkObject(name, QStringLiteral("ok"), QStringLiteral("可加载 %1。").arg(libraryName))
        : checkObject(name, QStringLiteral("missing"), missingMessage);
}

} // namespace

WorkerSession::WorkerSession(QObject* parent)
    : QObject(parent)
{
    connect(&socket_, &QLocalSocket::readyRead, this, &WorkerSession::readLines);
    connect(&socket_, &QLocalSocket::disconnected, qApp, &QCoreApplication::quit);
    connect(&timer_, &QTimer::timeout, this, &WorkerSession::tickTraining);
    timer_.setInterval(500);
}

bool WorkerSession::connectToServer(const QString& serverName)
{
    socket_.connectToServer(serverName);
    return socket_.waitForConnected(5000);
}

void WorkerSession::readLines()
{
    buffer_.append(socket_.readAll());
    int newline = buffer_.indexOf('\n');
    while (newline >= 0) {
        const QByteArray line = buffer_.left(newline);
        buffer_.remove(0, newline + 1);

        QString type;
        QJsonObject payload;
        QString requestId;
        QString error;
        if (aitrain::protocol::decodeMessage(line, &type, &payload, &requestId, &error)) {
            handleMessage(type, payload);
        } else {
            fail(QStringLiteral("Invalid protocol message: %1").arg(error));
        }

        newline = buffer_.indexOf('\n');
    }
}

void WorkerSession::tickTraining()
{
    if (!running_) {
        return;
    }

    ++step_;
    const int epoch = qMax(1, step_ / 2);
    const double progress = static_cast<double>(step_) / static_cast<double>(maxSteps_);
    const double loss = qMax(0.05, 1.2 * (1.0 - progress) + QRandomGenerator::global()->bounded(25) / 1000.0);
    const double quality = qMin(0.98, 0.35 + progress * 0.55 + QRandomGenerator::global()->bounded(20) / 1000.0);

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), qRound(progress * 100.0));
    progressPayload.insert(QStringLiteral("step"), step_);
    progressPayload.insert(QStringLiteral("epoch"), epoch);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject lossPayload;
    lossPayload.insert(QStringLiteral("taskId"), request_.taskId);
    lossPayload.insert(QStringLiteral("name"), QStringLiteral("loss"));
    lossPayload.insert(QStringLiteral("value"), loss);
    lossPayload.insert(QStringLiteral("step"), step_);
    lossPayload.insert(QStringLiteral("epoch"), epoch);
    send(QStringLiteral("metric"), lossPayload);

    QJsonObject mapPayload;
    mapPayload.insert(QStringLiteral("taskId"), request_.taskId);
    mapPayload.insert(QStringLiteral("name"), request_.taskType.contains(QStringLiteral("ocr"), Qt::CaseInsensitive) ? QStringLiteral("accuracy") : QStringLiteral("mAP50"));
    mapPayload.insert(QStringLiteral("value"), quality);
    mapPayload.insert(QStringLiteral("step"), step_);
    mapPayload.insert(QStringLiteral("epoch"), epoch);
    send(QStringLiteral("metric"), mapPayload);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 loss=%3 score=%4")
        .arg(epoch)
        .arg(step_)
        .arg(loss, 0, 'f', 4)
        .arg(quality, 0, 'f', 4));
    send(QStringLiteral("log"), logPayload);

    if (step_ >= maxSteps_) {
        complete();
    }
}

void WorkerSession::handleMessage(const QString& type, const QJsonObject& payload)
{
    if (type == QStringLiteral("startTrain")) {
        startTraining(aitrain::TrainingRequest::fromJson(payload));
    } else if (type == QStringLiteral("pause")) {
        pauseTraining();
    } else if (type == QStringLiteral("resume")) {
        resumeTraining();
    } else if (type == QStringLiteral("heartbeat")) {
        sendHeartbeat();
    } else if (type == QStringLiteral("environmentCheck")) {
        runEnvironmentCheck(payload);
    } else if (type == QStringLiteral("cancel")) {
        running_ = false;
        paused_ = false;
        timer_.stop();
        QJsonObject payloadObject;
        payloadObject.insert(QStringLiteral("taskId"), request_.taskId);
        payloadObject.insert(QStringLiteral("message"), QStringLiteral("Canceled by user"));
        send(QStringLiteral("canceled"), payloadObject);
        qApp->quit();
    } else {
        fail(QStringLiteral("Unsupported command: %1").arg(type));
    }
}

void WorkerSession::startTraining(const aitrain::TrainingRequest& request)
{
    request_ = request;
    step_ = 0;
    maxSteps_ = qMax(4, request.parameters.value(QStringLiteral("epochs")).toInt(20));
    running_ = true;
    paused_ = false;

    QDir().mkpath(request_.outputPath);
    QFile configFile(QDir(request_.outputPath).filePath(QStringLiteral("request.json")));
    if (configFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        configFile.write(QJsonDocument(request_.toJson()).toJson(QJsonDocument::Indented));
    }

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Worker accepted task %1 for plugin %2").arg(request_.taskId, request_.pluginId));
    send(QStringLiteral("log"), payload);
    timer_.start();
}

void WorkerSession::pauseTraining()
{
    if (!running_) {
        fail(QStringLiteral("Cannot pause because no training task is running"));
        return;
    }

    running_ = false;
    paused_ = true;
    timer_.stop();

    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), request_.taskId);
    payload.insert(QStringLiteral("message"), QStringLiteral("Training task paused"));
    send(QStringLiteral("paused"), payload);
}

void WorkerSession::resumeTraining()
{
    if (!paused_) {
        fail(QStringLiteral("Cannot resume because no training task is paused"));
        return;
    }

    paused_ = false;
    running_ = true;
    timer_.start();

    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), request_.taskId);
    payload.insert(QStringLiteral("message"), QStringLiteral("Training task resumed"));
    send(QStringLiteral("resumed"), payload);
}

void WorkerSession::sendHeartbeat()
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), request_.taskId);
    payload.insert(QStringLiteral("running"), running_);
    payload.insert(QStringLiteral("paused"), paused_);
    payload.insert(QStringLiteral("step"), step_);
    payload.insert(QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    send(QStringLiteral("heartbeat"), payload);
}

void WorkerSession::runEnvironmentCheck(const QJsonObject& payload)
{
    Q_UNUSED(payload)

    QJsonArray checks;
    checks.append(nvidiaSmiCheck());
    checks.append(dllCheck(QStringLiteral("CUDA Driver"), QStringLiteral("nvcuda"),
        QStringLiteral("无法加载 nvcuda，CUDA 驱动运行时不可用。")));
    checks.append(dllCheck(QStringLiteral("cuDNN"), QStringLiteral("cudnn64_8"),
        QStringLiteral("无法加载 cudnn64_8，后续真实训练需要配置 cuDNN DLL。")));
    checks.append(dllCheck(QStringLiteral("TensorRT"), QStringLiteral("nvinfer"),
        QStringLiteral("无法加载 nvinfer，TensorRT 导出和推理暂不可用。")));
    checks.append(dllCheck(QStringLiteral("ONNX Runtime"), QStringLiteral("onnxruntime"),
        QStringLiteral("无法加载 onnxruntime，ONNX 推理暂不可用。")));
    checks.append(dllCheck(QStringLiteral("LibTorch"), QStringLiteral("torch"),
        QStringLiteral("无法加载 torch，真实 LibTorch 训练暂不可用。")));
    checks.append(checkObject(QStringLiteral("Worker"), QStringLiteral("ok"), QStringLiteral("Worker 环境自检命令可用。")));

    QJsonObject result;
    result.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    result.insert(QStringLiteral("checks"), checks);
    send(QStringLiteral("environmentCheck"), result);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::send(const QString& type, const QJsonObject& payload)
{
    socket_.write(aitrain::protocol::encodeMessage(type, payload));
    socket_.flush();
}

void WorkerSession::fail(const QString& message)
{
    running_ = false;
    paused_ = false;
    timer_.stop();
    QJsonObject payload;
    payload.insert(QStringLiteral("message"), message);
    send(QStringLiteral("failed"), payload);
    socket_.waitForBytesWritten(1000);
    qApp->quit();
}

void WorkerSession::complete()
{
    running_ = false;
    paused_ = false;
    timer_.stop();

    const QString checkpointPath = QDir(request_.outputPath).filePath(QStringLiteral("checkpoint_best.aitrain"));
    QFile checkpoint(checkpointPath);
    if (checkpoint.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QJsonObject content;
        content.insert(QStringLiteral("taskId"), request_.taskId);
        content.insert(QStringLiteral("pluginId"), request_.pluginId);
        content.insert(QStringLiteral("taskType"), request_.taskType);
        content.insert(QStringLiteral("createdAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
        content.insert(QStringLiteral("note"), QStringLiteral("Platform scaffold checkpoint. Replace with native LibTorch weights in production plugins."));
        checkpoint.write(QJsonDocument(content).toJson(QJsonDocument::Indented));
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), request_.taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    artifact.insert(QStringLiteral("path"), checkpointPath);
    send(QStringLiteral("artifact"), artifact);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Training workflow completed"));
    send(QStringLiteral("completed"), payload);
    qApp->quit();
}
