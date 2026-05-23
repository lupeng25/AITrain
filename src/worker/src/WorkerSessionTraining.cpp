#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/ProductWorkflow.h"

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

void WorkerSession::startTraining(const aitrain::TrainingRequest& request)
{
    request_ = request;
    activeTaskId_ = request_.taskId;
    activeOutputPath_ = request_.outputPath;
    step_ = 0;
    maxSteps_ = qMax(4, request.parameters.value(QStringLiteral("epochs")).toInt(20));
    running_ = true;
    paused_ = false;
    canceled_ = false;

    QDir().mkpath(request_.outputPath);
    QFile configFile(QDir(request_.outputPath).filePath(QStringLiteral("request.json")));
    if (configFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        configFile.write(QJsonDocument(request_.toJson()).toJson(QJsonDocument::Indented));
    }

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Worker accepted task %1 for plugin %2").arg(request_.taskId, request_.pluginId));
    send(QStringLiteral("log"), payload);

    const QString backend = requestedTrainingBackend(request_);
    if (!isSupportedTrainingBackendId(backend, request_.parameters)) {
        failWithDetails(
            backend.isEmpty()
                ? QStringLiteral("Training task type '%1' does not have an official production backend. Select an official backend explicitly.").arg(request_.taskType)
                : QStringLiteral("Training backend '%1' is not supported for production training. Use Ultralytics YOLO or PaddleOCR official adapters.").arg(backend),
            QStringLiteral("unsupported_training_backend"),
            QJsonObject{
                {QStringLiteral("backend"), backend},
                {QStringLiteral("taskType"), request_.taskType},
                {QStringLiteral("outputPath"), request_.outputPath}});
        return;
    }

    if (request_.taskType.compare(QStringLiteral("detection"), Qt::CaseInsensitive) == 0) {
        runDetectionTraining();
        return;
    }
    if (request_.taskType.compare(QStringLiteral("segmentation"), Qt::CaseInsensitive) == 0) {
        runSegmentationTraining();
        return;
    }
    if (request_.taskType.compare(QStringLiteral("ocr_recognition"), Qt::CaseInsensitive) == 0
        || request_.taskType.compare(QStringLiteral("ocr_detection"), Qt::CaseInsensitive) == 0
        || request_.taskType.compare(QStringLiteral("ocr"), Qt::CaseInsensitive) == 0) {
        runOcrRecTraining();
        return;
    }

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

void WorkerSession::runDetectionTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }
    failWithDetails(
        QStringLiteral("Detection training requires an official Python backend. Use ultralytics_yolo_detect."),
        QStringLiteral("unsupported_training_backend"),
        QJsonObject{{QStringLiteral("taskType"), request_.taskType}, {QStringLiteral("outputPath"), request_.outputPath}});
}

void WorkerSession::runSegmentationTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }
    failWithDetails(
        QStringLiteral("Segmentation training requires an official Python backend. Use ultralytics_yolo_segment."),
        QStringLiteral("unsupported_training_backend"),
        QJsonObject{{QStringLiteral("taskType"), request_.taskType}, {QStringLiteral("outputPath"), request_.outputPath}});
}

void WorkerSession::runOcrRecTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }
    failWithDetails(
        QStringLiteral("OCR training requires an official Python backend. Use paddleocr_det_official or paddleocr_rec_official."),
        QStringLiteral("unsupported_training_backend"),
        QJsonObject{{QStringLiteral("taskType"), request_.taskType}, {QStringLiteral("outputPath"), request_.outputPath}});
}
