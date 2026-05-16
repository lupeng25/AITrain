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

    aitrain::DetectionTrainingOptions options;
    options.epochs = qMax(1, request_.parameters.value(QStringLiteral("epochs")).toInt(1));
    options.batchSize = qMax(1, request_.parameters.value(QStringLiteral("batchSize")).toInt(1));
    const int imageSize = qMax(1, request_.parameters.value(QStringLiteral("imageSize")).toInt(320));
    options.imageSize = QSize(imageSize, imageSize);
    options.gridSize = qBound(1, request_.parameters.value(QStringLiteral("gridSize")).toInt(4), 16);
    options.horizontalFlip = request_.parameters.value(QStringLiteral("horizontalFlip")).toBool(false);
    options.colorJitter = request_.parameters.value(QStringLiteral("colorJitter")).toBool(false);
    options.trainingBackend = request_.parameters.value(QStringLiteral("trainingBackend")).toString();
    options.resumeCheckpointPath = request_.parameters.value(QStringLiteral("resumeCheckpointPath")).toString();
    options.outputPath = request_.outputPath;

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), request_.taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("step"), 0);
    startProgress.insert(QStringLiteral("epoch"), 0);
    send(QStringLiteral("progress"), startProgress);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("Starting detection training with the tiny_linear_detector scaffold backend. Real YOLO-style LibTorch training is not implemented yet."));
    send(QStringLiteral("log"), logPayload);

    const aitrain::DetectionTrainingResult result = aitrain::trainDetectionBaseline(
        request_.datasetPath,
        options,
        [this](const aitrain::DetectionTrainingMetrics& metrics) {
            QCoreApplication::processEvents();
            while (paused_) {
                QThread::msleep(50);
                QCoreApplication::processEvents();
            }
            if (!running_) {
                return false;
            }

            step_ = metrics.step;
            const int progressBase = qMax(1, metrics.totalSteps);
            const int percent = qMin(99, qMax(1, qRound(100.0 * static_cast<double>(metrics.step) / static_cast<double>(progressBase))));

            QJsonObject progressPayload;
            progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
            progressPayload.insert(QStringLiteral("percent"), percent);
            progressPayload.insert(QStringLiteral("step"), metrics.step);
            progressPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("progress"), progressPayload);

            QJsonObject lossPayload;
            lossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            lossPayload.insert(QStringLiteral("name"), QStringLiteral("loss"));
            lossPayload.insert(QStringLiteral("value"), metrics.loss);
            lossPayload.insert(QStringLiteral("step"), metrics.step);
            lossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), lossPayload);

            QJsonObject objectnessLossPayload;
            objectnessLossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            objectnessLossPayload.insert(QStringLiteral("name"), QStringLiteral("objectnessLoss"));
            objectnessLossPayload.insert(QStringLiteral("value"), metrics.objectnessLoss);
            objectnessLossPayload.insert(QStringLiteral("step"), metrics.step);
            objectnessLossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), objectnessLossPayload);

            QJsonObject classLossPayload;
            classLossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            classLossPayload.insert(QStringLiteral("name"), QStringLiteral("classLoss"));
            classLossPayload.insert(QStringLiteral("value"), metrics.classLoss);
            classLossPayload.insert(QStringLiteral("step"), metrics.step);
            classLossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), classLossPayload);

            QJsonObject boxLossPayload;
            boxLossPayload.insert(QStringLiteral("taskId"), request_.taskId);
            boxLossPayload.insert(QStringLiteral("name"), QStringLiteral("boxLoss"));
            boxLossPayload.insert(QStringLiteral("value"), metrics.boxLoss);
            boxLossPayload.insert(QStringLiteral("step"), metrics.step);
            boxLossPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("metric"), boxLossPayload);

            QJsonObject stepLogPayload;
            stepLogPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 loss=%3 objLoss=%4 classLoss=%5 boxLoss=%6")
                .arg(metrics.epoch)
                .arg(metrics.step)
                .arg(metrics.loss, 0, 'f', 4)
                .arg(metrics.objectnessLoss, 0, 'f', 4)
                .arg(metrics.classLoss, 0, 'f', 4)
                .arg(metrics.boxLoss, 0, 'f', 4));
            send(QStringLiteral("log"), stepLogPayload);
            return true;
        });

    if (!result.ok) {
        if (canceled_) {
            socket_.waitForBytesWritten(1000);
            return;
        }
        fail(result.error);
        return;
    }

    running_ = false;
    paused_ = false;

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("step"), result.steps);
    progressPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), request_.taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    artifact.insert(QStringLiteral("path"), result.checkpointPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detector scaffold checkpoint"));
    artifact.insert(QStringLiteral("trainingBackend"), result.trainingBackend);
    artifact.insert(QStringLiteral("modelFamily"), result.modelFamily);
    artifact.insert(QStringLiteral("scaffold"), result.scaffold);
    send(QStringLiteral("artifact"), artifact);

    emitDetectionPreviewArtifacts(result.checkpointPath);

    QJsonObject precisionPayload;
    precisionPayload.insert(QStringLiteral("taskId"), request_.taskId);
    precisionPayload.insert(QStringLiteral("name"), QStringLiteral("precision"));
    precisionPayload.insert(QStringLiteral("value"), result.precision);
    precisionPayload.insert(QStringLiteral("step"), result.steps);
    precisionPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("metric"), precisionPayload);

    QJsonObject recallPayload;
    recallPayload.insert(QStringLiteral("taskId"), request_.taskId);
    recallPayload.insert(QStringLiteral("name"), QStringLiteral("recall"));
    recallPayload.insert(QStringLiteral("value"), result.recall);
    recallPayload.insert(QStringLiteral("step"), result.steps);
    recallPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("metric"), recallPayload);

    QJsonObject mapPayload;
    mapPayload.insert(QStringLiteral("taskId"), request_.taskId);
    mapPayload.insert(QStringLiteral("name"), QStringLiteral("mAP50"));
    mapPayload.insert(QStringLiteral("value"), result.map50);
    mapPayload.insert(QStringLiteral("step"), result.steps);
    mapPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("metric"), mapPayload);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detection training completed"));
    payload.insert(QStringLiteral("trainingBackend"), result.trainingBackend);
    payload.insert(QStringLiteral("modelFamily"), result.modelFamily);
    payload.insert(QStringLiteral("scaffold"), result.scaffold);
    send(QStringLiteral("completed"), payload);
    finishSession();
}

void WorkerSession::runSegmentationTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }

    aitrain::SegmentationTrainingOptions options;
    options.epochs = qMax(1, request_.parameters.value(QStringLiteral("epochs")).toInt(1));
    options.batchSize = qMax(1, request_.parameters.value(QStringLiteral("batchSize")).toInt(1));
    const int imageSize = qMax(1, request_.parameters.value(QStringLiteral("imageSize")).toInt(320));
    options.imageSize = QSize(imageSize, imageSize);
    options.learningRate = qMax(0.001, request_.parameters.value(QStringLiteral("learningRate")).toDouble(0.05));
    options.outputPath = request_.outputPath;

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), request_.taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("step"), 0);
    startProgress.insert(QStringLiteral("epoch"), 0);
    send(QStringLiteral("progress"), startProgress);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("Starting tiny mask segmentation scaffold. This is not full YOLO segmentation training."));
    send(QStringLiteral("log"), logPayload);

    const aitrain::SegmentationTrainingResult result = aitrain::trainSegmentationBaseline(
        request_.datasetPath,
        options,
        [this](const aitrain::SegmentationTrainingMetrics& metrics) {
            QCoreApplication::processEvents();
            while (paused_) {
                QThread::msleep(50);
                QCoreApplication::processEvents();
            }
            if (!running_) {
                return false;
            }

            step_ = metrics.step;
            const int progressBase = qMax(1, metrics.totalSteps);
            const int percent = qMin(99, qMax(1, qRound(100.0 * static_cast<double>(metrics.step) / static_cast<double>(progressBase))));

            QJsonObject progressPayload;
            progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
            progressPayload.insert(QStringLiteral("percent"), percent);
            progressPayload.insert(QStringLiteral("step"), metrics.step);
            progressPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("progress"), progressPayload);

            const auto sendMetric = [this, &metrics](const QString& name, double value) {
                QJsonObject payload;
                payload.insert(QStringLiteral("taskId"), request_.taskId);
                payload.insert(QStringLiteral("name"), name);
                payload.insert(QStringLiteral("value"), value);
                payload.insert(QStringLiteral("step"), metrics.step);
                payload.insert(QStringLiteral("epoch"), metrics.epoch);
                send(QStringLiteral("metric"), payload);
            };
            sendMetric(QStringLiteral("loss"), metrics.loss);
            sendMetric(QStringLiteral("maskLoss"), metrics.maskLoss);
            sendMetric(QStringLiteral("maskCoverage"), metrics.maskCoverage);
            sendMetric(QStringLiteral("maskIoU"), metrics.maskIou);
            sendMetric(QStringLiteral("precision"), metrics.precision);
            sendMetric(QStringLiteral("recall"), metrics.recall);
            sendMetric(QStringLiteral("segmentationMap50"), metrics.map50);

            QJsonObject stepLogPayload;
            stepLogPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 maskLoss=%3 maskCoverage=%4 maskIoU=%5 segmentationMap50=%6")
                .arg(metrics.epoch)
                .arg(metrics.step)
                .arg(metrics.maskLoss, 0, 'f', 4)
                .arg(metrics.maskCoverage, 0, 'f', 4)
                .arg(metrics.maskIou, 0, 'f', 4)
                .arg(metrics.map50, 0, 'f', 4));
            send(QStringLiteral("log"), stepLogPayload);
            return true;
        });

    if (!result.ok) {
        if (canceled_) {
            socket_.waitForBytesWritten(1000);
            return;
        }
        fail(result.error);
        return;
    }

    running_ = false;
    paused_ = false;

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("step"), result.steps);
    progressPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject checkpointArtifact;
    checkpointArtifact.insert(QStringLiteral("taskId"), request_.taskId);
    checkpointArtifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    checkpointArtifact.insert(QStringLiteral("path"), result.checkpointPath);
    checkpointArtifact.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold checkpoint"));
    send(QStringLiteral("artifact"), checkpointArtifact);

    if (!result.previewPath.isEmpty()) {
        QJsonObject previewArtifact;
        previewArtifact.insert(QStringLiteral("taskId"), request_.taskId);
        previewArtifact.insert(QStringLiteral("kind"), QStringLiteral("preview"));
        previewArtifact.insert(QStringLiteral("path"), result.previewPath);
        previewArtifact.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold preview"));
        send(QStringLiteral("artifact"), previewArtifact);
    }

    if (!result.maskPreviewPath.isEmpty()) {
        QJsonObject maskPreviewArtifact;
        maskPreviewArtifact.insert(QStringLiteral("taskId"), request_.taskId);
        maskPreviewArtifact.insert(QStringLiteral("kind"), QStringLiteral("mask_preview"));
        maskPreviewArtifact.insert(QStringLiteral("path"), result.maskPreviewPath);
        maskPreviewArtifact.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold mask preview"));
        send(QStringLiteral("artifact"), maskPreviewArtifact);
    }

    const auto sendFinalMetric = [this, &result, &options](const QString& name, double value) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("name"), name);
        payload.insert(QStringLiteral("value"), value);
        payload.insert(QStringLiteral("step"), result.steps);
        payload.insert(QStringLiteral("epoch"), options.epochs);
        send(QStringLiteral("metric"), payload);
    };
    sendFinalMetric(QStringLiteral("maskLoss"), result.finalLoss);
    sendFinalMetric(QStringLiteral("maskCoverage"), result.maskCoverage);
    sendFinalMetric(QStringLiteral("maskIoU"), result.maskIou);
    sendFinalMetric(QStringLiteral("precision"), result.precision);
    sendFinalMetric(QStringLiteral("recall"), result.recall);
    sendFinalMetric(QStringLiteral("segmentationMap50"), result.map50);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("Tiny mask segmentation scaffold completed"));
    send(QStringLiteral("completed"), payload);
    finishSession();
}

void WorkerSession::runOcrRecTraining()
{
    if (shouldUsePythonTrainer()) {
        runPythonTrainer();
        return;
    }

    aitrain::OcrRecTrainingOptions options;
    options.epochs = qMax(1, request_.parameters.value(QStringLiteral("epochs")).toInt(1));
    options.batchSize = qMax(1, request_.parameters.value(QStringLiteral("batchSize")).toInt(1));
    const int imageWidth = qMax(1, request_.parameters.value(QStringLiteral("imageWidth")).toInt(100));
    const int imageHeight = qMax(1, request_.parameters.value(QStringLiteral("imageHeight")).toInt(32));
    options.imageSize = QSize(imageWidth, imageHeight);
    options.learningRate = qMax(0.001, request_.parameters.value(QStringLiteral("learningRate")).toDouble(0.05));
    options.maxTextLength = qMax(1, request_.parameters.value(QStringLiteral("maxTextLength")).toInt(25));
    options.labelFilePath = request_.parameters.value(QStringLiteral("labelFilePath")).toString();
    options.dictionaryFilePath = request_.parameters.value(QStringLiteral("dictionaryFilePath")).toString();
    options.outputPath = request_.outputPath;

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), request_.taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("step"), 0);
    startProgress.insert(QStringLiteral("epoch"), 0);
    send(QStringLiteral("progress"), startProgress);

    QJsonObject logPayload;
    logPayload.insert(QStringLiteral("message"), QStringLiteral("Starting OCR recognition scaffold. This is not real CRNN/CTC OCR training."));
    send(QStringLiteral("log"), logPayload);

    const aitrain::OcrRecTrainingResult result = aitrain::trainOcrRecBaseline(
        request_.datasetPath,
        options,
        [this](const aitrain::OcrRecTrainingMetrics& metrics) {
            QCoreApplication::processEvents();
            while (paused_) {
                QThread::msleep(50);
                QCoreApplication::processEvents();
            }
            if (!running_) {
                return false;
            }

            step_ = metrics.step;
            const int progressBase = qMax(1, metrics.totalSteps);
            const int percent = qMin(99, qMax(1, qRound(100.0 * static_cast<double>(metrics.step) / static_cast<double>(progressBase))));

            QJsonObject progressPayload;
            progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
            progressPayload.insert(QStringLiteral("percent"), percent);
            progressPayload.insert(QStringLiteral("step"), metrics.step);
            progressPayload.insert(QStringLiteral("epoch"), metrics.epoch);
            send(QStringLiteral("progress"), progressPayload);

            const auto sendMetric = [this, &metrics](const QString& name, double value) {
                QJsonObject payload;
                payload.insert(QStringLiteral("taskId"), request_.taskId);
                payload.insert(QStringLiteral("name"), name);
                payload.insert(QStringLiteral("value"), value);
                payload.insert(QStringLiteral("step"), metrics.step);
                payload.insert(QStringLiteral("epoch"), metrics.epoch);
                send(QStringLiteral("metric"), payload);
            };
            sendMetric(QStringLiteral("loss"), metrics.loss);
            sendMetric(QStringLiteral("ctcLoss"), metrics.ctcLoss);
            sendMetric(QStringLiteral("accuracy"), metrics.accuracy);
            sendMetric(QStringLiteral("editDistance"), metrics.editDistance);

            QJsonObject stepLogPayload;
            stepLogPayload.insert(QStringLiteral("message"), QStringLiteral("epoch=%1 step=%2 ctcLoss=%3 accuracy=%4 editDistance=%5")
                .arg(metrics.epoch)
                .arg(metrics.step)
                .arg(metrics.ctcLoss, 0, 'f', 4)
                .arg(metrics.accuracy, 0, 'f', 4)
                .arg(metrics.editDistance, 0, 'f', 4));
            send(QStringLiteral("log"), stepLogPayload);
            return true;
        });

    if (!result.ok) {
        if (canceled_) {
            socket_.waitForBytesWritten(1000);
            return;
        }
        fail(result.error);
        return;
    }

    running_ = false;
    paused_ = false;

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), request_.taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("step"), result.steps);
    progressPayload.insert(QStringLiteral("epoch"), options.epochs);
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject checkpointArtifact;
    checkpointArtifact.insert(QStringLiteral("taskId"), request_.taskId);
    checkpointArtifact.insert(QStringLiteral("kind"), QStringLiteral("checkpoint"));
    checkpointArtifact.insert(QStringLiteral("path"), result.checkpointPath);
    checkpointArtifact.insert(QStringLiteral("message"), QStringLiteral("OCR recognition scaffold checkpoint"));
    send(QStringLiteral("artifact"), checkpointArtifact);

    if (!result.previewPath.isEmpty()) {
        QJsonObject previewArtifact;
        previewArtifact.insert(QStringLiteral("taskId"), request_.taskId);
        previewArtifact.insert(QStringLiteral("kind"), QStringLiteral("preview"));
        previewArtifact.insert(QStringLiteral("path"), result.previewPath);
        previewArtifact.insert(QStringLiteral("message"), QStringLiteral("OCR recognition scaffold preview"));
        send(QStringLiteral("artifact"), previewArtifact);
    }

    const auto sendFinalMetric = [this, &result, &options](const QString& name, double value) {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), request_.taskId);
        payload.insert(QStringLiteral("name"), name);
        payload.insert(QStringLiteral("value"), value);
        payload.insert(QStringLiteral("step"), result.steps);
        payload.insert(QStringLiteral("epoch"), options.epochs);
        send(QStringLiteral("metric"), payload);
    };
    sendFinalMetric(QStringLiteral("ctcLoss"), result.finalLoss);
    sendFinalMetric(QStringLiteral("accuracy"), result.accuracy);
    sendFinalMetric(QStringLiteral("editDistance"), result.editDistance);

    QJsonObject payload;
    payload.insert(QStringLiteral("message"), QStringLiteral("OCR recognition scaffold completed"));
    send(QStringLiteral("completed"), payload);
    finishSession();
}

void WorkerSession::emitDetectionPreviewArtifacts(const QString& checkpointPath)
{
    QString error;
    aitrain::DetectionBaselineCheckpoint checkpoint;
    if (!aitrain::loadDetectionBaselineCheckpoint(checkpointPath, &checkpoint, &error)) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not load detection checkpoint for preview: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }

    aitrain::DetectionDataset dataset;
    if (!dataset.load(request_.datasetPath, QStringLiteral("val"), &error)
        && !dataset.load(request_.datasetPath, QStringLiteral("train"), &error)) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not load detection dataset for preview: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }
    if (dataset.samples().isEmpty()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not create detection preview because the dataset split is empty."));
        send(QStringLiteral("log"), payload);
        return;
    }

    const QString imagePath = dataset.samples().first().imagePath;
    const QVector<aitrain::DetectionPrediction> predictions = aitrain::predictDetectionBaseline(checkpoint, imagePath, &error);
    if (!error.isEmpty()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not create detection predictions: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }

    QJsonArray predictionArray;
    for (const aitrain::DetectionPrediction& prediction : predictions) {
        predictionArray.append(aitrain::detectionPredictionToJson(prediction));
    }

    const QString predictionsPath = QDir(request_.outputPath).filePath(QStringLiteral("predictions_latest.json"));
    QFile predictionsFile(predictionsPath);
    if (predictionsFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QJsonObject document;
        document.insert(QStringLiteral("taskId"), request_.taskId);
        document.insert(QStringLiteral("checkpointPath"), checkpointPath);
        document.insert(QStringLiteral("imagePath"), imagePath);
        document.insert(QStringLiteral("predictions"), predictionArray);
        predictionsFile.write(QJsonDocument(document).toJson(QJsonDocument::Indented));
        predictionsFile.close();

        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), request_.taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("predictions"));
        artifact.insert(QStringLiteral("path"), predictionsPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detector predictions"));
        send(QStringLiteral("artifact"), artifact);
    } else {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not write detection predictions: %1").arg(predictionsPath));
        send(QStringLiteral("log"), payload);
    }

    const QImage preview = aitrain::renderDetectionPredictions(imagePath, predictions, &error);
    if (preview.isNull()) {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not render detection preview: %1").arg(error));
        send(QStringLiteral("log"), payload);
        return;
    }

    const QString previewPath = QDir(request_.outputPath).filePath(QStringLiteral("preview_latest.png"));
    if (preview.save(previewPath)) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), request_.taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("preview"));
        artifact.insert(QStringLiteral("path"), previewPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Tiny linear detector preview"));
        send(QStringLiteral("artifact"), artifact);
    } else {
        QJsonObject payload;
        payload.insert(QStringLiteral("message"), QStringLiteral("Could not write detection preview: %1").arg(previewPath));
        send(QStringLiteral("log"), payload);
    }
}
