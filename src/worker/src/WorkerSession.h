#pragma once

#include "aitrain/core/TaskModels.h"

#include <QLocalSocket>
#include <QObject>
#include <QTimer>

class WorkerSession : public QObject {
    Q_OBJECT

public:
    explicit WorkerSession(QObject* parent = nullptr);
    bool connectToServer(const QString& serverName);

private slots:
    void readLines();
    void tickTraining();

private:
    void handleMessage(const QString& type, const QJsonObject& payload);
    void startTraining(const aitrain::TrainingRequest& request);
    void pauseTraining();
    void resumeTraining();
    void sendHeartbeat();
    void runEnvironmentCheck(const QJsonObject& payload);
    void validateDataset(const QJsonObject& payload);
    void splitDataset(const QJsonObject& payload);
    void exportModel(const QJsonObject& payload);
    void runInference(const QJsonObject& payload);
    void runDetectionTraining();
    void runSegmentationTraining();
    void runOcrRecTraining();
    void emitDetectionPreviewArtifacts(const QString& checkpointPath);
    void send(const QString& type, const QJsonObject& payload);
    void fail(const QString& message);
    void complete();

    QLocalSocket socket_;
    QByteArray buffer_;
    QTimer timer_;
    aitrain::TrainingRequest request_;
    int step_ = 0;
    int maxSteps_ = 20;
    bool running_ = false;
    bool paused_ = false;
    bool canceled_ = false;
};
