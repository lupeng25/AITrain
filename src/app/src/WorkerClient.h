#pragma once

#include "aitrain/core/TaskModels.h"

#include <QLocalServer>
#include <QLocalSocket>
#include <QJsonObject>
#include <QObject>
#include <QProcess>

class WorkerClient : public QObject {
    Q_OBJECT

public:
    explicit WorkerClient(QObject* parent = nullptr);
    ~WorkerClient() override;

    bool startTraining(const QString& workerProgram, const aitrain::TrainingRequest& request, QString* error);
    bool requestEnvironmentCheck(const QString& workerProgram, QString* error);
    bool requestDatasetValidation(const QString& workerProgram, const QString& datasetPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId = {}, const QString& outputPath = {});
    bool requestDatasetSplit(const QString& workerProgram, const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId = {});
    bool requestDatasetCuration(const QString& workerProgram, const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId = {});
    bool requestDatasetSnapshot(const QString& workerProgram, const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options, QString* error, const QString& taskId = {});
    bool requestModelEvaluation(const QString& workerProgram, const QString& modelPath, const QString& datasetPath, const QString& outputPath, const QString& taskType, const QJsonObject& options, QString* error, const QString& taskId = {});
    bool requestModelBenchmark(const QString& workerProgram, const QString& modelPath, const QString& outputPath, const QJsonObject& options, QString* error, const QString& taskId = {});
    bool requestLocalPipeline(const QString& workerProgram, const QString& outputPath, const QString& templateId, const QJsonObject& options, QString* error, const QString& taskId = {});
    bool requestDeliveryReport(const QString& workerProgram, const QString& outputPath, const QJsonObject& context, QString* error, const QString& taskId = {});
    bool requestModelExport(const QString& workerProgram, const QString& checkpointPath, const QString& outputPath, const QString& format, QString* error, const QString& taskId = {});
    bool requestInference(const QString& workerProgram, const QString& checkpointPath, const QString& imagePath, const QString& outputPath, QString* error, const QString& taskId = {});
    void cancel();
    void pause();
    void resume();
    void requestHeartbeat();
    bool isRunning() const;

signals:
    void connected();
    void messageReceived(const QString& type, const QJsonObject& payload);
    void logLine(const QString& line);
    void finished(bool ok, const QString& message);
    void idle();

private slots:
    void acceptConnection();
    void readLines();
    void workerFinished(int exitCode, QProcess::ExitStatus status);

private:
    bool startWorkerCommand(const QString& workerProgram, const QString& commandType, const QJsonObject& payload, QString* error);
    void send(const QString& type, const QJsonObject& payload);
    void cleanupSocket();

    QLocalServer server_;
    QLocalSocket* socket_ = nullptr;
    QProcess process_;
    QByteArray buffer_;
    QString pendingCommandType_;
    QJsonObject pendingRequest_;
};
