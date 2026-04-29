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

    bool startTraining(const QString& workerProgram, const aitrain::TrainingRequest& request, QString* error);
    bool requestEnvironmentCheck(const QString& workerProgram, QString* error);
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

    QLocalServer server_;
    QLocalSocket* socket_ = nullptr;
    QProcess process_;
    QByteArray buffer_;
    QString pendingCommandType_;
    QJsonObject pendingRequest_;
};
