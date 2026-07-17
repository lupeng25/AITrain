#pragma once

#include "aitrain/core/WorkerProtocol.h"

#include <QLocalServer>
#include <QLocalSocket>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QTimer>

#include <optional>

class WorkerClient : public QObject {
    Q_OBJECT

public:
    enum class WorkerTerminalStatus {
        Succeeded,
        Failed,
        Canceled,
    };
    Q_ENUM(WorkerTerminalStatus)

    explicit WorkerClient(QObject* parent = nullptr);
    ~WorkerClient() override;

    bool startTask(const QString& workerProgram,
        const aitrain::worker_protocol::TaskCommand& command,
        QString* error);
    void cancel();
    bool isRunning() const;

signals:
    void connected();
    void taskEventReceived(const aitrain::worker_protocol::TaskEvent& event);
    void logLine(const QString& line);
    void finished(WorkerTerminalStatus status, const QString& message);
    void idle();

private slots:
    void acceptConnection();
    void readLines();
    void workerFinished(int exitCode, QProcess::ExitStatus status);
    void workerProcessError(QProcess::ProcessError error);

private:
    bool startWorkerCommand(const QString& workerProgram, const QString& commandType, const QJsonObject& payload, QString* error);
    void finalizeWorkerExit();
    void sendStartTask();
    void sendCancelTask();
    bool sendEnvelope(const aitrain::ProtocolEnvelope& envelope, QString* error = nullptr);
    void publishEvent(const aitrain::worker_protocol::TaskEvent& event);
    void rejectProtocol(const QString& message);
    void cleanupSocket();

    QLocalServer server_;
    QLocalSocket* socket_ = nullptr;
    QProcess process_;
    QByteArray buffer_;
    std::optional<aitrain::worker_protocol::TaskCommand> pendingCommand_;
    aitrain::RequestId activeRequestId_;
    aitrain::TaskId activeTaskId_;
    QString controlToken_;
    aitrain::ProtocolSequenceTracker incomingSequenceTracker_;
    quint64 outgoingSequence_ = 0;
    bool finishedEmitted_ = false;
    bool startTaskSent_ = false;
    bool terminalEnvelopeReceived_ = false;
    QTimer cancelTimer_;
    QTimer connectionTimer_;
    QTimer terminalShutdownTimer_;
    bool cancelRequested_ = false;
    bool workerReady_ = false;
    bool finishing_ = false;
    bool terminalDrainAttempted_ = false;
    int pendingExitCode_ = 0;
    QProcess::ExitStatus pendingExitStatus_ = QProcess::NormalExit;
};

Q_DECLARE_METATYPE(WorkerClient::WorkerTerminalStatus)
