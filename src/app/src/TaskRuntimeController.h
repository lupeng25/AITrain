#pragma once

#include "aitrain/core/WorkerProtocol.h"
#include "WorkerClient.h"

#include <QObject>

class TaskRuntimeController final : public QObject {
    Q_OBJECT

public:
    enum class State {
        Idle,
        Starting,
        Running,
        CancelRequested,
        Finalizing,
        Recovering
    };
    Q_ENUM(State)

    explicit TaskRuntimeController(QObject* parent = nullptr);

    bool start(const QString& workerProgram,
        const aitrain::worker_protocol::TaskCommand& command,
        QString* error);
    void cancel();
    bool isRunning() const;
    State state() const;
    const aitrain::TaskId& taskId() const;
    WorkerClient& workerClient();

signals:
    void stateChanged(TaskRuntimeController::State state);
    void taskEvent(const aitrain::worker_protocol::TaskEvent& event);
    void finished(WorkerClient::WorkerTerminalStatus status, const QString& message);
    void idle();

private:
    void setState(State state);

    WorkerClient worker_;
    State state_ = State::Idle;
    aitrain::TaskId taskId_;
};

Q_DECLARE_METATYPE(TaskRuntimeController::State)
