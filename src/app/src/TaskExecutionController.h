#pragma once

#include "aitrain/core/WorkerProtocol.h"
#include "WorkerClient.h"

#include <QObject>

class TaskExecutionController final : public QObject {
    Q_OBJECT

public:
    explicit TaskExecutionController(WorkerClient* worker, QObject* parent = nullptr);

    bool start(const QString& workerProgram,
        const aitrain::worker_protocol::TaskCommand& command,
        QString* error);
    void cancel();
    bool isRunning() const;

signals:
    void taskEvent(const aitrain::worker_protocol::TaskEvent& event);
    void finished(WorkerClient::WorkerTerminalStatus status, const QString& message);
    void idle();

private:
    WorkerClient* worker_ = nullptr;
};
