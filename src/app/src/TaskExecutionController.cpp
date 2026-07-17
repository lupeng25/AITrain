#include "TaskExecutionController.h"

#include "WorkerClient.h"

TaskExecutionController::TaskExecutionController(WorkerClient* worker, QObject* parent)
    : QObject(parent)
    , worker_(worker)
{
    Q_ASSERT(worker_);
    connect(worker_, &WorkerClient::taskEventReceived,
        this, &TaskExecutionController::taskEvent);
    connect(worker_, &WorkerClient::finished,
        this, &TaskExecutionController::finished);
    connect(worker_, &WorkerClient::idle,
        this, &TaskExecutionController::idle);
}

bool TaskExecutionController::start(const QString& workerProgram,
    const aitrain::worker_protocol::TaskCommand& command,
    QString* error)
{
    return worker_->startTask(workerProgram, command, error);
}

void TaskExecutionController::cancel()
{
    worker_->cancel();
}

bool TaskExecutionController::isRunning() const
{
    return worker_->isRunning();
}
