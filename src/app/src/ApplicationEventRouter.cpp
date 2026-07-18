#include "ApplicationEventRouter.h"

#include "WorkerClient.h"

#include <QJsonArray>
#include <QtMath>

namespace wp = aitrain::worker_protocol;

ApplicationEventRouter::ApplicationEventRouter(WorkerClient* worker, QObject* parent)
    : QObject(parent)
    , worker_(worker)
{
    Q_ASSERT(worker_);
    qRegisterMetaType<wp::TaskEvent>("aitrain::worker_protocol::TaskEvent");
    qRegisterMetaType<TaskViewState>("TaskViewState");
    connect(worker_, &WorkerClient::taskEventReceived,
        this, &ApplicationEventRouter::onTaskEvent);
}

TaskViewState ApplicationEventRouter::viewState(const QString& taskId) const
{
    return states_.value(taskId);
}

void ApplicationEventRouter::clear(const QString& taskId)
{
    states_.remove(taskId);
}

QString ApplicationEventRouter::taskIdForEvent(const wp::TaskEvent& event)
{
    return event.taskId.toString();
}

TaskViewState& ApplicationEventRouter::stateFor(const QString& taskId)
{
    TaskViewState& state = states_[taskId];
    state.taskId = taskId;
    return state;
}

void ApplicationEventRouter::onTaskEvent(const wp::TaskEvent& event)
{
    emit taskEventRouted(event);
    const QString taskId = taskIdForEvent(event);
    if (taskId.isEmpty()) {
        return;
    }

    TaskViewState& state = stateFor(taskId);
    switch (event.kind) {
    case wp::TaskEventKind::Ready:
        state.status = QStringLiteral("starting");
        break;
    case wp::TaskEventKind::Log:
        // 日志是可丢弃的活跃视图数据，最多保留最近 512 行；终态和 Artifact
        // 不通过该限额处理，避免背压导致事实丢失。
        state.logs.append(event.details.value(QStringLiteral("message")).toString());
        while (state.logs.size() > 512) state.logs.removeFirst();
        break;
    case wp::TaskEventKind::Progress: {
        int percent = event.details.value(QStringLiteral("percent")).toInt(-1);
        if (percent < 0 && event.details.contains(QStringLiteral("value"))) {
            percent = qRound(event.details.value(QStringLiteral("value")).toDouble() * 100.0);
        }
        state.progress = qBound(0, percent < 0 ? state.progress : percent, 100);
        state.status = QStringLiteral("running");
        break;
    }
    case wp::TaskEventKind::Metric: {
        TaskMetricView metric;
        metric.name = event.details.value(QStringLiteral("name")).toString();
        metric.value = event.details.value(QStringLiteral("value")).toDouble();
        metric.details = event.details;
        state.metrics.append(metric);
        ++state.metricSequence;
        while (state.metrics.size() > 1024) state.metrics.removeFirst();
        break;
    }
    case wp::TaskEventKind::Artifact: {
        TaskArtifactView artifact;
        artifact.artifactId = event.details.value(QStringLiteral("artifactId")).toString();
        artifact.kind = event.details.value(QStringLiteral("kind")).toString();
        artifact.relativePath = event.details.value(QStringLiteral("relativePath")).toString();
        state.artifacts.append(artifact);
        ++state.artifactSequence;
        while (state.artifacts.size() > 256) state.artifacts.removeFirst();
        break;
    }
    case wp::TaskEventKind::Succeeded:
        state.status = QStringLiteral("succeeded");
        state.terminal = true;
        state.progress = 100;
        state.terminalMessage = event.details.value(QStringLiteral("message")).toString();
        break;
    case wp::TaskEventKind::Failed:
        state.status = QStringLiteral("failed");
        state.terminal = true;
        state.terminalMessage = event.details.value(QStringLiteral("message")).toString();
        break;
    case wp::TaskEventKind::Canceled:
        state.status = QStringLiteral("canceled");
        state.terminal = true;
        state.terminalMessage = event.details.value(QStringLiteral("message")).toString();
        break;
    case wp::TaskEventKind::Result:
        // Workflow 结果不再作为页面事实源；Worker 已经持久化后只通知查询缓存失效。
        emit taskFactsInvalidated(taskId);
        break;
    }

    emit taskViewStateChanged(state);
    if (event.kind == wp::TaskEventKind::Succeeded
        || event.kind == wp::TaskEventKind::Failed
        || event.kind == wp::TaskEventKind::Canceled) {
        const QString terminalMessage = state.terminalMessage;
        emit taskFactsInvalidated(taskId);
        emit taskTerminalized(taskId, event.kind, terminalMessage);
        // Router 只保存活跃任务的瞬态投影；终态事实必须从 Query Service 重新读取。
        states_.remove(taskId);
    }
}
