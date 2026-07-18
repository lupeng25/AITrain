#pragma once

#include "aitrain/core/WorkerProtocol.h"

#include <QHash>
#include <QObject>
#include <QStringList>
#include <QVector>

struct TaskMetricView final {
    QString name;
    double value = 0.0;
    QJsonObject details;
};

struct TaskArtifactView final {
    QString artifactId;
    QString kind;
    QString relativePath;
};

struct TaskViewState final {
    QString taskId;
    int progress = 0;
    QString status;
    QString terminalMessage;
    QStringList logs;
    QVector<TaskMetricView> metrics;
    QVector<TaskArtifactView> artifacts;
    qint64 metricSequence = 0;
    qint64 artifactSequence = 0;
    bool terminal = false;
};

class WorkerClient;

class ApplicationEventRouter final : public QObject {
    Q_OBJECT

public:
    explicit ApplicationEventRouter(WorkerClient* worker, QObject* parent = nullptr);

    TaskViewState viewState(const QString& taskId) const;
    void clear(const QString& taskId);

signals:
    void taskEventRouted(const aitrain::worker_protocol::TaskEvent& event);
    void taskViewStateChanged(const TaskViewState& state);
    void taskFactsInvalidated(const QString& taskId);
    void taskTerminalized(const QString& taskId,
        aitrain::worker_protocol::TaskEventKind kind,
        const QString& message);

private slots:
    void onTaskEvent(const aitrain::worker_protocol::TaskEvent& event);

private:
    static QString taskIdForEvent(const aitrain::worker_protocol::TaskEvent& event);
    TaskViewState& stateFor(const QString& taskId);

    WorkerClient* worker_ = nullptr;
    QHash<QString, TaskViewState> states_;
};

Q_DECLARE_METATYPE(TaskMetricView)
Q_DECLARE_METATYPE(TaskArtifactView)
Q_DECLARE_METATYPE(TaskViewState)
