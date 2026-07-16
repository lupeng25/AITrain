#pragma once

#include "aitrain/workflow\ProjectQueryService.h"

#include <QObject>
#include <QVector>

struct TaskListItem final {
    QString taskId;
    QString capabilityId;
    QString taskType;
    QString state;
    QString stateLabel;
    QString updatedAt;
    QString message;
};

struct ArtifactFileItem final {
    QString artifactId;
    QString kind;
    QString relativePath;
    QString sha256;
    qint64 byteCount = 0;
    QString createdAt;
};

struct MetricItem final {
    QString name;
    double value = 0.0;
    QString occurredAt;
};

struct WorkflowStepItem final {
    QString workflowId;
    QString templateId;
    int ordinal = -1;
    QString kind;
    QString state;
    QString backend;
    QString outputArtifactId;
};

struct TaskArtifactDetails final {
    QString taskId;
    QString summary;
    QVector<ArtifactFileItem> artifacts;
    QVector<MetricItem> metrics;
    QVector<WorkflowStepItem> workflowSteps;
};

// “任务与产物”页面的  只读 Presenter。它只消费 ProjectQueryService
// 返回的持久化快照，不接收 Worker 消息，也不持有页面控件。
class TaskArtifactPresenter final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int taskCount READ taskCount NOTIFY taskRowsChanged)
    Q_PROPERTY(QString selectedTaskId READ selectedTaskId NOTIFY detailsChanged)
    Q_PROPERTY(int artifactCount READ artifactCount NOTIFY detailsChanged)
    Q_PROPERTY(int metricCount READ metricCount NOTIFY detailsChanged)
    Q_PROPERTY(int workflowStepCount READ workflowStepCount NOTIFY detailsChanged)

public:
    explicit TaskArtifactPresenter(const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    bool refresh(int limit = 200);
    bool selectTask(const QString& taskId);
    void clearSelection();

    int taskCount() const;
    QString selectedTaskId() const;
    int artifactCount() const;
    int metricCount() const;
    int workflowStepCount() const;
    QString lastError() const;
    const QVector<TaskListItem>& taskRows() const;
    const TaskArtifactDetails& details() const;

signals:
    void taskRowsChanged();
    void detailsChanged();
    void queryFailed(const QString& error);

private:
    void fail(const QString& error);

    const aitrain::ProjectQueryService* queryService_ = nullptr;
    QVector<TaskListItem> taskRows_;
    TaskArtifactDetails details_;
    QString lastError_;
};
