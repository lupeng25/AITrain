#pragma once

#include "aitrain/v2/ProjectQueryServiceV2.h"

#include <QObject>
#include <QVector>

struct TaskListItemV2 final {
    QString taskId;
    QString capabilityId;
    QString taskType;
    QString state;
    QString stateLabel;
    QString updatedAt;
    QString message;
};

struct ArtifactFileItemV2 final {
    QString artifactId;
    QString kind;
    QString relativePath;
    QString sha256;
    qint64 byteCount = 0;
    QString createdAt;
};

struct MetricItemV2 final {
    QString name;
    double value = 0.0;
    QString occurredAt;
};

struct WorkflowStepItemV2 final {
    QString workflowId;
    QString templateId;
    int ordinal = -1;
    QString kind;
    QString state;
    QString backend;
    QString outputArtifactId;
};

struct TaskArtifactDetailsV2 final {
    QString taskId;
    QString summary;
    QVector<ArtifactFileItemV2> artifacts;
    QVector<MetricItemV2> metrics;
    QVector<WorkflowStepItemV2> workflowSteps;
};

// “任务与产物”页面的 V2 只读 Presenter。它只消费 ProjectQueryServiceV2
// 返回的持久化快照，不接收 Worker 消息，也不持有页面控件。
class TaskArtifactPresenterV2 final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int taskCount READ taskCount NOTIFY taskRowsChanged)
    Q_PROPERTY(QString selectedTaskId READ selectedTaskId NOTIFY detailsChanged)
    Q_PROPERTY(int artifactCount READ artifactCount NOTIFY detailsChanged)
    Q_PROPERTY(int metricCount READ metricCount NOTIFY detailsChanged)
    Q_PROPERTY(int workflowStepCount READ workflowStepCount NOTIFY detailsChanged)

public:
    explicit TaskArtifactPresenterV2(const aitrain::v2::ProjectQueryServiceV2* queryService,
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
    const QVector<TaskListItemV2>& taskRows() const;
    const TaskArtifactDetailsV2& details() const;

signals:
    void taskRowsChanged();
    void detailsChanged();
    void queryFailed(const QString& error);

private:
    void fail(const QString& error);

    const aitrain::v2::ProjectQueryServiceV2* queryService_ = nullptr;
    QVector<TaskListItemV2> taskRows_;
    TaskArtifactDetailsV2 details_;
    QString lastError_;
};
