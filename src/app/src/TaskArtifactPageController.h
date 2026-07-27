#pragma once

#include "TaskArtifactPresenter.h"

#include <QObject>
#include <QPointer>

class TaskArtifactPage;

// 页面控制器拥有筛选状态、分页游标和 Presenter；MainWindow 只负责
// 把运行时取消命令接到全局 TaskRuntimeController。
class TaskArtifactPageController final : public QObject {
    Q_OBJECT

public:
    explicit TaskArtifactPageController(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    void attachPage(TaskArtifactPage* page);
    bool refresh(const aitrain::PageRequest& request = {100, {}});
    bool refreshSelected();
    void clearSelection();
    const QVector<TaskListItem>& taskRows() const;
    QString selectedTaskId() const;
    void setCancelable(bool cancelable);

signals:
    void cancelRequested();

private:
    void renderRows();
    void selectTask(const QString& taskId);

    TaskArtifactPresenter* presenter_ = nullptr;
    QPointer<TaskArtifactPage> page_;
    QString taskKindFilter_;
    QString taskStateFilter_;
    QString query_;
    bool cancelable_ = false;
};
