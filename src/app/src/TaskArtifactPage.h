#pragma once

#include "TaskArtifactPresenter.h"

#include <QScrollArea>

class QComboBox;
class QLineEdit;
class QPushButton;
class QTableView;
class TaskArtifactPanel;
class TaskListFilterProxyModel;
class TaskListTableModel;

// “任务与产物”页面只拥有视觉控件和表格模型。查询、分页、选择与
// 命令协调由 TaskArtifactPageController 负责。
class TaskArtifactPage final : public QScrollArea {
    Q_OBJECT

public:
    explicit TaskArtifactPage(QWidget* parent = nullptr);

    void setPresenter(TaskArtifactPresenter* presenter);
    void setRows(const QVector<TaskListItem>& rows, bool hasMore);
    void setDetails(const TaskArtifactDetails& details);
    void clearDetails();
    void setCancelable(bool cancelable);
    void applyFilter(const QString& taskKind, const QString& taskState,
        const QString& query);
    QString selectedTaskId() const;

signals:
    void refreshRequested();
    void loadMoreRequested();
    void cancelRequested();
    void filterChanged(const QString& taskKind, const QString& taskState,
        const QString& query);
    void selectedTaskChanged(const QString& taskId);

private:
    void ensureVisibleSelection();
    void emitFilterChanged();
    void emitSelectedTaskChanged();

    QComboBox* taskKindFilterCombo_ = nullptr;
    QComboBox* taskStateFilterCombo_ = nullptr;
    QLineEdit* taskSearchEdit_ = nullptr;
    QTableView* taskQueueTable_ = nullptr;
    TaskListTableModel* taskListTableModel_ = nullptr;
    TaskListFilterProxyModel* taskListFilterModel_ = nullptr;
    QPushButton* taskLoadMoreButton_ = nullptr;
    QPushButton* taskCancelButton_ = nullptr;
    TaskArtifactPanel* taskArtifactPanel_ = nullptr;
};
