#pragma once

#include "ProjectSummaryPresenter.h"
#include "TaskArtifactPresenter.h"

#include <QWidget>

class QLabel;
class QTableWidget;

enum class DashboardRoute {
    Project,
    Dataset,
    Training,
    TaskArtifact,
    ModelRegistry,
    RuntimeDelivery
};

struct DashboardViewModel final {
    bool projectOpen = false;
    QString projectName;
    QString environmentStatus;
    QString gpuStatus;
    int capabilityCount = 0;
    ProjectSummaryViewModel summary;
    QVector<TaskListItem> recentTasks;
};

// Dashboard 页面只渲染 Controller 提供的 ViewModel。
class DashboardWorkspacePage final : public QWidget {
    Q_OBJECT

public:
    explicit DashboardWorkspacePage(QWidget* parent = nullptr);

    void render(const DashboardViewModel& viewModel);

signals:
    void routeRequested(DashboardRoute route);

private:
    QLabel* projectStatusLabel_ = nullptr;
    QLabel* gpuStatusLabel_ = nullptr;
    QLabel* projectValueLabel_ = nullptr;
    QLabel* datasetValueLabel_ = nullptr;
    QLabel* taskValueLabel_ = nullptr;
    QLabel* nextStepLabel_ = nullptr;
    QTableWidget* recentTasksTable_ = nullptr;
};
