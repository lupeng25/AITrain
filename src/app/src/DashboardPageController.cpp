#include "DashboardPageController.h"

DashboardPageController::DashboardPageController(
    const aitrain::ProjectQueryService* queryService, QObject* parent)
    : QObject(parent)
    , summaryPresenter_(new ProjectSummaryPresenter(queryService, this))
    , recentTaskPresenter_(new TaskArtifactPresenter(queryService, this))
{
}

void DashboardPageController::attachPage(DashboardWorkspacePage* page)
{
    page_ = page;
    render();
}

void DashboardPageController::setContext(bool projectOpen,
    const QString& projectName, int capabilityCount,
    const QString& environmentStatus, const QString& gpuStatus)
{
    viewModel_.projectOpen = projectOpen;
    viewModel_.projectName = projectName;
    viewModel_.capabilityCount = capabilityCount;
    viewModel_.environmentStatus = environmentStatus;
    viewModel_.gpuStatus = gpuStatus;
}

void DashboardPageController::refresh()
{
    if (viewModel_.projectOpen) {
        summaryPresenter_->refresh();
        recentTaskPresenter_->refresh({10, {}});
    } else {
        summaryPresenter_->clear();
        recentTaskPresenter_->clearSelection();
    }
    viewModel_.summary = summaryPresenter_->viewModel();
    viewModel_.recentTasks = viewModel_.projectOpen
        ? recentTaskPresenter_->taskRows() : QVector<TaskListItem>{};
    render();
}

void DashboardPageController::render()
{
    if (page_) page_->render(viewModel_);
}
