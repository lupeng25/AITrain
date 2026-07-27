#include "TaskArtifactPageController.h"

#include "TaskArtifactPage.h"

TaskArtifactPageController::TaskArtifactPageController(
    const aitrain::ProjectQueryService* queryService, QObject* parent)
    : QObject(parent)
    , presenter_(new TaskArtifactPresenter(queryService, this))
{
    connect(presenter_, &TaskArtifactPresenter::taskRowsChanged,
        this, &TaskArtifactPageController::renderRows);
    connect(presenter_, &TaskArtifactPresenter::detailsChanged, this, [this]() {
        if (page_) page_->setDetails(presenter_->details());
    });
}

void TaskArtifactPageController::attachPage(TaskArtifactPage* page)
{
    if (page_ == page) return;
    if (page_) disconnect(page_, nullptr, this, nullptr);
    page_ = page;
    if (!page_) return;

    page_->setPresenter(presenter_);
    connect(page_, &TaskArtifactPage::refreshRequested, this, [this]() {
        refresh({100, {}});
    });
    connect(page_, &TaskArtifactPage::loadMoreRequested, this, [this]() {
        if (presenter_->loadMore()) renderRows();
    });
    connect(page_, &TaskArtifactPage::cancelRequested,
        this, &TaskArtifactPageController::cancelRequested);
    connect(page_, &TaskArtifactPage::filterChanged, this,
        [this](const QString& taskKind, const QString& taskState,
            const QString& query) {
            taskKindFilter_ = taskKind;
            taskStateFilter_ = taskState;
            query_ = query;
            if (page_) page_->applyFilter(taskKindFilter_, taskStateFilter_, query_);
        });
    connect(page_, &TaskArtifactPage::selectedTaskChanged,
        this, &TaskArtifactPageController::selectTask);
    renderRows();
    page_->setCancelable(cancelable_);
    if (!presenter_->selectedTaskId().isEmpty()) {
        page_->setDetails(presenter_->details());
    }
}

bool TaskArtifactPageController::refresh(const aitrain::PageRequest& request)
{
    const bool ok = presenter_->refresh(request);
    renderRows();
    return ok;
}

bool TaskArtifactPageController::refreshSelected()
{
    const QString taskId = selectedTaskId();
    if (taskId.isEmpty()) {
        clearSelection();
        return false;
    }
    return presenter_->selectTask(taskId);
}

void TaskArtifactPageController::clearSelection()
{
    presenter_->clearSelection();
    if (page_) page_->clearDetails();
}

const QVector<TaskListItem>& TaskArtifactPageController::taskRows() const
{
    return presenter_->taskRows();
}

QString TaskArtifactPageController::selectedTaskId() const
{
    return page_ ? page_->selectedTaskId() : presenter_->selectedTaskId();
}

void TaskArtifactPageController::setCancelable(bool cancelable)
{
    cancelable_ = cancelable;
    if (page_) page_->setCancelable(cancelable_);
}

void TaskArtifactPageController::renderRows()
{
    if (page_) {
        page_->setRows(presenter_->taskRows(), presenter_->hasMoreTasks());
    }
}

void TaskArtifactPageController::selectTask(const QString& taskId)
{
    if (taskId.isEmpty()) {
        clearSelection();
        return;
    }
    presenter_->selectTask(taskId);
}
