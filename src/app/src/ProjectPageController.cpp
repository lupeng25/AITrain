#include "ProjectPageController.h"

#include "ProjectSessionController.h"

ProjectPageController::ProjectPageController(
    const aitrain::ProjectQueryService* queryService,
    ProjectSessionController* sessionController, QObject* parent)
    : QObject(parent)
    , sessionController_(sessionController)
    , summaryPresenter_(new ProjectSummaryPresenter(queryService, this))
{
    Q_ASSERT(sessionController_);
    connect(sessionController_, &ProjectSessionController::busyChanged,
        this, [this](bool busy) {
            if (page_) page_->setBusy(busy);
        });
    connect(sessionController_, &ProjectSessionController::preparing,
        this, [this](const QString&, const QString&) {
            if (page_) {
                page_->setStatus(
                    tr("正在后台预检项目恢复，请稍候…"));
            }
        });
    connect(sessionController_, &ProjectSessionController::failed,
        this, [this](const QString& message) {
            if (page_) page_->showOperationError(message);
        });
    connect(sessionController_, &ProjectSessionController::activated,
        this, [this](const QString& name, const QString&, quint64) {
            setContext(true, name);
            refresh();
        });
}

void ProjectPageController::attachPage(ProjectWorkspacePage* page)
{
    page_ = page;
    connect(page_, &ProjectWorkspacePage::operationRequested,
        this, &ProjectPageController::request);
    page_->setBusy(sessionController_->isBusy());
    render();
}

void ProjectPageController::setContext(
    bool workspaceOpen, const QString& projectName)
{
    viewModel_.workspaceOpen = workspaceOpen;
    viewModel_.projectName = projectName;
}

void ProjectPageController::setDefaultRoot(const QString& root, bool apply)
{
    if (apply && page_) page_->setProjectRoot(root);
}

void ProjectPageController::refresh()
{
    if (viewModel_.workspaceOpen) {
        if (!summaryPresenter_->refresh()) {
            viewModel_.queryError = summaryPresenter_->lastError();
        } else {
            viewModel_.queryError.clear();
        }
    } else {
        summaryPresenter_->clear();
        viewModel_.queryError.clear();
    }
    viewModel_.summary = summaryPresenter_->viewModel();
    render();
}

void ProjectPageController::request(
    aitrain_app::ProjectSessionOperation operation)
{
    if (!page_) return;
    QString error;
    if (!sessionController_->request(
            operation, page_->projectName(), page_->projectRoot(), &error)) {
        page_->showOperationError(error);
        return;
    }
    emit sessionRequestAccepted();
}

void ProjectPageController::render()
{
    if (page_) page_->render(viewModel_);
}
