#include "EnvironmentPageController.h"

#include "EnvironmentPage.h"
#include "TaskRuntimeController.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QMessageBox>

EnvironmentPageController::EnvironmentPageController(
    const aitrain::ProjectQueryService* queryService,
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , presenter_(queryService, this)
    , taskRuntime_(taskRuntime)
{
    connect(&presenter_, &EnvironmentCheckPresenter::changed,
        this, [this]() {
            render();
            emit changed();
        });
}

void EnvironmentPageController::attach(EnvironmentWorkspacePage* page)
{
    page_ = page;
    connect(page_, &EnvironmentWorkspacePage::runRequested,
        this, &EnvironmentPageController::runCheck);
    render();
}

void EnvironmentPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
    if (!projectOpen_) {
        presenter_.clear();
    }
}

void EnvironmentPageController::setWorkerExecutable(const QString& executable)
{
    workerExecutable_ = executable;
}

bool EnvironmentPageController::selectTask(const QString& taskId)
{
    return presenter_.selectTask(taskId);
}

void EnvironmentPageController::clear()
{
    presenter_.clear();
}

QJsonObject EnvironmentPageController::report() const
{
    return presenter_.viewModel().report;
}

QString EnvironmentPageController::selectedTaskId() const
{
    return presenter_.viewModel().taskId;
}

void EnvironmentPageController::runCheck()
{
    if (taskRuntime_->isRunning()) {
        QMessageBox::warning(page_, tr("环境自检"),
            tr("Worker 正在执行任务，稍后再运行环境自检。"));
        return;
    }
    if (!projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(page_, tr("环境自检"), tr("请先打开项目。"));
        return;
    }
    presenter_.clear();
    page_->setChecking();
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::EnvironmentCheckCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    QString error;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("环境自检"), error);
        return;
    }
    emit taskStarted(taskId.toString(), QStringLiteral("environment_check"));
    emit runStarted();
}

void EnvironmentPageController::render()
{
    if (page_) {
        page_->renderReport(presenter_.viewModel().report);
    }
}
