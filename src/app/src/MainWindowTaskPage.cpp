#include "MainWindow.h"

#include "TaskArtifactPage.h"
#include "TaskArtifactPageController.h"
#include "TaskRuntimeController.h"

#include <QMessageBox>

using namespace aitrain_app;

QWidget* MainWindow::buildTaskQueuePage()
{
    taskArtifactPage_ = new TaskArtifactPage;
    taskArtifactPageController_->attachPage(taskArtifactPage_);
    connect(taskArtifactPageController_, &TaskArtifactPageController::cancelRequested,
        this, [this]() {
            if (taskController_->taskId().isValid() && taskController_->isRunning()) {
                // Worker/Core 是运行任务取消的唯一写入方；GUI 不直接双写状态。
                taskController_->cancel();
                return;
            }
            QMessageBox::information(this, uiText("任务队列"),
                uiText("只能取消当前 GUI 会话派发且仍在运行的任务。历史任务为只读。"));
        });
    return taskArtifactPage_;
}

void MainWindow::updateTaskCancelButton()
{
    taskArtifactPageController_->setCancelable(
        taskController_->taskId().isValid() && taskController_->isRunning());
}
