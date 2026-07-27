#include "MainWindow.h"

#include "TaskArtifactPageController.h"

QString MainWindow::selectedTaskId() const
{
    return taskArtifactPageController_
        ? taskArtifactPageController_->selectedTaskId() : QString();
}
