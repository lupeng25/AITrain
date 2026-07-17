#include "MainWindow.h"

#include "TaskArtifactPresenter.h"

#include <QTableWidgetItem>

QString MainWindow::selectedTaskId() const
{
    if (!taskQueueTable_ || taskQueueTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = taskQueueTable_->selectedItems().first()->row();
    auto* item = taskQueueTable_->item(row, 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

void MainWindow::updateSelectedTaskDetails()
{
    if (!taskQueueTable_ || !taskArtifactPresenter_ || !taskArtifactPanel_) {
        return;
    }
    if (taskQueueTable_->selectedItems().isEmpty()) {
        clearSelectedTaskDetails();
        return;
    }
    const int row = taskQueueTable_->selectedItems().first()->row();
    const QString taskId = taskQueueTable_->item(row, 0)
        ? taskQueueTable_->item(row, 0)->data(Qt::UserRole).toString()
        : QString();
    if (taskId.isEmpty()) {
        clearSelectedTaskDetails();
        return;
    }
    taskArtifactPresenter_->selectTask(taskId);
}
