#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "TaskArtifactPanel.h"
#include "TaskArtifactPresenter.h"

#include <QApplication>
#include <QClipboard>
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QMessageBox>
#include <QStatusBar>
#include <QTableWidgetItem>
#include <QUrl>

using namespace aitrain_app;

QString MainWindow::selectedTaskId() const
{
    if (!taskQueueTable_ || taskQueueTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = taskQueueTable_->selectedItems().first()->row();
    auto* item = taskQueueTable_->item(row, 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

QString MainWindow::selectedArtifactPath() const
{
    return taskArtifactPanel_ ? taskArtifactPanel_->selectedArtifactPath() : QString();
}

QString MainWindow::selectedEvaluationReportPath() const
{
    if (!evaluationReportTable_ || evaluationReportTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = evaluationReportTable_->selectedItems().first()->row();
    auto* item = evaluationReportTable_->item(row, 3);
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

void MainWindow::updateSelectedEvaluationReportDetails()
{
    if (!evaluationReportView_) {
        return;
    }
    const QString reportPath = selectedEvaluationReportPath();
    if (reportPath.isEmpty()) {
        evaluationReportView_->clear();
        return;
    }
    evaluationReportView_->loadReport(reportPath);
}

void MainWindow::openSelectedArtifactDirectory()
{
    const QString path = selectedArtifactPath();
    if (path.isEmpty()) {
        return;
    }
    const QFileInfo info(path);
    const QString directory = info.isDir() ? info.absoluteFilePath() : info.absolutePath();
    QDesktopServices::openUrl(QUrl::fromLocalFile(directory));
}

void MainWindow::copySelectedArtifactPath()
{
    const QString path = selectedArtifactPath();
    if (!path.isEmpty()) {
        QApplication::clipboard()->setText(QDir::toNativeSeparators(path));
        statusBar()->showMessage(uiText("产物路径已复制"), 3000);
    }
}

void MainWindow::useSelectedArtifactForInference()
{
    QMessageBox::information(this, uiText("推理"), uiText(" 推理不接受任务产物的裸路径。请先将模型导入为已验证的  模型包。"));
    showDeploymentTab(1);
}
