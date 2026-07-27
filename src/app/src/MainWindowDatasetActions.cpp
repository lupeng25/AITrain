#include "MainWindow.h"

#include "LanguageSupport.h"

#include <QStatusBar>

using namespace aitrain_app;

void MainWindow::openDatasetQualityFixList()
{
    showPage(TaskQueuePage, uiText("任务与产物"));
    statusBar()->showMessage(
        uiText("请按 Repair ArtifactId 查看受控问题清单。"), 5000);
}

void MainWindow::openDatasetQualityReport()
{
    showPage(TaskQueuePage, uiText("任务与产物"));
    statusBar()->showMessage(
        uiText("请按 Quality Report ArtifactId 查看受控报告。"), 5000);
}
