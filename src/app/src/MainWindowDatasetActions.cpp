#include "MainWindow.h"
#include "DatasetPageController.h"

void MainWindow::openDatasetQualityFixList()
{
    datasetPageController_->openQualityReport(true);
}

void MainWindow::openDatasetQualityReport()
{
    datasetPageController_->openQualityReport(false);
}
