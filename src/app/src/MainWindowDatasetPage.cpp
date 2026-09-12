#include "WorkbenchTranslation.h"
#include "MainWindow.h"

#include "DatasetPage.h"
#include "DatasetPageController.h"
#include "TrainingPage.h"

QWidget* MainWindow::buildDatasetPage()
{
    datasetPage_ = new DatasetWorkspacePage;
    datasetPageController_->attach(datasetPage_);
    connect(datasetPage_, &DatasetWorkspacePage::trainRequested, this, [this]() {
        showPage(TrainingPage, aitrain_app::workbenchText(QStringLiteral("训练")));
        refreshTrainingDefaults();
        if (trainingPage_) trainingPage_->setMode(TrainingWorkspacePage::Configuration);
    });
    connect(datasetPage_, &DatasetWorkspacePage::reportRequested, this, [this](bool repair) {
        if (repair) openDatasetQualityFixList();
        else openDatasetQualityReport();
    });
    return datasetPage_;
}
