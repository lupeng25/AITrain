#include "WorkbenchTranslation.h"
#include "MainWindow.h"
#include "TrainingPage.h"
#include "TrainingPageController.h"
#include "TaskArtifactPageController.h"
#include "ModelRegistryPageController.h"
#include "MainWindowSupport.h"

QWidget* MainWindow::buildTrainingPage()
{
    trainingPage_ = new TrainingWorkspacePage;
    trainingPageController_->attach(trainingPage_);
    connect(trainingPage_, &TrainingWorkspacePage::openTaskRequested, this,
        [this](const QString& id) {
            showPage(TaskQueuePage, aitrain_app::workbenchText(QStringLiteral("任务记录")));
            taskArtifactPageController_->openTask(id);
        });
    connect(trainingPage_, &TrainingWorkspacePage::modelRequested, this,
        [this](const QString& id) { showPage(ModelRegistryPage, aitrain_app::workbenchText(QStringLiteral("模型"))); modelRegistryPageController_->selectPackage(id); });
    return trainingPage_;
}
