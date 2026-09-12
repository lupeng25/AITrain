#include "WorkbenchTranslation.h"
#include "MainWindow.h"

#include "ModelRegistryPageController.h"
#include "RuntimeDeliveryPage.h"
#include "RuntimeDeliveryPageController.h"
#include "TaskArtifactPageController.h"

QWidget* MainWindow::buildDeploymentPage()
{
    runtimeDeliveryPage_ = new RuntimeDeliveryWorkspacePage(this);
    runtimeDeliveryPageController_->attach(runtimeDeliveryPage_);
    connect(runtimeDeliveryPage_, &RuntimeDeliveryWorkspacePage::openTaskRequested, this, [this](const QString& id) {
        showPage(TaskQueuePage, aitrain_app::workbenchText(QStringLiteral("任务记录"))); taskArtifactPageController_->openTask(id);
    });
    runtimeDeliveryPageController_->setModelPackages(
        modelRegistryPageController_->packages());
    return runtimeDeliveryPage_;
}
