#include "MainWindow.h"

#include "ModelRegistryPageController.h"
#include "RuntimeDeliveryPage.h"
#include "RuntimeDeliveryPageController.h"

QWidget* MainWindow::buildDeploymentPage()
{
    runtimeDeliveryPage_ = new RuntimeDeliveryWorkspacePage(this);
    runtimeDeliveryPageController_->attach(runtimeDeliveryPage_);
    runtimeDeliveryPageController_->setModelPackages(
        modelRegistryPageController_->packages());
    return runtimeDeliveryPage_;
}
