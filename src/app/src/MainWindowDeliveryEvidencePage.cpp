#include "WorkbenchTranslation.h"
#include "MainWindow.h"
#include "DeliveryEvidencePage.h"
#include "DeliveryEvidencePageController.h"
#include "TaskArtifactPageController.h"

QWidget* MainWindow::buildDeliveryEvidencePanel()
{
    deliveryEvidencePage_ = new DeliveryEvidenceWorkspacePage;
    deliveryEvidencePageController_->attach(deliveryEvidencePage_);
    connect(deliveryEvidencePage_, &DeliveryEvidenceWorkspacePage::openTaskRequested, this, [this](const QString& id) {
        showPage(TaskQueuePage, aitrain_app::workbenchText(QStringLiteral("任务记录"))); taskArtifactPageController_->openTask(id);
    });
    return deliveryEvidencePage_;
}
