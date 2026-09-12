#include "MainWindow.h"
#include "TaskRuntimeController.h"
#include "EnvironmentPageController.h"
#include "TrainingPageController.h"
#include "RuntimeDeliveryPageController.h"
#include "DeliveryEvidencePageController.h"
#include "DatasetPage.h"
#include "DatasetPageController.h"

#include "ApplicationEventRouter.h"
#include "EnvironmentCheckPresenter.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
using namespace aitrain_app;

void MainWindow::handleTaskViewStateChanged(const TaskViewState& state)
{
    if (!taskController_->taskId().isValid()
        || state.taskId != taskController_->taskId().toString()) {
        return;
    }

    const QString workflowKind = taskController_->workflowKind();
    datasetPageController_->applyTaskViewState(state);
    runtimeDeliveryPageController_->applyTaskViewState(state);
    deliveryEvidencePageController_->applyTaskViewState(state);
    if (workflowKind == QStringLiteral("training")) {
        trainingPageController_->applyTaskViewState(state);
    }
    if (workflowKind == QStringLiteral("dataset_conversion")
        && datasetPage_ && datasetPage_->datasetConversionProgressBar) {
        datasetPage_->datasetConversionProgressBar->setValue(state.progress);
    }

    if (!state.terminal) {
        updateTaskCancelButton();
        return;
    }

    if (workflowKind == QStringLiteral("dataset_conversion")) {
        datasetPageController_->setConversionRunning(false);
    }

    if (workflowKind == QStringLiteral("environment_check")) {
        environmentPageController_->selectTask(taskController_->taskId().toString());
    }

    QString workflowName = uiText("当前任务");
    if (workflowKind == QStringLiteral("training")) {
        workflowName = uiText("训练 Workflow");
    } else if (workflowKind == QStringLiteral("data_quality")) {
        workflowName = uiText("Data Quality");
    } else if (workflowKind == QStringLiteral("dataset_conversion")) {
        workflowName = uiText("Dataset Conversion");
    } else if (workflowKind == QStringLiteral("dataset_split")) {
        workflowName = uiText("Dataset Split");
    } else if (workflowKind == QStringLiteral("environment_check")) {
        workflowName = uiText("Environment Check");
    } else if (workflowKind == QStringLiteral("diagnostics")) {
        workflowName = uiText("Diagnostics Bundle");
    } else if (workflowKind == QStringLiteral("model_import")) {
        workflowName = uiText("模型导入");
    }

    appendLog(uiText("%1：%2：%3").arg(
        workflowName, state.status, state.terminalMessage));
    updateTaskCancelButton();
}
