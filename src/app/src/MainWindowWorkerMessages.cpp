#include "MainWindow.h"

#include "ApplicationEventRouter.h"
#include "EnvironmentCheckPresenter.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"

using namespace aitrain_app;

void MainWindow::handleTaskViewStateChanged(const TaskViewState& state)
{
    if (activeTaskId_.isEmpty() || state.taskId != activeTaskId_) {
        return;
    }

    if (activeWorkflowKind_ == QStringLiteral("training") && progressBar_) {
        progressBar_->setValue(state.progress);
    }
    if (activeWorkflowKind_ == QStringLiteral("dataset_conversion")
        && datasetConversionProgressBar_) {
        datasetConversionProgressBar_->setValue(state.progress);
    }

    if (!state.terminal) {
        updateTaskCancelButton();
        return;
    }

    if (activeWorkflowKind_ == QStringLiteral("dataset_conversion")) {
        setDatasetConversionFormRunning(false);
    }

    if (activeWorkflowKind_ == QStringLiteral("environment_check")
        && environmentCheckPresenter_) {
        environmentCheckPresenter_->selectTask(activeTaskId_);
        refreshEnvironmentReportView();
    }

    QString workflowName = uiText("当前任务");
    if (activeWorkflowKind_ == QStringLiteral("training")) {
        workflowName = uiText("训练 Workflow");
    } else if (activeWorkflowKind_ == QStringLiteral("data_quality")) {
        workflowName = uiText("Data Quality");
    } else if (activeWorkflowKind_ == QStringLiteral("dataset_conversion")) {
        workflowName = uiText("Dataset Conversion");
    } else if (activeWorkflowKind_ == QStringLiteral("dataset_split")) {
        workflowName = uiText("Dataset Split");
    } else if (activeWorkflowKind_ == QStringLiteral("environment_check")) {
        workflowName = uiText("Environment Check");
    } else if (activeWorkflowKind_ == QStringLiteral("diagnostics")) {
        workflowName = uiText("Diagnostics Bundle");
    }

    appendLog(uiText("%1：%2：%3").arg(
        workflowName, state.status, state.terminalMessage));
    activeTaskId_.clear();
    activeWorkflowKind_.clear();
    updateTaskCancelButton();
    updateRecentTasks();
    updateSelectedTaskDetails();
    updateProjectSummary();
    updateDashboardSummary();
    updateDeliveryAcceptanceSummary();
    updateModelRegistry();
}
