#include "WorkbenchTranslation.h"
#include "DiagnosticBundlePresenter.h"

DiagnosticBundlePresenter::DiagnosticBundlePresenter(
    const aitrain::ProjectQueryService* queryService, QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("DiagnosticBundlePresenter"));
}

bool DiagnosticBundlePresenter::selectTask(const QString& taskIdText)
{
    aitrain::TaskId taskId;
    QString error;
    if (!aitrain::TaskId::parse(taskIdText, &taskId, &error)) {
        clear();
        lastError_ = error;
        emit queryFailed(error);
        return false;
    }
    aitrain::TaskReadModel model;
    if (!queryService_ || !queryService_->taskDetails(taskId, &model, &error)) {
        clear();
        lastError_ = error.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("无法读取 Diagnostics 任务。")) : error;
        emit queryFailed(lastError_);
        return false;
    }
    bool diagnosticsWorkflow = false;
    for (const aitrain::WorkflowReadModel& workflow : model.workflows) {
        if (workflow.run.templateId == QStringLiteral("diagnostics")) {
            diagnosticsWorkflow = true;
            break;
        }
    }
    if (!diagnosticsWorkflow) {
        clear();
        lastError_ = aitrain_app::workbenchText(QStringLiteral("所选 Task 不是 Diagnostics Bundle 工作流。"));
        emit queryFailed(lastError_);
        return false;
    }

    DiagnosticBundleViewModel next;
    next.taskId = model.task.id.toString();
    next.state = aitrain::taskStateToString(model.task.state);
    next.failureCode = aitrain::failureCodeToString(model.task.failure.code);
    next.failureMessage = model.task.failure.message;
    for (const aitrain::ArtifactSnapshot& artifact : model.artifacts) {
        if (artifact.kind == QStringLiteral("diagnostic_bundle")) {
            next.diagnosticsArtifactId = artifact.id.toString();
        } else if (artifact.kind == QStringLiteral("evidence_bundle")) {
            next.evidenceArtifactId = artifact.id.toString();
        }
    }
    viewModel_ = next;
    lastError_.clear();
    emit changed();
    return true;
}

void DiagnosticBundlePresenter::clear()
{
    viewModel_ = {};
    emit changed();
}

const DiagnosticBundleViewModel& DiagnosticBundlePresenter::viewModel() const { return viewModel_; }
QString DiagnosticBundlePresenter::lastError() const { return lastError_; }
