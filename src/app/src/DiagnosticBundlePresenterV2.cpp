#include "DiagnosticBundlePresenterV2.h"

DiagnosticBundlePresenterV2::DiagnosticBundlePresenterV2(
    const aitrain::v2::ProjectQueryServiceV2* queryService, QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("DiagnosticBundlePresenterV2"));
}

bool DiagnosticBundlePresenterV2::selectTask(const QString& taskIdText)
{
    aitrain::v2::TaskId taskId;
    QString error;
    if (!aitrain::v2::TaskId::parse(taskIdText, &taskId, &error)) {
        clear();
        lastError_ = error;
        emit queryFailed(error);
        return false;
    }
    aitrain::v2::TaskReadModelV2 model;
    if (!queryService_ || !queryService_->taskDetails(taskId, &model, &error)) {
        clear();
        lastError_ = error.isEmpty() ? QStringLiteral("无法读取 Diagnostics V2 任务。") : error;
        emit queryFailed(lastError_);
        return false;
    }
    bool diagnosticsWorkflow = false;
    for (const aitrain::v2::WorkflowReadModelV2& workflow : model.workflows) {
        if (workflow.run.templateId == QStringLiteral("diagnostics_v2")) {
            diagnosticsWorkflow = true;
            break;
        }
    }
    if (!diagnosticsWorkflow) {
        clear();
        lastError_ = QStringLiteral("所选 Task 不是 Diagnostics Bundle V2 工作流。");
        emit queryFailed(lastError_);
        return false;
    }

    DiagnosticBundleViewModelV2 next;
    next.taskId = model.task.id.toString();
    next.state = aitrain::v2::taskStateToString(model.task.state);
    next.failureCode = aitrain::v2::failureCodeToString(model.task.failure.code);
    next.failureMessage = model.task.failure.message;
    for (const aitrain::v2::ArtifactSnapshotV2& artifact : model.artifacts) {
        if (artifact.kind == QStringLiteral("diagnostic_bundle_v2")) {
            next.diagnosticsArtifactId = artifact.id.toString();
        } else if (artifact.kind == QStringLiteral("evidence_bundle_v2")) {
            next.evidenceArtifactId = artifact.id.toString();
        }
    }
    viewModel_ = next;
    lastError_.clear();
    emit changed();
    return true;
}

void DiagnosticBundlePresenterV2::clear()
{
    viewModel_ = {};
    emit changed();
}

const DiagnosticBundleViewModelV2& DiagnosticBundlePresenterV2::viewModel() const { return viewModel_; }
QString DiagnosticBundlePresenterV2::lastError() const { return lastError_; }
