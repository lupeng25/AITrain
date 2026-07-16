#include "EnvironmentCheckPresenterV2.h"

EnvironmentCheckPresenterV2::EnvironmentCheckPresenterV2(
    const aitrain::v2::ProjectQueryServiceV2* queryService, QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("EnvironmentCheckPresenterV2"));
}

bool EnvironmentCheckPresenterV2::selectTask(const QString& taskIdText)
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
    QJsonObject report;
    if (!queryService_ || !queryService_->taskDetails(taskId, &model, &error)
        || !queryService_->environmentCheckReport(taskId, &report, &error)) {
        clear();
        lastError_ = error.isEmpty() ? QStringLiteral("无法读取 Environment Check V2 任务。") : error;
        emit queryFailed(lastError_);
        return false;
    }
    bool correctWorkflow = false;
    for (const aitrain::v2::WorkflowReadModelV2& workflow : model.workflows) {
        if (workflow.run.templateId == QStringLiteral("environment_check_v2")) {
            correctWorkflow = true;
            break;
        }
    }
    if (!correctWorkflow) {
        clear();
        lastError_ = QStringLiteral("所选 Task 不是 Environment Check V2 工作流。");
        emit queryFailed(lastError_);
        return false;
    }
    EnvironmentCheckViewModelV2 next;
    next.taskId = model.task.id.toString();
    next.state = aitrain::v2::taskStateToString(model.task.state);
    next.report = report;
    for (const aitrain::v2::ArtifactSnapshotV2& artifact : model.artifacts) {
        if (artifact.kind == QStringLiteral("environment_profiles_report_v2"))
            next.reportArtifactId = artifact.id.toString();
        else if (artifact.kind == QStringLiteral("evidence_bundle_v2"))
            next.evidenceArtifactId = artifact.id.toString();
    }
    viewModel_ = next;
    lastError_.clear();
    emit changed();
    return true;
}

void EnvironmentCheckPresenterV2::clear()
{
    viewModel_ = {};
    emit changed();
}

const EnvironmentCheckViewModelV2& EnvironmentCheckPresenterV2::viewModel() const { return viewModel_; }
QString EnvironmentCheckPresenterV2::lastError() const { return lastError_; }
