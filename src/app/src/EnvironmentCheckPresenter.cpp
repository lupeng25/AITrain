#include "EnvironmentCheckPresenter.h"

EnvironmentCheckPresenter::EnvironmentCheckPresenter(
    const aitrain::ProjectQueryService* queryService, QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("EnvironmentCheckPresenter"));
}

bool EnvironmentCheckPresenter::selectTask(const QString& taskIdText)
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
    QJsonObject report;
    if (!queryService_ || !queryService_->taskDetails(taskId, &model, &error)
        || !queryService_->environmentCheckReport(taskId, &report, &error)) {
        clear();
        lastError_ = error.isEmpty() ? QStringLiteral("无法读取 Environment Check  任务。") : error;
        emit queryFailed(lastError_);
        return false;
    }
    bool correctWorkflow = false;
    for (const aitrain::WorkflowReadModel& workflow : model.workflows) {
        if (workflow.run.templateId == QStringLiteral("environment_check")) {
            correctWorkflow = true;
            break;
        }
    }
    if (!correctWorkflow) {
        clear();
        lastError_ = QStringLiteral("所选 Task 不是 Environment Check  工作流。");
        emit queryFailed(lastError_);
        return false;
    }
    EnvironmentCheckViewModel next;
    next.taskId = model.task.id.toString();
    next.state = aitrain::taskStateToString(model.task.state);
    next.report = report;
    for (const aitrain::ArtifactSnapshot& artifact : model.artifacts) {
        if (artifact.kind == QStringLiteral("environment_profiles_report"))
            next.reportArtifactId = artifact.id.toString();
        else if (artifact.kind == QStringLiteral("evidence_bundle"))
            next.evidenceArtifactId = artifact.id.toString();
    }
    viewModel_ = next;
    lastError_.clear();
    emit changed();
    return true;
}

void EnvironmentCheckPresenter::clear()
{
    viewModel_ = {};
    emit changed();
}

const EnvironmentCheckViewModel& EnvironmentCheckPresenter::viewModel() const { return viewModel_; }
QString EnvironmentCheckPresenter::lastError() const { return lastError_; }
