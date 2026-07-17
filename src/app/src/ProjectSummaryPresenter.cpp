#include "ProjectSummaryPresenter.h"

ProjectSummaryPresenter::ProjectSummaryPresenter(
    const aitrain::ProjectQueryService* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("ProjectSummaryPresenter"));
}

bool ProjectSummaryPresenter::refresh()
{
    aitrain::ProjectSummaryReadModel summary;
    QString error;
    if (!queryService_ || !queryService_->projectSummary(&summary, &error)) {
        viewModel_ = ProjectSummaryViewModel();
        lastError_ = error.isEmpty()
            ? QStringLiteral("无法读取项目汇总。")
            : error;
        emit summaryChanged();
        emit queryFailed(lastError_);
        return false;
    }

    ProjectSummaryViewModel model;
    model.available = true;
    model.taskCount = summary.tasks.total();
    model.activeTaskCount = summary.tasks.queued + summary.tasks.starting
        + summary.tasks.running + summary.tasks.cancelRequested;
    model.succeededTaskCount = summary.tasks.succeeded;
    model.failedTaskCount = summary.tasks.failed;
    model.canceledTaskCount = summary.tasks.canceled;
    model.committedArtifactCount = summary.committedArtifactCount;
    model.datasetCount = summary.datasetCount;
    model.datasetVersionCount = summary.datasetVersionCount;
    model.datasetSnapshotCount = summary.datasetSnapshotCount;
    model.modelPackageCount = summary.modelPackageCount;
    model.verifiedModelPackageCount = summary.verifiedModelPackageCount;
    model.workflowRunCount = summary.workflowRunCount;
    model.evidenceAvailableWorkflowCount = summary.evidenceAvailableWorkflowCount;
    model.evidencePendingWorkflowCount = summary.evidencePendingWorkflowCount;

    viewModel_ = model;
    lastError_.clear();
    emit summaryChanged();
    return true;
}

void ProjectSummaryPresenter::clear()
{
    viewModel_ = ProjectSummaryViewModel();
    lastError_.clear();
    emit summaryChanged();
}

bool ProjectSummaryPresenter::available() const { return viewModel_.available; }
qint64 ProjectSummaryPresenter::taskCount() const { return viewModel_.taskCount; }
qint64 ProjectSummaryPresenter::datasetCount() const { return viewModel_.datasetCount; }
qint64 ProjectSummaryPresenter::datasetSnapshotCount() const { return viewModel_.datasetSnapshotCount; }
qint64 ProjectSummaryPresenter::modelPackageCount() const { return viewModel_.modelPackageCount; }
qint64 ProjectSummaryPresenter::committedArtifactCount() const { return viewModel_.committedArtifactCount; }
QString ProjectSummaryPresenter::lastError() const { return lastError_; }
const ProjectSummaryViewModel& ProjectSummaryPresenter::viewModel() const { return viewModel_; }
