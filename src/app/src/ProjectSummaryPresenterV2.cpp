#include "ProjectSummaryPresenterV2.h"

ProjectSummaryPresenterV2::ProjectSummaryPresenterV2(
    const aitrain::v2::ProjectQueryServiceV2* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("ProjectSummaryPresenterV2"));
}

bool ProjectSummaryPresenterV2::refresh()
{
    aitrain::v2::ProjectSummaryReadModelV2 summary;
    QString error;
    if (!queryService_ || !queryService_->projectSummary(&summary, &error)) {
        viewModel_ = ProjectSummaryViewModelV2();
        lastError_ = error.isEmpty()
            ? QStringLiteral("无法读取 V2 项目汇总。")
            : error;
        emit summaryChanged();
        emit queryFailed(lastError_);
        return false;
    }

    ProjectSummaryViewModelV2 model;
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

void ProjectSummaryPresenterV2::clear()
{
    viewModel_ = ProjectSummaryViewModelV2();
    lastError_.clear();
    emit summaryChanged();
}

bool ProjectSummaryPresenterV2::available() const { return viewModel_.available; }
qint64 ProjectSummaryPresenterV2::taskCount() const { return viewModel_.taskCount; }
qint64 ProjectSummaryPresenterV2::datasetCount() const { return viewModel_.datasetCount; }
qint64 ProjectSummaryPresenterV2::datasetSnapshotCount() const { return viewModel_.datasetSnapshotCount; }
qint64 ProjectSummaryPresenterV2::modelPackageCount() const { return viewModel_.modelPackageCount; }
qint64 ProjectSummaryPresenterV2::committedArtifactCount() const { return viewModel_.committedArtifactCount; }
QString ProjectSummaryPresenterV2::lastError() const { return lastError_; }
const ProjectSummaryViewModelV2& ProjectSummaryPresenterV2::viewModel() const { return viewModel_; }
