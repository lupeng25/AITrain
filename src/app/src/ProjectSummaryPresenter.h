#pragma once

#include "aitrain/workflow\ProjectQueryService.h"

#include <QObject>

// “总览 / 项目”页面可消费的项目级只读 ViewModel。
// 只保留聚合计数，不包含 Artifact 路径或 Worker payload。
struct ProjectSummaryViewModel final {
    bool available = false;
    qint64 taskCount = 0;
    qint64 activeTaskCount = 0;
    qint64 succeededTaskCount = 0;
    qint64 failedTaskCount = 0;
    qint64 canceledTaskCount = 0;
    qint64 committedArtifactCount = 0;
    qint64 datasetCount = 0;
    qint64 datasetVersionCount = 0;
    qint64 datasetSnapshotCount = 0;
    qint64 modelPackageCount = 0;
    qint64 verifiedModelPackageCount = 0;
    qint64 workflowRunCount = 0;
    qint64 evidenceAvailableWorkflowCount = 0;
    qint64 evidencePendingWorkflowCount = 0;
};

class ProjectSummaryPresenter final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool available READ available NOTIFY summaryChanged)
    Q_PROPERTY(qint64 taskCount READ taskCount NOTIFY summaryChanged)
    Q_PROPERTY(qint64 datasetCount READ datasetCount NOTIFY summaryChanged)
    Q_PROPERTY(qint64 datasetSnapshotCount READ datasetSnapshotCount NOTIFY summaryChanged)
    Q_PROPERTY(qint64 modelPackageCount READ modelPackageCount NOTIFY summaryChanged)
    Q_PROPERTY(qint64 committedArtifactCount READ committedArtifactCount NOTIFY summaryChanged)

public:
    explicit ProjectSummaryPresenter(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    bool refresh();
    void clear();

    bool available() const;
    qint64 taskCount() const;
    qint64 datasetCount() const;
    qint64 datasetSnapshotCount() const;
    qint64 modelPackageCount() const;
    qint64 committedArtifactCount() const;
    QString lastError() const;
    const ProjectSummaryViewModel& viewModel() const;

signals:
    void summaryChanged();
    void queryFailed(const QString& error);

private:
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    ProjectSummaryViewModel viewModel_;
    QString lastError_;
};
