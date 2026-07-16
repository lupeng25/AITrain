#pragma once

#include "aitrain/workflow\ProjectQueryService.h"

#include <QObject>

struct EnvironmentCheckViewModel final {
    QString taskId;
    QString state;
    QString reportArtifactId;
    QString evidenceArtifactId;
    QJsonObject report;
};

// 环境页只按 TaskId 查询已提交  报告；不消费 Worker 业务 payload，
// 不暴露临时或 Artifact 磁盘路径。
class EnvironmentCheckPresenter final : public QObject {
    Q_OBJECT

public:
    explicit EnvironmentCheckPresenter(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    bool selectTask(const QString& taskIdText);
    void clear();
    const EnvironmentCheckViewModel& viewModel() const;
    QString lastError() const;

signals:
    void changed();
    void queryFailed(const QString& error);

private:
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    EnvironmentCheckViewModel viewModel_;
    QString lastError_;
};
