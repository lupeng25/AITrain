#pragma once

#include "aitrain/v2/ProjectQueryServiceV2.h"

#include <QObject>

struct EnvironmentCheckViewModelV2 final {
    QString taskId;
    QString state;
    QString reportArtifactId;
    QString evidenceArtifactId;
    QJsonObject report;
};

// 环境页只按 TaskId 查询已提交 V2 报告；不消费 Worker 业务 payload，
// 不暴露临时或 Artifact 磁盘路径。
class EnvironmentCheckPresenterV2 final : public QObject {
    Q_OBJECT

public:
    explicit EnvironmentCheckPresenterV2(
        const aitrain::v2::ProjectQueryServiceV2* queryService,
        QObject* parent = nullptr);

    bool selectTask(const QString& taskIdText);
    void clear();
    const EnvironmentCheckViewModelV2& viewModel() const;
    QString lastError() const;

signals:
    void changed();
    void queryFailed(const QString& error);

private:
    const aitrain::v2::ProjectQueryServiceV2* queryService_ = nullptr;
    EnvironmentCheckViewModelV2 viewModel_;
    QString lastError_;
};
