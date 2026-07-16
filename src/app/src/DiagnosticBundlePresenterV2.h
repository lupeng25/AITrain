#pragma once

#include "aitrain/v2/ProjectQueryServiceV2.h"

#include <QObject>

struct DiagnosticBundleViewModelV2 final {
    QString taskId;
    QString state;
    QString diagnosticsArtifactId;
    QString evidenceArtifactId;
    QString failureCode;
    QString failureMessage;
};

// 诊断面板只按 TaskId 读取 V2 Query 快照；不消费 Worker payload，也不暴露磁盘路径。
class DiagnosticBundlePresenterV2 final : public QObject {
    Q_OBJECT

public:
    explicit DiagnosticBundlePresenterV2(
        const aitrain::v2::ProjectQueryServiceV2* queryService,
        QObject* parent = nullptr);

    bool selectTask(const QString& taskIdText);
    void clear();
    const DiagnosticBundleViewModelV2& viewModel() const;
    QString lastError() const;

signals:
    void changed();
    void queryFailed(const QString& error);

private:
    const aitrain::v2::ProjectQueryServiceV2* queryService_ = nullptr;
    DiagnosticBundleViewModelV2 viewModel_;
    QString lastError_;
};
