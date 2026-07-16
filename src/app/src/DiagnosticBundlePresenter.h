#pragma once

#include "aitrain/workflow\ProjectQueryService.h"

#include <QObject>

struct DiagnosticBundleViewModel final {
    QString taskId;
    QString state;
    QString diagnosticsArtifactId;
    QString evidenceArtifactId;
    QString failureCode;
    QString failureMessage;
};

// 诊断面板只按 TaskId 读取  Query 快照；不消费 Worker payload，也不暴露磁盘路径。
class DiagnosticBundlePresenter final : public QObject {
    Q_OBJECT

public:
    explicit DiagnosticBundlePresenter(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    bool selectTask(const QString& taskIdText);
    void clear();
    const DiagnosticBundleViewModel& viewModel() const;
    QString lastError() const;

signals:
    void changed();
    void queryFailed(const QString& error);

private:
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    DiagnosticBundleViewModel viewModel_;
    QString lastError_;
};
