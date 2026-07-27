#pragma once

#include "MainWindowSupport.h"
#include "aitrain/workflow/ProjectQueryService.h"

#include <QObject>

class TaskRuntimeController;

namespace aitrain {
class ProjectWorkspace;
}

// 项目 Session 的唯一 GUI 应用层协调器。它拥有 prepare/activate 状态，
// 但不拥有 SQL 连接；QSqlDatabase 始终由 ProjectWorkspace/ProjectDatabase 管理。
class ProjectSessionController final : public QObject {
    Q_OBJECT

public:
    explicit ProjectSessionController(TaskRuntimeController* taskRuntime,
        QObject* parent = nullptr);

    bool request(aitrain_app::ProjectSessionOperation operation,
        const QString& displayName, const QString& projectRoot,
        QString* error = nullptr);
    bool isBusy() const;
    bool isOpen() const;
    quint64 generation() const;
    QString currentRoot() const;
    QString currentDisplayName() const;
    aitrain::ProjectWorkspace* workspace();
    const aitrain::ProjectQueryService* queryService() const;

signals:
    void busyChanged(bool busy);
    void preparing(const QString& displayName, const QString& projectRoot);
    void activated(const QString& displayName, const QString& canonicalRoot,
        quint64 generation);
    void failed(const QString& message);

private:
    void finish(aitrain_app::ProjectSessionOperation operation,
        const QString& displayName, const QString& projectRoot,
        quint64 generation,
        const aitrain_app::ProjectOpenProbeResult& result);
    void setBusy(bool busy);

    aitrain::ProjectWorkspace workspace_;
    aitrain::ProjectQueryService queryService_;
    TaskRuntimeController* taskRuntime_ = nullptr;
    bool busy_ = false;
    quint64 generation_ = 0;
    QString pendingRoot_;
    QString currentRoot_;
    QString currentDisplayName_;
};
