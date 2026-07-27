#pragma once

#include "EnvironmentCheckPresenter.h"

#include <QObject>

class EnvironmentWorkspacePage;
class TaskRuntimeController;

class EnvironmentPageController final : public QObject
{
    Q_OBJECT

public:
    EnvironmentPageController(const aitrain::ProjectQueryService* queryService,
        TaskRuntimeController* taskRuntime, QObject* parent = nullptr);

    void attach(EnvironmentWorkspacePage* page);
    void setProjectContext(bool projectOpen, const QString& projectRoot);
    void setWorkerExecutable(const QString& executable);
    bool selectTask(const QString& taskId);
    void clear();
    void runCheck();
    QJsonObject report() const;
    QString selectedTaskId() const;

signals:
    void changed();
    void taskStarted(const QString& taskId, const QString& workflowKind);
    void runStarted();

private:
    void render();

    EnvironmentCheckPresenter presenter_;
    TaskRuntimeController* taskRuntime_ = nullptr;
    EnvironmentWorkspacePage* page_ = nullptr;
    bool projectOpen_ = false;
    QString projectRoot_;
    QString workerExecutable_;
};
