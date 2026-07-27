#pragma once

#include "ModelRegistryPresenter.h"

#include <QObject>
#include <QString>

class ModelRegistryWorkspacePage;
class TaskRuntimeController;

class ModelRegistryPageController final : public QObject
{
    Q_OBJECT

public:
    ModelRegistryPageController(const aitrain::ProjectQueryService* queryService,
        TaskRuntimeController* taskRuntime, QObject* parent = nullptr);

    void attach(ModelRegistryWorkspacePage* page);
    void setProjectContext(bool projectOpen, const QString& projectRoot);
    void setWorkerExecutable(const QString& executable);
    void refresh();
    void finishImport(bool succeeded, const QString& message);
    const QVector<ModelPackageListItem>& packages() const;

signals:
    void packagesChanged();
    void runtimeModelRequested(const QString& modelPackageId);
    void taskStarted(const QString& taskId, const QString& workflowKind);
    void importStarted();

private:
    void render();
    void browseSource();
    void browseManifest();
    void importModel();

    ModelRegistryPresenter presenter_;
    TaskRuntimeController* taskRuntime_ = nullptr;
    ModelRegistryWorkspacePage* page_ = nullptr;
    bool projectOpen_ = false;
    bool importInProgress_ = false;
    QString projectRoot_;
    QString workerExecutable_;
};
