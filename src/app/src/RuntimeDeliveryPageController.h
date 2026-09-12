#pragma once

#include "ModelRegistryPresenter.h"
#include "RuntimeDeliveryPage.h"
#include "aitrain/runtime/RuntimeCapabilityMatrix.h"

#include <QObject>

class TaskRuntimeController;
struct TaskViewState;

class RuntimeDeliveryPageController final : public QObject
{
    Q_OBJECT

public:
    explicit RuntimeDeliveryPageController(
        TaskRuntimeController* taskRuntime, QObject* parent = nullptr);

    void attach(RuntimeDeliveryWorkspacePage* page);
    void setQueryService(const aitrain::ProjectQueryService* query) { queryService_ = query; }
    void applyTaskViewState(const TaskViewState& state);
    void setProjectContext(bool projectOpen, const QString& projectRoot);
    void setWorkerExecutable(const QString& executable);
    void setModelPackages(const QVector<ModelPackageListItem>& packages);
    void refreshEnvironment();
    void selectModelPackageForInference(const QString& modelPackageId);
    void showTab(int tabIndex);

signals:
    void taskStarted(const QString& taskId, const QString& workflowKind);
    void runStarted();

private:
    void loadResult();
    void evaluateRoutes(RuntimeDeliveryMode mode,
        const QString& modelPackageId);
    void run(RuntimeDeliveryMode mode);
    const ModelPackageListItem* findPackage(const QString& id) const;

    TaskRuntimeController* taskRuntime_ = nullptr;
    RuntimeDeliveryWorkspacePage* page_ = nullptr;
    aitrain::RuntimeCapabilityMatrix matrix_;
    QVector<ModelPackageListItem> packages_;
    bool projectOpen_ = false;
    QString projectRoot_;
    QString workerExecutable_;
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    QString activeTaskId_;
    QString reportArtifactId_;
    QString reportRelativePath_;
    quint64 generation_ = 0;
};
