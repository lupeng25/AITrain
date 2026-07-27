#include "RuntimeDeliveryPageController.h"

#include "TaskRuntimeController.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QJsonObject>
#include <QMessageBox>

RuntimeDeliveryPageController::RuntimeDeliveryPageController(
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , taskRuntime_(taskRuntime)
    , matrix_(aitrain::EnvironmentSnapshot::capture())
{
}

void RuntimeDeliveryPageController::attach(
    RuntimeDeliveryWorkspacePage* page)
{
    page_ = page;
    connect(page_, &RuntimeDeliveryWorkspacePage::modelSelectionChanged,
        this, &RuntimeDeliveryPageController::evaluateRoutes);
    connect(page_, &RuntimeDeliveryWorkspacePage::runRequested,
        this, &RuntimeDeliveryPageController::run);
    page_->setModelPackages(packages_);
}

void RuntimeDeliveryPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
}

void RuntimeDeliveryPageController::setWorkerExecutable(
    const QString& executable)
{
    workerExecutable_ = executable;
}

void RuntimeDeliveryPageController::setModelPackages(
    const QVector<ModelPackageListItem>& packages)
{
    packages_ = packages;
    if (page_) {
        page_->setModelPackages(packages_);
        evaluateRoutes(RuntimeDeliveryMode::DeploymentValidation,
            page_->selectedModelPackageId(
                RuntimeDeliveryMode::DeploymentValidation));
        evaluateRoutes(RuntimeDeliveryMode::InferenceValidation,
            page_->selectedModelPackageId(
                RuntimeDeliveryMode::InferenceValidation));
    }
}

void RuntimeDeliveryPageController::refreshEnvironment()
{
    matrix_ = aitrain::RuntimeCapabilityMatrix(
        aitrain::EnvironmentSnapshot::capture());
    if (!page_) {
        return;
    }
    evaluateRoutes(RuntimeDeliveryMode::DeploymentValidation,
        page_->selectedModelPackageId(
            RuntimeDeliveryMode::DeploymentValidation));
    evaluateRoutes(RuntimeDeliveryMode::InferenceValidation,
        page_->selectedModelPackageId(
            RuntimeDeliveryMode::InferenceValidation));
}

const ModelPackageListItem* RuntimeDeliveryPageController::findPackage(
    const QString& id) const
{
    for (const ModelPackageListItem& package : packages_) {
        if (package.modelPackageId == id) {
            return &package;
        }
    }
    return nullptr;
}

void RuntimeDeliveryPageController::evaluateRoutes(
    RuntimeDeliveryMode mode, const QString& modelPackageId)
{
    if (!page_) {
        return;
    }
    const ModelPackageListItem* package = findPackage(modelPackageId);
    QStringList available;
    QStringList reasons;
    if (package) {
        QStringList seen;
        for (const QString& route : package->runtimeRoutes) {
            if (seen.contains(route)) {
                continue;
            }
            seen.append(route);
            const aitrain::RuntimeCapability capability = matrix_.query(
                {package->modelFamily, route});
            const bool executable =
                capability.executionAuthority
                    == aitrain::RuntimeExecutionAuthority::AitrainCpp
                && capability.productState
                    == aitrain::RuntimeProductState::Supported
                && capability.localReadiness
                    == aitrain::RuntimeLocalReadiness::Available;
            if (executable) {
                available.append(route);
            }
            reasons.append(QStringLiteral("%1：%2")
                .arg(route, capability.message));
        }
    }
    page_->setRouteEvaluation(mode, available, reasons);
}

void RuntimeDeliveryPageController::run(RuntimeDeliveryMode mode)
{
    if (taskRuntime_->isRunning()) {
        QMessageBox::warning(page_, tr("Runtime Delivery"),
            tr("Worker 正在执行任务，稍后再运行交付工作流。"));
        return;
    }
    const RuntimeDeliveryFormData form = page_->formData(mode);
    aitrain::ModelPackageId modelPackageId;
    aitrain::DatasetId datasetId;
    aitrain::DatasetVersionId datasetVersionId;
    aitrain::SnapshotId snapshotId;
    aitrain::ArtifactId snapshotArtifactId;
    QString error;
    if (!projectOpen_ || projectRoot_.isEmpty()
        || !aitrain::ModelPackageId::parse(
            form.modelPackageId, &modelPackageId, &error)
        || !aitrain::DatasetId::parse(
            form.sampleDatasetId, &datasetId, &error)
        || !aitrain::DatasetVersionId::parse(
            form.sampleDatasetVersionId, &datasetVersionId, &error)
        || !aitrain::SnapshotId::parse(
            form.sampleSnapshotId, &snapshotId, &error)
        || !aitrain::ArtifactId::parse(
            form.sampleSnapshotArtifactId, &snapshotArtifactId, &error)
        || form.sampleRelativePath.isEmpty()
        || form.runtimeRoute.isEmpty()) {
        QMessageBox::warning(page_, tr("Runtime Delivery"),
            tr("请选择模型包和当前可执行 Runtime 路线，并填写完整的样本 Snapshot 身份及 Artifact 内相对路径。"));
        return;
    }
    const ModelPackageListItem* package = findPackage(form.modelPackageId);
    if (!package) {
        QMessageBox::warning(page_, tr("Runtime Delivery"),
            tr("模型包目录已变化，请刷新后重试。"));
        return;
    }
    const aitrain::RuntimeCapability capability = matrix_.query(
        {package->modelFamily, form.runtimeRoute});
    if (capability.executionAuthority
            != aitrain::RuntimeExecutionAuthority::AitrainCpp
        || capability.productState != aitrain::RuntimeProductState::Supported
        || capability.localReadiness
            != aitrain::RuntimeLocalReadiness::Available) {
        evaluateRoutes(mode, form.modelPackageId);
        QMessageBox::warning(page_, tr("Runtime Delivery"),
            tr("所选 Runtime 路线当前已不可执行：%1")
                .arg(capability.message));
        return;
    }

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::RuntimeDeliveryCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.modelPackageId = modelPackageId.toString();
    command.runtimeRoute = form.runtimeRoute;
    command.sampleDatasetId = datasetId.toString();
    command.sampleDatasetVersionId = datasetVersionId.toString();
    command.sampleSnapshotId = snapshotId.toString();
    command.sampleSnapshotArtifactId = snapshotArtifactId.toString();
    command.sampleRelativePath = form.sampleRelativePath;
    command.options = QJsonObject{
        {QStringLiteral("benchmarkWarmup"), 3},
        {QStringLiteral("benchmarkIterations"), 20}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("Runtime Delivery"), error);
        return;
    }
    page_->setRunning(mode);
    emit taskStarted(taskId.toString(), QStringLiteral("runtime_delivery"));
    emit runStarted();
}

void RuntimeDeliveryPageController::selectModelPackageForInference(
    const QString& modelPackageId)
{
    if (page_) {
        page_->selectModelPackageForInference(modelPackageId);
        showTab(1);
    }
}

void RuntimeDeliveryPageController::showTab(int tabIndex)
{
    if (page_) {
        page_->showTab(tabIndex);
    }
}
