#include "WorkbenchTranslation.h"
#include "RuntimeDeliveryPageController.h"

#include "TaskRuntimeController.h"
#include "ProjectObjectSelectors.h"
#include "ApplicationEventRouter.h"
#include <QBuffer>
#include <QImageReader>
#include <QPointer>
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
    connect(page_, &RuntimeDeliveryWorkspacePage::selectSampleRequested, this, [this]() {
        aitrain_app::DatasetSelection selection;
        if (aitrain_app::selectProjectDataset(page_, queryService_, &selection, true)) page_->setDatasetSelection(selection);
    });
    connect(page_, &RuntimeDeliveryWorkspacePage::taskRequested, this, [this]() { if (!activeTaskId_.isEmpty()) emit page_->openTaskRequested(activeTaskId_); });
    connect(page_, &RuntimeDeliveryWorkspacePage::reportRequested, this, [this]() {
        if (!reportArtifactId_.isEmpty()) aitrain_app::showArtifactReport(page_, queryService_, reportArtifactId_, reportRelativePath_, aitrain_app::workbenchText(QStringLiteral("交付报告")));
    });
}

void RuntimeDeliveryPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    if (projectRoot_ != projectRoot) {
        ++generation_; activeTaskId_.clear(); reportArtifactId_.clear(); reportRelativePath_.clear();
        if (page_) page_->clearContext();
    }
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
                .arg(route, aitrain_app::workbenchText(capability.message)));
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
            tr("请选择模型、当前可执行路线，以及已提交数据版本中的验证样本。"));
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
    activeTaskId_ = taskId.toString();
    ++generation_; reportArtifactId_.clear(); reportRelativePath_.clear();
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

void RuntimeDeliveryPageController::applyTaskViewState(const TaskViewState& state)
{
    if (!page_ || state.taskId != activeTaskId_) return;
    page_->resultSummary->setText(!state.terminal ? aitrain_app::workbenchText(QStringLiteral("验证运行中 · %1%")).arg(state.progress)
        : state.status == QStringLiteral("succeeded") ? aitrain_app::workbenchText(QStringLiteral("验证与交付已完成，正在读取已提交结果。"))
        : aitrain_app::workbenchText(QStringLiteral("验证未完成：%1\n%2")).arg(state.status, state.terminalMessage));
    if (state.terminal) loadResult();
}

void RuntimeDeliveryPageController::loadResult()
{
    if (!queryService_) return;
    aitrain::TaskId task; QString error;
    if (!aitrain::TaskId::parse(activeTaskId_, &task, &error)) return;
    aitrain::TaskReadModel details;
    if (!queryService_->taskDetails(task, &details, &error)) { page_->resultSummary->setText(error); return; }
    auto artifacts = details.artifacts;
    QString cursor = details.artifactNextCursor;
    for (bool more = details.artifactsHasMore; more;) {
        const auto batch = queryService_->taskArtifacts(task, {100, cursor}, &error);
        if (!error.isEmpty()) break; artifacts += batch.items; cursor = batch.nextCursor; more = batch.hasMore;
    }
    QString imageArtifact, imagePath;
    for (const auto& artifact : artifacts) {
        QString fileCursor;
        do {
            const auto files = queryService_->artifactFiles(artifact.id, {100, fileCursor}, &error);
            if (!error.isEmpty()) break;
            for (const auto& file : files.items) {
                if (artifact.kind == QStringLiteral("runtime_delivery_report") && file.relativePath.endsWith(QStringLiteral("delivery_report.json"))) {
                    reportArtifactId_ = artifact.id.toString(); reportRelativePath_ = file.relativePath;
                }
                if (aitrain_app::isImageMember(file.relativePath) && file.relativePath.contains(QStringLiteral("overlay"))) { imageArtifact = artifact.id.toString(); imagePath = file.relativePath; }
            }
            fileCursor = files.hasMore ? files.nextCursor : QString();
        } while (!fileCursor.isEmpty());
    }
    page_->reportButton->setEnabled(!reportArtifactId_.isEmpty());
    if (!reportArtifactId_.isEmpty()) page_->resultSummary->setText(aitrain_app::workbenchText(QStringLiteral("交付报告已提交。查看报告了解本次运行状态、计时和部署检查结论。")));
    if (imageArtifact.isEmpty()) { page_->overlay->setText(aitrain_app::workbenchText(QStringLiteral("本次任务没有已提交的可视化结果；详细原因可在任务记录中查看。"))); return; }
    aitrain::ArtifactId id; if (!aitrain::ArtifactId::parse(imageArtifact, &id, &error)) return;
    const quint64 generation = generation_; QPointer<RuntimeDeliveryPageController> self(this);
    if (!queryService_->artifactFilePreviewAsync(id, imagePath, this,
        [self, generation](bool ok, aitrain::ArtifactFilePreview preview, QString message) {
            if (!self || !self->page_ || self->generation_ != generation) return;
            if (!ok) { self->page_->overlay->setText(message); return; }
            if (preview.truncated) { self->page_->overlay->setText(aitrain_app::workbenchText(QStringLiteral("图像超过 4 MB 预览上限；原文件仍保留在任务产物中。"))); return; }
            QBuffer buffer(&preview.content); buffer.open(QIODevice::ReadOnly); QImageReader reader(&buffer);
            const QSize size = reader.size();
            if (!size.isValid() || qint64(size.width()) * size.height() > 40000000) { self->page_->overlay->setText(aitrain_app::workbenchText(QStringLiteral("图像尺寸超过预览限制，可在任务产物中查看。"))); return; }
            reader.setScaledSize(size.scaled(1200, 900, Qt::KeepAspectRatio));
            const QImage image = reader.read();
            if (image.isNull()) self->page_->overlay->setText(aitrain_app::workbenchText(QStringLiteral("可视化文件无法解码。")));
            else self->page_->overlay->setImage(QPixmap::fromImage(image));
        }, 4 * 1024 * 1024, &error)) page_->overlay->setText(error);
}
