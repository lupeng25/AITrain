#include "ModelRegistryPageController.h"

#include "ModelRegistryPage.h"
#include "TaskRuntimeController.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonDocument>
#include <QMessageBox>

ModelRegistryPageController::ModelRegistryPageController(
    const aitrain::ProjectQueryService* queryService,
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , presenter_(queryService, this)
    , taskRuntime_(taskRuntime)
{
    connect(&presenter_, &ModelRegistryPresenter::modelPackagesChanged,
        this, [this]() {
            render();
            emit packagesChanged();
        });
}

void ModelRegistryPageController::attach(ModelRegistryWorkspacePage* page)
{
    page_ = page;
    connect(page_, &ModelRegistryWorkspacePage::refreshRequested,
        this, &ModelRegistryPageController::refresh);
    connect(page_, &ModelRegistryWorkspacePage::browseSourceRequested,
        this, &ModelRegistryPageController::browseSource);
    connect(page_, &ModelRegistryWorkspacePage::browseManifestRequested,
        this, &ModelRegistryPageController::browseManifest);
    connect(page_, &ModelRegistryWorkspacePage::importRequested,
        this, &ModelRegistryPageController::importModel);
    connect(page_, &ModelRegistryWorkspacePage::useForRuntimeRequested,
        this, &ModelRegistryPageController::runtimeModelRequested);
    render();
}

void ModelRegistryPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
    if (!projectOpen_) {
        presenter_.clear();
    }
}

void ModelRegistryPageController::setWorkerExecutable(const QString& executable)
{
    workerExecutable_ = executable;
}

void ModelRegistryPageController::refresh()
{
    if (!projectOpen_) {
        presenter_.clear();
        render();
        return;
    }
    presenter_.refresh({50, {}});
}

void ModelRegistryPageController::render()
{
    if (!page_) {
        return;
    }
    QString status;
    if (!projectOpen_) {
        status = tr("请先打开项目。");
    } else if (!presenter_.lastError().isEmpty()) {
        status = tr("读取模型包目录失败：%1").arg(presenter_.lastError());
    } else {
        status = tr("已登记模型包：%1。模型库只展示无路径 Manifest 与 lineage。")
            .arg(presenter_.modelPackageCount());
    }
    page_->renderPackages(presenter_.modelPackages(), status);
}

void ModelRegistryPageController::browseSource()
{
    const QString path = QFileDialog::getOpenFileName(
        page_, tr("选择待导入模型"), projectRoot_,
        tr("模型文件 (*.onnx);;所有文件 (*.*)"));
    if (!path.isEmpty()) {
        page_->setSourceFilePath(path);
    }
}

void ModelRegistryPageController::browseManifest()
{
    const QString path = QFileDialog::getOpenFileName(
        page_, tr("选择 Manifest 草稿"), projectRoot_,
        tr("JSON 文件 (*.json);;所有文件 (*.*)"));
    if (!path.isEmpty()) {
        page_->setManifestFilePath(path);
    }
}

void ModelRegistryPageController::importModel()
{
    if (taskRuntime_->isRunning()) {
        QMessageBox::warning(page_, tr("模型导入"),
            tr("Worker 正在执行任务，稍后再导入模型。"));
        return;
    }
    const QString sourcePath = page_->sourceFilePath();
    const QString manifestPath = page_->manifestFilePath();
    if (!projectOpen_ || !QFileInfo(sourcePath).isFile()
        || !QFileInfo(manifestPath).isFile()) {
        QMessageBox::warning(page_, tr("模型导入"),
            tr("请先打开项目，并选择常规模型文件和用户确认的 Manifest 草稿 JSON。"));
        return;
    }
    QFile manifestFile(manifestPath);
    if (!manifestFile.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(page_, tr("模型导入"),
            tr("无法读取 Manifest 草稿：%1").arg(manifestPath));
        return;
    }
    const QJsonDocument document = QJsonDocument::fromJson(manifestFile.readAll());
    if (!document.isObject()) {
        QMessageBox::warning(page_, tr("模型导入"),
            tr("Manifest 草稿必须是 JSON 对象。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::ModelImportCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.sourceFilePath = sourcePath;
    command.manifestDraft = document.object();
    QString error;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("模型导入"), error);
        return;
    }
    importInProgress_ = true;
    page_->setImportStatus(
        tr("正在导入模型并计算 SHA-256：%1")
            .arg(QDir::toNativeSeparators(sourcePath)));
    emit taskStarted(taskId.toString(), QStringLiteral("model_import"));
    emit importStarted();
}

void ModelRegistryPageController::finishImport(
    bool succeeded, const QString& message)
{
    if (!importInProgress_) {
        return;
    }
    importInProgress_ = false;
    if (page_) {
        page_->setImportStatus(succeeded
            ? tr("模型导入完成。")
            : tr("模型导入失败：%1").arg(message));
    }
    refresh();
}

const QVector<ModelPackageListItem>& ModelRegistryPageController::packages() const
{
    return presenter_.modelPackages();
}
