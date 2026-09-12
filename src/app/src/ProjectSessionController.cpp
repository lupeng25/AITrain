#include "WorkbenchTranslation.h"
#include "ProjectSessionController.h"

#include "TaskRuntimeController.h"
#include "aitrain/workflow/ProjectWorkspace.h"

#include <QDir>
#include <QFileInfo>

ProjectSessionController::ProjectSessionController(
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , queryService_(&workspace_)
    , taskRuntime_(taskRuntime)
{
}

bool ProjectSessionController::request(
    aitrain_app::ProjectSessionOperation operation,
    const QString& displayName, const QString& projectRoot, QString* error)
{
    const QString name = displayName.trimmed();
    QString root = QDir::cleanPath(
        QDir::fromNativeSeparators(projectRoot.trimmed()));
    const QFileInfo rootInfo(root);
    if (rootInfo.exists()) {
        const QString canonical = rootInfo.canonicalFilePath();
        if (!canonical.isEmpty()) root = QDir::cleanPath(canonical);
    }
    if (!taskRuntime_) {
        if (error) *error = aitrain_app::workbenchText(QStringLiteral("项目 Session Controller 未初始化。"));
        return false;
    }
    if (taskRuntime_->isRunning()) {
        if (error) *error = aitrain_app::workbenchText(QStringLiteral("ProjectBusy：Worker 正在执行任务。"));
        return false;
    }
    if (busy_) {
        if (error) *error = aitrain_app::workbenchText(QStringLiteral("项目打开操作正在进行。"));
        return false;
    }
    if (name.isEmpty() || root.isEmpty()) {
        if (error) *error = aitrain_app::workbenchText(QStringLiteral("项目名称和目录不能为空。"));
        return false;
    }

    // 当前 canonical root 再次打开视为刷新，不重复获取 Owner Lease。
    if (operation == aitrain_app::ProjectSessionOperation::Open
        && !currentRoot_.isEmpty()
        && QDir::cleanPath(root) == QDir::cleanPath(currentRoot_)
        && workspace_.isOpen()) {
        currentDisplayName_ = name;
        ++generation_;
        emit activated(currentDisplayName_, currentRoot_, generation_);
        return true;
    }

    const quint64 requestGeneration = ++generation_;
    pendingRoot_ = root;
    setBusy(true);
    emit preparing(name, root);
    aitrain_app::probeProjectOpenAsync(this, operation, root,
        [this, operation, name, root, requestGeneration](
            const aitrain_app::ProjectOpenProbeResult& result) {
            finish(operation, name, root, requestGeneration, result);
        });
    return true;
}

bool ProjectSessionController::isBusy() const { return busy_; }
bool ProjectSessionController::isOpen() const
{
    return workspace_.isOpen();
}
quint64 ProjectSessionController::generation() const { return generation_; }
QString ProjectSessionController::currentRoot() const { return currentRoot_; }
QString ProjectSessionController::currentDisplayName() const
{
    return currentDisplayName_;
}

aitrain::ProjectWorkspace* ProjectSessionController::workspace()
{
    return &workspace_;
}

const aitrain::ProjectQueryService* ProjectSessionController::queryService() const
{
    return &queryService_;
}

void ProjectSessionController::finish(
    aitrain_app::ProjectSessionOperation operation,
    const QString& displayName, const QString& projectRoot,
    quint64 generation,
    const aitrain_app::ProjectOpenProbeResult& result)
{
    Q_UNUSED(operation)
    if (!busy_ || generation != generation_ || projectRoot != pendingRoot_) {
        return;
    }
    pendingRoot_.clear();
    setBusy(false);
    if (!result.succeeded || !result.prepared.isValid()) {
        emit failed(result.error.isEmpty()
            ? aitrain_app::workbenchText(QStringLiteral("候选项目工作区预检失败。")) : result.error);
        return;
    }
    QString error;
    if (!workspace_.openPrepared(result.prepared, &error)) {
        emit failed(error.isEmpty()
            ? aitrain_app::workbenchText(QStringLiteral("项目打开凭证已失效，需要重新尝试。")) : error);
        return;
    }
    currentRoot_ = result.prepared.canonicalRoot;
    currentDisplayName_ = displayName;
    emit activated(currentDisplayName_, currentRoot_, generation_);
}

void ProjectSessionController::setBusy(bool busy)
{
    if (busy_ == busy) return;
    busy_ = busy;
    emit busyChanged(busy_);
}
