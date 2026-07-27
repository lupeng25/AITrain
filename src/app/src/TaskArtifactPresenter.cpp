#include "TaskArtifactPresenter.h"

#include <QDateTime>

#include <algorithm>
#include <utility>

namespace {

QString localTimeText(const QDateTime& value)
{
    return value.isValid()
        ? value.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))
        : QStringLiteral("--");
}

QString taskStateLabel(aitrain::TaskState state)
{
    switch (state) {
    case aitrain::TaskState::Created: return QStringLiteral("已创建");
    case aitrain::TaskState::Queued: return QStringLiteral("排队中");
    case aitrain::TaskState::Starting: return QStringLiteral("启动中");
    case aitrain::TaskState::Running: return QStringLiteral("运行中");
    case aitrain::TaskState::CancelRequested: return QStringLiteral("取消中");
    case aitrain::TaskState::Succeeded: return QStringLiteral("已完成");
    case aitrain::TaskState::Failed: return QStringLiteral("失败");
    case aitrain::TaskState::Canceled: return QStringLiteral("已取消");
    }
    return QStringLiteral("未知");
}

QString failureCatalogAction(aitrain::FailureCode code)
{
    switch (code) {
    case aitrain::FailureCode::Canceled:
        return QStringLiteral("任务已取消；如需重试，请重新派发同一 Workflow。");
    case aitrain::FailureCode::InvalidRequest:
        return QStringLiteral("检查请求中的登记 ID、格式和结构化参数。");
    case aitrain::FailureCode::InvalidDataset:
        return QStringLiteral("返回数据集页重新校验已登记 Snapshot。");
    case aitrain::FailureCode::ArtifactIncomplete:
        return QStringLiteral("在任务与产物页核对 committed Artifact 清单和哈希。");
    case aitrain::FailureCode::BackendUnsupported:
        return QStringLiteral("检查内置能力矩阵和当前 Profile 的后端边界。");
    case aitrain::FailureCode::RuntimeNotImplemented:
        return QStringLiteral("切换到产品支持的 runtime 路由；当前能力尚未实现。");
    case aitrain::FailureCode::DependencyMissing:
        return QStringLiteral("进入环境页补齐缺失依赖后重试。");
    case aitrain::FailureCode::SdkMissing:
        return QStringLiteral("安装并启用目标 SDK，再重新执行部署验证。");
    case aitrain::FailureCode::HardwareUnsupported:
        return QStringLiteral("当前硬件不在产品支持矩阵内，不能将结果标记为通过。");
    case aitrain::FailureCode::ArtifactIncompatible:
        return QStringLiteral("重新导入带完整 Manifest、入口和哈希的模型包。");
    case aitrain::FailureCode::ProcessCrashed:
        return QStringLiteral("查看诊断 Artifact 和官方日志，确认 Worker/Python 进程退出原因。");
    case aitrain::FailureCode::ProtocolViolation:
        return QStringLiteral("检查 Worker 控制面令牌、任务身份和序列号，不要重放旧帧。");
    case aitrain::FailureCode::Timeout:
        return QStringLiteral("检查环境和输入规模；长任务应通过 Worker 重新派发。");
    case aitrain::FailureCode::InternalError:
        return QStringLiteral("保留 Evidence 和诊断 Artifact，修复后重新运行 Workflow。");
    case aitrain::FailureCode::None:
        return QString();
    }
    return QStringLiteral("查看任务 Evidence 和诊断 Artifact。");
}

} // namespace

TaskArtifactPresenter::TaskArtifactPresenter(
    const aitrain::ProjectQueryService* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("TaskArtifactPresenter"));
}

bool TaskArtifactPresenter::refresh(const aitrain::PageRequest& request)
{
    QString error;
    const aitrain::Page<aitrain::TaskSnapshot> page = queryService_
        ? queryService_->recentTasks(request, &error)
        : aitrain::Page<aitrain::TaskSnapshot>();
    if (!error.isEmpty()) {
        taskRows_.clear();
        emit taskRowsChanged();
        fail(error);
        return false;
    }

    QVector<TaskListItem> rows;
    rows.reserve(page.items.size());
    for (const aitrain::TaskSnapshot& task : page.items) {
        TaskListItem row;
        row.taskId = task.id.toString();
        row.capabilityId = task.capabilityId;
        row.taskType = task.taskType;
        row.state = aitrain::taskStateToString(task.state);
        row.stateLabel = taskStateLabel(task.state);
        row.updatedAt = localTimeText(task.updatedAt);
        row.message = task.failure.isFailure() ? task.failure.message : QString();
        rows.append(row);
    }
    if (request.after.isEmpty()) taskRows_ = rows;
    else taskRows_ += rows;
    nextTaskCursor_ = page.nextCursor;
    hasMoreTasks_ = page.hasMore;
    lastError_.clear();
    emit taskRowsChanged();
    return true;
}

bool TaskArtifactPresenter::loadMore()
{
    return hasMoreTasks_ && refresh({100, nextTaskCursor_});
}

bool TaskArtifactPresenter::hasMoreTasks() const
{
    return hasMoreTasks_;
}

bool TaskArtifactPresenter::selectTask(const QString& taskIdText)
{
    aitrain::TaskId taskId;
    QString error;
    if (!aitrain::TaskId::parse(taskIdText, &taskId, &error)) {
        clearSelection();
        fail(error);
        return false;
    }

    aitrain::TaskReadModel model;
    if (!queryService_ || !queryService_->taskDetails(taskId, &model, &error)) {
        clearSelection();
        fail(error.isEmpty() ? QStringLiteral("无法读取任务详情。") : error);
        return false;
    }

    TaskArtifactDetails details;
    details.taskId = taskId.toString();
    if (model.task.failure.isFailure()) {
        details.failureCode = aitrain::failureCodeToString(model.task.failure.code);
        details.failureAction = model.task.failure.suggestedAction.trimmed();
        if (details.failureAction.isEmpty()) {
            details.failureAction = failureCatalogAction(model.task.failure.code);
        }
    }
    details_ = details;
    appendArtifacts(model.artifacts);
    appendMetrics(model.metrics);
    appendWorkflows(model.workflows);
    artifactCursor_ = model.artifactNextCursor;
    metricCursor_ = model.metricNextCursor;
    workflowCursor_ = model.workflowNextCursor;
    hasMoreArtifacts_ = model.artifactsHasMore;
    hasMoreMetrics_ = model.metricsHasMore;
    hasMoreWorkflows_ = model.workflowsHasMore;

    details_.summary = QStringLiteral(" 任务 %1：%2 / %3 / %4，%5 个已提交产物文件，%6 个指标点，%7 个工作流步骤")
        .arg(details_.taskId.left(8),
            model.task.taskType.isEmpty() ? QStringLiteral("未记录类型") : model.task.taskType,
            model.task.capabilityId.isEmpty() ? QStringLiteral("未记录能力") : model.task.capabilityId,
            taskStateLabel(model.task.state))
        .arg(details_.artifacts.size())
        .arg(details_.metrics.size())
        .arg(details_.workflowSteps.size());
    if (model.task.failure.isFailure()) {
        details_.summary.append(QStringLiteral("\n失败代码：%1\n失败摘要：%2\n建议：%3")
            .arg(details_.failureCode, model.task.failure.message, details_.failureAction));
    }

    lastError_.clear();
    emit detailsChanged();
    return true;
}

void TaskArtifactPresenter::appendArtifacts(
    const QVector<aitrain::ArtifactSnapshot>& artifacts)
{
    for (const aitrain::ArtifactSnapshot& artifact : artifacts) {
        if (artifact.files.isEmpty()) {
            ArtifactFileItem row;
            row.artifactId = artifact.id.toString();
            row.kind = artifact.kind;
            row.createdAt = localTimeText(artifact.createdAt);
            details_.artifacts.append(row);
            continue;
        }
        for (const aitrain::ArtifactFileSnapshot& file : artifact.files) {
            ArtifactFileItem row;
            row.artifactId = artifact.id.toString();
            row.kind = artifact.kind;
            row.relativePath = file.relativePath;
            row.sha256 = file.sha256;
            row.byteCount = file.byteCount;
            row.createdAt = localTimeText(artifact.createdAt);
            details_.artifacts.append(row);
        }
    }
}

void TaskArtifactPresenter::appendMetrics(
    const QVector<aitrain::MetricSnapshot>& metrics)
{
    for (const aitrain::MetricSnapshot& metric : metrics) {
        details_.metrics.append({metric.name, metric.value, localTimeText(metric.occurredAt)});
    }
}

void TaskArtifactPresenter::appendArtifactFiles(
    const aitrain::ArtifactId& artifactId,
    const QVector<aitrain::ArtifactFileSnapshot>& files)
{
    const auto artifact = std::find_if(details_.artifacts.cbegin(), details_.artifacts.cend(),
        [&artifactId](const ArtifactFileItem& item) {
            return item.artifactId == artifactId.toString();
        });
    if (artifact == details_.artifacts.cend()) return;
    for (const aitrain::ArtifactFileSnapshot& file : files) {
        details_.artifactFiles.append({artifact->artifactId, artifact->kind,
            file.relativePath, file.sha256, file.byteCount, artifact->createdAt});
    }
}

void TaskArtifactPresenter::appendWorkflows(
    const QVector<aitrain::WorkflowReadModel>& workflows)
{
    for (const aitrain::WorkflowReadModel& workflow : workflows) {
        for (const aitrain::WorkflowStepSnapshot& step : workflow.steps) {
            details_.workflowSteps.append({workflow.run.id.toString(), workflow.run.templateId,
                step.ordinal, step.kind, aitrain::workflowStepStateToString(step.state),
                step.backend, step.outputArtifactId.toString()});
        }
    }
}

bool TaskArtifactPresenter::loadMoreArtifacts()
{
    aitrain::TaskId taskId;
    QString error;
    if (!hasMoreArtifacts_ || !queryService_
        || !aitrain::TaskId::parse(details_.taskId, &taskId, &error)) return false;
    const auto page = queryService_->taskArtifacts(taskId, {50, artifactCursor_}, &error);
    if (!error.isEmpty()) {
        fail(error);
        return false;
    }
    appendArtifacts(page.items);
    artifactCursor_ = page.nextCursor;
    hasMoreArtifacts_ = page.hasMore;
    lastError_.clear();
    emit detailsChanged();
    return true;
}

bool TaskArtifactPresenter::selectArtifact(const QString& artifactIdText)
{
    aitrain::ArtifactId artifactId;
    QString error;
    if (!queryService_ || !aitrain::ArtifactId::parse(artifactIdText, &artifactId, &error)) {
        fail(error.isEmpty() ? QStringLiteral("Artifact 文件查询服务不可用。") : error);
        return false;
    }
    const auto artifact = std::find_if(details_.artifacts.cbegin(), details_.artifacts.cend(),
        [&artifactIdText](const ArtifactFileItem& item) {
            return item.artifactId == artifactIdText;
        });
    if (artifact == details_.artifacts.cend()) {
        fail(QStringLiteral("所选 Artifact 不属于当前任务页。"));
        return false;
    }
    const auto page = queryService_->artifactFiles(artifactId, {100, {}}, &error);
    if (!error.isEmpty()) {
        fail(error);
        return false;
    }
    details_.selectedArtifactId = artifactIdText;
    details_.artifactFiles.clear();
    appendArtifactFiles(artifactId, page.items);
    artifactFileCursor_ = page.nextCursor;
    hasMoreArtifactFiles_ = page.hasMore;
    lastError_.clear();
    emit detailsChanged();
    return true;
}

bool TaskArtifactPresenter::loadMoreArtifactFiles()
{
    aitrain::ArtifactId artifactId;
    QString error;
    if (!hasMoreArtifactFiles_ || !queryService_
        || !aitrain::ArtifactId::parse(
            details_.selectedArtifactId, &artifactId, &error)) return false;
    const auto page =
        queryService_->artifactFiles(artifactId, {100, artifactFileCursor_}, &error);
    if (!error.isEmpty()) {
        fail(error);
        return false;
    }
    appendArtifactFiles(artifactId, page.items);
    artifactFileCursor_ = page.nextCursor;
    hasMoreArtifactFiles_ = page.hasMore;
    lastError_.clear();
    emit detailsChanged();
    return true;
}

bool TaskArtifactPresenter::loadMoreMetrics()
{
    aitrain::TaskId taskId;
    QString error;
    if (!hasMoreMetrics_ || !queryService_
        || !aitrain::TaskId::parse(details_.taskId, &taskId, &error)) return false;
    const auto page = queryService_->taskMetrics(taskId, {100, metricCursor_}, &error);
    if (!error.isEmpty()) {
        fail(error);
        return false;
    }
    appendMetrics(page.items);
    metricCursor_ = page.nextCursor;
    hasMoreMetrics_ = page.hasMore;
    lastError_.clear();
    emit detailsChanged();
    return true;
}

bool TaskArtifactPresenter::loadMoreWorkflows()
{
    aitrain::TaskId taskId;
    QString error;
    if (!hasMoreWorkflows_ || !queryService_
        || !aitrain::TaskId::parse(details_.taskId, &taskId, &error)) return false;
    const auto page = queryService_->taskWorkflows(taskId, {50, workflowCursor_}, &error);
    if (!error.isEmpty()) {
        fail(error);
        return false;
    }
    appendWorkflows(page.items);
    workflowCursor_ = page.nextCursor;
    hasMoreWorkflows_ = page.hasMore;
    lastError_.clear();
    emit detailsChanged();
    return true;
}

bool TaskArtifactPresenter::hasMoreArtifacts() const { return hasMoreArtifacts_; }
bool TaskArtifactPresenter::hasMoreArtifactFiles() const { return hasMoreArtifactFiles_; }
bool TaskArtifactPresenter::hasMoreMetrics() const { return hasMoreMetrics_; }
bool TaskArtifactPresenter::hasMoreWorkflows() const { return hasMoreWorkflows_; }

bool TaskArtifactPresenter::previewArtifact(const QString& artifactIdText,
    const QString& relativePath,
    aitrain::ArtifactFilePreview* result,
    QString* error) const
{
    if (error) error->clear();
    aitrain::ArtifactId artifactId;
    if (!aitrain::ArtifactId::parse(artifactIdText, &artifactId, error)) {
        return false;
    }
    if (!queryService_) {
        if (error) *error = QStringLiteral("Artifact 预览查询服务不可用。");
        return false;
    }
    return queryService_->artifactFilePreview(artifactId, relativePath, result, 512 * 1024, error);
}

bool TaskArtifactPresenter::previewArtifactAsync(const QString& artifactIdText,
    const QString& relativePath,
    QObject* receiver,
    aitrain::ArtifactFilePreviewCallback callback,
    qint64 maxBytes,
    QString* error) const
{
    if (error) error->clear();
    aitrain::ArtifactId artifactId;
    if (!aitrain::ArtifactId::parse(artifactIdText, &artifactId, error)) {
        return false;
    }
    if (!queryService_) {
        if (error) *error = QStringLiteral("Artifact 异步预览查询服务不可用。");
        return false;
    }
    return queryService_->artifactFilePreviewAsync(artifactId, relativePath,
        receiver, std::move(callback), maxBytes, error);
}

void TaskArtifactPresenter::clearSelection()
{
    details_ = TaskArtifactDetails();
    artifactCursor_.clear();
    artifactFileCursor_.clear();
    metricCursor_.clear();
    workflowCursor_.clear();
    hasMoreArtifacts_ = false;
    hasMoreArtifactFiles_ = false;
    hasMoreMetrics_ = false;
    hasMoreWorkflows_ = false;
    emit detailsChanged();
}

int TaskArtifactPresenter::taskCount() const { return taskRows_.size(); }
QString TaskArtifactPresenter::selectedTaskId() const { return details_.taskId; }
int TaskArtifactPresenter::artifactCount() const { return details_.artifacts.size(); }
int TaskArtifactPresenter::metricCount() const { return details_.metrics.size(); }
int TaskArtifactPresenter::workflowStepCount() const { return details_.workflowSteps.size(); }
QString TaskArtifactPresenter::lastError() const { return lastError_; }
const QVector<TaskListItem>& TaskArtifactPresenter::taskRows() const { return taskRows_; }
const TaskArtifactDetails& TaskArtifactPresenter::details() const { return details_; }

void TaskArtifactPresenter::fail(const QString& error)
{
    lastError_ = error;
    emit queryFailed(error);
}
