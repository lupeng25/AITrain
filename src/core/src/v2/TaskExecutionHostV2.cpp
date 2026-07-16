#include "aitrain/v2/TaskExecutionHostV2.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSaveFile>

namespace aitrain::v2 {

TaskExecutionHostV2::TaskExecutionHostV2(TaskCoordinator* coordinator, ArtifactStoreV2* artifactStore)
    : coordinator_(coordinator)
    , artifactStore_(artifactStore)
{
}

bool TaskExecutionHostV2::start(const QString& capabilityId,
    const QString& taskType,
    const PythonAdapterLaunchV2& launch,
    TaskSnapshot* task,
    QString* error)
{
    if (!coordinator_ || adapterHost_.isRunning()) {
        if (error) {
            *error = QStringLiteral("V2 任务执行 Host 未连接协调器或已有运行任务。");
        }
        return false;
    }
    if (!coordinator_->createAndStartTask(capabilityId, taskType, &activeTask_, error)) {
        return false;
    }
    workflowTerminalHandler_ = {};
    adapterSettledHandler_ = {};
    adapterEventHandler_ = {};
    if (!startAdapter(launch, error)) {
        return false;
    }
    if (task) {
        *task = activeTask_;
    }
    return true;
}

bool TaskExecutionHostV2::startExistingTask(const TaskSnapshot& task,
    const PythonAdapterLaunchV2& launch,
    WorkflowTerminalHandler terminalHandler,
    QString* error,
    AdapterSettledHandler settledHandler,
    AdapterEventHandler eventHandler)
{
    if (!coordinator_ || adapterHost_.isRunning() || !task.id.isValid() || !task.requestId.isValid()
        || task.state != TaskState::Running || !terminalHandler) {
        if (error) *error = QStringLiteral("托管 Workflow Adapter 需要运行中 V2 任务和终态收口回调。");
        return false;
    }
    TaskSnapshot persisted;
    if (!coordinator_->storage() || !coordinator_->storage()->task(task.id, &persisted, error)
        || persisted.requestId != task.requestId || persisted.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("托管 Workflow Adapter 的任务未持久化为 Running。");
        return false;
    }
    activeTask_ = persisted;
    workflowTerminalHandler_ = std::move(terminalHandler);
    adapterSettledHandler_ = std::move(settledHandler);
    adapterEventHandler_ = std::move(eventHandler);
    return startAdapter(launch, error);
}

bool TaskExecutionHostV2::startAdapter(const PythonAdapterLaunchV2& launch, QString* error)
{
    lastSequence_ = 0;
    terminalEventSeen_ = false;
    lastError_.clear();
    artifactBundleId_ = {};
    artifactBundleStagingPath_.clear();
    artifactCandidateManifest_ = {};
    PythonAdapterLaunchV2 normalizedLaunch = launch;
    if (!coordinator_ || !coordinator_->storage()
        || !coordinator_->storage()->lastProtocolSequence(activeTask_.id, &normalizedLaunch.eventSequenceOffset, error)) {
        return false;
    }
    if (!adapterHost_.start(normalizedLaunch, activeTask_.requestId, activeTask_.id,
            [this](const ProtocolEnvelope& event) { consumeAdapterEvent(event); },
            [this](const PythonAdapterExitV2& outcome) { finishAdapter(outcome); }, error)) {
        const QString reason = error && !error->isEmpty()
            ? *error
            : QStringLiteral("V2 Python Adapter Host 无法启动。");
        QString terminalError;
        emitHostTerminal(QStringLiteral("event.failed"), QJsonObject{
            {QStringLiteral("message"), reason},
            {QStringLiteral("failureCode"), failureCodeToString(FailureCode::ProcessCrashed)}}, &terminalError);
        if (!terminalError.isEmpty()) {
            lastError_ = terminalError;
        }
        return false;
    }
    return true;
}

bool TaskExecutionHostV2::requestCancellation(const TaskId& taskId, QString* error)
{
    if (!activeTask_.id.isValid() || activeTask_.id != taskId) {
        if (error) {
            *error = QStringLiteral("V2 Adapter Host 不管理指定任务。");
        }
        return false;
    }
    if (!coordinator_->requestCancellation(taskId, error)) {
        return false;
    }
    if (adapterHost_.requestCancellation(error)) {
        return true;
    }
    const QString reason = error && !error->isEmpty()
        ? *error
        : QStringLiteral("V2 Python Adapter 已不可用，取消请求由 Host 收尾。");
    QString terminalError;
    const bool terminalized = emitHostTerminal(QStringLiteral("event.canceled"), QJsonObject{{QStringLiteral("message"), reason}}, &terminalError);
    if (!terminalized && !terminalError.isEmpty()) {
        lastError_ = terminalError;
    }
    return terminalized;
}

bool TaskExecutionHostV2::managesTask(const TaskId& taskId) const
{
    return activeTask_.id.isValid() && activeTask_.id == taskId;
}

bool TaskExecutionHostV2::isRunning() const
{
    return adapterHost_.isRunning();
}

AdapterEventEndpointV2 TaskExecutionHostV2::adapterEndpoint() const
{
    return adapterHost_.endpoint();
}

QString TaskExecutionHostV2::lastError() const
{
    return lastError_;
}

void TaskExecutionHostV2::consumeAdapterEvent(const ProtocolEnvelope& event)
{
    QString error;
    const bool terminal = event.kind == QStringLiteral("event.succeeded")
        || event.kind == QStringLiteral("event.failed") || event.kind == QStringLiteral("event.canceled");
    ArtifactId outputArtifactId;
    if (event.kind == QStringLiteral("event.succeeded") && !commitArtifactBundle(&outputArtifactId, &error)) {
        ProtocolEnvelope failure = event;
        failure.kind = QStringLiteral("event.failed");
        failure.payload = QJsonObject{
            {QStringLiteral("message"), QStringLiteral("无法原子提交 Python Adapter 产物：%1").arg(error)},
            {QStringLiteral("failureCode"), failureCodeToString(FailureCode::ArtifactIncomplete)}};
        if (!consumeTerminalEvent(failure, {}, &error)) {
            lastError_ = error;
        }
        lastSequence_ = failure.sequence;
        terminalEventSeen_ = true;
        abortArtifactBundle();
        QString ignored;
        adapterHost_.forceTerminate(&ignored);
        return;
    }
    if ((event.kind == QStringLiteral("event.failed") || event.kind == QStringLiteral("event.canceled"))) {
        abortArtifactBundle();
    }
    // 终态回调可能立即派发下一步或释放宿主；先转发适配器原始失败详情，
    // 确保 GUI/Worker 能保留官方后端给出的可诊断信息。
    if (terminal && adapterEventHandler_) {
        adapterEventHandler_(event);
    }
    const bool consumed = terminal
        ? consumeTerminalEvent(event, outputArtifactId, &error)
        : coordinator_->consumeWorkerEvent(event, &error);
    if (!consumed) {
        lastError_ = error;
        // 终态已经由事件服务器验过身份/顺序；Workflow 回调失败时不能在
        // 进程退出后再合成第二个终态事件。
        terminalEventSeen_ = terminal;
        QString ignored;
        adapterHost_.forceTerminate(&ignored);
        return;
    }
    lastSequence_ = event.sequence;
    if (event.kind == QStringLiteral("event.artifact_candidate") && !stageArtifactCandidate(event, &error)) {
        lastError_ = error;
        QString ignored;
        adapterHost_.forceTerminate(&ignored);
        return;
    }
    if (!terminal && adapterEventHandler_) {
        adapterEventHandler_(event);
    }
    terminalEventSeen_ = terminal;
}

bool TaskExecutionHostV2::consumeTerminalEvent(const ProtocolEnvelope& event,
    const ArtifactId& outputArtifactId,
    QString* error)
{
    if (!workflowTerminalHandler_) {
        return coordinator_->consumeWorkerEvent(event, error);
    }
    if (!coordinator_->recordWorkflowTerminalEvent(event, error)) {
        return false;
    }
    return workflowTerminalHandler_(event, outputArtifactId, error);
}

void TaskExecutionHostV2::finishAdapter(const PythonAdapterExitV2& outcome)
{
    if (terminalEventSeen_) {
        if (adapterSettledHandler_) {
            AdapterSettledHandler settled = std::move(adapterSettledHandler_);
            settled();
        }
        return;
    }
    abortArtifactBundle();
    QString error;
    if (outcome.cancelRequested) {
        emitHostTerminal(QStringLiteral("event.canceled"), QJsonObject{
            {QStringLiteral("message"), QStringLiteral("Python Adapter 在取消请求后退出，但未发送终态事件。")},
            {QStringLiteral("force"), outcome.forceTerminated}}, &error);
    } else {
        const QString message = !lastError_.isEmpty()
            ? lastError_
            : (outcome.diagnostic.isEmpty()
            ? QStringLiteral("Python Adapter 退出，但未发送终态事件。")
            : outcome.diagnostic);
        emitHostTerminal(QStringLiteral("event.failed"), QJsonObject{
            {QStringLiteral("message"), message},
            {QStringLiteral("failureCode"), failureCodeToString(FailureCode::ProcessCrashed)}}, &error);
    }
    if (!error.isEmpty()) {
        lastError_ = error;
    }
    if (terminalEventSeen_ && adapterSettledHandler_) {
        AdapterSettledHandler settled = std::move(adapterSettledHandler_);
        settled();
    }
}

bool TaskExecutionHostV2::stageArtifactCandidate(const ProtocolEnvelope& event, QString* error)
{
    if (!artifactStore_ || !coordinator_ || !coordinator_->storage()) {
        if (error) {
            *error = QStringLiteral("V2 Python Adapter 产物候选需要 Artifact Store 和 Storage。");
        }
        return false;
    }
    const QString kind = event.payload.value(QStringLiteral("kind")).toString().trimmed();
    const QString sourcePath = event.payload.value(QStringLiteral("path")).toString();
    const QFileInfo source(sourcePath);
    if (kind.isEmpty() || !source.exists() || !source.isFile() || source.isSymLink()) {
        if (error) {
            *error = QStringLiteral("Python Adapter 产物候选必须是存在的常规文件并具有 kind：%1").arg(sourcePath);
        }
        return false;
    }
    if (!artifactBundleId_.isValid()
        && !artifactStore_->begin(activeTask_.id, QStringLiteral("adapter_output_bundle"), &artifactBundleId_, &artifactBundleStagingPath_, error)) {
        return false;
    }
    const QString safeKind = QString(kind).replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]")), QStringLiteral("_"));
    const QString relativePath = QStringLiteral("%1/%2").arg(safeKind, source.fileName());
    const QString destination = QDir(artifactBundleStagingPath_).filePath(relativePath);
    if (QFileInfo::exists(destination) || !QDir().mkpath(QFileInfo(destination).absolutePath()) || !QFile::copy(source.absoluteFilePath(), destination)) {
        if (error) {
            *error = QStringLiteral("无法复制 Python Adapter 产物候选到 staging：%1").arg(sourcePath);
        }
        return false;
    }
    artifactCandidateManifest_.append(QJsonObject{{QStringLiteral("kind"), kind}, {QStringLiteral("relativePath"), relativePath}});
    return true;
}

bool TaskExecutionHostV2::commitArtifactBundle(ArtifactId* outputArtifactId, QString* error)
{
    if (!artifactBundleId_.isValid()) {
        return true;
    }
    QSaveFile candidatesFile(QDir(artifactBundleStagingPath_).filePath(QStringLiteral("candidates.json")));
    if (!candidatesFile.open(QIODevice::WriteOnly)
        || candidatesFile.write(QJsonDocument(QJsonObject{{QStringLiteral("candidates"), artifactCandidateManifest_}}).toJson(QJsonDocument::Compact)) < 0
        || !candidatesFile.commit()) {
        if (error) {
            *error = QStringLiteral("无法写入 Python Adapter 产物候选清单：%1").arg(candidatesFile.errorString());
        }
        return false;
    }
    QString artifactPath;
    const ArtifactId committedArtifactId = artifactBundleId_;
    if (!artifactStore_->commit(committedArtifactId, activeTask_.id, QStringLiteral("adapter_output_bundle"), artifactBundleStagingPath_,
            coordinator_->storage(), &artifactPath, error, {}, nullptr)) {
        return false;
    }
    artifactBundleId_ = {};
    artifactBundleStagingPath_.clear();
    artifactCandidateManifest_ = {};
    if (outputArtifactId) {
        *outputArtifactId = committedArtifactId;
    }
    return true;
}

void TaskExecutionHostV2::abortArtifactBundle()
{
    if (artifactBundleStagingPath_.isEmpty() || !artifactStore_) {
        return;
    }
    QString ignored;
    artifactStore_->abort(artifactBundleStagingPath_, &ignored);
    artifactBundleId_ = {};
    artifactBundleStagingPath_.clear();
    artifactCandidateManifest_ = {};
}

bool TaskExecutionHostV2::emitHostTerminal(const QString& kind, const QJsonObject& payload, QString* error)
{
    if (!activeTask_.id.isValid()) {
        if (error) {
            *error = QStringLiteral("V2 Adapter Host 没有活动任务，无法写入终态。");
        }
        return false;
    }
    ProtocolEnvelope terminal;
    terminal.messageId = MessageId::create();
    terminal.requestId = activeTask_.requestId;
    terminal.taskId = activeTask_.id;
    terminal.sequence = lastSequence_ + 1;
    terminal.kind = kind;
    terminal.timestamp = QDateTime::currentDateTimeUtc();
    terminal.payload = payload;
    if (!consumeTerminalEvent(terminal, {}, error)) {
        return false;
    }
    lastSequence_ = terminal.sequence;
    terminalEventSeen_ = true;
    return true;
}

} // namespace aitrain::v2
