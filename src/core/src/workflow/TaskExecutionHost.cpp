#include "aitrain/workflow/TaskExecutionHost.h"

#include "aitrain/protocol/ProtocolSanitizer.h"

#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSaveFile>

namespace aitrain {

TaskExecutionHost::TaskExecutionHost(TaskCoordinator* coordinator, ArtifactStore* artifactStore)
    : coordinator_(coordinator)
    , artifactStore_(artifactStore)
{
}

bool TaskExecutionHost::start(const QString& capabilityId,
    const QString& taskType,
    const PythonAdapterLaunch& launch,
    TaskSnapshot* task,
    QString* error)
{
    if (!coordinator_ || adapterHost_.isRunning()) {
        if (error) {
            *error = QStringLiteral(" 任务执行 Host 未连接协调器或已有运行任务。");
        }
        return false;
    }
    if (!coordinator_->createAndStartTask(capabilityId, taskType, &activeTask_, error)) {
        return false;
    }
    workflowTerminalHandler_ = {};
    adapterSettledHandler_ = {};
    adapterEventHandler_ = {};
    if (!startAdapter(launch, true, error)) {
        return false;
    }
    if (task) {
        *task = activeTask_;
    }
    return true;
}

bool TaskExecutionHost::startExistingTask(const TaskSnapshot& task,
    const PythonAdapterLaunch& launch,
    WorkflowTerminalHandler terminalHandler,
    QString* error,
    AdapterSettledHandler settledHandler,
    AdapterEventHandler eventHandler)
{
    if (!coordinator_ || adapterHost_.isRunning() || !task.id.isValid() || !task.requestId.isValid()
        || task.state != TaskState::Running || !terminalHandler) {
        if (error) *error = QStringLiteral("托管 Workflow Adapter 需要运行中  任务和终态收口回调。");
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
    // Existing Workflow steps are completed by the Worker caller when launch
    // preparation/start fails. Do not also synthesize a Core terminal here.
    return startAdapter(launch, false, error);
}

bool TaskExecutionHost::startAdapter(const PythonAdapterLaunch& launch,
    bool terminalizeStartFailure,
    QString* error)
{
    lastSequence_ = 0;
    terminalEventSeen_ = false;
    artifactCandidateRoots_.clear();
    lastError_.clear();
    artifactBundleId_ = {};
    artifactBundleStagingPath_.clear();
    artifactCandidateManifest_ = {};
    PythonAdapterLaunch normalizedLaunch = launch;
    if (!coordinator_ || !coordinator_->storage()
        || !coordinator_->storage()->lastProtocolSequence(activeTask_.id, &normalizedLaunch.eventSequenceOffset, error)) {
        return false;
    }
    lastSequence_ = normalizedLaunch.eventSequenceOffset;
    artifactCandidateRoots_.clear();
    for (const QString& root : normalizedLaunch.artifactCandidateRoots) {
        const QString absoluteRoot = QDir::cleanPath(QFileInfo(root).absoluteFilePath());
        if (!absoluteRoot.isEmpty() && !artifactCandidateRoots_.contains(absoluteRoot)) {
            artifactCandidateRoots_.append(absoluteRoot);
        }
    }
    if (!adapterHost_.start(normalizedLaunch, activeTask_.requestId, activeTask_.id,
            [this](const ProtocolEnvelope& event) { return consumeAdapterEvent(event); },
            [this](const PythonAdapterExit& outcome) { finishAdapter(outcome); }, error)) {
        if (!terminalizeStartFailure) {
            return false;
        }
        const QString reason = error && !error->isEmpty()
            ? *error
            : QStringLiteral(" Python Adapter Host 无法启动。");
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

bool TaskExecutionHost::requestCancellation(const TaskId& taskId, QString* error)
{
    if (!activeTask_.id.isValid() || activeTask_.id != taskId) {
        if (error) {
            *error = QStringLiteral(" Adapter Host 不管理指定任务。");
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
        : QStringLiteral(" Python Adapter 已不可用，取消请求由 Host 收尾。");
    QString terminalError;
    const bool terminalized = emitHostTerminal(QStringLiteral("event.canceled"), QJsonObject{{QStringLiteral("message"), reason}}, &terminalError);
    if (!terminalized && !terminalError.isEmpty()) {
        lastError_ = terminalError;
    }
    return terminalized;
}

bool TaskExecutionHost::managesTask(const TaskId& taskId) const
{
    return activeTask_.id.isValid() && activeTask_.id == taskId;
}

bool TaskExecutionHost::isRunning() const
{
    return adapterHost_.isRunning();
}

AdapterEventEndpoint TaskExecutionHost::adapterEndpoint() const
{
    return adapterHost_.endpoint();
}

QString TaskExecutionHost::lastError() const
{
    return lastError_;
}

bool TaskExecutionHost::consumeAdapterEvent(const ProtocolEnvelope& event)
{
    QString error;
    // 产物候选的原始 path 仅在本函数内部用于受控 staging 校验；进入
    // Core 回调、终态记录和 Worker 转发前统一移除所有物理路径字段。
    ProtocolEnvelope safeEvent = event;
    safeEvent.payload = protocol::redactPhysicalPathFields(event.payload);
    const bool terminal = event.kind == QStringLiteral("event.succeeded")
        || event.kind == QStringLiteral("event.failed") || event.kind == QStringLiteral("event.canceled");
    ArtifactId outputArtifactId;
    bool cancellationRequested = false;
    if (event.kind == QStringLiteral("event.succeeded") && coordinator_ && coordinator_->storage()) {
        TaskSnapshot persisted;
        if (!coordinator_->storage()->task(activeTask_.id, &persisted, &error)) {
            lastError_ = error;
            if (adapterEventHandler_) {
                ProtocolEnvelope diagnostic;
                diagnostic.kind = QStringLiteral("event.log");
                diagnostic.payload = QJsonObject{{QStringLiteral("message"),
                    QStringLiteral("读取任务状态失败：%1").arg(error)}};
                adapterEventHandler_(diagnostic);
            }
            QString ignored;
            adapterHost_.forceTerminate(&ignored);
            return false;
        }
        cancellationRequested = persisted.state == TaskState::CancelRequested;
    }
    if (event.kind == QStringLiteral("event.succeeded") && !cancellationRequested
        && !commitArtifactBundle(&outputArtifactId, &error)) {
        ProtocolEnvelope failure = event;
        failure.kind = QStringLiteral("event.failed");
        failure.payload = QJsonObject{
            {QStringLiteral("message"), QStringLiteral("无法原子提交 Python Adapter 产物：%1").arg(error)},
            {QStringLiteral("failureCode"), failureCodeToString(FailureCode::ArtifactIncomplete)}};
        const bool failureAccepted = consumeTerminalEvent(failure, {}, &error);
        if (!failureAccepted) {
            lastError_ = error;
            if (adapterEventHandler_) {
                ProtocolEnvelope diagnostic;
                diagnostic.kind = QStringLiteral("event.log");
                diagnostic.payload = QJsonObject{{QStringLiteral("message"),
                    QStringLiteral("提交失败终态失败：%1").arg(error)}};
                adapterEventHandler_(diagnostic);
            }
        } else if (adapterEventHandler_) {
            ProtocolEnvelope safeFailure = failure;
            safeFailure.payload = protocol::redactPhysicalPathFields(failure.payload);
            adapterEventHandler_(safeFailure);
        }
        lastSequence_ = failure.sequence;
        terminalEventSeen_ = true;
        abortArtifactBundle();
        QString ignored;
        adapterHost_.forceTerminate(&ignored);
        return failureAccepted;
    }
    if (cancellationRequested) {
        // 取消优先：Adapter 即使晚到 succeeded，也不得把成功产物提交为任务结果。
        abortArtifactBundle();
    }
    if ((event.kind == QStringLiteral("event.failed") || event.kind == QStringLiteral("event.canceled"))) {
        abortArtifactBundle();
    }
    const bool consumed = terminal
        ? consumeTerminalEvent(safeEvent, outputArtifactId, &error)
        : coordinator_->consumeWorkerEvent(safeEvent, &error);
    if (!consumed) {
        lastError_ = error;
        if (adapterEventHandler_) {
            ProtocolEnvelope diagnostic;
            diagnostic.kind = QStringLiteral("event.log");
            diagnostic.payload = QJsonObject{{QStringLiteral("message"),
                QStringLiteral("Adapter 事件被协调器拒绝：%1").arg(error)}};
            adapterEventHandler_(diagnostic);
        }
        qWarning().noquote() << QStringLiteral("[task adapter event rejected] %1").arg(error);
        QString ignored;
        adapterHost_.forceTerminate(&ignored);
        return false;
    }
    lastSequence_ = event.sequence;
    if (event.kind == QStringLiteral("event.artifact_candidate") && !stageArtifactCandidate(event, &error)) {
        lastError_ = error;
        if (adapterEventHandler_) {
            ProtocolEnvelope diagnostic;
            diagnostic.kind = QStringLiteral("event.log");
            diagnostic.payload = QJsonObject{{QStringLiteral("message"),
                QStringLiteral("Adapter 产物候选被拒绝：%1").arg(error)}};
            adapterEventHandler_(diagnostic);
        }
        qWarning().noquote() << QStringLiteral("[task artifact candidate rejected] %1").arg(error);
        QString ignored;
        adapterHost_.forceTerminate(&ignored);
        return false;
    }
    // 只有 Coordinator/Workflow terminal handler 已持久化接受终态，才向
    // Worker 转发并允许 AdapterEventServer 返回 terminal ACK。
    if (adapterEventHandler_) {
        adapterEventHandler_(safeEvent);
    }
    terminalEventSeen_ = terminalEventSeen_ || terminal;
    return true;
}

bool TaskExecutionHost::consumeTerminalEvent(const ProtocolEnvelope& event,
    const ArtifactId& outputArtifactId,
    QString* error)
{
    if (!workflowTerminalHandler_) {
        return coordinator_->consumeWorkerEvent(event, error);
    }
    if (!coordinator_->recordWorkflowTerminalEvent(event, error)) {
        return false;
    }
    ProtocolEnvelope effectiveEvent = event;
    TaskSnapshot persisted;
    if (coordinator_->storage()->task(event.taskId, &persisted, error)
        && persisted.state == TaskState::CancelRequested
        && event.kind != QStringLiteral("event.canceled")) {
        effectiveEvent.kind = QStringLiteral("event.canceled");
        effectiveEvent.payload = QJsonObject{
            {QStringLiteral("message"), QStringLiteral("任务已请求取消，忽略 Adapter 的晚到终态。")}};
        return workflowTerminalHandler_(effectiveEvent, {}, error);
    }
    return workflowTerminalHandler_(effectiveEvent, outputArtifactId, error);
}

void TaskExecutionHost::finishAdapter(const PythonAdapterExit& outcome)
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
    if (adapterEventHandler_) {
        ProtocolEnvelope diagnostic;
        diagnostic.kind = QStringLiteral("event.log");
        diagnostic.payload = QJsonObject{{QStringLiteral("message"),
            QStringLiteral("Adapter exit: %1 (exitCode=%2, terminal=%3)")
                .arg(outcome.diagnostic)
                .arg(outcome.exitCode)
                .arg(outcome.terminalEventSeen ? QStringLiteral("true") : QStringLiteral("false"))}};
        adapterEventHandler_(diagnostic);
    }
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
        const QString diagnostic = QStringLiteral("%1（exitCode=%2, normalExit=%3, forceTerminated=%4）")
            .arg(message).arg(outcome.exitCode).arg(outcome.normalExit ? QStringLiteral("true") : QStringLiteral("false"))
            .arg(outcome.forceTerminated ? QStringLiteral("true") : QStringLiteral("false"));
        emitHostTerminal(QStringLiteral("event.failed"), QJsonObject{
            {QStringLiteral("message"), diagnostic},
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

bool TaskExecutionHost::stageArtifactCandidate(const ProtocolEnvelope& event, QString* error)
{
    if (!artifactStore_ || !coordinator_ || !coordinator_->storage()) {
        if (error) {
            *error = QStringLiteral(" Python Adapter 产物候选需要 Artifact Store 和 Storage。");
        }
        return false;
    }
    const QString kind = event.payload.value(QStringLiteral("kind")).toString().trimmed();
    const QString declaredPath = event.payload.value(QStringLiteral("path")).toString().trimmed();
    const QString declaredRelativePath = event.payload.value(QStringLiteral("relativePath")).toString().trimmed();
    if (kind.isEmpty() || declaredPath.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Python Adapter 产物候选必须包含 kind 和 path。" );
        }
        return false;
    }
    if (artifactCandidateRoots_.isEmpty()) {
        if (error) *error = QStringLiteral("Python Adapter 产物候选缺少受控根目录。" );
        return false;
    }
    const QFileInfo declaredInfo(declaredPath);
    QString sourceCanonical;
    for (const QString& root : artifactCandidateRoots_) {
        const QString rootCanonical = QFileInfo(root).canonicalFilePath();
        if (rootCanonical.isEmpty()) continue;
        const QString resolvedPath = declaredInfo.isAbsolute()
            ? declaredInfo.absoluteFilePath()
            : QDir(rootCanonical).filePath(declaredPath);
        const QFileInfo source(resolvedPath);
        if (!source.exists() || !source.isFile() || source.isSymLink()) continue;
        const QString candidateCanonical = source.canonicalFilePath();
        const QString relativeToRoot = QDir(rootCanonical).relativeFilePath(candidateCanonical);
        const QString normalizedRelative = QDir::fromNativeSeparators(QDir::cleanPath(relativeToRoot));
        if (!candidateCanonical.isEmpty() && normalizedRelative != QStringLiteral("..")
            && !normalizedRelative.startsWith(QStringLiteral("../"))) {
            sourceCanonical = candidateCanonical;
            break;
        }
    }
    if (sourceCanonical.isEmpty()) {
        if (error) *error = QStringLiteral("Python Adapter 产物候选必须位于受控根目录内的常规文件：%1").arg(declaredPath);
        return false;
    }
    if (!artifactBundleId_.isValid()
        && !artifactStore_->begin(activeTask_.id, QStringLiteral("adapter_output_bundle"), &artifactBundleId_, &artifactBundleStagingPath_, error)) {
        return false;
    }
    const QString safeKind = QString(kind).replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]")), QStringLiteral("_"));
    QString relativePath = declaredRelativePath;
    if (relativePath.isEmpty()) {
        relativePath = QStringLiteral("%1/%2").arg(safeKind, QFileInfo(sourceCanonical).fileName());
    }
    relativePath = QDir::fromNativeSeparators(QDir::cleanPath(relativePath));
    const QFileInfo relativeInfo(relativePath);
    if (relativeInfo.isAbsolute() || relativePath == QStringLiteral(".")
        || relativePath == QStringLiteral("..") || relativePath.startsWith(QStringLiteral("../"))
        || relativePath.contains(QStringLiteral("/../"))) {
        if (error) *error = QStringLiteral("Python Adapter 产物 relativePath 无效。");
        return false;
    }
    const QString destination = QDir(artifactBundleStagingPath_).filePath(relativePath);
    if (QFileInfo::exists(destination) || !QDir().mkpath(QFileInfo(destination).absolutePath()) || !QFile::copy(sourceCanonical, destination)) {
        if (error) {
            *error = QStringLiteral("无法复制 Python Adapter 产物候选到 staging：%1").arg(declaredPath);
        }
        return false;
    }
    artifactCandidateManifest_.append(QJsonObject{{QStringLiteral("kind"), kind}, {QStringLiteral("relativePath"), relativePath}});
    return true;
}

bool TaskExecutionHost::commitArtifactBundle(ArtifactId* outputArtifactId, QString* error)
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

void TaskExecutionHost::abortArtifactBundle()
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

bool TaskExecutionHost::emitHostTerminal(const QString& kind, const QJsonObject& payload, QString* error)
{
    if (!activeTask_.id.isValid()) {
        if (error) {
            *error = QStringLiteral(" Adapter Host 没有活动任务，无法写入终态。");
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

} // namespace aitrain
