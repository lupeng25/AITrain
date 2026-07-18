#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/protocol/ProtocolSanitizer.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/runtime/RuntimeCapabilityMatrix.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QSaveFile>

namespace aitrain {
namespace {

constexpr int kDefaultProbeTimeoutMs = 3000;
constexpr int kDefaultProbeOutputBytes = 16 * 1024;
constexpr int kMaximumProbeOutputBytes = 64 * 1024;

bool canceled(const aitrain::CancellationCallback& callback)
{
    return aitrain::isCancellationRequested(callback);
}

Failure diagnosticsFailure(FailureCode code, const QString& message)
{
    Failure failure;
    failure.code = code;
    failure.message = message.isEmpty() ? QStringLiteral("diagnostics_failed") : message;
    failure.suggestedAction = code == FailureCode::Canceled
        ? QStringLiteral("如需继续，请重新发起 Diagnostics Bundle 。")
        : QStringLiteral("查看失败步骤和 Evidence，修复本机依赖或 Artifact 完整性后重试。");
    failure.occurredAt = QDateTime::currentDateTimeUtc();
    return failure;
}

bool writeFile(const QString& path, const QByteArray& bytes, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) *error = QStringLiteral("diagnostics_directory_create_failed");
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("diagnostics_file_write_failed:%1").arg(file.errorString());
        return false;
    }
    return true;
}

bool hashFile(const QString& path, QString* hash, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("diagnostics_artifact_unreadable");
        return false;
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray bytes = file.read(1024 * 1024);
        if (bytes.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("diagnostics_artifact_read_failed");
            return false;
        }
        digest.addData(bytes);
    }
    *hash = QString::fromLatin1(digest.result().toHex());
    return true;
}

bool verifyArtifact(const ArtifactSnapshot& artifact, const QString& root,
    QString* error)
{
    if (artifact.files.isEmpty() || !QFileInfo(root).isDir()) {
        if (error) *error = QStringLiteral("diagnostics_artifact_incomplete");
        return false;
    }
    for (const ArtifactFileSnapshot& expected : artifact.files) {
        const QString relative = QDir::cleanPath(expected.relativePath);
        const QString path = QDir(root).filePath(relative);
        const QFileInfo info(path);
        QString actualHash;
        if (relative.isEmpty() || relative == QStringLiteral("..")
            || relative.startsWith(QStringLiteral("../")) || QDir::isAbsolutePath(relative)
            || !info.isFile() || info.isSymLink() || info.size() != expected.byteCount
            || !hashFile(path, &actualHash, error) || actualHash != expected.sha256) {
            if (error && error->isEmpty()) *error = QStringLiteral("diagnostics_artifact_tampered");
            return false;
        }
    }
    return true;
}

bool commitFiles(ArtifactStore* store, ProjectStore* storage, const TaskId& taskId,
    const QString& kind, const QVector<QPair<QString, QByteArray>>& files,
    ArtifactId* result, QString* error, const aitrain::CancellationCallback& cancellation)
{
    ArtifactId id;
    QString staging;
    if (!store->begin(taskId, kind, &id, &staging, error)) return false;
    const auto abort = [&]() { QString ignored; store->abort(staging, &ignored); };
    for (const auto& file : files) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("diagnostics_canceled");
            abort();
            return false;
        }
        if (!writeFile(QDir(staging).filePath(file.first), file.second, error)) {
            abort();
            return false;
        }
    }
    QString committedPath;
    bool commitCanceled = false;
    if (!store->commit(id, taskId, kind, staging, storage, &committedPath, error,
            cancellation, &commitCanceled)) {
        if (QFileInfo::exists(staging)) abort();
        return false;
    }
    *result = id;
    return true;
}

QByteArray limited(const QByteArray& value, int limit, bool* truncated)
{
    *truncated = value.size() > limit;
    return value.left(limit);
}

QJsonObject runProbe(const QString& program, const QStringList& arguments,
    int timeoutMs, int outputLimit)
{
    QProcess process;
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.start(program, arguments, QIODevice::ReadOnly);
    QJsonObject result{{QStringLiteral("program"), program}};
    if (!process.waitForStarted(qMin(timeoutMs, 1000))) {
        result.insert(QStringLiteral("status"), QStringLiteral("unavailable"));
        result.insert(QStringLiteral("message"), process.errorString());
        return result;
    }
    const bool finished = process.waitForFinished(timeoutMs);
    if (!finished) {
        process.kill();
        process.waitForFinished(1000);
    }
    bool stdoutTruncated = false;
    bool stderrTruncated = false;
    const QByteArray standardOutput = limited(process.readAllStandardOutput(), outputLimit, &stdoutTruncated);
    const QByteArray standardError = limited(process.readAllStandardError(), outputLimit, &stderrTruncated);
    result.insert(QStringLiteral("status"), finished ? QStringLiteral("finished") : QStringLiteral("timeout"));
    result.insert(QStringLiteral("exitCode"), process.exitCode());
    result.insert(QStringLiteral("stdout"), QString::fromLocal8Bit(standardOutput).trimmed());
    result.insert(QStringLiteral("stderr"), QString::fromLocal8Bit(standardError).trimmed());
    result.insert(QStringLiteral("outputTruncated"), stdoutTruncated || stderrTruncated);
    return result;
}

QJsonObject taskStateCounts(const TaskStateCounts& counts)
{
    return {{QStringLiteral("created"), QString::number(counts.created)},
        {QStringLiteral("queued"), QString::number(counts.queued)},
        {QStringLiteral("starting"), QString::number(counts.starting)},
        {QStringLiteral("running"), QString::number(counts.running)},
        {QStringLiteral("cancelRequested"), QString::number(counts.cancelRequested)},
        {QStringLiteral("succeeded"), QString::number(counts.succeeded)},
        {QStringLiteral("failed"), QString::number(counts.failed)},
        {QStringLiteral("canceled"), QString::number(counts.canceled)}};
}

TaskState terminalState(const WorkflowRunExecutionResult& result)
{
    if (result.state == WorkflowStepState::Succeeded) return TaskState::Succeeded;
    if (result.state == WorkflowStepState::Canceled) return TaskState::Canceled;
    return TaskState::Failed;
}

bool validEnvironmentStatus(const QString& status)
{
    return status == QStringLiteral("ok") || status == QStringLiteral("warning")
        || status == QStringLiteral("missing") || status == QStringLiteral("hardware-blocked");
}

bool validEnvironmentCheck(const QJsonValue& value)
{
    if (!value.isObject()) return false;
    const QJsonObject check = value.toObject();
    return !check.value(QStringLiteral("name")).toString().trimmed().isEmpty()
        && validEnvironmentStatus(check.value(QStringLiteral("status")).toString())
        && check.value(QStringLiteral("message")).isString();
}

bool validEnvironmentFacts(const QJsonObject& facts)
{
    if (facts.value(QStringLiteral("checkedAt")).toString().trimmed().isEmpty()) return false;
    const QJsonArray checks = facts.value(QStringLiteral("checks")).toArray();
    const QJsonObject profiles = facts.value(QStringLiteral("profiles")).toObject();
    if (checks.isEmpty() || profiles.isEmpty()) return false;
    for (const QJsonValue& check : checks) {
        if (!validEnvironmentCheck(check)) return false;
    }
    for (auto it = profiles.constBegin(); it != profiles.constEnd(); ++it) {
        const QJsonObject profile = it.value().toObject();
        if (it.key().trimmed().isEmpty()
            || profile.value(QStringLiteral("title")).toString().trimmed().isEmpty()
            || !validEnvironmentStatus(profile.value(QStringLiteral("status")).toString())
            || !profile.value(QStringLiteral("checks")).isArray()
            || !profile.value(QStringLiteral("repairHints")).isArray()) return false;
        for (const QJsonValue& check : profile.value(QStringLiteral("checks")).toArray()) {
            if (!validEnvironmentCheck(check)) return false;
        }
        for (const QJsonValue& hint : profile.value(QStringLiteral("repairHints")).toArray()) {
            if (!hint.isString()) return false;
        }
    }
    return true;
}

} // namespace

bool ProjectWorkspace::runDiagnosticsWorkflow(const TaskId& taskId,
    const DiagnosticsWorkflowRequest& request, DiagnosticsWorkflowResult* result,
    QString* error, const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !result) {
        if (error) *error = QStringLiteral("运行 Diagnostics Bundle  需要已打开工作区、任务和输出对象。");
        return false;
    }
    *result = {};
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("Diagnostics Bundle  必须依附运行中任务。");
        return false;
    }

    const int taskLimit = qBound(1, request.options.value(QStringLiteral("taskLimit")).toInt(50), 100);
    const int artifactLimit = qBound(1, request.options.value(QStringLiteral("artifactLimitPerTask")).toInt(50), 100);
    const int probeTimeoutMs = qBound(500,
        request.options.value(QStringLiteral("probeTimeoutMs")).toInt(kDefaultProbeTimeoutMs), 10000);
    const int probeOutputBytes = qBound(1024,
        request.options.value(QStringLiteral("probeOutputBytes")).toInt(kDefaultProbeOutputBytes),
        kMaximumProbeOutputBytes);

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("diagnostics");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    QVector<WorkflowStepSnapshot> steps;
    const QStringList kinds{QStringLiteral("CollectDiagnostics"), QStringLiteral("RenderDiagnostics")};
    const QJsonObject parameterSummary{{QStringLiteral("taskLimit"), taskLimit},
        {QStringLiteral("artifactLimitPerTask"), artifactLimit},
        {QStringLiteral("probeTimeoutMs"), probeTimeoutMs},
        {QStringLiteral("probeOutputBytes"), probeOutputBytes}};
    for (int index = 0; index < kinds.size(); ++index) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = index;
        step.kind = kinds.at(index);
        step.backend = QStringLiteral("builtin_diagnostics");
        step.parameterSummary = parameterSummary;
        steps.append(step);
    }
    if (!storage_.createWorkflowRun(workflow, steps, error)) return false;

    QJsonObject facts;
    QJsonObject summary;
    QHash<QString, ArtifactId> outputs;
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult runResult;
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) -> WorkflowStepExecutionResult {
        if (canceled(stepCancellation)) {
            return {WorkflowStepState::Canceled, {}, diagnosticsFailure(FailureCode::Canceled,
                QStringLiteral("diagnostics_canceled_before_probe"))};
        }
        ArtifactId output;
        QString executionError;
        if (step.kind == QStringLiteral("CollectDiagnostics")) {
            ProjectSummarySnapshot projectSummary;
            if (!storage_.projectSummary(&projectSummary, &executionError)) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::InternalError, executionError)};
            }
            QJsonArray taskFacts;
            const QVector<TaskSnapshot> recentTasks = storage_.tasks(taskLimit, &executionError);
            if (!executionError.isEmpty()) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::InternalError, executionError)};
            }
            for (const TaskSnapshot& item : recentTasks) {
                if (canceled(stepCancellation)) {
                    return {WorkflowStepState::Canceled, {}, diagnosticsFailure(FailureCode::Canceled,
                        QStringLiteral("diagnostics_canceled_between_storage_reads"))};
                }
                QJsonArray artifacts;
                const QVector<ArtifactSnapshot> storedArtifacts = storage_.artifactsForTask(item.id, &executionError);
                if (!executionError.isEmpty()) {
                    return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::InternalError, executionError)};
                }
                for (int index = 0; index < storedArtifacts.size() && index < artifactLimit; ++index) {
                    const ArtifactSnapshot& artifact = storedArtifacts.at(index);
                    qint64 bytes = 0;
                    for (const ArtifactFileSnapshot& file : artifact.files) bytes += file.byteCount;
                    artifacts.append(QJsonObject{{QStringLiteral("artifactId"), artifact.id.toString()},
                        {QStringLiteral("kind"), artifact.kind},
                        {QStringLiteral("fileCount"), artifact.files.size()},
                        {QStringLiteral("byteCount"), QString::number(bytes)}});
                }
                taskFacts.append(QJsonObject{{QStringLiteral("taskId"), item.id.toString()},
                    {QStringLiteral("state"), taskStateToString(item.state)},
                    {QStringLiteral("capabilityId"), item.capabilityId},
                    {QStringLiteral("taskType"), item.taskType},
                    {QStringLiteral("failureCode"), failureCodeToString(item.failure.code)},
                    {QStringLiteral("artifacts"), artifacts},
                    {QStringLiteral("artifactListTruncated"), storedArtifacts.size() > artifactLimit}});
            }

            QJsonArray dependencies;
            const QVector<aitrain::RuntimeDependencyCheck> checks =
                aitrain::defaultRuntimeDependencyChecks(QCoreApplication::applicationDirPath());
            for (const aitrain::RuntimeDependencyCheck& check : checks) {
                dependencies.append(QJsonObject{{QStringLiteral("name"), check.name},
                    {QStringLiteral("libraryNames"), QJsonArray::fromStringList(check.libraryNames)},
                    {QStringLiteral("status"), check.status}, {QStringLiteral("message"), check.message}});
            }

            QJsonArray probes;
            if (canceled(stepCancellation)) {
                return {WorkflowStepState::Canceled, {}, diagnosticsFailure(FailureCode::Canceled,
                    QStringLiteral("diagnostics_canceled_before_python_probe"))};
            }
            probes.append(runProbe(QStringLiteral("python"), {QStringLiteral("--version")},
                probeTimeoutMs, probeOutputBytes));
            if (canceled(stepCancellation)) {
                return {WorkflowStepState::Canceled, {}, diagnosticsFailure(FailureCode::Canceled,
                    QStringLiteral("diagnostics_canceled_after_python_probe"))};
            }
            probes.append(runProbe(QStringLiteral("nvidia-smi"),
                {QStringLiteral("--query-gpu=name,compute_cap"), QStringLiteral("--format=csv,noheader")},
                probeTimeoutMs, probeOutputBytes));
            if (canceled(stepCancellation)) {
                return {WorkflowStepState::Canceled, {}, diagnosticsFailure(FailureCode::Canceled,
                    QStringLiteral("diagnostics_canceled_after_nvidia_probe"))};
            }

            facts = protocol::redactPhysicalPathFields(QJsonObject{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("diagnostic_facts")},
                {QStringLiteral("collectedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
                {QStringLiteral("projectSummary"), QJsonObject{
                    {QStringLiteral("tasks"), taskStateCounts(projectSummary.tasks)},
                    {QStringLiteral("committedArtifactCount"), QString::number(projectSummary.committedArtifactCount)},
                    {QStringLiteral("datasetCount"), QString::number(projectSummary.datasetCount)},
                    {QStringLiteral("modelPackageCount"), QString::number(projectSummary.modelPackageCount)},
                    {QStringLiteral("workflowRunCount"), QString::number(projectSummary.workflowRunCount)}}},
                {QStringLiteral("recentTasks"), taskFacts},
                {QStringLiteral("recentTaskListTruncated"), recentTasks.size() >= taskLimit},
                {QStringLiteral("capabilityRegistry"), aitrain::BuiltinCapabilityRegistry::instance().toJson()},
                {QStringLiteral("runtimeCapabilityMatrix"), RuntimeCapabilityMatrix().toJson()},
                {QStringLiteral("runtimeDependencies"), dependencies},
                {QStringLiteral("localProbes"), probes},
                {QStringLiteral("limitations"), QJsonArray{
                    QStringLiteral("外部同步探测在单次 waitForFinished 期间不可中断；取消只在探测前后检查。"),
                    QStringLiteral("探测输出有固定上限，超出内容会截断。"),
                    QStringLiteral("诊断事实只陈述  Storage/Query、内置能力注册表、运行时矩阵和有限本机探测结果。")}}});
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("diagnostic_facts"),
                    {{QStringLiteral("diagnostic_facts.json"), QJsonDocument(facts).toJson(QJsonDocument::Indented)}},
                    &output, &executionError, stepCancellation)) {
                const FailureCode code = canceled(stepCancellation) ? FailureCode::Canceled : FailureCode::ArtifactIncomplete;
                return {code == FailureCode::Canceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, diagnosticsFailure(code, executionError)};
            }
        } else if (step.kind == QStringLiteral("RenderDiagnostics")) {
            ArtifactSnapshot storedFacts;
            if (!storage_.artifact(step.inputArtifactId, &storedFacts, &executionError)
                || storedFacts.kind != QStringLiteral("diagnostic_facts")) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncompatible,
                    executionError.isEmpty() ? QStringLiteral("diagnostics_facts_identity_mismatch") : executionError)};
            }
            const QString factsRoot = artifactStore_->artifactPath(storedFacts.id);
            if (!verifyArtifact(storedFacts, factsRoot, &executionError)) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncompatible, executionError)};
            }
            QFile file(QDir(factsRoot).filePath(QStringLiteral("diagnostic_facts.json")));
            QJsonParseError parseError;
            if (!file.open(QIODevice::ReadOnly)) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncomplete,
                    QStringLiteral("diagnostics_facts_file_unreadable"))};
            }
            const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
            if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncompatible,
                    QStringLiteral("diagnostics_facts_json_invalid"))};
            }
            facts = document.object();
            const QJsonArray probes = facts.value(QStringLiteral("localProbes")).toArray();
            int unavailable = 0;
            for (const QJsonValue& probe : probes) {
                if (probe.toObject().value(QStringLiteral("status")).toString() != QStringLiteral("finished")) ++unavailable;
            }
            summary = {{QStringLiteral("taskCount"), facts.value(QStringLiteral("recentTasks")).toArray().size()},
                {QStringLiteral("probeCount"), probes.size()},
                {QStringLiteral("unavailableProbeCount"), unavailable},
                {QStringLiteral("dependencyCount"), facts.value(QStringLiteral("runtimeDependencies")).toArray().size()}};
            const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("diagnostic_bundle")},
                {QStringLiteral("factsArtifactId"), storedFacts.id.toString()},
                {QStringLiteral("summary"), summary}, {QStringLiteral("facts"), facts}};
            const QJsonObject manifest{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("diagnostic_bundle_manifest")},
                {QStringLiteral("factsArtifactId"), storedFacts.id.toString()},
                {QStringLiteral("files"), QJsonArray{QStringLiteral("diagnostic_bundle.json"),
                    QStringLiteral("diagnostic_summary.md"), QStringLiteral("diagnostic_manifest.json")}}};
            const QString markdown = QStringLiteral("# Diagnostics Bundle \n\n"
                "- 任务事实数：%1\n- 本机探测数：%2\n- 不可用/超时探测数：%3\n- 运行时依赖检查数：%4\n\n"
                "## 限制\n\n- 外部同步探测执行期间不可中断，取消在探测前后检查。\n- 探测输出最多保留 %5 字节。\n")
                .arg(summary.value(QStringLiteral("taskCount")).toInt())
                .arg(summary.value(QStringLiteral("probeCount")).toInt())
                .arg(summary.value(QStringLiteral("unavailableProbeCount")).toInt())
                .arg(summary.value(QStringLiteral("dependencyCount")).toInt())
                .arg(probeOutputBytes);
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("diagnostic_bundle"),
                    {{QStringLiteral("diagnostic_bundle.json"), QJsonDocument(report).toJson(QJsonDocument::Indented)},
                        {QStringLiteral("diagnostic_summary.md"), markdown.toUtf8()},
                        {QStringLiteral("diagnostic_manifest.json"), QJsonDocument(manifest).toJson(QJsonDocument::Indented)}},
                    &output, &executionError, stepCancellation)) {
                const FailureCode code = canceled(stepCancellation) ? FailureCode::Canceled : FailureCode::ArtifactIncomplete;
                return {code == FailureCode::Canceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, diagnosticsFailure(code, executionError)};
            }
        } else {
            return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::InternalError,
                QStringLiteral("diagnostics_unknown_step"))};
        }
        outputs.insert(step.kind, output);
        return {WorkflowStepState::Succeeded, output, {}};
    };

    if (!runner.run(workflow.id, executor, &runResult, error, cancellation)) return false;
    const TaskState finalState = terminalState(runResult);
    Failure finalFailure = runResult.failure;
    if (finalState != TaskState::Succeeded && !finalFailure.isFailure()) {
        finalFailure = diagnosticsFailure(finalState == TaskState::Canceled
            ? FailureCode::Canceled : FailureCode::InternalError, QStringLiteral("diagnostics_missing_terminal_failure"));
    }
    if (finalState == TaskState::Canceled) {
        TaskSnapshot current;
        if (!storage_.task(taskId, &current, error)) return false;
        if (current.state == TaskState::Running && !requestTaskCancellation(taskId, error)) return false;
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, finalState, finalFailure,
            QDateTime::currentDateTimeUtc(), error)) return false;
    EvidenceBundle evidence;
    EvidenceArtifactBundle committedEvidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)) return false;
    evidence.limitations.append(QStringLiteral("Diagnostics  的外部同步探测在单次 waitForFinished 期间不可中断；取消只在探测前后检查。"));
    evidence.limitations.append(QStringLiteral("Diagnostics  对探测输出实施固定上限，超出内容不会进入 Artifact。"));
    if (!commitEvidenceBundle(evidence, &committedEvidence, error)
        || !closeWorkflowTerminalization(workflow.id, error)) return false;

    result->workflowRunId = workflow.id;
    result->terminalState = finalState;
    result->factsArtifactId = outputs.value(QStringLiteral("CollectDiagnostics"));
    result->diagnosticsArtifactId = outputs.value(QStringLiteral("RenderDiagnostics"));
    result->evidenceArtifactId = committedEvidence.artifactId;
    result->summary = summary;
    result->failure = finalFailure;
    return true;
}

bool ProjectWorkspace::runEnvironmentCheckWorkflow(const TaskId& taskId,
    const EnvironmentCheckWorkflowRequest& request,
    EnvironmentCheckWorkflowResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !result) {
        if (error) *error = QStringLiteral("运行 Environment Check  需要已打开工作区、任务和输出对象。");
        return false;
    }
    *result = {};
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("Environment Check  必须依附运行中任务。");
        return false;
    }

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("environment_check");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    QVector<WorkflowStepSnapshot> steps;
    const QStringList kinds{QStringLiteral("ValidateEnvironmentFacts"),
        QStringLiteral("RenderEnvironmentReport")};
    for (int index = 0; index < kinds.size(); ++index) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = index;
        step.kind = kinds.at(index);
        step.backend = QStringLiteral("builtin_environment_check");
        step.parameterSummary = QJsonObject{{QStringLiteral("schemaVersion"), 2}};
        steps.append(step);
    }
    if (!storage_.createWorkflowRun(workflow, steps, error)) return false;

    QHash<QString, ArtifactId> outputs;
    QJsonObject summary;
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult runResult;
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) -> WorkflowStepExecutionResult {
        if (canceled(stepCancellation)) {
            return {WorkflowStepState::Canceled, {}, diagnosticsFailure(FailureCode::Canceled,
                QStringLiteral("environment_check_canceled"))};
        }
        ArtifactId output;
        QString executionError;
        if (step.kind == QStringLiteral("ValidateEnvironmentFacts")) {
            // Environment facts may originate from QProcess/SDK probes.  Strip
            // physical paths before validation and before the facts Artifact is
            // committed; the raw probe payload never crosses the persistence/UI
            // boundary.
            const QJsonObject facts = protocol::redactPhysicalPathFields(request.facts);
            const QByteArray serialized = QJsonDocument(facts).toJson(QJsonDocument::Compact);
            if (!validEnvironmentFacts(facts)
                || serialized.size() > 4 * 1024 * 1024) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::InvalidRequest,
                    QStringLiteral("environment_check_facts_invalid"))};
            }
            QJsonObject storedFacts = facts;
            storedFacts.insert(QStringLiteral("schemaVersion"), 2);
            storedFacts.insert(QStringLiteral("kind"), QStringLiteral("environment_facts"));
            if (!commitFiles(artifactStore_.get(), &storage_, taskId,
                    QStringLiteral("environment_facts"),
                    {{QStringLiteral("environment_facts.json"),
                        QJsonDocument(storedFacts).toJson(QJsonDocument::Indented)}},
                    &output, &executionError, stepCancellation)) {
                const FailureCode code = canceled(stepCancellation)
                    ? FailureCode::Canceled : FailureCode::ArtifactIncomplete;
                return {code == FailureCode::Canceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, diagnosticsFailure(code, executionError)};
            }
        } else if (step.kind == QStringLiteral("RenderEnvironmentReport")) {
            ArtifactSnapshot factsArtifact;
            if (!storage_.artifact(step.inputArtifactId, &factsArtifact, &executionError)
                || factsArtifact.kind != QStringLiteral("environment_facts")) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncompatible,
                    QStringLiteral("environment_check_facts_identity_mismatch"))};
            }
            const QString root = artifactStore_->artifactPath(factsArtifact.id);
            if (!verifyArtifact(factsArtifact, root, &executionError)) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncompatible,
                    executionError)};
            }
            QFile file(QDir(root).filePath(QStringLiteral("environment_facts.json")));
            QJsonParseError parseError;
            if (!file.open(QIODevice::ReadOnly)) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncomplete,
                    QStringLiteral("environment_check_facts_unreadable"))};
            }
            const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
            if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
                return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::ArtifactIncompatible,
                    QStringLiteral("environment_check_facts_json_invalid"))};
            }
            const QJsonObject facts = document.object();
            int ok = 0;
            int warning = 0;
            int missing = 0;
            int hardwareBlocked = 0;
            const auto countStatus = [&](const QString& status) {
                if (status == QStringLiteral("ok")) ++ok;
                else if (status == QStringLiteral("missing")) ++missing;
                else if (status == QStringLiteral("hardware-blocked")) ++hardwareBlocked;
                else ++warning;
            };
            for (const QJsonValue& value : facts.value(QStringLiteral("checks")).toArray())
                countStatus(value.toObject().value(QStringLiteral("status")).toString());
            const QJsonObject profiles = facts.value(QStringLiteral("profiles")).toObject();
            for (auto it = profiles.constBegin(); it != profiles.constEnd(); ++it)
                countStatus(it.value().toObject().value(QStringLiteral("status")).toString());
            summary = QJsonObject{{QStringLiteral("ok"), ok},
                {QStringLiteral("warning"), warning}, {QStringLiteral("missing"), missing},
                {QStringLiteral("hardwareBlocked"), hardwareBlocked},
                {QStringLiteral("profileCount"), profiles.size()}};
            QJsonObject report = facts;
            report.insert(QStringLiteral("schemaVersion"), 2);
            report.insert(QStringLiteral("kind"), QStringLiteral("environment_profiles_report"));
            report.insert(QStringLiteral("factsArtifactId"), factsArtifact.id.toString());
            report.insert(QStringLiteral("summary"), summary);
            const QJsonObject manifest{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("environment_profiles_report_manifest")},
                {QStringLiteral("factsArtifactId"), factsArtifact.id.toString()},
                {QStringLiteral("files"), QJsonArray{QStringLiteral("environment_profiles_report.json"),
                    QStringLiteral("environment_profiles_manifest.json")}}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId,
                    QStringLiteral("environment_profiles_report"),
                    {{QStringLiteral("environment_profiles_report.json"),
                         QJsonDocument(report).toJson(QJsonDocument::Indented)},
                        {QStringLiteral("environment_profiles_manifest.json"),
                         QJsonDocument(manifest).toJson(QJsonDocument::Indented)}},
                    &output, &executionError, stepCancellation)) {
                const FailureCode code = canceled(stepCancellation)
                    ? FailureCode::Canceled : FailureCode::ArtifactIncomplete;
                return {code == FailureCode::Canceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, diagnosticsFailure(code, executionError)};
            }
        } else {
            return {WorkflowStepState::Failed, {}, diagnosticsFailure(FailureCode::InternalError,
                QStringLiteral("environment_check_unknown_step"))};
        }
        outputs.insert(step.kind, output);
        return {WorkflowStepState::Succeeded, output, {}};
    };

    if (!runner.run(workflow.id, executor, &runResult, error, cancellation)) return false;
    const TaskState finalState = terminalState(runResult);
    Failure finalFailure = runResult.failure;
    if (finalState != TaskState::Succeeded && !finalFailure.isFailure()) {
        finalFailure = diagnosticsFailure(finalState == TaskState::Canceled
            ? FailureCode::Canceled : FailureCode::InternalError,
            QStringLiteral("environment_check_missing_terminal_failure"));
    }
    if (finalState == TaskState::Canceled) {
        TaskSnapshot current;
        if (!storage_.task(taskId, &current, error)) return false;
        if (current.state == TaskState::Running && !requestTaskCancellation(taskId, error)) return false;
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, finalState, finalFailure,
            QDateTime::currentDateTimeUtc(), error)) return false;
    EvidenceBundle evidence;
    EvidenceArtifactBundle committedEvidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)) return false;
    evidence.limitations.append(QStringLiteral("环境外部同步探测在单次调用期间不可中断；取消只在探测前后检查。"));
    if (!commitEvidenceBundle(evidence, &committedEvidence, error)
        || !closeWorkflowTerminalization(workflow.id, error)) return false;

    result->workflowRunId = workflow.id;
    result->terminalState = finalState;
    result->factsArtifactId = outputs.value(QStringLiteral("ValidateEnvironmentFacts"));
    result->reportArtifactId = outputs.value(QStringLiteral("RenderEnvironmentReport"));
    result->evidenceArtifactId = committedEvidence.artifactId;
    result->summary = summary;
    result->failure = finalFailure;
    return true;
}

bool ProjectWorkspace::environmentCheckReportForTask(
    const TaskId& taskId, QJsonObject* report, QString* error) const
{
    if (!isOpen() || !taskId.isValid() || !report) {
        if (error) *error = QStringLiteral("读取 Environment Check  报告需要已打开工作区和有效任务。");
        return false;
    }
    const QVector<ArtifactSnapshot> artifacts = storage_.artifactsForTask(taskId, error);
    if (error && !error->isEmpty()) return false;
    for (auto it = artifacts.crbegin(); it != artifacts.crend(); ++it) {
        if (it->kind != QStringLiteral("environment_profiles_report")) continue;
        const QString root = artifactStore_->artifactPath(it->id);
        if (!verifyArtifact(*it, root, error)) return false;
        QFile file(QDir(root).filePath(QStringLiteral("environment_profiles_report.json")));
        QJsonParseError parseError;
        if (!file.open(QIODevice::ReadOnly)) {
            if (error) *error = QStringLiteral("environment_check_report_unreadable");
            return false;
        }
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            if (error) *error = QStringLiteral("environment_check_report_invalid");
            return false;
        }
        const QJsonObject stored = document.object();
        QJsonArray safeChecks;
        for (const QJsonValue& value : stored.value(QStringLiteral("checks")).toArray()) {
            const QJsonObject check = value.toObject();
            safeChecks.append(QJsonObject{{QStringLiteral("name"), check.value(QStringLiteral("name"))},
                {QStringLiteral("status"), check.value(QStringLiteral("status"))},
                {QStringLiteral("message"), check.value(QStringLiteral("message"))}});
        }
        QJsonObject safeProfiles;
        const QJsonObject profiles = stored.value(QStringLiteral("profiles")).toObject();
        for (auto profileIt = profiles.constBegin(); profileIt != profiles.constEnd(); ++profileIt) {
            const QJsonObject profile = profileIt.value().toObject();
            QJsonArray profileChecks;
            for (const QJsonValue& value : profile.value(QStringLiteral("checks")).toArray()) {
                const QJsonObject check = value.toObject();
                profileChecks.append(QJsonObject{{QStringLiteral("name"), check.value(QStringLiteral("name"))},
                    {QStringLiteral("status"), check.value(QStringLiteral("status"))},
                    {QStringLiteral("message"), check.value(QStringLiteral("message"))}});
            }
            safeProfiles.insert(profileIt.key(), QJsonObject{
                {QStringLiteral("title"), profile.value(QStringLiteral("title"))},
                {QStringLiteral("status"), profile.value(QStringLiteral("status"))},
                {QStringLiteral("repairHints"), profile.value(QStringLiteral("repairHints"))},
                {QStringLiteral("checks"), profileChecks}});
        }
        *report = protocol::redactPhysicalPathFields(QJsonObject{{QStringLiteral("schemaVersion"), 2},
            {QStringLiteral("kind"), QStringLiteral("environment_profiles_report")},
            {QStringLiteral("checkedAt"), stored.value(QStringLiteral("checkedAt"))},
            {QStringLiteral("checks"), safeChecks}, {QStringLiteral("profiles"), safeProfiles},
            {QStringLiteral("summary"), stored.value(QStringLiteral("summary"))}});
        return true;
    }
    if (error) *error = QStringLiteral("所选任务没有已提交的 Environment Check  报告。");
    return false;
}

} // namespace aitrain
