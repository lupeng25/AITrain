#include "aitrain/storage/ProjectStore.h"
#include "aitrain/storage/ArtifactCatalogRepository.h"
#include "aitrain/storage/DatasetCatalogRepository.h"
#include "aitrain/storage/ModelCatalogRepository.h"
#include "aitrain/storage/ProjectMetaRepository.h"
#include "aitrain/storage/ProjectReadRepository.h"
#include "aitrain/storage/TaskEventRepository.h"
#include "aitrain/storage/WorkflowRepository.h"
#include "aitrain/storage/WorkflowTerminalizationStore.h"
#include "aitrain/domain/ArtifactMemberPath.h"
#include "aitrain/protocol/Protocol.h"
#include "aitrain/protocol/ProtocolSanitizer.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QDir>
#include <QHash>
#include <QSqlError>
#include <QSqlQuery>
#include <QSet>
#include <QVariant>
#include <QVector>

#include <cmath>
#include <limits>
#include <utility>

namespace aitrain {
namespace {

constexpr qint64 kFirstHostStateEventSequence = 4000000000000000000LL;
// durable Workflow terminal outbox 是本轮破坏性重构新增的持久化事实，
// 因此显式提升 Storage schema，旧数据库不会被静默当作当前结构打开。
constexpr int kStorageSchemaVersion = 13;
QString terminalPolicyText(WorkflowTerminalPolicy policy)
{
    return policy == WorkflowTerminalPolicy::EvidenceRequired
        ? QStringLiteral("evidence_required") : QStringLiteral("immediate");
}

bool parseTerminalPolicy(const QString& value, WorkflowTerminalPolicy* result)
{
    if (value == QStringLiteral("immediate")) {
        if (result) *result = WorkflowTerminalPolicy::Immediate;
        return true;
    }
    if (value == QStringLiteral("evidence_required")) {
        if (result) *result = WorkflowTerminalPolicy::EvidenceRequired;
        return true;
    }
    return false;
}

QString terminalizationStateText(WorkflowTerminalizationState state)
{
    switch (state) {
    case WorkflowTerminalizationState::Sealed: return QStringLiteral("sealed");
    case WorkflowTerminalizationState::EvidenceAttached: return QStringLiteral("evidence_attached");
    case WorkflowTerminalizationState::Closed: return QStringLiteral("closed");
    }
    return {};
}

bool parseTerminalizationState(const QString& value, WorkflowTerminalizationState* result)
{
    if (value == QStringLiteral("sealed")) {
        if (result) *result = WorkflowTerminalizationState::Sealed;
        return true;
    }
    if (value == QStringLiteral("evidence_attached")) {
        if (result) *result = WorkflowTerminalizationState::EvidenceAttached;
        return true;
    }
    if (value == QStringLiteral("closed")) {
        if (result) *result = WorkflowTerminalizationState::Closed;
        return true;
    }
    return false;
}

QString modelSourceSnapshotBindingText(ModelSourceSnapshotBinding binding)
{
    return binding == ModelSourceSnapshotBinding::ProjectSnapshot
        ? QStringLiteral("project_snapshot")
        : QStringLiteral("external_declared");
}

QString sqlError(const QSqlQuery& query)
{
    return query.lastError().text();
}

bool execute(QSqlDatabase database, const QString& statement, QString* error)
{
    QSqlQuery query(database);
    if (query.exec(statement)) {
        return true;
    }
    if (error) {
        *error = sqlError(query);
    }
    return false;
}

QString utcText(const QDateTime& value)
{
    return value.toUTC().toString(Qt::ISODateWithMs);
}

QString requiredText(const QString& value)
{
    return value.isNull() ? QStringLiteral("") : value;
}

ProjectErrorCode pageErrorCode(const QString& error)
{
    if (error.startsWith(QStringLiteral("InvalidPageCursor"))) {
        return ProjectErrorCode::InvalidPageCursor;
    }
    return error.isEmpty()
        ? ProjectErrorCode::None : ProjectErrorCode::SqlError;
}

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

bool hasTable(QSqlDatabase database, const QString& tableName, QString* error)
{
    QSqlQuery query(database);
    query.prepare(QStringLiteral("select 1 from sqlite_master where type = 'table' and name = :name"));
    query.bindValue(QStringLiteral(":name"), tableName);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return query.next();
}

bool parseTerminalization(QSqlQuery& query, WorkflowTerminalizationSnapshot* result, QString* error)
{
    WorkflowTerminalizationSnapshot parsed;
    if (!WorkflowRunId::parse(query.value(0).toString(), &parsed.workflowRunId, error)
        || !TaskId::parse(query.value(1).toString(), &parsed.taskId, error)
        || !parseTerminalizationState(query.value(2).toString(), &parsed.state)
        || !taskStateFromString(query.value(3).toString(), &parsed.terminalState)) {
        if (error && error->isEmpty()) *error = QStringLiteral("工作流终态封存记录包含无效状态或标识。");
        return false;
    }
    if (!failureCodeFromString(query.value(4).toString(), &parsed.failure.code)) {
        if (error) *error = QStringLiteral("工作流终态记录包含未知 FailureCode。");
        return false;
    }
    parsed.failure.message = query.value(5).toString();
    parsed.failure.suggestedAction = query.value(6).toString();
    parsed.failure.occurredAt = parseUtc(query.value(7).toString());
    parsed.terminalAt = parseUtc(query.value(8).toString());
    if (!query.value(9).toString().isEmpty()
        && !ArtifactId::parse(query.value(9).toString(), &parsed.evidenceArtifactId, error)) return false;
    parsed.evidenceAttemptCount = query.value(10).toInt();
    if (!failureCodeFromString(query.value(11).toString(), &parsed.lastEvidenceFailure.code)) {
        if (error) *error = QStringLiteral("Evidence 失败记录包含未知 FailureCode。");
        return false;
    }
    parsed.lastEvidenceFailure.message = query.value(12).toString();
    parsed.lastEvidenceFailure.suggestedAction = query.value(13).toString();
    parsed.lastEvidenceFailure.occurredAt = parseUtc(query.value(14).toString());
    parsed.sealedAt = parseUtc(query.value(15).toString());
    parsed.evidenceAttachedAt = parseUtc(query.value(16).toString());
    parsed.closedAt = parseUtc(query.value(17).toString());
    if (!isTerminalTaskState(parsed.terminalState) || !parsed.terminalAt.isValid()
        || !parsed.sealedAt.isValid() || parsed.evidenceAttemptCount < 0) {
        if (error) *error = QStringLiteral("工作流终态封存记录字段无效。");
        return false;
    }
    if (result) *result = parsed;
    return true;
}

bool artifactExistsInDatabase(QSqlDatabase database, const ArtifactId& artifactId, QString* error)
{
    QSqlQuery query(database);
    query.prepare(QStringLiteral("select 1 from artifacts where id = :id"));
    query.bindValue(QStringLiteral(":id"), artifactId.toString());
    if (!query.exec()) {
        if (error) *error = sqlError(query);
        return false;
    }
    if (!query.next()) {
        if (error) *error = QStringLiteral("工作流只能引用已提交的 Artifact。");
        return false;
    }
    return true;
}

bool completeTerminalFailure(TaskState terminalState, const Failure& failure)
{
    if (terminalState == TaskState::Succeeded) {
        return !failure.isFailure() && failure.message.isEmpty()
            && failure.suggestedAction.isEmpty() && !failure.occurredAt.isValid();
    }
    if (terminalState != TaskState::Failed && terminalState != TaskState::Canceled) return false;
    if (!failure.isFailure() || failure.message.trimmed().isEmpty()
        || failure.suggestedAction.trimmed().isEmpty() || !failure.occurredAt.isValid()) return false;
    return (terminalState == TaskState::Canceled) == (failure.code == FailureCode::Canceled);
}

bool completeEvidenceFailure(const Failure& failure)
{
    return failure.isFailure() && failure.code != FailureCode::Canceled
        && !failure.message.trimmed().isEmpty() && !failure.suggestedAction.trimmed().isEmpty()
        && failure.occurredAt.isValid();
}

bool sameFailure(const Failure& left, const Failure& right)
{
    return left.code == right.code && left.message == right.message
        && left.suggestedAction == right.suggestedAction
        && left.occurredAt.toUTC() == right.occurredAt.toUTC();
}

Failure cancellationWinsFailure(const Failure& original, const QDateTime& occurredAt)
{
    Failure failure = original;
    failure.code = FailureCode::Canceled;
    if (failure.message.trimmed().isEmpty()) {
        failure.message = QStringLiteral("任务已在适配器终态到达前收到取消请求。" );
    }
    if (failure.suggestedAction.trimmed().isEmpty()) {
        failure.suggestedAction = QStringLiteral("确认任务已停止后重新运行。" );
    }
    if (!failure.occurredAt.isValid()) {
        failure.occurredAt = occurredAt;
    }
    return failure;
}

} // namespace

ProjectStore::ProjectStore()
    : db_(database_.connection())
{
}

int ProjectStore::schemaVersion()
{
    return kStorageSchemaVersion;
}

ProjectStore::~ProjectStore()
{
    close();
}

void ProjectStore::swap(ProjectStore& other) noexcept
{
    using std::swap;
    database_.swap(other.database_);
    swap(artifactStoreRoot_, other.artifactStoreRoot_);
    swap(lastErrorCode_, other.lastErrorCode_);
}

bool ProjectStore::open(const QString& databasePath, QString* error)
{
    close();
    lastErrorCode_ = ProjectErrorCode::None;
    if (!database_.open(databasePath, error)) {
        lastErrorCode_ = ProjectErrorCode::SqlError;
        return false;
    }
    if (initialize(error)) {
        return true;
    }
    close();
    return false;
}

void ProjectStore::close()
{
    database_.close();
}

bool ProjectStore::isOpen() const
{
    return database_.isOpen();
}

ProjectErrorCode ProjectStore::lastErrorCode() const
{
    return lastErrorCode_;
}

bool ProjectStore::projectMeta(ProjectMetaSnapshot* result, QString* error) const
{
    if (!isOpen() || !result) {
        if (error) *error = QStringLiteral("读取 project_meta 需要已打开的 Storage 和输出对象。");
        return false;
    }
    ProjectMetaRepository repository(database_);
    return repository.read(result, error);
}

bool ProjectStore::advanceOpenGeneration(ProjectMetaSnapshot* result, QString* error)
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral("更新 open_generation 需要已打开的 Storage。");
        return false;
    }
    const QDateTime now = QDateTime::currentDateTimeUtc();
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    ProjectMetaRepository repository(database_);
    QString repositoryError;
    if (!repository.incrementOpenGeneration(now, &repositoryError)
        || !db_.commit()) {
        if (error) {
            *error = !repositoryError.isEmpty()
                ? repositoryError : db_.lastError().text();
        }
        db_.rollback();
        return false;
    }
    return result ? projectMeta(result, error) : true;
}

void ProjectStore::setArtifactStoreRoot(QString artifactStoreRoot)
{
    artifactStoreRoot_ = QDir::cleanPath(QDir(artifactStoreRoot).absolutePath());
}

bool ProjectStore::initialize(QString* error)
{
    QString lookupError;
    const bool hasMeta = hasTable(db_, QStringLiteral("project_meta"), &lookupError);
    if (!lookupError.isEmpty()) {
        if (error) {
            *error = lookupError;
        }
        return false;
    }
    const bool hasLegacySchema = hasTable(db_, QStringLiteral("schema_info"), &lookupError);
    if (!lookupError.isEmpty()) {
        if (error) *error = lookupError;
        lastErrorCode_ = ProjectErrorCode::SqlError;
        return false;
    }
    if (hasLegacySchema || (!hasMeta && hasTable(db_, QStringLiteral("tasks"), &lookupError))) {
        if (error) {
            *error = QStringLiteral(
                "SchemaRebuildRequired：检测到旧项目数据库；当前版本不迁移旧库，请显式重建项目。");
        }
        lastErrorCode_ = ProjectErrorCode::SchemaRebuildRequired;
        return false;
    }
    if (!lookupError.isEmpty()) {
        if (error) {
            *error = lookupError;
        }
        return false;
    }
    ProjectMetaSnapshot meta;
    ProjectMetaRepository metaRepository(database_);
    if (hasMeta && !metaRepository.read(&meta, error)) {
        lastErrorCode_ = error && error->contains(QStringLiteral("SchemaRebuildRequired"))
            ? ProjectErrorCode::SchemaRebuildRequired
            : ProjectErrorCode::ProjectMetaCorrupt;
        return false;
    }

    const QStringList pragmas = {
        QStringLiteral("pragma foreign_keys = on"),
        QStringLiteral("pragma journal_mode = wal"),
        QStringLiteral("pragma busy_timeout = 5000")
    };
    for (const QString& pragma : pragmas) {
        if (!execute(db_, pragma, error)) {
            return false;
        }
    }

    const QStringList statements = {
        QStringLiteral("create table if not exists project_meta (singleton integer primary key check(singleton = 1), project_id text not null unique, schema_version integer not null check(schema_version = 13), display_name text not null check(length(display_name) > 0), open_generation integer not null default 0 check(open_generation >= 0), created_at text not null, updated_at text not null, last_opened_at text)"),
        QStringLiteral("create table if not exists datasets (id text primary key, dataset_format text not null, created_at text not null)"),
        QStringLiteral("create table if not exists dataset_versions (id text primary key, dataset_id text not null references datasets(id) on delete restrict, root_hash text not null check(length(root_hash) = 64 and root_hash not glob '*[^0-9a-f]*'), created_at text not null, unique(dataset_id, root_hash))"),
        QStringLiteral("create table if not exists dataset_snapshots (id text primary key, dataset_version_id text not null references dataset_versions(id) on delete restrict, task_id text not null references tasks(id) on delete restrict, artifact_id text not null unique references artifacts(id) on delete restrict, root_path text not null, driver_id text not null, driver_version text not null, manifest_sha256 text not null check(length(manifest_sha256) = 64 and manifest_sha256 not glob '*[^0-9a-f]*'), file_count integer not null check(file_count >= 0), total_bytes integer not null check(total_bytes >= 0), created_at text not null)"),
        QStringLiteral("create table if not exists tasks (id text primary key, request_id text not null unique, state text not null check(state in ('created','queued','starting','running','cancel_requested','succeeded','failed','canceled')), capability_id text not null, task_type text not null, failure_code text not null default 'none', failure_details text not null default '', failure_suggested_action text not null default '', failure_occurred_at text, created_at text not null, updated_at text not null)"),
        QStringLiteral("create table if not exists task_events (id text primary key, task_id text not null references tasks(id) on delete restrict, request_id text not null, sequence integer not null check(sequence > 0), kind text not null, occurred_at text not null, payload_json text not null, unique(request_id, sequence))"),
        QStringLiteral("create table if not exists task_metrics (id text primary key, task_id text not null references tasks(id) on delete restrict, name text not null, value real not null, occurred_at text not null)"),
        QStringLiteral("create table if not exists artifacts (id text primary key, task_id text not null references tasks(id) on delete restrict, kind text not null, created_at text not null)"),
        QStringLiteral("create table if not exists artifact_files (id text primary key, artifact_id text not null references artifacts(id) on delete restrict, relative_path text not null, sha256 text not null check(length(sha256) = 64 and sha256 not glob '*[^0-9a-f]*'), byte_count integer not null check(byte_count >= 0), unique(artifact_id, relative_path))"),
        QStringLiteral("create table if not exists model_packages (id text primary key, model_family text not null, task_type text not null, source_backend text not null, source_task_id text not null references tasks(id) on delete restrict, source_snapshot_id text not null, source_snapshot_binding text not null check(source_snapshot_binding in ('project_snapshot','external_declared')), source_artifact_id text not null references artifacts(id) on delete restrict, source_artifact_sha256 text not null check(length(source_artifact_sha256) = 64 and source_artifact_sha256 not glob '*[^0-9a-f]*'), manifest_json text not null, verified integer not null check(verified in (0, 1)), created_at text not null)"),
        QStringLiteral("create table if not exists evaluation_reports (id text primary key, task_id text not null references tasks(id) on delete restrict, artifact_id text not null references artifacts(id) on delete restrict, created_at text not null)"),
        QStringLiteral("create table if not exists workflow_runs (id text primary key, task_id text not null references tasks(id) on delete restrict, template_id text not null, terminal_policy text not null check(terminal_policy in ('immediate','evidence_required')), created_at text not null)"),
        QStringLiteral("create table if not exists workflow_input_bindings (workflow_run_id text not null references workflow_runs(id) on delete restrict, role text not null, source_artifact_id text not null references artifacts(id) on delete restrict, source_task_id text not null references tasks(id) on delete restrict, source_artifact_kind text not null, dataset_id text references datasets(id) on delete restrict, dataset_snapshot_id text references dataset_snapshots(id) on delete restrict, dataset_version_id text references dataset_versions(id) on delete restrict, model_package_id text references model_packages(id) on delete restrict, manifest_sha256 text not null default '', root_hash text not null default '', bound_at text not null, primary key(workflow_run_id, role))"),
        QStringLiteral("create table if not exists workflow_steps (id text primary key, workflow_run_id text not null references workflow_runs(id) on delete restrict, ordinal integer not null check(ordinal >= 0), kind text not null, state text not null check(state in ('pending','running','succeeded','failed','canceled','skipped')), input_artifact_id text references artifacts(id) on delete restrict, output_artifact_id text references artifacts(id) on delete restrict, backend text not null, parameter_summary_json text not null, started_at text, finished_at text, failure_code text not null default 'none', failure_details text not null default '', failure_suggested_action text not null default '', failure_occurred_at text, retry_count integer not null default 0 check(retry_count >= 0), unique(workflow_run_id, ordinal))"),
        QStringLiteral("create table if not exists workflow_terminalizations (workflow_run_id text primary key references workflow_runs(id) on delete restrict, task_id text not null references tasks(id) on delete restrict, state text not null check(state in ('sealed','evidence_attached','closed')), terminal_state text not null check(terminal_state in ('succeeded','failed','canceled')), failure_code text not null, failure_details text not null, failure_suggested_action text not null, failure_occurred_at text, terminal_at text not null, evidence_artifact_id text unique references artifacts(id) on delete restrict, evidence_attempt_count integer not null default 0 check(evidence_attempt_count >= 0), last_evidence_failure_code text not null default 'none', last_evidence_failure_details text not null default '', last_evidence_failure_suggested_action text not null default '', last_evidence_failure_occurred_at text, sealed_at text not null, evidence_attached_at text, closed_at text, check((terminal_state = 'succeeded' and failure_code = 'none' and failure_details = '' and failure_suggested_action = '' and failure_occurred_at is null) or (terminal_state in ('failed','canceled') and failure_code <> 'none' and failure_details <> '' and failure_suggested_action <> '' and failure_occurred_at is not null)), check((terminal_state = 'canceled') = (failure_code = 'canceled')), check((last_evidence_failure_code = 'none' and last_evidence_failure_details = '' and last_evidence_failure_suggested_action = '' and last_evidence_failure_occurred_at is null) or (last_evidence_failure_code <> 'none' and last_evidence_failure_details <> '' and last_evidence_failure_suggested_action <> '' and last_evidence_failure_occurred_at is not null)), check((state = 'sealed' and evidence_artifact_id is null and evidence_attached_at is null and closed_at is null) or (state = 'evidence_attached' and evidence_artifact_id is not null and evidence_attached_at is not null and closed_at is null) or (state = 'closed' and evidence_artifact_id is not null and evidence_attached_at is not null and closed_at is not null)))"),
        QStringLiteral("create table if not exists workflow_terminal_outbox (message_id text primary key references task_events(id) on delete restrict, task_id text not null references tasks(id) on delete restrict, request_id text not null, workflow_run_id text not null references workflow_runs(id) on delete restrict, workflow_step_id text not null references workflow_steps(id) on delete restrict, sequence integer not null check(sequence > 0), kind text not null check(kind in ('event.succeeded','event.failed','event.canceled')), occurred_at text not null, payload_json text not null, output_artifact_id text references artifacts(id) on delete restrict, state text not null check(state in ('pending','applied')), created_at text not null, applied_at text, unique(request_id, sequence), check((state = 'pending' and applied_at is null) or (state = 'applied' and applied_at is not null)))"),
        QStringLiteral("create trigger if not exists trg_tasks_require_evidence_before_terminal before update of state on tasks when new.state in ('succeeded','failed','canceled') and old.state not in ('succeeded','failed','canceled') and exists(select 1 from workflow_runs w left join workflow_terminalizations z on z.workflow_run_id = w.id where w.task_id = new.id and w.terminal_policy = 'evidence_required' and (z.workflow_run_id is null or z.state = 'sealed')) begin select raise(abort, 'evidence_required workflow must attach evidence before task terminal'); end"),
        QStringLiteral("create index if not exists idx_tasks_updated_at_id on tasks(updated_at desc, id desc)"),
        QStringLiteral("create index if not exists idx_task_events_task_id on task_events(task_id, sequence)"),
        QStringLiteral("create index if not exists idx_artifacts_task_created_id on artifacts(task_id, created_at, id)"),
        QStringLiteral("create index if not exists idx_artifact_files_artifact_path on artifact_files(artifact_id, relative_path)"),
        QStringLiteral("create index if not exists idx_task_metrics_task_occurred_id on task_metrics(task_id, occurred_at, id)"),
        QStringLiteral("create index if not exists idx_artifacts_kind_created_at on artifacts(kind, created_at desc, id desc)"),
        QStringLiteral("create index if not exists idx_datasets_created_id on datasets(created_at desc, id desc)"),
        QStringLiteral("create index if not exists idx_dataset_versions_dataset_id on dataset_versions(dataset_id, created_at desc)"),
        QStringLiteral("create index if not exists idx_dataset_snapshots_task_id on dataset_snapshots(task_id, created_at desc)"),
        QStringLiteral("create index if not exists idx_workflow_runs_task_created_id on workflow_runs(task_id, created_at, id)"),
        QStringLiteral("create index if not exists idx_workflow_runs_policy_created_id on workflow_runs(terminal_policy, created_at, id)"),
        QStringLiteral("create index if not exists idx_workflow_steps_run_ordinal on workflow_steps(workflow_run_id, ordinal)"),
        QStringLiteral("create index if not exists idx_workflow_inputs_artifact on workflow_input_bindings(source_artifact_id)"),
        QStringLiteral("create index if not exists idx_workflow_terminalizations_pending on workflow_terminalizations(state, sealed_at, workflow_run_id)"),
        QStringLiteral("create index if not exists idx_workflow_terminal_outbox_pending on workflow_terminal_outbox(state, created_at, message_id)"),
        QStringLiteral("create index if not exists idx_model_packages_created_id on model_packages(created_at desc, id desc)"),
        QStringLiteral("create index if not exists idx_model_packages_source_task_id on model_packages(source_task_id)"),
        QStringLiteral("create index if not exists idx_model_packages_source_artifact_id on model_packages(source_artifact_id)")
    };

    if (!db_.transaction()) {
        if (error) {
            *error = db_.lastError().text();
        }
        return false;
    }
    for (const QString& statement : statements) {
        if (!execute(db_, statement, error)) {
            db_.rollback();
            return false;
        }
    }
    if (!hasMeta) {
        QDir projectDirectory = QFileInfo(db_.databaseName()).absoluteDir();
        projectDirectory.cdUp();
        QString displayName = projectDirectory.dirName().trimmed();
        if (displayName.isEmpty()) displayName = QStringLiteral("AITrain Project");
        const QDateTime now = QDateTime::currentDateTimeUtc();
        QSqlQuery metaQuery(db_);
        metaQuery.prepare(QStringLiteral(
            "insert into project_meta(singleton, project_id, schema_version, display_name, "
            "open_generation, created_at, updated_at, last_opened_at) "
            "values(1, :project_id, 13, :display_name, 0, :created_at, :updated_at, null)"));
        metaQuery.bindValue(QStringLiteral(":project_id"), ProjectId::create().toString());
        metaQuery.bindValue(QStringLiteral(":display_name"), displayName);
        metaQuery.bindValue(QStringLiteral(":created_at"), utcText(now));
        metaQuery.bindValue(QStringLiteral(":updated_at"), utcText(now));
        if (!metaQuery.exec()) {
            if (error) *error = sqlError(metaQuery);
            db_.rollback();
            lastErrorCode_ = ProjectErrorCode::SqlError;
            return false;
        }
    }
    if (!db_.commit()) {
        if (error) *error = db_.lastError().text();
        lastErrorCode_ = ProjectErrorCode::SqlError;
        return false;
    }
    lastErrorCode_ = ProjectErrorCode::None;
    return true;
}

bool ProjectStore::appendStateEvent(const TaskId& taskId,
    const RequestId& requestId,
    TaskState state,
    const Failure& failure,
    const QDateTime& occurredAt,
    QString* error)
{
    QSqlQuery sequenceQuery(db_);
    sequenceQuery.prepare(QStringLiteral("select coalesce(max(sequence), :previous_sequence) + 1 from task_events where request_id = :request_id and kind = :kind"));
    sequenceQuery.bindValue(QStringLiteral(":request_id"), requestId.toString());
    sequenceQuery.bindValue(QStringLiteral(":kind"), QStringLiteral("task.state_changed"));
    sequenceQuery.bindValue(QStringLiteral(":previous_sequence"), kFirstHostStateEventSequence - 1);
    if (!sequenceQuery.exec() || !sequenceQuery.next()) {
        if (error) {
            *error = sqlError(sequenceQuery);
        }
        return false;
    }

    QJsonObject payload;
    payload.insert(QStringLiteral("state"), taskStateToString(state));
    payload.insert(QStringLiteral("failureCode"), failureCodeToString(failure.code));
    payload.insert(QStringLiteral("failureMessage"), requiredText(failure.message));
    payload.insert(QStringLiteral("failureSuggestedAction"), requiredText(failure.suggestedAction));
    payload.insert(QStringLiteral("failureOccurredAt"), failure.isFailure()
            ? utcText(failure.occurredAt.isValid() ? failure.occurredAt : occurredAt)
            : QString());
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into task_events(id, task_id, request_id, sequence, kind, occurred_at, payload_json) values(:id, :task_id, :request_id, :sequence, :kind, :occurred_at, :payload_json)"));
    query.bindValue(QStringLiteral(":id"), MessageId::create().toString());
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    query.bindValue(QStringLiteral(":request_id"), requestId.toString());
    query.bindValue(QStringLiteral(":sequence"), sequenceQuery.value(0).toLongLong());
    query.bindValue(QStringLiteral(":kind"), QStringLiteral("task.state_changed"));
    query.bindValue(QStringLiteral(":occurred_at"), utcText(occurredAt));
    query.bindValue(QStringLiteral(":payload_json"), QString::fromUtf8(
        QJsonDocument(protocol::redactPhysicalPathFields(payload)).toJson(QJsonDocument::Compact)));
    if (query.exec()) {
        return true;
    }
    if (error) {
        *error = sqlError(query);
    }
    return false;
}

bool ProjectStore::createTask(const TaskSnapshot& task, QString* error)
{
    if (!task.id.isValid() || !task.requestId.isValid() || task.capabilityId.isEmpty() || task.taskType.isEmpty()) {
        if (error) {
            *error = QStringLiteral("创建任务需要有效 ID、能力标识和任务类型。");
        }
        return false;
    }
    if (task.state != TaskState::Created) {
        if (error) {
            *error = QStringLiteral("新任务必须处于 created 状态。");
        }
        return false;
    }
    const QDateTime now = task.createdAt.isValid() ? task.createdAt.toUTC() : QDateTime::currentDateTimeUtc();
    if (!db_.transaction()) {
        if (error) {
            *error = db_.lastError().text();
        }
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into tasks(id, request_id, state, capability_id, task_type, failure_code, failure_details, failure_suggested_action, failure_occurred_at, created_at, updated_at) values(:id, :request_id, :state, :capability_id, :task_type, :failure_code, :failure_details, :failure_suggested_action, :failure_occurred_at, :created_at, :updated_at)"));
    query.bindValue(QStringLiteral(":id"), task.id.toString());
    query.bindValue(QStringLiteral(":request_id"), task.requestId.toString());
    query.bindValue(QStringLiteral(":state"), taskStateToString(task.state));
    query.bindValue(QStringLiteral(":capability_id"), task.capabilityId);
    query.bindValue(QStringLiteral(":task_type"), task.taskType);
    query.bindValue(QStringLiteral(":failure_code"), failureCodeToString(task.failure.code));
    query.bindValue(QStringLiteral(":failure_details"), requiredText(task.failure.message));
    query.bindValue(QStringLiteral(":failure_suggested_action"), requiredText(task.failure.suggestedAction));
    query.bindValue(QStringLiteral(":failure_occurred_at"), task.failure.isFailure()
            ? utcText(task.failure.occurredAt.isValid() ? task.failure.occurredAt : now)
            : QVariant());
    query.bindValue(QStringLiteral(":created_at"), utcText(now));
    query.bindValue(QStringLiteral(":updated_at"), utcText(now));
    if (!query.exec() || !appendStateEvent(task.id, task.requestId, task.state, task.failure, now, error)) {
        if (error && error->isEmpty()) {
            *error = sqlError(query);
        }
        db_.rollback();
        return false;
    }
    return db_.commit();
}

bool ProjectStore::transitionTask(const TaskId& taskId,
    TaskState expectedState,
    TaskState nextState,
    const Failure& failure,
    QString* error)
{
    if (!taskId.isValid() || !isValidTaskStateTransition(expectedState, nextState)) {
        if (error) {
            *error = QStringLiteral("非法任务状态迁移。");
        }
        return false;
    }
    if (isTerminalTaskState(nextState) && !completeTerminalFailure(nextState, failure)) {
        if (error) *error = QStringLiteral("任务终态 Failure 字段不完整或与终态不匹配。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) {
            *error = db_.lastError().text();
        }
        return false;
    }
    QSqlQuery taskQuery(db_);
    taskQuery.prepare(QStringLiteral("select request_id from tasks where id = :id and state = :state"));
    taskQuery.bindValue(QStringLiteral(":id"), taskId.toString());
    taskQuery.bindValue(QStringLiteral(":state"), taskStateToString(expectedState));
    if (!taskQuery.exec() || !taskQuery.next()) {
        if (error) {
            *error = taskQuery.lastError().isValid()
                ? sqlError(taskQuery)
                : QStringLiteral("任务不存在或状态已被并发更新。");
        }
        db_.rollback();
        return false;
    }
    RequestId requestId;
    if (!RequestId::parse(taskQuery.value(0).toString(), &requestId, error)) {
        db_.rollback();
        return false;
    }
    const QDateTime now = QDateTime::currentDateTimeUtc();
    QSqlQuery updateQuery(db_);
    updateQuery.prepare(QStringLiteral("update tasks set state = :next_state, failure_code = :failure_code, failure_details = :failure_details, failure_suggested_action = :failure_suggested_action, failure_occurred_at = :failure_occurred_at, updated_at = :updated_at where id = :id and state = :expected_state"));
    updateQuery.bindValue(QStringLiteral(":next_state"), taskStateToString(nextState));
    updateQuery.bindValue(QStringLiteral(":failure_code"), failureCodeToString(failure.code));
    updateQuery.bindValue(QStringLiteral(":failure_details"), requiredText(failure.message));
    updateQuery.bindValue(QStringLiteral(":failure_suggested_action"), requiredText(failure.suggestedAction));
    updateQuery.bindValue(QStringLiteral(":failure_occurred_at"), failure.isFailure()
            ? utcText(failure.occurredAt.isValid() ? failure.occurredAt : now)
            : QVariant());
    updateQuery.bindValue(QStringLiteral(":updated_at"), utcText(now));
    updateQuery.bindValue(QStringLiteral(":id"), taskId.toString());
    updateQuery.bindValue(QStringLiteral(":expected_state"), taskStateToString(expectedState));
    if (!updateQuery.exec() || updateQuery.numRowsAffected() != 1
        || !appendStateEvent(taskId, requestId, nextState, failure, now, error)) {
        if (error && error->isEmpty()) {
            *error = sqlError(updateQuery);
        }
        db_.rollback();
        return false;
    }
    return db_.commit();
}

bool ProjectStore::markTaskInterruptedFailed(const TaskId& taskId, QString* error)
{
    if (!taskId.isValid()) {
        if (error) *error = QStringLiteral("Worker 丢失恢复需要有效任务 ID。");
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select state from tasks where id = :id"));
    query.bindValue(QStringLiteral(":id"), taskId.toString());
    if (!query.exec()) {
        if (error) *error = sqlError(query);
        return false;
    }
    if (!query.next()) {
        // Worker 可能在创建任务前就启动失败；此时没有需要收口的持久化事实。
        if (error) error->clear();
        return true;
    }
    TaskState currentState;
    if (!taskStateFromString(query.value(0).toString(), &currentState)) {
        if (error) *error = QStringLiteral("任务状态无法解析，拒绝 Worker 丢失恢复。");
        return false;
    }
    if (isTerminalTaskState(currentState)) {
        if (error) error->clear();
        return true;
    }
    const bool cancellationWasPending = currentState == TaskState::CancelRequested;
    const Failure failure{
        cancellationWasPending ? FailureCode::Canceled : FailureCode::ProcessCrashed,
        cancellationWasPending
            ? QStringLiteral("Worker 在取消请求后丢失，任务按取消完成收口。")
            : QStringLiteral("Worker 异常退出且未报告任务终态。"),
        cancellationWasPending
            ? QStringLiteral("如需继续，请重新发起该任务。")
            : defaultFailureSuggestedAction(FailureCode::ProcessCrashed),
        QDateTime::currentDateTimeUtc()};
    return transitionTask(taskId, currentState,
        cancellationWasPending ? TaskState::Canceled : TaskState::Failed, failure, error);
}

bool ProjectStore::markInterruptedTasksFailed(QString* error)
{
    QSqlQuery query(db_);
    if (!query.exec(QStringLiteral("select id, state from tasks where state in ('queued', 'starting','running','cancel_requested') and id not in (select task_id from workflow_runs where terminal_policy = 'evidence_required' and id not in (select workflow_run_id from workflow_terminalizations where state = 'closed'))"))) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    QVector<QPair<TaskId, TaskState>> interrupted;
    while (query.next()) {
        TaskId id;
        TaskState state;
        if (!TaskId::parse(query.value(0).toString(), &id, error) || !taskStateFromString(query.value(1).toString(), &state)) {
            return false;
        }
        interrupted.append({id, state});
    }
    for (const auto& item : interrupted) {
        const bool cancellationWasPending = item.second == TaskState::CancelRequested;
        const TaskState recoveredState = cancellationWasPending ? TaskState::Canceled : TaskState::Failed;
        const Failure recoveredFailure{
            cancellationWasPending ? FailureCode::Canceled : FailureCode::ProcessCrashed,
            cancellationWasPending
                ? QStringLiteral("应用在取消请求未收口时关闭，恢复时按取消完成。")
                : QStringLiteral("应用在任务未结束时关闭。"),
            cancellationWasPending
                ? QStringLiteral("如需继续，请重新发起该任务。")
                : defaultFailureSuggestedAction(FailureCode::ProcessCrashed),
            QDateTime::currentDateTimeUtc()};
        if (!transitionTask(item.first, item.second, recoveredState, recoveredFailure, error)) {
            return false;
        }
    }
    return true;
}

bool ProjectStore::recordProtocolEvent(const TaskId& taskId,
    const RequestId& requestId,
    const MessageId& messageId,
    quint64 sequence,
    const QString& kind,
    const QJsonObject& payload,
    const QDateTime& occurredAt,
    QString* error)
{
    if (!taskId.isValid() || !requestId.isValid() || !messageId.isValid() || sequence == 0
        || sequence > static_cast<quint64>(std::numeric_limits<qint64>::max())
        || kind.trimmed().isEmpty() || !occurredAt.isValid()) {
        if (error) {
            *error = QStringLiteral("协议事件字段无效。");
        }
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery owner(db_);
    owner.prepare(QStringLiteral("select 1 from tasks where id = :task_id and request_id = :request_id"));
    owner.bindValue(QStringLiteral(":task_id"), taskId.toString());
    owner.bindValue(QStringLiteral(":request_id"), requestId.toString());
    if (!owner.exec() || !owner.next()) {
        if (error) *error = owner.lastError().isValid() ? sqlError(owner)
            : QStringLiteral("协议事件不属于指定的任务和请求。" );
        db_.rollback();
        return false;
    }
    QSqlQuery sequenceQuery(db_);
    sequenceQuery.prepare(QStringLiteral("select coalesce(max(sequence), 0) from task_events where request_id = :request_id and kind <> 'task.state_changed'"));
    sequenceQuery.bindValue(QStringLiteral(":request_id"), requestId.toString());
    if (!sequenceQuery.exec() || !sequenceQuery.next()) {
        if (error) *error = sqlError(sequenceQuery);
        db_.rollback();
        return false;
    }
    if (sequenceQuery.value(0).toLongLong() >= static_cast<qint64>(sequence)) {
        if (error) *error = QStringLiteral("协议事件 sequence 必须严格递增或已重复。" );
        db_.rollback();
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into task_events(id, task_id, request_id, sequence, kind, occurred_at, payload_json) values(:id, :task_id, :request_id, :sequence, :kind, :occurred_at, :payload_json)"));
    query.bindValue(QStringLiteral(":id"), messageId.toString());
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    query.bindValue(QStringLiteral(":request_id"), requestId.toString());
    query.bindValue(QStringLiteral(":sequence"), static_cast<qint64>(sequence));
    query.bindValue(QStringLiteral(":kind"), kind.trimmed());
    query.bindValue(QStringLiteral(":occurred_at"), utcText(occurredAt));
    query.bindValue(QStringLiteral(":payload_json"), QString::fromUtf8(
        QJsonDocument(protocol::redactPhysicalPathFields(payload)).toJson(QJsonDocument::Compact)));
    if (query.exec()) {
        if (db_.commit()) return true;
        if (error) *error = db_.lastError().text();
        return false;
    }
    if (error) {
        *error = sqlError(query);
    }
    db_.rollback();
    return false;
}

bool ProjectStore::recordWorkflowTerminalEvent(const ProtocolEnvelope& envelope,
    const ArtifactId& outputArtifactId,
    bool* idempotent,
    QString* error)
{
    if (idempotent) *idempotent = false;
    const bool terminal = envelope.kind == QStringLiteral("event.succeeded")
        || envelope.kind == QStringLiteral("event.failed")
        || envelope.kind == QStringLiteral("event.canceled");
    if (!validateProtocolEnvelope(envelope, error) || !terminal
        || envelope.sequence > static_cast<quint64>(std::numeric_limits<qint64>::max())) {
        if (error && error->isEmpty()) *error = QStringLiteral("Workflow 终态事件字段无效。");
        return false;
    }
    if (envelope.kind != QStringLiteral("event.succeeded") && outputArtifactId.isValid()) {
        if (error) *error = QStringLiteral("失败或取消的 Workflow 终态不能携带输出 Artifact。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }

    QSqlQuery owner(db_);
    owner.prepare(QStringLiteral("select 1 from tasks where id = :task_id and request_id = :request_id"));
    owner.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
    owner.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
    if (!owner.exec() || !owner.next()) {
        if (error) *error = owner.lastError().isValid() ? sqlError(owner)
            : QStringLiteral("Workflow 终态事件不属于指定的任务和请求。");
        db_.rollback();
        return false;
    }

    const QString payloadJson = QString::fromUtf8(QJsonDocument(
        protocol::redactPhysicalPathFields(envelope.payload)).toJson(QJsonDocument::Compact));
    bool eventAlreadyStored = false;
    QSqlQuery duplicate(db_);
    duplicate.prepare(QStringLiteral("select id, task_id, kind, payload_json from task_events where request_id = :request_id and sequence = :sequence"));
    duplicate.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
    duplicate.bindValue(QStringLiteral(":sequence"), static_cast<qint64>(envelope.sequence));
    if (!duplicate.exec()) {
        if (error) *error = sqlError(duplicate);
        db_.rollback();
        return false;
    }
    if (duplicate.next()) {
        if (duplicate.value(0).toString() != envelope.messageId.toString()
            || duplicate.value(1).toString() != envelope.taskId.toString()
            || duplicate.value(2).toString() != envelope.kind
            || duplicate.value(3).toString() != payloadJson) {
            if (error) *error = QStringLiteral("Workflow 终态事件序号已被不同事件占用。");
            db_.rollback();
            return false;
        }
        eventAlreadyStored = true;

        // 已有 outbox 记录时，允许 Adapter 重复投递同一终态，即使
        // Workflow Step 已由上一次 handler 完成而不再是 Running。
        QSqlQuery existingOutbox(db_);
        existingOutbox.prepare(QStringLiteral("select task_id, coalesce(output_artifact_id,'') from workflow_terminal_outbox where message_id = :message_id"));
        existingOutbox.bindValue(QStringLiteral(":message_id"), envelope.messageId.toString());
        if (!existingOutbox.exec()) {
            if (error) *error = sqlError(existingOutbox);
            db_.rollback();
            return false;
        }
        if (existingOutbox.next()) {
            if (existingOutbox.value(0).toString() != envelope.taskId.toString()
                || existingOutbox.value(1).toString() != outputArtifactId.toString()) {
                if (error) *error = QStringLiteral("Workflow 终态 outbox 已按不同 Task/Artifact 事实登记。");
                db_.rollback();
                return false;
            }
            if (!db_.commit()) {
                if (error) *error = db_.lastError().text();
                return false;
            }
            if (idempotent) *idempotent = true;
            return true;
        }
    } else {
        QSqlQuery existingMessage(db_);
        existingMessage.prepare(QStringLiteral("select request_id, sequence from task_events where id = :id"));
        existingMessage.bindValue(QStringLiteral(":id"), envelope.messageId.toString());
        if (!existingMessage.exec()) {
            if (error) *error = sqlError(existingMessage);
            db_.rollback();
            return false;
        }
        if (existingMessage.next()) {
            if (error) *error = QStringLiteral("Workflow 终态事件 messageId 已用于另一条事件。");
            db_.rollback();
            return false;
        }
        QSqlQuery sequenceQuery(db_);
        sequenceQuery.prepare(QStringLiteral("select coalesce(max(sequence), 0) from task_events where request_id = :request_id and kind <> 'task.state_changed'"));
        sequenceQuery.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
        if (!sequenceQuery.exec() || !sequenceQuery.next()) {
            if (error) *error = sqlError(sequenceQuery);
            db_.rollback();
            return false;
        }
        if (sequenceQuery.value(0).toLongLong() >= static_cast<qint64>(envelope.sequence)) {
            if (error) *error = QStringLiteral("Workflow 终态事件 sequence 必须严格递增或已重复。");
            db_.rollback();
            return false;
        }
    }

    // 记录事件时绑定唯一的 Running Workflow Step，恢复时不依赖进程内
    // callback/closure，也不会把终态事件错误重放到后续步骤。
    QSqlQuery stepQuery(db_);
    stepQuery.prepare(QStringLiteral("select w.id, s.id from workflow_runs w join workflow_steps s on s.workflow_run_id = w.id where w.task_id = :task_id and s.state = 'running' order by s.ordinal asc limit 2"));
    stepQuery.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
    if (!stepQuery.exec()) {
        if (error) *error = sqlError(stepQuery);
        db_.rollback();
        return false;
    }
    if (!stepQuery.next()) {
        // 普通（非多步骤 Workflow）Adapter 仍然使用同一入口记录终态
        // 审计，但不创建 workflow outbox；只有声明过 Workflow 的任务才
        // 要求事件绑定到 Running Step，避免把晚到终态写入错误的运行。
        QSqlQuery workflowQuery(db_);
        workflowQuery.prepare(QStringLiteral("select count(*) from workflow_runs where task_id = :task_id"));
        workflowQuery.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
        if (!workflowQuery.exec() || !workflowQuery.next()) {
            if (error) *error = sqlError(workflowQuery);
            db_.rollback();
            return false;
        }
        if (workflowQuery.value(0).toInt() == 0) {
            if (outputArtifactId.isValid()) {
                QSqlQuery artifactQuery(db_);
                artifactQuery.prepare(QStringLiteral("select task_id from artifacts where id = :id"));
                artifactQuery.bindValue(QStringLiteral(":id"), outputArtifactId.toString());
                if (!artifactQuery.exec() || !artifactQuery.next()
                    || artifactQuery.value(0).toString() != envelope.taskId.toString()) {
                    if (error) *error = artifactQuery.lastError().isValid() ? sqlError(artifactQuery)
                        : QStringLiteral("Adapter 终态输出 Artifact 不属于根任务。");
                    db_.rollback();
                    return false;
                }
            }
            if (!eventAlreadyStored) {
                QSqlQuery insertEvent(db_);
                insertEvent.prepare(QStringLiteral("insert into task_events(id, task_id, request_id, sequence, kind, occurred_at, payload_json) values(:id, :task_id, :request_id, :sequence, :kind, :occurred_at, :payload_json)"));
                insertEvent.bindValue(QStringLiteral(":id"), envelope.messageId.toString());
                insertEvent.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
                insertEvent.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
                insertEvent.bindValue(QStringLiteral(":sequence"), static_cast<qint64>(envelope.sequence));
                insertEvent.bindValue(QStringLiteral(":kind"), envelope.kind);
                insertEvent.bindValue(QStringLiteral(":occurred_at"), utcText(envelope.timestamp));
                insertEvent.bindValue(QStringLiteral(":payload_json"), payloadJson);
                if (!insertEvent.exec()) {
                    if (error) *error = sqlError(insertEvent);
                    db_.rollback();
                    return false;
                }
            }
            if (!db_.commit()) {
                if (error) *error = db_.lastError().text();
                return false;
            }
            return true;
        }
        if (error) *error = QStringLiteral("Workflow 终态事件没有可绑定的 Running 步骤。");
        db_.rollback();
        return false;
    }
    const QString workflowRunText = stepQuery.value(0).toString();
    const QString workflowStepText = stepQuery.value(1).toString();
    WorkflowRunId workflowRunId;
    WorkflowStepId workflowStepId;
    if (!WorkflowRunId::parse(workflowRunText, &workflowRunId, error)
        || !WorkflowStepId::parse(workflowStepText, &workflowStepId, error)) {
        db_.rollback();
        return false;
    }
    if (stepQuery.next()) {
        if (error) *error = QStringLiteral("同一根任务存在多个 Running Workflow 步骤，拒绝绑定终态事件。");
        db_.rollback();
        return false;
    }

    if (outputArtifactId.isValid()) {
        QSqlQuery artifactQuery(db_);
        artifactQuery.prepare(QStringLiteral("select task_id from artifacts where id = :id"));
        artifactQuery.bindValue(QStringLiteral(":id"), outputArtifactId.toString());
        if (!artifactQuery.exec() || !artifactQuery.next()
            || artifactQuery.value(0).toString() != envelope.taskId.toString()) {
            if (error) *error = artifactQuery.lastError().isValid() ? sqlError(artifactQuery)
                : QStringLiteral("Workflow 终态输出 Artifact 不属于根任务。");
            db_.rollback();
            return false;
        }
    }

    if (!eventAlreadyStored) {
        QSqlQuery insertEvent(db_);
        insertEvent.prepare(QStringLiteral("insert into task_events(id, task_id, request_id, sequence, kind, occurred_at, payload_json) values(:id, :task_id, :request_id, :sequence, :kind, :occurred_at, :payload_json)"));
        insertEvent.bindValue(QStringLiteral(":id"), envelope.messageId.toString());
        insertEvent.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
        insertEvent.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
        insertEvent.bindValue(QStringLiteral(":sequence"), static_cast<qint64>(envelope.sequence));
        insertEvent.bindValue(QStringLiteral(":kind"), envelope.kind);
        insertEvent.bindValue(QStringLiteral(":occurred_at"), utcText(envelope.timestamp));
        insertEvent.bindValue(QStringLiteral(":payload_json"), payloadJson);
        if (!insertEvent.exec()) {
            if (error) *error = sqlError(insertEvent);
            db_.rollback();
            return false;
        }
    }

    QSqlQuery outbox(db_);
    outbox.prepare(QStringLiteral("select task_id, workflow_run_id, workflow_step_id, coalesce(output_artifact_id,''), state from workflow_terminal_outbox where message_id = :message_id"));
    outbox.bindValue(QStringLiteral(":message_id"), envelope.messageId.toString());
    if (!outbox.exec()) {
        if (error) *error = sqlError(outbox);
        db_.rollback();
        return false;
    }
    if (outbox.next()) {
        if (outbox.value(0).toString() != envelope.taskId.toString()
            || outbox.value(1).toString() != workflowRunId.toString()
            || outbox.value(2).toString() != workflowStepId.toString()
            || outbox.value(3).toString() != outputArtifactId.toString()) {
            if (error) *error = QStringLiteral("Workflow 终态 outbox 已按不同 Workflow/Artifact 事实登记。");
            db_.rollback();
            return false;
        }
        if (!db_.commit()) {
            if (error) *error = db_.lastError().text();
            return false;
        }
        if (idempotent) *idempotent = true;
        return true;
    }

    QSqlQuery insertOutbox(db_);
    insertOutbox.prepare(QStringLiteral("insert into workflow_terminal_outbox(message_id, task_id, request_id, workflow_run_id, workflow_step_id, sequence, kind, occurred_at, payload_json, output_artifact_id, state, created_at, applied_at) values(:message_id, :task_id, :request_id, :workflow_run_id, :workflow_step_id, :sequence, :kind, :occurred_at, :payload_json, :output_artifact_id, 'pending', :created_at, null)"));
    insertOutbox.bindValue(QStringLiteral(":message_id"), envelope.messageId.toString());
    insertOutbox.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
    insertOutbox.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
    insertOutbox.bindValue(QStringLiteral(":workflow_run_id"), workflowRunId.toString());
    insertOutbox.bindValue(QStringLiteral(":workflow_step_id"), workflowStepId.toString());
    insertOutbox.bindValue(QStringLiteral(":sequence"), static_cast<qint64>(envelope.sequence));
    insertOutbox.bindValue(QStringLiteral(":kind"), envelope.kind);
    insertOutbox.bindValue(QStringLiteral(":occurred_at"), utcText(envelope.timestamp));
    insertOutbox.bindValue(QStringLiteral(":payload_json"), payloadJson);
    insertOutbox.bindValue(QStringLiteral(":output_artifact_id"), outputArtifactId.isValid()
        ? QVariant(outputArtifactId.toString()) : QVariant());
    insertOutbox.bindValue(QStringLiteral(":created_at"), utcText(QDateTime::currentDateTimeUtc()));
    if (!insertOutbox.exec()) {
        if (error) *error = sqlError(insertOutbox);
        db_.rollback();
        return false;
    }
    if (!db_.commit()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    return true;
}

QVector<WorkflowTerminalEventSnapshot> ProjectStore::pendingWorkflowTerminalEvents(
    int limit, QString* error) const
{
    QVector<WorkflowTerminalEventSnapshot> results;
    if (limit <= 0) {
        if (error) *error = QStringLiteral("查询 Workflow 终态 outbox 需要正数 limit。");
        return results;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select message_id, request_id, task_id, workflow_run_id, workflow_step_id, sequence, kind, occurred_at, payload_json, coalesce(output_artifact_id,''), state from workflow_terminal_outbox where state = 'pending' order by created_at asc, message_id asc limit :limit"));
    query.bindValue(QStringLiteral(":limit"), limit);
    if (!query.exec()) {
        if (error) *error = sqlError(query);
        return {};
    }
    while (query.next()) {
        WorkflowTerminalEventSnapshot item;
        if (!MessageId::parse(query.value(0).toString(), &item.messageId, error)
            || !RequestId::parse(query.value(1).toString(), &item.requestId, error)
            || !TaskId::parse(query.value(2).toString(), &item.taskId, error)
            || !WorkflowRunId::parse(query.value(3).toString(), &item.workflowRunId, error)
            || !WorkflowStepId::parse(query.value(4).toString(), &item.workflowStepId, error)) {
            return {};
        }
        item.sequence = query.value(5).toULongLong();
        item.kind = query.value(6).toString();
        item.occurredAt = parseUtc(query.value(7).toString());
        const QJsonDocument document = QJsonDocument::fromJson(query.value(8).toString().toUtf8());
        if (!document.isObject()) {
            if (error) *error = QStringLiteral("Workflow 终态 outbox payload 不是 JSON 对象。");
            return {};
        }
        item.payload = document.object();
        const QString artifactText = query.value(9).toString();
        if (!artifactText.isEmpty() && !ArtifactId::parse(artifactText, &item.outputArtifactId, error)) return {};
        item.applied = query.value(10).toString() == QStringLiteral("applied");
        results.append(item);
    }
    return results;
}

bool ProjectStore::workflowTerminalEventApplied(const MessageId& messageId,
    bool* applied, QString* error, bool* exists) const
{
    if (!messageId.isValid() || !applied) {
        if (error) *error = QStringLiteral("查询 Workflow 终态 outbox 状态需要有效 messageId 和输出对象。");
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select state from workflow_terminal_outbox where message_id = :message_id"));
    query.bindValue(QStringLiteral(":message_id"), messageId.toString());
    if (!query.exec()) {
        if (error) *error = sqlError(query);
        return false;
    }
    if (!query.next()) {
        // 普通 Adapter 任务只写入 task_events，不会有 outbox 行；将其
        // 视为“未进入 Workflow outbox”，让调用方继续走普通终态回调。
        *applied = false;
        if (exists) *exists = false;
        if (error) error->clear();
        return true;
    }
    if (exists) *exists = true;
    *applied = query.value(0).toString() == QStringLiteral("applied");
    return true;
}

bool ProjectStore::workflowTerminalEventBinding(const MessageId& messageId,
    WorkflowRunId* workflowRunId, WorkflowStepId* workflowStepId,
    QString* error) const
{
    if (!messageId.isValid() || !workflowRunId || !workflowStepId) {
        if (error) *error = QStringLiteral("查询 Workflow 终态绑定需要有效 messageId 和输出对象。");
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral(
        "select workflow_run_id, workflow_step_id from workflow_terminal_outbox "
        "where message_id = :message_id"));
    query.bindValue(QStringLiteral(":message_id"), messageId.toString());
    if (!query.exec()) {
        if (error) *error = sqlError(query);
        return false;
    }
    if (!query.next()) {
        if (error) *error = QStringLiteral("Workflow 终态 outbox 记录不存在。");
        return false;
    }
    return WorkflowRunId::parse(query.value(0).toString(), workflowRunId, error)
        && WorkflowStepId::parse(query.value(1).toString(), workflowStepId, error);
}

bool ProjectStore::markWorkflowTerminalEventApplied(const MessageId& messageId,
    QString* error)
{
    if (!messageId.isValid()) {
        if (error) *error = QStringLiteral("标记 Workflow 终态 outbox 需要有效 messageId。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("update workflow_terminal_outbox set state = 'applied', applied_at = :applied_at where message_id = :message_id and state = 'pending'"));
    query.bindValue(QStringLiteral(":applied_at"), utcText(QDateTime::currentDateTimeUtc()));
    query.bindValue(QStringLiteral(":message_id"), messageId.toString());
    if (!query.exec()) {
        if (error) *error = sqlError(query);
        db_.rollback();
        return false;
    }
    if (query.numRowsAffected() == 0) {
        QSqlQuery existing(db_);
        existing.prepare(QStringLiteral("select state from workflow_terminal_outbox where message_id = :message_id"));
        existing.bindValue(QStringLiteral(":message_id"), messageId.toString());
        if (!existing.exec() || !existing.next()) {
            if (error) *error = existing.lastError().isValid() ? sqlError(existing)
                : QStringLiteral("Workflow 终态 outbox 记录不存在。");
            db_.rollback();
            return false;
        }
        if (existing.value(0).toString() != QStringLiteral("applied")) {
            if (error) *error = QStringLiteral("Workflow 终态 outbox 状态无法标记为 applied。");
            db_.rollback();
            return false;
        }
    }
    if (!db_.commit()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    return true;
}

bool ProjectStore::applyProtocolEvent(const ProtocolEnvelope& envelope,
    const ProtocolEventEffect& effect,
    ProtocolEventApplyResult* result,
    QString* error)
{
    ProtocolEventApplyResult localResult;
    if (!validateProtocolEnvelope(envelope, error)
        || envelope.sequence > static_cast<quint64>(std::numeric_limits<qint64>::max())) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("协议事件序号超出 SQLite INTEGER 安全范围。" );
        }
        return false;
    }
    if (effect.hasMetric() && (!std::isfinite(effect.metricValue)
        || effect.metricName.trimmed().isEmpty())) {
        if (error) *error = QStringLiteral("指标事件字段无效。" );
        return false;
    }
    if (effect.hasArtifact() && (!effect.artifactId.isValid()
        || effect.artifactKind.trimmed().isEmpty())) {
        if (error) *error = QStringLiteral("产物事件必须同时包含有效 Artifact ID 和 kind。" );
        return false;
    }
    if (effect.hasTerminal() && !isTerminalTaskState(effect.terminalState)) {
        if (error) *error = QStringLiteral("协议事件终态无效。" );
        return false;
    }
    if (effect.hasTerminal() && !completeTerminalFailure(effect.terminalState, effect.terminalFailure)) {
        if (error) *error = QStringLiteral("协议终态 Failure 字段不完整或与终态不匹配。");
        return false;
    }

    const QString payloadJson = QString::fromUtf8(QJsonDocument(
        protocol::redactPhysicalPathFields(envelope.payload)).toJson(QJsonDocument::Compact));
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }

    QSqlQuery taskQuery(db_);
    taskQuery.prepare(QStringLiteral("select state from tasks where id = :task_id and request_id = :request_id"));
    taskQuery.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
    taskQuery.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
    if (!taskQuery.exec() || !taskQuery.next()) {
        if (error) *error = taskQuery.lastError().isValid()
            ? sqlError(taskQuery) : QStringLiteral("协议事件不属于指定的任务和请求。" );
        db_.rollback();
        return false;
    }
    TaskState currentState;
    if (!taskStateFromString(taskQuery.value(0).toString(), &currentState)) {
        if (error) *error = QStringLiteral("任务状态无效。" );
        db_.rollback();
        return false;
    }
    if (!effect.hasTerminal() && isTerminalTaskState(currentState)) {
        if (error) *error = QStringLiteral("任务已进入终态，不能继续接收协议事件。" );
        db_.rollback();
        return false;
    }

    // 先识别同一 request/sequence，保证重放不会再次产生副作用。
    QSqlQuery duplicate(db_);
    duplicate.prepare(QStringLiteral("select id, task_id, kind, payload_json from task_events where request_id = :request_id and sequence = :sequence"));
    duplicate.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
    duplicate.bindValue(QStringLiteral(":sequence"), static_cast<qint64>(envelope.sequence));
    if (!duplicate.exec()) {
        if (error) *error = sqlError(duplicate);
        db_.rollback();
        return false;
    }
    if (duplicate.next()) {
        if (duplicate.value(0).toString() != envelope.messageId.toString()
            || duplicate.value(1).toString() != envelope.taskId.toString()
            || duplicate.value(2).toString() != envelope.kind
            || duplicate.value(3).toString() != payloadJson) {
            if (error) *error = QStringLiteral("协议事件序号已被不同事件占用。" );
            db_.rollback();
            return false;
        }
        localResult.idempotent = true;
        localResult.effectiveTaskState = currentState;
        if (!db_.commit()) {
            if (error) *error = db_.lastError().text();
            return false;
        }
        if (result) *result = localResult;
        return true;
    }

    QSqlQuery existingMessage(db_);
    existingMessage.prepare(QStringLiteral("select request_id, sequence from task_events where id = :id"));
    existingMessage.bindValue(QStringLiteral(":id"), envelope.messageId.toString());
    if (!existingMessage.exec()) {
        if (error) *error = sqlError(existingMessage);
        db_.rollback();
        return false;
    }
    if (existingMessage.next()) {
        if (error) *error = QStringLiteral("协议事件 messageId 已用于另一条事件。" );
        db_.rollback();
        return false;
    }

    QSqlQuery sequenceQuery(db_);
    sequenceQuery.prepare(QStringLiteral("select coalesce(max(sequence), 0) from task_events where request_id = :request_id and kind <> 'task.state_changed'"));
    sequenceQuery.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
    if (!sequenceQuery.exec() || !sequenceQuery.next()) {
        if (error) *error = sqlError(sequenceQuery);
        db_.rollback();
        return false;
    }
    const qint64 previousSequence = sequenceQuery.value(0).toLongLong();
    if (previousSequence < 0 || envelope.sequence <= static_cast<quint64>(previousSequence)) {
        if (error) *error = QStringLiteral("协议事件 sequence 必须严格递增。" );
        db_.rollback();
        return false;
    }

    TaskState effectiveState = currentState;
    Failure effectiveFailure = effect.terminalFailure;
    if (effect.hasTerminal()) {
        effectiveState = effect.terminalState;
        if (currentState == TaskState::CancelRequested && effectiveState != TaskState::Canceled) {
            effectiveState = TaskState::Canceled;
            effectiveFailure = cancellationWinsFailure(effectiveFailure, envelope.timestamp);
        }
        if (!isValidTaskStateTransition(currentState, effectiveState)) {
            if (error) *error = QStringLiteral("终态事件与当前任务状态不兼容。" );
            db_.rollback();
            return false;
        }
    }

    QSqlQuery insertEvent(db_);
    insertEvent.prepare(QStringLiteral("insert into task_events(id, task_id, request_id, sequence, kind, occurred_at, payload_json) values(:id, :task_id, :request_id, :sequence, :kind, :occurred_at, :payload_json)"));
    insertEvent.bindValue(QStringLiteral(":id"), envelope.messageId.toString());
    insertEvent.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
    insertEvent.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
    insertEvent.bindValue(QStringLiteral(":sequence"), static_cast<qint64>(envelope.sequence));
    insertEvent.bindValue(QStringLiteral(":kind"), envelope.kind);
    insertEvent.bindValue(QStringLiteral(":occurred_at"), utcText(envelope.timestamp));
    insertEvent.bindValue(QStringLiteral(":payload_json"), payloadJson);
    if (!insertEvent.exec()) {
        if (error) *error = sqlError(insertEvent);
        db_.rollback();
        return false;
    }

    if (effect.hasMetric()) {
        QSqlQuery metric(db_);
        metric.prepare(QStringLiteral("insert into task_metrics(id, task_id, name, value, occurred_at) values(:id, :task_id, :name, :value, :occurred_at)"));
        metric.bindValue(QStringLiteral(":id"), MessageId::create().toString());
        metric.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
        metric.bindValue(QStringLiteral(":name"), effect.metricName.trimmed());
        metric.bindValue(QStringLiteral(":value"), effect.metricValue);
        metric.bindValue(QStringLiteral(":occurred_at"), utcText(envelope.timestamp));
        if (!metric.exec()) {
            if (error) *error = sqlError(metric);
            db_.rollback();
            return false;
        }
    }

    if (effect.hasArtifact()) {
        QSqlQuery existingArtifact(db_);
        existingArtifact.prepare(QStringLiteral("select task_id, kind from artifacts where id = :id"));
        existingArtifact.bindValue(QStringLiteral(":id"), effect.artifactId.toString());
        if (!existingArtifact.exec()) {
            if (error) *error = sqlError(existingArtifact);
            db_.rollback();
            return false;
        }
        if (existingArtifact.next()) {
            if (error) *error = QStringLiteral("协议事件引用的 Artifact ID 已经登记，拒绝重复关联。" );
            db_.rollback();
            return false;
        }
        QSqlQuery artifact(db_);
        artifact.prepare(QStringLiteral("insert into artifacts(id, task_id, kind, created_at) values(:id, :task_id, :kind, :created_at)"));
        artifact.bindValue(QStringLiteral(":id"), effect.artifactId.toString());
        artifact.bindValue(QStringLiteral(":task_id"), envelope.taskId.toString());
        artifact.bindValue(QStringLiteral(":kind"), effect.artifactKind.trimmed());
        artifact.bindValue(QStringLiteral(":created_at"), utcText(envelope.timestamp));
        if (!artifact.exec()) {
            if (error) *error = sqlError(artifact);
            db_.rollback();
            return false;
        }
    }

    if (effect.hasTerminal()) {
        QSqlQuery update(db_);
        update.prepare(QStringLiteral("update tasks set state = :state, failure_code = :failure_code, failure_details = :details, failure_suggested_action = :action, failure_occurred_at = :occurred_at, updated_at = :updated_at where id = :id and request_id = :request_id and state = :expected_state"));
        update.bindValue(QStringLiteral(":state"), taskStateToString(effectiveState));
        update.bindValue(QStringLiteral(":failure_code"), failureCodeToString(effectiveFailure.code));
        update.bindValue(QStringLiteral(":details"), requiredText(effectiveFailure.message));
        update.bindValue(QStringLiteral(":action"), requiredText(effectiveFailure.suggestedAction));
        update.bindValue(QStringLiteral(":occurred_at"), effectiveFailure.isFailure()
            ? utcText(effectiveFailure.occurredAt.isValid() ? effectiveFailure.occurredAt : envelope.timestamp) : QVariant());
        update.bindValue(QStringLiteral(":updated_at"), utcText(envelope.timestamp));
        update.bindValue(QStringLiteral(":id"), envelope.taskId.toString());
        update.bindValue(QStringLiteral(":request_id"), envelope.requestId.toString());
        update.bindValue(QStringLiteral(":expected_state"), taskStateToString(currentState));
        if (!update.exec() || update.numRowsAffected() != 1
            || !appendStateEvent(envelope.taskId, envelope.requestId, effectiveState, effectiveFailure,
                envelope.timestamp, error)) {
            if (error && error->isEmpty()) *error = update.lastError().isValid()
                ? sqlError(update) : QStringLiteral("任务终态已被并发更新。" );
            db_.rollback();
            return false;
        }
    }

    localResult.effectiveTaskState = effectiveState;
    if (!db_.commit()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    if (result) *result = localResult;
    return true;
}

bool ProjectStore::lastProtocolSequence(const TaskId& taskId, quint64* result, QString* error) const
{
    if (!result || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral("读取最近协议事件序号需要有效任务和输出对象。");
        }
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select coalesce(max(sequence), 0) from task_events "
        "where task_id = :task_id and kind <> 'task.state_changed'"));
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    if (!query.exec() || !query.next()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    const qint64 sequence = query.value(0).toLongLong();
    if (sequence < 0) {
        if (error) {
            *error = QStringLiteral("存储中的协议事件序号无效。");
        }
        return false;
    }
    *result = static_cast<quint64>(sequence);
    return true;
}

bool ProjectStore::recordMetric(const TaskId& taskId, const QString& name, double value, const QDateTime& occurredAt, QString* error)
{
    if (!taskId.isValid() || name.isEmpty() || !std::isfinite(value) || !occurredAt.isValid()) {
        if (error) {
            *error = QStringLiteral("指标字段无效。");
        }
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into task_metrics(id, task_id, name, value, occurred_at) values(:id, :task_id, :name, :value, :occurred_at)"));
    query.bindValue(QStringLiteral(":id"), MessageId::create().toString());
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    query.bindValue(QStringLiteral(":name"), name);
    query.bindValue(QStringLiteral(":value"), value);
    query.bindValue(QStringLiteral(":occurred_at"), utcText(occurredAt));
    if (query.exec()) {
        return true;
    }
    if (error) {
        *error = sqlError(query);
    }
    return false;
}

bool ProjectStore::recordArtifact(const ArtifactId& artifactId, const TaskId& taskId, const QString& kind, const QDateTime& createdAt, QString* error)
{
    if (!artifactId.isValid() || !taskId.isValid() || kind.isEmpty() || !createdAt.isValid()) {
        if (error) {
            *error = QStringLiteral("产物字段无效。");
        }
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into artifacts(id, task_id, kind, created_at) values(:id, :task_id, :kind, :created_at)"));
    query.bindValue(QStringLiteral(":id"), artifactId.toString());
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    query.bindValue(QStringLiteral(":kind"), kind);
    query.bindValue(QStringLiteral(":created_at"), utcText(createdAt));
    if (query.exec()) {
        return true;
    }
    if (error) {
        *error = sqlError(query);
    }
    return false;
}

bool ProjectStore::recordArtifactWithFiles(const ArtifactId& artifactId,
    const TaskId& taskId,
    const QString& kind,
    const QVector<ArtifactFileSnapshot>& files,
    const QDateTime& createdAt,
    QString* error)
{
    if (files.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Artifact 至少需要一个文件记录。");
        }
        return false;
    }
    QSet<QString> normalizedPaths;
    for (const ArtifactFileSnapshot& file : files) {
        QString normalizedPath;
        if (!normalizeArtifactMemberPath(file.relativePath, &normalizedPath, error)
            || normalizedPath != file.relativePath
            || normalizedPaths.contains(normalizedPath.toCaseFolded())
            || !isSha256Hex(file.sha256) || file.byteCount < 0) {
            if (error) {
                *error = QStringLiteral("Artifact 文件记录无效。");
            }
            return false;
        }
        normalizedPaths.insert(normalizedPath.toCaseFolded());
    }
    if (!db_.transaction()) {
        if (error) {
            *error = db_.lastError().text();
        }
        return false;
    }
    if (!recordArtifact(artifactId, taskId, kind, createdAt, error)) {
        db_.rollback();
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into artifact_files(id, artifact_id, relative_path, sha256, byte_count) values(:id, :artifact_id, :relative_path, :sha256, :byte_count)"));
    for (const ArtifactFileSnapshot& file : files) {
        query.bindValue(QStringLiteral(":id"), MessageId::create().toString());
        query.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
        query.bindValue(QStringLiteral(":relative_path"), file.relativePath);
        query.bindValue(QStringLiteral(":sha256"), file.sha256);
        query.bindValue(QStringLiteral(":byte_count"), file.byteCount);
        if (!query.exec()) {
            if (error) {
                *error = sqlError(query);
            }
            db_.rollback();
            return false;
        }
    }
    if (db_.commit()) {
        return true;
    }
    if (error) {
        *error = db_.lastError().text();
    }
    return false;
}

bool ProjectStore::recordEvidenceArtifactWithFilesAndAttachTerminalization(
    const ArtifactId& artifactId,
    const TaskId& taskId,
    const WorkflowRunId& workflowRunId,
    const QVector<ArtifactFileSnapshot>& files,
    const QDateTime& createdAt,
    QString* error)
{
    if (!artifactId.isValid() || !taskId.isValid() || !workflowRunId.isValid()
        || !createdAt.isValid() || files.isEmpty()) {
        if (error) *error = QStringLiteral("原子提交 Evidence 需要有效标识、时间和文件清单。");
        return false;
    }
    QSet<QString> paths;
    for (const ArtifactFileSnapshot& file : files) {
        QString normalizedPath;
        if (!normalizeArtifactMemberPath(file.relativePath, &normalizedPath, error)
            || normalizedPath != file.relativePath || !isSha256Hex(file.sha256) || file.byteCount < 0
            || paths.contains(normalizedPath.toCaseFolded())) {
            if (error) *error = QStringLiteral("Evidence Artifact 文件记录无效或路径重复。");
            return false;
        }
        paths.insert(normalizedPath.toCaseFolded());
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery terminalization(db_);
    terminalization.prepare(QStringLiteral("select task_id, state, coalesce(evidence_artifact_id,'') from workflow_terminalizations where workflow_run_id = :id"));
    terminalization.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!terminalization.exec() || !terminalization.next()
        || terminalization.value(0).toString() != taskId.toString()) {
        if (error) *error = terminalization.lastError().isValid() ? sqlError(terminalization)
            : QStringLiteral("Evidence 必须关联同一根任务已封存的工作流终态。");
        db_.rollback();
        return false;
    }
    const QString terminalizationState = terminalization.value(1).toString();
    const QString attachedArtifact = terminalization.value(2).toString();
    if (terminalizationState != QStringLiteral("sealed")
        && !((terminalizationState == QStringLiteral("evidence_attached")
                || terminalizationState == QStringLiteral("closed"))
            && attachedArtifact == artifactId.toString())) {
        if (error) *error = QStringLiteral("工作流终态已关联不同 Evidence 或已经关闭。");
        db_.rollback();
        return false;
    }

    bool existing = false;
    QSqlQuery artifactQuery(db_);
    artifactQuery.prepare(QStringLiteral("select task_id, kind, created_at from artifacts where id = :id"));
    artifactQuery.bindValue(QStringLiteral(":id"), artifactId.toString());
    if (!artifactQuery.exec()) {
        if (error) *error = sqlError(artifactQuery);
        db_.rollback();
        return false;
    }
    if (artifactQuery.next()) {
        existing = true;
        if (artifactQuery.value(0).toString() != taskId.toString()
            || artifactQuery.value(1).toString() != QStringLiteral("evidence_bundle")
            || parseUtc(artifactQuery.value(2).toString()) != createdAt.toUTC()) {
            if (error) *error = QStringLiteral("Artifact ID 已按不同 Evidence 元数据登记。");
            db_.rollback();
            return false;
        }
        QSqlQuery filesQuery(db_);
        filesQuery.prepare(QStringLiteral("select relative_path, sha256, byte_count from artifact_files where artifact_id = :id"));
        filesQuery.bindValue(QStringLiteral(":id"), artifactId.toString());
        if (!filesQuery.exec()) {
            if (error) *error = sqlError(filesQuery);
            db_.rollback();
            return false;
        }
        QHash<QString, QPair<QString, qint64>> storedFiles;
        while (filesQuery.next()) {
            storedFiles.insert(filesQuery.value(0).toString(),
                {filesQuery.value(1).toString(), filesQuery.value(2).toLongLong()});
        }
        if (storedFiles.size() != files.size()) {
            if (error) *error = QStringLiteral("Artifact ID 已按不同 Evidence 文件清单登记。");
            db_.rollback();
            return false;
        }
        for (const ArtifactFileSnapshot& file : files) {
            if (!storedFiles.contains(file.relativePath)
                || storedFiles.value(file.relativePath).first != file.sha256
                || storedFiles.value(file.relativePath).second != file.byteCount) {
                if (error) *error = QStringLiteral("Artifact ID 已按不同 Evidence 文件清单登记。");
                db_.rollback();
                return false;
            }
        }
    }
    if (!existing) {
        QSqlQuery insertArtifact(db_);
        insertArtifact.prepare(QStringLiteral("insert into artifacts(id, task_id, kind, created_at) values(:id, :task_id, 'evidence_bundle', :created_at)"));
        insertArtifact.bindValue(QStringLiteral(":id"), artifactId.toString());
        insertArtifact.bindValue(QStringLiteral(":task_id"), taskId.toString());
        insertArtifact.bindValue(QStringLiteral(":created_at"), utcText(createdAt));
        if (!insertArtifact.exec()) {
            if (error) *error = sqlError(insertArtifact);
            db_.rollback();
            return false;
        }
        QSqlQuery insertFile(db_);
        insertFile.prepare(QStringLiteral("insert into artifact_files(id, artifact_id, relative_path, sha256, byte_count) values(:id, :artifact_id, :relative_path, :sha256, :byte_count)"));
        for (const ArtifactFileSnapshot& file : files) {
            insertFile.bindValue(QStringLiteral(":id"), MessageId::create().toString());
            insertFile.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
            insertFile.bindValue(QStringLiteral(":relative_path"), file.relativePath);
            insertFile.bindValue(QStringLiteral(":sha256"), file.sha256);
            insertFile.bindValue(QStringLiteral(":byte_count"), file.byteCount);
            if (!insertFile.exec()) {
                if (error) *error = sqlError(insertFile);
                db_.rollback();
                return false;
            }
        }
    }
    if (terminalizationState == QStringLiteral("sealed")) {
        QSqlQuery attach(db_);
        attach.prepare(QStringLiteral("update workflow_terminalizations set state = 'evidence_attached', evidence_artifact_id = :artifact_id, evidence_attached_at = :now where workflow_run_id = :id and state = 'sealed' and evidence_artifact_id is null"));
        attach.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
        attach.bindValue(QStringLiteral(":now"), utcText(QDateTime::currentDateTimeUtc()));
        attach.bindValue(QStringLiteral(":id"), workflowRunId.toString());
        if (!attach.exec() || attach.numRowsAffected() != 1) {
            if (error) *error = attach.lastError().isValid() ? sqlError(attach)
                : QStringLiteral("Evidence 关联状态已被并发更新。");
            db_.rollback();
            return false;
        }
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::registerDatasetSnapshot(DatasetSnapshotRecord* snapshot, QString* error)
{
    if (!snapshot || !snapshot->id.isValid() || !snapshot->taskId.isValid() || !snapshot->artifactId.isValid()
        || snapshot->rootPath.trimmed().isEmpty() || snapshot->datasetFormat.trimmed().isEmpty()
        || snapshot->driverId.trimmed().isEmpty() || snapshot->driverVersion.trimmed().isEmpty()
        || !isSha256Hex(snapshot->rootHash) || !isSha256Hex(snapshot->manifestSha256)
        || snapshot->fileCount < 0 || snapshot->totalBytes < 0) {
        if (error) *error = QStringLiteral("登记数据集快照需要完整且已校验的快照元数据。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }

    QSqlQuery artifactQuery(db_);
    artifactQuery.prepare(QStringLiteral("select task_id, kind from artifacts where id = :artifact_id"));
    artifactQuery.bindValue(QStringLiteral(":artifact_id"), snapshot->artifactId.toString());
    if (!artifactQuery.exec() || !artifactQuery.next()
        || artifactQuery.value(0).toString() != snapshot->taskId.toString()
        || artifactQuery.value(1).toString() != QStringLiteral("dataset_snapshot")) {
        if (error) *error = artifactQuery.lastError().isValid() ? sqlError(artifactQuery)
            : QStringLiteral("数据快照必须引用同一任务已提交的 dataset_snapshot Artifact。");
        db_.rollback();
        return false;
    }
    QSqlQuery manifestQuery(db_);
    manifestQuery.prepare(QStringLiteral("select 1 from artifact_files where artifact_id = :artifact_id and relative_path = 'dataset_snapshot.json' and sha256 = :sha256"));
    manifestQuery.bindValue(QStringLiteral(":artifact_id"), snapshot->artifactId.toString());
    manifestQuery.bindValue(QStringLiteral(":sha256"), snapshot->manifestSha256);
    if (!manifestQuery.exec() || !manifestQuery.next()) {
        if (error) *error = manifestQuery.lastError().isValid() ? sqlError(manifestQuery)
            : QStringLiteral("数据快照 Manifest 不属于指定 Artifact。 ");
        db_.rollback();
        return false;
    }

    if (artifactStoreRoot_.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("登记数据集快照需要配置 Artifact Store 根目录。");
        db_.rollback();
        return false;
    }
    const QString rootPath = QDir::cleanPath(QDir(snapshot->rootPath).absolutePath());
    const QString datasetFormat = snapshot->datasetFormat.trimmed();
    const QString expectedRootPath = QDir::cleanPath(QDir(artifactStoreRoot_).filePath(
        QStringLiteral("committed/%1").arg(snapshot->artifactId.toString())));
    if (rootPath.compare(expectedRootPath, Qt::CaseInsensitive) != 0) {
        if (error) *error = QStringLiteral("数据快照可执行根必须属于其 committed Snapshot Artifact。");
        db_.rollback();
        return false;
    }
    const DatasetId datasetId = snapshot->datasetId.isValid() ? snapshot->datasetId : DatasetId::create();
    QSqlQuery datasetLookup(db_);
    datasetLookup.prepare(QStringLiteral("select dataset_format from datasets where id = :id"));
    datasetLookup.bindValue(QStringLiteral(":id"), datasetId.toString());
    if (!datasetLookup.exec()) {
        if (error) *error = sqlError(datasetLookup);
        db_.rollback();
        return false;
    }
    if (datasetLookup.next()) {
        if (datasetLookup.value(0).toString() != datasetFormat) {
            if (error) *error = QStringLiteral("目标 DatasetId 已登记为不同格式。");
            db_.rollback();
            return false;
        }
    } else {
        QSqlQuery insertDataset(db_);
        insertDataset.prepare(QStringLiteral("insert into datasets(id, dataset_format, created_at) values(:id, :dataset_format, :created_at)"));
        insertDataset.bindValue(QStringLiteral(":id"), datasetId.toString());
        insertDataset.bindValue(QStringLiteral(":dataset_format"), datasetFormat);
        insertDataset.bindValue(QStringLiteral(":created_at"), utcText(QDateTime::currentDateTimeUtc()));
        if (!insertDataset.exec()) {
            if (error) *error = sqlError(insertDataset);
            db_.rollback();
            return false;
        }
    }

    DatasetVersionId versionId;
    QSqlQuery versionLookup(db_);
    versionLookup.prepare(QStringLiteral("select id from dataset_versions where dataset_id = :dataset_id and root_hash = :root_hash"));
    versionLookup.bindValue(QStringLiteral(":dataset_id"), datasetId.toString());
    versionLookup.bindValue(QStringLiteral(":root_hash"), snapshot->rootHash);
    if (!versionLookup.exec()) {
        if (error) *error = sqlError(versionLookup);
        db_.rollback();
        return false;
    }
    if (versionLookup.next()) {
        if (!DatasetVersionId::parse(versionLookup.value(0).toString(), &versionId, error)) {
            db_.rollback();
            return false;
        }
    } else {
        versionId = DatasetVersionId::create();
        QSqlQuery insertVersion(db_);
        insertVersion.prepare(QStringLiteral("insert into dataset_versions(id, dataset_id, root_hash, created_at) values(:id, :dataset_id, :root_hash, :created_at)"));
        insertVersion.bindValue(QStringLiteral(":id"), versionId.toString());
        insertVersion.bindValue(QStringLiteral(":dataset_id"), datasetId.toString());
        insertVersion.bindValue(QStringLiteral(":root_hash"), snapshot->rootHash);
        insertVersion.bindValue(QStringLiteral(":created_at"), utcText(QDateTime::currentDateTimeUtc()));
        if (!insertVersion.exec()) {
            if (error) *error = sqlError(insertVersion);
            db_.rollback();
            return false;
        }
    }

    const QDateTime createdAt = snapshot->createdAt.isValid() ? snapshot->createdAt.toUTC() : QDateTime::currentDateTimeUtc();
    QSqlQuery insertSnapshot(db_);
    insertSnapshot.prepare(QStringLiteral("insert into dataset_snapshots(id, dataset_version_id, task_id, artifact_id, root_path, driver_id, driver_version, manifest_sha256, file_count, total_bytes, created_at) values(:id, :dataset_version_id, :task_id, :artifact_id, :root_path, :driver_id, :driver_version, :manifest_sha256, :file_count, :total_bytes, :created_at)"));
    insertSnapshot.bindValue(QStringLiteral(":id"), snapshot->id.toString());
    insertSnapshot.bindValue(QStringLiteral(":dataset_version_id"), versionId.toString());
    insertSnapshot.bindValue(QStringLiteral(":task_id"), snapshot->taskId.toString());
    insertSnapshot.bindValue(QStringLiteral(":artifact_id"), snapshot->artifactId.toString());
    insertSnapshot.bindValue(QStringLiteral(":root_path"), rootPath);
    insertSnapshot.bindValue(QStringLiteral(":driver_id"), snapshot->driverId.trimmed());
    insertSnapshot.bindValue(QStringLiteral(":driver_version"), snapshot->driverVersion.trimmed());
    insertSnapshot.bindValue(QStringLiteral(":manifest_sha256"), snapshot->manifestSha256);
    insertSnapshot.bindValue(QStringLiteral(":file_count"), static_cast<qint64>(snapshot->fileCount));
    insertSnapshot.bindValue(QStringLiteral(":total_bytes"), snapshot->totalBytes);
    insertSnapshot.bindValue(QStringLiteral(":created_at"), utcText(createdAt));
    if (!insertSnapshot.exec()) {
        if (error) *error = sqlError(insertSnapshot);
        db_.rollback();
        return false;
    }
    if (!db_.commit()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    snapshot->datasetId = datasetId;
    snapshot->datasetVersionId = versionId;
    snapshot->rootPath = rootPath;
    snapshot->datasetFormat = datasetFormat;
    snapshot->createdAt = createdAt;
    return true;
}

bool ProjectStore::datasetSnapshot(const SnapshotId& snapshotId, DatasetSnapshotRecord* result, QString* error) const
{
    return DatasetCatalogRepository(database_).snapshot(
        snapshotId, result, error);
}

bool ProjectStore::datasetSnapshotForArtifact(const ArtifactId& artifactId,
    DatasetSnapshotRecord* result,
    QString* error) const
{
    return DatasetCatalogRepository(database_).snapshotForArtifact(
        artifactId, result, error);
}

Page<DatasetCatalogItem> ProjectStore::datasets(const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<DatasetCatalogItem> result =
        DatasetCatalogRepository(database_).page(request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

bool ProjectStore::artifactDiscardable(
    const ArtifactId& artifactId, bool* discardable, QString* error) const
{
    return ArtifactCatalogRepository(database_).isDiscardable(
        artifactId, discardable, error);
}

bool ProjectStore::removeUnreferencedArtifact(const ArtifactId& artifactId, QString* error)
{
    if (!artifactId.isValid()) {
        if (error) *error = QStringLiteral("删除 Artifact 需要有效 ID。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    bool discardable = false;
    if (!artifactDiscardable(artifactId, &discardable, error) || !discardable) {
        if (error && error->isEmpty()) *error = QStringLiteral("Artifact 已被引用，不能删除。");
        db_.rollback();
        return false;
    }
    QSqlQuery filesQuery(db_);
    filesQuery.prepare(QStringLiteral("delete from artifact_files where artifact_id = :artifact_id"));
    filesQuery.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
    QSqlQuery artifactQuery(db_);
    artifactQuery.prepare(QStringLiteral("delete from artifacts where id = :artifact_id"));
    artifactQuery.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
    if (!filesQuery.exec() || !artifactQuery.exec() || artifactQuery.numRowsAffected() != 1) {
        if (error) *error = artifactQuery.lastError().isValid() ? sqlError(artifactQuery) : QStringLiteral("Artifact 不存在或无法删除。");
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::registerModelPackage(const ModelPackageSnapshot& modelPackage, QString* error)
{
    const ModelManifest& manifest = modelPackage.manifest;
    if (!modelPackage.sourceArtifactId.isValid() || !validateModelManifest(manifest, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("注册模型包需要有效来源 Artifact 和 Manifest。");
        }
        return false;
    }

    QSqlQuery sourceQuery(db_);
    sourceQuery.prepare(QStringLiteral("select task_id from artifacts where id = :artifact_id"));
    sourceQuery.bindValue(QStringLiteral(":artifact_id"), modelPackage.sourceArtifactId.toString());
    if (!sourceQuery.exec() || !sourceQuery.next()) {
        if (error) *error = sourceQuery.lastError().isValid() ? sqlError(sourceQuery) : QStringLiteral("模型包来源 Artifact 不存在。");
        return false;
    }
    if (sourceQuery.value(0).toString() != manifest.sourceTaskId.toString()) {
        if (error) *error = QStringLiteral("Model Manifest 的 sourceTaskId 与来源 Artifact 不一致。");
        return false;
    }
    if (modelPackage.sourceSnapshotBinding == ModelSourceSnapshotBinding::ProjectSnapshot) {
        QSqlQuery snapshotQuery(db_);
        snapshotQuery.prepare(QStringLiteral(
            "select 1 from dataset_snapshots where id = :snapshot_id"));
        snapshotQuery.bindValue(QStringLiteral(":snapshot_id"), manifest.sourceSnapshotId.toString());
        if (!snapshotQuery.exec() || !snapshotQuery.next()) {
            if (error) {
                *error = snapshotQuery.lastError().isValid()
                    ? sqlError(snapshotQuery)
                    : QStringLiteral("项目训练模型必须关联已登记的 Dataset Snapshot。");
            }
            return false;
        }
    }

    QSqlQuery hashQuery(db_);
    hashQuery.prepare(QStringLiteral("select 1 from artifact_files where artifact_id = :artifact_id and relative_path = :relative_path and sha256 = :sha256"));
    hashQuery.bindValue(QStringLiteral(":artifact_id"), modelPackage.sourceArtifactId.toString());
    hashQuery.bindValue(QStringLiteral(":relative_path"), manifest.artifactEntryPath);
    hashQuery.bindValue(QStringLiteral(":sha256"), manifest.sourceArtifactSha256);
    if (!hashQuery.exec() || !hashQuery.next()) {
        if (error) *error = hashQuery.lastError().isValid() ? sqlError(hashQuery) : QStringLiteral("Model Manifest 的来源哈希不属于指定 Artifact。");
        return false;
    }

    QString manifestError;
    const QJsonObject encoded = encodeModelManifest(manifest, &manifestError);
    if (encoded.isEmpty()) {
        if (error) *error = manifestError;
        return false;
    }
    const QDateTime createdAt = modelPackage.createdAt.isValid() ? modelPackage.createdAt.toUTC() : QDateTime::currentDateTimeUtc();
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into model_packages(id, model_family, task_type, source_backend, source_task_id, source_snapshot_id, source_snapshot_binding, source_artifact_id, source_artifact_sha256, manifest_json, verified, created_at) values(:id, :model_family, :task_type, :source_backend, :source_task_id, :source_snapshot_id, :source_snapshot_binding, :source_artifact_id, :source_artifact_sha256, :manifest_json, :verified, :created_at)"));
    query.bindValue(QStringLiteral(":id"), manifest.modelPackageId.toString());
    query.bindValue(QStringLiteral(":model_family"), manifest.modelFamily);
    query.bindValue(QStringLiteral(":task_type"), manifest.taskType);
    query.bindValue(QStringLiteral(":source_backend"), manifest.sourceBackend);
    query.bindValue(QStringLiteral(":source_task_id"), manifest.sourceTaskId.toString());
    query.bindValue(QStringLiteral(":source_snapshot_id"), manifest.sourceSnapshotId.toString());
    query.bindValue(QStringLiteral(":source_snapshot_binding"),
        modelSourceSnapshotBindingText(modelPackage.sourceSnapshotBinding));
    query.bindValue(QStringLiteral(":source_artifact_id"), modelPackage.sourceArtifactId.toString());
    query.bindValue(QStringLiteral(":source_artifact_sha256"), manifest.sourceArtifactSha256);
    query.bindValue(QStringLiteral(":manifest_json"), QString::fromUtf8(QJsonDocument(encoded).toJson(QJsonDocument::Compact)));
    query.bindValue(QStringLiteral(":verified"), 1);
    query.bindValue(QStringLiteral(":created_at"), utcText(createdAt));
    if (query.exec()) return true;
    if (error) *error = sqlError(query);
    return false;
}

bool ProjectStore::modelPackage(const ModelPackageId& modelPackageId, ModelPackageSnapshot* result, QString* error) const
{
    return ModelCatalogRepository(database_).read(
        modelPackageId, result, error);
}

Page<ModelPackageSnapshot> ProjectStore::modelPackages(
    const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<ModelPackageSnapshot> result =
        ModelCatalogRepository(database_).page(request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

bool ProjectStore::projectSummary(ProjectSummarySnapshot* result, QString* error) const
{
    return ProjectReadRepository(database_).summary(result, error);
}

bool ProjectStore::createWorkflowRun(const WorkflowRunSnapshot& workflow,
    const QVector<WorkflowStepSnapshot>& steps,
    QString* error)
{
    return createWorkflowRunInternal(workflow, steps, nullptr, error);
}

bool ProjectStore::createWorkflowRunWithInput(const WorkflowRunSnapshot& workflow,
    const QVector<WorkflowStepSnapshot>& steps,
    const WorkflowInputBinding& input,
    QString* error)
{
    return createWorkflowRunInternal(workflow, steps, &input, error);
}

bool ProjectStore::createWorkflowRunInternal(const WorkflowRunSnapshot& workflow,
    const QVector<WorkflowStepSnapshot>& steps,
    const WorkflowInputBinding* input,
    QString* error)
{
    if (!workflow.id.isValid() || !workflow.taskId.isValid() || workflow.templateId.trimmed().isEmpty() || steps.isEmpty()) {
        if (error) *error = QStringLiteral("创建工作流需要有效运行 ID、根任务、模板和至少一个步骤。");
        return false;
    }
    bool exists = false;
    if (!taskExists(workflow.taskId, &exists, error) || !exists) {
        if (error && error->isEmpty()) *error = QStringLiteral("工作流根任务不存在。");
        return false;
    }
    QSet<int> ordinals;
    for (const WorkflowStepSnapshot& step : steps) {
        if (!step.id.isValid() || step.workflowRunId != workflow.id || step.ordinal < 0
            || step.kind.trimmed().isEmpty() || step.backend.trimmed().isEmpty()
            || step.state != WorkflowStepState::Pending || step.outputArtifactId.isValid()
            || step.retryCount != 0 || ordinals.contains(step.ordinal)) {
            if (error) *error = QStringLiteral("新工作流步骤必须具有唯一顺序、pending 状态和明确后端。");
            return false;
        }
        ordinals.insert(step.ordinal);
    }
    for (int ordinal = 0; ordinal < steps.size(); ++ordinal) {
        if (!ordinals.contains(ordinal)) {
            if (error) *error = QStringLiteral("工作流步骤顺序必须从零开始连续编号。");
            return false;
        }
    }
    if (input) {
        DatasetSnapshotRecord snapshot;
        ArtifactSnapshot artifactSnapshot;
        ModelPackageSnapshot modelPackageSnapshot;
        if (input->workflowRunId != workflow.id
            || input->role.trimmed().isEmpty()
            || !input->sourceArtifactId.isValid() || !input->sourceTaskId.isValid()
            || !artifact(input->sourceArtifactId, &artifactSnapshot, error)
            || artifactSnapshot.taskId != input->sourceTaskId
            || artifactSnapshot.kind != input->sourceArtifactKind) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("Workflow 外部输入必须引用匹配生产任务与 kind 的 committed Artifact。");
            }
            return false;
        }
        const bool datasetInput = input->role == QStringLiteral("dataset_snapshot");
        const bool modelInput = input->role == QStringLiteral("model_package");
        const QHash<QString, QString> artifactRoles{
            {QStringLiteral("dataset_repair_manifest"), QStringLiteral("dataset_repair_manifest")},
            {QStringLiteral("annotation_session"), QStringLiteral("annotation_session")}};
        if (datasetInput) {
            if (!input->datasetId.isValid() || !input->datasetSnapshotId.isValid()
                || !input->datasetVersionId.isValid() || input->modelPackageId.isValid()
                || !isSha256Hex(input->manifestSha256) || !isSha256Hex(input->rootHash)
                || input->sourceArtifactKind != QStringLiteral("dataset_snapshot")
                || !datasetSnapshot(input->datasetSnapshotId, &snapshot, error)
                || snapshot.artifactId != input->sourceArtifactId
                || snapshot.taskId != input->sourceTaskId
                || snapshot.datasetId != input->datasetId
                || snapshot.datasetVersionId != input->datasetVersionId
                || snapshot.manifestSha256 != input->manifestSha256
                || snapshot.rootHash != input->rootHash) {
                if (error && error->isEmpty()) {
                    *error = QStringLiteral("Workflow 外部输入必须与已登记 Dataset Snapshot lineage 完全一致。");
                }
                return false;
            }
        } else if (modelInput) {
            if (input->datasetId.isValid() || input->datasetSnapshotId.isValid()
                || input->datasetVersionId.isValid() || !input->modelPackageId.isValid()
                || !input->manifestSha256.isEmpty() || !input->rootHash.isEmpty()
                || !modelPackage(input->modelPackageId, &modelPackageSnapshot, error)
                || modelPackageSnapshot.sourceArtifactId != input->sourceArtifactId
                || modelPackageSnapshot.manifest.sourceTaskId != input->sourceTaskId) {
                if (error && error->isEmpty()) {
                    *error = QStringLiteral("Workflow 外部输入必须与已登记 Model Package lineage 完全一致。");
                }
                return false;
            }
        } else if (!artifactRoles.contains(input->role)
            || artifactRoles.value(input->role) != input->sourceArtifactKind
            || input->datasetId.isValid() || input->datasetSnapshotId.isValid()
            || input->datasetVersionId.isValid() || input->modelPackageId.isValid()
            || !input->manifestSha256.isEmpty() || !input->rootHash.isEmpty()) {
            if (error) *error = QStringLiteral("Workflow 外部 Artifact 输入角色或 lineage 字段无效。");
            return false;
        }
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    for (const WorkflowStepSnapshot& step : steps) {
        if (step.inputArtifactId.isValid()) {
            ArtifactSnapshot stepInput;
            if (!artifact(step.inputArtifactId, &stepInput, error)
                || (stepInput.taskId != workflow.taskId
                    && (!input || step.inputArtifactId != input->sourceArtifactId))) {
                if (error && error->isEmpty()) {
                    *error = QStringLiteral("工作流初始步骤输入必须属于根任务或已显式绑定的外部输入。");
                }
                db_.rollback();
                return false;
            }
        }
    }
    const QDateTime createdAt = workflow.createdAt.isValid() ? workflow.createdAt.toUTC() : QDateTime::currentDateTimeUtc();
    QSqlQuery runQuery(db_);
    runQuery.prepare(QStringLiteral("insert into workflow_runs(id, task_id, template_id, terminal_policy, created_at) values(:id, :task_id, :template_id, :terminal_policy, :created_at)"));
    runQuery.bindValue(QStringLiteral(":id"), workflow.id.toString());
    runQuery.bindValue(QStringLiteral(":task_id"), workflow.taskId.toString());
    runQuery.bindValue(QStringLiteral(":template_id"), workflow.templateId.trimmed());
    runQuery.bindValue(QStringLiteral(":terminal_policy"), terminalPolicyText(workflow.terminalPolicy));
    runQuery.bindValue(QStringLiteral(":created_at"), utcText(createdAt));
    if (!runQuery.exec()) {
        if (error) *error = sqlError(runQuery);
        db_.rollback();
        return false;
    }
    if (input) {
        QSqlQuery inputQuery(db_);
        inputQuery.prepare(QStringLiteral(
            "insert into workflow_input_bindings(workflow_run_id, role, source_artifact_id, source_task_id, "
            "source_artifact_kind, dataset_id, dataset_snapshot_id, dataset_version_id, model_package_id, "
            "manifest_sha256, root_hash, bound_at) values(:workflow_run_id, :role, :source_artifact_id, "
            ":source_task_id, :source_artifact_kind, :dataset_id, :dataset_snapshot_id, :dataset_version_id, "
            ":model_package_id, :manifest_sha256, :root_hash, :bound_at)"));
        inputQuery.bindValue(QStringLiteral(":workflow_run_id"), workflow.id.toString());
        inputQuery.bindValue(QStringLiteral(":role"), input->role);
        inputQuery.bindValue(QStringLiteral(":source_artifact_id"), input->sourceArtifactId.toString());
        inputQuery.bindValue(QStringLiteral(":source_task_id"), input->sourceTaskId.toString());
        inputQuery.bindValue(QStringLiteral(":source_artifact_kind"), input->sourceArtifactKind);
        inputQuery.bindValue(QStringLiteral(":dataset_id"), input->datasetId.isValid() ? input->datasetId.toString() : QVariant());
        inputQuery.bindValue(QStringLiteral(":dataset_snapshot_id"), input->datasetSnapshotId.isValid() ? input->datasetSnapshotId.toString() : QVariant());
        inputQuery.bindValue(QStringLiteral(":dataset_version_id"), input->datasetVersionId.isValid() ? input->datasetVersionId.toString() : QVariant());
        inputQuery.bindValue(QStringLiteral(":model_package_id"), input->modelPackageId.isValid() ? input->modelPackageId.toString() : QVariant());
        inputQuery.bindValue(QStringLiteral(":manifest_sha256"),
            input->manifestSha256.isEmpty() ? QStringLiteral("") : input->manifestSha256);
        inputQuery.bindValue(QStringLiteral(":root_hash"),
            input->rootHash.isEmpty() ? QStringLiteral("") : input->rootHash);
        inputQuery.bindValue(QStringLiteral(":bound_at"), utcText(
            input->boundAt.isValid() ? input->boundAt : createdAt));
        if (!inputQuery.exec()) {
            if (error) *error = sqlError(inputQuery);
            db_.rollback();
            return false;
        }
    }
    QSqlQuery stepQuery(db_);
    stepQuery.prepare(QStringLiteral("insert into workflow_steps(id, workflow_run_id, ordinal, kind, state, input_artifact_id, output_artifact_id, backend, parameter_summary_json, started_at, finished_at, failure_code, failure_details, failure_suggested_action, failure_occurred_at, retry_count) values(:id, :workflow_run_id, :ordinal, :kind, :state, :input_artifact_id, null, :backend, :parameters, null, null, 'none', '', '', null, 0)"));
    for (const WorkflowStepSnapshot& step : steps) {
        stepQuery.bindValue(QStringLiteral(":id"), step.id.toString());
        stepQuery.bindValue(QStringLiteral(":workflow_run_id"), workflow.id.toString());
        stepQuery.bindValue(QStringLiteral(":ordinal"), step.ordinal);
        stepQuery.bindValue(QStringLiteral(":kind"), step.kind.trimmed());
        stepQuery.bindValue(QStringLiteral(":state"), workflowStepStateToString(WorkflowStepState::Pending));
        stepQuery.bindValue(QStringLiteral(":input_artifact_id"), step.inputArtifactId.isValid() ? step.inputArtifactId.toString() : QVariant());
        stepQuery.bindValue(QStringLiteral(":backend"), step.backend.trimmed());
        stepQuery.bindValue(QStringLiteral(":parameters"), QString::fromUtf8(QJsonDocument(step.parameterSummary).toJson(QJsonDocument::Compact)));
        if (!stepQuery.exec()) {
            if (error) *error = sqlError(stepQuery);
            db_.rollback();
            return false;
        }
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::workflowRun(const WorkflowRunId& workflowRunId, WorkflowRunSnapshot* result, QString* error) const
{
    return WorkflowRepository(database_).readRun(
        workflowRunId, result, error);
}

bool ProjectStore::workflowInput(const WorkflowRunId& workflowRunId,
    const QString& role,
    WorkflowInputBinding* result,
    QString* error) const
{
    return WorkflowRepository(database_).readInput(
        workflowRunId, role, result, error);
}

QVector<WorkflowStepSnapshot> ProjectStore::workflowSteps(const WorkflowRunId& workflowRunId, QString* error) const
{
    return WorkflowRepository(database_).steps(workflowRunId, error);
}

bool ProjectStore::bindWorkflowStepInput(const WorkflowStepId& workflowStepId,
    const ArtifactId& inputArtifactId,
    QString* error)
{
    if (!workflowStepId.isValid() || !inputArtifactId.isValid()) {
        if (error) *error = QStringLiteral("绑定工作流步骤输入需要有效步骤和已提交 Artifact ID。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    if (!artifactExistsInDatabase(db_, inputArtifactId, error)) {
        db_.rollback();
        return false;
    }
    QSqlQuery ownership(db_);
    ownership.prepare(QStringLiteral(
        "select a.task_id, w.task_id, s.workflow_run_id "
        "from workflow_steps s join workflow_runs w on w.id = s.workflow_run_id "
        "join artifacts a on a.id = :artifact_id where s.id = :step_id"));
    ownership.bindValue(QStringLiteral(":artifact_id"), inputArtifactId.toString());
    ownership.bindValue(QStringLiteral(":step_id"), workflowStepId.toString());
    if (!ownership.exec() || !ownership.next()) {
        if (error) *error = ownership.lastError().isValid() ? sqlError(ownership)
            : QStringLiteral("工作流步骤或输入 Artifact 不存在。");
        db_.rollback();
        return false;
    }
    if (ownership.value(0).toString() != ownership.value(1).toString()) {
        QSqlQuery binding(db_);
        binding.prepare(QStringLiteral(
            "select 1 from workflow_input_bindings where workflow_run_id = :workflow_run_id "
            "and source_artifact_id = :artifact_id"));
        binding.bindValue(QStringLiteral(":workflow_run_id"), ownership.value(2).toString());
        binding.bindValue(QStringLiteral(":artifact_id"), inputArtifactId.toString());
        if (!binding.exec() || !binding.next()) {
            if (error) *error = binding.lastError().isValid() ? sqlError(binding)
                : QStringLiteral("禁止向步骤绑定未经 Workflow 授权的跨任务 Artifact。");
            db_.rollback();
            return false;
        }
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("update workflow_steps set input_artifact_id = :input_artifact_id where id = :id and state = 'pending' and input_artifact_id is null"));
    query.bindValue(QStringLiteral(":input_artifact_id"), inputArtifactId.toString());
    query.bindValue(QStringLiteral(":id"), workflowStepId.toString());
    if (!query.exec() || query.numRowsAffected() != 1) {
        if (error) *error = query.lastError().isValid() ? sqlError(query) : QStringLiteral("仅未绑定的 pending 工作流步骤可绑定输入 Artifact。");
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::transitionWorkflowStep(const WorkflowStepId& workflowStepId,
    WorkflowStepState expectedState,
    WorkflowStepState nextState,
    const ArtifactId& outputArtifactId,
    const Failure& failure,
    QString* error)
{
    if (!workflowStepId.isValid() || !isValidWorkflowStepTransition(expectedState, nextState)
        || (expectedState == WorkflowStepState::Failed && nextState == WorkflowStepState::Pending)) {
        if (error) *error = QStringLiteral("非法工作流步骤状态迁移；失败步骤必须通过显式重试恢复。");
        return false;
    }
    if (nextState == WorkflowStepState::Succeeded) {
        if (!outputArtifactId.isValid() || failure.isFailure()) {
            if (error) *error = QStringLiteral("成功工作流步骤必须提供已提交输出 Artifact，且不能带失败信息。");
            return false;
        }
    } else if (outputArtifactId.isValid()) {
        if (error) *error = QStringLiteral("只有成功工作流步骤可以登记输出 Artifact。");
        return false;
    }
    if (nextState == WorkflowStepState::Failed && !failure.isFailure()) {
        if (error) *error = QStringLiteral("失败工作流步骤必须记录失败原因。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    if (outputArtifactId.isValid() && !artifactExistsInDatabase(db_, outputArtifactId, error)) {
        db_.rollback();
        return false;
    }
    if (outputArtifactId.isValid()) {
        QSqlQuery ownership(db_);
        ownership.prepare(QStringLiteral(
            "select a.task_id, w.task_id, s.kind, s.workflow_run_id "
            "from workflow_steps s join workflow_runs w on w.id = s.workflow_run_id "
            "join artifacts a on a.id = :artifact_id where s.id = :step_id"));
        ownership.bindValue(QStringLiteral(":artifact_id"), outputArtifactId.toString());
        ownership.bindValue(QStringLiteral(":step_id"), workflowStepId.toString());
        if (!ownership.exec() || !ownership.next()) {
            if (error) *error = ownership.lastError().isValid() ? sqlError(ownership)
                : QStringLiteral("工作流步骤或输出 Artifact 不存在。");
            db_.rollback();
            return false;
        }
        if (ownership.value(0).toString() != ownership.value(1).toString()) {
            QSqlQuery borrowed(db_);
            borrowed.prepare(QStringLiteral(
                "select role from workflow_input_bindings where workflow_run_id = :workflow_run_id "
                "and source_artifact_id = :artifact_id"));
            borrowed.bindValue(QStringLiteral(":workflow_run_id"), ownership.value(3).toString());
            borrowed.bindValue(QStringLiteral(":artifact_id"), outputArtifactId.toString());
            const bool found = borrowed.exec() && borrowed.next();
            const QString stepKind = ownership.value(2).toString();
            const QString role = found ? borrowed.value(0).toString() : QString();
            const bool permittedBorrow = (stepKind == QStringLiteral("CreateSnapshot")
                    && role == QStringLiteral("dataset_snapshot"))
                || (stepKind == QStringLiteral("ImportOrResolveModel")
                    && role == QStringLiteral("model_package"));
            if (!found || !permittedBorrow) {
                if (error) *error = borrowed.lastError().isValid() ? sqlError(borrowed)
                    : QStringLiteral("步骤输出必须由根任务生成；仅声明的解析步骤可显式借用匹配角色的外部 Artifact。");
                db_.rollback();
                return false;
            }
        }
    }
    const QDateTime now = QDateTime::currentDateTimeUtc();
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("update workflow_steps set state = :next_state, output_artifact_id = case when :has_output = 1 then :output_artifact_id else output_artifact_id end, started_at = case when :next_state = 'running' then :now else started_at end, finished_at = case when :terminal = 1 then :now else finished_at end, failure_code = :failure_code, failure_details = :failure_details, failure_suggested_action = :failure_suggested_action, failure_occurred_at = :failure_occurred_at where id = :id and state = :expected_state"));
    query.bindValue(QStringLiteral(":next_state"), workflowStepStateToString(nextState));
    query.bindValue(QStringLiteral(":has_output"), outputArtifactId.isValid() ? 1 : 0);
    query.bindValue(QStringLiteral(":output_artifact_id"), outputArtifactId.isValid() ? outputArtifactId.toString() : QVariant());
    query.bindValue(QStringLiteral(":now"), utcText(now));
    query.bindValue(QStringLiteral(":terminal"), isTerminalWorkflowStepState(nextState) ? 1 : 0);
    query.bindValue(QStringLiteral(":failure_code"), failureCodeToString(failure.code));
    query.bindValue(QStringLiteral(":failure_details"), requiredText(failure.message));
    query.bindValue(QStringLiteral(":failure_suggested_action"), requiredText(failure.suggestedAction));
    query.bindValue(QStringLiteral(":failure_occurred_at"), failure.isFailure()
            ? utcText(failure.occurredAt.isValid() ? failure.occurredAt : now)
            : QVariant());
    query.bindValue(QStringLiteral(":id"), workflowStepId.toString());
    query.bindValue(QStringLiteral(":expected_state"), workflowStepStateToString(expectedState));
    if (!query.exec() || query.numRowsAffected() != 1) {
        if (error) *error = query.lastError().isValid() ? sqlError(query) : QStringLiteral("工作流步骤不存在或状态已被并发更新。");
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::terminalizeWorkflowStepAndSkipSuccessors(
    const WorkflowStepId& workflowStepId,
    WorkflowStepState expectedState,
    WorkflowStepState terminalState,
    const Failure& failure,
    QString* error)
{
    if (!workflowStepId.isValid()
        || (terminalState != WorkflowStepState::Failed
            && terminalState != WorkflowStepState::Canceled)
        || !failure.isFailure()
        || (expectedState != terminalState
            && !isValidWorkflowStepTransition(expectedState, terminalState))) {
        if (error) *error = QStringLiteral("原子收口 Workflow 步骤需要合法失败/取消终态和完整失败事实。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery current(db_);
    current.prepare(QStringLiteral(
        "select workflow_run_id, ordinal, state from workflow_steps where id = :id"));
    current.bindValue(QStringLiteral(":id"), workflowStepId.toString());
    if (!current.exec() || !current.next()) {
        if (error) *error = current.lastError().isValid() ? sqlError(current)
            : QStringLiteral("待原子收口的 Workflow 步骤不存在。");
        db_.rollback();
        return false;
    }
    const QString workflowRunId = current.value(0).toString();
    const int ordinal = current.value(1).toInt();
    const QString currentState = current.value(2).toString();
    const QString expectedText = workflowStepStateToString(expectedState);
    const QString terminalText = workflowStepStateToString(terminalState);
    if (currentState != expectedText && currentState != terminalText) {
        if (error) *error = QStringLiteral("Workflow 步骤状态已被并发更新，不能原子收口。");
        db_.rollback();
        return false;
    }
    QSqlQuery runningSuccessor(db_);
    runningSuccessor.prepare(QStringLiteral(
        "select 1 from workflow_steps where workflow_run_id = :workflow_run_id "
        "and ordinal > :ordinal and state = 'running' limit 1"));
    runningSuccessor.bindValue(QStringLiteral(":workflow_run_id"), workflowRunId);
    runningSuccessor.bindValue(QStringLiteral(":ordinal"), ordinal);
    if (!runningSuccessor.exec() || runningSuccessor.next()) {
        if (error) *error = runningSuccessor.lastError().isValid() ? sqlError(runningSuccessor)
            : QStringLiteral("Workflow 存在并发运行的后继步骤，拒绝隐藏为 skipped。");
        db_.rollback();
        return false;
    }
    const QDateTime now = QDateTime::currentDateTimeUtc();
    if (currentState == expectedText && expectedState != terminalState) {
        QSqlQuery updateCurrent(db_);
        updateCurrent.prepare(QStringLiteral(
            "update workflow_steps set state = :terminal_state, finished_at = :now, "
            "failure_code = :failure_code, failure_details = :failure_details, "
            "failure_suggested_action = :failure_suggested_action, failure_occurred_at = :failure_occurred_at "
            "where id = :id and state = :expected_state"));
        updateCurrent.bindValue(QStringLiteral(":terminal_state"), terminalText);
        updateCurrent.bindValue(QStringLiteral(":now"), utcText(now));
        updateCurrent.bindValue(QStringLiteral(":failure_code"), failureCodeToString(failure.code));
        updateCurrent.bindValue(QStringLiteral(":failure_details"), requiredText(failure.message));
        updateCurrent.bindValue(QStringLiteral(":failure_suggested_action"), requiredText(failure.suggestedAction));
        updateCurrent.bindValue(QStringLiteral(":failure_occurred_at"), utcText(
            failure.occurredAt.isValid() ? failure.occurredAt : now));
        updateCurrent.bindValue(QStringLiteral(":id"), workflowStepId.toString());
        updateCurrent.bindValue(QStringLiteral(":expected_state"), expectedText);
        if (!updateCurrent.exec() || updateCurrent.numRowsAffected() != 1) {
            if (error) *error = updateCurrent.lastError().isValid() ? sqlError(updateCurrent)
                : QStringLiteral("Workflow 步骤原子终态更新失败。");
            db_.rollback();
            return false;
        }
    }
    QSqlQuery skip(db_);
    skip.prepare(QStringLiteral(
        "update workflow_steps set state = 'skipped', finished_at = :now "
        "where workflow_run_id = :workflow_run_id and ordinal > :ordinal and state = 'pending'"));
    skip.bindValue(QStringLiteral(":now"), utcText(now));
    skip.bindValue(QStringLiteral(":workflow_run_id"), workflowRunId);
    skip.bindValue(QStringLiteral(":ordinal"), ordinal);
    if (!skip.exec()) {
        if (error) *error = sqlError(skip);
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::retryWorkflowStep(const WorkflowStepId& workflowStepId, QString* error)
{
    if (!workflowStepId.isValid()) {
        if (error) *error = QStringLiteral("重试工作流步骤需要有效 ID。");
        return false;
    }
    QSqlQuery stepQuery(db_);
    stepQuery.prepare(QStringLiteral("select workflow_run_id, ordinal, state from workflow_steps where id = :id"));
    stepQuery.bindValue(QStringLiteral(":id"), workflowStepId.toString());
    if (!stepQuery.exec() || !stepQuery.next()) {
        if (error) *error = stepQuery.lastError().isValid() ? sqlError(stepQuery)
            : QStringLiteral("工作流步骤不存在。");
        return false;
    }
    if (stepQuery.value(2).toString() != QStringLiteral("failed")) {
        if (error) *error = QStringLiteral("只有失败的工作流步骤可以重试。");
        return false;
    }
    const QString workflowRunId = stepQuery.value(0).toString();
    const int ordinal = stepQuery.value(1).toInt();
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral(
        "update workflow_steps set state = 'pending', started_at = null, finished_at = null, "
        "output_artifact_id = null, failure_code = 'none', failure_details = '', "
        "failure_suggested_action = '', failure_occurred_at = null, retry_count = retry_count + 1 "
        "where id = :id and state = 'failed'"));
    query.bindValue(QStringLiteral(":id"), workflowStepId.toString());
    if (!query.exec() || query.numRowsAffected() != 1) {
        if (error) *error = query.lastError().isValid() ? sqlError(query) : QStringLiteral("工作流步骤重试状态已被并发更新。");
        db_.rollback();
        return false;
    }
    QSqlQuery resetSuccessors(db_);
    resetSuccessors.prepare(QStringLiteral(
        "update workflow_steps set state = 'pending', input_artifact_id = null, output_artifact_id = null, "
        "started_at = null, finished_at = null, failure_code = 'none', failure_details = '', "
        "failure_suggested_action = '', failure_occurred_at = null "
        "where workflow_run_id = :workflow_run_id and ordinal > :ordinal and state = 'skipped'"));
    resetSuccessors.bindValue(QStringLiteral(":workflow_run_id"), workflowRunId);
    resetSuccessors.bindValue(QStringLiteral(":ordinal"), ordinal);
    if (!resetSuccessors.exec()) {
        if (error) *error = sqlError(resetSuccessors);
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::sealWorkflowTerminalization(const WorkflowRunId& workflowRunId,
    TaskState terminalState,
    const Failure& failure,
    const QDateTime& terminalAt,
    QString* error)
{
    if (!workflowRunId.isValid() || !terminalAt.isValid()
        || !completeTerminalFailure(terminalState, failure)) {
        if (error) *error = QStringLiteral("封存工作流终态需要有效终态、固定时间和完整失败字段。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery workflowQuery(db_);
    workflowQuery.prepare(QStringLiteral("select task_id, terminal_policy from workflow_runs where id = :id"));
    workflowQuery.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!workflowQuery.exec() || !workflowQuery.next()) {
        if (error) *error = workflowQuery.lastError().isValid() ? sqlError(workflowQuery)
            : QStringLiteral("待封存工作流不存在。");
        db_.rollback();
        return false;
    }
    TaskId taskId;
    WorkflowTerminalPolicy policy;
    if (!TaskId::parse(workflowQuery.value(0).toString(), &taskId, error)
        || !parseTerminalPolicy(workflowQuery.value(1).toString(), &policy)) {
        if (error && error->isEmpty()) *error = QStringLiteral("工作流终态策略无效。");
        db_.rollback();
        return false;
    }
    if (policy != WorkflowTerminalPolicy::EvidenceRequired) {
        if (error) *error = QStringLiteral("只有 evidence_required 工作流可以封存终态。");
        db_.rollback();
        return false;
    }
    QSqlQuery unfinishedQuery(db_);
    unfinishedQuery.prepare(QStringLiteral("select count(*) from workflow_steps where workflow_run_id = :id and state in ('pending','running')"));
    unfinishedQuery.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!unfinishedQuery.exec() || !unfinishedQuery.next() || unfinishedQuery.value(0).toInt() != 0) {
        if (error) *error = unfinishedQuery.lastError().isValid() ? sqlError(unfinishedQuery)
            : QStringLiteral("工作流步骤尚未全部进入终态，不能封存根任务终态。");
        db_.rollback();
        return false;
    }

    const QDateTime normalizedTerminalAt = terminalAt.toUTC();
    const QDateTime sealedAt = QDateTime::currentDateTimeUtc();
    QSqlQuery insert(db_);
    insert.prepare(QStringLiteral("insert or ignore into workflow_terminalizations(workflow_run_id, task_id, state, terminal_state, failure_code, failure_details, failure_suggested_action, failure_occurred_at, terminal_at, evidence_artifact_id, evidence_attempt_count, last_evidence_failure_code, last_evidence_failure_details, last_evidence_failure_suggested_action, last_evidence_failure_occurred_at, sealed_at, evidence_attached_at, closed_at) values(:workflow_run_id, :task_id, 'sealed', :terminal_state, :failure_code, :failure_details, :failure_suggested_action, :failure_occurred_at, :terminal_at, null, 0, 'none', '', '', null, :sealed_at, null, null)"));
    insert.bindValue(QStringLiteral(":workflow_run_id"), workflowRunId.toString());
    insert.bindValue(QStringLiteral(":task_id"), taskId.toString());
    insert.bindValue(QStringLiteral(":terminal_state"), taskStateToString(terminalState));
    insert.bindValue(QStringLiteral(":failure_code"), failureCodeToString(failure.code));
    insert.bindValue(QStringLiteral(":failure_details"), requiredText(failure.message));
    insert.bindValue(QStringLiteral(":failure_suggested_action"), requiredText(failure.suggestedAction));
    insert.bindValue(QStringLiteral(":failure_occurred_at"), failure.isFailure() ? utcText(failure.occurredAt) : QVariant());
    insert.bindValue(QStringLiteral(":terminal_at"), utcText(normalizedTerminalAt));
    insert.bindValue(QStringLiteral(":sealed_at"), utcText(sealedAt));
    if (!insert.exec()) {
        if (error) *error = sqlError(insert);
        db_.rollback();
        return false;
    }
    if (insert.numRowsAffected() == 0) {
        WorkflowTerminalizationSnapshot existing;
        QSqlQuery existingQuery(db_);
        existingQuery.prepare(QStringLiteral("select workflow_run_id, task_id, state, terminal_state, failure_code, failure_details, failure_suggested_action, coalesce(failure_occurred_at,''), terminal_at, coalesce(evidence_artifact_id,''), evidence_attempt_count, last_evidence_failure_code, last_evidence_failure_details, last_evidence_failure_suggested_action, coalesce(last_evidence_failure_occurred_at,''), sealed_at, coalesce(evidence_attached_at,''), coalesce(closed_at,'') from workflow_terminalizations where workflow_run_id = :id"));
        existingQuery.bindValue(QStringLiteral(":id"), workflowRunId.toString());
        if (!existingQuery.exec() || !existingQuery.next()
            || !parseTerminalization(existingQuery, &existing, error)
            || existing.taskId != taskId || existing.terminalState != terminalState
            || !sameFailure(existing.failure, failure)
            || existing.terminalAt != normalizedTerminalAt) {
            if (error && error->isEmpty()) *error = QStringLiteral("工作流终态已按不同事实封存，拒绝覆盖。");
            db_.rollback();
            return false;
        }
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::workflowTerminalization(const WorkflowRunId& workflowRunId,
    WorkflowTerminalizationSnapshot* result,
    QString* error) const
{
    return WorkflowTerminalizationStore(database_).read(
        workflowRunId, result, error);
}

bool ProjectStore::workflowTerminalizationExists(
    const WorkflowRunId& workflowRunId, bool* exists, QString* error) const
{
    return WorkflowTerminalizationStore(database_).exists(
        workflowRunId, exists, error);
}

QVector<WorkflowTerminalizationSnapshot> ProjectStore::pendingWorkflowTerminalizations(
    int limit,
    QString* error) const
{
    return WorkflowTerminalizationStore(database_).pending(limit, error);
}

QVector<WorkflowRunSnapshot> ProjectStore::pendingEvidenceRequiredWorkflows(
    int limit,
    QString* error) const
{
    return WorkflowTerminalizationStore(database_)
        .pendingEvidenceRequired(limit, error);
}

bool ProjectStore::attachWorkflowTerminalizationEvidence(const WorkflowRunId& workflowRunId,
    const ArtifactId& evidenceArtifactId,
    QString* error)
{
    if (!workflowRunId.isValid() || !evidenceArtifactId.isValid()) {
        if (error) *error = QStringLiteral("关联 Evidence 需要有效工作流和 Artifact ID。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery current(db_);
    current.prepare(QStringLiteral("select task_id, state, coalesce(evidence_artifact_id,'') from workflow_terminalizations where workflow_run_id = :id"));
    current.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!current.exec() || !current.next()) {
        if (error) *error = current.lastError().isValid() ? sqlError(current)
            : QStringLiteral("工作流终态尚未封存。");
        db_.rollback();
        return false;
    }
    const QString taskId = current.value(0).toString();
    const QString state = current.value(1).toString();
    const QString existingArtifact = current.value(2).toString();
    QSqlQuery artifactQuery(db_);
    artifactQuery.prepare(QStringLiteral("select task_id, kind from artifacts where id = :id"));
    artifactQuery.bindValue(QStringLiteral(":id"), evidenceArtifactId.toString());
    if (!artifactQuery.exec() || !artifactQuery.next()
        || artifactQuery.value(0).toString() != taskId
        || artifactQuery.value(1).toString() != QStringLiteral("evidence_bundle")) {
        if (error) *error = artifactQuery.lastError().isValid() ? sqlError(artifactQuery)
            : QStringLiteral("Evidence 必须是同一根任务已提交的 evidence_bundle Artifact。");
        db_.rollback();
        return false;
    }
    if (state != QStringLiteral("sealed")) {
        if (existingArtifact == evidenceArtifactId.toString()
            && (state == QStringLiteral("evidence_attached") || state == QStringLiteral("closed"))) {
            return db_.commit();
        }
        if (error) *error = QStringLiteral("工作流终态已关联不同 Evidence 或已经关闭。");
        db_.rollback();
        return false;
    }
    QSqlQuery update(db_);
    update.prepare(QStringLiteral("update workflow_terminalizations set state = 'evidence_attached', evidence_artifact_id = :artifact_id, evidence_attached_at = :now where workflow_run_id = :id and state = 'sealed' and evidence_artifact_id is null"));
    update.bindValue(QStringLiteral(":artifact_id"), evidenceArtifactId.toString());
    update.bindValue(QStringLiteral(":now"), utcText(QDateTime::currentDateTimeUtc()));
    update.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!update.exec() || update.numRowsAffected() != 1) {
        if (error) *error = update.lastError().isValid() ? sqlError(update)
            : QStringLiteral("Evidence 关联状态已被并发更新。");
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::recordWorkflowTerminalizationEvidenceFailure(const WorkflowRunId& workflowRunId,
    const Failure& failure,
    QString* error)
{
    if (!workflowRunId.isValid() || !completeEvidenceFailure(failure)) {
        if (error) *error = QStringLiteral("记录 Evidence 失败尝试需要完整失败字段。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery current(db_);
    current.prepare(QStringLiteral("select state, last_evidence_failure_code, last_evidence_failure_details, last_evidence_failure_suggested_action, coalesce(last_evidence_failure_occurred_at,'') from workflow_terminalizations where workflow_run_id = :id"));
    current.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!current.exec() || !current.next()) {
        if (error) *error = current.lastError().isValid() ? sqlError(current)
            : QStringLiteral("工作流终态尚未封存。");
        db_.rollback();
        return false;
    }
    Failure previous;
    failureCodeFromString(current.value(1).toString(), &previous.code);
    previous.message = current.value(2).toString();
    previous.suggestedAction = current.value(3).toString();
    previous.occurredAt = parseUtc(current.value(4).toString());
    if (current.value(0).toString() != QStringLiteral("sealed")) {
        if (error) *error = QStringLiteral("Evidence 已关联或工作流终态已关闭，不能记录失败尝试。");
        db_.rollback();
        return false;
    }
    if (sameFailure(previous, failure)) return db_.commit();
    QSqlQuery update(db_);
    update.prepare(QStringLiteral("update workflow_terminalizations set evidence_attempt_count = evidence_attempt_count + 1, last_evidence_failure_code = :code, last_evidence_failure_details = :details, last_evidence_failure_suggested_action = :action, last_evidence_failure_occurred_at = :occurred_at where workflow_run_id = :id and state = 'sealed' and evidence_artifact_id is null"));
    update.bindValue(QStringLiteral(":code"), failureCodeToString(failure.code));
    update.bindValue(QStringLiteral(":details"), failure.message);
    update.bindValue(QStringLiteral(":action"), failure.suggestedAction);
    update.bindValue(QStringLiteral(":occurred_at"), utcText(failure.occurredAt));
    update.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!update.exec() || update.numRowsAffected() != 1) {
        if (error) *error = update.lastError().isValid() ? sqlError(update)
            : QStringLiteral("Evidence 失败尝试状态已被并发更新。");
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::closeWorkflowTerminalization(
    const WorkflowRunId& workflowRunId, QString* error)
{
    if (!workflowRunId.isValid()) {
        if (error) *error = QStringLiteral("关闭工作流终态需要有效工作流 ID。");
        return false;
    }
    if (!db_.transaction()) {
        if (error) *error = db_.lastError().text();
        return false;
    }
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select t.workflow_run_id, t.task_id, t.state, t.terminal_state, t.failure_code, t.failure_details, t.failure_suggested_action, coalesce(t.failure_occurred_at,''), t.terminal_at, coalesce(t.evidence_artifact_id,''), t.evidence_attempt_count, t.last_evidence_failure_code, t.last_evidence_failure_details, t.last_evidence_failure_suggested_action, coalesce(t.last_evidence_failure_occurred_at,''), t.sealed_at, coalesce(t.evidence_attached_at,''), coalesce(t.closed_at,''), task.state, task.request_id, a.task_id, a.kind from workflow_terminalizations t join tasks task on task.id = t.task_id left join artifacts a on a.id = t.evidence_artifact_id where t.workflow_run_id = :id"));
    query.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!query.exec() || !query.next()) {
        if (error) *error = query.lastError().isValid() ? sqlError(query)
            : QStringLiteral("工作流终态封存不存在。");
        db_.rollback();
        return false;
    }
    WorkflowTerminalizationSnapshot terminalization;
    if (!parseTerminalization(query, &terminalization, error)) {
        db_.rollback();
        return false;
    }
    TaskState currentTaskState;
    RequestId requestId;
    if (!taskStateFromString(query.value(18).toString(), &currentTaskState)
        || !RequestId::parse(query.value(19).toString(), &requestId, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("根任务状态记录无效。");
        db_.rollback();
        return false;
    }
    if (terminalization.state == WorkflowTerminalizationState::Closed) {
        if (currentTaskState == terminalization.terminalState) return db_.commit();
        if (error) *error = QStringLiteral("已关闭终态与根任务状态不一致。");
        db_.rollback();
        return false;
    }
    if (terminalization.state != WorkflowTerminalizationState::EvidenceAttached
        || query.value(20).toString() != terminalization.taskId.toString()
        || query.value(21).toString() != QStringLiteral("evidence_bundle")) {
        if (error) *error = QStringLiteral("关闭工作流终态前必须关联同任务 evidence_bundle Artifact。");
        db_.rollback();
        return false;
    }
    if (!isValidTaskStateTransition(currentTaskState, terminalization.terminalState)) {
        if (error) *error = QStringLiteral("根任务不存在、状态已被并发更新或终态迁移非法。");
        db_.rollback();
        return false;
    }
    QSqlQuery updateTask(db_);
    updateTask.prepare(QStringLiteral("update tasks set state = :next_state, failure_code = :failure_code, failure_details = :failure_details, failure_suggested_action = :failure_suggested_action, failure_occurred_at = :failure_occurred_at, updated_at = :terminal_at where id = :id and state = :expected_state"));
    updateTask.bindValue(QStringLiteral(":next_state"), taskStateToString(terminalization.terminalState));
    updateTask.bindValue(QStringLiteral(":failure_code"), failureCodeToString(terminalization.failure.code));
    updateTask.bindValue(QStringLiteral(":failure_details"), requiredText(terminalization.failure.message));
    updateTask.bindValue(QStringLiteral(":failure_suggested_action"), requiredText(terminalization.failure.suggestedAction));
    updateTask.bindValue(QStringLiteral(":failure_occurred_at"), terminalization.failure.isFailure()
        ? utcText(terminalization.failure.occurredAt) : QVariant());
    updateTask.bindValue(QStringLiteral(":terminal_at"), utcText(terminalization.terminalAt));
    updateTask.bindValue(QStringLiteral(":id"), terminalization.taskId.toString());
    updateTask.bindValue(QStringLiteral(":expected_state"), taskStateToString(currentTaskState));
    if (!updateTask.exec() || updateTask.numRowsAffected() != 1
        || !appendStateEvent(terminalization.taskId, requestId, terminalization.terminalState,
            terminalization.failure, terminalization.terminalAt, error)) {
        if (error && error->isEmpty()) *error = updateTask.lastError().isValid()
            ? sqlError(updateTask) : QStringLiteral("根任务终态已被并发更新。");
        db_.rollback();
        return false;
    }
    QSqlQuery close(db_);
    close.prepare(QStringLiteral("update workflow_terminalizations set state = 'closed', closed_at = :closed_at where workflow_run_id = :id and state = 'evidence_attached'"));
    close.bindValue(QStringLiteral(":closed_at"), utcText(QDateTime::currentDateTimeUtc()));
    close.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!close.exec() || close.numRowsAffected() != 1) {
        if (error) *error = close.lastError().isValid() ? sqlError(close)
            : QStringLiteral("工作流终态关闭状态已被并发更新。");
        db_.rollback();
        return false;
    }
    if (db_.commit()) return true;
    if (error) *error = db_.lastError().text();
    return false;
}

bool ProjectStore::artifact(const ArtifactId& artifactId, ArtifactSnapshot* result, QString* error) const
{
    return ArtifactCatalogRepository(database_).read(artifactId, result, error);
}

Page<ArtifactFileSnapshot> ProjectStore::artifactFiles(
    const ArtifactId& artifactId, const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<ArtifactFileSnapshot> result =
        ArtifactCatalogRepository(database_).files(
            artifactId, request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

Page<ArtifactSnapshot> ProjectStore::artifactsForTask(
    const TaskId& taskId, const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<ArtifactSnapshot> result =
        ArtifactCatalogRepository(database_).forTask(
            taskId, request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

Page<DeliveryEvidenceCandidate> ProjectStore::deliveryEvidenceCandidates(
    const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<DeliveryEvidenceCandidate> result =
        ProjectReadRepository(database_).deliveryEvidence(
            request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

Page<MetricSnapshot> ProjectStore::metricsForTask(
    const TaskId& taskId, const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<MetricSnapshot> result = TaskEventRepository(database_).metrics(
        taskId, request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

bool ProjectStore::taskExists(const TaskId& taskId, bool* exists, QString* error) const
{
    return TaskEventRepository(database_).exists(taskId, exists, error);
}

bool ProjectStore::artifactExists(const ArtifactId& artifactId, bool* exists, QString* error) const
{
    return ArtifactCatalogRepository(database_).exists(
        artifactId, exists, error);
}

bool ProjectStore::task(const TaskId& taskId, TaskSnapshot* result, QString* error) const
{
    return TaskEventRepository(database_).read(taskId, result, error);
}

Page<TaskSnapshot> ProjectStore::tasks(const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<TaskSnapshot> result =
        TaskEventRepository(database_).page(request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

Page<WorkflowRunSnapshot> ProjectStore::workflowRunsForTask(
    const TaskId& taskId, const PageRequest& request, QString* error) const
{
    QString repositoryError;
    Page<WorkflowRunSnapshot> result =
        WorkflowRepository(database_).runsForTask(
            taskId, request, &repositoryError);
    lastErrorCode_ = pageErrorCode(repositoryError);
    if (error) *error = repositoryError;
    return result;
}

int ProjectStore::eventCount(const TaskId& taskId, QString* error) const
{
    return TaskEventRepository(database_).eventCount(taskId, error);
}

int ProjectStore::metricCount(const TaskId& taskId, QString* error) const
{
    return TaskEventRepository(database_).metricCount(taskId, error);
}

int ProjectStore::artifactCount(const TaskId& taskId, QString* error) const
{
    return ArtifactCatalogRepository(database_).countForTask(taskId, error);
}

int ProjectStore::artifactFileCount(const ArtifactId& artifactId, QString* error) const
{
    return ArtifactCatalogRepository(database_).fileCount(artifactId, error);
}

} // namespace aitrain
