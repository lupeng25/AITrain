#pragma once

#include "aitrain/domain/DomainTypes.h"
#include "aitrain/model/ModelManifest.h"

#include <QDateTime>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QVector>

namespace aitrain {

struct ProtocolEnvelope;

struct TaskSnapshot final {
    TaskId id;
    RequestId requestId;
    TaskState state = TaskState::Created;
    QString capabilityId;
    QString taskType;
    QDateTime createdAt;
    QDateTime updatedAt;
    Failure failure;
};

struct ArtifactFileSnapshot final {
    QString relativePath;
    QString sha256;
    qint64 byteCount = 0;
};

// Artifact 和其文件清单必须一起读取，供 Evidence 等只读消费者使用。
// 该快照不包含可写磁盘路径，避免上层绕过 Artifact Store 的完整性边界。
struct ArtifactSnapshot final {
    ArtifactId id;
    TaskId taskId;
    QString kind;
    QDateTime createdAt;
    QVector<ArtifactFileSnapshot> files;
};

// 交付证据查询的一次性只读候选。任务、Artifact 和文件清单由同一条
// SQL 查询加载，避免查询服务先取任务再逐任务加载 Artifact。
struct DeliveryEvidenceCandidate final {
    TaskSnapshot task;
    ArtifactSnapshot artifact;
};

struct MetricSnapshot final {
    QString name;
    double value = 0.0;
    QDateTime occurredAt;
};

// 协议事件的副作用必须与事件记录、任务终态在同一个 SQLite 事务中完成。
// effect 中未设置的字段表示该事件没有对应副作用。
struct ProtocolEventEffect final {
    QString metricName;
    double metricValue = 0.0;
    ArtifactId artifactId;
    QString artifactKind;
    TaskState terminalState = TaskState::Running;
    Failure terminalFailure;

    bool hasMetric() const { return !metricName.trimmed().isEmpty(); }
    bool hasArtifact() const { return artifactId.isValid() || !artifactKind.trimmed().isEmpty(); }
    bool hasTerminal() const { return terminalState != TaskState::Running; }
};

struct ProtocolEventApplyResult final {
    bool idempotent = false;
    TaskState effectiveTaskState = TaskState::Created;
};

// 数据快照既是训练输入，也是可审计的 Artifact 引用；不保存可直接执行的裸路径。
struct DatasetSnapshotRecord final {
    DatasetId datasetId;
    DatasetVersionId datasetVersionId;
    SnapshotId id;
    TaskId taskId;
    ArtifactId artifactId;
    QString rootPath;
    QString datasetFormat;
    QString driverId;
    QString driverVersion;
    QString rootHash;
    QString manifestSha256;
    qsizetype fileCount = 0;
    qint64 totalBytes = 0;
    QDateTime createdAt;
};

// GUI 数据集目录只消费持久化身份与摘要，不暴露原始根目录或 Artifact 物理路径。
struct DatasetCatalogItem final {
    DatasetId datasetId;
    QString datasetFormat;
    qint64 versionCount = 0;
    qint64 snapshotCount = 0;
    DatasetVersionId latestVersionId;
    SnapshotId latestSnapshotId;
    ArtifactId latestArtifactId;
    QString latestRootHash;
    qsizetype latestFileCount = 0;
    QDateTime latestCreatedAt;
};

struct ModelPackageSnapshot final {
    ModelManifest manifest;
    ArtifactId sourceArtifactId;
    QDateTime createdAt;
};

// Dashboard/Presenter 可安全消费的项目级聚合事实。这里仅包含  SQLite 中
// 已持久化的计数，不暴露数据集根目录、Artifact 磁盘目录或 Worker payload。
struct TaskStateCounts final {
    qint64 created = 0;
    qint64 queued = 0;
    qint64 starting = 0;
    qint64 running = 0;
    qint64 cancelRequested = 0;
    qint64 succeeded = 0;
    qint64 failed = 0;
    qint64 canceled = 0;

    qint64 total() const
    {
        return created + queued + starting + running + cancelRequested
            + succeeded + failed + canceled;
    }
};

struct ProjectSummarySnapshot final {
    TaskStateCounts tasks;
    qint64 committedArtifactCount = 0;
    qint64 datasetCount = 0;
    qint64 datasetVersionCount = 0;
    qint64 datasetSnapshotCount = 0;
    qint64 modelPackageCount = 0;
    qint64 verifiedModelPackageCount = 0;
    qint64 workflowRunCount = 0;
    qint64 evidenceRequiredWorkflowCount = 0;
    qint64 evidenceAvailableWorkflowCount = 0;
    qint64 evidencePendingWorkflowCount = 0;
};

enum class WorkflowTerminalPolicy {
    Immediate,
    EvidenceRequired
};

enum class WorkflowTerminalizationState {
    Sealed,
    EvidenceAttached,
    Closed
};

struct WorkflowRunSnapshot final {
    WorkflowRunId id;
    TaskId taskId;
    QString templateId;
    WorkflowTerminalPolicy terminalPolicy = WorkflowTerminalPolicy::Immediate;
    QDateTime createdAt;
};

struct WorkflowStepSnapshot final {
    WorkflowStepId id;
    WorkflowRunId workflowRunId;
    int ordinal = -1;
    QString kind;
    WorkflowStepState state = WorkflowStepState::Pending;
    ArtifactId inputArtifactId;
    ArtifactId outputArtifactId;
    QString backend;
    QJsonObject parameterSummary;
    QDateTime startedAt;
    QDateTime finishedAt;
    Failure failure;
    int retryCount = 0;
};

// 跨任务只读输入必须在 Workflow 创建时原子绑定，不能借助 step output 或
// parameter_summary 暗中注入。首个角色固定为 dataset_snapshot。
struct WorkflowInputBinding final {
    WorkflowRunId workflowRunId;
    QString role;
    ArtifactId sourceArtifactId;
    TaskId sourceTaskId;
    QString sourceArtifactKind;
    DatasetId datasetId;
    SnapshotId datasetSnapshotId;
    DatasetVersionId datasetVersionId;
    ModelPackageId modelPackageId;
    QString manifestSha256;
    QString rootHash;
    QDateTime boundAt;
};

// Evidence 门控工作流在根任务真正进入终态前先封存不可变的终态事实。
// terminalAt 和 failure 一经 seal 即不可修改；重启恢复只能补交 Evidence 或关闭。
struct WorkflowTerminalizationSnapshot final {
    WorkflowRunId workflowRunId;
    TaskId taskId;
    WorkflowTerminalizationState state = WorkflowTerminalizationState::Sealed;
    TaskState terminalState = TaskState::Failed;
    Failure failure;
    QDateTime terminalAt;
    ArtifactId evidenceArtifactId;
    int evidenceAttemptCount = 0;
    Failure lastEvidenceFailure;
    QDateTime sealedAt;
    QDateTime evidenceAttachedAt;
    QDateTime closedAt;
};

class ProjectStore final {
public:
    ProjectStore();
    ~ProjectStore();

    // 当前项目尚未上线，不提供旧数据库迁移；新工作区必须使用这一版完整 schema。
    static int schemaVersion();

    ProjectStore(const ProjectStore&) = delete;
    ProjectStore& operator=(const ProjectStore&) = delete;

    bool open(const QString& databasePath, QString* error = nullptr);
    void close();
    bool isOpen() const;
    void setArtifactStoreRoot(QString artifactStoreRoot);

    bool createTask(const TaskSnapshot& task, QString* error = nullptr);
    bool transitionTask(const TaskId& taskId,
        TaskState expectedState,
        TaskState nextState,
        const Failure& failure = {},
        QString* error = nullptr);
    bool markInterruptedTasksFailed(QString* error = nullptr);
    bool recordProtocolEvent(const TaskId& taskId,
        const RequestId& requestId,
        const MessageId& messageId,
        quint64 sequence,
        const QString& kind,
        const QJsonObject& payload,
        const QDateTime& occurredAt,
        QString* error = nullptr);
    // 原子应用 Adapter 事件：事件、Metric/Artifact 副作用、任务终态和状态审计
    // 要么全部落库，要么全部回滚。重复的同一事件返回 idempotent=true。
    bool applyProtocolEvent(const ProtocolEnvelope& envelope,
        const ProtocolEventEffect& effect,
        ProtocolEventApplyResult* result = nullptr,
        QString* error = nullptr);
    bool recordMetric(const TaskId& taskId, const QString& name, double value, const QDateTime& occurredAt, QString* error = nullptr);
    bool recordArtifact(const ArtifactId& artifactId, const TaskId& taskId, const QString& kind, const QDateTime& createdAt, QString* error = nullptr);
    bool recordArtifactWithFiles(const ArtifactId& artifactId,
        const TaskId& taskId,
        const QString& kind,
        const QVector<ArtifactFileSnapshot>& files,
        const QDateTime& createdAt,
        QString* error = nullptr);
    bool recordEvidenceArtifactWithFilesAndAttachTerminalization(
        const ArtifactId& artifactId,
        const TaskId& taskId,
        const WorkflowRunId& workflowRunId,
        const QVector<ArtifactFileSnapshot>& files,
        const QDateTime& createdAt,
        QString* error = nullptr);
    bool registerDatasetSnapshot(DatasetSnapshotRecord* snapshot, QString* error = nullptr);
    bool datasetSnapshot(const SnapshotId& snapshotId, DatasetSnapshotRecord* result, QString* error = nullptr) const;
    bool datasetSnapshotForArtifact(const ArtifactId& artifactId,
        DatasetSnapshotRecord* result,
        QString* error = nullptr) const;
    QVector<DatasetCatalogItem> datasets(int limit, QString* error = nullptr) const;
    bool removeUnreferencedArtifact(const ArtifactId& artifactId, QString* error = nullptr);
    bool registerModelPackage(const ModelPackageSnapshot& modelPackage, QString* error = nullptr);
    bool modelPackage(const ModelPackageId& modelPackageId, ModelPackageSnapshot* result, QString* error = nullptr) const;
    QVector<ModelPackageSnapshot> modelPackages(int limit, QString* error = nullptr) const;
    bool projectSummary(ProjectSummarySnapshot* result, QString* error = nullptr) const;
    bool createWorkflowRun(const WorkflowRunSnapshot& workflow,
        const QVector<WorkflowStepSnapshot>& steps,
        QString* error = nullptr);
    bool createWorkflowRunWithInput(const WorkflowRunSnapshot& workflow,
        const QVector<WorkflowStepSnapshot>& steps,
        const WorkflowInputBinding& input,
        QString* error = nullptr);
    bool workflowInput(const WorkflowRunId& workflowRunId,
        const QString& role,
        WorkflowInputBinding* result,
        QString* error = nullptr) const;
    bool workflowRun(const WorkflowRunId& workflowRunId, WorkflowRunSnapshot* result, QString* error = nullptr) const;
    QVector<WorkflowStepSnapshot> workflowSteps(const WorkflowRunId& workflowRunId, QString* error = nullptr) const;
    bool bindWorkflowStepInput(const WorkflowStepId& workflowStepId,
        const ArtifactId& inputArtifactId,
        QString* error = nullptr);
    bool transitionWorkflowStep(const WorkflowStepId& workflowStepId,
        WorkflowStepState expectedState,
        WorkflowStepState nextState,
        const ArtifactId& outputArtifactId = {},
        const Failure& failure = {},
        QString* error = nullptr);
    // 将失败/取消步骤与所有后继 pending 步骤在同一 SQLite 事务中收口。
    // 若当前步骤已经处于相同终态，则执行幂等的半收口修复。
    bool terminalizeWorkflowStepAndSkipSuccessors(const WorkflowStepId& workflowStepId,
        WorkflowStepState expectedState,
        WorkflowStepState terminalState,
        const Failure& failure,
        QString* error = nullptr);
    bool retryWorkflowStep(const WorkflowStepId& workflowStepId, QString* error = nullptr);
    bool sealWorkflowTerminalization(const WorkflowRunId& workflowRunId,
        TaskState terminalState,
        const Failure& failure,
        const QDateTime& terminalAt,
        QString* error = nullptr);
    bool workflowTerminalization(const WorkflowRunId& workflowRunId,
        WorkflowTerminalizationSnapshot* result,
        QString* error = nullptr) const;
    QVector<WorkflowTerminalizationSnapshot> pendingWorkflowTerminalizations(
        int limit,
        QString* error = nullptr) const;
    QVector<WorkflowTerminalizationSnapshot> pendingWorkflowTerminalizations(
        int limit,
        int offset,
        QString* error) const;
    QVector<WorkflowRunSnapshot> pendingEvidenceRequiredWorkflows(
        int limit,
        QString* error = nullptr) const;
    QVector<WorkflowRunSnapshot> pendingEvidenceRequiredWorkflows(
        int limit,
        int offset,
        QString* error) const;
    bool attachWorkflowTerminalizationEvidence(const WorkflowRunId& workflowRunId,
        const ArtifactId& evidenceArtifactId,
        QString* error = nullptr);
    bool recordWorkflowTerminalizationEvidenceFailure(const WorkflowRunId& workflowRunId,
        const Failure& failure,
        QString* error = nullptr);
    bool closeWorkflowTerminalization(const WorkflowRunId& workflowRunId,
        TaskState expectedTaskState,
        QString* error = nullptr);
    bool taskExists(const TaskId& taskId, bool* exists, QString* error = nullptr) const;
    bool artifactExists(const ArtifactId& artifactId, bool* exists, QString* error = nullptr) const;
    bool task(const TaskId& taskId, TaskSnapshot* result, QString* error = nullptr) const;
    QVector<TaskSnapshot> tasks(int limit, QString* error = nullptr) const;
    bool artifact(const ArtifactId& artifactId, ArtifactSnapshot* result, QString* error = nullptr) const;
    QVector<ArtifactSnapshot> artifactsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<DeliveryEvidenceCandidate> deliveryEvidenceCandidates(
        int limit, QString* error = nullptr) const;
    QVector<MetricSnapshot> metricsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowRunSnapshot> workflowRunsForTask(const TaskId& taskId, QString* error = nullptr) const;
    // Adapter 事件可在同一根任务的多个 Workflow 步骤中连续产生；返回最近一条
    // 外部协议事件的序号，排除由宿主单独分配的 task.state_changed 审计序号。
    bool lastProtocolSequence(const TaskId& taskId, quint64* result, QString* error = nullptr) const;
    int eventCount(const TaskId& taskId, QString* error = nullptr) const;
    int metricCount(const TaskId& taskId, QString* error = nullptr) const;
    int artifactCount(const TaskId& taskId, QString* error = nullptr) const;
    int artifactFileCount(const ArtifactId& artifactId, QString* error = nullptr) const;


private:
    bool createWorkflowRunInternal(const WorkflowRunSnapshot& workflow,
        const QVector<WorkflowStepSnapshot>& steps,
        const WorkflowInputBinding* input,
        QString* error);
    bool initialize(QString* error);
    bool appendStateEvent(const TaskId& taskId,
        const RequestId& requestId,
        TaskState state,
        const Failure& failure,
        const QDateTime& occurredAt,
        QString* error);

    QString connectionName_;
    QString artifactStoreRoot_;
    QSqlDatabase db_;
};

} // namespace aitrain
