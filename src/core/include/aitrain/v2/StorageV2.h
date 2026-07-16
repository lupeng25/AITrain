#pragma once

#include "aitrain/v2/DomainTypes.h"
#include "aitrain/v2/ModelManifestV2.h"

#include <QDateTime>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QVector>

namespace aitrain::v2 {

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
struct ArtifactSnapshotV2 final {
    ArtifactId id;
    TaskId taskId;
    QString kind;
    QDateTime createdAt;
    QVector<ArtifactFileSnapshot> files;
};

struct MetricSnapshotV2 final {
    QString name;
    double value = 0.0;
    QDateTime occurredAt;
};

// 数据快照既是训练输入，也是可审计的 Artifact 引用；不保存可直接执行的裸路径。
struct DatasetSnapshotRecordV2 final {
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
struct DatasetCatalogItemV2 final {
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

struct ModelPackageSnapshotV2 final {
    ModelManifestV2 manifest;
    ArtifactId sourceArtifactId;
    QDateTime createdAt;
};

// Dashboard/Presenter 可安全消费的项目级聚合事实。这里仅包含 V2 SQLite 中
// 已持久化的计数，不暴露数据集根目录、Artifact 磁盘目录或 Worker payload。
struct TaskStateCountsV2 final {
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

struct ProjectSummarySnapshotV2 final {
    TaskStateCountsV2 tasks;
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

enum class WorkflowTerminalPolicyV2 {
    Immediate,
    EvidenceRequired
};

enum class WorkflowTerminalizationStateV2 {
    Sealed,
    EvidenceAttached,
    Closed
};

struct WorkflowRunSnapshotV2 final {
    WorkflowRunId id;
    TaskId taskId;
    QString templateId;
    WorkflowTerminalPolicyV2 terminalPolicy = WorkflowTerminalPolicyV2::Immediate;
    QDateTime createdAt;
};

struct WorkflowStepSnapshotV2 final {
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
struct WorkflowInputBindingV2 final {
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
struct WorkflowTerminalizationSnapshotV2 final {
    WorkflowRunId workflowRunId;
    TaskId taskId;
    WorkflowTerminalizationStateV2 state = WorkflowTerminalizationStateV2::Sealed;
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

class StorageV2 final {
public:
    StorageV2();
    ~StorageV2();

    StorageV2(const StorageV2&) = delete;
    StorageV2& operator=(const StorageV2&) = delete;

    bool open(const QString& databasePath, QString* error = nullptr);
    void close();
    bool isOpen() const;

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
    bool registerDatasetSnapshot(DatasetSnapshotRecordV2* snapshot, QString* error = nullptr);
    bool datasetSnapshot(const SnapshotId& snapshotId, DatasetSnapshotRecordV2* result, QString* error = nullptr) const;
    bool datasetSnapshotForArtifact(const ArtifactId& artifactId,
        DatasetSnapshotRecordV2* result,
        QString* error = nullptr) const;
    QVector<DatasetCatalogItemV2> datasets(int limit, QString* error = nullptr) const;
    bool removeUnreferencedArtifact(const ArtifactId& artifactId, QString* error = nullptr);
    bool registerModelPackage(const ModelPackageSnapshotV2& modelPackage, QString* error = nullptr);
    bool modelPackage(const ModelPackageId& modelPackageId, ModelPackageSnapshotV2* result, QString* error = nullptr) const;
    QVector<ModelPackageSnapshotV2> modelPackages(int limit, QString* error = nullptr) const;
    bool projectSummary(ProjectSummarySnapshotV2* result, QString* error = nullptr) const;
    bool createWorkflowRun(const WorkflowRunSnapshotV2& workflow,
        const QVector<WorkflowStepSnapshotV2>& steps,
        QString* error = nullptr);
    bool createWorkflowRunWithInput(const WorkflowRunSnapshotV2& workflow,
        const QVector<WorkflowStepSnapshotV2>& steps,
        const WorkflowInputBindingV2& input,
        QString* error = nullptr);
    bool workflowInput(const WorkflowRunId& workflowRunId,
        const QString& role,
        WorkflowInputBindingV2* result,
        QString* error = nullptr) const;
    bool workflowRun(const WorkflowRunId& workflowRunId, WorkflowRunSnapshotV2* result, QString* error = nullptr) const;
    QVector<WorkflowStepSnapshotV2> workflowSteps(const WorkflowRunId& workflowRunId, QString* error = nullptr) const;
    bool bindWorkflowStepInput(const WorkflowStepId& workflowStepId,
        const ArtifactId& inputArtifactId,
        QString* error = nullptr);
    bool transitionWorkflowStep(const WorkflowStepId& workflowStepId,
        WorkflowStepState expectedState,
        WorkflowStepState nextState,
        const ArtifactId& outputArtifactId = {},
        const Failure& failure = {},
        QString* error = nullptr);
    bool retryWorkflowStep(const WorkflowStepId& workflowStepId, QString* error = nullptr);
    bool sealWorkflowTerminalization(const WorkflowRunId& workflowRunId,
        TaskState terminalState,
        const Failure& failure,
        const QDateTime& terminalAt,
        QString* error = nullptr);
    bool workflowTerminalization(const WorkflowRunId& workflowRunId,
        WorkflowTerminalizationSnapshotV2* result,
        QString* error = nullptr) const;
    QVector<WorkflowTerminalizationSnapshotV2> pendingWorkflowTerminalizations(
        int limit,
        QString* error = nullptr) const;
    QVector<WorkflowRunSnapshotV2> pendingEvidenceRequiredWorkflows(
        int limit,
        QString* error = nullptr) const;
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
    bool artifact(const ArtifactId& artifactId, ArtifactSnapshotV2* result, QString* error = nullptr) const;
    QVector<ArtifactSnapshotV2> artifactsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<MetricSnapshotV2> metricsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowRunSnapshotV2> workflowRunsForTask(const TaskId& taskId, QString* error = nullptr) const;
    // Adapter 事件可在同一根任务的多个 Workflow 步骤中连续产生；返回最近一条
    // 外部协议事件的序号，排除由宿主单独分配的 task.state_changed 审计序号。
    bool lastProtocolSequence(const TaskId& taskId, quint64* result, QString* error = nullptr) const;
    int eventCount(const TaskId& taskId, QString* error = nullptr) const;
    int metricCount(const TaskId& taskId, QString* error = nullptr) const;
    int artifactCount(const TaskId& taskId, QString* error = nullptr) const;
    int artifactFileCount(const ArtifactId& artifactId, QString* error = nullptr) const;

private:
    bool createWorkflowRunInternal(const WorkflowRunSnapshotV2& workflow,
        const QVector<WorkflowStepSnapshotV2>& steps,
        const WorkflowInputBindingV2* input,
        QString* error);
    bool initialize(QString* error);
    bool appendStateEvent(const TaskId& taskId,
        const RequestId& requestId,
        TaskState state,
        const Failure& failure,
        const QDateTime& occurredAt,
        QString* error);

    QString connectionName_;
    QSqlDatabase db_;
};

} // namespace aitrain::v2
