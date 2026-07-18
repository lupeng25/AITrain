#pragma once

#include "aitrain/model/ModelImportService.h"
#include "aitrain/runtime/ModelPackageRuntimeService.h"
#include "aitrain/runtime/RuntimeInvocation.h"
#include "aitrain/workflow/EvidenceBundle.h"
#include "aitrain/dataset/DatasetSnapshot.h"
#include "aitrain/dataset/DatasetConversionService.h"
#include "aitrain/workflow/TaskExecutionHost.h"
#include "aitrain/workflow/WorkflowRunner.h"
#include "aitrain/core/Cancellation.h"

#include <functional>
#include <QByteArray>
#include <QDateTime>
#include <QHash>
#include <QStringList>
#include <memory>

class QObject;

namespace aitrain {

struct RuntimeArtifactCandidate final {
    QString kind;
    QString sourcePath;
};

struct RuntimeArtifactBundle final {
    ArtifactId artifactId;
    QString artifactPath;
    QHash<QString, QString> pathsByKind;
};

// GUI/Query 的只读 Artifact 预览结果。内容来自已提交文件并经过清单哈希复验，
// 不向上层暴露 Artifact Store 物理路径。
struct ArtifactFilePreview final {
    QString relativePath;
    QString sha256;
    qint64 byteCount = 0;
    QByteArray content;
    bool truncated = false;
};

// 由当前线程根据已提交 Artifact 清单解析出的不可变文件读取来源。
// 该结构只供 Query Service 的异步文件校验使用；后台任务不得捕获
// ProjectWorkspace、ProjectStore 或 QSqlDatabase。
struct CommittedArtifactFileReadSource final {
    QString artifactRoot;
    QString absolutePath;
    ArtifactFileSnapshot expected;
};

// Artifact 文件预览回调。元数据查询在调用线程执行，文件内容和完整 SHA-256
// 校验在受控线程池执行；回调总是排队回到 receiver 所在线程。
using ArtifactFilePreviewCallback = std::function<void(
    bool success, ArtifactFilePreview preview, QString error)>;

struct EvidenceArtifactBundle final {
    ArtifactId artifactId;
    QString artifactPath;
    QHash<QString, QString> pathsByKind;
};

struct DatasetSnapshotCommitRequest final {
    QString datasetRoot;
    QString datasetFormat;
    QString driverId;
    QString driverVersion;
    DatasetSnapshotOptions options;
    // 新数据集导入/转换边界可预分配稳定 DatasetId；为空时仍由 Storage 生成。
    DatasetId datasetId;
};

struct DatasetSnapshotArtifactBundle final {
    DatasetSnapshotRecord snapshot;
    QString artifactPath;
    QString manifestPath;
    QJsonObject manifest;
};

struct DatasetSnapshotImportWorkflowRequest final {
    // 这是显式外部数据导入边界；其余快照消费者只允许使用持久化身份。
    QString sourcePath;
    QString sourceFormat;
    DatasetId targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DatasetSnapshotImportWorkflowResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId importPlanArtifactId;
    DatasetSnapshotRecord datasetSnapshot;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

struct DataQualityWorkflowRequest final {
    SnapshotId snapshotId;
    QJsonObject options;
    // GUI/Worker 边界必须同时携带完整登记身份；Core 在 ValidateSnapshot
    // 步骤内核对四者，避免只凭一个可误填的 SnapshotId 运行。
    DatasetId datasetId;
    DatasetVersionId datasetVersionId;
    ArtifactId snapshotArtifactId;
};

struct DataQualityWorkflowResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId snapshotValidationArtifactId;
    ArtifactId qualityAnalysisArtifactId;
    ArtifactId repairManifestArtifactId;
    ArtifactId qualityReportArtifactId;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
};

// Diagnostics  只接受策略选项；项目根目录和 TaskId 由调用边界提供，
// 结果只返回持久化身份，不把 Artifact 磁盘路径泄漏到 Worker/GUI。
struct DiagnosticsWorkflowRequest final {
    QJsonObject options;
};

struct DiagnosticsWorkflowResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId factsArtifactId;
    ArtifactId diagnosticsArtifactId;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

struct ExternalAcceptanceEvidenceImportRequest final {
    // 仅允许在显式外部验收证据导入边界使用裸路径；内容会在提交前
    // 经过严格 schema 校验，之后上层只使用 ArtifactId。
    QString sourcePath;
};

struct ExternalAcceptanceEvidenceImportResult final {
    ArtifactId evidenceArtifactId;
    QString evidenceKind;
    QString status;
    QString producer;
    QDateTime observedAt;
    QStringList limitations;
    QJsonObject summary;
    Failure failure;
};

// 环境探测由 Worker 执行，Core 只接受结构化事实并负责验证、持久化、
// Evidence 收口。结果边界只暴露身份，不暴露临时目录或报告路径。
struct EnvironmentCheckWorkflowRequest final {
    QJsonObject facts;
};

struct EnvironmentCheckWorkflowResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId factsArtifactId;
    ArtifactId reportArtifactId;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

struct DatasetConversionWorkflowRequest final {
    // 外部源路径只允许出现在这一显式数据导入边界。
    QString sourcePath;
    QString sourceFormat;
    QString targetFormat;
    DatasetId targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DatasetConversionWorkflowResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId conversionArtifactId;
    DatasetSnapshotRecord datasetSnapshot;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

struct DatasetSplitWorkflowRequest final {
    DatasetId sourceDatasetId;
    DatasetVersionId sourceDatasetVersionId;
    SnapshotId sourceSnapshotId;
    ArtifactId sourceSnapshotArtifactId;
    DatasetId targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DatasetSplitWorkflowResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId splitPlanArtifactId;
    ArtifactId splitArtifactId;
    DatasetSnapshotRecord datasetSnapshot;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

enum class AnnotationSyncStatus {
    Inspected,
    ChangesDetected,
    NoChanges,
    InvalidSession,
    Conflict,
    Canceled
};

struct AnnotationSessionCreateRequest final {
    ArtifactId repairManifestArtifactId;
    // 仅作为外部工具的临时工作目录 locator；不会写入 Artifact、Workflow 参数或数据库。
    QString workingDirectory;
    QJsonObject toolParameters;
};

struct AnnotationSessionCreateResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId sessionArtifactId;
    ArtifactId evidenceArtifactId;
    AnnotationSyncStatus status = AnnotationSyncStatus::InvalidSession;
};

struct AnnotationSessionSyncRequest final {
    ArtifactId sessionArtifactId;
    // 外部工具可写目录，仅在本次同步中读取并重新校验，不作为持久化事实。
    QString workingDirectory;
};

struct AnnotationSessionSyncResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    AnnotationSyncStatus status = AnnotationSyncStatus::InvalidSession;
    ArtifactId inspectionArtifactId;
    ArtifactId changesArtifactId;
    ArtifactId syncReportArtifactId;
    ArtifactId evidenceArtifactId;
    DatasetSnapshotRecord datasetSnapshot;
};

struct RuntimeDeliveryWorkflowRequest final {
    ModelPackageId modelPackageId;
    DatasetId sampleDatasetId;
    DatasetVersionId sampleDatasetVersionId;
    SnapshotId sampleSnapshotId;
    ArtifactId sampleSnapshotArtifactId;
    QString sampleRelativePath;
    QString runtimeRoute;
    QJsonObject options;
};

struct RuntimeDeliveryWorkflowResult final {
    WorkflowRunId workflowRunId;
    WorkflowStepState state = WorkflowStepState::Pending;
    ArtifactId finalOutputArtifactId;
    EvidenceArtifactBundle evidence;
    RuntimeStatus runtimeStatus = RuntimeStatus::RuntimeNotImplemented;
    bool runtimeStatusObserved = false;
    Failure failure;
};

// 裸路径仅允许出现在这一显式导入边界。每个报告必须绑定一个已登记的数据集
// Snapshot；调用方可以提供 SnapshotId，或提供该 Snapshot 的 committed ArtifactId。
// 两者同时提供时必须解析到同一条持久化记录。
struct OcrOfficialReportImportSource final {
    QString reportPath;
    SnapshotId datasetSnapshotId;
    ArtifactId datasetSnapshotArtifactId;
};

struct OcrOfficialReportImportRequest final {
    OcrOfficialReportImportSource det;
    OcrOfficialReportImportSource rec;
    OcrOfficialReportImportSource system;
    QString acceptanceCohortId;
    QString customerDomainId;
    QString evidenceClass;
};

struct OcrOfficialReportImportResult final {
    ArtifactId detReportArtifactId;
    ArtifactId recReportArtifactId;
    ArtifactId systemReportArtifactId;
    ArtifactId evidenceArtifactId;
    Failure failure;
};

// OCR 验收只消费已提交的官方报告 Artifact。报告磁盘路径、客户数据目录和
// Python Adapter 输出目录都不能越过 ArtifactId 边界进入该请求。
struct OcrAcceptanceWorkflowRequest final {
    ArtifactId detReportArtifactId;
    ArtifactId recReportArtifactId;
    ArtifactId systemReportArtifactId;
    int minimumDetSamples = 1;
    int minimumRecSamples = 1;
    int minimumSystemSamples = 1;
    double minimumDetHmean = 0.50;
    double minimumRecAccuracy = 0.70;
    double maximumRecCer = 0.30;
    double minimumSystemAccuracy = 0.70;
};

struct OcrAcceptanceWorkflowResult final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId resolvedEvidenceArtifactId;
    ArtifactId officialReportValidationArtifactId;
    ArtifactId thresholdEvaluationArtifactId;
    ArtifactId acceptanceReportArtifactId;
    ArtifactId evidenceArtifactId;
    bool productionAccepted = false;
    Failure failure;
};

struct TrainingWorkflowRequest final {
    DatasetId datasetId;
    DatasetVersionId datasetVersionId;
    SnapshotId snapshotId;
    ArtifactId snapshotArtifactId;
    QString templateId;
    QString trainingBackend;
    QString evaluationBackend;
    QString exportBackend;
    QString deploymentBackend;
    QJsonObject parameterSummary;
    bool requireEvidenceBeforeTerminal = false;
};

struct TrainingWorkflowDispatch final {
    WorkflowRunId workflowRunId;
    WorkflowStepDispatch dispatch;
};

struct VerifiedWorkflowArtifactFile final {
    QString relativePath;
    QString absolutePath;
    QString sha256;
    qint64 byteCount = 0;
};

// 后端请求只能使用该解析结果中的绝对路径；解析过程重新校验 Storage 已登记的
// 文件长度和 SHA-256，拒绝符号链接、越界和提交后被篡改的 Artifact。
struct VerifiedTrainingWorkflowInput final {
    ArtifactId artifactId;
    QString artifactPath;
    QVector<VerifiedWorkflowArtifactFile> files;
};

struct TrainingWorkflowAdapterConfig final {
    QString pythonProgram;
    QString trainersRoot;
    QString deploymentSampleRelativePath;
    QProcessEnvironment environment;
    int cancellationGraceMs = 5000;
};

struct TrainingWorkflowAdapterLaunch final {
    PythonAdapterLaunch launch;
    QString requestPath;
    QJsonObject request;
};

// RegisterModel 步骤的不可变输出。模型包始终引用 Export 步骤的原始 Artifact；
// 此 Artifact 仅保存由已验证 sidecar 派生的最终 Manifest，供后续报告追溯。
struct TrainingModelRegistration final {
    ModelPackageSnapshot modelPackage;
    RuntimeArtifactBundle registrationArtifact;
};

struct TrainingDeploymentInvocation final {
    QJsonObject invocation;
    ArtifactId sourceArtifactId;
};

// 候选工作区在后台线程完成完整 open/recovery 后交给 GUI 的不可变激活凭证。
// 只携带文件元数据，不携带 QSqlDatabase、ProjectStore 或 ArtifactStore。
struct ProjectWorkspacePreparedOpen final {
    QString normalizedRoot;
    // 覆盖恢复会读取的所有持久化文件与暂存树，避免仅依赖父目录
    // mtime/文件大小导致同大小快速修改绕过激活校验。
    QString fingerprintSha256;

    bool isValid() const
    {
        return !normalizedRoot.isEmpty() && fingerprintSha256.size() == 64;
    }
};

// 真实后端步骤终态已落盘后才触发；调用方据此构建并派发下一步请求，不能直接
// 修改 Workflow Step 状态。
using TrainingWorkflowDispatchHandler = std::function<void(const TrainingWorkflowDispatch&)>;
using TrainingWorkflowAdapterEventHandler = std::function<void(const ProtocolEnvelope&)>;
using RuntimeAdapterFactory = std::function<std::unique_ptr<RuntimeAdapter>(const QString& runtimeRoute)>;

//  项目边界：GUI 通过模型包 ID 进行导入、浏览与运行时调用准备，
// 不持有可直接推断模型类型的裸模型路径。
class ProjectWorkspace final {
public:
    ProjectWorkspace();

    bool open(const QString& projectRoot, QString* error = nullptr);
    // 在调用线程创建临时 ProjectWorkspace，完整执行恢复并返回值类型凭证。
    // 候选对象及其 QSqlDatabase 始终留在调用线程，适合由 QThreadPool 调用。
    static bool prepareOpen(const QString& projectRoot,
        ProjectWorkspacePreparedOpen* prepared, QString* error = nullptr);
    // 仅在凭证指纹仍匹配时激活。恢复已由 prepareOpen 完成，GUI 不重复扫描/恢复。
    bool openPrepared(const ProjectWorkspacePreparedOpen& prepared,
        QString* error = nullptr);
    // Worker 异常退出后立即收口当前任务；应用重启恢复仍会再次执行同一套
    // 幂等检查，因此该入口不会依赖 GUI 的瞬态事件是否成功送达。
    bool recoverAfterWorkerLoss(const TaskId& taskId, QString* error = nullptr);
    void close();
    bool isOpen() const;

    bool importModel(const ModelImportRequest& request,
        ModelImportResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool startTask(const TaskId& taskId,
        const QString& capabilityId,
        const QString& taskType,
        TaskSnapshot* task,
        QString* error = nullptr);
    bool requestTaskCancellation(const TaskId& taskId, QString* error = nullptr);
    bool finalizeTask(const TaskId& taskId,
        TaskState terminalState,
        const Failure& failure = {},
        QString* error = nullptr);
    bool commitDatasetSnapshot(const TaskId& taskId,
        const DatasetSnapshotCommitRequest& request,
        DatasetSnapshotArtifactBundle* result,
        QString* error = nullptr);
    bool runDatasetSnapshotImportWorkflow(const TaskId& taskId,
        const DatasetSnapshotImportWorkflowRequest& request,
        DatasetSnapshotImportWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runDataQualityWorkflow(const TaskId& taskId,
        const DataQualityWorkflowRequest& request,
        DataQualityWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runDiagnosticsWorkflow(const TaskId& taskId,
        const DiagnosticsWorkflowRequest& request,
        DiagnosticsWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool importExternalAcceptanceEvidence(const TaskId& taskId,
        const ExternalAcceptanceEvidenceImportRequest& request,
        ExternalAcceptanceEvidenceImportResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runEnvironmentCheckWorkflow(const TaskId& taskId,
        const EnvironmentCheckWorkflowRequest& request,
        EnvironmentCheckWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool environmentCheckReportForTask(const TaskId& taskId,
        QJsonObject* report,
        QString* error = nullptr) const;
    bool runDatasetConversionWorkflow(const TaskId& taskId,
        const DatasetConversionWorkflowRequest& request,
        DatasetConversionWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runDatasetSplitWorkflow(const TaskId& taskId,
        const DatasetSplitWorkflowRequest& request,
        DatasetSplitWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool createAnnotationSession(const TaskId& taskId,
        const AnnotationSessionCreateRequest& request,
        AnnotationSessionCreateResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool syncAnnotationSession(const TaskId& taskId,
        const AnnotationSessionSyncRequest& request,
        AnnotationSessionSyncResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool beginTrainingWorkflow(const TaskId& taskId,
        const TrainingWorkflowRequest& request,
        TrainingWorkflowDispatch* result,
        QString* error = nullptr);
    bool completeTrainingWorkflowStep(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const WorkflowStepExecutionResult& execution,
        TrainingWorkflowDispatch* result,
        QString* error = nullptr);
    bool startTrainingWorkflowAdapterStep(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const PythonAdapterLaunch& launch,
        TrainingWorkflowDispatchHandler nextStepHandler = {},
        QString* error = nullptr,
        TrainingWorkflowAdapterEventHandler eventHandler = {});
    bool requestTrainingWorkflowAdapterCancellation(const TaskId& taskId, QString* error = nullptr);
    bool isTrainingWorkflowAdapterRunning() const;
    AdapterEventEndpoint trainingWorkflowAdapterEndpoint() const;
    bool resolveTrainingWorkflowStepInput(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        VerifiedTrainingWorkflowInput* result,
        QString* error = nullptr) const;
    bool prepareTrainingWorkflowAdapterLaunch(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const TrainingWorkflowAdapterConfig& config,
        TrainingWorkflowAdapterLaunch* result,
        QString* error = nullptr) const;
    bool registerTrainingWorkflowModel(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        TrainingModelRegistration* result,
        QString* error = nullptr);
    bool prepareTrainingWorkflowDeploymentInvocation(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const QString& deploymentSampleRelativePath,
        TrainingDeploymentInvocation* result,
        QString* error = nullptr) const;
    bool renderTrainingWorkflowDeliveryReport(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        RuntimeArtifactBundle* result,
        QString* error = nullptr);
    bool runRuntimeDeliveryWorkflow(const TaskId& taskId,
        const RuntimeDeliveryWorkflowRequest& request,
        RuntimeDeliveryWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {},
        RuntimeAdapterFactory adapterFactory = {});
    bool runOcrAcceptanceWorkflow(const TaskId& taskId,
        const OcrAcceptanceWorkflowRequest& request,
        OcrAcceptanceWorkflowResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool importOcrOfficialReports(const TaskId& taskId,
        const OcrOfficialReportImportRequest& request,
        OcrOfficialReportImportResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    QString runtimeStagingPath(const TaskId& taskId) const;
    bool commitRuntimeArtifacts(const TaskId& taskId,
        const QString& bundleKind,
        const QVector<RuntimeArtifactCandidate>& candidates,
        RuntimeArtifactBundle* result,
        QString* error = nullptr);
    bool buildWorkflowEvidenceBundle(const WorkflowRunId& workflowRunId,
        EvidenceBundle* result,
        QString* error = nullptr) const;
    bool commitEvidenceBundle(const EvidenceBundle& bundle,
        EvidenceArtifactBundle* result,
        QString* error = nullptr);
    bool recordWorkflowEvidenceFailure(const WorkflowRunId& workflowRunId,
        const Failure& failure,
        QString* error = nullptr);
    bool closeWorkflowTerminalization(const WorkflowRunId& workflowRunId,
        QString* error = nullptr);
    // 训练调用点的过渡别名；通用 Runtime/Workflow 代码必须使用上面的语义中立 API。
    bool cleanupRuntimeStaging(const TaskId& taskId, QString* error = nullptr);
    QVector<TaskSnapshot> tasks(int limit, QString* error = nullptr) const;
    bool task(const TaskId& taskId, TaskSnapshot* result, QString* error = nullptr) const;
    bool artifact(const ArtifactId& artifactId, ArtifactSnapshot* result, QString* error = nullptr) const;
    QVector<ArtifactSnapshot> artifactsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<DeliveryEvidenceCandidate> deliveryEvidenceCandidates(
        int limit, QString* error = nullptr) const;
    bool readCommittedArtifactFile(const ArtifactId& artifactId,
        const QString& relativePath,
        ArtifactFilePreview* result,
        qint64 maxBytes = 512 * 1024,
        QString* error = nullptr) const;
    // 已经从同一条候选查询加载过清单时使用此重载，避免再次查询 Artifact 元数据。
    bool readCommittedArtifactFile(const ArtifactSnapshot& snapshot,
        const QString& relativePath,
        ArtifactFilePreview* result,
        qint64 maxBytes = 512 * 1024,
        QString* error = nullptr) const;
    bool prepareCommittedArtifactFileRead(const ArtifactSnapshot& snapshot,
        const QString& relativePath,
        CommittedArtifactFileReadSource* result,
        QString* error = nullptr) const;
    // 先在当前线程读取 committed Artifact 清单，再在后台读取文件并复验
    // SHA-256。后台任务不持有 ProjectStore/QSqlDatabase。
    bool readCommittedArtifactFileAsync(const ArtifactId& artifactId,
        const QString& relativePath,
        QObject* receiver,
        ArtifactFilePreviewCallback callback,
        qint64 maxBytes = 512 * 1024,
        QString* error = nullptr) const;
    QVector<MetricSnapshot> metricsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowRunSnapshot> workflowRunsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowStepSnapshot> workflowSteps(const WorkflowRunId& workflowRunId, QString* error = nullptr) const;
    QVector<ModelPackageSnapshot> modelPackages(int limit, QString* error = nullptr) const;
    QVector<DatasetCatalogItem> datasets(int limit, QString* error = nullptr) const;
    bool projectSummary(ProjectSummarySnapshot* result, QString* error = nullptr) const;
    QString workspacePath() const;

private:
    bool openInternal(const QString& projectRoot, bool recover, QString* error);
    static bool capturePreparedOpenFingerprint(const QString& normalizedRoot,
        ProjectWorkspacePreparedOpen* prepared, QString* error);
    static bool preparedOpenFingerprintMatches(const ProjectWorkspacePreparedOpen& prepared,
        QString* error);
    bool recoverPendingWorkflowTerminalEvents(QString* error);
    bool recoverEvidenceGatedWorkflows(QString* error);
    bool recoverRuntimeStaging(QString* error);

    ProjectStore storage_;
    std::unique_ptr<ArtifactStore> artifactStore_;
    std::unique_ptr<TaskCoordinator> taskCoordinator_;
    std::unique_ptr<TaskExecutionHost> trainingAdapterHost_;
    QString workspacePath_;
};

} // namespace aitrain
