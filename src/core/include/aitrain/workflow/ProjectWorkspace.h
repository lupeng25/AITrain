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
#include <QHash>
#include <memory>

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
    QString runtimeRoute;
    QString sampleImagePath;
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
    QVector<ArtifactSnapshot> artifactsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<MetricSnapshot> metricsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowRunSnapshot> workflowRunsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowStepSnapshot> workflowSteps(const WorkflowRunId& workflowRunId, QString* error = nullptr) const;
    QVector<ModelPackageSnapshot> modelPackages(int limit, QString* error = nullptr) const;
    QVector<DatasetCatalogItem> datasets(int limit, QString* error = nullptr) const;
    bool projectSummary(ProjectSummarySnapshot* result, QString* error = nullptr) const;
    QString workspacePath() const;

private:
    bool recoverEvidenceGatedWorkflows(QString* error);

    ProjectStore storage_;
    std::unique_ptr<ArtifactStore> artifactStore_;
    std::unique_ptr<TaskCoordinator> taskCoordinator_;
    std::unique_ptr<TaskExecutionHost> trainingAdapterHost_;
    QString workspacePath_;
};

} // namespace aitrain
