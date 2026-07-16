#pragma once

#include "aitrain/v2/ModelImportServiceV2.h"
#include "aitrain/v2/ModelPackageRuntimeServiceV2.h"
#include "aitrain/v2/RuntimeInvocationV2.h"
#include "aitrain/v2/EvidenceBundleV2.h"
#include "aitrain/v2/DatasetSnapshotV2.h"
#include "aitrain/v2/DatasetConversionServiceV2.h"
#include "aitrain/v2/TaskExecutionHostV2.h"
#include "aitrain/v2/WorkflowRunnerV2.h"
#include "aitrain/core/Cancellation.h"

#include <functional>
#include <QHash>
#include <memory>

namespace aitrain::v2 {

struct RuntimeArtifactCandidateV2 final {
    QString kind;
    QString sourcePath;
};

struct RuntimeArtifactBundleV2 final {
    ArtifactId artifactId;
    QString artifactPath;
    QHash<QString, QString> pathsByKind;
};

struct EvidenceArtifactBundleV2 final {
    ArtifactId artifactId;
    QString artifactPath;
    QHash<QString, QString> pathsByKind;
};

struct DatasetSnapshotCommitRequestV2 final {
    QString datasetRoot;
    QString datasetFormat;
    QString driverId;
    QString driverVersion;
    DatasetSnapshotOptions options;
    // 新数据集导入/转换边界可预分配稳定 DatasetId；为空时仍由 Storage 生成。
    DatasetId datasetId;
};

struct DatasetSnapshotArtifactBundleV2 final {
    DatasetSnapshotRecordV2 snapshot;
    QString artifactPath;
    QString manifestPath;
    QJsonObject manifest;
};

struct DatasetSnapshotImportWorkflowRequestV2 final {
    // 这是显式外部数据导入边界；其余快照消费者只允许使用持久化身份。
    QString sourcePath;
    QString sourceFormat;
    DatasetId targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DatasetSnapshotImportWorkflowResultV2 final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId importPlanArtifactId;
    DatasetSnapshotRecordV2 datasetSnapshot;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

struct DataQualityWorkflowRequestV2 final {
    SnapshotId snapshotId;
    QJsonObject options;
    // GUI/Worker 边界必须同时携带完整登记身份；Core 在 ValidateSnapshot
    // 步骤内核对四者，避免只凭一个可误填的 SnapshotId 运行。
    DatasetId datasetId;
    DatasetVersionId datasetVersionId;
    ArtifactId snapshotArtifactId;
};

struct DataQualityWorkflowResultV2 final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId snapshotValidationArtifactId;
    ArtifactId qualityAnalysisArtifactId;
    ArtifactId repairManifestArtifactId;
    ArtifactId qualityReportArtifactId;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
};

// Diagnostics V2 只接受策略选项；项目根目录和 TaskId 由调用边界提供，
// 结果只返回持久化身份，不把 Artifact 磁盘路径泄漏到 Worker/GUI。
struct DiagnosticsWorkflowRequestV2 final {
    QJsonObject options;
};

struct DiagnosticsWorkflowResultV2 final {
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
struct EnvironmentCheckWorkflowRequestV2 final {
    QJsonObject facts;
};

struct EnvironmentCheckWorkflowResultV2 final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId factsArtifactId;
    ArtifactId reportArtifactId;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

struct DatasetConversionWorkflowRequestV2 final {
    // 外部源路径只允许出现在这一显式数据导入边界。
    QString sourcePath;
    QString sourceFormat;
    QString targetFormat;
    DatasetId targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DatasetConversionWorkflowResultV2 final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId conversionArtifactId;
    DatasetSnapshotRecordV2 datasetSnapshot;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

struct DatasetSplitWorkflowRequestV2 final {
    DatasetId sourceDatasetId;
    DatasetVersionId sourceDatasetVersionId;
    SnapshotId sourceSnapshotId;
    ArtifactId sourceSnapshotArtifactId;
    DatasetId targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DatasetSplitWorkflowResultV2 final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId splitPlanArtifactId;
    ArtifactId splitArtifactId;
    DatasetSnapshotRecordV2 datasetSnapshot;
    ArtifactId evidenceArtifactId;
    QJsonObject summary;
    Failure failure;
};

enum class AnnotationSyncStatusV2 {
    Inspected,
    ChangesDetected,
    NoChanges,
    InvalidSession,
    Conflict,
    Canceled
};

struct AnnotationSessionCreateRequestV2 final {
    ArtifactId repairManifestArtifactId;
    // 仅作为外部工具的临时工作目录 locator；不会写入 Artifact、Workflow 参数或数据库。
    QString workingDirectory;
    QJsonObject toolParameters;
};

struct AnnotationSessionCreateResultV2 final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    ArtifactId sessionArtifactId;
    ArtifactId evidenceArtifactId;
    AnnotationSyncStatusV2 status = AnnotationSyncStatusV2::InvalidSession;
};

struct AnnotationSessionSyncRequestV2 final {
    ArtifactId sessionArtifactId;
    // 外部工具可写目录，仅在本次同步中读取并重新校验，不作为持久化事实。
    QString workingDirectory;
};

struct AnnotationSessionSyncResultV2 final {
    WorkflowRunId workflowRunId;
    TaskState terminalState = TaskState::Failed;
    AnnotationSyncStatusV2 status = AnnotationSyncStatusV2::InvalidSession;
    ArtifactId inspectionArtifactId;
    ArtifactId changesArtifactId;
    ArtifactId syncReportArtifactId;
    ArtifactId evidenceArtifactId;
    DatasetSnapshotRecordV2 datasetSnapshot;
};

struct RuntimeDeliveryWorkflowRequestV2 final {
    ModelPackageId modelPackageId;
    QString runtimeRoute;
    QString sampleImagePath;
    QJsonObject options;
};

struct RuntimeDeliveryWorkflowResultV2 final {
    WorkflowRunId workflowRunId;
    WorkflowStepState state = WorkflowStepState::Pending;
    ArtifactId finalOutputArtifactId;
    EvidenceArtifactBundleV2 evidence;
    RuntimeStatusV2 runtimeStatus = RuntimeStatusV2::RuntimeNotImplemented;
    bool runtimeStatusObserved = false;
    Failure failure;
};

// 裸路径仅允许出现在这一显式导入边界。每个报告必须绑定一个已登记的数据集
// Snapshot；调用方可以提供 SnapshotId，或提供该 Snapshot 的 committed ArtifactId。
// 两者同时提供时必须解析到同一条持久化记录。
struct OcrOfficialReportImportSourceV2 final {
    QString reportPath;
    SnapshotId datasetSnapshotId;
    ArtifactId datasetSnapshotArtifactId;
};

struct OcrOfficialReportImportRequestV2 final {
    OcrOfficialReportImportSourceV2 det;
    OcrOfficialReportImportSourceV2 rec;
    OcrOfficialReportImportSourceV2 system;
    QString acceptanceCohortId;
    QString customerDomainId;
    QString evidenceClass;
};

struct OcrOfficialReportImportResultV2 final {
    ArtifactId detReportArtifactId;
    ArtifactId recReportArtifactId;
    ArtifactId systemReportArtifactId;
    ArtifactId evidenceArtifactId;
    Failure failure;
};

// OCR 验收只消费已提交的官方报告 Artifact。报告磁盘路径、客户数据目录和
// Python Adapter 输出目录都不能越过 ArtifactId 边界进入该请求。
struct OcrAcceptanceWorkflowRequestV2 final {
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

struct OcrAcceptanceWorkflowResultV2 final {
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

struct TrainingWorkflowRequestV2 final {
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

struct TrainingWorkflowDispatchV2 final {
    WorkflowRunId workflowRunId;
    WorkflowStepDispatchV2 dispatch;
};

struct VerifiedWorkflowArtifactFileV2 final {
    QString relativePath;
    QString absolutePath;
    QString sha256;
    qint64 byteCount = 0;
};

// 后端请求只能使用该解析结果中的绝对路径；解析过程重新校验 Storage 已登记的
// 文件长度和 SHA-256，拒绝符号链接、越界和提交后被篡改的 Artifact。
struct VerifiedTrainingWorkflowInputV2 final {
    ArtifactId artifactId;
    QString artifactPath;
    QVector<VerifiedWorkflowArtifactFileV2> files;
};

struct TrainingWorkflowAdapterConfigV2 final {
    QString pythonProgram;
    QString trainersRoot;
    QString deploymentSampleRelativePath;
    QProcessEnvironment environment;
    int cancellationGraceMs = 5000;
};

struct TrainingWorkflowAdapterLaunchV2 final {
    PythonAdapterLaunchV2 launch;
    QString requestPath;
    QJsonObject request;
};

// RegisterModel 步骤的不可变输出。模型包始终引用 Export 步骤的原始 Artifact；
// 此 Artifact 仅保存由已验证 sidecar 派生的最终 Manifest，供后续报告追溯。
struct TrainingModelRegistrationV2 final {
    ModelPackageSnapshotV2 modelPackage;
    RuntimeArtifactBundleV2 registrationArtifact;
};

struct TrainingDeploymentInvocationV2 final {
    QJsonObject invocation;
    ArtifactId sourceArtifactId;
};

// 真实后端步骤终态已落盘后才触发；调用方据此构建并派发下一步请求，不能直接
// 修改 Workflow Step 状态。
using TrainingWorkflowDispatchHandlerV2 = std::function<void(const TrainingWorkflowDispatchV2&)>;
using TrainingWorkflowAdapterEventHandlerV2 = std::function<void(const ProtocolEnvelope&)>;
using RuntimeAdapterFactoryV2 = std::function<std::unique_ptr<RuntimeAdapterV2>(const QString& runtimeRoute)>;

// V2 项目边界：GUI 通过模型包 ID 进行导入、浏览与运行时调用准备，
// 不持有可直接推断模型类型的裸模型路径。
class ProjectWorkspaceV2 final {
public:
    ProjectWorkspaceV2();

    bool open(const QString& projectRoot, QString* error = nullptr);
    void close();
    bool isOpen() const;

    bool importModel(const ModelImportRequestV2& request,
        ModelImportResultV2* result,
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
        const DatasetSnapshotCommitRequestV2& request,
        DatasetSnapshotArtifactBundleV2* result,
        QString* error = nullptr);
    bool runDatasetSnapshotImportWorkflow(const TaskId& taskId,
        const DatasetSnapshotImportWorkflowRequestV2& request,
        DatasetSnapshotImportWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runDataQualityWorkflow(const TaskId& taskId,
        const DataQualityWorkflowRequestV2& request,
        DataQualityWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runDiagnosticsWorkflow(const TaskId& taskId,
        const DiagnosticsWorkflowRequestV2& request,
        DiagnosticsWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runEnvironmentCheckWorkflow(const TaskId& taskId,
        const EnvironmentCheckWorkflowRequestV2& request,
        EnvironmentCheckWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool environmentCheckReportForTask(const TaskId& taskId,
        QJsonObject* report,
        QString* error = nullptr) const;
    bool runDatasetConversionWorkflow(const TaskId& taskId,
        const DatasetConversionWorkflowRequestV2& request,
        DatasetConversionWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool runDatasetSplitWorkflow(const TaskId& taskId,
        const DatasetSplitWorkflowRequestV2& request,
        DatasetSplitWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool createAnnotationSession(const TaskId& taskId,
        const AnnotationSessionCreateRequestV2& request,
        AnnotationSessionCreateResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool syncAnnotationSession(const TaskId& taskId,
        const AnnotationSessionSyncRequestV2& request,
        AnnotationSessionSyncResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool beginTrainingWorkflow(const TaskId& taskId,
        const TrainingWorkflowRequestV2& request,
        TrainingWorkflowDispatchV2* result,
        QString* error = nullptr);
    bool completeTrainingWorkflowStep(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const WorkflowStepExecutionResultV2& execution,
        TrainingWorkflowDispatchV2* result,
        QString* error = nullptr);
    bool startTrainingWorkflowAdapterStep(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const PythonAdapterLaunchV2& launch,
        TrainingWorkflowDispatchHandlerV2 nextStepHandler = {},
        QString* error = nullptr,
        TrainingWorkflowAdapterEventHandlerV2 eventHandler = {});
    bool requestTrainingWorkflowAdapterCancellation(const TaskId& taskId, QString* error = nullptr);
    bool isTrainingWorkflowAdapterRunning() const;
    AdapterEventEndpointV2 trainingWorkflowAdapterEndpoint() const;
    bool resolveTrainingWorkflowStepInput(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        VerifiedTrainingWorkflowInputV2* result,
        QString* error = nullptr) const;
    bool prepareTrainingWorkflowAdapterLaunch(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const TrainingWorkflowAdapterConfigV2& config,
        TrainingWorkflowAdapterLaunchV2* result,
        QString* error = nullptr) const;
    bool registerTrainingWorkflowModel(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        TrainingModelRegistrationV2* result,
        QString* error = nullptr);
    bool prepareTrainingWorkflowDeploymentInvocation(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const QString& deploymentSampleRelativePath,
        TrainingDeploymentInvocationV2* result,
        QString* error = nullptr) const;
    bool renderTrainingWorkflowDeliveryReport(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        RuntimeArtifactBundleV2* result,
        QString* error = nullptr);
    bool runRuntimeDeliveryWorkflow(const TaskId& taskId,
        const RuntimeDeliveryWorkflowRequestV2& request,
        RuntimeDeliveryWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {},
        RuntimeAdapterFactoryV2 adapterFactory = {});
    bool runOcrAcceptanceWorkflow(const TaskId& taskId,
        const OcrAcceptanceWorkflowRequestV2& request,
        OcrAcceptanceWorkflowResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    bool importOcrOfficialReports(const TaskId& taskId,
        const OcrOfficialReportImportRequestV2& request,
        OcrOfficialReportImportResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    QString runtimeStagingPath(const TaskId& taskId) const;
    bool commitRuntimeArtifacts(const TaskId& taskId,
        const QString& bundleKind,
        const QVector<RuntimeArtifactCandidateV2>& candidates,
        RuntimeArtifactBundleV2* result,
        QString* error = nullptr);
    bool buildWorkflowEvidenceBundle(const WorkflowRunId& workflowRunId,
        EvidenceBundleV2* result,
        QString* error = nullptr) const;
    bool commitEvidenceBundle(const EvidenceBundleV2& bundle,
        EvidenceArtifactBundleV2* result,
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
    QVector<ArtifactSnapshotV2> artifactsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<MetricSnapshotV2> metricsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowRunSnapshotV2> workflowRunsForTask(const TaskId& taskId, QString* error = nullptr) const;
    QVector<WorkflowStepSnapshotV2> workflowSteps(const WorkflowRunId& workflowRunId, QString* error = nullptr) const;
    QVector<ModelPackageSnapshotV2> modelPackages(int limit, QString* error = nullptr) const;
    QVector<DatasetCatalogItemV2> datasets(int limit, QString* error = nullptr) const;
    bool projectSummary(ProjectSummarySnapshotV2* result, QString* error = nullptr) const;
    QString workspacePath() const;

private:
    bool recoverEvidenceGatedWorkflows(QString* error);

    StorageV2 storage_;
    std::unique_ptr<ArtifactStoreV2> artifactStore_;
    std::unique_ptr<TaskCoordinator> taskCoordinator_;
    std::unique_ptr<TaskExecutionHostV2> trainingAdapterHost_;
    QString workspacePath_;
};

} // namespace aitrain::v2
