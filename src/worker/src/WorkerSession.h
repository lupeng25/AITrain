#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/workflow/ProjectWorkspace.h"

#include <QJsonArray>
#include <QElapsedTimer>
#include <QLocalSocket>
#include <QObject>
#include <QProcess>
#include <QTimer>
#include <QVector>

#include <memory>

class WorkerSession : public QObject {
    Q_OBJECT

public:
    explicit WorkerSession(QObject* parent = nullptr);
    bool connectToServer(const QString& serverName,
        const aitrain::RequestId& requestId,
        const aitrain::TaskId& taskId,
        const QString& controlToken);

private slots:
    void readLines();
    void handleSocketDisconnected();

private:
    void handleCommand(const aitrain::worker_protocol::TaskCommand& command);
    void cancelCommand();
    bool requestCancellationForActiveWorkflow();
    void requestCancellationForTrackedWorkflows();
    void runEnvironmentCheckWorkflow(const aitrain::worker_protocol::EnvironmentCheckCommand& command);
    void runDatasetSplitWorkflow(const aitrain::worker_protocol::DatasetSplitCommand& command);
    void runDatasetConversionWorkflow(const aitrain::worker_protocol::DatasetConversionCommand& command);
    void runDataQualityWorkflow(const aitrain::worker_protocol::DataQualityCommand& command);
    void runDiagnosticsWorkflow(const aitrain::worker_protocol::DiagnosticsCommand& command);
    void importExternalAcceptanceEvidence(const aitrain::worker_protocol::ExternalAcceptanceEvidenceImportCommand& command);
    void createAnnotationSession(const aitrain::worker_protocol::AnnotationSessionCreateCommand& command);
    void syncAnnotationSession(const aitrain::worker_protocol::AnnotationSessionSyncCommand& command);
    void runDatasetSnapshotImportWorkflow(const aitrain::worker_protocol::DatasetSnapshotImportCommand& command);
    void importOcrOfficialReports(const aitrain::worker_protocol::OcrOfficialReportImportCommand& command);
    void runOcrAcceptanceWorkflow(const aitrain::worker_protocol::OcrAcceptanceCommand& command);
    void runRuntimeDeliveryWorkflow(const aitrain::worker_protocol::RuntimeDeliveryCommand& command);
    void importModel(const aitrain::worker_protocol::ModelImportCommand& command);
    void runTrainingWorkflow(const aitrain::worker_protocol::TrainingCommand& command);
    void dispatchTrainingWorkflow(const aitrain::TrainingWorkflowDispatch& dispatch);
    void runTrainingWorkflowLocalStep(const aitrain::TrainingWorkflowDispatch& dispatch);
    void finishTrainingWorkflow(const aitrain::TrainingWorkflowDispatch& dispatch);
    void forwardTrainingWorkflowAdapterEvent(const aitrain::ProtocolEnvelope& event);
    void cancelTrainingWorkflow();
    void send(const QString& type, const QJsonObject& payload);
    bool acceptControlEnvelope(const aitrain::ProtocolEnvelope& envelope);
    void rejectControlProtocol(const QString& message);
    aitrain::CancellationCallback cancellationCallback();
    aitrain::CancellationCallback pollingCancellationCallback(int timeoutMs = 0);
    void shutdownPythonTrainer(const QString& reason, bool notifyClient);
    bool pollPendingCancel(int timeoutMs = 100);
    void sendCanceledAndFinish(const QString& taskId, const QString& message);
    void finishSession();
    void maybeFinishSessionAfterWrite();
    void fail(const QString& message);
    void failWithDetails(const QString& message, const QString& errorCode, const QJsonObject& details = {});

    QLocalSocket socket_;
    QByteArray buffer_;
    bool running_ = false;
    bool canceled_ = false;
    QProcess pythonTrainerProcess_;
    bool interceptPythonTrainerMessages_ = false;
    bool finishingSession_ = false;
    bool startTaskReceived_ = false;
    bool terminalEnvelopeSent_ = false;
    QTimer terminalDrainTimer_;
    QElapsedTimer terminalDrainElapsed_;
    bool terminalQuitScheduled_ = false;
    // 控制面只允许有限的待写缓存；日志/进度/指标是可丢弃事件，不能反向
    // 把生产任务的速度绑定到 GUI 消费速度。终态 payload 会带出丢弃计数。
    quint64 droppedControlEventCount_ = 0;
    quint64 outgoingSequence_ = 0;
    aitrain::RequestId controlRequestId_;
    aitrain::TaskId controlTaskId_;
    QString controlToken_;
    aitrain::ProtocolSequenceTracker incomingSequenceTracker_;
    QString activeTaskId_;
    QString activeCommand_;
    std::unique_ptr<aitrain::ProjectWorkspace> trainingWorkspace_;
    aitrain::TaskId trainingWorkflowTaskId_;
    aitrain::WorkflowRunId trainingWorkflowRunId_;
    QString trainingWorkflowDeploymentSampleRelativePath_;
    aitrain::TrainingWorkflowAdapterConfig trainingWorkflowAdapterConfig_;
    std::unique_ptr<aitrain::ProjectWorkspace> runtimeDeliveryWorkspace_;
    aitrain::TaskId runtimeDeliveryTaskId_;
    bool runtimeDeliveryRunning_ = false;
    std::unique_ptr<aitrain::ProjectWorkspace> annotationWorkspace_;
    aitrain::TaskId annotationTaskId_;
    bool annotationRunning_ = false;
    std::unique_ptr<aitrain::ProjectWorkspace> ocrAcceptanceWorkspace_;
    aitrain::TaskId ocrAcceptanceTaskId_;
    bool ocrAcceptanceRunning_ = false;
    std::unique_ptr<aitrain::ProjectWorkspace> dataQualityWorkspace_;
    aitrain::TaskId dataQualityTaskId_;
    bool dataQualityRunning_ = false;
    std::unique_ptr<aitrain::ProjectWorkspace> datasetConversionWorkspace_;
    aitrain::TaskId datasetConversionTaskId_;
    bool datasetConversionRunning_ = false;
    std::unique_ptr<aitrain::ProjectWorkspace> datasetSnapshotImportWorkspace_;
    aitrain::TaskId datasetSnapshotImportTaskId_;
    bool datasetSnapshotImportRunning_ = false;
    std::unique_ptr<aitrain::ProjectWorkspace> datasetSplitWorkspace_;
    aitrain::TaskId datasetSplitTaskId_;
    bool datasetSplitRunning_ = false;
    std::unique_ptr<aitrain::ProjectWorkspace> diagnosticsWorkspace_;
    aitrain::TaskId diagnosticsTaskId_;
    bool diagnosticsRunning_ = false;
};
