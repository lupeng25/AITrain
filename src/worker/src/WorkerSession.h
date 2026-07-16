#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/v2/ProjectWorkspaceV2.h"

#include <QJsonArray>
#include <QLocalSocket>
#include <QObject>
#include <QProcess>
#include <QVector>

#include <memory>

class WorkerSession : public QObject {
    Q_OBJECT

public:
    explicit WorkerSession(QObject* parent = nullptr);
    bool connectToServer(const QString& serverName,
        const aitrain::v2::RequestId& requestId,
        const aitrain::v2::TaskId& taskId);

private slots:
    void readLines();
    void handleSocketDisconnected();

private:
    struct CommandBinding {
        QString command;
        void (WorkerSession::*handler)(const QJsonObject&);
    };

    static QVector<CommandBinding> commandBindings();

    void handleMessage(const QString& type, const QJsonObject& payload);
    void runEnvironmentCheckWorkflowV2Command(const QJsonObject& payload);
    void runDatasetSplitWorkflowV2Command(const QJsonObject& payload);
    void runDatasetConversionWorkflowV2Command(const QJsonObject& payload);
    void runDataQualityWorkflowV2Command(const QJsonObject& payload);
    void runDiagnosticsWorkflowV2Command(const QJsonObject& payload);
    void createAnnotationSessionV2Command(const QJsonObject& payload);
    void syncAnnotationSessionV2Command(const QJsonObject& payload);
    void runDatasetSnapshotImportWorkflowV2Command(const QJsonObject& payload);
    void importOcrOfficialReportsV2Command(const QJsonObject& payload);
    void runOcrAcceptanceWorkflowV2Command(const QJsonObject& payload);
    void runRuntimeDeliveryWorkflowV2Command(const QJsonObject& payload);
    void importModelV2Command(const QJsonObject& payload);
    void runTrainingWorkflowV2Command(const QJsonObject& payload);
    void cancelCommand(const QJsonObject& payload);
    void runEnvironmentCheckWorkflowV2(const QJsonObject& payload);
    void runDatasetSplitWorkflowV2(const QJsonObject& payload);
    void runDatasetConversionWorkflowV2(const QJsonObject& payload);
    void runDataQualityWorkflowV2(const QJsonObject& payload);
    void runDiagnosticsWorkflowV2(const QJsonObject& payload);
    void createAnnotationSessionV2(const QJsonObject& payload);
    void syncAnnotationSessionV2(const QJsonObject& payload);
    void runDatasetSnapshotImportWorkflowV2(const QJsonObject& payload);
    void importOcrOfficialReportsV2(const QJsonObject& payload);
    void runOcrAcceptanceWorkflowV2(const QJsonObject& payload);
    void runRuntimeDeliveryWorkflowV2(const QJsonObject& payload);
    void importModelV2(const QJsonObject& payload);
    void runTrainingWorkflowV2(const QJsonObject& payload);
    void dispatchTrainingWorkflowV2(const aitrain::v2::TrainingWorkflowDispatchV2& dispatch);
    void runTrainingWorkflowV2LocalStep(const aitrain::v2::TrainingWorkflowDispatchV2& dispatch);
    void finishTrainingWorkflowV2(const aitrain::v2::TrainingWorkflowDispatchV2& dispatch);
    void forwardTrainingWorkflowAdapterEvent(const aitrain::v2::ProtocolEnvelope& event);
    void cancelTrainingWorkflowV2();
    void send(const QString& type, const QJsonObject& payload);
    bool acceptControlEnvelope(const aitrain::v2::ProtocolEnvelope& envelope);
    void rejectControlProtocol(const QString& message);
    aitrain::CancellationCallback cancellationCallback();
    aitrain::CancellationCallback pollingCancellationCallback(int timeoutMs = 0);
    void shutdownPythonTrainer(const QString& reason, bool notifyClient);
    bool pollPendingCancel(int timeoutMs = 100);
    void sendCanceledAndFinish(const QString& taskId, const QString& message);
    void finishSession();
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
    quint64 outgoingSequence_ = 0;
    aitrain::v2::RequestId controlRequestId_;
    aitrain::v2::TaskId controlTaskId_;
    aitrain::v2::ProtocolV2SequenceTracker incomingSequenceTracker_;
    QString activeTaskId_;
    QString activeCommand_;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> trainingWorkspaceV2_;
    aitrain::v2::TaskId trainingWorkflowTaskIdV2_;
    aitrain::v2::WorkflowRunId trainingWorkflowRunIdV2_;
    QString trainingWorkflowDeploymentSampleRelativePath_;
    aitrain::v2::TrainingWorkflowAdapterConfigV2 trainingWorkflowAdapterConfigV2_;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> runtimeDeliveryWorkspaceV2_;
    aitrain::v2::TaskId runtimeDeliveryTaskIdV2_;
    bool runtimeDeliveryRunningV2_ = false;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> annotationWorkspaceV2_;
    aitrain::v2::TaskId annotationTaskIdV2_;
    bool annotationRunningV2_ = false;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> ocrAcceptanceWorkspaceV2_;
    aitrain::v2::TaskId ocrAcceptanceTaskIdV2_;
    bool ocrAcceptanceRunningV2_ = false;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> dataQualityWorkspaceV2_;
    aitrain::v2::TaskId dataQualityTaskIdV2_;
    bool dataQualityRunningV2_ = false;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> datasetConversionWorkspaceV2_;
    aitrain::v2::TaskId datasetConversionTaskIdV2_;
    bool datasetConversionRunningV2_ = false;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> datasetSnapshotImportWorkspaceV2_;
    aitrain::v2::TaskId datasetSnapshotImportTaskIdV2_;
    bool datasetSnapshotImportRunningV2_ = false;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> datasetSplitWorkspaceV2_;
    aitrain::v2::TaskId datasetSplitTaskIdV2_;
    bool datasetSplitRunningV2_ = false;
    std::unique_ptr<aitrain::v2::ProjectWorkspaceV2> diagnosticsWorkspaceV2_;
    aitrain::v2::TaskId diagnosticsTaskIdV2_;
    bool diagnosticsRunningV2_ = false;
};
