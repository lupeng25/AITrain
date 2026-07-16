#pragma once

#include "aitrain/v2/ProtocolV2.h"

#include <QLocalServer>
#include <QLocalSocket>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QTimer>

class WorkerClient : public QObject {
    Q_OBJECT

public:
    explicit WorkerClient(QObject* parent = nullptr);
    ~WorkerClient() override;

    bool requestEnvironmentCheckWorkflowV2(const QString& workerProgram,
        const QString& projectRoot, QString* error, const QString& taskId = {});
    bool requestDatasetSplitWorkflowV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& sourceDatasetId,
        const QString& sourceDatasetVersionId,
        const QString& sourceSnapshotId,
        const QString& sourceSnapshotArtifactId,
        const QString& targetDatasetId,
        const QString& targetDatasetName,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestDatasetConversionWorkflowV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& sourcePath,
        const QString& sourceFormat,
        const QString& targetFormat,
        const QString& targetDatasetId,
        const QString& targetDatasetName,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestDataQualityWorkflowV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& datasetId,
        const QString& datasetVersionId,
        const QString& snapshotId,
        const QString& snapshotArtifactId,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestAnnotationSessionCreateV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& repairManifestArtifactId,
        const QString& workingDirectory,
        const QJsonObject& toolSummary,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestAnnotationSessionSyncV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& sessionArtifactId,
        const QString& workingDirectory,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestDatasetSnapshotImportWorkflowV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& sourcePath,
        const QString& sourceFormat,
        const QString& targetDatasetId,
        const QString& targetDatasetName,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestOcrOfficialReportImportV2(const QString& workerProgram,
        const QString& projectRoot,
        const QJsonObject& det,
        const QJsonObject& rec,
        const QJsonObject& system,
        const QString& acceptanceCohortId,
        const QString& customerDomainId,
        const QString& evidenceClass,
        QString* error,
        const QString& taskId = {});
    bool requestOcrAcceptanceWorkflowV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& detReportArtifactId,
        const QString& recReportArtifactId,
        const QString& systemReportArtifactId,
        const QJsonObject& thresholds,
        QString* error,
        const QString& taskId = {});
    bool requestDiagnosticsWorkflowV2(const QString& workerProgram,
        const QString& projectRoot,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestRuntimeDeliveryWorkflowV2(const QString& workerProgram,
        const QString& projectRoot,
        const QString& modelPackageId,
        const QString& runtimeRoute,
        const QString& sampleImagePath,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestModelImportV2(const QString& workerProgram, const QString& projectRoot, const QString& sourceFilePath, const QJsonObject& manifestDraft, QString* error, const QString& taskId = {});
    // 完整 V2 训练工作流由 Worker 独占持久化和步骤收口；GUI 只传递已校验的请求并展示事件。
    bool requestTrainingWorkflowV2(const QString& workerProgram, const QJsonObject& request, QString* error);
    void cancel();
    bool isRunning() const;

signals:
    void connected();
    void messageReceived(const QString& type, const QJsonObject& payload);
    void logLine(const QString& line);
    void finished(bool ok, const QString& message);
    void idle();

private slots:
    void acceptConnection();
    void readLines();
    void workerFinished(int exitCode, QProcess::ExitStatus status);
    void workerProcessError(QProcess::ProcessError error);

private:
    bool startWorkerCommand(const QString& workerProgram, const QString& commandType, const QJsonObject& payload, QString* error);
    void finalizeWorkerExit();
    void sendStartTask();
    void sendCancelTask();
    bool sendEnvelope(const aitrain::v2::ProtocolEnvelope& envelope, QString* error = nullptr);
    void rejectProtocol(const QString& message);
    void cleanupSocket();

    QLocalServer server_;
    QLocalSocket* socket_ = nullptr;
    QProcess process_;
    QByteArray buffer_;
    QString pendingCommandType_;
    QJsonObject pendingRequest_;
    aitrain::v2::RequestId activeRequestId_;
    aitrain::v2::TaskId activeTaskId_;
    aitrain::v2::ProtocolV2SequenceTracker incomingSequenceTracker_;
    quint64 outgoingSequence_ = 0;
    bool finishedEmitted_ = false;
    bool startTaskSent_ = false;
    bool terminalEnvelopeReceived_ = false;
    QTimer cancelTimer_;
    QTimer terminalShutdownTimer_;
    bool cancelRequested_ = false;
    bool finishing_ = false;
    int pendingExitCode_ = 0;
    QProcess::ExitStatus pendingExitStatus_ = QProcess::NormalExit;
};
