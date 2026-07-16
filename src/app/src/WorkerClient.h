#pragma once

#include "aitrain/protocol/Protocol.h"

#include <QLocalServer>
#include <QLocalSocket>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QTimer>

class WorkerClient : public QObject {
    Q_OBJECT

public:
    enum class WorkerTerminalStatus {
        Succeeded,
        Failed,
        Canceled,
    };
    Q_ENUM(WorkerTerminalStatus)

    explicit WorkerClient(QObject* parent = nullptr);
    ~WorkerClient() override;

    bool requestEnvironmentCheckWorkflow(const QString& workerProgram,
        const QString& projectRoot, QString* error, const QString& taskId = {});
    bool requestDatasetSplitWorkflow(const QString& workerProgram,
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
    bool requestDatasetConversionWorkflow(const QString& workerProgram,
        const QString& projectRoot,
        const QString& sourcePath,
        const QString& sourceFormat,
        const QString& targetFormat,
        const QString& targetDatasetId,
        const QString& targetDatasetName,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestDataQualityWorkflow(const QString& workerProgram,
        const QString& projectRoot,
        const QString& datasetId,
        const QString& datasetVersionId,
        const QString& snapshotId,
        const QString& snapshotArtifactId,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestAnnotationSessionCreate(const QString& workerProgram,
        const QString& projectRoot,
        const QString& repairManifestArtifactId,
        const QString& workingDirectory,
        const QJsonObject& toolSummary,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestAnnotationSessionSync(const QString& workerProgram,
        const QString& projectRoot,
        const QString& sessionArtifactId,
        const QString& workingDirectory,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestDatasetSnapshotImportWorkflow(const QString& workerProgram,
        const QString& projectRoot,
        const QString& sourcePath,
        const QString& sourceFormat,
        const QString& targetDatasetId,
        const QString& targetDatasetName,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestOcrOfficialReportImport(const QString& workerProgram,
        const QString& projectRoot,
        const QJsonObject& det,
        const QJsonObject& rec,
        const QJsonObject& system,
        const QString& acceptanceCohortId,
        const QString& customerDomainId,
        const QString& evidenceClass,
        QString* error,
        const QString& taskId = {});
    bool requestOcrAcceptanceWorkflow(const QString& workerProgram,
        const QString& projectRoot,
        const QString& detReportArtifactId,
        const QString& recReportArtifactId,
        const QString& systemReportArtifactId,
        const QJsonObject& thresholds,
        QString* error,
        const QString& taskId = {});
    bool requestDiagnosticsWorkflow(const QString& workerProgram,
        const QString& projectRoot,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestRuntimeDeliveryWorkflow(const QString& workerProgram,
        const QString& projectRoot,
        const QString& modelPackageId,
        const QString& runtimeRoute,
        const QString& sampleImagePath,
        const QJsonObject& options,
        QString* error,
        const QString& taskId = {});
    bool requestModelImport(const QString& workerProgram, const QString& projectRoot, const QString& sourceFilePath, const QJsonObject& manifestDraft, QString* error, const QString& taskId = {});
    // 完整  训练工作流由 Worker 独占持久化和步骤收口；GUI 只传递已校验的请求并展示事件。
    bool requestTrainingWorkflow(const QString& workerProgram, const QJsonObject& request, QString* error);
    void cancel();
    bool isRunning() const;

signals:
    void connected();
    void messageReceived(const QString& type, const QJsonObject& payload);
    void logLine(const QString& line);
    void finished(WorkerTerminalStatus status, const QString& message);
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
    bool sendEnvelope(const aitrain::ProtocolEnvelope& envelope, QString* error = nullptr);
    void rejectProtocol(const QString& message);
    void cleanupSocket();

    QLocalServer server_;
    QLocalSocket* socket_ = nullptr;
    QProcess process_;
    QByteArray buffer_;
    QString pendingCommandType_;
    QJsonObject pendingRequest_;
    aitrain::RequestId activeRequestId_;
    aitrain::TaskId activeTaskId_;
    aitrain::ProtocolSequenceTracker incomingSequenceTracker_;
    quint64 outgoingSequence_ = 0;
    bool finishedEmitted_ = false;
    bool startTaskSent_ = false;
    bool terminalEnvelopeReceived_ = false;
    QTimer cancelTimer_;
    QTimer connectionTimer_;
    QTimer terminalShutdownTimer_;
    bool cancelRequested_ = false;
    bool workerReady_ = false;
    bool finishing_ = false;
    int pendingExitCode_ = 0;
    QProcess::ExitStatus pendingExitStatus_ = QProcess::NormalExit;
};

Q_DECLARE_METATYPE(WorkerClient::WorkerTerminalStatus)
