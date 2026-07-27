#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/artifact/VerifiedArtifactReader.h"
#include "aitrain/storage/ProjectStore.h"

#include <functional>

namespace aitrain {

enum class ArtifactCommitFailPoint {
    AfterDirectoryRenameBeforeDatabase,
    AfterDatabaseBeforeJournalRemoval
};

enum class ArtifactCommitPhase {
    None,
    Begun,
    Prepared,
    FilesCommitted,
    CatalogCommitted
};

enum class ArtifactCommitStatus {
    Committed,
    PendingRecovery,
    RejectedBeforeFilesCommitted
};

struct ArtifactCommitResult final {
    ArtifactCommitStatus status =
        ArtifactCommitStatus::RejectedBeforeFilesCommitted;
    bool cleanupPending = false;

    ArtifactCommitResult() = default;
    ArtifactCommitResult(ArtifactCommitStatus value, bool cleanup = false)
        : status(value), cleanupPending(cleanup) {}
    ArtifactCommitResult(bool committed)
        : status(committed ? ArtifactCommitStatus::Committed
                           : ArtifactCommitStatus::RejectedBeforeFilesCommitted) {}

    operator bool() const
    {
        return status == ArtifactCommitStatus::Committed;
    }
};

enum class ArtifactDiscardStatus {
    Discarded,
    CleanupPending,
    NotDiscardable,
    Failed
};

struct ArtifactDiscardResult final {
    ArtifactDiscardStatus status = ArtifactDiscardStatus::Failed;

    operator bool() const
    {
        return status == ArtifactDiscardStatus::Discarded
            || status == ArtifactDiscardStatus::CleanupPending;
    }
};

using ArtifactCommitFailureInjector = std::function<bool(ArtifactCommitFailPoint)>;

class ArtifactStore final {
public:
    explicit ArtifactStore(QString rootPath,
        ArtifactCommitFailureInjector failureInjector = {});

    bool begin(const TaskId& taskId, const QString& kind, ArtifactId* artifactId, QString* stagingPath, QString* error = nullptr);
    ArtifactCommitResult commit(const ArtifactId& artifactId,
        const TaskId& taskId,
        const QString& kind,
        const QString& stagingPath,
        ProjectStore* storage,
        QString* artifactPath,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {},
        bool* canceled = nullptr,
        const WorkflowRunId& workflowRunId = {},
        ArtifactCommitPhase* phase = nullptr);
    bool abort(const QString& stagingPath, QString* error = nullptr);
    ArtifactDiscardResult discardCommitted(
        const ArtifactId& artifactId, ProjectStore* storage, QString* error = nullptr);
    bool recoverStaging(ProjectStore* storage, QStringList* diagnostics, QString* error = nullptr);
    bool openVerified(const ArtifactSnapshot& artifact,
        VerifiedArtifactDirectory* result,
        ArtifactReadError* readError = nullptr,
        QString* error = nullptr) const;

    QString rootPath() const;

private:
    QString artifactPath(const ArtifactId& artifactId) const;

    QString rootPath_;
    ArtifactCommitFailureInjector failureInjector_;
};

} // namespace aitrain
