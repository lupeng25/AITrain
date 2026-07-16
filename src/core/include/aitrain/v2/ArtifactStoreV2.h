#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/v2/StorageV2.h"

#include <functional>

namespace aitrain::v2 {

enum class ArtifactCommitFailPointV2 {
    AfterDirectoryRenameBeforeDatabase,
    AfterDatabaseBeforeJournalRemoval
};

using ArtifactCommitFailureInjectorV2 = std::function<bool(ArtifactCommitFailPointV2)>;

class ArtifactStoreV2 final {
public:
    explicit ArtifactStoreV2(QString rootPath,
        ArtifactCommitFailureInjectorV2 failureInjector = {});

    bool begin(const TaskId& taskId, const QString& kind, ArtifactId* artifactId, QString* stagingPath, QString* error = nullptr);
    bool commit(const ArtifactId& artifactId,
        const TaskId& taskId,
        const QString& kind,
        const QString& stagingPath,
        StorageV2* storage,
        QString* artifactPath,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {},
        bool* canceled = nullptr,
        const WorkflowRunId& workflowRunId = {});
    bool abort(const QString& stagingPath, QString* error = nullptr);
    bool discardCommitted(const ArtifactId& artifactId, StorageV2* storage, QString* error = nullptr);
    bool recoverStaging(StorageV2* storage, QStringList* diagnostics, QString* error = nullptr);

    QString rootPath() const;
    QString artifactPath(const ArtifactId& artifactId) const;

private:
    QString rootPath_;
    ArtifactCommitFailureInjectorV2 failureInjector_;
};

} // namespace aitrain::v2
