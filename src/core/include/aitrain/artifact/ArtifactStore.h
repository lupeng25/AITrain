#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/storage/ProjectStore.h"

#include <functional>

namespace aitrain {

enum class ArtifactCommitFailPoint {
    AfterDirectoryRenameBeforeDatabase,
    AfterDatabaseBeforeJournalRemoval
};

using ArtifactCommitFailureInjector = std::function<bool(ArtifactCommitFailPoint)>;

class ArtifactStore final {
public:
    explicit ArtifactStore(QString rootPath,
        ArtifactCommitFailureInjector failureInjector = {});

    bool begin(const TaskId& taskId, const QString& kind, ArtifactId* artifactId, QString* stagingPath, QString* error = nullptr);
    bool commit(const ArtifactId& artifactId,
        const TaskId& taskId,
        const QString& kind,
        const QString& stagingPath,
        ProjectStore* storage,
        QString* artifactPath,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {},
        bool* canceled = nullptr,
        const WorkflowRunId& workflowRunId = {});
    bool abort(const QString& stagingPath, QString* error = nullptr);
    bool discardCommitted(const ArtifactId& artifactId, ProjectStore* storage, QString* error = nullptr);
    bool recoverStaging(ProjectStore* storage, QStringList* diagnostics, QString* error = nullptr);

    QString rootPath() const;
    QString artifactPath(const ArtifactId& artifactId) const;

private:
    QString rootPath_;
    ArtifactCommitFailureInjector failureInjector_;
};

} // namespace aitrain
