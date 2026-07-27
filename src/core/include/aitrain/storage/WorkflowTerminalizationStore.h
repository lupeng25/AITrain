#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QString>
#include <QVector>

namespace aitrain {

class ProjectDatabase;
struct WorkflowTerminalizationSnapshot;
struct WorkflowRunSnapshot;

// Evidence 门控终态三态记录的 SQL 入口。attach/close 等跨表事务仍由
// ProjectStore Unit of Work 负责。
class WorkflowTerminalizationStore final {
public:
    explicit WorkflowTerminalizationStore(const ProjectDatabase& database);

    bool exists(const WorkflowRunId& workflowRunId, bool* result,
        QString* error = nullptr) const;
    bool read(const WorkflowRunId& workflowRunId,
        WorkflowTerminalizationSnapshot* result,
        QString* error = nullptr) const;
    QVector<WorkflowTerminalizationSnapshot> pending(
        int limit, QString* error = nullptr) const;
    QVector<WorkflowRunSnapshot> pendingEvidenceRequired(
        int limit, QString* error = nullptr) const;

private:
    const ProjectDatabase& database_;
};

} // namespace aitrain
