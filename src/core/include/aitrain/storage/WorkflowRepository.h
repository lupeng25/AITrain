#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QString>
#include <QVector>

namespace aitrain {

class ProjectDatabase;
struct PageRequest;
template<class T> struct Page;
struct WorkflowRunSnapshot;
struct WorkflowInputBinding;
struct WorkflowStepSnapshot;

// Workflow Run/Step 模板执行事实的 SQL 入口。
class WorkflowRepository final {
public:
    explicit WorkflowRepository(const ProjectDatabase& database);

    bool readRun(const WorkflowRunId& workflowRunId,
        WorkflowRunSnapshot* result, QString* error = nullptr) const;
    bool readInput(const WorkflowRunId& workflowRunId, const QString& role,
        WorkflowInputBinding* result, QString* error = nullptr) const;
    QVector<WorkflowStepSnapshot> steps(const WorkflowRunId& workflowRunId,
        QString* error = nullptr) const;
    Page<WorkflowRunSnapshot> runsForTask(const TaskId& taskId,
        const PageRequest& request, QString* error = nullptr) const;

private:
    const ProjectDatabase& database_;
};

} // namespace aitrain
