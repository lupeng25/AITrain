#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QString>

namespace aitrain {

class ProjectDatabase;
struct PageRequest;
template<class T> struct Page;
struct MetricSnapshot;
struct TaskSnapshot;

// 任务聚合与其事件流的 SQL 入口。当前先承接任务身份读取；事件写入和
// 终态跨表事务仍由 ProjectStore Unit of Work 协调。
class TaskEventRepository final {
public:
    explicit TaskEventRepository(const ProjectDatabase& database);

    bool exists(const TaskId& taskId, bool* result,
        QString* error = nullptr) const;
    bool read(const TaskId& taskId, TaskSnapshot* result,
        QString* error = nullptr) const;
    int eventCount(const TaskId& taskId, QString* error = nullptr) const;
    int metricCount(const TaskId& taskId, QString* error = nullptr) const;
    Page<TaskSnapshot> page(
        const PageRequest& request, QString* error = nullptr) const;
    Page<MetricSnapshot> metrics(const TaskId& taskId,
        const PageRequest& request, QString* error = nullptr) const;

private:
    int countForTask(const QString& table, const TaskId& taskId,
        QString* error) const;

    const ProjectDatabase& database_;
};

} // namespace aitrain
