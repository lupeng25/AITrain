#pragma once

#include <QDateTime>
#include <QString>

namespace aitrain {

class ProjectDatabase;
struct ProjectMetaSnapshot;

// project_meta 单行聚合的唯一 SQL 访问点。Repository 共用
// ProjectDatabase，不创建连接，也不自行开启或提交事务。
class ProjectMetaRepository final {
public:
    explicit ProjectMetaRepository(const ProjectDatabase& database);

    bool read(ProjectMetaSnapshot* result, QString* error = nullptr) const;
    bool incrementOpenGeneration(const QDateTime& now,
        QString* error = nullptr);

private:
    const ProjectDatabase& database_;
};

} // namespace aitrain
