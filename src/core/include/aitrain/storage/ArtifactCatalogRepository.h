#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QString>
#include <QStringList>

namespace aitrain {

class ProjectDatabase;
struct PageRequest;
template<class T> struct Page;
struct ArtifactFileSnapshot;
struct ArtifactSnapshot;

// Artifact catalog 与文件 inventory 的 SQL 入口。物理文件校验仍由
// VerifiedArtifactReader/ArtifactStore 负责。
class ArtifactCatalogRepository final {
public:
    explicit ArtifactCatalogRepository(const ProjectDatabase& database);

    bool exists(const ArtifactId& artifactId, bool* result,
        QString* error = nullptr) const;
    bool read(const ArtifactId& artifactId, ArtifactSnapshot* result,
        QString* error = nullptr) const;
    bool isDiscardable(const ArtifactId& artifactId, bool* result,
        QString* error = nullptr) const;
    int countForTask(const TaskId& taskId, QString* error = nullptr) const;
    int fileCount(const ArtifactId& artifactId,
        QString* error = nullptr) const;
    Page<ArtifactFileSnapshot> files(const ArtifactId& artifactId,
        const PageRequest& request, QString* error = nullptr) const;
    Page<ArtifactSnapshot> forTask(const TaskId& taskId,
        const PageRequest& request, QString* error = nullptr) const;
    Page<ArtifactSnapshot> catalog(const QStringList& kinds,
        const PageRequest& request, QString* error = nullptr) const;

private:
    const ProjectDatabase& database_;
};

} // namespace aitrain
