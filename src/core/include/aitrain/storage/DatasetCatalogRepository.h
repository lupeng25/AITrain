#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QString>

namespace aitrain {

class ProjectDatabase;
struct PageRequest;
template<class T> struct Page;
struct DatasetCatalogItem;
struct DatasetSnapshotRecord;

// 数据集目录与不可变 Snapshot 身份的 SQL 入口。
class DatasetCatalogRepository final {
public:
    explicit DatasetCatalogRepository(const ProjectDatabase& database);

    bool snapshot(const SnapshotId& snapshotId, DatasetSnapshotRecord* result,
        QString* error = nullptr) const;
    bool snapshotForArtifact(const ArtifactId& artifactId,
        DatasetSnapshotRecord* result, QString* error = nullptr) const;
    Page<DatasetCatalogItem> page(const PageRequest& request,
        QString* error = nullptr) const;

private:
    bool executeSnapshotQuery(const QString& whereColumn, const QString& id,
        const QString& notFoundMessage, DatasetSnapshotRecord* result,
        QString* error) const;

    const ProjectDatabase& database_;
};

} // namespace aitrain
