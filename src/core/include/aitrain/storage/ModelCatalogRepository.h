#pragma once

#include "aitrain/domain/Pagination.h"

#include "aitrain/domain/DomainTypes.h"

#include <QString>

namespace aitrain {

class ProjectDatabase;
struct PageRequest;
template<class T> struct Page;
struct ModelPackageSnapshot;

// 模型包 Manifest、来源 Artifact 与 Snapshot binding 的 SQL 入口。
class ModelCatalogRepository final {
public:
    explicit ModelCatalogRepository(const ProjectDatabase& database);

    bool read(const ModelPackageId& modelPackageId,
        ModelPackageSnapshot* result, QString* error = nullptr) const;
    Page<ModelPackageSnapshot> page(const PageRequest& request,
        QString* error = nullptr, const CatalogFilter& filter = {}) const;

private:
    const ProjectDatabase& database_;
};

} // namespace aitrain
