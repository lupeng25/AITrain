#pragma once

#include "aitrain/domain/Pagination.h"

#include <QString>

namespace aitrain {

class ProjectDatabase;
struct PageRequest;
template<class T> struct Page;
struct DeliveryEvidenceCandidate;
struct ProjectSummarySnapshot;

// 跨聚合只读投影的 SQL 入口；不参与业务写事务。
class ProjectReadRepository final {
public:
    explicit ProjectReadRepository(const ProjectDatabase& database);

    bool summary(ProjectSummarySnapshot* result,
        QString* error = nullptr) const;
    Page<DeliveryEvidenceCandidate> deliveryEvidence(
        const PageRequest& request, QString* error = nullptr, const CatalogFilter& filter = {}) const;

private:
    const ProjectDatabase& database_;
};

} // namespace aitrain
