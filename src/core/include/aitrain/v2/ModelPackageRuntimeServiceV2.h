#pragma once

#include "aitrain/v2/RuntimeAdapterV2.h"
#include "aitrain/v2/RuntimeCapabilityMatrixV2.h"
#include "aitrain/v2/StorageV2.h"

namespace aitrain::v2 {

class ModelPackageRuntimeServiceV2 final {
public:
    ModelPackageRuntimeServiceV2(const StorageV2* storage, QString artifactStoreRoot);

    bool resolve(const ModelPackageId& modelPackageId,
        const QString& runtimeRoute,
        RuntimeModelLocationV2* location,
        RuntimeCapabilityV2* capability = nullptr,
        QString* error = nullptr) const;

private:
    const StorageV2* storage_ = nullptr;
    QString artifactStoreRoot_;
    RuntimeCapabilityMatrixV2 matrix_;
};

} // namespace aitrain::v2
