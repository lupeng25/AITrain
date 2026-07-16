#pragma once

#include "aitrain/runtime/RuntimeAdapter.h"
#include "aitrain/runtime/RuntimeCapabilityMatrix.h"
#include "aitrain/storage/ProjectStore.h"

namespace aitrain {

class ModelPackageRuntimeService final {
public:
    ModelPackageRuntimeService(const ProjectStore* storage, QString artifactStoreRoot);

    bool resolve(const ModelPackageId& modelPackageId,
        const QString& runtimeRoute,
        RuntimeModelLocation* location,
        RuntimeCapability* capability = nullptr,
        QString* error = nullptr) const;

private:
    const ProjectStore* storage_ = nullptr;
    QString artifactStoreRoot_;
    RuntimeCapabilityMatrix matrix_;
};

} // namespace aitrain
