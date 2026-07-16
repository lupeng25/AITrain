#pragma once

#include "aitrain/runtime/RuntimeAdapter.h"

#include <QJsonObject>
#include <QStringList>

namespace aitrain {

enum class RuntimeCapabilityStatus {
    Supported,
    UnsupportedByProduct,
    RuntimeNotImplemented,
    RequiresSdk,
    RequiresDependency,
    RequiresHardware,
    RequiresExternalEvidence
};

struct RuntimeCapabilityQuery final {
    QString modelFamily;
    QString runtimeRoute;
};

struct RuntimeCapability final {
    RuntimeCapabilityStatus status = RuntimeCapabilityStatus::UnsupportedByProduct;
    QString modelFamily;
    QString runtimeRoute;
    QString message;
    QStringList limitations;
    RuntimeStatus runtimeStatus = RuntimeStatus::RuntimeNotImplemented;

    QJsonObject toJson() const;
};

QString runtimeCapabilityStatusToString(RuntimeCapabilityStatus status);

class RuntimeCapabilityMatrix final {
public:
    RuntimeCapability query(const RuntimeCapabilityQuery& query) const;
    QJsonObject toJson() const;
};

} // namespace aitrain
