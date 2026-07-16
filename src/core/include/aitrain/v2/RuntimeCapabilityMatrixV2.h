#pragma once

#include "aitrain/v2/RuntimeAdapterV2.h"

#include <QJsonObject>
#include <QStringList>

namespace aitrain::v2 {

enum class RuntimeCapabilityStatusV2 {
    Supported,
    UnsupportedByProduct,
    RuntimeNotImplemented,
    RequiresSdk,
    RequiresDependency,
    RequiresHardware,
    RequiresExternalEvidence
};

struct RuntimeCapabilityQueryV2 final {
    QString modelFamily;
    QString runtimeRoute;
};

struct RuntimeCapabilityV2 final {
    RuntimeCapabilityStatusV2 status = RuntimeCapabilityStatusV2::UnsupportedByProduct;
    QString modelFamily;
    QString runtimeRoute;
    QString message;
    QStringList limitations;
    RuntimeStatusV2 runtimeStatus = RuntimeStatusV2::RuntimeNotImplemented;

    QJsonObject toJson() const;
};

QString runtimeCapabilityStatusV2ToString(RuntimeCapabilityStatusV2 status);

class RuntimeCapabilityMatrixV2 final {
public:
    RuntimeCapabilityV2 query(const RuntimeCapabilityQueryV2& query) const;
    QJsonObject toJson() const;
};

} // namespace aitrain::v2
