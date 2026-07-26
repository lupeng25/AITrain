#pragma once

#include "aitrain/product/ProductCapabilityContract.h"
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
    RuntimeExecutionAuthority executionAuthority = RuntimeExecutionAuthority::AitrainCpp;
    RuntimeProductState productState = RuntimeProductState::UnsupportedByProduct;
    RuntimeLocalReadiness localReadiness = RuntimeLocalReadiness::NotApplicable;
    QString modelFamily;
    QString runtimeRoute;
    QString message;
    QStringList limitations;
    RuntimeStatus runtimeStatus = RuntimeStatus::RuntimeNotImplemented;

    QJsonObject toJson() const;
};

struct RuntimeReadinessSnapshot final {
    QString runtimeRoute;
    RuntimeLocalReadiness readiness = RuntimeLocalReadiness::NotApplicable;
    QString message;
};

struct EnvironmentSnapshot final {
    QVector<RuntimeReadinessSnapshot> runtimeReadiness;

    static EnvironmentSnapshot capture();
    RuntimeReadinessSnapshot readinessFor(const QString& runtimeRoute) const;
};

QString runtimeCapabilityStatusToString(RuntimeCapabilityStatus status);

class RuntimeCapabilityMatrix final {
public:
    explicit RuntimeCapabilityMatrix(
        EnvironmentSnapshot environment = EnvironmentSnapshot::capture());
    RuntimeCapability query(const RuntimeCapabilityQuery& query) const;
    QJsonObject toJson() const;

private:
    EnvironmentSnapshot environment_;
};

} // namespace aitrain
