#pragma once

#include "aitrain/v2/ModelManifestV2.h"

#include <QJsonObject>

namespace aitrain::v2 {

enum class RuntimeStatusV2 {
    Available,
    RuntimeNotImplemented,
    SdkMissing,
    DependencyMissing,
    HardwareUnsupported,
    ArtifactIncompatible
};

struct RuntimeModelLocationV2 final {
    ModelManifestV2 manifest;
    QString artifactDirectory;
};

struct RuntimeOperationResultV2 final {
    RuntimeStatusV2 status = RuntimeStatusV2::RuntimeNotImplemented;
    QString message;
    QJsonObject details;
};

class RuntimeAdapterV2 {
public:
    virtual ~RuntimeAdapterV2() = default;

    virtual QString runtimeRoute() const = 0;
    virtual RuntimeOperationResultV2 probe(const RuntimeModelLocationV2& model) const = 0;
    virtual RuntimeOperationResultV2 validateModel(const RuntimeModelLocationV2& model) const = 0;
    RuntimeOperationResultV2 load(const RuntimeModelLocationV2& model) const { return validateModel(model); }
    virtual RuntimeOperationResultV2 infer(const RuntimeModelLocationV2& model, const QJsonObject& request) const = 0;
    virtual RuntimeOperationResultV2 benchmark(const RuntimeModelLocationV2& model, const QJsonObject& request) const = 0;
    virtual RuntimeOperationResultV2 deploymentValidate(const RuntimeModelLocationV2& model, const QJsonObject& request) const = 0;
};

QString runtimeStatusV2ToString(RuntimeStatusV2 status);
RuntimeOperationResultV2 validateRuntimeModelV2(const RuntimeModelLocationV2& model, const QString& runtimeRoute);

} // namespace aitrain::v2
