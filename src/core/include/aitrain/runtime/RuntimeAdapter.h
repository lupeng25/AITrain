#pragma once

#include "aitrain/model/ModelManifest.h"

#include <QJsonObject>

namespace aitrain {

enum class RuntimeStatus {
    Available,
    RuntimeNotImplemented,
    SdkMissing,
    DependencyMissing,
    HardwareUnsupported,
    ArtifactIncompatible
};

struct RuntimeModelLocation final {
    ModelManifest manifest;
    QString artifactDirectory;
};

struct RuntimeOperationResult final {
    RuntimeStatus status = RuntimeStatus::RuntimeNotImplemented;
    QString message;
    QJsonObject details;
};

class RuntimeAdapter {
public:
    virtual ~RuntimeAdapter() = default;

    virtual QString runtimeRoute() const = 0;
    virtual RuntimeOperationResult probe(const RuntimeModelLocation& model) const = 0;
    virtual RuntimeOperationResult validateModel(const RuntimeModelLocation& model) const = 0;
    RuntimeOperationResult load(const RuntimeModelLocation& model) const { return validateModel(model); }
    virtual RuntimeOperationResult infer(const RuntimeModelLocation& model, const QJsonObject& request) const = 0;
    virtual RuntimeOperationResult deploymentValidate(const RuntimeModelLocation& model, const QJsonObject& request) const = 0;
};

QString runtimeStatusToString(RuntimeStatus status);
RuntimeOperationResult validateRuntimeModel(const RuntimeModelLocation& model, const QString& runtimeRoute);

} // namespace aitrain
