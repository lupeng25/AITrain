#pragma once

#include "aitrain/runtime/RuntimeAdapter.h"

namespace aitrain {

class NcnnRuntimeAdapter final : public RuntimeAdapter {
public:
    QString runtimeRoute() const override;
    RuntimeOperationResult probe(const RuntimeModelLocation& model) const override;
    RuntimeOperationResult validateModel(const RuntimeModelLocation& model) const override;
    RuntimeOperationResult infer(const RuntimeModelLocation& model, const QJsonObject& request) const override;
    RuntimeOperationResult benchmark(const RuntimeModelLocation& model, const QJsonObject& request) const override;
    RuntimeOperationResult deploymentValidate(const RuntimeModelLocation& model, const QJsonObject& request) const override;
};

} // namespace aitrain
