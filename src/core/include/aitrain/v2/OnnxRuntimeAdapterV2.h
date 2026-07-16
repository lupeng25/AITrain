#pragma once

#include "aitrain/v2/RuntimeAdapterV2.h"

namespace aitrain::v2 {

class OnnxRuntimeAdapterV2 final : public RuntimeAdapterV2 {
public:
    QString runtimeRoute() const override;
    RuntimeOperationResultV2 probe(const RuntimeModelLocationV2& model) const override;
    RuntimeOperationResultV2 validateModel(const RuntimeModelLocationV2& model) const override;
    RuntimeOperationResultV2 infer(const RuntimeModelLocationV2& model, const QJsonObject& request) const override;
    RuntimeOperationResultV2 benchmark(const RuntimeModelLocationV2& model, const QJsonObject& request) const override;
    RuntimeOperationResultV2 deploymentValidate(const RuntimeModelLocationV2& model, const QJsonObject& request) const override;
};

} // namespace aitrain::v2
