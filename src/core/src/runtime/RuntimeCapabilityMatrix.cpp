#include "aitrain/runtime/RuntimeCapabilityMatrix.h"

#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/product/ProductCapabilityContract.h"

#include <QJsonArray>

#include <utility>

namespace aitrain {
namespace {

QString normalized(const QString& value)
{
    return value.trimmed().toLower();
}

RuntimeCapability result(const RuntimeRouteContract* contract,
    RuntimeCapabilityStatus status,
    const QString& family,
    const QString& route,
    const QString& message,
    const QStringList& limitations = {})
{
    RuntimeStatus runtimeStatus = RuntimeStatus::RuntimeNotImplemented;
    RuntimeLocalReadiness localReadiness = RuntimeLocalReadiness::NotApplicable;
    if (status == RuntimeCapabilityStatus::Supported) {
        runtimeStatus = RuntimeStatus::Available;
        localReadiness = RuntimeLocalReadiness::Available;
    } else if (status == RuntimeCapabilityStatus::RequiresSdk) {
        runtimeStatus = RuntimeStatus::SdkMissing;
        localReadiness = RuntimeLocalReadiness::SdkMissing;
    } else if (status == RuntimeCapabilityStatus::RequiresDependency) {
        runtimeStatus = RuntimeStatus::DependencyMissing;
        localReadiness = RuntimeLocalReadiness::DependencyMissing;
    } else if (status == RuntimeCapabilityStatus::RequiresHardware) {
        runtimeStatus = RuntimeStatus::HardwareUnsupported;
        localReadiness = RuntimeLocalReadiness::HardwareUnsupported;
    } else if (status == RuntimeCapabilityStatus::RequiresExternalEvidence) {
        localReadiness = RuntimeLocalReadiness::ExternalEvidenceRequired;
    }
    RuntimeCapability value;
    value.status = status;
    value.executionAuthority = contract
        ? contract->executionAuthority
        : RuntimeExecutionAuthority::AitrainCpp;
    value.productState = contract
        ? contract->productState
        : RuntimeProductState::UnsupportedByProduct;
    value.localReadiness = localReadiness;
    value.modelFamily = family;
    value.runtimeRoute = route;
    value.message = message;
    value.limitations = limitations;
    value.runtimeStatus = runtimeStatus;
    return value;
}

} // namespace

QString runtimeCapabilityStatusToString(RuntimeCapabilityStatus status)
{
    switch (status) {
    case RuntimeCapabilityStatus::Supported: return QStringLiteral("supported");
    case RuntimeCapabilityStatus::UnsupportedByProduct: return QStringLiteral("unsupported_by_product");
    case RuntimeCapabilityStatus::RuntimeNotImplemented: return QStringLiteral("runtime_not_implemented");
    case RuntimeCapabilityStatus::RequiresSdk: return QStringLiteral("requires_sdk");
    case RuntimeCapabilityStatus::RequiresDependency: return QStringLiteral("requires_dependency");
    case RuntimeCapabilityStatus::RequiresHardware: return QStringLiteral("requires_hardware");
    case RuntimeCapabilityStatus::RequiresExternalEvidence: return QStringLiteral("requires_external_evidence");
    }
    return QStringLiteral("unsupported_by_product");
}

QJsonObject RuntimeCapability::toJson() const
{
    return {{QStringLiteral("status"), runtimeCapabilityStatusToString(status)},
        {QStringLiteral("executionAuthority"), runtimeExecutionAuthorityToString(executionAuthority)},
        {QStringLiteral("productState"), runtimeProductStateToString(productState)},
        {QStringLiteral("localReadiness"), runtimeLocalReadinessToString(localReadiness)},
        {QStringLiteral("modelFamily"), modelFamily}, {QStringLiteral("runtimeRoute"), runtimeRoute},
        {QStringLiteral("message"), message}, {QStringLiteral("limitations"), QJsonArray::fromStringList(limitations)},
        {QStringLiteral("runtimeStatus"), runtimeStatusToString(runtimeStatus)}};
}

EnvironmentSnapshot EnvironmentSnapshot::capture()
{
    EnvironmentSnapshot snapshot;
    const bool onnxAvailable = isOnnxRuntimeInferenceAvailable();
    snapshot.runtimeReadiness.append({
        QStringLiteral("aitrain_onnxruntime"),
        onnxAvailable
            ? RuntimeLocalReadiness::Available
            : RuntimeLocalReadiness::DependencyMissing,
        onnxAvailable
            ? QStringLiteral("ONNX Runtime 依赖可用。")
            : QStringLiteral("当前构建缺少 ONNX Runtime 依赖。")});
    const NcnnBackendStatus ncnn = ncnnBackendStatus();
    snapshot.runtimeReadiness.append({
        QStringLiteral("aitrain_ncnn"),
        !ncnn.sdkAvailable ? RuntimeLocalReadiness::SdkMissing
            : (ncnn.inferenceAvailable ? RuntimeLocalReadiness::Available
                                       : RuntimeLocalReadiness::NotApplicable),
        ncnn.message});
    return snapshot;
}

RuntimeReadinessSnapshot EnvironmentSnapshot::readinessFor(
    const QString& runtimeRoute) const
{
    const QString route = normalized(runtimeRoute);
    for (const RuntimeReadinessSnapshot& readiness : runtimeReadiness) {
        if (normalized(readiness.runtimeRoute) == route) return readiness;
    }
    return {route, RuntimeLocalReadiness::NotApplicable,
        QStringLiteral("该 Runtime 路线不适用本机执行探测。")};
}

RuntimeCapabilityMatrix::RuntimeCapabilityMatrix(EnvironmentSnapshot environment)
    : environment_(std::move(environment))
{
}

RuntimeCapability RuntimeCapabilityMatrix::query(const RuntimeCapabilityQuery& request) const
{
    const QString family = normalized(request.modelFamily);
    const QString route = normalized(request.runtimeRoute);
    if (family.isEmpty() || route.isEmpty()) {
        return result(nullptr, RuntimeCapabilityStatus::UnsupportedByProduct, family, route,
            QStringLiteral("Runtime 矩阵查询需要模型族与运行时路由。"));
    }
    RuntimeRouteContract contract;
    if (!ProductCapabilityContract::instance().resolveRuntimeRoute(family, route, &contract)) {
        return result(nullptr, RuntimeCapabilityStatus::UnsupportedByProduct, family, route,
            QStringLiteral("未知或未注册的 Runtime 路由。"));
    }
    if (contract.productState == RuntimeProductState::NotImplemented) {
        return result(&contract, RuntimeCapabilityStatus::RuntimeNotImplemented, family, route,
            contract.limitations.isEmpty()
                ? QStringLiteral("该产品 Runtime 路线尚未实现。")
                : contract.limitations.constFirst(),
            contract.limitations);
    }
    if (contract.productState == RuntimeProductState::UnsupportedByProduct) {
        return result(&contract, RuntimeCapabilityStatus::UnsupportedByProduct, family, route,
            QStringLiteral("该模型族与 Runtime 组合不在产品支持范围内。"),
            contract.limitations);
    }
    if (contract.executionAuthority == RuntimeExecutionAuthority::WorkerManaged) {
        return result(&contract, RuntimeCapabilityStatus::RequiresExternalEvidence, family, route,
            QStringLiteral("该路线由 Worker 管理的官方 Python Runtime 执行，不属于 AITrain C++ Runtime Delivery。"),
            contract.limitations);
    }
    if (contract.executionAuthority == RuntimeExecutionAuthority::OfficialEvidence) {
        return result(&contract, RuntimeCapabilityStatus::RequiresExternalEvidence, family, route,
            QStringLiteral("该路线仅通过官方工具链报告和验收证据收口。"),
            contract.limitations);
    }
    if (route == QStringLiteral("aitrain_onnxruntime")) {
        const RuntimeReadinessSnapshot readiness = environment_.readinessFor(route);
        if (readiness.readiness != RuntimeLocalReadiness::Available) {
            return result(&contract, RuntimeCapabilityStatus::RequiresDependency, family, route,
                readiness.message, contract.limitations);
        }
        return result(&contract, RuntimeCapabilityStatus::Supported, family, route,
            QStringLiteral("由 Manifest 指定 decoder 的 AITrain ONNX Runtime 路线可用。"),
            contract.limitations);
    }
    if (route == QStringLiteral("aitrain_ncnn")) {
        const RuntimeReadinessSnapshot readiness = environment_.readinessFor(route);
        if (readiness.readiness == RuntimeLocalReadiness::SdkMissing) {
            return result(&contract, RuntimeCapabilityStatus::RequiresSdk, family, route,
                readiness.message, contract.limitations);
        }
        if (readiness.readiness != RuntimeLocalReadiness::Available) {
            return result(&contract, RuntimeCapabilityStatus::RuntimeNotImplemented, family, route,
                readiness.message, contract.limitations);
        }
        return result(&contract, RuntimeCapabilityStatus::Supported, family, route,
            QStringLiteral("NCNN 运行时可用。"), contract.limitations);
    }
    return result(&contract, RuntimeCapabilityStatus::RuntimeNotImplemented, family, route,
        QStringLiteral("产品合同已登记该路线，但 AITrain C++ Runtime Adapter 尚未实现。"),
        contract.limitations);
}

QJsonObject RuntimeCapabilityMatrix::toJson() const
{
    QJsonArray entries;
    for (const RuntimeRouteContract& route :
        ProductCapabilityContract::instance().runtimeRoutes()) {
        entries.append(query({route.modelFamily, route.routeId}).toJson());
    }
    return {{QStringLiteral("entries"), entries}};
}

} // namespace aitrain
