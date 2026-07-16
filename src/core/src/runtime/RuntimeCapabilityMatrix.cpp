#include "aitrain/runtime/RuntimeCapabilityMatrix.h"

#include "aitrain/core/VisionModelRuntime.h"

#include <QJsonArray>

namespace aitrain {
namespace {

QString normalized(const QString& value)
{
    return value.trimmed().toLower();
}

RuntimeCapability result(RuntimeCapabilityStatus status, const QString& family, const QString& route,
    const QString& message, const QStringList& limitations = {})
{
    RuntimeStatus runtimeStatus = RuntimeStatus::RuntimeNotImplemented;
    if (status == RuntimeCapabilityStatus::Supported
        || status == RuntimeCapabilityStatus::RequiresExternalEvidence) {
        runtimeStatus = RuntimeStatus::Available;
    } else if (status == RuntimeCapabilityStatus::RequiresSdk) {
        runtimeStatus = RuntimeStatus::SdkMissing;
    } else if (status == RuntimeCapabilityStatus::RequiresDependency) {
        runtimeStatus = RuntimeStatus::DependencyMissing;
    } else if (status == RuntimeCapabilityStatus::RequiresHardware) {
        runtimeStatus = RuntimeStatus::HardwareUnsupported;
    }
    return {status, family, route, message, limitations, runtimeStatus};
}

bool onnxFamily(const QString& family)
{
    return family == QStringLiteral("yolo_detection") || family == QStringLiteral("yolo_segmentation")
        || family == QStringLiteral("yolo_obb") || family == QStringLiteral("semantic_segmentation");
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
        {QStringLiteral("modelFamily"), modelFamily}, {QStringLiteral("runtimeRoute"), runtimeRoute},
        {QStringLiteral("message"), message}, {QStringLiteral("limitations"), QJsonArray::fromStringList(limitations)},
        {QStringLiteral("runtimeStatus"), runtimeStatusToString(runtimeStatus)}};
}

RuntimeCapability RuntimeCapabilityMatrix::query(const RuntimeCapabilityQuery& request) const
{
    const QString family = normalized(request.modelFamily);
    const QString route = normalized(request.runtimeRoute);
    if (family.isEmpty() || route.isEmpty()) {
        return result(RuntimeCapabilityStatus::UnsupportedByProduct, family, route, QStringLiteral("Runtime 矩阵查询需要模型族与运行时路由。"));
    }
    if (route == QStringLiteral("aitrain_onnxruntime")) {
        if (!onnxFamily(family)) {
            return result(RuntimeCapabilityStatus::UnsupportedByProduct, family, route,
                QStringLiteral("该模型族不在 AITrain ONNX Runtime 产品部署范围内。"));
        }
        if (!aitrain::isOnnxRuntimeInferenceAvailable()) {
            return result(RuntimeCapabilityStatus::RequiresDependency, family, route, QStringLiteral("当前构建缺少 ONNX Runtime 依赖。"));
        }
        return result(RuntimeCapabilityStatus::Supported, family, route,
            QStringLiteral("由 Manifest 指定 decoder 的 AITrain ONNX Runtime 路线可用。"));
    }
    if (route == QStringLiteral("aitrain_ncnn")) {
        if (family != QStringLiteral("yolo_detection") && family != QStringLiteral("yolo_segmentation")) {
            return result(RuntimeCapabilityStatus::UnsupportedByProduct, family, route,
                QStringLiteral("NCNN 仅支持已验证的 YOLO Detection/Segmentation 路线。"));
        }
        const aitrain::NcnnBackendStatus ncnn = aitrain::ncnnBackendStatus();
        if (!ncnn.sdkAvailable) return result(RuntimeCapabilityStatus::RequiresSdk, family, route, ncnn.message);
        if (!ncnn.inferenceAvailable) return result(RuntimeCapabilityStatus::RuntimeNotImplemented, family, route, ncnn.message);
        return result(RuntimeCapabilityStatus::Supported, family, route, QStringLiteral("NCNN 运行时可用。"));
    }
    if (route == QStringLiteral("aitrain_tensorrt")) {
        if (family != QStringLiteral("yolo_detection") && family != QStringLiteral("yolo_segmentation")) {
            return result(RuntimeCapabilityStatus::UnsupportedByProduct, family, route,
                QStringLiteral("TensorRT 不支持该模型族的产品部署。"));
        }
        const aitrain::TensorRtBackendStatus tensorRt = aitrain::tensorRtBackendStatus();
        if (!tensorRt.sdkAvailable) return result(RuntimeCapabilityStatus::RequiresSdk, family, route, tensorRt.message);
        if (tensorRt.status == QStringLiteral("dependency_missing") || !tensorRt.dependenciesAvailable) {
            return result(RuntimeCapabilityStatus::RequiresDependency, family, route, tensorRt.message);
        }
        if (!tensorRt.hardwareSupported) return result(RuntimeCapabilityStatus::RequiresHardware, family, route, tensorRt.message);
        if (!tensorRt.inferenceAvailable) return result(RuntimeCapabilityStatus::RuntimeNotImplemented, family, route, tensorRt.message);
        return result(RuntimeCapabilityStatus::Supported, family, route,
            QStringLiteral("TensorRT runtime 可用；engine 仍只对当前探测到的 GPU 有效。"),
            {QStringLiteral("跨 GPU 部署必须重新执行 engine 兼容性验证。")});
    }
    if (route == QStringLiteral("anomalib_python") && family == QStringLiteral("anomaly_detection")) {
        return result(RuntimeCapabilityStatus::RequiresExternalEvidence, family, route,
            QStringLiteral("异常检测仅使用 Worker 管理的 Anomalib Python Artifact Runtime。"),
            {QStringLiteral("不声明 AITrain C++ ONNX/TensorRT/NCNN 异常检测运行时。")});
    }
    if (route == QStringLiteral("paddleocr_official") && (family == QStringLiteral("ocr_detection") || family == QStringLiteral("ocr_recognition"))) {
        return result(RuntimeCapabilityStatus::RequiresExternalEvidence, family, route,
            QStringLiteral("OCR 只接受 PaddleOCR 官方工具链与报告作为运行/验收证据。"));
    }
    return result(RuntimeCapabilityStatus::UnsupportedByProduct, family, route, QStringLiteral("未知或未注册的 Runtime 路由。"));
}

QJsonObject RuntimeCapabilityMatrix::toJson() const
{
    QJsonArray entries;
    const QStringList families = {QStringLiteral("yolo_detection"), QStringLiteral("yolo_segmentation"),
        QStringLiteral("yolo_obb"), QStringLiteral("semantic_segmentation"), QStringLiteral("anomaly_detection"),
        QStringLiteral("ocr_detection"), QStringLiteral("ocr_recognition")};
    const QStringList routes = {QStringLiteral("aitrain_onnxruntime"), QStringLiteral("aitrain_ncnn"),
        QStringLiteral("aitrain_tensorrt"), QStringLiteral("anomalib_python"), QStringLiteral("paddleocr_official")};
    for (const QString& family : families) {
        for (const QString& route : routes) entries.append(query({family, route}).toJson());
    }
    return {{QStringLiteral("entries"), entries}};
}

} // namespace aitrain
