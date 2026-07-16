#include "DetectionTrainerInternal.h"

#include "aitrain/core/Deployment.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPainter>
#include <QProcess>
#include <QQueue>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QtEndian>
#include <QtMath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#ifdef AITRAIN_WITH_TENSORRT_SDK
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif
namespace aitrain {

using namespace detection_detail;

QJsonObject TensorRtBackendStatus::toJson() const
{
    return QJsonObject{
        {QStringLiteral("sdkAvailable"), sdkAvailable},
        {QStringLiteral("dependenciesAvailable"), dependenciesAvailable},
        {QStringLiteral("hardwareDetected"), hardwareDetected},
        {QStringLiteral("hardwareSupported"), hardwareSupported},
        {QStringLiteral("exportAvailable"), exportAvailable},
        {QStringLiteral("inferenceAvailable"), inferenceAvailable},
        {QStringLiteral("computeCapabilityMajor"), computeCapabilityMajor},
        {QStringLiteral("computeCapabilityMinor"), computeCapabilityMinor},
        {QStringLiteral("sdkStatus"), sdkStatus},
        {QStringLiteral("engineBuildStatus"), engineBuildStatus},
        {QStringLiteral("runtimeInferenceStatus"), runtimeInferenceStatus},
        {QStringLiteral("engineBuildMessage"), engineBuildMessage},
        {QStringLiteral("runtimeInferenceMessage"), runtimeInferenceMessage},
        {QStringLiteral("status"), status},
        {QStringLiteral("message"), message}
    };
}

TensorRtBackendStatus tensorRtBackendStatus()
{
    TensorRtBackendStatus status;
#ifdef AITRAIN_WITH_TENSORRT_SDK
    status.sdkAvailable = true;
    status.sdkStatus = QStringLiteral("available");
    TensorRtRuntimeLibraries libraries;
    QString dependencyError;
    if (!loadTensorRtCore(&libraries, &dependencyError)
        || !loadCudaRuntime(&libraries, &dependencyError)) {
        status.status = QStringLiteral("dependency_missing");
        status.engineBuildStatus = QStringLiteral("dependency_missing");
        status.runtimeInferenceStatus = QStringLiteral("dependency_missing");
        status.runtimeInferenceMessage = dependencyError.isEmpty()
            ? QStringLiteral("TensorRT/CUDA 运行时依赖不可用。")
            : dependencyError;
        status.engineBuildMessage = status.runtimeInferenceMessage;
        status.message = status.runtimeInferenceMessage;
        return status;
    }
    status.dependenciesAvailable = true;
    QString parserError;
    const bool parserAvailable = loadTensorRtParser(&libraries, &parserError);
    status.engineBuildStatus = parserAvailable ? QStringLiteral("available") : QStringLiteral("dependency_missing");
    status.engineBuildMessage = parserAvailable
        ? QStringLiteral("TensorRT engine builder 与 ONNX parser 依赖可用。")
        : parserError;
    int deviceCount = 0;
    const cudaError_t countStatus = libraries.cudaGetDeviceCount(&deviceCount);
    if (countStatus != cudaSuccess) {
        status.status = countStatus == cudaErrorNoDevice
            ? QStringLiteral("hardware_unsupported")
            : QStringLiteral("dependency_missing");
        status.engineBuildStatus = status.status;
        status.runtimeInferenceStatus = status.status;
        status.runtimeInferenceMessage = QStringLiteral("CUDA 设备探测失败：%1").arg(cudaErrorText(libraries, countStatus));
        status.engineBuildMessage = status.runtimeInferenceMessage;
        status.message = status.runtimeInferenceMessage;
        return status;
    }
    if (deviceCount <= 0) {
        status.status = QStringLiteral("hardware_unsupported");
        status.engineBuildStatus = status.status;
        status.runtimeInferenceStatus = status.status;
        status.runtimeInferenceMessage = QStringLiteral("未检测到可用于 TensorRT 的 CUDA GPU。");
        status.engineBuildMessage = status.runtimeInferenceMessage;
        status.message = status.runtimeInferenceMessage;
        return status;
    }
    status.hardwareDetected = true;
    for (int index = 0; index < deviceCount; ++index) {
        cudaDeviceProp properties{};
        const cudaError_t propertyStatus = libraries.cudaGetDeviceProperties(&properties, index);
        if (propertyStatus != cudaSuccess) {
            status.status = QStringLiteral("dependency_missing");
            status.engineBuildStatus = status.status;
            status.runtimeInferenceStatus = status.status;
            status.runtimeInferenceMessage = QStringLiteral("读取 CUDA GPU 属性失败：%1").arg(cudaErrorText(libraries, propertyStatus));
            status.engineBuildMessage = status.runtimeInferenceMessage;
            status.message = status.runtimeInferenceMessage;
            return status;
        }
        if (properties.major > status.computeCapabilityMajor
            || (properties.major == status.computeCapabilityMajor && properties.minor > status.computeCapabilityMinor)) {
            status.computeCapabilityMajor = properties.major;
            status.computeCapabilityMinor = properties.minor;
        }
    }
    status.hardwareSupported = status.computeCapabilityMajor > 7
        || (status.computeCapabilityMajor == 7 && status.computeCapabilityMinor >= 5);
    if (!status.hardwareSupported) {
        status.status = QStringLiteral("hardware_unsupported");
        status.engineBuildStatus = status.status;
        status.runtimeInferenceStatus = status.status;
        status.runtimeInferenceMessage = QStringLiteral("TensorRT 产品路线要求 GPU compute capability >= 7.5；当前最高为 %1.%2。")
            .arg(status.computeCapabilityMajor).arg(status.computeCapabilityMinor);
        status.engineBuildMessage = status.runtimeInferenceMessage;
        status.message = status.runtimeInferenceMessage;
        return status;
    }
    status.exportAvailable = parserAvailable;
    status.inferenceAvailable = false;
    status.runtimeInferenceStatus = QStringLiteral("runtime_not_implemented");
    status.status = QStringLiteral("runtime_not_implemented");
    status.runtimeInferenceMessage = QStringLiteral("官方 YOLO TensorRT decoder 尚未实现，不能声明 runtime inference 成功。");
    status.message = status.runtimeInferenceMessage;
#else
    status.sdkAvailable = false;
    status.sdkStatus = QStringLiteral("sdk_missing");
    status.engineBuildStatus = QStringLiteral("sdk_missing");
    status.runtimeInferenceStatus = QStringLiteral("sdk_missing");
    status.engineBuildMessage = QStringLiteral("TensorRT SDK 未在配置阶段找到，无法构建 engine。");
    status.runtimeInferenceMessage = QStringLiteral("TensorRT SDK 未在配置阶段找到，无法执行 runtime inference。");
    status.exportAvailable = false;
    status.inferenceAvailable = false;
    status.status = QStringLiteral("sdk_missing");
    status.message = QStringLiteral("TensorRT SDK was not found at configure time. Set AITRAIN_TENSORRT_ROOT, TENSORRT_ROOT, or TRT_ROOT before enabling real TensorRT export/inference.");
#endif
    return status;
}

bool isTensorRtInferenceAvailable()
{
    return tensorRtBackendStatus().inferenceAvailable;
}

QVector<DetectionPrediction> predictDetectionTensorRt(
    const QString& enginePath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error)
{
#ifndef AITRAIN_WITH_TENSORRT_SDK
    Q_UNUSED(enginePath)
    Q_UNUSED(imagePath)
    Q_UNUSED(options)
    if (error) {
        *error = QStringLiteral("TensorRT inference is not available: %1").arg(tensorRtBackendStatus().message);
    }
    return {};
#else
    return predictTensorRtEngine(enginePath, imagePath, options, error);
#endif
}


} // namespace aitrain
