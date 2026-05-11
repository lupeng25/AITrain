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
        {QStringLiteral("exportAvailable"), exportAvailable},
        {QStringLiteral("inferenceAvailable"), inferenceAvailable},
        {QStringLiteral("status"), status},
        {QStringLiteral("message"), message}
    };
}

TensorRtBackendStatus tensorRtBackendStatus()
{
    TensorRtBackendStatus status;
#ifdef AITRAIN_WITH_TENSORRT_SDK
    status.sdkAvailable = true;
    status.exportAvailable = true;
    status.inferenceAvailable = true;
    status.status = QStringLiteral("backend_available");
    status.message = QStringLiteral("TensorRT backend is compiled. Runtime DLLs are resolved lazily from runtimes/tensorrt, configured roots, or PATH.");
#else
    status.sdkAvailable = false;
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

