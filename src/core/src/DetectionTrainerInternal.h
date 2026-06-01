#pragma once

#include "aitrain/core/DetectionTrainer.h"

#include <QByteArray>
#include <QColor>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QLibrary>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QVector>
#include <vector>

#ifdef AITRAIN_WITH_TENSORRT_SDK
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif

namespace aitrain {
namespace detection_detail {

double clamp01(double value);
double sigmoid(double value);
QJsonObject boxObject(const DetectionBox& box);
double boxIou(const DetectionBox& left, const DetectionBox& right);

QStringList stringListFromArray(const QJsonArray& array);
QStringList classNamesFromYoloDataYaml(const QString& yamlPath);
QJsonObject loadUltralyticsTrainingReport(const QString& onnxPath);
QStringList ultralyticsClassNames(const QString& onnxPath);
QVector<float> yoloImageTensorFromLetterbox(const QImage& image, const QSize& inputSize, LetterboxTransform* transform);
DetectionBox yoloBoxFromInputPixels(
    double xCenter,
    double yCenter,
    double width,
    double height,
    int classId,
    const QSize& inputSize,
    const LetterboxTransform& transform);
QVector<DetectionPrediction> yoloPredictionsFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    const QStringList& classNames,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error);
QColor overlayColorForClass(int classId, int alpha);
QImage maskFromPrototype(
    const QVector<float>& coefficients,
    const float* prototypes,
    const std::vector<int64_t>& prototypeShape,
    const DetectionBox& box,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    double threshold,
    double* maskArea);
QVector<SegmentationPrediction> yoloSegmentationPredictionsFromOutputs(
    const float* boxesAndMasks,
    const std::vector<int64_t>& boxesShape,
    const float* prototypes,
    const std::vector<int64_t>& prototypeShape,
    const QStringList& classNames,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error);
QVector<DetectionPrediction> ncnnDflPredictionsFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    const QStringList& classNames,
    const QVector<int>& strides,
    int regMax,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error);
QVector<SegmentationPrediction> ncnnDflSegmentationPredictionsFromOutputs(
    const float* boxes,
    const std::vector<int64_t>& boxesShape,
    const float* maskFeatures,
    int maskFeatureRows,
    int maskFeatureColumns,
    const float* prototypes,
    const std::vector<int64_t>& prototypeShape,
    const QStringList& classNames,
    const QVector<int>& strides,
    int regMax,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error);
QStringList readOcrDictionary(const QString& path);
QJsonObject loadOcrRecReport(const QString& onnxPath);
QJsonObject loadOcrDetReport(const QString& onnxPath);
QVector<float> ocrImageTensor(const QImage& image, int width, int height);
QVector<float> ocrDetImageTensor(const QImage& image, int width, int height);
OcrRecPrediction ocrPredictionFromLogits(
    const float* logits,
    const std::vector<int64_t>& shape,
    const QStringList& dictionary,
    int blankIndex,
    QString* error);
QVector<float> ocrDetProbabilityMapFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    QSize* mapSize,
    QString* error);

QString onnxExportReportPath(const QString& onnxPath);
bool writeJsonObject(const QString& path, const QJsonObject& object, QString* error);
QJsonObject loadOnnxExportConfig(const QString& onnxPath);

#ifdef AITRAIN_WITH_TENSORRT_SDK
using CreateInferBuilderInternalFn = void* (*)(void*, int32_t);
using CreateInferRuntimeInternalFn = void* (*)(void*, int32_t);
using CreateNvOnnxParserInternalFn = void* (*)(void*, void*, int32_t);
using CudaMallocFn = cudaError_t (*)(void**, size_t);
using CudaFreeFn = cudaError_t (*)(void*);
using CudaMemcpyFn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
using CudaDeviceSynchronizeFn = cudaError_t (*)();
using CudaGetErrorStringFn = const char* (*)(cudaError_t);

struct TensorRtRuntimeLibraries {
    QLibrary nvinfer;
    QLibrary nvonnxparser;
    QLibrary cudart;
    CreateInferBuilderInternalFn createInferBuilder = nullptr;
    CreateInferRuntimeInternalFn createInferRuntime = nullptr;
    CreateNvOnnxParserInternalFn createNvOnnxParser = nullptr;
    CudaMallocFn cudaMalloc = nullptr;
    CudaFreeFn cudaFree = nullptr;
    CudaMemcpyFn cudaMemcpy = nullptr;
    CudaDeviceSynchronizeFn cudaDeviceSynchronize = nullptr;
    CudaGetErrorStringFn cudaGetErrorString = nullptr;
};

bool loadTensorRtCore(TensorRtRuntimeLibraries* libraries, QString* error);
bool loadTensorRtParser(TensorRtRuntimeLibraries* libraries, QString* error);
bool loadCudaRuntime(TensorRtRuntimeLibraries* libraries, QString* error);
bool writeTensorRtEngineFromOnnx(const QByteArray& onnxModel, const QString& outputPath, bool fp16, QString* error);
QVector<DetectionPrediction> predictTensorRtEngine(
    const QString& enginePath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error);
#endif

} // namespace detection_detail
} // namespace aitrain
