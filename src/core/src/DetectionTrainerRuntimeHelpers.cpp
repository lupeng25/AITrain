#include "DetectionTrainerInternal.h"

#include "aitrain/core/Deployment.h"

#include <QDir>
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLibrary>
#include <QMap>
#include <QPainter>
#include <QQueue>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QtEndian>
#include <QtMath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#ifdef AITRAIN_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#ifdef AITRAIN_WITH_TENSORRT_SDK
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif

namespace aitrain {
namespace detection_detail {

double clamp01(double value)
{
    return qBound(0.0, value, 1.0);
}

double safeLog(double value)
{
    return qLn(qMax(value, 1.0e-12));
}

QString onnxExportReportPath(const QString& onnxPath)
{
    const QFileInfo info(onnxPath);
    return info.absoluteDir().filePath(QStringLiteral("%1.aitrain-export.json").arg(info.completeBaseName()));
}

bool writeJsonObject(const QString& path, const QJsonObject& object, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Cannot create directory for JSON artifact: %1").arg(QFileInfo(path).absolutePath());
        }
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write JSON artifact: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    file.close();
    return true;
}

QJsonObject loadOnnxExportConfig(const QString& onnxPath)
{
    const QStringList candidates = {
        onnxExportReportPath(onnxPath),
        QStringLiteral("%1.aitrain-export.json").arg(onnxPath)
    };
    for (const QString& candidate : candidates) {
        QFile file(candidate);
        if (!file.open(QIODevice::ReadOnly)) {
            continue;
        }
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
        if (document.isObject()) {
            return document.object();
        }
    }
    return {};
}

#ifdef AITRAIN_WITH_TENSORRT_SDK
class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kWARNING && message) {
            messages.append(QString::fromUtf8(message));
        }
    }

    QString joinedMessages() const
    {
        return messages.join(QStringLiteral("; "));
    }

private:
    QStringList messages;
};

bool loadRuntimeLibrary(
    QLibrary* library,
    const QString& name,
    const QStringList& libraryNames,
    QString* error)
{
    const RuntimeDependencyCheck check = checkRuntimeDependency(
        name,
        libraryNames,
        QStringLiteral("Required by the TensorRT backend."));
    if (check.status != QStringLiteral("ok")) {
        if (error) {
            *error = check.message;
        }
        return false;
    }

    const QFileInfo resolvedInfo(check.resolvedPath);
    if (resolvedInfo.isAbsolute()) {
        const QByteArray directory = QFile::encodeName(QDir::toNativeSeparators(resolvedInfo.absolutePath()));
        const QByteArray path = qgetenv("PATH");
#ifdef Q_OS_WIN
        const char separator = ';';
#else
        const char separator = ':';
#endif
        if (!path.split(separator).contains(directory)) {
            qputenv("PATH", directory + QByteArray(1, separator) + path);
        }
    }
    library->setFileName(check.resolvedPath);
    if (!library->load()) {
        if (error) {
            *error = QStringLiteral("Cannot load %1 runtime library: %2").arg(name, library->errorString());
        }
        return false;
    }
    return true;
}

template <typename Function>
bool resolveRuntimeSymbol(QLibrary& library, const char* symbolName, Function* function, QString* error)
{
    const QFunctionPointer pointer = library.resolve(symbolName);
    if (!pointer) {
        if (error) {
            *error = QStringLiteral("TensorRT runtime library is missing symbol %1").arg(QString::fromLatin1(symbolName));
        }
        return false;
    }
    *function = reinterpret_cast<Function>(pointer);
    return true;
}

bool loadTensorRtCore(TensorRtRuntimeLibraries* libraries, QString* error)
{
    if (!loadRuntimeLibrary(
            &libraries->nvinfer,
            QStringLiteral("TensorRT"),
            QStringList() << QStringLiteral("nvinfer") << QStringLiteral("nvinfer_10") << QStringLiteral("nvinfer_8"),
            error)) {
        return false;
    }
    return resolveRuntimeSymbol(libraries->nvinfer, "createInferBuilder_INTERNAL", &libraries->createInferBuilder, error)
        && resolveRuntimeSymbol(libraries->nvinfer, "createInferRuntime_INTERNAL", &libraries->createInferRuntime, error);
}

bool loadTensorRtParser(TensorRtRuntimeLibraries* libraries, QString* error)
{
    if (!loadRuntimeLibrary(
            &libraries->nvonnxparser,
            QStringLiteral("TensorRT ONNX Parser"),
            QStringList() << QStringLiteral("nvonnxparser") << QStringLiteral("nvonnxparser_10") << QStringLiteral("nvonnxparser_8"),
            error)) {
        return false;
    }
    return resolveRuntimeSymbol(libraries->nvonnxparser, "createNvOnnxParser_INTERNAL", &libraries->createNvOnnxParser, error);
}

bool loadCudaRuntime(TensorRtRuntimeLibraries* libraries, QString* error)
{
    if (!loadRuntimeLibrary(
            &libraries->cudart,
            QStringLiteral("CUDA Runtime"),
            QStringList() << QStringLiteral("cudart64_13") << QStringLiteral("cudart64_12") << QStringLiteral("cudart64_120") << QStringLiteral("cudart64_110"),
            error)) {
        return false;
    }
    return resolveRuntimeSymbol(libraries->cudart, "cudaMalloc", &libraries->cudaMalloc, error)
        && resolveRuntimeSymbol(libraries->cudart, "cudaFree", &libraries->cudaFree, error)
        && resolveRuntimeSymbol(libraries->cudart, "cudaMemcpy", &libraries->cudaMemcpy, error)
        && resolveRuntimeSymbol(libraries->cudart, "cudaDeviceSynchronize", &libraries->cudaDeviceSynchronize, error)
        && resolveRuntimeSymbol(libraries->cudart, "cudaGetErrorString", &libraries->cudaGetErrorString, error);
}

QString cudaErrorText(const TensorRtRuntimeLibraries& libraries, cudaError_t code)
{
    const char* message = libraries.cudaGetErrorString ? libraries.cudaGetErrorString(code) : nullptr;
    return message ? QString::fromUtf8(message) : QStringLiteral("CUDA error %1").arg(static_cast<int>(code));
}

QString parserErrorText(const nvonnxparser::IParser& parser)
{
    QStringList errors;
    const int count = parser.getNbErrors();
    for (int index = 0; index < count; ++index) {
        const nvonnxparser::IParserError* parserError = parser.getError(index);
        if (parserError && parserError->desc()) {
            errors.append(QString::fromUtf8(parserError->desc()));
        }
    }
    return errors.join(QStringLiteral("; "));
}

qint64 tensorElementCount(const nvinfer1::Dims& dims)
{
    if (dims.nbDims <= 0) {
        return 0;
    }
    qint64 count = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            return -1;
        }
        count *= dims.d[index];
    }
    return count;
}

bool writeTensorRtEngineFromOnnx(
    const QByteArray& onnxModel,
    const QString& outputPath,
    bool fp16,
    QString* error)
{
    TensorRtRuntimeLibraries libraries;
    if (!loadTensorRtCore(&libraries, error) || !loadTensorRtParser(&libraries, error)) {
        return false;
    }

    TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(
        static_cast<nvinfer1::IBuilder*>(libraries.createInferBuilder(&logger, NV_TENSORRT_VERSION)));
    if (!builder) {
        if (error) {
            *error = QStringLiteral("Cannot create TensorRT builder: %1").arg(logger.joinedMessages());
        }
        return false;
    }

    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !config) {
        if (error) {
            *error = QStringLiteral("Cannot create TensorRT network/config: %1").arg(logger.joinedMessages());
        }
        return false;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t{1} << 30);
    if (fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    std::unique_ptr<nvonnxparser::IParser> parser(
        static_cast<nvonnxparser::IParser*>(libraries.createNvOnnxParser(network.get(), &logger, NV_ONNX_PARSER_VERSION)));
    if (!parser) {
        if (error) {
            *error = QStringLiteral("Cannot create TensorRT ONNX parser: %1").arg(logger.joinedMessages());
        }
        return false;
    }
    if (!parser->parse(onnxModel.constData(), static_cast<size_t>(onnxModel.size()))) {
        const QString parserErrors = parserErrorText(*parser);
        if (error) {
            *error = parserErrors.isEmpty()
                ? QStringLiteral("TensorRT ONNX parser rejected the tiny detector model: %1").arg(logger.joinedMessages())
                : QStringLiteral("TensorRT ONNX parser rejected the tiny detector model: %1").arg(parserErrors);
        }
        return false;
    }

    std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized || !serialized->data() || serialized->size() == 0) {
        if (error) {
            *error = QStringLiteral("TensorRT engine build failed: %1").arg(logger.joinedMessages());
        }
        return false;
    }

    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write TensorRT engine artifact: %1").arg(outputPath);
        }
        return false;
    }
    file.write(static_cast<const char*>(serialized->data()), static_cast<qint64>(serialized->size()));
    file.close();
    return true;
}

struct TensorRtTensorInfo {
    QString name;
    QByteArray nameBytes;
    nvinfer1::TensorIOMode mode = nvinfer1::TensorIOMode::kNONE;
    nvinfer1::Dims shape;
    qint64 elementCount = 0;
};

QVector<DetectionPrediction> predictTensorRtEngine(
    const QString& enginePath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error)
{
    QFile engineFile(enginePath);
    if (!engineFile.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("TensorRT engine does not exist or cannot be read: %1").arg(enginePath);
        }
        return {};
    }
    const QByteArray engineBytes = engineFile.readAll();
    if (engineBytes.isEmpty()) {
        if (error) {
            *error = QStringLiteral("TensorRT engine is empty: %1").arg(enginePath);
        }
        return {};
    }

    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for TensorRT detection prediction: %1").arg(imagePath);
        }
        return {};
    }

    TensorRtRuntimeLibraries libraries;
    if (!loadTensorRtCore(&libraries, error) || !loadCudaRuntime(&libraries, error)) {
        return {};
    }

    TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        static_cast<nvinfer1::IRuntime*>(libraries.createInferRuntime(&logger, NV_TENSORRT_VERSION)));
    if (!runtime) {
        if (error) {
            *error = QStringLiteral("Cannot create TensorRT runtime: %1").arg(logger.joinedMessages());
        }
        return {};
    }

    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engineBytes.constData(), static_cast<size_t>(engineBytes.size())));
    if (!engine) {
        if (error) {
            *error = QStringLiteral("Cannot deserialize TensorRT engine: %1").arg(logger.joinedMessages());
        }
        return {};
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        if (error) {
            *error = QStringLiteral("Cannot create TensorRT execution context: %1").arg(logger.joinedMessages());
        }
        return {};
    }

    QVector<TensorRtTensorInfo> tensors;
    TensorRtTensorInfo input;
    bool hasInput = false;
    const int tensorCount = engine->getNbIOTensors();
    tensors.reserve(tensorCount);
    for (int index = 0; index < tensorCount; ++index) {
        const char* tensorName = engine->getIOTensorName(index);
        if (!tensorName) {
            continue;
        }
        TensorRtTensorInfo info;
        info.nameBytes = QByteArray(tensorName);
        info.name = QString::fromUtf8(tensorName);
        info.mode = engine->getTensorIOMode(tensorName);
        info.shape = engine->getTensorShape(tensorName);
        info.elementCount = tensorElementCount(info.shape);
        if (engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT || info.elementCount <= 0) {
            if (error) {
                *error = QStringLiteral("TensorRT tiny detector tensor %1 must be fixed-shape float32").arg(info.name);
            }
            return {};
        }
        if (info.mode == nvinfer1::TensorIOMode::kINPUT) {
            input = info;
            hasInput = true;
        }
        tensors.append(info);
    }

    if (!hasInput || input.shape.nbDims != 2 || input.shape.d[0] <= 0 || input.shape.d[1] != 7) {
        if (error) {
            *error = QStringLiteral("TensorRT tiny detector expects one [cells, 7] input tensor");
        }
        return {};
    }
    const int cellCount = static_cast<int>(input.shape.d[0]);
    const int featureCount = static_cast<int>(input.shape.d[1]);
    const int gridSize = qMax(1, qRound(qSqrt(static_cast<double>(cellCount))));
    QVector<float> inputFeatures = tinyDetectorFeatureInput(image, cellCount, featureCount, gridSize, error);
    if (inputFeatures.isEmpty()) {
        return {};
    }

    QVector<void*> deviceBuffers;
    auto freeDeviceBuffers = [&libraries, &deviceBuffers]() {
        for (void* buffer : deviceBuffers) {
            if (buffer) {
                libraries.cudaFree(buffer);
            }
        }
        deviceBuffers.clear();
    };
    auto fail = [&freeDeviceBuffers, error](const QString& message) {
        freeDeviceBuffers();
        if (error) {
            *error = message;
        }
        return QVector<DetectionPrediction>{};
    };

    void* inputDevice = nullptr;
    const size_t inputBytes = static_cast<size_t>(inputFeatures.size()) * sizeof(float);
    cudaError_t cudaResult = libraries.cudaMalloc(&inputDevice, inputBytes);
    if (cudaResult != cudaSuccess) {
        return fail(QStringLiteral("CUDA allocation failed for TensorRT input: %1").arg(cudaErrorText(libraries, cudaResult)));
    }
    deviceBuffers.append(inputDevice);
    cudaResult = libraries.cudaMemcpy(inputDevice, inputFeatures.constData(), inputBytes, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        return fail(QStringLiteral("CUDA copy failed for TensorRT input: %1").arg(cudaErrorText(libraries, cudaResult)));
    }

    QVector<float> objectness;
    QVector<float> classProbabilities;
    QVector<float> boxes;
    QVector<void*> outputDevices;
    for (const TensorRtTensorInfo& tensor : tensors) {
        void* devicePointer = nullptr;
        if (tensor.mode == nvinfer1::TensorIOMode::kINPUT) {
            devicePointer = inputDevice;
        } else if (tensor.mode == nvinfer1::TensorIOMode::kOUTPUT) {
            const size_t outputBytes = static_cast<size_t>(tensor.elementCount) * sizeof(float);
            cudaResult = libraries.cudaMalloc(&devicePointer, outputBytes);
            if (cudaResult != cudaSuccess) {
                return fail(QStringLiteral("CUDA allocation failed for TensorRT output %1: %2").arg(tensor.name, cudaErrorText(libraries, cudaResult)));
            }
            deviceBuffers.append(devicePointer);
            outputDevices.append(devicePointer);
            if (tensor.name == QStringLiteral("objectness")) {
                objectness.resize(static_cast<int>(tensor.elementCount));
            } else if (tensor.name == QStringLiteral("class_probabilities")) {
                classProbabilities.resize(static_cast<int>(tensor.elementCount));
            } else if (tensor.name == QStringLiteral("boxes")) {
                boxes.resize(static_cast<int>(tensor.elementCount));
            }
        }

        if (!devicePointer || !context->setTensorAddress(tensor.nameBytes.constData(), devicePointer)) {
            return fail(QStringLiteral("Cannot bind TensorRT tensor address: %1").arg(tensor.name));
        }
    }
    Q_UNUSED(outputDevices)

    if (objectness.size() != cellCount || boxes.size() != cellCount * 4
        || classProbabilities.isEmpty() || classProbabilities.size() % cellCount != 0) {
        return fail(QStringLiteral("TensorRT tiny detector output shapes are invalid"));
    }
    const int classCount = classProbabilities.size() / cellCount;

    if (!context->enqueueV3(nullptr)) {
        return fail(QStringLiteral("TensorRT inference enqueue failed: %1").arg(logger.joinedMessages()));
    }
    cudaResult = libraries.cudaDeviceSynchronize();
    if (cudaResult != cudaSuccess) {
        return fail(QStringLiteral("CUDA synchronize failed after TensorRT inference: %1").arg(cudaErrorText(libraries, cudaResult)));
    }

    for (const TensorRtTensorInfo& tensor : tensors) {
        if (tensor.mode != nvinfer1::TensorIOMode::kOUTPUT) {
            continue;
        }
        float* hostBuffer = nullptr;
        if (tensor.name == QStringLiteral("objectness")) {
            hostBuffer = objectness.data();
        } else if (tensor.name == QStringLiteral("class_probabilities")) {
            hostBuffer = classProbabilities.data();
        } else if (tensor.name == QStringLiteral("boxes")) {
            hostBuffer = boxes.data();
        }
        if (!hostBuffer) {
            continue;
        }
        void* devicePointer = nullptr;
        if (!context->getTensorAddress(tensor.nameBytes.constData())) {
            return fail(QStringLiteral("TensorRT output tensor address is not available: %1").arg(tensor.name));
        }
        devicePointer = const_cast<void*>(context->getTensorAddress(tensor.nameBytes.constData()));
        cudaResult = libraries.cudaMemcpy(
            hostBuffer,
            devicePointer,
            static_cast<size_t>(tensor.elementCount) * sizeof(float),
            cudaMemcpyDeviceToHost);
        if (cudaResult != cudaSuccess) {
            return fail(QStringLiteral("CUDA copy failed for TensorRT output %1: %2").arg(tensor.name, cudaErrorText(libraries, cudaResult)));
        }
    }

    const QJsonObject exportConfig = loadOnnxExportConfig(enginePath);
    const QStringList classNames = stringListFromArray(exportConfig.value(QStringLiteral("classNames")).toArray());
    QVector<DetectionPrediction> predictions = tinyDetectorPredictionsFromOutputs(
        objectness.constData(),
        classProbabilities.constData(),
        boxes.constData(),
        cellCount,
        classCount,
        classNames,
        options);
    freeDeviceBuffers();
    return predictions;
}
#endif

} // namespace detection_detail

} // namespace aitrain
