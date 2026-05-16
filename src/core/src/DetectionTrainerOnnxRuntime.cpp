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
#include <exception>
#include <limits>
#include <memory>
#include <vector>
#ifdef AITRAIN_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif
namespace aitrain {

using namespace detection_detail;

bool isOnnxRuntimeInferenceAvailable()
{
#ifdef AITRAIN_WITH_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

QString inferOnnxModelFamily(const QString& onnxPath)
{
    return inferOnnxModelFamily(onnxPath, nullptr);
}

QString inferOnnxModelFamily(const QString& onnxPath, QString* warning)
{
    if (warning) {
        warning->clear();
    }
    const QJsonObject config = loadOnnxExportConfig(onnxPath);
    const QString configuredFamily = config.value(QStringLiteral("modelFamily")).toString();
    const QString configuredBackend = config.value(QStringLiteral("backend")).toString();
    if (configuredFamily == QStringLiteral("yolo_segmentation")
        || configuredBackend == QStringLiteral("ultralytics_yolo_segment")) {
        return QStringLiteral("yolo_segmentation");
    }
    if (configuredFamily == QStringLiteral("ocr_recognition")
        || configuredBackend == QStringLiteral("paddleocr_rec")) {
        return QStringLiteral("ocr_recognition");
    }
    if (configuredFamily == QStringLiteral("ocr_detection")
        || configuredBackend == QStringLiteral("paddleocr_det_official")) {
        return QStringLiteral("ocr_detection");
    }
    if (configuredFamily == QStringLiteral("yolo_detection")
        || configuredBackend == QStringLiteral("ultralytics_yolo_detect")) {
        return QStringLiteral("yolo_detection");
    }

    const QJsonObject detReport = loadOcrDetReport(onnxPath);
    if (!detReport.isEmpty()) {
        return QStringLiteral("ocr_detection");
    }
    const QJsonObject ocrReport = loadOcrRecReport(onnxPath);
    if (!ocrReport.isEmpty()) {
        return QStringLiteral("ocr_recognition");
    }
    const QJsonObject yoloReport = loadUltralyticsTrainingReport(onnxPath);
    if (yoloReport.value(QStringLiteral("backend")).toString() == QStringLiteral("ultralytics_yolo_segment")) {
        return QStringLiteral("yolo_segmentation");
    }
    if (yoloReport.value(QStringLiteral("backend")).toString() == QStringLiteral("ultralytics_yolo_detect")) {
        return QStringLiteral("yolo_detection");
    }

#ifdef AITRAIN_WITH_ONNXRUNTIME
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aitrain");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
#ifdef Q_OS_WIN
        const std::wstring modelPath = QDir::toNativeSeparators(onnxPath).toStdWString();
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
#else
        const QByteArray modelPath = QFile::encodeName(onnxPath);
        Ort::Session session(env, modelPath.constData(), sessionOptions);
#endif
        if (session.GetInputCount() == 1) {
            const std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (inputShape.size() == 4 && inputShape.at(1) == 1) {
                return QStringLiteral("ocr_recognition");
            }
            if (inputShape.size() == 4 && inputShape.at(1) == 3 && session.GetOutputCount() == 1) {
                const std::vector<int64_t> outputShape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
                if ((outputShape.size() == 4 && outputShape.at(1) == 1)
                    || (outputShape.size() == 3 && outputShape.at(0) == 1)
                    || outputShape.size() == 2) {
                    return QStringLiteral("ocr_detection");
                }
            }
            if (session.GetOutputCount() >= 2) {
                for (size_t index = 0; index < session.GetOutputCount(); ++index) {
                    const std::vector<int64_t> outputShape = session.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
                    if (outputShape.size() == 4) {
                        return QStringLiteral("yolo_segmentation");
                    }
                }
            }
            if (inputShape.size() == 4 && inputShape.at(1) == 3) {
                return QStringLiteral("yolo_detection");
            }
        }
    } catch (const std::exception& exception) {
        if (warning) {
            *warning = QStringLiteral("ONNX model-family inference failed for %1: %2")
                .arg(onnxPath, QString::fromUtf8(exception.what()));
        }
    } catch (...) {
        if (warning) {
            *warning = QStringLiteral("ONNX model-family inference failed for %1: unknown exception").arg(onnxPath);
        }
    }
#else
    if (warning) {
        *warning = QStringLiteral("ONNX Runtime SDK is not enabled; model-family inference is limited to sidecar/report metadata.");
    }
#endif
    return {};
}
QVector<DetectionPrediction> predictDetectionOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error)
{
#ifndef AITRAIN_WITH_ONNXRUNTIME
    Q_UNUSED(onnxPath)
    Q_UNUSED(imagePath)
    Q_UNUSED(options)
    if (error) {
        *error = QStringLiteral("ONNX Runtime inference is not enabled. Configure AITRAIN_ONNXRUNTIME_ROOT with an ONNX Runtime SDK to enable .onnx inference.");
    }
    return {};
#else
    if (!QFileInfo::exists(onnxPath)) {
        if (error) {
            *error = QStringLiteral("ONNX model does not exist: %1").arg(onnxPath);
        }
        return {};
    }

    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for ONNX detection prediction: %1").arg(imagePath);
        }
        return {};
    }

    try {
        const QJsonObject exportConfig = loadOnnxExportConfig(onnxPath);
        const QStringList classNames = stringListFromArray(exportConfig.value(QStringLiteral("classNames")).toArray());
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aitrain");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef Q_OS_WIN
        const std::wstring modelPath = QDir::toNativeSeparators(onnxPath).toStdWString();
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
#else
        const QByteArray modelPath = QFile::encodeName(onnxPath);
        Ort::Session session(env, modelPath.constData(), sessionOptions);
#endif
        Ort::AllocatorWithDefaultOptions allocator;

        if (session.GetInputCount() != 1) {
            if (error) {
                *error = QStringLiteral("ONNX detection inference expects exactly one input tensor");
            }
            return {};
        }
        Ort::TypeInfo inputType = session.GetInputTypeInfo(0);
        const std::vector<int64_t> inputShape = inputType.GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() == 4) {
            const int channels = static_cast<int>(inputShape.at(1));
            const int inputHeight = static_cast<int>(inputShape.at(2));
            const int inputWidth = static_cast<int>(inputShape.at(3));
            if (channels != 3 || inputHeight <= 0 || inputWidth <= 0) {
                if (error) {
                    *error = QStringLiteral("YOLO detection ONNX input shape must be [1, 3, height, width]");
                }
                return {};
            }

            LetterboxTransform transform;
            const QSize inputSize(inputWidth, inputHeight);
            QVector<float> input = yoloImageTensorFromLetterbox(image, inputSize, &transform);
            std::vector<int64_t> tensorShape = inputShape;
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                input.data(),
                static_cast<size_t>(input.size()),
                tensorShape.data(),
                tensorShape.size());

            auto inputName = session.GetInputNameAllocated(0, allocator);
            std::vector<Ort::AllocatedStringPtr> outputNameHolders;
            std::vector<const char*> outputNames;
            const size_t outputCount = session.GetOutputCount();
            outputNameHolders.reserve(outputCount);
            outputNames.reserve(outputCount);
            for (size_t outputIndex = 0; outputIndex < outputCount; ++outputIndex) {
                outputNameHolders.emplace_back(session.GetOutputNameAllocated(outputIndex, allocator));
                outputNames.push_back(outputNameHolders.back().get());
            }
            if (outputNames.empty()) {
                if (error) {
                    *error = QStringLiteral("YOLO detection ONNX model has no outputs");
                }
                return {};
            }

            const char* inputNames[] = { inputName.get() };
            std::vector<Ort::Value> outputs = session.Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputTensor,
                1,
                outputNames.data(),
                outputNames.size());

            const QStringList classNames = ultralyticsClassNames(onnxPath);
            const std::vector<int64_t> outputShape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
            return yoloPredictionsFromOutput(
                outputs.front().GetTensorData<float>(),
                outputShape,
                classNames,
                inputSize,
                transform,
                options,
                error);
        }

        if (session.GetOutputCount() < 3) {
            if (error) {
                *error = QStringLiteral("ONNX tiny detector expects at least three outputs; YOLO detection expects a 4D image input.");
            }
            return {};
        }
        if (inputShape.size() != 2 || inputShape.at(0) <= 0 || inputShape.at(1) <= 0) {
            if (error) {
                *error = QStringLiteral("ONNX tiny detector input shape must be [cells, features]");
            }
            return {};
        }
        const int cellCount = static_cast<int>(inputShape.at(0));
        const int featureCount = static_cast<int>(inputShape.at(1));
        const int gridSize = qMax(1, qRound(qSqrt(static_cast<double>(cellCount))));
        QVector<float> input = tinyDetectorFeatureInput(image, cellCount, featureCount, gridSize, error);
        if (input.isEmpty()) {
            return {};
        }

        std::vector<int64_t> tensorShape = inputShape;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            static_cast<size_t>(input.size()),
            tensorShape.data(),
            tensorShape.size());

        auto inputName = session.GetInputNameAllocated(0, allocator);
        std::vector<Ort::AllocatedStringPtr> outputNameHolders;
        std::vector<const char*> outputNames;
        const size_t outputCount = session.GetOutputCount();
        outputNameHolders.reserve(outputCount);
        outputNames.reserve(outputCount);
        for (size_t outputIndex = 0; outputIndex < outputCount; ++outputIndex) {
            outputNameHolders.emplace_back(session.GetOutputNameAllocated(outputIndex, allocator));
            outputNames.push_back(outputNameHolders.back().get());
        }

        const char* inputNames[] = { inputName.get() };
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames.data(),
            outputNames.size());

        int objectnessIndex = -1;
        int classIndex = -1;
        int boxIndex = -1;
        for (int index = 0; index < static_cast<int>(outputNames.size()); ++index) {
            const QString name = QString::fromUtf8(outputNames.at(index));
            if (name == QStringLiteral("objectness")) objectnessIndex = index;
            if (name == QStringLiteral("class_probabilities")) classIndex = index;
            if (name == QStringLiteral("boxes")) boxIndex = index;
        }
        if (objectnessIndex < 0 || classIndex < 0 || boxIndex < 0) {
            if (error) {
                *error = QStringLiteral("ONNX tiny detector outputs must include objectness, class_probabilities, and boxes");
            }
            return {};
        }

        auto tensorShapeInfo = [](const Ort::Value& value) {
            return value.GetTensorTypeAndShapeInfo().GetShape();
        };
        const std::vector<int64_t> objectnessShape = tensorShapeInfo(outputs.at(objectnessIndex));
        const std::vector<int64_t> classShape = tensorShapeInfo(outputs.at(classIndex));
        const std::vector<int64_t> boxShape = tensorShapeInfo(outputs.at(boxIndex));
        if (objectnessShape.size() != 2 || classShape.size() != 2 || boxShape.size() != 2
            || objectnessShape.at(0) != cellCount || objectnessShape.at(1) != 1
            || classShape.at(0) != cellCount || classShape.at(1) <= 0
            || boxShape.at(0) != cellCount || boxShape.at(1) != 4) {
            if (error) {
                *error = QStringLiteral("ONNX tiny detector output shapes are invalid");
            }
            return {};
        }

        const float* objectness = outputs.at(objectnessIndex).GetTensorData<float>();
        const float* classProbabilities = outputs.at(classIndex).GetTensorData<float>();
        const float* boxes = outputs.at(boxIndex).GetTensorData<float>();
        const int classCount = static_cast<int>(classShape.at(1));
        return tinyDetectorPredictionsFromOutputs(
            objectness,
            classProbabilities,
            boxes,
            cellCount,
            classCount,
            classNames,
            options);
    } catch (const Ort::Exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX Runtime inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    } catch (const std::exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    }
#endif
}

QVector<SegmentationPrediction> predictSegmentationOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error)
{
#ifndef AITRAIN_WITH_ONNXRUNTIME
    Q_UNUSED(onnxPath)
    Q_UNUSED(imagePath)
    Q_UNUSED(options)
    if (error) {
        *error = QStringLiteral("ONNX Runtime inference is not enabled. Configure AITRAIN_ONNXRUNTIME_ROOT with an ONNX Runtime SDK to enable .onnx inference.");
    }
    return {};
#else
    if (!QFileInfo::exists(onnxPath)) {
        if (error) {
            *error = QStringLiteral("ONNX segmentation model does not exist: %1").arg(onnxPath);
        }
        return {};
    }
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for ONNX segmentation prediction: %1").arg(imagePath);
        }
        return {};
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aitrain");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef Q_OS_WIN
        const std::wstring modelPath = QDir::toNativeSeparators(onnxPath).toStdWString();
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
#else
        const QByteArray modelPath = QFile::encodeName(onnxPath);
        Ort::Session session(env, modelPath.constData(), sessionOptions);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        if (session.GetInputCount() != 1 || session.GetOutputCount() < 2) {
            if (error) {
                *error = QStringLiteral("YOLO segmentation ONNX expects one input and at least two outputs");
            }
            return {};
        }
        const std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() != 4 || inputShape.at(1) != 3 || inputShape.at(2) <= 0 || inputShape.at(3) <= 0) {
            if (error) {
                *error = QStringLiteral("YOLO segmentation ONNX input shape must be [1, 3, height, width]");
            }
            return {};
        }

        const QSize inputSize(static_cast<int>(inputShape.at(3)), static_cast<int>(inputShape.at(2)));
        LetterboxTransform transform;
        QVector<float> input = yoloImageTensorFromLetterbox(image, inputSize, &transform);
        std::vector<int64_t> tensorShape = {1, 3, inputSize.height(), inputSize.width()};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            static_cast<size_t>(input.size()),
            tensorShape.data(),
            tensorShape.size());

        auto inputName = session.GetInputNameAllocated(0, allocator);
        std::vector<Ort::AllocatedStringPtr> outputNameHolders;
        std::vector<const char*> outputNames;
        const size_t outputCount = session.GetOutputCount();
        outputNameHolders.reserve(outputCount);
        outputNames.reserve(outputCount);
        for (size_t outputIndex = 0; outputIndex < outputCount; ++outputIndex) {
            outputNameHolders.emplace_back(session.GetOutputNameAllocated(outputIndex, allocator));
            outputNames.push_back(outputNameHolders.back().get());
        }
        const char* inputNames[] = { inputName.get() };
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames.data(),
            outputNames.size());

        int boxesIndex = 0;
        int prototypeIndex = 1;
        for (int index = 0; index < static_cast<int>(outputs.size()); ++index) {
            const std::vector<int64_t> shape = outputs.at(index).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 4) {
                prototypeIndex = index;
            } else if (shape.size() == 3) {
                boxesIndex = index;
            }
        }
        const QStringList classNames = ultralyticsClassNames(onnxPath);
        return yoloSegmentationPredictionsFromOutputs(
            outputs.at(boxesIndex).GetTensorData<float>(),
            outputs.at(boxesIndex).GetTensorTypeAndShapeInfo().GetShape(),
            outputs.at(prototypeIndex).GetTensorData<float>(),
            outputs.at(prototypeIndex).GetTensorTypeAndShapeInfo().GetShape(),
            classNames,
            inputSize,
            transform,
            options,
            error);
    } catch (const Ort::Exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX Runtime segmentation inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    } catch (const std::exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX segmentation inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    }
#endif
}

OcrRecPrediction predictOcrRecOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    QString* error)
{
#ifndef AITRAIN_WITH_ONNXRUNTIME
    Q_UNUSED(onnxPath)
    Q_UNUSED(imagePath)
    if (error) {
        *error = QStringLiteral("ONNX Runtime inference is not enabled. Configure AITRAIN_ONNXRUNTIME_ROOT with an ONNX Runtime SDK to enable .onnx inference.");
    }
    return {};
#else
    if (!QFileInfo::exists(onnxPath)) {
        if (error) {
            *error = QStringLiteral("OCR Rec ONNX model does not exist: %1").arg(onnxPath);
        }
        return {};
    }
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for OCR Rec ONNX prediction: %1").arg(imagePath);
        }
        return {};
    }

    const QJsonObject report = loadOcrRecReport(onnxPath);
    const int reportWidth = report.value(QStringLiteral("imageWidth")).toInt(96);
    const int reportHeight = report.value(QStringLiteral("imageHeight")).toInt(32);
    const int blankIndex = report.value(QStringLiteral("blankIndex")).toInt(0);
    QString dictPath = report.value(QStringLiteral("dictPath")).toString();
    if (dictPath.isEmpty()) {
        dictPath = QFileInfo(onnxPath).absoluteDir().filePath(QStringLiteral("dict.txt"));
    }
    const QStringList dictionary = readOcrDictionary(dictPath);
    if (dictionary.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Cannot read OCR dictionary for ONNX prediction: %1").arg(dictPath);
        }
        return {};
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aitrain");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef Q_OS_WIN
        const std::wstring modelPath = QDir::toNativeSeparators(onnxPath).toStdWString();
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
#else
        const QByteArray modelPath = QFile::encodeName(onnxPath);
        Ort::Session session(env, modelPath.constData(), sessionOptions);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) {
            if (error) {
                *error = QStringLiteral("OCR Rec ONNX expects one input and at least one output");
            }
            return {};
        }
        const std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() != 4 || inputShape.at(1) != 1) {
            if (error) {
                *error = QStringLiteral("OCR Rec ONNX input shape must be [1, 1, height, width]");
            }
            return {};
        }
        const int inputHeight = inputShape.at(2) > 0 ? static_cast<int>(inputShape.at(2)) : reportHeight;
        const int inputWidth = inputShape.at(3) > 0 ? static_cast<int>(inputShape.at(3)) : reportWidth;
        QVector<float> input = ocrImageTensor(image, inputWidth, inputHeight);
        std::vector<int64_t> tensorShape = {1, 1, inputHeight, inputWidth};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            static_cast<size_t>(input.size()),
            tensorShape.data(),
            tensorShape.size());

        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        const char* inputNames[] = { inputName.get() };
        const char* outputNames[] = { outputName.get() };
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);
        return ocrPredictionFromLogits(
            outputs.front().GetTensorData<float>(),
            outputs.front().GetTensorTypeAndShapeInfo().GetShape(),
            dictionary,
            blankIndex,
            error);
    } catch (const Ort::Exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX Runtime OCR Rec inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    } catch (const std::exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX OCR Rec inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    }
#endif
}

QVector<OcrDetPrediction> predictOcrDetOnnxRuntime(
    const QString& onnxPath,
    const QString& imagePath,
    const OcrDetPostprocessOptions& options,
    QString* error)
{
#ifndef AITRAIN_WITH_ONNXRUNTIME
    Q_UNUSED(onnxPath)
    Q_UNUSED(imagePath)
    Q_UNUSED(options)
    if (error) {
        *error = QStringLiteral("ONNX Runtime inference is not enabled. Configure AITRAIN_ONNXRUNTIME_ROOT with an ONNX Runtime SDK to enable .onnx inference.");
    }
    return {};
#else
    if (!QFileInfo::exists(onnxPath)) {
        if (error) {
            *error = QStringLiteral("OCR Det ONNX model does not exist: %1").arg(onnxPath);
        }
        return {};
    }
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for OCR Det ONNX prediction: %1").arg(imagePath);
        }
        return {};
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aitrain");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef Q_OS_WIN
        const std::wstring modelPath = QDir::toNativeSeparators(onnxPath).toStdWString();
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
#else
        const QByteArray modelPath = QFile::encodeName(onnxPath);
        Ort::Session session(env, modelPath.constData(), sessionOptions);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) {
            if (error) {
                *error = QStringLiteral("OCR Det DB ONNX expects one input and at least one output");
            }
            return {};
        }
        const std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() != 4 || inputShape.at(1) != 3) {
            if (error) {
                *error = QStringLiteral("OCR Det DB ONNX input shape must be [1, 3, height, width]");
            }
            return {};
        }
        const auto alignToStride = [](int value) {
            return qMax(32, ((value + 31) / 32) * 32);
        };
        const int inputHeight = inputShape.at(2) > 0 ? static_cast<int>(inputShape.at(2)) : alignToStride(image.height());
        const int inputWidth = inputShape.at(3) > 0 ? static_cast<int>(inputShape.at(3)) : alignToStride(image.width());
        if (inputHeight <= 0 || inputWidth <= 0) {
            if (error) {
                *error = QStringLiteral("OCR Det DB ONNX input dimensions are invalid");
            }
            return {};
        }

        QVector<float> input = ocrDetImageTensor(image, inputWidth, inputHeight);
        std::vector<int64_t> tensorShape = {1, 3, inputHeight, inputWidth};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            input.data(),
            static_cast<size_t>(input.size()),
            tensorShape.data(),
            tensorShape.size());

        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        const char* inputNames[] = { inputName.get() };
        const char* outputNames[] = { outputName.get() };
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);

        QSize mapSize;
        const QVector<float> probabilityMap = ocrDetProbabilityMapFromOutput(
            outputs.front().GetTensorData<float>(),
            outputs.front().GetTensorTypeAndShapeInfo().GetShape(),
            &mapSize,
            error);
        if (probabilityMap.isEmpty()) {
            return {};
        }
        return postProcessPaddleOcrDetDbMap(probabilityMap, mapSize, image.size(), options, error);
    } catch (const Ort::Exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX Runtime OCR Det inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    } catch (const std::exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX OCR Det inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    }
#endif
}

} // namespace aitrain
