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

namespace {
int exportImageSize(const QJsonObject& config)
{
    const QJsonObject args = config.value(QStringLiteral("ultralyticsExportArgs")).toObject();
    const QJsonValue rawSize = args.value(QStringLiteral("imgsz"));
    if (rawSize.isDouble()) {
        return qMax(32, rawSize.toInt(640));
    }
    if (rawSize.isString()) {
        bool ok = false;
        const int parsed = rawSize.toString().trimmed().toInt(&ok);
        if (ok) {
            return qMax(32, parsed);
        }
    }
    if (rawSize.isArray() && !rawSize.toArray().isEmpty()) {
        return qMax(32, rawSize.toArray().first().toInt(640));
    }
    return 640;
}

QSize yoloInputSizeFromShape(const std::vector<int64_t>& inputShape, const QJsonObject& config)
{
    if (inputShape.size() != 4) {
        return {};
    }
    int height = static_cast<int>(inputShape.at(2));
    int width = static_cast<int>(inputShape.at(3));
    if (height <= 0 || width <= 0) {
        const int imageSize = exportImageSize(config);
        height = height <= 0 ? imageSize : height;
        width = width <= 0 ? imageSize : width;
    }
    return QSize(width, height);
}

QVector<double> doubleVectorFromArray(const QJsonArray& array, const QVector<double>& fallback)
{
    if (array.isEmpty()) {
        return fallback;
    }
    QVector<double> values;
    values.reserve(array.size());
    for (const QJsonValue& value : array) {
        values.append(value.toDouble());
    }
    return values.isEmpty() ? fallback : values;
}

QSize semanticInputSizeFromShape(const std::vector<int64_t>& inputShape, const QJsonObject& config)
{
    if (inputShape.size() != 4) {
        return {};
    }
    int height = static_cast<int>(inputShape.at(2));
    int width = static_cast<int>(inputShape.at(3));
    if (height <= 0) {
        height = config.value(QStringLiteral("inputHeight")).toInt(config.value(QStringLiteral("imageSize")).toInt(256));
    }
    if (width <= 0) {
        width = config.value(QStringLiteral("inputWidth")).toInt(config.value(QStringLiteral("imageSize")).toInt(height));
    }
    return height > 0 && width > 0 ? QSize(width, height) : QSize();
}

QVector<float> semanticImageTensor(const QImage& image, const QSize& inputSize, const QJsonObject& config)
{
    const QJsonObject normalization = config.value(QStringLiteral("normalization")).toObject();
    const QVector<double> mean = doubleVectorFromArray(normalization.value(QStringLiteral("mean")).toArray(), {0.485, 0.456, 0.406});
    const QVector<double> std = doubleVectorFromArray(normalization.value(QStringLiteral("std")).toArray(), {0.229, 0.224, 0.225});
    const double scale = normalization.value(QStringLiteral("scale")).toDouble(1.0 / 255.0);
    const QImage resized = image.convertToFormat(QImage::Format_RGB888).scaled(inputSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QVector<float> tensor;
    tensor.resize(3 * inputSize.width() * inputSize.height());
    const int planeSize = inputSize.width() * inputSize.height();
    for (int y = 0; y < inputSize.height(); ++y) {
        const uchar* line = resized.constScanLine(y);
        for (int x = 0; x < inputSize.width(); ++x) {
            const int pixelIndex = y * inputSize.width() + x;
            const int byteIndex = x * 3;
            const double red = static_cast<double>(line[byteIndex]) * scale;
            const double green = static_cast<double>(line[byteIndex + 1]) * scale;
            const double blue = static_cast<double>(line[byteIndex + 2]) * scale;
            tensor[pixelIndex] = static_cast<float>((red - mean.value(0, 0.485)) / std.value(0, 0.229));
            tensor[planeSize + pixelIndex] = static_cast<float>((green - mean.value(1, 0.456)) / std.value(1, 0.224));
            tensor[2 * planeSize + pixelIndex] = static_cast<float>((blue - mean.value(2, 0.406)) / std.value(2, 0.225));
        }
    }
    return tensor;
}

QVector<QRgb> grayscaleColorTable()
{
    QVector<QRgb> table;
    table.reserve(256);
    for (int index = 0; index < 256; ++index) {
        table.append(qRgb(index, index, index));
    }
    return table;
}

QJsonObject pixelCountsForMask(const QImage& mask)
{
    QJsonObject counts;
    for (int y = 0; y < mask.height(); ++y) {
        for (int x = 0; x < mask.width(); ++x) {
            const int classId = qGray(mask.pixel(x, y));
            const QString key = QString::number(classId);
            counts.insert(key, counts.value(key).toDouble() + 1.0);
        }
    }
    return counts;
}

bool jsonBoolValue(const QJsonValue& value, bool defaultValue = false)
{
    if (value.isBool()) {
        return value.toBool();
    }
    if (value.isString()) {
        const QString text = value.toString().trimmed().toLower();
        if (text == QStringLiteral("true") || text == QStringLiteral("1") || text == QStringLiteral("yes")) {
            return true;
        }
        if (text == QStringLiteral("false") || text == QStringLiteral("0") || text == QStringLiteral("no")
            || text == QStringLiteral("auto") || text.isEmpty()) {
            return false;
        }
    }
    if (value.isDouble()) {
        return value.toInt() != 0;
    }
    return defaultValue;
}

int reportEndToEndFlag(const QJsonObject& report)
{
    const QJsonObject args = report.value(QStringLiteral("ultralyticsExportArgs")).toObject();
    if (!args.contains(QStringLiteral("end2end"))) {
        return -1;
    }
    return jsonBoolValue(args.value(QStringLiteral("end2end")), false) ? 1 : 0;
}

bool yoloEndToEndFromExportConfig(const QString& onnxPath, const QJsonObject& config)
{
    int flag = reportEndToEndFlag(config);
    if (flag >= 0) {
        return flag == 1;
    }
    flag = reportEndToEndFlag(config.value(QStringLiteral("trainingReport")).toObject());
    if (flag >= 0) {
        return flag == 1;
    }
    flag = reportEndToEndFlag(loadUltralyticsTrainingReport(onnxPath));
    return flag == 1;
}
} // namespace

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
    if (configuredFamily == QStringLiteral("semantic_segmentation")
        || configuredBackend == QStringLiteral("smp_semantic_segmentation")) {
        return QStringLiteral("semantic_segmentation");
    }
    if (configuredFamily == QStringLiteral("yolo_obb")
        || configuredBackend == QStringLiteral("ultralytics_yolo_obb")) {
        return QStringLiteral("yolo_obb");
    }
    if (configuredFamily == QStringLiteral("yolo_segmentation")
        || configuredBackend == QStringLiteral("ultralytics_yolo_segment")) {
        return QStringLiteral("yolo_segmentation");
    }
    if (configuredFamily == QStringLiteral("ocr_recognition")
        || configuredBackend == QStringLiteral("paddleocr_rec_official")
        || configuredBackend == QStringLiteral("paddleocr_ppocrv4_rec")) {
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
    if (yoloReport.value(QStringLiteral("backend")).toString() == QStringLiteral("ultralytics_yolo_obb")) {
        return QStringLiteral("yolo_obb");
    }
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
            const QSize inputSize = yoloInputSizeFromShape(inputShape, exportConfig);
            if (channels != 3 || inputSize.isEmpty()) {
                if (error) {
                    *error = QStringLiteral("YOLO detection ONNX input shape must be [1, 3, height, width]");
                }
                return {};
            }

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
            if (yoloEndToEndFromExportConfig(onnxPath, exportConfig)) {
                return yoloEndToEndPredictionsFromOutput(
                    outputs.front().GetTensorData<float>(),
                    outputShape,
                    classNames,
                    inputSize,
                    transform,
                    options,
                    error);
            }
            return yoloPredictionsFromOutput(
                outputs.front().GetTensorData<float>(),
                outputShape,
                classNames,
                inputSize,
                transform,
                options,
                error);
        }

        if (error) {
            *error = QStringLiteral("Unsupported detection ONNX input shape. Production detection inference expects an official YOLO ONNX model with [1, 3, height, width] input.");
        }
        return {};
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

QVector<ObbPrediction> predictObbOnnxRuntime(
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
            *error = QStringLiteral("ONNX OBB model does not exist: %1").arg(onnxPath);
        }
        return {};
    }

    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for ONNX OBB prediction: %1").arg(imagePath);
        }
        return {};
    }

    try {
        const QJsonObject exportConfig = loadOnnxExportConfig(onnxPath);
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
                *error = QStringLiteral("YOLO OBB ONNX expects one input and at least one output");
            }
            return {};
        }
        const std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const QSize inputSize = yoloInputSizeFromShape(inputShape, exportConfig);
        if (inputShape.size() != 4 || inputShape.at(1) != 3 || inputSize.isEmpty()) {
            if (error) {
                *error = QStringLiteral("YOLO OBB ONNX input shape must be [1, 3, height, width]");
            }
            return {};
        }

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
        if (outputs.empty()) {
            if (error) {
                *error = QStringLiteral("YOLO OBB ONNX model returned no outputs");
            }
            return {};
        }

        const QStringList classNames = ultralyticsClassNames(onnxPath);
        const std::vector<int64_t> outputShape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        return yoloObbPredictionsFromOutput(
            outputs.front().GetTensorData<float>(),
            outputShape,
            classNames,
            inputSize,
            transform,
            options,
            error);
    } catch (const Ort::Exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX Runtime OBB inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    } catch (const std::exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX OBB inference failed: %1").arg(QString::fromUtf8(exception.what()));
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
        const QJsonObject exportConfig = loadOnnxExportConfig(onnxPath);
        const std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const QSize inputSize = yoloInputSizeFromShape(inputShape, exportConfig);
        if (inputShape.size() != 4 || inputShape.at(1) != 3 || inputSize.isEmpty()) {
            if (error) {
                *error = QStringLiteral("YOLO segmentation ONNX input shape must be [1, 3, height, width]");
            }
            return {};
        }

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
        if (yoloEndToEndFromExportConfig(onnxPath, exportConfig)) {
            return yoloEndToEndSegmentationPredictionsFromOutputs(
                outputs.at(boxesIndex).GetTensorData<float>(),
                outputs.at(boxesIndex).GetTensorTypeAndShapeInfo().GetShape(),
                outputs.at(prototypeIndex).GetTensorData<float>(),
                outputs.at(prototypeIndex).GetTensorTypeAndShapeInfo().GetShape(),
                classNames,
                inputSize,
                transform,
                options,
                error);
        }
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

SemanticSegmentationPrediction predictSemanticSegmentationOnnxRuntime(
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
            *error = QStringLiteral("ONNX semantic segmentation model does not exist: %1").arg(onnxPath);
        }
        return {};
    }
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for ONNX semantic segmentation prediction: %1").arg(imagePath);
        }
        return {};
    }

    try {
        const QJsonObject exportConfig = loadOnnxExportConfig(onnxPath);
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
                *error = QStringLiteral("Semantic segmentation ONNX expects one input and at least one output.");
            }
            return {};
        }
        const std::vector<int64_t> inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const QSize inputSize = semanticInputSizeFromShape(inputShape, exportConfig);
        if (inputShape.size() != 4 || inputShape.at(1) != 3 || inputSize.isEmpty()) {
            if (error) {
                *error = QStringLiteral("Semantic segmentation ONNX input shape must be [1, 3, height, width].");
            }
            return {};
        }

        QVector<float> input = semanticImageTensor(image, inputSize, exportConfig);
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
        if (outputs.empty()) {
            if (error) {
                *error = QStringLiteral("Semantic segmentation ONNX did not return outputs.");
            }
            return {};
        }

        const std::vector<int64_t> outputShape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() != 4 || outputShape.at(0) != 1 || outputShape.at(1) <= 0 || outputShape.at(2) <= 0 || outputShape.at(3) <= 0) {
            if (error) {
                *error = QStringLiteral("Semantic segmentation ONNX output must be logits shaped [1, classes, height, width].");
            }
            return {};
        }
        const int classCount = static_cast<int>(outputShape.at(1));
        const int outputHeight = static_cast<int>(outputShape.at(2));
        const int outputWidth = static_cast<int>(outputShape.at(3));
        if (classCount > 255) {
            if (error) {
                *error = QStringLiteral("Semantic segmentation class count exceeds 255; Mask PNG runtime supports one-byte class ids.");
            }
            return {};
        }

        QStringList classNames = stringListFromArray(exportConfig.value(QStringLiteral("classNames")).toArray());
        while (classNames.size() < classCount) {
            classNames.append(QStringLiteral("class_%1").arg(classNames.size()));
        }

        const float* logits = outputs.front().GetTensorData<float>();
        QImage mask(outputWidth, outputHeight, QImage::Format_Indexed8);
        mask.setColorTable(grayscaleColorTable());
        for (int y = 0; y < outputHeight; ++y) {
            uchar* line = mask.scanLine(y);
            for (int x = 0; x < outputWidth; ++x) {
                int bestClass = 0;
                float bestValue = logits[y * outputWidth + x];
                for (int classId = 1; classId < classCount; ++classId) {
                    const float value = logits[classId * outputHeight * outputWidth + y * outputWidth + x];
                    if (value > bestValue) {
                        bestValue = value;
                        bestClass = classId;
                    }
                }
                line[x] = static_cast<uchar>(qBound(0, bestClass, 255));
            }
        }
        QImage sourceMask = mask.scaled(image.size(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
        if (sourceMask.format() != QImage::Format_Indexed8) {
            sourceMask = sourceMask.convertToFormat(QImage::Format_Indexed8);
            sourceMask.setColorTable(grayscaleColorTable());
        }

        SemanticSegmentationPrediction prediction;
        prediction.mask = sourceMask;
        prediction.classNames = classNames;
        prediction.pixelCounts = pixelCountsForMask(sourceMask);
        prediction.sourceSize = image.size();
        prediction.modelSize = QSize(outputWidth, outputHeight);
        return prediction;
    } catch (const Ort::Exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX Runtime semantic segmentation inference failed: %1").arg(QString::fromUtf8(exception.what()));
        }
        return {};
    } catch (const std::exception& exception) {
        if (error) {
            *error = QStringLiteral("ONNX semantic segmentation inference failed: %1").arg(QString::fromUtf8(exception.what()));
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
