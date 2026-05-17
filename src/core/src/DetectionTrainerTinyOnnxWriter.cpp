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
void appendProtoVarint(QByteArray* output, quint64 value)
{
    while (value >= 0x80) {
        output->append(static_cast<char>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    output->append(static_cast<char>(value));
}

void appendProtoKey(QByteArray* output, int fieldNumber, int wireType)
{
    appendProtoVarint(output, (static_cast<quint64>(fieldNumber) << 3) | static_cast<quint64>(wireType));
}

void appendProtoInt64(QByteArray* output, int fieldNumber, qint64 value)
{
    appendProtoKey(output, fieldNumber, 0);
    appendProtoVarint(output, static_cast<quint64>(value));
}

void appendProtoInt32(QByteArray* output, int fieldNumber, int value)
{
    appendProtoKey(output, fieldNumber, 0);
    appendProtoVarint(output, static_cast<quint64>(value));
}

void appendProtoBytes(QByteArray* output, int fieldNumber, const QByteArray& bytes)
{
    appendProtoKey(output, fieldNumber, 2);
    appendProtoVarint(output, static_cast<quint64>(bytes.size()));
    output->append(bytes);
}

void appendProtoString(QByteArray* output, int fieldNumber, const QString& value)
{
    appendProtoBytes(output, fieldNumber, value.toUtf8());
}

void appendProtoMessage(QByteArray* output, int fieldNumber, const QByteArray& message)
{
    appendProtoBytes(output, fieldNumber, message);
}

QByteArray onnxShapeDimension(qint64 value)
{
    QByteArray dimension;
    appendProtoInt64(&dimension, 1, value);
    return dimension;
}

QByteArray onnxShape(const QVector<qint64>& dimensions)
{
    QByteArray shape;
    for (qint64 dimension : dimensions) {
        appendProtoMessage(&shape, 1, onnxShapeDimension(dimension));
    }
    return shape;
}

QByteArray onnxTensorType(const QVector<qint64>& dimensions)
{
    QByteArray tensorType;
    appendProtoInt32(&tensorType, 1, 1); // TensorProto.FLOAT
    appendProtoMessage(&tensorType, 2, onnxShape(dimensions));
    return tensorType;
}

QByteArray onnxType(const QVector<qint64>& dimensions)
{
    QByteArray type;
    appendProtoMessage(&type, 1, onnxTensorType(dimensions));
    return type;
}

QByteArray onnxValueInfo(const QString& name, const QVector<qint64>& dimensions)
{
    QByteArray valueInfo;
    appendProtoString(&valueInfo, 1, name);
    appendProtoMessage(&valueInfo, 2, onnxType(dimensions));
    return valueInfo;
}

QByteArray floatRawData(const QVector<double>& values)
{
    QByteArray raw;
    raw.reserve(values.size() * static_cast<int>(sizeof(float)));
    for (double value : values) {
        const float asFloat = static_cast<float>(value);
        quint32 bits = 0;
        static_assert(sizeof(bits) == sizeof(asFloat), "Unexpected float size");
        std::memcpy(&bits, &asFloat, sizeof(bits));
        bits = qToLittleEndian(bits);
        raw.append(reinterpret_cast<const char*>(&bits), sizeof(bits));
    }
    return raw;
}

QByteArray onnxTensor(const QString& name, const QVector<qint64>& dimensions, const QVector<double>& values)
{
    QByteArray tensor;
    for (qint64 dimension : dimensions) {
        appendProtoInt64(&tensor, 1, dimension);
    }
    appendProtoInt32(&tensor, 2, 1); // TensorProto.FLOAT
    appendProtoString(&tensor, 8, name);
    appendProtoBytes(&tensor, 9, floatRawData(values));
    return tensor;
}

QByteArray onnxNode(const QString& name, const QString& opType, const QStringList& inputs, const QStringList& outputs)
{
    QByteArray node;
    for (const QString& input : inputs) {
        appendProtoString(&node, 1, input);
    }
    for (const QString& output : outputs) {
        appendProtoString(&node, 2, output);
    }
    appendProtoString(&node, 3, name);
    appendProtoString(&node, 4, opType);
    return node;
}

QByteArray onnxOpsetImport(qint64 version)
{
    QByteArray opset;
    appendProtoInt64(&opset, 2, version);
    return opset;
}

QVector<double> transposeLinearWeights(const QVector<double>& weights, int rows, int featureCount)
{
    QVector<double> transposed;
    transposed.reserve(weights.size());
    for (int featureIndex = 0; featureIndex < featureCount; ++featureIndex) {
        for (int row = 0; row < rows; ++row) {
            transposed.append(weights.at(weightIndex(row, featureIndex, featureCount)));
        }
    }
    return transposed;
}

QByteArray tinyDetectorOnnxModel(const DetectionBaselineCheckpoint& checkpoint, QString* error)
{
    const int featureCount = checkpoint.featureCount;
    const int classCount = featureCount > 0 ? checkpoint.classWeights.size() / featureCount : 0;
    const int cellCount = qMax(1, checkpoint.gridSize * checkpoint.gridSize);
    if (featureCount <= 0 || classCount <= 0
        || checkpoint.objectnessWeights.size() != featureCount
        || checkpoint.classWeights.size() != classCount * featureCount
        || checkpoint.boxWeights.size() != 4 * featureCount) {
        if (error) {
            *error = QStringLiteral("Tiny detector checkpoint weights cannot be represented as ONNX tensors");
        }
        return {};
    }

    QVector<double> objectnessWeights;
    objectnessWeights.reserve(featureCount);
    for (int featureIndex = 0; featureIndex < featureCount; ++featureIndex) {
        objectnessWeights.append(checkpoint.objectnessWeights.at(featureIndex));
    }

    QByteArray graph;
    appendProtoMessage(&graph, 1, onnxNode(
        QStringLiteral("objectness_gemm"),
        QStringLiteral("Gemm"),
        QStringList() << QStringLiteral("features") << QStringLiteral("objectness_weight"),
        QStringList() << QStringLiteral("objectness_logits")));
    appendProtoMessage(&graph, 1, onnxNode(
        QStringLiteral("objectness_sigmoid"),
        QStringLiteral("Sigmoid"),
        QStringList() << QStringLiteral("objectness_logits"),
        QStringList() << QStringLiteral("objectness")));
    appendProtoMessage(&graph, 1, onnxNode(
        QStringLiteral("class_gemm"),
        QStringLiteral("Gemm"),
        QStringList() << QStringLiteral("features") << QStringLiteral("class_weight"),
        QStringList() << QStringLiteral("class_logits")));
    appendProtoMessage(&graph, 1, onnxNode(
        QStringLiteral("class_softmax"),
        QStringLiteral("Softmax"),
        QStringList() << QStringLiteral("class_logits"),
        QStringList() << QStringLiteral("class_probabilities")));
    appendProtoMessage(&graph, 1, onnxNode(
        QStringLiteral("box_gemm"),
        QStringLiteral("Gemm"),
        QStringList() << QStringLiteral("features") << QStringLiteral("box_weight"),
        QStringList() << QStringLiteral("box_logits")));
    appendProtoMessage(&graph, 1, onnxNode(
        QStringLiteral("box_sigmoid"),
        QStringLiteral("Sigmoid"),
        QStringList() << QStringLiteral("box_logits"),
        QStringList() << QStringLiteral("boxes")));
    appendProtoString(&graph, 2, QStringLiteral("aitrain_tiny_detector"));
    appendProtoMessage(&graph, 5, onnxTensor(QStringLiteral("objectness_weight"), QVector<qint64>{featureCount, 1}, objectnessWeights));
    appendProtoMessage(&graph, 5, onnxTensor(QStringLiteral("class_weight"), QVector<qint64>{featureCount, classCount}, transposeLinearWeights(checkpoint.classWeights, classCount, featureCount)));
    appendProtoMessage(&graph, 5, onnxTensor(QStringLiteral("box_weight"), QVector<qint64>{featureCount, 4}, transposeLinearWeights(checkpoint.boxWeights, 4, featureCount)));
    appendProtoMessage(&graph, 11, onnxValueInfo(QStringLiteral("features"), QVector<qint64>{cellCount, featureCount}));
    appendProtoMessage(&graph, 12, onnxValueInfo(QStringLiteral("objectness"), QVector<qint64>{cellCount, 1}));
    appendProtoMessage(&graph, 12, onnxValueInfo(QStringLiteral("class_probabilities"), QVector<qint64>{cellCount, classCount}));
    appendProtoMessage(&graph, 12, onnxValueInfo(QStringLiteral("boxes"), QVector<qint64>{cellCount, 4}));

    QByteArray model;
    appendProtoInt64(&model, 1, 8); // ONNX IR version.
    appendProtoString(&model, 2, QStringLiteral("AITrain Studio"));
    appendProtoString(&model, 3, QStringLiteral("0.1.0"));
    appendProtoString(&model, 6, QStringLiteral("Tiny detector ONNX model core. Image preprocessing and NMS are handled by AITrain runtime code."));
    appendProtoMessage(&model, 7, graph);
    appendProtoMessage(&model, 8, onnxOpsetImport(13));
    return model;
}

bool writeTinyDetectorOnnxModel(const DetectionBaselineCheckpoint& checkpoint, const QString& outputPath, QString* error)
{
    const QByteArray model = tinyDetectorOnnxModel(checkpoint, error);
    if (model.isEmpty()) {
        return false;
    }

    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write ONNX export artifact: %1").arg(outputPath);
        }
        return false;
    }
    file.write(model);
    file.close();
    return true;
}

QJsonObject shapeObject(const QString& name, const QVector<qint64>& dimensions)
{
    QJsonArray shape;
    for (qint64 dimension : dimensions) {
        shape.append(static_cast<double>(dimension));
    }
    return QJsonObject{
        {QStringLiteral("name"), name},
        {QStringLiteral("dtype"), QStringLiteral("float32")},
        {QStringLiteral("shape"), shape}
    };
}

QJsonObject tinyDetectorExportConfig(
    const DetectionBaselineCheckpoint& checkpoint,
    const QString& checkpointPath,
    const QString& exportPath,
    const QString& format)
{
    const int classCount = checkpoint.featureCount > 0 ? checkpoint.classWeights.size() / checkpoint.featureCount : 0;
    const int cellCount = qMax(1, checkpoint.gridSize * checkpoint.gridSize);
    QJsonArray outputs;
    outputs.append(shapeObject(QStringLiteral("objectness"), QVector<qint64>{cellCount, 1}));
    outputs.append(shapeObject(QStringLiteral("class_probabilities"), QVector<qint64>{cellCount, classCount}));
    outputs.append(shapeObject(QStringLiteral("boxes"), QVector<qint64>{cellCount, 4}));

    return QJsonObject{
        {QStringLiteral("format"), format},
        {QStringLiteral("backend"), QStringLiteral("tiny_detector")},
        {QStringLiteral("checkpointSchemaVersion"), checkpoint.checkpointSchemaVersion},
        {QStringLiteral("trainingBackend"), checkpoint.trainingBackend},
        {QStringLiteral("modelFamily"), checkpoint.modelFamily},
        {QStringLiteral("scaffold"), checkpoint.scaffold},
        {QStringLiteral("modelArchitecture"), checkpoint.modelArchitecture},
        {QStringLiteral("phase8"), checkpoint.phase8},
        {QStringLiteral("sourceCheckpoint"), checkpointPath},
        {QStringLiteral("exportPath"), exportPath},
        {QStringLiteral("modelType"), checkpoint.type},
        {QStringLiteral("datasetPath"), checkpoint.datasetPath},
        {QStringLiteral("classNames"), QJsonArray::fromStringList(checkpoint.classNames)},
        {QStringLiteral("imageWidth"), checkpoint.imageSize.width()},
        {QStringLiteral("imageHeight"), checkpoint.imageSize.height()},
        {QStringLiteral("gridSize"), checkpoint.gridSize},
        {QStringLiteral("cellCount"), cellCount},
        {QStringLiteral("featureCount"), checkpoint.featureCount},
        {QStringLiteral("input"), shapeObject(QStringLiteral("features"), QVector<qint64>{cellCount, checkpoint.featureCount})},
        {QStringLiteral("outputs"), outputs},
        {QStringLiteral("shapeValidation"), QJsonObject{
            {QStringLiteral("ok"), checkpoint.featureCount > 0 && classCount > 0},
            {QStringLiteral("message"), QStringLiteral("Tiny detector ONNX graph exposes fixed feature, objectness, class probability, and box tensors.")}
        }},
        {QStringLiteral("postprocess"), QJsonObject{
            {QStringLiteral("nms"), QStringLiteral("AITrain runtime")},
            {QStringLiteral("classMapping"), QStringLiteral("sidecar classNames")}
        }},
        {QStringLiteral("metrics"), QJsonObject{
            {QStringLiteral("finalLoss"), checkpoint.finalLoss},
            {QStringLiteral("precision"), checkpoint.precision},
            {QStringLiteral("recall"), checkpoint.recall},
            {QStringLiteral("mAP50"), checkpoint.map50}
        }}
    };
}

} // namespace detection_detail

} // namespace aitrain
