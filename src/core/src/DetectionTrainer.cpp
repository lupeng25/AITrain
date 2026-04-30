#include "aitrain/core/DetectionTrainer.h"

#include "aitrain/core/Deployment.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLibrary>
#include <QtEndian>
#include <QtMath>
#include <algorithm>
#include <cstring>
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
namespace {

double clamp01(double value)
{
    return qBound(0.0, value, 1.0);
}

double safeLog(double value)
{
    return qLn(qMax(value, 1.0e-12));
}

double binaryCrossEntropy(double prediction, double target)
{
    const double clamped = qBound(1.0e-12, prediction, 1.0 - 1.0e-12);
    return -(target * qLn(clamped) + (1.0 - target) * qLn(1.0 - clamped));
}

QVector<double> softmax(const QVector<double>& logits)
{
    QVector<double> probabilities(logits.size(), 0.0);
    if (logits.isEmpty()) {
        return probabilities;
    }

    double maxLogit = logits.first();
    for (double value : logits) {
        maxLogit = qMax(maxLogit, value);
    }

    double sum = 0.0;
    for (int index = 0; index < logits.size(); ++index) {
        probabilities[index] = qExp(logits.at(index) - maxLogit);
        sum += probabilities[index];
    }
    if (sum <= 0.0) {
        return probabilities;
    }
    for (double& value : probabilities) {
        value /= sum;
    }
    return probabilities;
}

DetectionBox averageTargetBox(const QVector<DetectionBox>& boxes)
{
    DetectionBox target;
    if (boxes.isEmpty()) {
        target.classId = 0;
        target.xCenter = 0.5;
        target.yCenter = 0.5;
        target.width = 0.1;
        target.height = 0.1;
        return target;
    }

    for (const DetectionBox& box : boxes) {
        target.xCenter += box.xCenter;
        target.yCenter += box.yCenter;
        target.width += box.width;
        target.height += box.height;
    }
    const double count = static_cast<double>(boxes.size());
    target.classId = boxes.first().classId;
    target.xCenter /= count;
    target.yCenter /= count;
    target.width /= count;
    target.height /= count;
    return target;
}

double squared(double value)
{
    return value * value;
}

double sigmoid(double value)
{
    return 1.0 / (1.0 + qExp(-value));
}

double logit(double value)
{
    const double clamped = qBound(1.0e-5, value, 1.0 - 1.0e-5);
    return qLn(clamped / (1.0 - clamped));
}

QJsonArray doubleArray(const QVector<double>& values)
{
    QJsonArray array;
    for (double value : values) {
        array.append(value);
    }
    return array;
}

QJsonObject boxObject(const DetectionBox& box)
{
    QJsonObject object;
    object.insert(QStringLiteral("classId"), box.classId);
    object.insert(QStringLiteral("xCenter"), box.xCenter);
    object.insert(QStringLiteral("yCenter"), box.yCenter);
    object.insert(QStringLiteral("width"), box.width);
    object.insert(QStringLiteral("height"), box.height);
    return object;
}

bool boxFromObject(const QJsonObject& object, DetectionBox* box, QString* error)
{
    if (!box) {
        if (error) {
            *error = QStringLiteral("DetectionBox output is null");
        }
        return false;
    }
    if (!object.contains(QStringLiteral("classId"))
        || !object.contains(QStringLiteral("xCenter"))
        || !object.contains(QStringLiteral("yCenter"))
        || !object.contains(QStringLiteral("width"))
        || !object.contains(QStringLiteral("height"))) {
        if (error) {
            *error = QStringLiteral("Checkpoint priorBox is incomplete");
        }
        return false;
    }
    box->classId = object.value(QStringLiteral("classId")).toInt(-1);
    box->xCenter = object.value(QStringLiteral("xCenter")).toDouble();
    box->yCenter = object.value(QStringLiteral("yCenter")).toDouble();
    box->width = object.value(QStringLiteral("width")).toDouble();
    box->height = object.value(QStringLiteral("height")).toDouble();
    if (box->classId < 0 || box->width <= 0.0 || box->height <= 0.0) {
        if (error) {
            *error = QStringLiteral("Checkpoint priorBox values are invalid");
        }
        return false;
    }
    return true;
}

QString tinyDetectorBackendId()
{
    return QStringLiteral("tiny_linear_detector");
}

QString phase8DetectionModelFamily()
{
    return QStringLiteral("yolo_style_detection_scaffold");
}

QString yoloStyleLibTorchBackendId()
{
    return QStringLiteral("yolo_style_libtorch");
}

QString normalizedDetectionTrainingBackend(const QString& backend)
{
    const QString normalized = backend.trimmed().toLower();
    if (normalized.isEmpty() || normalized == QStringLiteral("auto")) {
        return tinyDetectorBackendId();
    }
    return normalized;
}

QJsonObject tinyDetectorModelArchitecture(int gridSize, int featureCount)
{
    QJsonArray losses;
    losses.append(QStringLiteral("objectness_bce"));
    losses.append(QStringLiteral("class_cross_entropy"));
    losses.append(QStringLiteral("box_mse"));

    return QJsonObject{
        {QStringLiteral("family"), QStringLiteral("tiny_linear_detector_scaffold")},
        {QStringLiteral("targetFamily"), QStringLiteral("yolo_style_detection")},
        {QStringLiteral("backbone"), QStringLiteral("handcrafted_grid_features")},
        {QStringLiteral("neck"), QStringLiteral("none")},
        {QStringLiteral("head"), QStringLiteral("linear_objectness_class_box")},
        {QStringLiteral("optimizer"), QStringLiteral("manual_sgd")},
        {QStringLiteral("losses"), losses},
        {QStringLiteral("gridSize"), gridSize},
        {QStringLiteral("featureCount"), featureCount},
        {QStringLiteral("realYoloStyleTraining"), false}
    };
}

QJsonObject phase8ScaffoldMetadata(const QString& requestedBackend)
{
    return QJsonObject{
        {QStringLiteral("status"), QStringLiteral("scaffold_backend")},
        {QStringLiteral("requestedBackend"), requestedBackend.trimmed().isEmpty()
            ? QStringLiteral("auto")
            : requestedBackend.trimmed()},
        {QStringLiteral("activeBackend"), tinyDetectorBackendId()},
        {QStringLiteral("nextBackend"), yoloStyleLibTorchBackendId()},
        {QStringLiteral("realYoloStyleTraining"), false},
        {QStringLiteral("message"), QStringLiteral("Phase 8 admission metadata is present; real YOLO-style LibTorch training is not implemented in this build.")}
    };
}

struct TinyDetectorModel {
    int classCount = 0;
    int gridSize = 4;
    int featureCount = 7;
    QVector<double> objectnessWeights;
    QVector<double> classWeights;
    QVector<double> boxWeights;
};

struct TinyDetectorCellForward {
    int cellIndex = 0;
    QVector<double> features;
    double objectnessLogit = 0.0;
    double objectness = 0.0;
    QVector<double> classLogits;
    QVector<double> classProbabilities;
    DetectionBox box;
};

struct TinyDetectorForward {
    QVector<TinyDetectorCellForward> cells;
    TinyDetectorCellForward best;
};

int weightIndex(int row, int column, int featureCount)
{
    return row * featureCount + column;
}

int bestClass(const QVector<double>& logits);

int targetCellIndex(const DetectionBox& box, int gridSize)
{
    const int safeGrid = qMax(1, gridSize);
    const int column = qBound(0, static_cast<int>(box.xCenter * safeGrid), safeGrid - 1);
    const int row = qBound(0, static_cast<int>(box.yCenter * safeGrid), safeGrid - 1);
    return row * safeGrid + column;
}

TinyDetectorModel createTinyDetectorModel(int classCount, int gridSize)
{
    TinyDetectorModel model;
    model.classCount = classCount;
    model.gridSize = qMax(1, gridSize);
    model.objectnessWeights = QVector<double>(model.featureCount, 0.0);
    model.objectnessWeights[0] = logit(0.05);
    model.classWeights = QVector<double>(classCount * model.featureCount, 0.0);
    model.boxWeights = QVector<double>(4 * model.featureCount, 0.0);
    model.boxWeights[weightIndex(0, 0, model.featureCount)] = logit(0.5);
    model.boxWeights[weightIndex(1, 0, model.featureCount)] = logit(0.5);
    model.boxWeights[weightIndex(2, 0, model.featureCount)] = logit(0.25);
    model.boxWeights[weightIndex(3, 0, model.featureCount)] = logit(0.25);
    return model;
}

QVector<double> imageFeatures(const QImage& input, int cellIndex, int gridSize)
{
    QImage image = input.convertToFormat(QImage::Format_RGB888);
    QVector<double> features;
    const int safeGrid = qMax(1, gridSize);
    const int row = cellIndex / safeGrid;
    const int column = cellIndex % safeGrid;
    features << 1.0 << 0.0 << 0.0 << 0.0 << 1.0
             << (static_cast<double>(column) + 0.5) / static_cast<double>(safeGrid)
             << (static_cast<double>(row) + 0.5) / static_cast<double>(safeGrid);
    if (image.isNull() || image.width() <= 0 || image.height() <= 0) {
        return features;
    }

    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    const int pixels = image.width() * image.height();
    for (int y = 0; y < image.height(); ++y) {
        const uchar* line = image.constScanLine(y);
        for (int x = 0; x < image.width(); ++x) {
            red += line[x * 3];
            green += line[x * 3 + 1];
            blue += line[x * 3 + 2];
        }
    }

    features[1] = red / static_cast<double>(pixels * 255);
    features[2] = green / static_cast<double>(pixels * 255);
    features[3] = blue / static_cast<double>(pixels * 255);
    features[4] = static_cast<double>(image.width()) / static_cast<double>(qMax(1, image.height()));
    return features;
}

QVector<DetectionBox> horizontalFlipBoxes(QVector<DetectionBox> boxes)
{
    for (DetectionBox& box : boxes) {
        box.xCenter = clamp01(1.0 - box.xCenter);
    }
    return boxes;
}

QImage brightnessJitterImage(const QImage& input, double factor)
{
    QImage image = input.convertToFormat(QImage::Format_RGB888);
    for (int y = 0; y < image.height(); ++y) {
        uchar* line = image.scanLine(y);
        for (int x = 0; x < image.width(); ++x) {
            line[x * 3] = static_cast<uchar>(qBound(0, qRound(line[x * 3] * factor), 255));
            line[x * 3 + 1] = static_cast<uchar>(qBound(0, qRound(line[x * 3 + 1] * factor), 255));
            line[x * 3 + 2] = static_cast<uchar>(qBound(0, qRound(line[x * 3 + 2] * factor), 255));
        }
    }
    return image;
}

TinyDetectorForward forwardTinyDetector(const TinyDetectorModel& model, const QImage& image)
{
    TinyDetectorForward output;
    const int cells = qMax(1, model.gridSize * model.gridSize);
    output.cells.reserve(cells);
    for (int cellIndex = 0; cellIndex < cells; ++cellIndex) {
        TinyDetectorCellForward cell;
        cell.cellIndex = cellIndex;
        cell.features = imageFeatures(image, cellIndex, model.gridSize);

        for (int featureIndex = 0; featureIndex < model.featureCount; ++featureIndex) {
            cell.objectnessLogit += model.objectnessWeights.at(featureIndex) * cell.features.at(featureIndex);
        }
        cell.objectness = sigmoid(cell.objectnessLogit);

        cell.classLogits = QVector<double>(model.classCount, 0.0);
        for (int classIndex = 0; classIndex < model.classCount; ++classIndex) {
            double value = 0.0;
            for (int featureIndex = 0; featureIndex < model.featureCount; ++featureIndex) {
                value += model.classWeights.at(weightIndex(classIndex, featureIndex, model.featureCount)) * cell.features.at(featureIndex);
            }
            cell.classLogits[classIndex] = value;
        }
        cell.classProbabilities = softmax(cell.classLogits);

        cell.box.classId = bestClass(cell.classLogits);
        double boxValues[4] = {};
        for (int coordinate = 0; coordinate < 4; ++coordinate) {
            double value = 0.0;
            for (int featureIndex = 0; featureIndex < model.featureCount; ++featureIndex) {
                value += model.boxWeights.at(weightIndex(coordinate, featureIndex, model.featureCount)) * cell.features.at(featureIndex);
            }
            boxValues[coordinate] = sigmoid(value);
        }
        cell.box.xCenter = clamp01(boxValues[0]);
        cell.box.yCenter = clamp01(boxValues[1]);
        cell.box.width = qBound(1.0e-4, boxValues[2], 1.0);
        cell.box.height = qBound(1.0e-4, boxValues[3], 1.0);
        output.cells.append(cell);
        if (cellIndex == 0 || cell.objectness > output.best.objectness) {
            output.best = cell;
        }
    }
    return output;
}

int bestClass(const QVector<double>& logits)
{
    if (logits.isEmpty()) {
        return 0;
    }
    int bestIndex = 0;
    for (int index = 1; index < logits.size(); ++index) {
        if (logits.at(index) > logits.at(bestIndex)) {
            bestIndex = index;
        }
    }
    return bestIndex;
}

double boxIou(const DetectionBox& left, const DetectionBox& right)
{
    const double leftX1 = left.xCenter - left.width / 2.0;
    const double leftY1 = left.yCenter - left.height / 2.0;
    const double leftX2 = left.xCenter + left.width / 2.0;
    const double leftY2 = left.yCenter + left.height / 2.0;

    const double rightX1 = right.xCenter - right.width / 2.0;
    const double rightY1 = right.yCenter - right.height / 2.0;
    const double rightX2 = right.xCenter + right.width / 2.0;
    const double rightY2 = right.yCenter + right.height / 2.0;

    const double intersectionWidth = qMax(0.0, qMin(leftX2, rightX2) - qMax(leftX1, rightX1));
    const double intersectionHeight = qMax(0.0, qMin(leftY2, rightY2) - qMax(leftY1, rightY1));
    const double intersection = intersectionWidth * intersectionHeight;
    const double leftArea = qMax(0.0, left.width) * qMax(0.0, left.height);
    const double rightArea = qMax(0.0, right.width) * qMax(0.0, right.height);
    const double unionArea = leftArea + rightArea - intersection;
    if (unionArea <= 0.0) {
        return 0.0;
    }
    return intersection / unionArea;
}

struct EvaluationResult {
    double precision = 0.0;
    double recall = 0.0;
    double map50 = 0.0;
};

struct EvaluationPrediction {
    int sampleIndex = 0;
    DetectionBox box;
    double confidence = 0.0;
};

EvaluationResult evaluateBaseline(
    const QString& datasetPath,
    const TinyDetectorModel& model,
    const QSize& imageSize,
    const DetectionDataset& trainDataset)
{
    DetectionDataset evalDataset;
    QString error;
    const DetectionDataset* dataset = &trainDataset;
    if (evalDataset.load(datasetPath, QStringLiteral("val"), &error)) {
        dataset = &evalDataset;
    }

    int targets = 0;
    QVector<EvaluationPrediction> predictions;
    QVector<QVector<bool>> matchedTargets;

    const QVector<DetectionSample> samples = dataset->samples();
    for (int sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
        const DetectionSample& sample = samples.at(sampleIndex);
        QImage image(sample.imagePath);
        if (image.isNull()) {
            continue;
        }
        const TinyDetectorForward output = forwardTinyDetector(model, letterboxImage(image, imageSize));
        EvaluationPrediction prediction;
        prediction.sampleIndex = sampleIndex;
        prediction.box = output.best.box;
        const int classId = prediction.box.classId;
        const double classConfidence = classId >= 0 && classId < output.best.classProbabilities.size()
            ? output.best.classProbabilities.at(classId)
            : 0.0;
        prediction.confidence = output.best.objectness * classConfidence;
        predictions.append(prediction);
        targets += sample.boxes.size();
        matchedTargets.append(QVector<bool>(sample.boxes.size(), false));
    }

    std::sort(predictions.begin(), predictions.end(), [](const EvaluationPrediction& left, const EvaluationPrediction& right) {
        return left.confidence > right.confidence;
    });

    int truePositive = 0;
    int falsePositive = 0;
    QVector<double> recalls;
    QVector<double> precisions;
    for (const EvaluationPrediction& prediction : predictions) {
        bool matched = false;
        const DetectionSample& sample = samples.at(prediction.sampleIndex);
        for (int targetIndex = 0; targetIndex < sample.boxes.size(); ++targetIndex) {
            const DetectionBox& target = sample.boxes.at(targetIndex);
            if (!matchedTargets[prediction.sampleIndex].at(targetIndex)
                && target.classId == prediction.box.classId
                && boxIou(prediction.box, target) >= 0.5) {
                matchedTargets[prediction.sampleIndex][targetIndex] = true;
                matched = true;
                break;
            }
        }
        if (matched) {
            ++truePositive;
        } else {
            ++falsePositive;
        }
        precisions.append(static_cast<double>(truePositive) / static_cast<double>(qMax(1, truePositive + falsePositive)));
        recalls.append(targets > 0 ? static_cast<double>(truePositive) / static_cast<double>(targets) : 0.0);
    }

    QVector<double> recallEnvelope;
    QVector<double> precisionEnvelope;
    recallEnvelope.append(0.0);
    precisionEnvelope.append(0.0);
    for (int index = 0; index < recalls.size(); ++index) {
        recallEnvelope.append(recalls.at(index));
        precisionEnvelope.append(precisions.at(index));
    }
    recallEnvelope.append(1.0);
    precisionEnvelope.append(0.0);
    for (int index = precisionEnvelope.size() - 2; index >= 0; --index) {
        precisionEnvelope[index] = qMax(precisionEnvelope.at(index), precisionEnvelope.at(index + 1));
    }
    double ap50 = 0.0;
    for (int index = 1; index < recallEnvelope.size(); ++index) {
        const double recallDelta = recallEnvelope.at(index) - recallEnvelope.at(index - 1);
        if (recallDelta > 0.0) {
            ap50 += recallDelta * precisionEnvelope.at(index);
        }
    }

    EvaluationResult result;
    result.precision = !predictions.isEmpty() ? static_cast<double>(truePositive) / static_cast<double>(predictions.size()) : 0.0;
    result.recall = targets > 0 ? static_cast<double>(truePositive) / static_cast<double>(targets) : 0.0;
    result.map50 = ap50;
    return result;
}

QStringList stringListFromArray(const QJsonArray& array)
{
    QStringList values;
    for (const QJsonValue& value : array) {
        values.append(value.toString());
    }
    return values;
}

QVector<double> doubleVectorFromArray(const QJsonArray& array)
{
    QVector<double> values;
    for (const QJsonValue& value : array) {
        values.append(value.toDouble());
    }
    return values;
}

QVector<float> tinyDetectorFeatureInput(const QImage& image, int cellCount, int featureCount, int gridSize, QString* error)
{
    if (featureCount != 7) {
        if (error) {
            *error = QStringLiteral("ONNX tiny detector expects 7 input features but model declares %1").arg(featureCount);
        }
        return {};
    }
    if (cellCount <= 0) {
        if (error) {
            *error = QStringLiteral("ONNX tiny detector input cell count is invalid");
        }
        return {};
    }

    QVector<float> input;
    input.reserve(cellCount * featureCount);
    for (int cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
        const QVector<double> features = imageFeatures(image, cellIndex, gridSize);
        for (double feature : features) {
            input.append(static_cast<float>(feature));
        }
    }
    return input;
}

QVector<DetectionPrediction> tinyDetectorPredictionsFromOutputs(
    const float* objectness,
    const float* classProbabilities,
    const float* boxes,
    int cellCount,
    int classCount,
    const QStringList& classNames,
    const DetectionInferenceOptions& options)
{
    QVector<DetectionPrediction> predictions;
    predictions.reserve(cellCount);
    for (int cellIndex = 0; cellIndex < cellCount; ++cellIndex) {
        int bestClassIndex = 0;
        for (int candidate = 1; candidate < classCount; ++candidate) {
            if (classProbabilities[cellIndex * classCount + candidate] > classProbabilities[cellIndex * classCount + bestClassIndex]) {
                bestClassIndex = candidate;
            }
        }

        DetectionPrediction prediction;
        prediction.box.classId = bestClassIndex;
        prediction.box.xCenter = clamp01(boxes[cellIndex * 4]);
        prediction.box.yCenter = clamp01(boxes[cellIndex * 4 + 1]);
        prediction.box.width = qBound(1.0e-4, static_cast<double>(boxes[cellIndex * 4 + 2]), 1.0);
        prediction.box.height = qBound(1.0e-4, static_cast<double>(boxes[cellIndex * 4 + 3]), 1.0);
        prediction.objectness = qBound(0.0, static_cast<double>(objectness[cellIndex]), 1.0);
        prediction.confidence = prediction.objectness
            * qBound(0.0, static_cast<double>(classProbabilities[cellIndex * classCount + bestClassIndex]), 1.0);
        prediction.className = bestClassIndex >= 0 && bestClassIndex < classNames.size()
            ? classNames.at(bestClassIndex)
            : QStringLiteral("class_%1").arg(bestClassIndex);
        predictions.append(prediction);
    }
    return postProcessDetectionPredictions(predictions, options);
}

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

QString onnxExportReportPath(const QString& onnxPath)
{
    const QFileInfo info(onnxPath);
    return info.absoluteDir().filePath(QStringLiteral("%1.aitrain-export.json").arg(info.completeBaseName()));
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

} // namespace

DetectionTrainingResult trainDetectionBaseline(
    const QString& datasetPath,
    const DetectionTrainingOptions& options,
    const DetectionTrainingCallback& callback)
{
    DetectionTrainingResult result;
    const QString trainingBackend = normalizedDetectionTrainingBackend(options.trainingBackend);
    result.trainingBackend = trainingBackend;
    result.modelFamily = phase8DetectionModelFamily();
    result.scaffold = trainingBackend == tinyDetectorBackendId();
    result.modelArchitecture = tinyDetectorModelArchitecture(qBound(1, options.gridSize, 16), 7);
    if (trainingBackend != tinyDetectorBackendId()) {
        result.error = QStringLiteral("Detection training backend '%1' is not available in this build. Phase 8 currently exposes the tiny_linear_detector scaffold; real YOLO-style LibTorch training is not implemented yet.")
            .arg(trainingBackend);
        return result;
    }

    QString error;
    DetectionDataset dataset;
    if (!dataset.load(datasetPath, QStringLiteral("train"), &error)) {
        result.error = error;
        return result;
    }
    if (dataset.info().classCount <= 0) {
        result.error = QStringLiteral("Detection dataset has no classes");
        return result;
    }

    const int epochs = qMax(1, options.epochs);
    const int batchSize = qMax(1, options.batchSize);
    const int stepsThisRun = epochs * ((dataset.size() + batchSize - 1) / batchSize);
    const QSize imageSize = options.imageSize.isValid() && !options.imageSize.isEmpty()
        ? options.imageSize
        : QSize(320, 320);
    const double learningRate = qBound(1.0e-5, options.learningRate, 1.0);
    const int gridSize = qBound(1, options.gridSize, 16);

    QString outputPath = options.outputPath;
    if (outputPath.isEmpty()) {
        outputPath = QDir(datasetPath).filePath(QStringLiteral("runs/minimal_detection_baseline"));
    }
    if (!QDir().mkpath(outputPath)) {
        result.error = QStringLiteral("Cannot create training output directory: %1").arg(outputPath);
        return result;
    }

    int step = 0;
    int totalSteps = stepsThisRun;
    TinyDetectorModel model = createTinyDetectorModel(dataset.info().classCount, gridSize);
    if (!options.resumeCheckpointPath.isEmpty()) {
        DetectionBaselineCheckpoint checkpoint;
        if (!loadDetectionBaselineCheckpoint(options.resumeCheckpointPath, &checkpoint, &error)) {
            result.error = error;
            return result;
        }
        if (checkpoint.type != QStringLiteral("tiny_linear_detector")
            || checkpoint.featureCount <= 0
            || checkpoint.classWeights.size() % checkpoint.featureCount != 0
            || checkpoint.classWeights.size() / checkpoint.featureCount != dataset.info().classCount
            || checkpoint.boxWeights.size() != 4 * checkpoint.featureCount
            || checkpoint.objectnessWeights.size() != checkpoint.featureCount) {
            result.error = QStringLiteral("Resume checkpoint is not compatible with the current detection dataset");
            return result;
        }
        model.featureCount = checkpoint.featureCount;
        model.gridSize = qMax(1, checkpoint.gridSize);
        model.classCount = dataset.info().classCount;
        model.objectnessWeights = checkpoint.objectnessWeights;
        model.classWeights = checkpoint.classWeights;
        model.boxWeights = checkpoint.boxWeights;
        step = qMax(0, checkpoint.steps);
        totalSteps = step + stepsThisRun;
    }
    result.modelArchitecture = tinyDetectorModelArchitecture(model.gridSize, model.featureCount);

    double finalLoss = 0.0;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        DetectionDataLoader loader(dataset, batchSize, imageSize);
        while (loader.hasNext()) {
            DetectionBatch batch;
            if (!loader.next(&batch, &error)) {
                result.error = error;
                return result;
            }

            double batchClassLoss = 0.0;
            double batchBoxLoss = 0.0;
            double batchObjectnessLoss = 0.0;
            int targetCount = 0;
            auto trainSample = [&](const QImage& image, const QVector<DetectionBox>& boxes) -> bool {
                const DetectionBox target = averageTargetBox(boxes);
                if (target.classId < 0 || target.classId >= model.classCount) {
                    result.error = QStringLiteral("Training target class is out of range");
                    return false;
                }

                const TinyDetectorForward output = forwardTinyDetector(model, image);
                const int positiveCell = targetCellIndex(target, model.gridSize);
                for (const TinyDetectorCellForward& cell : output.cells) {
                    const double expectedObjectness = cell.cellIndex == positiveCell ? 1.0 : 0.0;
                    batchObjectnessLoss += binaryCrossEntropy(cell.objectness, expectedObjectness);
                    const double objectnessGradient = cell.objectness - expectedObjectness;
                    for (int featureIndex = 0; featureIndex < model.featureCount; ++featureIndex) {
                        model.objectnessWeights[featureIndex] -= learningRate * objectnessGradient * cell.features.at(featureIndex);
                    }
                }

                const TinyDetectorCellForward& positiveOutput = output.cells.at(positiveCell);
                batchClassLoss += -safeLog(positiveOutput.classProbabilities.at(target.classId));
                for (int classIndex = 0; classIndex < model.classCount; ++classIndex) {
                    const double expected = classIndex == target.classId ? 1.0 : 0.0;
                    const double classGradient = positiveOutput.classProbabilities.at(classIndex) - expected;
                    for (int featureIndex = 0; featureIndex < model.featureCount; ++featureIndex) {
                        model.classWeights[weightIndex(classIndex, featureIndex, model.featureCount)]
                            -= learningRate * classGradient * positiveOutput.features.at(featureIndex);
                    }
                }

                const double boxLoss = (
                    squared(positiveOutput.box.xCenter - target.xCenter)
                    + squared(positiveOutput.box.yCenter - target.yCenter)
                    + squared(positiveOutput.box.width - target.width)
                    + squared(positiveOutput.box.height - target.height)) / 4.0;
                batchBoxLoss += boxLoss;

                const double predictedValues[4] = {
                    positiveOutput.box.xCenter,
                    positiveOutput.box.yCenter,
                    positiveOutput.box.width,
                    positiveOutput.box.height
                };
                const double targetValues[4] = {
                    target.xCenter,
                    target.yCenter,
                    target.width,
                    target.height
                };
                for (int coordinate = 0; coordinate < 4; ++coordinate) {
                    const double boxGradient = 0.5
                        * (predictedValues[coordinate] - targetValues[coordinate])
                        * predictedValues[coordinate]
                        * (1.0 - predictedValues[coordinate]);
                    for (int featureIndex = 0; featureIndex < model.featureCount; ++featureIndex) {
                        model.boxWeights[weightIndex(coordinate, featureIndex, model.featureCount)]
                            -= learningRate * boxGradient * positiveOutput.features.at(featureIndex);
                    }
                }
                ++targetCount;
                return true;
            };

            for (int sampleIndex = 0; sampleIndex < batch.boxes.size(); ++sampleIndex) {
                const QImage image = batch.images.at(sampleIndex);
                const QVector<DetectionBox> boxes = batch.boxes.at(sampleIndex);
                if (!trainSample(image, boxes)) {
                    return result;
                }
                if (options.horizontalFlip) {
                    if (!trainSample(image.mirrored(true, false), horizontalFlipBoxes(boxes))) {
                        return result;
                    }
                }
                if (options.colorJitter) {
                    if (!trainSample(brightnessJitterImage(image, 1.08), boxes)) {
                        return result;
                    }
                }
            }

            if (targetCount == 0) {
                continue;
            }

            ++step;
            batchClassLoss /= static_cast<double>(targetCount);
            batchBoxLoss /= static_cast<double>(targetCount);
            batchObjectnessLoss /= static_cast<double>(qMax(1, targetCount * model.gridSize * model.gridSize));
            finalLoss = batchObjectnessLoss + batchClassLoss + batchBoxLoss;

            DetectionTrainingMetrics metrics;
            metrics.epoch = epoch;
            metrics.step = step;
            metrics.totalSteps = totalSteps;
            metrics.objectnessLoss = batchObjectnessLoss;
            metrics.classLoss = batchClassLoss;
            metrics.boxLoss = batchBoxLoss;
            metrics.loss = finalLoss;
            if (callback && !callback(metrics)) {
                result.error = QStringLiteral("Training canceled");
                result.steps = step;
                result.finalLoss = finalLoss;
                return result;
            }
        }
    }

    const EvaluationResult evaluation = evaluateBaseline(datasetPath, model, imageSize, dataset);
    QImage neutralImage(imageSize, QImage::Format_RGB888);
    neutralImage.fill(Qt::gray);
    const TinyDetectorForward neutralOutput = forwardTinyDetector(model, neutralImage);

    QJsonObject checkpoint;
    checkpoint.insert(QStringLiteral("type"), tinyDetectorBackendId());
    checkpoint.insert(QStringLiteral("checkpointSchemaVersion"), 2);
    checkpoint.insert(QStringLiteral("trainingBackend"), tinyDetectorBackendId());
    checkpoint.insert(QStringLiteral("modelFamily"), phase8DetectionModelFamily());
    checkpoint.insert(QStringLiteral("scaffold"), true);
    checkpoint.insert(QStringLiteral("modelArchitecture"), tinyDetectorModelArchitecture(model.gridSize, model.featureCount));
    checkpoint.insert(QStringLiteral("phase8"), phase8ScaffoldMetadata(options.trainingBackend));
    checkpoint.insert(QStringLiteral("datasetPath"), QDir::cleanPath(datasetPath));
    checkpoint.insert(QStringLiteral("resumeFrom"), options.resumeCheckpointPath);
    checkpoint.insert(QStringLiteral("epochs"), epochs);
    checkpoint.insert(QStringLiteral("batchSize"), batchSize);
    checkpoint.insert(QStringLiteral("imageWidth"), imageSize.width());
    checkpoint.insert(QStringLiteral("imageHeight"), imageSize.height());
    checkpoint.insert(QStringLiteral("gridSize"), model.gridSize);
    checkpoint.insert(QStringLiteral("horizontalFlip"), options.horizontalFlip);
    checkpoint.insert(QStringLiteral("colorJitter"), options.colorJitter);
    checkpoint.insert(QStringLiteral("featureCount"), model.featureCount);
    checkpoint.insert(QStringLiteral("steps"), step);
    checkpoint.insert(QStringLiteral("finalLoss"), finalLoss);
    checkpoint.insert(QStringLiteral("precision"), evaluation.precision);
    checkpoint.insert(QStringLiteral("recall"), evaluation.recall);
    checkpoint.insert(QStringLiteral("mAP50"), evaluation.map50);
    checkpoint.insert(QStringLiteral("classLogits"), doubleArray(neutralOutput.best.classLogits));
    checkpoint.insert(QStringLiteral("objectnessWeights"), doubleArray(model.objectnessWeights));
    checkpoint.insert(QStringLiteral("classWeights"), doubleArray(model.classWeights));
    checkpoint.insert(QStringLiteral("boxWeights"), doubleArray(model.boxWeights));
    checkpoint.insert(QStringLiteral("priorBox"), boxObject(neutralOutput.best.box));
    checkpoint.insert(QStringLiteral("classNames"), QJsonArray::fromStringList(dataset.info().classNames));

    const QString checkpointPath = QDir(outputPath).filePath(QStringLiteral("checkpoint_latest.aitrain"));
    QFile file(checkpointPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        result.error = QStringLiteral("Cannot write checkpoint: %1").arg(checkpointPath);
        return result;
    }
    file.write(QJsonDocument(checkpoint).toJson(QJsonDocument::Indented));
    file.close();

    result.ok = true;
    result.checkpointPath = checkpointPath;
    result.trainingBackend = tinyDetectorBackendId();
    result.modelFamily = phase8DetectionModelFamily();
    result.scaffold = true;
    result.modelArchitecture = tinyDetectorModelArchitecture(model.gridSize, model.featureCount);
    result.steps = step;
    result.finalLoss = finalLoss;
    result.precision = evaluation.precision;
    result.recall = evaluation.recall;
    result.map50 = evaluation.map50;
    return result;
}

bool loadDetectionBaselineCheckpoint(
    const QString& checkpointPath,
    DetectionBaselineCheckpoint* checkpoint,
    QString* error)
{
    if (!checkpoint) {
        if (error) {
            *error = QStringLiteral("DetectionBaselineCheckpoint output is null");
        }
        return false;
    }

    QFile file(checkpointPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("Cannot open checkpoint: %1").arg(checkpointPath);
        }
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Invalid checkpoint JSON: %1").arg(parseError.errorString());
        }
        return false;
    }

    const QJsonObject object = document.object();
    DetectionBaselineCheckpoint loaded;
    loaded.type = object.value(QStringLiteral("type")).toString();
    if (loaded.type != QStringLiteral("minimal_detection_baseline")
        && loaded.type != tinyDetectorBackendId()) {
        if (error) {
            *error = QStringLiteral("Unsupported detection checkpoint type: %1").arg(loaded.type);
        }
        return false;
    }

    loaded.checkpointSchemaVersion = object.value(QStringLiteral("checkpointSchemaVersion")).toInt(
        loaded.type == tinyDetectorBackendId() ? 2 : 1);
    loaded.trainingBackend = object.value(QStringLiteral("trainingBackend")).toString(loaded.type);
    if (loaded.trainingBackend.isEmpty()) {
        loaded.trainingBackend = loaded.type;
    }
    loaded.modelFamily = object.value(QStringLiteral("modelFamily")).toString(
        loaded.type == tinyDetectorBackendId() ? phase8DetectionModelFamily() : QStringLiteral("minimal_detection_baseline"));
    loaded.scaffold = object.contains(QStringLiteral("scaffold"))
        ? object.value(QStringLiteral("scaffold")).toBool(loaded.type == tinyDetectorBackendId())
        : loaded.type == tinyDetectorBackendId();
    loaded.modelArchitecture = object.value(QStringLiteral("modelArchitecture")).toObject();
    if (loaded.modelArchitecture.isEmpty() && loaded.type == tinyDetectorBackendId()) {
        loaded.modelArchitecture = tinyDetectorModelArchitecture(
            object.value(QStringLiteral("gridSize")).toInt(1),
            object.value(QStringLiteral("featureCount")).toInt(0));
    }
    loaded.phase8 = object.value(QStringLiteral("phase8")).toObject();
    if (loaded.phase8.isEmpty() && loaded.type == tinyDetectorBackendId()) {
        loaded.phase8 = phase8ScaffoldMetadata(QString());
    }
    loaded.datasetPath = object.value(QStringLiteral("datasetPath")).toString();
    loaded.imageSize = QSize(
        object.value(QStringLiteral("imageWidth")).toInt(),
        object.value(QStringLiteral("imageHeight")).toInt());
    loaded.gridSize = object.value(QStringLiteral("gridSize")).toInt(1);
    loaded.featureCount = object.value(QStringLiteral("featureCount")).toInt(0);
    loaded.steps = object.value(QStringLiteral("steps")).toInt();
    loaded.finalLoss = object.value(QStringLiteral("finalLoss")).toDouble();
    loaded.precision = object.value(QStringLiteral("precision")).toDouble();
    loaded.recall = object.value(QStringLiteral("recall")).toDouble();
    loaded.map50 = object.value(QStringLiteral("mAP50")).toDouble();
    loaded.classNames = stringListFromArray(object.value(QStringLiteral("classNames")).toArray());
    loaded.classLogits = doubleVectorFromArray(object.value(QStringLiteral("classLogits")).toArray());
    loaded.objectnessWeights = doubleVectorFromArray(object.value(QStringLiteral("objectnessWeights")).toArray());
    loaded.classWeights = doubleVectorFromArray(object.value(QStringLiteral("classWeights")).toArray());
    loaded.boxWeights = doubleVectorFromArray(object.value(QStringLiteral("boxWeights")).toArray());
    if (loaded.classLogits.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Checkpoint classLogits are empty");
        }
        return false;
    }
    if (!boxFromObject(object.value(QStringLiteral("priorBox")).toObject(), &loaded.priorBox, error)) {
        return false;
    }
    if (loaded.type == tinyDetectorBackendId()) {
        if (loaded.featureCount <= 0
            || loaded.classWeights.isEmpty()
            || loaded.boxWeights.isEmpty()
            || loaded.classWeights.size() % loaded.featureCount != 0
            || loaded.boxWeights.size() != 4 * loaded.featureCount) {
            if (error) {
                *error = QStringLiteral("Tiny detector checkpoint weights are invalid");
            }
            return false;
        }
        if (loaded.objectnessWeights.isEmpty()) {
            loaded.objectnessWeights = QVector<double>(loaded.featureCount, 0.0);
            loaded.objectnessWeights[0] = logit(0.5);
        }
        if (loaded.objectnessWeights.size() != loaded.featureCount) {
            if (error) {
                *error = QStringLiteral("Tiny detector objectness weights are invalid");
            }
            return false;
        }
    }

    *checkpoint = loaded;
    return true;
}

TinyDetectorModel modelFromCheckpoint(const DetectionBaselineCheckpoint& checkpoint)
{
    TinyDetectorModel model;
    model.featureCount = checkpoint.featureCount;
    model.gridSize = qMax(1, checkpoint.gridSize);
    model.classCount = checkpoint.featureCount > 0 ? checkpoint.classWeights.size() / checkpoint.featureCount : 0;
    model.objectnessWeights = checkpoint.objectnessWeights.isEmpty()
        ? QVector<double>(model.featureCount, 0.0)
        : checkpoint.objectnessWeights;
    model.classWeights = checkpoint.classWeights;
    model.boxWeights = checkpoint.boxWeights;
    return model;
}

QVector<DetectionPrediction> predictDetectionBaseline(
    const DetectionBaselineCheckpoint& checkpoint,
    const QString& imagePath,
    QString* error)
{
    return predictDetectionBaseline(checkpoint, imagePath, DetectionInferenceOptions(), error);
}

QVector<DetectionPrediction> predictDetectionBaseline(
    const DetectionBaselineCheckpoint& checkpoint,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error)
{
    if (checkpoint.type != QStringLiteral("minimal_detection_baseline")
        && checkpoint.type != QStringLiteral("tiny_linear_detector")) {
        if (error) {
            *error = QStringLiteral("Unsupported detection checkpoint type: %1").arg(checkpoint.type);
        }
        return {};
    }
    if (checkpoint.classLogits.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Checkpoint classLogits are empty");
        }
        return {};
    }
    if (!QFileInfo::exists(imagePath) || !QImageReader(imagePath).size().isValid()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for detection prediction: %1").arg(imagePath);
        }
        return {};
    }

    DetectionPrediction prediction;
    QVector<double> probabilities;
    int classId = 0;
    if (checkpoint.type == QStringLiteral("tiny_linear_detector")) {
        QImage image(imagePath);
        if (image.isNull()) {
            if (error) {
                *error = QStringLiteral("Cannot read image for detection prediction: %1").arg(imagePath);
            }
            return {};
        }
        const QSize targetSize = checkpoint.imageSize.isValid() && !checkpoint.imageSize.isEmpty()
            ? checkpoint.imageSize
            : image.size();
        const TinyDetectorForward output = forwardTinyDetector(modelFromCheckpoint(checkpoint), letterboxImage(image, targetSize));
        probabilities = output.best.classProbabilities;
        classId = output.best.box.classId;
        prediction.box = output.best.box;
        prediction.objectness = output.best.objectness;
    } else {
        probabilities = softmax(checkpoint.classLogits);
        classId = bestClass(checkpoint.classLogits);
        prediction.box = checkpoint.priorBox;
        prediction.box.classId = classId;
        prediction.objectness = 1.0;
    }
    prediction.confidence = (classId >= 0 && classId < probabilities.size() ? probabilities.at(classId) : 0.0) * prediction.objectness;
    if (classId >= 0 && classId < checkpoint.classNames.size()) {
        prediction.className = checkpoint.classNames.at(classId);
    } else {
        prediction.className = QStringLiteral("class_%1").arg(classId);
    }

    return postProcessDetectionPredictions({prediction}, options);
}

bool isOnnxRuntimeInferenceAvailable()
{
#ifdef AITRAIN_WITH_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

QJsonObject detectionTrainingBackendStatus()
{
    QJsonArray backends;
    backends.append(QJsonObject{
        {QStringLiteral("id"), tinyDetectorBackendId()},
        {QStringLiteral("available"), true},
        {QStringLiteral("scaffold"), true},
        {QStringLiteral("modelFamily"), phase8DetectionModelFamily()}
    });
    backends.append(QJsonObject{
        {QStringLiteral("id"), yoloStyleLibTorchBackendId()},
        {QStringLiteral("available"), false},
        {QStringLiteral("scaffold"), false},
        {QStringLiteral("modelFamily"), QStringLiteral("yolo_style_detection")},
        {QStringLiteral("message"), QStringLiteral("LibTorch YOLO-style detection training is planned for Phase 8 and is not implemented in this build.")}
    });

    return QJsonObject{
        {QStringLiteral("phase"), 8},
        {QStringLiteral("checkpointSchemaVersion"), 2},
        {QStringLiteral("activeBackend"), tinyDetectorBackendId()},
        {QStringLiteral("activeBackendScaffold"), true},
        {QStringLiteral("realYoloStyleTrainingAvailable"), false},
        {QStringLiteral("nextBackend"), yoloStyleLibTorchBackendId()},
        {QStringLiteral("availableBackends"), backends},
        {QStringLiteral("message"), QStringLiteral("Phase 8 can run locally through the tiny detector scaffold while the real YOLO-style LibTorch backend is implemented.")}
    };
}

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

        if (session.GetInputCount() != 1 || session.GetOutputCount() < 3) {
            if (error) {
                *error = QStringLiteral("ONNX tiny detector expects one input and at least three outputs");
            }
            return {};
        }

        Ort::TypeInfo inputType = session.GetInputTypeInfo(0);
        const std::vector<int64_t> inputShape = inputType.GetTensorTypeAndShapeInfo().GetShape();
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

QVector<DetectionPrediction> postProcessDetectionPredictions(
    const QVector<DetectionPrediction>& predictions,
    const DetectionInferenceOptions& options)
{
    if (options.maxDetections <= 0 || predictions.isEmpty()) {
        return {};
    }

    QVector<DetectionPrediction> candidates;
    candidates.reserve(predictions.size());
    for (const DetectionPrediction& prediction : predictions) {
        if (prediction.confidence >= options.confidenceThreshold) {
            candidates.append(prediction);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const DetectionPrediction& left, const DetectionPrediction& right) {
        return left.confidence > right.confidence;
    });

    const double iouThreshold = qBound(0.0, options.iouThreshold, 1.0);
    QVector<DetectionPrediction> selected;
    selected.reserve(qMin(options.maxDetections, candidates.size()));
    for (const DetectionPrediction& candidate : candidates) {
        bool suppressed = false;
        for (const DetectionPrediction& accepted : selected) {
            if (candidate.box.classId == accepted.box.classId && boxIou(candidate.box, accepted.box) > iouThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            selected.append(candidate);
            if (selected.size() >= options.maxDetections) {
                break;
            }
        }
    }
    return selected;
}

QJsonObject detectionPredictionToJson(const DetectionPrediction& prediction)
{
    QJsonObject box;
    box.insert(QStringLiteral("classId"), prediction.box.classId);
    box.insert(QStringLiteral("xCenter"), prediction.box.xCenter);
    box.insert(QStringLiteral("yCenter"), prediction.box.yCenter);
    box.insert(QStringLiteral("width"), prediction.box.width);
    box.insert(QStringLiteral("height"), prediction.box.height);

    QJsonObject object;
    object.insert(QStringLiteral("classId"), prediction.box.classId);
    object.insert(QStringLiteral("className"), prediction.className);
    object.insert(QStringLiteral("objectness"), prediction.objectness);
    object.insert(QStringLiteral("confidence"), prediction.confidence);
    object.insert(QStringLiteral("box"), box);
    return object;
}

QImage renderDetectionPredictions(
    const QString& imagePath,
    const QVector<DetectionPrediction>& predictions,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output = image.convertToFormat(QImage::Format_RGB888);
    const QColor color(220, 40, 40);
    const int thickness = qMax(2, output.width() / 240);
    auto drawPoint = [&output, &color](int x, int y) {
        if (x >= 0 && x < output.width() && y >= 0 && y < output.height()) {
            output.setPixelColor(x, y, color);
        }
    };
    for (const DetectionPrediction& prediction : predictions) {
        const double imageWidth = static_cast<double>(output.width());
        const double imageHeight = static_cast<double>(output.height());
        const int x = qRound((prediction.box.xCenter - prediction.box.width / 2.0) * imageWidth);
        const int y = qRound((prediction.box.yCenter - prediction.box.height / 2.0) * imageHeight);
        const int width = qRound(prediction.box.width * imageWidth);
        const int height = qRound(prediction.box.height * imageHeight);
        QRect rect(x, y, width, height);
        rect = rect.intersected(output.rect());
        if (rect.isEmpty()) {
            continue;
        }

        for (int offset = 0; offset < thickness; ++offset) {
            for (int px = rect.left(); px <= rect.right(); ++px) {
                drawPoint(px, rect.top() + offset);
                drawPoint(px, rect.bottom() - offset);
            }
            for (int py = rect.top(); py <= rect.bottom(); ++py) {
                drawPoint(rect.left() + offset, py);
                drawPoint(rect.right() - offset, py);
            }
        }
    }
    return output;
}

DetectionExportResult exportDetectionCheckpoint(
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format)
{
    DetectionExportResult result;
    const QString normalizedFormat = format.isEmpty() ? QStringLiteral("tiny_detector_json") : format.toLower();
    const bool tensorRtFormat = normalizedFormat == QStringLiteral("tensorrt")
        || normalizedFormat == QStringLiteral("tensorrt_fp16");
    result.format = normalizedFormat;
    result.sourceCheckpointPath = checkpointPath;
    if (normalizedFormat != QStringLiteral("tiny_detector_json")
        && normalizedFormat != QStringLiteral("onnx")
        && !tensorRtFormat) {
        if (normalizedFormat.startsWith(QStringLiteral("tensorrt"))) {
            result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
        } else {
            result.error = QStringLiteral("Unsupported detection export format: %1").arg(normalizedFormat);
        }
        return result;
    }

    QString error;
    DetectionBaselineCheckpoint checkpoint;
    if (!loadDetectionBaselineCheckpoint(checkpointPath, &checkpoint, &error)) {
        result.error = error;
        return result;
    }
    if (checkpoint.type != QStringLiteral("tiny_linear_detector")) {
        result.error = QStringLiteral("Only tiny_linear_detector checkpoints can be exported by this scaffold exporter");
        return result;
    }

    QString finalOutputPath = outputPath;
    if (finalOutputPath.isEmpty()) {
        finalOutputPath = QFileInfo(checkpointPath).absoluteDir().filePath(
            normalizedFormat == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (tensorRtFormat ? QStringLiteral("model.engine") : QStringLiteral("model.aitrain-export.json")));
    }
    if (QFileInfo(finalOutputPath).isDir()) {
        finalOutputPath = QDir(finalOutputPath).filePath(
            normalizedFormat == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (tensorRtFormat ? QStringLiteral("model.engine") : QStringLiteral("model.aitrain-export.json")));
    }
    if (!QDir().mkpath(QFileInfo(finalOutputPath).absolutePath())) {
        result.error = QStringLiteral("Cannot create export directory: %1").arg(QFileInfo(finalOutputPath).absolutePath());
        return result;
    }

    if (normalizedFormat == QStringLiteral("onnx")) {
        if (!writeTinyDetectorOnnxModel(checkpoint, finalOutputPath, &result.error)) {
            return result;
        }
        const QString reportPath = onnxExportReportPath(finalOutputPath);
        const QJsonObject config = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
        if (!writeJsonObject(reportPath, config, &result.error)) {
            return result;
        }
        result.ok = true;
        result.exportPath = finalOutputPath;
        result.reportPath = reportPath;
        result.config = config;
        return result;
    }

    if (tensorRtFormat) {
#ifndef AITRAIN_WITH_TENSORRT_SDK
        result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
        return result;
#else
        const QByteArray onnxModel = tinyDetectorOnnxModel(checkpoint, &result.error);
        if (onnxModel.isEmpty()) {
            return result;
        }
        const bool fp16 = normalizedFormat == QStringLiteral("tensorrt_fp16");
        if (!writeTensorRtEngineFromOnnx(onnxModel, finalOutputPath, fp16, &result.error)) {
            return result;
        }

        const QString reportPath = onnxExportReportPath(finalOutputPath);
        QJsonObject config = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
        config.insert(QStringLiteral("backend"), QStringLiteral("tensorrt_tiny_detector"));
        config.insert(QStringLiteral("tensorRt"), QJsonObject{
            {QStringLiteral("precision"), fp16 ? QStringLiteral("fp16") : QStringLiteral("fp32")},
            {QStringLiteral("workspaceBytes"), static_cast<double>(size_t{1} << 30)},
            {QStringLiteral("dynamicShape"), false},
            {QStringLiteral("engineCache"), true}
        });
        if (!writeJsonObject(reportPath, config, &result.error)) {
            return result;
        }
        result.ok = true;
        result.exportPath = finalOutputPath;
        result.reportPath = reportPath;
        result.config = config;
        return result;
#endif
    }

    QJsonObject exportObject = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
    exportObject.insert(QStringLiteral("sourceCheckpoint"), checkpointPath);
    exportObject.insert(QStringLiteral("note"), QStringLiteral("Scaffold export for AITrain tiny detector. This is not ONNX."));
    exportObject.insert(QStringLiteral("type"), checkpoint.type);
    exportObject.insert(QStringLiteral("datasetPath"), checkpoint.datasetPath);
    exportObject.insert(QStringLiteral("imageWidth"), checkpoint.imageSize.width());
    exportObject.insert(QStringLiteral("imageHeight"), checkpoint.imageSize.height());
    exportObject.insert(QStringLiteral("gridSize"), checkpoint.gridSize);
    exportObject.insert(QStringLiteral("featureCount"), checkpoint.featureCount);
    exportObject.insert(QStringLiteral("classNames"), QJsonArray::fromStringList(checkpoint.classNames));
    exportObject.insert(QStringLiteral("classLogits"), doubleArray(checkpoint.classLogits));
    exportObject.insert(QStringLiteral("objectnessWeights"), doubleArray(checkpoint.objectnessWeights));
    exportObject.insert(QStringLiteral("classWeights"), doubleArray(checkpoint.classWeights));
    exportObject.insert(QStringLiteral("boxWeights"), doubleArray(checkpoint.boxWeights));
    exportObject.insert(QStringLiteral("priorBox"), boxObject(checkpoint.priorBox));
    exportObject.insert(QStringLiteral("metrics"), QJsonObject{
        {QStringLiteral("finalLoss"), checkpoint.finalLoss},
        {QStringLiteral("precision"), checkpoint.precision},
        {QStringLiteral("recall"), checkpoint.recall},
        {QStringLiteral("mAP50"), checkpoint.map50}
    });

    QFile file(finalOutputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        result.error = QStringLiteral("Cannot write export artifact: %1").arg(finalOutputPath);
        return result;
    }
    file.write(QJsonDocument(exportObject).toJson(QJsonDocument::Indented));
    file.close();

    result.ok = true;
    result.exportPath = finalOutputPath;
    result.reportPath = finalOutputPath;
    result.config = exportObject;
    return result;
}

} // namespace aitrain
