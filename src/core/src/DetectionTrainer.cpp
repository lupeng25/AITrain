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

} // namespace detection_detail

} // namespace aitrain
