#include "aitrain/core/DetectionTrainer.h"

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
namespace {

QJsonObject loadOnnxExportConfig(const QString& onnxPath);
QString onnxExportReportPath(const QString& onnxPath);

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

QString unquoteYamlScalar(QString value)
{
    value = value.trimmed();
    if ((value.startsWith(QLatin1Char('"')) && value.endsWith(QLatin1Char('"')))
        || (value.startsWith(QLatin1Char('\'')) && value.endsWith(QLatin1Char('\'')))) {
        value = value.mid(1, value.size() - 2);
    }
    return value;
}

QStringList classNamesFromYoloDataYaml(const QString& yamlPath)
{
    QFile file(yamlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    const QString text = QString::fromUtf8(file.readAll());
    QRegularExpression inlineNames(QStringLiteral("(?m)^\\s*names\\s*:\\s*\\[([^\\]]*)\\]"));
    const QRegularExpressionMatch inlineMatch = inlineNames.match(text);
    if (inlineMatch.hasMatch()) {
        QStringList names;
        for (const QString& raw : inlineMatch.captured(1).split(QLatin1Char(','))) {
            const QString name = unquoteYamlScalar(raw);
            if (!name.isEmpty()) {
                names.append(name);
            }
        }
        return names;
    }

    QStringList names;
    QRegularExpression blockItem(QStringLiteral("(?m)^\\s*(\\d+)\\s*:\\s*(.+)\\s*$"));
    QRegularExpressionMatchIterator iterator = blockItem.globalMatch(text);
    QMap<int, QString> indexedNames;
    while (iterator.hasNext()) {
        const QRegularExpressionMatch match = iterator.next();
        indexedNames.insert(match.captured(1).toInt(), unquoteYamlScalar(match.captured(2)));
    }
    for (auto it = indexedNames.constBegin(); it != indexedNames.constEnd(); ++it) {
        names.append(it.value());
    }
    return names;
}

QJsonObject loadUltralyticsTrainingReport(const QString& onnxPath)
{
    const QFileInfo onnxInfo(onnxPath);
    const QDir weightsDir = onnxInfo.absoluteDir();
    const QStringList candidates = {
        weightsDir.absoluteFilePath(QStringLiteral("ultralytics_training_report.json")),
        weightsDir.absoluteFilePath(QStringLiteral("../ultralytics_training_report.json")),
        weightsDir.absoluteFilePath(QStringLiteral("../../ultralytics_training_report.json")),
        weightsDir.absoluteFilePath(QStringLiteral("../../../ultralytics_training_report.json")),
        weightsDir.absoluteFilePath(QStringLiteral("../../../../ultralytics_training_report.json"))
    };
    for (const QString& candidate : candidates) {
        QFile file(QDir::cleanPath(candidate));
        if (!file.open(QIODevice::ReadOnly)) {
            continue;
        }
        const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
        if (document.isObject()) {
            const QString backend = document.object().value(QStringLiteral("backend")).toString();
            if (backend == QStringLiteral("ultralytics_yolo_detect")
                || backend == QStringLiteral("ultralytics_yolo_segment")) {
                return document.object();
            }
        }
    }
    return {};
}

QStringList ultralyticsClassNames(const QString& onnxPath)
{
    const QJsonObject tinyConfig = loadOnnxExportConfig(onnxPath);
    QStringList classNames = stringListFromArray(tinyConfig.value(QStringLiteral("classNames")).toArray());
    if (!classNames.isEmpty()) {
        return classNames;
    }

    const QJsonObject report = loadUltralyticsTrainingReport(onnxPath);
    const QString dataYaml = report.value(QStringLiteral("dataYaml")).toString();
    if (!dataYaml.isEmpty()) {
        classNames = classNamesFromYoloDataYaml(dataYaml);
    }
    if (classNames.isEmpty()) {
        classNames.append(QStringLiteral("class_0"));
    }
    return classNames;
}

QVector<float> yoloImageTensorFromLetterbox(const QImage& image, const QSize& inputSize, LetterboxTransform* transform)
{
    const QImage letterboxed = letterboxImage(image, inputSize, transform).convertToFormat(QImage::Format_RGB888);
    QVector<float> tensor;
    tensor.resize(3 * inputSize.width() * inputSize.height());
    const int planeSize = inputSize.width() * inputSize.height();
    for (int y = 0; y < inputSize.height(); ++y) {
        const uchar* scanline = letterboxed.constScanLine(y);
        for (int x = 0; x < inputSize.width(); ++x) {
            const int pixelIndex = y * inputSize.width() + x;
            tensor[pixelIndex] = static_cast<float>(scanline[x * 3]) / 255.0f;
            tensor[planeSize + pixelIndex] = static_cast<float>(scanline[x * 3 + 1]) / 255.0f;
            tensor[planeSize * 2 + pixelIndex] = static_cast<float>(scanline[x * 3 + 2]) / 255.0f;
        }
    }
    return tensor;
}

DetectionBox yoloBoxFromInputPixels(
    double xCenter,
    double yCenter,
    double width,
    double height,
    int classId,
    const QSize& inputSize,
    const LetterboxTransform& transform)
{
    const double x1 = (xCenter - width / 2.0 - transform.padX) / qMax(1.0e-12, transform.scale);
    const double y1 = (yCenter - height / 2.0 - transform.padY) / qMax(1.0e-12, transform.scale);
    const double x2 = (xCenter + width / 2.0 - transform.padX) / qMax(1.0e-12, transform.scale);
    const double y2 = (yCenter + height / 2.0 - transform.padY) / qMax(1.0e-12, transform.scale);
    Q_UNUSED(inputSize)

    const double sourceWidth = qMax(1, transform.sourceSize.width());
    const double sourceHeight = qMax(1, transform.sourceSize.height());
    const double clampedX1 = qBound(0.0, x1, sourceWidth);
    const double clampedY1 = qBound(0.0, y1, sourceHeight);
    const double clampedX2 = qBound(0.0, x2, sourceWidth);
    const double clampedY2 = qBound(0.0, y2, sourceHeight);

    DetectionBox box;
    box.classId = classId;
    box.xCenter = clamp01((clampedX1 + clampedX2) / 2.0 / sourceWidth);
    box.yCenter = clamp01((clampedY1 + clampedY2) / 2.0 / sourceHeight);
    box.width = qBound(1.0e-6, (clampedX2 - clampedX1) / sourceWidth, 1.0);
    box.height = qBound(1.0e-6, (clampedY2 - clampedY1) / sourceHeight, 1.0);
    return box;
}

QVector<DetectionPrediction> yoloPredictionsFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    const QStringList& classNames,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error)
{
    if (shape.size() != 3 || shape.at(0) != 1) {
        if (error) {
            *error = QStringLiteral("YOLO detection ONNX output shape must be [1, attributes, anchors] or [1, anchors, attributes]");
        }
        return {};
    }

    const int classCount = qMax(1, classNames.size());
    int anchorCount = 0;
    int attributeCount = 0;
    bool attributesFirst = false;
    if (shape.at(1) >= 4 + classCount && shape.at(2) > 0) {
        attributesFirst = true;
        attributeCount = static_cast<int>(shape.at(1));
        anchorCount = static_cast<int>(shape.at(2));
    } else if (shape.at(2) >= 4 + classCount && shape.at(1) > 0) {
        attributesFirst = false;
        anchorCount = static_cast<int>(shape.at(1));
        attributeCount = static_cast<int>(shape.at(2));
    } else {
        if (error) {
            *error = QStringLiteral("YOLO detection ONNX output does not contain box and class attributes");
        }
        return {};
    }

    auto valueAt = [output, anchorCount, attributeCount, attributesFirst](int anchor, int attribute) -> float {
        return attributesFirst
            ? output[attribute * anchorCount + anchor]
            : output[anchor * attributeCount + attribute];
    };

    QVector<DetectionPrediction> predictions;
    predictions.reserve(anchorCount);
    for (int anchor = 0; anchor < anchorCount; ++anchor) {
        int bestClassIndex = 0;
        double bestClassScore = static_cast<double>(valueAt(anchor, 4));
        for (int classIndex = 1; classIndex < classCount && 4 + classIndex < attributeCount; ++classIndex) {
            const double score = static_cast<double>(valueAt(anchor, 4 + classIndex));
            if (score > bestClassScore) {
                bestClassScore = score;
                bestClassIndex = classIndex;
            }
        }
        const double objectness = attributeCount > 4 + classCount
            ? qBound(0.0, static_cast<double>(valueAt(anchor, 4 + classCount)), 1.0)
            : 1.0;
        const double confidence = qBound(0.0, bestClassScore * objectness, 1.0);
        if (confidence < options.confidenceThreshold) {
            continue;
        }

        DetectionPrediction prediction;
        prediction.box = yoloBoxFromInputPixels(
            static_cast<double>(valueAt(anchor, 0)),
            static_cast<double>(valueAt(anchor, 1)),
            qMax(0.0, static_cast<double>(valueAt(anchor, 2))),
            qMax(0.0, static_cast<double>(valueAt(anchor, 3))),
            bestClassIndex,
            inputSize,
            transform);
        prediction.className = bestClassIndex >= 0 && bestClassIndex < classNames.size()
            ? classNames.at(bestClassIndex)
            : QStringLiteral("class_%1").arg(bestClassIndex);
        prediction.objectness = objectness;
        prediction.confidence = confidence;
        predictions.append(prediction);
    }
    return postProcessDetectionPredictions(predictions, options);
}

QColor overlayColorForClass(int classId, int alpha)
{
    static const QVector<QColor> colors = {
        QColor(46, 204, 113),
        QColor(52, 152, 219),
        QColor(241, 196, 15),
        QColor(231, 76, 60),
        QColor(155, 89, 182),
        QColor(26, 188, 156)
    };
    QColor color = colors.at(qAbs(classId) % colors.size());
    color.setAlpha(alpha);
    return color;
}

struct SegmentationCandidate {
    DetectionPrediction detection;
    QVector<float> maskCoefficients;
};

QVector<SegmentationCandidate> postProcessSegmentationCandidates(
    QVector<SegmentationCandidate> candidates,
    const DetectionInferenceOptions& options)
{
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [options](const SegmentationCandidate& candidate) {
        return candidate.detection.confidence < options.confidenceThreshold;
    }), candidates.end());
    std::sort(candidates.begin(), candidates.end(), [](const SegmentationCandidate& left, const SegmentationCandidate& right) {
        return left.detection.confidence > right.detection.confidence;
    });

    QVector<SegmentationCandidate> selected;
    for (const SegmentationCandidate& candidate : candidates) {
        bool suppress = false;
        for (const SegmentationCandidate& accepted : selected) {
            if (candidate.detection.box.classId == accepted.detection.box.classId
                && boxIou(candidate.detection.box, accepted.detection.box) > options.iouThreshold) {
                suppress = true;
                break;
            }
        }
        if (!suppress) {
            selected.append(candidate);
            if (selected.size() >= options.maxDetections) {
                break;
            }
        }
    }
    return selected;
}

QImage maskFromPrototype(
    const QVector<float>& coefficients,
    const float* prototypes,
    const std::vector<int64_t>& prototypeShape,
    const DetectionBox& box,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    double threshold,
    double* maskArea)
{
    if (prototypeShape.size() != 4 || prototypeShape.at(0) != 1 || prototypeShape.at(1) <= 0
        || prototypeShape.at(2) <= 0 || prototypeShape.at(3) <= 0) {
        return {};
    }

    const int maskDim = static_cast<int>(prototypeShape.at(1));
    const int protoHeight = static_cast<int>(prototypeShape.at(2));
    const int protoWidth = static_cast<int>(prototypeShape.at(3));
    if (coefficients.size() < maskDim || inputSize.isEmpty() || transform.sourceSize.isEmpty()) {
        return {};
    }

    const int sourceWidth = qMax(1, transform.sourceSize.width());
    const int sourceHeight = qMax(1, transform.sourceSize.height());
    const int xMin = qBound(0, qFloor((box.xCenter - box.width / 2.0) * sourceWidth), sourceWidth - 1);
    const int xMax = qBound(0, qCeil((box.xCenter + box.width / 2.0) * sourceWidth), sourceWidth);
    const int yMin = qBound(0, qFloor((box.yCenter - box.height / 2.0) * sourceHeight), sourceHeight - 1);
    const int yMax = qBound(0, qCeil((box.yCenter + box.height / 2.0) * sourceHeight), sourceHeight);

    QImage mask(transform.sourceSize, QImage::Format_ARGB32);
    mask.fill(Qt::transparent);
    int activePixels = 0;
    for (int y = yMin; y < yMax; ++y) {
        QRgb* scanline = reinterpret_cast<QRgb*>(mask.scanLine(y));
        const double inputY = static_cast<double>(y) * transform.scale + transform.padY;
        const int protoY = qBound(0, qFloor(inputY / qMax(1, inputSize.height()) * protoHeight), protoHeight - 1);
        for (int x = xMin; x < xMax; ++x) {
            const double inputX = static_cast<double>(x) * transform.scale + transform.padX;
            const int protoX = qBound(0, qFloor(inputX / qMax(1, inputSize.width()) * protoWidth), protoWidth - 1);
            double logit = 0.0;
            const int protoPixel = protoY * protoWidth + protoX;
            for (int index = 0; index < maskDim; ++index) {
                logit += static_cast<double>(coefficients.at(index))
                    * static_cast<double>(prototypes[index * protoHeight * protoWidth + protoPixel]);
            }
            if (sigmoid(logit) >= threshold) {
                scanline[x] = qRgba(255, 255, 255, 180);
                ++activePixels;
            }
        }
    }
    if (maskArea) {
        *maskArea = static_cast<double>(activePixels) / static_cast<double>(qMax(1, sourceWidth * sourceHeight));
    }
    return mask;
}

QVector<SegmentationPrediction> yoloSegmentationPredictionsFromOutputs(
    const float* boxesAndMasks,
    const std::vector<int64_t>& boxesShape,
    const float* prototypes,
    const std::vector<int64_t>& prototypeShape,
    const QStringList& classNames,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error)
{
    if (boxesShape.size() != 3 || boxesShape.at(0) != 1 || prototypeShape.size() != 4 || prototypeShape.at(0) != 1) {
        if (error) {
            *error = QStringLiteral("YOLO segmentation ONNX outputs must be [1, attributes, anchors] and [1, maskDim, maskH, maskW]");
        }
        return {};
    }

    const int maskDim = static_cast<int>(prototypeShape.at(1));
    int anchorCount = 0;
    int attributeCount = 0;
    bool attributesFirst = false;
    if (boxesShape.at(1) >= 4 + maskDim && boxesShape.at(2) > 0) {
        attributesFirst = true;
        attributeCount = static_cast<int>(boxesShape.at(1));
        anchorCount = static_cast<int>(boxesShape.at(2));
    } else if (boxesShape.at(2) >= 4 + maskDim && boxesShape.at(1) > 0) {
        attributesFirst = false;
        anchorCount = static_cast<int>(boxesShape.at(1));
        attributeCount = static_cast<int>(boxesShape.at(2));
    } else {
        if (error) {
            *error = QStringLiteral("YOLO segmentation ONNX output does not contain box and mask attributes");
        }
        return {};
    }

    int classCount = attributeCount - 4 - maskDim;
    if (!classNames.isEmpty()) {
        classCount = qMin(classCount, classNames.size());
    }
    if (classCount <= 0) {
        if (error) {
            *error = QStringLiteral("YOLO segmentation ONNX output does not contain class scores");
        }
        return {};
    }

    auto valueAt = [boxesAndMasks, anchorCount, attributeCount, attributesFirst](int anchor, int attribute) -> float {
        return attributesFirst
            ? boxesAndMasks[attribute * anchorCount + anchor]
            : boxesAndMasks[anchor * attributeCount + attribute];
    };

    QVector<SegmentationCandidate> candidates;
    for (int anchor = 0; anchor < anchorCount; ++anchor) {
        int bestClassIndex = 0;
        double bestClassScore = static_cast<double>(valueAt(anchor, 4));
        for (int classIndex = 1; classIndex < classCount; ++classIndex) {
            const double score = static_cast<double>(valueAt(anchor, 4 + classIndex));
            if (score > bestClassScore) {
                bestClassScore = score;
                bestClassIndex = classIndex;
            }
        }
        const double confidence = qBound(0.0, bestClassScore, 1.0);
        if (confidence < options.confidenceThreshold) {
            continue;
        }

        SegmentationCandidate candidate;
        candidate.detection.box = yoloBoxFromInputPixels(
            static_cast<double>(valueAt(anchor, 0)),
            static_cast<double>(valueAt(anchor, 1)),
            qMax(0.0, static_cast<double>(valueAt(anchor, 2))),
            qMax(0.0, static_cast<double>(valueAt(anchor, 3))),
            bestClassIndex,
            inputSize,
            transform);
        candidate.detection.className = bestClassIndex >= 0 && bestClassIndex < classNames.size()
            ? classNames.at(bestClassIndex)
            : QStringLiteral("class_%1").arg(bestClassIndex);
        candidate.detection.objectness = 1.0;
        candidate.detection.confidence = confidence;
        candidate.maskCoefficients.reserve(maskDim);
        for (int index = 0; index < maskDim; ++index) {
            candidate.maskCoefficients.append(valueAt(anchor, 4 + classCount + index));
        }
        candidates.append(candidate);
    }

    const QVector<SegmentationCandidate> selected = postProcessSegmentationCandidates(candidates, options);
    QVector<SegmentationPrediction> predictions;
    predictions.reserve(selected.size());
    constexpr double maskThreshold = 0.5;
    for (const SegmentationCandidate& candidate : selected) {
        SegmentationPrediction prediction;
        prediction.detection = candidate.detection;
        prediction.maskThreshold = maskThreshold;
        prediction.mask = maskFromPrototype(
            candidate.maskCoefficients,
            prototypes,
            prototypeShape,
            candidate.detection.box,
            inputSize,
            transform,
            maskThreshold,
            &prediction.maskArea);
        predictions.append(prediction);
    }
    return predictions;
}

QStringList readOcrDictionary(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    QStringList characters;
    for (const QString& line : QString::fromUtf8(file.readAll()).split(QLatin1Char('\n'))) {
        const QString value = line.trimmed();
        if (!value.isEmpty()) {
            characters.append(value);
        }
    }
    return characters;
}

QJsonObject loadOcrRecReport(const QString& onnxPath)
{
    const QFileInfo onnxInfo(onnxPath);
    const QStringList candidates = {
        onnxInfo.absoluteDir().filePath(QStringLiteral("paddleocr_rec_training_report.json")),
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
            const QJsonObject object = document.object();
            if (object.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_rec")
                || object.value(QStringLiteral("modelFamily")).toString() == QStringLiteral("ocr_recognition")) {
                return object;
            }
        }
    }
    return {};
}

QJsonObject loadOcrDetReport(const QString& onnxPath)
{
    const QFileInfo onnxInfo(onnxPath);
    const QStringList candidates = {
        onnxInfo.absoluteDir().filePath(QStringLiteral("paddleocr_official_det_report.json")),
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
            const QJsonObject object = document.object();
            if (object.value(QStringLiteral("backend")).toString() == QStringLiteral("paddleocr_det_official")
                || object.value(QStringLiteral("modelFamily")).toString() == QStringLiteral("ocr_detection")) {
                return object;
            }
        }
    }
    return {};
}

QVector<float> ocrImageTensor(const QImage& image, int width, int height)
{
    const QImage gray = image.convertToFormat(QImage::Format_Grayscale8).scaled(width, height, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QVector<float> tensor;
    tensor.resize(width * height);
    for (int y = 0; y < height; ++y) {
        const uchar* scanline = gray.constScanLine(y);
        for (int x = 0; x < width; ++x) {
            tensor[y * width + x] = static_cast<float>(scanline[x]) / 255.0f;
        }
    }
    return tensor;
}

QVector<float> ocrDetImageTensor(const QImage& image, int width, int height)
{
    const QImage rgb = image.convertToFormat(QImage::Format_RGB888).scaled(width, height, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QVector<float> tensor;
    tensor.resize(3 * width * height);
    const int planeSize = width * height;
    for (int y = 0; y < height; ++y) {
        const uchar* scanline = rgb.constScanLine(y);
        for (int x = 0; x < width; ++x) {
            const int pixelIndex = y * width + x;
            tensor[pixelIndex] = static_cast<float>(scanline[x * 3]) / 255.0f;
            tensor[planeSize + pixelIndex] = static_cast<float>(scanline[x * 3 + 1]) / 255.0f;
            tensor[planeSize * 2 + pixelIndex] = static_cast<float>(scanline[x * 3 + 2]) / 255.0f;
        }
    }
    return tensor;
}

OcrRecPrediction ocrPredictionFromLogits(
    const float* logits,
    const std::vector<int64_t>& shape,
    const QStringList& dictionary,
    int blankIndex,
    QString* error)
{
    int timesteps = 0;
    int classCount = 0;
    if (shape.size() == 3 && shape.at(0) == 1 && shape.at(1) > 0 && shape.at(2) > 0) {
        timesteps = static_cast<int>(shape.at(1));
        classCount = static_cast<int>(shape.at(2));
    } else if (shape.size() == 2 && shape.at(0) > 0 && shape.at(1) > 0) {
        timesteps = static_cast<int>(shape.at(0));
        classCount = static_cast<int>(shape.at(1));
    } else {
        if (error) {
            *error = QStringLiteral("OCR Rec ONNX output must be [1, timesteps, classes] or [timesteps, classes]");
        }
        return {};
    }

    OcrRecPrediction prediction;
    prediction.blankIndex = blankIndex;
    int previous = -1;
    double confidenceSum = 0.0;
    for (int step = 0; step < timesteps; ++step) {
        const float* row = logits + step * classCount;
        int bestIndex = 0;
        float bestLogit = row[0];
        float maxLogit = row[0];
        for (int index = 1; index < classCount; ++index) {
            maxLogit = qMax(maxLogit, row[index]);
            if (row[index] > bestLogit) {
                bestLogit = row[index];
                bestIndex = index;
            }
        }
        double denominator = 0.0;
        for (int index = 0; index < classCount; ++index) {
            denominator += qExp(static_cast<double>(row[index] - maxLogit));
        }
        confidenceSum += qExp(static_cast<double>(bestLogit - maxLogit)) / qMax(1.0e-12, denominator);
        prediction.tokens.append(bestIndex);
        if (bestIndex != blankIndex && bestIndex != previous) {
            const int dictionaryIndex = bestIndex > blankIndex ? bestIndex - 1 : bestIndex;
            if (dictionaryIndex >= 0 && dictionaryIndex < dictionary.size()) {
                prediction.text.append(dictionary.at(dictionaryIndex));
            }
        }
        previous = bestIndex;
    }
    prediction.confidence = timesteps > 0 ? confidenceSum / static_cast<double>(timesteps) : 0.0;
    return prediction;
}

QVector<float> ocrDetProbabilityMapFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    QSize* mapSize,
    QString* error)
{
    if (!output) {
        if (error) {
            *error = QStringLiteral("OCR Det ONNX output tensor is null");
        }
        return {};
    }

    int height = 0;
    int width = 0;
    if (shape.size() == 4 && shape.at(0) == 1 && shape.at(1) == 1 && shape.at(2) > 0 && shape.at(3) > 0) {
        height = static_cast<int>(shape.at(2));
        width = static_cast<int>(shape.at(3));
    } else if (shape.size() == 3 && shape.at(0) == 1 && shape.at(1) > 0 && shape.at(2) > 0) {
        height = static_cast<int>(shape.at(1));
        width = static_cast<int>(shape.at(2));
    } else if (shape.size() == 2 && shape.at(0) > 0 && shape.at(1) > 0) {
        height = static_cast<int>(shape.at(0));
        width = static_cast<int>(shape.at(1));
    } else {
        if (error) {
            *error = QStringLiteral("OCR Det DB ONNX output must be [1,1,H,W], [1,H,W], or [H,W]");
        }
        return {};
    }

    QVector<float> probabilities;
    probabilities.resize(width * height);
    for (int index = 0; index < probabilities.size(); ++index) {
        probabilities[index] = qBound(0.0f, output[index], 1.0f);
    }
    if (mapSize) {
        *mapSize = QSize(width, height);
    }
    return probabilities;
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

QString inferOnnxModelFamily(const QString& onnxPath)
{
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
    } catch (...) {
    }
#endif
    return {};
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
        const int inputHeight = inputShape.at(2) > 0 ? static_cast<int>(inputShape.at(2)) : image.height();
        const int inputWidth = inputShape.at(3) > 0 ? static_cast<int>(inputShape.at(3)) : image.width();
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

QVector<OcrDetPrediction> postProcessPaddleOcrDetDbMap(
    const QVector<float>& probabilityMap,
    const QSize& mapSize,
    const QSize& sourceSize,
    const OcrDetPostprocessOptions& options,
    QString* error)
{
    if (mapSize.isEmpty() || sourceSize.isEmpty() || probabilityMap.size() != mapSize.width() * mapSize.height()) {
        if (error) {
            *error = QStringLiteral("OCR Det DB probability map size is invalid");
        }
        return {};
    }

    const double binaryThreshold = qBound(0.0, options.binaryThreshold, 1.0);
    const double boxThreshold = qBound(0.0, options.boxThreshold, 1.0);
    const int minArea = qMax(1, options.minArea);
    const int maxDetections = qMax(0, options.maxDetections);
    if (maxDetections == 0) {
        return {};
    }

    const int width = mapSize.width();
    const int height = mapSize.height();
    QVector<uchar> visited(width * height, 0);
    QVector<OcrDetPrediction> detections;

    auto indexOf = [width](int x, int y) {
        return y * width + x;
    };

    for (int startY = 0; startY < height; ++startY) {
        for (int startX = 0; startX < width; ++startX) {
            const int startIndex = indexOf(startX, startY);
            if (visited.at(startIndex) || probabilityMap.at(startIndex) < binaryThreshold) {
                continue;
            }

            QQueue<QPoint> queue;
            queue.enqueue(QPoint(startX, startY));
            visited[startIndex] = 1;
            int minX = startX;
            int maxX = startX;
            int minY = startY;
            int maxY = startY;
            int area = 0;
            double confidenceSum = 0.0;

            while (!queue.isEmpty()) {
                const QPoint point = queue.dequeue();
                const int pointIndex = indexOf(point.x(), point.y());
                ++area;
                confidenceSum += static_cast<double>(probabilityMap.at(pointIndex));
                minX = qMin(minX, point.x());
                maxX = qMax(maxX, point.x());
                minY = qMin(minY, point.y());
                maxY = qMax(maxY, point.y());

                const QPoint neighbors[] = {
                    QPoint(point.x() + 1, point.y()),
                    QPoint(point.x() - 1, point.y()),
                    QPoint(point.x(), point.y() + 1),
                    QPoint(point.x(), point.y() - 1)
                };
                for (const QPoint& neighbor : neighbors) {
                    if (neighbor.x() < 0 || neighbor.x() >= width || neighbor.y() < 0 || neighbor.y() >= height) {
                        continue;
                    }
                    const int neighborIndex = indexOf(neighbor.x(), neighbor.y());
                    if (visited.at(neighborIndex) || probabilityMap.at(neighborIndex) < binaryThreshold) {
                        continue;
                    }
                    visited[neighborIndex] = 1;
                    queue.enqueue(neighbor);
                }
            }

            const double confidence = area > 0 ? confidenceSum / static_cast<double>(area) : 0.0;
            if (area < minArea || confidence < boxThreshold) {
                continue;
            }

            const double sx = static_cast<double>(sourceSize.width()) / static_cast<double>(qMax(1, width));
            const double sy = static_cast<double>(sourceSize.height()) / static_cast<double>(qMax(1, height));
            const double left = qBound(0.0, static_cast<double>(minX) * sx, static_cast<double>(sourceSize.width()));
            const double top = qBound(0.0, static_cast<double>(minY) * sy, static_cast<double>(sourceSize.height()));
            const double right = qBound(0.0, static_cast<double>(maxX + 1) * sx, static_cast<double>(sourceSize.width()));
            const double bottom = qBound(0.0, static_cast<double>(maxY + 1) * sy, static_cast<double>(sourceSize.height()));

            OcrDetPrediction prediction;
            prediction.confidence = confidence;
            prediction.pixelArea = area;
            prediction.polygon = {
                QPointF(left, top),
                QPointF(right, top),
                QPointF(right, bottom),
                QPointF(left, bottom)
            };
            const double sourceWidth = qMax(1, sourceSize.width());
            const double sourceHeight = qMax(1, sourceSize.height());
            prediction.box.classId = 0;
            prediction.box.xCenter = clamp01((left + right) / 2.0 / sourceWidth);
            prediction.box.yCenter = clamp01((top + bottom) / 2.0 / sourceHeight);
            prediction.box.width = qBound(1.0e-6, (right - left) / sourceWidth, 1.0);
            prediction.box.height = qBound(1.0e-6, (bottom - top) / sourceHeight, 1.0);
            detections.append(prediction);
        }
    }

    std::sort(detections.begin(), detections.end(), [](const OcrDetPrediction& left, const OcrDetPrediction& right) {
        return left.confidence > right.confidence;
    });
    if (detections.size() > maxDetections) {
        detections.resize(maxDetections);
    }
    return detections;
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

QJsonObject segmentationPredictionToJson(const SegmentationPrediction& prediction)
{
    QJsonObject object = detectionPredictionToJson(prediction.detection);
    object.insert(QStringLiteral("taskType"), QStringLiteral("segmentation"));
    object.insert(QStringLiteral("maskArea"), prediction.maskArea);
    object.insert(QStringLiteral("maskThreshold"), prediction.maskThreshold);
    object.insert(QStringLiteral("hasMask"), !prediction.mask.isNull());
    return object;
}

QJsonObject ocrRecPredictionToJson(const OcrRecPrediction& prediction)
{
    QJsonArray tokens;
    for (int token : prediction.tokens) {
        tokens.append(token);
    }
    return QJsonObject{
        {QStringLiteral("taskType"), QStringLiteral("ocr_recognition")},
        {QStringLiteral("text"), prediction.text},
        {QStringLiteral("confidence"), prediction.confidence},
        {QStringLiteral("blankIndex"), prediction.blankIndex},
        {QStringLiteral("tokens"), tokens}
    };
}

QJsonObject ocrDetPredictionToJson(const OcrDetPrediction& prediction)
{
    QJsonArray points;
    for (const QPointF& point : prediction.polygon) {
        QJsonArray item;
        item.append(point.x());
        item.append(point.y());
        points.append(item);
    }

    QJsonObject object;
    object.insert(QStringLiteral("taskType"), QStringLiteral("ocr_detection"));
    object.insert(QStringLiteral("confidence"), prediction.confidence);
    object.insert(QStringLiteral("pixelArea"), prediction.pixelArea);
    object.insert(QStringLiteral("points"), points);
    object.insert(QStringLiteral("box"), boxObject(prediction.box));
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

QImage renderSegmentationPredictions(
    const QString& imagePath,
    const QVector<SegmentationPrediction>& predictions,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render segmentation prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output = image.convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&output);
    painter.setRenderHint(QPainter::Antialiasing, true);
    for (const SegmentationPrediction& prediction : predictions) {
        if (!prediction.mask.isNull()) {
            const QColor fillColor = overlayColorForClass(prediction.detection.box.classId, 95);
            const int maskHeight = qMin(output.height(), prediction.mask.height());
            const int maskWidth = qMin(output.width(), prediction.mask.width());
            for (int y = 0; y < maskHeight; ++y) {
                const QRgb* maskLine = reinterpret_cast<const QRgb*>(prediction.mask.constScanLine(y));
                for (int x = 0; x < maskWidth; ++x) {
                    if (qAlpha(maskLine[x]) > 0) {
                        painter.fillRect(QRect(x, y, 1, 1), fillColor);
                    }
                }
            }
        }

        const double imageWidth = static_cast<double>(output.width());
        const double imageHeight = static_cast<double>(output.height());
        QRect rect(
            qRound((prediction.detection.box.xCenter - prediction.detection.box.width / 2.0) * imageWidth),
            qRound((prediction.detection.box.yCenter - prediction.detection.box.height / 2.0) * imageHeight),
            qRound(prediction.detection.box.width * imageWidth),
            qRound(prediction.detection.box.height * imageHeight));
        rect = rect.intersected(output.rect());
        if (rect.isEmpty()) {
            continue;
        }
        painter.setPen(QPen(overlayColorForClass(prediction.detection.box.classId, 230), qMax(2, output.width() / 240)));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(rect);
    }
    painter.end();
    return output;
}

QString pixelGlyph(QChar ch)
{
    ch = ch.toLower();
    switch (ch.toLatin1()) {
    case 'a': return QStringLiteral("01110""10001""10001""11111""10001""10001""10001");
    case 'b': return QStringLiteral("11110""10001""10001""11110""10001""10001""11110");
    case 'c': return QStringLiteral("01111""10000""10000""10000""10000""10000""01111");
    case 'd': return QStringLiteral("11110""10001""10001""10001""10001""10001""11110");
    case 'e': return QStringLiteral("11111""10000""10000""11110""10000""10000""11111");
    case 'f': return QStringLiteral("11111""10000""10000""11110""10000""10000""10000");
    case 'g': return QStringLiteral("01111""10000""10000""10111""10001""10001""01111");
    case 'h': return QStringLiteral("10001""10001""10001""11111""10001""10001""10001");
    case 'i': return QStringLiteral("11111""00100""00100""00100""00100""00100""11111");
    case 'j': return QStringLiteral("00111""00010""00010""00010""00010""10010""01100");
    case 'k': return QStringLiteral("10001""10010""10100""11000""10100""10010""10001");
    case 'l': return QStringLiteral("10000""10000""10000""10000""10000""10000""11111");
    case 'm': return QStringLiteral("10001""11011""10101""10101""10001""10001""10001");
    case 'n': return QStringLiteral("10001""11001""10101""10011""10001""10001""10001");
    case 'o': return QStringLiteral("01110""10001""10001""10001""10001""10001""01110");
    case 'p': return QStringLiteral("11110""10001""10001""11110""10000""10000""10000");
    case 'q': return QStringLiteral("01110""10001""10001""10001""10101""10010""01101");
    case 'r': return QStringLiteral("11110""10001""10001""11110""10100""10010""10001");
    case 's': return QStringLiteral("01111""10000""10000""01110""00001""00001""11110");
    case 't': return QStringLiteral("11111""00100""00100""00100""00100""00100""00100");
    case 'u': return QStringLiteral("10001""10001""10001""10001""10001""10001""01110");
    case 'v': return QStringLiteral("10001""10001""10001""10001""10001""01010""00100");
    case 'w': return QStringLiteral("10001""10001""10001""10101""10101""10101""01010");
    case 'x': return QStringLiteral("10001""10001""01010""00100""01010""10001""10001");
    case 'y': return QStringLiteral("10001""10001""01010""00100""00100""00100""00100");
    case 'z': return QStringLiteral("11111""00001""00010""00100""01000""10000""11111");
    case '0': return QStringLiteral("01110""10001""10011""10101""11001""10001""01110");
    case '1': return QStringLiteral("00100""01100""00100""00100""00100""00100""01110");
    case '2': return QStringLiteral("01110""10001""00001""00010""00100""01000""11111");
    case '3': return QStringLiteral("11110""00001""00001""01110""00001""00001""11110");
    case '4': return QStringLiteral("00010""00110""01010""10010""11111""00010""00010");
    case '5': return QStringLiteral("11111""10000""10000""11110""00001""00001""11110");
    case '6': return QStringLiteral("01110""10000""10000""11110""10001""10001""01110");
    case '7': return QStringLiteral("11111""00001""00010""00100""01000""01000""01000");
    case '8': return QStringLiteral("01110""10001""10001""01110""10001""10001""01110");
    case '9': return QStringLiteral("01110""10001""10001""01111""00001""00001""01110");
    default: return QStringLiteral("11111""10001""00010""00100""00000""00100""00100");
    }
}

void drawPixelText(QImage* image, QPoint origin, const QString& text, const QColor& color, int scale)
{
    if (!image || image->isNull()) {
        return;
    }
    int cursorX = origin.x();
    for (const QChar ch : text.left(32)) {
        if (ch.isSpace()) {
            cursorX += 4 * scale;
            continue;
        }
        const QString glyph = pixelGlyph(ch);
        for (int row = 0; row < 7; ++row) {
            for (int col = 0; col < 5; ++col) {
                if (glyph.at(row * 5 + col) != QLatin1Char('1')) {
                    continue;
                }
                for (int dy = 0; dy < scale; ++dy) {
                    for (int dx = 0; dx < scale; ++dx) {
                        const int x = cursorX + col * scale + dx;
                        const int y = origin.y() + row * scale + dy;
                        if (image->rect().contains(x, y)) {
                            image->setPixelColor(x, y, color);
                        }
                    }
                }
            }
        }
        cursorX += 6 * scale;
        if (cursorX >= image->width()) {
            break;
        }
    }
}

QImage renderOcrRecPrediction(
    const QString& imagePath,
    const OcrRecPrediction& prediction,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render OCR Rec prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output(image.width(), image.height() + 40, QImage::Format_ARGB32);
    output.fill(Qt::white);
    QPainter painter(&output);
    painter.drawImage(QPoint(0, 0), image.convertToFormat(QImage::Format_ARGB32));
    painter.fillRect(QRect(0, image.height(), output.width(), 40), QColor(20, 20, 20));
    painter.end();
    const QString text = prediction.text.isEmpty() ? QStringLiteral("empty") : prediction.text;
    drawPixelText(&output, QPoint(8, image.height() + 8), text, Qt::white, 3);
    return output;
}

QImage renderOcrDetPredictions(
    const QString& imagePath,
    const QVector<OcrDetPrediction>& predictions,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render OCR Det prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output = image.convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&output);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor(245, 158, 11), qMax(2, output.width() / 240)));
    painter.setBrush(QColor(245, 158, 11, 45));
    for (const OcrDetPrediction& prediction : predictions) {
        if (prediction.polygon.size() < 4) {
            continue;
        }
        QPolygonF polygon;
        for (const QPointF& point : prediction.polygon) {
            polygon << point;
        }
        painter.drawPolygon(polygon);
    }
    painter.end();
    return output;
}

QJsonObject yoloOnnxExportConfig(const QString& sourceOnnxPath, const QString& exportPath, const QString& format);

QString ncnnConverterExecutableName()
{
#ifdef Q_OS_WIN
    return QStringLiteral("onnx2ncnn.exe");
#else
    return QStringLiteral("onnx2ncnn");
#endif
}

QString unquotedPath(QString value)
{
    value = value.trimmed();
    if (value.size() >= 2
        && ((value.startsWith(QLatin1Char('"')) && value.endsWith(QLatin1Char('"')))
            || (value.startsWith(QLatin1Char('\'')) && value.endsWith(QLatin1Char('\''))))) {
        value = value.mid(1, value.size() - 2);
    }
    return QDir::fromNativeSeparators(value);
}

QString existingExecutablePath(const QString& candidate)
{
    if (candidate.trimmed().isEmpty()) {
        return {};
    }
    const QFileInfo info(unquotedPath(candidate));
    return info.exists() && info.isFile() ? info.absoluteFilePath() : QString();
}

QString ncnnParamPathForOutput(const QString& outputPath, const QString& sourcePath)
{
    QString finalOutputPath = outputPath;
    if (finalOutputPath.isEmpty()) {
        finalOutputPath = QFileInfo(sourcePath).absoluteDir().filePath(QStringLiteral("model.param"));
    }
    if (QFileInfo(finalOutputPath).isDir()) {
        finalOutputPath = QDir(finalOutputPath).filePath(QStringLiteral("model.param"));
    }

    QFileInfo outputInfo(finalOutputPath);
    const QString suffix = outputInfo.suffix().toLower();
    if (suffix == QStringLiteral("bin")) {
        finalOutputPath = outputInfo.absoluteDir().filePath(QStringLiteral("%1.param").arg(outputInfo.completeBaseName()));
    } else if (suffix != QStringLiteral("param")) {
        finalOutputPath = outputInfo.absoluteDir().filePath(QStringLiteral("%1.param").arg(outputInfo.fileName()));
    }
    return QDir::cleanPath(finalOutputPath);
}

QString ncnnBinPathForParam(const QString& paramPath)
{
    const QFileInfo info(paramPath);
    return info.absoluteDir().filePath(QStringLiteral("%1.bin").arg(info.completeBaseName()));
}

struct NcnnConverterResolution {
    QString executablePath;
    QString message;
};

NcnnConverterResolution resolveNcnnOnnx2Ncnn()
{
    const QString configured = unquotedPath(QString::fromLocal8Bit(qgetenv("AITRAIN_NCNN_ONNX2NCNN")));
    if (!configured.isEmpty()) {
        const QString executable = existingExecutablePath(configured);
        if (!executable.isEmpty()) {
            return {executable, QString()};
        }
        return {{}, QStringLiteral("Configured NCNN converter was not found: %1").arg(configured)};
    }

    const QString executableName = ncnnConverterExecutableName();
    QStringList candidates;
    const auto appendRootCandidates = [&candidates, &executableName](const QString& rootValue) {
        const QString root = unquotedPath(rootValue);
        if (root.isEmpty()) {
            return;
        }
        const QDir rootDir(root);
        candidates << rootDir.filePath(QStringLiteral("bin/%1").arg(executableName))
                   << rootDir.filePath(QStringLiteral("tools/onnx/%1").arg(executableName))
                   << rootDir.filePath(executableName);
    };
    appendRootCandidates(QString::fromLocal8Bit(qgetenv("AITRAIN_NCNN_ROOT")));
    appendRootCandidates(QString::fromLocal8Bit(qgetenv("NCNN_ROOT")));

    const QDir appDir(QCoreApplication::applicationDirPath());
    candidates << appDir.filePath(QStringLiteral("runtimes/ncnn/%1").arg(executableName))
               << appDir.filePath(QStringLiteral("../runtimes/ncnn/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/bin/%1").arg(executableName))
               << QDir::current().filePath(QStringLiteral(".deps/ncnn/tools/onnx/%1").arg(executableName));

    for (const QString& candidate : candidates) {
        const QString executable = existingExecutablePath(candidate);
        if (!executable.isEmpty()) {
            return {executable, QString()};
        }
    }

    const QString pathExecutable = QStandardPaths::findExecutable(QStringLiteral("onnx2ncnn"));
    if (!pathExecutable.isEmpty()) {
        return {pathExecutable, QString()};
    }

    return {{}, QStringLiteral("NCNN export requires onnx2ncnn. Set AITRAIN_NCNN_ONNX2NCNN to onnx2ncnn.exe or AITRAIN_NCNN_ROOT to an NCNN install root.")};
}

bool isWindowsCommandScript(const QString& path)
{
#ifdef Q_OS_WIN
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == QStringLiteral("bat") || suffix == QStringLiteral("cmd");
#else
    Q_UNUSED(path);
    return false;
#endif
}

bool runOnnx2Ncnn(
    const QString& sourceOnnxPath,
    const QString& paramPath,
    const QString& binPath,
    QString* converterPath,
    QString* error)
{
    const NcnnConverterResolution converter = resolveNcnnOnnx2Ncnn();
    if (converter.executablePath.isEmpty()) {
        if (error) {
            *error = converter.message;
        }
        return false;
    }

    if (!QFileInfo::exists(sourceOnnxPath)) {
        if (error) {
            *error = QStringLiteral("Cannot read source ONNX model for NCNN export: %1").arg(sourceOnnxPath);
        }
        return false;
    }
    if (!QDir().mkpath(QFileInfo(paramPath).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Cannot create NCNN export directory: %1").arg(QFileInfo(paramPath).absolutePath());
        }
        return false;
    }

    QFile::remove(paramPath);
    QFile::remove(binPath);

    QProcess process;
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.setWorkingDirectory(QFileInfo(paramPath).absolutePath());
#ifdef Q_OS_WIN
    if (isWindowsCommandScript(converter.executablePath)) {
        process.start(QStringLiteral("cmd.exe"), QStringList()
            << QStringLiteral("/D")
            << QStringLiteral("/C")
            << QDir::toNativeSeparators(converter.executablePath)
            << QDir::toNativeSeparators(sourceOnnxPath)
            << QDir::toNativeSeparators(paramPath)
            << QDir::toNativeSeparators(binPath));
    } else
#endif
    {
        process.start(converter.executablePath, QStringList() << sourceOnnxPath << paramPath << binPath);
    }

    if (!process.waitForStarted(5000)) {
        if (error) {
            *error = QStringLiteral("Could not start NCNN converter %1: %2").arg(converter.executablePath, process.errorString());
        }
        return false;
    }
    if (!process.waitForFinished(-1)) {
        if (error) {
            *error = QStringLiteral("NCNN converter did not finish: %1").arg(process.errorString());
        }
        return false;
    }

    const QString converterOutput = QString::fromLocal8Bit(process.readAll()).trimmed();
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        if (error) {
            *error = QStringLiteral("NCNN converter failed with exit code %1: %2").arg(process.exitCode()).arg(converterOutput);
        }
        return false;
    }
    if (!QFileInfo::exists(paramPath) || !QFileInfo::exists(binPath)) {
        if (error) {
            *error = QStringLiteral("NCNN converter finished but did not produce expected .param/.bin files. Output: %1").arg(converterOutput);
        }
        return false;
    }

    if (converterPath) {
        *converterPath = converter.executablePath;
    }
    return true;
}

QJsonObject ncnnMetadata(
    const QString& paramPath,
    const QString& binPath,
    const QString& converterPath,
    const QString& sourceOnnxPath)
{
    return QJsonObject{
        {QStringLiteral("paramPath"), paramPath},
        {QStringLiteral("binPath"), binPath},
        {QStringLiteral("converter"), converterPath},
        {QStringLiteral("sourceOnnx"), sourceOnnxPath},
        {QStringLiteral("runtime"), QStringLiteral("ncnn")},
        {QStringLiteral("note"), QStringLiteral("NCNN runtime inference is not implemented in AITrain Studio yet; this export produces param/bin deployment artifacts.")}
    };
}

QJsonObject ncnnOnnxExportConfig(
    const QString& sourceOnnxPath,
    const QString& paramPath,
    const QString& binPath,
    const QString& converterPath)
{
    const QString modelFamily = inferOnnxModelFamily(sourceOnnxPath);
    QJsonObject config;
    if (modelFamily == QStringLiteral("yolo_detection") || modelFamily == QStringLiteral("yolo_segmentation")) {
        config = yoloOnnxExportConfig(sourceOnnxPath, paramPath, QStringLiteral("ncnn"));
    } else if (modelFamily == QStringLiteral("ocr_recognition")) {
        config = QJsonObject{
            {QStringLiteral("format"), QStringLiteral("ncnn")},
            {QStringLiteral("backend"), QStringLiteral("paddleocr_rec")},
            {QStringLiteral("modelFamily"), QStringLiteral("ocr_recognition")},
            {QStringLiteral("scaffold"), false},
            {QStringLiteral("sourceCheckpoint"), sourceOnnxPath},
            {QStringLiteral("sourceOnnx"), sourceOnnxPath},
            {QStringLiteral("exportPath"), paramPath},
            {QStringLiteral("trainingReport"), loadOcrRecReport(sourceOnnxPath)}
        };
    } else {
        config = QJsonObject{
            {QStringLiteral("format"), QStringLiteral("ncnn")},
            {QStringLiteral("backend"), QStringLiteral("onnx2ncnn")},
            {QStringLiteral("modelFamily"), modelFamily.isEmpty() ? QStringLiteral("unknown_onnx") : modelFamily},
            {QStringLiteral("scaffold"), false},
            {QStringLiteral("sourceCheckpoint"), sourceOnnxPath},
            {QStringLiteral("sourceOnnx"), sourceOnnxPath},
            {QStringLiteral("exportPath"), paramPath}
        };
    }
    config.insert(QStringLiteral("ncnn"), ncnnMetadata(paramPath, binPath, converterPath, sourceOnnxPath));
    return config;
}

QJsonObject yoloOnnxExportConfig(const QString& sourceOnnxPath, const QString& exportPath, const QString& format)
{
    const QStringList classNames = ultralyticsClassNames(sourceOnnxPath);
    QJsonObject report = loadUltralyticsTrainingReport(sourceOnnxPath);
    const bool segmentation = report.value(QStringLiteral("backend")).toString() == QStringLiteral("ultralytics_yolo_segment");
    return QJsonObject{
        {QStringLiteral("format"), format},
        {QStringLiteral("backend"), segmentation ? QStringLiteral("ultralytics_yolo_segment") : QStringLiteral("ultralytics_yolo_detect")},
        {QStringLiteral("modelFamily"), segmentation ? QStringLiteral("yolo_segmentation") : QStringLiteral("yolo_detection")},
        {QStringLiteral("scaffold"), false},
        {QStringLiteral("sourceCheckpoint"), sourceOnnxPath},
        {QStringLiteral("sourceOnnx"), sourceOnnxPath},
        {QStringLiteral("exportPath"), exportPath},
        {QStringLiteral("classNames"), QJsonArray::fromStringList(classNames)},
        {QStringLiteral("trainingReport"), report},
        {QStringLiteral("postprocess"), QJsonObject{
            {QStringLiteral("decoder"), segmentation ? QStringLiteral("yolo_v8_segmentation") : QStringLiteral("yolo_v8_detection")},
            {QStringLiteral("nms"), QStringLiteral("AITrain runtime")},
            {QStringLiteral("coordinates"), QStringLiteral("letterbox_to_original_image")}
        }}
    };
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
    const bool ncnnFormat = normalizedFormat == QStringLiteral("ncnn");
    result.format = normalizedFormat;
    result.sourceCheckpointPath = checkpointPath;
    if (normalizedFormat != QStringLiteral("tiny_detector_json")
        && normalizedFormat != QStringLiteral("onnx")
        && !ncnnFormat
        && !tensorRtFormat) {
        if (normalizedFormat.startsWith(QStringLiteral("tensorrt"))) {
            result.error = QStringLiteral("TensorRT export is not available: %1").arg(tensorRtBackendStatus().message);
        } else {
            result.error = QStringLiteral("Unsupported detection export format: %1").arg(normalizedFormat);
        }
        return result;
    }

    const bool sourceIsOnnx = QFileInfo(checkpointPath).suffix().toLower() == QStringLiteral("onnx");
    if (sourceIsOnnx) {
        QString finalOutputPath = outputPath;
        if (finalOutputPath.isEmpty()) {
            finalOutputPath = ncnnFormat
                ? QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("model.param"))
                : QFileInfo(checkpointPath).absoluteDir().filePath(
                    tensorRtFormat ? QStringLiteral("model.engine") : QFileInfo(checkpointPath).fileName());
        }
        if (QFileInfo(finalOutputPath).isDir()) {
            finalOutputPath = QDir(finalOutputPath).filePath(
                ncnnFormat ? QStringLiteral("model.param") : (tensorRtFormat ? QStringLiteral("model.engine") : QFileInfo(checkpointPath).fileName()));
        }
        if (ncnnFormat) {
            finalOutputPath = ncnnParamPathForOutput(finalOutputPath, checkpointPath);
        }
        if (!QDir().mkpath(QFileInfo(finalOutputPath).absolutePath())) {
            result.error = QStringLiteral("Cannot create export directory: %1").arg(QFileInfo(finalOutputPath).absolutePath());
            return result;
        }

        if (normalizedFormat == QStringLiteral("onnx")) {
            if (QFileInfo(checkpointPath).absoluteFilePath() != QFileInfo(finalOutputPath).absoluteFilePath()) {
                QFile::remove(finalOutputPath);
                if (!QFile::copy(checkpointPath, finalOutputPath)) {
                    result.error = QStringLiteral("Cannot copy ONNX model to export path: %1").arg(finalOutputPath);
                    return result;
                }
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            const QJsonObject config = yoloOnnxExportConfig(checkpointPath, finalOutputPath, normalizedFormat);
            if (!writeJsonObject(reportPath, config, &result.error)) {
                return result;
            }
            result.ok = true;
            result.exportPath = finalOutputPath;
            result.reportPath = reportPath;
            result.config = config;
            return result;
        }

        if (ncnnFormat) {
            const QString binPath = ncnnBinPathForParam(finalOutputPath);
            QString converterPath;
            if (!runOnnx2Ncnn(checkpointPath, finalOutputPath, binPath, &converterPath, &result.error)) {
                return result;
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            const QJsonObject config = ncnnOnnxExportConfig(checkpointPath, finalOutputPath, binPath, converterPath);
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
            QFile onnxFile(checkpointPath);
            if (!onnxFile.open(QIODevice::ReadOnly)) {
                result.error = QStringLiteral("Cannot read source ONNX model for TensorRT export: %1").arg(checkpointPath);
                return result;
            }
            const QByteArray onnxModel = onnxFile.readAll();
            const bool fp16 = normalizedFormat == QStringLiteral("tensorrt_fp16");
            if (!writeTensorRtEngineFromOnnx(onnxModel, finalOutputPath, fp16, &result.error)) {
                return result;
            }
            const QString reportPath = onnxExportReportPath(finalOutputPath);
            QJsonObject config = yoloOnnxExportConfig(checkpointPath, finalOutputPath, normalizedFormat);
            config.insert(QStringLiteral("backend"), QStringLiteral("tensorrt_ultralytics_yolo_detect"));
            config.insert(QStringLiteral("tensorRt"), QJsonObject{
                {QStringLiteral("precision"), fp16 ? QStringLiteral("fp16") : QStringLiteral("fp32")},
                {QStringLiteral("workspaceBytes"), static_cast<double>(size_t{1} << 30)},
                {QStringLiteral("sourceOnnx"), checkpointPath}
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
                : (ncnnFormat ? QStringLiteral("model.param") : (tensorRtFormat ? QStringLiteral("model.engine") : QStringLiteral("model.aitrain-export.json"))));
    }
    if (QFileInfo(finalOutputPath).isDir()) {
        finalOutputPath = QDir(finalOutputPath).filePath(
            normalizedFormat == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (ncnnFormat ? QStringLiteral("model.param") : (tensorRtFormat ? QStringLiteral("model.engine") : QStringLiteral("model.aitrain-export.json"))));
    }
    if (ncnnFormat) {
        finalOutputPath = ncnnParamPathForOutput(finalOutputPath, checkpointPath);
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

    if (ncnnFormat) {
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            result.error = QStringLiteral("Cannot create temporary directory for NCNN export.");
            return result;
        }
        const QString tempOnnxPath = tempDir.filePath(QStringLiteral("tiny_detector.onnx"));
        if (!writeTinyDetectorOnnxModel(checkpoint, tempOnnxPath, &result.error)) {
            return result;
        }

        const QString binPath = ncnnBinPathForParam(finalOutputPath);
        QString converterPath;
        if (!runOnnx2Ncnn(tempOnnxPath, finalOutputPath, binPath, &converterPath, &result.error)) {
            return result;
        }

        const QString reportPath = onnxExportReportPath(finalOutputPath);
        QJsonObject config = tinyDetectorExportConfig(checkpoint, checkpointPath, finalOutputPath, normalizedFormat);
        config.insert(QStringLiteral("ncnn"), ncnnMetadata(finalOutputPath, binPath, converterPath, tempOnnxPath));
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
