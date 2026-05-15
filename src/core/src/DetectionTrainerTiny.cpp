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
namespace aitrain {

using namespace detection_detail;

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


} // namespace aitrain
