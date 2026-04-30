#include "aitrain/core/OcrRecTrainer.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace aitrain {
namespace {

int editDistance(const QString& left, const QString& right)
{
    QVector<int> previous(right.size() + 1);
    QVector<int> current(right.size() + 1);
    for (int j = 0; j <= right.size(); ++j) {
        previous[j] = j;
    }
    for (int i = 1; i <= left.size(); ++i) {
        current[0] = i;
        for (int j = 1; j <= right.size(); ++j) {
            const int cost = left.at(i - 1) == right.at(j - 1) ? 0 : 1;
            current[j] = qMin(qMin(previous.at(j) + 1, current.at(j - 1) + 1), previous.at(j - 1) + cost);
        }
        previous = current;
    }
    return previous.at(right.size());
}

struct OcrEvaluation {
    double accuracy = 0.0;
    double editDistance = 0.0;
};

OcrEvaluation evaluateBatch(const OcrRecBatch& batch, const OcrRecDictionary& dictionary)
{
    OcrEvaluation evaluation;
    if (batch.labels.isEmpty()) {
        return evaluation;
    }

    int exactMatches = 0;
    double distance = 0.0;
    for (int index = 0; index < batch.labels.size(); ++index) {
        const QString prediction = decodeOcrText(batch.labels.at(index), dictionary, false);
        const QString target = batch.texts.at(index);
        if (prediction == target) {
            ++exactMatches;
        }
        distance += static_cast<double>(editDistance(prediction, target));
    }

    evaluation.accuracy = static_cast<double>(exactMatches) / static_cast<double>(batch.labels.size());
    evaluation.editDistance = distance / static_cast<double>(batch.labels.size());
    return evaluation;
}

double averageLabelDensity(const OcrRecBatch& batch, int maxTextLength)
{
    if (batch.labelLengths.isEmpty()) {
        return 0.0;
    }

    double total = 0.0;
    const int denominator = qMax(1, maxTextLength);
    for (const int length : batch.labelLengths) {
        total += qMin(1.0, static_cast<double>(length) / static_cast<double>(denominator));
    }
    return total / static_cast<double>(batch.labelLengths.size());
}

QJsonArray encodedArray(const QVector<int>& encoded)
{
    QJsonArray array;
    for (const int token : encoded) {
        array.append(token);
    }
    return array;
}

bool writeCheckpoint(
    const QString& path,
    const QString& datasetPath,
    const OcrRecDataset& dataset,
    const OcrRecTrainingOptions& options,
    const OcrRecTrainingResult& result,
    QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write OCR checkpoint: %1").arg(path);
        }
        return false;
    }

    QJsonObject checkpoint;
    checkpoint.insert(QStringLiteral("type"), QStringLiteral("tiny_ocr_recognition_scaffold"));
    checkpoint.insert(QStringLiteral("datasetPath"), QDir::cleanPath(datasetPath));
    checkpoint.insert(QStringLiteral("labelFilePath"), dataset.labelFilePath());
    checkpoint.insert(QStringLiteral("dictionaryPath"), dataset.dictionary().path);
    checkpoint.insert(QStringLiteral("createdAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    checkpoint.insert(QStringLiteral("note"), QStringLiteral("Scaffold checkpoint for PaddleOCR Rec admission. Replace with native CRNN/CTC weights before claiming real OCR training."));
    checkpoint.insert(QStringLiteral("epochs"), options.epochs);
    checkpoint.insert(QStringLiteral("batchSize"), options.batchSize);
    checkpoint.insert(QStringLiteral("imageWidth"), options.imageSize.width());
    checkpoint.insert(QStringLiteral("imageHeight"), options.imageSize.height());
    checkpoint.insert(QStringLiteral("maxTextLength"), options.maxTextLength);
    checkpoint.insert(QStringLiteral("steps"), result.steps);
    checkpoint.insert(QStringLiteral("finalLoss"), result.finalLoss);
    checkpoint.insert(QStringLiteral("ctcLoss"), result.finalLoss);
    checkpoint.insert(QStringLiteral("accuracy"), result.accuracy);
    checkpoint.insert(QStringLiteral("editDistance"), result.editDistance);
    checkpoint.insert(QStringLiteral("dictionary"), QJsonArray::fromStringList(dataset.dictionary().characters));
    checkpoint.insert(QStringLiteral("modelHead"), QStringLiteral("label_echo_ctc_scaffold"));

    file.write(QJsonDocument(checkpoint).toJson(QJsonDocument::Indented));
    return true;
}

bool writePreview(const QString& path, const OcrRecDataset& dataset, QString* error)
{
    if (dataset.samples().isEmpty()) {
        if (error) {
            *error = QStringLiteral("Cannot write OCR preview because dataset is empty");
        }
        return false;
    }

    const OcrRecSample sample = dataset.samples().first();
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write OCR preview: %1").arg(path);
        }
        return false;
    }

    QJsonObject preview;
    preview.insert(QStringLiteral("imagePath"), sample.imagePath);
    preview.insert(QStringLiteral("label"), sample.label);
    preview.insert(QStringLiteral("prediction"), decodeOcrText(sample.encodedLabel, dataset.dictionary(), false));
    preview.insert(QStringLiteral("encodedLabel"), encodedArray(sample.encodedLabel));
    preview.insert(QStringLiteral("note"), QStringLiteral("OCR scaffold preview echoes the encoded label; this is not model inference."));
    file.write(QJsonDocument(preview).toJson(QJsonDocument::Indented));
    return true;
}

} // namespace

OcrRecTrainingResult trainOcrRecBaseline(
    const QString& datasetPath,
    const OcrRecTrainingOptions& options,
    const OcrRecTrainingCallback& callback)
{
    OcrRecTrainingResult result;

    QString error;
    OcrRecDataset dataset;
    if (!dataset.load(datasetPath, options.labelFilePath, options.dictionaryFilePath, options.maxTextLength, &error)) {
        result.error = error;
        return result;
    }

    const int epochs = qMax(1, options.epochs);
    const int batchSize = qMax(1, options.batchSize);
    const QSize imageSize = options.imageSize.isValid() && !options.imageSize.isEmpty()
        ? options.imageSize
        : QSize(100, 32);
    const int maxTextLength = qMax(1, options.maxTextLength);
    const int batchesPerEpoch = qMax(1, (dataset.size() + batchSize - 1) / batchSize);
    const int totalSteps = epochs * batchesPerEpoch;
    const double learningRate = qMax(0.001, options.learningRate);

    OcrRecDataLoader loader(dataset, batchSize, imageSize);
    double finalLoss = 0.0;
    OcrEvaluation finalEvaluation;
    int step = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        loader.reset();
        while (loader.hasNext()) {
            ++step;

            OcrRecBatch batch;
            if (!loader.next(&batch, &error)) {
                result.error = error;
                return result;
            }

            const OcrEvaluation evaluation = evaluateBatch(batch, dataset.dictionary());
            const double density = averageLabelDensity(batch, maxTextLength);
            const double progress = static_cast<double>(step) / static_cast<double>(qMax(1, totalSteps));
            const double ctcLoss = qMax(0.01, (1.0 - qMin(0.95, density)) / (1.0 + learningRate * 10.0 * progress * static_cast<double>(epochs)));
            finalLoss = ctcLoss;
            finalEvaluation = evaluation;

            if (callback) {
                OcrRecTrainingMetrics metrics;
                metrics.epoch = epoch;
                metrics.step = step;
                metrics.totalSteps = totalSteps;
                metrics.loss = ctcLoss;
                metrics.ctcLoss = ctcLoss;
                metrics.accuracy = evaluation.accuracy;
                metrics.editDistance = evaluation.editDistance;
                if (!callback(metrics)) {
                    result.error = QStringLiteral("OCR recognition scaffold training canceled");
                    return result;
                }
            }
        }
    }

    QString outputPath = options.outputPath;
    if (outputPath.isEmpty()) {
        outputPath = QDir(datasetPath).filePath(QStringLiteral("runs/ocr_recognition_scaffold"));
    }
    if (!QDir().mkpath(outputPath)) {
        result.error = QStringLiteral("Cannot create OCR output directory: %1").arg(outputPath);
        return result;
    }

    result.ok = true;
    result.steps = step;
    result.finalLoss = finalLoss;
    result.accuracy = finalEvaluation.accuracy;
    result.editDistance = finalEvaluation.editDistance;
    result.checkpointPath = QDir(outputPath).filePath(QStringLiteral("checkpoint_latest.aitrain"));
    if (!writeCheckpoint(result.checkpointPath, datasetPath, dataset, options, result, &result.error)) {
        result.ok = false;
        return result;
    }

    result.previewPath = QDir(outputPath).filePath(QStringLiteral("preview_latest.json"));
    if (!writePreview(result.previewPath, dataset, &result.error)) {
        result.ok = false;
        return result;
    }

    return result;
}

} // namespace aitrain
