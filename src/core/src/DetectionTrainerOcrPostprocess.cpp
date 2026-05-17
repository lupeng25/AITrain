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

} // namespace detection_detail

} // namespace aitrain
