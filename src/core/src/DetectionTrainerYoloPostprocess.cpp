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

} // namespace detection_detail

} // namespace aitrain
