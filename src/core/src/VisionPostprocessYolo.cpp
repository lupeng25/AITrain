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
#include <QPointF>
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

struct YoloOutputLayout {
    int anchorCount = 0;
    int attributeCount = 0;
    bool attributesFirst = false;
};

bool selectYoloOutputLayout(const std::vector<int64_t>& shape, int minimumAttributeCount, YoloOutputLayout* layout)
{
    if (!layout || shape.size() != 3 || shape.at(0) != 1 || minimumAttributeCount <= 0) {
        return false;
    }

    const int64_t first = shape.at(1);
    const int64_t second = shape.at(2);
    if (first <= 0 || second <= 0
        || first > std::numeric_limits<int>::max()
        || second > std::numeric_limits<int>::max()) {
        return false;
    }

    const int64_t minimum = static_cast<int64_t>(minimumAttributeCount);
    const bool firstCanBeAttributes = first >= minimum;
    const bool secondCanBeAttributes = second >= minimum;
    if (!firstCanBeAttributes && !secondCanBeAttributes) {
        return false;
    }

    bool useFirstAsAttributes = firstCanBeAttributes;
    if (firstCanBeAttributes && secondCanBeAttributes) {
        const int64_t firstDistance = first - minimum;
        const int64_t secondDistance = second - minimum;
        useFirstAsAttributes = firstDistance == secondDistance
            ? first <= second
            : firstDistance < secondDistance;
    }

    layout->attributesFirst = useFirstAsAttributes;
    layout->attributeCount = static_cast<int>(useFirstAsAttributes ? first : second);
    layout->anchorCount = static_cast<int>(useFirstAsAttributes ? second : first);
    return true;
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
                || backend == QStringLiteral("ultralytics_yolo_segment")
                || backend == QStringLiteral("ultralytics_yolo_obb")) {
                return document.object();
            }
        }
    }
    return {};
}

QJsonObject loadUltralyticsTrainingReportFile(const QString& reportPath, const QString& referencePath)
{
    QString resolvedReportPath = reportPath;
    if (QFileInfo(resolvedReportPath).isRelative()) {
        resolvedReportPath = QDir(QFileInfo(referencePath).absolutePath()).absoluteFilePath(resolvedReportPath);
    }
    QFile file(QDir::cleanPath(resolvedReportPath));
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject()) {
        return {};
    }
    const QJsonObject report = document.object();
    const QString backend = report.value(QStringLiteral("backend")).toString();
    if (backend != QStringLiteral("ultralytics_yolo_detect")
        && backend != QStringLiteral("ultralytics_yolo_segment")
        && backend != QStringLiteral("ultralytics_yolo_obb")) {
        return {};
    }
    return report;
}

QStringList classNamesFromTrainingReport(const QJsonObject& report, const QString& referencePath)
{
    QString dataYaml = report.value(QStringLiteral("dataYaml")).toString();
    if (dataYaml.isEmpty()) {
        return {};
    }
    if (QFileInfo(dataYaml).isRelative()) {
        dataYaml = QDir(QFileInfo(referencePath).absolutePath()).absoluteFilePath(dataYaml);
    }
    return classNamesFromYoloDataYaml(QDir::cleanPath(dataYaml));
}

QStringList ultralyticsClassNames(const QString& onnxPath)
{
    const QJsonObject exportConfig = loadOnnxExportConfig(onnxPath);
    QStringList classNames = stringListFromArray(exportConfig.value(QStringLiteral("classNames")).toArray());
    if (!classNames.isEmpty()) {
        return classNames;
    }

    classNames = classNamesFromTrainingReport(exportConfig.value(QStringLiteral("trainingReport")).toObject(), onnxPath);
    if (!classNames.isEmpty()) {
        return classNames;
    }

    QString sourceTrainingReport = exportConfig.value(QStringLiteral("sourceTrainingReport")).toString();
    if (!sourceTrainingReport.isEmpty()) {
        if (QFileInfo(sourceTrainingReport).isRelative()) {
            sourceTrainingReport = QDir(QFileInfo(onnxPath).absolutePath()).absoluteFilePath(sourceTrainingReport);
        }
        classNames = classNamesFromTrainingReport(
            loadUltralyticsTrainingReportFile(sourceTrainingReport, onnxPath),
            sourceTrainingReport);
        if (!classNames.isEmpty()) {
            return classNames;
        }
    }

    const QJsonObject report = loadUltralyticsTrainingReport(onnxPath);
    classNames = classNamesFromTrainingReport(report, onnxPath);
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

DetectionBox yoloBoxFromInputCorners(
    double x1,
    double y1,
    double x2,
    double y2,
    int classId,
    const QSize& inputSize,
    const LetterboxTransform& transform)
{
    const double left = (qMin(x1, x2) - transform.padX) / qMax(1.0e-12, transform.scale);
    const double top = (qMin(y1, y2) - transform.padY) / qMax(1.0e-12, transform.scale);
    const double right = (qMax(x1, x2) - transform.padX) / qMax(1.0e-12, transform.scale);
    const double bottom = (qMax(y1, y2) - transform.padY) / qMax(1.0e-12, transform.scale);
    Q_UNUSED(inputSize)

    const double sourceWidth = qMax(1, transform.sourceSize.width());
    const double sourceHeight = qMax(1, transform.sourceSize.height());
    const double clampedX1 = qBound(0.0, left, sourceWidth);
    const double clampedY1 = qBound(0.0, top, sourceHeight);
    const double clampedX2 = qBound(0.0, right, sourceWidth);
    const double clampedY2 = qBound(0.0, bottom, sourceHeight);

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
    YoloOutputLayout layout;
    if (!selectYoloOutputLayout(shape, 4 + classCount, &layout)) {
        if (error) {
            *error = QStringLiteral("YOLO detection ONNX output does not contain box and class attributes");
        }
        return {};
    }
    const int anchorCount = layout.anchorCount;
    const int attributeCount = layout.attributeCount;
    const bool attributesFirst = layout.attributesFirst;

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

double signedPolygonArea(const QVector<QPointF>& polygon)
{
    if (polygon.size() < 3) {
        return 0.0;
    }
    double area = 0.0;
    for (int index = 0; index < polygon.size(); ++index) {
        const QPointF& current = polygon.at(index);
        const QPointF& next = polygon.at((index + 1) % polygon.size());
        area += current.x() * next.y() - next.x() * current.y();
    }
    return area * 0.5;
}

double polygonAreaPixels(const QVector<QPointF>& polygon)
{
    return qAbs(signedPolygonArea(polygon));
}

double crossProduct(const QPointF& a, const QPointF& b, const QPointF& c)
{
    return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
}

QPointF lineIntersection(const QPointF& p1, const QPointF& p2, const QPointF& q1, const QPointF& q2)
{
    const double a1 = p2.y() - p1.y();
    const double b1 = p1.x() - p2.x();
    const double c1 = a1 * p1.x() + b1 * p1.y();
    const double a2 = q2.y() - q1.y();
    const double b2 = q1.x() - q2.x();
    const double c2 = a2 * q1.x() + b2 * q1.y();
    const double determinant = a1 * b2 - a2 * b1;
    if (qAbs(determinant) < 1.0e-12) {
        return p2;
    }
    return QPointF((b2 * c1 - b1 * c2) / determinant, (a1 * c2 - a2 * c1) / determinant);
}

QVector<QPointF> clipConvexPolygon(const QVector<QPointF>& subject, const QVector<QPointF>& clipper)
{
    QVector<QPointF> output = subject;
    if (subject.size() < 3 || clipper.size() < 3) {
        return {};
    }
    const bool clipperCcw = signedPolygonArea(clipper) >= 0.0;
    for (int edge = 0; edge < clipper.size(); ++edge) {
        const QPointF a = clipper.at(edge);
        const QPointF b = clipper.at((edge + 1) % clipper.size());
        const QVector<QPointF> input = output;
        output.clear();
        if (input.isEmpty()) {
            break;
        }

        auto inside = [&](const QPointF& point) {
            const double cross = crossProduct(a, b, point);
            return clipperCcw ? cross >= -1.0e-9 : cross <= 1.0e-9;
        };

        QPointF previous = input.last();
        bool previousInside = inside(previous);
        for (const QPointF& current : input) {
            const bool currentInside = inside(current);
            if (currentInside) {
                if (!previousInside) {
                    output.append(lineIntersection(previous, current, a, b));
                }
                output.append(current);
            } else if (previousInside) {
                output.append(lineIntersection(previous, current, a, b));
            }
            previous = current;
            previousInside = currentInside;
        }
    }
    return output;
}

double polygonIou(const QVector<QPointF>& left, const QVector<QPointF>& right)
{
    const double leftArea = polygonAreaPixels(left);
    const double rightArea = polygonAreaPixels(right);
    if (leftArea <= 0.0 || rightArea <= 0.0) {
        return 0.0;
    }
    const QVector<QPointF> intersection = clipConvexPolygon(left, right);
    const double intersectionArea = polygonAreaPixels(intersection);
    const double unionArea = leftArea + rightArea - intersectionArea;
    return unionArea > 0.0 ? intersectionArea / unionArea : 0.0;
}

DetectionBox boundingBoxForPoints(const QVector<QPointF>& points, int classId, const QSize& sourceSize)
{
    const double sourceWidth = qMax(1, sourceSize.width());
    const double sourceHeight = qMax(1, sourceSize.height());
    double left = sourceWidth;
    double top = sourceHeight;
    double right = 0.0;
    double bottom = 0.0;
    for (const QPointF& point : points) {
        left = qMin(left, qBound(0.0, point.x(), sourceWidth));
        top = qMin(top, qBound(0.0, point.y(), sourceHeight));
        right = qMax(right, qBound(0.0, point.x(), sourceWidth));
        bottom = qMax(bottom, qBound(0.0, point.y(), sourceHeight));
    }
    DetectionBox box;
    box.classId = classId;
    box.xCenter = clamp01((left + right) / 2.0 / sourceWidth);
    box.yCenter = clamp01((top + bottom) / 2.0 / sourceHeight);
    box.width = qBound(1.0e-6, (right - left) / sourceWidth, 1.0);
    box.height = qBound(1.0e-6, (bottom - top) / sourceHeight, 1.0);
    return box;
}

QVector<QPointF> obbPointsFromInputPixels(
    double xCenter,
    double yCenter,
    double width,
    double height,
    double rotation,
    const LetterboxTransform& transform)
{
    const double scale = qMax(1.0e-12, transform.scale);
    const double cosValue = qCos(rotation);
    const double sinValue = qSin(rotation);
    const double halfWidth = width / 2.0;
    const double halfHeight = height / 2.0;
    const QVector<QPointF> local = {
        QPointF(-halfWidth, -halfHeight),
        QPointF(halfWidth, -halfHeight),
        QPointF(halfWidth, halfHeight),
        QPointF(-halfWidth, halfHeight)
    };
    QVector<QPointF> points;
    points.reserve(4);
    const double sourceWidth = qMax(1, transform.sourceSize.width());
    const double sourceHeight = qMax(1, transform.sourceSize.height());
    for (const QPointF& point : local) {
        const double inputX = xCenter + point.x() * cosValue - point.y() * sinValue;
        const double inputY = yCenter + point.x() * sinValue + point.y() * cosValue;
        const double sourceX = (inputX - transform.padX) / scale;
        const double sourceY = (inputY - transform.padY) / scale;
        points.append(QPointF(qBound(0.0, sourceX, sourceWidth), qBound(0.0, sourceY, sourceHeight)));
    }
    return points;
}

QVector<ObbPrediction> postProcessObbPredictions(QVector<ObbPrediction> candidates, const DetectionInferenceOptions& options)
{
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [options](const ObbPrediction& candidate) {
        return candidate.detection.confidence < options.confidenceThreshold || candidate.points.size() < 4;
    }), candidates.end());
    std::sort(candidates.begin(), candidates.end(), [](const ObbPrediction& left, const ObbPrediction& right) {
        return left.detection.confidence > right.detection.confidence;
    });

    QVector<ObbPrediction> selected;
    for (const ObbPrediction& candidate : candidates) {
        bool suppress = false;
        for (const ObbPrediction& accepted : selected) {
            if (candidate.detection.box.classId == accepted.detection.box.classId
                && polygonIou(candidate.points, accepted.points) > options.iouThreshold) {
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

QVector<ObbPrediction> yoloObbPredictionsFromOutput(
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
            *error = QStringLiteral("YOLO OBB ONNX output shape must be [1, attributes, anchors] or [1, anchors, attributes]");
        }
        return {};
    }

    const int classCount = qMax(1, classNames.size());
    YoloOutputLayout layout;
    if (!selectYoloOutputLayout(shape, 5 + classCount, &layout)) {
        if (error) {
            *error = QStringLiteral("YOLO OBB ONNX output does not contain xywh, class scores, and angle attributes");
        }
        return {};
    }
    const int anchorCount = layout.anchorCount;
    const int attributeCount = layout.attributeCount;
    const bool attributesFirst = layout.attributesFirst;

    const int usableClassCount = qMin(classCount, qMax(0, attributeCount - 5));
    if (usableClassCount <= 0) {
        if (error) {
            *error = QStringLiteral("YOLO OBB ONNX output does not contain class scores");
        }
        return {};
    }
    const int angleIndex = 4 + usableClassCount;

    auto valueAt = [output, anchorCount, attributeCount, attributesFirst](int anchor, int attribute) -> float {
        return attributesFirst
            ? output[attribute * anchorCount + anchor]
            : output[anchor * attributeCount + attribute];
    };

    QVector<ObbPrediction> candidates;
    candidates.reserve(anchorCount);
    const double sourceWidth = qMax(1, transform.sourceSize.width());
    const double sourceHeight = qMax(1, transform.sourceSize.height());
    const double scale = qMax(1.0e-12, transform.scale);
    for (int anchor = 0; anchor < anchorCount; ++anchor) {
        int bestClassIndex = 0;
        double bestClassScore = static_cast<double>(valueAt(anchor, 4));
        for (int classIndex = 1; classIndex < usableClassCount; ++classIndex) {
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

        const double inputX = static_cast<double>(valueAt(anchor, 0));
        const double inputY = static_cast<double>(valueAt(anchor, 1));
        const double inputWidth = qMax(0.0, static_cast<double>(valueAt(anchor, 2)));
        const double inputHeight = qMax(0.0, static_cast<double>(valueAt(anchor, 3)));
        if (inputWidth <= 0.0 || inputHeight <= 0.0) {
            continue;
        }
        const double rotation = static_cast<double>(valueAt(anchor, angleIndex));
        QVector<QPointF> points = obbPointsFromInputPixels(inputX, inputY, inputWidth, inputHeight, rotation, transform);
        if (polygonAreaPixels(points) <= 1.0e-6) {
            continue;
        }

        ObbPrediction prediction;
        prediction.detection.box = boundingBoxForPoints(points, bestClassIndex, transform.sourceSize);
        prediction.detection.className = bestClassIndex >= 0 && bestClassIndex < classNames.size()
            ? classNames.at(bestClassIndex)
            : QStringLiteral("class_%1").arg(bestClassIndex);
        prediction.detection.objectness = 1.0;
        prediction.detection.confidence = confidence;
        prediction.xCenter = qBound(0.0, (inputX - transform.padX) / scale, sourceWidth);
        prediction.yCenter = qBound(0.0, (inputY - transform.padY) / scale, sourceHeight);
        prediction.width = qMax(1.0e-6, inputWidth / scale);
        prediction.height = qMax(1.0e-6, inputHeight / scale);
        prediction.rotation = rotation;
        prediction.points = points;
        candidates.append(prediction);
    }
    return postProcessObbPredictions(candidates, options);
}

QVector<DetectionPrediction> yoloEndToEndPredictionsFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    const QStringList& classNames,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error)
{
    if (shape.size() != 3 || shape.at(0) != 1 || shape.at(1) <= 0 || shape.at(2) != 6) {
        if (error) {
            *error = QStringLiteral("YOLO end-to-end detection output shape must be [1, detections, 6]");
        }
        return {};
    }

    const int detectionCount = static_cast<int>(shape.at(1));
    constexpr int attributeCount = 6;

    auto valueAt = [output, attributeCount](int detection, int attribute) -> float {
        return output[detection * attributeCount + attribute];
    };

    QVector<DetectionPrediction> predictions;
    predictions.reserve(detectionCount);
    for (int detection = 0; detection < detectionCount; ++detection) {
        const double classValue = static_cast<double>(valueAt(detection, 5));
        const int classId = qRound(classValue);
        if (classId < 0 || qAbs(classValue - static_cast<double>(classId)) > 1.0e-3) {
            if (error) {
                *error = QStringLiteral("YOLO end-to-end detection class_id must be an integer attribute");
            }
            return {};
        }
        const double confidence = qBound(0.0, static_cast<double>(valueAt(detection, 4)), 1.0);
        if (confidence < options.confidenceThreshold) {
            continue;
        }
        DetectionPrediction prediction;
        prediction.box = yoloBoxFromInputCorners(
            static_cast<double>(valueAt(detection, 0)),
            static_cast<double>(valueAt(detection, 1)),
            static_cast<double>(valueAt(detection, 2)),
            static_cast<double>(valueAt(detection, 3)),
            classId,
            inputSize,
            transform);
        prediction.className = classId >= 0 && classId < classNames.size()
            ? classNames.at(classId)
            : QStringLiteral("class_%1").arg(classId);
        prediction.objectness = 1.0;
        prediction.confidence = confidence;
        predictions.append(prediction);
    }

    std::sort(predictions.begin(), predictions.end(), [](const DetectionPrediction& left, const DetectionPrediction& right) {
        return left.confidence > right.confidence;
    });
    if (predictions.size() > options.maxDetections) {
        predictions.resize(options.maxDetections);
    }
    return predictions;
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
    YoloOutputLayout layout;
    if (!selectYoloOutputLayout(boxesShape, 4 + maskDim + 1, &layout)) {
        if (error) {
            *error = QStringLiteral("YOLO segmentation ONNX output does not contain box and mask attributes");
        }
        return {};
    }
    const int anchorCount = layout.anchorCount;
    const int attributeCount = layout.attributeCount;
    const bool attributesFirst = layout.attributesFirst;

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

QVector<SegmentationPrediction> yoloEndToEndSegmentationPredictionsFromOutputs(
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
            *error = QStringLiteral("YOLO end-to-end segmentation outputs must be [1, detections, 6 + maskDim] and [1, maskDim, maskH, maskW]");
        }
        return {};
    }

    const int maskDim = static_cast<int>(prototypeShape.at(1));
    if (maskDim <= 0) {
        if (error) {
            *error = QStringLiteral("YOLO end-to-end segmentation prototype output does not contain mask channels");
        }
        return {};
    }

    const int expectedAttributeCount = 6 + maskDim;
    if (boxesShape.at(1) <= 0 || boxesShape.at(2) != expectedAttributeCount) {
        if (error) {
            *error = QStringLiteral("YOLO end-to-end segmentation outputs must be [1, detections, 6 + maskDim] and [1, maskDim, maskH, maskW]");
        }
        return {};
    }

    const int detectionCount = static_cast<int>(boxesShape.at(1));
    const int attributeCount = expectedAttributeCount;
    auto valueAt = [boxesAndMasks, attributeCount](int detection, int attribute) -> float {
        return boxesAndMasks[detection * attributeCount + attribute];
    };

    QVector<SegmentationCandidate> candidates;
    candidates.reserve(detectionCount);
    for (int detection = 0; detection < detectionCount; ++detection) {
        const double classValue = static_cast<double>(valueAt(detection, 5));
        const int classId = qRound(classValue);
        if (classId < 0 || qAbs(classValue - static_cast<double>(classId)) > 1.0e-3) {
            if (error) {
                *error = QStringLiteral("YOLO end-to-end segmentation class_id must be an integer attribute");
            }
            return {};
        }
        const double confidence = qBound(0.0, static_cast<double>(valueAt(detection, 4)), 1.0);
        if (confidence < options.confidenceThreshold) {
            continue;
        }

        SegmentationCandidate candidate;
        candidate.detection.box = yoloBoxFromInputCorners(
            static_cast<double>(valueAt(detection, 0)),
            static_cast<double>(valueAt(detection, 1)),
            static_cast<double>(valueAt(detection, 2)),
            static_cast<double>(valueAt(detection, 3)),
            classId,
            inputSize,
            transform);
        candidate.detection.className = classId >= 0 && classId < classNames.size()
            ? classNames.at(classId)
            : QStringLiteral("class_%1").arg(classId);
        candidate.detection.objectness = 1.0;
        candidate.detection.confidence = confidence;
        candidate.maskCoefficients.reserve(maskDim);
        for (int index = 0; index < maskDim; ++index) {
            candidate.maskCoefficients.append(valueAt(detection, 6 + index));
        }
        candidates.append(candidate);
    }

    std::sort(candidates.begin(), candidates.end(), [](const SegmentationCandidate& left, const SegmentationCandidate& right) {
        return left.detection.confidence > right.detection.confidence;
    });
    if (candidates.size() > options.maxDetections) {
        candidates.resize(options.maxDetections);
    }

    QVector<SegmentationPrediction> predictions;
    predictions.reserve(candidates.size());
    constexpr double maskThreshold = 0.5;
    for (const SegmentationCandidate& candidate : candidates) {
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
