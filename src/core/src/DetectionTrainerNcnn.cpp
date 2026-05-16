#include "DetectionTrainerInternal.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonValue>
#include <QRegularExpression>
#include <QSet>
#include <QtMath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#ifdef AITRAIN_WITH_NCNN
#include <net.h>
#endif

namespace aitrain {

using namespace detection_detail;

namespace {

struct NcnnParamMetadata {
    QString inputBlob;
    QStringList outputBlobs;
    QSize inputSize;
};

struct NcnnRuntimeConfig {
    bool hasSidecar = false;
    bool hasExplicitConfig = false;
    QString modelFamily;
    QStringList classNames;
    QString inputBlob;
    QStringList outputBlobs;
    QSize inputSize = QSize(640, 640);
    QString decoder = QStringLiteral("auto");
    QVector<int> strides = {8, 16, 32};
    int regMax = 16;
    QString binPath;
};

struct NcnnTensor {
    QVector<float> values;
    std::vector<int64_t> yoloShape;
    std::vector<int64_t> prototypeShape;
    int rows = 0;
    int columns = 0;
};

struct DflCandidate {
    DetectionPrediction detection;
    int anchorIndex = -1;
};

QString ncnnBinPathForParamPath(const QString& paramPath)
{
    QFileInfo info(paramPath);
    return info.absoluteDir().filePath(QStringLiteral("%1.bin").arg(info.completeBaseName()));
}

QStringList stringListFromJsonValue(const QJsonValue& value)
{
    QStringList result;
    if (value.isArray()) {
        for (const QJsonValue& item : value.toArray()) {
            const QString text = item.toString().trimmed();
            if (!text.isEmpty()) {
                result.append(text);
            }
        }
    } else {
        const QString text = value.toString().trimmed();
        if (!text.isEmpty()) {
            result = text.split(QLatin1Char(','), QString::SkipEmptyParts);
            for (QString& item : result) {
                item = item.trimmed();
            }
        }
    }
    return result;
}

QVector<int> intVectorFromJsonValue(const QJsonValue& value, const QVector<int>& fallback)
{
    QVector<int> result;
    if (value.isArray()) {
        for (const QJsonValue& item : value.toArray()) {
            const int number = item.toInt();
            if (number > 0) {
                result.append(number);
            }
        }
    }
    return result.isEmpty() ? fallback : result;
}

QSize inputSizeFromJsonValue(const QJsonValue& value, const QSize& fallback)
{
    if (value.isDouble()) {
        const int size = value.toInt();
        return size > 0 ? QSize(size, size) : fallback;
    }
    if (value.isObject()) {
        const QJsonObject object = value.toObject();
        const int width = object.value(QStringLiteral("width")).toInt(object.value(QStringLiteral("w")).toInt());
        const int height = object.value(QStringLiteral("height")).toInt(object.value(QStringLiteral("h")).toInt());
        if (width > 0 && height > 0) {
            return QSize(width, height);
        }
    }
    return fallback;
}

QString normalizedNcnnDecoder(QString decoder)
{
    decoder = decoder.trimmed().toLower();
    if (decoder == QStringLiteral("yolo_v8_detection")
        || decoder == QStringLiteral("yolo_v8_segmentation")
        || decoder == QStringLiteral("yolov8")
        || decoder == QStringLiteral("ultralytics")
        || decoder == QStringLiteral("ultralytics_yolo")) {
        return QStringLiteral("ultralytics_output");
    }
    if (decoder == QStringLiteral("pnnx")
        || decoder == QStringLiteral("yolov8_dfl")
        || decoder == QStringLiteral("dfl")) {
        return QStringLiteral("dfl");
    }
    return decoder.isEmpty() ? QStringLiteral("auto") : decoder;
}

QString decoderFromPostprocess(const QJsonObject& object)
{
    const QString decoder = normalizedNcnnDecoder(object.value(QStringLiteral("decoder")).toString());
    if (!decoder.isEmpty() && decoder != QStringLiteral("auto")) {
        return decoder;
    }
    return {};
}

void mergeNcnnConfigObject(NcnnRuntimeConfig* config, const QJsonObject& object)
{
    if (!config || object.isEmpty()) {
        return;
    }

    const QString inputBlob = object.value(QStringLiteral("inputBlob")).toString().trimmed();
    if (!inputBlob.isEmpty()) {
        config->inputBlob = inputBlob;
    }

    const QStringList outputBlobs = stringListFromJsonValue(object.value(QStringLiteral("outputBlobs")));
    if (!outputBlobs.isEmpty()) {
        config->outputBlobs = outputBlobs;
    }

    const QString decoder = normalizedNcnnDecoder(object.value(QStringLiteral("decoder")).toString());
    if (!decoder.isEmpty() && decoder != QStringLiteral("auto")) {
        config->decoder = decoder;
    }

    config->inputSize = inputSizeFromJsonValue(object.value(QStringLiteral("inputSize")), config->inputSize);
    config->strides = intVectorFromJsonValue(object.value(QStringLiteral("strides")), config->strides);
    config->regMax = qMax(1, object.value(QStringLiteral("regMax")).toInt(config->regMax));

    const QString binPath = object.value(QStringLiteral("binPath")).toString().trimmed();
    if (!binPath.isEmpty()) {
        config->binPath = binPath;
    }
}

void mergeTopLevelConfig(NcnnRuntimeConfig* config, const QJsonObject& object)
{
    if (!config || object.isEmpty()) {
        return;
    }

    const QString modelFamily = object.value(QStringLiteral("modelFamily")).toString().trimmed();
    if (!modelFamily.isEmpty()) {
        config->modelFamily = modelFamily;
    }

    const QString backend = object.value(QStringLiteral("backend")).toString();
    if (config->modelFamily.isEmpty()) {
        if (backend == QStringLiteral("ultralytics_yolo_segment")) {
            config->modelFamily = QStringLiteral("yolo_segmentation");
        } else if (backend == QStringLiteral("ultralytics_yolo_detect")) {
            config->modelFamily = QStringLiteral("yolo_detection");
        }
    }

    const QStringList classNames = stringListFromJsonValue(object.value(QStringLiteral("classNames")));
    if (!classNames.isEmpty()) {
        config->classNames = classNames;
    }

    const QString postprocessDecoder = decoderFromPostprocess(object.value(QStringLiteral("postprocess")).toObject());
    if (!postprocessDecoder.isEmpty()) {
        config->decoder = postprocessDecoder;
    }

    mergeNcnnConfigObject(config, object);
    mergeNcnnConfigObject(config, object.value(QStringLiteral("ncnn")).toObject());
}

NcnnParamMetadata parseNcnnParamMetadata(const QString& paramPath)
{
    NcnnParamMetadata metadata;
    QFile file(paramPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return metadata;
    }

    QSet<QString> produced;
    QSet<QString> consumed;
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#')) || line == QStringLiteral("7767517")) {
            continue;
        }
        const QStringList tokens = line.split(QRegularExpression(QStringLiteral("\\s+")), QString::SkipEmptyParts);
        if (tokens.size() < 4) {
            continue;
        }
        bool okBottom = false;
        bool okTop = false;
        const int bottomCount = tokens.at(2).toInt(&okBottom);
        const int topCount = tokens.at(3).toInt(&okTop);
        if (!okBottom || !okTop || bottomCount < 0 || topCount < 0 || tokens.size() < 4 + bottomCount + topCount) {
            continue;
        }

        for (int index = 0; index < bottomCount; ++index) {
            consumed.insert(tokens.at(4 + index));
        }
        QStringList topBlobs;
        for (int index = 0; index < topCount; ++index) {
            const QString blob = tokens.at(4 + bottomCount + index);
            topBlobs.append(blob);
            produced.insert(blob);
        }

        if (tokens.at(0) == QStringLiteral("Input") && !topBlobs.isEmpty()) {
            metadata.inputBlob = topBlobs.first();
            int width = 0;
            int height = 0;
            for (int index = 4 + bottomCount + topCount; index < tokens.size(); ++index) {
                const QString token = tokens.at(index);
                const int equalIndex = token.indexOf(QLatin1Char('='));
                if (equalIndex <= 0) {
                    continue;
                }
                const int key = token.left(equalIndex).toInt();
                const int value = token.mid(equalIndex + 1).toInt();
                if (key == 0) width = value;
                if (key == 1) height = value;
            }
            if (width > 0 && height > 0) {
                metadata.inputSize = QSize(width, height);
            }
        }
    }

    QStringList outputBlobs;
    for (const QString& blob : produced) {
        if (!consumed.contains(blob)) {
            outputBlobs.append(blob);
        }
    }
    outputBlobs.sort();
    metadata.outputBlobs = outputBlobs;
    return metadata;
}

NcnnRuntimeConfig resolveNcnnRuntimeConfig(const QString& paramPath, const QJsonObject& runtimeOptions = QJsonObject())
{
    NcnnRuntimeConfig config;
    config.binPath = ncnnBinPathForParamPath(paramPath);

    const NcnnParamMetadata paramMetadata = parseNcnnParamMetadata(paramPath);
    if (!paramMetadata.inputBlob.isEmpty()) {
        config.inputBlob = paramMetadata.inputBlob;
    }
    if (!paramMetadata.outputBlobs.isEmpty()) {
        config.outputBlobs = paramMetadata.outputBlobs;
    }
    if (paramMetadata.inputSize.isValid() && !paramMetadata.inputSize.isEmpty()) {
        config.inputSize = paramMetadata.inputSize;
    }

    const QJsonObject sidecar = loadOnnxExportConfig(paramPath);
    if (!sidecar.isEmpty()) {
        config.hasSidecar = true;
        mergeTopLevelConfig(&config, sidecar);
    }
    if (!runtimeOptions.isEmpty()) {
        config.hasExplicitConfig = true;
        mergeTopLevelConfig(&config, runtimeOptions);
    }

    config.decoder = normalizedNcnnDecoder(config.decoder);
    if (config.decoder == QStringLiteral("auto") && config.hasSidecar) {
        config.decoder = QStringLiteral("ultralytics_output");
    }
    if (config.inputSize.isEmpty()) {
        config.inputSize = QSize(640, 640);
    }
    if (config.binPath.isEmpty()) {
        config.binPath = ncnnBinPathForParamPath(paramPath);
    }
    return config;
}

bool validateNcnnRuntimeConfig(const NcnnRuntimeConfig& config, const QString& expectedFamily, QString* error)
{
    if (!config.hasSidecar && !config.hasExplicitConfig) {
        if (error) {
            *error = QStringLiteral("NCNN runtime inference requires an AITrain sidecar or explicit NCNN runtime config.");
        }
        return false;
    }
    if (config.modelFamily != expectedFamily) {
        if (error) {
            *error = QStringLiteral("NCNN runtime expected %1 but sidecar/config declares %2.")
                .arg(expectedFamily, config.modelFamily.isEmpty() ? QStringLiteral("<empty>") : config.modelFamily);
        }
        return false;
    }
    if (config.inputBlob.isEmpty()) {
        if (error) {
            *error = QStringLiteral("NCNN runtime config is missing inputBlob.");
        }
        return false;
    }
    if (config.outputBlobs.isEmpty()) {
        if (error) {
            *error = QStringLiteral("NCNN runtime config is missing outputBlobs.");
        }
        return false;
    }
    if (config.decoder != QStringLiteral("ultralytics_output") && config.decoder != QStringLiteral("dfl")) {
        if (error) {
            *error = QStringLiteral("Unsupported NCNN YOLO decoder: %1.").arg(config.decoder);
        }
        return false;
    }
    if (config.inputSize.isEmpty()) {
        if (error) {
            *error = QStringLiteral("NCNN runtime config has invalid inputSize.");
        }
        return false;
    }
    return true;
}

QStringList classNamesWithFallback(QStringList names, int classCount)
{
    while (names.size() < classCount) {
        names.append(QStringLiteral("class_%1").arg(names.size()));
    }
    if (names.size() > classCount) {
        names = names.mid(0, classCount);
    }
    return names;
}

double softmaxDistance(const QVector<double>& logits)
{
    if (logits.isEmpty()) {
        return 0.0;
    }
    const double maxLogit = *std::max_element(logits.cbegin(), logits.cend());
    double denominator = 0.0;
    QVector<double> probabilities;
    probabilities.reserve(logits.size());
    for (const double logitValue : logits) {
        const double value = qExp(logitValue - maxLogit);
        probabilities.append(value);
        denominator += value;
    }
    if (denominator <= 0.0) {
        return 0.0;
    }
    double distance = 0.0;
    for (int index = 0; index < probabilities.size(); ++index) {
        distance += static_cast<double>(index) * probabilities.at(index) / denominator;
    }
    return distance;
}

QVector<DflCandidate> dflCandidatesFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    const NcnnRuntimeConfig& config,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error)
{
    if (shape.size() != 3 || shape.at(0) != 1) {
        if (error) {
            *error = QStringLiteral("NCNN DFL output shape must be [1, anchors, attributes] or [1, attributes, anchors].");
        }
        return {};
    }

    const int regAttributes = config.regMax * 4;
    int anchorCount = 0;
    int attributeCount = 0;
    bool attributesFirst = false;
    if (shape.at(1) > 0 && shape.at(2) > regAttributes) {
        anchorCount = static_cast<int>(shape.at(1));
        attributeCount = static_cast<int>(shape.at(2));
    } else if (shape.at(1) > regAttributes && shape.at(2) > 0) {
        attributesFirst = true;
        attributeCount = static_cast<int>(shape.at(1));
        anchorCount = static_cast<int>(shape.at(2));
    } else {
        if (error) {
            *error = QStringLiteral("NCNN DFL output does not contain bbox distribution and class scores.");
        }
        return {};
    }
    const int classCount = attributeCount - regAttributes;
    if (classCount <= 0) {
        if (error) {
            *error = QStringLiteral("NCNN DFL output has no class scores.");
        }
        return {};
    }

    int expectedAnchors = 0;
    for (const int stride : config.strides) {
        expectedAnchors += (inputSize.width() / stride) * (inputSize.height() / stride);
    }
    if (expectedAnchors != anchorCount) {
        if (error) {
            *error = QStringLiteral("NCNN DFL output anchor count %1 does not match input/stride grid count %2.")
                .arg(anchorCount)
                .arg(expectedAnchors);
        }
        return {};
    }

    const QStringList classNames = classNamesWithFallback(config.classNames, classCount);
    auto valueAt = [output, anchorCount, attributeCount, attributesFirst](int anchor, int attribute) -> float {
        return attributesFirst
            ? output[attribute * anchorCount + anchor]
            : output[anchor * attributeCount + attribute];
    };

    QVector<DflCandidate> candidates;
    int anchorOffset = 0;
    for (const int stride : config.strides) {
        const int gridW = inputSize.width() / stride;
        const int gridH = inputSize.height() / stride;
        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                const int anchor = anchorOffset + y * gridW + x;
                int bestClass = 0;
                double bestScore = -std::numeric_limits<double>::infinity();
                for (int classIndex = 0; classIndex < classCount; ++classIndex) {
                    const double score = sigmoid(static_cast<double>(valueAt(anchor, regAttributes + classIndex)));
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = classIndex;
                    }
                }
                if (bestScore < options.confidenceThreshold) {
                    continue;
                }

                double distances[4] = {0.0, 0.0, 0.0, 0.0};
                for (int side = 0; side < 4; ++side) {
                    QVector<double> logits;
                    logits.reserve(config.regMax);
                    for (int index = 0; index < config.regMax; ++index) {
                        logits.append(static_cast<double>(valueAt(anchor, side * config.regMax + index)));
                    }
                    distances[side] = softmaxDistance(logits) * static_cast<double>(stride);
                }

                const double centerX = (static_cast<double>(x) + 0.5) * static_cast<double>(stride);
                const double centerY = (static_cast<double>(y) + 0.5) * static_cast<double>(stride);
                const double x0 = centerX - distances[0];
                const double y0 = centerY - distances[1];
                const double x1 = centerX + distances[2];
                const double y1 = centerY + distances[3];

                DflCandidate candidate;
                candidate.anchorIndex = anchor;
                candidate.detection.box = yoloBoxFromInputPixels(
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                    qMax(0.0, x1 - x0),
                    qMax(0.0, y1 - y0),
                    bestClass,
                    inputSize,
                    transform);
                candidate.detection.className = classNames.at(bestClass);
                candidate.detection.objectness = 1.0;
                candidate.detection.confidence = qBound(0.0, bestScore, 1.0);
                candidates.append(candidate);
            }
        }
        anchorOffset += gridW * gridH;
    }
    return candidates;
}

QVector<DflCandidate> selectDflCandidates(QVector<DflCandidate> candidates, const DetectionInferenceOptions& options)
{
    if (options.maxDetections <= 0 || candidates.isEmpty()) {
        return {};
    }
    std::sort(candidates.begin(), candidates.end(), [](const DflCandidate& left, const DflCandidate& right) {
        return left.detection.confidence > right.detection.confidence;
    });
    const double iouThreshold = qBound(0.0, options.iouThreshold, 1.0);
    QVector<DflCandidate> selected;
    for (const DflCandidate& candidate : candidates) {
        bool suppressed = false;
        for (const DflCandidate& accepted : selected) {
            if (candidate.detection.box.classId == accepted.detection.box.classId
                && boxIou(candidate.detection.box, accepted.detection.box) > iouThreshold) {
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

QVector<DetectionPrediction> predictionsFromDflCandidates(const QVector<DflCandidate>& candidates)
{
    QVector<DetectionPrediction> predictions;
    predictions.reserve(candidates.size());
    for (const DflCandidate& candidate : candidates) {
        predictions.append(candidate.detection);
    }
    return predictions;
}

QVector<SegmentationPrediction> segmentationFromDflOutputs(
    const NcnnTensor& boxes,
    const NcnnTensor& maskFeatures,
    const NcnnTensor& prototypes,
    const NcnnRuntimeConfig& config,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error)
{
    QVector<DflCandidate> candidates = dflCandidatesFromOutput(
        boxes.values.constData(),
        boxes.yoloShape,
        config,
        inputSize,
        transform,
        options,
        error);
    if (!error || error->isEmpty()) {
        candidates = selectDflCandidates(candidates, options);
    }
    if (candidates.isEmpty() || (error && !error->isEmpty())) {
        return {};
    }
    if (maskFeatures.rows <= 0 || maskFeatures.columns <= 0 || maskFeatures.rows < boxes.rows) {
        if (error) {
            *error = QStringLiteral("NCNN DFL segmentation mask feature output shape is invalid.");
        }
        return {};
    }
    if (prototypes.prototypeShape.size() != 4) {
        if (error) {
            *error = QStringLiteral("NCNN DFL segmentation prototype output shape is invalid.");
        }
        return {};
    }

    constexpr double maskThreshold = 0.5;
    QVector<SegmentationPrediction> predictions;
    predictions.reserve(candidates.size());
    for (const DflCandidate& candidate : candidates) {
        if (candidate.anchorIndex < 0 || candidate.anchorIndex >= maskFeatures.rows) {
            continue;
        }
        QVector<float> coefficients;
        coefficients.reserve(maskFeatures.columns);
        const int offset = candidate.anchorIndex * maskFeatures.columns;
        for (int index = 0; index < maskFeatures.columns; ++index) {
            coefficients.append(maskFeatures.values.at(offset + index));
        }

        SegmentationPrediction prediction;
        prediction.detection = candidate.detection;
        prediction.maskThreshold = maskThreshold;
        prediction.mask = maskFromPrototype(
            coefficients,
            prototypes.values.constData(),
            prototypes.prototypeShape,
            candidate.detection.box,
            inputSize,
            transform,
            maskThreshold,
            &prediction.maskArea);
        predictions.append(prediction);
    }
    return predictions;
}

#ifdef AITRAIN_WITH_NCNN

QVector<uchar> contiguousRgbPixels(const QImage& image)
{
    QVector<uchar> pixels;
    pixels.resize(image.width() * image.height() * 3);
    for (int y = 0; y < image.height(); ++y) {
        const uchar* scanline = image.constScanLine(y);
        memcpy(pixels.data() + y * image.width() * 3, scanline, image.width() * 3);
    }
    return pixels;
}

ncnn::Mat ncnnInputFromImage(const QImage& image, const QSize& inputSize, LetterboxTransform* transform)
{
    const QImage letterboxed = letterboxImage(image, inputSize, transform).convertToFormat(QImage::Format_RGB888);
    const QVector<uchar> pixels = contiguousRgbPixels(letterboxed);
    ncnn::Mat input = ncnn::Mat::from_pixels(pixels.constData(), ncnn::Mat::PIXEL_RGB, inputSize.width(), inputSize.height());
    const float normVals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    input.substract_mean_normalize(nullptr, normVals);
    return input;
}

NcnnTensor tensorFromMat(const ncnn::Mat& mat)
{
    NcnnTensor tensor;
    if (mat.empty()) {
        return tensor;
    }

    if (mat.dims == 1) {
        tensor.rows = 1;
        tensor.columns = mat.w;
        tensor.values.reserve(mat.w);
        const float* row = static_cast<const float*>(mat.data);
        for (int x = 0; x < mat.w; ++x) {
            tensor.values.append(row[x]);
        }
        tensor.yoloShape = {1, tensor.rows, tensor.columns};
    } else if (mat.dims == 2) {
        tensor.rows = mat.h;
        tensor.columns = mat.w;
        tensor.values.reserve(mat.w * mat.h);
        for (int y = 0; y < mat.h; ++y) {
            const float* row = mat.row(y);
            for (int x = 0; x < mat.w; ++x) {
                tensor.values.append(row[x]);
            }
        }
        tensor.yoloShape = {1, tensor.rows, tensor.columns};
    } else if (mat.dims == 3) {
        tensor.rows = mat.h * mat.c;
        tensor.columns = mat.w;
        tensor.values.reserve(mat.w * mat.h * mat.c);
        for (int c = 0; c < mat.c; ++c) {
            const ncnn::Mat channel = mat.channel(c);
            for (int y = 0; y < mat.h; ++y) {
                const float* row = channel.row(y);
                for (int x = 0; x < mat.w; ++x) {
                    tensor.values.append(row[x]);
                }
            }
        }
        tensor.yoloShape = {1, tensor.rows, tensor.columns};
        tensor.prototypeShape = {1, mat.c, mat.h, mat.w};
    }
    return tensor;
}

bool loadNcnnNet(const QString& paramPath, const QString& binPath, ncnn::Net* net, QString* error)
{
    if (!net) {
        return false;
    }
    net->clear();
    net->opt = ncnn::Option();
    net->opt.num_threads = 1;
    net->opt.use_packing_layout = false;
#if NCNN_VULKAN
    net->opt.use_vulkan_compute = false;
#endif

    const QByteArray paramBytes = QFile::encodeName(paramPath);
    const QByteArray binBytes = QFile::encodeName(binPath);
    if (net->load_param(paramBytes.constData()) != 0) {
        if (error) {
            *error = QStringLiteral("NCNN failed to load param file: %1").arg(paramPath);
        }
        return false;
    }
    if (net->load_model(binBytes.constData()) != 0) {
        if (error) {
            *error = QStringLiteral("NCNN failed to load bin file: %1").arg(binPath);
        }
        return false;
    }
    return true;
}

bool runNcnnModel(
    const QString& paramPath,
    const NcnnRuntimeConfig& config,
    const QImage& image,
    QVector<NcnnTensor>* outputs,
    LetterboxTransform* transform,
    QString* error)
{
    ncnn::Net net;
    if (!loadNcnnNet(paramPath, config.binPath, &net, error)) {
        return false;
    }
    ncnn::Mat input = ncnnInputFromImage(image, config.inputSize, transform);
    ncnn::Extractor extractor = net.create_extractor();
    const QByteArray inputBlob = config.inputBlob.toUtf8();
    if (extractor.input(inputBlob.constData(), input) != 0) {
        if (error) {
            *error = QStringLiteral("NCNN failed to bind input blob: %1").arg(config.inputBlob);
        }
        return false;
    }

    outputs->clear();
    outputs->reserve(config.outputBlobs.size());
    for (const QString& outputBlob : config.outputBlobs) {
        ncnn::Mat output;
        const QByteArray outputName = outputBlob.toUtf8();
        if (extractor.extract(outputName.constData(), output) != 0) {
            if (error) {
                *error = QStringLiteral("NCNN failed to extract output blob: %1").arg(outputBlob);
            }
            return false;
        }
        outputs->append(tensorFromMat(output));
    }
    return true;
}

#endif

} // namespace

namespace detection_detail {

namespace {

void setTensorMatrixShapeFromYoloShape(
    NcnnTensor* tensor,
    const std::vector<int64_t>& shape,
    int regAttributes,
    int classCount)
{
    if (!tensor || shape.size() != 3 || shape.at(0) != 1) {
        return;
    }
    const int minimumAttributes = regAttributes + qMax(1, classCount);
    if (shape.at(1) > 0 && shape.at(2) >= minimumAttributes) {
        tensor->rows = static_cast<int>(shape.at(1));
        tensor->columns = static_cast<int>(shape.at(2));
    } else if (shape.at(1) >= minimumAttributes && shape.at(2) > 0) {
        tensor->rows = static_cast<int>(shape.at(2));
        tensor->columns = static_cast<int>(shape.at(1));
    }
}

NcnnRuntimeConfig dflConfigForInternalDecode(
    const QStringList& classNames,
    const QVector<int>& strides,
    int regMax)
{
    NcnnRuntimeConfig config;
    config.hasExplicitConfig = true;
    config.modelFamily = QStringLiteral("yolo_detection");
    config.classNames = classNames;
    config.decoder = QStringLiteral("dfl");
    config.strides = strides.isEmpty() ? QVector<int>{8, 16, 32} : strides;
    config.regMax = qMax(1, regMax);
    return config;
}

} // namespace

QVector<DetectionPrediction> ncnnDflPredictionsFromOutput(
    const float* output,
    const std::vector<int64_t>& shape,
    const QStringList& classNames,
    const QVector<int>& strides,
    int regMax,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error)
{
    if (!output) {
        if (error) {
            *error = QStringLiteral("NCNN DFL output is null.");
        }
        return {};
    }
    NcnnRuntimeConfig config = dflConfigForInternalDecode(classNames, strides, regMax);
    config.inputSize = inputSize;
    const QVector<DflCandidate> candidates = dflCandidatesFromOutput(
        output,
        shape,
        config,
        inputSize,
        transform,
        options,
        error);
    return predictionsFromDflCandidates(selectDflCandidates(candidates, options));
}

QVector<SegmentationPrediction> ncnnDflSegmentationPredictionsFromOutputs(
    const float* boxes,
    const std::vector<int64_t>& boxesShape,
    const float* maskFeatures,
    int maskFeatureRows,
    int maskFeatureColumns,
    const float* prototypes,
    const std::vector<int64_t>& prototypeShape,
    const QStringList& classNames,
    const QVector<int>& strides,
    int regMax,
    const QSize& inputSize,
    const LetterboxTransform& transform,
    const DetectionInferenceOptions& options,
    QString* error)
{
    if (!boxes || !maskFeatures || !prototypes) {
        if (error) {
            *error = QStringLiteral("NCNN DFL segmentation output is null.");
        }
        return {};
    }

    NcnnRuntimeConfig config = dflConfigForInternalDecode(classNames, strides, regMax);
    config.modelFamily = QStringLiteral("yolo_segmentation");
    config.inputSize = inputSize;

    NcnnTensor boxesTensor;
    boxesTensor.yoloShape = boxesShape;
    const int64_t boxesValueCount = boxesShape.size() == 3
        ? boxesShape.at(0) * boxesShape.at(1) * boxesShape.at(2)
        : 0;
    if (boxesValueCount > 0) {
        boxesTensor.values.reserve(static_cast<int>(boxesValueCount));
        for (int64_t index = 0; index < boxesValueCount; ++index) {
            boxesTensor.values.append(boxes[index]);
        }
    }
    setTensorMatrixShapeFromYoloShape(&boxesTensor, boxesShape, config.regMax * 4, classNames.size());

    NcnnTensor maskTensor;
    maskTensor.rows = maskFeatureRows;
    maskTensor.columns = maskFeatureColumns;
    const int maskValueCount = qMax(0, maskFeatureRows) * qMax(0, maskFeatureColumns);
    maskTensor.values.reserve(maskValueCount);
    for (int index = 0; index < maskValueCount; ++index) {
        maskTensor.values.append(maskFeatures[index]);
    }

    NcnnTensor prototypeTensor;
    prototypeTensor.prototypeShape = prototypeShape;
    const int64_t prototypeValueCount = prototypeShape.size() == 4
        ? prototypeShape.at(0) * prototypeShape.at(1) * prototypeShape.at(2) * prototypeShape.at(3)
        : 0;
    if (prototypeValueCount > 0) {
        prototypeTensor.values.reserve(static_cast<int>(prototypeValueCount));
        for (int64_t index = 0; index < prototypeValueCount; ++index) {
            prototypeTensor.values.append(prototypes[index]);
        }
    }

    return segmentationFromDflOutputs(
        boxesTensor,
        maskTensor,
        prototypeTensor,
        config,
        inputSize,
        transform,
        options,
        error);
}

} // namespace detection_detail

QJsonObject NcnnBackendStatus::toJson() const
{
    return QJsonObject{
        {QStringLiteral("sdkAvailable"), sdkAvailable},
        {QStringLiteral("inferenceAvailable"), inferenceAvailable},
        {QStringLiteral("status"), status},
        {QStringLiteral("message"), message}
    };
}

NcnnBackendStatus ncnnBackendStatus()
{
    NcnnBackendStatus status;
#ifdef AITRAIN_WITH_NCNN
    status.sdkAvailable = true;
    status.inferenceAvailable = true;
    status.status = QStringLiteral("backend_available");
    status.message = QStringLiteral("NCNN SDK is compiled into this build. Runtime inference is available for supported YOLO detection and segmentation NCNN artifacts.");
#else
    status.sdkAvailable = false;
    status.inferenceAvailable = false;
    status.status = QStringLiteral("sdk_missing");
    status.message = QStringLiteral("NCNN inference is not enabled. Configure AITRAIN_NCNN_ROOT with an NCNN SDK and rebuild.");
#endif
    return status;
}

bool isNcnnInferenceAvailable()
{
    return ncnnBackendStatus().inferenceAvailable;
}

QString inferNcnnModelFamily(const QString& paramPath)
{
    const NcnnRuntimeConfig config = resolveNcnnRuntimeConfig(paramPath);
    return config.modelFamily;
}

QVector<DetectionPrediction> predictDetectionNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error)
{
    return predictDetectionNcnnRuntime(paramPath, imagePath, options, QJsonObject(), error);
}

QVector<DetectionPrediction> predictDetectionNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    const QJsonObject& runtimeOptions,
    QString* error)
{
#ifndef AITRAIN_WITH_NCNN
    Q_UNUSED(paramPath)
    Q_UNUSED(imagePath)
    Q_UNUSED(options)
    Q_UNUSED(runtimeOptions)
    if (error) {
        *error = ncnnBackendStatus().message;
    }
    return {};
#else
    if (!QFileInfo::exists(paramPath)) {
        if (error) {
            *error = QStringLiteral("NCNN param file does not exist: %1").arg(paramPath);
        }
        return {};
    }
    const NcnnRuntimeConfig config = resolveNcnnRuntimeConfig(paramPath, runtimeOptions);
    if (!QFileInfo::exists(config.binPath)) {
        if (error) {
            *error = QStringLiteral("NCNN bin file does not exist: %1").arg(config.binPath);
        }
        return {};
    }
    if (!validateNcnnRuntimeConfig(config, QStringLiteral("yolo_detection"), error)) {
        return {};
    }

    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for NCNN detection prediction: %1").arg(imagePath);
        }
        return {};
    }

    QVector<NcnnTensor> outputs;
    LetterboxTransform transform;
    if (!runNcnnModel(paramPath, config, image, &outputs, &transform, error)) {
        return {};
    }
    if (outputs.isEmpty()) {
        if (error) {
            *error = QStringLiteral("NCNN detection model did not produce outputs.");
        }
        return {};
    }
    if (config.decoder == QStringLiteral("dfl")) {
        const QVector<DflCandidate> candidates = dflCandidatesFromOutput(
            outputs.first().values.constData(),
            outputs.first().yoloShape,
            config,
            config.inputSize,
            transform,
            options,
            error);
        return predictionsFromDflCandidates(selectDflCandidates(candidates, options));
    }
    return yoloPredictionsFromOutput(
        outputs.first().values.constData(),
        outputs.first().yoloShape,
        config.classNames,
        config.inputSize,
        transform,
        options,
        error);
#endif
}

QVector<SegmentationPrediction> predictSegmentationNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    QString* error)
{
    return predictSegmentationNcnnRuntime(paramPath, imagePath, options, QJsonObject(), error);
}

QVector<SegmentationPrediction> predictSegmentationNcnnRuntime(
    const QString& paramPath,
    const QString& imagePath,
    const DetectionInferenceOptions& options,
    const QJsonObject& runtimeOptions,
    QString* error)
{
#ifndef AITRAIN_WITH_NCNN
    Q_UNUSED(paramPath)
    Q_UNUSED(imagePath)
    Q_UNUSED(options)
    Q_UNUSED(runtimeOptions)
    if (error) {
        *error = ncnnBackendStatus().message;
    }
    return {};
#else
    if (!QFileInfo::exists(paramPath)) {
        if (error) {
            *error = QStringLiteral("NCNN param file does not exist: %1").arg(paramPath);
        }
        return {};
    }
    const NcnnRuntimeConfig config = resolveNcnnRuntimeConfig(paramPath, runtimeOptions);
    if (!QFileInfo::exists(config.binPath)) {
        if (error) {
            *error = QStringLiteral("NCNN bin file does not exist: %1").arg(config.binPath);
        }
        return {};
    }
    if (!validateNcnnRuntimeConfig(config, QStringLiteral("yolo_segmentation"), error)) {
        return {};
    }

    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot read image for NCNN segmentation prediction: %1").arg(imagePath);
        }
        return {};
    }

    QVector<NcnnTensor> outputs;
    LetterboxTransform transform;
    if (!runNcnnModel(paramPath, config, image, &outputs, &transform, error)) {
        return {};
    }
    if (config.decoder == QStringLiteral("dfl")) {
        if (outputs.size() < 3) {
            if (error) {
                *error = QStringLiteral("NCNN DFL segmentation requires out0 boxes, out1 mask features, and out2 prototypes.");
            }
            return {};
        }
        return segmentationFromDflOutputs(
            outputs.at(0),
            outputs.at(1),
            outputs.at(2),
            config,
            config.inputSize,
            transform,
            options,
            error);
    }

    if (outputs.size() < 2) {
        if (error) {
            *error = QStringLiteral("NCNN YOLO segmentation requires boxes/masks and prototype outputs.");
        }
        return {};
    }

    int boxesIndex = 0;
    int prototypeIndex = 1;
    for (int index = 0; index < outputs.size(); ++index) {
        if (outputs.at(index).prototypeShape.size() == 4) {
            prototypeIndex = index;
        } else {
            boxesIndex = index;
        }
    }
    return yoloSegmentationPredictionsFromOutputs(
        outputs.at(boxesIndex).values.constData(),
        outputs.at(boxesIndex).yoloShape,
        outputs.at(prototypeIndex).values.constData(),
        outputs.at(prototypeIndex).prototypeShape,
        config.classNames,
        config.inputSize,
        transform,
        options,
        error);
#endif
}

} // namespace aitrain
