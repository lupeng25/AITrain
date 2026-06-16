#include "aitrain/core/DatasetValidators.h"

#include "YoloDatasetLayout.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QMap>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QSize>
#include <QSet>
#include <QTextStream>

namespace aitrain {
namespace {

constexpr int kDefaultMaxIssues = 100;
constexpr int kDefaultMaxFiles = 5000;

struct YoloSample {
    QString imagePath;
    QString labelPath;
    QString fileName;
    QString baseName;
};

struct OcrSample {
    QString imagePath;
    QString text;
    QString fileName;
};

struct OcrDetSample {
    QString imagePath;
    QString relativeImagePath;
    QString labelJson;
    QString fileName;
};

struct SemanticMaskSample {
    QString imagePath;
    QString maskPath;
    QString fileName;
    QString baseName;
    QString split;
};

void addIssue(DatasetValidationResult& result,
    const QString& severity,
    const QString& code,
    const QString& filePath,
    int line,
    const QString& message)
{
    DatasetValidationResult::Issue issue;
    issue.severity = severity;
    issue.code = code;
    issue.filePath = filePath;
    issue.line = line;
    issue.message = message;
    result.issues.append(issue);

    const QString location = line > 0
        ? QStringLiteral("%1:%2").arg(filePath).arg(line)
        : filePath;
    const QString text = filePath.isEmpty()
        ? message
        : QStringLiteral("%1 %2").arg(location, message);
    if (severity == QStringLiteral("error")) {
        result.ok = false;
        result.errors.append(text);
    } else {
        result.warnings.append(text);
    }
}

bool issueLimitReached(DatasetValidationResult& result, int maxIssues)
{
    if (result.issues.size() < maxIssues) {
        return false;
    }
    addIssue(result, QStringLiteral("warning"), QStringLiteral("truncated"), QString(), 0,
        QStringLiteral("校验问题过多，已截断结果。"));
    return true;
}

QStringList splitFields(const QString& line)
{
    return line.split(QRegularExpression(QStringLiteral("\\s+")),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
        QString::SkipEmptyParts
#else
        Qt::SkipEmptyParts
#endif
    );
}

QStringList imageNameFilters()
{
    return {
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.png"),
        QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff")
    };
}

QFileInfoList imageFiles(const QDir& directory)
{
    QFileInfoList files;
    for (const QString& filter : imageNameFilters()) {
        files.append(directory.entryInfoList({filter}, QDir::Files, QDir::Name));
    }
    return files;
}

QString canonicalImageKey(const QFileInfo& imageInfo)
{
    QString key = imageInfo.canonicalFilePath();
    if (key.isEmpty()) {
        key = imageInfo.absoluteFilePath();
    }
    key = QDir::cleanPath(key);
#if defined(Q_OS_WIN)
    key = key.toLower();
#endif
    return key;
}

bool validateReadableImageFile(
    const QString& imagePath,
    DatasetValidationResult& result,
    const QString& code,
    const QString& context)
{
    const QFileInfo imageInfo(imagePath);
    if (!imageInfo.exists()) {
        return false;
    }
    if (imageInfo.size() <= 0) {
        addIssue(result, QStringLiteral("error"), code, imagePath, 0,
            QStringLiteral("%1图片文件为空，无法用于训练。").arg(context));
        return false;
    }

    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    if (!reader.canRead()) {
        const QString readerError = reader.errorString().trimmed();
        addIssue(result, QStringLiteral("error"), code, imagePath, 0,
            readerError.isEmpty()
                ? QStringLiteral("%1图片无法解码。").arg(context)
                : QStringLiteral("%1图片无法解码：%2。").arg(context, readerError));
        return false;
    }
    const QSize size = reader.size();
    if (!size.isValid() || size.isEmpty()) {
        addIssue(result, QStringLiteral("error"), code, imagePath, 0,
            QStringLiteral("%1图片尺寸无效，无法用于训练。").arg(context));
        return false;
    }
    return true;
}

QStringList readClassesTxt(const QString& classesPath, DatasetValidationResult& result)
{
    QStringList classes;
    QFile file(classesPath);
    if (!file.exists()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("missing_classes_txt"), classesPath, 0,
            QStringLiteral("语义分割 Mask PNG 数据集缺少 classes.txt。"));
        return classes;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("classes_txt_unreadable"), classesPath, 0,
            QStringLiteral("无法读取 classes.txt。"));
        return classes;
    }
    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            addIssue(result, QStringLiteral("warning"), QStringLiteral("empty_class_name"), classesPath, lineNumber,
                QStringLiteral("classes.txt 包含空类别名，已忽略该行。"));
            continue;
        }
        classes.append(line);
    }
    if (classes.isEmpty()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("empty_classes_txt"), classesPath, 0,
            QStringLiteral("classes.txt 至少需要一个类别名，0 默认作为背景类别。"));
    }
    return classes;
}

bool semanticMaskFormatAcceptable(const QImage& mask)
{
    return mask.format() == QImage::Format_Grayscale8
#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
        || mask.format() == QImage::Format_Alpha8
#endif
        || mask.format() == QImage::Format_Indexed8
        || mask.allGray();
}

QJsonObject semanticClassPixelCounts(const QString& datasetPath, int classCount, int ignoreIndex)
{
    QJsonObject counts;
    const QDir root(datasetPath);
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir maskDir(root.filePath(QStringLiteral("masks/%1").arg(split)));
        if (!maskDir.exists()) {
            continue;
        }
        const QFileInfoList masks = maskDir.entryInfoList({QStringLiteral("*.png")}, QDir::Files, QDir::Name);
        for (const QFileInfo& maskInfo : masks) {
            const QImage mask(maskInfo.absoluteFilePath());
            if (mask.isNull()) {
                continue;
            }
            for (int y = 0; y < mask.height(); ++y) {
                for (int x = 0; x < mask.width(); ++x) {
                    const int classId = qGray(mask.pixel(x, y));
                    if (classId == ignoreIndex || classId < 0 || (classCount > 0 && classId >= classCount)) {
                        continue;
                    }
                    const QString key = QString::number(classId);
                    counts.insert(key, counts.value(key).toDouble() + 1.0);
                }
            }
        }
    }
    return counts;
}

int parseClassCount(const QString& yamlPath, DatasetValidationResult& result)
{
    QFile file(yamlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("data_yaml_unreadable"), yamlPath, 0,
            QStringLiteral("无法读取 data.yaml。"));
        return -1;
    }

    int classCount = -1;
    int namesCount = -1;
    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        QString line = QString::fromUtf8(file.readLine()).trimmed();
        const int commentIndex = line.indexOf(QLatin1Char('#'));
        if (commentIndex >= 0) {
            line = line.left(commentIndex).trimmed();
        }
        if (line.startsWith(QStringLiteral("nc:"))) {
            bool ok = false;
            classCount = line.mid(3).trimmed().toInt(&ok);
            if (!ok || classCount <= 0) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_nc"), yamlPath, lineNumber,
                    QStringLiteral("nc 必须是正整数。"));
            }
        } else if (line.startsWith(QStringLiteral("names:"))) {
            const QString names = line.mid(6).trimmed();
            if (names.startsWith(QLatin1Char('[')) && names.endsWith(QLatin1Char(']'))) {
                const QString inner = names.mid(1, names.size() - 2);
                namesCount = inner.split(QLatin1Char(','),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
                    QString::SkipEmptyParts
#else
                    Qt::SkipEmptyParts
#endif
                ).size();
            }
        }
    }

    if (classCount < 0 && namesCount > 0) {
        classCount = namesCount;
    }
    if (classCount < 0) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("missing_class_count"), yamlPath, 0,
            QStringLiteral("data.yaml 缺少 nc 或 names。"));
    } else if (namesCount > 0 && namesCount != classCount) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("class_count_mismatch"), yamlPath, 0,
            QStringLiteral("names 数量与 nc 不一致。"));
    }
    return classCount;
}

bool parseNormalizedDouble(const QString& token, double* value)
{
    bool ok = false;
    const double parsed = token.toDouble(&ok);
    if (!ok || parsed < 0.0 || parsed > 1.0) {
        return false;
    }
    if (value) {
        *value = parsed;
    }
    return true;
}

void validateClassId(const QString& token,
    int classCount,
    DatasetValidationResult& result,
    const QString& filePath,
    int lineNumber)
{
    bool ok = false;
    const int classId = token.toInt(&ok);
    if (!ok || classId < 0) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_class_id"), filePath, lineNumber,
            QStringLiteral("class id 必须是非负整数。"));
        return;
    }
    if (classCount > 0 && classId >= classCount) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("class_id_out_of_range"), filePath, lineNumber,
            QStringLiteral("class id 超出 data.yaml 的类别范围。"));
    }
}

double polygonArea(const QVector<double>& coordinates)
{
    double area = 0.0;
    const int points = coordinates.size() / 2;
    for (int index = 0; index < points; ++index) {
        const int next = (index + 1) % points;
        area += coordinates.at(index * 2) * coordinates.at(next * 2 + 1);
        area -= coordinates.at(next * 2) * coordinates.at(index * 2 + 1);
    }
    return qAbs(area) * 0.5;
}

void validateLabelFile(const QFileInfo& labelInfo,
    int classCount,
    bool segmentation,
    bool allowEmptyLabels,
    DatasetValidationResult& result,
    int maxIssues)
{
    QFile file(labelInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("label_unreadable"), labelInfo.absoluteFilePath(), 0,
            QStringLiteral("无法读取标注文件。"));
        return;
    }

    bool hasRows = false;
    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        hasRows = true;
        const QStringList parts = splitFields(line);
        if (!segmentation && parts.size() != 5) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_yolo_detection_row"), labelInfo.absoluteFilePath(), lineNumber,
                QStringLiteral("YOLO 检测标注必须是 5 列：class x_center y_center width height。"));
            if (issueLimitReached(result, maxIssues)) return;
            continue;
        }
        if (segmentation && (parts.size() < 7 || parts.size() % 2 == 0)) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_yolo_segmentation_row"), labelInfo.absoluteFilePath(), lineNumber,
                QStringLiteral("YOLO 分割标注必须是 class 后接至少 3 个 polygon 点，坐标数量为偶数。"));
            if (issueLimitReached(result, maxIssues)) return;
            continue;
        }

        validateClassId(parts.first(), classCount, result, labelInfo.absoluteFilePath(), lineNumber);
        QVector<double> coordinates;
        for (int index = 1; index < parts.size(); ++index) {
            double value = 0.0;
            if (!parseNormalizedDouble(parts.at(index), &value)) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("coordinate_out_of_range"), labelInfo.absoluteFilePath(), lineNumber,
                    QStringLiteral("坐标必须是 [0,1] 范围内的数字。"));
                break;
            }
            coordinates.append(value);
        }
        if (!segmentation && coordinates.size() == 4 && (coordinates.at(2) <= 0.0 || coordinates.at(3) <= 0.0)) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_bbox_size"), labelInfo.absoluteFilePath(), lineNumber,
                QStringLiteral("bbox 宽高必须大于 0。"));
        }
        if (segmentation && coordinates.size() >= 6 && polygonArea(coordinates) < 0.000001) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("polygon_too_small"), labelInfo.absoluteFilePath(), lineNumber,
                QStringLiteral("polygon 面积过小。"));
        }
        if (issueLimitReached(result, maxIssues)) {
            return;
        }
    }

    if (!hasRows && !allowEmptyLabels) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("empty_label"), labelInfo.absoluteFilePath(), 0,
            QStringLiteral("标注文件为空，当前配置不允许空标注图片。"));
    }
}

QString yoloLabelSummary(const QString& labelPath, bool segmentation)
{
    QFile file(labelPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return QStringLiteral("标注不可读");
    }

    int rows = 0;
    int minClass = -1;
    int maxClass = -1;
    int maxPoints = 0;
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const QStringList parts = splitFields(line);
        if (parts.isEmpty()) {
            continue;
        }
        bool ok = false;
        const int classId = parts.first().toInt(&ok);
        if (ok) {
            minClass = minClass < 0 ? classId : qMin(minClass, classId);
            maxClass = maxClass < 0 ? classId : qMax(maxClass, classId);
        }
        if (segmentation && parts.size() > 1) {
            maxPoints = qMax(maxPoints, (parts.size() - 1) / 2);
        }
        ++rows;
    }
    if (rows == 0) {
        return QStringLiteral("空标注");
    }
    const QString classText = minClass == maxClass
        ? QStringLiteral("class=%1").arg(minClass)
        : QStringLiteral("class=%1..%2").arg(minClass).arg(maxClass);
    return segmentation
        ? QStringLiteral("polygon=%1, maxPoints=%2, %3").arg(rows).arg(maxPoints).arg(classText)
        : QStringLiteral("bbox=%1, %2").arg(rows).arg(classText);
}

DatasetValidationResult validateYoloDataset(const QString& datasetPath, const QJsonObject& options, bool segmentation)
{
    DatasetValidationResult result;
    const int maxIssues = options.value(QStringLiteral("maxIssues")).toInt(kDefaultMaxIssues);
    const int maxFiles = options.value(QStringLiteral("maxFiles")).toInt(kDefaultMaxFiles);
    const bool allowEmptyLabels = options.value(QStringLiteral("allowEmptyLabels")).toBool(false);
    const QDir root(datasetPath);

    if (!root.exists()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("dataset_missing"), datasetPath, 0,
            QStringLiteral("数据集目录不存在。"));
        return result;
    }

    QString yamlError;
    const YoloDataYaml layout = parseYoloDataYaml(datasetPath, &yamlError);
    if (!layout.exists) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("missing_data_yaml"), layout.yamlPath, 0,
            QStringLiteral("缺少 data.yaml。"));
    } else if (!yamlError.isEmpty()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_data_yaml"), layout.yamlPath, 0, yamlError);
    }
    const int classCount = layout.exists ? layout.classCount : -1;
    if (layout.exists && classCount < 0) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("missing_class_count"), layout.yamlPath, 0,
            QStringLiteral("data.yaml must define nc or names."));
    }

    int inspectedFiles = 0;
    QSet<QString> seenImages;
    const QStringList splits = {QStringLiteral("train"), QStringLiteral("val")};
    for (const QString& split : splits) {
        const YoloSplitPaths splitPaths = yoloSplitPaths(layout, split);
        const QDir imageDir(splitPaths.imageDir);
        const QDir labelDir(splitPaths.labelDir);
        if (!imageDir.exists()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("missing_image_split"), imageDir.path(), 0,
                QStringLiteral("缺少图片目录。"));
            continue;
        }
        if (!labelDir.exists()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("missing_label_split"), labelDir.path(), 0,
                QStringLiteral("缺少标注目录。"));
            continue;
        }

        const QFileInfoList images = imageFiles(imageDir);
        if (images.isEmpty()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("empty_image_split"), imageDir.path(), 0,
                QStringLiteral("图片目录为空。"));
        }

        for (const QFileInfo& imageInfo : images) {
            const QString imageKey = canonicalImageKey(imageInfo);
            if (seenImages.contains(imageKey)) {
                continue;
            }
            seenImages.insert(imageKey);
            if (++inspectedFiles > maxFiles) {
                addIssue(result, QStringLiteral("warning"), QStringLiteral("file_limit"), datasetPath, 0,
                    QStringLiteral("数据集较大，已在 %1 个样本后截断校验。").arg(maxFiles));
                return result;
            }
            ++result.sampleCount;
            validateReadableImageFile(imageInfo.absoluteFilePath(), result, QStringLiteral("invalid_image"), QStringLiteral("YOLO "));
            if (issueLimitReached(result, maxIssues)) {
                return result;
            }
            const QString labelPath = labelDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".txt"));
            const QFileInfo labelInfo(labelPath);
            if (result.previewSamples.size() < 20) {
                result.previewSamples.append(QStringLiteral("%1\t%2").arg(imageInfo.absoluteFilePath(), labelInfo.exists()
                    ? yoloLabelSummary(labelInfo.absoluteFilePath(), segmentation)
                    : QStringLiteral("缺少标注文件")));
            }
            if (!labelInfo.exists()) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("missing_label"), labelPath, 0,
                    QStringLiteral("图片缺少对应标注文件：%1。").arg(imageInfo.fileName()));
                if (issueLimitReached(result, maxIssues)) return result;
                continue;
            }
            validateLabelFile(labelInfo, classCount, segmentation, allowEmptyLabels, result, maxIssues);
            if (issueLimitReached(result, maxIssues)) {
                return result;
            }
        }
    }

    if (result.sampleCount == 0) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("no_samples"), datasetPath, 0,
            QStringLiteral("未找到可校验的样本。"));
    }
    return result;
}

QVector<YoloSample> collectYoloSamples(const QString& datasetPath, DatasetSplitResult& result)
{
    QVector<YoloSample> samples;
    QString yamlError;
    const YoloDataYaml layout = parseYoloDataYaml(datasetPath, &yamlError);
    if (!yamlError.isEmpty()) {
        result.ok = false;
        result.errors.append(yamlError);
        return samples;
    }
    const QStringList splits = {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")};
    QSet<QString> seenImages;
    bool duplicateIgnored = false;
    for (const QString& split : splits) {
        const YoloSplitPaths splitPaths = yoloSplitPaths(layout, split);
        const QDir imageDir(splitPaths.imageDir);
        const QDir labelDir(splitPaths.labelDir);
        if (!imageDir.exists() || !labelDir.exists()) {
            continue;
        }
        for (const QFileInfo& imageInfo : imageFiles(imageDir)) {
            const QString imageKey = canonicalImageKey(imageInfo);
            if (seenImages.contains(imageKey)) {
                duplicateIgnored = true;
                continue;
            }
            seenImages.insert(imageKey);
            const QFileInfo labelInfo(labelDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".txt")));
            if (!labelInfo.exists()) {
                result.ok = false;
                result.errors.append(QStringLiteral("缺少标注文件：%1").arg(labelInfo.absoluteFilePath()));
                continue;
            }
            YoloSample sample;
            sample.imagePath = imageInfo.absoluteFilePath();
            sample.labelPath = labelInfo.absoluteFilePath();
            sample.fileName = imageInfo.fileName();
            sample.baseName = imageInfo.completeBaseName();
            samples.append(sample);
        }
    }
    if (duplicateIgnored) {
        result.warnings.append(QStringLiteral("Duplicate YOLO images were ignored while collecting split samples."));
    }
    return samples;
}

QVector<OcrSample> collectOcrSamples(const QString& datasetPath, const QString& labelFilePath, DatasetSplitResult& result)
{
    QVector<OcrSample> samples;
    const QDir root(datasetPath);
    QFile file(labelFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        result.ok = false;
        result.errors.append(QStringLiteral("无法读取 OCR 标签文件：%1").arg(labelFilePath));
        return samples;
    }

    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const int split = line.indexOf(QLatin1Char('\t'));
        if (split <= 0) {
            result.ok = false;
            result.errors.append(QStringLiteral("%1:%2 OCR 标签行格式错误。").arg(labelFilePath).arg(lineNumber));
            continue;
        }
        const QString relativeImagePath = line.left(split).trimmed();
        const QString absoluteImagePath = root.filePath(relativeImagePath);
        if (!QFileInfo::exists(absoluteImagePath)) {
            result.ok = false;
            result.errors.append(QStringLiteral("OCR 图片不存在：%1").arg(absoluteImagePath));
            continue;
        }
        OcrSample sample;
        sample.imagePath = absoluteImagePath;
        sample.text = line.mid(split + 1);
        sample.fileName = QFileInfo(relativeImagePath).fileName();
        samples.append(sample);
    }
    return samples;
}

QVector<OcrDetSample> collectOcrDetSamples(const QString& datasetPath, const QString& labelFilePath, DatasetSplitResult& result)
{
    QVector<OcrDetSample> samples;
    const QDir root(datasetPath);
    QFile file(labelFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        result.ok = false;
        result.errors.append(QStringLiteral("Cannot read PaddleOCR Det label file: %1").arg(labelFilePath));
        return samples;
    }

    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const int split = line.indexOf(QLatin1Char('\t'));
        if (split <= 0) {
            result.ok = false;
            result.errors.append(QStringLiteral("%1:%2 PaddleOCR Det label row format is invalid.").arg(labelFilePath).arg(lineNumber));
            continue;
        }
        const QString relativeImagePath = line.left(split).trimmed();
        const QString absoluteImagePath = root.filePath(relativeImagePath);
        if (!QFileInfo::exists(absoluteImagePath)) {
            result.ok = false;
            result.errors.append(QStringLiteral("PaddleOCR Det image does not exist: %1").arg(absoluteImagePath));
            continue;
        }
        OcrDetSample sample;
        sample.imagePath = absoluteImagePath;
        sample.relativeImagePath = relativeImagePath;
        sample.labelJson = line.mid(split + 1).trimmed();
        sample.fileName = QFileInfo(relativeImagePath).fileName();
        samples.append(sample);
    }
    return samples;
}

QVector<SemanticMaskSample> collectSemanticMaskSamples(const QString& datasetPath, DatasetSplitResult& result)
{
    QVector<SemanticMaskSample> samples;
    const QDir root(datasetPath);
    QSet<QString> seenImages;
    bool duplicateIgnored = false;
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir imageDir(root.filePath(QStringLiteral("images/%1").arg(split)));
        const QDir maskDir(root.filePath(QStringLiteral("masks/%1").arg(split)));
        if (!imageDir.exists() || !maskDir.exists()) {
            continue;
        }
        for (const QFileInfo& imageInfo : imageFiles(imageDir)) {
            const QString imageKey = canonicalImageKey(imageInfo);
            if (seenImages.contains(imageKey)) {
                duplicateIgnored = true;
                continue;
            }
            seenImages.insert(imageKey);
            const QFileInfo maskInfo(maskDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".png")));
            if (!maskInfo.exists()) {
                result.ok = false;
                result.errors.append(QStringLiteral("缺少语义分割 mask：%1").arg(maskInfo.absoluteFilePath()));
                continue;
            }
            SemanticMaskSample sample;
            sample.imagePath = imageInfo.absoluteFilePath();
            sample.maskPath = maskInfo.absoluteFilePath();
            sample.fileName = imageInfo.fileName();
            sample.baseName = imageInfo.completeBaseName();
            sample.split = split;
            samples.append(sample);
        }
    }
    if (duplicateIgnored) {
        result.warnings.append(QStringLiteral("Duplicate semantic segmentation images were ignored while collecting split samples."));
    }
    return samples;
}

void shuffleSamples(QVector<YoloSample>& samples, quint32 seed)
{
    QRandomGenerator rng(seed);
    for (int index = samples.size() - 1; index > 0; --index) {
        const int swapIndex = static_cast<int>(rng.bounded(static_cast<quint32>(index + 1)));
        qSwap(samples[index], samples[swapIndex]);
    }
}

void shuffleSamples(QVector<OcrDetSample>& samples, quint32 seed)
{
    QRandomGenerator rng(seed);
    for (int index = samples.size() - 1; index > 0; --index) {
        const int swapIndex = static_cast<int>(rng.bounded(static_cast<quint32>(index + 1)));
        qSwap(samples[index], samples[swapIndex]);
    }
}

void shuffleSamples(QVector<SemanticMaskSample>& samples, quint32 seed)
{
    QRandomGenerator rng(seed);
    for (int index = samples.size() - 1; index > 0; --index) {
        const int swapIndex = static_cast<int>(rng.bounded(static_cast<quint32>(index + 1)));
        qSwap(samples[index], samples[swapIndex]);
    }
}

bool copyFileReplacing(const QString& sourcePath, const QString& targetPath, QStringList& errors)
{
    QDir().mkpath(QFileInfo(targetPath).absolutePath());
    if (QFileInfo::exists(targetPath) && !QFile::remove(targetPath)) {
        errors.append(QStringLiteral("无法覆盖文件：%1").arg(targetPath));
        return false;
    }
    if (!QFile::copy(sourcePath, targetPath)) {
        errors.append(QStringLiteral("复制失败：%1 -> %2").arg(sourcePath, targetPath));
        return false;
    }
    return true;
}

QString splitNameForIndex(int index, int trainCount, int valCount)
{
    if (index < trainCount) {
        return QStringLiteral("train");
    }
    if (index < trainCount + valCount) {
        return QStringLiteral("val");
    }
    return QStringLiteral("test");
}

void calculateSplitCounts(int total,
    double trainRatio,
    double valRatio,
    double testRatio,
    int* trainCount,
    int* valCount,
    int* testCount)
{
    const double ratioSum = trainRatio + valRatio + testRatio;
    *trainCount = qRound((trainRatio / ratioSum) * total);
    *valCount = qRound((valRatio / ratioSum) * total);
    if (*trainCount <= 0 && total > 0) {
        *trainCount = 1;
    }
    if (*trainCount + *valCount > total) {
        *valCount = qMax(0, total - *trainCount);
    }
    *testCount = total - *trainCount - *valCount;
    if (testRatio > 0.0 && *testCount <= 0 && total >= 3) {
        if (*valCount > 1) {
            --(*valCount);
        } else if (*trainCount > 1) {
            --(*trainCount);
        }
        *testCount = total - *trainCount - *valCount;
    }
}

bool validateSplitRatios(double trainRatio, double valRatio, double testRatio, DatasetSplitResult& result)
{
    const double ratioSum = trainRatio + valRatio + testRatio;
    if (trainRatio <= 0.0 || valRatio < 0.0 || testRatio < 0.0 || ratioSum <= 0.0) {
        result.ok = false;
        result.errors.append(QStringLiteral("划分比例不合法。"));
        return false;
    }
    return true;
}

DatasetSplitResult splitYoloDataset(const QString& datasetPath,
    const QString& outputPath,
    const QJsonObject& options,
    bool segmentation)
{
    DatasetSplitResult result;
    result.outputPath = outputPath;

    const DatasetValidationResult validation = segmentation
        ? validateYoloSegmentationDataset(datasetPath, options)
        : validateYoloDetectionDataset(datasetPath, options);
    if (!validation.ok) {
        result.ok = false;
        result.errors.append(segmentation
            ? QStringLiteral("源数据集未通过 YOLO 分割校验，已取消划分。")
            : QStringLiteral("源数据集未通过 YOLO 检测校验，已取消划分。"));
        result.errors.append(validation.errors);
        return result;
    }

    const double trainRatio = options.value(QStringLiteral("trainRatio")).toDouble(0.8);
    const double valRatio = options.value(QStringLiteral("valRatio")).toDouble(0.2);
    const double testRatio = options.value(QStringLiteral("testRatio")).toDouble(0.0);
    if (!validateSplitRatios(trainRatio, valRatio, testRatio, result)) {
        return result;
    }

    QVector<YoloSample> samples = collectYoloSamples(datasetPath, result);
    if (!result.ok) {
        return result;
    }
    if (samples.isEmpty()) {
        result.ok = false;
        result.errors.append(segmentation
            ? QStringLiteral("没有可划分的 YOLO 分割样本。")
            : QStringLiteral("没有可划分的 YOLO 检测样本。"));
        return result;
    }

    const quint32 seed = static_cast<quint32>(options.value(QStringLiteral("seed")).toInt(42));
    shuffleSamples(samples, seed);

    calculateSplitCounts(samples.size(), trainRatio, valRatio, testRatio, &result.trainCount, &result.valCount, &result.testCount);

    const QDir outputRoot(outputPath);
    QDir().mkpath(outputRoot.path());
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        QDir().mkpath(outputRoot.filePath(QStringLiteral("images/%1").arg(split)));
        QDir().mkpath(outputRoot.filePath(QStringLiteral("labels/%1").arg(split)));
    }

    for (int index = 0; index < samples.size(); ++index) {
        const QString split = splitNameForIndex(index, result.trainCount, result.valCount);
        const YoloSample& sample = samples.at(index);
        const QString imageTarget = outputRoot.filePath(QStringLiteral("images/%1/%2").arg(split, sample.fileName));
        const QString labelTarget = outputRoot.filePath(QStringLiteral("labels/%1/%2.txt").arg(split, sample.baseName));
        copyFileReplacing(sample.imagePath, imageTarget, result.errors);
        copyFileReplacing(sample.labelPath, labelTarget, result.errors);
    }

    QString yamlError;
    const YoloDataYaml sourceLayout = parseYoloDataYaml(datasetPath, &yamlError);
    if (!yamlError.isEmpty()) {
        result.errors.append(yamlError);
    }
    writeNormalizedYoloDataYaml(outputRoot.path(), sourceLayout, result.testCount > 0, &result.errors);

    if (!result.errors.isEmpty()) {
        result.ok = false;
    }

    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("sourcePath"), datasetPath);
    report.insert(QStringLiteral("format"), segmentation ? QStringLiteral("yolo_segmentation") : QStringLiteral("yolo_detection"));
    report.insert(QStringLiteral("seed"), static_cast<int>(seed));
    report.insert(QStringLiteral("trainRatio"), trainRatio);
    report.insert(QStringLiteral("valRatio"), valRatio);
    report.insert(QStringLiteral("testRatio"), testRatio);
    QFile reportFile(outputRoot.filePath(QStringLiteral("split_report.json")));
    if (reportFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        reportFile.write(QJsonDocument(report).toJson(QJsonDocument::Indented));
    } else {
        result.warnings.append(QStringLiteral("无法写入 split_report.json。"));
    }

    return result;
}

DatasetValidationResult validateSemanticMaskDataset(const QString& datasetPath, const QJsonObject& options)
{
    DatasetValidationResult result;
    const int maxIssues = options.value(QStringLiteral("maxIssues")).toInt(kDefaultMaxIssues);
    const int maxFiles = options.value(QStringLiteral("maxFiles")).toInt(kDefaultMaxFiles);
    const int ignoreIndex = options.value(QStringLiteral("ignoreIndex")).toInt(255);
    const QDir root(datasetPath);

    if (!root.exists()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("dataset_missing"), datasetPath, 0,
            QStringLiteral("数据集目录不存在。"));
        return result;
    }
    if (ignoreIndex < 0 || ignoreIndex > 255) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_ignore_index"), datasetPath, 0,
            QStringLiteral("ignoreIndex 必须在 0..255 范围内。"));
        return result;
    }

    const QString classesPath = root.filePath(QStringLiteral("classes.txt"));
    const QStringList classNames = readClassesTxt(classesPath, result);
    const int classCount = classNames.size();

    int inspectedFiles = 0;
    QMap<int, qint64> totalClassPixels;
    QJsonObject splitCounts;
    const QStringList requiredSplits = {QStringLiteral("train"), QStringLiteral("val")};
    const QStringList allSplits = {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")};
    for (const QString& split : allSplits) {
        const bool required = requiredSplits.contains(split);
        const QDir imageDir(root.filePath(QStringLiteral("images/%1").arg(split)));
        const QDir maskDir(root.filePath(QStringLiteral("masks/%1").arg(split)));
        if (!imageDir.exists() || !maskDir.exists()) {
            if (required) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("missing_semantic_split"), root.path(), 0,
                    QStringLiteral("语义分割数据集必须包含 images/%1 和 masks/%1。").arg(split));
            }
            continue;
        }

        const QFileInfoList images = imageFiles(imageDir);
        if (required && images.isEmpty()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("empty_image_split"), imageDir.path(), 0,
                QStringLiteral("图片目录为空。"));
        }
        int splitSampleCount = 0;
        for (const QFileInfo& imageInfo : images) {
            if (++inspectedFiles > maxFiles) {
                addIssue(result, QStringLiteral("warning"), QStringLiteral("file_limit"), datasetPath, 0,
                    QStringLiteral("数据集较大，已在 %1 个样本后截断校验。").arg(maxFiles));
                return result;
            }
            ++result.sampleCount;
            ++splitSampleCount;
            validateReadableImageFile(imageInfo.absoluteFilePath(), result, QStringLiteral("invalid_image"), QStringLiteral("Semantic segmentation "));
            QImageReader imageReader(imageInfo.absoluteFilePath());
            imageReader.setAutoTransform(true);
            const QSize imageSize = imageReader.size();

            const QString maskPath = maskDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".png"));
            const QFileInfo maskInfo(maskPath);
            if (result.previewSamples.size() < 20) {
                result.previewSamples.append(QStringLiteral("%1\tmask=%2").arg(imageInfo.absoluteFilePath(), maskPath));
            }
            if (!maskInfo.exists()) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("missing_mask"), maskPath, 0,
                    QStringLiteral("图片缺少同 stem 的单通道 PNG mask：%1。").arg(imageInfo.fileName()));
                if (issueLimitReached(result, maxIssues)) return result;
                continue;
            }
            if (maskInfo.suffix().compare(QStringLiteral("png"), Qt::CaseInsensitive) != 0) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("mask_not_png"), maskInfo.absoluteFilePath(), 0,
                    QStringLiteral("语义分割 mask 必须是 PNG 文件。"));
                if (issueLimitReached(result, maxIssues)) return result;
            }
            if (maskInfo.size() <= 0) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("empty_mask_file"), maskInfo.absoluteFilePath(), 0,
                    QStringLiteral("mask 文件为空。"));
                if (issueLimitReached(result, maxIssues)) return result;
                continue;
            }

            QImageReader maskReader(maskInfo.absoluteFilePath());
            const QSize maskSize = maskReader.size();
            QImage mask(maskInfo.absoluteFilePath());
            if (mask.isNull() || !maskSize.isValid() || maskSize.isEmpty()) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_mask"), maskInfo.absoluteFilePath(), 0,
                    QStringLiteral("mask 无法解码或尺寸无效。"));
                if (issueLimitReached(result, maxIssues)) return result;
                continue;
            }
            if (imageSize.isValid() && !imageSize.isEmpty() && maskSize != imageSize) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("mask_size_mismatch"), maskInfo.absoluteFilePath(), 0,
                    QStringLiteral("mask 尺寸必须与图片一致。"));
                if (issueLimitReached(result, maxIssues)) return result;
            }
            if (!semanticMaskFormatAcceptable(mask)) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("mask_not_single_channel"), maskInfo.absoluteFilePath(), 0,
                    QStringLiteral("mask 必须是单通道或灰度 PNG，像素值表示 class id。"));
                if (issueLimitReached(result, maxIssues)) return result;
            }

            QSet<int> presentClasses;
            int foregroundPixels = 0;
            for (int y = 0; y < mask.height(); ++y) {
                for (int x = 0; x < mask.width(); ++x) {
                    const int classId = qGray(mask.pixel(x, y));
                    if (classId == ignoreIndex) {
                        continue;
                    }
                    if (classId < 0 || (classCount > 0 && classId >= classCount)) {
                        addIssue(result, QStringLiteral("error"), QStringLiteral("class_id_out_of_range"), maskInfo.absoluteFilePath(), 0,
                            QStringLiteral("mask 像素 class id=%1 超出 classes.txt 类别范围。").arg(classId));
                        if (issueLimitReached(result, maxIssues)) return result;
                        x = mask.width();
                        y = mask.height();
                        break;
                    }
                    presentClasses.insert(classId);
                    totalClassPixels[classId] += 1;
                    if (classId > 0) {
                        ++foregroundPixels;
                    }
                }
            }
            if (foregroundPixels == 0) {
                addIssue(result, QStringLiteral("warning"), QStringLiteral("empty_mask"), maskInfo.absoluteFilePath(), 0,
                    QStringLiteral("mask 没有前景类别像素，仅包含背景或 ignore。"));
            }
            if (presentClasses.size() <= 1) {
                addIssue(result, QStringLiteral("warning"), QStringLiteral("single_class_coverage"), maskInfo.absoluteFilePath(), 0,
                    QStringLiteral("mask 只覆盖单一类别，训练前请确认这不是误标。"));
            }
            if (issueLimitReached(result, maxIssues)) {
                return result;
            }
        }
        splitCounts.insert(split, splitSampleCount);
    }

    if (result.sampleCount == 0) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("no_samples"), datasetPath, 0,
            QStringLiteral("未找到语义分割 Mask PNG 样本。"));
    }
    for (int classId = 0; classId < classCount; ++classId) {
        if (!totalClassPixels.contains(classId)) {
            addIssue(result, QStringLiteral("warning"), QStringLiteral("missing_class_pixels"), classesPath, classId + 1,
                QStringLiteral("类别 %1 (%2) 没有出现在任何 mask 像素中。").arg(classId).arg(classNames.value(classId)));
        }
    }
    Q_UNUSED(splitCounts)
    return result;
}

} // namespace

QJsonObject DatasetSplitResult::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("ok"), ok);
    object.insert(QStringLiteral("trainCount"), trainCount);
    object.insert(QStringLiteral("valCount"), valCount);
    object.insert(QStringLiteral("testCount"), testCount);
    object.insert(QStringLiteral("outputPath"), outputPath);
    object.insert(QStringLiteral("errors"), QJsonArray::fromStringList(errors));
    object.insert(QStringLiteral("warnings"), QJsonArray::fromStringList(warnings));
    return object;
}

DatasetValidationResult validateYoloDetectionDataset(const QString& datasetPath, const QJsonObject& options)
{
    return validateYoloDataset(datasetPath, options, false);
}

DatasetValidationResult validateYoloSegmentationDataset(const QString& datasetPath, const QJsonObject& options)
{
    return validateYoloDataset(datasetPath, options, true);
}

DatasetValidationResult validateSemanticSegmentationMaskDataset(const QString& datasetPath, const QJsonObject& options)
{
    return validateSemanticMaskDataset(datasetPath, options);
}

DatasetValidationResult validatePaddleOcrDetDataset(const QString& datasetPath, const QJsonObject& options)
{
    DatasetValidationResult result;
    const int maxIssues = options.value(QStringLiteral("maxIssues")).toInt(kDefaultMaxIssues);
    const int maxFiles = options.value(QStringLiteral("maxFiles")).toInt(kDefaultMaxFiles);
    const QDir root(datasetPath);
    if (!root.exists()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("dataset_missing"), datasetPath, 0,
            QStringLiteral("数据集目录不存在。"));
        return result;
    }

    QString labelFilePath = options.value(QStringLiteral("labelFile")).toString();
    if (labelFilePath.isEmpty()) {
        labelFilePath = QFileInfo::exists(root.filePath(QStringLiteral("det_gt.txt")))
            ? root.filePath(QStringLiteral("det_gt.txt"))
            : root.filePath(QStringLiteral("det_gt_train.txt"));
    }
    QFile labelFile(labelFilePath);
    if (!labelFile.exists()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("missing_label_file"), labelFilePath, 0,
            QStringLiteral("缺少 PaddleOCR Det 标签文件。"));
        return result;
    }
    if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("label_file_unreadable"), labelFilePath, 0,
            QStringLiteral("无法读取 PaddleOCR Det 标签文件。"));
        return result;
    }

    QSet<QString> seenImages;
    int lineNumber = 0;
    while (!labelFile.atEnd()) {
        ++lineNumber;
        if (result.sampleCount >= maxFiles) {
            addIssue(result, QStringLiteral("warning"), QStringLiteral("file_limit"), datasetPath, 0,
                QStringLiteral("数据集较大，已在 %1 个样本后截断校验。").arg(maxFiles));
            return result;
        }
        const QString line = QString::fromUtf8(labelFile.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const int split = line.indexOf(QLatin1Char('\t'));
        if (split <= 0) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_det_row"), labelFilePath, lineNumber,
                QStringLiteral("标签行必须是 '<image path>\\t<json boxes>'。"));
            if (issueLimitReached(result, maxIssues)) return result;
            continue;
        }

        const QString imagePath = line.left(split).trimmed();
        const QString jsonText = line.mid(split + 1).trimmed();
        if (imagePath.isEmpty() || jsonText.isEmpty()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("empty_det_row"), labelFilePath, lineNumber,
                QStringLiteral("PaddleOCR Det 图片路径和 JSON 标注不能为空。"));
            if (issueLimitReached(result, maxIssues)) return result;
            continue;
        }
        if (seenImages.contains(imagePath)) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("duplicate_det_sample"), labelFilePath, lineNumber,
                QStringLiteral("存在重复 PaddleOCR Det 样本。"));
        }
        seenImages.insert(imagePath);

        const QString absoluteImagePath = root.filePath(imagePath);
        if (result.previewSamples.size() < 20) {
            result.previewSamples.append(QStringLiteral("%1\t%2").arg(absoluteImagePath, jsonText.left(120)));
        }
        if (!QFileInfo::exists(absoluteImagePath)) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("missing_det_image"), absoluteImagePath, 0,
                QStringLiteral("PaddleOCR Det 图片不存在。"));
        } else {
            validateReadableImageFile(absoluteImagePath, result, QStringLiteral("invalid_image"), QStringLiteral("PaddleOCR Det "));
        }

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(jsonText.toUtf8(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isArray()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_det_json"), labelFilePath, lineNumber,
                QStringLiteral("PaddleOCR Det 标注必须是 JSON 数组。"));
            if (issueLimitReached(result, maxIssues)) return result;
            continue;
        }
        const QJsonArray boxes = document.array();
        if (boxes.isEmpty()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("empty_det_boxes"), labelFilePath, lineNumber,
                QStringLiteral("PaddleOCR Det 标注至少需要一个文本框。"));
        }
        for (const QJsonValue& boxValue : boxes) {
            const QJsonObject box = boxValue.toObject();
            if (!box.contains(QStringLiteral("transcription"))) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("missing_transcription"), labelFilePath, lineNumber,
                    QStringLiteral("PaddleOCR Det 文本框缺少 transcription。"));
                break;
            }
            const QJsonArray points = box.value(QStringLiteral("points")).toArray();
            if (points.size() < 4) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("det_points_too_few"), labelFilePath, lineNumber,
                    QStringLiteral("PaddleOCR Det 文本框至少需要 4 个点。"));
                break;
            }
            bool pointsOk = true;
            for (const QJsonValue& pointValue : points) {
                const QJsonArray point = pointValue.toArray();
                if (point.size() < 2
                    || !point.at(0).isDouble()
                    || !point.at(1).isDouble()
                    || point.at(0).toDouble() < 0.0
                    || point.at(1).toDouble() < 0.0) {
                    pointsOk = false;
                    break;
                }
            }
            if (!pointsOk) {
                addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_det_point"), labelFilePath, lineNumber,
                    QStringLiteral("PaddleOCR Det 点坐标必须是非负数字。"));
                break;
            }
        }

        ++result.sampleCount;
        if (issueLimitReached(result, maxIssues)) {
            return result;
        }
    }

    if (result.sampleCount == 0) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("no_det_samples"), labelFilePath, 0,
            QStringLiteral("未找到 PaddleOCR Det 样本。"));
    }
    return result;
}

DatasetValidationResult validatePaddleOcrRecDataset(const QString& datasetPath, const QJsonObject& options)
{
    DatasetValidationResult result;
    const int maxIssues = options.value(QStringLiteral("maxIssues")).toInt(kDefaultMaxIssues);
    const int maxTextLength = options.value(QStringLiteral("maxTextLength")).toInt(25);
    const QDir root(datasetPath);
    if (!root.exists()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("dataset_missing"), datasetPath, 0,
            QStringLiteral("数据集目录不存在。"));
        return result;
    }

    QString labelFilePath = options.value(QStringLiteral("labelFile")).toString();
    if (labelFilePath.isEmpty()) {
        labelFilePath = QFileInfo::exists(root.filePath(QStringLiteral("rec_gt.txt")))
            ? root.filePath(QStringLiteral("rec_gt.txt"))
            : root.filePath(QStringLiteral("rec_gt_train.txt"));
    }
    QFile labelFile(labelFilePath);
    if (!labelFile.exists()) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("missing_label_file"), labelFilePath, 0,
            QStringLiteral("缺少 PaddleOCR Rec 标签文件。"));
        return result;
    }
    if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("label_file_unreadable"), labelFilePath, 0,
            QStringLiteral("无法读取 PaddleOCR Rec 标签文件。"));
        return result;
    }

    QSet<QChar> dictionary;
    const QString dictionaryPath = options.value(QStringLiteral("dictionaryFile")).toString(root.filePath(QStringLiteral("dict.txt")));
    if (QFileInfo::exists(dictionaryPath)) {
        QFile dictFile(dictionaryPath);
        if (!dictFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("dictionary_unreadable"), dictionaryPath, 0,
                QStringLiteral("无法读取字符字典。"));
        } else {
            while (!dictFile.atEnd()) {
                const QString line = QString::fromUtf8(dictFile.readLine()).trimmed();
                if (!line.isEmpty()) {
                    dictionary.insert(line.at(0));
                }
            }
        }
    }

    QSet<QString> seenImages;
    int lineNumber = 0;
    while (!labelFile.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(labelFile.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const int split = line.indexOf(QLatin1Char('\t'));
        if (split <= 0) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("invalid_ocr_row"), labelFilePath, lineNumber,
                QStringLiteral("标签行必须是 '<image path>\\t<label>'。"));
            if (issueLimitReached(result, maxIssues)) return result;
            continue;
        }
        const QString imagePath = line.left(split).trimmed();
        const QString text = line.mid(split + 1);
        if (text.isEmpty()) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("empty_ocr_label"), labelFilePath, lineNumber,
                QStringLiteral("OCR 标签不能为空。"));
        }
        if (text.size() > maxTextLength) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("ocr_label_too_long"), labelFilePath, lineNumber,
                QStringLiteral("OCR 标签长度超过 maxTextLength。"));
        }
        if (seenImages.contains(imagePath)) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("duplicate_ocr_sample"), labelFilePath, lineNumber,
                QStringLiteral("存在重复 OCR 样本。"));
        }
        seenImages.insert(imagePath);

        const QString absoluteImagePath = root.filePath(imagePath);
        if (result.previewSamples.size() < 20) {
            result.previewSamples.append(QStringLiteral("%1\t%2").arg(absoluteImagePath, text));
        }
        if (!QFileInfo::exists(absoluteImagePath)) {
            addIssue(result, QStringLiteral("error"), QStringLiteral("missing_ocr_image"), absoluteImagePath, 0,
                QStringLiteral("OCR 图片不存在。"));
        } else {
            validateReadableImageFile(absoluteImagePath, result, QStringLiteral("invalid_image"), QStringLiteral("PaddleOCR Rec "));
        }
        if (!dictionary.isEmpty()) {
            for (const QChar ch : text) {
                if (!dictionary.contains(ch)) {
                    addIssue(result, QStringLiteral("error"), QStringLiteral("char_not_in_dictionary"), labelFilePath, lineNumber,
                        QStringLiteral("字符不在字典中：%1。").arg(ch));
                    break;
                }
            }
        }
        ++result.sampleCount;
        if (issueLimitReached(result, maxIssues)) {
            return result;
        }
    }

    if (result.sampleCount == 0) {
        addIssue(result, QStringLiteral("error"), QStringLiteral("no_ocr_samples"), labelFilePath, 0,
            QStringLiteral("未找到 OCR 识别样本。"));
    }
    return result;
}

DatasetSplitResult splitYoloDetectionDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options)
{
    return splitYoloDataset(datasetPath, outputPath, options, false);
}

DatasetSplitResult splitYoloSegmentationDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options)
{
    return splitYoloDataset(datasetPath, outputPath, options, true);
}

DatasetSplitResult splitSemanticSegmentationMaskDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options)
{
    DatasetSplitResult result;
    result.outputPath = outputPath;

    const DatasetValidationResult validation = validateSemanticSegmentationMaskDataset(datasetPath, options);
    if (!validation.ok) {
        result.ok = false;
        result.errors.append(QStringLiteral("源数据集未通过语义分割 Mask PNG 校验，已取消划分。"));
        result.errors.append(validation.errors);
        return result;
    }

    const double trainRatio = options.value(QStringLiteral("trainRatio")).toDouble(0.8);
    const double valRatio = options.value(QStringLiteral("valRatio")).toDouble(0.2);
    const double testRatio = options.value(QStringLiteral("testRatio")).toDouble(0.0);
    if (!validateSplitRatios(trainRatio, valRatio, testRatio, result)) {
        return result;
    }

    QVector<SemanticMaskSample> samples = collectSemanticMaskSamples(datasetPath, result);
    if (!result.ok) {
        return result;
    }
    if (samples.isEmpty()) {
        result.ok = false;
        result.errors.append(QStringLiteral("没有可划分的语义分割 Mask PNG 样本。"));
        return result;
    }

    const quint32 seed = static_cast<quint32>(options.value(QStringLiteral("seed")).toInt(42));
    shuffleSamples(samples, seed);
    calculateSplitCounts(samples.size(), trainRatio, valRatio, testRatio, &result.trainCount, &result.valCount, &result.testCount);

    const QDir outputRoot(outputPath);
    QDir().mkpath(outputRoot.path());
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        QDir().mkpath(outputRoot.filePath(QStringLiteral("images/%1").arg(split)));
        QDir().mkpath(outputRoot.filePath(QStringLiteral("masks/%1").arg(split)));
    }

    for (int index = 0; index < samples.size(); ++index) {
        const QString split = splitNameForIndex(index, result.trainCount, result.valCount);
        const SemanticMaskSample& sample = samples.at(index);
        const QString imageTarget = outputRoot.filePath(QStringLiteral("images/%1/%2").arg(split, sample.fileName));
        const QString maskTarget = outputRoot.filePath(QStringLiteral("masks/%1/%2.png").arg(split, sample.baseName));
        copyFileReplacing(sample.imagePath, imageTarget, result.errors);
        copyFileReplacing(sample.maskPath, maskTarget, result.errors);
    }
    copyFileReplacing(QDir(datasetPath).filePath(QStringLiteral("classes.txt")), outputRoot.filePath(QStringLiteral("classes.txt")), result.errors);

    if (!result.errors.isEmpty()) {
        result.ok = false;
    }

    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("sourcePath"), datasetPath);
    report.insert(QStringLiteral("format"), QStringLiteral("semantic_segmentation_mask"));
    report.insert(QStringLiteral("seed"), static_cast<int>(seed));
    report.insert(QStringLiteral("trainRatio"), trainRatio);
    report.insert(QStringLiteral("valRatio"), valRatio);
    report.insert(QStringLiteral("testRatio"), testRatio);
    QFile reportFile(outputRoot.filePath(QStringLiteral("split_report.json")));
    if (reportFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        reportFile.write(QJsonDocument(report).toJson(QJsonDocument::Indented));
    } else {
        result.warnings.append(QStringLiteral("无法写入 split_report.json。"));
    }

    return result;
}

DatasetSplitResult splitPaddleOcrDetDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options)
{
    DatasetSplitResult result;
    result.outputPath = outputPath;

    const DatasetValidationResult validation = validatePaddleOcrDetDataset(datasetPath, options);
    if (!validation.ok) {
        result.ok = false;
        result.errors.append(QStringLiteral("源数据集未通过 PaddleOCR Det 校验，已取消划分。"));
        result.errors.append(validation.errors);
        return result;
    }

    const double trainRatio = options.value(QStringLiteral("trainRatio")).toDouble(0.8);
    const double valRatio = options.value(QStringLiteral("valRatio")).toDouble(0.2);
    const double testRatio = options.value(QStringLiteral("testRatio")).toDouble(0.0);
    if (!validateSplitRatios(trainRatio, valRatio, testRatio, result)) {
        return result;
    }

    const QDir root(datasetPath);
    const QString labelFilePath = QFileInfo::exists(root.filePath(QStringLiteral("det_gt.txt")))
        ? root.filePath(QStringLiteral("det_gt.txt"))
        : root.filePath(QStringLiteral("det_gt_train.txt"));
    QVector<OcrDetSample> samples = collectOcrDetSamples(datasetPath, labelFilePath, result);
    if (!result.ok) {
        return result;
    }
    if (samples.isEmpty()) {
        result.ok = false;
        result.errors.append(QStringLiteral("没有可划分的 PaddleOCR Det 样本。"));
        return result;
    }

    const quint32 seed = static_cast<quint32>(options.value(QStringLiteral("seed")).toInt(42));
    shuffleSamples(samples, seed);
    calculateSplitCounts(samples.size(), trainRatio, valRatio, testRatio, &result.trainCount, &result.valCount, &result.testCount);

    const QDir outputRoot(outputPath);
    QDir().mkpath(outputRoot.path());
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        QDir().mkpath(outputRoot.filePath(QStringLiteral("images/%1").arg(split)));
    }

    QStringList allRows;
    QStringList trainRows;
    QStringList valRows;
    QStringList testRows;
    for (int index = 0; index < samples.size(); ++index) {
        const QString split = splitNameForIndex(index, result.trainCount, result.valCount);
        const OcrDetSample& sample = samples.at(index);
        QString fileName = sample.fileName;
        if (fileName.isEmpty()) {
            fileName = QStringLiteral("sample_%1.png").arg(index + 1);
        }
        const QString targetRelative = QStringLiteral("images/%1/%2_%3").arg(split).arg(index + 1, 4, 10, QLatin1Char('0')).arg(fileName);
        const QString targetPath = outputRoot.filePath(targetRelative);
        copyFileReplacing(sample.imagePath, targetPath, result.errors);
        const QString row = QStringLiteral("%1\t%2").arg(targetRelative, sample.labelJson);
        allRows.append(row);
        if (split == QStringLiteral("train")) {
            trainRows.append(row);
        } else if (split == QStringLiteral("val")) {
            valRows.append(row);
        } else {
            testRows.append(row);
        }
    }

    auto writeRows = [&result](const QString& path, const QStringList& rows) {
        QFile file(path);
        QDir().mkpath(QFileInfo(path).absolutePath());
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
            result.errors.append(QStringLiteral("无法写入 PaddleOCR Det 标签文件：%1").arg(path));
            result.ok = false;
            return;
        }
        QTextStream stream(&file);
        stream.setCodec("UTF-8");
        for (const QString& row : rows) {
            stream << row << '\n';
        }
    };
    writeRows(outputRoot.filePath(QStringLiteral("det_gt.txt")), allRows);
    writeRows(outputRoot.filePath(QStringLiteral("det_gt_train.txt")), trainRows);
    writeRows(outputRoot.filePath(QStringLiteral("det_gt_val.txt")), valRows);
    writeRows(outputRoot.filePath(QStringLiteral("det_gt_test.txt")), testRows);

    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("sourcePath"), datasetPath);
    report.insert(QStringLiteral("format"), QStringLiteral("paddleocr_det"));
    report.insert(QStringLiteral("seed"), static_cast<int>(seed));
    report.insert(QStringLiteral("trainRatio"), trainRatio);
    report.insert(QStringLiteral("valRatio"), valRatio);
    report.insert(QStringLiteral("testRatio"), testRatio);
    QFile reportFile(outputRoot.filePath(QStringLiteral("split_report.json")));
    if (reportFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        reportFile.write(QJsonDocument(report).toJson(QJsonDocument::Indented));
    } else {
        result.warnings.append(QStringLiteral("无法写入 split_report.json。"));
    }
    if (!result.errors.isEmpty()) {
        result.ok = false;
    }
    return result;
}

DatasetSplitResult splitPaddleOcrRecDataset(const QString& datasetPath, const QString& outputPath, const QJsonObject& options)
{
    DatasetSplitResult result;
    result.outputPath = outputPath;

    const DatasetValidationResult validation = validatePaddleOcrRecDataset(datasetPath, options);
    if (!validation.ok) {
        result.ok = false;
        result.errors.append(QStringLiteral("源数据集未通过 PaddleOCR Rec 校验，已取消划分。"));
        result.errors.append(validation.errors);
        return result;
    }

    const double trainRatio = options.value(QStringLiteral("trainRatio")).toDouble(0.8);
    const double valRatio = options.value(QStringLiteral("valRatio")).toDouble(0.2);
    const double testRatio = options.value(QStringLiteral("testRatio")).toDouble(0.0);
    if (!validateSplitRatios(trainRatio, valRatio, testRatio, result)) {
        return result;
    }

    const QDir root(datasetPath);
    const QString labelFilePath = QFileInfo::exists(root.filePath(QStringLiteral("rec_gt.txt")))
        ? root.filePath(QStringLiteral("rec_gt.txt"))
        : root.filePath(QStringLiteral("rec_gt_train.txt"));
    QVector<OcrSample> samples = collectOcrSamples(datasetPath, labelFilePath, result);
    if (!result.ok) {
        return result;
    }
    if (samples.isEmpty()) {
        result.ok = false;
        result.errors.append(QStringLiteral("没有可划分的 PaddleOCR Rec 样本。"));
        return result;
    }

    const quint32 seed = static_cast<quint32>(options.value(QStringLiteral("seed")).toInt(42));
    QRandomGenerator rng(seed);
    for (int index = samples.size() - 1; index > 0; --index) {
        const int swapIndex = static_cast<int>(rng.bounded(static_cast<quint32>(index + 1)));
        qSwap(samples[index], samples[swapIndex]);
    }
    calculateSplitCounts(samples.size(), trainRatio, valRatio, testRatio, &result.trainCount, &result.valCount, &result.testCount);

    const QDir outputRoot(outputPath);
    QDir().mkpath(outputRoot.path());
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        QDir().mkpath(outputRoot.filePath(QStringLiteral("images/%1").arg(split)));
    }

    QStringList allRows;
    QStringList trainRows;
    QStringList valRows;
    QStringList testRows;
    for (int index = 0; index < samples.size(); ++index) {
        const QString split = splitNameForIndex(index, result.trainCount, result.valCount);
        const OcrSample& sample = samples.at(index);
        QString fileName = sample.fileName;
        if (fileName.isEmpty()) {
            fileName = QStringLiteral("sample_%1.png").arg(index + 1);
        }
        const QString targetRelative = QStringLiteral("images/%1/%2_%3").arg(split).arg(index + 1, 4, 10, QLatin1Char('0')).arg(fileName);
        const QString targetPath = outputRoot.filePath(targetRelative);
        copyFileReplacing(sample.imagePath, targetPath, result.errors);
        const QString row = QStringLiteral("%1\t%2").arg(targetRelative, sample.text);
        allRows.append(row);
        if (split == QStringLiteral("train")) {
            trainRows.append(row);
        } else if (split == QStringLiteral("val")) {
            valRows.append(row);
        } else {
            testRows.append(row);
        }
    }

    auto writeRows = [&result](const QString& path, const QStringList& rows) {
        QFile file(path);
        QDir().mkpath(QFileInfo(path).absolutePath());
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
            result.errors.append(QStringLiteral("无法写入 OCR 标签文件：%1").arg(path));
            result.ok = false;
            return;
        }
        QTextStream stream(&file);
        stream.setCodec("UTF-8");
        for (const QString& row : rows) {
            stream << row << '\n';
        }
    };
    writeRows(outputRoot.filePath(QStringLiteral("rec_gt.txt")), allRows);
    writeRows(outputRoot.filePath(QStringLiteral("rec_gt_train.txt")), trainRows);
    writeRows(outputRoot.filePath(QStringLiteral("rec_gt_val.txt")), valRows);
    writeRows(outputRoot.filePath(QStringLiteral("rec_gt_test.txt")), testRows);

    const QString dictionaryPath = options.value(QStringLiteral("dictionaryFile")).toString(root.filePath(QStringLiteral("dict.txt")));
    if (QFileInfo::exists(dictionaryPath)) {
        copyFileReplacing(dictionaryPath, outputRoot.filePath(QStringLiteral("dict.txt")), result.errors);
    } else {
        result.warnings.append(QStringLiteral("未找到 OCR Rec 字典文件；split 输出不包含 dict.txt，将依赖训练参数或官方预置字典。"));
    }

    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("sourcePath"), datasetPath);
    report.insert(QStringLiteral("format"), QStringLiteral("paddleocr_rec"));
    report.insert(QStringLiteral("seed"), static_cast<int>(seed));
    report.insert(QStringLiteral("trainRatio"), trainRatio);
    report.insert(QStringLiteral("valRatio"), valRatio);
    report.insert(QStringLiteral("testRatio"), testRatio);
    QFile reportFile(outputRoot.filePath(QStringLiteral("split_report.json")));
    if (reportFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        reportFile.write(QJsonDocument(report).toJson(QJsonDocument::Indented));
    } else {
        result.warnings.append(QStringLiteral("无法写入 split_report.json。"));
    }
    if (!result.errors.isEmpty()) {
        result.ok = false;
    }
    return result;
}

} // namespace aitrain
