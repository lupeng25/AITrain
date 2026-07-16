#include "aitrain/v2/DatasetDriverV2.h"

#include "aitrain/core/SemanticMask.h"

#include <QDir>
#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QVariant>

#include <algorithm>

namespace aitrain::v2 {
namespace {

constexpr auto kSemanticMaskFormat = "semantic_segmentation_mask";

QStringList semanticImageFilters()
{
    return {QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"), QStringLiteral("*.png"), QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"), QStringLiteral("*.tiff")};
}

bool canceled(const DatasetOperationContext& context)
{
    return context.isCancellationRequested && context.isCancellationRequested();
}

void report(const DatasetOperationContext& context, const QString& code, const QString& message)
{
    if (context.reportDiagnostic) {
        context.reportDiagnostic(code, message);
    }
}

void appendIssue(QJsonArray* issues, const QString& code, const QString& path, const QString& message)
{
    issues->append(QJsonObject{{QStringLiteral("code"), code}, {QStringLiteral("path"), path}, {QStringLiteral("message"), message}});
}

bool sourcePathIsWithinRoot(const QDir& root, const QString& path)
{
    const QString relative = QDir::cleanPath(root.relativeFilePath(path));
    return relative != QStringLiteral("..") && !relative.startsWith(QStringLiteral("../"));
}

bool safeRelativePath(const QString& path)
{
    const QString clean = QDir::cleanPath(path);
    return !clean.isEmpty() && !QDir::isAbsolutePath(clean) && clean != QStringLiteral("..") && !clean.startsWith(QStringLiteral("../"));
}

QString splitForIndex(int index, int total, double trainRatio, double valRatio)
{
    const double position = static_cast<double>(index) / static_cast<double>(total);
    if (position < trainRatio) {
        return QStringLiteral("train");
    }
    if (position < trainRatio + valRatio) {
        return QStringLiteral("val");
    }
    return QStringLiteral("test");
}

QString splitPlanHash(const QJsonObject& manifest)
{
    QJsonObject copy = manifest;
    copy.remove(QStringLiteral("planHash"));
    return QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(copy).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
}

bool fileIdentity(const QString& path, QString* sha256, qint64* bytes, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("dataset_split_file_unreadable:%1").arg(path);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("dataset_split_file_read_failed:%1").arg(path);
            return false;
        }
        hash.addData(block);
    }
    if (sha256) *sha256 = QString::fromLatin1(hash.result().toHex());
    if (bytes) *bytes = QFileInfo(file).size();
    return true;
}

} // namespace

bool DatasetDriverRegistryV2::registerDriver(const DatasetDriverV2* driver, QString* error)
{
    if (!driver || driver->id().trimmed().isEmpty() || driver->version().trimmed().isEmpty()) {
        if (error) {
            *error = QStringLiteral("Dataset Driver 必须提供非空 ID 和版本。");
        }
        return false;
    }
    const QStringList driverFormats = driver->supportedFormats();
    if (driverFormats.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Dataset Driver 必须声明至少一种格式。");
        }
        return false;
    }
    for (const QString& format : driverFormats) {
        const QString normalizedFormat = format.trimmed().toLower();
        if (normalizedFormat.isEmpty() || driversByFormat_.contains(normalizedFormat)) {
            if (error) {
                *error = QStringLiteral("Dataset 格式已注册或无效：%1").arg(format);
            }
            return false;
        }
    }
    for (const QString& format : driverFormats) {
        driversByFormat_.insert(format.trimmed().toLower(), driver);
    }
    return true;
}

const DatasetDriverV2* DatasetDriverRegistryV2::driverForFormat(const QString& format) const
{
    return driversByFormat_.value(format.trimmed().toLower(), nullptr);
}

QStringList DatasetDriverRegistryV2::formats() const
{
    QStringList values = driversByFormat_.keys();
    std::sort(values.begin(), values.end());
    return values;
}

QString SemanticMaskDatasetDriverV2::id() const
{
    return QStringLiteral("semantic_mask");
}

QString SemanticMaskDatasetDriverV2::version() const
{
    return QStringLiteral("2.0");
}

QStringList SemanticMaskDatasetDriverV2::supportedFormats() const
{
    return {QString::fromLatin1(kSemanticMaskFormat)};
}

bool SemanticMaskDatasetDriverV2::detect(const QString& sourcePath, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const
{
    const QDir root(sourcePath);
    if (!root.exists() || !QFileInfo::exists(root.filePath(QStringLiteral("classes.txt")))
        || !QDir(root.filePath(QStringLiteral("images"))).exists() || !QDir(root.filePath(QStringLiteral("masks"))).exists()) {
        return false;
    }
    return inspect(sourcePath, QString::fromLatin1(kSemanticMaskFormat), inspection, context, error);
}

bool SemanticMaskDatasetDriverV2::inspect(const QString& sourcePath, const QString& format, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const
{
    if (!inspection || format.trimmed().toLower() != QLatin1String(kSemanticMaskFormat)) {
        if (error) {
            *error = QStringLiteral("语义分割 Mask Driver 仅支持 semantic_segmentation_mask 格式。");
        }
        return false;
    }
    const QDir root(sourcePath);
    if (!root.exists()) {
        if (error) {
            *error = QStringLiteral("数据集目录不存在：%1").arg(sourcePath);
        }
        return false;
    }
    DatasetInspection inspected;
    inspected.sourcePath = root.absolutePath();
    inspected.format = QString::fromLatin1(kSemanticMaskFormat);
    QJsonArray issues;
    QJsonArray classes;
    QFile classesFile(root.filePath(QStringLiteral("classes.txt")));
    if (!classesFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        appendIssue(&issues, QStringLiteral("classes_missing"), classesFile.fileName(), QStringLiteral("缺少 classes.txt。"));
    } else {
        while (!classesFile.atEnd()) {
            const QString value = QString::fromUtf8(classesFile.readLine()).trimmed();
            if (!value.isEmpty()) {
                classes.append(value);
            }
        }
        if (classes.isEmpty()) {
            appendIssue(&issues, QStringLiteral("classes_empty"), classesFile.fileName(), QStringLiteral("classes.txt 至少需要一个类别。"));
        }
    }

    const QStringList splits = {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")};
    for (const QString& split : splits) {
        const QDir imageDirectory(root.filePath(QStringLiteral("images/%1").arg(split)));
        if (!imageDirectory.exists()) {
            continue;
        }
        const QFileInfoList images = imageDirectory.entryInfoList(semanticImageFilters(), QDir::Files, QDir::Name);
        for (const QFileInfo& imageInfo : images) {
            if (canceled(context)) {
                if (error) {
                    *error = QStringLiteral("dataset_inspection_canceled");
                }
                return false;
            }
            ++inspected.sampleCount;
            const QString maskPath = root.filePath(QStringLiteral("masks/%1/%2.png").arg(split, imageInfo.completeBaseName()));
            const QImage image(imageInfo.absoluteFilePath());
            const QImage mask(maskPath);
            if (image.isNull()) {
                appendIssue(&issues, QStringLiteral("image_unreadable"), imageInfo.absoluteFilePath(), QStringLiteral("图像无法读取。"));
                continue;
            }
            if (mask.isNull()) {
                appendIssue(&issues, QStringLiteral("mask_missing_or_unreadable"), maskPath, QStringLiteral("缺少或无法读取同名 PNG 掩码。"));
                continue;
            }
            if (!isSupportedSemanticMask(mask)) {
                appendIssue(&issues, QStringLiteral("mask_format_unsupported"), maskPath, QStringLiteral("掩码必须是 Grayscale8 或 Indexed8。"));
                continue;
            }
            if (mask.size() != image.size()) {
                appendIssue(&issues, QStringLiteral("mask_size_mismatch"), maskPath, QStringLiteral("掩码尺寸必须与图像一致。"));
                continue;
            }
            const int classCount = classes.size();
            for (int y = 0; y < mask.height(); ++y) {
                for (int x = 0; x < mask.width(); ++x) {
                    const int classId = semanticMaskClassId(mask, x, y);
                    if (classId != 255 && (classId < 0 || classId >= classCount)) {
                        appendIssue(&issues, QStringLiteral("class_id_out_of_range"), maskPath,
                            QStringLiteral("掩码包含超出 classes.txt 范围的类别 ID。"));
                        y = mask.height();
                        break;
                    }
                }
            }
            if (context.reportProgress) {
                context.reportProgress(static_cast<int>(inspected.sampleCount), QStringLiteral("已检查语义分割样本。"));
            }
        }
    }
    if (inspected.sampleCount == 0) {
        appendIssue(&issues, QStringLiteral("samples_empty"), inspected.sourcePath, QStringLiteral("未找到语义分割图像样本。"));
    }
    inspected.details = QJsonObject{{QStringLiteral("classes"), classes}, {QStringLiteral("issues"), issues}};
    *inspection = inspected;
    return true;
}

bool SemanticMaskDatasetDriverV2::validate(const DatasetInspection& inspection, DatasetValidationResult* validation, const DatasetOperationContext& context, QString* error) const
{
    if (!validation || inspection.format != QLatin1String(kSemanticMaskFormat)) {
        if (error) {
            *error = QStringLiteral("语义分割 Mask 校验参数无效。");
        }
        return false;
    }
    if (canceled(context)) {
        if (error) {
            *error = QStringLiteral("dataset_validation_canceled");
        }
        return false;
    }
    validation->issues = inspection.details.value(QStringLiteral("issues")).toArray();
    validation->details = inspection.details;
    validation->valid = validation->issues.isEmpty();
    if (!validation->valid) {
        report(context, QStringLiteral("semantic_mask_invalid"), QStringLiteral("语义分割 Mask 数据集未通过校验。"));
    }
    return true;
}

bool SemanticMaskDatasetDriverV2::planSplit(const DatasetInspection& inspection, const QJsonObject& options, DatasetSplitPlan* plan, const DatasetOperationContext& context, QString* error) const
{
    if (!plan || inspection.format != QLatin1String(kSemanticMaskFormat)
        || !inspection.details.value(QStringLiteral("issues")).toArray().isEmpty()) {
        if (error) {
            *error = QStringLiteral("语义分割 Mask 拆分计划需要已通过校验的数据集和输出对象。");
        }
        return false;
    }
    const double trainRatio = options.value(QStringLiteral("trainRatio")).toDouble(0.8);
    const double valRatio = options.value(QStringLiteral("valRatio")).toDouble(0.2);
    const double testRatio = options.value(QStringLiteral("testRatio")).toDouble(0.0);
    if (trainRatio < 0.0 || valRatio < 0.0 || testRatio < 0.0 || qAbs(trainRatio + valRatio + testRatio - 1.0) > 0.000001) {
        if (error) {
            *error = QStringLiteral("trainRatio、valRatio、testRatio 必须为非负且总和为 1。");
        }
        return false;
    }
    const QDir root(inspection.sourcePath);
    struct Sample final {
        QString imageRelativePath;
        QString maskRelativePath;
        QString imageSha256;
        QString maskSha256;
        qint64 imageBytes = 0;
        qint64 maskBytes = 0;
        QString name;
        QString orderKey;
    };
    QVector<Sample> samples;
    const QString seed = options.value(QStringLiteral("seed")).toVariant().toString();
    const QStringList sourceSplits = {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")};
    for (const QString& sourceSplit : sourceSplits) {
        const QDir imageDirectory(root.filePath(QStringLiteral("images/%1").arg(sourceSplit)));
        const QFileInfoList images = imageDirectory.entryInfoList(semanticImageFilters(), QDir::Files, QDir::Name);
        for (const QFileInfo& image : images) {
            if (canceled(context)) {
                if (error) {
                    *error = QStringLiteral("dataset_split_plan_canceled");
                }
                return false;
            }
            const QString imagePath = image.absoluteFilePath();
            const QString maskPath = root.filePath(QStringLiteral("masks/%1/%2.png").arg(sourceSplit, image.completeBaseName()));
            if (!sourcePathIsWithinRoot(root, imagePath) || !sourcePathIsWithinRoot(root, maskPath) || !QFileInfo::exists(maskPath)) {
                if (error) {
                    *error = QStringLiteral("拆分计划发现无效样本配对：%1").arg(imagePath);
                }
                return false;
            }
            const QString relative = QDir::cleanPath(root.relativeFilePath(imagePath));
            const QString orderKey = QString::fromLatin1(QCryptographicHash::hash((seed + QStringLiteral("\n") + relative).toUtf8(), QCryptographicHash::Sha256).toHex());
            QString imageSha256;
            QString maskSha256;
            qint64 imageBytes = 0;
            qint64 maskBytes = 0;
            if (!fileIdentity(imagePath, &imageSha256, &imageBytes, error)
                || !fileIdentity(maskPath, &maskSha256, &maskBytes, error)) {
                return false;
            }
            samples.append({relative, QDir::cleanPath(root.relativeFilePath(maskPath)),
                imageSha256, maskSha256, imageBytes, maskBytes, image.fileName(), orderKey});
        }
    }
    if (samples.isEmpty()) {
        if (error) {
            *error = QStringLiteral("语义分割 Mask 数据集没有可拆分样本。");
        }
        return false;
    }
    std::sort(samples.begin(), samples.end(), [](const Sample& left, const Sample& right) {
        return left.orderKey == right.orderKey ? left.imageRelativePath < right.imageRelativePath : left.orderKey < right.orderKey;
    });
    QJsonArray entries;
    for (int index = 0; index < samples.size(); ++index) {
        const Sample& sample = samples.at(index);
        const QString destinationSplit = splitForIndex(index, samples.size(), trainRatio, valRatio);
        const QString targetImage = QStringLiteral("images/%1/%2").arg(destinationSplit, sample.name);
        const QString targetMask = QStringLiteral("masks/%1/%2.png").arg(destinationSplit, QFileInfo(sample.name).completeBaseName());
        entries.append(QJsonObject{{QStringLiteral("sourceImageRelativePath"), sample.imageRelativePath},
            {QStringLiteral("sourceImageSha256"), sample.imageSha256},
            {QStringLiteral("sourceImageBytes"), QString::number(sample.imageBytes)},
            {QStringLiteral("sourceMaskRelativePath"), sample.maskRelativePath},
            {QStringLiteral("sourceMaskSha256"), sample.maskSha256},
            {QStringLiteral("sourceMaskBytes"), QString::number(sample.maskBytes)},
            {QStringLiteral("targetImage"), targetImage}, {QStringLiteral("targetMask"), targetMask}, {QStringLiteral("split"), destinationSplit}});
    }
    QJsonObject manifest{{QStringLiteral("schemaVersion"), 2}, {QStringLiteral("format"), inspection.format},
        {QStringLiteral("classes"), inspection.details.value(QStringLiteral("classes")).toArray()},
        {QStringLiteral("seed"), seed}, {QStringLiteral("ratios"), QJsonObject{{QStringLiteral("train"), trainRatio}, {QStringLiteral("val"), valRatio}, {QStringLiteral("test"), testRatio}}},
        {QStringLiteral("entries"), entries}};
    manifest.insert(QStringLiteral("planHash"), splitPlanHash(manifest));
    plan->format = inspection.format;
    plan->sourceRoot = root.absolutePath();
    plan->planHash = manifest.value(QStringLiteral("planHash")).toString();
    plan->manifest = manifest;
    return true;
}

bool SemanticMaskDatasetDriverV2::materializeSplit(const DatasetSplitPlan& plan, const QString& stagingPath, const DatasetOperationContext& context, QString* error) const
{
    if (plan.format != QLatin1String(kSemanticMaskFormat) || stagingPath.isEmpty()
        || plan.planHash.isEmpty() || plan.planHash != splitPlanHash(plan.manifest)) {
        if (error) {
            *error = QStringLiteral("语义分割 Mask 拆分计划无效或已被篡改。");
        }
        return false;
    }
    const QDir sourceRoot(plan.sourceRoot);
    const QDir staging(stagingPath);
    if (!sourceRoot.exists() || (staging.exists() && !staging.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty())) {
        if (error) {
            *error = QStringLiteral("拆分 staging 必须为空，且源数据集必须存在。");
        }
        return false;
    }
    if (!QDir().mkpath(staging.absolutePath())) {
        if (error) {
            *error = QStringLiteral("无法创建拆分 staging：%1").arg(stagingPath);
        }
        return false;
    }
    const QJsonArray classes = plan.manifest.value(QStringLiteral("classes")).toArray();
    QSaveFile classesFile(staging.filePath(QStringLiteral("classes.txt")));
    QByteArray classesText;
    for (const QJsonValue& item : classes) {
        classesText += item.toString().toUtf8() + '\n';
    }
    if (!classesFile.open(QIODevice::WriteOnly) || classesFile.write(classesText) != classesText.size() || !classesFile.commit()) {
        if (error) {
            *error = QStringLiteral("无法写入拆分 classes.txt：%1").arg(classesFile.errorString());
        }
        return false;
    }
    const QJsonArray entries = plan.manifest.value(QStringLiteral("entries")).toArray();
    for (int index = 0; index < entries.size(); ++index) {
        if (canceled(context)) {
            if (error) {
                *error = QStringLiteral("dataset_split_materialize_canceled");
            }
            return false;
        }
        const QJsonObject entry = entries.at(index).toObject();
        const QString sourceImageRelative = entry.value(QStringLiteral("sourceImageRelativePath")).toString();
        const QString sourceMaskRelative = entry.value(QStringLiteral("sourceMaskRelativePath")).toString();
        const QString sourceImage = sourceRoot.filePath(sourceImageRelative);
        const QString sourceMask = sourceRoot.filePath(sourceMaskRelative);
        const QString targetImage = entry.value(QStringLiteral("targetImage")).toString();
        const QString targetMask = entry.value(QStringLiteral("targetMask")).toString();
        QString imageSha256;
        QString maskSha256;
        qint64 imageBytes = 0;
        qint64 maskBytes = 0;
        if (!safeRelativePath(sourceImageRelative) || !safeRelativePath(sourceMaskRelative)
            || !sourcePathIsWithinRoot(sourceRoot, sourceImage) || !sourcePathIsWithinRoot(sourceRoot, sourceMask)
            || !safeRelativePath(targetImage) || !safeRelativePath(targetMask)) {
            if (error) {
                *error = QStringLiteral("拆分计划包含不安全路径。");
            }
            return false;
        }
        if (!fileIdentity(sourceImage, &imageSha256, &imageBytes, error)
            || !fileIdentity(sourceMask, &maskSha256, &maskBytes, error)
            || imageSha256 != entry.value(QStringLiteral("sourceImageSha256")).toString()
            || maskSha256 != entry.value(QStringLiteral("sourceMaskSha256")).toString()
            || imageBytes != entry.value(QStringLiteral("sourceImageBytes")).toString().toLongLong()
            || maskBytes != entry.value(QStringLiteral("sourceMaskBytes")).toString().toLongLong()) {
            if (error && error->isEmpty()) *error = QStringLiteral("dataset_source_changed_after_plan");
            return false;
        }
        const QString imageDestination = staging.filePath(targetImage);
        const QString maskDestination = staging.filePath(targetMask);
        if (QFileInfo::exists(imageDestination) || QFileInfo::exists(maskDestination)
            || !QDir().mkpath(QFileInfo(imageDestination).absolutePath()) || !QDir().mkpath(QFileInfo(maskDestination).absolutePath())
            || !QFile::copy(sourceImage, imageDestination) || !QFile::copy(sourceMask, maskDestination)) {
            if (error) {
                *error = QStringLiteral("无法物化拆分样本：%1").arg(sourceImage);
            }
            return false;
        }
        if (context.reportProgress) {
            context.reportProgress(index + 1, QStringLiteral("已物化语义分割样本。"));
        }
    }
    QSaveFile manifestFile(staging.filePath(QStringLiteral("split_plan.json")));
    if (!manifestFile.open(QIODevice::WriteOnly)
        || manifestFile.write(QJsonDocument(plan.manifest).toJson(QJsonDocument::Indented)) < 0
        || !manifestFile.commit()) {
        if (error) {
            *error = QStringLiteral("无法写入拆分计划清单：%1").arg(manifestFile.errorString());
        }
        return false;
    }
    return true;
}

bool SemanticMaskDatasetDriverV2::snapshot(const DatasetInspection& inspection, const QString& manifestPath, const DatasetSnapshotOptions& options, DatasetSnapshotResult* result, QString* error) const
{
    if (inspection.format != QLatin1String(kSemanticMaskFormat)) {
        if (error) {
            *error = QStringLiteral("语义分割 Mask 快照格式无效。");
        }
        return false;
    }
    DatasetSnapshotOptions snapshotOptions = options;
    snapshotOptions.classDefinitions = inspection.details.value(QStringLiteral("classes")).toArray();
    return createDatasetSnapshotV2(inspection.sourcePath, manifestPath, inspection.format, id(), version(), snapshotOptions, result, error);
}

} // namespace aitrain::v2
