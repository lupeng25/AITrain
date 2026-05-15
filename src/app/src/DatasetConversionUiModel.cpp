#include "DatasetConversionUiModel.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain_app {
namespace {

struct DatasetConversionPair {
    const char* source;
    const char* const* targets;
    int targetCount;
};

const char* const cocoTargets[] = {"yolo_detection", "yolo_segmentation"};
const char* const vocTargets[] = {"yolo_detection"};
const char* const yoloDetectionTargets[] = {"coco_json", "voc_xml"};
const char* const yoloSegmentationTargets[] = {"coco_json"};

const DatasetConversionPair conversionMatrix[] = {
    {"coco_json", cocoTargets, 2},
    {"voc_xml", vocTargets, 1},
    {"yolo_detection", yoloDetectionTargets, 2},
    {"yolo_segmentation", yoloSegmentationTargets, 1},
};

void appendIfPresent(QStringList* messages, const QString& message)
{
    if (!message.isEmpty()) {
        messages->append(message);
    }
}

int errorFieldCount(const DatasetConversionValidation& validation)
{
    int count = 0;
    if (!validation.sourceFormatError.isEmpty()) {
        ++count;
    }
    if (!validation.targetFormatError.isEmpty()) {
        ++count;
    }
    if (!validation.inputPathError.isEmpty()) {
        ++count;
    }
    if (!validation.outputPathError.isEmpty()) {
        ++count;
    }
    return count;
}

} // namespace

QString datasetConversionFormatLabel(const QString& format)
{
    if (format == QStringLiteral("coco_json")) {
        return QStringLiteral("COCO JSON");
    }
    if (format == QStringLiteral("voc_xml")) {
        return QStringLiteral("Pascal VOC XML");
    }
    if (format == QStringLiteral("yolo_detection")) {
        return QStringLiteral("YOLO Detection");
    }
    if (format == QStringLiteral("yolo_segmentation")) {
        return QStringLiteral("YOLO Segmentation");
    }
    return format;
}

QStringList supportedDatasetConversionSourceFormats()
{
    return QStringList({QStringLiteral("coco_json"),
        QStringLiteral("voc_xml"),
        QStringLiteral("yolo_detection"),
        QStringLiteral("yolo_segmentation")});
}

QStringList supportedDatasetConversionTargets(const QString& sourceFormat)
{
    for (const DatasetConversionPair& pair : conversionMatrix) {
        if (sourceFormat == QLatin1String(pair.source)) {
            QStringList targets;
            for (int index = 0; index < pair.targetCount; ++index) {
                targets.append(QLatin1String(pair.targets[index]));
            }
            return targets;
        }
    }
    return {};
}

bool isSupportedDatasetConversionPair(const QString& sourceFormat, const QString& targetFormat)
{
    return supportedDatasetConversionTargets(sourceFormat).contains(targetFormat);
}

QString normalizedDatasetConversionPath(const QString& path)
{
    const QString trimmed = QDir::fromNativeSeparators(path.trimmed());
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

DatasetConversionValidation validateDatasetConversionForm(const DatasetConversionForm& form)
{
    DatasetConversionValidation validation;
    if (form.workerRunning) {
        validation.summary = QStringLiteral("Worker 正在执行任务，稍后再转换数据集。");
        validation.messages.append(validation.summary);
        return validation;
    }

    const QString sourceFormat = form.sourceFormat.trimmed();
    const QString targetFormat = form.targetFormat.trimmed();

    if (sourceFormat.isEmpty()) {
        validation.sourceFormatError = QStringLiteral("请选择源格式。");
    } else if (!supportedDatasetConversionSourceFormats().contains(sourceFormat)) {
        validation.sourceFormatError = QStringLiteral("当前不支持该源格式。");
    }

    if (targetFormat.isEmpty()) {
        validation.targetFormatError = QStringLiteral("请选择目标格式。");
    } else if (!sourceFormat.isEmpty()
        && !isSupportedDatasetConversionPair(sourceFormat, targetFormat)) {
        validation.targetFormatError = QStringLiteral("当前源格式不支持转换到该目标格式。");
    }

    QString normalizedInputPath;
    if (form.inputPath.trimmed().isEmpty()) {
        validation.inputPathError = QStringLiteral("请选择输入路径。");
    } else {
        normalizedInputPath = normalizedDatasetConversionPath(form.inputPath);
        const QFileInfo inputInfo(normalizedInputPath);
        if (!inputInfo.exists()) {
            validation.inputPathError = QStringLiteral("输入路径不存在。");
        } else if (sourceFormat == QStringLiteral("coco_json")) {
            if (!inputInfo.isFile() || inputInfo.suffix().compare(QStringLiteral("json"), Qt::CaseInsensitive) != 0) {
                validation.inputPathError = QStringLiteral("COCO 输入路径必须是 JSON 文件。");
            }
        } else if (sourceFormat == QStringLiteral("voc_xml")) {
            if (!inputInfo.isDir()
                && !(inputInfo.isFile() && inputInfo.suffix().compare(QStringLiteral("xml"), Qt::CaseInsensitive) == 0)) {
                validation.inputPathError = QStringLiteral("VOC 输入路径必须是 XML 文件或目录。");
            }
        } else if (!inputInfo.isDir()) {
            validation.inputPathError = QStringLiteral("输入路径必须是目录。");
        }
    }

    if (form.outputPath.trimmed().isEmpty()) {
        validation.outputPathError = QStringLiteral("请选择输出目录。");
    } else {
        const QString normalizedOutputPath = normalizedDatasetConversionPath(form.outputPath);
        const QFileInfo outputInfo(normalizedOutputPath);
        if (outputInfo.exists() && !outputInfo.isDir()) {
            validation.outputPathError = QStringLiteral("输出路径必须是目录。");
        } else if (!normalizedInputPath.isEmpty()
#ifdef Q_OS_WIN
            && normalizedOutputPath.compare(normalizedInputPath, Qt::CaseInsensitive) == 0
#else
            && normalizedOutputPath == normalizedInputPath
#endif
        ) {
            validation.outputPathError = QStringLiteral("输出目录不能与输入路径相同。");
        } else {
            const QDir outputParent = outputInfo.absoluteDir();
            const QFileInfo outputParentInfo(outputParent.absolutePath());
            if (!outputParent.exists()) {
                validation.outputPathError = QStringLiteral("输出目录的父目录不存在。");
            } else if (!outputParentInfo.isWritable()) {
                validation.outputPathError = QStringLiteral("输出目录的父目录不可写。");
            }
        }
    }

    appendIfPresent(&validation.messages, validation.sourceFormatError);
    appendIfPresent(&validation.messages, validation.targetFormatError);
    appendIfPresent(&validation.messages, validation.inputPathError);
    appendIfPresent(&validation.messages, validation.outputPathError);

    validation.ok = validation.messages.isEmpty();
    if (validation.ok) {
        validation.summary = QStringLiteral("可以开始转换。");
    } else {
        validation.summary = QStringLiteral("请修正 %1 个字段后再转换。").arg(errorFieldCount(validation));
    }
    return validation;
}

} // namespace aitrain_app
