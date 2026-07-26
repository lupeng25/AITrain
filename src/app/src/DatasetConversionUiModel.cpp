#include "DatasetConversionUiModel.h"
#include "aitrain/product/ProductCapabilityContract.h"

#include <QDir>
#include <QFileInfo>

namespace aitrain_app {
namespace {

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
    if (format == QStringLiteral("yolo_obb")) {
        return QStringLiteral("YOLO OBB");
    }
    if (format == QStringLiteral("xanylabeling_xlabel")) {
        return QStringLiteral("X-AnyLabeling XLABEL");
    }
    return format;
}

QStringList supportedDatasetConversionSourceFormats()
{
    QStringList formats;
    for (const aitrain::DatasetConversionRouteContract& route :
        aitrain::ProductCapabilityContract::instance().datasetConversionRoutes()) {
        if (!formats.contains(route.sourceFormat)) formats.append(route.sourceFormat);
    }
    return formats;
}

QStringList supportedDatasetConversionTargets(const QString& sourceFormat)
{
    const QString source = sourceFormat.trimmed().toLower();
    QStringList targets;
    for (const aitrain::DatasetConversionRouteContract& route :
        aitrain::ProductCapabilityContract::instance().datasetConversionRoutes()) {
        if (route.sourceFormat == source && !targets.contains(route.targetFormat)) {
            targets.append(route.targetFormat);
        }
    }
    return targets;
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

    appendIfPresent(&validation.messages, validation.sourceFormatError);
    appendIfPresent(&validation.messages, validation.targetFormatError);
    appendIfPresent(&validation.messages, validation.inputPathError);

    validation.ok = validation.messages.isEmpty();
    if (validation.ok) {
        validation.summary = QStringLiteral("可以开始转换。");
    } else {
        validation.summary = QStringLiteral("请修正 %1 个字段后再转换。").arg(errorFieldCount(validation));
    }
    return validation;
}

} // namespace aitrain_app
