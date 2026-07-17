#include "aitrain/core/CapabilityRegistry.h"

#include <QJsonArray>

namespace aitrain {
namespace {

QString canonicalTaskType(const QString& value)
{
    return value.trimmed().toLower();
}

QString canonicalDatasetFormat(const QString& value)
{
    return value.trimmed().toLower();
}

QString canonicalBackendId(const QString& value)
{
    return value.trimmed().toLower();
}

QStringList yoloModelPresets(bool segmentation)
{
    const QStringList families = {
        QStringLiteral("yolov8"),
        QStringLiteral("yolo11"),
        QStringLiteral("yolo12"),
        QStringLiteral("yolo26")
    };
    const QStringList scales = {
        QStringLiteral("n"),
        QStringLiteral("s"),
        QStringLiteral("m"),
        QStringLiteral("l"),
        QStringLiteral("x")
    };
    QStringList presets;
    if (!segmentation) {
        for (const QString& scale : scales) {
            presets << QStringLiteral("yolov5%1.yaml").arg(scale)
                    << QStringLiteral("yolov5%1u.pt").arg(scale);
        }
    }
    for (const QString& family : families) {
        for (const QString& scale : scales) {
            const QString stem = segmentation
                ? QStringLiteral("%1%2-seg").arg(family, scale)
                : QStringLiteral("%1%2").arg(family, scale);
            presets << QStringLiteral("%1.yaml").arg(stem)
                    << QStringLiteral("%1.pt").arg(stem);
        }
    }
    if (!segmentation) {
        for (const QString& scale : scales) {
            presets << QStringLiteral("yolov8%1-p2.yaml").arg(scale)
                    << QStringLiteral("yolov8%1-p6.yaml").arg(scale);
        }
    }
    return presets;
}

QStringList yoloObbModelPresets()
{
    const QStringList scales = {
        QStringLiteral("n"),
        QStringLiteral("s"),
        QStringLiteral("m"),
        QStringLiteral("l"),
        QStringLiteral("x")
    };
    QStringList presets;
    for (const QString& scale : scales) {
        presets << QStringLiteral("yolo11%1-obb.pt").arg(scale);
    }
    for (const QString& scale : scales) {
        presets << QStringLiteral("yolo11%1-obb.yaml").arg(scale);
    }
    return presets;
}

BackendDescriptor makeBackend(
    const QString& id,
    const QString& displayName,
    const QStringList& taskTypes,
    const QStringList& datasetFormats,
    const QStringList& presets,
    const QStringList& exportFormats,
    const QString& runtime,
    const QString& devicePolicy,
    const QStringList& limitations = {})
{
    BackendDescriptor value;
    value.id = id;
    value.displayName = displayName;
    value.taskTypes = taskTypes;
    value.datasetFormats = datasetFormats;
    value.modelPresets = presets;
    value.exportFormats = exportFormats;
    value.runtime = runtime;
    value.devicePolicy = devicePolicy;
    value.limitations = limitations;
    return value;
}

CapabilityDescriptor makeCapability(
    const QString& id,
    const QString& displayName,
    const QStringList& taskTypes,
    const QStringList& datasetFormats,
    const QStringList& backendIds,
    const QStringList& limitations = {})
{
    CapabilityDescriptor value;
    value.id = id;
    value.displayName = displayName;
    value.taskTypes = taskTypes;
    value.datasetFormats = datasetFormats;
    value.backendIds = backendIds;
    value.limitations = limitations;
    return value;
}

QJsonArray jsonArray(const QStringList& values)
{
    return QJsonArray::fromStringList(values);
}

} // namespace

QJsonObject BackendDescriptor::toJson() const
{
    return QJsonObject{
        {QStringLiteral("id"), id},
        {QStringLiteral("displayName"), displayName},
        {QStringLiteral("taskTypes"), jsonArray(taskTypes)},
        {QStringLiteral("datasetFormats"), jsonArray(datasetFormats)},
        {QStringLiteral("modelPresets"), jsonArray(modelPresets)},
        {QStringLiteral("exportFormats"), jsonArray(exportFormats)},
        {QStringLiteral("runtime"), runtime},
        {QStringLiteral("devicePolicy"), devicePolicy},
        {QStringLiteral("supportsCancel"), true},
        {QStringLiteral("limitations"), jsonArray(limitations)}};
}

QJsonObject CapabilityDescriptor::toJson() const
{
    return QJsonObject{
        {QStringLiteral("id"), id},
        {QStringLiteral("displayName"), displayName},
        {QStringLiteral("taskTypes"), jsonArray(taskTypes)},
        {QStringLiteral("datasetFormats"), jsonArray(datasetFormats)},
        {QStringLiteral("backendIds"), jsonArray(backendIds)},
        {QStringLiteral("limitations"), jsonArray(limitations)}};
}

const BuiltinCapabilityRegistry& BuiltinCapabilityRegistry::instance()
{
    static const BuiltinCapabilityRegistry registry;
    return registry;
}

BuiltinCapabilityRegistry::BuiltinCapabilityRegistry()
{
    backends_ = {
        makeBackend(QStringLiteral("ultralytics_yolo_detect"), QStringLiteral("Ultralytics YOLO Detection"),
            {QStringLiteral("detection")}, {QStringLiteral("yolo_detection")},
            yoloModelPresets(false),
            {QStringLiteral("onnx"), QStringLiteral("ncnn"), QStringLiteral("tensorrt")},
            QStringLiteral("aitrain_yolo_runtime"), QStringLiteral("gpu_recommended")),
        makeBackend(QStringLiteral("ultralytics_yolo_segment"), QStringLiteral("Ultralytics YOLO Segmentation"),
            {QStringLiteral("segmentation")}, {QStringLiteral("yolo_segmentation")},
            yoloModelPresets(true),
            {QStringLiteral("onnx"), QStringLiteral("ncnn"), QStringLiteral("tensorrt")},
            QStringLiteral("aitrain_yolo_runtime"), QStringLiteral("gpu_recommended")),
        makeBackend(QStringLiteral("ultralytics_yolo_obb"), QStringLiteral("Ultralytics YOLO OBB"),
            {QStringLiteral("obb_detection")}, {QStringLiteral("yolo_obb")},
            yoloObbModelPresets(), {QStringLiteral("onnx")},
            QStringLiteral("aitrain_onnxruntime"), QStringLiteral("gpu_recommended"),
            {QStringLiteral("OBB v1 仅支持 ONNX Runtime 部署。")}),
        makeBackend(QStringLiteral("smp_semantic_segmentation"), QStringLiteral("SMP Semantic Segmentation"),
            {QStringLiteral("semantic_segmentation")}, {QStringLiteral("semantic_segmentation_mask")},
            {QStringLiteral("smp_unet_resnet34"), QStringLiteral("smp_unetplusplus_resnet34"),
                QStringLiteral("smp_fpn_resnet34"), QStringLiteral("smp_deeplabv3plus_resnet50"),
                QStringLiteral("smp_segformer_mit_b0")},
            {QStringLiteral("onnx")},
            QStringLiteral("aitrain_onnxruntime"), QStringLiteral("cpu_supported"),
            {QStringLiteral("SMP 不支持 NCNN 或 TensorRT 导出。")}),
        makeBackend(QStringLiteral("anomalib_patchcore"), QStringLiteral("Anomalib PatchCore"),
            {QStringLiteral("anomaly_detection")}, {QStringLiteral("anomaly_folder")},
            {QStringLiteral("anomalib_patchcore_wide_resnet50_2")}, {},
            QStringLiteral("anomalib_python"), QStringLiteral("cpu_supported"),
            {QStringLiteral("异常检测仅使用 Worker 管理的 Anomalib Python Runtime。")}),
        makeBackend(QStringLiteral("anomalib_efficientad"), QStringLiteral("Anomalib EfficientAD"),
            {QStringLiteral("anomaly_detection")}, {QStringLiteral("anomaly_folder")},
            {QStringLiteral("anomalib_efficientad_s")}, {},
            QStringLiteral("anomalib_python"), QStringLiteral("gpu_recommended"),
            {QStringLiteral("EfficientAD 需要显式 ImageNet 数据目录且 batchSize 固定为 1。")}),
        makeBackend(QStringLiteral("paddleocr_det_official"), QStringLiteral("PaddleOCR Detection"),
            {QStringLiteral("ocr_detection")}, {QStringLiteral("paddleocr_det")},
            {QStringLiteral("PP-OCRv5_mobile_det"), QStringLiteral("PP-OCRv5_server_det"),
                QStringLiteral("PP-OCRv6_tiny_det"), QStringLiteral("PP-OCRv6_small_det"),
                QStringLiteral("PP-OCRv6_medium_det"), QStringLiteral("PP-OCRv4_mobile_det")},
            {},
            QStringLiteral("paddleocr_official"), QStringLiteral("cpu_supported"),
            {QStringLiteral("OCR 交付与验收仅使用 PaddleOCR 官方报告。")}),
        makeBackend(QStringLiteral("paddleocr_rec_official"), QStringLiteral("PaddleOCR Recognition"),
            {QStringLiteral("ocr_recognition")}, {QStringLiteral("paddleocr_rec")},
            {QStringLiteral("PP-OCRv5_mobile_rec"), QStringLiteral("PP-OCRv5_server_rec"),
                QStringLiteral("en_PP-OCRv5_mobile_rec"), QStringLiteral("PP-OCRv6_tiny_rec"),
                QStringLiteral("PP-OCRv6_small_rec"), QStringLiteral("PP-OCRv6_medium_rec"),
                QStringLiteral("PP-OCRv4_mobile_rec")},
            {},
            QStringLiteral("paddleocr_official"), QStringLiteral("cpu_supported"),
            {QStringLiteral("OCR 交付与验收仅使用 PaddleOCR 官方报告。")})};

    capabilities_ = {
        makeCapability(QStringLiteral("yolo"), QStringLiteral("YOLO"),
            {QStringLiteral("detection"), QStringLiteral("segmentation"), QStringLiteral("obb_detection")},
            {QStringLiteral("yolo_detection"), QStringLiteral("yolo_segmentation"), QStringLiteral("yolo_obb")},
            {QStringLiteral("ultralytics_yolo_detect"), QStringLiteral("ultralytics_yolo_segment"), QStringLiteral("ultralytics_yolo_obb")}),
        makeCapability(QStringLiteral("semantic_segmentation"), QStringLiteral("专用语义分割"),
            {QStringLiteral("semantic_segmentation")}, {QStringLiteral("semantic_segmentation_mask")},
            {QStringLiteral("smp_semantic_segmentation")}),
        makeCapability(QStringLiteral("anomaly_detection"), QStringLiteral("异常检测与定位"),
            {QStringLiteral("anomaly_detection")}, {QStringLiteral("anomaly_folder")},
            {QStringLiteral("anomalib_patchcore"), QStringLiteral("anomalib_efficientad")}),
        makeCapability(QStringLiteral("paddleocr"), QStringLiteral("PaddleOCR"),
            {QStringLiteral("ocr_detection"), QStringLiteral("ocr_recognition")},
            {QStringLiteral("paddleocr_det"), QStringLiteral("paddleocr_rec")},
            {QStringLiteral("paddleocr_det_official"), QStringLiteral("paddleocr_rec_official")},
            {QStringLiteral("OCR 仅提供官方链路与证据材料。")}),
        makeCapability(QStringLiteral("dataset_interop"), QStringLiteral("数据集互操作"),
            {QStringLiteral("dataset_conversion")},
            {QStringLiteral("coco_json"), QStringLiteral("voc_xml"), QStringLiteral("yolo_detection"),
                QStringLiteral("yolo_segmentation"), QStringLiteral("yolo_obb"), QStringLiteral("xanylabeling_xlabel")}, {},
            {QStringLiteral("不提供 LabelMe 作为产品数据格式。")})};
}

QVector<CapabilityDescriptor> BuiltinCapabilityRegistry::capabilities() const
{
    return capabilities_;
}

QVector<BackendDescriptor> BuiltinCapabilityRegistry::backends() const
{
    return backends_;
}

CapabilityDescriptor BuiltinCapabilityRegistry::capability(const QString& id) const
{
    const QString normalizedId = id.trimmed().toLower();
    for (const CapabilityDescriptor& value : capabilities_) {
        if (value.id == normalizedId) {
            return value;
        }
    }
    return {};
}

BackendDescriptor BuiltinCapabilityRegistry::backend(const QString& id) const
{
    const QString normalizedId = canonicalBackendId(id);
    for (const BackendDescriptor& value : backends_) {
        if (value.id == normalizedId) {
            return value;
        }
    }
    return {};
}

QStringList BuiltinCapabilityRegistry::taskTypesForCapability(const QString& capabilityId) const
{
    return capability(capabilityId).taskTypes;
}

QStringList BuiltinCapabilityRegistry::datasetFormatsForTask(const QString& taskType) const
{
    const QString normalizedTaskType = canonicalTaskType(taskType);
    QStringList values;
    for (const BackendDescriptor& value : backends_) {
        if (value.taskTypes.contains(normalizedTaskType)) {
            for (const QString& format : value.datasetFormats) {
                if (!values.contains(format)) {
                    values.append(format);
                }
            }
        }
    }
    return values;
}

QStringList BuiltinCapabilityRegistry::backendsForTask(const QString& taskType, const QString& datasetFormat) const
{
    const QString normalizedTaskType = canonicalTaskType(taskType);
    const QString normalizedDatasetFormat = canonicalDatasetFormat(datasetFormat);
    QStringList values;
    for (const BackendDescriptor& value : backends_) {
        if (value.taskTypes.contains(normalizedTaskType)
            && (normalizedDatasetFormat.isEmpty() || value.datasetFormats.contains(normalizedDatasetFormat))) {
            values.append(value.id);
        }
    }
    return values;
}

bool BuiltinCapabilityRegistry::supports(const QString& capabilityId,
    const QString& taskType,
    const QString& datasetFormat,
    const QString& backendId,
    QString* error) const
{
    const QString normalizedTaskType = canonicalTaskType(taskType);
    const QString normalizedDatasetFormat = canonicalDatasetFormat(datasetFormat);
    const QString normalizedBackendId = canonicalBackendId(backendId);
    const CapabilityDescriptor selectedCapability = capability(capabilityId);
    const BackendDescriptor selectedBackend = backend(normalizedBackendId);
    const bool supported = !selectedCapability.id.isEmpty()
        && !selectedBackend.id.isEmpty()
        && selectedCapability.taskTypes.contains(normalizedTaskType)
        && selectedCapability.datasetFormats.contains(normalizedDatasetFormat)
        && selectedCapability.backendIds.contains(normalizedBackendId)
        && selectedBackend.taskTypes.contains(normalizedTaskType)
        && selectedBackend.datasetFormats.contains(normalizedDatasetFormat);
    if (!supported && error) {
        *error = QStringLiteral("内置能力不支持 capability=%1 taskType=%2 datasetFormat=%3 backend=%4。")
            .arg(capabilityId, taskType, datasetFormat, backendId);
    }
    return supported;
}

QJsonObject BuiltinCapabilityRegistry::toJson() const
{
    QJsonArray capabilityArray;
    for (const CapabilityDescriptor& value : capabilities_) {
        capabilityArray.append(value.toJson());
    }
    QJsonArray backendArray;
    for (const BackendDescriptor& value : backends_) {
        backendArray.append(value.toJson());
    }
    return QJsonObject{
        {QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("capabilities"), capabilityArray},
        {QStringLiteral("backends"), backendArray}};
}

} // namespace aitrain
