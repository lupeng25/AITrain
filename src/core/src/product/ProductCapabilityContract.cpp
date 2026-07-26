#include "aitrain/product/ProductCapabilityContract.h"

#include <QJsonArray>
#include <QSet>

namespace aitrain {
namespace {

QString normalized(const QString& value)
{
    return value.trimmed().toLower();
}

QStringList yoloModelPresets(bool segmentation)
{
    const QStringList families{
        QStringLiteral("yolov8"),
        QStringLiteral("yolo11"),
        QStringLiteral("yolo12"),
        QStringLiteral("yolo26")};
    const QStringList scales{
        QStringLiteral("n"), QStringLiteral("s"), QStringLiteral("m"),
        QStringLiteral("l"), QStringLiteral("x")};
    QStringList presets;
    if (!segmentation) {
        for (const QString& scale : scales) {
            presets.append(QStringLiteral("yolov5%1.yaml").arg(scale));
            presets.append(QStringLiteral("yolov5%1u.pt").arg(scale));
        }
    }
    for (const QString& family : families) {
        for (const QString& scale : scales) {
            const QString stem = segmentation
                ? QStringLiteral("%1%2-seg").arg(family, scale)
                : QStringLiteral("%1%2").arg(family, scale);
            presets.append(QStringLiteral("%1.yaml").arg(stem));
            presets.append(QStringLiteral("%1.pt").arg(stem));
        }
    }
    if (!segmentation) {
        for (const QString& scale : scales) {
            presets.append(QStringLiteral("yolov8%1-p2.yaml").arg(scale));
            presets.append(QStringLiteral("yolov8%1-p6.yaml").arg(scale));
        }
    }
    return presets;
}

QStringList yoloObbModelPresets()
{
    const QStringList scales{
        QStringLiteral("n"), QStringLiteral("s"), QStringLiteral("m"),
        QStringLiteral("l"), QStringLiteral("x")};
    QStringList presets;
    for (const QString& scale : scales) {
        presets.append(QStringLiteral("yolo11%1-obb.pt").arg(scale));
    }
    for (const QString& scale : scales) {
        presets.append(QStringLiteral("yolo11%1-obb.yaml").arg(scale));
    }
    return presets;
}

QJsonArray strings(const QStringList& values)
{
    return QJsonArray::fromStringList(values);
}

QJsonObject capabilityJson(const CapabilityContract& value)
{
    return {
        {QStringLiteral("id"), value.id},
        {QStringLiteral("displayName"), value.displayName},
        {QStringLiteral("taskTypes"), strings(value.taskTypes)},
        {QStringLiteral("datasetFormats"), strings(value.datasetFormats)},
        {QStringLiteral("backendIds"), strings(value.backendIds)},
        {QStringLiteral("limitations"), strings(value.limitations)}};
}

QJsonObject pythonProfileJson(const PythonEnvironmentProfile& value)
{
    return {
        {QStringLiteral("id"), value.id},
        {QStringLiteral("dedicatedEnvironmentVariable"), value.dedicatedEnvironmentVariable},
        {QStringLiteral("requirementsFile"), value.requirementsFile},
        {QStringLiteral("requiredModules"), strings(value.requiredModules)},
        {QStringLiteral("requiresPaddleOcrSource"), value.requiresPaddleOcrSource}};
}

QJsonObject trainingBackendJson(const TrainingBackendContract& value)
{
    return {
        {QStringLiteral("id"), value.id},
        {QStringLiteral("displayName"), value.displayName},
        {QStringLiteral("capabilityId"), value.capabilityId},
        {QStringLiteral("taskType"), value.taskType},
        {QStringLiteral("datasetFormat"), value.datasetFormat},
        {QStringLiteral("modelPresets"), strings(value.modelPresets)},
        {QStringLiteral("exportFormats"), strings(value.exportFormats)},
        {QStringLiteral("runtime"), value.legacyRuntimeId},
        {QStringLiteral("devicePolicy"), value.devicePolicy},
        {QStringLiteral("supportsCancel"), value.supportsCancel},
        {QStringLiteral("pythonProfileId"), value.pythonProfileId},
        {QStringLiteral("officialArtifactFormat"), value.officialArtifactFormat},
        {QStringLiteral("modelFamily"), value.modelFamily},
        {QStringLiteral("decoder"), value.decoder},
        {QStringLiteral("runtimeRoutes"), strings(value.runtimeRouteIds)},
        {QStringLiteral("limitations"), strings(value.limitations)}};
}

QJsonObject runtimeRouteJson(const RuntimeRouteContract& value)
{
    return {
        {QStringLiteral("modelFamily"), value.modelFamily},
        {QStringLiteral("routeId"), value.routeId},
        {QStringLiteral("executionAuthority"), runtimeExecutionAuthorityToString(value.executionAuthority)},
        {QStringLiteral("productState"), runtimeProductStateToString(value.productState)},
        {QStringLiteral("acceptedArtifactFormats"), strings(value.acceptedArtifactFormats)},
        {QStringLiteral("limitations"), strings(value.limitations)}};
}

QJsonObject datasetConversionJson(const DatasetConversionRouteContract& value)
{
    return {
        {QStringLiteral("sourceFormat"), value.sourceFormat},
        {QStringLiteral("targetFormat"), value.targetFormat},
        {QStringLiteral("sourceSemantics"), value.sourceSemantics},
        {QStringLiteral("targetSemantics"), value.targetSemantics},
        {QStringLiteral("limitations"), strings(value.limitations)}};
}

} // namespace

QString runtimeExecutionAuthorityToString(RuntimeExecutionAuthority authority)
{
    switch (authority) {
    case RuntimeExecutionAuthority::AitrainCpp: return QStringLiteral("aitrain_cpp");
    case RuntimeExecutionAuthority::WorkerManaged: return QStringLiteral("worker_managed");
    case RuntimeExecutionAuthority::OfficialEvidence: return QStringLiteral("official_evidence");
    }
    return QStringLiteral("aitrain_cpp");
}

QString runtimeProductStateToString(RuntimeProductState state)
{
    switch (state) {
    case RuntimeProductState::Supported: return QStringLiteral("supported");
    case RuntimeProductState::NotImplemented: return QStringLiteral("not_implemented");
    case RuntimeProductState::UnsupportedByProduct: return QStringLiteral("unsupported_by_product");
    }
    return QStringLiteral("unsupported_by_product");
}

const ProductCapabilityContract& ProductCapabilityContract::instance()
{
    static const ProductCapabilityContract contract;
    return contract;
}

ProductCapabilityContract::ProductCapabilityContract()
{
    pythonProfiles_ = {
        {QStringLiteral("yolo"), QStringLiteral("AITRAIN_YOLO_PYTHON_EXECUTABLE"),
            QStringLiteral("requirements-yolo.txt"),
            {QStringLiteral("ultralytics"), QStringLiteral("torch"), QStringLiteral("onnx"),
                QStringLiteral("onnxruntime")}, false},
        {QStringLiteral("smp_semantic_segmentation"), QStringLiteral("AITRAIN_SMP_PYTHON_EXECUTABLE"),
            QStringLiteral("requirements-smp.txt"),
            {QStringLiteral("segmentation_models_pytorch"), QStringLiteral("torch"),
                QStringLiteral("torchvision"), QStringLiteral("timm"), QStringLiteral("onnx"),
                QStringLiteral("onnxruntime")}, false},
        {QStringLiteral("anomaly_detection"), QStringLiteral("AITRAIN_ANOMALIB_PYTHON_EXECUTABLE"),
            QStringLiteral("requirements-anomaly.txt"),
            {QStringLiteral("anomalib"), QStringLiteral("torch"), QStringLiteral("torchvision"),
                QStringLiteral("lightning"), QStringLiteral("timm"), QStringLiteral("PIL"),
                QStringLiteral("numpy"), QStringLiteral("cv2")}, false},
        {QStringLiteral("ocr"), QStringLiteral("AITRAIN_OCR_PYTHON_EXECUTABLE"),
            QStringLiteral("requirements-ocr.txt"),
            {QStringLiteral("paddle"), QStringLiteral("paddleocr")}, true}};

    trainingBackends_ = {
        {QStringLiteral("ultralytics_yolo_detect"), QStringLiteral("Ultralytics YOLO Detection"),
            QStringLiteral("yolo"), QStringLiteral("detection"), QStringLiteral("yolo_detection"),
            yoloModelPresets(false),
            {QStringLiteral("onnx"), QStringLiteral("ncnn"), QStringLiteral("tensorrt")},
            QStringLiteral("aitrain_yolo_runtime"), QStringLiteral("gpu_recommended"), true,
            QStringLiteral("yolo"), QStringLiteral("onnx"), QStringLiteral("yolo_detection"),
            QStringLiteral("yolo_detection_v8"), {QStringLiteral("aitrain_onnxruntime")}, {}},
        {QStringLiteral("ultralytics_yolo_segment"), QStringLiteral("Ultralytics YOLO Segmentation"),
            QStringLiteral("yolo"), QStringLiteral("segmentation"), QStringLiteral("yolo_segmentation"),
            yoloModelPresets(true),
            {QStringLiteral("onnx"), QStringLiteral("ncnn"), QStringLiteral("tensorrt")},
            QStringLiteral("aitrain_yolo_runtime"), QStringLiteral("gpu_recommended"), true,
            QStringLiteral("yolo"), QStringLiteral("onnx"), QStringLiteral("yolo_segmentation"),
            QStringLiteral("yolo_segmentation_v8"), {QStringLiteral("aitrain_onnxruntime")}, {}},
        {QStringLiteral("ultralytics_yolo_obb"), QStringLiteral("Ultralytics YOLO OBB"),
            QStringLiteral("yolo"), QStringLiteral("obb_detection"), QStringLiteral("yolo_obb"),
            yoloObbModelPresets(), {QStringLiteral("onnx")},
            QStringLiteral("aitrain_onnxruntime"), QStringLiteral("gpu_recommended"), true,
            QStringLiteral("yolo"), QStringLiteral("onnx"), QStringLiteral("yolo_obb"),
            QStringLiteral("yolo_obb_v8"), {QStringLiteral("aitrain_onnxruntime")},
            {QStringLiteral("OBB v1 仅支持 ONNX Runtime 部署。")}},
        {QStringLiteral("smp_semantic_segmentation"), QStringLiteral("SMP Semantic Segmentation"),
            QStringLiteral("semantic_segmentation"), QStringLiteral("semantic_segmentation"),
            QStringLiteral("semantic_segmentation_mask"),
            {QStringLiteral("smp_unet_resnet34"), QStringLiteral("smp_unetplusplus_resnet34"),
                QStringLiteral("smp_fpn_resnet34"), QStringLiteral("smp_deeplabv3plus_resnet50"),
                QStringLiteral("smp_segformer_mit_b0")},
            {QStringLiteral("onnx")}, QStringLiteral("aitrain_onnxruntime"),
            QStringLiteral("cpu_supported"), true, QStringLiteral("smp_semantic_segmentation"),
            QStringLiteral("onnx"), QStringLiteral("semantic_segmentation"),
            QStringLiteral("smp_semantic_segmentation"), {QStringLiteral("aitrain_onnxruntime")},
            {QStringLiteral("SMP 仅支持 AITrain ONNX Runtime。"),
                QStringLiteral("SMP 不支持 NCNN 或 TensorRT 导出。")}},
        {QStringLiteral("anomalib_patchcore"), QStringLiteral("Anomalib PatchCore"),
            QStringLiteral("anomaly_detection"), QStringLiteral("anomaly_detection"),
            QStringLiteral("anomaly_folder"),
            {QStringLiteral("anomalib_patchcore_wide_resnet50_2")}, {},
            QStringLiteral("anomalib_python"), QStringLiteral("cpu_supported"), true,
            QStringLiteral("anomaly_detection"), QStringLiteral("anomalib_bundle"),
            QStringLiteral("anomaly_detection"), QStringLiteral("anomalib_python_sidecar_v1"),
            {QStringLiteral("anomalib_python")},
            {QStringLiteral("异常检测仅使用 Worker 管理的 Anomalib Python Runtime。")}},
        {QStringLiteral("anomalib_efficientad"), QStringLiteral("Anomalib EfficientAD"),
            QStringLiteral("anomaly_detection"), QStringLiteral("anomaly_detection"),
            QStringLiteral("anomaly_folder"), {QStringLiteral("anomalib_efficientad_s")}, {},
            QStringLiteral("anomalib_python"), QStringLiteral("gpu_recommended"), true,
            QStringLiteral("anomaly_detection"), QStringLiteral("anomalib_bundle"),
            QStringLiteral("anomaly_detection"), QStringLiteral("anomalib_python_sidecar_v1"),
            {QStringLiteral("anomalib_python")},
            {QStringLiteral("EfficientAD 需要显式 ImageNet 数据目录且 batchSize 固定为 1。")}},
        {QStringLiteral("paddleocr_det_official"), QStringLiteral("PaddleOCR Detection"),
            QStringLiteral("paddleocr"), QStringLiteral("ocr_detection"), QStringLiteral("paddleocr_det"),
            {QStringLiteral("PP-OCRv5_mobile_det"), QStringLiteral("PP-OCRv5_server_det"),
                QStringLiteral("PP-OCRv6_tiny_det"), QStringLiteral("PP-OCRv6_small_det"),
                QStringLiteral("PP-OCRv6_medium_det"), QStringLiteral("PP-OCRv4_mobile_det")},
            {}, QStringLiteral("paddleocr_official"), QStringLiteral("cpu_supported"), true,
            QStringLiteral("ocr"), QStringLiteral("paddleocr_inference_bundle"),
            QStringLiteral("ocr_detection"), QStringLiteral("paddleocr_official_det_v1"),
            {QStringLiteral("paddleocr_official")},
            {QStringLiteral("OCR 交付与验收仅使用 PaddleOCR 官方报告。")}},
        {QStringLiteral("paddleocr_rec_official"), QStringLiteral("PaddleOCR Recognition"),
            QStringLiteral("paddleocr"), QStringLiteral("ocr_recognition"), QStringLiteral("paddleocr_rec"),
            {QStringLiteral("PP-OCRv5_mobile_rec"), QStringLiteral("PP-OCRv5_server_rec"),
                QStringLiteral("en_PP-OCRv5_mobile_rec"), QStringLiteral("PP-OCRv6_tiny_rec"),
                QStringLiteral("PP-OCRv6_small_rec"), QStringLiteral("PP-OCRv6_medium_rec"),
                QStringLiteral("PP-OCRv4_mobile_rec")},
            {}, QStringLiteral("paddleocr_official"), QStringLiteral("cpu_supported"), true,
            QStringLiteral("ocr"), QStringLiteral("paddleocr_inference_bundle"),
            QStringLiteral("ocr_recognition"), QStringLiteral("paddleocr_official_rec_v1"),
            {QStringLiteral("paddleocr_official")},
            {QStringLiteral("OCR 交付与验收仅使用 PaddleOCR 官方报告。")}}};

    capabilities_ = {
        {QStringLiteral("yolo"), QStringLiteral("YOLO"),
            {QStringLiteral("detection"), QStringLiteral("segmentation"), QStringLiteral("obb_detection")},
            {QStringLiteral("yolo_detection"), QStringLiteral("yolo_segmentation"), QStringLiteral("yolo_obb")},
            {QStringLiteral("ultralytics_yolo_detect"), QStringLiteral("ultralytics_yolo_segment"),
                QStringLiteral("ultralytics_yolo_obb")}, {}},
        {QStringLiteral("semantic_segmentation"), QStringLiteral("专用语义分割"),
            {QStringLiteral("semantic_segmentation")}, {QStringLiteral("semantic_segmentation_mask")},
            {QStringLiteral("smp_semantic_segmentation")}, {}},
        {QStringLiteral("anomaly_detection"), QStringLiteral("异常检测与定位"),
            {QStringLiteral("anomaly_detection")}, {QStringLiteral("anomaly_folder")},
            {QStringLiteral("anomalib_patchcore"), QStringLiteral("anomalib_efficientad")}, {}},
        {QStringLiteral("paddleocr"), QStringLiteral("PaddleOCR"),
            {QStringLiteral("ocr_detection"), QStringLiteral("ocr_recognition")},
            {QStringLiteral("paddleocr_det"), QStringLiteral("paddleocr_rec")},
            {QStringLiteral("paddleocr_det_official"), QStringLiteral("paddleocr_rec_official")},
            {QStringLiteral("OCR 仅提供官方链路与证据材料。")}},
        {QStringLiteral("dataset_interop"), QStringLiteral("数据集互操作"),
            {QStringLiteral("dataset_conversion")},
            {QStringLiteral("coco_json"), QStringLiteral("voc_xml"), QStringLiteral("yolo_detection"),
                QStringLiteral("yolo_segmentation"), QStringLiteral("yolo_obb"),
                QStringLiteral("xanylabeling_xlabel")},
            {}, {QStringLiteral("产品 Workflow 只开放具备完整 Artifact/Driver 合同的转换路线。")}}};

    runtimeRoutes_ = {
        {QStringLiteral("yolo_detection"), QStringLiteral("aitrain_onnxruntime"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::Supported,
            {QStringLiteral("onnx")}, {}},
        {QStringLiteral("yolo_segmentation"), QStringLiteral("aitrain_onnxruntime"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::Supported,
            {QStringLiteral("onnx")}, {}},
        {QStringLiteral("yolo_obb"), QStringLiteral("aitrain_onnxruntime"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::Supported,
            {QStringLiteral("onnx")}, {QStringLiteral("OBB v1 仅支持 ONNX Runtime。")}},
        {QStringLiteral("semantic_segmentation"), QStringLiteral("aitrain_onnxruntime"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::Supported,
            {QStringLiteral("onnx")}, {QStringLiteral("SMP 仅支持 ONNX Runtime。")}},
        {QStringLiteral("yolo_detection"), QStringLiteral("aitrain_ncnn"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::Supported,
            {QStringLiteral("ncnn")}, {}},
        {QStringLiteral("yolo_segmentation"), QStringLiteral("aitrain_ncnn"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::Supported,
            {QStringLiteral("ncnn")}, {}},
        {QStringLiteral("yolo_detection"), QStringLiteral("aitrain_tensorrt"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::NotImplemented,
            {QStringLiteral("tensorrt_engine")},
            {QStringLiteral("TensorRT engine 可导出，但产品推理解码器尚未实现。")}},
        {QStringLiteral("yolo_segmentation"), QStringLiteral("aitrain_tensorrt"),
            RuntimeExecutionAuthority::AitrainCpp, RuntimeProductState::NotImplemented,
            {QStringLiteral("tensorrt_engine")},
            {QStringLiteral("TensorRT engine 可导出，但产品推理解码器尚未实现。")}},
        {QStringLiteral("anomaly_detection"), QStringLiteral("anomalib_python"),
            RuntimeExecutionAuthority::WorkerManaged, RuntimeProductState::Supported,
            {QStringLiteral("anomalib_bundle")},
            {QStringLiteral("不声明 AITrain C++ ONNX/TensorRT/NCNN 异常检测运行时。")}},
        {QStringLiteral("ocr_detection"), QStringLiteral("paddleocr_official"),
            RuntimeExecutionAuthority::OfficialEvidence, RuntimeProductState::Supported,
            {QStringLiteral("paddleocr_inference_bundle")},
            {QStringLiteral("OCR 只接受 PaddleOCR 官方工具链与报告。")}},
        {QStringLiteral("ocr_recognition"), QStringLiteral("paddleocr_official"),
            RuntimeExecutionAuthority::OfficialEvidence, RuntimeProductState::Supported,
            {QStringLiteral("paddleocr_inference_bundle")},
            {QStringLiteral("OCR 只接受 PaddleOCR 官方工具链与报告。")}}};

    datasetConversionRoutes_ = {
        {QStringLiteral("coco_json"), QStringLiteral("yolo_detection"),
            QStringLiteral("bbox"), QStringLiteral("bbox"), {}},
        {QStringLiteral("coco_json"), QStringLiteral("yolo_segmentation"),
            QStringLiteral("instance_polygon"), QStringLiteral("polygon"),
            {QStringLiteral("COCO RLE 不属于该转换路线。")}},
        {QStringLiteral("voc_xml"), QStringLiteral("yolo_detection"),
            QStringLiteral("bbox"), QStringLiteral("bbox"), {}}};
}

const QVector<CapabilityContract>& ProductCapabilityContract::capabilities() const
{
    return capabilities_;
}

const QVector<TrainingBackendContract>& ProductCapabilityContract::trainingBackends() const
{
    return trainingBackends_;
}

const QVector<RuntimeRouteContract>& ProductCapabilityContract::runtimeRoutes() const
{
    return runtimeRoutes_;
}

const QVector<DatasetConversionRouteContract>& ProductCapabilityContract::datasetConversionRoutes() const
{
    return datasetConversionRoutes_;
}

const QVector<PythonEnvironmentProfile>& ProductCapabilityContract::pythonProfiles() const
{
    return pythonProfiles_;
}

bool ProductCapabilityContract::resolveTrainingBackend(const QString& id,
    TrainingBackendContract* result) const
{
    const QString requested = normalized(id);
    for (const TrainingBackendContract& backend : trainingBackends_) {
        if (backend.id == requested) {
            if (result) *result = backend;
            return true;
        }
    }
    return false;
}

bool ProductCapabilityContract::resolveRuntimeRoute(const QString& modelFamily,
    const QString& routeId,
    RuntimeRouteContract* result) const
{
    const QString family = normalized(modelFamily);
    const QString route = normalized(routeId);
    for (const RuntimeRouteContract& candidate : runtimeRoutes_) {
        if (candidate.modelFamily == family && candidate.routeId == route) {
            if (result) *result = candidate;
            return true;
        }
    }
    return false;
}

bool ProductCapabilityContract::resolvePythonProfile(const QString& id,
    PythonEnvironmentProfile* result) const
{
    const QString requested = normalized(id);
    for (const PythonEnvironmentProfile& profile : pythonProfiles_) {
        if (profile.id == requested) {
            if (result) *result = profile;
            return true;
        }
    }
    return false;
}

bool ProductCapabilityContract::supportsDatasetConversion(const QString& sourceFormat,
    const QString& targetFormat) const
{
    const QString source = normalized(sourceFormat);
    const QString target = normalized(targetFormat);
    for (const DatasetConversionRouteContract& route : datasetConversionRoutes_) {
        if (route.sourceFormat == source && route.targetFormat == target) return true;
    }
    return false;
}

QStringList ProductCapabilityContract::validationErrors() const
{
    QStringList errors;
    QSet<QString> capabilityIds;
    QSet<QString> backendIds;
    QSet<QString> pythonIds;
    QSet<QString> runtimeKeys;

    for (const CapabilityContract& capability : capabilities_) {
        if (capability.id.isEmpty() || capabilityIds.contains(capability.id)) {
            errors.append(QStringLiteral("Capability ID 为空或重复：%1").arg(capability.id));
        }
        capabilityIds.insert(capability.id);
    }
    for (const PythonEnvironmentProfile& profile : pythonProfiles_) {
        if (profile.id.isEmpty() || pythonIds.contains(profile.id)) {
            errors.append(QStringLiteral("Python Profile ID 为空或重复：%1").arg(profile.id));
        }
        pythonIds.insert(profile.id);
    }
    for (const RuntimeRouteContract& route : runtimeRoutes_) {
        const QString key = route.modelFamily + QLatin1Char('/') + route.routeId;
        if (route.modelFamily.isEmpty() || route.routeId.isEmpty() || runtimeKeys.contains(key)) {
            errors.append(QStringLiteral("Runtime 路线为空或重复：%1").arg(key));
        }
        runtimeKeys.insert(key);
    }
    for (const TrainingBackendContract& backend : trainingBackends_) {
        if (backend.id.isEmpty() || backendIds.contains(backend.id)) {
            errors.append(QStringLiteral("训练 backend ID 为空或重复：%1").arg(backend.id));
        }
        backendIds.insert(backend.id);
        if (!capabilityIds.contains(backend.capabilityId)) {
            errors.append(QStringLiteral("训练 backend 未关联有效 Capability：%1").arg(backend.id));
        }
        if (!pythonIds.contains(backend.pythonProfileId)) {
            errors.append(QStringLiteral("训练 backend 未关联有效 Python Profile：%1").arg(backend.id));
        }
        for (const QString& routeId : backend.runtimeRouteIds) {
            if (!runtimeKeys.contains(backend.modelFamily + QLatin1Char('/') + routeId)) {
                errors.append(QStringLiteral("训练 backend 未关联有效 Runtime 路线：%1/%2")
                        .arg(backend.id, routeId));
            }
        }
    }
    for (const CapabilityContract& capability : capabilities_) {
        for (const QString& backendId : capability.backendIds) {
            if (!backendIds.contains(backendId)) {
                errors.append(QStringLiteral("Capability 引用了未知 backend：%1/%2")
                        .arg(capability.id, backendId));
            }
        }
    }
    return errors;
}

QJsonObject ProductCapabilityContract::toJson() const
{
    QJsonArray capabilities;
    for (const CapabilityContract& value : capabilities_) capabilities.append(capabilityJson(value));
    QJsonArray backends;
    for (const TrainingBackendContract& value : trainingBackends_) backends.append(trainingBackendJson(value));
    QJsonArray routes;
    for (const RuntimeRouteContract& value : runtimeRoutes_) routes.append(runtimeRouteJson(value));
    QJsonArray conversions;
    for (const DatasetConversionRouteContract& value : datasetConversionRoutes_) {
        conversions.append(datasetConversionJson(value));
    }
    QJsonArray python;
    for (const PythonEnvironmentProfile& value : pythonProfiles_) python.append(pythonProfileJson(value));
    const QStringList errors = validationErrors();
    return {
        {QStringLiteral("contractsValid"), errors.isEmpty()},
        {QStringLiteral("validationErrors"), strings(errors)},
        {QStringLiteral("capabilities"), capabilities},
        {QStringLiteral("trainingBackends"), backends},
        {QStringLiteral("runtimeRoutes"), routes},
        {QStringLiteral("datasetConversionRoutes"), conversions},
        {QStringLiteral("pythonProfiles"), python}};
}

} // namespace aitrain
