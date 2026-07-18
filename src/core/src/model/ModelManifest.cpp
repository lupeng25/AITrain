#include "aitrain/model/ModelManifest.h"
#include "aitrain/domain/ArtifactMemberPath.h"

#include <QJsonArray>
#include <QDir>
#include <QRegularExpression>

namespace aitrain {
namespace {

bool validSha256(const QString& value)
{
    static const QRegularExpression expression(QStringLiteral("^[0-9a-f]{64}$"));
    return expression.match(value).hasMatch();
}

bool validateTensor(const TensorContract& tensor, const QString& role, QString* error)
{
    if (tensor.name.trimmed().isEmpty() || tensor.layout.trimmed().isEmpty() || tensor.shape.isEmpty()) {
        if (error) *error = QStringLiteral("Model Manifest %1 tensor 缺少 name、layout 或 shape。").arg(role);
        return false;
    }
    for (qint64 dimension : tensor.shape) {
        if (dimension == 0 || dimension < -1) {
            if (error) *error = QStringLiteral("Model Manifest tensor shape 仅允许正数或 -1 动态维度。");
            return false;
        }
    }
    return true;
}

QJsonArray encodeTensors(const QVector<TensorContract>& tensors)
{
    QJsonArray values;
    for (const TensorContract& tensor : tensors) {
        QJsonArray shape;
        for (qint64 dimension : tensor.shape) shape.append(dimension);
        values.append(QJsonObject{{QStringLiteral("name"), tensor.name}, {QStringLiteral("layout"), tensor.layout}, {QStringLiteral("shape"), shape}});
    }
    return values;
}

bool decodeTensors(const QJsonValue& value, QVector<TensorContract>* tensors, const QString& role, QString* error)
{
    if (!value.isArray()) {
        if (error) *error = QStringLiteral("Model Manifest %1 必须为数组。").arg(role);
        return false;
    }
    for (const QJsonValue& item : value.toArray()) {
        const QJsonObject object = item.toObject();
        if (!item.isObject() || !object.value(QStringLiteral("shape")).isArray()) {
            if (error) *error = QStringLiteral("Model Manifest %1 tensor 格式无效。").arg(role);
            return false;
        }
        TensorContract tensor;
        tensor.name = object.value(QStringLiteral("name")).toString();
        tensor.layout = object.value(QStringLiteral("layout")).toString();
        for (const QJsonValue& dimension : object.value(QStringLiteral("shape")).toArray()) {
            if (!dimension.isDouble()) {
                if (error) *error = QStringLiteral("Model Manifest tensor shape 必须为整数。");
                return false;
            }
            const qint64 parsed = static_cast<qint64>(dimension.toDouble());
            if (static_cast<double>(parsed) != dimension.toDouble()) {
                if (error) *error = QStringLiteral("Model Manifest tensor shape 必须为整数。");
                return false;
            }
            tensor.shape.append(parsed);
        }
        if (!validateTensor(tensor, role, error)) return false;
        tensors->append(tensor);
    }
    return !tensors->isEmpty();
}

} // namespace

bool validateModelManifest(const ModelManifest& manifest, QString* error)
{
    QString normalizedEntryPath;
    const QString artifactFormat = manifest.artifactFormat.trimmed().toLower();
    const bool isOnnx = artifactFormat == QStringLiteral("onnx");
    const bool isNcnn = artifactFormat == QStringLiteral("ncnn");
    const bool isTensorRtEngine = artifactFormat == QStringLiteral("tensorrt_engine");
    const bool isAnomalibBundle = artifactFormat == QStringLiteral("anomalib_bundle");
    const bool isPaddleOcrBundle = artifactFormat == QStringLiteral("paddleocr_inference_bundle");
    if (!manifest.modelPackageId.isValid() || !manifest.sourceTaskId.isValid() || !manifest.sourceSnapshotId.isValid()
        || manifest.modelFamily.trimmed().isEmpty() || manifest.taskType.trimmed().isEmpty() || manifest.sourceBackend.trimmed().isEmpty()
        || !validSha256(manifest.sourceArtifactSha256)
        || !normalizeArtifactMemberPath(manifest.artifactEntryPath, &normalizedEntryPath, error)
        || normalizedEntryPath != manifest.artifactEntryPath
        || (!isOnnx && !isNcnn && !isTensorRtEngine && !isAnomalibBundle && !isPaddleOcrBundle)
        || manifest.preprocessing.isEmpty() || manifest.postprocessing.isEmpty() || manifest.decoder.trimmed().isEmpty()
        || manifest.classNames.isEmpty() || manifest.exporterVersion.trimmed().isEmpty()
        || manifest.runtimeRoutes.isEmpty() || !manifest.verified) {
        if (error) *error = QStringLiteral("Model Manifest  缺少必需字段、未验证、来源哈希或产物入口路径无效。");
        return false;
    }
    if (isOnnx && (manifest.inputs.isEmpty() || manifest.outputs.isEmpty() || manifest.opset < 1)) {
        if (error) *error = QStringLiteral("ONNX Model Manifest 必须携带 input/output tensor 与有效 opset。");
        return false;
    }
    if ((isNcnn || isTensorRtEngine) && (manifest.inputs.isEmpty() || manifest.outputs.isEmpty() || manifest.opset != 0)) {
        if (error) *error = QStringLiteral("NCNN/TensorRT Model Manifest 必须携带 input/output tensor，且不得伪造 ONNX opset。");
        return false;
    }
    if (isNcnn && (manifest.runtimeRoutes.size() != 1
        || manifest.runtimeRoutes.constFirst() != QStringLiteral("aitrain_ncnn"))) {
        if (error) *error = QStringLiteral("NCNN Model Manifest 的 runtime route 必须且只能是 aitrain_ncnn。");
        return false;
    }
    if (isTensorRtEngine && (manifest.runtimeRoutes.size() != 1
        || manifest.runtimeRoutes.constFirst() != QStringLiteral("aitrain_tensorrt"))) {
        if (error) *error = QStringLiteral("TensorRT engine Manifest 的 runtime route 必须且只能是 aitrain_tensorrt。");
        return false;
    }
    if (isAnomalibBundle && (manifest.opset != 0 || manifest.runtimeRoutes.size() != 1
        || manifest.runtimeRoutes.constFirst() != QStringLiteral("anomalib_python"))) {
        if (error) *error = QStringLiteral("Anomalib bundle 不得声明 ONNX opset，且 runtime route 必须且只能是 anomalib_python。");
        return false;
    }
    if (isPaddleOcrBundle && (manifest.opset != 0 || manifest.runtimeRoutes.size() != 1
        || manifest.runtimeRoutes.constFirst() != QStringLiteral("paddleocr_official"))) {
        if (error) *error = QStringLiteral("PaddleOCR inference bundle 不得声明 ONNX opset，且 runtime route 必须且只能是 paddleocr_official。");
        return false;
    }
    for (const TensorContract& tensor : manifest.inputs) if (!validateTensor(tensor, QStringLiteral("input"), error)) return false;
    for (const TensorContract& tensor : manifest.outputs) if (!validateTensor(tensor, QStringLiteral("output"), error)) return false;
    for (const QString& route : manifest.runtimeRoutes) {
        if (route.trimmed().isEmpty()) {
            if (error) *error = QStringLiteral("Model Manifest  runtime route 不能为空。");
            return false;
        }
    }
    return true;
}

QJsonObject encodeModelManifest(const ModelManifest& manifest, QString* error)
{
    if (!validateModelManifest(manifest, error)) return {};
    return QJsonObject{{QStringLiteral("schemaVersion"), kModelManifestSchemaVersion},
        {QStringLiteral("modelPackageId"), manifest.modelPackageId.toString()}, {QStringLiteral("modelFamily"), manifest.modelFamily},
        {QStringLiteral("taskType"), manifest.taskType}, {QStringLiteral("sourceBackend"), manifest.sourceBackend},
        {QStringLiteral("sourceTaskId"), manifest.sourceTaskId.toString()}, {QStringLiteral("sourceSnapshotId"), manifest.sourceSnapshotId.toString()},
        {QStringLiteral("sourceArtifactSha256"), manifest.sourceArtifactSha256}, {QStringLiteral("artifactEntryPath"), manifest.artifactEntryPath},
        {QStringLiteral("artifactFormat"), manifest.artifactFormat}, {QStringLiteral("inputs"), encodeTensors(manifest.inputs)},
        {QStringLiteral("outputs"), encodeTensors(manifest.outputs)}, {QStringLiteral("preprocessing"), manifest.preprocessing},
        {QStringLiteral("postprocessing"), manifest.postprocessing}, {QStringLiteral("decoder"), manifest.decoder},
        {QStringLiteral("classNames"), QJsonArray::fromStringList(manifest.classNames)}, {QStringLiteral("opset"), manifest.opset},
        {QStringLiteral("exporterVersion"), manifest.exporterVersion}, {QStringLiteral("runtimeRoutes"), QJsonArray::fromStringList(manifest.runtimeRoutes)},
        {QStringLiteral("verified"), manifest.verified}, {QStringLiteral("limitations"), QJsonArray::fromStringList(manifest.limitations)}};
}

bool decodeModelManifest(const QJsonObject& object, ModelManifest* manifest, QString* error)
{
    if (object.value(QStringLiteral("schemaVersion")).toInt(-1) != kModelManifestSchemaVersion) {
        if (error) *error = QStringLiteral("不支持的 Model Manifest schemaVersion。");
        return false;
    }
    ModelManifest parsed;
    if (!ModelPackageId::parse(object.value(QStringLiteral("modelPackageId")).toString(), &parsed.modelPackageId, error)
        || !TaskId::parse(object.value(QStringLiteral("sourceTaskId")).toString(), &parsed.sourceTaskId, error)
        || !SnapshotId::parse(object.value(QStringLiteral("sourceSnapshotId")).toString(), &parsed.sourceSnapshotId, error)) return false;
    parsed.modelFamily = object.value(QStringLiteral("modelFamily")).toString();
    parsed.taskType = object.value(QStringLiteral("taskType")).toString();
    parsed.sourceBackend = object.value(QStringLiteral("sourceBackend")).toString();
    parsed.sourceArtifactSha256 = object.value(QStringLiteral("sourceArtifactSha256")).toString();
    parsed.artifactEntryPath = object.value(QStringLiteral("artifactEntryPath")).toString();
    parsed.artifactFormat = object.value(QStringLiteral("artifactFormat")).toString();
    const bool tensorContractRequired = parsed.artifactFormat == QStringLiteral("onnx")
        || parsed.artifactFormat == QStringLiteral("ncnn")
        || parsed.artifactFormat == QStringLiteral("tensorrt_engine");
    const bool inputsDecoded = object.value(QStringLiteral("inputs")).isArray()
        && (!object.value(QStringLiteral("inputs")).toArray().isEmpty()
            ? decodeTensors(object.value(QStringLiteral("inputs")), &parsed.inputs, QStringLiteral("inputs"), error)
            : !tensorContractRequired);
    const bool outputsDecoded = object.value(QStringLiteral("outputs")).isArray()
        && (!object.value(QStringLiteral("outputs")).toArray().isEmpty()
            ? decodeTensors(object.value(QStringLiteral("outputs")), &parsed.outputs, QStringLiteral("outputs"), error)
            : !tensorContractRequired);
    if (!inputsDecoded || !outputsDecoded
        || !object.value(QStringLiteral("preprocessing")).isObject() || !object.value(QStringLiteral("postprocessing")).isObject()
        || !object.value(QStringLiteral("classNames")).isArray() || !object.value(QStringLiteral("runtimeRoutes")).isArray()
        || !object.value(QStringLiteral("verified")).isBool()) {
        if (error && error->isEmpty()) *error = QStringLiteral("Model Manifest  JSON 字段类型无效。");
        return false;
    }
    parsed.preprocessing = object.value(QStringLiteral("preprocessing")).toObject();
    parsed.postprocessing = object.value(QStringLiteral("postprocessing")).toObject();
    parsed.decoder = object.value(QStringLiteral("decoder")).toString();
    for (const QJsonValue& value : object.value(QStringLiteral("classNames")).toArray()) parsed.classNames.append(value.toString());
    parsed.opset = object.value(QStringLiteral("opset")).toInt();
    parsed.exporterVersion = object.value(QStringLiteral("exporterVersion")).toString();
    for (const QJsonValue& value : object.value(QStringLiteral("runtimeRoutes")).toArray()) parsed.runtimeRoutes.append(value.toString());
    parsed.verified = object.value(QStringLiteral("verified")).toBool();
    for (const QJsonValue& value : object.value(QStringLiteral("limitations")).toArray()) parsed.limitations.append(value.toString());
    if (!validateModelManifest(parsed, error)) return false;
    if (manifest) *manifest = parsed;
    return true;
}

bool decodeModelManifestImportDraft(const QJsonObject& object, ModelManifest* manifest, QString* error)
{
    if (!object.contains(QStringLiteral("modelPackageId")) || !object.contains(QStringLiteral("sourceSnapshotId"))) {
        if (error) *error = QStringLiteral("导入 Manifest 草稿必须明确 modelPackageId 和 sourceSnapshotId。");
        return false;
    }
    QJsonObject normalized = object;
    // 这两个字段由导入过程建立证据链，不能由外部文件决定。
    normalized.insert(QStringLiteral("sourceTaskId"), TaskId::create().toString());
    normalized.insert(QStringLiteral("sourceArtifactSha256"), QString(64, QLatin1Char('0')));
    return decodeModelManifest(normalized, manifest, error);
}

ModelManifestStatus modelManifestStatus(const QJsonObject* object, QString* error)
{
    if (!object || object->isEmpty()) return ModelManifestStatus::Unclassified;
    ModelManifest manifest;
    return decodeModelManifest(*object, &manifest, error) ? ModelManifestStatus::Valid : ModelManifestStatus::Invalid;
}

bool canUseModelManifestForRuntime(const QJsonObject* object, const QString& runtimeRoute, QString* error)
{
    ModelManifest manifest;
    if (modelManifestStatus(object, error) != ModelManifestStatus::Valid || !decodeModelManifest(*object, &manifest, error)) return false;
    if (!manifest.runtimeRoutes.contains(runtimeRoute)) {
        if (error) *error = QStringLiteral("Model Manifest 未声明目标 runtime route：%1").arg(runtimeRoute);
        return false;
    }
    return true;
}

} // namespace aitrain
