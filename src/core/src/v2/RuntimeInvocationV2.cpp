#include "aitrain/v2/RuntimeInvocationV2.h"

#include <QDir>

namespace aitrain::v2 {
namespace {

bool validateInvocation(const RuntimeInvocationV2& invocation, QString* error)
{
    if (invocation.runtimeRoute.trimmed().isEmpty()
        || invocation.imagePath.trimmed().isEmpty()
        || invocation.outputPath.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("Runtime Invocation V2 缺少 runtimeRoute、imagePath 或 outputPath。");
        return false;
    }
    const RuntimeOperationResultV2 admission = validateRuntimeModelV2(invocation.model, invocation.runtimeRoute);
    if (admission.status != RuntimeStatusV2::Available) {
        if (error) *error = admission.message;
        return false;
    }
    return true;
}

} // namespace

QJsonObject encodeRuntimeInvocationV2(const RuntimeInvocationV2& invocation, QString* error)
{
    if (!validateInvocation(invocation, error)) return {};
    const QJsonObject manifest = encodeModelManifestV2(invocation.model.manifest, error);
    if (manifest.isEmpty()) return {};
    return QJsonObject{
        {QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("runtimeRoute"), invocation.runtimeRoute},
        {QStringLiteral("model"), QJsonObject{
             {QStringLiteral("manifest"), manifest},
             {QStringLiteral("artifactDirectory"), QDir::cleanPath(invocation.model.artifactDirectory)}}},
        {QStringLiteral("imagePath"), invocation.imagePath},
        {QStringLiteral("outputPath"), invocation.outputPath},
        {QStringLiteral("options"), invocation.options}};
}

bool decodeRuntimeInvocationV2(const QJsonObject& object, RuntimeInvocationV2* invocation, QString* error)
{
    if (!invocation || object.value(QStringLiteral("schemaVersion")).toInt(-1) != 2
        || !object.value(QStringLiteral("model")).isObject() || !object.value(QStringLiteral("options")).isObject()) {
        if (error) *error = QStringLiteral("Runtime Invocation V2 JSON 格式无效。");
        return false;
    }
    const QJsonObject model = object.value(QStringLiteral("model")).toObject();
    if (!model.value(QStringLiteral("manifest")).isObject()) {
        if (error) *error = QStringLiteral("Runtime Invocation V2 缺少 Model Manifest。");
        return false;
    }
    RuntimeInvocationV2 parsed;
    if (!decodeModelManifestV2(model.value(QStringLiteral("manifest")).toObject(), &parsed.model.manifest, error)) return false;
    parsed.model.artifactDirectory = QDir::cleanPath(model.value(QStringLiteral("artifactDirectory")).toString());
    parsed.runtimeRoute = object.value(QStringLiteral("runtimeRoute")).toString();
    parsed.imagePath = object.value(QStringLiteral("imagePath")).toString();
    parsed.outputPath = object.value(QStringLiteral("outputPath")).toString();
    parsed.options = object.value(QStringLiteral("options")).toObject();
    if (!validateInvocation(parsed, error)) return false;
    *invocation = parsed;
    return true;
}

} // namespace aitrain::v2
