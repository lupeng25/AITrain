#include "aitrain/v2/ModelPackageRuntimeServiceV2.h"

#include <QDir>

namespace aitrain::v2 {

ModelPackageRuntimeServiceV2::ModelPackageRuntimeServiceV2(const StorageV2* storage, QString artifactStoreRoot)
    : storage_(storage)
    , artifactStoreRoot_(QDir::cleanPath(std::move(artifactStoreRoot)))
{
}

bool ModelPackageRuntimeServiceV2::resolve(const ModelPackageId& modelPackageId,
    const QString& runtimeRoute,
    RuntimeModelLocationV2* location,
    RuntimeCapabilityV2* capability,
    QString* error) const
{
    if (!storage_ || !storage_->isOpen() || !modelPackageId.isValid() || !location || runtimeRoute.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("解析 V2 模型运行时需要已打开存储、模型包、运行时路由和输出对象。" );
        return false;
    }
    ModelPackageSnapshotV2 modelPackage;
    if (!storage_->modelPackage(modelPackageId, &modelPackage, error)) return false;
    const RuntimeCapabilityV2 evaluated = matrix_.query({modelPackage.manifest.modelFamily, runtimeRoute});
    if (capability) *capability = evaluated;
    if (evaluated.status != RuntimeCapabilityStatusV2::Supported) {
        if (error) *error = QStringLiteral("模型包不能进入目标运行时：%1").arg(evaluated.message);
        return false;
    }
    RuntimeModelLocationV2 resolved;
    resolved.manifest = modelPackage.manifest;
    resolved.artifactDirectory = QDir(artifactStoreRoot_).filePath(
        QStringLiteral("artifacts/%1").arg(modelPackage.sourceArtifactId.toString()));
    const RuntimeOperationResultV2 admission = validateRuntimeModelV2(resolved, runtimeRoute);
    if (admission.status != RuntimeStatusV2::Available) {
        if (error) *error = admission.message;
        return false;
    }
    *location = resolved;
    return true;
}

} // namespace aitrain::v2
