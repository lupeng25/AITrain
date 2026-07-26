#include "aitrain/runtime/ModelPackageRuntimeService.h"

#include <QDir>

namespace aitrain {

ModelPackageRuntimeService::ModelPackageRuntimeService(const ProjectStore* storage, QString artifactStoreRoot)
    : storage_(storage)
    , artifactStoreRoot_(QDir::cleanPath(std::move(artifactStoreRoot)))
{
}

bool ModelPackageRuntimeService::resolve(const ModelPackageId& modelPackageId,
    const QString& runtimeRoute,
    RuntimeModelLocation* location,
    RuntimeCapability* capability,
    QString* error) const
{
    if (!storage_ || !storage_->isOpen() || !modelPackageId.isValid() || !location || runtimeRoute.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("解析  模型运行时需要已打开存储、模型包、运行时路由和输出对象。" );
        return false;
    }
    ModelPackageSnapshot modelPackage;
    if (!storage_->modelPackage(modelPackageId, &modelPackage, error)) return false;
    const RuntimeCapability evaluated = matrix_.query({modelPackage.manifest.modelFamily, runtimeRoute});
    if (capability) *capability = evaluated;
    if (evaluated.executionAuthority != RuntimeExecutionAuthority::AitrainCpp
        || evaluated.productState != RuntimeProductState::Supported
        || evaluated.localReadiness != RuntimeLocalReadiness::Available
        || evaluated.status != RuntimeCapabilityStatus::Supported) {
        if (error) *error = QStringLiteral("模型包不能进入目标运行时：%1").arg(evaluated.message);
        return false;
    }
    RuntimeModelLocation resolved;
    resolved.manifest = modelPackage.manifest;
    resolved.artifactDirectory = QDir(artifactStoreRoot_).filePath(
        QStringLiteral("committed/%1").arg(modelPackage.sourceArtifactId.toString()));
    const RuntimeOperationResult admission = validateRuntimeModel(resolved, runtimeRoute);
    if (admission.status != RuntimeStatus::Available) {
        if (error) *error = admission.message;
        return false;
    }
    *location = resolved;
    return true;
}

} // namespace aitrain
