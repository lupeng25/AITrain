#include "aitrain/v2/CapabilityPlannerV2.h"

#include "aitrain/core/CapabilityRegistry.h"

#include <QCryptographicHash>
#include <QJsonDocument>

namespace aitrain::v2 {
namespace {

QString normalized(const QString& value)
{
    return value.trimmed().toLower();
}

QString planHash(const QJsonObject& plan)
{
    return QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(plan).toJson(QJsonDocument::Compact), QCryptographicHash::Sha256).toHex());
}

} // namespace

QJsonObject ExecutionPlanV2::toJson() const
{
    return QJsonObject{{QStringLiteral("capabilityId"), capabilityId},
        {QStringLiteral("taskType"), taskType},
        {QStringLiteral("datasetFormat"), datasetFormat},
        {QStringLiteral("trainingBackend"), trainingBackend},
        {QStringLiteral("evaluationBackend"), evaluationBackend},
        {QStringLiteral("exportFormat"), exportFormat},
        {QStringLiteral("runtimeRoute"), runtimeRoute},
        {QStringLiteral("summaryHash"), summaryHash}};
}

bool CapabilityPlannerV2::plan(const ExecutionRequestV2& request, ExecutionPlanV2* result, QString* error) const
{
    if (!result) {
        if (error) {
            *error = QStringLiteral("能力规划需要输出对象。");
        }
        return false;
    }
    ExecutionPlanV2 plan;
    plan.capabilityId = normalized(request.capabilityId);
    plan.taskType = normalized(request.taskType);
    plan.datasetFormat = normalized(request.datasetFormat);
    plan.trainingBackend = normalized(request.trainingBackend);
    plan.evaluationBackend = normalized(request.evaluationBackend);
    plan.exportFormat = normalized(request.exportFormat);
    plan.runtimeRoute = normalized(request.runtimeRoute);
    if (plan.capabilityId.isEmpty() || plan.taskType.isEmpty() || plan.datasetFormat.isEmpty()
        || plan.trainingBackend.isEmpty() || plan.evaluationBackend.isEmpty() || plan.runtimeRoute.isEmpty()) {
        if (error) {
            *error = QStringLiteral("能力规划需要完整的 capability、任务、数据集、训练/评估后端和运行时路由。");
        }
        return false;
    }

    const BuiltinCapabilityRegistry& registry = BuiltinCapabilityRegistry::instance();
    if (!registry.supports(plan.capabilityId, plan.taskType, plan.datasetFormat, plan.trainingBackend, error)) {
        return false;
    }
    if (!registry.supports(plan.capabilityId, plan.taskType, plan.datasetFormat, plan.evaluationBackend, error)) {
        return false;
    }
    const BackendDescriptor training = registry.backend(plan.trainingBackend);
    const BackendDescriptor evaluation = registry.backend(plan.evaluationBackend);
    if (training.runtime != plan.runtimeRoute || evaluation.runtime != plan.runtimeRoute) {
        if (error) {
            *error = QStringLiteral("请求的运行时路由与训练或评估后端不一致。");
        }
        return false;
    }
    if (!plan.exportFormat.isEmpty() && !training.exportFormats.contains(plan.exportFormat)) {
        if (error) {
            *error = QStringLiteral("训练后端不支持请求的导出格式：%1").arg(plan.exportFormat);
        }
        return false;
    }
    plan.summaryHash = planHash(QJsonObject{{QStringLiteral("capabilityId"), plan.capabilityId},
        {QStringLiteral("taskType"), plan.taskType},
        {QStringLiteral("datasetFormat"), plan.datasetFormat},
        {QStringLiteral("trainingBackend"), plan.trainingBackend},
        {QStringLiteral("evaluationBackend"), plan.evaluationBackend},
        {QStringLiteral("exportFormat"), plan.exportFormat},
        {QStringLiteral("runtimeRoute"), plan.runtimeRoute}});
    *result = plan;
    return true;
}

bool CapabilityPlannerV2::verify(const ExecutionRequestV2& request, const QString& expectedSummaryHash, ExecutionPlanV2* result, QString* error) const
{
    ExecutionPlanV2 plan;
    if (!this->plan(request, &plan, error)) {
        return false;
    }
    if (expectedSummaryHash.trimmed().isEmpty() || plan.summaryHash != expectedSummaryHash.trimmed().toLower()) {
        if (error) {
            *error = QStringLiteral("ExecutionPlan 摘要不匹配，Worker 必须拒绝执行。");
        }
        return false;
    }
    if (result) {
        *result = plan;
    }
    return true;
}

} // namespace aitrain::v2
