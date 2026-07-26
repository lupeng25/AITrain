#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/product/ProductCapabilityContract.h"

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
        {QStringLiteral("supportsCancel"), supportsCancel},
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
    const ProductCapabilityContract& contract = ProductCapabilityContract::instance();
    for (const TrainingBackendContract& source : contract.trainingBackends()) {
        BackendDescriptor backend;
        backend.id = source.id;
        backend.displayName = source.displayName;
        backend.taskTypes = QStringList{source.taskType};
        backend.datasetFormats = QStringList{source.datasetFormat};
        backend.modelPresets = source.modelPresets;
        backend.exportFormats = source.exportFormats;
        backend.runtime = source.legacyRuntimeId;
        backend.devicePolicy = source.devicePolicy;
        backend.supportsCancel = source.supportsCancel;
        backend.limitations = source.limitations;
        backends_.append(backend);
    }
    for (const CapabilityContract& source : contract.capabilities()) {
        CapabilityDescriptor capability;
        capability.id = source.id;
        capability.displayName = source.displayName;
        capability.taskTypes = source.taskTypes;
        capability.datasetFormats = source.datasetFormats;
        capability.backendIds = source.backendIds;
        capability.limitations = source.limitations;
        capabilities_.append(capability);
    }
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
