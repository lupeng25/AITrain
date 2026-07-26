#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

// 编译期内置能力的唯一事实来源。GUI、Worker 和环境检查均必须通过此处
// 查询任务、数据集、后端和部署边界，不能再各自维护字符串矩阵。
struct BackendDescriptor {
    QString id;
    QString displayName;
    QStringList taskTypes;
    QStringList datasetFormats;
    QStringList modelPresets;
    QStringList exportFormats;
    QString runtime;
    QString devicePolicy;
    bool supportsCancel = false;
    QStringList limitations;

    QJsonObject toJson() const;
};

struct CapabilityDescriptor {
    QString id;
    QString displayName;
    QStringList taskTypes;
    QStringList datasetFormats;
    QStringList backendIds;
    QStringList limitations;

    QJsonObject toJson() const;
};

class BuiltinCapabilityRegistry final {
public:
    static const BuiltinCapabilityRegistry& instance();

    QVector<CapabilityDescriptor> capabilities() const;
    QVector<BackendDescriptor> backends() const;
    CapabilityDescriptor capability(const QString& id) const;
    BackendDescriptor backend(const QString& id) const;
    QStringList taskTypesForCapability(const QString& capabilityId) const;
    QStringList datasetFormatsForTask(const QString& taskType) const;
    QStringList backendsForTask(const QString& taskType, const QString& datasetFormat = QString()) const;
    bool supports(const QString& capabilityId,
        const QString& taskType,
        const QString& datasetFormat,
        const QString& backendId,
        QString* error = nullptr) const;
    QJsonObject toJson() const;

private:
    BuiltinCapabilityRegistry();

    QVector<CapabilityDescriptor> capabilities_;
    QVector<BackendDescriptor> backends_;
};

} // namespace aitrain
