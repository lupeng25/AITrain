#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

enum class RuntimeExecutionAuthority {
    AitrainCpp,
    WorkerManaged,
    OfficialEvidence
};

enum class RuntimeProductState {
    Supported,
    NotImplemented,
    UnsupportedByProduct
};

enum class RuntimeLocalReadiness {
    Available,
    SdkMissing,
    DependencyMissing,
    HardwareUnsupported,
    ExternalEvidenceRequired,
    NotApplicable
};

struct CapabilityContract final {
    QString id;
    QString displayName;
    QStringList taskTypes;
    QStringList datasetFormats;
    QStringList backendIds;
    QStringList limitations;
};

struct PythonEnvironmentProfile final {
    QString id;
    QString dedicatedEnvironmentVariable;
    QString requirementsFile;
    QStringList requiredModules;
    bool requiresPaddleOcrSource = false;
};

struct TrainingBackendContract final {
    QString id;
    QString displayName;
    QString capabilityId;
    QString taskType;
    QString datasetFormat;
    QStringList modelPresets;
    QStringList exportFormats;
    QString legacyRuntimeId;
    QString devicePolicy;
    bool supportsCancel = false;
    QString pythonProfileId;
    QString officialArtifactFormat;
    QString modelFamily;
    QString decoder;
    QStringList runtimeRouteIds;
    QStringList limitations;
};

struct RuntimeRouteContract final {
    QString modelFamily;
    QString routeId;
    RuntimeExecutionAuthority executionAuthority = RuntimeExecutionAuthority::AitrainCpp;
    RuntimeProductState productState = RuntimeProductState::UnsupportedByProduct;
    QStringList acceptedArtifactFormats;
    QStringList limitations;
};

struct DatasetConversionRouteContract final {
    QString sourceFormat;
    QString targetFormat;
    QString sourceSemantics;
    QString targetSemantics;
    QStringList limitations;
};

QString runtimeExecutionAuthorityToString(RuntimeExecutionAuthority authority);
QString runtimeProductStateToString(RuntimeProductState state);
QString runtimeLocalReadinessToString(RuntimeLocalReadiness readiness);

class ProductCapabilityContract final {
public:
    static const ProductCapabilityContract& instance();

    const QVector<CapabilityContract>& capabilities() const;
    const QVector<TrainingBackendContract>& trainingBackends() const;
    const QVector<RuntimeRouteContract>& runtimeRoutes() const;
    const QVector<DatasetConversionRouteContract>& datasetConversionRoutes() const;
    const QVector<PythonEnvironmentProfile>& pythonProfiles() const;

    bool resolveTrainingBackend(const QString& id, TrainingBackendContract* result) const;
    bool resolveRuntimeRoute(const QString& modelFamily,
        const QString& routeId,
        RuntimeRouteContract* result) const;
    bool resolvePythonProfile(const QString& id, PythonEnvironmentProfile* result) const;
    bool supportsDatasetConversion(const QString& sourceFormat, const QString& targetFormat) const;

    QStringList validationErrors() const;
    QJsonObject toJson() const;

private:
    ProductCapabilityContract();

    QVector<CapabilityContract> capabilities_;
    QVector<TrainingBackendContract> trainingBackends_;
    QVector<RuntimeRouteContract> runtimeRoutes_;
    QVector<DatasetConversionRouteContract> datasetConversionRoutes_;
    QVector<PythonEnvironmentProfile> pythonProfiles_;
};

} // namespace aitrain
