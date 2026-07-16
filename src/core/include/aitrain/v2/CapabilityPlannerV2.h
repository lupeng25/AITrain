#pragma once

#include <QJsonObject>

namespace aitrain::v2 {

struct ExecutionRequestV2 final {
    QString capabilityId;
    QString taskType;
    QString datasetFormat;
    QString trainingBackend;
    QString evaluationBackend;
    QString exportFormat;
    QString runtimeRoute;
};

struct ExecutionPlanV2 final {
    QString capabilityId;
    QString taskType;
    QString datasetFormat;
    QString trainingBackend;
    QString evaluationBackend;
    QString exportFormat;
    QString runtimeRoute;
    QString summaryHash;

    QJsonObject toJson() const;
};

class CapabilityPlannerV2 final {
public:
    bool plan(const ExecutionRequestV2& request, ExecutionPlanV2* result, QString* error = nullptr) const;
    bool verify(const ExecutionRequestV2& request, const QString& expectedSummaryHash, ExecutionPlanV2* result, QString* error = nullptr) const;
};

} // namespace aitrain::v2
