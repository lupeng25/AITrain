#pragma once

#include <QJsonObject>

namespace aitrain {

struct ExecutionRequest final {
    QString capabilityId;
    QString taskType;
    QString datasetFormat;
    QString trainingBackend;
    QString evaluationBackend;
    QString exportFormat;
    QString runtimeRoute;
};

struct ExecutionPlan final {
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

class CapabilityPlanner final {
public:
    bool plan(const ExecutionRequest& request, ExecutionPlan* result, QString* error = nullptr) const;
    bool verify(const ExecutionRequest& request, const QString& expectedSummaryHash, ExecutionPlan* result, QString* error = nullptr) const;
};

} // namespace aitrain
