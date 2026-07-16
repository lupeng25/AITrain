#pragma once

#include "aitrain/v2/DatasetDriverV2.h"

#include "aitrain/core/DatasetValidation.h"
#include "aitrain/core/DatasetValidators.h"

#include <functional>

namespace aitrain::v2 {

class ValidatedDatasetDriverV2 : public DatasetDriverV2 {
public:
    using Validator = std::function<aitrain::DatasetValidationResult(const QString&, const QJsonObject&)>;
    using Splitter = std::function<aitrain::DatasetSplitResult(const QString&, const QString&, const QJsonObject&)>;
    using LayoutDetector = std::function<bool(const QString&)>;

    QString id() const override;
    QString version() const override;
    QStringList supportedFormats() const override;
    bool detect(const QString& sourcePath, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const override;
    bool inspect(const QString& sourcePath, const QString& format, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const override;
    bool validate(const DatasetInspection& inspection, DatasetValidationResult* validation, const DatasetOperationContext& context, QString* error) const override;
    bool planSplit(const DatasetInspection& inspection, const QJsonObject& options, DatasetSplitPlan* plan, const DatasetOperationContext& context, QString* error) const override;
    bool materializeSplit(const DatasetSplitPlan& plan, const QString& stagingPath, const DatasetOperationContext& context, QString* error) const override;
    bool snapshot(const DatasetInspection& inspection, const QString& manifestPath, const DatasetSnapshotOptions& options, DatasetSnapshotResult* result, QString* error) const override;

protected:
    ValidatedDatasetDriverV2(QString driverId, QString format, Validator validator, Splitter splitter, LayoutDetector layoutDetector);

private:
    QString driverId_;
    QString format_;
    Validator validator_;
    Splitter splitter_;
    LayoutDetector layoutDetector_;
};

} // namespace aitrain::v2
