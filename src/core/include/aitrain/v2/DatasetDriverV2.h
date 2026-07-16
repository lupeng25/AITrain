#pragma once

#include "aitrain/v2/DatasetSnapshotV2.h"

#include <QHash>

#include <functional>

namespace aitrain::v2 {

struct DatasetOperationContext final {
    std::function<bool()> isCancellationRequested;
    std::function<void(int percent, const QString& message)> reportProgress;
    std::function<void(const QString& code, const QString& message)> reportDiagnostic;
};

struct DatasetInspection final {
    QString sourcePath;
    QString format;
    qsizetype sampleCount = 0;
    QJsonObject details;
};

struct DatasetValidationResult final {
    bool valid = false;
    QJsonArray issues;
    QJsonObject details;
};

struct DatasetSplitPlan final {
    QString format;
    QString sourceRoot;
    QString planHash;
    QJsonObject manifest;
};

class DatasetDriverV2 {
public:
    virtual ~DatasetDriverV2() = default;

    virtual QString id() const = 0;
    virtual QString version() const = 0;
    virtual QStringList supportedFormats() const = 0;
    virtual bool detect(const QString& sourcePath, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool inspect(const QString& sourcePath, const QString& format, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool validate(const DatasetInspection& inspection, DatasetValidationResult* validation, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool planSplit(const DatasetInspection& inspection, const QJsonObject& options, DatasetSplitPlan* plan, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool materializeSplit(const DatasetSplitPlan& plan, const QString& stagingPath, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool snapshot(const DatasetInspection& inspection, const QString& manifestPath, const DatasetSnapshotOptions& options, DatasetSnapshotResult* result, QString* error) const = 0;
};

class DatasetDriverRegistryV2 final {
public:
    bool registerDriver(const DatasetDriverV2* driver, QString* error = nullptr);
    const DatasetDriverV2* driverForFormat(const QString& format) const;
    QStringList formats() const;

private:
    QHash<QString, const DatasetDriverV2*> driversByFormat_;
};

class SemanticMaskDatasetDriverV2 final : public DatasetDriverV2 {
public:
    QString id() const override;
    QString version() const override;
    QStringList supportedFormats() const override;
    bool detect(const QString& sourcePath, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const override;
    bool inspect(const QString& sourcePath, const QString& format, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const override;
    bool validate(const DatasetInspection& inspection, DatasetValidationResult* validation, const DatasetOperationContext& context, QString* error) const override;
    bool planSplit(const DatasetInspection& inspection, const QJsonObject& options, DatasetSplitPlan* plan, const DatasetOperationContext& context, QString* error) const override;
    bool materializeSplit(const DatasetSplitPlan& plan, const QString& stagingPath, const DatasetOperationContext& context, QString* error) const override;
    bool snapshot(const DatasetInspection& inspection, const QString& manifestPath, const DatasetSnapshotOptions& options, DatasetSnapshotResult* result, QString* error) const override;
};

} // namespace aitrain::v2
