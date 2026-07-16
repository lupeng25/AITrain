#pragma once

#include "aitrain/dataset/DatasetSnapshot.h"

#include <QHash>

#include <functional>

namespace aitrain {

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

struct DatasetDriverValidationResult final {
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

class DatasetDriver {
public:
    virtual ~DatasetDriver() = default;

    virtual QString id() const = 0;
    virtual QString version() const = 0;
    virtual QStringList supportedFormats() const = 0;
    virtual bool detect(const QString& sourcePath, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool inspect(const QString& sourcePath, const QString& format, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool validate(const DatasetInspection& inspection, DatasetDriverValidationResult* validation, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool planSplit(const DatasetInspection& inspection, const QJsonObject& options, DatasetSplitPlan* plan, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool materializeSplit(const DatasetSplitPlan& plan, const QString& stagingPath, const DatasetOperationContext& context, QString* error) const = 0;
    virtual bool snapshot(const DatasetInspection& inspection, const QString& manifestPath, const DatasetSnapshotOptions& options, DatasetSnapshotResult* result, QString* error) const = 0;
};

class DatasetDriverRegistry final {
public:
    bool registerDriver(const DatasetDriver* driver, QString* error = nullptr);
    const DatasetDriver* driverForFormat(const QString& format) const;
    QStringList formats() const;

private:
    QHash<QString, const DatasetDriver*> driversByFormat_;
};

class SemanticMaskDatasetDriver final : public DatasetDriver {
public:
    QString id() const override;
    QString version() const override;
    QStringList supportedFormats() const override;
    bool detect(const QString& sourcePath, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const override;
    bool inspect(const QString& sourcePath, const QString& format, DatasetInspection* inspection, const DatasetOperationContext& context, QString* error) const override;
    bool validate(const DatasetInspection& inspection, DatasetDriverValidationResult* validation, const DatasetOperationContext& context, QString* error) const override;
    bool planSplit(const DatasetInspection& inspection, const QJsonObject& options, DatasetSplitPlan* plan, const DatasetOperationContext& context, QString* error) const override;
    bool materializeSplit(const DatasetSplitPlan& plan, const QString& stagingPath, const DatasetOperationContext& context, QString* error) const override;
    bool snapshot(const DatasetInspection& inspection, const QString& manifestPath, const DatasetSnapshotOptions& options, DatasetSnapshotResult* result, QString* error) const override;
};

} // namespace aitrain
