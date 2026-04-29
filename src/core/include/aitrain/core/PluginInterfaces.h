#pragma once

#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QString>
#include <QStringList>

namespace aitrain {

struct PluginManifest {
    QString id;
    QString name;
    QString version;
    QString description;
    QStringList taskTypes;
    QStringList datasetFormats;
    QStringList exportFormats;
    QJsonArray parameterSchema;
    bool requiresGpu = false;

    QJsonObject toJson() const;
    static PluginManifest fromJson(const QJsonObject& object);
};

struct DatasetValidationResult {
    bool ok = true;
    int sampleCount = 0;
    QStringList errors;
    QStringList warnings;

    QJsonObject toJson() const;
};

struct InferenceResult {
    QImage overlay;
    QJsonObject raw;
};

class IDatasetAdapter {
public:
    virtual ~IDatasetAdapter() = default;
    virtual QString formatId() const = 0;
    virtual DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) = 0;
};

class ITrainer {
public:
    virtual ~ITrainer() = default;
    virtual QString backendName() const = 0;
};

class IValidator {
public:
    virtual ~IValidator() = default;
    virtual QString backendName() const = 0;
};

class IExporter {
public:
    virtual ~IExporter() = default;
    virtual QStringList supportedFormats() const = 0;
};

class IInferencer {
public:
    virtual ~IInferencer() = default;
    virtual QString backendName() const = 0;
};

class IModelPlugin {
public:
    virtual ~IModelPlugin() = default;
    virtual PluginManifest manifest() const = 0;
    virtual IDatasetAdapter* datasetAdapter(const QString& formatId) = 0;
    virtual ITrainer* trainer() = 0;
    virtual IValidator* validator() = 0;
    virtual IExporter* exporter() = 0;
    virtual IInferencer* inferencer() = 0;
};

} // namespace aitrain

#define AITrainModelPluginInterface_iid "com.aitrainstudio.IModelPlugin/1.0"
Q_DECLARE_INTERFACE(aitrain::IModelPlugin, AITrainModelPluginInterface_iid)

