#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QJsonObject>
#include <QJsonValue>
#include <QObject>

namespace {

class AnomalyFolderDatasetAdapter final : public aitrain::IDatasetAdapter {
public:
    QString formatId() const override { return QStringLiteral("anomaly_folder"); }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        return aitrain::validateAnomalyFolderDataset(datasetPath, options);
    }
};

class AnomalibTrainer final : public aitrain::ITrainer {
public:
    QString backendName() const override { return QStringLiteral("Anomalib PatchCore/EfficientAD Python backend"); }
};

class AnomalibValidator final : public aitrain::IValidator {
public:
    QString backendName() const override { return QStringLiteral("Anomalib Python evaluator"); }
};

class AnomalibExporter final : public aitrain::IExporter {
public:
    QStringList supportedFormats() const override { return {}; }
};

class AnomalibInferencer final : public aitrain::IInferencer {
public:
    QString backendName() const override { return QStringLiteral("Anomalib Python inferencer"); }
};

} // namespace

class AnomalyDetectionPlugin final : public QObject, public aitrain::IModelPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID AITrainModelPluginInterface_iid FILE "../anomaly_detection.json")
    Q_INTERFACES(aitrain::IModelPlugin)

public:
    aitrain::PluginManifest manifest() const override
    {
        aitrain::PluginManifest manifest;
        manifest.id = QStringLiteral("com.aitrain.plugins.anomaly_detection");
        manifest.name = QStringLiteral("Anomaly Detection");
        manifest.version = QStringLiteral("0.1.0");
        manifest.description = QStringLiteral("Anomalib PatchCore/EfficientAD route for MVTec-compatible anomaly_folder datasets and Python runtime artifacts.");
        manifest.taskTypes = QStringList() << QStringLiteral("anomaly_detection");
        manifest.datasetFormats = QStringList() << QStringLiteral("anomaly_folder");
        manifest.exportFormats = exporter_.supportedFormats();
        manifest.requiresGpu = false;

        auto addParam = [](const QString& key, const QString& label, const QString& type, const QJsonValue& defaultValue) {
            QJsonObject object;
            object.insert(QStringLiteral("key"), key);
            object.insert(QStringLiteral("label"), label);
            object.insert(QStringLiteral("type"), type);
            object.insert(QStringLiteral("default"), defaultValue);
            return object;
        };
        manifest.parameterSchema.append(addParam(QStringLiteral("trainingBackend"), QStringLiteral("Backend"), QStringLiteral("enum"), QStringLiteral("anomalib_patchcore")));
        manifest.parameterSchema.append(addParam(QStringLiteral("modelPreset"), QStringLiteral("Preset"), QStringLiteral("string"), QStringLiteral("anomalib_patchcore_wide_resnet50_2")));
        manifest.parameterSchema.append(addParam(QStringLiteral("device"), QStringLiteral("Device"), QStringLiteral("string"), QStringLiteral("cpu")));
        manifest.parameterSchema.append(addParam(QStringLiteral("workers"), QStringLiteral("Workers"), QStringLiteral("int"), 0));
        manifest.parameterSchema.append(addParam(QStringLiteral("thresholdStrategy"), QStringLiteral("Threshold Strategy"), QStringLiteral("enum"), QStringLiteral("quantile")));
        manifest.parameterSchema.append(addParam(QStringLiteral("quantile"), QStringLiteral("Quantile"), QStringLiteral("float"), 0.995));
        manifest.parameterSchema.append(addParam(QStringLiteral("backbone"), QStringLiteral("PatchCore Backbone"), QStringLiteral("string"), QStringLiteral("wide_resnet50_2")));
        manifest.parameterSchema.append(addParam(QStringLiteral("layers"), QStringLiteral("PatchCore Layers"), QStringLiteral("string"), QStringLiteral("layer2,layer3")));
        manifest.parameterSchema.append(addParam(QStringLiteral("coresetSamplingRatio"), QStringLiteral("Coreset Ratio"), QStringLiteral("float"), 0.1));
        manifest.parameterSchema.append(addParam(QStringLiteral("numNeighbors"), QStringLiteral("Neighbors"), QStringLiteral("int"), 9));
        manifest.parameterSchema.append(addParam(QStringLiteral("modelSize"), QStringLiteral("EfficientAD Size"), QStringLiteral("string"), QStringLiteral("s")));
        manifest.parameterSchema.append(addParam(QStringLiteral("imagenetDir"), QStringLiteral("EfficientAD ImageNet Dir"), QStringLiteral("path"), QString()));
        return manifest;
    }

    aitrain::IDatasetAdapter* datasetAdapter(const QString& formatId) override
    {
        return formatId == adapter_.formatId() ? &adapter_ : nullptr;
    }

    aitrain::ITrainer* trainer() override { return &trainer_; }
    aitrain::IValidator* validator() override { return &validator_; }
    aitrain::IExporter* exporter() override { return &exporter_; }
    aitrain::IInferencer* inferencer() override { return &inferencer_; }

private:
    AnomalyFolderDatasetAdapter adapter_;
    AnomalibTrainer trainer_;
    AnomalibValidator validator_;
    AnomalibExporter exporter_;
    AnomalibInferencer inferencer_;
};

#include "AnomalyDetectionPlugin.moc"
