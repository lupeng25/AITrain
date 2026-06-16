#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QJsonObject>
#include <QJsonValue>
#include <QObject>

namespace {

class SemanticMaskDatasetAdapter final : public aitrain::IDatasetAdapter {
public:
    QString formatId() const override { return QStringLiteral("semantic_segmentation_mask"); }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        return aitrain::validateSemanticSegmentationMaskDataset(datasetPath, options);
    }
};

class SmpTrainer final : public aitrain::ITrainer {
public:
    QString backendName() const override { return QStringLiteral("SMP semantic segmentation Python backend"); }
};

class SmpValidator final : public aitrain::IValidator {
public:
    QString backendName() const override { return QStringLiteral("SMP ONNX Runtime semantic segmentation evaluator"); }
};

class SmpExporter final : public aitrain::IExporter {
public:
    QStringList supportedFormats() const override { return QStringList() << QStringLiteral("onnx"); }
};

class SmpInferencer final : public aitrain::IInferencer {
public:
    QString backendName() const override { return QStringLiteral("ONNX Runtime semantic segmentation inferencer"); }
};

} // namespace

class SemanticSegmentationPlugin final : public QObject, public aitrain::IModelPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID AITrainModelPluginInterface_iid FILE "../semantic_segmentation.json")
    Q_INTERFACES(aitrain::IModelPlugin)

public:
    aitrain::PluginManifest manifest() const override
    {
        aitrain::PluginManifest manifest;
        manifest.id = QStringLiteral("com.aitrain.plugins.semantic_segmentation");
        manifest.name = QStringLiteral("Semantic Segmentation");
        manifest.version = QStringLiteral("0.1.0");
        manifest.description = QStringLiteral("Dedicated SMP semantic segmentation route for class-id Mask PNG datasets and ONNX Runtime deployment.");
        manifest.taskTypes = QStringList() << QStringLiteral("semantic_segmentation");
        manifest.datasetFormats = QStringList() << QStringLiteral("semantic_segmentation_mask");
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
        manifest.parameterSchema.append(addParam(QStringLiteral("epochs"), QStringLiteral("Epochs"), QStringLiteral("int"), 10));
        manifest.parameterSchema.append(addParam(QStringLiteral("batchSize"), QStringLiteral("Batch Size"), QStringLiteral("int"), 4));
        manifest.parameterSchema.append(addParam(QStringLiteral("imageSize"), QStringLiteral("Image Size"), QStringLiteral("int"), 256));
        manifest.parameterSchema.append(addParam(QStringLiteral("learningRate"), QStringLiteral("Learning Rate"), QStringLiteral("float"), 0.001));
        manifest.parameterSchema.append(addParam(QStringLiteral("ignoreIndex"), QStringLiteral("Ignore Index"), QStringLiteral("int"), 255));
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
    SemanticMaskDatasetAdapter adapter_;
    SmpTrainer trainer_;
    SmpValidator validator_;
    SmpExporter exporter_;
    SmpInferencer inferencer_;
};

#include "SemanticSegmentationPlugin.moc"
