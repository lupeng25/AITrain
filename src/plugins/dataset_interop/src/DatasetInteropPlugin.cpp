#include "aitrain/core/PluginInterfaces.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QObject>

#include <utility>

namespace {

class GenericDatasetAdapter final : public aitrain::IDatasetAdapter {
public:
    explicit GenericDatasetAdapter(QString id)
        : id_(std::move(id))
    {
    }

    QString formatId() const override { return id_; }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        Q_UNUSED(options)
        aitrain::DatasetValidationResult result;
        const QDir root(datasetPath);
        if (!root.exists()) {
            result.ok = false;
            result.errors.append(QStringLiteral("Dataset path does not exist."));
            return result;
        }

        if (id_ == QStringLiteral("coco_json")) {
            const QFileInfoList jsonFiles = root.entryInfoList({QStringLiteral("*.json")}, QDir::Files);
            result.sampleCount = jsonFiles.size();
            if (jsonFiles.isEmpty()) {
                result.ok = false;
                result.errors.append(QStringLiteral("No COCO JSON annotation files found."));
            }
            return result;
        }

        if (id_ == QStringLiteral("voc_xml")) {
            const QFileInfoList xmlFiles = root.entryInfoList({QStringLiteral("*.xml")}, QDir::Files);
            result.sampleCount = xmlFiles.size();
            if (xmlFiles.isEmpty()) {
                result.ok = false;
                result.errors.append(QStringLiteral("No VOC XML annotation files found."));
            }
            return result;
        }

        result.warnings.append(QStringLiteral("Generic validation only confirms the dataset directory exists."));
        result.sampleCount = root.entryInfoList(QDir::Files).size();
        return result;
    }

private:
    QString id_;
};

class NullTrainer final : public aitrain::ITrainer {
public:
    QString backendName() const override { return QStringLiteral("dataset-only"); }
};

class NullValidator final : public aitrain::IValidator {
public:
    QString backendName() const override { return QStringLiteral("dataset-only"); }
};

class NullExporter final : public aitrain::IExporter {
public:
    QStringList supportedFormats() const override { return {}; }
};

class NullInferencer final : public aitrain::IInferencer {
public:
    QString backendName() const override { return QStringLiteral("dataset-only"); }
};

} // namespace

class DatasetInteropPlugin final : public QObject, public aitrain::IModelPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID AITrainModelPluginInterface_iid FILE "../dataset_interop.json")
    Q_INTERFACES(aitrain::IModelPlugin)

public:
    DatasetInteropPlugin()
        : coco_(QStringLiteral("coco_json"))
        , voc_(QStringLiteral("voc_xml"))
        , labelme_(QStringLiteral("labelme_json"))
    {
    }

    aitrain::PluginManifest manifest() const override
    {
        aitrain::PluginManifest manifest;
        manifest.id = QStringLiteral("com.aitrain.plugins.dataset_interop");
        manifest.name = QStringLiteral("Dataset Interop");
        manifest.version = QStringLiteral("0.1.0");
        manifest.description = QStringLiteral("Dataset import/export validation scaffold for third-party annotation tools.");
        manifest.taskTypes = QStringList() << QStringLiteral("dataset_conversion");
        manifest.datasetFormats = QStringList() << QStringLiteral("coco_json") << QStringLiteral("voc_xml") << QStringLiteral("labelme_json");
        manifest.requiresGpu = false;
        return manifest;
    }

    aitrain::IDatasetAdapter* datasetAdapter(const QString& formatId) override
    {
        if (formatId == coco_.formatId()) return &coco_;
        if (formatId == voc_.formatId()) return &voc_;
        if (formatId == labelme_.formatId()) return &labelme_;
        return nullptr;
    }

    aitrain::ITrainer* trainer() override { return &trainer_; }
    aitrain::IValidator* validator() override { return &validator_; }
    aitrain::IExporter* exporter() override { return &exporter_; }
    aitrain::IInferencer* inferencer() override { return &inferencer_; }

private:
    GenericDatasetAdapter coco_;
    GenericDatasetAdapter voc_;
    GenericDatasetAdapter labelme_;
    NullTrainer trainer_;
    NullValidator validator_;
    NullExporter exporter_;
    NullInferencer inferencer_;
};

#include "DatasetInteropPlugin.moc"
