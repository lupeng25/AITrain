#include "aitrain/core/PluginInterfaces.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QObject>
#include <QXmlStreamReader>

#include <utility>

namespace {

QFileInfoList datasetFiles(const QFileInfo& input, const QStringList& nameFilters)
{
    QFileInfoList files;
    if (input.isFile()) {
        const QString fileName = input.fileName();
        for (const QString& filter : nameFilters) {
            if (QDir::match(filter, fileName)) {
                files.append(input);
                break;
            }
        }
        return files;
    }
    const QDir root(input.absoluteFilePath());
    return root.entryInfoList(nameFilters, QDir::Files);
}

bool validateCocoFile(const QFileInfo& fileInfo, aitrain::DatasetValidationResult* result)
{
    QFile file(fileInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        result->errors.append(QStringLiteral("%1: cannot read COCO JSON file.").arg(fileInfo.fileName()));
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        result->errors.append(QStringLiteral("%1: COCO annotation must be a JSON object.").arg(fileInfo.fileName()));
        return false;
    }

    const QJsonObject root = document.object();
    const QJsonValue images = root.value(QStringLiteral("images"));
    const QJsonValue annotations = root.value(QStringLiteral("annotations"));
    if (!images.isArray()) {
        result->errors.append(QStringLiteral("%1: COCO annotation is missing an images array.").arg(fileInfo.fileName()));
        return false;
    }
    if (!annotations.isArray()) {
        result->errors.append(QStringLiteral("%1: COCO annotation is missing an annotations array.").arg(fileInfo.fileName()));
        return false;
    }
    const QJsonArray imageArray = images.toArray();
    if (imageArray.isEmpty()) {
        result->warnings.append(QStringLiteral("%1: COCO images array is empty.").arg(fileInfo.fileName()));
    } else {
        const QString firstFileName = imageArray.first().toObject().value(QStringLiteral("file_name")).toString();
        if (!firstFileName.isEmpty()) {
            result->previewSamples.append(firstFileName);
        }
    }
    result->sampleCount += imageArray.size();
    return true;
}

bool validateVocFile(const QFileInfo& fileInfo, aitrain::DatasetValidationResult* result)
{
    QFile file(fileInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        result->errors.append(QStringLiteral("%1: cannot read VOC XML file.").arg(fileInfo.fileName()));
        return false;
    }

    QXmlStreamReader reader(&file);
    QString rootName;
    QString filename;
    bool sawObject = false;
    while (!reader.atEnd()) {
        reader.readNext();
        if (!reader.isStartElement()) {
            continue;
        }
        const QString elementName = reader.name().toString();
        if (rootName.isEmpty()) {
            rootName = elementName;
        }
        if (elementName == QStringLiteral("filename")) {
            filename = reader.readElementText().trimmed();
        } else if (elementName == QStringLiteral("object")) {
            sawObject = true;
        }
    }
    if (reader.hasError()) {
        result->errors.append(QStringLiteral("%1: invalid VOC XML: %2").arg(fileInfo.fileName(), reader.errorString()));
        return false;
    }
    if (rootName != QStringLiteral("annotation")) {
        result->errors.append(QStringLiteral("%1: VOC XML root must be annotation.").arg(fileInfo.fileName()));
        return false;
    }
    if (filename.isEmpty()) {
        result->errors.append(QStringLiteral("%1: VOC XML is missing filename.").arg(fileInfo.fileName()));
        return false;
    }
    if (!sawObject) {
        result->errors.append(QStringLiteral("%1: VOC XML is missing object entries.").arg(fileInfo.fileName()));
        return false;
    }
    result->previewSamples.append(filename);
    ++result->sampleCount;
    return true;
}

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
        const QFileInfo inputInfo(datasetPath);
        if (!inputInfo.exists()) {
            result.ok = false;
            result.errors.append(QStringLiteral("Dataset path does not exist."));
            return result;
        }

        if (id_ == QStringLiteral("coco_json")) {
            const QFileInfoList jsonFiles = datasetFiles(inputInfo, {QStringLiteral("*.json")});
            if (jsonFiles.isEmpty()) {
                result.ok = false;
                result.errors.append(QStringLiteral("No COCO JSON annotation files found."));
                return result;
            }
            for (const QFileInfo& fileInfo : jsonFiles) {
                validateCocoFile(fileInfo, &result);
            }
            result.ok = result.errors.isEmpty();
            return result;
        }

        if (id_ == QStringLiteral("voc_xml")) {
            const QFileInfoList xmlFiles = datasetFiles(inputInfo, {QStringLiteral("*.xml")});
            if (xmlFiles.isEmpty()) {
                result.ok = false;
                result.errors.append(QStringLiteral("No VOC XML annotation files found."));
                return result;
            }
            for (const QFileInfo& fileInfo : xmlFiles) {
                validateVocFile(fileInfo, &result);
            }
            result.ok = result.errors.isEmpty();
            return result;
        }

        const QDir root(inputInfo.absoluteFilePath());
        if (!root.exists()) {
            result.ok = false;
            result.errors.append(QStringLiteral("Dataset path must be a directory for this dataset format."));
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
