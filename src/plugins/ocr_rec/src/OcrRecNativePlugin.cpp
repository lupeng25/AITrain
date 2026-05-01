#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QJsonObject>
#include <QObject>

namespace {

class PaddleOcrRecAdapter final : public aitrain::IDatasetAdapter {
public:
    QString formatId() const override { return QStringLiteral("paddleocr_rec"); }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        return aitrain::validatePaddleOcrRecDataset(datasetPath, options);
    }
};

class PaddleOcrDetAdapter final : public aitrain::IDatasetAdapter {
public:
    QString formatId() const override { return QStringLiteral("paddleocr_det"); }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        return aitrain::validatePaddleOcrDetDataset(datasetPath, options);
    }
};

class NativeTrainer final : public aitrain::ITrainer {
public:
    QString backendName() const override { return QStringLiteral("LibTorch/CUDA CTC OCR scaffold"); }
};

class NativeValidator final : public aitrain::IValidator {
public:
    QString backendName() const override { return QStringLiteral("OCR accuracy/edit-distance validator scaffold"); }
};

class NativeExporter final : public aitrain::IExporter {
public:
    QStringList supportedFormats() const override { return {}; }
};

class NativeInferencer final : public aitrain::IInferencer {
public:
    QString backendName() const override { return QStringLiteral("OCR recognition inferencer scaffold"); }
};

} // namespace

class OcrRecNativePlugin final : public QObject, public aitrain::IModelPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID AITrainModelPluginInterface_iid FILE "../ocr_rec.json")
    Q_INTERFACES(aitrain::IModelPlugin)

public:
    aitrain::PluginManifest manifest() const override
    {
        aitrain::PluginManifest manifest;
        manifest.id = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        manifest.name = QStringLiteral("PaddleOCR Rec Native");
        manifest.version = QStringLiteral("0.1.0");
        manifest.description = QStringLiteral("PaddleOCR dataset adapters and OCR scaffold/plugin entry points.");
        manifest.taskTypes = QStringList()
            << QStringLiteral("ocr_detection")
            << QStringLiteral("ocr_recognition")
            << QStringLiteral("ocr");
        manifest.datasetFormats = QStringList()
            << QStringLiteral("paddleocr_det")
            << QStringLiteral("paddleocr_rec");
        manifest.exportFormats = exporter_.supportedFormats();
        manifest.requiresGpu = true;

        auto addParam = [](const QString& key, const QString& label, const QString& type, const QJsonValue& defaultValue) {
            QJsonObject object;
            object.insert(QStringLiteral("key"), key);
            object.insert(QStringLiteral("label"), label);
            object.insert(QStringLiteral("type"), type);
            object.insert(QStringLiteral("default"), defaultValue);
            return object;
        };
        manifest.parameterSchema.append(addParam(QStringLiteral("epochs"), QStringLiteral("Epochs"), QStringLiteral("int"), 100));
        manifest.parameterSchema.append(addParam(QStringLiteral("batchSize"), QStringLiteral("Batch Size"), QStringLiteral("int"), 64));
        manifest.parameterSchema.append(addParam(QStringLiteral("maxTextLength"), QStringLiteral("Max Text Length"), QStringLiteral("int"), 25));
        return manifest;
    }

    aitrain::IDatasetAdapter* datasetAdapter(const QString& formatId) override
    {
        if (formatId == detAdapter_.formatId()) {
            return &detAdapter_;
        }
        return formatId == recAdapter_.formatId() ? &recAdapter_ : nullptr;
    }

    aitrain::ITrainer* trainer() override { return &trainer_; }
    aitrain::IValidator* validator() override { return &validator_; }
    aitrain::IExporter* exporter() override { return &exporter_; }
    aitrain::IInferencer* inferencer() override { return &inferencer_; }

private:
    PaddleOcrDetAdapter detAdapter_;
    PaddleOcrRecAdapter recAdapter_;
    NativeTrainer trainer_;
    NativeValidator validator_;
    NativeExporter exporter_;
    NativeInferencer inferencer_;
};

#include "OcrRecNativePlugin.moc"
