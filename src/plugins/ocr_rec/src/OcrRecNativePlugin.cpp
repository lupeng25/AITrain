#include "aitrain/core/PluginInterfaces.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonObject>
#include <QObject>
#include <QSet>

namespace {

class PaddleOcrRecAdapter final : public aitrain::IDatasetAdapter {
public:
    QString formatId() const override { return QStringLiteral("paddleocr_rec"); }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        aitrain::DatasetValidationResult result;
        const QDir root(datasetPath);
        if (!root.exists()) {
            result.ok = false;
            result.errors.append(QStringLiteral("Dataset path does not exist."));
            return result;
        }

        const QString labelFilePath = options.value(QStringLiteral("labelFile")).toString(
            root.filePath(QStringLiteral("rec_gt.txt")));
        QFile labelFile(labelFilePath);
        if (!labelFile.exists()) {
            result.ok = false;
            result.errors.append(QStringLiteral("Missing PaddleOCR rec label file: %1").arg(labelFilePath));
            return result;
        }
        if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            result.ok = false;
            result.errors.append(QStringLiteral("Cannot open label file."));
            return result;
        }

        QSet<QChar> dictionary;
        const QString dictFilePath = options.value(QStringLiteral("dictionaryFile")).toString();
        if (!dictFilePath.isEmpty()) {
            QFile dictFile(dictFilePath);
            if (dictFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                while (!dictFile.atEnd()) {
                    const QString line = QString::fromUtf8(dictFile.readLine()).trimmed();
                    if (!line.isEmpty()) {
                        dictionary.insert(line.at(0));
                    }
                }
            }
        }

        int lineNumber = 0;
        while (!labelFile.atEnd()) {
            ++lineNumber;
            const QString line = QString::fromUtf8(labelFile.readLine()).trimmed();
            if (line.isEmpty()) {
                continue;
            }
            const int split = line.indexOf(QLatin1Char('\t'));
            if (split <= 0) {
                result.ok = false;
                result.errors.append(QStringLiteral("Line %1 must be '<image path>\\t<label>'.").arg(lineNumber));
                continue;
            }
            const QString imagePath = line.left(split);
            const QString text = line.mid(split + 1);
            if (!QFileInfo::exists(root.filePath(imagePath))) {
                result.warnings.append(QStringLiteral("Missing image: %1").arg(imagePath));
            }
            if (!dictionary.isEmpty()) {
                for (const QChar ch : text) {
                    if (!dictionary.contains(ch)) {
                        result.ok = false;
                        result.errors.append(QStringLiteral("Line %1 contains char not in dictionary: %2").arg(lineNumber).arg(ch));
                        break;
                    }
                }
            }
            ++result.sampleCount;
            if (result.errors.size() > 50) {
                result.warnings.append(QStringLiteral("Too many errors; validation truncated."));
                break;
            }
        }

        if (result.sampleCount == 0) {
            result.ok = false;
            result.errors.append(QStringLiteral("No OCR recognition samples found."));
        }
        return result;
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
    QStringList supportedFormats() const override { return QStringList() << QStringLiteral("onnx") << QStringLiteral("tensorrt"); }
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
        manifest.description = QStringLiteral("PaddleOCR recognition-format compatible C++ CTC training plugin scaffold.");
        manifest.taskTypes = QStringList() << QStringLiteral("ocr_recognition");
        manifest.datasetFormats = QStringList() << QStringLiteral("paddleocr_rec");
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
        return formatId == adapter_.formatId() ? &adapter_ : nullptr;
    }

    aitrain::ITrainer* trainer() override { return &trainer_; }
    aitrain::IValidator* validator() override { return &validator_; }
    aitrain::IExporter* exporter() override { return &exporter_; }
    aitrain::IInferencer* inferencer() override { return &inferencer_; }

private:
    PaddleOcrRecAdapter adapter_;
    NativeTrainer trainer_;
    NativeValidator validator_;
    NativeExporter exporter_;
    NativeInferencer inferencer_;
};

#include "OcrRecNativePlugin.moc"
