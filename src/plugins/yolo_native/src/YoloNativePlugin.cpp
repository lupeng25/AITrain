#include "aitrain/core/PluginInterfaces.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonObject>
#include <QObject>
#include <QRegularExpression>
#include <QTextStream>

namespace {

class YoloDatasetAdapter final : public aitrain::IDatasetAdapter {
public:
    QString formatId() const override { return QStringLiteral("yolo_txt"); }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        Q_UNUSED(options)
        aitrain::DatasetValidationResult result;
        const QDir root(datasetPath);
        const QDir labels(root.filePath(QStringLiteral("labels")));
        const QDir images(root.filePath(QStringLiteral("images")));

        if (!root.exists()) {
            result.ok = false;
            result.errors.append(QStringLiteral("Dataset path does not exist."));
            return result;
        }
        if (!labels.exists()) {
            result.ok = false;
            result.errors.append(QStringLiteral("Missing labels directory."));
        }
        if (!images.exists()) {
            result.warnings.append(QStringLiteral("Missing images directory; validation only checks labels."));
        }

        const QFileInfoList labelFiles = labels.entryInfoList({QStringLiteral("*.txt")}, QDir::Files);
        result.sampleCount = labelFiles.size();
        if (labelFiles.isEmpty()) {
            result.ok = false;
            result.errors.append(QStringLiteral("No YOLO label files found."));
            return result;
        }

        int inspected = 0;
        for (const QFileInfo& fileInfo : labelFiles) {
            if (++inspected > 500) {
                result.warnings.append(QStringLiteral("Validation stopped after 500 label files."));
                break;
            }
            QFile file(fileInfo.absoluteFilePath());
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                result.ok = false;
                result.errors.append(QStringLiteral("Cannot open %1").arg(fileInfo.fileName()));
                continue;
            }
            int lineNumber = 0;
            while (!file.atEnd()) {
                ++lineNumber;
                const QString line = QString::fromUtf8(file.readLine()).trimmed();
                if (line.isEmpty()) {
                    continue;
                }
                const QStringList parts = line.split(QRegularExpression(QStringLiteral("\\s+")),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
                    QString::SkipEmptyParts
#else
                    Qt::SkipEmptyParts
#endif
                );
                if (parts.size() != 5 && parts.size() < 7) {
                    result.ok = false;
                    result.errors.append(QStringLiteral("%1:%2 invalid YOLO row").arg(fileInfo.fileName()).arg(lineNumber));
                    continue;
                }
                bool classOk = false;
                parts.first().toInt(&classOk);
                if (!classOk) {
                    result.ok = false;
                    result.errors.append(QStringLiteral("%1:%2 class id is not integer").arg(fileInfo.fileName()).arg(lineNumber));
                }
                for (int i = 1; i < parts.size(); ++i) {
                    bool ok = false;
                    const double value = parts.at(i).toDouble(&ok);
                    if (!ok || value < 0.0 || value > 1.0) {
                        result.ok = false;
                        result.errors.append(QStringLiteral("%1:%2 coordinate out of [0,1]").arg(fileInfo.fileName()).arg(lineNumber));
                        break;
                    }
                }
                if (result.errors.size() > 50) {
                    result.warnings.append(QStringLiteral("Too many errors; validation truncated."));
                    return result;
                }
            }
        }
        return result;
    }
};

class NativeTrainer final : public aitrain::ITrainer {
public:
    QString backendName() const override { return QStringLiteral("LibTorch/CUDA native YOLO scaffold"); }
};

class NativeValidator final : public aitrain::IValidator {
public:
    QString backendName() const override { return QStringLiteral("YOLO metric validator scaffold"); }
};

class NativeExporter final : public aitrain::IExporter {
public:
    QStringList supportedFormats() const override { return QStringList() << QStringLiteral("onnx") << QStringLiteral("tensorrt"); }
};

class NativeInferencer final : public aitrain::IInferencer {
public:
    QString backendName() const override { return QStringLiteral("ONNX Runtime/TensorRT YOLO inferencer scaffold"); }
};

} // namespace

class YoloNativePlugin final : public QObject, public aitrain::IModelPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID AITrainModelPluginInterface_iid FILE "../yolo_native.json")
    Q_INTERFACES(aitrain::IModelPlugin)

public:
    aitrain::PluginManifest manifest() const override
    {
        aitrain::PluginManifest manifest;
        manifest.id = QStringLiteral("com.aitrain.plugins.yolo_native");
        manifest.name = QStringLiteral("YOLO Native");
        manifest.version = QStringLiteral("0.1.0");
        manifest.description = QStringLiteral("C++ native YOLO-style detection and segmentation plugin scaffold.");
        manifest.taskTypes = QStringList() << QStringLiteral("detection") << QStringLiteral("segmentation");
        manifest.datasetFormats = QStringList() << QStringLiteral("yolo_txt");
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
        manifest.parameterSchema.append(addParam(QStringLiteral("batchSize"), QStringLiteral("Batch Size"), QStringLiteral("int"), 16));
        manifest.parameterSchema.append(addParam(QStringLiteral("imageSize"), QStringLiteral("Image Size"), QStringLiteral("int"), 640));
        manifest.parameterSchema.append(addParam(QStringLiteral("learningRate"), QStringLiteral("Learning Rate"), QStringLiteral("float"), 0.01));
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
    YoloDatasetAdapter adapter_;
    NativeTrainer trainer_;
    NativeValidator validator_;
    NativeExporter exporter_;
    NativeInferencer inferencer_;
};

#include "YoloNativePlugin.moc"
