#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QJsonObject>
#include <QObject>

#include <utility>

namespace {

class YoloDatasetAdapter final : public aitrain::IDatasetAdapter {
public:
    explicit YoloDatasetAdapter(QString formatId)
        : formatId_(std::move(formatId))
    {
    }

    QString formatId() const override { return formatId_; }

    aitrain::DatasetValidationResult validateDataset(const QString& datasetPath, const QJsonObject& options) override
    {
        return formatId_ == QStringLiteral("yolo_segmentation")
            ? aitrain::validateYoloSegmentationDataset(datasetPath, options)
            : aitrain::validateYoloDetectionDataset(datasetPath, options);
    }

private:
    QString formatId_;
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
    QStringList supportedFormats() const override { return QStringList() << QStringLiteral("tiny_detector_json") << QStringLiteral("onnx") << QStringLiteral("ncnn"); }
};

class NativeInferencer final : public aitrain::IInferencer {
public:
    QString backendName() const override { return QStringLiteral("Tiny detector checkpoint inferencer scaffold; ONNX Runtime/TensorRT pending"); }
};

} // namespace

class YoloNativePlugin final : public QObject, public aitrain::IModelPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID AITrainModelPluginInterface_iid FILE "../yolo_native.json")
    Q_INTERFACES(aitrain::IModelPlugin)

public:
    YoloNativePlugin()
        : detectionAdapter_(QStringLiteral("yolo_detection"))
        , segmentationAdapter_(QStringLiteral("yolo_segmentation"))
    {
    }

    aitrain::PluginManifest manifest() const override
    {
        aitrain::PluginManifest manifest;
        manifest.id = QStringLiteral("com.aitrain.plugins.yolo_native");
        manifest.name = QStringLiteral("YOLO Native");
        manifest.version = QStringLiteral("0.1.0");
        manifest.description = QStringLiteral("C++ native YOLO-style detection and segmentation plugin scaffold.");
        manifest.taskTypes = QStringList() << QStringLiteral("detection") << QStringLiteral("segmentation");
        manifest.datasetFormats = QStringList() << QStringLiteral("yolo_detection") << QStringLiteral("yolo_segmentation");
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
        if (formatId == detectionAdapter_.formatId() || formatId == QStringLiteral("yolo_txt")) return &detectionAdapter_;
        if (formatId == segmentationAdapter_.formatId()) return &segmentationAdapter_;
        return nullptr;
    }

    aitrain::ITrainer* trainer() override { return &trainer_; }
    aitrain::IValidator* validator() override { return &validator_; }
    aitrain::IExporter* exporter() override { return &exporter_; }
    aitrain::IInferencer* inferencer() override { return &inferencer_; }

private:
    YoloDatasetAdapter detectionAdapter_;
    YoloDatasetAdapter segmentationAdapter_;
    NativeTrainer trainer_;
    NativeValidator validator_;
    NativeExporter exporter_;
    NativeInferencer inferencer_;
};

#include "YoloNativePlugin.moc"
