#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/v2/NcnnRuntimeAdapterV2.h"
#include "aitrain/v2/RuntimeCapabilityMatrixV2.h"
#include "aitrain/v2/TensorRtRuntimeAdapterV2.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QTemporaryDir>
#include <QTest>

namespace {

aitrain::v2::ModelManifestV2 baseManifest(const QByteArray& bytes, const QString& entry)
{
    aitrain::v2::ModelManifestV2 manifest;
    manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    manifest.modelFamily = QStringLiteral("yolo_detection");
    manifest.taskType = QStringLiteral("detection");
    manifest.sourceBackend = QStringLiteral("ultralytics_yolo_detect");
    manifest.sourceTaskId = aitrain::v2::TaskId::create();
    manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
    manifest.sourceArtifactSha256 = QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex());
    manifest.artifactEntryPath = entry;
    manifest.inputs = {{QStringLiteral("in0"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    manifest.outputs = {{QStringLiteral("out0"), QStringLiteral("NCN"), {1, 84, -1}}};
    manifest.preprocessing = QJsonObject{{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    manifest.postprocessing = QJsonObject{{QStringLiteral("id"), QStringLiteral("route_owned_decoder")}};
    manifest.classNames = QStringList{QStringLiteral("part")};
    manifest.exporterVersion = QStringLiteral("aitrain-runtime-contract-test");
    manifest.verified = true;
    return manifest;
}

bool writeFile(const QString& path, const QByteArray& bytes)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}

aitrain::v2::RuntimeModelLocationV2 ncnnLocation(QTemporaryDir* directory)
{
    const QByteArray param("7767517\n2 2\nInput images 0 1 in0\nMemoryData output 1 1 in0 out0 0=1\n");
    const QByteArray bin(4, '\0');
    writeFile(directory->filePath(QStringLiteral("model.param")), param);
    writeFile(directory->filePath(QStringLiteral("model.bin")), bin);
    aitrain::v2::RuntimeModelLocationV2 location;
    location.manifest = baseManifest(param, QStringLiteral("model.param"));
    location.manifest.artifactFormat = QStringLiteral("ncnn");
    location.manifest.opset = 0;
    location.manifest.decoder = QStringLiteral("ncnn_yolo_detection_ultralytics_v1");
    location.manifest.postprocessing.insert(QStringLiteral("ncnn"), QJsonObject{
        {QStringLiteral("binSha256"), QString::fromLatin1(QCryptographicHash::hash(bin, QCryptographicHash::Sha256).toHex())}});
    location.manifest.runtimeRoutes = QStringList{QStringLiteral("aitrain_ncnn")};
    location.artifactDirectory = directory->path();
    return location;
}

aitrain::v2::RuntimeModelLocationV2 tensorRtLocation(QTemporaryDir* directory)
{
    const QByteArray engine("serialized-engine-contract-fixture");
    writeFile(directory->filePath(QStringLiteral("model.engine")), engine);
    aitrain::v2::RuntimeModelLocationV2 location;
    location.manifest = baseManifest(engine, QStringLiteral("model.engine"));
    location.manifest.artifactFormat = QStringLiteral("tensorrt_engine");
    location.manifest.opset = 0;
    location.manifest.decoder = QStringLiteral("tensorrt_yolo_detection_v8");
    location.manifest.runtimeRoutes = QStringList{QStringLiteral("aitrain_tensorrt")};
    location.artifactDirectory = directory->path();
    return location;
}

} // namespace

class V2RuntimeAdapterTests : public QObject {
    Q_OBJECT
private slots:
    void statusCatalogIsStable();
    void ncnnRequiresExplicitManifestContract();
    void ncnnSegmentationContractDoesNotUseFamilyGuessing();
    void ncnnRejectsForbiddenProductFamiliesBeforeSdkProbe();
    void tensorRtSeparatesManifestSdkDependencyHardwareAndDecoderStates();
    void runtimeMatrixExportsUnifiedRuntimeStatus();
};

void V2RuntimeAdapterTests::statusCatalogIsStable()
{
    QCOMPARE(aitrain::v2::runtimeStatusV2ToString(aitrain::v2::RuntimeStatusV2::Available), QStringLiteral("available"));
    QCOMPARE(aitrain::v2::runtimeStatusV2ToString(aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented), QStringLiteral("runtime_not_implemented"));
    QCOMPARE(aitrain::v2::runtimeStatusV2ToString(aitrain::v2::RuntimeStatusV2::SdkMissing), QStringLiteral("sdk_missing"));
    QCOMPARE(aitrain::v2::runtimeStatusV2ToString(aitrain::v2::RuntimeStatusV2::DependencyMissing), QStringLiteral("dependency_missing"));
    QCOMPARE(aitrain::v2::runtimeStatusV2ToString(aitrain::v2::RuntimeStatusV2::HardwareUnsupported), QStringLiteral("hardware_unsupported"));
    QCOMPARE(aitrain::v2::runtimeStatusV2ToString(aitrain::v2::RuntimeStatusV2::ArtifactIncompatible), QStringLiteral("artifact_incompatible"));
}

void V2RuntimeAdapterTests::ncnnRequiresExplicitManifestContract()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::RuntimeModelLocationV2 location = ncnnLocation(&directory);
    aitrain::v2::NcnnRuntimeAdapterV2 adapter;
    const aitrain::v2::RuntimeOperationResultV2 validation = adapter.validateModel(location);
    QCOMPARE(validation.status, aitrain::v2::RuntimeStatusV2::Available);
    QCOMPARE(validation.details.value(QStringLiteral("inputBlob")).toString(), QStringLiteral("in0"));
    QCOMPARE(validation.details.value(QStringLiteral("outputBlobs")).toArray().first().toString(), QStringLiteral("out0"));
    QCOMPARE(validation.details.value(QStringLiteral("decoder")).toString(), QStringLiteral("ncnn_yolo_detection_ultralytics_v1"));

    const aitrain::NcnnBackendStatus backend = aitrain::ncnnBackendStatus();
    const aitrain::v2::RuntimeStatusV2 expectedProbe = !backend.sdkAvailable
        ? aitrain::v2::RuntimeStatusV2::SdkMissing
        : (backend.inferenceAvailable ? aitrain::v2::RuntimeStatusV2::Available
                                      : aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
    QCOMPARE(adapter.probe(location).status, expectedProbe);

    location.manifest.decoder = QStringLiteral("auto");
    QCOMPARE(adapter.validateModel(location).status, aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
    location = ncnnLocation(&directory);
    QVERIFY(writeFile(directory.filePath(QStringLiteral("model.bin")), QByteArray("tampered")));
    QCOMPARE(adapter.validateModel(location).status, aitrain::v2::RuntimeStatusV2::ArtifactIncompatible);
    location = ncnnLocation(&directory);
    QFile::remove(directory.filePath(QStringLiteral("model.bin")));
    QCOMPARE(adapter.validateModel(location).status, aitrain::v2::RuntimeStatusV2::ArtifactIncompatible);
}

void V2RuntimeAdapterTests::ncnnSegmentationContractDoesNotUseFamilyGuessing()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::RuntimeModelLocationV2 location = ncnnLocation(&directory);
    location.manifest.modelFamily = QStringLiteral("yolo_segmentation");
    location.manifest.taskType = QStringLiteral("instance_segmentation");
    location.manifest.decoder = QStringLiteral("ncnn_yolo_segmentation_ultralytics_v1");
    location.manifest.outputs.append(
        {QStringLiteral("proto"), QStringLiteral("NCHW"), {1, 32, 160, 160}});
    const aitrain::v2::RuntimeOperationResultV2 validation =
        aitrain::v2::NcnnRuntimeAdapterV2().validateModel(location);
    QCOMPARE(validation.status, aitrain::v2::RuntimeStatusV2::Available);
    QCOMPARE(validation.details.value(QStringLiteral("outputBlobs")).toArray().size(), 2);

    location.manifest.decoder = QStringLiteral("ncnn_yolo_segmentation_dfl_v1");
    QCOMPARE(aitrain::v2::NcnnRuntimeAdapterV2().validateModel(location).status,
        aitrain::v2::RuntimeStatusV2::ArtifactIncompatible);
    location.manifest.outputs.append(
        {QStringLiteral("proto_dfl"), QStringLiteral("NCHW"), {1, 32, 160, 160}});
    location.manifest.postprocessing.insert(QStringLiteral("ncnn"), QJsonObject{
        {QStringLiteral("binSha256"), location.manifest.postprocessing.value(QStringLiteral("ncnn")).toObject().value(QStringLiteral("binSha256"))},
        {QStringLiteral("strides"), QJsonArray{8, 16, 32}}, {QStringLiteral("regMax"), 16}});
    QCOMPARE(aitrain::v2::NcnnRuntimeAdapterV2().validateModel(location).status,
        aitrain::v2::RuntimeStatusV2::Available);
}

void V2RuntimeAdapterTests::ncnnRejectsForbiddenProductFamiliesBeforeSdkProbe()
{
    const QStringList forbiddenFamilies{QStringLiteral("yolo_obb"), QStringLiteral("semantic_segmentation"),
        QStringLiteral("anomaly_detection"), QStringLiteral("ocr_detection"), QStringLiteral("ocr_recognition")};
    for (const QString& family : forbiddenFamilies) {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        aitrain::v2::RuntimeModelLocationV2 location = ncnnLocation(&directory);
        location.manifest.modelFamily = family;
        location.manifest.taskType = family;
        QCOMPARE(aitrain::v2::NcnnRuntimeAdapterV2().probe(location).status,
            aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
    }
}

void V2RuntimeAdapterTests::tensorRtSeparatesManifestSdkDependencyHardwareAndDecoderStates()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::RuntimeModelLocationV2 location = tensorRtLocation(&directory);
    aitrain::v2::TensorRtRuntimeAdapterV2 adapter;
    QCOMPARE(adapter.validateModel(location).status, aitrain::v2::RuntimeStatusV2::Available);

    const aitrain::TensorRtBackendStatus backend = aitrain::tensorRtBackendStatus();
    aitrain::v2::RuntimeStatusV2 expected = aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented;
    if (!backend.sdkAvailable) expected = aitrain::v2::RuntimeStatusV2::SdkMissing;
    else if (backend.status == QStringLiteral("dependency_missing") || !backend.dependenciesAvailable)
        expected = aitrain::v2::RuntimeStatusV2::DependencyMissing;
    else if (!backend.hardwareSupported) expected = aitrain::v2::RuntimeStatusV2::HardwareUnsupported;
    else if (backend.inferenceAvailable) expected = aitrain::v2::RuntimeStatusV2::Available;
    QCOMPARE(adapter.probe(location).status, expected);

    location.manifest.decoder = QStringLiteral("tensorrt_unknown_decoder");
    QCOMPARE(adapter.probe(location).status, aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
    location = tensorRtLocation(&directory);
    location.manifest.modelFamily = QStringLiteral("yolo_obb");
    QCOMPARE(adapter.probe(location).status, aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
}

void V2RuntimeAdapterTests::runtimeMatrixExportsUnifiedRuntimeStatus()
{
    aitrain::v2::RuntimeCapabilityMatrixV2 matrix;
    const aitrain::v2::RuntimeCapabilityV2 obb = matrix.query(
        {QStringLiteral("yolo_obb"), QStringLiteral("aitrain_ncnn")});
    QCOMPARE(obb.status, aitrain::v2::RuntimeCapabilityStatusV2::UnsupportedByProduct);
    QCOMPARE(obb.runtimeStatus, aitrain::v2::RuntimeStatusV2::RuntimeNotImplemented);
    QCOMPARE(obb.toJson().value(QStringLiteral("runtimeStatus")).toString(), QStringLiteral("runtime_not_implemented"));

    const aitrain::v2::RuntimeCapabilityV2 tensorRt = matrix.query(
        {QStringLiteral("yolo_detection"), QStringLiteral("aitrain_tensorrt")});
    QCOMPARE(tensorRt.toJson().value(QStringLiteral("runtimeStatus")).toString(),
        aitrain::v2::runtimeStatusV2ToString(tensorRt.runtimeStatus));
    QVERIFY(tensorRt.status == aitrain::v2::RuntimeCapabilityStatusV2::RequiresSdk
        || tensorRt.status == aitrain::v2::RuntimeCapabilityStatusV2::RequiresDependency
        || tensorRt.status == aitrain::v2::RuntimeCapabilityStatusV2::RequiresHardware
        || tensorRt.status == aitrain::v2::RuntimeCapabilityStatusV2::RuntimeNotImplemented
        || tensorRt.status == aitrain::v2::RuntimeCapabilityStatusV2::Supported);
}

QTEST_MAIN(V2RuntimeAdapterTests)
#include "tst_v2_runtime_adapters.moc"
