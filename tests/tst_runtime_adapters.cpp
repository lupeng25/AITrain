#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/runtime/NcnnRuntimeAdapter.h"
#include "aitrain/runtime/RuntimeCapabilityMatrix.h"
#include "aitrain/runtime/TensorRtRuntimeAdapter.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QTemporaryDir>
#include <QTest>

namespace {

aitrain::ModelManifest baseManifest(const QByteArray& bytes, const QString& entry)
{
    aitrain::ModelManifest manifest;
    manifest.modelPackageId = aitrain::ModelPackageId::create();
    manifest.modelFamily = QStringLiteral("yolo_detection");
    manifest.taskType = QStringLiteral("detection");
    manifest.sourceBackend = QStringLiteral("ultralytics_yolo_detect");
    manifest.sourceTaskId = aitrain::TaskId::create();
    manifest.sourceSnapshotId = aitrain::SnapshotId::create();
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

aitrain::RuntimeModelLocation ncnnLocation(QTemporaryDir* directory)
{
    const QByteArray param("7767517\n2 2\nInput images 0 1 in0\nMemoryData output 1 1 in0 out0 0=1\n");
    const QByteArray bin(4, '\0');
    writeFile(directory->filePath(QStringLiteral("model.param")), param);
    writeFile(directory->filePath(QStringLiteral("model.bin")), bin);
    aitrain::RuntimeModelLocation location;
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

aitrain::RuntimeModelLocation tensorRtLocation(QTemporaryDir* directory)
{
    const QByteArray engine("serialized-engine-contract-fixture");
    writeFile(directory->filePath(QStringLiteral("model.engine")), engine);
    aitrain::RuntimeModelLocation location;
    location.manifest = baseManifest(engine, QStringLiteral("model.engine"));
    location.manifest.artifactFormat = QStringLiteral("tensorrt_engine");
    location.manifest.opset = 0;
    location.manifest.decoder = QStringLiteral("tensorrt_yolo_detection_v8");
    location.manifest.runtimeRoutes = QStringList{QStringLiteral("aitrain_tensorrt")};
    location.artifactDirectory = directory->path();
    return location;
}

} // namespace

class RuntimeAdapterTests : public QObject {
    Q_OBJECT
private slots:
    void statusCatalogIsStable();
    void ncnnRequiresExplicitManifestContract();
    void ncnnSegmentationContractDoesNotUseFamilyGuessing();
    void ncnnRejectsForbiddenProductFamiliesBeforeSdkProbe();
    void tensorRtSeparatesManifestSdkDependencyHardwareAndDecoderStates();
    void runtimeMatrixExportsUnifiedRuntimeStatus();
};

void RuntimeAdapterTests::statusCatalogIsStable()
{
    QCOMPARE(aitrain::runtimeStatusToString(aitrain::RuntimeStatus::Available), QStringLiteral("available"));
    QCOMPARE(aitrain::runtimeStatusToString(aitrain::RuntimeStatus::RuntimeNotImplemented), QStringLiteral("runtime_not_implemented"));
    QCOMPARE(aitrain::runtimeStatusToString(aitrain::RuntimeStatus::SdkMissing), QStringLiteral("sdk_missing"));
    QCOMPARE(aitrain::runtimeStatusToString(aitrain::RuntimeStatus::DependencyMissing), QStringLiteral("dependency_missing"));
    QCOMPARE(aitrain::runtimeStatusToString(aitrain::RuntimeStatus::HardwareUnsupported), QStringLiteral("hardware_unsupported"));
    QCOMPARE(aitrain::runtimeStatusToString(aitrain::RuntimeStatus::ArtifactIncompatible), QStringLiteral("artifact_incompatible"));
}

void RuntimeAdapterTests::ncnnRequiresExplicitManifestContract()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::RuntimeModelLocation location = ncnnLocation(&directory);
    aitrain::NcnnRuntimeAdapter adapter;
    const aitrain::RuntimeOperationResult validation = adapter.validateModel(location);
    QCOMPARE(validation.status, aitrain::RuntimeStatus::Available);
    QCOMPARE(validation.details.value(QStringLiteral("inputBlob")).toString(), QStringLiteral("in0"));
    QCOMPARE(validation.details.value(QStringLiteral("outputBlobs")).toArray().first().toString(), QStringLiteral("out0"));
    QCOMPARE(validation.details.value(QStringLiteral("decoder")).toString(), QStringLiteral("ncnn_yolo_detection_ultralytics_v1"));

    const aitrain::NcnnBackendStatus backend = aitrain::ncnnBackendStatus();
    const aitrain::RuntimeStatus expectedProbe = !backend.sdkAvailable
        ? aitrain::RuntimeStatus::SdkMissing
        : (backend.inferenceAvailable ? aitrain::RuntimeStatus::Available
                                      : aitrain::RuntimeStatus::RuntimeNotImplemented);
    QCOMPARE(adapter.probe(location).status, expectedProbe);

    location.manifest.decoder = QStringLiteral("auto");
    QCOMPARE(adapter.validateModel(location).status, aitrain::RuntimeStatus::RuntimeNotImplemented);
    location = ncnnLocation(&directory);
    QVERIFY(writeFile(directory.filePath(QStringLiteral("model.bin")), QByteArray("tampered")));
    QCOMPARE(adapter.validateModel(location).status, aitrain::RuntimeStatus::ArtifactIncompatible);
    location = ncnnLocation(&directory);
    QFile::remove(directory.filePath(QStringLiteral("model.bin")));
    QCOMPARE(adapter.validateModel(location).status, aitrain::RuntimeStatus::ArtifactIncompatible);
}

void RuntimeAdapterTests::ncnnSegmentationContractDoesNotUseFamilyGuessing()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::RuntimeModelLocation location = ncnnLocation(&directory);
    location.manifest.modelFamily = QStringLiteral("yolo_segmentation");
    location.manifest.taskType = QStringLiteral("instance_segmentation");
    location.manifest.decoder = QStringLiteral("ncnn_yolo_segmentation_ultralytics_v1");
    location.manifest.outputs.append(
        {QStringLiteral("proto"), QStringLiteral("NCHW"), {1, 32, 160, 160}});
    const aitrain::RuntimeOperationResult validation =
        aitrain::NcnnRuntimeAdapter().validateModel(location);
    QCOMPARE(validation.status, aitrain::RuntimeStatus::Available);
    QCOMPARE(validation.details.value(QStringLiteral("outputBlobs")).toArray().size(), 2);

    location.manifest.decoder = QStringLiteral("ncnn_yolo_segmentation_dfl_v1");
    QCOMPARE(aitrain::NcnnRuntimeAdapter().validateModel(location).status,
        aitrain::RuntimeStatus::ArtifactIncompatible);
    location.manifest.outputs.append(
        {QStringLiteral("proto_dfl"), QStringLiteral("NCHW"), {1, 32, 160, 160}});
    location.manifest.postprocessing.insert(QStringLiteral("ncnn"), QJsonObject{
        {QStringLiteral("binSha256"), location.manifest.postprocessing.value(QStringLiteral("ncnn")).toObject().value(QStringLiteral("binSha256"))},
        {QStringLiteral("strides"), QJsonArray{8, 16, 32}}, {QStringLiteral("regMax"), 16}});
    QCOMPARE(aitrain::NcnnRuntimeAdapter().validateModel(location).status,
        aitrain::RuntimeStatus::Available);
}

void RuntimeAdapterTests::ncnnRejectsForbiddenProductFamiliesBeforeSdkProbe()
{
    const QStringList forbiddenFamilies{QStringLiteral("yolo_obb"), QStringLiteral("semantic_segmentation"),
        QStringLiteral("anomaly_detection"), QStringLiteral("ocr_detection"), QStringLiteral("ocr_recognition")};
    for (const QString& family : forbiddenFamilies) {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        aitrain::RuntimeModelLocation location = ncnnLocation(&directory);
        location.manifest.modelFamily = family;
        location.manifest.taskType = family;
        QCOMPARE(aitrain::NcnnRuntimeAdapter().probe(location).status,
            aitrain::RuntimeStatus::RuntimeNotImplemented);
    }
}

void RuntimeAdapterTests::tensorRtSeparatesManifestSdkDependencyHardwareAndDecoderStates()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::RuntimeModelLocation location = tensorRtLocation(&directory);
    aitrain::TensorRtRuntimeAdapter adapter;
    QCOMPARE(adapter.validateModel(location).status, aitrain::RuntimeStatus::Available);

    const aitrain::TensorRtBackendStatus backend = aitrain::tensorRtBackendStatus();
    aitrain::RuntimeStatus expected = aitrain::RuntimeStatus::RuntimeNotImplemented;
    if (!backend.sdkAvailable) expected = aitrain::RuntimeStatus::SdkMissing;
    else if (backend.status == QStringLiteral("dependency_missing") || !backend.dependenciesAvailable)
        expected = aitrain::RuntimeStatus::DependencyMissing;
    else if (!backend.hardwareSupported) expected = aitrain::RuntimeStatus::HardwareUnsupported;
    else if (backend.inferenceAvailable) expected = aitrain::RuntimeStatus::Available;
    QCOMPARE(adapter.probe(location).status, expected);

    location.manifest.decoder = QStringLiteral("tensorrt_unknown_decoder");
    QCOMPARE(adapter.probe(location).status, aitrain::RuntimeStatus::RuntimeNotImplemented);
    location = tensorRtLocation(&directory);
    location.manifest.modelFamily = QStringLiteral("yolo_obb");
    QCOMPARE(adapter.probe(location).status, aitrain::RuntimeStatus::RuntimeNotImplemented);
}

void RuntimeAdapterTests::runtimeMatrixExportsUnifiedRuntimeStatus()
{
    aitrain::RuntimeCapabilityMatrix matrix;
    const aitrain::RuntimeCapability obb = matrix.query(
        {QStringLiteral("yolo_obb"), QStringLiteral("aitrain_ncnn")});
    QCOMPARE(obb.status, aitrain::RuntimeCapabilityStatus::UnsupportedByProduct);
    QCOMPARE(obb.runtimeStatus, aitrain::RuntimeStatus::RuntimeNotImplemented);
    QCOMPARE(obb.toJson().value(QStringLiteral("runtimeStatus")).toString(), QStringLiteral("runtime_not_implemented"));

    const aitrain::RuntimeCapability tensorRt = matrix.query(
        {QStringLiteral("yolo_detection"), QStringLiteral("aitrain_tensorrt")});
    QCOMPARE(tensorRt.toJson().value(QStringLiteral("runtimeStatus")).toString(),
        aitrain::runtimeStatusToString(tensorRt.runtimeStatus));
    QVERIFY(tensorRt.status == aitrain::RuntimeCapabilityStatus::RequiresSdk
        || tensorRt.status == aitrain::RuntimeCapabilityStatus::RequiresDependency
        || tensorRt.status == aitrain::RuntimeCapabilityStatus::RequiresHardware
        || tensorRt.status == aitrain::RuntimeCapabilityStatus::RuntimeNotImplemented
        || tensorRt.status == aitrain::RuntimeCapabilityStatus::Supported);
}

QTEST_MAIN(RuntimeAdapterTests)
#include "tst_runtime_adapters.moc"
