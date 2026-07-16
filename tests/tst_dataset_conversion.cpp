#include "aitrain/dataset/BuiltinDatasetDrivers.h"
#include "aitrain/dataset/DatasetConversionService.h"

#include <QColor>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>

#include <memory>

namespace {

bool writeImage(const QString& path, int marker = 0)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QImage image(16, 12, QImage::Format_RGB32);
    image.fill(QColor(marker, 255 - marker, 32));
    return image.save(path);
}

bool writeUtf8(const QString& path, const QString& text)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QFile file(path);
    const QByteArray bytes = text.toUtf8();
    return file.open(QIODevice::WriteOnly | QIODevice::Truncate)
        && file.write(bytes) == bytes.size();
}

bool createSparseFile(const QString& path, qint64 bytes)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QFile file(path);
    return file.open(QIODevice::WriteOnly | QIODevice::Truncate) && file.resize(bytes);
}

QString createCocoFixtureWithFileName(const QString& root, const QString& relative)
{
    if (!writeImage(QDir(root).filePath(relative), 31)) qFatal("write named COCO image failed");
    const QJsonObject coco{
        {QStringLiteral("images"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
             {QStringLiteral("file_name"), relative}, {QStringLiteral("width"), 16}, {QStringLiteral("height"), 12}}}},
        {QStringLiteral("categories"), QJsonArray{QJsonObject{{QStringLiteral("id"), 7},
             {QStringLiteral("name"), QStringLiteral("零件")}}}},
        {QStringLiteral("annotations"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
             {QStringLiteral("image_id"), 1}, {QStringLiteral("category_id"), 7},
             {QStringLiteral("bbox"), QJsonArray{2, 2, 8, 6}}}}}};
    const QString path = QDir(root).filePath(QStringLiteral("annotations.json"));
    if (!writeUtf8(path, QString::fromUtf8(QJsonDocument(coco).toJson(QJsonDocument::Compact)))) {
        qFatal("write named COCO json failed");
    }
    return path;
}

aitrain::TaskSnapshot createTask(aitrain::ProjectStore* storage)
{
    aitrain::TaskSnapshot task;
    task.id = aitrain::TaskId::create();
    task.requestId = aitrain::RequestId::create();
    task.capabilityId = QStringLiteral("dataset_conversion");
    task.taskType = QStringLiteral("dataset_conversion");
    task.state = aitrain::TaskState::Created;
    task.createdAt = QDateTime::currentDateTimeUtc();
    task.updatedAt = task.createdAt;
    QString error;
    if (!storage->createTask(task, &error)) qFatal("createTask failed: %s", qPrintable(error));
    return task;
}

QString createCocoFixture(const QString& root, int imageCount, bool duplicateBasename = false, bool segmentation = false)
{
    QJsonArray images;
    QJsonArray annotations;
    for (int index = 0; index < imageCount; ++index) {
        const QString relative = duplicateBasename
            ? QStringLiteral("images/%1/a.png").arg(index)
            : QStringLiteral("images/样本 %1.png").arg(index);
        if (!writeImage(QDir(root).filePath(relative), index % 200)) qFatal("writeImage failed");
        images.append(QJsonObject{{QStringLiteral("id"), index + 1},
            {QStringLiteral("file_name"), relative}, {QStringLiteral("width"), 16}, {QStringLiteral("height"), 12}});
        QJsonObject annotation{{QStringLiteral("id"), index + 1},
            {QStringLiteral("image_id"), index + 1}, {QStringLiteral("category_id"), 7},
            {QStringLiteral("bbox"), QJsonArray{2, 2, 8, 6}}};
        if (segmentation) {
            annotation.insert(QStringLiteral("segmentation"),
                QJsonArray{QJsonArray{2, 2, 10, 2, 10, 8, 2, 8}});
        }
        annotations.append(annotation);
    }
    const QJsonObject coco{{QStringLiteral("images"), images},
        {QStringLiteral("categories"), QJsonArray{QJsonObject{{QStringLiteral("id"), 7}, {QStringLiteral("name"), QStringLiteral("零件")}}}},
        {QStringLiteral("annotations"), annotations}};
    const QString path = QDir(root).filePath(QStringLiteral("标注 文件.json"));
    if (!writeUtf8(path, QString::fromUtf8(QJsonDocument(coco).toJson(QJsonDocument::Compact)))) qFatal("write coco failed");
    return path;
}

QString createVocFixture(const QString& root)
{
    const QString annotations = QDir(root).filePath(QStringLiteral("Annotations"));
    const QString images = QDir(root).filePath(QStringLiteral("JPEGImages"));
    if (!writeImage(QDir(images).filePath(QStringLiteral("零件 a.png")), 17)) qFatal("write VOC image failed");
    const QString xml = QStringLiteral(
        "<annotation><filename>零件 a.png</filename><size><width>16</width><height>12</height></size>"
        "<object><name>缺陷</name><bndbox><xmin>2</xmin><ymin>2</ymin><xmax>10</xmax><ymax>8</ymax>"
        "</bndbox></object></annotation>");
    if (!writeUtf8(QDir(annotations).filePath(QStringLiteral("零件 a.xml")), xml)) qFatal("write VOC XML failed");
    return annotations;
}

struct Fixture final {
    QTemporaryDir directory;
    aitrain::ProjectStore storage;
    std::unique_ptr<aitrain::ArtifactStore> artifacts;
    aitrain::DatasetDriverRegistry drivers;

    Fixture()
    {
        if (!directory.isValid()) qFatal("temporary directory unavailable");
        QString error;
        if (!storage.open(directory.filePath(QStringLiteral("workspace.sqlite")), &error)) qFatal("storage open failed: %s", qPrintable(error));
        artifacts = std::make_unique<aitrain::ArtifactStore>(directory.filePath(QStringLiteral("artifact-store")));
        if (!aitrain::registerBuiltinDatasetDrivers(&drivers, &error)) qFatal("drivers failed: %s", qPrintable(error));
    }
};

aitrain::DatasetArtifactConversionRequest requestFor(const QString& sourcePath)
{
    aitrain::DatasetArtifactConversionRequest request;
    request.sourcePath = sourcePath;
    request.sourceFormat = QStringLiteral("coco_json");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.options.insert(QStringLiteral("copyImages"), true);
    return request;
}

aitrain::DatasetArtifactConversionRequest requestForPair(const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetFormat)
{
    auto request = requestFor(sourcePath);
    request.sourceFormat = sourceFormat;
    request.targetFormat = targetFormat;
    return request;
}

} // namespace

class DatasetConversionTests final : public QObject {
    Q_OBJECT

private slots:
    void supportedEntryPoints_data();
    void supportedEntryPoints();
    void rejectedRoutesAreExplicitBackendUnsupported_data();
    void rejectedRoutesAreExplicitBackendUnsupported();
    void commitsOnlyAfterTargetDriverValidation();
    void targetConflictIsRejectedBeforeMaterialization();
    void sourceChangeAfterPlanLeavesNoArtifact();
    void vocImageChangeAfterPlanLeavesNoArtifact();
    void imageOutsideFrozenSourceRootLeavesNoArtifact();
    void injectedWriteFailureLeavesNoArtifact();
    void unplannedOutputIsRejected();
    void cancellationDuringCopyLeavesNoArtifact();
    void sourceAndArtifactRootOverlapIsRejected();
    void manySmallFilesAreFrozenAndCommitted();
    void sparseLargeFileIsHashedWithoutLargeAllocation();
    void portableRelativePathBoundaryIsDeterministic();
    void injectedIoFailuresLeaveNoArtifact_data();
    void injectedIoFailuresLeaveNoArtifact();
    void abandonedStagingIsRecoveredWithoutArtifact();
};

void DatasetConversionTests::supportedEntryPoints_data()
{
    QTest::addColumn<QString>("sourceFormat");
    QTest::addColumn<QString>("targetFormat");
    QTest::newRow("coco-detection") << QStringLiteral("coco_json") << QStringLiteral("yolo_detection");
    QTest::newRow("coco-segmentation") << QStringLiteral("coco_json") << QStringLiteral("yolo_segmentation");
    QTest::newRow("voc-detection") << QStringLiteral("voc_xml") << QStringLiteral("yolo_detection");
}

void DatasetConversionTests::supportedEntryPoints()
{
    QFETCH(QString, sourceFormat);
    QFETCH(QString, targetFormat);
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = sourceFormat == QStringLiteral("voc_xml")
        ? createVocFixture(fixture.directory.filePath(QStringLiteral("voc-source")))
        : createCocoFixture(fixture.directory.filePath(QStringLiteral("coco-source")), 1, false,
              targetFormat == QStringLiteral("yolo_segmentation"));
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY2(service.convert(task.id, requestForPair(source, sourceFormat, targetFormat), &result, &error), qPrintable(error));
    QVERIFY(result.targetValidation.valid);
    QCOMPARE(result.plan.value(QStringLiteral("sourceFormat")).toString(), sourceFormat);
    QCOMPARE(result.plan.value(QStringLiteral("targetFormat")).toString(), targetFormat);
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 1);
}

void DatasetConversionTests::rejectedRoutesAreExplicitBackendUnsupported_data()
{
    QTest::addColumn<QString>("sourceFormat");
    QTest::addColumn<QString>("targetFormat");
    QTest::addColumn<QString>("reason");
    QTest::newRow("voc-segmentation") << QStringLiteral("voc_xml") << QStringLiteral("yolo_segmentation")
                                        << QStringLiteral("voc_has_no_polygon_semantics");
    QTest::newRow("yolo-coco") << QStringLiteral("yolo_detection") << QStringLiteral("coco_json")
                                 << QStringLiteral("coco_or_voc_target_has_no_driver");
    QTest::newRow("yolo-voc") << QStringLiteral("yolo_detection") << QStringLiteral("voc_xml")
                                << QStringLiteral("coco_or_voc_target_has_no_driver");
    QTest::newRow("yolo-xlabel") << QStringLiteral("yolo_segmentation") << QStringLiteral("xanylabeling_xlabel")
                                   << QStringLiteral("external_cli_output_cannot_be_frozen_before_materialize");
    QTest::newRow("xlabel-yolo") << QStringLiteral("xanylabeling_xlabel") << QStringLiteral("yolo_obb")
                                   << QStringLiteral("external_cli_output_cannot_be_frozen_before_materialize");
    QTest::newRow("ocr-cross-family") << QStringLiteral("paddleocr_rec") << QStringLiteral("yolo_detection")
                                       << QStringLiteral("no_target_driver");
}

void DatasetConversionTests::rejectedRoutesAreExplicitBackendUnsupported()
{
    QFETCH(QString, sourceFormat);
    QFETCH(QString, targetFormat);
    QFETCH(QString, reason);
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = fixture.directory.filePath(QStringLiteral("unsupported-source"));
    QVERIFY(QDir().mkpath(source));
    QVERIFY(writeUtf8(QDir(source).filePath(QStringLiteral("input.txt")), QStringLiteral("input")));
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestForPair(source, sourceFormat, targetFormat), &result, &error));
    QVERIFY2(error.startsWith(QStringLiteral("dataset_conversion_backend_unsupported:")), qPrintable(error));
    QVERIFY2(error.endsWith(reason), qPrintable(error));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
    QVERIFY(!QDir(fixture.directory.filePath(QStringLiteral("artifact-store/.staging"))).exists());
}

void DatasetConversionTests::commitsOnlyAfterTargetDriverValidation()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("含 空格的源")), 2);
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY2(service.convert(task.id, requestFor(source), &result, &error), qPrintable(error));
    QVERIFY(result.targetValidation.valid);
    QVERIFY(QFileInfo::exists(result.planPath));
    QVERIFY(QFileInfo::exists(result.conversionReportPath));
    QVERIFY(QFileInfo::exists(QDir(result.artifactPath).filePath(QStringLiteral("data.yaml"))));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 1);
    aitrain::ArtifactSnapshot artifact;
    QVERIFY(fixture.storage.artifact(result.artifactId, &artifact, &error));
    QCOMPARE(artifact.kind, QStringLiteral("dataset_conversion"));
    QVERIFY(artifact.files.size() >= 7);
    QCOMPARE(result.plan.value(QStringLiteral("overwritePolicy")).toString(), QStringLiteral("reject"));
}

void DatasetConversionTests::targetConflictIsRejectedBeforeMaterialization()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("conflict-source")), 2, true);
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
    QVERIFY2(error.startsWith(QStringLiteral("dataset_conversion_target_conflict:")), qPrintable(error));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
    QCOMPARE(QDir(fixture.directory.filePath(QStringLiteral("artifact-store/.staging")))
                 .entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 0);
}

void DatasetConversionTests::sourceChangeAfterPlanLeavesNoArtifact()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("source-change")), 1);
    bool changed = false;
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers,
        [&](aitrain::DatasetConversionFailPoint point, const QString&) {
            if (point == aitrain::DatasetConversionFailPoint::AfterPlan) {
                changed = writeUtf8(source, QStringLiteral("{}"));
            }
            return false;
        });
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
    QVERIFY(changed);
    QCOMPARE(error, QStringLiteral("dataset_conversion_source_changed_after_plan"));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
}

void DatasetConversionTests::vocImageChangeAfterPlanLeavesNoArtifact()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString root = fixture.directory.filePath(QStringLiteral("voc-source-change"));
    const QString source = createVocFixture(root);
    const QString imagePath = QDir(root).filePath(QStringLiteral("JPEGImages/零件 a.png"));
    bool changed = false;
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers,
        [&](aitrain::DatasetConversionFailPoint point, const QString&) {
            if (point == aitrain::DatasetConversionFailPoint::AfterPlan) {
                changed = writeUtf8(imagePath, QStringLiteral("changed"));
            }
            return false;
        });
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id,
        requestForPair(source, QStringLiteral("voc_xml"), QStringLiteral("yolo_detection")),
        &result, &error));
    QVERIFY(changed);
    QCOMPARE(error, QStringLiteral("dataset_conversion_source_changed_after_plan"));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
}

void DatasetConversionTests::imageOutsideFrozenSourceRootLeavesNoArtifact()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString sourceRoot = fixture.directory.filePath(QStringLiteral("frozen-source"));
    const QString outsideImage = fixture.directory.filePath(QStringLiteral("outside/样本.png"));
    QVERIFY(writeImage(outsideImage, 23));
    const QJsonObject coco{
        {QStringLiteral("images"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
             {QStringLiteral("file_name"), outsideImage}, {QStringLiteral("width"), 16}, {QStringLiteral("height"), 12}}}},
        {QStringLiteral("categories"), QJsonArray{QJsonObject{{QStringLiteral("id"), 7},
             {QStringLiteral("name"), QStringLiteral("零件")}}}},
        {QStringLiteral("annotations"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
             {QStringLiteral("image_id"), 1}, {QStringLiteral("category_id"), 7},
             {QStringLiteral("bbox"), QJsonArray{2, 2, 8, 6}}}}}};
    const QString source = QDir(sourceRoot).filePath(QStringLiteral("annotations.json"));
    QVERIFY(writeUtf8(source, QString::fromUtf8(QJsonDocument(coco).toJson(QJsonDocument::Compact))));
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
    QVERIFY2(error.startsWith(QStringLiteral("dataset_conversion_source_image_not_frozen:")), qPrintable(error));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
}

void DatasetConversionTests::injectedWriteFailureLeavesNoArtifact()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("write-failure")), 2);
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers,
        [](aitrain::DatasetConversionFailPoint point, const QString&) {
            return point == aitrain::DatasetConversionFailPoint::AfterMaterialize;
        });
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
    QCOMPARE(error, QStringLiteral("dataset_conversion_injected_failure"));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
    QVERIFY(!QDir(fixture.directory.filePath(QStringLiteral("artifact-store/.staging"))).exists()
        || QDir(fixture.directory.filePath(QStringLiteral("artifact-store/.staging")))
               .entryList(QDir::Dirs | QDir::NoDotAndDotDot).isEmpty());
}

void DatasetConversionTests::unplannedOutputIsRejected()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("unplanned-output")), 1);
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers,
        [](aitrain::DatasetConversionFailPoint point, const QString& staging) {
            if (point == aitrain::DatasetConversionFailPoint::AfterMaterialize) {
                writeUtf8(QDir(staging).filePath(QStringLiteral("未计划 文件.bin")), QStringLiteral("rogue"));
            }
            return false;
        });
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
    QVERIFY2(error.startsWith(QStringLiteral("dataset_conversion_output_plan_mismatch:")), qPrintable(error));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
}

void DatasetConversionTests::cancellationDuringCopyLeavesNoArtifact()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("cancel-source")), 30);
    bool planWritten = false;
    int pollsAfterPlan = 0;
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers,
        [&](aitrain::DatasetConversionFailPoint point, const QString&) {
            if (point == aitrain::DatasetConversionFailPoint::AfterPlan) planWritten = true;
            return false;
        });
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error, [&]() {
        // plan 后完整源复核约消耗 62 次轮询；再允许转换器复制若干样本后取消。
        return planWritten && ++pollsAfterPlan > 75;
    }));
    QVERIFY(error.contains(QStringLiteral("canceled")));
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
}

void DatasetConversionTests::sourceAndArtifactRootOverlapIsRejected()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY(storage.open(directory.filePath(QStringLiteral("workspace.sqlite")), &error));
    const auto task = createTask(&storage);
    const QString sourceRoot = directory.filePath(QStringLiteral("source"));
    const QString source = createCocoFixture(sourceRoot, 1);
    aitrain::ArtifactStore artifacts(QDir(sourceRoot).filePath(QStringLiteral("nested-artifact-store")));
    aitrain::DatasetDriverRegistry drivers;
    QVERIFY(aitrain::registerBuiltinDatasetDrivers(&drivers, &error));
    aitrain::DatasetConversionService service(&artifacts, &storage, &drivers);
    aitrain::DatasetArtifactConversion result;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
    QCOMPARE(error, QStringLiteral("dataset_conversion_source_artifact_root_overlap"));
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
    QVERIFY(!QFileInfo::exists(QDir(sourceRoot).filePath(QStringLiteral("nested-artifact-store/.staging"))));
}

void DatasetConversionTests::manySmallFilesAreFrozenAndCommitted()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString root = fixture.directory.filePath(QStringLiteral("many-small-files"));
    const QString source = createCocoFixture(root, 16);
    for (int index = 0; index < 192; ++index) {
        QVERIFY(writeUtf8(QDir(root).filePath(QStringLiteral("sidecars/批次-%1/meta-%2.txt")
            .arg(index % 8).arg(index)), QStringLiteral("frozen-%1").arg(index)));
    }
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY2(service.convert(task.id, requestFor(source), &result, &error), qPrintable(error));
    QVERIFY(result.plan.value(QStringLiteral("sourceFiles")).toArray().size() >= 209);
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 1);
}

void DatasetConversionTests::sparseLargeFileIsHashedWithoutLargeAllocation()
{
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString root = fixture.directory.filePath(QStringLiteral("sparse-large-file"));
    const QString source = createCocoFixture(root, 1);
    constexpr qint64 kSparseBytes = 32LL * 1024LL * 1024LL;
    const QString sparseRelative = QStringLiteral("sidecars/large-sparse.bin");
    QVERIFY(createSparseFile(QDir(root).filePath(sparseRelative), kSparseBytes));

    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY2(service.convert(task.id, requestFor(source), &result, &error), qPrintable(error));
    bool found = false;
    for (const QJsonValue& value : result.plan.value(QStringLiteral("sourceFiles")).toArray()) {
        const QJsonObject file = value.toObject();
        if (file.value(QStringLiteral("relativePath")).toString() == sparseRelative) {
            QCOMPARE(file.value(QStringLiteral("bytes")).toString(), QString::number(kSparseBytes));
            QCOMPARE(file.value(QStringLiteral("sha256")).toString().size(), 64);
            found = true;
        }
    }
    QVERIFY(found);
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 1);
}

void DatasetConversionTests::portableRelativePathBoundaryIsDeterministic()
{
    {
        Fixture fixture;
        const auto task = createTask(&fixture.storage);
        const QString allowedName = QString(120, QLatin1Char('a')) + QStringLiteral(".png");
        const QString source = createCocoFixtureWithFileName(
            fixture.directory.filePath(QStringLiteral("portable-allowed")),
            QStringLiteral("images/%1").arg(allowedName));
        aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
        aitrain::DatasetArtifactConversion result;
        QString error;
        QVERIFY2(service.convert(task.id, requestFor(source), &result, &error), qPrintable(error));
        QCOMPARE(fixture.storage.artifactCount(task.id, &error), 1);
    }
    {
        Fixture fixture;
        const auto task = createTask(&fixture.storage);
        const QString rejectedRelative = QStringLiteral("images/%1.png").arg(QString(161, QLatin1Char('b')));
        const QJsonObject coco{
            {QStringLiteral("images"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
                 {QStringLiteral("file_name"), rejectedRelative}, {QStringLiteral("width"), 16}, {QStringLiteral("height"), 12}}}},
            {QStringLiteral("categories"), QJsonArray{QJsonObject{{QStringLiteral("id"), 7},
                 {QStringLiteral("name"), QStringLiteral("零件")}}}},
            {QStringLiteral("annotations"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
                 {QStringLiteral("image_id"), 1}, {QStringLiteral("category_id"), 7},
                 {QStringLiteral("bbox"), QJsonArray{2, 2, 8, 6}}}}}};
        const QString source = QDir(fixture.directory.filePath(QStringLiteral("portable-rejected")))
                                   .filePath(QStringLiteral("annotations.json"));
        QVERIFY(writeUtf8(source, QString::fromUtf8(QJsonDocument(coco).toJson(QJsonDocument::Compact))));
        aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
        aitrain::DatasetArtifactConversion result;
        QString error;
        QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
        QVERIFY2(error.startsWith(QStringLiteral("dataset_conversion_relative_path_too_long:")), qPrintable(error));
        QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
    }
}

void DatasetConversionTests::injectedIoFailuresLeaveNoArtifact_data()
{
    QTest::addColumn<int>("operation");
    QTest::addColumn<QString>("failureCode");
    QTest::newRow("disk-full-during-materialize")
        << static_cast<int>(aitrain::DatasetConversionIoOperation::MaterializeWrite)
        << QStringLiteral("dataset_conversion_io_disk_full");
    QTest::newRow("report-target-locked")
        << static_cast<int>(aitrain::DatasetConversionIoOperation::ReportWrite)
        << QStringLiteral("dataset_conversion_io_target_locked");
    QTest::newRow("commit-target-busy")
        << static_cast<int>(aitrain::DatasetConversionIoOperation::Commit)
        << QStringLiteral("dataset_conversion_io_commit_target_busy");
}

void DatasetConversionTests::injectedIoFailuresLeaveNoArtifact()
{
    QFETCH(int, operation);
    QFETCH(QString, failureCode);
    Fixture fixture;
    const auto task = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("io-fault")), 2);
    const auto selected = static_cast<aitrain::DatasetConversionIoOperation>(operation);
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers,
        {}, [selected, failureCode](aitrain::DatasetConversionIoOperation point, const QString&) {
            return point == selected ? failureCode : QString();
        });
    aitrain::DatasetArtifactConversion result;
    QString error;
    QVERIFY(!service.convert(task.id, requestFor(source), &result, &error));
    QCOMPARE(error, failureCode);
    QCOMPARE(fixture.storage.artifactCount(task.id, &error), 0);
    const QDir staging(fixture.directory.filePath(QStringLiteral("artifact-store/.staging")));
    QVERIFY(!staging.exists() || staging.entryList(QDir::Dirs | QDir::NoDotAndDotDot).isEmpty());
}

void DatasetConversionTests::abandonedStagingIsRecoveredWithoutArtifact()
{
    Fixture fixture;
    QString error;
    const auto committedTask = createTask(&fixture.storage);
    const QString source = createCocoFixture(fixture.directory.filePath(QStringLiteral("committed-source")), 1);
    aitrain::DatasetConversionService service(fixture.artifacts.get(), &fixture.storage, &fixture.drivers);
    aitrain::DatasetArtifactConversion committed;
    QVERIFY2(service.convert(committedTask.id, requestFor(source), &committed, &error), qPrintable(error));
    QVERIFY(QFileInfo::exists(committed.artifactPath));

    const auto task = createTask(&fixture.storage);
    aitrain::ArtifactId artifactId;
    QString staging;
    QVERIFY(fixture.artifacts->begin(task.id, QStringLiteral("dataset_conversion"), &artifactId, &staging, &error));
    QVERIFY(writeUtf8(QDir(staging).filePath(QStringLiteral("partial.bin")), QStringLiteral("partial")));
    QVERIFY(fixture.storage.transitionTask(task.id, aitrain::TaskState::Created,
        aitrain::TaskState::Failed,
        {aitrain::FailureCode::ProcessCrashed, QStringLiteral("worker_killed"),
            QStringLiteral("清理中断的转换 staging。"), QDateTime::currentDateTimeUtc()}, &error));

    const auto activeTask = createTask(&fixture.storage);
    aitrain::ArtifactId activeArtifactId;
    QString activeStaging;
    QVERIFY(fixture.artifacts->begin(activeTask.id, QStringLiteral("dataset_conversion"),
        &activeArtifactId, &activeStaging, &error));
    QVERIFY(writeUtf8(QDir(activeStaging).filePath(QStringLiteral("owned-by-running-task.bin")),
        QStringLiteral("keep")));

    QStringList diagnostics;
    QVERIFY2(fixture.artifacts->recoverStaging(&fixture.storage, &diagnostics, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(staging));
    bool exists = true;
    QVERIFY(fixture.storage.artifactExists(artifactId, &exists, &error));
    QVERIFY(!exists);
    QVERIFY(QFileInfo::exists(activeStaging));
    QVERIFY(QFileInfo::exists(committed.artifactPath));
    QVERIFY(fixture.storage.artifactExists(committed.artifactId, &exists, &error));
    QVERIFY(exists);
    QVERIFY(!diagnostics.isEmpty());
}

QTEST_MAIN(DatasetConversionTests)
#include "tst_dataset_conversion.moc"
