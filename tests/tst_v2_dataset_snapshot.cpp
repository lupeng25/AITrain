#include "aitrain/v2/DatasetSnapshotV2.h"
#include "aitrain/v2/DatasetDriverV2.h"
#include "aitrain/v2/AnomalyFolderDatasetDriverV2.h"
#include "aitrain/v2/BuiltinDatasetDriversV2.h"
#include "aitrain/v2/PaddleOcrDetDatasetDriverV2.h"
#include "aitrain/v2/PaddleOcrRecDatasetDriverV2.h"
#include "aitrain/v2/YoloDetectionDatasetDriverV2.h"
#include "aitrain/v2/YoloObbDatasetDriverV2.h"
#include "aitrain/v2/YoloSegmentationDatasetDriverV2.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QTemporaryDir>
#include <QTest>

#include <utility>

namespace {

bool writeFile(const QString& path, const QByteArray& content)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        return false;
    }
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(content) == content.size();
}

bool createSnapshot(const QString& root, const QString& output, aitrain::v2::DatasetSnapshotResult* result, QString* error, const aitrain::v2::DatasetSnapshotOptions& options = {})
{
    return aitrain::v2::createDatasetSnapshotV2(root, output, QStringLiteral("yolo_detection"),
        QStringLiteral("yolo_detection"), QStringLiteral("2.0"), options, result, error);
}

bool writeTinyImage(const QString& path)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        return false;
    }
    QImage image(8, 8, QImage::Format_RGB32);
    image.fill(Qt::white);
    return image.save(path);
}

bool exerciseDriver(const aitrain::v2::DatasetDriverV2& driver, const QString& root, const QString& outputRoot, QString* error)
{
    aitrain::v2::DatasetOperationContext context;
    aitrain::v2::DatasetInspection inspection;
    if (!driver.detect(root, &inspection, context, error)) {
        return false;
    }
    aitrain::v2::DatasetValidationResult validation;
    if (!driver.validate(inspection, &validation, context, error) || !validation.valid) {
        return false;
    }
    QJsonObject options{{QStringLiteral("trainRatio"), 1.0}, {QStringLiteral("valRatio"), 0.0},
        {QStringLiteral("testRatio"), 0.0}, {QStringLiteral("seed"), 17}};
    aitrain::v2::DatasetSplitPlan first;
    aitrain::v2::DatasetSplitPlan second;
    if (!driver.planSplit(inspection, options, &first, context, error)
        || !driver.planSplit(inspection, options, &second, context, error)
        || first.planHash != second.planHash) {
        return false;
    }
    const QJsonArray entries = first.manifest.value(QStringLiteral("entries")).toArray();
    if (entries.isEmpty()) {
        return false;
    }
    for (const QJsonValue& value : entries) {
        const QJsonObject entry = value.toObject();
        if (entry.value(QStringLiteral("targetRelativePath")).toString().isEmpty()
            || entry.value(QStringLiteral("sha256")).toString().isEmpty()
            || (!entry.contains(QStringLiteral("sourceRelativePath")) && !entry.contains(QStringLiteral("inlineBase64")))) {
            return false;
        }
    }
    const QString staging = QDir(outputRoot).filePath(driver.id() + QStringLiteral("-staging"));
    if (!driver.materializeSplit(first, staging, context, error)
        || !QFileInfo::exists(QDir(staging).filePath(QStringLiteral("split_plan_v2.json")))) {
        return false;
    }
    aitrain::v2::DatasetSnapshotResult snapshot;
    return driver.snapshot(inspection, QDir(outputRoot).filePath(driver.id() + QStringLiteral("-snapshot.json")), {}, &snapshot, error)
        && snapshot.manifest.value(QStringLiteral("complete")).toBool()
        && snapshot.manifest.value(QStringLiteral("driver")).toObject().value(QStringLiteral("id")).toString() == driver.id();
}

} // namespace

class V2DatasetSnapshotTests : public QObject {
    Q_OBJECT

private slots:
    void createsCompleteDeterministicManifest();
    void rejectsFileLimitWithoutWritingManifest();
    void cancellationDoesNotWriteManifest();
    void datasetDriverRegistryRejectsOverlappingFormats();
    void semanticMaskDriverPreservesPaletteIndexes();
    void migratedDriversProvideDeterministicV2Contract();
    void migratedDriverRejectsChangedSourceAndUnsafeStaging();
    void migratedDriverDetectsInvalidRecognizableLayout();
};

class TestDatasetDriver final : public aitrain::v2::DatasetDriverV2 {
public:
    explicit TestDatasetDriver(QStringList formats)
        : formats_(std::move(formats))
    {
    }

    QString id() const override { return QStringLiteral("test_driver"); }
    QString version() const override { return QStringLiteral("2.0"); }
    QStringList supportedFormats() const override { return formats_; }
    bool detect(const QString&, aitrain::v2::DatasetInspection*, const aitrain::v2::DatasetOperationContext&, QString*) const override { return false; }
    bool inspect(const QString&, const QString&, aitrain::v2::DatasetInspection*, const aitrain::v2::DatasetOperationContext&, QString*) const override { return false; }
    bool validate(const aitrain::v2::DatasetInspection&, aitrain::v2::DatasetValidationResult*, const aitrain::v2::DatasetOperationContext&, QString*) const override { return false; }
    bool planSplit(const aitrain::v2::DatasetInspection&, const QJsonObject&, aitrain::v2::DatasetSplitPlan*, const aitrain::v2::DatasetOperationContext&, QString*) const override { return false; }
    bool materializeSplit(const aitrain::v2::DatasetSplitPlan&, const QString&, const aitrain::v2::DatasetOperationContext&, QString*) const override { return false; }
    bool snapshot(const aitrain::v2::DatasetInspection&, const QString&, const aitrain::v2::DatasetSnapshotOptions&, aitrain::v2::DatasetSnapshotResult*, QString*) const override { return false; }

private:
    QStringList formats_;
};

void V2DatasetSnapshotTests::createsCompleteDeterministicManifest()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("dataset"));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("images/a.jpg")), QByteArray("image-a")));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("labels/a.txt")), QByteArray("0 0.5 0.5 1 1")));
    aitrain::v2::DatasetSnapshotResult first;
    QString error;
    QVERIFY2(createSnapshot(root, directory.filePath(QStringLiteral("first.json")), &first, &error), qPrintable(error));
    QVERIFY(first.manifest.value(QStringLiteral("complete")).toBool());
    QCOMPARE(first.fileCount, static_cast<qsizetype>(2));
    QCOMPARE(first.manifest.value(QStringLiteral("files")).toArray().size(), 2);

    aitrain::v2::DatasetSnapshotResult second;
    QVERIFY2(createSnapshot(root, directory.filePath(QStringLiteral("second.json")), &second, &error), qPrintable(error));
    QCOMPARE(second.rootHash, first.rootHash);
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("labels/a.txt")), QByteArray("0 0.4 0.4 1 1")));
    aitrain::v2::DatasetSnapshotResult changed;
    QVERIFY2(createSnapshot(root, directory.filePath(QStringLiteral("changed.json")), &changed, &error), qPrintable(error));
    QVERIFY(changed.rootHash != first.rootHash);
}

void V2DatasetSnapshotTests::rejectsFileLimitWithoutWritingManifest()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("dataset"));
    for (int index = 0; index < 20001; ++index) {
        QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("samples/%1.txt").arg(index, 5, 10, QLatin1Char('0'))), QByteArray("x")));
    }
    aitrain::v2::DatasetSnapshotOptions options;
    options.maxFileCount = 20000;
    aitrain::v2::DatasetSnapshotResult result;
    QString error;
    const QString manifestPath = directory.filePath(QStringLiteral("limit.json"));
    QVERIFY(!createSnapshot(root, manifestPath, &result, &error, options));
    QVERIFY(error.contains(QStringLiteral("file_limit_exceeded")));
    QVERIFY(!QFileInfo::exists(manifestPath));
}

void V2DatasetSnapshotTests::cancellationDoesNotWriteManifest()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("dataset"));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("one.txt")), QByteArray("one")));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("two.txt")), QByteArray("two")));
    int polls = 0;
    aitrain::v2::DatasetSnapshotOptions options;
    options.isCancellationRequested = [&polls]() { return ++polls > 2; };
    aitrain::v2::DatasetSnapshotResult result;
    QString error;
    const QString manifestPath = directory.filePath(QStringLiteral("canceled.json"));
    QVERIFY(!createSnapshot(root, manifestPath, &result, &error, options));
    QCOMPARE(error, QStringLiteral("snapshot_canceled"));
    QVERIFY(!QFileInfo::exists(manifestPath));
}

void V2DatasetSnapshotTests::datasetDriverRegistryRejectsOverlappingFormats()
{
    aitrain::v2::DatasetDriverRegistryV2 registry;
    TestDatasetDriver yoloDriver({QStringLiteral("yolo_detection"), QStringLiteral("yolo_obb")});
    TestDatasetDriver conflictingDriver({QStringLiteral("yolo_detection")});
    QString error;
    QVERIFY2(registry.registerDriver(&yoloDriver, &error), qPrintable(error));
    QVERIFY(registry.driverForFormat(QStringLiteral("YOLO_DETECTION")) == static_cast<const aitrain::v2::DatasetDriverV2*>(&yoloDriver));
    QVERIFY(!registry.registerDriver(&conflictingDriver, &error));
    QVERIFY(error.contains(QStringLiteral("已注册")));
    QCOMPARE(registry.formats(), QStringList({QStringLiteral("yolo_detection"), QStringLiteral("yolo_obb")}));
}

void V2DatasetSnapshotTests::semanticMaskDriverPreservesPaletteIndexes()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("semantic"));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("classes.txt")), QByteArray("background\nobject\n")));
    QImage image(4, 4, QImage::Format_RGB32);
    image.fill(Qt::white);
    QVERIFY(QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train"))));
    QVERIFY(image.save(QDir(root).filePath(QStringLiteral("images/train/sample.png"))));
    QImage mask(4, 4, QImage::Format_Indexed8);
    QVector<QRgb> palette(2, qRgb(0, 0, 0));
    palette[1] = qRgb(255, 0, 0);
    mask.setColorTable(palette);
    mask.fill(1);
    QVERIFY(QDir().mkpath(QDir(root).filePath(QStringLiteral("masks/train"))));
    QVERIFY(mask.save(QDir(root).filePath(QStringLiteral("masks/train/sample.png"))));

    aitrain::v2::SemanticMaskDatasetDriverV2 driver;
    aitrain::v2::DatasetInspection inspection;
    aitrain::v2::DatasetOperationContext context;
    QString error;
    QVERIFY2(driver.detect(root, &inspection, context, &error), qPrintable(error));
    QCOMPARE(inspection.sampleCount, static_cast<qsizetype>(1));
    aitrain::v2::DatasetValidationResult validation;
    QVERIFY2(driver.validate(inspection, &validation, context, &error), qPrintable(error));
    QVERIFY(validation.valid);

    QJsonObject splitOptions;
    splitOptions.insert(QStringLiteral("trainRatio"), 1.0);
    splitOptions.insert(QStringLiteral("valRatio"), 0.0);
    splitOptions.insert(QStringLiteral("testRatio"), 0.0);
    splitOptions.insert(QStringLiteral("seed"), QStringLiteral("fixture-seed"));
    aitrain::v2::DatasetSplitPlan firstPlan;
    aitrain::v2::DatasetSplitPlan secondPlan;
    QVERIFY2(driver.planSplit(inspection, splitOptions, &firstPlan, context, &error), qPrintable(error));
    QVERIFY2(driver.planSplit(inspection, splitOptions, &secondPlan, context, &error), qPrintable(error));
    QCOMPARE(firstPlan.planHash, secondPlan.planHash);
    QVERIFY(!firstPlan.manifest.contains(QStringLiteral("sourceRoot")));
    const QJsonObject entry = firstPlan.manifest.value(QStringLiteral("entries")).toArray().at(0).toObject();
    QVERIFY(!entry.contains(QStringLiteral("sourceImage")));
    QVERIFY(!entry.contains(QStringLiteral("sourceMask")));
    QVERIFY(!QDir::isAbsolutePath(entry.value(QStringLiteral("sourceImageRelativePath")).toString()));
    QVERIFY(!QDir::isAbsolutePath(entry.value(QStringLiteral("sourceMaskRelativePath")).toString()));
    QCOMPARE(entry.value(QStringLiteral("sourceImageSha256")).toString().size(), 64);
    QCOMPARE(entry.value(QStringLiteral("sourceMaskSha256")).toString().size(), 64);
    QVERIFY(entry.value(QStringLiteral("sourceImageBytes")).toString().toLongLong() > 0);
    QVERIFY(entry.value(QStringLiteral("sourceMaskBytes")).toString().toLongLong() > 0);
    const QString stagingPath = directory.filePath(QStringLiteral("staging"));
    QVERIFY2(driver.materializeSplit(firstPlan, stagingPath, context, &error), qPrintable(error));
    QVERIFY(QFileInfo::exists(QDir(stagingPath).filePath(QStringLiteral("images/train/sample.png"))));
    QVERIFY(QFileInfo::exists(QDir(stagingPath).filePath(QStringLiteral("masks/train/sample.png"))));
    QVERIFY(QFileInfo::exists(QDir(stagingPath).filePath(QStringLiteral("split_plan.json"))));

    aitrain::v2::DatasetSnapshotResult snapshot;
    QVERIFY2(driver.snapshot(inspection, directory.filePath(QStringLiteral("semantic-snapshot.json")), {}, &snapshot, &error), qPrintable(error));
    QCOMPARE(snapshot.manifest.value(QStringLiteral("driver")).toObject().value(QStringLiteral("id")).toString(), QStringLiteral("semantic_mask"));
    QCOMPARE(snapshot.manifest.value(QStringLiteral("classDefinitions")).toArray().size(), 2);
}

void V2DatasetSnapshotTests::migratedDriversProvideDeterministicV2Contract()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    QString error;

    const QString detectionRoot = directory.filePath(QStringLiteral("detection"));
    QVERIFY(writeFile(QDir(detectionRoot).filePath(QStringLiteral("data.yaml")), QByteArray("nc: 1\nnames: [item]\n")));
    QVERIFY(writeTinyImage(QDir(detectionRoot).filePath(QStringLiteral("images/train/a.png"))));
    QVERIFY(writeTinyImage(QDir(detectionRoot).filePath(QStringLiteral("images/val/b.png"))));
    QVERIFY(writeFile(QDir(detectionRoot).filePath(QStringLiteral("labels/train/a.txt")), QByteArray("0 0.5 0.5 0.25 0.25\n")));
    QVERIFY(writeFile(QDir(detectionRoot).filePath(QStringLiteral("labels/val/b.txt")), QByteArray("0 0.5 0.5 0.25 0.25\n")));
    aitrain::v2::YoloDetectionDatasetDriverV2 detection;
    QVERIFY2(exerciseDriver(detection, detectionRoot, directory.path(), &error), qPrintable(error));

    const QString segmentationRoot = directory.filePath(QStringLiteral("segmentation"));
    QVERIFY(writeFile(QDir(segmentationRoot).filePath(QStringLiteral("data.yaml")), QByteArray("nc: 1\nnames: [part]\n")));
    QVERIFY(writeTinyImage(QDir(segmentationRoot).filePath(QStringLiteral("images/train/a.png"))));
    QVERIFY(writeTinyImage(QDir(segmentationRoot).filePath(QStringLiteral("images/val/b.png"))));
    const QByteArray polygon("0 0.1 0.1 0.8 0.1 0.9 0.5 0.5 0.9 0.1 0.7\n");
    QVERIFY(writeFile(QDir(segmentationRoot).filePath(QStringLiteral("labels/train/a.txt")), polygon));
    QVERIFY(writeFile(QDir(segmentationRoot).filePath(QStringLiteral("labels/val/b.txt")), polygon));
    aitrain::v2::YoloSegmentationDatasetDriverV2 segmentation;
    QVERIFY2(exerciseDriver(segmentation, segmentationRoot, directory.path(), &error), qPrintable(error));

    const QString obbRoot = directory.filePath(QStringLiteral("obb"));
    QVERIFY(writeFile(QDir(obbRoot).filePath(QStringLiteral("data.yaml")), QByteArray("task: obb\nnc: 1\nnames: [ship]\n")));
    QVERIFY(writeTinyImage(QDir(obbRoot).filePath(QStringLiteral("images/train/a.png"))));
    QVERIFY(writeTinyImage(QDir(obbRoot).filePath(QStringLiteral("images/val/b.png"))));
    const QByteArray quad("0 0.2 0.2 0.8 0.2 0.8 0.7 0.2 0.7\n");
    QVERIFY(writeFile(QDir(obbRoot).filePath(QStringLiteral("labels/train/a.txt")), quad));
    QVERIFY(writeFile(QDir(obbRoot).filePath(QStringLiteral("labels/val/b.txt")), quad));
    aitrain::v2::YoloObbDatasetDriverV2 obb;
    QVERIFY2(exerciseDriver(obb, obbRoot, directory.path(), &error), qPrintable(error));

    const QString anomalyRoot = directory.filePath(QStringLiteral("anomaly"));
    QVERIFY(writeTinyImage(QDir(anomalyRoot).filePath(QStringLiteral("train/good/a.png"))));
    QVERIFY(writeTinyImage(QDir(anomalyRoot).filePath(QStringLiteral("test/good/b.png"))));
    aitrain::v2::AnomalyFolderDatasetDriverV2 anomaly;
    QVERIFY2(exerciseDriver(anomaly, anomalyRoot, directory.path(), &error), qPrintable(error));

    const QString detRoot = directory.filePath(QStringLiteral("ocr-det"));
    QVERIFY(writeTinyImage(QDir(detRoot).filePath(QStringLiteral("images/a.png"))));
    const QByteArray detRow("images/a.png\t[{\"transcription\":\"a\",\"points\":[[0,0],[7,0],[7,7],[0,7]]}]\n");
    QVERIFY(writeFile(QDir(detRoot).filePath(QStringLiteral("det_gt.txt")), detRow));
    aitrain::v2::PaddleOcrDetDatasetDriverV2 det;
    QVERIFY2(exerciseDriver(det, detRoot, directory.path(), &error), qPrintable(error));

    const QString recRoot = directory.filePath(QStringLiteral("ocr-rec"));
    QVERIFY(writeTinyImage(QDir(recRoot).filePath(QStringLiteral("images/a.png"))));
    QVERIFY(writeFile(QDir(recRoot).filePath(QStringLiteral("dict.txt")), QByteArray("a\n")));
    QVERIFY(writeFile(QDir(recRoot).filePath(QStringLiteral("rec_gt.txt")), QByteArray("images/a.png\ta\n")));
    aitrain::v2::PaddleOcrRecDatasetDriverV2 rec;
    QVERIFY2(exerciseDriver(rec, recRoot, directory.path(), &error), qPrintable(error));

    aitrain::v2::DatasetDriverRegistryV2 registry;
    QVERIFY(registry.registerDriver(&detection, &error));
    QVERIFY(registry.registerDriver(&segmentation, &error));
    QVERIFY(registry.registerDriver(&obb, &error));
    QVERIFY(registry.registerDriver(&anomaly, &error));
    QVERIFY(registry.registerDriver(&det, &error));
    QVERIFY(registry.registerDriver(&rec, &error));
    QCOMPARE(registry.formats().size(), 6);

    aitrain::v2::DatasetDriverRegistryV2 builtins;
    QVERIFY2(aitrain::v2::registerBuiltinDatasetDriversV2(&builtins, &error), qPrintable(error));
    QCOMPARE(builtins.formats(), QStringList({QStringLiteral("anomaly_folder"), QStringLiteral("paddleocr_det"),
        QStringLiteral("paddleocr_rec"), QStringLiteral("semantic_segmentation_mask"), QStringLiteral("yolo_detection"),
        QStringLiteral("yolo_obb"), QStringLiteral("yolo_segmentation")}));
}

void V2DatasetSnapshotTests::migratedDriverRejectsChangedSourceAndUnsafeStaging()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("source"));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("data.yaml")), QByteArray("nc: 1\nnames: [item]\n")));
    QVERIFY(writeTinyImage(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
    QVERIFY(writeTinyImage(QDir(root).filePath(QStringLiteral("images/val/b.png"))));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QByteArray("0 0.5 0.5 0.25 0.25\n")));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QByteArray("0 0.5 0.5 0.25 0.25\n")));
    aitrain::v2::YoloDetectionDatasetDriverV2 driver;
    aitrain::v2::DatasetOperationContext context;
    aitrain::v2::DatasetInspection inspection;
    QString error;
    QVERIFY(driver.detect(root, &inspection, context, &error));
    aitrain::v2::DatasetSplitPlan plan;
    const QJsonObject options{{QStringLiteral("trainRatio"), 1.0}, {QStringLiteral("valRatio"), 0.0}, {QStringLiteral("testRatio"), 0.0}};
    QVERIFY(driver.planSplit(inspection, options, &plan, context, &error));
    QVERIFY(!driver.materializeSplit(plan, QDir(root).filePath(QStringLiteral("unsafe")), context, &error));
    QCOMPARE(error, QStringLiteral("dataset_split_staging_unsafe_or_not_empty"));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QByteArray("0 0.4 0.4 0.2 0.2\n")));
    QVERIFY(!driver.materializeSplit(plan, directory.filePath(QStringLiteral("staging")), context, &error));
    QCOMPARE(error, QStringLiteral("dataset_source_changed_after_plan"));
}

void V2DatasetSnapshotTests::migratedDriverDetectsInvalidRecognizableLayout()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("invalid-yolo"));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("data.yaml")), QByteArray("nc: 1\nnames: [item]\n")));
    QVERIFY(writeTinyImage(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
    QVERIFY(writeFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QByteArray("bad label\n")));
    aitrain::v2::YoloDetectionDatasetDriverV2 driver;
    aitrain::v2::DatasetInspection inspection;
    aitrain::v2::DatasetOperationContext context;
    QString error;
    QVERIFY2(driver.detect(root, &inspection, context, &error), qPrintable(error));
    QCOMPARE(inspection.format, QStringLiteral("yolo_detection"));
    QVERIFY(!inspection.details.value(QStringLiteral("valid")).toBool());
    aitrain::v2::DatasetValidationResult validation;
    QVERIFY(driver.validate(inspection, &validation, context, &error));
    QVERIFY(!validation.valid);
    QVERIFY(!validation.issues.isEmpty());
    QVERIFY(!validation.issues.first().toObject().value(QStringLiteral("code")).toString().isEmpty());
}

QTEST_MAIN(V2DatasetSnapshotTests)
#include "tst_v2_dataset_snapshot.moc"
