#include "TestSupport.h"

#include "aitrain/core/DatasetConversion.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QImage>
#include <QTemporaryDir>
#include <QTest>

class DatasetConversionTests : public QObject {
    Q_OBJECT

private slots:
    void unsupportedFormatFails();
    void cocoDetectionConvertsBboxToYolo();
    void cocoSegmentationConvertsPolygonToYolo();
    void cocoSegmentationSkipsRleMasks();
    void vocXmlConvertsBoxesToYoloDetection();
    void yoloDetectionConvertsToCoco();
    void yoloDetectionConvertsToVocXml();
    void yoloSegmentationConvertsToCocoPolygons();
    void copyImagesFalseKeepsReferencedPaths();
    void invalidYoloLabelsReportIssues();
    void yoloDataYamlCustomSplitPathsAreHonored();
    void missingYoloSourcePathFailsAsSourceReadFailed();
    void yoloDetectionToVocReportsDuplicateOutputTargets();
};

void DatasetConversionTests::unsupportedFormatFails()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());

    aitrain::DatasetConversionRequest request;
    request.sourcePath = temp.path();
    request.outputPath = QDir(temp.path()).filePath(QStringLiteral("out"));
    request.sourceFormat = QStringLiteral("unknown_format");
    request.targetFormat = QStringLiteral("yolo_detection");

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY(!result.ok);
    QCOMPARE(result.errorCode, QStringLiteral("unsupported_source_format"));
    QVERIFY(result.reportPath.isEmpty());
}

void writeTinyPngWithSize(const QString& path, int width, int height)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QImage image(width, height, QImage::Format_RGB888);
    image.fill(Qt::white);
    QVERIFY(image.save(path));
}

QString readTextForConversionTest(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    return QString::fromUtf8(file.readAll());
}

QJsonObject readJsonObjectForConversionTest(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QJsonDocument::fromJson(file.readAll()).object();
}

QJsonObject firstObjectWithValue(const QJsonArray& array, const QString& key, const QJsonValue& expected)
{
    for (const QJsonValue& value : array) {
        const QJsonObject object = value.toObject();
        if (object.value(key) == expected) {
            return object;
        }
    }
    return {};
}

bool issuesContainCode(const QVector<aitrain::DatasetConversionIssue>& issues, const QString& code)
{
    for (const aitrain::DatasetConversionIssue& issue : issues) {
        if (issue.code == code) {
            return true;
        }
    }
    return false;
}

void writeYoloDetectionFixture(const QDir& root)
{
    writeTinyPngWithSize(root.filePath(QStringLiteral("images/train/a.png")), 100, 80);
    writeTinyPngWithSize(root.filePath(QStringLiteral("images/val/b.png")), 120, 60);
    writeTextFile(root.filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.500000 0.500000 0.500000 0.400000\n"));
    writeTextFile(root.filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("1 0.250000 0.500000 0.300000 0.500000\n"));
    writeTextFile(root.filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: widget\n  1: part\n"));
}

void writeYoloSegmentationFixture(const QDir& root)
{
    writeTinyPngWithSize(root.filePath(QStringLiteral("images/train/seg.png")), 100, 100);
    writeTinyPngWithSize(root.filePath(QStringLiteral("images/val/seg_val.png")), 100, 100);
    writeTextFile(root.filePath(QStringLiteral("labels/train/seg.txt")),
        QStringLiteral("0 0.100000 0.100000 0.900000 0.100000 0.900000 0.900000 0.100000 0.900000\n"));
    writeTextFile(root.filePath(QStringLiteral("labels/val/seg_val.txt")),
        QStringLiteral("0 0.200000 0.200000 0.800000 0.200000 0.800000 0.800000 0.200000 0.800000\n"));
    writeTextFile(root.filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: part\n"));
}

void DatasetConversionTests::cocoDetectionConvertsBboxToYolo()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());

    writeTinyPng(root.filePath(QStringLiteral("images/a.png")));
    writeTextFile(root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("{\"images\":[{\"id\":1,\"file_name\":\"images/a.png\",\"width\":8,\"height\":8}],"
                       "\"categories\":[{\"id\":7,\"name\":\"widget\"}],"
                       "\"annotations\":[{\"id\":10,\"image_id\":1,\"category_id\":7,\"bbox\":[2,2,4,2]}]}"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.filePath(QStringLiteral("annotations.json"));
    request.sourceFormat = QStringLiteral("coco_json");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(result.convertedSampleCount, 1);
    QCOMPARE(result.convertedAnnotationCount, 1);

    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted/data.yaml"))));
    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted/images/train/a.png"))));
    QFile labelFile(root.filePath(QStringLiteral("converted/labels/train/a.txt")));
    QVERIFY(labelFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString label = QString::fromUtf8(labelFile.readAll()).trimmed();
    QCOMPARE(label, QStringLiteral("0 0.500000 0.375000 0.500000 0.250000"));

    QFile reportFile(result.reportPath);
    QVERIFY(reportFile.open(QIODevice::ReadOnly));
    const QJsonObject report = QJsonDocument::fromJson(reportFile.readAll()).object();
    QCOMPARE(report.value(QStringLiteral("sourceFormat")).toString(), QStringLiteral("coco_json"));
    QCOMPARE(report.value(QStringLiteral("targetFormat")).toString(), QStringLiteral("yolo_detection"));
    QVERIFY(report.value(QStringLiteral("targetValidation")).toObject().value(QStringLiteral("ok")).toBool());
}

void DatasetConversionTests::cocoSegmentationConvertsPolygonToYolo()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPngWithSize(root.filePath(QStringLiteral("images/seg.png")), 100, 100);
    writeTextFile(root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("{\"images\":[{\"id\":1,\"file_name\":\"images/seg.png\",\"width\":100,\"height\":100}],"
                       "\"categories\":[{\"id\":1,\"name\":\"part\"}],"
                       "\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":1,"
                       "\"segmentation\":[[10,10,90,10,90,90,10,90]]}]}"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.filePath(QStringLiteral("annotations.json"));
    request.sourceFormat = QStringLiteral("coco_json");
    request.targetFormat = QStringLiteral("yolo_segmentation");
    request.outputPath = root.filePath(QStringLiteral("converted_seg"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QFile segLabelFile(root.filePath(QStringLiteral("converted_seg/labels/train/seg.txt")));
    QVERIFY(segLabelFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString label = QString::fromUtf8(segLabelFile.readAll()).trimmed();
    QCOMPARE(label, QStringLiteral("0 0.100000 0.100000 0.900000 0.100000 0.900000 0.900000 0.100000 0.900000"));
    QVERIFY(result.targetValidation.ok);
}

void DatasetConversionTests::cocoSegmentationSkipsRleMasks()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPngWithSize(root.filePath(QStringLiteral("images/rle.png")), 20, 20);
    writeTextFile(root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("{\"images\":[{\"id\":1,\"file_name\":\"images/rle.png\",\"width\":20,\"height\":20}],"
                       "\"categories\":[{\"id\":1,\"name\":\"mask\"}],"
                       "\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":1,"
                       "\"segmentation\":{\"counts\":\"abc\",\"size\":[20,20]}}]}"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.filePath(QStringLiteral("annotations.json"));
    request.sourceFormat = QStringLiteral("coco_json");
    request.targetFormat = QStringLiteral("yolo_segmentation");
    request.outputPath = root.filePath(QStringLiteral("converted_rle"));

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY(!result.ok);
    QCOMPARE(result.errorCode, QStringLiteral("no_convertible_samples"));
    QVERIFY(!result.issues.isEmpty());
    QCOMPARE(result.issues.first().code, QStringLiteral("rle_not_supported"));
}

void DatasetConversionTests::vocXmlConvertsBoxesToYoloDetection()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPngWithSize(root.filePath(QStringLiteral("JPEGImages/a.png")), 100, 80);
    writeTextFile(root.filePath(QStringLiteral("Annotations/a.xml")),
        QStringLiteral("<annotation>"
                       "<filename>a.png</filename>"
                       "<size><width>100</width><height>80</height></size>"
                       "<object><name>part</name><bndbox>"
                       "<xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>36</ymax>"
                       "</bndbox></object>"
                       "</annotation>"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.filePath(QStringLiteral("Annotations"));
    request.sourceFormat = QStringLiteral("voc_xml");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted_voc"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QFile vocLabelFile(root.filePath(QStringLiteral("converted_voc/labels/train/a.txt")));
    QVERIFY(vocLabelFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString label = QString::fromUtf8(vocLabelFile.readAll()).trimmed();
    QCOMPARE(label, QStringLiteral("0 0.250000 0.350000 0.300000 0.200000"));
}

void DatasetConversionTests::yoloDetectionConvertsToCoco()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_yolo")));
    writeYoloDetectionFixture(source);

    aitrain::DatasetConversionRequest request;
    request.sourcePath = source.absolutePath();
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("coco_json");
    request.outputPath = root.filePath(QStringLiteral("converted_coco"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(result.convertedSampleCount, 2);
    QCOMPARE(result.convertedAnnotationCount, 2);

    const QJsonObject train = readJsonObjectForConversionTest(root.filePath(QStringLiteral("converted_coco/annotations/train.json")));
    const QJsonArray trainImages = train.value(QStringLiteral("images")).toArray();
    const QJsonArray trainAnnotations = train.value(QStringLiteral("annotations")).toArray();
    const QJsonArray categories = train.value(QStringLiteral("categories")).toArray();
    QCOMPARE(trainImages.size(), 1);
    QCOMPARE(trainAnnotations.size(), 1);
    QCOMPARE(categories.size(), 2);
    QCOMPARE(categories.at(0).toObject().value(QStringLiteral("name")).toString(), QStringLiteral("widget"));

    const QJsonArray bbox = trainAnnotations.at(0).toObject().value(QStringLiteral("bbox")).toArray();
    QCOMPARE(bbox.size(), 4);
    QCOMPARE(bbox.at(0).toDouble(), 25.0);
    QCOMPARE(bbox.at(1).toDouble(), 24.0);
    QCOMPARE(bbox.at(2).toDouble(), 50.0);
    QCOMPARE(bbox.at(3).toDouble(), 32.0);

    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_coco/images/train/a.png"))));
    QVERIFY(QFileInfo::exists(result.reportPath));
}

void DatasetConversionTests::yoloDetectionConvertsToVocXml()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_yolo")));
    writeYoloDetectionFixture(source);

    aitrain::DatasetConversionRequest request;
    request.sourcePath = source.absolutePath();
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("voc_xml");
    request.outputPath = root.filePath(QStringLiteral("converted_voc"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(result.convertedSampleCount, 2);
    QCOMPARE(result.convertedAnnotationCount, 2);

    const QString xml = readTextForConversionTest(root.filePath(QStringLiteral("converted_voc/Annotations/a.xml")));
    QVERIFY(xml.contains(QStringLiteral("<filename>a.png</filename>")));
    QVERIFY(xml.contains(QStringLiteral("<name>widget</name>")));
    QVERIFY(xml.contains(QStringLiteral("<xmin>25</xmin>")));
    QVERIFY(xml.contains(QStringLiteral("<ymin>24</ymin>")));
    QVERIFY(xml.contains(QStringLiteral("<xmax>75</xmax>")));
    QVERIFY(xml.contains(QStringLiteral("<ymax>56</ymax>")));
    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_voc/JPEGImages/a.png"))));
}

void DatasetConversionTests::yoloSegmentationConvertsToCocoPolygons()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_yolo_seg")));
    writeYoloSegmentationFixture(source);

    aitrain::DatasetConversionRequest request;
    request.sourcePath = source.absolutePath();
    request.sourceFormat = QStringLiteral("yolo_segmentation");
    request.targetFormat = QStringLiteral("coco_json");
    request.outputPath = root.filePath(QStringLiteral("converted_coco_seg"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(result.convertedSampleCount, 2);
    QCOMPARE(result.convertedAnnotationCount, 2);

    const QJsonObject train = readJsonObjectForConversionTest(root.filePath(QStringLiteral("converted_coco_seg/annotations/train.json")));
    const QJsonArray annotations = train.value(QStringLiteral("annotations")).toArray();
    QCOMPARE(annotations.size(), 1);
    const QJsonArray segmentation = annotations.at(0).toObject().value(QStringLiteral("segmentation")).toArray();
    QCOMPARE(segmentation.size(), 1);
    const QJsonArray polygon = segmentation.at(0).toArray();
    QCOMPARE(polygon.at(0).toDouble(), 10.0);
    QCOMPARE(polygon.at(1).toDouble(), 10.0);
    QCOMPARE(polygon.at(6).toDouble(), 10.0);
    QCOMPARE(polygon.at(7).toDouble(), 90.0);
    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_coco_seg/images/train/seg.png"))));
}

void DatasetConversionTests::copyImagesFalseKeepsReferencedPaths()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_yolo")));
    writeYoloDetectionFixture(source);

    aitrain::DatasetConversionRequest request;
    request.sourcePath = source.absolutePath();
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("coco_json");
    request.outputPath = root.filePath(QStringLiteral("converted_reference"));
    request.options.insert(QStringLiteral("copyImages"), false);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QVERIFY(!QFileInfo::exists(root.filePath(QStringLiteral("converted_reference/images/train/a.png"))));
    const QJsonObject train = readJsonObjectForConversionTest(root.filePath(QStringLiteral("converted_reference/annotations/train.json")));
    const QString fileName = train.value(QStringLiteral("images")).toArray().at(0).toObject().value(QStringLiteral("file_name")).toString();
    QCOMPARE(fileName, QStringLiteral("images/train/a.png"));
    const QJsonObject report = readJsonObjectForConversionTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("imagePolicy")).toString(), QStringLiteral("referenced"));
}

void DatasetConversionTests::invalidYoloLabelsReportIssues()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_yolo_invalid")));
    writeTinyPngWithSize(source.filePath(QStringLiteral("images/train/a.png")), 100, 80);
    writeTextFile(source.filePath(QStringLiteral("labels/train/a.txt")),
        QStringLiteral("0 0.500000 0.500000 0.500000 0.400000\n9 0.5 0.5 0.2 0.2\n0 0.5 0.5 -0.2 0.2\n"));
    writeTextFile(source.filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: .\ntrain: images/train\nval: images/train\nnc: 1\nnames:\n  0: widget\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = source.absolutePath();
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("coco_json");
    request.outputPath = root.filePath(QStringLiteral("converted_invalid"));

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(result.convertedAnnotationCount, 1);
    QVERIFY(issuesContainCode(result.issues, QStringLiteral("unknown_class_id")));
    QVERIFY(issuesContainCode(result.issues, QStringLiteral("invalid_bbox")));
}

void DatasetConversionTests::yoloDataYamlCustomSplitPathsAreHonored()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_custom_yolo")));
    writeTinyPngWithSize(source.filePath(QStringLiteral("dataset/train/images/custom_train.png")), 100, 80);
    writeTinyPngWithSize(source.filePath(QStringLiteral("dataset/validation/images/custom_val.png")), 120, 60);
    writeTextFile(source.filePath(QStringLiteral("dataset/train/labels/custom_train.txt")),
        QStringLiteral("0 0.500000 0.500000 0.500000 0.400000\n"));
    writeTextFile(source.filePath(QStringLiteral("dataset/validation/labels/custom_val.txt")),
        QStringLiteral("0 0.250000 0.500000 0.300000 0.500000\n"));
    writeTextFile(source.filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: dataset\ntrain: train/images\nval: validation/images\nnc: 1\nnames: [custom]\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = source.absolutePath();
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("coco_json");
    request.outputPath = root.filePath(QStringLiteral("converted_custom"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(result.convertedSampleCount, 2);
    QCOMPARE(result.convertedAnnotationCount, 2);

    const QJsonObject train = readJsonObjectForConversionTest(root.filePath(QStringLiteral("converted_custom/annotations/train.json")));
    const QString trainFile = train.value(QStringLiteral("images")).toArray().at(0).toObject().value(QStringLiteral("file_name")).toString();
    QCOMPARE(trainFile, QStringLiteral("train/images/custom_train.png"));
    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_custom/train/images/custom_train.png"))));

    const QJsonObject val = readJsonObjectForConversionTest(root.filePath(QStringLiteral("converted_custom/annotations/val.json")));
    const QString valFile = val.value(QStringLiteral("images")).toArray().at(0).toObject().value(QStringLiteral("file_name")).toString();
    QCOMPARE(valFile, QStringLiteral("validation/images/custom_val.png"));
    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_custom/validation/images/custom_val.png"))));
}

void DatasetConversionTests::missingYoloSourcePathFailsAsSourceReadFailed()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.filePath(QStringLiteral("does_not_exist"));
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("coco_json");
    request.outputPath = root.filePath(QStringLiteral("converted_missing"));

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY(!result.ok);
    QCOMPARE(result.errorCode, QStringLiteral("source_read_failed"));
    QVERIFY(result.reportPath.isEmpty());
}

void DatasetConversionTests::yoloDetectionToVocReportsDuplicateOutputTargets()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_dataset_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_duplicate_yolo")));
    writeTinyPngWithSize(source.filePath(QStringLiteral("images/train/left/dup.png")), 100, 80);
    writeTinyPngWithSize(source.filePath(QStringLiteral("images/train/right/dup.png")), 120, 60);
    writeTextFile(source.filePath(QStringLiteral("labels/train/left/dup.txt")),
        QStringLiteral("0 0.500000 0.500000 0.500000 0.400000\n"));
    writeTextFile(source.filePath(QStringLiteral("labels/train/right/dup.txt")),
        QStringLiteral("0 0.250000 0.500000 0.300000 0.500000\n"));
    writeTextFile(source.filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: .\ntrain: images/train\nval: images/train\nnc: 1\nnames: [duplicate]\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = source.absolutePath();
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("voc_xml");
    request.outputPath = root.filePath(QStringLiteral("converted_duplicate_voc"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(result.convertedSampleCount, 1);
    QCOMPARE(result.convertedAnnotationCount, 1);
    QVERIFY(issuesContainCode(result.issues, QStringLiteral("duplicate_output_target")));
    const QString xml = readTextForConversionTest(root.filePath(QStringLiteral("converted_duplicate_voc/Annotations/dup.xml")));
    QCOMPARE(xml.count(QStringLiteral("<object>")), 1);
}

QTEST_MAIN(DatasetConversionTests)
#include "tst_dataset_conversion.moc"
