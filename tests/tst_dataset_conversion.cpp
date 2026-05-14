#include "TestSupport.h"

#include "aitrain/core/DatasetConversion.h"

#include <QFile>
#include <QJsonDocument>
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

QTEST_MAIN(DatasetConversionTests)
#include "tst_dataset_conversion.moc"
