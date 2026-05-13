#include "TestSupport.h"

#include "aitrain/core/DatasetConversion.h"

#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>

class DatasetConversionTests : public QObject {
    Q_OBJECT

private slots:
    void unsupportedFormatFails();
    void cocoDetectionConvertsBboxToYolo();
};

void DatasetConversionTests::unsupportedFormatFails()
{
    QTemporaryDir temp;
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

namespace {

QString readTextFile(const QString& path)
{
    QFile file(path);
    QVERIFY(file.open(QIODevice::ReadOnly | QIODevice::Text));
    return QString::fromUtf8(file.readAll());
}

QJsonObject readJsonObjectForTest(const QString& path)
{
    QFile file(path);
    QVERIFY(file.open(QIODevice::ReadOnly));
    return QJsonDocument::fromJson(file.readAll()).object();
}

} // namespace

void DatasetConversionTests::cocoDetectionConvertsBboxToYolo()
{
    QTemporaryDir temp;
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
    const QString label = readTextFile(root.filePath(QStringLiteral("converted/labels/train/a.txt"))).trimmed();
    QCOMPARE(label, QStringLiteral("0 0.500000 0.375000 0.500000 0.250000"));

    const QJsonObject report = readJsonObjectForTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("sourceFormat")).toString(), QStringLiteral("coco_json"));
    QCOMPARE(report.value(QStringLiteral("targetFormat")).toString(), QStringLiteral("yolo_detection"));
    QVERIFY(report.value(QStringLiteral("targetValidation")).toObject().value(QStringLiteral("ok")).toBool());
}

QTEST_MAIN(DatasetConversionTests)
#include "tst_dataset_conversion.moc"
