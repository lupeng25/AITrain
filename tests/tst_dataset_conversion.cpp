#include "TestSupport.h"

#include "aitrain/core/DatasetConversion.h"

#include <QTemporaryDir>
#include <QTest>

class DatasetConversionTests : public QObject {
    Q_OBJECT

private slots:
    void unsupportedFormatFails();
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

QTEST_MAIN(DatasetConversionTests)
#include "tst_dataset_conversion.moc"
