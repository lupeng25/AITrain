#include "DatasetConversionUiModel.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QTest>

class DatasetConversionUiTests : public QObject {
    Q_OBJECT

private slots:
    void sourceFormatsAreFixed();
    void targetFormatsFollowConversionMatrix();
    void unsupportedSourceHasNoTargets();
    void validFormPassesPreflight();
    void cocoJsonFileInputPassesPreflight();
    void validFormTrimsFormatFields();
    void unsupportedPairAndMissingInputAreRejected();
    void invalidSourceStillReportsTargetPairError();
    void workerRunningIsRejected();
};

void DatasetConversionUiTests::sourceFormatsAreFixed()
{
    const QStringList formats = aitrain_app::supportedDatasetConversionSourceFormats();
    QCOMPARE(formats, QStringList({QStringLiteral("coco_json"),
                          QStringLiteral("voc_xml")}));
}

void DatasetConversionUiTests::targetFormatsFollowConversionMatrix()
{
    QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("coco_json")),
        QStringList({QStringLiteral("yolo_detection"), QStringLiteral("yolo_segmentation")}));
    QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("voc_xml")),
        QStringList({QStringLiteral("yolo_detection")}));
    QVERIFY(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("yolo_detection")).isEmpty());
}

void DatasetConversionUiTests::unsupportedSourceHasNoTargets()
{
    QVERIFY(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("paddleocr_rec")).isEmpty());
}

void DatasetConversionUiTests::validFormPassesPreflight()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    QVERIFY(root.mkpath(QStringLiteral("input")));

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral("voc_xml");
    form.targetFormat = QStringLiteral("yolo_detection");
    const QString xmlPath = root.filePath(QStringLiteral("input/sample.xml"));
    QFile xml(xmlPath);
    QVERIFY(xml.open(QIODevice::WriteOnly));
    xml.write("<annotation/>");
    xml.close();
    form.inputPath = xmlPath;
    form.workerRunning = false;

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(validation.ok);
    QCOMPARE(validation.summary, QStringLiteral("可以开始转换。"));
    QVERIFY(validation.messages.isEmpty());
}

void DatasetConversionUiTests::cocoJsonFileInputPassesPreflight()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    QFile annotations(root.filePath(QStringLiteral("annotations.json")));
    QVERIFY(annotations.open(QIODevice::WriteOnly));
    annotations.write("{}");
    annotations.close();

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral("coco_json");
    form.targetFormat = QStringLiteral("yolo_detection");
    form.inputPath = annotations.fileName();

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(validation.ok);
    QVERIFY(validation.messages.isEmpty());
}

void DatasetConversionUiTests::validFormTrimsFormatFields()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    QVERIFY(root.mkpath(QStringLiteral("input")));

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral(" voc_xml ");
    form.targetFormat = QStringLiteral(" yolo_detection ");
    const QString xmlPath = root.filePath(QStringLiteral("input/sample.xml"));
    QFile xml(xmlPath);
    QVERIFY(xml.open(QIODevice::WriteOnly));
    xml.write("<annotation/>");
    xml.close();
    form.inputPath = xmlPath;

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(validation.ok);
    QCOMPARE(validation.summary, QStringLiteral("可以开始转换。"));
    QVERIFY(validation.messages.isEmpty());
}

void DatasetConversionUiTests::unsupportedPairAndMissingInputAreRejected()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral("voc_xml");
    form.targetFormat = QStringLiteral("coco_json");
    form.inputPath = root.filePath(QStringLiteral("missing"));

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(!validation.ok);
    QCOMPARE(validation.targetFormatError, QStringLiteral("当前源格式不支持转换到该目标格式。"));
    QCOMPARE(validation.inputPathError, QStringLiteral("输入路径不存在。"));
}

void DatasetConversionUiTests::invalidSourceStillReportsTargetPairError()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral("paddleocr_rec");
    form.targetFormat = QStringLiteral("yolo_detection");
    form.inputPath = root.filePath(QStringLiteral("missing"));

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(!validation.ok);
    QCOMPARE(validation.sourceFormatError, QStringLiteral("当前不支持该源格式。"));
    QCOMPARE(validation.targetFormatError, QStringLiteral("当前源格式不支持转换到该目标格式。"));
    QCOMPARE(validation.inputPathError, QStringLiteral("输入路径不存在。"));
    QCOMPARE(validation.summary, QStringLiteral("请修正 3 个字段后再转换。"));
}

void DatasetConversionUiTests::workerRunningIsRejected()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral("coco_json");
    form.targetFormat = QStringLiteral("yolo_detection");
    form.inputPath = temp.path();
    form.workerRunning = true;

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(!validation.ok);
    QCOMPARE(validation.summary, QStringLiteral("Worker 正在执行任务，稍后再转换数据集。"));
    QCOMPARE(validation.messages, QStringList({QStringLiteral("Worker 正在执行任务，稍后再转换数据集。")}));
}

QTEST_MAIN(DatasetConversionUiTests)
#include "tst_dataset_conversion_ui.moc"
