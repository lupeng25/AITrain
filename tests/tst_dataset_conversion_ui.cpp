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
    void validFormTrimsFormatFields();
    void sameInputAndOutputDirectoryIsRejected();
    void unsupportedPairAndMissingInputAreRejected();
    void invalidSourceStillReportsTargetPairError();
    void workerRunningIsRejected();
};

void DatasetConversionUiTests::sourceFormatsAreFixed()
{
    const QStringList formats = aitrain_app::supportedDatasetConversionSourceFormats();
    QCOMPARE(formats, QStringList({QStringLiteral("coco_json"),
                          QStringLiteral("voc_xml"),
                          QStringLiteral("yolo_detection"),
                          QStringLiteral("yolo_segmentation")}));
}

void DatasetConversionUiTests::targetFormatsFollowConversionMatrix()
{
    QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("coco_json")),
        QStringList({QStringLiteral("yolo_detection"), QStringLiteral("yolo_segmentation")}));
    QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("voc_xml")),
        QStringList({QStringLiteral("yolo_detection")}));
    QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("yolo_detection")),
        QStringList({QStringLiteral("coco_json"), QStringLiteral("voc_xml")}));
    QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("yolo_segmentation")),
        QStringList({QStringLiteral("coco_json")}));
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
    form.sourceFormat = QStringLiteral("coco_json");
    form.targetFormat = QStringLiteral("yolo_detection");
    form.inputPath = root.filePath(QStringLiteral("input"));
    form.outputPath = root.filePath(QStringLiteral("output"));
    form.workerRunning = false;

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(validation.ok);
    QCOMPARE(validation.summary, QStringLiteral("可以开始转换。"));
    QVERIFY(validation.messages.isEmpty());
}

void DatasetConversionUiTests::validFormTrimsFormatFields()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    QVERIFY(root.mkpath(QStringLiteral("input")));

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral(" coco_json ");
    form.targetFormat = QStringLiteral(" yolo_detection ");
    form.inputPath = root.filePath(QStringLiteral("input"));
    form.outputPath = root.filePath(QStringLiteral("output"));

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(validation.ok);
    QCOMPARE(validation.summary, QStringLiteral("可以开始转换。"));
    QVERIFY(validation.messages.isEmpty());
}

void DatasetConversionUiTests::sameInputAndOutputDirectoryIsRejected()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = QStringLiteral("coco_json");
    form.targetFormat = QStringLiteral("yolo_detection");
    form.inputPath = temp.path();
    form.outputPath = temp.path();

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(!validation.ok);
    QCOMPARE(validation.outputPathError, QStringLiteral("输出目录不能与输入目录相同。"));
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
    form.outputPath = root.filePath(QStringLiteral("output"));

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(!validation.ok);
    QCOMPARE(validation.targetFormatError, QStringLiteral("当前源格式不支持转换到该目标格式。"));
    QCOMPARE(validation.inputPathError, QStringLiteral("输入目录不存在。"));
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
    form.outputPath = root.filePath(QStringLiteral("output"));

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(!validation.ok);
    QCOMPARE(validation.sourceFormatError, QStringLiteral("当前不支持该源格式。"));
    QCOMPARE(validation.targetFormatError, QStringLiteral("当前源格式不支持转换到该目标格式。"));
    QCOMPARE(validation.inputPathError, QStringLiteral("输入目录不存在。"));
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
    form.outputPath = QDir(temp.path()).filePath(QStringLiteral("output"));
    form.workerRunning = true;

    const aitrain_app::DatasetConversionValidation validation = aitrain_app::validateDatasetConversionForm(form);
    QVERIFY(!validation.ok);
    QCOMPARE(validation.summary, QStringLiteral("Worker 正在执行任务，稍后再转换数据集。"));
    QCOMPARE(validation.messages, QStringList({QStringLiteral("Worker 正在执行任务，稍后再转换数据集。")}));
}

QTEST_MAIN(DatasetConversionUiTests)
#include "tst_dataset_conversion_ui.moc"
