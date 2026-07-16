#include "TestSupport.h"

#include "aitrain/core/SemanticMask.h"

class DatasetTests : public QObject {
    Q_OBJECT

private slots:
    void yoloDetectionDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("1 0.5 0.5 0.20 0.20\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);
        QVERIFY(!valid.previewSamples.isEmpty());
        QVERIFY(valid.previewSamples.first().contains(QStringLiteral("bbox=")));

        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("3 0.5 0.5 0.25 0.25\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY(!invalid.ok);
        QVERIFY(!invalid.issues.isEmpty());
        QCOMPARE(invalid.issues.first().line, 1);
    }

    void yoloObbDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("task: obb\nnc: 2\nnames: [ship, plane]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")),
            QStringLiteral("0 0.20 0.20 0.80 0.20 0.80 0.70 0.20 0.70\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")),
            QStringLiteral("1 0.30 0.30 0.70 0.30 0.70 0.60 0.30 0.60\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloObbDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);
        QVERIFY(!valid.previewSamples.isEmpty());
        QVERIFY(valid.previewSamples.first().contains(QStringLiteral("obb=")));

        struct Case {
            QString line;
            QString code;
        };
        const QVector<Case> cases = {
            {QStringLiteral("0 0.20 0.20 0.80 0.20 0.80 0.70\n"), QStringLiteral("invalid_yolo_obb_row")},
            {QStringLiteral("0 0.20 0.20 0.80 0.20 0.80 0.70 0.20 0.70 0.30\n"), QStringLiteral("invalid_yolo_obb_row")},
            {QStringLiteral("3 0.20 0.20 0.80 0.20 0.80 0.70 0.20 0.70\n"), QStringLiteral("class_id_out_of_range")},
            {QStringLiteral("0 0.20 0.20 1.20 0.20 0.80 0.70 0.20 0.70\n"), QStringLiteral("coordinate_out_of_range")},
            {QStringLiteral("0 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20\n"), QStringLiteral("obb_degenerate_quad")}
        };
        for (const Case& item : cases) {
            writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), item.line);
            const aitrain::DatasetValidationResult invalid = aitrain::validateYoloObbDataset(root);
            QVERIFY(!invalid.ok);
            QVERIFY2(jsonArrayContainsCode(invalid.toJson().value(QStringLiteral("issues")).toArray(), item.code),
                qPrintable(QString::fromUtf8(QJsonDocument(invalid.toJson()).toJson(QJsonDocument::Compact))));
        }
    }

    void yoloSegmentationFourPointPolygonWarnsAmbiguousObb()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [ship]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")),
            QStringLiteral("0 0.20 0.20 0.80 0.20 0.80 0.70 0.20 0.70\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")),
            QStringLiteral("0 0.30 0.30 0.70 0.30 0.70 0.60 0.30 0.60\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QVERIFY(jsonArrayContainsCode(valid.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("ambiguous_four_point_polygon")));
    }

    void yoloObbDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("obb-source"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("task: obb\nnc: 1\nnames: [ship]\n"));
        for (int index = 0; index < 4; ++index) {
            const QString split = index < 3 ? QStringLiteral("train") : QStringLiteral("val");
            const QString name = QStringLiteral("sample_%1").arg(index);
            writeTinyPng(QDir(root).filePath(QStringLiteral("images/%1/%2.jpg").arg(split, name)));
            writeTextFile(QDir(root).filePath(QStringLiteral("labels/%1/%2.txt").arg(split, name)),
                QStringLiteral("0 0.20 0.20 0.80 0.20 0.80 0.70 0.20 0.70\n"));
        }

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.25);
        options.insert(QStringLiteral("testRatio"), 0.25);
        options.insert(QStringLiteral("seed"), 7);
        const QString output = dir.filePath(QStringLiteral("obb-normalized"));
        const aitrain::DatasetSplitResult result = aitrain::splitYoloObbDataset(root, output, options);
        QVERIFY2(result.ok, qPrintable(result.errors.join(QStringLiteral("\n"))));
        QCOMPARE(result.trainCount, 2);
        QCOMPARE(result.valCount, 1);
        QCOMPARE(result.testCount, 1);
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("data.yaml"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("split_report.json"))));
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("labels/train"))).entryInfoList(QStringList() << QStringLiteral("*.txt"), QDir::Files).size(), 2);

        QFile reportFile(QDir(output).filePath(QStringLiteral("split_report.json")));
        QVERIFY(reportFile.open(QIODevice::ReadOnly | QIODevice::Text));
        const QJsonObject report = QJsonDocument::fromJson(reportFile.readAll()).object();
        QCOMPARE(report.value(QStringLiteral("format")).toString(), QStringLiteral("yolo_obb"));
    }

    void datasetValidationRejectsUnreadableImages()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString yoloRoot = dir.filePath(QStringLiteral("bad-yolo"));
        writeTextFile(QDir(yoloRoot).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTextFile(QDir(yoloRoot).filePath(QStringLiteral("images/train/bad.png")), QStringLiteral("not an image\n"));
        writeTinyPng(QDir(yoloRoot).filePath(QStringLiteral("images/val/good.png")));
        writeTextFile(QDir(yoloRoot).filePath(QStringLiteral("labels/train/bad.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(yoloRoot).filePath(QStringLiteral("labels/val/good.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        const aitrain::DatasetValidationResult yolo = aitrain::validateYoloDetectionDataset(yoloRoot);
        QVERIFY(!yolo.ok);
        QVERIFY(jsonArrayContainsCode(yolo.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("invalid_image")));

        const QString detRoot = dir.filePath(QStringLiteral("bad-ocr-det"));
        writeTextFile(QDir(detRoot).filePath(QStringLiteral("images/a.png")), QStringLiteral("not an image\n"));
        writeTextFile(QDir(detRoot).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\t[{\"transcription\":\"x\",\"points\":[[0,0],[1,0],[1,1],[0,1]]}]\n"));
        const aitrain::DatasetValidationResult det = aitrain::validatePaddleOcrDetDataset(detRoot);
        QVERIFY(!det.ok);
        QVERIFY(jsonArrayContainsCode(det.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("invalid_image")));

        const QString recRoot = dir.filePath(QStringLiteral("bad-ocr-rec"));
        writeTextFile(QDir(recRoot).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\n"));
        writeTextFile(QDir(recRoot).filePath(QStringLiteral("images/a.png")), QStringLiteral("not an image\n"));
        writeTextFile(QDir(recRoot).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.png\ta\n"));
        const aitrain::DatasetValidationResult rec = aitrain::validatePaddleOcrRecDataset(recRoot);
        QVERIFY(!rec.ok);
        QVERIFY(jsonArrayContainsCode(rec.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("invalid_image")));
    }

    void yoloDetectionDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("source"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        for (int index = 0; index < 4; ++index) {
            const QString split = index < 3 ? QStringLiteral("train") : QStringLiteral("val");
            const QString name = QStringLiteral("sample_%1").arg(index);
            writeTinyPng(QDir(root).filePath(QStringLiteral("images/%1/%2.jpg").arg(split, name)));
            writeTextFile(QDir(root).filePath(QStringLiteral("labels/%1/%2.txt").arg(split, name)), QStringLiteral("0 0.5 0.5 0.2 0.2\n"));
        }

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.25);
        options.insert(QStringLiteral("testRatio"), 0.25);
        options.insert(QStringLiteral("seed"), 7);
        const QString output = dir.filePath(QStringLiteral("normalized"));
        const aitrain::DatasetSplitResult result = aitrain::splitYoloDetectionDataset(root, output, options);
        QVERIFY2(result.ok, qPrintable(result.errors.join(QStringLiteral("\n"))));
        QCOMPARE(result.trainCount, 2);
        QCOMPARE(result.valCount, 1);
        QCOMPARE(result.testCount, 1);
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("data.yaml"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("split_report.json"))));
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/train"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 2);
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/val"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 1);
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/test"))).entryInfoList(QStringList() << QStringLiteral("*.jpg"), QDir::Files).size(), 1);
        QVERIFY(QFileInfo::exists(QDir(root).filePath(QStringLiteral("images/train/sample_0.jpg"))));
    }

    void yoloDetectionCustomDataYamlPaths()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("custom-detection"));
        const QDir rootDir(root);
        writeTextFile(rootDir.filePath(QStringLiteral("data.yaml")),
            QStringLiteral("path : raw\ntrain : custom/images/train\nval : custom/images/val\nnc : 2\nnames :\n  0: widget\n  1: part\n"));
        writeTinyPng(rootDir.filePath(QStringLiteral("raw/custom/images/train/a.jpg")));
        writeTinyPng(rootDir.filePath(QStringLiteral("raw/custom/images/val/b.jpg")));
        writeTextFile(rootDir.filePath(QStringLiteral("raw/custom/labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(rootDir.filePath(QStringLiteral("raw/custom/labels/val/b.txt")), QStringLiteral("1 0.5 0.5 0.20 0.20\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);

        QString error;
        aitrain::DetectionDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));
        QCOMPARE(dataset.size(), 1);
        QCOMPARE(dataset.info().classNames.at(1), QStringLiteral("part"));
        QVERIFY(dataset.samples().first().imagePath.contains(QStringLiteral("raw/custom/images/train")));

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        options.insert(QStringLiteral("seed"), 11);
        const QString output = dir.filePath(QStringLiteral("normalized-detection"));
        const aitrain::DatasetSplitResult split = aitrain::splitYoloDetectionDataset(root, output, options);
        QVERIFY2(split.ok, qPrintable(split.errors.join(QStringLiteral("\n"))));
        QCOMPARE(split.trainCount, 1);
        QCOMPARE(split.valCount, 1);

        QFile dataYaml(QDir(output).filePath(QStringLiteral("data.yaml")));
        QVERIFY(dataYaml.open(QIODevice::ReadOnly | QIODevice::Text));
        const QString dataYamlText = QString::fromUtf8(dataYaml.readAll());
        QVERIFY(dataYamlText.contains(QStringLiteral("path: .")));
        QVERIFY(dataYamlText.contains(QStringLiteral("train: images/train")));
        QVERIFY(!dataYamlText.contains(QStringLiteral("custom/images")));

        const aitrain::DatasetValidationResult normalized = aitrain::validateYoloDetectionDataset(output);
        QVERIFY2(normalized.ok, qPrintable(normalized.errors.join(QStringLiteral("\n"))));
        QCOMPARE(normalized.sampleCount, 2);
    }

    void yoloSplitIgnoresDuplicateCustomSplitImages()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("duplicate-splits"));
        const QDir rootDir(root);
        writeTextFile(rootDir.filePath(QStringLiteral("data.yaml")),
            QStringLiteral("path: .\ntrain: images/shared\nval: images/shared\nnc: 1\nnames: [item]\n"));
        writeTinyPng(rootDir.filePath(QStringLiteral("images/shared/a.jpg")));
        writeTinyPng(rootDir.filePath(QStringLiteral("images/shared/b.jpg")));
        writeTextFile(rootDir.filePath(QStringLiteral("labels/shared/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(rootDir.filePath(QStringLiteral("labels/shared/b.txt")), QStringLiteral("0 0.5 0.5 0.20 0.20\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloDetectionDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        options.insert(QStringLiteral("seed"), 7);
        const QString output = dir.filePath(QStringLiteral("normalized-duplicates"));
        const aitrain::DatasetSplitResult split = aitrain::splitYoloDetectionDataset(root, output, options);
        QVERIFY2(split.ok, qPrintable(split.errors.join(QStringLiteral("\n"))));
        QCOMPARE(split.trainCount + split.valCount + split.testCount, 2);
        QCOMPARE(split.warnings.filter(QStringLiteral("Duplicate YOLO images")).size(), 1);

        const aitrain::DatasetValidationResult normalized = aitrain::validateYoloDetectionDataset(output);
        QVERIFY2(normalized.ok, qPrintable(normalized.errors.join(QStringLiteral("\n"))));
        QCOMPARE(normalized.sampleCount, 2);
    }

    void yoloSegmentationAndOcrDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString segRoot = dir.filePath(QStringLiteral("seg-source"));
        writeTinySegmentationDataset(segRoot);
        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        const QString segOutput = dir.filePath(QStringLiteral("seg-normalized"));
        const aitrain::DatasetSplitResult segResult = aitrain::splitYoloSegmentationDataset(segRoot, segOutput, options);
        QVERIFY2(segResult.ok, qPrintable(segResult.errors.join(QStringLiteral("\n"))));
        QVERIFY(QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("data.yaml"))));
        QVERIFY(QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("labels/train/a.txt")))
            || QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("labels/val/a.txt"))));
        QVERIFY(QFileInfo::exists(QDir(segOutput).filePath(QStringLiteral("split_report.json"))));

        const QString ocrRoot = dir.filePath(QStringLiteral("ocr-source"));
        writeTinyOcrRecDataset(ocrRoot);
        const QString ocrOutput = dir.filePath(QStringLiteral("ocr-normalized"));
        const aitrain::DatasetSplitResult ocrResult = aitrain::splitPaddleOcrRecDataset(ocrRoot, ocrOutput, options);
        QVERIFY2(ocrResult.ok, qPrintable(ocrResult.errors.join(QStringLiteral("\n"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("dict.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("rec_gt.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("rec_gt_train.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("rec_gt_val.txt"))));
        QVERIFY(QFileInfo::exists(QDir(ocrOutput).filePath(QStringLiteral("split_report.json"))));
        const aitrain::DatasetValidationResult validation = aitrain::validatePaddleOcrRecDataset(ocrOutput);
        QVERIFY2(validation.ok, qPrintable(validation.errors.join(QStringLiteral("\n"))));
    }

    void paddleOcrRecSplitAllowsMissingDictionary()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("ocr-source-without-dict"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/b.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.png\taz\nimages/b.png\tza\n"));

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        const QString output = dir.filePath(QStringLiteral("ocr-normalized-without-dict"));
        const aitrain::DatasetSplitResult result = aitrain::splitPaddleOcrRecDataset(root, output, options);
        QVERIFY2(result.ok, qPrintable(result.errors.join(QStringLiteral("\n"))));
        QVERIFY(!QFileInfo::exists(QDir(output).filePath(QStringLiteral("dict.txt"))));
        QVERIFY(result.warnings.join(QStringLiteral("\n")).contains(QStringLiteral("字典")));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("rec_gt.txt"))));

        const aitrain::DatasetValidationResult validation = aitrain::validatePaddleOcrRecDataset(output);
        QVERIFY2(validation.ok, qPrintable(validation.errors.join(QStringLiteral("\n"))));
    }

    void yoloSegmentationCustomDataYamlPaths()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("custom-segmentation"));
        const QDir rootDir(root);
        writeTextFile(rootDir.filePath(QStringLiteral("data.yaml")),
            QStringLiteral("path : raw\ntrain : custom/images/train\nval : custom/images/val\nnc : 1\nnames :\n  - part\n"));
        writeTinyPng(rootDir.filePath(QStringLiteral("raw/custom/images/train/a.png")));
        writeTinyPng(rootDir.filePath(QStringLiteral("raw/custom/images/val/b.png")));
        writeTextFile(rootDir.filePath(QStringLiteral("raw/custom/labels/train/a.txt")),
            QStringLiteral("0 0.10 0.10 0.80 0.10 0.80 0.80 0.10 0.80\n"));
        writeTextFile(rootDir.filePath(QStringLiteral("raw/custom/labels/val/b.txt")),
            QStringLiteral("0 0.20 0.20 0.70 0.20 0.70 0.70 0.20 0.70\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);

        QString error;
        aitrain::SegmentationDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("val"), &error), qPrintable(error));
        QCOMPARE(dataset.size(), 1);
        QVERIFY(dataset.samples().first().imagePath.contains(QStringLiteral("raw/custom/images/val")));

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        options.insert(QStringLiteral("seed"), 13);
        const QString output = dir.filePath(QStringLiteral("normalized-segmentation"));
        const aitrain::DatasetSplitResult split = aitrain::splitYoloSegmentationDataset(root, output, options);
        QVERIFY2(split.ok, qPrintable(split.errors.join(QStringLiteral("\n"))));

        const aitrain::DatasetValidationResult normalized = aitrain::validateYoloSegmentationDataset(output);
        QVERIFY2(normalized.ok, qPrintable(normalized.errors.join(QStringLiteral("\n"))));
        QCOMPARE(normalized.sampleCount, 2);
    }

    void detectionDatasetLoadsSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [cat, dog]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/b.txt")), QStringLiteral("1 0.4 0.4 0.20 0.30\n"));

        QString error;
        const aitrain::DetectionDatasetInfo info = aitrain::readDetectionDatasetInfo(root, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(info.classCount, 2);
        QCOMPARE(info.classNames.size(), 2);
        QCOMPARE(info.classNames.at(1), QStringLiteral("dog"));

        aitrain::DetectionDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));
        QCOMPARE(dataset.size(), 2);
        QCOMPARE(dataset.info().classCount, 2);
        QCOMPARE(dataset.samples().first().boxes.size(), 1);
        QCOMPARE(dataset.samples().first().boxes.first().classId, 0);
        QCOMPARE(dataset.samples().at(1).boxes.first().classId, 1);
    }

    void detectionDatasetRejectsInvalidLabel()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("2 0.5 0.5 0.25 0.25\n"));

        QString error;
        aitrain::DetectionDataset dataset;
        QVERIFY(!dataset.load(root, QStringLiteral("train"), &error));
        QVERIFY(error.contains(QStringLiteral("class id")));
    }

    void detectionDataLoaderBuildsLetterboxBatch()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [item]\n"));

        QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train")));
        QImage wideImage(16, 8, QImage::Format_RGB888);
        wideImage.fill(Qt::white);
        QVERIFY(wideImage.save(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.5 0.5\n"));

        QString error;
        aitrain::DetectionDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));

        aitrain::DetectionDataLoader loader(dataset, 1, QSize(32, 32));
        QVERIFY(loader.hasNext());
        aitrain::DetectionBatch batch;
        QVERIFY2(loader.next(&batch, &error), qPrintable(error));
        QCOMPARE(batch.images.size(), 1);
        QCOMPARE(batch.images.first().size(), QSize(32, 32));
        QCOMPARE(batch.boxes.first().size(), 1);
        QCOMPARE(batch.boxes.first().first().classId, 0);
        QCOMPARE(batch.boxes.first().first().xCenter, 0.5);
        QCOMPARE(batch.boxes.first().first().yCenter, 0.5);
        QCOMPARE(batch.boxes.first().first().width, 0.5);
        QCOMPARE(batch.boxes.first().first().height, 0.25);
        QVERIFY(!loader.hasNext());
    }

    void yoloSegmentationDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [part]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.jpg")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/val/b.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.1 0.1 0.8 0.1 0.8 0.8\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("0 0.1 0.1 0.8 0.1 0.8 0.8\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QVERIFY(!valid.previewSamples.isEmpty());
        QVERIFY(valid.previewSamples.first().contains(QStringLiteral("polygon=")));

        writeTextFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), QStringLiteral("0 0.1 0.1 0.2\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validateYoloSegmentationDataset(root);
        QVERIFY(!invalid.ok);
    }

    void semanticMaskDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("semantic"));
        writeTinySemanticMaskDataset(root);

        const aitrain::DatasetValidationResult valid = aitrain::validateSemanticSegmentationMaskDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);
        QVERIFY(!valid.previewSamples.isEmpty());

        QFile::remove(QDir(root).filePath(QStringLiteral("masks/val/b.png")));
        const aitrain::DatasetValidationResult missing = aitrain::validateSemanticSegmentationMaskDataset(root);
        QVERIFY(!missing.ok);
        QVERIFY(jsonArrayContainsCode(missing.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("missing_mask")));
    }

    void semanticMaskDatasetRejectsBadMasks()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString sizeRoot = dir.filePath(QStringLiteral("semantic-size"));
        writeTinySemanticMaskDataset(sizeRoot);
        writeTinyMaskPng(QDir(sizeRoot).filePath(QStringLiteral("masks/train/a.png")), 4, 4, 1);
        const aitrain::DatasetValidationResult sizeMismatch = aitrain::validateSemanticSegmentationMaskDataset(sizeRoot);
        QVERIFY(!sizeMismatch.ok);
        QVERIFY(jsonArrayContainsCode(sizeMismatch.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("mask_size_mismatch")));

        const QString classRoot = dir.filePath(QStringLiteral("semantic-class"));
        writeTinySemanticMaskDataset(classRoot);
        writeTinyMaskPng(QDir(classRoot).filePath(QStringLiteral("masks/train/a.png")), 8, 8, 9);
        const aitrain::DatasetValidationResult classRange = aitrain::validateSemanticSegmentationMaskDataset(classRoot);
        QVERIFY(!classRange.ok);
        QVERIFY(jsonArrayContainsCode(classRange.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("class_id_out_of_range")));

        const QString ignoreRoot = dir.filePath(QStringLiteral("semantic-ignore"));
        writeTinySemanticMaskDataset(ignoreRoot);
        QImage ignoreMask(8, 8, QImage::Format_Grayscale8);
        ignoreMask.fill(255);
        QVERIFY(ignoreMask.save(QDir(ignoreRoot).filePath(QStringLiteral("masks/train/a.png"))));
        const aitrain::DatasetValidationResult ignoreValid = aitrain::validateSemanticSegmentationMaskDataset(ignoreRoot);
        QVERIFY2(ignoreValid.ok, qPrintable(ignoreValid.errors.join(QStringLiteral("\n"))));
        QVERIFY(jsonArrayContainsCode(ignoreValid.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("empty_mask")));
    }

    void semanticMaskDatasetPreservesPaletteIndexes()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("semantic-palette"));
        writeTinySemanticMaskDataset(root);

        QImage paletteMask(8, 8, QImage::Format_Indexed8);
        QVector<QRgb> palette(3, qRgb(0, 0, 0));
        palette[1] = qRgb(255, 0, 0);
        palette[2] = qRgb(0, 255, 0);
        paletteMask.setColorTable(palette);
        paletteMask.fill(1);
        const QString maskPath = QDir(root).filePath(QStringLiteral("masks/train/a.png"));
        QVERIFY(paletteMask.save(maskPath));

        const QImage reloaded(maskPath);
        QCOMPARE(reloaded.format(), QImage::Format_Indexed8);
        QCOMPARE(aitrain::semanticMaskClassId(reloaded, 0, 0), 1);
        const aitrain::DatasetValidationResult validation = aitrain::validateSemanticSegmentationMaskDataset(root);
        QVERIFY2(validation.ok, qPrintable(validation.errors.join(QStringLiteral("\n"))));
    }

    void semanticMaskDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("semantic-source"));
        writeTinySemanticMaskDataset(root);

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        options.insert(QStringLiteral("seed"), 5);
        const QString output = dir.filePath(QStringLiteral("semantic-normalized"));
        const aitrain::DatasetSplitResult split = aitrain::splitSemanticSegmentationMaskDataset(root, output, options);
        QVERIFY2(split.ok, qPrintable(split.errors.join(QStringLiteral("\n"))));
        QCOMPARE(split.trainCount, 1);
        QCOMPARE(split.valCount, 1);
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("classes.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("split_report.json"))));
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("images/train"))).entryInfoList(QStringList() << QStringLiteral("*.png"), QDir::Files).size(), 1);
        QCOMPARE(QDir(QDir(output).filePath(QStringLiteral("masks/train"))).entryInfoList(QStringList() << QStringLiteral("*.png"), QDir::Files).size(), 1);
        const aitrain::DatasetValidationResult normalized = aitrain::validateSemanticSegmentationMaskDataset(output);
        QVERIFY2(normalized.ok, qPrintable(normalized.errors.join(QStringLiteral("\n"))));
    }

    void ocrRecDatasetLoadsDictionaryAndLabels()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);

        QString error;
        aitrain::OcrRecDataset dataset;
        QVERIFY2(dataset.load(root, QString(), QString(), 8, &error), qPrintable(error));
        QCOMPARE(dataset.size(), 2);
        QCOMPARE(dataset.dictionary().characters.size(), 4);
        QCOMPARE(dataset.samples().first().label, QStringLiteral("ab12"));
        QCOMPARE(dataset.samples().first().encodedLabel.size(), 4);
        QCOMPARE(dataset.samples().first().encodedLabel.at(0), 1);
        QCOMPARE(dataset.samples().first().encodedLabel.at(1), 2);
        QCOMPARE(dataset.samples().first().encodedLabel.at(2), 3);
        QCOMPARE(dataset.samples().first().encodedLabel.at(3), 4);
        QCOMPARE(aitrain::decodeOcrText(dataset.samples().first().encodedLabel, dataset.dictionary()), QStringLiteral("ab12"));
    }

    void ocrRecDatasetRejectsUnknownDictionaryCharacter()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinyOcrRecDataset(root);
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.png\taz\n"));

        QString error;
        aitrain::OcrRecDataset dataset;
        QVERIFY(!dataset.load(root, QString(), QString(), 8, &error));
        QVERIFY(error.contains(QStringLiteral("dictionary")));
    }

    void ocrRecDataLoaderBuildsPaddedBatch()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\nb\n"));

        QDir().mkpath(QDir(root).filePath(QStringLiteral("images")));
        QImage wideImage(16, 8, QImage::Format_RGB888);
        wideImage.fill(Qt::black);
        QVERIFY(wideImage.save(QDir(root).filePath(QStringLiteral("images/a.png"))));
        QImage tallImage(8, 16, QImage::Format_RGB888);
        tallImage.fill(Qt::black);
        QVERIFY(tallImage.save(QDir(root).filePath(QStringLiteral("images/b.png"))));
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.png\tab\nimages/b.png\tba\n"));

        QString error;
        aitrain::OcrRecDataset dataset;
        QVERIFY2(dataset.load(root, QString(), QString(), 8, &error), qPrintable(error));

        aitrain::OcrRecDataLoader loader(dataset, 2, QSize(32, 16));
        aitrain::OcrRecBatch batch;
        QVERIFY2(loader.next(&batch, &error), qPrintable(error));
        QCOMPARE(batch.images.size(), 2);
        QCOMPARE(batch.labels.size(), 2);
        QCOMPARE(batch.labelLengths.size(), 2);
        QCOMPARE(batch.labelLengths.at(0), 2);
        QCOMPARE(batch.labelLengths.at(1), 2);
        QCOMPARE(batch.texts.first(), QStringLiteral("ab"));
        QCOMPARE(batch.images.first().size(), QSize(32, 16));
        QCOMPARE(qRed(batch.images.first().pixel(31, 15)), 0);
        QCOMPARE(qRed(batch.images.at(1).pixel(0, 15)), 0);
        QCOMPARE(qRed(batch.images.at(1).pixel(31, 15)), 255);
        QVERIFY(!loader.hasNext());
    }

    void segmentationDatasetLoadsMasksAndOverlay()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTinySegmentationDataset(root);

        QString error;
        aitrain::SegmentationDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));
        QCOMPARE(dataset.size(), 1);
        QCOMPARE(dataset.info().classCount, 1);
        QCOMPARE(dataset.samples().first().polygons.size(), 1);
        QCOMPARE(dataset.samples().first().polygons.first().points.size(), 4);

        const QImage mask = aitrain::polygonToMask(dataset.samples().first().polygons.first().points, QSize(8, 8));
        QVERIFY(!mask.isNull());
        QCOMPARE(mask.size(), QSize(8, 8));
        QVERIFY(qAlpha(mask.pixel(4, 4)) > 0);
        QCOMPARE(qAlpha(mask.pixel(0, 0)), 0);

        const QImage overlay = aitrain::renderSegmentationOverlay(
            dataset.samples().first().imagePath,
            dataset.samples().first().polygons,
            &error);
        QVERIFY2(!overlay.isNull(), qPrintable(error));
        QCOMPARE(overlay.size(), QSize(8, 8));
    }

    void segmentationDataLoaderBuildsAlignedBatchMasks()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 2\nnames: [part, edge]\n"));

        QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train")));
        QImage wideImage(16, 8, QImage::Format_RGB888);
        wideImage.fill(Qt::white);
        QVERIFY(wideImage.save(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
        QImage tallImage(8, 16, QImage::Format_RGB888);
        tallImage.fill(Qt::white);
        QVERIFY(tallImage.save(QDir(root).filePath(QStringLiteral("images/train/b.png"))));

        writeTextFile(
            QDir(root).filePath(QStringLiteral("labels/train/a.txt")),
            QStringLiteral("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
                           "1 0.05 0.05 0.15 0.05 0.15 0.20 0.05 0.20\n"));
        writeTextFile(
            QDir(root).filePath(QStringLiteral("labels/train/b.txt")),
            QStringLiteral("1 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"));

        QString error;
        aitrain::SegmentationDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));

        aitrain::SegmentationDataLoader loader(dataset, 2, QSize(32, 32));
        QVERIFY(loader.hasNext());
        aitrain::SegmentationBatch batch;
        QVERIFY2(loader.next(&batch, &error), qPrintable(error));
        QCOMPARE(batch.images.size(), 2);
        QCOMPARE(batch.masks.size(), 2);
        QCOMPARE(batch.polygons.size(), 2);
        QCOMPARE(batch.imagePaths.size(), 2);
        QCOMPARE(batch.images.first().size(), QSize(32, 32));
        QCOMPARE(batch.masks.first().size(), QSize(32, 32));

        QCOMPARE(batch.polygons.first().size(), 2);
        QCOMPARE(batch.polygons.first().first().classId, 0);
        QVERIFY(qAbs(batch.polygons.first().first().points.first().x() - 0.25) < 0.001);
        QVERIFY(qAbs(batch.polygons.first().first().points.first().y() - 0.375) < 0.001);
        QCOMPARE(qAlpha(batch.masks.first().pixel(16, 4)), 0);
        QCOMPARE(qRed(batch.masks.first().pixel(16, 16)), 1);
        QVERIFY(qAlpha(batch.masks.first().pixel(16, 16)) > 0);
        QCOMPARE(qRed(batch.masks.first().pixel(3, 10)), 2);

        QCOMPARE(batch.polygons.at(1).first().classId, 1);
        QCOMPARE(qAlpha(batch.masks.at(1).pixel(4, 16)), 0);
        QCOMPARE(qRed(batch.masks.at(1).pixel(16, 16)), 2);
        QVERIFY(!loader.hasNext());
    }

    void segmentationDataLoaderRejectsInvalidPolygons()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("dataset"));
        writeTextFile(QDir(root).filePath(QStringLiteral("data.yaml")), QStringLiteral("nc: 1\nnames: [part]\n"));
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/train/a.png")));
        writeTextFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.1 0.1 0.2 0.2 0.3 0.3\n"));

        QString error;
        aitrain::SegmentationDataset dataset;
        QVERIFY2(dataset.load(root, QStringLiteral("train"), &error), qPrintable(error));

        aitrain::SegmentationDataLoader loader(dataset, 1, QSize(16, 16));
        aitrain::SegmentationBatch batch;
        QVERIFY(!loader.next(&batch, &error));
        QVERIFY(error.contains(QStringLiteral("Invalid segmentation polygon")));
    }


    void paddleOcrDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("images/a.jpg")));
        writeTextFile(QDir(root).filePath(QStringLiteral("dict.txt")), QStringLiteral("a\nb\nc\n"));
        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/a.jpg\tabc\n"));

        const aitrain::DatasetValidationResult valid = aitrain::validatePaddleOcrRecDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 1);
        QVERIFY(!valid.previewSamples.isEmpty());

        writeTextFile(QDir(root).filePath(QStringLiteral("rec_gt.txt")), QStringLiteral("images/missing.jpg\taz\n"));
        const aitrain::DatasetValidationResult invalid = aitrain::validatePaddleOcrRecDataset(root);
        QVERIFY(!invalid.ok);
        QVERIFY(invalid.errors.join(QStringLiteral("\n")).contains(QStringLiteral("字典")));
    }

    void paddleOcrDetDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyOcrDetDataset(root);

        const aitrain::DatasetValidationResult valid = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QCOMPARE(valid.sampleCount, 2);
        QVERIFY(!valid.previewSamples.isEmpty());

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/missing.png\t[{\"transcription\":\"x\",\"points\":[[0,0],[1,0],[1,1],[0,1]]}]\n"));
        const aitrain::DatasetValidationResult missing = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!missing.ok);
        QVERIFY(missing.errors.join(QStringLiteral("\n")).contains(QStringLiteral("不存在")));

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\tbad-json\n"));
        const aitrain::DatasetValidationResult badJson = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!badJson.ok);
        QVERIFY(badJson.errors.join(QStringLiteral("\n")).contains(QStringLiteral("JSON")));

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\t[{\"transcription\":\"x\",\"points\":[[0,0],[1,0],[1,1]]}]\n"));
        const aitrain::DatasetValidationResult tooFew = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!tooFew.ok);
        QVERIFY(tooFew.errors.join(QStringLiteral("\n")).contains(QStringLiteral("4")));

        writeTextFile(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            QStringLiteral("images/a.png\t[{\"transcription\":\"x\",\"points\":[[0,0],[1,0],[1,1],[0,1]]}]\n"
                           "images/a.png\t[{\"transcription\":\"y\",\"points\":[[0,0],[1,0],[1,1],[0,1]]}]\n"));
        const aitrain::DatasetValidationResult duplicate = aitrain::validatePaddleOcrDetDataset(root);
        QVERIFY(!duplicate.ok);
        QVERIFY(duplicate.errors.join(QStringLiteral("\n")).contains(QStringLiteral("重复")));
    }

    void paddleOcrDetDatasetSplit()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.filePath(QStringLiteral("source"));
        writeTinyOcrDetDataset(root);
        const QString output = dir.filePath(QStringLiteral("split"));

        QJsonObject options;
        options.insert(QStringLiteral("trainRatio"), 0.5);
        options.insert(QStringLiteral("valRatio"), 0.5);
        options.insert(QStringLiteral("testRatio"), 0.0);
        options.insert(QStringLiteral("seed"), 7);
        const aitrain::DatasetSplitResult result = aitrain::splitPaddleOcrDetDataset(root, output, options);
        QVERIFY2(result.ok, qPrintable(result.errors.join(QStringLiteral("\n"))));
        QCOMPARE(result.trainCount, 1);
        QCOMPARE(result.valCount, 1);
        QCOMPARE(result.testCount, 0);
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt_train.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt_val.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("det_gt_test.txt"))));
        QVERIFY(QFileInfo::exists(QDir(output).filePath(QStringLiteral("split_report.json"))));

        const aitrain::DatasetValidationResult validation = aitrain::validatePaddleOcrDetDataset(output);
        QVERIFY2(validation.ok, qPrintable(validation.errors.join(QStringLiteral("\n"))));
        QCOMPARE(validation.sampleCount, 2);
    }

    void anomalyFolderDatasetValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("train/good/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("val/good/b.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("test/anomaly/ng.png")));
        writeTinyMaskPng(QDir(root).filePath(QStringLiteral("masks/test/anomaly/ng.png")));

        const aitrain::DatasetValidationResult valid = aitrain::validateAnomalyFolderDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        const QJsonObject metadata = valid.toJson().value(QStringLiteral("metadata")).toObject();
        QCOMPARE(valid.sampleCount, 3);
        QCOMPARE(metadata.value(QStringLiteral("normalCount")).toInt(), 2);
        QCOMPARE(metadata.value(QStringLiteral("anomalyCount")).toInt(), 1);
        QCOMPARE(metadata.value(QStringLiteral("maskCount")).toInt(), 1);
        QVERIFY(!metadata.value(QStringLiteral("evaluationLimited")).toBool());
    }

    void anomalyFolderMvtecAliasValidation()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("train/good/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("test/good/b.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("test/scratch/ng.png")));
        writeTinyMaskPng(QDir(root).filePath(QStringLiteral("ground_truth/scratch/ng_mask.png")));

        const aitrain::DatasetValidationResult valid = aitrain::validateAnomalyFolderDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        const QJsonObject metadata = valid.toJson().value(QStringLiteral("metadata")).toObject();
        QCOMPARE(metadata.value(QStringLiteral("anomalyCount")).toInt(), 1);
        QCOMPARE(metadata.value(QStringLiteral("maskCount")).toInt(), 1);
    }

    void anomalyFolderMvtecSameStemMissingMaskWarns()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("train/good/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("test/scratch/000.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("test/crack/000.png")));
        writeTinyMaskPng(QDir(root).filePath(QStringLiteral("ground_truth/scratch/000_mask.png")));

        const aitrain::DatasetValidationResult valid = aitrain::validateAnomalyFolderDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QVERIFY(jsonArrayContainsCode(valid.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("missing_anomaly_mask")));
    }

    void anomalyFolderGoodOnlyLimited()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("train/good/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("test/good/b.png")));

        const aitrain::DatasetValidationResult valid = aitrain::validateAnomalyFolderDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        const QJsonObject metadata = valid.toJson().value(QStringLiteral("metadata")).toObject();
        QVERIFY(metadata.value(QStringLiteral("evaluationLimited")).toBool());
        QVERIFY(jsonArrayContainsCode(valid.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("good_only_dataset")));
    }

    void anomalyFolderMaskMismatchWarns()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString root = dir.path();
        writeTinyPng(QDir(root).filePath(QStringLiteral("train/good/a.png")));
        writeTinyPng(QDir(root).filePath(QStringLiteral("test/anomaly/ng.png")));
        writeTinyMaskPng(QDir(root).filePath(QStringLiteral("masks/test/anomaly/other.png")));

        const aitrain::DatasetValidationResult valid = aitrain::validateAnomalyFolderDataset(root);
        QVERIFY2(valid.ok, qPrintable(valid.errors.join(QStringLiteral("\n"))));
        QVERIFY(jsonArrayContainsCode(valid.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("missing_anomaly_mask")));
    }

    void anomalyFolderEmptyFails()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const aitrain::DatasetValidationResult invalid = aitrain::validateAnomalyFolderDataset(dir.path());
        QVERIFY(!invalid.ok);
        QVERIFY(jsonArrayContainsCode(invalid.toJson().value(QStringLiteral("issues")).toArray(), QStringLiteral("missing_train_good")));
    }

};

QTEST_MAIN(DatasetTests)
#include "tst_datasets.moc"
