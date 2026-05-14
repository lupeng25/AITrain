# Dataset Conversion Matrix v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first stable YOLO/COCO/VOC conversion matrix v2 for detection and polygon segmentation, with core conversion tests and Worker routing.

**Architecture:** Keep parsing and writing in `src/core`, expose the matrix through the existing `DatasetConversionRequest` / `DatasetConversionResult` API, and route product-facing work through `aitrain_worker`. Do not add GUI entry points in this phase.

**Tech Stack:** C++20, Qt 5.12-compatible Core/Gui/Xml/Network/Test, QtTest, CMake, JSON/DOM APIs, existing AITrain Worker JSON Lines protocol.

---

## Current Workspace Note

This plan starts from the current workspace state, where these files are already dirty from the previous VOC baseline work:

- `CMakeLists.txt`
- `src/core/CMakeLists.txt`
- `src/core/src/DatasetConversion.cpp`
- `tests/tst_dataset_conversion.cpp`
- `docs/superpowers/specs/2026-05-14-dataset-conversion-v2-design.zh-CN.md`

Do not revert those files. Treat the current VOC XML to YOLO detection fix and Qt XML dependency as baseline work. Commit it before starting the v2 matrix tasks so later commits stay reviewable.

## File Structure

- Modify `src/core/include/aitrain/core/DatasetConversion.h`: add backward-compatible report fields to `DatasetConversionResult`.
- Modify `src/core/src/DatasetConversion.cpp`: add YOLO source parsing, COCO writer, VOC writer, report enrichment, and dispatch for reverse conversions.
- Modify `src/worker/src/WorkerSession.h`: declare `convertDataset(...)`.
- Modify `src/worker/src/WorkerSession.cpp`: route the `convertDataset` Worker command.
- Modify `src/worker/src/WorkerSessionDatasetCommands.cpp`: implement Worker conversion command and artifact emission.
- Modify `src/app/src/WorkerClient.h`: add `requestDatasetConversion(...)`.
- Modify `src/app/src/WorkerClient.cpp`: send Worker `convertDataset` request.
- Modify `tests/tst_dataset_conversion.cpp`: add core matrix tests.
- Modify one Worker QtTest file, preferably `tests/tst_repository_workflow.cpp`: add one Worker conversion route test.

No GUI page, SQLite schema, plugin interface, training backend, inference path, evaluation path, or packaging workflow should be changed.

---

### Task 0: Checkpoint Current VOC Baseline

**Files:**
- Stage: `CMakeLists.txt`
- Stage: `src/core/CMakeLists.txt`
- Stage: `src/core/src/DatasetConversion.cpp`
- Stage: `tests/tst_dataset_conversion.cpp`
- Stage: `docs/superpowers/specs/2026-05-14-dataset-conversion-v2-design.zh-CN.md`

- [ ] **Step 1: Verify current baseline**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: build succeeds and CTest reports `100% tests passed, 0 tests failed out of 6`.

- [ ] **Step 2: Commit only the baseline and Chinese spec**

Run:

```powershell
git add CMakeLists.txt src/core/CMakeLists.txt src/core/src/DatasetConversion.cpp tests/tst_dataset_conversion.cpp docs/superpowers/specs/2026-05-14-dataset-conversion-v2-design.zh-CN.md
git commit -m "feat: add VOC dataset conversion baseline"
```

Expected: one commit is created. Do not stage unrelated files.

---

### Task 1: Add Failing Core Tests for Reverse Matrix

**Files:**
- Modify: `tests/tst_dataset_conversion.cpp`

- [ ] **Step 1: Add test slot declarations**

In `DatasetConversionTests`, add these private slots after `vocXmlConvertsBoxesToYoloDetection()`:

```cpp
void yoloDetectionConvertsToCoco();
void yoloDetectionConvertsToVocXml();
void yoloSegmentationConvertsToCocoPolygons();
void copyImagesFalseKeepsReferencedPaths();
void invalidYoloLabelsReportIssues();
```

- [ ] **Step 2: Add test helper functions**

Place these helpers near the existing `writeTinyPngWithSize(...)` helper:

```cpp
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
```

- [ ] **Step 3: Add `yoloDetectionConvertsToCoco`**

Append this test body before `QTEST_MAIN`:

```cpp
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
```

- [ ] **Step 4: Add `yoloDetectionConvertsToVocXml`**

Append:

```cpp
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
```

- [ ] **Step 5: Add `yoloSegmentationConvertsToCocoPolygons`**

Append:

```cpp
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
```

- [ ] **Step 6: Add copy/reference and invalid-label tests**

Append:

```cpp
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
```

- [ ] **Step 7: Run the new tests and verify they fail**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: build succeeds, `aitrain_dataset_conversion_tests` fails because `yolo_detection` and `yolo_segmentation` are not accepted source formats yet.

- [ ] **Step 8: Keep red tests uncommitted**

Run:

```powershell
git status --short
```

Expected: `tests/tst_dataset_conversion.cpp` is modified and uncommitted. Do not commit the red test state.

---

### Task 2: Extend Result Reporting Fields

**Files:**
- Modify: `src/core/include/aitrain/core/DatasetConversion.h`
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add backward-compatible result fields**

In `DatasetConversionResult`, after `QJsonObject classMap;`, add:

```cpp
int conversionMatrixVersion = 0;
bool copyImages = true;
QString imagePolicy;
QJsonObject outputFiles;
QJsonObject splitCounts;
QJsonObject sourceValidation;
```

- [ ] **Step 2: Serialize the new fields**

In `DatasetConversionResult::toJson()`, after `classMap`, add:

```cpp
object.insert(QStringLiteral("conversionMatrixVersion"), conversionMatrixVersion);
object.insert(QStringLiteral("copyImages"), copyImages);
object.insert(QStringLiteral("imagePolicy"), imagePolicy);
object.insert(QStringLiteral("outputFiles"), outputFiles);
object.insert(QStringLiteral("splitCounts"), splitCounts);
object.insert(QStringLiteral("sourceValidation"), sourceValidation);
```

- [ ] **Step 3: Set v2 metadata in existing converters**

In `convertCoco(...)` and `convertVoc(...)`, after result source/output initialization, add:

```cpp
result.conversionMatrixVersion = 2;
result.copyImages = request.options.value(QStringLiteral("copyImages")).toBool(true);
result.imagePolicy = result.copyImages ? QStringLiteral("copied") : QStringLiteral("referenced");
```

Then replace local `copyImages` initialization with:

```cpp
const bool copyImages = result.copyImages;
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
ctest --test-dir .\build-vscode --output-on-failure -R aitrain_dataset_conversion_tests
```

Expected: existing tests still pass if the test binary is already built. If CTest reports stale binary behavior, continue to the next full build step in Task 3.

- [ ] **Step 5: Keep report field changes uncommitted**

Run:

```powershell
git status --short
```

Expected: `src/core/include/aitrain/core/DatasetConversion.h`, `src/core/src/DatasetConversion.cpp`, and `tests/tst_dataset_conversion.cpp` remain modified. The branch still has intentionally red v2 tests, so do not commit yet.

---

### Task 3: Implement YOLO Source Parser

**Files:**
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add internal YOLO model structs**

Inside the anonymous namespace, near `VocAnnotation`, add:

```cpp
struct YoloClass {
    int id = 0;
    QString name;
};

struct YoloAnnotation {
    int classId = -1;
    QVector<double> values;
    QString sourceFile;
    int lineNumber = 0;
};

struct YoloImage {
    int id = 0;
    QString split;
    QString fileName;
    QString relativeImagePath;
    QString sourceImagePath;
    QString labelPath;
    int width = 0;
    int height = 0;
    QVector<YoloAnnotation> annotations;
};

struct YoloDataset {
    QString rootPath;
    QVector<YoloClass> classes;
    QVector<YoloImage> images;
    QJsonObject splitCounts;
};
```

- [ ] **Step 2: Add image and YAML helpers**

Add:

```cpp
QJsonArray stringListToJsonArray(const QStringList& values)
{
    QJsonArray array;
    for (const QString& value : values) {
        array.append(value);
    }
    return array;
}

QStringList parseYoloNamesFromYaml(const QString& yamlText)
{
    QStringList names;
    const QStringList lines = yamlText.split(QLatin1Char('\n'));
    bool inBlockNames = false;
    for (const QString& rawLine : lines) {
        const QString line = rawLine.trimmed();
        if (line.startsWith(QStringLiteral("names: ["))) {
            QString inner = line;
            inner.remove(QStringLiteral("names:"));
            inner = inner.trimmed();
            inner.remove(QLatin1Char('['));
            inner.remove(QLatin1Char(']'));
            for (QString part : inner.split(QLatin1Char(','), Qt::SkipEmptyParts)) {
                part = part.trimmed();
                part.remove(QLatin1Char('\''));
                part.remove(QLatin1Char('"'));
                if (!part.isEmpty()) {
                    names.append(part);
                }
            }
            inBlockNames = false;
            continue;
        }
        if (line == QStringLiteral("names:")) {
            inBlockNames = true;
            continue;
        }
        if (inBlockNames) {
            const int colon = line.indexOf(QLatin1Char(':'));
            if (colon < 0) {
                continue;
            }
            QString value = line.mid(colon + 1).trimmed();
            value.remove(QLatin1Char('\''));
            value.remove(QLatin1Char('"'));
            if (!value.isEmpty()) {
                names.append(value);
            }
        }
    }
    return names;
}

bool readImageSize(const QString& path, int* width, int* height)
{
    QImage image(path);
    if (image.isNull()) {
        return false;
    }
    if (width) {
        *width = image.width();
    }
    if (height) {
        *height = image.height();
    }
    return true;
}
```

- [ ] **Step 3: Add YOLO label parser**

Add:

```cpp
QVector<YoloAnnotation> parseYoloLabelFile(
    const QString& labelPath,
    bool segmentation,
    DatasetConversionResult* result)
{
    QVector<YoloAnnotation> annotations;
    QFile file(labelPath);
    if (!file.exists()) {
        return annotations;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (result) {
            result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("label_read_failed"), labelPath, QString(), QString(),
                QStringLiteral("Cannot read YOLO label file.")));
        }
        return annotations;
    }
    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty()) {
            continue;
        }
        const QStringList parts = line.split(QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts);
        const int minimumSize = segmentation ? 7 : 5;
        if (parts.size() < minimumSize || (segmentation && parts.size() % 2 == 0)) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_yolo_label"), labelPath, QString(), QString(),
                    QStringLiteral("YOLO label row has an invalid column count.")));
                ++result->skippedAnnotationCount;
            }
            continue;
        }
        YoloAnnotation annotation;
        annotation.classId = parts.first().toInt();
        annotation.sourceFile = labelPath;
        annotation.lineNumber = lineNumber;
        for (int index = 1; index < parts.size(); ++index) {
            annotation.values.append(parts.at(index).toDouble());
        }
        annotations.append(annotation);
    }
    return annotations;
}
```

Also add `#include <QRegularExpression>` and `#include <QImage>` if not already present.

- [ ] **Step 4: Add YOLO dataset parser**

Add:

```cpp
YoloDataset parseYoloDataset(const DatasetConversionRequest& request, bool segmentation, DatasetConversionResult* result)
{
    YoloDataset dataset;
    dataset.rootPath = QDir::cleanPath(request.sourcePath);
    const QDir root(dataset.rootPath);

    QStringList names;
    QFile yamlFile(root.filePath(QStringLiteral("data.yaml")));
    if (yamlFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        names = parseYoloNamesFromYaml(QString::fromUtf8(yamlFile.readAll()));
    }
    if (names.isEmpty()) {
        names.append(QStringLiteral("class_0"));
        if (result) {
            result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("missing_class_names"), dataset.rootPath, QString(), QString(),
                QStringLiteral("YOLO data.yaml names are missing; generated class_0.")));
        }
    }
    for (int index = 0; index < names.size(); ++index) {
        YoloClass yoloClass;
        yoloClass.id = index;
        yoloClass.name = names.at(index);
        dataset.classes.append(yoloClass);
        if (result) {
            result->classMap.insert(QString::number(index), yoloClass.name);
        }
    }

    int nextImageId = 1;
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val")}) {
        const QDir imageDir(root.filePath(QStringLiteral("images/%1").arg(split)));
        const QDir labelDir(root.filePath(QStringLiteral("labels/%1").arg(split)));
        if (!imageDir.exists()) {
            continue;
        }
        QDirIterator it(imageDir.absolutePath(), QStringList{QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"), QStringLiteral("*.png"), QStringLiteral("*.bmp")}, QDir::Files);
        int splitCount = 0;
        while (it.hasNext()) {
            const QString imagePath = it.next();
            YoloImage image;
            image.id = nextImageId++;
            image.split = split;
            image.fileName = QFileInfo(imagePath).fileName();
            image.relativeImagePath = QStringLiteral("images/%1/%2").arg(split, image.fileName);
            image.sourceImagePath = imagePath;
            image.labelPath = labelDir.filePath(QFileInfo(image.fileName).completeBaseName() + QStringLiteral(".txt"));
            if (!readImageSize(image.sourceImagePath, &image.width, &image.height)) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_read_failed"), image.labelPath, image.sourceImagePath, QString(),
                        QStringLiteral("Cannot read image dimensions.")));
                    ++result->skippedSampleCount;
                }
                continue;
            }
            image.annotations = parseYoloLabelFile(image.labelPath, segmentation, result);
            dataset.images.append(image);
            ++splitCount;
        }
        dataset.splitCounts.insert(split, splitCount);
    }
    return dataset;
}
```

- [ ] **Step 5: Run compile to catch parser errors**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: build succeeds; reverse matrix tests still fail because writers and dispatch are not implemented.

- [ ] **Step 6: Keep parser work uncommitted**

Run:

```powershell
git status --short
```

Expected: parser changes remain in the working tree. Do not commit until all core matrix tests are green.

---

### Task 4: Add COCO Writer for YOLO Detection and Segmentation

**Files:**
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add geometry helpers**

Add:

```cpp
double clampUnit(double value)
{
    return qMin(1.0, qMax(0.0, value));
}

QJsonArray bboxFromYoloValues(const QVector<double>& values, int width, int height)
{
    const double cx = clampUnit(values.value(0)) * width;
    const double cy = clampUnit(values.value(1)) * height;
    const double boxW = clampUnit(values.value(2)) * width;
    const double boxH = clampUnit(values.value(3)) * height;
    return QJsonArray{cx - boxW / 2.0, cy - boxH / 2.0, boxW, boxH};
}

double polygonArea(const QJsonArray& polygon)
{
    double area = 0.0;
    for (int index = 0; index + 3 < polygon.size(); index += 2) {
        const double x1 = polygon.at(index).toDouble();
        const double y1 = polygon.at(index + 1).toDouble();
        const int next = (index + 2 < polygon.size()) ? index + 2 : 0;
        const double x2 = polygon.at(next).toDouble();
        const double y2 = polygon.at(next + 1).toDouble();
        area += x1 * y2 - x2 * y1;
    }
    return qAbs(area) / 2.0;
}

QJsonArray bboxFromPolygon(const QJsonArray& polygon)
{
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = 0.0;
    double maxY = 0.0;
    for (int index = 0; index + 1 < polygon.size(); index += 2) {
        const double x = polygon.at(index).toDouble();
        const double y = polygon.at(index + 1).toDouble();
        minX = qMin(minX, x);
        minY = qMin(minY, y);
        maxX = qMax(maxX, x);
        maxY = qMax(maxY, y);
    }
    return QJsonArray{minX, minY, maxX - minX, maxY - minY};
}
```

Add `#include <limits>` if needed.

- [ ] **Step 2: Add image copy helper for arbitrary target path**

Add:

```cpp
bool copyImageToRelativePath(const QString& sourceImagePath, const QString& outputPath, const QString& relativePath, QString* error)
{
    if (!QFileInfo::exists(sourceImagePath)) {
        if (error) {
            *error = QStringLiteral("Image file does not exist: %1").arg(sourceImagePath);
        }
        return false;
    }
    const QString targetPath = QDir(outputPath).filePath(relativePath);
    QDir().mkpath(QFileInfo(targetPath).absolutePath());
    if (QFileInfo::exists(targetPath) && !QFile::remove(targetPath)) {
        if (error) {
            *error = QStringLiteral("Cannot clear existing image target: %1").arg(targetPath);
        }
        return false;
    }
    if (!QFile::copy(sourceImagePath, targetPath)) {
        if (error) {
            *error = QStringLiteral("Cannot copy image: %1").arg(sourceImagePath);
        }
        return false;
    }
    return true;
}
```

- [ ] **Step 3: Add `writeCocoSplit`**

Add:

```cpp
bool writeCocoSplit(const YoloDataset& dataset,
    const QString& split,
    bool segmentation,
    const DatasetConversionRequest& request,
    DatasetConversionResult* result,
    QString* annotationPath)
{
    QJsonArray images;
    QJsonArray annotations;
    QJsonArray categories;
    for (const YoloClass& yoloClass : dataset.classes) {
        categories.append(QJsonObject{{QStringLiteral("id"), yoloClass.id}, {QStringLiteral("name"), yoloClass.name}});
    }

    int annotationId = 1;
    for (const YoloImage& image : dataset.images) {
        if (image.split != split) {
            continue;
        }
        const QString outputRelativeImagePath = image.relativeImagePath;
        if (result && result->copyImages) {
            QString copyError;
            if (!copyImageToRelativePath(image.sourceImagePath, request.outputPath, outputRelativeImagePath, &copyError)) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), image.labelPath, image.sourceImagePath, QString(), copyError));
                ++result->skippedSampleCount;
                continue;
            }
        }
        images.append(QJsonObject{
            {QStringLiteral("id"), image.id},
            {QStringLiteral("file_name"), outputRelativeImagePath},
            {QStringLiteral("width"), image.width},
            {QStringLiteral("height"), image.height}
        });
        bool convertedImageAnnotation = false;
        for (const YoloAnnotation& annotation : image.annotations) {
            if (annotation.classId < 0 || annotation.classId >= dataset.classes.size()) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("unknown_class_id"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                        QStringLiteral("YOLO annotation references an unknown class id.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            if (!segmentation) {
                if (annotation.values.size() != 4 || annotation.values.at(2) <= 0.0 || annotation.values.at(3) <= 0.0) {
                    if (result) {
                        result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                            QStringLiteral("YOLO detection bbox is invalid.")));
                        ++result->skippedAnnotationCount;
                    }
                    continue;
                }
                const QJsonArray bbox = bboxFromYoloValues(annotation.values, image.width, image.height);
                annotations.append(QJsonObject{
                    {QStringLiteral("id"), annotationId++},
                    {QStringLiteral("image_id"), image.id},
                    {QStringLiteral("category_id"), annotation.classId},
                    {QStringLiteral("bbox"), bbox},
                    {QStringLiteral("area"), bbox.at(2).toDouble() * bbox.at(3).toDouble()},
                    {QStringLiteral("iscrowd"), 0}
                });
                convertedImageAnnotation = true;
                if (result) {
                    ++result->convertedAnnotationCount;
                }
                continue;
            }
            if (annotation.values.size() < 6 || annotation.values.size() % 2 != 0) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                        QStringLiteral("YOLO segmentation polygon is invalid.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            QJsonArray polygon;
            for (int index = 0; index + 1 < annotation.values.size(); index += 2) {
                polygon.append(clampUnit(annotation.values.at(index)) * image.width);
                polygon.append(clampUnit(annotation.values.at(index + 1)) * image.height);
            }
            const double area = polygonArea(polygon);
            if (area <= 0.0) {
                if (result) {
                    result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                        QStringLiteral("YOLO segmentation polygon area is invalid.")));
                    ++result->skippedAnnotationCount;
                }
                continue;
            }
            annotations.append(QJsonObject{
                {QStringLiteral("id"), annotationId++},
                {QStringLiteral("image_id"), image.id},
                {QStringLiteral("category_id"), annotation.classId},
                {QStringLiteral("segmentation"), QJsonArray{polygon}},
                {QStringLiteral("bbox"), bboxFromPolygon(polygon)},
                {QStringLiteral("area"), area},
                {QStringLiteral("iscrowd"), 0}
            });
            convertedImageAnnotation = true;
            if (result) {
                ++result->convertedAnnotationCount;
            }
        }
        if (convertedImageAnnotation && result) {
            ++result->convertedSampleCount;
        }
    }

    if (images.isEmpty() && annotations.isEmpty()) {
        return true;
    }
    const QString path = QDir(request.outputPath).filePath(QStringLiteral("annotations/%1.json").arg(split));
    const QJsonObject root{
        {QStringLiteral("images"), images},
        {QStringLiteral("annotations"), annotations},
        {QStringLiteral("categories"), categories}
    };
    QString writeError;
    if (!writeJsonFile(path, root, &writeError)) {
        if (result) {
            result->errorCode = QStringLiteral("output_write_failed");
            result->errorMessage = writeError;
        }
        return false;
    }
    if (annotationPath) {
        *annotationPath = path;
    }
    return true;
}
```

- [ ] **Step 4: Add `convertYoloToCoco`**

Add:

```cpp
DatasetConversionResult convertYoloToCoco(const DatasetConversionRequest& request, bool segmentation)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);
    result.conversionMatrixVersion = 2;
    result.copyImages = request.options.value(QStringLiteral("copyImages")).toBool(true);
    result.imagePolicy = result.copyImages ? QStringLiteral("copied") : QStringLiteral("referenced");

    const YoloDataset dataset = parseYoloDataset(request, segmentation, &result);
    result.sampleCount = dataset.images.size();
    result.splitCounts = dataset.splitCounts;
    QDir().mkpath(result.outputPath);

    QString trainAnnotations;
    QString valAnnotations;
    if (!writeCocoSplit(dataset, QStringLiteral("train"), segmentation, request, &result, &trainAnnotations)
        || !writeCocoSplit(dataset, QStringLiteral("val"), segmentation, request, &result, &valAnnotations)) {
        return result;
    }
    if (result.convertedSampleCount <= 0 || result.convertedAnnotationCount <= 0) {
        result.errorCode = QStringLiteral("no_convertible_samples");
        result.errorMessage = QStringLiteral("YOLO dataset did not contain convertible annotations.");
        return result;
    }

    QJsonArray annotationFiles;
    if (!trainAnnotations.isEmpty()) {
        annotationFiles.append(trainAnnotations);
    }
    if (!valAnnotations.isEmpty()) {
        annotationFiles.append(valAnnotations);
    }
    result.outputFiles.insert(QStringLiteral("annotations"), annotationFiles);
    result.outputFiles.insert(QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("images")));
    result.ok = true;

    result.reportPath = QDir(result.outputPath).filePath(QStringLiteral("dataset_conversion_report.json"));
    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("convertedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    QString writeError;
    if (!writeJsonFile(result.reportPath, report, &writeError)) {
        result.ok = false;
        result.errorCode = QStringLiteral("report_write_failed");
        result.errorMessage = writeError;
    }
    return result;
}
```

- [ ] **Step 5: Dispatch YOLO to COCO**

In `convertDataset(...)`, update supported source formats:

```cpp
if (sourceFormat != QStringLiteral("coco_json")
    && sourceFormat != QStringLiteral("voc_xml")
    && sourceFormat != QStringLiteral("labelme_json")
    && sourceFormat != QStringLiteral("yolo_detection")
    && sourceFormat != QStringLiteral("yolo_segmentation")) {
```

Before the final `not_implemented` return, add:

```cpp
if (sourceFormat == QStringLiteral("yolo_detection") && targetFormat == QStringLiteral("coco_json")) {
    return convertYoloToCoco(request, false);
}
if (sourceFormat == QStringLiteral("yolo_segmentation") && targetFormat == QStringLiteral("coco_json")) {
    return convertYoloToCoco(request, true);
}
```

- [ ] **Step 6: Run focused test**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: `yoloDetectionConvertsToCoco`, `yoloSegmentationConvertsToCocoPolygons`, `copyImagesFalseKeepsReferencedPaths`, and `invalidYoloLabelsReportIssues` pass. `yoloDetectionConvertsToVocXml` still fails.

- [ ] **Step 7: Keep COCO writer work uncommitted**

Run:

```powershell
git status --short
```

Expected: COCO reverse conversion changes remain in the working tree because the VOC reverse test is still red.

---

### Task 5: Add Pascal VOC XML Writer

**Files:**
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add XML text escape helper**

Add:

```cpp
QString xmlEscaped(QString value)
{
    value.replace(QLatin1Char('&'), QStringLiteral("&amp;"));
    value.replace(QLatin1Char('<'), QStringLiteral("&lt;"));
    value.replace(QLatin1Char('>'), QStringLiteral("&gt;"));
    value.replace(QLatin1Char('"'), QStringLiteral("&quot;"));
    value.replace(QLatin1Char('\''), QStringLiteral("&apos;"));
    return value;
}
```

- [ ] **Step 2: Add VOC XML writer helper**

Add:

```cpp
bool writeVocXmlForImage(
    const YoloDataset& dataset,
    const YoloImage& image,
    const DatasetConversionRequest& request,
    DatasetConversionResult* result)
{
    QStringList objects;
    for (const YoloAnnotation& annotation : image.annotations) {
        if (annotation.classId < 0 || annotation.classId >= dataset.classes.size()) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("unknown_class_id"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                    QStringLiteral("YOLO annotation references an unknown class id.")));
                ++result->skippedAnnotationCount;
            }
            continue;
        }
        if (annotation.values.size() != 4 || annotation.values.at(2) <= 0.0 || annotation.values.at(3) <= 0.0) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                    QStringLiteral("YOLO detection bbox is invalid.")));
                ++result->skippedAnnotationCount;
            }
            continue;
        }
        const QJsonArray bbox = bboxFromYoloValues(annotation.values, image.width, image.height);
        const int xmin = qMax(0, qRound(bbox.at(0).toDouble()));
        const int ymin = qMax(0, qRound(bbox.at(1).toDouble()));
        const int xmax = qMin(image.width, qRound(bbox.at(0).toDouble() + bbox.at(2).toDouble()));
        const int ymax = qMin(image.height, qRound(bbox.at(1).toDouble() + bbox.at(3).toDouble()));
        if (xmax <= xmin || ymax <= ymin) {
            if (result) {
                result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), annotation.sourceFile, image.sourceImagePath, QString::number(annotation.classId),
                    QStringLiteral("YOLO detection bbox became empty after clamping.")));
                ++result->skippedAnnotationCount;
            }
            continue;
        }
        objects.append(QStringLiteral("  <object>\n"
                                      "    <name>%1</name>\n"
                                      "    <pose>Unspecified</pose>\n"
                                      "    <truncated>0</truncated>\n"
                                      "    <difficult>0</difficult>\n"
                                      "    <bndbox>\n"
                                      "      <xmin>%2</xmin>\n"
                                      "      <ymin>%3</ymin>\n"
                                      "      <xmax>%4</xmax>\n"
                                      "      <ymax>%5</ymax>\n"
                                      "    </bndbox>\n"
                                      "  </object>\n")
            .arg(xmlEscaped(dataset.classes.at(annotation.classId).name))
            .arg(xmin)
            .arg(ymin)
            .arg(xmax)
            .arg(ymax));
        if (result) {
            ++result->convertedAnnotationCount;
        }
    }
    if (objects.isEmpty()) {
        return true;
    }

    const QString imageName = QFileInfo(image.sourceImagePath).fileName();
    const QString copiedImagePath = QDir(request.outputPath).filePath(QStringLiteral("JPEGImages/%1").arg(imageName));
    if (result && result->copyImages) {
        QString copyError;
        if (!copyImageToRelativePath(image.sourceImagePath, request.outputPath, QStringLiteral("JPEGImages/%1").arg(imageName), &copyError)) {
            result->issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), image.labelPath, image.sourceImagePath, QString(), copyError));
            ++result->skippedSampleCount;
            return false;
        }
    }

    const QString xml = QStringLiteral("<annotation>\n"
                                       "  <folder>JPEGImages</folder>\n"
                                       "  <filename>%1</filename>\n"
                                       "  <path>%2</path>\n"
                                       "  <size>\n"
                                       "    <width>%3</width>\n"
                                       "    <height>%4</height>\n"
                                       "    <depth>3</depth>\n"
                                       "  </size>\n"
                                       "%5"
                                       "</annotation>\n")
        .arg(xmlEscaped(imageName))
        .arg(xmlEscaped(result && result->copyImages ? copiedImagePath : image.sourceImagePath))
        .arg(image.width)
        .arg(image.height)
        .arg(objects.join(QString()));
    QString writeError;
    const QString xmlPath = QDir(request.outputPath).filePath(QStringLiteral("Annotations/%1.xml").arg(QFileInfo(imageName).completeBaseName()));
    if (!writeTextFile(xmlPath, xml, &writeError)) {
        if (result) {
            result->errorCode = QStringLiteral("output_write_failed");
            result->errorMessage = writeError;
        }
        return false;
    }
    if (result) {
        ++result->convertedSampleCount;
    }
    return true;
}
```

- [ ] **Step 3: Add `convertYoloToVoc`**

Add:

```cpp
DatasetConversionResult convertYoloToVoc(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);
    result.conversionMatrixVersion = 2;
    result.copyImages = request.options.value(QStringLiteral("copyImages")).toBool(true);
    result.imagePolicy = result.copyImages ? QStringLiteral("copied") : QStringLiteral("referenced");

    const YoloDataset dataset = parseYoloDataset(request, false, &result);
    result.sampleCount = dataset.images.size();
    result.splitCounts = dataset.splitCounts;
    QDir().mkpath(result.outputPath);
    for (const YoloImage& image : dataset.images) {
        if (!writeVocXmlForImage(dataset, image, request, &result)) {
            if (!result.errorCode.isEmpty()) {
                return result;
            }
        }
    }
    if (result.convertedSampleCount <= 0 || result.convertedAnnotationCount <= 0) {
        result.errorCode = QStringLiteral("no_convertible_samples");
        result.errorMessage = QStringLiteral("YOLO dataset did not contain convertible detection annotations.");
        return result;
    }
    result.outputFiles.insert(QStringLiteral("xmlRoot"), QDir(result.outputPath).filePath(QStringLiteral("Annotations")));
    result.outputFiles.insert(QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("JPEGImages")));
    result.ok = true;

    result.reportPath = QDir(result.outputPath).filePath(QStringLiteral("dataset_conversion_report.json"));
    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("convertedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    QString writeError;
    if (!writeJsonFile(result.reportPath, report, &writeError)) {
        result.ok = false;
        result.errorCode = QStringLiteral("report_write_failed");
        result.errorMessage = writeError;
    }
    return result;
}
```

- [ ] **Step 4: Dispatch YOLO detection to VOC**

In `convertDataset(...)`, before the final `not_implemented` return, add:

```cpp
if (sourceFormat == QStringLiteral("yolo_detection") && targetFormat == QStringLiteral("voc_xml")) {
    return convertYoloToVoc(request);
}
if (sourceFormat == QStringLiteral("yolo_segmentation") && targetFormat == QStringLiteral("voc_xml")) {
    result.errorCode = QStringLiteral("unsupported_target_format");
    result.errorMessage = QStringLiteral("YOLO segmentation to Pascal VOC XML is not supported.");
    return result;
}
```

- [ ] **Step 5: Run full core conversion tests**

Run:

```powershell
ctest --test-dir .\build-vscode --output-on-failure -R aitrain_dataset_conversion_tests
```

Expected: `aitrain_dataset_conversion_tests` passes.

- [ ] **Step 6: Run full harness**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: all 6 CTest executables pass.

- [ ] **Step 7: Commit green core conversion matrix**

Run:

```powershell
git add src/core/include/aitrain/core/DatasetConversion.h src/core/src/DatasetConversion.cpp tests/tst_dataset_conversion.cpp
git commit -m "feat: add dataset conversion matrix v2"
```

Expected: one green core conversion commit is created after `aitrain_dataset_conversion_tests` and the full harness have passed.

---

### Task 6: Add Worker and WorkerClient Conversion Route

**Files:**
- Modify: `src/worker/src/WorkerSession.h`
- Modify: `src/worker/src/WorkerSession.cpp`
- Modify: `src/worker/src/WorkerSessionDatasetCommands.cpp`
- Modify: `src/app/src/WorkerClient.h`
- Modify: `src/app/src/WorkerClient.cpp`

- [ ] **Step 1: Include DatasetConversion in Worker dataset commands**

At the top of `WorkerSessionDatasetCommands.cpp`, add:

```cpp
#include "aitrain/core/DatasetConversion.h"
```

- [ ] **Step 2: Declare Worker conversion method**

In `WorkerSession.h`, after `void splitDataset(const QJsonObject& payload);`, add:

```cpp
void convertDataset(const QJsonObject& payload);
```

- [ ] **Step 3: Route Worker command**

In `WorkerSession::handleMessage(...)`, after the `splitDataset` branch, add:

```cpp
} else if (type == QStringLiteral("convertDataset")) {
    convertDataset(payload);
```

- [ ] **Step 4: Implement Worker conversion command**

In `WorkerSessionDatasetCommands.cpp`, after `splitDataset(...)`, add:

```cpp
void WorkerSession::convertDataset(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    aitrain::DatasetConversionRequest request;
    request.sourcePath = payload.value(QStringLiteral("sourcePath")).toString();
    request.sourceFormat = payload.value(QStringLiteral("sourceFormat")).toString();
    request.targetFormat = payload.value(QStringLiteral("targetFormat")).toString();
    request.outputPath = payload.value(QStringLiteral("outputPath")).toString();
    request.options = payload.value(QStringLiteral("options")).toObject();

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始转换数据集。"));
    send(QStringLiteral("progress"), startProgress);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("数据集转换完成。"));
    send(QStringLiteral("progress"), doneProgress);

    QJsonObject response = result.toJson();
    response.insert(QStringLiteral("taskId"), taskId);
    send(QStringLiteral("datasetConversion"), response);

    if (!result.reportPath.isEmpty() && QFileInfo::exists(result.reportPath)) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("dataset_conversion_report"));
        artifact.insert(QStringLiteral("path"), result.reportPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Dataset conversion report"));
        send(QStringLiteral("artifact"), artifact);
    }

    QJsonObject terminal;
    terminal.insert(QStringLiteral("taskId"), taskId);
    terminal.insert(QStringLiteral("message"), result.ok
        ? QStringLiteral("Dataset conversion completed")
        : QStringLiteral("Dataset conversion failed"));
    if (!result.errorCode.isEmpty()) {
        terminal.insert(QStringLiteral("errorCode"), result.errorCode);
    }
    if (!result.errorMessage.isEmpty()) {
        terminal.insert(QStringLiteral("errorMessage"), result.errorMessage);
    }
    send(result.ok ? QStringLiteral("completed") : QStringLiteral("failed"), terminal);
    finishSession();
}
```

- [ ] **Step 5: Add WorkerClient request method declaration**

In `WorkerClient.h`, after `requestDatasetSplit(...)`, add:

```cpp
bool requestDatasetConversion(const QString& workerProgram, const QString& sourcePath, const QString& outputPath, const QString& sourceFormat, const QString& targetFormat, const QJsonObject& options, QString* error, const QString& taskId = {});
```

- [ ] **Step 6: Add WorkerClient implementation**

In `WorkerClient.cpp`, after `requestDatasetSplit(...)`, add:

```cpp
bool WorkerClient::requestDatasetConversion(const QString& workerProgram, const QString& sourcePath, const QString& outputPath, const QString& sourceFormat, const QString& targetFormat, const QJsonObject& options, QString* error, const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("sourcePath"), sourcePath);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("sourceFormat"), sourceFormat);
    payload.insert(QStringLiteral("targetFormat"), targetFormat);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("convertDataset"), payload, error);
}
```

- [ ] **Step 7: Build Worker route**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: build succeeds and all existing tests pass. No Worker conversion test exists yet.

- [ ] **Step 8: Commit Worker route**

Run:

```powershell
git add src/worker/src/WorkerSession.h src/worker/src/WorkerSession.cpp src/worker/src/WorkerSessionDatasetCommands.cpp src/app/src/WorkerClient.h src/app/src/WorkerClient.cpp
git commit -m "feat: route dataset conversion through worker"
```

Expected: one Worker route commit is created.

---

### Task 7: Add Worker Conversion Test

**Files:**
- Modify: `tests/tst_repository_workflow.cpp`

- [ ] **Step 1: Add test slot declaration**

In the relevant QtTest class in `tst_repository_workflow.cpp`, add:

```cpp
void workerRunsDatasetConversion();
```

- [ ] **Step 2: Add test body**

Append this test near other Worker command tests:

```cpp
void RepositoryWorkflowTests::workerRunsDatasetConversion()
{
    QTemporaryDir temp(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("aitrain_worker_conversion_XXXXXX")));
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QDir source(root.filePath(QStringLiteral("source_yolo")));
    writeTinyPng(QDir(source.path()).filePath(QStringLiteral("images/train/a.png")));
    writeTinyPng(QDir(source.path()).filePath(QStringLiteral("images/val/a.png")));
    writeTextFile(QDir(source.path()).filePath(QStringLiteral("labels/train/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    writeTextFile(QDir(source.path()).filePath(QStringLiteral("labels/val/a.txt")), QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    writeTextFile(QDir(source.path()).filePath(QStringLiteral("data.yaml")), QStringLiteral("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"));

    WorkerClient client;
    QVector<QPair<QString, QJsonObject>> messages;
    bool finished = false;
    bool ok = false;
    QString finishedMessage;
    connect(&client, &WorkerClient::messageReceived, this, [&messages](const QString& type, const QJsonObject& payload) {
        messages.append(qMakePair(type, payload));
    });
    connect(&client, &WorkerClient::finished, this, [&finished, &ok, &finishedMessage](bool success, const QString& message) {
        finished = true;
        ok = success;
        finishedMessage = message;
    });

    QString error;
    QVERIFY2(client.requestDatasetConversion(
                 workerExecutablePath(),
                 source.absolutePath(),
                 root.filePath(QStringLiteral("converted_coco")),
                 QStringLiteral("yolo_detection"),
                 QStringLiteral("coco_json"),
                 QJsonObject{{QStringLiteral("copyImages"), true}},
                 &error,
                 QStringLiteral("dataset-conversion-test")),
        qPrintable(error));

    QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
    QVERIFY2(ok, qPrintable(finishedMessage));
    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_coco/annotations/train.json"))));
    QVERIFY(QFileInfo::exists(root.filePath(QStringLiteral("converted_coco/dataset_conversion_report.json"))));

    bool sawConversionPayload = false;
    bool sawReportArtifact = false;
    for (const auto& message : messages) {
        if (message.first == QStringLiteral("datasetConversion")) {
            sawConversionPayload = true;
        }
        if (message.first == QStringLiteral("artifact")
            && message.second.value(QStringLiteral("kind")).toString() == QStringLiteral("dataset_conversion_report")) {
            sawReportArtifact = true;
        }
    }
    QVERIFY(sawConversionPayload);
    QVERIFY(sawReportArtifact);
}
```

- [ ] **Step 3: Run Worker test**

Run:

```powershell
ctest --test-dir .\build-vscode --output-on-failure -R aitrain_repository_workflow_tests
```

Expected: `aitrain_repository_workflow_tests` passes.

- [ ] **Step 4: Run full harness**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: all 6 CTest executables pass.

- [ ] **Step 5: Commit Worker test**

Run:

```powershell
git add tests/tst_repository_workflow.cpp
git commit -m "test: cover worker dataset conversion"
```

Expected: one test commit is created.

---

### Task 8: Final Verification and Cleanup

**Files:**
- Review all modified source, test, and docs files.

- [ ] **Step 1: Check status**

Run:

```powershell
git status --short
```

Expected: only intentional files are modified. No build artifacts, `.deps` files, or `.pdb` files are staged.

- [ ] **Step 2: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Run full harness**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: configure, build, and all CTest executables pass.

- [ ] **Step 4: Inspect final conversion report shape**

Run:

```powershell
Get-Content .\build-vscode\tests\aitrain_dataset_conversion_tests_ctest.txt -Tail 40
```

Expected: `aitrain_dataset_conversion_tests` reports all dataset conversion tests passed.

- [ ] **Step 5: Final commit if any verification-only edits were needed**

Run this only when Step 1 shows intentional unstaged edits:

```powershell
git add src/core/include/aitrain/core/DatasetConversion.h src/core/src/DatasetConversion.cpp src/worker/src/WorkerSession.h src/worker/src/WorkerSession.cpp src/worker/src/WorkerSessionDatasetCommands.cpp src/app/src/WorkerClient.h src/app/src/WorkerClient.cpp tests/tst_dataset_conversion.cpp tests/tst_repository_workflow.cpp
git commit -m "chore: finalize dataset conversion matrix v2"
```

Expected: either no commit is needed because the tree is clean, or one small final commit captures verification fixes.

---

## Final Acceptance Checklist

- [ ] `coco_json -> yolo_detection` still works.
- [ ] `coco_json -> yolo_segmentation` still works.
- [ ] `voc_xml -> yolo_detection` still works.
- [ ] `yolo_detection -> coco_json` works with train/val split output.
- [ ] `yolo_detection -> voc_xml` works with VOC XML and copied JPEGImages.
- [ ] `yolo_segmentation -> coco_json` writes polygon segmentation arrays.
- [ ] `copyImages=false` avoids image copies and reports `imagePolicy=referenced`.
- [ ] Invalid YOLO labels create actionable issues while valid annotations still convert.
- [ ] Worker emits `datasetConversion`, `artifact`, and terminal `completed`/`failed`.
- [ ] `.\tools\harness-check.ps1` passes.
