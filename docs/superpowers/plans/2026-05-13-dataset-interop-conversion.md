# Dataset Interop Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Worker-managed conversion from COCO JSON, Pascal VOC XML, and LabelMe JSON into AITrain trainable YOLO detection or YOLO segmentation datasets.

**Architecture:** Add a focused `DatasetConversion` core module with deterministic conversion reports and post-conversion YOLO validation. Route conversion through `aitrain_worker`, expose a compact conversion strip on the existing Qt dataset page, and record conversion plus validation reports through existing task artifact flow without changing SQLite schema.

**Tech Stack:** C++20, Qt 5.12+ Core/Gui/Widgets/Network/Sql/Test, CMake, QtTest, existing `WorkerClient`, `WorkerSession`, `DatasetValidators`, `ProjectRepository`, `InfoPanel`, `StatusPill`, and `QStringLiteral` UI text.

---

## File Map

Create:

- `src/core/include/aitrain/core/DatasetConversion.h`: public request/result structs and `convertDataset(...)` API.
- `src/core/src/DatasetConversion.cpp`: conversion implementation for COCO, VOC, LabelMe, YOLO output writing, report JSON, and post-conversion validation.
- `tests/tst_dataset_conversion.cpp`: core and Worker tests for conversion formats and failure behavior.

Modify:

- `src/core/CMakeLists.txt`: add new header/source to `aitrain_core`.
- `tests/CMakeLists.txt`: add `aitrain_dataset_conversion_tests`.
- `src/worker/src/WorkerSession.h`: declare `convertDataset(...)`.
- `src/worker/src/WorkerSession.cpp`: dispatch `convertDataset`.
- `src/worker/src/WorkerSessionDatasetCommands.cpp`: implement Worker command and artifact emission.
- `src/app/src/WorkerClient.h`: add `requestDatasetConversion(...)`.
- `src/app/src/WorkerClient.cpp`: serialize and send `convertDataset`.
- `src/app/src/MainWindow.h`: add slots and widgets for conversion controls; add handler declaration.
- `src/app/src/MainWindowDatasetPage.cpp`: add compact conversion section to existing dataset workbench.
- `src/app/src/MainWindowActions.cpp`: implement browse/start conversion actions.
- `src/app/src/MainWindowWorkerMessages.cpp`: route `datasetConversion` messages.
- `src/app/src/MainWindowRecords.cpp` or existing dataset list refresh path only if conversion result needs a local UI summary helper.
- `src/app/src/LanguageSupport.cpp`: add English mappings for new Chinese UI strings.
- `docs/user-guide.md`: document converting external formats.
- `docs/training-backends.md`: document normalized dataset boundary.
- `docs/harness/current-status.md`: add a new phase/status entry after implementation.
- `docs/product-roadmap-local-training-platform.md`: add the dataset interop conversion closeout notes.

Do not modify:

- `ProjectRepository` schema.
- Plugin interfaces.
- Training backends.
- Python trainer adapters.

---

### Task 1: Core API and Build Wiring

**Files:**
- Create: `src/core/include/aitrain/core/DatasetConversion.h`
- Create: `src/core/src/DatasetConversion.cpp`
- Modify: `src/core/CMakeLists.txt`
- Test: `tests/tst_dataset_conversion.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add a failing compile-level test for the new API**

Create `tests/tst_dataset_conversion.cpp` with this initial test fixture:

```cpp
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
```

- [ ] **Step 2: Register the failing test target**

Modify `tests/CMakeLists.txt`:

```cmake
add_aitrain_qt_test(aitrain_dataset_conversion_tests tst_dataset_conversion.cpp)
```

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
```

Expected: FAIL because `aitrain/core/DatasetConversion.h` does not exist.

- [ ] **Step 3: Add the public header**

Create `src/core/include/aitrain/core/DatasetConversion.h`:

```cpp
#pragma once

#include "aitrain/core/PluginInterfaces.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QString>

namespace aitrain {

struct DatasetConversionIssue {
    QString severity;
    QString code;
    QString sourceFile;
    QString imagePath;
    QString category;
    QString message;

    QJsonObject toJson() const;
};

struct DatasetConversionRequest {
    QString sourcePath;
    QString sourceFormat;
    QString targetFormat;
    QString outputPath;
    QJsonObject options;
};

struct DatasetConversionResult {
    bool ok = false;
    QString errorCode;
    QString errorMessage;
    QString sourceFormat;
    QString targetFormat;
    QString sourcePath;
    QString outputPath;
    QString reportPath;
    QString validationReportPath;
    int sampleCount = 0;
    int convertedSampleCount = 0;
    int skippedSampleCount = 0;
    int annotationCount = 0;
    int convertedAnnotationCount = 0;
    int skippedAnnotationCount = 0;
    QJsonObject classMap;
    QVector<DatasetConversionIssue> issues;
    DatasetValidationResult targetValidation;

    QJsonObject toJson() const;
};

DatasetConversionResult convertDataset(const DatasetConversionRequest& request);

} // namespace aitrain
```

- [ ] **Step 4: Add a minimal implementation**

Create `src/core/src/DatasetConversion.cpp`:

```cpp
#include "aitrain/core/DatasetConversion.h"

#include "aitrain/core/DatasetValidators.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>

namespace aitrain {

QJsonObject DatasetConversionIssue::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("severity"), severity);
    object.insert(QStringLiteral("code"), code);
    object.insert(QStringLiteral("sourceFile"), sourceFile);
    object.insert(QStringLiteral("imagePath"), imagePath);
    object.insert(QStringLiteral("category"), category);
    object.insert(QStringLiteral("message"), message);
    return object;
}

QJsonObject DatasetConversionResult::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("ok"), ok);
    object.insert(QStringLiteral("errorCode"), errorCode);
    object.insert(QStringLiteral("errorMessage"), errorMessage);
    object.insert(QStringLiteral("sourceFormat"), sourceFormat);
    object.insert(QStringLiteral("targetFormat"), targetFormat);
    object.insert(QStringLiteral("sourcePath"), sourcePath);
    object.insert(QStringLiteral("outputPath"), outputPath);
    object.insert(QStringLiteral("reportPath"), reportPath);
    object.insert(QStringLiteral("validationReportPath"), validationReportPath);
    object.insert(QStringLiteral("sampleCount"), sampleCount);
    object.insert(QStringLiteral("convertedSampleCount"), convertedSampleCount);
    object.insert(QStringLiteral("skippedSampleCount"), skippedSampleCount);
    object.insert(QStringLiteral("annotationCount"), annotationCount);
    object.insert(QStringLiteral("convertedAnnotationCount"), convertedAnnotationCount);
    object.insert(QStringLiteral("skippedAnnotationCount"), skippedAnnotationCount);
    object.insert(QStringLiteral("classMap"), classMap);
    object.insert(QStringLiteral("targetValidation"), targetValidation.toJson());

    QJsonArray issueArray;
    for (const DatasetConversionIssue& issue : issues) {
        issueArray.append(issue.toJson());
    }
    object.insert(QStringLiteral("issues"), issueArray);
    return object;
}

DatasetConversionResult convertDataset(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = request.sourceFormat;
    result.targetFormat = request.targetFormat;
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);

    const QString sourceFormat = request.sourceFormat.trimmed().toLower();
    if (sourceFormat != QStringLiteral("coco_json")
        && sourceFormat != QStringLiteral("voc_xml")
        && sourceFormat != QStringLiteral("labelme_json")) {
        result.errorCode = QStringLiteral("unsupported_source_format");
        result.errorMessage = QStringLiteral("Unsupported dataset source format: %1").arg(request.sourceFormat);
        return result;
    }

    result.errorCode = QStringLiteral("not_implemented");
    result.errorMessage = QStringLiteral("Dataset conversion parser is not implemented yet.");
    return result;
}

} // namespace aitrain
```

- [ ] **Step 5: Wire the files into core CMake**

Modify `src/core/CMakeLists.txt`:

```cmake
    include/aitrain/core/DatasetConversion.h
```

Place it next to `DatasetValidators.h`.

Add:

```cmake
    src/DatasetConversion.cpp
```

Place it next to `DatasetValidators.cpp`.

- [ ] **Step 6: Run the API test**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: PASS for `unsupportedFormatFails`.

- [ ] **Step 7: Commit**

```powershell
git add src/core/include/aitrain/core/DatasetConversion.h src/core/src/DatasetConversion.cpp src/core/CMakeLists.txt tests/tst_dataset_conversion.cpp tests/CMakeLists.txt
git commit -m "test: add dataset conversion API shell"
```

---

### Task 2: COCO Detection Conversion

**Files:**
- Modify: `tests/tst_dataset_conversion.cpp`
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add the failing COCO bbox test**

Add these helper functions near the top of `tests/tst_dataset_conversion.cpp`:

```cpp
namespace {

void writeTextFile(const QString& path, const QString& text)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
    file.write(text.toUtf8());
}

void writeTinyPng(const QString& path, int width = 100, int height = 80)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QImage image(width, height, QImage::Format_RGB888);
    image.fill(Qt::white);
    QVERIFY(image.save(path));
}

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
```

Add a private slot:

```cpp
void cocoDetectionConvertsBboxToYolo();
```

Add the test:

```cpp
void DatasetConversionTests::cocoDetectionConvertsBboxToYolo()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());

    writeTinyPng(root.filePath(QStringLiteral("images/a.png")), 100, 80);
    writeTextFile(root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("{"
                       "\"images\":[{\"id\":1,\"file_name\":\"images/a.png\",\"width\":100,\"height\":80}],"
                       "\"categories\":[{\"id\":7,\"name\":\"widget\"}],"
                       "\"annotations\":[{\"id\":10,\"image_id\":1,\"category_id\":7,\"bbox\":[10,20,30,16]}]"
                       "}"));

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
    QCOMPARE(label, QStringLiteral("0 0.250000 0.350000 0.300000 0.200000"));

    const QJsonObject report = readJsonObjectForTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("sourceFormat")).toString(), QStringLiteral("coco_json"));
    QCOMPARE(report.value(QStringLiteral("targetFormat")).toString(), QStringLiteral("yolo_detection"));
    QVERIFY(report.value(QStringLiteral("targetValidation")).toObject().value(QStringLiteral("ok")).toBool());
}
```

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: FAIL with `not_implemented`.

- [ ] **Step 2: Implement shared file/report helpers**

In `DatasetConversion.cpp`, add helpers inside an anonymous namespace before `namespace aitrain` closes:

```cpp
namespace {

QString normalizedFormat(const QString& value)
{
    return value.trimmed().toLower();
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write JSON file: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

bool writeTextFile(const QString& path, const QString& content, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot write text file: %1").arg(path);
        }
        return false;
    }
    QTextStream stream(&file);
    stream.setCodec("UTF-8");
    stream << content;
    return true;
}

QString yoloNumber(double value)
{
    return QString::number(value, 'f', 6);
}

QString safeBaseName(const QString& path)
{
    QString base = QFileInfo(path).completeBaseName();
    if (base.trimmed().isEmpty()) {
        base = QStringLiteral("sample");
    }
    return base.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9_.-]")), QStringLiteral("_"));
}

bool copyImageToTrain(const QString& sourceImagePath, const QString& outputPath, QString* copiedRelativePath, QString* error)
{
    const QFileInfo sourceInfo(sourceImagePath);
    const QString relative = QStringLiteral("images/train/%1").arg(sourceInfo.fileName());
    const QString destination = QDir(outputPath).filePath(relative);
    QDir().mkpath(QFileInfo(destination).absolutePath());
    if (QFileInfo::exists(destination)) {
        QFile::remove(destination);
    }
    if (!QFile::copy(sourceImagePath, destination)) {
        if (error) {
            *error = QStringLiteral("Cannot copy image %1 to %2").arg(sourceImagePath, destination);
        }
        return false;
    }
    if (copiedRelativePath) {
        *copiedRelativePath = relative;
    }
    return true;
}

DatasetConversionIssue issue(
    const QString& severity,
    const QString& code,
    const QString& sourceFile,
    const QString& imagePath,
    const QString& category,
    const QString& message)
{
    DatasetConversionIssue item;
    item.severity = severity;
    item.code = code;
    item.sourceFile = sourceFile;
    item.imagePath = imagePath;
    item.category = category;
    item.message = message;
    return item;
}

} // namespace
```

Add required includes:

```cpp
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QTextStream>
#include <algorithm>
```

- [ ] **Step 3: Implement COCO detection conversion**

Add a private helper in `DatasetConversion.cpp`:

```cpp
DatasetConversionResult convertCoco(const DatasetConversionRequest& request)
{
    DatasetConversionResult result;
    result.sourceFormat = QStringLiteral("coco_json");
    result.targetFormat = normalizedFormat(request.targetFormat);
    result.sourcePath = QDir::cleanPath(request.sourcePath);
    result.outputPath = QDir::cleanPath(request.outputPath);

    QFile file(result.sourcePath);
    if (!file.open(QIODevice::ReadOnly)) {
        result.errorCode = QStringLiteral("source_read_failed");
        result.errorMessage = QStringLiteral("Cannot read COCO JSON: %1").arg(result.sourcePath);
        return result;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        result.errorCode = QStringLiteral("source_parse_failed");
        result.errorMessage = QStringLiteral("Cannot parse COCO JSON %1: %2").arg(result.sourcePath, parseError.errorString());
        return result;
    }

    const QJsonObject root = document.object();
    const QJsonArray images = root.value(QStringLiteral("images")).toArray();
    const QJsonArray categories = root.value(QStringLiteral("categories")).toArray();
    const QJsonArray annotations = root.value(QStringLiteral("annotations")).toArray();
    result.sampleCount = images.size();
    result.annotationCount = annotations.size();

    QMap<int, QJsonObject> imagesById;
    for (const QJsonValue& value : images) {
        const QJsonObject image = value.toObject();
        imagesById.insert(image.value(QStringLiteral("id")).toInt(), image);
    }

    QMap<int, QString> categoryNames;
    for (const QJsonValue& value : categories) {
        const QJsonObject category = value.toObject();
        categoryNames.insert(category.value(QStringLiteral("id")).toInt(), category.value(QStringLiteral("name")).toString().trimmed());
    }

    QMap<int, int> categoryIdToClassId;
    int nextClassId = 0;
    for (auto it = categoryNames.constBegin(); it != categoryNames.constEnd(); ++it) {
        if (it.value().isEmpty()) {
            continue;
        }
        categoryIdToClassId.insert(it.key(), nextClassId);
        result.classMap.insert(QString::number(nextClassId), it.value());
        ++nextClassId;
    }

    QMap<int, QStringList> labelsByImageId;
    for (const QJsonValue& value : annotations) {
        const QJsonObject annotation = value.toObject();
        const int imageId = annotation.value(QStringLiteral("image_id")).toInt();
        const int categoryId = annotation.value(QStringLiteral("category_id")).toInt();
        if (!imagesById.contains(imageId) || !categoryIdToClassId.contains(categoryId)) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("missing_image_or_category"), result.sourcePath, {}, QString::number(categoryId), QStringLiteral("COCO annotation references a missing image or category.")));
            ++result.skippedAnnotationCount;
            continue;
        }
        const QJsonArray bbox = annotation.value(QStringLiteral("bbox")).toArray();
        if (bbox.size() != 4) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), result.sourcePath, imagesById.value(imageId).value(QStringLiteral("file_name")).toString(), QString::number(categoryId), QStringLiteral("COCO bbox must contain four numbers.")));
            ++result.skippedAnnotationCount;
            continue;
        }
        const QJsonObject image = imagesById.value(imageId);
        const double width = image.value(QStringLiteral("width")).toDouble();
        const double height = image.value(QStringLiteral("height")).toDouble();
        const double x = bbox.at(0).toDouble();
        const double y = bbox.at(1).toDouble();
        const double w = bbox.at(2).toDouble();
        const double h = bbox.at(3).toDouble();
        if (width <= 0.0 || height <= 0.0 || w <= 0.0 || h <= 0.0) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_bbox"), result.sourcePath, image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId), QStringLiteral("COCO bbox or image size is invalid.")));
            ++result.skippedAnnotationCount;
            continue;
        }
        const QString row = QStringLiteral("%1 %2 %3 %4 %5")
            .arg(categoryIdToClassId.value(categoryId))
            .arg(yoloNumber((x + w / 2.0) / width))
            .arg(yoloNumber((y + h / 2.0) / height))
            .arg(yoloNumber(w / width))
            .arg(yoloNumber(h / height));
        labelsByImageId[imageId].append(row);
        ++result.convertedAnnotationCount;
    }

    QDir().mkpath(result.outputPath);
    int convertedSamples = 0;
    for (auto it = labelsByImageId.constBegin(); it != labelsByImageId.constEnd(); ++it) {
        const QJsonObject image = imagesById.value(it.key());
        const QString fileName = image.value(QStringLiteral("file_name")).toString();
        const QString sourceImagePath = QFileInfo(fileName).isAbsolute()
            ? fileName
            : QFileInfo(result.sourcePath).absoluteDir().filePath(fileName);
        QString copyError;
        QString copiedRelativePath;
        if (!copyImageToTrain(sourceImagePath, result.outputPath, &copiedRelativePath, &copyError)) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("image_copy_failed"), result.sourcePath, fileName, {}, copyError));
            ++result.skippedSampleCount;
            continue;
        }
        QString writeError;
        const QString labelPath = QDir(result.outputPath).filePath(QStringLiteral("labels/train/%1.txt").arg(safeBaseName(copiedRelativePath)));
        if (!writeTextFile(labelPath, it.value().join(QLatin1Char('\n')) + QLatin1Char('\n'), &writeError)) {
            result.errorCode = QStringLiteral("output_write_failed");
            result.errorMessage = writeError;
            return result;
        }
        ++convertedSamples;
    }
    result.convertedSampleCount = convertedSamples;
    result.skippedSampleCount += qMax(0, result.sampleCount - convertedSamples);

    if (result.convertedSampleCount <= 0 || result.convertedAnnotationCount <= 0) {
        result.errorCode = QStringLiteral("no_convertible_samples");
        result.errorMessage = QStringLiteral("COCO dataset did not contain convertible annotations.");
        return result;
    }

    QStringList names;
    for (int index = 0; index < result.classMap.size(); ++index) {
        names.append(result.classMap.value(QString::number(index)).toString());
    }
    QString yaml = QStringLiteral("path: .\ntrain: images/train\nval: images/train\nnc: %1\nnames:\n").arg(names.size());
    for (int index = 0; index < names.size(); ++index) {
        yaml += QStringLiteral("  %1: %2\n").arg(index).arg(names.at(index));
    }
    QString writeError;
    if (!writeTextFile(QDir(result.outputPath).filePath(QStringLiteral("data.yaml")), yaml, &writeError)) {
        result.errorCode = QStringLiteral("output_write_failed");
        result.errorMessage = writeError;
        return result;
    }

    result.targetValidation = validateYoloDetectionDataset(result.outputPath);
    result.ok = result.targetValidation.ok;
    if (!result.ok) {
        result.errorCode = QStringLiteral("target_validation_failed");
        result.errorMessage = QStringLiteral("Converted YOLO detection dataset failed validation.");
    }

    result.reportPath = QDir(result.outputPath).filePath(QStringLiteral("dataset_conversion_report.json"));
    QJsonObject report = result.toJson();
    report.insert(QStringLiteral("convertedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    report.insert(QStringLiteral("artifacts"), QJsonObject{
        {QStringLiteral("dataYaml"), QDir(result.outputPath).filePath(QStringLiteral("data.yaml"))},
        {QStringLiteral("imagesRoot"), QDir(result.outputPath).filePath(QStringLiteral("images"))},
        {QStringLiteral("labelsRoot"), QDir(result.outputPath).filePath(QStringLiteral("labels"))}
    });
    if (!writeJsonFile(result.reportPath, report, &writeError)) {
        result.ok = false;
        result.errorCode = QStringLiteral("report_write_failed");
        result.errorMessage = writeError;
    }
    return result;
}
```

Add needed includes:

```cpp
#include <QMap>
```

Update `convertDataset(...)` to call `convertCoco(request)` when `sourceFormat == "coco_json"` and target is `yolo_detection`.

- [ ] **Step 4: Run the COCO detection test**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: PASS for unsupported format and COCO detection.

- [ ] **Step 5: Commit**

```powershell
git add tests/tst_dataset_conversion.cpp src/core/src/DatasetConversion.cpp
git commit -m "feat: convert COCO bbox datasets to YOLO"
```

---

### Task 3: COCO Segmentation and RLE Reporting

**Files:**
- Modify: `tests/tst_dataset_conversion.cpp`
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add failing tests for COCO polygon and RLE skip**

Add private slots:

```cpp
void cocoSegmentationConvertsPolygonToYolo();
void cocoSegmentationSkipsRleMasks();
```

Add polygon test:

```cpp
void DatasetConversionTests::cocoSegmentationConvertsPolygonToYolo()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPng(root.filePath(QStringLiteral("images/seg.png")), 100, 100);
    writeTextFile(root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("{"
                       "\"images\":[{\"id\":1,\"file_name\":\"images/seg.png\",\"width\":100,\"height\":100}],"
                       "\"categories\":[{\"id\":1,\"name\":\"part\"}],"
                       "\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":1,"
                       "\"segmentation\":[[10,10,90,10,90,90,10,90]]}]"
                       "}"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.filePath(QStringLiteral("annotations.json"));
    request.sourceFormat = QStringLiteral("coco_json");
    request.targetFormat = QStringLiteral("yolo_segmentation");
    request.outputPath = root.filePath(QStringLiteral("converted_seg"));
    request.options.insert(QStringLiteral("copyImages"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    const QString label = readTextFile(root.filePath(QStringLiteral("converted_seg/labels/train/seg.txt"))).trimmed();
    QCOMPARE(label, QStringLiteral("0 0.100000 0.100000 0.900000 0.100000 0.900000 0.900000 0.100000 0.900000"));
    QVERIFY(result.targetValidation.ok);
}
```

Add RLE test:

```cpp
void DatasetConversionTests::cocoSegmentationSkipsRleMasks()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPng(root.filePath(QStringLiteral("images/rle.png")), 20, 20);
    writeTextFile(root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("{"
                       "\"images\":[{\"id\":1,\"file_name\":\"images/rle.png\",\"width\":20,\"height\":20}],"
                       "\"categories\":[{\"id\":1,\"name\":\"mask\"}],"
                       "\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":1,"
                       "\"segmentation\":{\"counts\":\"abc\",\"size\":[20,20]}}]"
                       "}"));

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
```

Run the test target. Expected: FAIL because segmentation is not implemented.

- [ ] **Step 2: Extend COCO conversion for segmentation**

In `convertCoco(...)`, branch by `result.targetFormat`. For `yolo_segmentation`, read `segmentation` as an array of polygon arrays:

```cpp
if (result.targetFormat == QStringLiteral("yolo_segmentation")) {
    const QJsonValue segmentation = annotation.value(QStringLiteral("segmentation"));
    if (segmentation.isObject()) {
        result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("rle_not_supported"), result.sourcePath, image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId), QStringLiteral("COCO RLE masks are not converted in this version.")));
        ++result.skippedAnnotationCount;
        continue;
    }
    const QJsonArray segments = segmentation.toArray();
    if (segments.isEmpty()) {
        result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), result.sourcePath, image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId), QStringLiteral("COCO polygon segmentation is empty.")));
        ++result.skippedAnnotationCount;
        continue;
    }
    for (const QJsonValue& segmentValue : segments) {
        const QJsonArray polygon = segmentValue.toArray();
        if (polygon.size() < 6 || polygon.size() % 2 != 0) {
            result.issues.append(issue(QStringLiteral("warning"), QStringLiteral("invalid_polygon"), result.sourcePath, image.value(QStringLiteral("file_name")).toString(), QString::number(categoryId), QStringLiteral("COCO polygon must contain at least three x/y points.")));
            ++result.skippedAnnotationCount;
            continue;
        }
        QStringList parts;
        parts.append(QString::number(categoryIdToClassId.value(categoryId)));
        for (int i = 0; i + 1 < polygon.size(); i += 2) {
            parts.append(yoloNumber(polygon.at(i).toDouble() / width));
            parts.append(yoloNumber(polygon.at(i + 1).toDouble() / height));
        }
        labelsByImageId[imageId].append(parts.join(QLatin1Char(' ')));
        ++result.convertedAnnotationCount;
    }
    continue;
}
```

Use `validateYoloSegmentationDataset(result.outputPath)` when `targetFormat == "yolo_segmentation"`.

- [ ] **Step 3: Run tests**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: PASS for COCO detection, COCO segmentation, RLE skip, unsupported format.

- [ ] **Step 4: Commit**

```powershell
git add tests/tst_dataset_conversion.cpp src/core/src/DatasetConversion.cpp
git commit -m "feat: convert COCO polygons to YOLO segmentation"
```

---

### Task 4: Pascal VOC Detection Conversion

**Files:**
- Modify: `tests/tst_dataset_conversion.cpp`
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add failing VOC test**

Add private slot:

```cpp
void vocXmlConvertsBoxesToYoloDetection();
```

Add test:

```cpp
void DatasetConversionTests::vocXmlConvertsBoxesToYoloDetection()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPng(root.filePath(QStringLiteral("JPEGImages/a.png")), 100, 80);
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
    const QString label = readTextFile(root.filePath(QStringLiteral("converted_voc/labels/train/a.txt"))).trimmed();
    QCOMPARE(label, QStringLiteral("0 0.250000 0.350000 0.300000 0.200000"));
}
```

Run the test target. Expected: FAIL with unsupported or not implemented.

- [ ] **Step 2: Implement VOC XML parsing**

In `DatasetConversion.cpp`, add:

```cpp
#include <QDirIterator>
#include <QXmlStreamReader>
```

Create a helper that reads one XML file:

```cpp
struct VocObject {
    QString name;
    double xmin = 0.0;
    double ymin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
};

struct VocAnnotation {
    QString filename;
    int width = 0;
    int height = 0;
    QVector<VocObject> objects;
};
```

Parse with `QXmlStreamReader` by reading element text for `filename`, `path`, `width`, `height`, `name`, `xmin`, `ymin`, `xmax`, `ymax`. Resolve image path by checking:

```text
<sourceDir>/<filename>
<sourceDir>/../JPEGImages/<filename>
<sourceDir>/../images/<filename>
```

Create class ids by sorted unique object names:

```cpp
QStringList classNames;
classNames.removeDuplicates();
std::sort(classNames.begin(), classNames.end());
```

Write YOLO rows:

```cpp
const double boxW = object.xmax - object.xmin;
const double boxH = object.ymax - object.ymin;
const double cx = object.xmin + boxW / 2.0;
const double cy = object.ymin + boxH / 2.0;
```

Reject `targetFormat == "yolo_segmentation"` with:

```cpp
result.errorCode = QStringLiteral("unsupported_target_format");
result.errorMessage = QStringLiteral("Pascal VOC XML conversion supports YOLO detection only.");
```

- [ ] **Step 3: Run tests**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: PASS for VOC and previous tests.

- [ ] **Step 4: Commit**

```powershell
git add tests/tst_dataset_conversion.cpp src/core/src/DatasetConversion.cpp
git commit -m "feat: convert Pascal VOC boxes to YOLO"
```

---

### Task 5: LabelMe Detection and Segmentation Conversion

**Files:**
- Modify: `tests/tst_dataset_conversion.cpp`
- Modify: `src/core/src/DatasetConversion.cpp`

- [ ] **Step 1: Add failing LabelMe tests**

Add private slots:

```cpp
void labelMeRectangleConvertsToYoloDetection();
void labelMePolygonConvertsToYoloSegmentation();
void labelMePolygonConvertsToDetectionBoxWhenEnabled();
```

Add rectangle test:

```cpp
void DatasetConversionTests::labelMeRectangleConvertsToYoloDetection()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPng(root.filePath(QStringLiteral("a.png")), 100, 80);
    writeTextFile(root.filePath(QStringLiteral("a.json")),
        QStringLiteral("{\"imagePath\":\"a.png\",\"imageWidth\":100,\"imageHeight\":80,"
                       "\"shapes\":[{\"label\":\"box\",\"shape_type\":\"rectangle\","
                       "\"points\":[[10,20],[40,36]]}]}"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.path();
    request.sourceFormat = QStringLiteral("labelme_json");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted_labelme_det"));

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(readTextFile(root.filePath(QStringLiteral("converted_labelme_det/labels/train/a.txt"))).trimmed(),
             QStringLiteral("0 0.250000 0.350000 0.300000 0.200000"));
}
```

Add polygon segmentation test:

```cpp
void DatasetConversionTests::labelMePolygonConvertsToYoloSegmentation()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPng(root.filePath(QStringLiteral("poly.png")), 100, 100);
    writeTextFile(root.filePath(QStringLiteral("poly.json")),
        QStringLiteral("{\"imagePath\":\"poly.png\",\"imageWidth\":100,\"imageHeight\":100,"
                       "\"shapes\":[{\"label\":\"part\",\"shape_type\":\"polygon\","
                       "\"points\":[[10,10],[90,10],[90,90],[10,90]]}]}"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.path();
    request.sourceFormat = QStringLiteral("labelme_json");
    request.targetFormat = QStringLiteral("yolo_segmentation");
    request.outputPath = root.filePath(QStringLiteral("converted_labelme_seg"));

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(readTextFile(root.filePath(QStringLiteral("converted_labelme_seg/labels/train/poly.txt"))).trimmed(),
             QStringLiteral("0 0.100000 0.100000 0.900000 0.100000 0.900000 0.900000 0.100000 0.900000"));
}
```

Add polygon-to-box test:

```cpp
void DatasetConversionTests::labelMePolygonConvertsToDetectionBoxWhenEnabled()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPng(root.filePath(QStringLiteral("polybox.png")), 100, 100);
    writeTextFile(root.filePath(QStringLiteral("polybox.json")),
        QStringLiteral("{\"imagePath\":\"polybox.png\",\"imageWidth\":100,\"imageHeight\":100,"
                       "\"shapes\":[{\"label\":\"part\",\"shape_type\":\"polygon\","
                       "\"points\":[[10,10],[90,20],[70,80]]}]}"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.path();
    request.sourceFormat = QStringLiteral("labelme_json");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted_labelme_polybox"));
    request.options.insert(QStringLiteral("polygonToBox"), true);

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));
    QCOMPARE(readTextFile(root.filePath(QStringLiteral("converted_labelme_polybox/labels/train/polybox.txt"))).trimmed(),
             QStringLiteral("0 0.500000 0.450000 0.800000 0.700000"));
}
```

Run the test target. Expected: FAIL because LabelMe is not implemented.

- [ ] **Step 2: Implement LabelMe parsing**

Add `convertLabelMe(...)` that:

- Iterates `*.json` under `sourcePath`.
- Reads each JSON as an object.
- Resolves `imagePath` relative to the JSON file directory.
- Reads `imageWidth` and `imageHeight`.
- Builds sorted class names from `shapes[].label`.
- Handles `rectangle` by using the two points as bbox corners.
- Handles `polygon` by either writing YOLO polygon rows or enclosing bbox rows when `targetFormat == "yolo_detection"` and `polygonToBox=true`.
- Records unsupported shapes as `unsupported_shape`.

Use this row creation logic:

```cpp
QString yoloBoxRow(int classId, double xmin, double ymin, double xmax, double ymax, double width, double height)
{
    const double boxW = xmax - xmin;
    const double boxH = ymax - ymin;
    return QStringLiteral("%1 %2 %3 %4 %5")
        .arg(classId)
        .arg(yoloNumber((xmin + boxW / 2.0) / width))
        .arg(yoloNumber((ymin + boxH / 2.0) / height))
        .arg(yoloNumber(boxW / width))
        .arg(yoloNumber(boxH / height));
}
```

For polygon rows:

```cpp
QStringList parts;
parts.append(QString::number(classId));
for (const QPointF& point : points) {
    parts.append(yoloNumber(point.x() / width));
    parts.append(yoloNumber(point.y() / height));
}
labels.append(parts.join(QLatin1Char(' ')));
```

Write the same YOLO output layout and validation report as COCO/VOC.

- [ ] **Step 3: Run tests**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: PASS for all conversion tests.

- [ ] **Step 4: Commit**

```powershell
git add tests/tst_dataset_conversion.cpp src/core/src/DatasetConversion.cpp
git commit -m "feat: convert LabelMe annotations to YOLO"
```

---

### Task 6: Worker Command and Client Request

**Files:**
- Modify: `src/worker/src/WorkerSession.h`
- Modify: `src/worker/src/WorkerSession.cpp`
- Modify: `src/worker/src/WorkerSessionDatasetCommands.cpp`
- Modify: `src/app/src/WorkerClient.h`
- Modify: `src/app/src/WorkerClient.cpp`
- Modify: `tests/tst_dataset_conversion.cpp`

- [ ] **Step 1: Add a failing Worker integration test**

Add private slot:

```cpp
void workerConvertsDatasetAndEmitsArtifacts();
```

Add test:

```cpp
void DatasetConversionTests::workerConvertsDatasetAndEmitsArtifacts()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    writeTinyPng(root.filePath(QStringLiteral("images/a.png")), 100, 80);
    writeTextFile(root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("{"
                       "\"images\":[{\"id\":1,\"file_name\":\"images/a.png\",\"width\":100,\"height\":80}],"
                       "\"categories\":[{\"id\":1,\"name\":\"widget\"}],"
                       "\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":1,\"bbox\":[10,20,30,16]}]"
                       "}"));

    WorkerClient client;
    QVector<QString> messageTypes;
    QVector<QJsonObject> artifacts;
    connect(&client, &WorkerClient::messageReceived, this, [&](const QString& type, const QJsonObject& payload) {
        messageTypes.append(type);
        if (type == QStringLiteral("artifact")) {
            artifacts.append(payload);
        }
    });

    QSignalSpy idleSpy(&client, &WorkerClient::idle);
    QString error;
    QVERIFY2(client.requestDatasetConversion(
        workerExecutablePath(),
        root.filePath(QStringLiteral("annotations.json")),
        QStringLiteral("coco_json"),
        QStringLiteral("yolo_detection"),
        root.filePath(QStringLiteral("worker_converted")),
        QJsonObject{{QStringLiteral("copyImages"), true}},
        &error,
        QStringLiteral("conversion-test-task")), qPrintable(error));
    QVERIFY(idleSpy.wait(30000));

    QVERIFY(messageTypes.contains(QStringLiteral("datasetConversion")));
    QVERIFY(messageTypes.contains(QStringLiteral("datasetValidation")));
    QVERIFY(messageTypes.contains(QStringLiteral("completed")));

    bool hasConversionArtifact = false;
    bool hasValidationArtifact = false;
    for (const QJsonObject& artifact : artifacts) {
        hasConversionArtifact = hasConversionArtifact || artifact.value(QStringLiteral("kind")).toString() == QStringLiteral("dataset_conversion_report");
        hasValidationArtifact = hasValidationArtifact || artifact.value(QStringLiteral("kind")).toString() == QStringLiteral("dataset_validation_report");
    }
    QVERIFY(hasConversionArtifact);
    QVERIFY(hasValidationArtifact);
}
```

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: FAIL because `WorkerClient::requestDatasetConversion` does not exist.

- [ ] **Step 2: Add WorkerClient API**

In `WorkerClient.h`, add:

```cpp
bool requestDatasetConversion(
    const QString& workerProgram,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QString& outputPath,
    const QJsonObject& options,
    QString* error,
    const QString& taskId = {});
```

In `WorkerClient.cpp`, add:

```cpp
bool WorkerClient::requestDatasetConversion(
    const QString& workerProgram,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QString& outputPath,
    const QJsonObject& options,
    QString* error,
    const QString& taskId)
{
    QJsonObject payload;
    payload.insert(QStringLiteral("taskId"), taskId);
    payload.insert(QStringLiteral("sourcePath"), sourcePath);
    payload.insert(QStringLiteral("sourceFormat"), sourceFormat);
    payload.insert(QStringLiteral("targetFormat"), targetFormat);
    payload.insert(QStringLiteral("outputPath"), outputPath);
    payload.insert(QStringLiteral("options"), options);
    return startWorkerCommand(workerProgram, QStringLiteral("convertDataset"), payload, error);
}
```

- [ ] **Step 3: Add Worker dispatch**

In `WorkerSession.h`, add:

```cpp
void convertDataset(const QJsonObject& payload);
```

In `WorkerSession.cpp`, add this branch before `validateDataset` or near other dataset commands:

```cpp
} else if (type == QStringLiteral("convertDataset")) {
    convertDataset(payload);
```

- [ ] **Step 4: Implement Worker command**

In `WorkerSessionDatasetCommands.cpp`, include:

```cpp
#include "aitrain/core/DatasetConversion.h"
```

Add:

```cpp
void WorkerSession::convertDataset(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始转换数据集。"));
    send(QStringLiteral("progress"), startProgress);

    aitrain::DatasetConversionRequest request;
    request.sourcePath = payload.value(QStringLiteral("sourcePath")).toString();
    request.sourceFormat = payload.value(QStringLiteral("sourceFormat")).toString();
    request.targetFormat = payload.value(QStringLiteral("targetFormat")).toString();
    request.outputPath = payload.value(QStringLiteral("outputPath")).toString();
    request.options = payload.value(QStringLiteral("options")).toObject();

    const aitrain::DatasetConversionResult result = aitrain::convertDataset(request);
    QJsonObject response = result.toJson();
    response.insert(QStringLiteral("taskId"), taskId);

    if (!result.reportPath.isEmpty()) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("dataset_conversion_report"));
        artifact.insert(QStringLiteral("path"), result.reportPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Dataset conversion report"));
        send(QStringLiteral("artifact"), artifact);
    }

    if (!result.validationReportPath.isEmpty()) {
        QJsonObject artifact;
        artifact.insert(QStringLiteral("taskId"), taskId);
        artifact.insert(QStringLiteral("kind"), QStringLiteral("dataset_validation_report"));
        artifact.insert(QStringLiteral("path"), result.validationReportPath);
        artifact.insert(QStringLiteral("message"), QStringLiteral("Converted dataset validation report"));
        send(QStringLiteral("artifact"), artifact);
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("数据集转换完成。"));
    send(QStringLiteral("progress"), doneProgress);

    send(QStringLiteral("datasetConversion"), response);
    if (!response.value(QStringLiteral("targetValidation")).toObject().isEmpty()) {
        QJsonObject validation = response.value(QStringLiteral("targetValidation")).toObject();
        validation.insert(QStringLiteral("taskId"), taskId);
        validation.insert(QStringLiteral("datasetPath"), result.outputPath);
        validation.insert(QStringLiteral("format"), result.targetFormat);
        validation.insert(QStringLiteral("reportPath"), result.validationReportPath);
        send(QStringLiteral("datasetValidation"), validation);
    }

    QJsonObject terminal;
    terminal.insert(QStringLiteral("taskId"), taskId);
    terminal.insert(QStringLiteral("message"), result.ok
        ? QStringLiteral("Dataset conversion completed")
        : QStringLiteral("Dataset conversion failed: %1").arg(result.errorMessage));
    send(result.ok ? QStringLiteral("completed") : QStringLiteral("failed"), terminal);
    finishSession();
}
```

- [ ] **Step 5: Ensure core writes validation report path**

In `DatasetConversion.cpp`, after validation, write:

```cpp
result.validationReportPath = QDir(result.outputPath).filePath(QStringLiteral("dataset_validation_report.json"));
QJsonObject validationJson = result.targetValidation.toJson();
validationJson.insert(QStringLiteral("datasetPath"), result.outputPath);
validationJson.insert(QStringLiteral("format"), result.targetFormat);
validationJson.insert(QStringLiteral("reportPath"), result.validationReportPath);
if (!writeJsonFile(result.validationReportPath, validationJson, &writeError)) {
    result.ok = false;
    result.errorCode = QStringLiteral("validation_report_write_failed");
    result.errorMessage = writeError;
}
```

Apply this in the shared finalization path used by COCO, VOC, and LabelMe.

- [ ] **Step 6: Run Worker test**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: PASS including `workerConvertsDatasetAndEmitsArtifacts`.

- [ ] **Step 7: Commit**

```powershell
git add src/worker/src/WorkerSession.h src/worker/src/WorkerSession.cpp src/worker/src/WorkerSessionDatasetCommands.cpp src/app/src/WorkerClient.h src/app/src/WorkerClient.cpp src/core/src/DatasetConversion.cpp tests/tst_dataset_conversion.cpp
git commit -m "feat: route dataset conversion through worker"
```

---

### Task 7: GUI Conversion Controls

**Files:**
- Modify: `src/app/src/MainWindow.h`
- Modify: `src/app/src/MainWindowDatasetPage.cpp`
- Modify: `src/app/src/MainWindowActions.cpp`
- Modify: `src/app/src/MainWindowWorkerMessages.cpp`
- Modify: `src/app/src/LanguageSupport.cpp`

- [ ] **Step 1: Add GUI members and slots**

In `MainWindow.h`, add private slots:

```cpp
void browseDatasetConversionSource();
void browseDatasetConversionOutput();
void convertDataset();
```

Add private handler declaration:

```cpp
void handleDatasetConversionMessage(const QJsonObject& payload);
```

Add widget members near existing dataset fields:

```cpp
QLineEdit* datasetConversionSourceEdit_ = nullptr;
QComboBox* datasetConversionSourceFormatCombo_ = nullptr;
QComboBox* datasetConversionTargetFormatCombo_ = nullptr;
QLineEdit* datasetConversionOutputEdit_ = nullptr;
QCheckBox* datasetConversionCopyImagesCheck_ = nullptr;
QCheckBox* datasetConversionAllowEmptyLabelsCheck_ = nullptr;
QCheckBox* datasetConversionPolygonToBoxCheck_ = nullptr;
QLabel* datasetConversionSummaryLabel_ = nullptr;
```

- [ ] **Step 2: Add compact UI section**

In `MainWindowDatasetPage.cpp`, inside `buildDatasetPage()` after the existing split action strip and before the result panel is assembled, create:

```cpp
auto* conversionPanel = new QFrame;
conversionPanel->setObjectName(QStringLiteral("ActionStrip"));
auto* conversionLayout = new QGridLayout(conversionPanel);
conversionLayout->setContentsMargins(10, 8, 10, 8);
conversionLayout->setHorizontalSpacing(10);
conversionLayout->setVerticalSpacing(8);

datasetConversionSourceEdit_ = new QLineEdit;
datasetConversionSourceEdit_->setPlaceholderText(QStringLiteral("COCO JSON 文件，或 VOC / LabelMe 标注目录"));
auto* conversionSourceBrowseButton = new QPushButton(QStringLiteral("选择源"));
connect(conversionSourceBrowseButton, &QPushButton::clicked, this, &MainWindow::browseDatasetConversionSource);

auto* conversionSourceRow = new QWidget;
auto* conversionSourceRowLayout = new QHBoxLayout(conversionSourceRow);
conversionSourceRowLayout->setContentsMargins(0, 0, 0, 0);
conversionSourceRowLayout->addWidget(datasetConversionSourceEdit_, 1);
conversionSourceRowLayout->addWidget(conversionSourceBrowseButton);

datasetConversionSourceFormatCombo_ = new QComboBox;
datasetConversionSourceFormatCombo_->addItem(QStringLiteral("COCO JSON"), QStringLiteral("coco_json"));
datasetConversionSourceFormatCombo_->addItem(QStringLiteral("VOC XML"), QStringLiteral("voc_xml"));
datasetConversionSourceFormatCombo_->addItem(QStringLiteral("LabelMe JSON"), QStringLiteral("labelme_json"));

datasetConversionTargetFormatCombo_ = new QComboBox;
datasetConversionTargetFormatCombo_->addItem(QStringLiteral("YOLO 检测"), QStringLiteral("yolo_detection"));
datasetConversionTargetFormatCombo_->addItem(QStringLiteral("YOLO 分割"), QStringLiteral("yolo_segmentation"));

datasetConversionOutputEdit_ = new QLineEdit;
datasetConversionOutputEdit_->setPlaceholderText(QStringLiteral("默认输出到当前项目 datasets/normalized"));
auto* conversionOutputBrowseButton = new QPushButton(QStringLiteral("选择输出"));
connect(conversionOutputBrowseButton, &QPushButton::clicked, this, &MainWindow::browseDatasetConversionOutput);

auto* conversionOutputRow = new QWidget;
auto* conversionOutputRowLayout = new QHBoxLayout(conversionOutputRow);
conversionOutputRowLayout->setContentsMargins(0, 0, 0, 0);
conversionOutputRowLayout->addWidget(datasetConversionOutputEdit_, 1);
conversionOutputRowLayout->addWidget(conversionOutputBrowseButton);

datasetConversionCopyImagesCheck_ = new QCheckBox(QStringLiteral("复制图片"));
datasetConversionCopyImagesCheck_->setChecked(true);
datasetConversionAllowEmptyLabelsCheck_ = new QCheckBox(QStringLiteral("允许空标签"));
datasetConversionAllowEmptyLabelsCheck_->setChecked(false);
datasetConversionPolygonToBoxCheck_ = new QCheckBox(QStringLiteral("polygon 转 bbox"));
datasetConversionPolygonToBoxCheck_->setChecked(true);

auto* conversionButton = primaryButton(QStringLiteral("转换为训练数据集"));
connect(conversionButton, &QPushButton::clicked, this, &MainWindow::convertDataset);
datasetConversionSummaryLabel_ = mutedLabel(QStringLiteral("将 COCO / VOC / LabelMe 转为可校验的 YOLO 数据集。"));
allowLabelToShrink(datasetConversionSummaryLabel_);

conversionLayout->addWidget(new QLabel(QStringLiteral("转换源")), 0, 0);
conversionLayout->addWidget(conversionSourceRow, 0, 1, 1, 3);
conversionLayout->addWidget(new QLabel(QStringLiteral("源格式")), 1, 0);
conversionLayout->addWidget(datasetConversionSourceFormatCombo_, 1, 1);
conversionLayout->addWidget(new QLabel(QStringLiteral("目标格式")), 1, 2);
conversionLayout->addWidget(datasetConversionTargetFormatCombo_, 1, 3);
conversionLayout->addWidget(new QLabel(QStringLiteral("输出目录")), 2, 0);
conversionLayout->addWidget(conversionOutputRow, 2, 1, 1, 3);
conversionLayout->addWidget(datasetConversionCopyImagesCheck_, 3, 0);
conversionLayout->addWidget(datasetConversionAllowEmptyLabelsCheck_, 3, 1);
conversionLayout->addWidget(datasetConversionPolygonToBoxCheck_, 3, 2);
conversionLayout->addWidget(conversionButton, 3, 3);
conversionLayout->addWidget(datasetConversionSummaryLabel_, 4, 0, 1, 4);
conversionLayout->setColumnStretch(1, 1);
conversionLayout->setColumnStretch(3, 1);

inputPanel->bodyLayout()->addWidget(conversionPanel);
```

- [ ] **Step 3: Implement browse/start actions**

In `MainWindowActions.cpp`, add:

```cpp
void MainWindow::browseDatasetConversionSource()
{
    const QString sourceFormat = datasetConversionSourceFormatCombo_
        ? datasetConversionSourceFormatCombo_->currentData().toString()
        : QString();
    QString selected;
    if (sourceFormat == QStringLiteral("coco_json")) {
        selected = QFileDialog::getOpenFileName(this, uiText("选择 COCO JSON"), QString(), QStringLiteral("JSON (*.json);;All files (*.*)"));
    } else {
        selected = QFileDialog::getExistingDirectory(this, uiText("选择标注目录"));
    }
    if (selected.isEmpty()) {
        return;
    }
    datasetConversionSourceEdit_->setText(QDir::toNativeSeparators(selected));
    if (datasetConversionOutputEdit_ && datasetConversionOutputEdit_->text().trimmed().isEmpty()) {
        const QString baseName = QFileInfo(selected).completeBaseName().isEmpty()
            ? QFileInfo(selected).fileName()
            : QFileInfo(selected).completeBaseName();
        const QString targetFormat = datasetConversionTargetFormatCombo_
            ? datasetConversionTargetFormatCombo_->currentData().toString()
            : QStringLiteral("yolo_detection");
        const QString outputPath = currentProjectPath_.isEmpty()
            ? QFileInfo(selected).absoluteDir().filePath(QStringLiteral("normalized/%1-%2").arg(baseName, targetFormat))
            : QDir(currentProjectPath_).filePath(QStringLiteral("datasets/normalized/%1-%2").arg(baseName, targetFormat));
        datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(QDir::cleanPath(outputPath)));
    }
}

void MainWindow::browseDatasetConversionOutput()
{
    const QString selected = QFileDialog::getExistingDirectory(this, uiText("选择转换输出目录"));
    if (!selected.isEmpty() && datasetConversionOutputEdit_) {
        datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(selected));
    }
}

void MainWindow::convertDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据集转换"), uiText("Worker 正在执行任务，稍后再转换数据集。"));
        return;
    }
    const QString sourcePath = QDir::fromNativeSeparators(datasetConversionSourceEdit_ ? datasetConversionSourceEdit_->text().trimmed() : QString());
    const QString sourceFormat = datasetConversionSourceFormatCombo_ ? datasetConversionSourceFormatCombo_->currentData().toString() : QString();
    const QString targetFormat = datasetConversionTargetFormatCombo_ ? datasetConversionTargetFormatCombo_->currentData().toString() : QString();
    QString outputPath = QDir::fromNativeSeparators(datasetConversionOutputEdit_ ? datasetConversionOutputEdit_->text().trimmed() : QString());
    if (sourcePath.isEmpty() || sourceFormat.isEmpty() || targetFormat.isEmpty()) {
        QMessageBox::warning(this, uiText("数据集转换"), uiText("请选择转换源、源格式和目标格式。"));
        return;
    }
    if (outputPath.isEmpty()) {
        const QString baseName = QFileInfo(sourcePath).completeBaseName().isEmpty()
            ? QFileInfo(sourcePath).fileName()
            : QFileInfo(sourcePath).completeBaseName();
        outputPath = currentProjectPath_.isEmpty()
            ? QFileInfo(sourcePath).absoluteDir().filePath(QStringLiteral("normalized/%1-%2").arg(baseName, targetFormat))
            : QDir(currentProjectPath_).filePath(QStringLiteral("datasets/normalized/%1-%2").arg(baseName, targetFormat));
        datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(QDir::cleanPath(outputPath)));
    }

    QJsonObject options;
    options.insert(QStringLiteral("copyImages"), datasetConversionCopyImagesCheck_ ? datasetConversionCopyImagesCheck_->isChecked() : true);
    options.insert(QStringLiteral("allowEmptyLabels"), datasetConversionAllowEmptyLabelsCheck_ ? datasetConversionAllowEmptyLabelsCheck_->isChecked() : false);
    options.insert(QStringLiteral("polygonToBox"), datasetConversionPolygonToBoxCheck_ ? datasetConversionPolygonToBoxCheck_->isChecked() : true);
    options.insert(QStringLiteral("maxIssues"), 500);

    QString taskId;
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Validate,
            QStringLiteral("dataset_conversion"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集转换中。"));
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestDatasetConversion(workerExecutablePath(), sourcePath, sourceFormat, targetFormat, outputPath, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据集转换"), error);
        return;
    }
    if (datasetConversionSummaryLabel_) {
        datasetConversionSummaryLabel_->setText(uiText("正在转换数据集：%1").arg(QDir::toNativeSeparators(outputPath)));
    }
    workerPill_->setStatus(uiText("数据集转换中"), StatusPill::Tone::Info);
}
```

- [ ] **Step 4: Route conversion Worker messages**

In `MainWindowWorkerMessages.cpp`, inside `handleWorkerMessage(...)`, add:

```cpp
    } else if (type == QStringLiteral("datasetConversion")) {
        handleDatasetConversionMessage(payload);
```

Add handler implementation in the same file:

```cpp
void MainWindow::handleDatasetConversionMessage(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString targetFormat = payload.value(QStringLiteral("targetFormat")).toString();
    const QString reportPath = payload.value(QStringLiteral("reportPath")).toString();
    const int convertedSamples = payload.value(QStringLiteral("convertedSampleCount")).toInt();
    const int convertedAnnotations = payload.value(QStringLiteral("convertedAnnotationCount")).toInt();
    const QString message = ok
        ? uiText("转换完成：%1 个样本，%2 个标注。报告：%3").arg(convertedSamples).arg(convertedAnnotations).arg(QDir::toNativeSeparators(reportPath))
        : uiText("转换失败：%1").arg(payload.value(QStringLiteral("errorMessage")).toString());
    if (datasetConversionSummaryLabel_) {
        datasetConversionSummaryLabel_->setText(message);
    }
    if (ok) {
        if (datasetPathEdit_) {
            datasetPathEdit_->setText(QDir::toNativeSeparators(outputPath));
        }
        if (datasetFormatCombo_) {
            const int index = datasetFormatCombo_->findData(targetFormat);
            if (index >= 0) {
                datasetFormatCombo_->setCurrentIndex(index);
            }
        }
        currentDatasetPath_ = outputPath;
        currentDatasetFormat_ = targetFormat;
        currentDatasetValid_ = payload.value(QStringLiteral("targetValidation")).toObject().value(QStringLiteral("ok")).toBool();
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
        updateDatasetList();
    }
}
```

- [ ] **Step 5: Add translations**

In `LanguageSupport.cpp`, add mappings for new UI strings:

```cpp
{QStringLiteral("数据集转换"), QStringLiteral("Dataset Conversion")},
{QStringLiteral("Worker 正在执行任务，稍后再转换数据集。"), QStringLiteral("Worker is running a task. Convert the dataset later.")},
{QStringLiteral("请选择转换源、源格式和目标格式。"), QStringLiteral("Select conversion source, source format, and target format.")},
{QStringLiteral("数据集转换中。"), QStringLiteral("Dataset conversion in progress.")},
{QStringLiteral("数据集转换中"), QStringLiteral("Converting dataset")},
{QStringLiteral("转换为训练数据集"), QStringLiteral("Convert to Training Dataset")},
{QStringLiteral("将 COCO / VOC / LabelMe 转为可校验的 YOLO 数据集。"), QStringLiteral("Convert COCO / VOC / LabelMe into a YOLO dataset that can be validated.")},
{QStringLiteral("COCO JSON 文件，或 VOC / LabelMe 标注目录"), QStringLiteral("COCO JSON file, or VOC / LabelMe annotation directory")},
{QStringLiteral("转换完成：%1 个样本，%2 个标注。报告：%3"), QStringLiteral("Conversion complete: %1 sample(s), %2 annotation(s). Report: %3")},
{QStringLiteral("转换失败：%1"), QStringLiteral("Conversion failed: %1")}
```

- [ ] **Step 6: Build app target**

Run:

```powershell
cmake --build build-vscode --target AITrainStudio
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/app/src/MainWindow.h src/app/src/MainWindowDatasetPage.cpp src/app/src/MainWindowActions.cpp src/app/src/MainWindowWorkerMessages.cpp src/app/src/LanguageSupport.cpp
git commit -m "feat: add dataset conversion UI"
```

---

### Task 8: Documentation and Status

**Files:**
- Modify: `docs/user-guide.md`
- Modify: `docs/training-backends.md`
- Modify: `docs/harness/current-status.md`
- Modify: `docs/product-roadmap-local-training-platform.md`

- [ ] **Step 1: Update user guide**

In `docs/user-guide.md`, add a subsection under dataset import/validation:

```markdown
### 外部标注格式转换

“数据集”页可以把常见外部标注格式转换为 AITrain 可训练的 YOLO 数据集：

| 源格式 | 目标格式 | 说明 |
|---|---|---|
| COCO JSON | YOLO 检测 | 使用 `bbox` 转换为 YOLO bbox 标签。 |
| COCO JSON | YOLO 分割 | 使用 polygon `segmentation` 转换为 YOLO polygon 标签；RLE mask 会被记录为跳过项。 |
| VOC XML | YOLO 检测 | 使用 `object/name/bndbox` 转换为 YOLO bbox 标签。 |
| LabelMe JSON | YOLO 检测 | `rectangle` 转 bbox；启用 polygon 转 bbox 时会使用外接框。 |
| LabelMe JSON | YOLO 分割 | `polygon` 转 YOLO polygon 标签。 |

转换不会修改原始数据。输出默认写入当前项目的 `datasets/normalized`，并生成 `dataset_conversion_report.json` 和 `dataset_validation_report.json`。转换后仍以目标格式校验结果为准，只有校验通过的数据集才进入训练主流程。
```

- [ ] **Step 2: Update backend docs**

In `docs/training-backends.md`, add after the dataset layout section:

```markdown
External COCO JSON, Pascal VOC XML, and LabelMe JSON are import formats, not training layouts. AITrain converts them into YOLO detection or YOLO segmentation directories through the Worker-managed `convertDataset` command, then runs the existing YOLO validator. OCR conversion is not inferred from these formats.
```

- [ ] **Step 3: Update current status**

In `docs/harness/current-status.md`, add a new phase row after Phase 49 or the current latest local phase:

```markdown
| Phase 50 Lite: Dataset interop conversion | Done locally | Added Worker-managed conversion from COCO JSON, Pascal VOC XML, and LabelMe JSON into normalized YOLO detection/segmentation datasets. Conversion writes `dataset_conversion_report.json`, runs target YOLO validation, records conversion and validation artifacts, and exposes a compact conversion workflow in the existing dataset page. This does not add OCR conversion, Datumaro/Python conversion backends, new training algorithms, SQLite schema changes, or plugin interface changes. |
```

If phase numbering has advanced by the time this plan is executed, use the next available phase number and keep the wording.

- [ ] **Step 4: Update product roadmap**

In `docs/product-roadmap-local-training-platform.md`, add a short section after Phase 49 Lite:

```markdown
## Dataset interop conversion

AITrain now supports a focused external-format import loop: COCO JSON, Pascal VOC XML, and LabelMe JSON can be converted through the Worker into normalized YOLO detection or YOLO segmentation datasets. The conversion path writes a machine-readable report, preserves source data, runs existing target validation, and records artifacts in the task history.

This remains a local data-loop enhancement. It does not introduce Datumaro, OCR conversion, cloud dataset management, or new model training backends.
```

- [ ] **Step 5: Run document checks**

Run:

```powershell
git diff --check
.\tools\harness-context.ps1
```

Expected: both pass.

- [ ] **Step 6: Commit**

```powershell
git add docs/user-guide.md docs/training-backends.md docs/harness/current-status.md docs/product-roadmap-local-training-platform.md
git commit -m "docs: document dataset interop conversion"
```

---

### Task 9: Full Verification and UI Walkthrough

**Files:**
- No source edits expected unless verification exposes a concrete issue.

- [ ] **Step 1: Run focused conversion test**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_tests
.\build-vscode\tests\aitrain_dataset_conversion_tests.exe
```

Expected: all tests pass.

- [ ] **Step 2: Run full harness**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: configure/build/CTest pass.

- [ ] **Step 3: Run UI walkthrough**

Run:

```powershell
$pages = @('总览','项目','数据集','样本复核','训练实验','任务与产物','模型库','评估报告','模型导出','推理验证','交付验收','插件','环境','设置')
C:\Users\73200\.codex\skills\qt-gui-walkthrough\scripts\qt_walkthrough.ps1 `
  -AppPath .\build-vscode\bin\AITrainStudio.exe `
  -WorkingDirectory .\build-vscode\bin `
  -OutDir .\.deps\ui-walkthrough-dataset-conversion `
  -PageNames $pages `
  -Width 1280 `
  -Height 820
```

Expected: dataset page controls fit the viewport, no horizontal overflow findings, and existing dataset actions remain visible or clearly reachable by vertical scrolling.

- [ ] **Step 4: Final status check**

Run:

```powershell
git status --short
git log --oneline -5
```

Expected: clean working tree after commits, with conversion feature commits visible.

---

## Self-Review Notes

Spec coverage:

- External formats: covered by Tasks 2-5.
- Worker command and artifacts: covered by Task 6.
- GUI controls and message handling: covered by Task 7.
- No SQLite schema change: preserved by using existing task/artifact records.
- No new training backend: preserved; only dataset conversion and target validation are added.
- Documentation and status: covered by Task 8.
- Verification: covered by Task 9.

Risk notes:

- If duplicate output image filenames appear across COCO or LabelMe subdirectories, implementation must make destination names unique by prefixing a stable hash or source index. Add a test for that before release if real datasets show collisions.
- If `copyImages=false` fails existing YOLO validation because labels and images are not colocated in the expected layout, keep the GUI default enabled and treat disabled copy mode as advanced.
- If the dataset page becomes crowded at 1280x820, move conversion into a compact collapsible `InfoPanel` inside the same page rather than creating a new navigation page.
