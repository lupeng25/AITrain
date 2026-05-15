# 数据集转换 GUI 入口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在数据集页“数据集操作”区内联新增数据集格式转换入口，支持已实现转换矩阵、轻量预检、Worker 调用、表单内进度/日志/取消和结果展示。

**Architecture:** 把转换矩阵、格式显示名和表单预检拆到轻量 `DatasetConversionUiModel`，用 QtTest 覆盖，不依赖真实 `MainWindow`。`MainWindowDatasetPage.cpp` 只创建控件和连接信号，`MainWindowActions.cpp` 负责启动/取消 Worker，`MainWindowWorkerMessages.cpp` 负责进度和结果回写。

**Tech Stack:** C++20, Qt 5.12 Widgets/Core/Test, CMake, existing `WorkerClient`, existing `WorkerSession::convertDataset`, existing `aitrain::convertDataset`.

---

## 完成记录

已于 2026-05-15 本地完成。

- 最终实现提交：`3579239 feat: show dataset conversion worker results`。
- 验证：转换相关 focused tests、`.\tools\harness-check.ps1`、`git diff --check`、1280x820 Qt walkthrough（`.deps\ui-walkthrough-dataset-conversion\walkthrough-summary.json`）均通过。
- 范围边界保持不变：未修改 core 转换语义、未修改 SQLite schema、未修改插件接口、未自动登记转换产物为数据集。

---

## Scope Check

本计划只实现设计规格中的 GUI 入口，不扩展 core 转换矩阵，不新增 SQLite schema，不新增插件接口，不自动登记转换产物为数据集。任务是单一 UI/Worker 编排功能，适合一个实施计划。

## File Structure

- Create: `src/app/src/DatasetConversionUiModel.h`
  - 纯 QtCore helper：格式矩阵、格式显示名、路径规范化、预检结果结构。
- Create: `src/app/src/DatasetConversionUiModel.cpp`
  - 实现 helper，不依赖 `MainWindow`，便于 QtTest 覆盖。
- Create: `tests/tst_dataset_conversion_ui.cpp`
  - 覆盖源/目标联动矩阵、路径预检、输入输出相同拦截、Worker running 拦截。
- Modify: `src/app/CMakeLists.txt`
  - 把 helper 加入 `AITrainStudio`。
- Modify: `tests/CMakeLists.txt`
  - 把 helper 加入测试目标，并注册 `aitrain_dataset_conversion_ui_tests`。
- Modify: `src/app/src/MainWindow.h`
  - 新增转换控件成员、槽函数和小型 helper 声明。
- Modify: `src/app/src/MainWindowDatasetPage.cpp`
  - 在数据集操作面板中新增转换表单，连接信号，选择数据集时同步转换输入。
- Modify: `src/app/src/MainWindowActions.cpp`
  - 新增浏览输入/输出目录、启动转换、取消转换、预检显示、表单 running 状态。
- Modify: `src/app/src/MainWindowWorkerMessages.cpp`
  - 识别 `datasetConversion` 消息，更新转换结果；进度消息同步转换表单进度。
- Modify: `src/app/src/MainWindow.cpp`
  - Worker finished 时兜底恢复转换表单运行态。
- Modify: `src/app/src/LanguageSupport.cpp`
  - 为新增中文 UI 文案补充英文 fallback 翻译，保持语言切换体验。

---

### Task 1: Add Dataset Conversion UI Model Tests

**Files:**
- Create: `tests/tst_dataset_conversion_ui.cpp`
- Modify: `tests/CMakeLists.txt`

- [x] **Step 1: Write the failing test file**

Create `tests/tst_dataset_conversion_ui.cpp`:

```cpp
#include "DatasetConversionUiModel.h"

#include <QtTest>
#include <QDir>
#include <QTemporaryDir>

class DatasetConversionUiTests : public QObject {
    Q_OBJECT

private slots:
    void exposesOnlyImplementedSourceFormats()
    {
        const QStringList sources = aitrain_app::supportedDatasetConversionSourceFormats();
        QCOMPARE(sources, QStringList()
            << QStringLiteral("coco_json")
            << QStringLiteral("voc_xml")
            << QStringLiteral("yolo_detection")
            << QStringLiteral("yolo_segmentation"));
    }

    void filtersTargetsBySourceFormat()
    {
        QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("coco_json")),
            QStringList() << QStringLiteral("yolo_detection") << QStringLiteral("yolo_segmentation"));
        QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("voc_xml")),
            QStringList() << QStringLiteral("yolo_detection"));
        QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("yolo_detection")),
            QStringList() << QStringLiteral("coco_json") << QStringLiteral("voc_xml"));
        QCOMPARE(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("yolo_segmentation")),
            QStringList() << QStringLiteral("coco_json"));
        QVERIFY(aitrain_app::supportedDatasetConversionTargets(QStringLiteral("paddleocr_rec")).isEmpty());
    }

    void validatesWritableDifferentOutputDirectory()
    {
        QTemporaryDir temp;
        QVERIFY(temp.isValid());
        QDir root(temp.path());
        QVERIFY(root.mkpath(QStringLiteral("input")));
        QVERIFY(root.mkpath(QStringLiteral("outputs")));

        aitrain_app::DatasetConversionForm form;
        form.sourceFormat = QStringLiteral("coco_json");
        form.targetFormat = QStringLiteral("yolo_detection");
        form.inputPath = root.filePath(QStringLiteral("input"));
        form.outputPath = root.filePath(QStringLiteral("outputs/converted"));
        form.workerRunning = false;

        const aitrain_app::DatasetConversionValidation validation =
            aitrain_app::validateDatasetConversionForm(form);

        QVERIFY2(validation.ok, qPrintable(validation.summary));
        QVERIFY(validation.messages.isEmpty());
    }

    void rejectsSameInputAndOutputDirectory()
    {
        QTemporaryDir temp;
        QVERIFY(temp.isValid());

        aitrain_app::DatasetConversionForm form;
        form.sourceFormat = QStringLiteral("coco_json");
        form.targetFormat = QStringLiteral("yolo_detection");
        form.inputPath = temp.path();
        form.outputPath = QDir::toNativeSeparators(temp.path());
        form.workerRunning = false;

        const aitrain_app::DatasetConversionValidation validation =
            aitrain_app::validateDatasetConversionForm(form);

        QVERIFY(!validation.ok);
        QCOMPARE(validation.outputPathError, QStringLiteral("输出目录不能与输入目录相同。"));
    }

    void rejectsUnsupportedPairAndMissingInput()
    {
        QTemporaryDir temp;
        QVERIFY(temp.isValid());
        QDir root(temp.path());
        QVERIFY(root.mkpath(QStringLiteral("outputs")));

        aitrain_app::DatasetConversionForm form;
        form.sourceFormat = QStringLiteral("voc_xml");
        form.targetFormat = QStringLiteral("yolo_segmentation");
        form.inputPath = root.filePath(QStringLiteral("missing-input"));
        form.outputPath = root.filePath(QStringLiteral("outputs/converted"));
        form.workerRunning = false;

        const aitrain_app::DatasetConversionValidation validation =
            aitrain_app::validateDatasetConversionForm(form);

        QVERIFY(!validation.ok);
        QCOMPARE(validation.targetFormatError, QStringLiteral("当前源格式不支持转换到该目标格式。"));
        QCOMPARE(validation.inputPathError, QStringLiteral("输入目录不存在。"));
    }

    void rejectsWhenWorkerIsRunning()
    {
        QTemporaryDir temp;
        QVERIFY(temp.isValid());
        QDir root(temp.path());
        QVERIFY(root.mkpath(QStringLiteral("input")));
        QVERIFY(root.mkpath(QStringLiteral("outputs")));

        aitrain_app::DatasetConversionForm form;
        form.sourceFormat = QStringLiteral("coco_json");
        form.targetFormat = QStringLiteral("yolo_detection");
        form.inputPath = root.filePath(QStringLiteral("input"));
        form.outputPath = root.filePath(QStringLiteral("outputs/converted"));
        form.workerRunning = true;

        const aitrain_app::DatasetConversionValidation validation =
            aitrain_app::validateDatasetConversionForm(form);

        QVERIFY(!validation.ok);
        QCOMPARE(validation.summary, QStringLiteral("Worker 正在执行任务，稍后再转换数据集。"));
    }
};

QTEST_MAIN(DatasetConversionUiTests)
#include "tst_dataset_conversion_ui.moc"
```

- [x] **Step 2: Register the failing test target**

Modify `tests/CMakeLists.txt` so every test target can compile the new helper and the new test is registered:

```cmake
function(add_aitrain_qt_test target source)
    add_executable(${target}
        ../src/app/src/WorkerClient.h
        ../src/app/src/WorkerClient.cpp
        ../src/app/src/DatasetConversionUiModel.h
        ../src/app/src/DatasetConversionUiModel.cpp
        TestSupport.h
        TestSupport.cpp
        ${source}
    )
```

Add this line after `aitrain_dataset_conversion_tests`:

```cmake
add_aitrain_qt_test(aitrain_dataset_conversion_ui_tests tst_dataset_conversion_ui.cpp)
```

- [x] **Step 3: Run the new test to verify it fails**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_ui_tests
```

Expected: FAIL because `DatasetConversionUiModel.h` and `DatasetConversionUiModel.cpp` do not exist yet.

- [x] **Step 4: Keep the failing test uncommitted**

Do not commit this state. The next task adds the implementation and commits the passing helper plus test together, so no commit leaves the branch unable to build.

---

### Task 2: Implement Dataset Conversion UI Model

**Files:**
- Create: `src/app/src/DatasetConversionUiModel.h`
- Create: `src/app/src/DatasetConversionUiModel.cpp`
- Modify: `src/app/CMakeLists.txt`
- Test: `tests/tst_dataset_conversion_ui.cpp`

- [x] **Step 1: Add helper header**

Create `src/app/src/DatasetConversionUiModel.h`:

```cpp
#pragma once

#include <QString>
#include <QStringList>

namespace aitrain_app {

struct DatasetConversionForm {
    QString sourceFormat;
    QString targetFormat;
    QString inputPath;
    QString outputPath;
    bool workerRunning = false;
};

struct DatasetConversionValidation {
    bool ok = false;
    QString summary;
    QString sourceFormatError;
    QString targetFormatError;
    QString inputPathError;
    QString outputPathError;
    QStringList messages;
};

QString datasetConversionFormatLabel(const QString& format);
QStringList supportedDatasetConversionSourceFormats();
QStringList supportedDatasetConversionTargets(const QString& sourceFormat);
bool isSupportedDatasetConversionPair(const QString& sourceFormat, const QString& targetFormat);
QString normalizedDatasetConversionPath(const QString& path);
DatasetConversionValidation validateDatasetConversionForm(const DatasetConversionForm& form);

} // namespace aitrain_app
```

- [x] **Step 2: Add helper implementation**

Create `src/app/src/DatasetConversionUiModel.cpp`:

```cpp
#include "DatasetConversionUiModel.h"

#include <QDir>
#include <QFileInfo>
#include <QMap>

namespace aitrain_app {
namespace {

const QMap<QString, QStringList>& conversionMatrix()
{
    static const QMap<QString, QStringList> matrix = {
        {QStringLiteral("coco_json"), QStringList()
            << QStringLiteral("yolo_detection")
            << QStringLiteral("yolo_segmentation")},
        {QStringLiteral("voc_xml"), QStringList()
            << QStringLiteral("yolo_detection")},
        {QStringLiteral("yolo_detection"), QStringList()
            << QStringLiteral("coco_json")
            << QStringLiteral("voc_xml")},
        {QStringLiteral("yolo_segmentation"), QStringList()
            << QStringLiteral("coco_json")}
    };
    return matrix;
}

void appendFieldError(QString* field, DatasetConversionValidation* validation, const QString& message)
{
    if (!field || !validation || !field->isEmpty()) {
        return;
    }
    *field = message;
    validation->messages.append(message);
}

QString absoluteCleanPath(const QString& path)
{
    const QString normalized = QDir::fromNativeSeparators(path.trimmed());
    if (normalized.isEmpty()) {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(normalized).absoluteFilePath());
}

} // namespace

QString datasetConversionFormatLabel(const QString& format)
{
    if (format == QStringLiteral("coco_json")) {
        return QStringLiteral("COCO JSON");
    }
    if (format == QStringLiteral("voc_xml")) {
        return QStringLiteral("Pascal VOC XML");
    }
    if (format == QStringLiteral("yolo_detection")) {
        return QStringLiteral("YOLO Detection");
    }
    if (format == QStringLiteral("yolo_segmentation")) {
        return QStringLiteral("YOLO Segmentation");
    }
    return format;
}

QStringList supportedDatasetConversionSourceFormats()
{
    return conversionMatrix().keys();
}

QStringList supportedDatasetConversionTargets(const QString& sourceFormat)
{
    return conversionMatrix().value(sourceFormat);
}

bool isSupportedDatasetConversionPair(const QString& sourceFormat, const QString& targetFormat)
{
    return supportedDatasetConversionTargets(sourceFormat).contains(targetFormat);
}

QString normalizedDatasetConversionPath(const QString& path)
{
    return absoluteCleanPath(path);
}

DatasetConversionValidation validateDatasetConversionForm(const DatasetConversionForm& form)
{
    DatasetConversionValidation validation;

    if (form.workerRunning) {
        validation.summary = QStringLiteral("Worker 正在执行任务，稍后再转换数据集。");
        validation.messages.append(validation.summary);
        return validation;
    }

    const QString sourceFormat = form.sourceFormat.trimmed();
    const QString targetFormat = form.targetFormat.trimmed();
    const QString inputPath = QDir::fromNativeSeparators(form.inputPath.trimmed());
    const QString outputPath = QDir::fromNativeSeparators(form.outputPath.trimmed());

    if (sourceFormat.isEmpty()) {
        appendFieldError(&validation.sourceFormatError, &validation, QStringLiteral("请选择源格式。"));
    } else if (!conversionMatrix().contains(sourceFormat)) {
        appendFieldError(&validation.sourceFormatError, &validation, QStringLiteral("当前不支持该源格式。"));
    }

    if (targetFormat.isEmpty()) {
        appendFieldError(&validation.targetFormatError, &validation, QStringLiteral("请选择目标格式。"));
    } else if (!sourceFormat.isEmpty() && !isSupportedDatasetConversionPair(sourceFormat, targetFormat)) {
        appendFieldError(&validation.targetFormatError, &validation, QStringLiteral("当前源格式不支持转换到该目标格式。"));
    }

    if (inputPath.isEmpty()) {
        appendFieldError(&validation.inputPathError, &validation, QStringLiteral("请选择输入目录。"));
    } else {
        const QFileInfo inputInfo(inputPath);
        if (!inputInfo.exists()) {
            appendFieldError(&validation.inputPathError, &validation, QStringLiteral("输入目录不存在。"));
        } else if (!inputInfo.isDir()) {
            appendFieldError(&validation.inputPathError, &validation, QStringLiteral("输入路径必须是目录。"));
        }
    }

    if (outputPath.isEmpty()) {
        appendFieldError(&validation.outputPathError, &validation, QStringLiteral("请选择输出目录。"));
    } else {
        const QFileInfo outputInfo(outputPath);
        if (outputInfo.exists() && !outputInfo.isDir()) {
            appendFieldError(&validation.outputPathError, &validation, QStringLiteral("输出路径必须是目录。"));
        }

        const QString normalizedInput = absoluteCleanPath(inputPath);
        const QString normalizedOutput = absoluteCleanPath(outputPath);
        if (!normalizedInput.isEmpty()
            && !normalizedOutput.isEmpty()
            && QString::compare(normalizedInput, normalizedOutput, Qt::CaseInsensitive) == 0) {
            appendFieldError(&validation.outputPathError, &validation, QStringLiteral("输出目录不能与输入目录相同。"));
        }

        const QDir parentDir = outputInfo.absoluteDir();
        if (!parentDir.exists()) {
            appendFieldError(&validation.outputPathError, &validation, QStringLiteral("输出目录的父目录不存在。"));
        } else if (!QFileInfo(parentDir.absolutePath()).isWritable()) {
            appendFieldError(&validation.outputPathError, &validation, QStringLiteral("输出目录的父目录不可写。"));
        }
    }

    validation.ok = validation.messages.isEmpty();
    validation.summary = validation.ok
        ? QStringLiteral("可以开始转换。")
        : QStringLiteral("请修正 %1 个字段后再转换。").arg(validation.messages.size());
    return validation;
}

} // namespace aitrain_app
```

- [x] **Step 3: Add helper to app target**

Modify `src/app/CMakeLists.txt` and insert these files near `MainWindowSupport`:

```cmake
    src/DatasetConversionUiModel.h
    src/DatasetConversionUiModel.cpp
```

- [x] **Step 4: Run focused UI model test**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_ui_tests
ctest --test-dir build-vscode -R aitrain_dataset_conversion_ui_tests --output-on-failure
```

Expected: PASS.

- [x] **Step 5: Commit helper implementation**

```powershell
git add src/app/CMakeLists.txt src/app/src/DatasetConversionUiModel.h src/app/src/DatasetConversionUiModel.cpp tests/CMakeLists.txt tests/tst_dataset_conversion_ui.cpp
git commit -m "feat: add dataset conversion gui model"
```

---

### Task 3: Add MainWindow Declarations and Dataset Page Form

**Files:**
- Modify: `src/app/src/MainWindow.h`
- Modify: `src/app/src/MainWindowDatasetPage.cpp`

- [x] **Step 1: Add declarations to `MainWindow.h`**

Add `class QPushButton;` near the existing forward declarations:

```cpp
class QPushButton;
```

Add these private slots near the dataset slots:

```cpp
    void browseDatasetConversionInput();
    void browseDatasetConversionOutput();
    void updateDatasetConversionTargetFormats();
    void startDatasetConversion();
    void cancelDatasetConversion();
```

Add these private helper declarations near `updateDatasetSplitResult`:

```cpp
    void updateDatasetConversionResult(const QJsonObject& payload);
    void setDatasetConversionFormRunning(bool running);
    void clearDatasetConversionErrors();
    void appendDatasetConversionLog(const QString& text);
    void refreshDatasetConversionDefaultsFromCurrentDataset();
```

Add this state field near `currentDatasetFormat_`:

```cpp
    QString currentDatasetConversionTaskId_;
```

Add these widget members near the existing dataset widgets:

```cpp
    QComboBox* datasetConversionSourceFormatCombo_ = nullptr;
    QComboBox* datasetConversionTargetFormatCombo_ = nullptr;
    QLineEdit* datasetConversionInputEdit_ = nullptr;
    QLineEdit* datasetConversionOutputEdit_ = nullptr;
    QLabel* datasetConversionStatusLabel_ = nullptr;
    QLabel* datasetConversionSourceErrorLabel_ = nullptr;
    QLabel* datasetConversionTargetErrorLabel_ = nullptr;
    QLabel* datasetConversionInputErrorLabel_ = nullptr;
    QLabel* datasetConversionOutputErrorLabel_ = nullptr;
    QLabel* datasetConversionResultLabel_ = nullptr;
    QPushButton* datasetConversionStartButton_ = nullptr;
    QPushButton* datasetConversionCancelButton_ = nullptr;
    QProgressBar* datasetConversionProgressBar_ = nullptr;
    QPlainTextEdit* datasetConversionLog_ = nullptr;
```

- [x] **Step 2: Include the new helper in `MainWindowDatasetPage.cpp`**

Add this include:

```cpp
#include "DatasetConversionUiModel.h"
```

- [x] **Step 3: Build the conversion form inside `buildDatasetPage()`**

Insert this block after the split ratio row is added and before `inputPanel->bodyLayout()->addWidget(datasetActionStrip);`:

```cpp
    auto* conversionPanel = new QFrame;
    conversionPanel->setObjectName(QStringLiteral("ActionStrip"));
    auto* conversionLayout = new QVBoxLayout(conversionPanel);
    conversionLayout->setContentsMargins(10, 8, 10, 8);
    conversionLayout->setSpacing(8);

    auto* conversionTitle = new QLabel(QStringLiteral("格式转换"));
    conversionTitle->setObjectName(QStringLiteral("SectionTitle"));
    datasetConversionStatusLabel_ = inlineStatusLabel(QStringLiteral("选择源格式、目标格式和输出目录后开始转换。"));
    allowLabelToShrink(datasetConversionStatusLabel_);
    conversionLayout->addWidget(conversionTitle);
    conversionLayout->addWidget(datasetConversionStatusLabel_);

    auto* conversionForm = new QFormLayout;
    datasetConversionSourceFormatCombo_ = new QComboBox;
    for (const QString& format : supportedDatasetConversionSourceFormats()) {
        addComboItem(datasetConversionSourceFormatCombo_, datasetConversionFormatLabel(format), format);
    }
    connect(datasetConversionSourceFormatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &MainWindow::updateDatasetConversionTargetFormats);

    datasetConversionTargetFormatCombo_ = new QComboBox;
    datasetConversionInputEdit_ = new QLineEdit;
    datasetConversionInputEdit_->setPlaceholderText(QStringLiteral("选择待转换的数据集目录"));
    auto* browseConversionInputButton = new QPushButton(QStringLiteral("选择输入"));
    connect(browseConversionInputButton, &QPushButton::clicked, this, &MainWindow::browseDatasetConversionInput);

    auto* conversionInputRow = new QWidget;
    auto* conversionInputLayout = new QHBoxLayout(conversionInputRow);
    conversionInputLayout->setContentsMargins(0, 0, 0, 0);
    conversionInputLayout->addWidget(datasetConversionInputEdit_);
    conversionInputLayout->addWidget(browseConversionInputButton);

    datasetConversionOutputEdit_ = new QLineEdit;
    datasetConversionOutputEdit_->setPlaceholderText(QStringLiteral("选择不同于输入目录的输出目录"));
    auto* browseConversionOutputButton = new QPushButton(QStringLiteral("选择输出"));
    connect(browseConversionOutputButton, &QPushButton::clicked, this, &MainWindow::browseDatasetConversionOutput);

    auto* conversionOutputRow = new QWidget;
    auto* conversionOutputLayout = new QHBoxLayout(conversionOutputRow);
    conversionOutputLayout->setContentsMargins(0, 0, 0, 0);
    conversionOutputLayout->addWidget(datasetConversionOutputEdit_);
    conversionOutputLayout->addWidget(browseConversionOutputButton);

    conversionForm->addRow(QStringLiteral("源格式"), datasetConversionSourceFormatCombo_);
    conversionForm->addRow(QStringLiteral("目标格式"), datasetConversionTargetFormatCombo_);
    conversionForm->addRow(QStringLiteral("输入目录"), conversionInputRow);
    conversionForm->addRow(QStringLiteral("输出目录"), conversionOutputRow);
    conversionLayout->addLayout(conversionForm);

    datasetConversionSourceErrorLabel_ = mutedLabel(QString());
    datasetConversionTargetErrorLabel_ = mutedLabel(QString());
    datasetConversionInputErrorLabel_ = mutedLabel(QString());
    datasetConversionOutputErrorLabel_ = mutedLabel(QString());
    datasetConversionSourceErrorLabel_->hide();
    datasetConversionTargetErrorLabel_->hide();
    datasetConversionInputErrorLabel_->hide();
    datasetConversionOutputErrorLabel_->hide();
    conversionLayout->addWidget(datasetConversionSourceErrorLabel_);
    conversionLayout->addWidget(datasetConversionTargetErrorLabel_);
    conversionLayout->addWidget(datasetConversionInputErrorLabel_);
    conversionLayout->addWidget(datasetConversionOutputErrorLabel_);

    auto* conversionButtonRow = new QWidget;
    auto* conversionButtonLayout = new QHBoxLayout(conversionButtonRow);
    conversionButtonLayout->setContentsMargins(0, 0, 0, 0);
    datasetConversionStartButton_ = primaryButton(QStringLiteral("转换数据集"));
    datasetConversionCancelButton_ = dangerButton(QStringLiteral("取消转换"));
    datasetConversionCancelButton_->setEnabled(false);
    connect(datasetConversionStartButton_, &QPushButton::clicked, this, &MainWindow::startDatasetConversion);
    connect(datasetConversionCancelButton_, &QPushButton::clicked, this, &MainWindow::cancelDatasetConversion);
    conversionButtonLayout->addWidget(datasetConversionStartButton_);
    conversionButtonLayout->addWidget(datasetConversionCancelButton_);
    conversionButtonLayout->addStretch(1);
    conversionLayout->addWidget(conversionButtonRow);

    datasetConversionProgressBar_ = new QProgressBar;
    datasetConversionProgressBar_->setRange(0, 100);
    datasetConversionProgressBar_->setValue(0);
    conversionLayout->addWidget(datasetConversionProgressBar_);

    datasetConversionResultLabel_ = mutedLabel(QStringLiteral("转换结果会显示在这里。"));
    allowLabelToShrink(datasetConversionResultLabel_);
    conversionLayout->addWidget(datasetConversionResultLabel_);

    datasetConversionLog_ = new QPlainTextEdit;
    datasetConversionLog_->setReadOnly(true);
    datasetConversionLog_->setMinimumHeight(96);
    datasetConversionLog_->setPlainText(QStringLiteral("等待转换。"));
    conversionLayout->addWidget(datasetConversionLog_);

    inputPanel->bodyLayout()->addWidget(conversionPanel);
    updateDatasetConversionTargetFormats();
```

- [x] **Step 4: Synchronize dataset selection into conversion form**

Inside the existing `datasetListTable_` selection lambda, after `currentDatasetFormat_ = format;`, insert:

```cpp
            refreshDatasetConversionDefaultsFromCurrentDataset();
```

Inside `MainWindow::browseDataset()`, after `currentDatasetFormat_ = selectedFormat.isEmpty() ? detectedFormat : selectedFormat;`, insert:

```cpp
        refreshDatasetConversionDefaultsFromCurrentDataset();
```

- [x] **Step 5: Build to catch declaration/layout errors**

Run:

```powershell
cmake --build build-vscode --target AITrainStudio
```

Expected: FAIL because the new slots and helpers are declared but not implemented yet.

- [x] **Step 6: Keep the form shell uncommitted**

Do not commit this state. The next task implements the declared methods and commits the passing UI shell plus actions together, so no commit leaves `AITrainStudio` unable to build.

---

### Task 4: Implement Form Behavior and Worker Start/Cancel

**Files:**
- Modify: `src/app/src/MainWindowActions.cpp`
- Modify: `src/app/src/MainWindowDatasetPage.cpp`
- Test: `tests/tst_dataset_conversion_ui.cpp`

- [x] **Step 1: Include helper in `MainWindowActions.cpp`**

Add:

```cpp
#include "DatasetConversionUiModel.h"
```

- [x] **Step 2: Add local helper for default output paths**

In the anonymous namespace of `MainWindowActions.cpp`, add:

```cpp
QString defaultDatasetConversionOutputPath(const QString& sourcePath, const QString& projectPath, const QString& targetFormat)
{
    const QString cleanSource = QDir::cleanPath(QDir::fromNativeSeparators(sourcePath.trimmed()));
    const QString datasetName = QFileInfo(cleanSource).fileName().isEmpty()
        ? QStringLiteral("dataset")
        : QFileInfo(cleanSource).fileName();
    const QString suffix = targetFormat.isEmpty()
        ? QStringLiteral("converted")
        : targetFormat;
    if (!projectPath.trimmed().isEmpty()) {
        return QDir(projectPath).filePath(QStringLiteral("datasets/converted/%1-%2").arg(datasetName, suffix));
    }
    return QDir(cleanSource).absoluteFilePath(QStringLiteral("../converted/%1-%2").arg(datasetName, suffix));
}

void setFieldErrorLabel(QLabel* label, const QString& text)
{
    if (!label) {
        return;
    }
    label->setText(text);
    label->setVisible(!text.isEmpty());
}
```

- [x] **Step 3: Implement target filtering and default sync**

Add these methods to `MainWindowActions.cpp`:

```cpp
void MainWindow::updateDatasetConversionTargetFormats()
{
    if (!datasetConversionSourceFormatCombo_ || !datasetConversionTargetFormatCombo_) {
        return;
    }
    const QString sourceFormat = comboCurrentDataOrText(datasetConversionSourceFormatCombo_);
    const QString previousTarget = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);

    datasetConversionTargetFormatCombo_->blockSignals(true);
    datasetConversionTargetFormatCombo_->clear();
    for (const QString& targetFormat : supportedDatasetConversionTargets(sourceFormat)) {
        addComboItem(datasetConversionTargetFormatCombo_, datasetConversionFormatLabel(targetFormat), targetFormat);
    }
    const int previousIndex = datasetConversionTargetFormatCombo_->findData(previousTarget);
    if (previousIndex >= 0) {
        datasetConversionTargetFormatCombo_->setCurrentIndex(previousIndex);
    }
    datasetConversionTargetFormatCombo_->blockSignals(false);
}

void MainWindow::refreshDatasetConversionDefaultsFromCurrentDataset()
{
    if (!datasetConversionInputEdit_ || !datasetConversionSourceFormatCombo_) {
        return;
    }

    const QString path = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    const QString format = currentDatasetFormat_.isEmpty() ? currentDatasetFormat() : currentDatasetFormat_;
    if (!path.isEmpty()) {
        datasetConversionInputEdit_->setText(QDir::toNativeSeparators(path));
    }
    if (!format.isEmpty() && datasetConversionSourceFormatCombo_->findData(format) >= 0) {
        setComboCurrentData(datasetConversionSourceFormatCombo_, format);
        updateDatasetConversionTargetFormats();
    }
    if (datasetConversionOutputEdit_ && datasetConversionOutputEdit_->text().trimmed().isEmpty() && !path.isEmpty()) {
        const QString target = datasetConversionTargetFormatCombo_
            ? comboCurrentDataOrText(datasetConversionTargetFormatCombo_)
            : QString();
        datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(
            defaultDatasetConversionOutputPath(path, currentProjectPath_, target)));
    }
}
```

- [x] **Step 4: Implement browsing and validation display**

Add:

```cpp
void MainWindow::browseDatasetConversionInput()
{
    const QString directory = QFileDialog::getExistingDirectory(this, uiText("选择待转换数据集目录"));
    if (directory.isEmpty()) {
        return;
    }
    datasetConversionInputEdit_->setText(QDir::toNativeSeparators(directory));
    const QString detectedFormat = detectDatasetFormatFromPath(directory);
    if (!detectedFormat.isEmpty() && datasetConversionSourceFormatCombo_->findData(detectedFormat) >= 0) {
        setComboCurrentData(datasetConversionSourceFormatCombo_, detectedFormat);
        updateDatasetConversionTargetFormats();
    }
    const QString target = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
    datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(
        defaultDatasetConversionOutputPath(directory, currentProjectPath_, target)));
}

void MainWindow::browseDatasetConversionOutput()
{
    const QString directory = QFileDialog::getExistingDirectory(this, uiText("选择转换输出目录"));
    if (!directory.isEmpty()) {
        datasetConversionOutputEdit_->setText(QDir::toNativeSeparators(directory));
    }
}

void MainWindow::clearDatasetConversionErrors()
{
    setFieldErrorLabel(datasetConversionSourceErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionTargetErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionInputErrorLabel_, QString());
    setFieldErrorLabel(datasetConversionOutputErrorLabel_, QString());
}

void MainWindow::appendDatasetConversionLog(const QString& text)
{
    if (!datasetConversionLog_ || text.trimmed().isEmpty()) {
        return;
    }
    if (datasetConversionLog_->toPlainText() == QStringLiteral("等待转换。")) {
        datasetConversionLog_->clear();
    }
    datasetConversionLog_->appendPlainText(text.trimmed());
}
```

- [x] **Step 5: Implement running state and cancellation**

Add:

```cpp
void MainWindow::setDatasetConversionFormRunning(bool running)
{
    if (datasetConversionSourceFormatCombo_) {
        datasetConversionSourceFormatCombo_->setEnabled(!running);
    }
    if (datasetConversionTargetFormatCombo_) {
        datasetConversionTargetFormatCombo_->setEnabled(!running);
    }
    if (datasetConversionInputEdit_) {
        datasetConversionInputEdit_->setEnabled(!running);
    }
    if (datasetConversionOutputEdit_) {
        datasetConversionOutputEdit_->setEnabled(!running);
    }
    if (datasetConversionStartButton_) {
        datasetConversionStartButton_->setEnabled(!running);
    }
    if (datasetConversionCancelButton_) {
        datasetConversionCancelButton_->setEnabled(running);
    }
}

void MainWindow::cancelDatasetConversion()
{
    if (!worker_.isRunning()) {
        return;
    }
    worker_.cancel();
    if (datasetConversionStatusLabel_) {
        datasetConversionStatusLabel_->setText(uiText("正在取消数据集转换。"));
    }
    appendDatasetConversionLog(uiText("正在取消数据集转换。"));
}
```

- [x] **Step 6: Implement Worker start**

Add:

```cpp
void MainWindow::startDatasetConversion()
{
    clearDatasetConversionErrors();

    aitrain_app::DatasetConversionForm form;
    form.sourceFormat = comboCurrentDataOrText(datasetConversionSourceFormatCombo_);
    form.targetFormat = comboCurrentDataOrText(datasetConversionTargetFormatCombo_);
    form.inputPath = datasetConversionInputEdit_ ? datasetConversionInputEdit_->text() : QString();
    form.outputPath = datasetConversionOutputEdit_ ? datasetConversionOutputEdit_->text() : QString();
    form.workerRunning = worker_.isRunning();

    const aitrain_app::DatasetConversionValidation validation =
        validateDatasetConversionForm(form);
    if (!validation.ok) {
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(validation.summary);
        }
        setFieldErrorLabel(datasetConversionSourceErrorLabel_, validation.sourceFormatError);
        setFieldErrorLabel(datasetConversionTargetErrorLabel_, validation.targetFormatError);
        setFieldErrorLabel(datasetConversionInputErrorLabel_, validation.inputPathError);
        setFieldErrorLabel(datasetConversionOutputErrorLabel_, validation.outputPathError);
        return;
    }

    const QString sourcePath = normalizedDatasetConversionPath(form.inputPath);
    const QString outputPath = normalizedDatasetConversionPath(form.outputPath);
    if (!QDir().mkpath(outputPath)) {
        const QString message = uiText("无法创建输出目录：%1").arg(QDir::toNativeSeparators(outputPath));
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(message);
        }
        setFieldErrorLabel(datasetConversionOutputErrorLabel_, message);
        return;
    }

    QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    if (repository_.isOpen()) {
        const QString createdTaskId = createRepositoryTask(
            aitrain::TaskKind::Validate,
            QStringLiteral("dataset_conversion"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集格式转换中。"),
            taskId);
        if (createdTaskId.isEmpty()) {
            return;
        }
        taskId = createdTaskId;
    }

    QJsonObject options;
    options.insert(QStringLiteral("copyImages"), true);
    options.insert(QStringLiteral("maxIssues"), 200);

    if (datasetConversionProgressBar_) {
        datasetConversionProgressBar_->setValue(0);
    }
    if (datasetConversionLog_) {
        datasetConversionLog_->clear();
    }
    if (datasetConversionStatusLabel_) {
        datasetConversionStatusLabel_->setText(uiText("正在通过 Worker 转换数据集。"));
    }
    if (datasetConversionResultLabel_) {
        datasetConversionResultLabel_->setText(uiText("等待转换结果。"));
    }

    currentDatasetConversionTaskId_ = taskId;
    setDatasetConversionFormRunning(true);
    appendDatasetConversionLog(uiText("开始转换数据集。"));

    QString error;
    if (!worker_.requestDatasetConversion(
            workerExecutablePath(),
            sourcePath,
            outputPath,
            form.sourceFormat,
            form.targetFormat,
            options,
            &error,
            taskId)) {
        if (repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        currentDatasetConversionTaskId_.clear();
        setDatasetConversionFormRunning(false);
        const QString message = uiText("无法启动数据集转换：%1").arg(error);
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(message);
        }
        appendDatasetConversionLog(message);
        QMessageBox::critical(this, uiText("数据集转换"), error);
        return;
    }

    workerPill_->setStatus(uiText("数据集转换中"), StatusPill::Tone::Info);
    statusBar()->showMessage(uiText("正在转换数据集"), 3000);
}
```

- [x] **Step 7: Build app**

Run:

```powershell
cmake --build build-vscode --target AITrainStudio
```

Expected: PASS after all declared methods are implemented.

- [x] **Step 8: Commit form behavior**

```powershell
git add src/app/src/MainWindow.h src/app/src/MainWindowDatasetPage.cpp src/app/src/MainWindowActions.cpp
git commit -m "feat: wire dataset conversion form actions"
```

---

### Task 5: Handle Conversion Messages and Worker Completion

**Files:**
- Modify: `src/app/src/MainWindowWorkerMessages.cpp`
- Modify: `src/app/src/MainWindow.cpp`

- [x] **Step 1: Route `datasetConversion` messages**

In `MainWindow::handleWorkerMessage`, insert after `datasetSplit`:

```cpp
    } else if (type == QStringLiteral("datasetConversion")) {
        updateDatasetConversionResult(payload);
```

- [x] **Step 2: Update conversion progress inside `handleProgressMessage`**

Add after the global progress bar update:

```cpp
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    if (!currentDatasetConversionTaskId_.isEmpty()
        && taskId == currentDatasetConversionTaskId_
        && datasetConversionProgressBar_) {
        datasetConversionProgressBar_->setValue(payload.value(QStringLiteral("percent")).toInt());
    }
```

Keep the existing status-bar message block, and add this inside that block after the existing `statusBar()->showMessage` call:

```cpp
        if (!currentDatasetConversionTaskId_.isEmpty()
            && payload.value(QStringLiteral("taskId")).toString() == currentDatasetConversionTaskId_) {
            appendDatasetConversionLog(message);
        }
```

- [x] **Step 3: Implement result rendering**

Add this method to `MainWindowWorkerMessages.cpp`:

```cpp
void MainWindow::updateDatasetConversionResult(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString reportPath = payload.value(QStringLiteral("reportPath")).toString();
    const QString errorCode = payload.value(QStringLiteral("errorCode")).toString();
    const QString errorMessage = payload.value(QStringLiteral("errorMessage")).toString();
    const int convertedSamples = payload.value(QStringLiteral("convertedSampleCount")).toInt();
    const int skippedSamples = payload.value(QStringLiteral("skippedSampleCount")).toInt();
    const int convertedAnnotations = payload.value(QStringLiteral("convertedAnnotationCount")).toInt();
    const int skippedAnnotations = payload.value(QStringLiteral("skippedAnnotationCount")).toInt();

    if (datasetConversionProgressBar_) {
        datasetConversionProgressBar_->setValue(ok ? 100 : datasetConversionProgressBar_->value());
    }

    if (ok) {
        const QString summary = uiText("转换完成：%1 个样本，跳过 %2 个样本；%3 条标注，跳过 %4 条标注。")
            .arg(convertedSamples)
            .arg(skippedSamples)
            .arg(convertedAnnotations)
            .arg(skippedAnnotations);
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(summary);
        }
        if (datasetConversionResultLabel_) {
            datasetConversionResultLabel_->setText(uiText("输出：%1\n报告：%2")
                .arg(QDir::toNativeSeparators(outputPath),
                     reportPath.isEmpty() ? uiText("未返回报告路径") : QDir::toNativeSeparators(reportPath)));
        }
        appendDatasetConversionLog(summary);
    } else {
        const QString summary = errorCode.isEmpty()
            ? uiText("转换失败：%1").arg(errorMessage.isEmpty() ? uiText("Worker 未返回详细错误。") : errorMessage)
            : uiText("转换失败：%1 | %2").arg(errorCode, errorMessage);
        if (datasetConversionStatusLabel_) {
            datasetConversionStatusLabel_->setText(summary);
        }
        if (datasetConversionResultLabel_) {
            datasetConversionResultLabel_->setText(summary);
        }
        appendDatasetConversionLog(summary);
    }

    const QByteArray json = QJsonDocument(payload).toJson(QJsonDocument::Indented);
    appendDatasetConversionLog(QString::fromUtf8(json));
    setDatasetConversionFormRunning(false);
    currentDatasetConversionTaskId_.clear();
}
```

- [x] **Step 4: Reset conversion UI on canceled/failed task-state messages**

Inside `handleTaskStateMessage`, in the `canceled` branch before `currentTaskId_.clear();`, add:

```cpp
        if (!currentDatasetConversionTaskId_.isEmpty() && canceledTaskId == currentDatasetConversionTaskId_) {
            setDatasetConversionFormRunning(false);
            currentDatasetConversionTaskId_.clear();
            if (datasetConversionStatusLabel_) {
                datasetConversionStatusLabel_->setText(uiText("数据集转换已取消。"));
            }
        }
```

Inside the `failed` branch after `updateModelRegistry();`, add:

```cpp
        if (!currentDatasetConversionTaskId_.isEmpty() && failedTaskId == currentDatasetConversionTaskId_) {
            setDatasetConversionFormRunning(false);
            currentDatasetConversionTaskId_.clear();
            if (datasetConversionStatusLabel_) {
                datasetConversionStatusLabel_->setText(uiText("数据集转换失败：%1").arg(failedMessage));
            }
        }
```

- [x] **Step 5: Reset conversion UI from Worker finished fallback**

In `src/app/src/MainWindow.cpp`, inside the `WorkerClient::finished` lambda after the existing `appendLog` call, add:

```cpp
        if (!currentDatasetConversionTaskId_.isEmpty()) {
            setDatasetConversionFormRunning(false);
            if (!ok && datasetConversionStatusLabel_) {
                datasetConversionStatusLabel_->setText(uiText("数据集转换失败：%1").arg(message));
            }
            currentDatasetConversionTaskId_.clear();
        }
```

- [x] **Step 6: Run focused build**

Run:

```powershell
cmake --build build-vscode --target AITrainStudio
```

Expected: PASS.

- [x] **Step 7: Commit message handling**

```powershell
git add src/app/src/MainWindowWorkerMessages.cpp src/app/src/MainWindow.cpp
git commit -m "feat: show dataset conversion worker results"
```

---

### Task 6: Add English Fallback Text and Final Verification

**Files:**
- Modify: `src/app/src/LanguageSupport.cpp`
- Test: full harness and UI walkthrough

- [x] **Step 1: Add fallback translations**

In `fallbackEnglishTranslation`, add these entries to the static hash:

```cpp
        {QStringLiteral("格式转换"), QStringLiteral("Format Conversion")},
        {QStringLiteral("选择源格式、目标格式和输出目录后开始转换。"), QStringLiteral("Choose source format, target format, and output folder before conversion.")},
        {QStringLiteral("选择待转换的数据集目录"), QStringLiteral("Choose the dataset folder to convert")},
        {QStringLiteral("选择不同于输入目录的输出目录"), QStringLiteral("Choose an output folder different from the input folder")},
        {QStringLiteral("选择输入"), QStringLiteral("Choose Input")},
        {QStringLiteral("选择输出"), QStringLiteral("Choose Output")},
        {QStringLiteral("源格式"), QStringLiteral("Source Format")},
        {QStringLiteral("目标格式"), QStringLiteral("Target Format")},
        {QStringLiteral("输入目录"), QStringLiteral("Input Folder")},
        {QStringLiteral("输出目录"), QStringLiteral("Output Folder")},
        {QStringLiteral("转换数据集"), QStringLiteral("Convert Dataset")},
        {QStringLiteral("取消转换"), QStringLiteral("Cancel Conversion")},
        {QStringLiteral("转换结果会显示在这里。"), QStringLiteral("Conversion results will appear here.")},
        {QStringLiteral("等待转换。"), QStringLiteral("Waiting for conversion.")},
        {QStringLiteral("选择待转换数据集目录"), QStringLiteral("Choose Dataset Folder to Convert")},
        {QStringLiteral("选择转换输出目录"), QStringLiteral("Choose Conversion Output Folder")},
        {QStringLiteral("正在取消数据集转换。"), QStringLiteral("Canceling dataset conversion.")},
        {QStringLiteral("可以开始转换。"), QStringLiteral("Ready to convert.")},
        {QStringLiteral("请选择源格式。"), QStringLiteral("Choose a source format.")},
        {QStringLiteral("当前不支持该源格式。"), QStringLiteral("This source format is not supported.")},
        {QStringLiteral("请选择目标格式。"), QStringLiteral("Choose a target format.")},
        {QStringLiteral("当前源格式不支持转换到该目标格式。"), QStringLiteral("The selected source format cannot convert to that target format.")},
        {QStringLiteral("请选择输入目录。"), QStringLiteral("Choose an input folder.")},
        {QStringLiteral("输入目录不存在。"), QStringLiteral("Input folder does not exist.")},
        {QStringLiteral("输入路径必须是目录。"), QStringLiteral("Input path must be a folder.")},
        {QStringLiteral("请选择输出目录。"), QStringLiteral("Choose an output folder.")},
        {QStringLiteral("输出路径必须是目录。"), QStringLiteral("Output path must be a folder.")},
        {QStringLiteral("输出目录不能与输入目录相同。"), QStringLiteral("Output folder cannot be the same as the input folder.")},
        {QStringLiteral("输出目录的父目录不存在。"), QStringLiteral("Output folder parent does not exist.")},
        {QStringLiteral("输出目录的父目录不可写。"), QStringLiteral("Output folder parent is not writable.")},
        {QStringLiteral("请修正 %1 个字段后再转换。"), QStringLiteral("Fix %1 fields before converting.")},
        {QStringLiteral("Worker 正在执行任务，稍后再转换数据集。"), QStringLiteral("Worker is running a task. Convert the dataset later.")},
        {QStringLiteral("无法创建输出目录：%1"), QStringLiteral("Failed to create output folder: %1")},
        {QStringLiteral("数据集格式转换中。"), QStringLiteral("Dataset format conversion in progress.")},
        {QStringLiteral("正在通过 Worker 转换数据集。"), QStringLiteral("Converting the dataset through Worker.")},
        {QStringLiteral("等待转换结果。"), QStringLiteral("Waiting for conversion results.")},
        {QStringLiteral("开始转换数据集。"), QStringLiteral("Starting dataset conversion.")},
        {QStringLiteral("无法启动数据集转换：%1"), QStringLiteral("Failed to start dataset conversion: %1")},
        {QStringLiteral("数据集转换"), QStringLiteral("Dataset Conversion")},
        {QStringLiteral("数据集转换中"), QStringLiteral("Converting dataset")},
        {QStringLiteral("正在转换数据集"), QStringLiteral("Converting dataset")},
        {QStringLiteral("转换完成：%1 个样本，跳过 %2 个样本；%3 条标注，跳过 %4 条标注。"), QStringLiteral("Conversion complete: %1 samples, %2 skipped samples; %3 annotations, %4 skipped annotations.")},
        {QStringLiteral("输出：%1\n报告：%2"), QStringLiteral("Output: %1\nReport: %2")},
        {QStringLiteral("未返回报告路径"), QStringLiteral("No report path returned")},
        {QStringLiteral("转换失败：%1"), QStringLiteral("Conversion failed: %1")},
        {QStringLiteral("Worker 未返回详细错误。"), QStringLiteral("Worker did not return a detailed error.")},
        {QStringLiteral("转换失败：%1 | %2"), QStringLiteral("Conversion failed: %1 | %2")},
        {QStringLiteral("数据集转换已取消。"), QStringLiteral("Dataset conversion canceled.")},
        {QStringLiteral("数据集转换失败：%1"), QStringLiteral("Dataset conversion failed: %1")},
```

- [x] **Step 2: Run focused tests**

Run:

```powershell
cmake --build build-vscode --target aitrain_dataset_conversion_ui_tests
ctest --test-dir build-vscode -R "aitrain_dataset_conversion_ui_tests|aitrain_dataset_conversion_tests|aitrain_repository_workflow_tests" --output-on-failure
```

Expected: PASS.

- [x] **Step 3: Run full harness**

Run:

```powershell
.\tools\harness-check.ps1
```

Expected: configure/build succeeds and all CTest tests pass.

- [x] **Step 4: Run UI walkthrough**

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

Expected: PASS or produces screenshots with no horizontal overflow on the 数据集 page. If the walkthrough cannot run because license or GUI automation is unavailable, record the exact failure and verify at least the 数据集 page manually before claiming UI validation.

- [x] **Step 5: Check whitespace and status**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` returns success. `git status --short` lists only intentional source, test, translation, and plan/spec changes until commit.

- [x] **Step 6: Commit final UI verification updates**

```powershell
git add src/app/src/LanguageSupport.cpp
git commit -m "chore: localize dataset conversion gui text"
```

If Task 6 only changes `LanguageSupport.cpp`, this commit should contain only that file.

---

## Final Acceptance

- `aitrain_dataset_conversion_ui_tests` passes.
- Existing `aitrain_dataset_conversion_tests` still pass.
- Existing Worker-backed repository workflow conversion tests still pass.
- `.\tools\harness-check.ps1` passes.
- UI walkthrough at 1280x820 verifies the 数据集 page has no horizontal overflow and the new conversion form is usable.
- No core conversion semantics changed.
- No SQLite schema changed.
- No plugin interface changed.
- Conversion result is not automatically registered as a dataset.

## Implementation Notes

- Use `QStringLiteral` for all new visible UI text.
- Keep `DatasetConversionUiModel` free of `QWidget` and `MainWindow` dependencies.
- Keep actual conversion work inside Worker/core through `WorkerClient::requestDatasetConversion`.
- Use `aitrain::TaskKind::Validate` with `taskType=dataset_conversion` for repository task history because the current enum has no conversion kind and this plan must not alter schema or task enums.
- Do not update `ProjectRepository::upsertDatasetValidation` from conversion result handling.
