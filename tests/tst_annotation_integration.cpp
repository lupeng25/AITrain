#include "aitrain/core/AnnotationIntegration.h"

#include "TestSupport.h"

#include <QDir>
#include <QFile>
#include <QFileDevice>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QTest>

class AnnotationIntegrationTests : public QObject {
    Q_OBJECT

private slots:
    void missingExecutableWritesEnvironmentReport();
    void environmentPrefersExplicitExecutableOverEnv();
    void xAnyCliMissingConversionFailsWithReport();
    void xAnyCliYoloCustomSplitPathsUseFlatStaging();
    void xAnyCliYoloDuplicateBasenamesAcrossSplitsUseDistinctStaging();
    void xAnyCliYoloMissingLabelsStageAsEmptyFiles();
    void xAnyCliYoloStagingStripsBomFromLabels();
    void xAnyCliXLabelToYoloPassesImagesPath();
    void xAnyCliXLabelToYoloFallsBackToImagesSubdirForBasenameImagePath();
    void xAnyCliXLabelToYoloHandlesUnicodeSpacePaths();
    void xAnyCliCancellationReturnsCanceled();
};

namespace {

void writeTextFileForAnnotationTest(const QString& path, const QString& text)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
    file.write(text.toUtf8());
}

void writeBinaryFileForAnnotationTest(const QString& path, const QByteArray& bytes)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(bytes);
}

QJsonObject readJsonObjectForAnnotationTest(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QJsonDocument::fromJson(file.readAll()).object();
}

QJsonObject disabledDiscoveryOptions()
{
    QJsonObject options;
    options.insert(QStringLiteral("xAnyLabelingExecutable"), QStringLiteral("Z:/missing/X-AnyLabeling.exe"));
    options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);
    return options;
}

QString writeFakeXAnyExecutable(const QDir& root, bool writeXLabelJson = false)
{
#ifdef Q_OS_WIN
    const QString scriptPath = root.filePath(QStringLiteral("fake_xany.cmd"));
    if (writeXLabelJson) {
        writeTextFileForAnnotationTest(scriptPath,
            QStringLiteral("@echo off\r\n"
                           "setlocal enabledelayedexpansion\r\n"
                           "set \"IMAGES=\"\r\n"
                           "set \"OUTPUT=\"\r\n"
                           ":parse\r\n"
                           "if \"%~1\"==\"\" goto write\r\n"
                           "if \"%~1\"==\"--images\" (\r\n"
                           "  set \"IMAGES=%~2\"\r\n"
                           "  shift\r\n"
                           "  shift\r\n"
                           "  goto parse\r\n"
                           ")\r\n"
                           "if \"%~1\"==\"--output\" (\r\n"
                           "  set \"OUTPUT=%~2\"\r\n"
                           "  shift\r\n"
                           "  shift\r\n"
                           "  goto parse\r\n"
                           ")\r\n"
                           "shift\r\n"
                           "goto parse\r\n"
                           ":write\r\n"
                           "if \"%IMAGES%\"==\"\" exit /b 2\r\n"
                           "if \"%OUTPUT%\"==\"\" exit /b 2\r\n"
                           "if not exist \"%OUTPUT%\" mkdir \"%OUTPUT%\"\r\n"
                           "pushd \"%IMAGES%\" || exit /b 3\r\n"
                           "for %%F in (*) do (\r\n"
                           "  set \"BASE=%%~nxF\"\r\n"
                           "  set \"NAME=%%~nF\"\r\n"
                           "  >\"%OUTPUT%\\!NAME!.json\" echo {\"imagePath\":\"!BASE!\",\"shapes\":[]}\r\n"
                           ")\r\n"
                           "popd\r\n"
                           "exit /b 0\r\n"));
    } else {
        writeTextFileForAnnotationTest(scriptPath, QStringLiteral("@echo off\r\nexit /b 0\r\n"));
    }
#else
    const QString scriptPath = root.filePath(QStringLiteral("fake_xany.sh"));
    if (writeXLabelJson) {
        writeTextFileForAnnotationTest(scriptPath,
            QStringLiteral("#!/bin/sh\n"
                           "images=\"\"\n"
                           "output=\"\"\n"
                           "while [ \"$#\" -gt 0 ]; do\n"
                           "  case \"$1\" in\n"
                           "    --images) images=\"$2\"; shift 2 ;;\n"
                           "    --output) output=\"$2\"; shift 2 ;;\n"
                           "    *) shift ;;\n"
                           "  esac\n"
                           "done\n"
                           "[ -n \"$images\" ] || exit 2\n"
                           "[ -n \"$output\" ] || exit 2\n"
                           "mkdir -p \"$output\"\n"
                           "for file in \"$images\"/*; do\n"
                           "  [ -f \"$file\" ] || continue\n"
                           "  base=$(basename \"$file\")\n"
                           "  name=${base%.*}\n"
                           "  printf '{\"imagePath\":\"%s\",\"shapes\":[]}\\n' \"$base\" > \"$output/$name.json\"\n"
                           "done\n"
                           "exit 0\n"));
    } else {
        writeTextFileForAnnotationTest(scriptPath, QStringLiteral("#!/bin/sh\nexit 0\n"));
    }
    QFile::setPermissions(scriptPath, QFile::permissions(scriptPath) | QFileDevice::ExeOwner | QFileDevice::ExeUser);
#endif
    return QFileInfo(scriptPath).absoluteFilePath();
}

QString writeLabelEchoXAnyExecutable(const QDir& root)
{
#ifdef Q_OS_WIN
    const QString scriptPath = root.filePath(QStringLiteral("fake_xany_label_echo.cmd"));
    writeTextFileForAnnotationTest(scriptPath,
        QStringLiteral("@echo off\r\n"
                       "setlocal enabledelayedexpansion\r\n"
                       "set \"IMAGES=\"\r\n"
                       "set \"LABELS=\"\r\n"
                       "set \"OUTPUT=\"\r\n"
                       ":parse\r\n"
                       "if \"%~1\"==\"\" goto write\r\n"
                       "if \"%~1\"==\"--images\" (\r\n"
                       "  set \"IMAGES=%~2\"\r\n"
                       "  shift\r\n"
                       "  shift\r\n"
                       "  goto parse\r\n"
                       ")\r\n"
                       "if \"%~1\"==\"--labels\" (\r\n"
                       "  set \"LABELS=%~2\"\r\n"
                       "  shift\r\n"
                       "  shift\r\n"
                       "  goto parse\r\n"
                       ")\r\n"
                       "if \"%~1\"==\"--output\" (\r\n"
                       "  set \"OUTPUT=%~2\"\r\n"
                       "  shift\r\n"
                       "  shift\r\n"
                       "  goto parse\r\n"
                       ")\r\n"
                       "shift\r\n"
                       "goto parse\r\n"
                       ":write\r\n"
                       "if \"%IMAGES%\"==\"\" exit /b 2\r\n"
                       "if \"%LABELS%\"==\"\" exit /b 2\r\n"
                       "if \"%OUTPUT%\"==\"\" exit /b 2\r\n"
                       "if not exist \"%OUTPUT%\" mkdir \"%OUTPUT%\"\r\n"
                       "for %%L in (\"%LABELS%\\*.txt\") do (\r\n"
                       "  copy /Y \"%%~fL\" \"%OUTPUT%\\observed_label.txt\" >nul\r\n"
                       "  goto copied\r\n"
                       ")\r\n"
                       ":copied\r\n"
                       "pushd \"%IMAGES%\" || exit /b 3\r\n"
                       "for %%F in (*) do (\r\n"
                       "  set \"BASE=%%~nxF\"\r\n"
                       "  set \"NAME=%%~nF\"\r\n"
                       "  >\"%OUTPUT%\\!NAME!.json\" echo {\"imagePath\":\"!BASE!\",\"shapes\":[]}\r\n"
                       ")\r\n"
                       "popd\r\n"
                       "exit /b 0\r\n"));
#else
    const QString scriptPath = root.filePath(QStringLiteral("fake_xany_label_echo.sh"));
    writeTextFileForAnnotationTest(scriptPath,
        QStringLiteral("#!/bin/sh\n"
                       "images=\"\"\n"
                       "labels=\"\"\n"
                       "output=\"\"\n"
                       "while [ \"$#\" -gt 0 ]; do\n"
                       "  case \"$1\" in\n"
                       "    --images) images=\"$2\"; shift 2 ;;\n"
                       "    --labels) labels=\"$2\"; shift 2 ;;\n"
                       "    --output) output=\"$2\"; shift 2 ;;\n"
                       "    *) shift ;;\n"
                       "  esac\n"
                       "done\n"
                       "[ -n \"$images\" ] || exit 2\n"
                       "[ -n \"$labels\" ] || exit 2\n"
                       "[ -n \"$output\" ] || exit 2\n"
                       "mkdir -p \"$output\"\n"
                       "for label in \"$labels\"/*.txt; do\n"
                       "  [ -f \"$label\" ] || continue\n"
                       "  cp \"$label\" \"$output/observed_label.txt\"\n"
                       "  break\n"
                       "done\n"
                       "for file in \"$images\"/*; do\n"
                       "  [ -f \"$file\" ] || continue\n"
                       "  base=$(basename \"$file\")\n"
                       "  name=${base%.*}\n"
                       "  printf '{\"imagePath\":\"%s\",\"shapes\":[]}\\n' \"$base\" > \"$output/$name.json\"\n"
                       "done\n"
                       "exit 0\n"));
    QFile::setPermissions(scriptPath, QFile::permissions(scriptPath) | QFileDevice::ExeOwner | QFileDevice::ExeUser);
#endif
    return QFileInfo(scriptPath).absoluteFilePath();
}

QString writeSlowFakeXAnyExecutable(const QDir& root)
{
#ifdef Q_OS_WIN
    const QString scriptPath = root.filePath(QStringLiteral("slow_xany.cmd"));
    writeTextFileForAnnotationTest(scriptPath,
        QStringLiteral("@echo off\r\n"
                       "ping -n 6 127.0.0.1 >nul\r\n"
                       "exit /b 0\r\n"));
#else
    const QString scriptPath = root.filePath(QStringLiteral("slow_xany.sh"));
    writeTextFileForAnnotationTest(scriptPath,
        QStringLiteral("#!/bin/sh\n"
                       "sleep 5\n"
                       "exit 0\n"));
    QFile::setPermissions(scriptPath, QFile::permissions(scriptPath) | QFileDevice::ExeOwner | QFileDevice::ExeUser);
#endif
    return QFileInfo(scriptPath).absoluteFilePath();
}

QStringList stringListFromJsonArrayForAnnotationTest(const QJsonArray& array)
{
    QStringList values;
    for (const QJsonValue& value : array) {
        values.append(value.toString());
    }
    return values;
}

QString argumentValueForAnnotationTest(const QStringList& arguments, const QString& key)
{
    const int index = arguments.indexOf(key);
    if (index < 0 || index + 1 >= arguments.size()) {
        return {};
    }
    return arguments.at(index + 1);
}

} // namespace

void AnnotationIntegrationTests::missingExecutableWritesEnvironmentReport()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());

    const aitrain::WorkflowResult result =
        aitrain::inspectXAnyLabelingEnvironment(temp.path(), disabledDiscoveryOptions());
    QVERIFY(result.ok);
    QVERIFY(QFileInfo::exists(result.reportPath));
    QCOMPARE(result.payload.value(QStringLiteral("status")).toString(), QStringLiteral("missing"));
    QCOMPARE(result.payload.value(QStringLiteral("kind")).toString(), QStringLiteral("xanylabeling_environment_report"));
}

void AnnotationIntegrationTests::environmentPrefersExplicitExecutableOverEnv()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString envExecutable = writeFakeXAnyExecutable(QDir(root.filePath(QStringLiteral("env_tool"))));
    const QString requestedExecutable = writeFakeXAnyExecutable(QDir(root.filePath(QStringLiteral("requested_tool"))));
    ScopedEnvVar env("AITRAIN_XANYLABELING_EXE", envExecutable.toLocal8Bit());

    QJsonObject options;
    options.insert(QStringLiteral("xAnyLabelingExecutable"), requestedExecutable);
    options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::WorkflowResult result =
        aitrain::inspectXAnyLabelingEnvironment(root.filePath(QStringLiteral("environment")), options);
    QVERIFY2(result.ok, qPrintable(result.error));
    QCOMPARE(QDir::cleanPath(result.payload.value(QStringLiteral("executable")).toString()),
        QDir::cleanPath(requestedExecutable));

    const QStringList candidates = stringListFromJsonArrayForAnnotationTest(
        result.payload.value(QStringLiteral("candidates")).toArray());
    QVERIFY(!candidates.isEmpty());
    QCOMPARE(QDir::cleanPath(candidates.first()), QDir::cleanPath(requestedExecutable));
    QVERIFY(candidates.contains(envExecutable));
}

void AnnotationIntegrationTests::xAnyCliMissingConversionFailsWithReport()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());

    aitrain::DatasetConversionRequest request;
    request.sourcePath = root.filePath(QStringLiteral("dataset"));
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("xanylabeling_xlabel");
    request.outputPath = root.filePath(QStringLiteral("converted"));
    request.options = disabledDiscoveryOptions();

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY(!result.ok);
    QCOMPARE(result.errorCode, QStringLiteral("xanylabeling_missing"));
    QVERIFY(QFileInfo::exists(result.reportPath));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("conversionEngine")).toString(), QStringLiteral("xanylabeling_cli"));
    QCOMPARE(report.value(QStringLiteral("errorCode")).toString(), QStringLiteral("xanylabeling_missing"));
}

void AnnotationIntegrationTests::xAnyCliYoloCustomSplitPathsUseFlatStaging()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString datasetPath = root.filePath(QStringLiteral("dataset"));
    QVERIFY(QDir().mkpath(QDir(datasetPath).filePath(QStringLiteral("raw/custom/images/train"))));
    QVERIFY(QDir().mkpath(QDir(datasetPath).filePath(QStringLiteral("raw/custom/images/val"))));
    QVERIFY(QDir().mkpath(QDir(datasetPath).filePath(QStringLiteral("raw/custom/labels/train"))));
    QVERIFY(QDir().mkpath(QDir(datasetPath).filePath(QStringLiteral("raw/custom/labels/val"))));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("raw/custom/images/train/a.jpg")),
        QStringLiteral("fake image\n"));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("raw/custom/images/val/b.jpg")),
        QStringLiteral("fake image\n"));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("raw/custom/labels/train/a.txt")),
        QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("raw/custom/labels/val/b.txt")),
        QStringLiteral("1 0.5 0.5 0.20 0.20\n"));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: raw\ntrain: custom/images/train\nval: custom/images/val\nnc: 2\nnames: [widget, part]\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = datasetPath;
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("xanylabeling_xlabel");
    request.outputPath = root.filePath(QStringLiteral("converted"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeFakeXAnyExecutable(root, true));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("stagedInput")).toBool(), true);
    QCOMPARE(report.value(QStringLiteral("stagedImageCount")).toInt(), 2);
    QCOMPARE(report.value(QStringLiteral("stagedLabelCount")).toInt(), 2);
    QCOMPARE(report.value(QStringLiteral("persistedImageCount")).toInt(), 2);
    QCOMPARE(report.value(QStringLiteral("rewrittenXLabelCount")).toInt(), 2);
    const QStringList arguments = stringListFromJsonArrayForAnnotationTest(
        report.value(QStringLiteral("command")).toObject().value(QStringLiteral("arguments")).toArray());
    QVERIFY(QDir::cleanPath(argumentValueForAnnotationTest(arguments, QStringLiteral("--images")))
        != QDir::cleanPath(QDir(datasetPath).filePath(QStringLiteral("raw/custom/images"))));
    QVERIFY(QDir::cleanPath(argumentValueForAnnotationTest(arguments, QStringLiteral("--labels")))
        != QDir::cleanPath(QDir(datasetPath).filePath(QStringLiteral("raw/custom/labels"))));
    QCOMPARE(QFileInfo(argumentValueForAnnotationTest(arguments, QStringLiteral("--images"))).fileName(), QStringLiteral("images"));
    QCOMPARE(QFileInfo(argumentValueForAnnotationTest(arguments, QStringLiteral("--labels"))).fileName(), QStringLiteral("labels"));
    QVERIFY(QFileInfo::exists(QDir(request.outputPath).filePath(QStringLiteral("images/train__a.jpg"))));
    QVERIFY(QFileInfo::exists(QDir(request.outputPath).filePath(QStringLiteral("images/val__b.jpg"))));
    const QJsonObject trainLabel = readJsonObjectForAnnotationTest(QDir(request.outputPath).filePath(QStringLiteral("train__a.json")));
    const QJsonObject valLabel = readJsonObjectForAnnotationTest(QDir(request.outputPath).filePath(QStringLiteral("val__b.json")));
    QCOMPARE(trainLabel.value(QStringLiteral("imagePath")).toString(), QStringLiteral("images/train__a.jpg"));
    QCOMPARE(valLabel.value(QStringLiteral("imagePath")).toString(), QStringLiteral("images/val__b.jpg"));
}

void AnnotationIntegrationTests::xAnyCliYoloDuplicateBasenamesAcrossSplitsUseDistinctStaging()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString datasetPath = root.filePath(QStringLiteral("dataset"));
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("images/%1/a.jpg").arg(split)),
            QStringLiteral("fake image\n"));
        writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("labels/%1/a.txt").arg(split)),
            QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    }
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: [widget]\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = datasetPath;
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("xanylabeling_xlabel");
    request.outputPath = root.filePath(QStringLiteral("converted"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeFakeXAnyExecutable(root, true));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("stagedImageCount")).toInt(), 3);
    QCOMPARE(report.value(QStringLiteral("stagedLabelCount")).toInt(), 3);
    QCOMPARE(report.value(QStringLiteral("persistedImageCount")).toInt(), 3);
    QCOMPARE(report.value(QStringLiteral("rewrittenXLabelCount")).toInt(), 3);
    for (const QString& stagedName : {QStringLiteral("train__a"), QStringLiteral("val__a"), QStringLiteral("test__a")}) {
        QVERIFY(QFileInfo::exists(QDir(request.outputPath).filePath(QStringLiteral("images/%1.jpg").arg(stagedName))));
        const QJsonObject label = readJsonObjectForAnnotationTest(
            QDir(request.outputPath).filePath(QStringLiteral("%1.json").arg(stagedName)));
        QCOMPARE(label.value(QStringLiteral("imagePath")).toString(), QStringLiteral("images/%1.jpg").arg(stagedName));
    }
}

void AnnotationIntegrationTests::xAnyCliYoloMissingLabelsStageAsEmptyFiles()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString datasetPath = root.filePath(QStringLiteral("dataset"));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("images/train/negative.jpg")),
        QStringLiteral("fake image\n"));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: .\ntrain: images/train\nval: images/train\nnc: 1\nnames: [widget]\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = datasetPath;
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("xanylabeling_xlabel");
    request.outputPath = root.filePath(QStringLiteral("converted"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeLabelEchoXAnyExecutable(root));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("stagedImageCount")).toInt(), 1);
    QCOMPARE(report.value(QStringLiteral("stagedLabelCount")).toInt(), 0);
    QCOMPARE(report.value(QStringLiteral("emptyLabelCount")).toInt(), 1);
    QFile observed(QDir(request.outputPath).filePath(QStringLiteral("observed_label.txt")));
    QVERIFY(observed.open(QIODevice::ReadOnly));
    QCOMPARE(observed.readAll(), QByteArray());
}

void AnnotationIntegrationTests::xAnyCliYoloStagingStripsBomFromLabels()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString datasetPath = root.filePath(QStringLiteral("dataset"));
    QVERIFY(QDir().mkpath(QDir(datasetPath).filePath(QStringLiteral("images/train"))));
    QVERIFY(QDir().mkpath(QDir(datasetPath).filePath(QStringLiteral("labels/train"))));
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("images/train/a.jpg")),
        QStringLiteral("fake image\n"));
    QByteArray label;
    label.append(char(0xEF));
    label.append(char(0xBB));
    label.append(char(0xBF));
    label.append("0 0.5 0.5 0.25 0.25\n");
    writeBinaryFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("labels/train/a.txt")), label);
    writeTextFileForAnnotationTest(QDir(datasetPath).filePath(QStringLiteral("data.yaml")),
        QStringLiteral("path: .\ntrain: images/train\nval: images/train\nnc: 1\nnames: [widget]\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = datasetPath;
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetFormat = QStringLiteral("xanylabeling_xlabel");
    request.outputPath = root.filePath(QStringLiteral("converted"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeLabelEchoXAnyExecutable(root));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));

    QFile observed(QDir(request.outputPath).filePath(QStringLiteral("observed_label.txt")));
    QVERIFY(observed.open(QIODevice::ReadOnly));
    const QByteArray observedBytes = observed.readAll();
    QVERIFY(!observedBytes.startsWith(QByteArray::fromHex("efbbbf")));
    QVERIFY(observedBytes.startsWith("0 "));
}

void AnnotationIntegrationTests::xAnyCliXLabelToYoloPassesImagesPath()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString sourcePath = root.filePath(QStringLiteral("xlabel"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("images/sample.jpg")),
        QStringLiteral("fake image\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("sample.json")),
        QStringLiteral("{\"imagePath\":\"images/sample.jpg\",\"imageHeight\":10,\"imageWidth\":10,\"shapes\":[]}\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("classes.txt")),
        QStringLiteral("widget\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = sourcePath;
    request.sourceFormat = QStringLiteral("xanylabeling_xlabel");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted_yolo"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeFakeXAnyExecutable(root, true));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    const QStringList arguments = stringListFromJsonArrayForAnnotationTest(
        report.value(QStringLiteral("command")).toObject().value(QStringLiteral("arguments")).toArray());
    QCOMPARE(QDir::cleanPath(argumentValueForAnnotationTest(arguments, QStringLiteral("--images"))),
        QDir::cleanPath(QDir(sourcePath).filePath(QStringLiteral("images"))));
    QCOMPARE(QDir::cleanPath(argumentValueForAnnotationTest(arguments, QStringLiteral("--labels"))),
        QDir::cleanPath(sourcePath));
    QCOMPARE(QDir::cleanPath(report.value(QStringLiteral("resolvedImagesPath")).toString()),
        QDir::cleanPath(QDir(sourcePath).filePath(QStringLiteral("images"))));
}

void AnnotationIntegrationTests::xAnyCliXLabelToYoloFallsBackToImagesSubdirForBasenameImagePath()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString sourcePath = root.filePath(QStringLiteral("xlabel"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("images/sample.jpg")),
        QStringLiteral("fake image\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("sample.json")),
        QStringLiteral("{\"imagePath\":\"sample.jpg\",\"imageHeight\":10,\"imageWidth\":10,\"shapes\":[]}\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("classes.txt")),
        QStringLiteral("widget\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = sourcePath;
    request.sourceFormat = QStringLiteral("xanylabeling_xlabel");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted_yolo"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeFakeXAnyExecutable(root));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    const QStringList arguments = stringListFromJsonArrayForAnnotationTest(
        report.value(QStringLiteral("command")).toObject().value(QStringLiteral("arguments")).toArray());
    QCOMPARE(QDir::cleanPath(argumentValueForAnnotationTest(arguments, QStringLiteral("--images"))),
        QDir::cleanPath(QDir(sourcePath).filePath(QStringLiteral("images"))));
    QCOMPARE(QDir::cleanPath(report.value(QStringLiteral("resolvedImagesPath")).toString()),
        QDir::cleanPath(QDir(sourcePath).filePath(QStringLiteral("images"))));
}

void AnnotationIntegrationTests::xAnyCliXLabelToYoloHandlesUnicodeSpacePaths()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString sourcePath = root.filePath(QStringLiteral("xlabel 中文 空格"));
    const QString imagePath = QDir(sourcePath).filePath(QStringLiteral("images/样本 1.jpg"));
    writeTextFileForAnnotationTest(imagePath, QStringLiteral("fake image\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("sample.json")),
        QStringLiteral("{\"imagePath\":\"images/样本 1.jpg\",\"imageHeight\":10,\"imageWidth\":10,\"shapes\":[]}\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("classes.txt")),
        QStringLiteral("widget\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = sourcePath;
    request.sourceFormat = QStringLiteral("xanylabeling_xlabel");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted yolo 中文"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeFakeXAnyExecutable(root));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request);
    QVERIFY2(result.ok, qPrintable(result.errorMessage));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    const QStringList arguments = stringListFromJsonArrayForAnnotationTest(
        report.value(QStringLiteral("command")).toObject().value(QStringLiteral("arguments")).toArray());
    QCOMPARE(QDir::cleanPath(argumentValueForAnnotationTest(arguments, QStringLiteral("--images"))),
        QDir::cleanPath(QDir(sourcePath).filePath(QStringLiteral("images"))));
    QCOMPARE(QDir::cleanPath(argumentValueForAnnotationTest(arguments, QStringLiteral("--labels"))),
        QDir::cleanPath(sourcePath));
    QCOMPARE(QDir::cleanPath(report.value(QStringLiteral("resolvedImagesPath")).toString()),
        QDir::cleanPath(QDir(sourcePath).filePath(QStringLiteral("images"))));
}

void AnnotationIntegrationTests::xAnyCliCancellationReturnsCanceled()
{
    QTemporaryDir temp;
    QVERIFY(temp.isValid());
    const QDir root(temp.path());
    const QString sourcePath = root.filePath(QStringLiteral("xlabel"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("images/sample.jpg")),
        QStringLiteral("fake image\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("sample.json")),
        QStringLiteral("{\"imagePath\":\"images/sample.jpg\",\"imageHeight\":10,\"imageWidth\":10,\"shapes\":[]}\n"));
    writeTextFileForAnnotationTest(QDir(sourcePath).filePath(QStringLiteral("classes.txt")),
        QStringLiteral("widget\n"));

    aitrain::DatasetConversionRequest request;
    request.sourcePath = sourcePath;
    request.sourceFormat = QStringLiteral("xanylabeling_xlabel");
    request.targetFormat = QStringLiteral("yolo_detection");
    request.outputPath = root.filePath(QStringLiteral("converted_yolo"));
    request.options.insert(QStringLiteral("xAnyLabelingExecutable"), writeSlowFakeXAnyExecutable(root));
    request.options.insert(QStringLiteral("disableXAnyLabelingAutoDiscovery"), true);

    int cancelChecks = 0;
    const aitrain::DatasetConversionResult result =
        aitrain::convertDatasetWithXAnyLabelingCli(request, [&cancelChecks]() {
            ++cancelChecks;
            return cancelChecks > 1;
        });
    QVERIFY(!result.ok);
    QCOMPARE(result.errorCode, QStringLiteral("canceled"));
    QCOMPARE(result.errorMessage, QStringLiteral("Canceled by user"));
    QVERIFY(QFileInfo::exists(result.reportPath));

    const QJsonObject report = readJsonObjectForAnnotationTest(result.reportPath);
    QCOMPARE(report.value(QStringLiteral("process")).toObject().value(QStringLiteral("status")).toString(),
        QStringLiteral("canceled"));
}

QTEST_MAIN(AnnotationIntegrationTests)
#include "tst_annotation_integration.moc"
