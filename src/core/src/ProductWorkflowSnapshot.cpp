#include "aitrain/core/ProductWorkflow.h"

#include "ProductWorkflowSupport.h"
#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/OcrRecDataset.h"
#include "aitrain/core/SegmentationDataset.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMap>
#include <QRegularExpression>
#include <QSet>
#include <QTextStream>
#include <QThread>

#include <algorithm>
namespace aitrain {
using namespace workflow_detail;
namespace {
QString snapshotFileRole(const QString& relativePath, const QFileInfo& fileInfo)
{
    const QString path = QDir::fromNativeSeparators(relativePath).toLower();
    const QString name = fileInfo.fileName().toLower();
    const QString suffix = fileInfo.suffix().toLower();
    if (isImageFile(suffix)) {
        return QStringLiteral("image");
    }
    if (name == QStringLiteral("data.yaml") || name == QStringLiteral("data.yml") || suffix == QStringLiteral("yaml") || suffix == QStringLiteral("yml")) {
        return QStringLiteral("config");
    }
    if (name == QStringLiteral("dict.txt")) {
        return QStringLiteral("dict");
    }
    if (name.startsWith(QStringLiteral("rec_gt")) || name.startsWith(QStringLiteral("det_gt")) || path.contains(QStringLiteral("/rec_gt")) || path.contains(QStringLiteral("/det_gt"))) {
        return QStringLiteral("ocr_gt");
    }
    if ((path.startsWith(QStringLiteral("labels/")) || path.contains(QStringLiteral("/labels/"))) && suffix == QStringLiteral("txt")) {
        return QStringLiteral("label");
    }
    return QStringLiteral("other");
}

bool isSnapshotKeyRole(const QString& role)
{
    return role == QStringLiteral("config")
        || role == QStringLiteral("dict")
        || role == QStringLiteral("ocr_gt");
}

QJsonObject validationIssueCounts(const DatasetValidationResult& validation)
{
    QJsonObject counts;
    for (const DatasetValidationResult::Issue& issue : validation.issues) {
        counts.insert(issue.code, counts.value(issue.code).toInt() + 1);
    }
    return counts;
}

QJsonObject countYoloClasses(const QString& datasetPath)
{
    QJsonObject classCounts;
    const QDir root(datasetPath);
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir labelDir(root.filePath(QStringLiteral("labels/%1").arg(split)));
        if (!labelDir.exists()) {
            continue;
        }
        const QFileInfoList labels = labelDir.entryInfoList({QStringLiteral("*.txt")}, QDir::Files, QDir::Name);
        for (const QFileInfo& labelInfo : labels) {
            QFile file(labelInfo.absoluteFilePath());
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                continue;
            }
            while (!file.atEnd()) {
                const QString line = QString::fromUtf8(file.readLine()).trimmed();
                if (line.isEmpty()) {
                    continue;
                }
                const QString classId = line.section(QLatin1Char(' '), 0, 0).trimmed();
                if (!classId.isEmpty()) {
                    classCounts.insert(classId, classCounts.value(classId).toInt() + 1);
                }
            }
        }
    }
    return classCounts;
}

QJsonObject countImageSplits(const QString& datasetPath)
{
    QJsonObject splits;
    const QDir root(datasetPath);
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir imageDir(root.filePath(QStringLiteral("images/%1").arg(split)));
        int count = 0;
        if (imageDir.exists()) {
            for (const QString& filter : imageNameFilters()) {
                count += imageDir.entryInfoList({filter}, QDir::Files).size();
            }
        }
        splits.insert(split, count);
    }
    return splits;
}

QString classDistributionCsv(const QJsonObject& counts)
{
    QString csv = QStringLiteral("class,count\n");
    const QStringList keys = counts.keys();
    for (const QString& key : keys) {
        csv.append(QStringLiteral("%1,%2\n").arg(key).arg(counts.value(key).toInt()));
    }
    return csv;
}
} // namespace
WorkflowResult createDatasetSnapshotReport(const QString& datasetPath, const QString& outputPath, const QString& format, const QJsonObject& options)
{
    const QDir root(datasetPath);
    if (!root.exists()) {
        return failedResult(QStringLiteral("Dataset directory does not exist: %1").arg(datasetPath));
    }
    const int maxFiles = options.value(QStringLiteral("maxFiles")).toInt(20000);
    const QFileInfoList files = collectFilesRecursive(datasetPath, maxFiles);
    QJsonArray fileArray;
    QJsonArray keyFileArray;
    QJsonObject roleCounts;
    QCryptographicHash manifestHash(QCryptographicHash::Sha256);
    qint64 totalBytes = 0;
    QString error;
    for (const QFileInfo& fileInfo : files) {
        qint64 fileSize = 0;
        const QByteArray hash = fileSha256(fileInfo.absoluteFilePath(), &fileSize, &error);
        if (!error.isEmpty()) {
            return failedResult(error);
        }
        totalBytes += fileSize;
        const QString relativePath = cleanRelativePath(root, fileInfo.absoluteFilePath());
        const QString role = snapshotFileRole(relativePath, fileInfo);
        QJsonObject fileObject;
        fileObject.insert(QStringLiteral("path"), relativePath);
        fileObject.insert(QStringLiteral("role"), role);
        fileObject.insert(QStringLiteral("size"), QString::number(fileSize));
        fileObject.insert(QStringLiteral("mtime"), fileInfo.lastModified().toUTC().toString(Qt::ISODateWithMs));
        fileObject.insert(QStringLiteral("sha256"), QString::fromLatin1(hash));
        fileArray.append(fileObject);
        manifestHash.addData(relativePath.toUtf8());
        manifestHash.addData("\0", 1);
        manifestHash.addData(hash);
        manifestHash.addData("\0", 1);
        roleCounts.insert(role, roleCounts.value(role).toInt() + 1);
        if (isSnapshotKeyRole(role)) {
            QJsonObject keyFile;
            keyFile.insert(QStringLiteral("path"), relativePath);
            keyFile.insert(QStringLiteral("role"), role);
            keyFile.insert(QStringLiteral("sha256"), QString::fromLatin1(hash));
            keyFileArray.append(keyFile);
        }
    }

    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 1);
    manifest.insert(QStringLiteral("kind"), QStringLiteral("dataset_snapshot"));
    manifest.insert(QStringLiteral("createdAt"), nowIso());
    manifest.insert(QStringLiteral("datasetPath"), datasetPath);
    manifest.insert(QStringLiteral("format"), format);
    manifest.insert(QStringLiteral("fileCount"), files.size());
    manifest.insert(QStringLiteral("totalBytes"), QString::number(totalBytes));
    manifest.insert(QStringLiteral("contentHash"), QString::fromLatin1(manifestHash.result().toHex()));
    manifest.insert(QStringLiteral("splits"), countImageSplits(datasetPath));
    manifest.insert(QStringLiteral("roleCounts"), roleCounts);
    manifest.insert(QStringLiteral("keyFiles"), keyFileArray);
    manifest.insert(QStringLiteral("imageCount"), roleCounts.value(QStringLiteral("image")).toInt());
    manifest.insert(QStringLiteral("labelCount"), roleCounts.value(QStringLiteral("label")).toInt());
    manifest.insert(QStringLiteral("files"), fileArray);

    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("dataset_snapshot_manifest.json"));
    if (!writeJsonFile(reportPath, manifest, &error)) {
        return failedResult(error);
    }
    manifest.insert(QStringLiteral("manifestPath"), reportPath);
    return resultFromReport(reportPath, manifest);
}
} // namespace aitrain
