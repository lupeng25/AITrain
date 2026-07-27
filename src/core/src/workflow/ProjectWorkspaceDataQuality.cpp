#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/dataset/BuiltinDatasetDrivers.h"
#include "aitrain/core/SemanticMask.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSet>

#include <algorithm>

namespace aitrain {
namespace {

struct VerifiedSnapshot final {
    DatasetSnapshotRecord record;
    QString rootPath;
    QJsonObject manifest;
    QHash<QString, QJsonObject> files;
};

struct QualityAnalysis final {
    QJsonArray issues;
    QJsonObject summary;
};

bool canceled(const aitrain::CancellationCallback& cancellation)
{
    return aitrain::isCancellationRequested(cancellation);
}

Failure makeFailure(FailureCode code, const QString& message, const QString& action = {})
{
    const QString suggestedAction = action.trimmed().isEmpty()
        ? QStringLiteral("查看该步骤的 Evidence 和 Artifact 完整性后重新运行数据质量 Workflow。")
        : action;
    return {code, message, suggestedAction, QDateTime::currentDateTimeUtc()};
}

bool safeRelative(const QString& path)
{
    const QString clean = QDir::cleanPath(path);
    return !clean.isEmpty() && !QDir::isAbsolutePath(clean) && clean != QStringLiteral("..")
        && !clean.startsWith(QStringLiteral("../"));
}

bool readAndHash(const QString& path, QByteArray* bytes, QString* sha256,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("quality.snapshot.file_unreadable:%1").arg(path);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    QByteArray content;
    while (!file.atEnd()) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("quality.canceled");
            return false;
        }
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("quality.snapshot.file_read_failed:%1").arg(path);
            return false;
        }
        hash.addData(block);
        if (bytes) content.append(block);
    }
    if (bytes) *bytes = content;
    if (sha256) *sha256 = QString::fromLatin1(hash.result().toHex());
    return true;
}

bool loadVerifiedSnapshot(ProjectStore* storage, ArtifactStore* artifacts,
    const SnapshotId& snapshotId, VerifiedSnapshot* result,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    DatasetSnapshotRecord record;
    ArtifactSnapshot artifact;
    if (!storage->datasetSnapshot(snapshotId, &record, error)
        || !storage->artifact(record.artifactId, &artifact, error)
        || artifact.kind != QStringLiteral("dataset_snapshot")) {
        if (error && error->isEmpty()) *error = QStringLiteral("quality.snapshot.artifact_invalid");
        return false;
    }
    VerifiedArtifactDirectory directory;
    if (!artifacts->openVerified(
            artifact, &directory, nullptr, error)) return false;
    const auto manifestFile = std::find_if(artifact.files.cbegin(), artifact.files.cend(), [](const ArtifactFileSnapshot& file) {
        return file.relativePath == QStringLiteral("dataset_snapshot.json");
    });
    const auto verifiedManifest = std::find_if(directory.files.cbegin(),
        directory.files.cend(), [](const VerifiedArtifactFile& file) {
            return file.relativePath == QStringLiteral("dataset_snapshot.json");
        });
    QByteArray manifestBytes;
    QString manifestHash;
    if (manifestFile == artifact.files.cend()
        || verifiedManifest == directory.files.cend()
        || !readAndHash(verifiedManifest->absolutePath,
            &manifestBytes, &manifestHash, cancellation, error)
        || manifestHash != manifestFile->sha256 || manifestHash != record.manifestSha256
        || manifestBytes.size() != manifestFile->byteCount) {
        if (error && error->isEmpty()) *error = QStringLiteral("quality.snapshot.manifest_integrity_failed");
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(manifestBytes, &parseError);
    const QJsonObject manifest = document.object();
    if (parseError.error != QJsonParseError::NoError || !document.isObject()
        || manifest.value(QStringLiteral("schemaVersion")).toInt() != 2
        || !manifest.value(QStringLiteral("complete")).toBool()
        || manifest.value(QStringLiteral("snapshotId")).toString() != record.id.toString()
        || manifest.value(QStringLiteral("datasetFormat")).toString() != record.datasetFormat
        || manifest.value(QStringLiteral("rootHash")).toString() != record.rootHash) {
        if (error) *error = QStringLiteral("quality.snapshot.manifest_contract_invalid");
        return false;
    }
    QHash<QString, QJsonObject> declaredFiles;
    for (const QJsonValue& value : manifest.value(QStringLiteral("files")).toArray()) {
        const QJsonObject item = value.toObject();
        const QString relative = QDir::cleanPath(item.value(QStringLiteral("relativePath")).toString());
        if (!safeRelative(relative) || declaredFiles.contains(relative.toCaseFolded())
            || !isSha256Hex(item.value(QStringLiteral("sha256")).toString())) {
            if (error) *error = QStringLiteral("quality.snapshot.file_manifest_invalid:%1").arg(relative);
            return false;
        }
        declaredFiles.insert(relative.toCaseFolded(), item);
    }
    if (declaredFiles.size() != record.fileCount) {
        if (error) *error = QStringLiteral("quality.snapshot.file_count_mismatch");
        return false;
    }
    // rootPath 仅作临时读取 locator；每个文件必须重新匹配 committed manifest。
    // Workflow 参数、问题清单和报告均不持久化该裸目录。
    const QString rootPath = directory.absolutePath;
    const QDir source(rootPath);
    if (!source.exists()) {
        if (error) *error = QStringLiteral("quality.snapshot.source_unavailable");
        return false;
    }
    for (auto it = declaredFiles.cbegin(); it != declaredFiles.cend(); ++it) {
        const QJsonObject declared = it.value();
        const QString relative = declared.value(QStringLiteral("relativePath")).toString();
        const QString absolute = source.filePath(relative);
        QString actualHash;
        if (!readAndHash(absolute, nullptr, &actualHash, cancellation, error)
            || QFileInfo(absolute).size() != declared.value(QStringLiteral("bytes")).toString().toLongLong()
            || actualHash != declared.value(QStringLiteral("sha256")).toString()) {
            if (error && error->isEmpty()) *error = QStringLiteral("quality.snapshot.source_changed:%1").arg(relative);
            return false;
        }
    }
    result->record = record;
    result->rootPath = rootPath;
    result->manifest = manifest;
    result->files = declaredFiles;
    return true;
}

QJsonObject qualityIssue(const QString& code, const QString& severity, const QString& samplePath,
    const QString& sourcePath, int line, const QString& message, const QString& action)
{
    return {{QStringLiteral("code"), code}, {QStringLiteral("severity"), severity},
        {QStringLiteral("sampleRelativePath"), samplePath}, {QStringLiteral("sourceRelativePath"), sourcePath},
        {QStringLiteral("line"), line}, {QStringLiteral("message"), message},
        {QStringLiteral("suggestedAction"), action}};
}

QStringList snapshotRelativeFiles(const VerifiedSnapshot& snapshot)
{
    QStringList files;
    files.reserve(snapshot.files.size());
    for (auto it = snapshot.files.cbegin(); it != snapshot.files.cend(); ++it) {
        files.append(it.value().value(QStringLiteral("relativePath")).toString());
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool isImageRelativePath(const QString& path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == QStringLiteral("jpg") || suffix == QStringLiteral("jpeg")
        || suffix == QStringLiteral("png") || suffix == QStringLiteral("bmp")
        || suffix == QStringLiteral("tif") || suffix == QStringLiteral("tiff");
}

double polygonArea(const QVector<double>& coordinates)
{
    if (coordinates.size() < 6 || coordinates.size() % 2 != 0) return 0.0;
    double twiceArea = 0.0;
    const int pointCount = coordinates.size() / 2;
    for (int index = 0; index < pointCount; ++index) {
        const int next = (index + 1) % pointCount;
        twiceArea += coordinates.at(index * 2) * coordinates.at(next * 2 + 1)
            - coordinates.at(next * 2) * coordinates.at(index * 2 + 1);
    }
    return qAbs(twiceArea) * 0.5;
}

bool parseYoloCoordinates(const QList<QByteArray>& fields, QVector<double>* coordinates)
{
    if (!coordinates || fields.size() < 4) return false;
    coordinates->clear();
    for (int index = 1; index < fields.size(); ++index) {
        bool ok = false;
        const double value = fields.at(index).toDouble(&ok);
        if (!ok) return false;
        coordinates->append(value);
    }
    return true;
}

QString yoloImageForLabel(const QString& labelPath, const QHash<QString, QJsonObject>& files)
{
    QString stem = QDir::cleanPath(labelPath);
    stem.replace(QStringLiteral("labels/"), QStringLiteral("images/"));
    stem = QFileInfo(stem).path() + QLatin1Char('/') + QFileInfo(stem).completeBaseName();
    const QStringList suffixes{QStringLiteral(".jpg"), QStringLiteral(".jpeg"), QStringLiteral(".png"),
        QStringLiteral(".bmp"), QStringLiteral(".tif"), QStringLiteral(".tiff")};
    for (const QString& suffix : suffixes) {
        const QString candidate = stem + suffix;
        if (files.contains(candidate.toCaseFolded())) return candidate;
    }
    return {};
}

bool analyzeYoloDetection(const VerifiedSnapshot& snapshot, const QJsonObject& options,
    QualityAnalysis* analysis, const aitrain::CancellationCallback& cancellation, QString* error)
{
    const double minimumArea = options.value(QStringLiteral("minimumNormalizedBoxArea")).toDouble(0.01);
    int labelCount = 0;
    int boxCount = 0;
    for (auto it = snapshot.files.cbegin(); it != snapshot.files.cend(); ++it) {
        const QString relative = it.value().value(QStringLiteral("relativePath")).toString();
        if (!relative.startsWith(QStringLiteral("labels/")) || !relative.endsWith(QStringLiteral(".txt"))) continue;
        ++labelCount;
        QByteArray bytes;
        if (!readAndHash(QDir(snapshot.rootPath).filePath(relative), &bytes, nullptr, cancellation, error)) return false;
        const QList<QByteArray> lines = bytes.split('\n');
        int sampleBoxes = 0;
        for (int index = 0; index < lines.size(); ++index) {
            if (canceled(cancellation)) {
                if (error) *error = QStringLiteral("quality.canceled");
                return false;
            }
            const QList<QByteArray> fields = lines.at(index).simplified().split(' ');
            if (fields.size() < 5 || lines.at(index).trimmed().isEmpty()) continue;
            bool widthOk = false;
            bool heightOk = false;
            const double width = fields.at(3).toDouble(&widthOk);
            const double height = fields.at(4).toDouble(&heightOk);
            if (!widthOk || !heightOk) continue;
            ++sampleBoxes;
            ++boxCount;
            if (width * height < minimumArea) {
                analysis->issues.append(qualityIssue(QStringLiteral("quality.yolo_detection.bbox_too_small"),
                    QStringLiteral("warning"), yoloImageForLabel(relative, snapshot.files), relative, index + 1,
                    QStringLiteral("标注框归一化面积低于质量阈值。"), QStringLiteral("review_bbox")));
            }
        }
        if (sampleBoxes == 0) {
            analysis->issues.append(qualityIssue(QStringLiteral("quality.yolo_detection.empty_label"),
                QStringLiteral("warning"), yoloImageForLabel(relative, snapshot.files), relative, 0,
                QStringLiteral("标签文件没有有效目标。"), QStringLiteral("review_empty_label")));
        }
    }
    analysis->summary = {{QStringLiteral("format"), QStringLiteral("yolo_detection")},
        {QStringLiteral("labelFileCount"), labelCount}, {QStringLiteral("boxCount"), boxCount},
        {QStringLiteral("issueCount"), analysis->issues.size()},
        {QStringLiteral("minimumNormalizedBoxArea"), minimumArea}};
    return true;
}

bool analyzeYoloPolygon(const VerifiedSnapshot& snapshot, const QJsonObject& options,
    const QString& format, const QString& thresholdOption, const QString& issueCode,
    const QString& message, const QString& action, QualityAnalysis* analysis,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    const double minimumArea = qMax(0.0, options.value(thresholdOption).toDouble(0.01));
    int labelCount = 0;
    int polygonCount = 0;
    const QStringList files = snapshotRelativeFiles(snapshot);
    for (const QString& relative : files) {
        if (!relative.startsWith(QStringLiteral("labels/")) || !relative.endsWith(QStringLiteral(".txt"))) continue;
        ++labelCount;
        QByteArray bytes;
        if (!readAndHash(QDir(snapshot.rootPath).filePath(relative), &bytes, nullptr, cancellation, error)) return false;
        const QList<QByteArray> lines = bytes.split('\n');
        for (int index = 0; index < lines.size(); ++index) {
            if (canceled(cancellation)) {
                if (error) *error = QStringLiteral("quality.canceled");
                return false;
            }
            const QByteArray row = lines.at(index).trimmed();
            if (row.isEmpty()) continue;
            const QList<QByteArray> fields = row.simplified().split(' ');
            QVector<double> coordinates;
            if (!parseYoloCoordinates(fields, &coordinates)) continue;
            ++polygonCount;
            if (polygonArea(coordinates) < minimumArea) {
                analysis->issues.append(qualityIssue(issueCode, QStringLiteral("warning"),
                    yoloImageForLabel(relative, snapshot.files), relative, index + 1, message, action));
            }
        }
    }
    analysis->summary = {{QStringLiteral("format"), format},
        {QStringLiteral("labelFileCount"), labelCount}, {QStringLiteral("polygonCount"), polygonCount},
        {QStringLiteral("issueCount"), analysis->issues.size()},
        {QStringLiteral("minimumNormalizedPolygonArea"), minimumArea}};
    return true;
}

QString semanticImageForMask(const QString& maskPath, const QHash<QString, QJsonObject>& files)
{
    QString stem = QDir::cleanPath(maskPath);
    stem.replace(QStringLiteral("masks/"), QStringLiteral("images/"));
    stem = QFileInfo(stem).path() + QLatin1Char('/') + QFileInfo(stem).completeBaseName();
    const QStringList suffixes{QStringLiteral(".jpg"), QStringLiteral(".jpeg"), QStringLiteral(".png"),
        QStringLiteral(".bmp"), QStringLiteral(".tif"), QStringLiteral(".tiff")};
    for (const QString& suffix : suffixes) {
        const QString candidate = stem + suffix;
        if (files.contains(candidate.toCaseFolded())) return candidate;
    }
    return {};
}

bool analyzeSemanticMask(const VerifiedSnapshot& snapshot, const QJsonObject& options,
    QualityAnalysis* analysis, const aitrain::CancellationCallback& cancellation, QString* error)
{
    const int ignoreIndex = options.value(QStringLiteral("ignoreIndex")).toInt(255);
    int maskCount = 0;
    int foregroundMaskCount = 0;
    const QStringList files = snapshotRelativeFiles(snapshot);
    for (const QString& relative : files) {
        if (!relative.startsWith(QStringLiteral("masks/")) || !relative.endsWith(QStringLiteral(".png"))) continue;
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("quality.canceled");
            return false;
        }
        ++maskCount;
        const QImage mask(QDir(snapshot.rootPath).filePath(relative));
        bool hasForeground = false;
        for (int y = 0; y < mask.height() && !hasForeground; ++y) {
            for (int x = 0; x < mask.width(); ++x) {
                const int classId = aitrain::semanticMaskClassId(mask, x, y);
                if (classId != 0 && classId != ignoreIndex) {
                    hasForeground = true;
                    break;
                }
            }
        }
        if (hasForeground) {
            ++foregroundMaskCount;
        } else {
            analysis->issues.append(qualityIssue(QStringLiteral("quality.semantic_mask.no_foreground_pixels"),
                QStringLiteral("warning"), semanticImageForMask(relative, snapshot.files), relative, 0,
                QStringLiteral("Mask 仅包含背景或 ignore 像素。"), QStringLiteral("review_background_only_mask")));
        }
    }
    analysis->summary = {{QStringLiteral("format"), QStringLiteral("semantic_segmentation_mask")},
        {QStringLiteral("maskCount"), maskCount}, {QStringLiteral("foregroundMaskCount"), foregroundMaskCount},
        {QStringLiteral("issueCount"), analysis->issues.size()}, {QStringLiteral("ignoreIndex"), ignoreIndex}};
    return true;
}

bool analyzeAnomalyFolder(const VerifiedSnapshot& snapshot, QualityAnalysis* analysis,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    int normalCount = 0;
    int anomalyCount = 0;
    QString firstNormal;
    const QStringList files = snapshotRelativeFiles(snapshot);
    for (const QString& relative : files) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("quality.canceled");
            return false;
        }
        if (!isImageRelativePath(relative)) continue;
        const QStringList parts = QDir::cleanPath(relative).split(QLatin1Char('/'));
        if (parts.size() < 3) continue;
        const bool normal = (parts.at(0) == QStringLiteral("train")
                || parts.at(0) == QStringLiteral("val") || parts.at(0) == QStringLiteral("test"))
            && parts.at(1).compare(QStringLiteral("good"), Qt::CaseInsensitive) == 0;
        const bool anomaly = (parts.at(0) == QStringLiteral("val") || parts.at(0) == QStringLiteral("test"))
            && parts.at(1).compare(QStringLiteral("good"), Qt::CaseInsensitive) != 0;
        if (normal) {
            ++normalCount;
            if (firstNormal.isEmpty()) firstNormal = relative;
        } else if (anomaly) {
            ++anomalyCount;
        }
    }
    if (anomalyCount == 0 && !firstNormal.isEmpty()) {
        analysis->issues.append(qualityIssue(QStringLiteral("quality.anomaly_folder.good_only_dataset"),
            QStringLiteral("warning"), firstNormal, firstNormal, 0,
            QStringLiteral("Snapshot 仅包含正常样本，无法形成异常样本评估结论。"),
            QStringLiteral("collect_anomaly_evaluation_samples")));
    }
    analysis->summary = {{QStringLiteral("format"), QStringLiteral("anomaly_folder")},
        {QStringLiteral("normalImageCount"), normalCount}, {QStringLiteral("anomalyImageCount"), anomalyCount},
        {QStringLiteral("evaluationLimited"), anomalyCount == 0},
        {QStringLiteral("issueCount"), analysis->issues.size()}};
    return true;
}

bool analyzePaddleOcrDet(const VerifiedSnapshot& snapshot, const QJsonObject& options,
    QualityAnalysis* analysis, const aitrain::CancellationCallback& cancellation, QString* error)
{
    const double minimumArea = qMax(0.0, options.value(QStringLiteral("minimumTextPolygonAreaPixels")).toDouble(16.0));
    int sampleCount = 0;
    int polygonCount = 0;
    const QStringList files = snapshotRelativeFiles(snapshot);
    for (const QString& relative : files) {
        const QString fileName = QFileInfo(relative).fileName();
        if (!fileName.startsWith(QStringLiteral("det_gt")) || !fileName.endsWith(QStringLiteral(".txt"))) continue;
        QByteArray bytes;
        if (!readAndHash(QDir(snapshot.rootPath).filePath(relative), &bytes, nullptr, cancellation, error)) return false;
        const QList<QByteArray> lines = bytes.split('\n');
        for (int index = 0; index < lines.size(); ++index) {
            if (canceled(cancellation)) {
                if (error) *error = QStringLiteral("quality.canceled");
                return false;
            }
            const QByteArray row = lines.at(index).trimmed();
            if (row.isEmpty()) continue;
            const int tab = row.indexOf('\t');
            if (tab <= 0) continue;
            ++sampleCount;
            const QString image = QDir::cleanPath(QString::fromUtf8(row.left(tab)));
            QJsonParseError parseError;
            const QJsonDocument document = QJsonDocument::fromJson(row.mid(tab + 1), &parseError);
            if (parseError.error != QJsonParseError::NoError || !document.isArray()) continue;
            const QJsonArray boxes = document.array();
            for (const QJsonValue& value : boxes) {
                QVector<double> coordinates;
                for (const QJsonValue& pointValue : value.toObject().value(QStringLiteral("points")).toArray()) {
                    const QJsonArray point = pointValue.toArray();
                    if (point.size() >= 2) {
                        coordinates.append(point.at(0).toDouble());
                        coordinates.append(point.at(1).toDouble());
                    }
                }
                ++polygonCount;
                if (polygonArea(coordinates) < minimumArea) {
                    analysis->issues.append(qualityIssue(QStringLiteral("quality.paddleocr_det.text_polygon_too_small"),
                        QStringLiteral("warning"), image, relative, index + 1,
                        QStringLiteral("文本多边形像素面积低于质量阈值。"), QStringLiteral("review_text_polygon")));
                }
            }
        }
    }
    analysis->summary = {{QStringLiteral("format"), QStringLiteral("paddleocr_det")},
        {QStringLiteral("sampleCount"), sampleCount}, {QStringLiteral("polygonCount"), polygonCount},
        {QStringLiteral("issueCount"), analysis->issues.size()},
        {QStringLiteral("minimumTextPolygonAreaPixels"), minimumArea}};
    return true;
}

bool analyzePaddleOcrRec(const VerifiedSnapshot& snapshot, const QJsonObject& options,
    QualityAnalysis* analysis, const aitrain::CancellationCallback& cancellation, QString* error)
{
    const int maximumLength = qMax(1, options.value(QStringLiteral("maximumLabelLength")).toInt(32));
    QSet<QString> seenImages;
    int sampleCount = 0;
    for (auto it = snapshot.files.cbegin(); it != snapshot.files.cend(); ++it) {
        const QString relative = it.value().value(QStringLiteral("relativePath")).toString();
        const QString fileName = QFileInfo(relative).fileName();
        if (!fileName.startsWith(QStringLiteral("rec_gt")) || !fileName.endsWith(QStringLiteral(".txt"))) continue;
        QByteArray bytes;
        if (!readAndHash(QDir(snapshot.rootPath).filePath(relative), &bytes, nullptr, cancellation, error)) return false;
        const QList<QByteArray> lines = bytes.split('\n');
        for (int index = 0; index < lines.size(); ++index) {
            if (canceled(cancellation)) {
                if (error) *error = QStringLiteral("quality.canceled");
                return false;
            }
            const QByteArray row = lines.at(index).trimmed();
            if (row.isEmpty()) continue;
            const int tab = row.indexOf('\t');
            if (tab <= 0) continue;
            ++sampleCount;
            const QString image = QDir::cleanPath(QString::fromUtf8(row.left(tab)));
            const QString label = QString::fromUtf8(row.mid(tab + 1));
            if (seenImages.contains(image.toCaseFolded())) {
                analysis->issues.append(qualityIssue(QStringLiteral("quality.paddleocr_rec.duplicate_image_reference"),
                    QStringLiteral("warning"), image, relative, index + 1,
                    QStringLiteral("同一图像在识别标签清单中重复出现。"), QStringLiteral("review_duplicate_reference")));
            }
            seenImages.insert(image.toCaseFolded());
            if (label.size() > maximumLength) {
                analysis->issues.append(qualityIssue(QStringLiteral("quality.paddleocr_rec.label_too_long"),
                    QStringLiteral("warning"), image, relative, index + 1,
                    QStringLiteral("识别文本长度超过质量阈值。"), QStringLiteral("review_transcription")));
            }
        }
    }
    analysis->summary = {{QStringLiteral("format"), QStringLiteral("paddleocr_rec")},
        {QStringLiteral("sampleCount"), sampleCount}, {QStringLiteral("issueCount"), analysis->issues.size()},
        {QStringLiteral("maximumLabelLength"), maximumLength}};
    return true;
}

bool analyzeQuality(const VerifiedSnapshot& snapshot, const QJsonObject& options,
    QualityAnalysis* analysis, const aitrain::CancellationCallback& cancellation, QString* error)
{
    if (snapshot.record.datasetFormat == QStringLiteral("yolo_detection")) {
        return analyzeYoloDetection(snapshot, options, analysis, cancellation, error);
    }
    if (snapshot.record.datasetFormat == QStringLiteral("paddleocr_rec")) {
        return analyzePaddleOcrRec(snapshot, options, analysis, cancellation, error);
    }
    if (snapshot.record.datasetFormat == QStringLiteral("yolo_segmentation")) {
        return analyzeYoloPolygon(snapshot, options, QStringLiteral("yolo_segmentation"),
            QStringLiteral("minimumNormalizedPolygonArea"),
            QStringLiteral("quality.yolo_segmentation.polygon_too_small"),
            QStringLiteral("分割多边形归一化面积低于质量阈值。"), QStringLiteral("review_polygon"),
            analysis, cancellation, error);
    }
    if (snapshot.record.datasetFormat == QStringLiteral("yolo_obb")) {
        return analyzeYoloPolygon(snapshot, options, QStringLiteral("yolo_obb"),
            QStringLiteral("minimumNormalizedObbArea"),
            QStringLiteral("quality.yolo_obb.quad_too_small"),
            QStringLiteral("OBB 四边形归一化面积低于质量阈值。"), QStringLiteral("review_obb_quad"),
            analysis, cancellation, error);
    }
    if (snapshot.record.datasetFormat == QStringLiteral("semantic_segmentation_mask")) {
        return analyzeSemanticMask(snapshot, options, analysis, cancellation, error);
    }
    if (snapshot.record.datasetFormat == QStringLiteral("anomaly_folder")) {
        return analyzeAnomalyFolder(snapshot, analysis, cancellation, error);
    }
    if (snapshot.record.datasetFormat == QStringLiteral("paddleocr_det")) {
        return analyzePaddleOcrDet(snapshot, options, analysis, cancellation, error);
    }
    if (error) *error = QStringLiteral("quality.format_not_implemented:%1").arg(snapshot.record.datasetFormat);
    return false;
}

bool writeFile(const QString& path, const QByteArray& bytes, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) *error = QStringLiteral("quality.artifact_directory_failed:%1").arg(path);
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("quality.artifact_write_failed:%1").arg(path);
        return false;
    }
    return true;
}

bool commitFiles(ArtifactStore* artifacts, ProjectStore* storage, const TaskId& taskId,
    const QString& kind, const QVector<QPair<QString, QByteArray>>& files, ArtifactId* artifactId,
    QString* error, const aitrain::CancellationCallback& cancellation)
{
    ArtifactId id;
    QString staging;
    if (!artifacts->begin(taskId, kind, &id, &staging, error)) return false;
    const auto abort = [&]() { QString ignored; artifacts->abort(staging, &ignored); };
    for (const auto& file : files) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("quality.canceled");
            abort();
            return false;
        }
        if (!safeRelative(file.first) || !writeFile(QDir(staging).filePath(file.first), file.second, error)) {
            abort();
            return false;
        }
    }
    QString committedPath;
    bool commitCanceled = false;
    if (!artifacts->commit(id, taskId, kind, staging, storage, &committedPath, error,
            cancellation, &commitCanceled)) {
        if (QFileInfo::exists(staging)) abort();
        return false;
    }
    *artifactId = id;
    return true;
}

QByteArray jsonBytes(const QJsonObject& object)
{
    return QJsonDocument(object).toJson(QJsonDocument::Indented);
}

TaskState taskStateForResult(const WorkflowRunExecutionResult& result)
{
    if (result.state == WorkflowStepState::Succeeded) return TaskState::Succeeded;
    if (result.state == WorkflowStepState::Canceled) return TaskState::Canceled;
    return TaskState::Failed;
}

Failure terminalFailure(const WorkflowRunExecutionResult& result)
{
    if (result.state == WorkflowStepState::Succeeded) return {};
    if (result.failure.isFailure()) {
        Failure normalized = result.failure;
        if (normalized.suggestedAction.trimmed().isEmpty()) {
            normalized.suggestedAction = result.state == WorkflowStepState::Canceled
                ? QStringLiteral("可从同一 Snapshot 重新运行数据质量 Workflow。")
                : QStringLiteral("查看失败步骤 Evidence，修复输入或 Artifact 后重新运行。");
        }
        if (!normalized.occurredAt.isValid()) normalized.occurredAt = QDateTime::currentDateTimeUtc();
        return normalized;
    }
    return makeFailure(result.state == WorkflowStepState::Canceled ? FailureCode::Canceled : FailureCode::InternalError,
        QStringLiteral("数据质量 Workflow 未返回完整终态。"));
}

} // namespace

bool ProjectWorkspace::runDataQualityWorkflow(const TaskId& taskId,
    const DataQualityWorkflowRequest& request, DataQualityWorkflowResult* result,
    QString* error, const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !request.snapshotId.isValid() || !result) {
        if (error) *error = QStringLiteral("运行数据质量 Workflow 需要已打开工作区、任务、快照和输出对象。");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("数据质量 Workflow 必须依附运行中任务。");
        return false;
    }
    DatasetSnapshotRecord snapshotRecord;
    ArtifactSnapshot snapshotArtifact;
    if (!storage_.datasetSnapshot(request.snapshotId, &snapshotRecord, error)
        || !storage_.artifact(snapshotRecord.artifactId, &snapshotArtifact, error)
        || snapshotArtifact.kind != QStringLiteral("dataset_snapshot")) return false;

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("dataset_quality");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    const QStringList kinds{QStringLiteral("ValidateSnapshot"), QStringLiteral("AnalyzeQuality"),
        QStringLiteral("ProduceRepairManifest"), QStringLiteral("RenderQualityReport")};
    QVector<WorkflowStepSnapshot> steps;
    const QJsonObject parameters{{QStringLiteral("datasetSnapshotId"), request.snapshotId.toString()},
        {QStringLiteral("datasetId"), request.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), request.datasetVersionId.toString()},
        {QStringLiteral("snapshotArtifactId"), request.snapshotArtifactId.toString()},
        {QStringLiteral("datasetFormat"), snapshotRecord.datasetFormat},
        {QStringLiteral("qualityOptions"), request.options}};
    for (int index = 0; index < kinds.size(); ++index) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = index;
        step.kind = kinds.at(index);
        step.backend = QStringLiteral("builtin_dataset_quality");
        step.parameterSummary = parameters;
        if (index == 0) step.inputArtifactId = snapshotRecord.artifactId;
        steps.append(step);
    }
    WorkflowInputBinding input;
    input.workflowRunId = workflow.id;
    input.role = QStringLiteral("dataset_snapshot");
    input.sourceArtifactId = snapshotRecord.artifactId;
    input.sourceTaskId = snapshotRecord.taskId;
    input.sourceArtifactKind = snapshotArtifact.kind;
    input.datasetId = snapshotRecord.datasetId;
    input.datasetSnapshotId = snapshotRecord.id;
    input.datasetVersionId = snapshotRecord.datasetVersionId;
    input.manifestSha256 = snapshotRecord.manifestSha256;
    input.rootHash = snapshotRecord.rootHash;
    if (!storage_.createWorkflowRunWithInput(workflow, steps, input, error)) return false;

    VerifiedSnapshot verifiedSnapshot;
    QualityAnalysis analysis;
    QJsonObject repairManifest;
    QJsonObject qualityReport;
    QHash<QString, ArtifactId> outputs;
    QString executionError;
    DatasetDriverRegistry drivers;
    if (!registerBuiltinDatasetDrivers(&drivers, &executionError)) {
        if (error) *error = executionError;
        return false;
    }
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult runResult;
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) -> WorkflowStepExecutionResult {
        if (canceled(stepCancellation)) {
            return {WorkflowStepState::Canceled, {}, makeFailure(FailureCode::Canceled,
                QStringLiteral("数据质量步骤已取消。"), QStringLiteral("可从同一 Snapshot 重新运行质量检查。"))};
        }
        ArtifactId output;
        if (step.kind == QStringLiteral("ValidateSnapshot")) {
            if ((request.datasetId.isValid() && request.datasetId != snapshotRecord.datasetId)
                || (request.datasetVersionId.isValid()
                    && request.datasetVersionId != snapshotRecord.datasetVersionId)
                || (request.snapshotArtifactId.isValid()
                    && request.snapshotArtifactId != snapshotRecord.artifactId)) {
                return {WorkflowStepState::Failed, {}, makeFailure(FailureCode::ArtifactIncompatible,
                    QStringLiteral("quality.snapshot_identity_mismatch"),
                    QStringLiteral("从  数据集快照记录重新复制 DatasetId、DatasetVersionId、SnapshotId 和 ArtifactId。"))};
            }
            if (!loadVerifiedSnapshot(&storage_, artifactStore_.get(), request.snapshotId,
                    &verifiedSnapshot, stepCancellation, &executionError)) {
                const bool wasCanceled = executionError == QStringLiteral("quality.canceled");
                return {wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    makeFailure(wasCanceled ? FailureCode::Canceled : FailureCode::ArtifactIncompatible,
                        executionError, QStringLiteral("重新创建完整 Dataset Snapshot 后再运行质量检查。"))};
            }
            const DatasetDriver* driver = drivers.driverForFormat(verifiedSnapshot.record.datasetFormat);
            DatasetInspection inspection;
            DatasetDriverValidationResult validation;
            DatasetOperationContext context;
            context.isCancellationRequested = stepCancellation;
            if (!driver || !driver->inspect(verifiedSnapshot.rootPath, verifiedSnapshot.record.datasetFormat,
                    &inspection, context, &executionError)
                || !driver->validate(inspection, &validation, context, &executionError)
                || !validation.valid) {
                if (executionError.isEmpty()) executionError = QStringLiteral("quality.snapshot.driver_validation_failed");
                return {WorkflowStepState::Failed, {}, makeFailure(FailureCode::InvalidDataset, executionError,
                    QStringLiteral("修复 Driver 校验问题并创建新 Snapshot。"))};
            }
            const QJsonObject report{{QStringLiteral("schemaVersion"), 2}, {QStringLiteral("valid"), true},
                {QStringLiteral("datasetSnapshotId"), request.snapshotId.toString()},
                {QStringLiteral("snapshotArtifactId"), verifiedSnapshot.record.artifactId.toString()},
                {QStringLiteral("datasetFormat"), verifiedSnapshot.record.datasetFormat},
                {QStringLiteral("driver"), QJsonObject{{QStringLiteral("id"), driver->id()},
                    {QStringLiteral("version"), driver->version()}}},
                {QStringLiteral("rootHash"), verifiedSnapshot.record.rootHash},
                {QStringLiteral("verifiedFileCount"), QString::number(verifiedSnapshot.files.size())}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("dataset_snapshot_validation"),
                    {{QStringLiteral("snapshot_validation.json"), jsonBytes(report)}}, &output, &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, makeFailure(FailureCode::ArtifactIncomplete, executionError)};
            }
        } else if (step.kind == QStringLiteral("AnalyzeQuality")) {
            if (!analyzeQuality(verifiedSnapshot, request.options, &analysis, stepCancellation, &executionError)) {
                const bool wasCanceled = executionError == QStringLiteral("quality.canceled");
                return {wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    makeFailure(wasCanceled ? FailureCode::Canceled : FailureCode::BackendUnsupported,
                        executionError, QStringLiteral("确认 Dataset Driver 格式属于  七格式质量规则范围。"))};
            }
            const QJsonObject report{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("datasetSnapshotId"), request.snapshotId.toString()},
                {QStringLiteral("datasetFormat"), verifiedSnapshot.record.datasetFormat},
                {QStringLiteral("summary"), analysis.summary}, {QStringLiteral("issues"), analysis.issues}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("dataset_quality_analysis"),
                    {{QStringLiteral("quality_analysis.json"), jsonBytes(report)},
                        {QStringLiteral("problem_samples.json"), jsonBytes(QJsonObject{{QStringLiteral("issues"), analysis.issues}})}},
                    &output, &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, makeFailure(FailureCode::ArtifactIncomplete, executionError)};
            }
        } else if (step.kind == QStringLiteral("ProduceRepairManifest")) {
            QJsonArray actions;
            for (const QJsonValue& value : analysis.issues) {
                const QJsonObject item = value.toObject();
                actions.append(QJsonObject{{QStringLiteral("issueCode"), item.value(QStringLiteral("code"))},
                    {QStringLiteral("severity"), item.value(QStringLiteral("severity"))},
                    {QStringLiteral("sampleRelativePath"), item.value(QStringLiteral("sampleRelativePath"))},
                    {QStringLiteral("sourceRelativePath"), item.value(QStringLiteral("sourceRelativePath"))},
                    {QStringLiteral("line"), item.value(QStringLiteral("line"))},
                    {QStringLiteral("action"), item.value(QStringLiteral("suggestedAction"))}});
            }
            repairManifest = {{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("dataset_repair_manifest")},
                {QStringLiteral("datasetSnapshotId"), request.snapshotId.toString()},
                {QStringLiteral("datasetFormat"), verifiedSnapshot.record.datasetFormat},
                {QStringLiteral("mutatesSource"), false}, {QStringLiteral("actions"), actions}};
            const QJsonObject xany{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("xanylabeling_review_manifest")},
                {QStringLiteral("datasetSnapshotId"), request.snapshotId.toString()},
                {QStringLiteral("samples"), actions}};
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("dataset_repair_manifest"),
                    {{QStringLiteral("repair_manifest.json"), jsonBytes(repairManifest)},
                        {QStringLiteral("xanylabeling_review_manifest.json"), jsonBytes(xany)}},
                    &output, &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, makeFailure(FailureCode::ArtifactIncomplete, executionError)};
            }
        } else if (step.kind == QStringLiteral("RenderQualityReport")) {
            qualityReport = {};
            qualityReport.insert(QStringLiteral("schemaVersion"), 2);
            qualityReport.insert(QStringLiteral("datasetSnapshotId"), request.snapshotId.toString());
            qualityReport.insert(QStringLiteral("datasetFormat"), verifiedSnapshot.record.datasetFormat);
            qualityReport.insert(QStringLiteral("summary"), analysis.summary);
            qualityReport.insert(QStringLiteral("issues"), analysis.issues);
            qualityReport.insert(QStringLiteral("repairActionCount"),
                repairManifest.value(QStringLiteral("actions")).toArray().size());
            qualityReport.insert(QStringLiteral("limitations"), QJsonArray{
                QStringLiteral("质量规则只生成复核建议，不修改原始标签。"),
                QStringLiteral("该报告不代表客户域模型精度或生产验收。")});
            const QString markdown = QStringLiteral("# 数据集质量报告\n\n- Snapshot：%1\n- 格式：%2\n- 问题数：%3\n\n该报告只生成复核建议，不修改原始标签。\n")
                .arg(request.snapshotId.toString(), verifiedSnapshot.record.datasetFormat).arg(analysis.issues.size());
            const QString html = QStringLiteral("<!doctype html><meta charset=\"utf-8\"><title>数据集质量报告</title>"
                "<h1>数据集质量报告</h1><p>Snapshot：%1</p><p>格式：%2</p><p>问题数：%3</p>"
                "<p>该报告只生成复核建议，不修改原始标签。</p>")
                .arg(request.snapshotId.toString().toHtmlEscaped(), verifiedSnapshot.record.datasetFormat.toHtmlEscaped())
                .arg(analysis.issues.size());
            if (!commitFiles(artifactStore_.get(), &storage_, taskId, QStringLiteral("dataset_quality_report"),
                    {{QStringLiteral("quality_report.json"), jsonBytes(qualityReport)},
                        {QStringLiteral("quality_report.md"), markdown.toUtf8()},
                        {QStringLiteral("quality_report.html"), html.toUtf8()}},
                    &output, &executionError, stepCancellation)) {
                return {WorkflowStepState::Failed, {}, makeFailure(FailureCode::ArtifactIncomplete, executionError)};
            }
        } else {
            return {WorkflowStepState::Failed, {}, makeFailure(FailureCode::InternalError,
                QStringLiteral("quality.workflow.unknown_step:%1").arg(step.kind))};
        }
        outputs.insert(step.kind, output);
        return {WorkflowStepState::Succeeded, output, {}};
    };
    if (!runner.run(workflow.id, executor, &runResult, error, cancellation)) return false;
    const TaskState terminalState = taskStateForResult(runResult);
    const Failure finalFailure = terminalFailure(runResult);
    if (terminalState == TaskState::Canceled) {
        TaskSnapshot current;
        if (!storage_.task(taskId, &current, error)) return false;
        if (current.state == TaskState::Running && !requestTaskCancellation(taskId, error)) return false;
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, terminalState, finalFailure,
            QDateTime::currentDateTimeUtc(), error)) return false;
    EvidenceBundle evidence;
    EvidenceArtifactBundle committedEvidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)
        || !commitEvidenceBundle(evidence, &committedEvidence, error)
        || !closeWorkflowTerminalization(workflow.id, error)) return false;
    result->workflowRunId = workflow.id;
    result->terminalState = terminalState;
    result->snapshotValidationArtifactId = outputs.value(QStringLiteral("ValidateSnapshot"));
    result->qualityAnalysisArtifactId = outputs.value(QStringLiteral("AnalyzeQuality"));
    result->repairManifestArtifactId = outputs.value(QStringLiteral("ProduceRepairManifest"));
    result->qualityReportArtifactId = outputs.value(QStringLiteral("RenderQualityReport"));
    result->evidenceArtifactId = committedEvidence.artifactId;
    result->summary = qualityReport.value(QStringLiteral("summary")).toObject();
    return true;
}

} // namespace aitrain
