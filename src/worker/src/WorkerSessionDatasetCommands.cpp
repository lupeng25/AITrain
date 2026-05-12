#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/JsonProtocol.h"
#include "aitrain/core/OcrRecTrainer.h"
#include "aitrain/core/ProductWorkflow.h"
#include "aitrain/core/SegmentationTrainer.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QStandardPaths>
#include <QThread>

using namespace worker_support;
void WorkerSession::validateDataset(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(datasetPath).absoluteDir().absolutePath(), taskId);
    }

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始校验数据集。"));
    send(QStringLiteral("progress"), startProgress);

    aitrain::DatasetValidationResult result;
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        result = aitrain::validateYoloDetectionDataset(datasetPath, options);
    } else if (format == QStringLiteral("yolo_segmentation")) {
        result = aitrain::validateYoloSegmentationDataset(datasetPath, options);
    } else if (format == QStringLiteral("paddleocr_det")) {
        result = aitrain::validatePaddleOcrDetDataset(datasetPath, options);
    } else if (format == QStringLiteral("paddleocr_rec")) {
        result = aitrain::validatePaddleOcrRecDataset(datasetPath, options);
    } else {
        result.ok = false;
        aitrain::DatasetValidationResult::Issue issue;
        issue.severity = QStringLiteral("error");
        issue.code = QStringLiteral("unsupported_format");
        issue.filePath = datasetPath;
        issue.message = QStringLiteral("Worker 不支持该数据集格式：%1。").arg(format);
        result.issues.append(issue);
        result.errors.append(issue.message);
    }

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("数据集校验完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject response = result.toJson();
    response.insert(QStringLiteral("taskId"), taskId);
    response.insert(QStringLiteral("datasetPath"), datasetPath);
    response.insert(QStringLiteral("outputPath"), outputPath);
    response.insert(QStringLiteral("format"), format);
    response.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("dataset_validation_report.json"));
    response.insert(QStringLiteral("reportPath"), reportPath);
    QString writeError;
    if (!writeJsonFile(reportPath, response, &writeError)) {
        fail(writeError);
        return;
    }
    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("dataset_validation_report"));
    artifact.insert(QStringLiteral("path"), reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Dataset validation report"));
    send(QStringLiteral("artifact"), artifact);
    send(QStringLiteral("datasetValidation"), response);
    QJsonObject terminal;
    terminal.insert(QStringLiteral("taskId"), taskId);
    terminal.insert(QStringLiteral("message"), result.ok
        ? QStringLiteral("Dataset validation completed")
        : QStringLiteral("Dataset validation failed"));
    send(result.ok ? QStringLiteral("completed") : QStringLiteral("failed"), terminal);
    finishSession();
}

void WorkerSession::splitDataset(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();

    QJsonObject startProgress;
    startProgress.insert(QStringLiteral("taskId"), taskId);
    startProgress.insert(QStringLiteral("percent"), 0);
    startProgress.insert(QStringLiteral("message"), QStringLiteral("开始划分数据集。"));
    send(QStringLiteral("progress"), startProgress);

    aitrain::DatasetSplitResult result;
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        result = aitrain::splitYoloDetectionDataset(datasetPath, outputPath, options);
    } else if (format == QStringLiteral("yolo_segmentation")) {
        result = aitrain::splitYoloSegmentationDataset(datasetPath, outputPath, options);
    } else if (format == QStringLiteral("paddleocr_det")) {
        result = aitrain::splitPaddleOcrDetDataset(datasetPath, outputPath, options);
    } else if (format == QStringLiteral("paddleocr_rec")) {
        result = aitrain::splitPaddleOcrRecDataset(datasetPath, outputPath, options);
    } else {
        result.ok = false;
        result.outputPath = outputPath;
        result.errors.append(QStringLiteral("当前仅支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec 数据集划分。"));
    }

    QJsonObject progressPayload;
    progressPayload.insert(QStringLiteral("taskId"), taskId);
    progressPayload.insert(QStringLiteral("percent"), 100);
    progressPayload.insert(QStringLiteral("message"), QStringLiteral("数据集划分完成。"));
    send(QStringLiteral("progress"), progressPayload);

    QJsonObject response = result.toJson();
    response.insert(QStringLiteral("taskId"), taskId);
    response.insert(QStringLiteral("datasetPath"), datasetPath);
    response.insert(QStringLiteral("format"), format);
    response.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    const QString reportPath = QDir(outputPath).filePath(QStringLiteral("split_report.json"));
    response.insert(QStringLiteral("reportPath"), reportPath);
    if (!QFileInfo::exists(reportPath)) {
        QString writeError;
        if (!writeJsonFile(reportPath, response, &writeError)) {
            fail(writeError);
            return;
        }
    }
    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("dataset_split_report"));
    artifact.insert(QStringLiteral("path"), reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Dataset split report"));
    send(QStringLiteral("artifact"), artifact);
    send(QStringLiteral("datasetSplit"), response);
    QJsonObject terminal;
    terminal.insert(QStringLiteral("taskId"), taskId);
    terminal.insert(QStringLiteral("message"), result.ok
        ? QStringLiteral("Dataset split completed")
        : QStringLiteral("Dataset split failed"));
    send(result.ok ? QStringLiteral("completed") : QStringLiteral("failed"), terminal);
    finishSession();
}

void WorkerSession::curateDataset(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(datasetPath).absoluteDir().absolutePath(), taskId);
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始生成数据质量报告。"));
    send(QStringLiteral("progress"), progress);

    const aitrain::WorkflowResult result = aitrain::curateDatasetQualityReport(datasetPath, outputPath, format, options);
    if (!result.ok) {
        fail(QStringLiteral("Dataset quality report failed for %1 (%2): %3").arg(datasetPath, format, result.error));
        return;
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("dataset_quality_report"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Dataset quality report"));
    send(QStringLiteral("artifact"), artifact);

    const QString csvPath = result.payload.value(QStringLiteral("classDistributionPath")).toString();
    if (!csvPath.isEmpty()) {
        QJsonObject csvArtifact;
        csvArtifact.insert(QStringLiteral("taskId"), taskId);
        csvArtifact.insert(QStringLiteral("kind"), QStringLiteral("class_distribution"));
        csvArtifact.insert(QStringLiteral("path"), csvPath);
        csvArtifact.insert(QStringLiteral("message"), QStringLiteral("Class distribution CSV"));
        send(QStringLiteral("artifact"), csvArtifact);
    }
    const QString problemPath = result.payload.value(QStringLiteral("problemSamplesPath")).toString();
    if (!problemPath.isEmpty()) {
        QJsonObject problemArtifact;
        problemArtifact.insert(QStringLiteral("taskId"), taskId);
        problemArtifact.insert(QStringLiteral("kind"), QStringLiteral("problem_samples"));
        problemArtifact.insert(QStringLiteral("path"), problemPath);
        problemArtifact.insert(QStringLiteral("message"), QStringLiteral("Problem sample list"));
        send(QStringLiteral("artifact"), problemArtifact);
    }
    for (const auto& item : {
             qMakePair(QStringLiteral("image_statistics"), QStringLiteral("imageStatisticsPath")),
             qMakePair(QStringLiteral("split_distribution"), QStringLiteral("splitDistributionPath")),
             qMakePair(QStringLiteral("xanylabeling_fix_list"), QStringLiteral("xAnyLabelingFixListPath")),
             qMakePair(QStringLiteral("xanylabeling_fix_manifest"), QStringLiteral("xAnyLabelingFixManifestPath")),
             qMakePair(QStringLiteral("dataset_rework_sample_set"), QStringLiteral("reworkSampleSetPath")),
             qMakePair(QStringLiteral("prelabel_candidates"), QStringLiteral("prelabelCandidatesPath")),
             qMakePair(QStringLiteral("dataset_training_readiness"), QStringLiteral("trainingReadinessPath"))}) {
        const QString path = result.payload.value(item.second).toString();
        if (!path.isEmpty()) {
            QJsonObject extraArtifact;
            extraArtifact.insert(QStringLiteral("taskId"), taskId);
            extraArtifact.insert(QStringLiteral("kind"), item.first);
            extraArtifact.insert(QStringLiteral("path"), path);
            extraArtifact.insert(QStringLiteral("message"), QStringLiteral("Dataset quality artifact"));
            send(QStringLiteral("artifact"), extraArtifact);
        }
    }

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("数据质量报告完成。"));
    send(QStringLiteral("progress"), doneProgress);
    send(QStringLiteral("datasetQuality"), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Dataset quality report completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

void WorkerSession::createDatasetSnapshot(const QJsonObject& payload)
{
    const QString taskId = payload.value(QStringLiteral("taskId")).toString();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    if (outputPath.isEmpty()) {
        outputPath = defaultTaskOutputPath(QFileInfo(datasetPath).absoluteDir().absolutePath(), taskId);
    }

    QJsonObject progress;
    progress.insert(QStringLiteral("taskId"), taskId);
    progress.insert(QStringLiteral("percent"), 0);
    progress.insert(QStringLiteral("message"), QStringLiteral("开始生成数据集快照。"));
    send(QStringLiteral("progress"), progress);

    const aitrain::WorkflowResult result = aitrain::createDatasetSnapshotReport(datasetPath, outputPath, format, options);
    if (!result.ok) {
        fail(result.error);
        return;
    }

    QJsonObject artifact;
    artifact.insert(QStringLiteral("taskId"), taskId);
    artifact.insert(QStringLiteral("kind"), QStringLiteral("dataset_snapshot_manifest"));
    artifact.insert(QStringLiteral("path"), result.reportPath);
    artifact.insert(QStringLiteral("message"), QStringLiteral("Dataset snapshot manifest"));
    send(QStringLiteral("artifact"), artifact);

    QJsonObject doneProgress;
    doneProgress.insert(QStringLiteral("taskId"), taskId);
    doneProgress.insert(QStringLiteral("percent"), 100);
    doneProgress.insert(QStringLiteral("message"), QStringLiteral("数据集快照完成。"));
    send(QStringLiteral("progress"), doneProgress);
    send(QStringLiteral("datasetSnapshot"), result.payload);
    socket_.waitForBytesWritten(1000);

    QJsonObject completed;
    completed.insert(QStringLiteral("taskId"), taskId);
    completed.insert(QStringLiteral("message"), QStringLiteral("Dataset snapshot completed"));
    send(QStringLiteral("completed"), completed);
    finishSession();
}

