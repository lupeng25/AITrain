#pragma once

#include "aitrain/core/TaskModels.h"

#include <QJsonArray>
#include <QString>
#include <QVector>

struct PendingTrainingTask {
    QString taskId;
    aitrain::TrainingRequest request;
    bool needsSnapshot = false;
    int datasetId = 0;
    QString datasetFormat;
};

struct DatasetWorkbenchState {
    QString currentConversionTaskId;
    QString currentPath;
    QString currentFormat;
    QString latestQualityFixListPath;
    QString latestQualityFixManifestPath;
    QString latestQualityReportPath;
    QString latestReviewListPath;
    bool currentValid = false;
    QJsonArray sampleReviewSamples;
};

struct TrainingRunState {
    QString currentTaskId;
    QVector<PendingTrainingTask> pendingTrainingTasks;
    PendingTrainingTask activeSnapshotTrainingTask;
    bool hasActiveSnapshotTrainingTask = false;
};

struct ModelArtifactState {
    QString latestExportPath;
    QString latestInferenceOutputPath;
    QString latestEvaluationReportPath;
    QString latestBenchmarkReportPath;
};

struct DeliveryAcceptanceState {
    QString latestDeploymentValidationReportPath;
    QString latestCustomerOcrAcceptanceReportPath;
    QString latestDiagnosticBundlePath;
};

struct MainWindowState {
    DatasetWorkbenchState dataset;
    TrainingRunState training;
    ModelArtifactState artifacts;
    DeliveryAcceptanceState delivery;
};
