#pragma once

#include <QJsonArray>
#include <QString>
#include <QVector>

struct DatasetWorkbenchState {
    QString currentPath;
    QString currentFormat;
    QString latestQualityFixListPath;
    QString latestQualityFixManifestPath;
    QString latestQualityReportPath;
    QString latestAnnotationSessionArtifactId;
    QString latestAnnotationEvidenceArtifactId;
    QString latestAnnotationSyncReportArtifactId;
    QString annotationWorkingDirectory;
    bool currentValid = false;
    QJsonArray sampleReviewSamples;
};

struct ModelArtifactState {
    QString latestExportPath;
    QString latestInferenceOutputPath;
    QString latestEvaluationReportPath;
    QString latestBenchmarkReportPath;
};

struct DeliveryAcceptanceState {
    QString latestDeploymentValidationReportPath;
};

struct MainWindowState {
    DatasetWorkbenchState dataset;
    ModelArtifactState artifacts;
    DeliveryAcceptanceState delivery;
};
