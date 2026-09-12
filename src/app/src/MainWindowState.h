#pragma once

#include <QJsonArray>
#include <QString>
#include <QVector>

struct DatasetWorkbenchState {
    QString currentPath;
    QString currentFormat;
    QString currentDisplayName;
    QString currentSampleRelativePath;
    QString currentDatasetId;
    QString currentDatasetVersionId;
    QString currentSnapshotId;
    QString currentSnapshotArtifactId;
    QString latestAnnotationSessionArtifactId;
    QString latestAnnotationEvidenceArtifactId;
    QString latestAnnotationSyncReportArtifactId;
    QString sampleReviewArtifactId;
    QString annotationWorkingDirectory;
    QString latestQualityTaskId;
    QString latestQualityArtifactId;
    QString latestRepairArtifactId;
    bool currentValid = false;
    QJsonArray sampleReviewSamples;
};

struct MainWindowState {
    DatasetWorkbenchState dataset;
};
