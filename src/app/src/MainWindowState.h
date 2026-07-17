#pragma once

#include <QJsonArray>
#include <QString>
#include <QVector>

struct DatasetWorkbenchState {
    QString currentPath;
    QString currentFormat;
    QString currentDatasetId;
    QString currentDatasetVersionId;
    QString currentSnapshotId;
    QString currentSnapshotArtifactId;
    QString latestAnnotationSessionArtifactId;
    QString latestAnnotationEvidenceArtifactId;
    QString latestAnnotationSyncReportArtifactId;
    QString annotationWorkingDirectory;
    bool currentValid = false;
    QJsonArray sampleReviewSamples;
};

struct MainWindowState {
    DatasetWorkbenchState dataset;
};
