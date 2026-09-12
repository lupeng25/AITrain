#pragma once

#include "aitrain/workflow/ProjectQueryService.h"
#include "WorkbenchLabels.h"

class QWidget;

namespace aitrain_app {

struct DatasetSelection final {
    aitrain::DatasetSnapshotReadModel snapshot;
    QString displayName;
    QString sampleRelativePath;
};

bool isImageMember(const QString& relativePath);
QString selectProjectArtifact(QWidget* parent,
    const aitrain::ProjectQueryService* query, const QStringList& kinds,
    const QString& title);
bool selectProjectDataset(QWidget* parent,
    const aitrain::ProjectQueryService* query, DatasetSelection* selection,
    bool requireSample = false);
void showArtifactReport(QWidget* parent, const aitrain::ProjectQueryService* query,
    const QString& artifactId, const QString& relativePath,
    const QString& title = QString());

} // namespace aitrain_app
