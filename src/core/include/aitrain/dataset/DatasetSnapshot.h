#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QJsonArray>
#include <QJsonObject>

#include <functional>

namespace aitrain {

struct DatasetSnapshotOptions final {
    qsizetype maxFileCount = 1000000;
    QJsonArray classDefinitions;
    std::function<bool()> isCancellationRequested;
    std::function<void(qsizetype filesVisited)> progress;
};

struct DatasetSnapshotResult final {
    SnapshotId snapshotId;
    QString manifestPath;
    QString rootHash;
    qsizetype fileCount = 0;
    qint64 totalBytes = 0;
    QJsonObject manifest;
};

bool createDatasetSnapshot(const QString& datasetRoot,
    const QString& manifestPath,
    const QString& datasetFormat,
    const QString& driverId,
    const QString& driverVersion,
    const DatasetSnapshotOptions& options,
    DatasetSnapshotResult* result,
    QString* error = nullptr);

} // namespace aitrain
