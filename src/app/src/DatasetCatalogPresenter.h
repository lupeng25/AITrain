#pragma once

#include "aitrain/workflow\\ProjectQueryService.h"

#include <QObject>
#include <QVector>

// 数据集目录页面消费的无路径只读 ViewModel。
struct DatasetCatalogListItem final {
    QString datasetId;
    QString datasetFormat;
    qint64 versionCount = 0;
    qint64 snapshotCount = 0;
    QString latestVersionId;
    QString latestSnapshotId;
    QString latestArtifactId;
    QString latestRootHash;
    qsizetype latestFileCount = 0;
    QString latestCreatedAt;
};

class DatasetCatalogPresenter final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int datasetCount READ datasetCount NOTIFY datasetsChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY queryFailed)

public:
    explicit DatasetCatalogPresenter(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    bool refresh(int limit = 200);
    void clear();

    int datasetCount() const;
    QString lastError() const;
    const QVector<DatasetCatalogListItem>& datasets() const;

signals:
    void datasetsChanged();
    void queryFailed(const QString& error);

private:
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    QVector<DatasetCatalogListItem> datasets_;
    QString lastError_;
};
