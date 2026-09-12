#pragma once

#include "aitrain/workflow\\ProjectQueryService.h"

#include <QObject>
#include <QVector>

// 数据集目录页面消费的无路径只读 ViewModel。
struct DatasetCatalogListItem final {
    QString datasetId;
    QString displayName;
    QString datasetFormat;
    qint64 versionCount = 0;
    qint64 snapshotCount = 0;
    QString latestVersionId;
    QString latestSnapshotId;
    QString latestArtifactId;
    QString latestRootHash;
    qsizetype latestFileCount = 0;
    QString latestCreatedAt;
    QString latestSourceTaskId;
    QString latestQualityTaskId;
};

class DatasetCatalogPresenter final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int datasetCount READ datasetCount NOTIFY datasetsChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY queryFailed)

public:
    explicit DatasetCatalogPresenter(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    void setCatalogFilter(const aitrain::CatalogFilter& filter) { filter_ = filter; }
    bool refresh(const aitrain::PageRequest& request = {50, {}});
    bool loadMore();
    bool hasMore() const;
    QString nextCursor() const { return nextCursor_; }
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
    aitrain::CatalogFilter filter_;
    QString lastError_;
    QString nextCursor_;
    bool hasMore_ = false;
};
