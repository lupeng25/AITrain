#include "DatasetCatalogPresenter.h"

#include <QDateTime>

namespace {

QString localTimeText(const QDateTime& value)
{
    return value.isValid()
        ? value.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))
        : QStringLiteral("--");
}

} // namespace

DatasetCatalogPresenter::DatasetCatalogPresenter(
    const aitrain::ProjectQueryService* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("DatasetCatalogPresenter"));
}

bool DatasetCatalogPresenter::refresh(const aitrain::PageRequest& request)
{
    QString error;
    const aitrain::Page<aitrain::DatasetCatalogReadModel> page = queryService_
        ? queryService_->datasetCatalog(request, &error)
        : aitrain::Page<aitrain::DatasetCatalogReadModel>();
    if (!queryService_ && error.isEmpty()) {
        error = QStringLiteral("数据集目录 Presenter 缺少项目查询服务。");
    }
    if (!error.isEmpty()) {
        datasets_.clear();
        lastError_ = error;
        emit datasetsChanged();
        emit queryFailed(lastError_);
        return false;
    }

    QVector<DatasetCatalogListItem> rows;
    rows.reserve(page.items.size());
    for (const aitrain::DatasetCatalogReadModel& model : page.items) {
        DatasetCatalogListItem row;
        row.datasetId = model.datasetId.toString();
        row.datasetFormat = model.datasetFormat;
        row.versionCount = model.versionCount;
        row.snapshotCount = model.snapshotCount;
        row.latestVersionId = model.latestVersionId.toString();
        row.latestSnapshotId = model.latestSnapshotId.toString();
        row.latestArtifactId = model.latestArtifactId.toString();
        row.latestRootHash = model.latestRootHash;
        row.latestFileCount = model.latestFileCount;
        row.latestCreatedAt = localTimeText(model.latestCreatedAt);
        rows.append(row);
    }

    if (request.after.isEmpty()) datasets_ = rows;
    else datasets_ += rows;
    nextCursor_ = page.nextCursor;
    hasMore_ = page.hasMore;
    lastError_.clear();
    emit datasetsChanged();
    return true;
}

bool DatasetCatalogPresenter::loadMore()
{
    return hasMore_ && refresh({50, nextCursor_});
}

bool DatasetCatalogPresenter::hasMore() const { return hasMore_; }

void DatasetCatalogPresenter::clear()
{
    datasets_.clear();
    lastError_.clear();
    nextCursor_.clear();
    hasMore_ = false;
    emit datasetsChanged();
}

int DatasetCatalogPresenter::datasetCount() const { return datasets_.size(); }
QString DatasetCatalogPresenter::lastError() const { return lastError_; }
const QVector<DatasetCatalogListItem>& DatasetCatalogPresenter::datasets() const { return datasets_; }
