#include "ModelRegistryPresenter.h"

#include <QDateTime>

namespace {

QString localTimeText(const QDateTime& value)
{
    return value.isValid()
        ? value.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))
        : QStringLiteral("--");
}

} // namespace

ModelRegistryPresenter::ModelRegistryPresenter(
    const aitrain::ProjectQueryService* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("ModelRegistryPresenter"));
}

bool ModelRegistryPresenter::refresh(const aitrain::PageRequest& request)
{
    QString error;
    const aitrain::Page<aitrain::ModelPackageReadModel> page = queryService_
        ? queryService_->modelPackages(request, &error)
        : aitrain::Page<aitrain::ModelPackageReadModel>();
    if (!queryService_ && error.isEmpty()) {
        error = QStringLiteral("模型库 Presenter 缺少项目查询服务。");
    }
    if (!error.isEmpty()) {
        modelPackages_.clear();
        lastError_ = error;
        emit modelPackagesChanged();
        emit queryFailed(lastError_);
        return false;
    }

    QVector<ModelPackageListItem> rows;
    rows.reserve(page.items.size());
    for (const aitrain::ModelPackageReadModel& model : page.items) {
        ModelPackageListItem row;
        row.modelPackageId = model.modelPackageId.toString();
        row.sourceTaskId = model.sourceTaskId.toString();
        row.sourceSnapshotId = model.sourceSnapshotId.toString();
        row.sourceArtifactId = model.sourceArtifactId.toString();
        row.sourceArtifactSha256 = model.sourceArtifactSha256;
        row.modelFamily = model.modelFamily;
        row.taskType = model.taskType;
        row.sourceBackend = model.sourceBackend;
        row.artifactFormat = model.artifactFormat;
        row.decoder = model.decoder;
        row.exporterVersion = model.exporterVersion;
        row.runtimeRoutes = model.runtimeRoutes;
        row.limitations = model.limitations;
        row.verified = model.verified;
        row.createdAt = localTimeText(model.createdAt);
        rows.append(row);
    }

    if (request.after.isEmpty()) modelPackages_ = rows;
    else modelPackages_ += rows;
    nextCursor_ = page.nextCursor;
    hasMore_ = page.hasMore;
    lastError_.clear();
    emit modelPackagesChanged();
    return true;
}

bool ModelRegistryPresenter::loadMore()
{
    return hasMore_ && refresh({50, nextCursor_});
}

bool ModelRegistryPresenter::hasMore() const { return hasMore_; }

void ModelRegistryPresenter::clear()
{
    modelPackages_.clear();
    lastError_.clear();
    nextCursor_.clear();
    hasMore_ = false;
    emit modelPackagesChanged();
}

int ModelRegistryPresenter::modelPackageCount() const { return modelPackages_.size(); }
QString ModelRegistryPresenter::lastError() const { return lastError_; }
const QVector<ModelPackageListItem>& ModelRegistryPresenter::modelPackages() const
{
    return modelPackages_;
}
