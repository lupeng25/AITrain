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

bool ModelRegistryPresenter::refresh(int limit)
{
    QString error;
    const QVector<aitrain::ModelPackageReadModel> models = queryService_
        ? queryService_->modelPackages(limit, &error)
        : QVector<aitrain::ModelPackageReadModel>();
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
    rows.reserve(models.size());
    for (const aitrain::ModelPackageReadModel& model : models) {
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

    modelPackages_ = rows;
    lastError_.clear();
    emit modelPackagesChanged();
    return true;
}

void ModelRegistryPresenter::clear()
{
    modelPackages_.clear();
    lastError_.clear();
    emit modelPackagesChanged();
}

int ModelRegistryPresenter::modelPackageCount() const { return modelPackages_.size(); }
QString ModelRegistryPresenter::lastError() const { return lastError_; }
const QVector<ModelPackageListItem>& ModelRegistryPresenter::modelPackages() const
{
    return modelPackages_;
}
