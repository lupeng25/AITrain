#include "ModelRegistryPresenterV2.h"

#include <QDateTime>

namespace {

QString localTimeText(const QDateTime& value)
{
    return value.isValid()
        ? value.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))
        : QStringLiteral("--");
}

} // namespace

ModelRegistryPresenterV2::ModelRegistryPresenterV2(
    const aitrain::v2::ProjectQueryServiceV2* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("ModelRegistryPresenterV2"));
}

bool ModelRegistryPresenterV2::refresh(int limit)
{
    QString error;
    const QVector<aitrain::v2::ModelPackageReadModelV2> models = queryService_
        ? queryService_->modelPackages(limit, &error)
        : QVector<aitrain::v2::ModelPackageReadModelV2>();
    if (!queryService_ && error.isEmpty()) {
        error = QStringLiteral("模型库 Presenter 缺少 V2 项目查询服务。");
    }
    if (!error.isEmpty()) {
        modelPackages_.clear();
        lastError_ = error;
        emit modelPackagesChanged();
        emit queryFailed(lastError_);
        return false;
    }

    QVector<ModelPackageListItemV2> rows;
    rows.reserve(models.size());
    for (const aitrain::v2::ModelPackageReadModelV2& model : models) {
        ModelPackageListItemV2 row;
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

void ModelRegistryPresenterV2::clear()
{
    modelPackages_.clear();
    lastError_.clear();
    emit modelPackagesChanged();
}

int ModelRegistryPresenterV2::modelPackageCount() const { return modelPackages_.size(); }
QString ModelRegistryPresenterV2::lastError() const { return lastError_; }
const QVector<ModelPackageListItemV2>& ModelRegistryPresenterV2::modelPackages() const
{
    return modelPackages_;
}
