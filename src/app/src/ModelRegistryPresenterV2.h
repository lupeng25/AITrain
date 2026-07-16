#pragma once

#include "aitrain/v2/ProjectQueryServiceV2.h"

#include <QObject>
#include <QStringList>
#include <QVector>

// 模型库页面可直接消费的无路径 ViewModel。模型选择只使用 ModelPackageId，
// 来源追溯只使用 V2 Task/Snapshot/Artifact 身份。
struct ModelPackageListItemV2 final {
    QString modelPackageId;
    QString sourceTaskId;
    QString sourceSnapshotId;
    QString sourceArtifactId;
    QString sourceArtifactSha256;
    QString modelFamily;
    QString taskType;
    QString sourceBackend;
    QString artifactFormat;
    QString decoder;
    QString exporterVersion;
    QStringList runtimeRoutes;
    QStringList limitations;
    bool verified = false;
    QString createdAt;
};

class ModelRegistryPresenterV2 final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int modelPackageCount READ modelPackageCount NOTIFY modelPackagesChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY queryFailed)

public:
    explicit ModelRegistryPresenterV2(
        const aitrain::v2::ProjectQueryServiceV2* queryService,
        QObject* parent = nullptr);

    bool refresh(int limit = 200);
    void clear();

    int modelPackageCount() const;
    QString lastError() const;
    const QVector<ModelPackageListItemV2>& modelPackages() const;

signals:
    void modelPackagesChanged();
    void queryFailed(const QString& error);

private:
    const aitrain::v2::ProjectQueryServiceV2* queryService_ = nullptr;
    QVector<ModelPackageListItemV2> modelPackages_;
    QString lastError_;
};
