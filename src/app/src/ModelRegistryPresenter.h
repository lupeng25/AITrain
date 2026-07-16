#pragma once

#include "aitrain/workflow\ProjectQueryService.h"

#include <QObject>
#include <QStringList>
#include <QVector>

// 模型库页面可直接消费的无路径 ViewModel。模型选择只使用 ModelPackageId，
// 来源追溯只使用  Task/Snapshot/Artifact 身份。
struct ModelPackageListItem final {
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

class ModelRegistryPresenter final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int modelPackageCount READ modelPackageCount NOTIFY modelPackagesChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY queryFailed)

public:
    explicit ModelRegistryPresenter(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    bool refresh(int limit = 200);
    void clear();

    int modelPackageCount() const;
    QString lastError() const;
    const QVector<ModelPackageListItem>& modelPackages() const;

signals:
    void modelPackagesChanged();
    void queryFailed(const QString& error);

private:
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    QVector<ModelPackageListItem> modelPackages_;
    QString lastError_;
};
