#pragma once

#include "ModelRegistryPresenter.h"

#include "WorkbenchWidgets.h"

class QLabel;
class QLineEdit;
class QTableWidget;

class ModelRegistryWorkspacePage final : public aitrain_app::WorkspaceViewHost
{
    Q_OBJECT

public:
    enum View { Catalog, Detail, Import, Technical };
    explicit ModelRegistryWorkspacePage(QWidget* parent = nullptr);
    QPushButton* moreButton = nullptr;
    void selectPackage(const QString& id, bool openDetail = false);

    QString sourceFilePath() const;
    QString manifestFilePath() const;
    QString selectedModelPackageId() const;
    void setSourceFilePath(const QString& path);
    void setManifestFilePath(const QString& path);
    void setImportStatus(const QString& status);
    void renderPackages(const QVector<ModelPackageListItem>& packages,
        const QString& status);

signals:
    void moreRequested();
    void reportsRequested();
    void sourceTaskRequested(const QString& taskId);
    void latestValidationRequested(const QString& taskId);
    void editManifestRequested();
    void refreshRequested();
    void browseSourceRequested();
    void browseManifestRequested();
    void importRequested();
    void useForRuntimeRequested(const QString& modelPackageId);

private:
    void showDetails();
    QVector<ModelPackageListItem> packages_;
    QLabel* detailSummary_ = nullptr;
    QLabel* technicalSummary_ = nullptr;
    QLineEdit* sourceEdit_ = nullptr;
    QLineEdit* manifestEdit_ = nullptr;
    QLabel* importStatusLabel_ = nullptr;
    QLabel* summaryLabel_ = nullptr;
    QTableWidget* packageTable_ = nullptr;
};
