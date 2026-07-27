#pragma once

#include "ModelRegistryPresenter.h"

#include <QWidget>

class QLabel;
class QLineEdit;
class QTableWidget;

class ModelRegistryWorkspacePage final : public QWidget
{
    Q_OBJECT

public:
    explicit ModelRegistryWorkspacePage(QWidget* parent = nullptr);

    QString sourceFilePath() const;
    QString manifestFilePath() const;
    QString selectedModelPackageId() const;
    void setSourceFilePath(const QString& path);
    void setManifestFilePath(const QString& path);
    void setImportStatus(const QString& status);
    void renderPackages(const QVector<ModelPackageListItem>& packages,
        const QString& status);

signals:
    void refreshRequested();
    void browseSourceRequested();
    void browseManifestRequested();
    void importRequested();
    void useForRuntimeRequested(const QString& modelPackageId);

private:
    QLineEdit* sourceEdit_ = nullptr;
    QLineEdit* manifestEdit_ = nullptr;
    QLabel* importStatusLabel_ = nullptr;
    QLabel* summaryLabel_ = nullptr;
    QTableWidget* packageTable_ = nullptr;
};
