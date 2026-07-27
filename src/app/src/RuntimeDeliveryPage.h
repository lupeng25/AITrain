#pragma once

#include "ModelRegistryPresenter.h"

#include <QStringList>
#include <QWidget>

class QComboBox;
class QLabel;
class QLineEdit;
class QTabWidget;

struct RuntimeDeliveryFormData final
{
    QString modelPackageId;
    QString runtimeRoute;
    QString sampleDatasetId;
    QString sampleDatasetVersionId;
    QString sampleSnapshotId;
    QString sampleSnapshotArtifactId;
    QString sampleRelativePath;
};

enum class RuntimeDeliveryMode {
    DeploymentValidation,
    InferenceValidation
};

class RuntimeDeliveryWorkspacePage final : public QWidget
{
    Q_OBJECT

public:
    explicit RuntimeDeliveryWorkspacePage(QWidget* parent = nullptr);

    RuntimeDeliveryFormData formData(RuntimeDeliveryMode mode) const;
    QString selectedModelPackageId(RuntimeDeliveryMode mode) const;
    void setModelPackages(const QVector<ModelPackageListItem>& packages);
    void setRouteEvaluation(RuntimeDeliveryMode mode,
        const QStringList& availableRoutes,
        const QStringList& reasons);
    void setRunning(RuntimeDeliveryMode mode);
    void selectModelPackageForInference(const QString& modelPackageId);
    void showTab(int tabIndex);

signals:
    void modelSelectionChanged(RuntimeDeliveryMode mode,
        const QString& modelPackageId);
    void runRequested(RuntimeDeliveryMode mode);

private:
    struct FormControls final {
        QComboBox* model = nullptr;
        QComboBox* route = nullptr;
        QLineEdit* datasetId = nullptr;
        QLineEdit* datasetVersionId = nullptr;
        QLineEdit* snapshotId = nullptr;
        QLineEdit* snapshotArtifactId = nullptr;
        QLineEdit* relativePath = nullptr;
        QLabel* reasons = nullptr;
        QLabel* result = nullptr;
        QLabel* overlay = nullptr;
    };

    QWidget* buildForm(RuntimeDeliveryMode mode);
    FormControls& controls(RuntimeDeliveryMode mode);
    const FormControls& controls(RuntimeDeliveryMode mode) const;

    QTabWidget* tabs_ = nullptr;
    FormControls deployment_;
    FormControls inference_;
};

Q_DECLARE_METATYPE(RuntimeDeliveryMode)
