#pragma once
#include "ModelRegistryPresenter.h"
#include "WorkbenchWidgets.h"
#include "ProjectObjectSelectors.h"
class QComboBox;
struct RuntimeDeliveryFormData final {
    QString modelPackageId, runtimeRoute, sampleDatasetId, sampleDatasetVersionId;
    QString sampleSnapshotId, sampleSnapshotArtifactId, sampleRelativePath;
};
enum class RuntimeDeliveryMode { DeploymentValidation, InferenceValidation };
class RuntimeDeliveryWorkspacePage final : public aitrain_app::WorkspaceViewHost {
    Q_OBJECT
public:
    explicit RuntimeDeliveryWorkspacePage(QWidget* parent = nullptr);
    RuntimeDeliveryFormData formData(RuntimeDeliveryMode mode) const;
    QString selectedModelPackageId(RuntimeDeliveryMode mode) const;
    void setModelPackages(const QVector<ModelPackageListItem>& packages);
    void setRouteEvaluation(RuntimeDeliveryMode mode, const QStringList& availableRoutes, const QStringList& reasons);
    void setRunning(RuntimeDeliveryMode mode);
    void selectModelPackageForInference(const QString& id);
    void showTab(int index);
    void setDatasetSelection(const aitrain_app::DatasetSelection& selected);
    void clearContext();
    QLabel* resultSummary = nullptr;
    aitrain_app::ImagePreviewLabel* overlay = nullptr;
    QPushButton* reportButton = nullptr;
    QPushButton* taskButton = nullptr;
signals:
    void modelSelectionChanged(RuntimeDeliveryMode mode, const QString& modelPackageId);
    void runRequested(RuntimeDeliveryMode mode);
    void selectSampleRequested();
    void reportRequested();
    void taskRequested();
    void openTaskRequested(const QString& taskId);
private:
    QComboBox* model_ = nullptr;
    QComboBox* route_ = nullptr;
    QLabel* sample_ = nullptr;
    QLabel* reasons_ = nullptr;
    RuntimeDeliveryFormData binding_;
};
Q_DECLARE_METATYPE(RuntimeDeliveryMode)
