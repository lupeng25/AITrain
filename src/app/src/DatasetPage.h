#pragma once

#include "WorkbenchWidgets.h"

class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QTableWidget;

class DatasetWorkspacePage final : public aitrain_app::WorkspaceViewHost
{
    Q_OBJECT

public:
    enum View { Catalog, Detail, Import, Split, Conversion, Quality, Review, Annotation, Technical };
    explicit DatasetWorkspacePage(QWidget* parent = nullptr);
    void setSelectionAvailable(bool available);
    void showView(View view);

    template<class T>
    T* control(const QString& objectName) const
    {
        return findChild<T*>(objectName);
    }

    QLabel* catalogStatusLabel = nullptr;
    aitrain_app::ImagePreviewLabel* sampleImageLabel = nullptr;
    QLabel* sampleStatusLabel = nullptr;
    QLabel* operationStatusLabel = nullptr;
    QComboBox* snapshotCombo = nullptr;
    QPushButton* previousPageButton = nullptr;
    QPushButton* nextPageButton = nullptr;
    QPushButton* sampleLoadMoreButton = nullptr;
    QPushButton* snapshotLoadMoreButton = nullptr;
    QTableWidget* datasetListTable = nullptr;
    QLineEdit* datasetPathEdit = nullptr;
    QLabel* datasetProbeStatusLabel = nullptr;
    QComboBox* datasetFormatCombo = nullptr;
    QLineEdit* dataQualityDatasetIdEdit = nullptr;
    QLineEdit* dataQualityDatasetVersionIdEdit = nullptr;
    QLineEdit* dataQualitySnapshotIdEdit = nullptr;
    QLineEdit* dataQualitySnapshotArtifactIdEdit = nullptr;
    QLineEdit* splitSourceDatasetIdEdit = nullptr;
    QLineEdit* splitSourceDatasetVersionIdEdit = nullptr;
    QLineEdit* splitSourceSnapshotIdEdit = nullptr;
    QLineEdit* splitSourceSnapshotArtifactIdEdit = nullptr;
    QLineEdit* splitTargetDatasetIdEdit = nullptr;
    QLineEdit* splitTargetDatasetNameEdit = nullptr;
    QLineEdit* splitTrainRatioEdit = nullptr;
    QLineEdit* splitValRatioEdit = nullptr;
    QLineEdit* splitTestRatioEdit = nullptr;
    QLineEdit* splitSeedEdit = nullptr;
    QLineEdit* datasetSnapshotTargetDatasetIdEdit = nullptr;
    QLineEdit* datasetSnapshotTargetDatasetNameEdit = nullptr;
    QComboBox* datasetConversionSourceFormatCombo = nullptr;
    QComboBox* datasetConversionTargetFormatCombo = nullptr;
    QLineEdit* datasetConversionInputEdit = nullptr;
    QLabel* datasetConversionProbeStatusLabel = nullptr;
    QLineEdit* datasetConversionTargetDatasetIdEdit = nullptr;
    QLineEdit* datasetConversionTargetDatasetNameEdit = nullptr;
    QLabel* datasetConversionStatusLabel = nullptr;
    QLabel* datasetConversionSourceErrorLabel = nullptr;
    QLabel* datasetConversionTargetErrorLabel = nullptr;
    QLabel* datasetConversionInputErrorLabel = nullptr;
    QLabel* datasetConversionResultLabel = nullptr;
    QPushButton* datasetConversionStartButton = nullptr;
    QPushButton* datasetConversionCancelButton = nullptr;
    QPushButton* datasetConversionBrowseInputButton = nullptr;
    QProgressBar* datasetConversionProgressBar = nullptr;
    QPlainTextEdit* datasetConversionLog = nullptr;
    QLabel* validationSummaryLabel = nullptr;
    QLabel* datasetRepairLoopLabel = nullptr;
    QLabel* datasetDetailLabel = nullptr;
    QLabel* annotationToolStatusLabel = nullptr;
    QTableWidget* validationIssuesTable = nullptr;
    QTableWidget* datasetRepairLoopTable = nullptr;
    QTableWidget* datasetPreviewTable = nullptr;
    QPlainTextEdit* validationOutput = nullptr;
    QLineEdit* reviewSamplePathEdit = nullptr;
    QComboBox* reviewSourceFilterCombo = nullptr;
    QComboBox* reviewReasonFilterCombo = nullptr;
    QLineEdit* reviewSearchEdit = nullptr;
    QTableWidget* sampleReviewTable = nullptr;
    QLabel* sampleReviewSummaryLabel = nullptr;

signals:
    void importRequested();
    void importVersionRequested();
    void qualityRequested();
    void splitRequested();
    void conversionRequested();
    void cancelRequested();
    void browseDatasetRequested();
    void browseConversionRequested();
    void conversionSourceChanged();
    void refreshRequested();
    void nextPageRequested();
    void previousPageRequested();
    void trainRequested();
    void snapshotChanged(int index);
    void moreSnapshotsRequested();
    void moreSamplesRequested();
    void sampleSelected(int row);
    void reportRequested(bool repairList);
    void createAnnotationRequested();
    void syncAnnotationRequested();
    void chooseReviewRequested();
    void loadReviewRequested();
    void reviewFilterChanged();
    void openReviewSampleRequested();

private:
    QList<QPushButton*> selectionActions_;
};
