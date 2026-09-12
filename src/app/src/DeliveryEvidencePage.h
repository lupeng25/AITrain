#pragma once

#include "WorkbenchWidgets.h"

class QComboBox;
class QLabel;
class QLineEdit;
class QTableWidget;

class DeliveryEvidenceWorkspacePage final : public aitrain_app::WorkspaceViewHost
{
    Q_OBJECT

public:
    enum View { Catalog, Import, Acceptance, Diagnostics };
    explicit DeliveryEvidenceWorkspacePage(QWidget* parent = nullptr);

    QLabel* acceptanceSummaryLabel = nullptr;
    QTableWidget* acceptanceTable = nullptr;
    QLineEdit* ocrDetReportEdit = nullptr;
    QLineEdit* ocrRecReportEdit = nullptr;
    QLineEdit* ocrSystemReportEdit = nullptr;
    QLineEdit* ocrDetSnapshotIdEdit = nullptr;
    QLineEdit* ocrDetSnapshotArtifactIdEdit = nullptr;
    QLineEdit* ocrRecSnapshotIdEdit = nullptr;
    QLineEdit* ocrRecSnapshotArtifactIdEdit = nullptr;
    QLineEdit* ocrSystemSnapshotIdEdit = nullptr;
    QLineEdit* ocrSystemSnapshotArtifactIdEdit = nullptr;
    QLineEdit* ocrCohortIdEdit = nullptr;
    QLineEdit* ocrDomainIdEdit = nullptr;
    QComboBox* ocrEvidenceClassCombo = nullptr;
    QLineEdit* ocrDetReportArtifactIdEdit = nullptr;
    QLineEdit* ocrRecReportArtifactIdEdit = nullptr;
    QLineEdit* ocrSystemReportArtifactIdEdit = nullptr;
    QLineEdit* ocrMinDetHmeanEdit = nullptr;
    QLineEdit* ocrMinAccEdit = nullptr;
    QLineEdit* ocrMaxCerEdit = nullptr;
    QLineEdit* ocrMinSystemAccEdit = nullptr;
    QLabel* ocrStatusLabel = nullptr;
    QLabel* diagnosticsStatusLabel = nullptr;
    QLabel* snapshotLabels[3] = {};
    QLabel* reportLabels[3] = {};
    QPushButton* moreButton = nullptr;
signals:
    void browseRequested(QLineEdit* target);
    void bindSnapshotRequested(int index);
    void selectReportRequested(int index);
    void refreshRequested();
    void moreRequested();
    void importOcrRequested();
    void acceptanceRequested();
    void diagnosticsRequested();
    void externalImportRequested();
    void openSelectedRequested();
    void openTaskRequested(const QString& taskId);
};
