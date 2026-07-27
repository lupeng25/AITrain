#pragma once

#include <QScrollArea>

class QComboBox;
class QLabel;
class QLineEdit;
class QTableWidget;

class DeliveryEvidenceWorkspacePage final : public QScrollArea
{
    Q_OBJECT

public:
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
};
