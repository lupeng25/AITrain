#pragma once

#include <QHash>
#include <QWidget>

class QLabel;
class QPlainTextEdit;
class QTableWidget;

class EvaluationReportView : public QWidget {
    Q_OBJECT

public:
    explicit EvaluationReportView(QWidget* parent = nullptr);

    void clear();
    bool loadReport(const QString& reportPath);

private slots:
    void updateArtifactPreview();
    void updateSamplePreview();

private:
    void configureTable(QTableWidget* table, bool stretchLast = true) const;
    void populateMetrics(const QJsonObject& report);
    void populatePerClass(const QJsonObject& report);
    void populateOfficialArtifacts(const QJsonObject& report);
    void populateSamples(const QJsonObject& report);
    void showPreviewImage(const QString& imagePath);
    void showEmptyState(const QString& text);

    QLabel* statusLabel_ = nullptr;
    QLabel* summaryLabel_ = nullptr;
    QTableWidget* metricsTable_ = nullptr;
    QTableWidget* perClassTable_ = nullptr;
    QTableWidget* officialArtifactsTable_ = nullptr;
    QTableWidget* sampleTable_ = nullptr;
    QLabel* previewLabel_ = nullptr;
    QPlainTextEdit* detailText_ = nullptr;
    QString currentReportPath_;
    QHash<int, QString> artifactPreviewPaths_;
    QHash<int, QString> artifactDetailTexts_;
    QHash<int, QString> samplePreviewPaths_;
    QHash<int, QString> sampleDetailTexts_;
};
