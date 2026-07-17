#pragma once

#include <QByteArray>
#include <QHash>
#include <QJsonObject>
#include <QWidget>

#include <functional>

class QLabel;
class QPlainTextEdit;
class QTableWidget;

class EvaluationReportView : public QWidget {
    Q_OBJECT

public:
    using ArtifactPreviewProvider = std::function<bool(
        const QString& relativePath, QByteArray* content, QString* error)>;

    explicit EvaluationReportView(QWidget* parent = nullptr);

    void clear();
    void setArtifactPreviewProvider(ArtifactPreviewProvider provider);
    bool loadReportData(const QByteArray& data, const QString& relativePath = QString());

private slots:
    void updateArtifactPreview();
    void updateSamplePreview();

private:
    void configureTable(QTableWidget* table, bool stretchLast = true) const;
    void populateMetrics(const QJsonObject& report);
    void populatePerClass(const QJsonObject& report);
    void populateOfficialArtifacts(const QJsonObject& report);
    void populateSamples(const QJsonObject& report);
    bool loadReportObject(const QJsonObject& report);
    QString resolveArtifactRelativePath(const QString& declaredPath) const;
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
    QString currentReportRelativePath_;
    ArtifactPreviewProvider artifactPreviewProvider_;
    QHash<int, QString> artifactPreviewPaths_;
    QHash<int, QString> artifactDetailTexts_;
    QHash<int, QString> samplePreviewPaths_;
    QHash<int, QString> sampleDetailTexts_;
};
