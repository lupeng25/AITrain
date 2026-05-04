#pragma once

#include <QHash>
#include <QWidget>

class QLabel;
class QPlainTextEdit;
class QStackedWidget;
class QTableWidget;

class EvaluationReportView : public QWidget {
    Q_OBJECT

public:
    explicit EvaluationReportView(QWidget* parent = nullptr);

    void clear();
    bool loadReport(const QString& reportPath);

private slots:
    void updateErrorPreview();

private:
    void configureTable(QTableWidget* table, bool stretchLast = true) const;
    void populateMetrics(const QJsonObject& report);
    void populatePerClass(const QJsonObject& report);
    void populateConfusion(const QJsonObject& report);
    void populateErrors(const QJsonObject& report);
    void showOverlayImage(const QString& overlayPath);
    void showEmptyState(const QString& text);

    QLabel* statusLabel_ = nullptr;
    QLabel* summaryLabel_ = nullptr;
    QTableWidget* metricsTable_ = nullptr;
    QTableWidget* perClassTable_ = nullptr;
    QTableWidget* confusionTable_ = nullptr;
    QTableWidget* errorTable_ = nullptr;
    QLabel* overlayLabel_ = nullptr;
    QPlainTextEdit* detailText_ = nullptr;
    QString currentReportPath_;
    QHash<int, QString> rowOverlayPaths_;
    QHash<int, QString> rowDetailTexts_;
};
