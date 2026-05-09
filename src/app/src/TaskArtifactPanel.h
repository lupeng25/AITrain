#pragma once

#include "aitrain/core/TaskModels.h"

#include <QWidget>
#include <QVector>

class EvaluationReportView;
class QLabel;
class QPlainTextEdit;
class QStackedWidget;
class QTableWidget;

class TaskArtifactPanel : public QWidget {
    Q_OBJECT

public:
    explicit TaskArtifactPanel(QWidget* parent = nullptr);

    void clear();
    void setTaskSummary(const QString& summary);
    void setArtifacts(const QVector<aitrain::ArtifactRecord>& artifacts);
    void setMetrics(const QVector<aitrain::MetricPoint>& metrics);
    void setExports(const QVector<aitrain::ExportRecord>& exports);
    QString selectedArtifactPath() const;

signals:
    void openDirectoryRequested();
    void copyPathRequested();
    void useForInferenceRequested();
    void useForExportRequested();
    void registerModelRequested();
    void evaluateRequested();
    void benchmarkRequested();
    void deliveryReportRequested();

private:
    void configureTable(QTableWidget* table) const;
    void clearTableWithPlaceholder(QTableWidget* table, const QString& placeholder);
    void updatePreviewFromSelection();
    void previewArtifactPath(const QString& path);

    QLabel* selectedTaskSummaryLabel_ = nullptr;
    QTableWidget* artifactTable_ = nullptr;
    QTableWidget* metricTable_ = nullptr;
    QTableWidget* exportTable_ = nullptr;
    QLabel* imagePreviewLabel_ = nullptr;
    QPlainTextEdit* previewText_ = nullptr;
    QStackedWidget* previewStack_ = nullptr;
    EvaluationReportView* evaluationReportView_ = nullptr;
};
