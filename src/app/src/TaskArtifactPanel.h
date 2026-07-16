#pragma once

#include "TaskArtifactPresenter.h"

#include <QWidget>
#include <QVector>

class EvaluationReportView;
class QLabel;
class QPlainTextEdit;
class QStackedWidget;
class QTableWidget;
class QTabWidget;

class TaskArtifactPanel : public QWidget {
    Q_OBJECT

public:
    explicit TaskArtifactPanel(QWidget* parent = nullptr);

    void clear();
    void setTaskSummary(const QString& summary);
    void setDetails(const TaskArtifactDetails& details);
    QString selectedArtifactPath() const;
    int artifactRowCount() const;
    int metricRowCount() const;
    int workflowStepRowCount() const;

signals:
    void openDirectoryRequested();
    void copyPathRequested();
    void useForInferenceRequested();
    void registerModelRequested();

private:
    void configureTable(QTableWidget* table) const;
    void clearTableWithPlaceholder(QTableWidget* table, const QString& placeholder);
    void updatePreviewFromSelection();
    void previewArtifactPath(const QString& path);

    QLabel* selectedTaskSummaryLabel_ = nullptr;
    QTableWidget* artifactTable_ = nullptr;
    QTableWidget* metricTable_ = nullptr;
    QTableWidget* exportTable_ = nullptr;
    QTabWidget* detailTabs_ = nullptr;
    QLabel* imagePreviewLabel_ = nullptr;
    QPlainTextEdit* previewText_ = nullptr;
    QStackedWidget* previewStack_ = nullptr;
    EvaluationReportView* evaluationReportView_ = nullptr;
    QWidget* legacyArtifactActions_ = nullptr;
};
