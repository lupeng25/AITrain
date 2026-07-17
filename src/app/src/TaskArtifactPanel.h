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

    void setPresenter(TaskArtifactPresenter* presenter);
    void clear();
    void setTaskSummary(const QString& summary);
    void setDetails(const TaskArtifactDetails& details);
    int artifactRowCount() const;
    int metricRowCount() const;
    int workflowStepRowCount() const;

private:
    void configureTable(QTableWidget* table) const;
    void clearTableWithPlaceholder(QTableWidget* table, const QString& placeholder);
    void updatePreviewFromSelection();
    void previewSelectedArtifact();

    QLabel* selectedTaskSummaryLabel_ = nullptr;
    QTableWidget* artifactTable_ = nullptr;
    QTableWidget* metricTable_ = nullptr;
    QTableWidget* exportTable_ = nullptr;
    QTabWidget* detailTabs_ = nullptr;
    QLabel* imagePreviewLabel_ = nullptr;
    QPlainTextEdit* previewText_ = nullptr;
    QStackedWidget* previewStack_ = nullptr;
    EvaluationReportView* evaluationReportView_ = nullptr;
    TaskArtifactPresenter* presenter_ = nullptr;
    QString selectedArtifactId_;
    QString selectedRelativePath_;
};
