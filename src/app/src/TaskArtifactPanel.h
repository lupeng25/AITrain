#pragma once

#include "TaskArtifactPresenter.h"

#include <QWidget>
#include <QVector>

class EvaluationReportView;
class QLabel;
class QPlainTextEdit;
class QPushButton;
class QStackedWidget;
class QTableView;
class QTableWidget;
class QComboBox;
class ArtifactFileTableModel;
class ArtifactTableModel;
class MetricTableModel;

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
    QTableView* artifactTable_ = nullptr;
    QTableView* artifactFileTable_ = nullptr;
    QTableView* metricTable_ = nullptr;
    QTableWidget* exportTable_ = nullptr;
    ArtifactTableModel* artifactModel_ = nullptr;
    ArtifactFileTableModel* artifactFileModel_ = nullptr;
    MetricTableModel* metricModel_ = nullptr;
    QPushButton* artifactLoadMoreButton_ = nullptr;
    QPushButton* artifactFileLoadMoreButton_ = nullptr;
    QPushButton* metricLoadMoreButton_ = nullptr;
    QPushButton* workflowLoadMoreButton_ = nullptr;
    QStackedWidget* detailTabs_ = nullptr;
    QLabel* imagePreviewLabel_ = nullptr;
    QPlainTextEdit* previewText_ = nullptr;
    QStackedWidget* previewStack_ = nullptr;
    EvaluationReportView* evaluationReportView_ = nullptr;
    TaskArtifactPresenter* presenter_ = nullptr;
    QString selectedArtifactId_;
    QString selectedRelativePath_;
    quint64 previewGeneration_ = 0;
};
