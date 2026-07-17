#include "TaskArtifactPanel.h"

#include "EvaluationReportView.h"
#include "MainWindowSupport.h"

#include <QAbstractItemView>
#include <QByteArray>
#include <QFrame>
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QLabel>
#include <QFileInfo>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QScrollArea>
#include <QSize>
#include <QSizePolicy>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QVBoxLayout>

using namespace aitrain_app;

namespace {

QString formatArtifactJsonText(const QByteArray& data)
{
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(data, &error);
    if (error.error != QJsonParseError::NoError) {
        return QString::fromUtf8(data);
    }
    return QString::fromUtf8(document.toJson(QJsonDocument::Indented));
}

QString suffixFor(const QString& relativePath)
{
    return QFileInfo(relativePath).suffix().toLower();
}

} // namespace

TaskArtifactPanel::TaskArtifactPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(12);

    selectedTaskSummaryLabel_ = inlineStatusLabel(QStringLiteral("请选择一个任务查看产物、指标和工作流。"));
    selectedTaskSummaryLabel_->setObjectName(QStringLiteral("TaskDetailSummary"));
    selectedTaskSummaryLabel_->setMinimumHeight(40);
    selectedTaskSummaryLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

    artifactTable_ = new QTableWidget(0, 4);
    artifactTable_->setObjectName(QStringLiteral("TaskArtifactTable"));
    artifactTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("产物类型") << QStringLiteral("包内相对路径")
        << QStringLiteral("完整性") << QStringLiteral("提交时间"));
    configureTable(artifactTable_);
    artifactTable_->setWordWrap(true);
    artifactTable_->setMinimumHeight(170);
    artifactTable_->verticalHeader()->setDefaultSectionSize(42);
    artifactTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    artifactTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    artifactTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    artifactTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    connect(artifactTable_, &QTableWidget::itemSelectionChanged, this, &TaskArtifactPanel::updatePreviewFromSelection);

    metricTable_ = new QTableWidget(0, 4);
    metricTable_->setObjectName(QStringLiteral("TaskMetricTable"));
    metricTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("指标") << QStringLiteral("值") << QStringLiteral("发生时间") << QStringLiteral("来源"));
    configureTable(metricTable_);
    metricTable_->setMinimumHeight(160);
    metricTable_->verticalHeader()->setDefaultSectionSize(38);
    metricTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    metricTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    metricTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    metricTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);

    exportTable_ = new QTableWidget(0, 3);
    exportTable_->setObjectName(QStringLiteral("TaskWorkflowTable"));
    exportTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("步骤") << QStringLiteral("状态 / 后端") << QStringLiteral("输出 Artifact"));
    configureTable(exportTable_);
    exportTable_->setWordWrap(true);
    exportTable_->setMinimumHeight(160);
    exportTable_->verticalHeader()->setDefaultSectionSize(42);
    exportTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    exportTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    exportTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    connect(exportTable_, &QTableWidget::itemSelectionChanged, this, &TaskArtifactPanel::updatePreviewFromSelection);

    imagePreviewLabel_ = new QLabel(QStringLiteral("暂无产物预览"));
    imagePreviewLabel_->setObjectName(QStringLiteral("ArtifactPreviewCanvas"));
    imagePreviewLabel_->setAlignment(Qt::AlignCenter);
    imagePreviewLabel_->setMinimumHeight(160);
    imagePreviewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    previewText_ = new QPlainTextEdit;
    previewText_->setObjectName(QStringLiteral("ArtifactPreviewText"));
    previewText_->setReadOnly(true);
    previewText_->setMinimumHeight(120);
    previewText_->setPlainText(QStringLiteral("选择一个已提交文件后显示摘要。"));
    auto* defaultPreview = new QWidget;
    auto* defaultLayout = new QVBoxLayout(defaultPreview);
    defaultLayout->setContentsMargins(0, 0, 0, 0);
    defaultLayout->setSpacing(10);
    defaultLayout->addWidget(imagePreviewLabel_, 1);
    defaultLayout->addWidget(previewText_, 2);
    previewStack_ = new QStackedWidget;
    previewStack_->setMinimumHeight(180);
    previewStack_->addWidget(defaultPreview);

    evaluationReportView_ = new EvaluationReportView;
    auto* evaluationScroll = new QScrollArea;
    evaluationScroll->setWidget(evaluationReportView_);
    evaluationScroll->setWidgetResizable(true);
    evaluationScroll->setFrameShape(QFrame::NoFrame);
    previewStack_->addWidget(evaluationScroll);

    auto* artifactTab = new QWidget;
    auto* artifactTabLayout = new QVBoxLayout(artifactTab);
    artifactTabLayout->setContentsMargins(0, 0, 0, 0);
    artifactTabLayout->addWidget(artifactTable_);
    auto* metricTab = new QWidget;
    auto* metricTabLayout = new QVBoxLayout(metricTab);
    metricTabLayout->setContentsMargins(0, 0, 0, 0);
    metricTabLayout->addWidget(metricTable_);
    auto* exportTab = new QWidget;
    auto* exportTabLayout = new QVBoxLayout(exportTab);
    exportTabLayout->setContentsMargins(0, 0, 0, 0);
    exportTabLayout->addWidget(exportTable_);
    auto* previewTab = new QWidget;
    auto* previewTabLayout = new QVBoxLayout(previewTab);
    previewTabLayout->setContentsMargins(0, 0, 0, 0);
    previewTabLayout->addWidget(previewStack_);

    detailTabs_ = new QTabWidget;
    detailTabs_->setObjectName(QStringLiteral("TaskDetailTabs"));
    detailTabs_->addTab(artifactTab, uiText("产物"));
    detailTabs_->addTab(metricTab, uiText("指标"));
    detailTabs_->addTab(exportTab, uiText("工作流"));
    detailTabs_->addTab(previewTab, uiText("预览"));
    connect(detailTabs_, &QTabWidget::currentChanged, this, &TaskArtifactPanel::updatePreviewFromSelection);
    detailTabs_->setMinimumHeight(300);
    detailTabs_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    previewStack_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    layout->addWidget(selectedTaskSummaryLabel_);
    layout->addWidget(detailTabs_, 1);
    clear();
}

void TaskArtifactPanel::setPresenter(TaskArtifactPresenter* presenter)
{
    presenter_ = presenter;
    previewSelectedArtifact();
}

void TaskArtifactPanel::clear()
{
    setTaskSummary(uiText("请选择一个任务查看产物、指标和工作流。"));
    clearTableWithPlaceholder(artifactTable_, uiText("暂无产物"));
    clearTableWithPlaceholder(metricTable_, uiText("暂无指标"));
    clearTableWithPlaceholder(exportTable_, uiText("暂无工作流步骤"));
    selectedArtifactId_.clear();
    selectedRelativePath_.clear();
    previewSelectedArtifact();
}

void TaskArtifactPanel::setTaskSummary(const QString& summary)
{
    if (selectedTaskSummaryLabel_) selectedTaskSummaryLabel_->setText(summary);
}

void TaskArtifactPanel::setDetails(const TaskArtifactDetails& details)
{
    setTaskSummary(details.summary.isEmpty()
        ? uiText("请选择一个任务查看已提交产物、指标和工作流。") : details.summary);

    artifactTable_->setRowCount(0);
    for (const ArtifactFileItem& artifact : details.artifacts) {
        const int row = artifactTable_->rowCount();
        artifactTable_->insertRow(row);
        artifactTable_->setItem(row, 0, new QTableWidgetItem(artifact.kind));
        auto* relativePath = new QTableWidgetItem(artifact.relativePath.isEmpty()
            ? QStringLiteral("（无文件清单）") : artifact.relativePath);
        relativePath->setData(Qt::UserRole, artifact.relativePath);
        relativePath->setData(Qt::UserRole + 1, artifact.artifactId);
        artifactTable_->setItem(row, 1, relativePath);
        artifactTable_->setItem(row, 2, new QTableWidgetItem(
            QStringLiteral("SHA-256 %1 · %2 bytes")
                .arg(artifact.sha256.isEmpty() ? QStringLiteral("--") : artifact.sha256)
                .arg(artifact.byteCount)));
        artifactTable_->setItem(row, 3, new QTableWidgetItem(artifact.createdAt));
    }
    if (details.artifacts.isEmpty()) {
        clearTableWithPlaceholder(artifactTable_, uiText("暂无已提交产物"));
    } else {
        artifactTable_->selectRow(0);
    }

    metricTable_->setRowCount(0);
    for (const MetricItem& metric : details.metrics) {
        const int row = metricTable_->rowCount();
        metricTable_->insertRow(row);
        metricTable_->setItem(row, 0, new QTableWidgetItem(metric.name));
        metricTable_->setItem(row, 1, new QTableWidgetItem(QString::number(metric.value, 'g', 12)));
        metricTable_->setItem(row, 2, new QTableWidgetItem(metric.occurredAt));
        metricTable_->setItem(row, 3, new QTableWidgetItem(QStringLiteral("持久化事件")));
    }
    if (details.metrics.isEmpty()) clearTableWithPlaceholder(metricTable_, uiText("暂无指标"));

    exportTable_->setRowCount(0);
    for (const WorkflowStepItem& step : details.workflowSteps) {
        const int row = exportTable_->rowCount();
        exportTable_->insertRow(row);
        exportTable_->setItem(row, 0, new QTableWidgetItem(
            QStringLiteral("%1. %2").arg(step.ordinal + 1).arg(step.kind)));
        exportTable_->setItem(row, 1, new QTableWidgetItem(
            QStringLiteral("%1 / %2").arg(step.state,
                step.backend.isEmpty() ? QStringLiteral("--") : step.backend)));
        auto* output = new QTableWidgetItem(step.outputArtifactId.isEmpty()
            ? QStringLiteral("--") : step.outputArtifactId.left(8));
        output->setData(Qt::UserRole + 1, step.outputArtifactId);
        exportTable_->setItem(row, 2, output);
    }
    if (details.workflowSteps.isEmpty()) clearTableWithPlaceholder(exportTable_, uiText("暂无工作流步骤"));
    previewSelectedArtifact();
}

int TaskArtifactPanel::artifactRowCount() const { return artifactTable_ ? artifactTable_->rowCount() : 0; }
int TaskArtifactPanel::metricRowCount() const { return metricTable_ ? metricTable_->rowCount() : 0; }
int TaskArtifactPanel::workflowStepRowCount() const { return exportTable_ ? exportTable_->rowCount() : 0; }

void TaskArtifactPanel::configureTable(QTableWidget* table) const
{
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setStretchLastSection(true);
}

void TaskArtifactPanel::clearTableWithPlaceholder(QTableWidget* table, const QString& placeholder)
{
    if (!table) return;
    table->clearSelection();
    table->setRowCount(0);
    table->insertRow(0);
    table->setItem(0, 0, new QTableWidgetItem(placeholder));
    for (int column = 1; column < table->columnCount(); ++column)
        table->setItem(0, column, new QTableWidgetItem(QString()));
}

void TaskArtifactPanel::updatePreviewFromSelection()
{
    previewSelectedArtifact();
}

void TaskArtifactPanel::previewSelectedArtifact()
{
    if (!previewText_ || !imagePreviewLabel_ || !previewStack_) return;
    previewStack_->setCurrentIndex(0);
    if (evaluationReportView_) evaluationReportView_->clear();
    imagePreviewLabel_->clear();
    imagePreviewLabel_->setVisible(false);
    imagePreviewLabel_->setText(uiText("暂无产物预览"));
    previewText_->setVisible(true);
    previewText_->clear();

    if (!detailTabs_ || detailTabs_->currentIndex() != 0
        || !artifactTable_ || artifactTable_->selectedItems().isEmpty()) {
        imagePreviewLabel_->setVisible(true);
        previewText_->setVisible(false);
        return;
    }
    const int row = artifactTable_->selectedItems().first()->row();
    auto* item = artifactTable_->item(row, 1);
    selectedRelativePath_ = item ? item->data(Qt::UserRole).toString() : QString();
    selectedArtifactId_ = item ? item->data(Qt::UserRole + 1).toString() : QString();
    if (selectedArtifactId_.isEmpty() || selectedRelativePath_.isEmpty() || !presenter_) {
        imagePreviewLabel_->setVisible(true);
        previewText_->setVisible(false);
        return;
    }

    aitrain::ArtifactFilePreview preview;
    QString error;
    if (!presenter_->previewArtifact(selectedArtifactId_, selectedRelativePath_, &preview, &error)) {
        previewText_->setPlainText(uiText("无法读取已提交 Artifact 预览：%1").arg(error));
        return;
    }
    const QString suffix = suffixFor(preview.relativePath);
    const QString fileName = QFileInfo(preview.relativePath).fileName();
    if (suffix == QStringLiteral("json") && fileName == QStringLiteral("evaluation_report.json") && evaluationReportView_) {
        evaluationReportView_->loadReportData(preview.content, preview.relativePath);
        const QString artifactId = selectedArtifactId_;
        evaluationReportView_->setArtifactPreviewProvider(
            [this, artifactId](const QString& relativePath, QByteArray* content, QString* error) {
                aitrain::ArtifactFilePreview related;
                if (!presenter_ || !presenter_->previewArtifact(
                        artifactId, relativePath, &related, error)) {
                    return false;
                }
                if (content) *content = related.content;
                return true;
            });
        previewStack_->setCurrentIndex(1);
        return;
    }
    if (QStringList{QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("bmp"), QStringLiteral("webp")}.contains(suffix)) {
        QPixmap image;
        if (image.loadFromData(preview.content)) {
            imagePreviewLabel_->setVisible(true);
            imagePreviewLabel_->setPixmap(image.scaled(
                imagePreviewLabel_->size().boundedTo(QSize(520, 360)),
                Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        previewText_->setPlainText(uiText("图片产物\nArtifact 相对项：%1\n尺寸：%2 x %3\n大小：%4 bytes")
            .arg(preview.relativePath).arg(image.width()).arg(image.height()).arg(preview.byteCount));
        return;
    }
    const bool textLike = QStringList{QStringLiteral("json"), QStringLiteral("yaml"), QStringLiteral("yml"),
        QStringLiteral("txt"), QStringLiteral("csv"), QStringLiteral("log"), QStringLiteral("md")}.contains(suffix);
    if (textLike) {
        QString text = suffix == QStringLiteral("json")
            ? formatArtifactJsonText(preview.content) : QString::fromUtf8(preview.content);
        if (preview.truncated) text.append(QStringLiteral("\n\n[文件超过 512KB，仅显示前部内容]"));
        previewText_->setPlainText(text);
        return;
    }
    previewText_->setPlainText(uiText("已提交模型/二进制产物\nArtifact 相对项：%1\nSHA-256：%2\n大小：%3 bytes")
        .arg(preview.relativePath, preview.sha256).arg(preview.byteCount));
}
