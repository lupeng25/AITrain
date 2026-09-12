#include "WorkbenchTranslation.h"
#include "TaskArtifactPanel.h"

#include "EvaluationReportView.h"
#include "MainWindowSupport.h"
#include "TaskArtifactTableModels.h"

#include <QAbstractItemView>
#include <QByteArray>
#include <QFrame>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QLabel>
#include <QFileInfo>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QScrollArea>
#include <QSize>
#include <QSizePolicy>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTableView>
#include <QComboBox>
#include <QSignalBlocker>
#include <QVBoxLayout>

#include <utility>

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

    selectedTaskSummaryLabel_ = inlineStatusLabel(aitrain_app::workbenchText(QStringLiteral("请选择一个任务查看产物、指标和工作流。")));
    selectedTaskSummaryLabel_->setObjectName(QStringLiteral("TaskDetailSummary"));
    selectedTaskSummaryLabel_->setTextFormat(Qt::PlainText);
    selectedTaskSummaryLabel_->setMinimumHeight(40);
    selectedTaskSummaryLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

    artifactTable_ = new QTableView;
    artifactTable_->setObjectName(QStringLiteral("TaskArtifactTable"));
    artifactModel_ = new ArtifactTableModel(artifactTable_);
    artifactTable_->setModel(artifactModel_);
    artifactTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    artifactTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    artifactTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    artifactTable_->verticalHeader()->setVisible(false);
    artifactTable_->setWordWrap(true);
    artifactTable_->setMinimumHeight(170);
    artifactTable_->verticalHeader()->setDefaultSectionSize(42);
    artifactTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    artifactTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    artifactTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    artifactTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    connect(artifactTable_->selectionModel(), &QItemSelectionModel::currentChanged,
        this, [this]() {
            if (!presenter_ || !artifactTable_->currentIndex().isValid()) return;
            presenter_->selectArtifact(artifactTable_->currentIndex()
                .data(ArtifactTableModel::ArtifactIdRole).toString());
        });

    artifactFileTable_ = new QTableView;
    artifactFileTable_->setObjectName(QStringLiteral("TaskArtifactFileTable"));
    artifactFileModel_ = new ArtifactFileTableModel(artifactFileTable_);
    artifactFileTable_->setModel(artifactFileModel_);
    artifactFileTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    artifactFileTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    artifactFileTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    artifactFileTable_->verticalHeader()->setVisible(false);
    artifactFileTable_->setWordWrap(true);
    artifactFileTable_->setMinimumHeight(170);
    artifactFileTable_->verticalHeader()->setDefaultSectionSize(42);
    artifactFileTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    artifactFileTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    artifactFileTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    artifactFileTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    connect(artifactFileTable_->selectionModel(), &QItemSelectionModel::currentChanged,
        this, [this]() { updatePreviewFromSelection(); });

    metricTable_ = new QTableView;
    metricTable_->setObjectName(QStringLiteral("TaskMetricTable"));
    metricModel_ = new MetricTableModel(metricTable_);
    metricTable_->setModel(metricModel_);
    metricTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    metricTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    metricTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    metricTable_->verticalHeader()->setVisible(false);
    metricTable_->setMinimumHeight(160);
    metricTable_->verticalHeader()->setDefaultSectionSize(38);
    metricTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    metricTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    metricTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    metricTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);

    exportTable_ = new QTableWidget(0, 3);
    exportTable_->setObjectName(QStringLiteral("TaskWorkflowTable"));
    exportTable_->setHorizontalHeaderLabels(QStringList()
        << aitrain_app::workbenchText(QStringLiteral("步骤")) << aitrain_app::workbenchText(QStringLiteral("状态 / 后端")) << aitrain_app::workbenchText(QStringLiteral("输出 Artifact")));
    configureTable(exportTable_);
    exportTable_->setWordWrap(true);
    exportTable_->setMinimumHeight(160);
    exportTable_->verticalHeader()->setDefaultSectionSize(42);
    exportTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    exportTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    exportTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    connect(exportTable_, &QTableWidget::itemSelectionChanged, this, &TaskArtifactPanel::updatePreviewFromSelection);

    imagePreviewLabel_ = new QLabel(aitrain_app::workbenchText(QStringLiteral("暂无产物预览")));
    imagePreviewLabel_->setObjectName(QStringLiteral("ArtifactPreviewCanvas"));
    imagePreviewLabel_->setAlignment(Qt::AlignCenter);
    imagePreviewLabel_->setMinimumHeight(160);
    imagePreviewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    previewText_ = new QPlainTextEdit;
    previewText_->setObjectName(QStringLiteral("ArtifactPreviewText"));
    previewText_->setReadOnly(true);
    previewText_->setMinimumHeight(120);
    previewText_->setPlainText(aitrain_app::workbenchText(QStringLiteral("选择一个已提交文件后显示摘要。")));
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
    artifactLoadMoreButton_ = new QPushButton(uiText("加载更多产物"));
    artifactLoadMoreButton_->setObjectName(QStringLiteral("ArtifactLoadMoreButton"));
    connect(artifactLoadMoreButton_, &QPushButton::clicked, this, [this]() {
        if (presenter_) presenter_->loadMoreArtifacts();
    });
    artifactTabLayout->addWidget(artifactLoadMoreButton_, 0, Qt::AlignHCenter);
    auto* artifactFileTab = new QWidget;
    auto* artifactFileTabLayout = new QVBoxLayout(artifactFileTab);
    artifactFileTabLayout->setContentsMargins(0, 0, 0, 0);
    artifactFileTabLayout->addWidget(artifactFileTable_);
    artifactFileLoadMoreButton_ = new QPushButton(uiText("加载更多文件"));
    artifactFileLoadMoreButton_->setObjectName(QStringLiteral("ArtifactFileLoadMoreButton"));
    connect(artifactFileLoadMoreButton_, &QPushButton::clicked, this, [this]() {
        if (presenter_) presenter_->loadMoreArtifactFiles();
    });
    artifactFileTabLayout->addWidget(
        artifactFileLoadMoreButton_, 0, Qt::AlignHCenter);
    auto* metricTab = new QWidget;
    auto* metricTabLayout = new QVBoxLayout(metricTab);
    metricTabLayout->setContentsMargins(0, 0, 0, 0);
    metricTabLayout->addWidget(metricTable_);
    metricLoadMoreButton_ = new QPushButton(uiText("加载更多指标"));
    metricLoadMoreButton_->setObjectName(QStringLiteral("MetricLoadMoreButton"));
    connect(metricLoadMoreButton_, &QPushButton::clicked, this, [this]() {
        if (presenter_) presenter_->loadMoreMetrics();
    });
    metricTabLayout->addWidget(metricLoadMoreButton_, 0, Qt::AlignHCenter);
    auto* exportTab = new QWidget;
    auto* exportTabLayout = new QVBoxLayout(exportTab);
    exportTabLayout->setContentsMargins(0, 0, 0, 0);
    exportTabLayout->addWidget(exportTable_);
    workflowLoadMoreButton_ = new QPushButton(uiText("加载更多工作流"));
    workflowLoadMoreButton_->setObjectName(QStringLiteral("WorkflowLoadMoreButton"));
    connect(workflowLoadMoreButton_, &QPushButton::clicked, this, [this]() {
        if (presenter_) presenter_->loadMoreWorkflows();
    });
    exportTabLayout->addWidget(workflowLoadMoreButton_, 0, Qt::AlignHCenter);
    auto* previewTab = new QWidget;
    auto* previewTabLayout = new QVBoxLayout(previewTab);
    previewTabLayout->setContentsMargins(0, 0, 0, 0);
    previewTabLayout->addWidget(previewStack_);

    detailTabs_ = new QStackedWidget;
    detailTabs_->setObjectName(QStringLiteral("TaskDetailViews"));
    detailTabs_->addWidget(artifactTab); detailTabs_->addWidget(artifactFileTab); detailTabs_->addWidget(metricTab); detailTabs_->addWidget(exportTab); detailTabs_->addWidget(previewTab);
    auto* section = new QComboBox; section->setObjectName(QStringLiteral("TaskDetailSection"));
    section->addItems({aitrain_app::workbenchText(QStringLiteral("产物目录")), aitrain_app::workbenchText(QStringLiteral("选中产物的文件")), aitrain_app::workbenchText(QStringLiteral("指标")), aitrain_app::workbenchText(QStringLiteral("工作流")), aitrain_app::workbenchText(QStringLiteral("文件预览"))});
    layout->addWidget(section, 0, Qt::AlignLeft);
    connect(section, QOverload<int>::of(&QComboBox::currentIndexChanged), detailTabs_, &QStackedWidget::setCurrentIndex);
    connect(detailTabs_, &QStackedWidget::currentChanged, section, &QComboBox::setCurrentIndex);
    connect(detailTabs_, &QStackedWidget::currentChanged, this, &TaskArtifactPanel::updatePreviewFromSelection);
    auto* filesButton = new QPushButton(aitrain_app::workbenchText(QStringLiteral("查看选中产物的文件"))); artifactTabLayout->addWidget(filesButton, 0, Qt::AlignLeft);
    connect(filesButton, &QPushButton::clicked, this, [this]() { if (artifactTable_->currentIndex().isValid()) detailTabs_->setCurrentIndex(1); });
    auto* previewButton = new QPushButton(aitrain_app::workbenchText(QStringLiteral("预览选中文件"))); artifactFileTabLayout->addWidget(previewButton, 0, Qt::AlignLeft);
    connect(previewButton, &QPushButton::clicked, this, [this]() { if (artifactFileTable_->currentIndex().isValid()) detailTabs_->setCurrentIndex(4); });
    connect(artifactTable_, &QTableView::doubleClicked, this, [this](const QModelIndex&) { detailTabs_->setCurrentIndex(1); });
    connect(artifactFileTable_, &QTableView::doubleClicked, this, [this](const QModelIndex&) { detailTabs_->setCurrentIndex(4); });
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
    if (artifactModel_) artifactModel_->setFiles({});
    if (artifactFileModel_) artifactFileModel_->setRows({});
    if (metricModel_) metricModel_->setRows({});
    if (artifactLoadMoreButton_) artifactLoadMoreButton_->setEnabled(false);
    if (artifactFileLoadMoreButton_) artifactFileLoadMoreButton_->setEnabled(false);
    if (metricLoadMoreButton_) metricLoadMoreButton_->setEnabled(false);
    if (workflowLoadMoreButton_) workflowLoadMoreButton_->setEnabled(false);
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

    const QString previousFile = artifactFileTable_->currentIndex().data(ArtifactFileTableModel::RelativePathRole).toString();
    const QSignalBlocker artifactBlocker(artifactTable_->selectionModel());
    const QSignalBlocker fileBlocker(artifactFileTable_->selectionModel());
    artifactModel_->setFiles(details.artifacts);
    artifactFileModel_->setRows(details.artifactFiles);
    for (int row = 0; row < artifactModel_->rowCount(); ++row) if (artifactModel_->index(row, 0).data(ArtifactTableModel::ArtifactIdRole).toString() == details.selectedArtifactId) artifactTable_->selectRow(row);
    for (int row = 0; row < artifactFileModel_->rowCount(); ++row) if (artifactFileModel_->index(row, 0).data(ArtifactFileTableModel::RelativePathRole).toString() == previousFile) artifactFileTable_->selectRow(row);
    metricModel_->setRows(details.metrics);
    artifactLoadMoreButton_->setEnabled(presenter_ && presenter_->hasMoreArtifacts());
    artifactFileLoadMoreButton_->setEnabled(
        presenter_ && presenter_->hasMoreArtifactFiles());
    metricLoadMoreButton_->setEnabled(presenter_ && presenter_->hasMoreMetrics());

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
    workflowLoadMoreButton_->setEnabled(presenter_ && presenter_->hasMoreWorkflows());
    previewSelectedArtifact();
}

int TaskArtifactPanel::artifactRowCount() const { return artifactModel_ ? artifactModel_->rowCount() : 0; }
int TaskArtifactPanel::metricRowCount() const { return metricModel_ ? metricModel_->rowCount() : 0; }
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
    ++previewGeneration_;
    if (previewGeneration_ == 0) ++previewGeneration_;
    const quint64 generation = previewGeneration_;
    if (!previewText_ || !imagePreviewLabel_ || !previewStack_) return;
    previewStack_->setCurrentIndex(0);
    if (evaluationReportView_) evaluationReportView_->clear();
    imagePreviewLabel_->clear();
    imagePreviewLabel_->setVisible(false);
    imagePreviewLabel_->setText(uiText("暂无产物预览"));
    previewText_->setVisible(true);
    previewText_->clear();
    selectedArtifactId_.clear();
    selectedRelativePath_.clear();

    if (!detailTabs_ || detailTabs_->currentIndex() != 4
        || !artifactFileTable_ || !artifactFileTable_->currentIndex().isValid()) {
        imagePreviewLabel_->setVisible(true);
        previewText_->setVisible(false);
        return;
    }
    const QModelIndex selected = artifactFileTable_->currentIndex();
    selectedRelativePath_ =
        selected.data(ArtifactFileTableModel::RelativePathRole).toString();
    selectedArtifactId_ =
        selected.data(ArtifactFileTableModel::ArtifactIdRole).toString();
    const qint64 byteCount =
        selected.data(ArtifactFileTableModel::ByteCountRole).toLongLong();
    if (selectedArtifactId_.isEmpty() || selectedRelativePath_.isEmpty()) {
        imagePreviewLabel_->setVisible(true);
        previewText_->setVisible(false);
        return;
    }
    if (!presenter_) {
        imagePreviewLabel_->setVisible(true);
        previewText_->setVisible(true);
        previewText_->setPlainText(uiText("Artifact 预览查询服务不可用。"));
        return;
    }

    previewText_->setPlainText(uiText("正在后台读取并校验 Artifact：%1\n大小：%2 bytes")
        .arg(selectedRelativePath_).arg(byteCount));
    const QString requestedArtifactId = selectedArtifactId_;
    const QString requestedRelativePath = selectedRelativePath_;
    QPointer<TaskArtifactPanel> self(this);
    QString requestError;
    if (!presenter_->previewArtifactAsync(requestedArtifactId, requestedRelativePath, this,
        [self, generation, requestedArtifactId, requestedRelativePath]
        (bool success, aitrain::ArtifactFilePreview preview, QString error) {
            if (!self || self->previewGeneration_ != generation
                || self->selectedArtifactId_ != requestedArtifactId
                || self->selectedRelativePath_ != requestedRelativePath) {
                return;
            }
            if (!success) {
                self->previewText_->setPlainText(uiText("无法读取已提交 Artifact 预览：%1").arg(error));
                return;
            }
            const QString suffix = suffixFor(preview.relativePath);
            const QString fileName = QFileInfo(preview.relativePath).fileName();
            if (suffix == QStringLiteral("json") && fileName == QStringLiteral("evaluation_report.json")
                && self->evaluationReportView_) {
                self->evaluationReportView_->loadReportData(preview.content, preview.relativePath);
                const QString artifactId = requestedArtifactId;
                QPointer<TaskArtifactPanel> panel = self;
                QPointer<EvaluationReportView> reportView = self->evaluationReportView_;
                reportView->setArtifactPreviewProvider(
                    [panel, reportView, artifactId](const QString& relativePath,
                        EvaluationReportView::ArtifactPreviewCallback callback) {
                        if (!panel || !reportView || !panel->presenter_) {
                            callback(false, {}, aitrain_app::workbenchText(QStringLiteral("Artifact 预览查询服务不可用。")));
                            return;
                        }
                        QString requestError;
                        const auto callbackForWorker = [callback]
                            (bool ok, aitrain::ArtifactFilePreview related, QString error) mutable {
                                callback(ok, std::move(related.content), std::move(error));
                            };
                        if (!panel->presenter_->previewArtifactAsync(artifactId, relativePath,
                                reportView,
                                callbackForWorker, 4 * 1024 * 1024, &requestError)) {
                            callback(false, {}, requestError);
                        }
                    });
                self->previewStack_->setCurrentIndex(1);
                return;
            }
            if (QStringList{QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"),
                    QStringLiteral("bmp"), QStringLiteral("webp")}.contains(suffix)) {
                QPixmap image;
                if (image.loadFromData(preview.content)) {
                    self->imagePreviewLabel_->setVisible(true);
                    self->imagePreviewLabel_->setPixmap(image.scaled(
                        self->imagePreviewLabel_->size().boundedTo(QSize(520, 360)),
                        Qt::KeepAspectRatio, Qt::SmoothTransformation));
                } else {
                    self->imagePreviewLabel_->setVisible(true);
                    self->imagePreviewLabel_->setText(uiText("图片产物无法解码。"));
                }
                self->previewText_->setPlainText(uiText("图片产物\nArtifact 相对项：%1\n尺寸：%2 x %3\n大小：%4 bytes")
                    .arg(preview.relativePath).arg(image.width()).arg(image.height()).arg(preview.byteCount));
                return;
            }
            const bool textLike = QStringList{QStringLiteral("json"), QStringLiteral("yaml"),
                QStringLiteral("yml"), QStringLiteral("txt"), QStringLiteral("csv"),
                QStringLiteral("log"), QStringLiteral("md")}.contains(suffix);
            if (textLike) {
                QString text = suffix == QStringLiteral("json")
                    ? formatArtifactJsonText(preview.content) : QString::fromUtf8(preview.content);
                if (preview.truncated) text.append(aitrain_app::workbenchText(QStringLiteral("\n\n[文件超过 4MB，仅显示前部内容]")));
                self->previewText_->setPlainText(text);
                return;
            }
            self->previewText_->setPlainText(uiText("已提交模型/二进制产物\nArtifact 相对项：%1\nSHA-256：%2\n大小：%3 bytes")
                .arg(preview.relativePath, preview.sha256).arg(preview.byteCount));
        }, 4 * 1024 * 1024, &requestError)) {
        // Metadata 快照阶段失败时不会进入线程池，直接呈现可操作错误。
        previewText_->setPlainText(uiText("无法读取已提交 Artifact 预览：%1").arg(requestError));
    }
}
