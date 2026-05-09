#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildModelRegistryPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerRefreshButton = primaryButton(QStringLiteral("刷新模型库"));
    connect(headerRefreshButton, &QPushButton::clicked, this, &MainWindow::refreshModelRegistry);

    auto* toolbar = new InfoPanel(QStringLiteral("模型库"));
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionGrid = new QGridLayout(actionStrip);
    actionGrid->setContentsMargins(10, 8, 10, 8);
    actionGrid->setHorizontalSpacing(10);
    actionGrid->setVerticalSpacing(8);
    auto* refreshButton = primaryButton(QStringLiteral("刷新模型库"));
    auto* inferButton = new QPushButton(QStringLiteral("选中模型用于推理"));
    auto* exportButton = new QPushButton(QStringLiteral("选中模型用于导出"));
    auto* pipelineButton = new QPushButton(QStringLiteral("执行本地流水线"));
    auto* reportsButton = new QPushButton(QStringLiteral("查看评估报告"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshModelRegistry);
    connect(inferButton, &QPushButton::clicked, this, [this]() {
        if (!modelVersionTable_ || modelVersionTable_->selectedItems().isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("请先选择一个模型版本。"));
            return;
        }
        const int row = modelVersionTable_->selectedItems().first()->row();
        QString path = modelVersionTable_->item(row, 4) ? modelVersionTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        if (path.isEmpty()) {
            path = modelVersionTable_->item(row, 3) ? modelVersionTable_->item(row, 3)->data(Qt::UserRole).toString() : QString();
        }
        if (path.isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("选中模型版本没有可用 checkpoint 或 ONNX 路径。"));
            return;
        }
        if (inferenceCheckpointEdit_) {
            inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        }
        showPage(InferencePage, uiText("推理验证"));
    });
    connect(exportButton, &QPushButton::clicked, this, [this]() {
        if (!modelVersionTable_ || modelVersionTable_->selectedItems().isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("请先选择一个模型版本。"));
            return;
        }
        const int row = modelVersionTable_->selectedItems().first()->row();
        QString path = modelVersionTable_->item(row, 3) ? modelVersionTable_->item(row, 3)->data(Qt::UserRole).toString() : QString();
        if (path.isEmpty()) {
            path = modelVersionTable_->item(row, 4) ? modelVersionTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        }
        if (path.isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("选中模型版本没有可用 checkpoint 或 ONNX 路径。"));
            return;
        }
        if (conversionCheckpointEdit_) {
            conversionCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        }
        showPage(ConversionPage, uiText("模型导出"));
    });
    connect(pipelineButton, &QPushButton::clicked, this, &MainWindow::runLocalPipelinePlanFromCurrentDataset);
    connect(reportsButton, &QPushButton::clicked, this, &MainWindow::openEvaluationReportsPage);
    actionGrid->addWidget(refreshButton, 0, 0);
    actionGrid->addWidget(inferButton, 0, 1);
    actionGrid->addWidget(exportButton, 0, 2);
    actionGrid->addWidget(pipelineButton, 1, 0);
    actionGrid->addWidget(reportsButton, 1, 1);
    for (int column = 0; column < 3; ++column) {
        actionGrid->setColumnStretch(column, 1);
    }
    modelRegistrySummaryLabel_ = mutedLabel(QStringLiteral("训练产物可从“任务与产物”注册为模型版本；评估报告已拆分到独立页面，模型库聚焦版本管理、导出和推理入口。"));
    allowLabelToShrink(modelRegistrySummaryLabel_);
    toolbar->bodyLayout()->addWidget(actionStrip);
    toolbar->bodyLayout()->addWidget(modelRegistrySummaryLabel_);

    auto* splitter = new QSplitter(Qt::Vertical);

    auto* modelPanel = new InfoPanel(QStringLiteral("模型版本"));
    modelVersionTable_ = new QTableWidget(0, 8);
    modelVersionTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("模型")
        << QStringLiteral("版本")
        << QStringLiteral("状态")
        << QStringLiteral("Checkpoint")
        << QStringLiteral("ONNX")
        << QStringLiteral("来源任务")
        << QStringLiteral("交付摘要")
        << QStringLiteral("更新时间"));
    configureTable(modelVersionTable_);
    modelVersionTable_->setWordWrap(true);
    modelVersionTable_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    modelVersionTable_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    modelVersionTable_->verticalHeader()->setDefaultSectionSize(42);
    modelVersionTable_->horizontalHeader()->setStretchLastSection(false);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::Stretch);
    modelVersionTable_->horizontalHeader()->setSectionResizeMode(7, QHeaderView::ResizeToContents);
    modelPanel->bodyLayout()->addWidget(modelVersionTable_);

    auto* pipelinePanel = new InfoPanel(QStringLiteral("流水线记录"));
    pipelineRunTable_ = new QTableWidget(0, 5);
    pipelineRunTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("名称")
        << QStringLiteral("模板")
        << QStringLiteral("状态")
        << QStringLiteral("摘要")
        << QStringLiteral("更新时间"));
    configureTable(pipelineRunTable_);
    pipelineRunTable_->setWordWrap(true);
    pipelineRunTable_->verticalHeader()->setDefaultSectionSize(42);
    pipelineRunTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    pipelineRunTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    pipelineRunTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    pipelineRunTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    pipelineRunTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    pipelinePanel->bodyLayout()->addWidget(pipelineRunTable_);
    splitter->addWidget(modelPanel);
    splitter->addWidget(pipelinePanel);
    splitter->setChildrenCollapsible(false);
    splitter->setStretchFactor(0, 4);
    splitter->setStretchFactor(1, 2);
    splitter->setSizes(QList<int>() << 520 << 260);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("MODEL REGISTRY"),
        QStringLiteral("模型库工作台"),
        QStringLiteral("管理训练产物注册后的模型版本、来源 lineage、交付摘要和本地流水线入口。"),
        headerRefreshButton,
        QStringList()
            << QStringLiteral("Versioned Models")
            << QStringLiteral("Lineage")
            << QStringLiteral("Export")
            << QStringLiteral("Inference")));
    layout->addWidget(toolbar);
    layout->addWidget(splitter, 1);
    return page;
}

QWidget* MainWindow::buildEvaluationReportsPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerRefreshButton = primaryButton(QStringLiteral("刷新评估报告"));
    connect(headerRefreshButton, &QPushButton::clicked, this, &MainWindow::refreshModelRegistry);

    auto* toolbar = new InfoPanel(QStringLiteral("评估报告"));
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* row = new QHBoxLayout(actionStrip);
    row->setContentsMargins(10, 8, 10, 8);
    row->setSpacing(10);
    auto* refreshButton = primaryButton(QStringLiteral("刷新评估报告"));
    auto* backToModelsButton = new QPushButton(QStringLiteral("查看模型库"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshModelRegistry);
    connect(backToModelsButton, &QPushButton::clicked, this, [this]() {
        showPage(ModelRegistryPage, uiText("模型库"));
    });
    row->addWidget(refreshButton);
    row->addWidget(backToModelsButton);
    row->addStretch();
    toolbar->bodyLayout()->addWidget(actionStrip);
    auto* reportHint = mutedLabel(QStringLiteral("集中查看最近评估报告、任务类型、报告路径和详细可视化结果；模型版本管理保留在“模型库”。"));
    allowLabelToShrink(reportHint);
    toolbar->bodyLayout()->addWidget(reportHint);

    auto* splitter = new QSplitter(Qt::Vertical);

    auto* reportPanel = new InfoPanel(QStringLiteral("最近评估报告"));
    evaluationReportTable_ = new QTableWidget(0, 5);
    evaluationReportTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务")
        << QStringLiteral("类型")
        << QStringLiteral("模型")
        << QStringLiteral("报告")
        << QStringLiteral("时间"));
    configureTable(evaluationReportTable_);
    evaluationReportTable_->setWordWrap(true);
    evaluationReportTable_->verticalHeader()->setDefaultSectionSize(42);
    evaluationReportTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    evaluationReportTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    evaluationReportTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    evaluationReportTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    evaluationReportTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    connect(evaluationReportTable_, &QTableWidget::itemSelectionChanged, this, &MainWindow::updateSelectedEvaluationReportDetails);
    reportPanel->bodyLayout()->addWidget(evaluationReportTable_);

    auto* reportDetailPanel = new InfoPanel(QStringLiteral("评估报告详情"));
    evaluationReportView_ = new EvaluationReportView;
    auto* evaluationReportScroll = new QScrollArea;
    evaluationReportScroll->setWidget(evaluationReportView_);
    evaluationReportScroll->setWidgetResizable(true);
    evaluationReportScroll->setFrameShape(QFrame::NoFrame);
    evaluationReportScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    evaluationReportScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    reportDetailPanel->bodyLayout()->addWidget(evaluationReportScroll);

    splitter->addWidget(reportPanel);
    splitter->addWidget(reportDetailPanel);
    splitter->setChildrenCollapsible(false);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 3);
    splitter->setSizes(QList<int>() << 260 << 560);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("EVALUATION REPORTS"),
        QStringLiteral("评估报告工作台"),
        QStringLiteral("集中查看模型评估摘要、关键指标、分类别表现、混淆矩阵、错误样本和 overlay 详情。"),
        headerRefreshButton,
        QStringList()
            << QStringLiteral("Metrics")
            << QStringLiteral("Per-class")
            << QStringLiteral("Confusion Matrix")
            << QStringLiteral("Error Overlay")));
    layout->addWidget(toolbar);
    layout->addWidget(splitter, 1);
    return page;
}
