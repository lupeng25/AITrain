#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"

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
    layout->setContentsMargins(18, 0, 18, 18);
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
    auto* inferButton = new QPushButton(QStringLiteral("选中  模型包用于推理"));
    auto* reportsButton = new QPushButton(QStringLiteral("查看评估报告"));
    auto* comparisonInferButton = new QPushButton(QStringLiteral("对比候选用于推理"));
    auto* comparisonReportButton = new QPushButton(QStringLiteral("打开候选报告"));
    connect(inferButton, &QPushButton::clicked, this, [this]() {
        if (!ModelPackageTable_ || ModelPackageTable_->selectedItems().isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("请先选择一个已验证  模型包。"));
            return;
        }
        const int row = ModelPackageTable_->selectedItems().first()->row();
        const QString modelPackageId = ModelPackageTable_->item(row, 0)
            ? ModelPackageTable_->item(row, 0)->data(Qt::UserRole).toString()
            : QString();
        if (modelPackageId.isEmpty() || !inferenceModelPackageCombo_) {
            QMessageBox::information(this, uiText("模型库"), uiText("选中行不包含可用的  模型包 ID。"));
            return;
        }
        const int comboIndex = inferenceModelPackageCombo_->findData(modelPackageId);
        if (comboIndex < 0) {
            QMessageBox::warning(this, uiText("模型库"), uiText("模型包目录已变更，请刷新模型库后重试。"));
            return;
        }
        inferenceModelPackageCombo_->setCurrentIndex(comboIndex);
        showDeploymentTab(1);
    });
    connect(reportsButton, &QPushButton::clicked, this, &MainWindow::openEvaluationReportsPage);
    comparisonInferButton->setEnabled(false);
    comparisonReportButton->setEnabled(false);
    comparisonInferButton->setToolTip(uiText("旧裸路径模型对比已停用，请使用  模型包。"));
    comparisonReportButton->setToolTip(uiText("旧评估报告路径入口已停用。"));
    actionGrid->addWidget(inferButton, 0, 0);
    actionGrid->addWidget(reportsButton, 0, 1);
    for (int column = 0; column < 2; ++column) {
        actionGrid->setColumnStretch(column, 1);
    }
    modelRegistrySummaryLabel_ = mutedLabel(uiText("推理与部署验证只使用已登记且经 Manifest 校验的  模型包；旧模型版本和评估记录仅用于迁移期审计。"));
    allowLabelToShrink(modelRegistrySummaryLabel_);
    toolbar->bodyLayout()->addWidget(actionStrip);
    toolbar->bodyLayout()->addWidget(modelRegistrySummaryLabel_);

    auto* ModelPackagePanel = new InfoPanel(QStringLiteral("已验证  模型包"));
    auto* importForm = new QFormLayout;
    importForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    importForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    modelImportSourceEdit_ = new QLineEdit;
    modelImportSourceEdit_->setPlaceholderText(QStringLiteral("选择待导入的常规模型文件（例如 .onnx）"));
    modelImportManifestEdit_ = new QLineEdit;
    modelImportManifestEdit_->setPlaceholderText(QStringLiteral("选择用户确认的  Manifest 草稿 JSON"));
    const auto makeImportPathRow = [this](QLineEdit* edit, const QString& title, const QString& filter) {
        auto* row = new QWidget;
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        auto* browseButton = new QPushButton(uiText("选择文件"));
        connect(browseButton, &QPushButton::clicked, this, [this, edit, title, filter]() {
            const QString file = QFileDialog::getOpenFileName(this, title, currentProjectPath_, filter);
            if (!file.isEmpty()) edit->setText(QDir::toNativeSeparators(file));
        });
        rowLayout->addWidget(edit, 1);
        rowLayout->addWidget(browseButton);
        return row;
    };
    importForm->addRow(QStringLiteral("模型文件"), makeImportPathRow(modelImportSourceEdit_, uiText("选择待导入模型"), QStringLiteral("Model files (*.onnx);;All files (*.*)")));
    importForm->addRow(QStringLiteral("Manifest 草稿"), makeImportPathRow(modelImportManifestEdit_, uiText("选择  Manifest 草稿"), QStringLiteral("JSON files (*.json);;All files (*.*)")));
    ModelPackagePanel->bodyLayout()->addLayout(importForm);
    auto* importActionStrip = new QFrame;
    importActionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* importActionLayout = new QHBoxLayout(importActionStrip);
    importActionLayout->setContentsMargins(10, 8, 10, 8);
    modelImportResultLabel_ = mutedLabel(uiText("导入由 Worker 执行；Manifest 草稿必须明确模型语义、张量契约、来源快照和已验证状态。导入过程将生成任务 ID 与模型 SHA-256。"));
    allowLabelToShrink(modelImportResultLabel_);
    auto* importButton = primaryButton(uiText("导入  模型包"));
    connect(importButton, &QPushButton::clicked, this, &MainWindow::importModelPackage);
    importActionLayout->addWidget(modelImportResultLabel_, 1);
    importActionLayout->addWidget(importButton);
    ModelPackagePanel->bodyLayout()->addWidget(importActionStrip);
    ModelPackageTable_ = new QTableWidget(0, 6);
    ModelPackageTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("模型包 ID")
        << QStringLiteral("模型族")
        << QStringLiteral("任务")
        << QStringLiteral("来源后端")
        << QStringLiteral("解码器")
        << QStringLiteral("登记时间"));
    configureTable(ModelPackageTable_);
    ModelPackageTable_->setWordWrap(true);
    ModelPackageTable_->verticalHeader()->setDefaultSectionSize(42);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    ModelPackagePanel->bodyLayout()->addWidget(ModelPackageTable_);

    auto* modelPanel = new InfoPanel(QStringLiteral("旧模型版本（迁移期审计）"));
    modelVersionTable_ = new QTableWidget(0, 8);
    modelVersionTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("模型")
        << QStringLiteral("版本")
        << QStringLiteral("状态")
        << QStringLiteral("Checkpoint")
        << QStringLiteral("ONNX")
        << QStringLiteral("来源任务")
        << uiText("交付摘要")
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

    auto* comparisonPanel = new InfoPanel(QStringLiteral("模型对比看板"));
    modelComparisonSummaryLabel_ = mutedLabel(QStringLiteral("选择模型版本或运行评估/基准后，这里会按主指标、延迟和限制项给出可交付排序。"));
    allowLabelToShrink(modelComparisonSummaryLabel_);
    modelComparisonTable_ = new QTableWidget(0, 8);
    modelComparisonTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("排名")
        << QStringLiteral("来源")
        << QStringLiteral("模型 / 版本")
        << QStringLiteral("任务")
        << QStringLiteral("主指标")
        << QStringLiteral("部署基准")
        << QStringLiteral("限制")
        << QStringLiteral("建议"));
    configureTable(modelComparisonTable_);
    modelComparisonTable_->setWordWrap(true);
    modelComparisonTable_->verticalHeader()->setDefaultSectionSize(44);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    modelComparisonTable_->horizontalHeader()->setSectionResizeMode(7, QHeaderView::Stretch);
    auto* comparisonActionStrip = new QFrame;
    comparisonActionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* comparisonActionLayout = new QGridLayout(comparisonActionStrip);
    comparisonActionLayout->setContentsMargins(10, 8, 10, 8);
    comparisonActionLayout->setHorizontalSpacing(10);
    comparisonActionLayout->addWidget(comparisonInferButton, 0, 0);
    comparisonActionLayout->addWidget(comparisonReportButton, 0, 1);
    for (int column = 0; column < 3; ++column) {
        comparisonActionLayout->setColumnStretch(column, 1);
    }
    comparisonPanel->bodyLayout()->addWidget(modelComparisonSummaryLabel_);
    comparisonPanel->bodyLayout()->addWidget(comparisonActionStrip);
    comparisonPanel->bodyLayout()->addWidget(modelComparisonTable_);

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
    modelWorkspaceTabs_ = new QTabWidget;
    modelWorkspaceTabs_->setObjectName(QStringLiteral("ModelWorkspaceTabs"));
    modelWorkspaceTabs_->addTab(ModelPackagePanel, uiText(" 模型包"));
    modelWorkspaceTabs_->addTab(modelPanel, uiText("旧模型版本"));
    modelWorkspaceTabs_->addTab(buildEvaluationReportsPanel(), uiText("评估报告"));
    modelWorkspaceTabs_->addTab(comparisonPanel, uiText("模型对比"));
    modelWorkspaceTabs_->addTab(pipelinePanel, uiText("流水线记录"));

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("MODEL REGISTRY"),
        uiText("模型库工作台"),
        uiText("管理模型版本、评估报告、对比和流水线记录。"),
        headerRefreshButton,
        QStringList()
            << uiText("版本模型")
            << uiText("评估报告")
            << uiText("模型对比")
            << uiText("流水线")));
    layout->addWidget(toolbar);
    layout->addWidget(modelWorkspaceTabs_, 1);
    return page;
}

QWidget* MainWindow::buildEvaluationReportsPanel()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

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
        showModelWorkspaceTab(0);
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
        << uiText("报告")
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
    connect(evaluationReportView_, &QObject::destroyed, this, [this]() {
        evaluationReportView_ = nullptr;
    });
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

    layout->addWidget(toolbar);
    layout->addWidget(splitter, 1);
    return page;
}
