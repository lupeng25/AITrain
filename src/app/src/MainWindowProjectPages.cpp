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

QWidget* MainWindow::buildDashboardPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    projectLabel_ = inlineStatusLabel(QStringLiteral("未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    projectLabel_->setObjectName(QStringLiteral("ProjectWorkspaceStatus"));
    gpuLabel_ = inlineStatusLabel(QStringLiteral("GPU / 运行时：未执行环境自检"));
    allowLabelToShrink(projectLabel_);
    allowLabelToShrink(gpuLabel_);

    auto* grid = new QGridLayout;
    grid->setSpacing(10);
    auto* projectCard = createMetricCard(QStringLiteral("项目"), QStringLiteral("未打开"), QStringLiteral("当前本地工作目录"));
    dashboardProjectValue_ = projectCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(projectCard, 0, 0);
    auto* datasetCard = createMetricCard(QStringLiteral("数据集"), QStringLiteral("0"), QStringLiteral(" 快照 / 数据集"));
    dashboardDatasetValue_ = datasetCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    dashboardDatasetValue_->setObjectName(QStringLiteral("DashboardDatasetSummary"));
    grid->addWidget(datasetCard, 0, 1);
    auto* taskCard = createMetricCard(QStringLiteral("任务"), QStringLiteral("0"), QStringLiteral("训练、校验、导出、推理记录"));
    dashboardTaskValue_ = taskCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    dashboardTaskValue_->setObjectName(QStringLiteral("DashboardTaskSummary"));
    grid->addWidget(taskCard, 0, 2);

    auto* bottom = new QWidget;
    auto* bottomLayout = new QHBoxLayout(bottom);
    bottomLayout->setContentsMargins(0, 0, 0, 0);
    bottomLayout->setSpacing(12);

    auto* workflowPanel = new InfoPanel(QStringLiteral("下一步"));
    dashboardNextStepLabel_ = emptyStateLabel(QStringLiteral("打开项目后，按 数据集 -> 训练实验 -> 任务与产物 -> 部署验证 的顺序完成本机训练闭环。"));
    allowLabelToShrink(dashboardNextStepLabel_);
    workflowPanel->bodyLayout()->addWidget(dashboardNextStepLabel_);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QGridLayout(actionStrip);
    actionLayout->setContentsMargins(12, 12, 12, 12);
    actionLayout->setSpacing(10);
    auto* projectButton = primaryButton(QStringLiteral("打开项目"));
    auto* datasetButton = new QPushButton(QStringLiteral("导入 / 校验数据"));
    auto* trainingButton = new QPushButton(QStringLiteral("启动训练实验"));
    auto* artifactButton = new QPushButton(QStringLiteral("查看任务与产物"));
    auto* modelRegistryButton = new QPushButton(QStringLiteral("模型库"));
    auto* inferenceButton = new QPushButton(QStringLiteral("部署验证"));
    connect(projectButton, &QPushButton::clicked, this, [this]() { showPage(ProjectPage, uiText("项目")); });
    connect(datasetButton, &QPushButton::clicked, this, [this]() { showPage(DatasetPage, uiText("数据集")); });
    connect(trainingButton, &QPushButton::clicked, this, [this]() { showPage(TrainingPage, uiText("训练实验")); });
    connect(artifactButton, &QPushButton::clicked, this, [this]() { showPage(TaskQueuePage, uiText("任务与产物")); });
    connect(modelRegistryButton, &QPushButton::clicked, this, [this]() { showPage(ModelRegistryPage, uiText("模型库")); });
    connect(inferenceButton, &QPushButton::clicked, this, [this]() { showPage(DeploymentPage, uiText("部署验证")); });
    actionLayout->addWidget(projectButton, 0, 0);
    actionLayout->addWidget(datasetButton, 0, 1);
    actionLayout->addWidget(trainingButton, 1, 0);
    actionLayout->addWidget(artifactButton, 1, 1);
    actionLayout->addWidget(modelRegistryButton, 2, 0);
    actionLayout->addWidget(inferenceButton, 2, 1);
    workflowPanel->bodyLayout()->addWidget(actionStrip);
    workflowPanel->bodyLayout()->addStretch();

    auto* recentPanel = new InfoPanel(QStringLiteral("最近任务"));
    recentTasksTable_ = new QTableWidget(0, 5);
    recentTasksTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务")
        << QStringLiteral("内置能力")
        << QStringLiteral("类型")
        << QStringLiteral("状态")
        << QStringLiteral("消息"));
    configureTable(recentTasksTable_);
    recentTasksTable_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    recentPanel->bodyLayout()->addWidget(recentTasksTable_);

    bottomLayout->addWidget(workflowPanel, 3, Qt::AlignTop);
    bottomLayout->addWidget(recentPanel, 4);

    layout->addWidget(projectLabel_);
    layout->addWidget(gpuLabel_);
    layout->addLayout(grid);
    layout->addWidget(bottom, 1);
    return page;
}

QWidget* MainWindow::buildProjectPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    auto* headerOpenButton = primaryButton(uiText("创建 / 打开项目"));
    projectOpenButton_ = headerOpenButton;

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("WorkspaceToolbar"));
    auto* headerRoot = new QHBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 10, 14, 10);
    headerRoot->setSpacing(12);
    auto* contextLabel = new QLabel(uiText("本地项目与元数据"));
    contextLabel->setObjectName(QStringLiteral("WorkspaceToolbarTitle"));
    projectConsoleStatusLabel_ = inlineStatusLabel(uiText("未打开项目。"));
    projectConsoleStatusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus"));
    auto* policyStatus = inlineStatusLabel(uiText("工作区由 .aitrain 管理，产物和元数据通过登记身份访问。"));
    policyStatus->setObjectName(QStringLiteral("WorkspaceToolbarMeta"));
    allowLabelToShrink(projectConsoleStatusLabel_);
    allowLabelToShrink(policyStatus);
    headerRoot->addWidget(contextLabel);
    headerRoot->addWidget(projectConsoleStatusLabel_);
    headerRoot->addWidget(policyStatus, 1);
    headerRoot->addWidget(headerOpenButton);

    auto* formPanel = new InfoPanel(QStringLiteral("项目设置"));
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFormAlignment(Qt::AlignTop);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    projectNameEdit_ = new QLineEdit(uiText("本地训练项目"));
    projectNameEdit_->setObjectName(QStringLiteral("ProjectNameEdit"));
    projectRootEdit_ = new QLineEdit(QDir::toNativeSeparators(configuredDefaultProjectPath()));
    projectRootEdit_->setObjectName(QStringLiteral("ProjectRootEdit"));
    auto* browseButton = new QPushButton(uiText("选择目录"));

    connect(browseButton, &QPushButton::clicked, this, [this]() {
        const QString directory = QFileDialog::getExistingDirectory(this, uiText("选择项目目录"));
        if (!directory.isEmpty()) {
            projectRootEdit_->setText(QDir::toNativeSeparators(directory));
        }
    });
    connect(headerOpenButton, &QPushButton::clicked, this, &MainWindow::createProject);

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(projectRootEdit_);
    pathLayout->addWidget(browseButton);
    form->addRow(QStringLiteral("项目名称"), projectNameEdit_);
    form->addRow(QStringLiteral("项目目录"), pathRow);
    formPanel->bodyLayout()->addLayout(form);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QGridLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setHorizontalSpacing(10);
    actionLayout->setVerticalSpacing(8);
    auto* projectActionHint = mutedLabel(uiText("打开项目后，项目摘要只读取 .aitrain/project.sqlite 中已持久化的事实。"));
    allowLabelToShrink(projectActionHint);
    actionLayout->addWidget(projectActionHint, 0, 0);
    actionLayout->setColumnStretch(0, 1);
    formPanel->bodyLayout()->addWidget(actionStrip);
    formPanel->bodyLayout()->addStretch();

    auto* summaryPanel = new InfoPanel(QStringLiteral("项目摘要"));
    auto* summaryGrid = new QGridLayout;
    summaryGrid->setHorizontalSpacing(10);
    summaryGrid->setVerticalSpacing(10);
    auto* pathCard = createCompactSummaryCard(QStringLiteral("当前项目"), QStringLiteral("未打开"), QStringLiteral("项目登记身份"));
    projectPathSummaryLabel_ = pathCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* sqliteCard = createCompactSummaryCard(QStringLiteral("SQLite"), QStringLiteral("未连接"), QStringLiteral("项目元数据状态"));
    projectSqliteSummaryLabel_ = sqliteCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* datasetCard = createCompactSummaryCard(QStringLiteral("数据集"), QStringLiteral("0"), QStringLiteral("已登记数据集"));
    projectDatasetSummaryLabel_ = datasetCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    projectDatasetSummaryLabel_->setObjectName(QStringLiteral("ProjectDatasetSummary"));
    auto* taskCard = createCompactSummaryCard(QStringLiteral("任务"), QStringLiteral("0"), QStringLiteral("训练、校验、导出、推理"));
    projectTaskSummaryLabel_ = taskCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    projectTaskSummaryLabel_->setObjectName(QStringLiteral("ProjectTaskSummary"));
    auto* exportCard = createCompactSummaryCard(QStringLiteral("模型包"), QStringLiteral("0"), QStringLiteral("已登记模型包"));
    projectExportSummaryLabel_ = exportCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    projectExportSummaryLabel_->setObjectName(QStringLiteral("ProjectModelPackageSummary"));
    summaryGrid->addWidget(pathCard, 0, 0, 1, 2);
    summaryGrid->addWidget(sqliteCard, 0, 2);
    summaryGrid->addWidget(datasetCard, 1, 0);
    summaryGrid->addWidget(taskCard, 1, 1);
    summaryGrid->addWidget(exportCard, 1, 2);
    summaryGrid->setColumnStretch(0, 1);
    summaryGrid->setColumnStretch(1, 1);
    summaryGrid->setColumnStretch(2, 1);
    summaryPanel->bodyLayout()->addLayout(summaryGrid);

    auto* structurePanel = new InfoPanel(QStringLiteral("标准目录结构"));
    auto* structure = new QPlainTextEdit;
    structure->setReadOnly(true);
    structure->setMaximumHeight(170);
    structure->setPlainText(QStringLiteral(".aitrain/\n  artifacts/\n    committed/\n    .staging/\n  project.sqlite"));
    structurePanel->bodyLayout()->addWidget(structure);
    structurePanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("项目页只负责创建和打开工作区；训练、导出和推理仍通过 Worker 执行，GUI 不打开或复制物理产物路径。")));
    summaryPanel->bodyLayout()->addWidget(structurePanel);

    layout->addWidget(headerPanel);
    layout->addWidget(formPanel);
    layout->addWidget(summaryPanel);
    layout->addStretch();
    page->setWidget(content);
    updateProjectSummary();
    return page;
}
