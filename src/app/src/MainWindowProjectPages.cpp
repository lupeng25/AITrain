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

QWidget* MainWindow::buildDashboardPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    projectLabel_ = inlineStatusLabel(QStringLiteral("未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    gpuLabel_ = inlineStatusLabel(QStringLiteral("GPU / 运行时：未执行环境自检"));
    allowLabelToShrink(projectLabel_);
    allowLabelToShrink(gpuLabel_);

    auto* grid = new QGridLayout;
    grid->setSpacing(12);
    auto* projectCard = createMetricCard(QStringLiteral("项目"), QStringLiteral("未打开"), QStringLiteral("当前本地工作目录"));
    dashboardProjectValue_ = projectCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(projectCard, 0, 0);
    auto* datasetCard = createMetricCard(QStringLiteral("数据集"), QStringLiteral("0"), QStringLiteral("已登记并校验的数据集"));
    dashboardDatasetValue_ = datasetCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(datasetCard, 0, 1);
    auto* taskCard = createMetricCard(QStringLiteral("任务"), QStringLiteral("0"), QStringLiteral("训练、校验、导出、推理记录"));
    dashboardTaskValue_ = taskCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(taskCard, 0, 2);
    auto* modelCard = createMetricCard(QStringLiteral("模型版本"), QStringLiteral("0"), QStringLiteral("模型库 / 导出产物"));
    dashboardModelValue_ = modelCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(modelCard, 0, 3);
    auto* pluginCard = createMetricCard(QStringLiteral("插件"), QStringLiteral("0"), QStringLiteral("已加载能力插件"));
    dashboardPluginValue_ = pluginCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(pluginCard, 1, 0);
    auto* environmentCard = createMetricCard(QStringLiteral("环境"), QStringLiteral("待检测"), QStringLiteral("CUDA / TensorRT / Worker"));
    dashboardEnvironmentValue_ = environmentCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(environmentCard, 1, 1);

    auto* bottom = new QSplitter(Qt::Horizontal);

    auto* workflowPanel = new InfoPanel(QStringLiteral("下一步"));
    dashboardNextStepLabel_ = emptyStateLabel(QStringLiteral("打开项目后，按 数据集 -> 训练实验 -> 任务与产物 -> 模型导出 -> 推理验证 的顺序完成本机训练闭环。"));
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
    auto* inferenceButton = new QPushButton(QStringLiteral("推理验证"));
    connect(projectButton, &QPushButton::clicked, this, [this]() { showPage(ProjectPage, uiText("项目")); });
    connect(datasetButton, &QPushButton::clicked, this, [this]() { showPage(DatasetPage, uiText("数据集")); });
    connect(trainingButton, &QPushButton::clicked, this, [this]() { showPage(TrainingPage, uiText("训练实验")); });
    connect(artifactButton, &QPushButton::clicked, this, [this]() { showPage(TaskQueuePage, uiText("任务与产物")); });
    connect(modelRegistryButton, &QPushButton::clicked, this, [this]() { showPage(ModelRegistryPage, uiText("模型库")); });
    connect(inferenceButton, &QPushButton::clicked, this, [this]() { showPage(InferencePage, uiText("推理验证")); });
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
        << QStringLiteral("插件")
        << QStringLiteral("类型")
        << QStringLiteral("状态")
        << QStringLiteral("消息"));
    configureTable(recentTasksTable_);
    recentPanel->bodyLayout()->addWidget(recentTasksTable_);

    bottom->addWidget(workflowPanel);
    bottom->addWidget(recentPanel);
    bottom->setStretchFactor(0, 3);
    bottom->setStretchFactor(1, 4);

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
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerOpenButton = primaryButton(QStringLiteral("创建 / 打开项目"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("PROJECT CONSOLE"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("项目控制台"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("统一管理本机训练项目、SQLite 元数据、数据集目录、运行目录和模型产物目录。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(headerOpenButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* statusCaption = new QLabel(QStringLiteral("状态"));
    statusCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* policyCaption = new QLabel(QStringLiteral("目录"));
    policyCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    projectConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("未打开项目。"));
    projectConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    auto* policyStatus = inlineStatusLabel(QStringLiteral("项目会生成 datasets、runs、models 和 project.sqlite。"));
    policyStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(projectConsoleStatusLabel_);
    allowLabelToShrink(policyStatus);
    headerGrid->addWidget(statusCaption, 0, 0);
    headerGrid->addWidget(projectConsoleStatusLabel_, 0, 1);
    headerGrid->addWidget(policyCaption, 1, 0);
    headerGrid->addWidget(policyStatus, 1, 1);
    headerRoot->addLayout(headerGrid);

    auto* formPanel = new InfoPanel(QStringLiteral("项目设置"));
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFormAlignment(Qt::AlignTop);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    projectNameEdit_ = new QLineEdit(uiText("本地训练项目"));
    projectRootEdit_ = new QLineEdit(QDir::toNativeSeparators(configuredDefaultProjectPath()));
    auto* browseButton = new QPushButton(QStringLiteral("选择目录"));
    auto* createButton = primaryButton(QStringLiteral("创建 / 打开项目"));

    connect(browseButton, &QPushButton::clicked, this, [this]() {
        const QString directory = QFileDialog::getExistingDirectory(this, uiText("选择项目目录"));
        if (!directory.isEmpty()) {
            projectRootEdit_->setText(QDir::toNativeSeparators(directory));
        }
    });
    connect(headerOpenButton, &QPushButton::clicked, this, &MainWindow::createProject);
    connect(createButton, &QPushButton::clicked, this, &MainWindow::createProject);

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
    auto* projectActionHint = mutedLabel(QStringLiteral("打开项目后，数据集、任务、导出记录和环境检查都会写入 project.sqlite。"));
    allowLabelToShrink(projectActionHint);
    actionLayout->addWidget(projectActionHint, 0, 0);
    actionLayout->addWidget(createButton, 1, 0, Qt::AlignRight);
    actionLayout->setColumnStretch(0, 1);
    formPanel->bodyLayout()->addWidget(actionStrip);
    formPanel->bodyLayout()->addStretch();

    auto* summaryPanel = new InfoPanel(QStringLiteral("项目摘要"));
    auto* summaryList = new QVBoxLayout;
    summaryList->setSpacing(10);
    auto* pathCard = createCompactSummaryCard(QStringLiteral("项目路径"), QStringLiteral("未打开"), QStringLiteral("当前项目根目录"));
    projectPathSummaryLabel_ = pathCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* sqliteCard = createCompactSummaryCard(QStringLiteral("SQLite"), QStringLiteral("未连接"), QStringLiteral("项目元数据状态"));
    projectSqliteSummaryLabel_ = sqliteCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* datasetCard = createCompactSummaryCard(QStringLiteral("数据集"), QStringLiteral("0"), QStringLiteral("已登记数据集"));
    projectDatasetSummaryLabel_ = datasetCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* taskCard = createCompactSummaryCard(QStringLiteral("任务"), QStringLiteral("0"), QStringLiteral("训练、校验、导出、推理"));
    projectTaskSummaryLabel_ = taskCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* exportCard = createCompactSummaryCard(QStringLiteral("模型导出"), QStringLiteral("0"), QStringLiteral("已记录导出产物"));
    projectExportSummaryLabel_ = exportCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    summaryList->addWidget(pathCard);
    summaryList->addWidget(sqliteCard);
    summaryList->addWidget(datasetCard);
    summaryList->addWidget(taskCard);
    summaryList->addWidget(exportCard);
    summaryPanel->bodyLayout()->addLayout(summaryList);

    auto* structurePanel = new InfoPanel(QStringLiteral("标准目录结构"));
    auto* structure = new QPlainTextEdit;
    structure->setReadOnly(true);
    structure->setMaximumHeight(170);
    structure->setPlainText(QStringLiteral("datasets/\n  raw/\n  normalized/\nruns/\n  <task-id>/\nmodels/\n  exported/\nproject.sqlite"));
    structurePanel->bodyLayout()->addWidget(structure);
    structurePanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("项目页只负责创建和打开工作区；训练、导出和推理仍通过 Worker 执行。")));
    summaryPanel->bodyLayout()->addWidget(structurePanel);

    layout->addWidget(headerPanel);
    layout->addWidget(formPanel);
    layout->addWidget(summaryPanel);
    layout->addStretch();
    page->setWidget(content);
    updateProjectSummary();
    return page;
}
