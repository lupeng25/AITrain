#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidgetItem>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

using namespace aitrain_app;

QWidget* MainWindow::buildTopBar()
{
    auto* topBar = new QFrame;
    topBar->setObjectName(QStringLiteral("TopBar"));
    topBar->setFixedHeight(72);

    auto* layout = new QHBoxLayout(topBar);
    layout->setContentsMargins(22, 10, 22, 10);
    layout->setSpacing(16);

    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    pageTitle_ = new QLabel(tr("首页"));
    pageTitle_->setObjectName(QStringLiteral("PageTitle"));
    pageCaption_ = new QLabel;
    pageCaption_->setObjectName(QStringLiteral("PageCaption"));
    titleLayout->addWidget(pageTitle_);
    titleLayout->addWidget(pageCaption_);

    headerProjectLabel_ = new QLabel(tr("项目：未打开"));
    headerProjectLabel_->setObjectName(QStringLiteral("MutedText"));
    workerPill_ = new StatusPill;
    workerPill_->setStatus(tr("Worker 空闲"), StatusPill::Tone::Neutral);
    pluginPill_ = new StatusPill;
    gpuPill_ = new StatusPill;
    gpuPill_->setStatus(tr("GPU 未检测"), StatusPill::Tone::Warning);
    licensePill_ = new StatusPill;
    licensePill_->setStatus(licenseOwner_.isEmpty()
            ? tr("已注册")
            : tr("授权：%1").arg(licenseOwner_),
        StatusPill::Tone::Success);
    licensePill_->setToolTip(licenseExpiry_.isEmpty()
            ? tr("离线授权已验证")
            : tr("授权有效期：%1").arg(licenseExpiry_));
    auto* languageSwitch = new QFrame;
    languageSwitch->setObjectName(QStringLiteral("LanguageSwitch"));
    auto* languageLayout = new QHBoxLayout(languageSwitch);
    languageLayout->setContentsMargins(2, 2, 2, 2);
    languageLayout->setSpacing(0);
    topBarZhLanguageButton_ = new QToolButton;
    topBarZhLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    topBarZhLanguageButton_->setText(QStringLiteral("中"));
    topBarZhLanguageButton_->setCheckable(true);
    topBarZhLanguageButton_->setCursor(Qt::PointingHandCursor);
    topBarZhLanguageButton_->setToolTip(uiText("切换到中文，重启后生效"));
    topBarEnLanguageButton_ = new QToolButton;
    topBarEnLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    topBarEnLanguageButton_->setText(QStringLiteral("EN"));
    topBarEnLanguageButton_->setCheckable(true);
    topBarEnLanguageButton_->setCursor(Qt::PointingHandCursor);
    topBarEnLanguageButton_->setToolTip(uiText("切换到英文，重启后生效"));
    updateLanguageButtonState();
    languageLayout->addWidget(topBarZhLanguageButton_);
    languageLayout->addWidget(topBarEnLanguageButton_);
    connect(topBarZhLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("zh_CN"));
    });
    connect(topBarEnLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("en_US"));
    });

    layout->addWidget(titleBlock, 1);
    layout->addWidget(headerProjectLabel_);
    layout->addWidget(workerPill_);
    layout->addWidget(pluginPill_);
    layout->addWidget(gpuPill_);
    layout->addWidget(licensePill_);
    layout->addWidget(languageSwitch);
    return topBar;
}

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

QWidget* MainWindow::buildDatasetPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerValidateButton = primaryButton(QStringLiteral("校验数据集"));
    connect(headerValidateButton, &QPushButton::clicked, this, &MainWindow::validateDataset);

    auto* inputPanel = new InfoPanel(QStringLiteral("数据集操作"));
    auto* form = new QFormLayout;
    datasetPathEdit_ = new QLineEdit;
    auto* browseButton = new QPushButton(QStringLiteral("选择数据集"));
    connect(browseButton, &QPushButton::clicked, this, &MainWindow::browseDataset);

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(datasetPathEdit_);
    pathLayout->addWidget(browseButton);

    datasetFormatCombo_ = new QComboBox;
    connect(datasetFormatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        currentDatasetFormat_ = currentDatasetFormat();
        currentDatasetValid_ = false;
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
    });
    auto* validateButton = primaryButton(QStringLiteral("校验数据集"));
    connect(validateButton, &QPushButton::clicked, this, &MainWindow::validateDataset);
    form->addRow(QStringLiteral("数据集目录"), pathRow);
    form->addRow(QStringLiteral("格式"), datasetFormatCombo_);

    splitOutputEdit_ = new QLineEdit;
    splitOutputEdit_->setPlaceholderText(QStringLiteral("默认输出到当前项目 datasets/normalized"));
    splitTrainRatioEdit_ = new QLineEdit(QStringLiteral("0.8"));
    splitValRatioEdit_ = new QLineEdit(QStringLiteral("0.2"));
    splitTestRatioEdit_ = new QLineEdit(QStringLiteral("0.0"));
    splitSeedEdit_ = new QLineEdit(QStringLiteral("42"));
    auto* splitButton = new QPushButton(QStringLiteral("划分数据集"));
    connect(splitButton, &QPushButton::clicked, this, &MainWindow::splitDataset);
    auto* curateButton = new QPushButton(QStringLiteral("生成质量报告"));
    connect(curateButton, &QPushButton::clicked, this, &MainWindow::curateDataset);
    auto* snapshotButton = new QPushButton(QStringLiteral("创建数据快照"));
    connect(snapshotButton, &QPushButton::clicked, this, &MainWindow::createDatasetSnapshot);
    auto* openFixListButton = new QPushButton(QStringLiteral("打开问题清单"));
    connect(openFixListButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityFixList);
    auto* fixWithXAnyButton = new QPushButton(QStringLiteral("X-AnyLabeling 修复"));
    connect(fixWithXAnyButton, &QPushButton::clicked, this, &MainWindow::launchXAnyLabelingForQualityFix);
    auto* ratioRow = new QWidget;
    auto* ratioLayout = new QHBoxLayout(ratioRow);
    ratioLayout->setContentsMargins(0, 0, 0, 0);
    ratioLayout->addWidget(splitTrainRatioEdit_);
    ratioLayout->addWidget(splitValRatioEdit_);
    ratioLayout->addWidget(splitTestRatioEdit_);
    ratioLayout->addWidget(splitSeedEdit_);
    form->addRow(QStringLiteral("输出目录"), splitOutputEdit_);
    form->addRow(QStringLiteral("train / val / test / seed"), ratioRow);
    inputPanel->bodyLayout()->addLayout(form);
    auto* datasetActionStrip = new QFrame;
    datasetActionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* datasetActionGrid = new QGridLayout(datasetActionStrip);
    datasetActionGrid->setContentsMargins(10, 8, 10, 8);
    datasetActionGrid->setHorizontalSpacing(10);
    datasetActionGrid->setVerticalSpacing(8);
    datasetActionGrid->addWidget(validateButton, 0, 0);
    datasetActionGrid->addWidget(splitButton, 0, 1);
    datasetActionGrid->addWidget(snapshotButton, 0, 2);
    datasetActionGrid->addWidget(curateButton, 1, 0);
    datasetActionGrid->addWidget(openFixListButton, 1, 1);
    datasetActionGrid->addWidget(fixWithXAnyButton, 1, 2);
    for (int column = 0; column < 3; ++column) {
        datasetActionGrid->setColumnStretch(column, 1);
    }
    inputPanel->bodyLayout()->addWidget(datasetActionStrip);

    auto* splitter = new QSplitter(Qt::Horizontal);
    auto* resultPanel = new InfoPanel(QStringLiteral("所选数据集详情"));
    datasetDetailLabel_ = inlineStatusLabel(QStringLiteral("选择或导入数据集后显示格式、样本数、校验状态和最近报告。"));
    validationSummaryLabel_ = mutedLabel(QStringLiteral("请选择数据集目录和格式，然后执行校验。"));
    allowLabelToShrink(datasetDetailLabel_);
    allowLabelToShrink(validationSummaryLabel_);
    validationIssuesTable_ = new QTableWidget(0, 5);
    validationIssuesTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("级别")
        << QStringLiteral("代码")
        << QStringLiteral("文件")
        << QStringLiteral("行号")
        << QStringLiteral("说明"));
    configureTable(validationIssuesTable_);
    validationIssuesTable_->verticalHeader()->setDefaultSectionSize(38);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    validationOutput_ = new QPlainTextEdit;
    validationOutput_->setReadOnly(true);
    validationOutput_->setMinimumHeight(130);
    validationOutput_->setPlainText(QStringLiteral("校验报告 JSON 会显示在这里。"));
    resultPanel->bodyLayout()->addWidget(datasetDetailLabel_);
    resultPanel->bodyLayout()->addWidget(validationSummaryLabel_);
    resultPanel->bodyLayout()->addWidget(validationIssuesTable_, 2);
    resultPanel->bodyLayout()->addWidget(validationOutput_);

    auto* toolsPanel = new InfoPanel(QStringLiteral("数据集库与样本预览"));
    datasetListTable_ = new QTableWidget(0, 5);
    datasetListTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("数据集")
        << QStringLiteral("格式")
        << QStringLiteral("状态")
        << QStringLiteral("样本")
        << QStringLiteral("路径"));
    configureTable(datasetListTable_);
    datasetListTable_->setWordWrap(true);
    datasetListTable_->verticalHeader()->setDefaultSectionSize(40);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    connect(datasetListTable_, &QTableWidget::itemSelectionChanged, this, [this]() {
        if (!datasetListTable_ || datasetListTable_->selectedItems().isEmpty()) {
            return;
        }
        const int row = datasetListTable_->selectedItems().first()->row();
        const QString path = datasetListTable_->item(row, 4) ? datasetListTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        const QString format = datasetListTable_->item(row, 1) ? datasetListTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
        if (!path.isEmpty()) {
            datasetPathEdit_->setText(QDir::toNativeSeparators(path));
            const int formatIndex = datasetFormatCombo_->findData(format);
            if (formatIndex >= 0) {
                datasetFormatCombo_->setCurrentIndex(formatIndex);
            }
            currentDatasetPath_ = path;
            currentDatasetFormat_ = format;
            currentDatasetValid_ = datasetListTable_->item(row, 2)
                && datasetListTable_->item(row, 2)->data(Qt::UserRole).toString() == QStringLiteral("valid");
            updateTrainingSelectionSummary();
            refreshTrainingDefaults();
        }
    });
    datasetPreviewTable_ = new QTableWidget(0, 2);
    datasetPreviewTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("样本") << QStringLiteral("标签 / 说明"));
    configureTable(datasetPreviewTable_);
    datasetPreviewTable_->setWordWrap(true);
    datasetPreviewTable_->verticalHeader()->setDefaultSectionSize(40);
    datasetPreviewTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetPreviewTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    auto* annotationPanel = new QGroupBox(QStringLiteral("外部标注工具"));
    annotationPanel->setMinimumHeight(136);
    annotationPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    auto* annotationLayout = new QVBoxLayout(annotationPanel);
    annotationLayout->setContentsMargins(10, 12, 10, 8);
    annotationLayout->setSpacing(6);
    auto* annotationSummary = mutedLabel(QStringLiteral("X-AnyLabeling：检测导出 YOLO bbox，分割导出 YOLO polygon；PaddleOCR 使用 det_gt / rec_gt + dict。"));
    annotationToolStatusLabel_ = inlineStatusLabel(xAnyLabelingStatusText());
    allowLabelToShrink(annotationSummary);
    allowLabelToShrink(annotationToolStatusLabel_);
    annotationSummary->setMinimumHeight(28);
    annotationToolStatusLabel_->setMinimumHeight(28);
    annotationToolStatusLabel_->setToolTip(QDir::toNativeSeparators(resolvedXAnyLabelingProgram()));
    auto* launchAnnotationToolButton = new QPushButton(QStringLiteral("启动 X-AnyLabeling"));
    auto* refreshAnnotationStatusButton = new QPushButton(QStringLiteral("检测状态"));
    auto* refreshDatasetAfterAnnotationButton = new QPushButton(QStringLiteral("标注后刷新 / 重新校验"));
    auto* openDatasetDirButton = new QPushButton(QStringLiteral("打开数据目录"));
    connect(refreshAnnotationStatusButton, &QPushButton::clicked, this, &MainWindow::updateAnnotationToolStatus);
    connect(refreshDatasetAfterAnnotationButton, &QPushButton::clicked, this, &MainWindow::refreshAfterAnnotation);
    connect(openDatasetDirButton, &QPushButton::clicked, this, [this]() {
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        if (datasetPath.isEmpty()) {
            QMessageBox::information(this, uiText("标注工具"), uiText("请先选择数据集目录。"));
            return;
        }
        QDesktopServices::openUrl(QUrl::fromLocalFile(datasetPath));
    });
    connect(launchAnnotationToolButton, &QPushButton::clicked, this, [this]() {
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        if (datasetPath.isEmpty()) {
            QMessageBox::information(this, uiText("标注工具"), uiText("请先选择数据集目录。"));
            return;
        }
        const QString program = resolvedXAnyLabelingProgram();
        if (program.isEmpty()) {
            updateAnnotationToolStatus();
            QMessageBox::warning(this,
                QStringLiteral("X-AnyLabeling"),
                uiText("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或将 X-AnyLabeling.exe 放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"));
            return;
        }
        const QStringList arguments = {QStringLiteral("--filename"), datasetPath, QStringLiteral("--no-auto-update-check")};
        if (QProcess::startDetached(program, arguments)) {
            updateAnnotationToolStatus();
            statusBar()->showMessage(uiText("已启动 X-AnyLabeling：%1").arg(QDir::toNativeSeparators(datasetPath)), 4000);
            return;
        }
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(program)));
    });
    auto* annotationActionRow = new QWidget;
    auto* annotationActionGrid = new QGridLayout(annotationActionRow);
    annotationActionGrid->setContentsMargins(0, 0, 0, 0);
    annotationActionGrid->setHorizontalSpacing(10);
    annotationActionGrid->setVerticalSpacing(0);
    annotationActionGrid->addWidget(launchAnnotationToolButton, 0, 0);
    annotationActionGrid->addWidget(refreshAnnotationStatusButton, 0, 1);
    annotationActionGrid->addWidget(refreshDatasetAfterAnnotationButton, 0, 2);
    annotationActionGrid->addWidget(openDatasetDirButton, 0, 3);
    for (int column = 0; column < 4; ++column) {
        annotationActionGrid->setColumnStretch(column, 1);
    }
    annotationLayout->addWidget(annotationSummary);
    annotationLayout->addWidget(annotationToolStatusLabel_);
    annotationLayout->addWidget(annotationActionRow);
    auto* datasetLibraryTab = new QWidget;
    auto* datasetLibraryLayout = new QVBoxLayout(datasetLibraryTab);
    datasetLibraryLayout->setContentsMargins(0, 0, 0, 0);
    datasetLibraryLayout->setSpacing(10);
    auto* datasetLibraryCaption = mutedLabel(QStringLiteral("已登记数据集"));
    allowLabelToShrink(datasetLibraryCaption);
    datasetLibraryLayout->addWidget(datasetLibraryCaption);
    datasetLibraryLayout->addWidget(datasetListTable_, 1);
    datasetLibraryLayout->addWidget(annotationPanel);

    auto* samplePreviewTab = new QWidget;
    auto* samplePreviewLayout = new QVBoxLayout(samplePreviewTab);
    samplePreviewLayout->setContentsMargins(0, 0, 0, 0);
    samplePreviewLayout->setSpacing(10);
    auto* samplePreviewHint = mutedLabel(QStringLiteral("划分会复制到新目录，不修改原始数据；支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec。"));
    allowLabelToShrink(samplePreviewHint);
    samplePreviewLayout->addWidget(datasetPreviewTable_, 1);
    samplePreviewLayout->addWidget(samplePreviewHint);

    auto* datasetToolsTabs = new QTabWidget;
    datasetToolsTabs->setObjectName(QStringLiteral("DatasetToolsTabs"));
    datasetToolsTabs->addTab(datasetLibraryTab, uiText("数据集库"));
    datasetToolsTabs->addTab(samplePreviewTab, uiText("样本预览"));
    toolsPanel->bodyLayout()->addWidget(datasetToolsTabs, 1);

    splitter->addWidget(toolsPanel);
    splitter->addWidget(resultPanel);
    splitter->setChildrenCollapsible(false);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);
    splitter->setSizes(QList<int>() << 480 << 620);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("DATASET VALIDATION"),
        QStringLiteral("数据集工作台"),
        QStringLiteral("导入、校验、划分、快照和标注工具入口统一在这里；训练前必须先完成当前格式校验。"),
        headerValidateButton,
        QStringList()
            << QStringLiteral("YOLO BBox")
            << QStringLiteral("YOLO Polygon")
            << QStringLiteral("PaddleOCR Det")
            << QStringLiteral("PaddleOCR Rec")));
    layout->addWidget(inputPanel);
    layout->addWidget(splitter, 1);
    page->setWidget(content);
    return page;
}

QWidget* MainWindow::buildTrainingPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(12);

    pluginCombo_ = new QComboBox;
    taskTypeCombo_ = new QComboBox;
    trainingBackendCombo_ = new QComboBox;
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("ultralytics_yolo_detect")), QStringLiteral("ultralytics_yolo_detect"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("ultralytics_yolo_segment")), QStringLiteral("ultralytics_yolo_segment"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_det_official")), QStringLiteral("paddleocr_det_official"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_rec")), QStringLiteral("paddleocr_rec"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_rec_official")), QStringLiteral("paddleocr_rec_official"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_system_official")), QStringLiteral("paddleocr_system_official"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("tiny_linear_detector")), QStringLiteral("tiny_linear_detector"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("python_mock")), QStringLiteral("python_mock"));
    modelPresetCombo_ = new QComboBox;
    modelPresetCombo_->setEditable(true);
    modelPresetCombo_->addItems(QStringList()
        << QStringLiteral("yolov8n.yaml")
        << QStringLiteral("yolov8n-seg.yaml")
        << QStringLiteral("yolo11n.yaml")
        << QStringLiteral("yolo11n-seg.yaml")
        << QStringLiteral("yolo12n.yaml")
        << QStringLiteral("yolo12n-seg.yaml")
        << QStringLiteral("PP-OCRv4_mobile_det")
        << QStringLiteral("paddle_ctc_smoke")
        << QStringLiteral("PP-OCRv4_mobile_rec")
        << QStringLiteral("PP-OCRv4_det_rec_system")
        << QStringLiteral("diagnostic"));
    epochsEdit_ = new QLineEdit(QStringLiteral("20"));
    batchEdit_ = new QLineEdit(QStringLiteral("8"));
    imageSizeEdit_ = new QLineEdit(QStringLiteral("640"));
    gridSizeEdit_ = new QLineEdit(QStringLiteral("4"));
    resumeCheckpointEdit_ = new QLineEdit;
    resumeCheckpointEdit_->setPlaceholderText(QStringLiteral("可选：选择已有 checkpoint 继续训练"));
    horizontalFlipCheck_ = new QCheckBox(QStringLiteral("水平翻转增强"));
    colorJitterCheck_ = new QCheckBox(QStringLiteral("亮度扰动增强"));
    connect(pluginCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        taskTypeCombo_->clear();
        auto* plugin = pluginManager_.pluginById(pluginCombo_->currentData().toString());
        if (plugin) {
            addTaskTypeItems(taskTypeCombo_, plugin->manifest().taskTypes);
        }
        refreshTrainingDefaults();
    });
    connect(taskTypeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::refreshTrainingDefaults);
    connect(trainingBackendCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        if (modelPresetCombo_ && trainingBackendCombo_) {
            modelPresetCombo_->setCurrentText(defaultModelForBackend(trainingBackendCombo_->currentData().toString()));
        }
        updateTrainingSelectionSummary();
    });
    connect(modelPresetCombo_, &QComboBox::currentTextChanged, this, &MainWindow::updateTrainingSelectionSummary);
    connect(epochsEdit_, &QLineEdit::textChanged, this, &MainWindow::updateTrainingSelectionSummary);
    connect(batchEdit_, &QLineEdit::textChanged, this, &MainWindow::updateTrainingSelectionSummary);
    connect(imageSizeEdit_, &QLineEdit::textChanged, this, &MainWindow::updateTrainingSelectionSummary);
    trainingDatasetSummaryLabel_ = inlineStatusLabel(QStringLiteral("当前数据集：未选择。请先在数据集页导入并通过校验。"));
    trainingDatasetSummaryLabel_->setMinimumHeight(34);
    allowLabelToShrink(trainingDatasetSummaryLabel_);
    trainingBackendHintLabel_ = mutedLabel(QStringLiteral("官方后端会由 Worker 启动独立 Python 进程；scaffold 后端只用于高级诊断。"));
    allowLabelToShrink(trainingBackendHintLabel_);
    trainingRunSummaryLabel_ = inlineStatusLabel(QStringLiteral("等待配置训练实验。"));
    trainingRunSummaryLabel_->setMinimumHeight(42);
    allowLabelToShrink(trainingRunSummaryLabel_);

    auto* startButton = primaryButton(QStringLiteral("启动训练"));
    startButton->setObjectName(QStringLiteral("GreenButton"));
    auto* pauseButton = new QPushButton(QStringLiteral("暂停任务"));
    auto* resumeButton = new QPushButton(QStringLiteral("继续任务"));
    auto* cancelButton = dangerButton(QStringLiteral("取消任务"));
    connect(startButton, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(pauseButton, &QPushButton::clicked, &worker_, &WorkerClient::pause);
    connect(resumeButton, &QPushButton::clicked, &worker_, &WorkerClient::resume);
    connect(cancelButton, &QPushButton::clicked, &worker_, &WorkerClient::cancel);

    trainingDatasetSummaryLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    trainingRunSummaryLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));

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
    auto* kicker = new QLabel(QStringLiteral("LOCAL TRAINING WORKBENCH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("训练实验"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("按数据集类型优先选择官方 YOLO / OCR 后端；运行结果沉淀到任务与产物。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerRoot->addLayout(headerTop);

    auto* actionLayout = new QHBoxLayout;
    actionLayout->setContentsMargins(0, 0, 0, 0);
    actionLayout->setSpacing(10);
    actionLayout->addWidget(startButton);
    actionLayout->addWidget(pauseButton);
    actionLayout->addWidget(resumeButton);
    actionLayout->addWidget(cancelButton);
    actionLayout->addStretch();
    headerRoot->addLayout(actionLayout);

    auto* headerLayout = new QGridLayout;
    headerLayout->setHorizontalSpacing(12);
    headerLayout->setVerticalSpacing(8);
    headerLayout->setColumnStretch(0, 0);
    headerLayout->setColumnStretch(1, 1);
    auto* datasetHeader = new QLabel(QStringLiteral("数据集"));
    datasetHeader->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* summaryHeader = new QLabel(QStringLiteral("摘要"));
    summaryHeader->setObjectName(QStringLiteral("ExperimentMeta"));
    headerLayout->addWidget(datasetHeader, 0, 0);
    headerLayout->addWidget(trainingDatasetSummaryLabel_, 0, 1);
    headerLayout->addWidget(summaryHeader, 1, 0);
    headerLayout->addWidget(trainingRunSummaryLabel_, 1, 1);
    headerRoot->addLayout(headerLayout);

    auto* setupPanel = new InfoPanel(QStringLiteral("实验参数"));
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    form->addRow(QStringLiteral("任务类型"), taskTypeCombo_);
    form->addRow(QStringLiteral("训练后端"), trainingBackendCombo_);
    form->addRow(QStringLiteral("模型预设"), modelPresetCombo_);
    form->addRow(QStringLiteral("Epochs"), epochsEdit_);
    form->addRow(QStringLiteral("Batch Size"), batchEdit_);
    form->addRow(QStringLiteral("Image Size"), imageSizeEdit_);
    setupPanel->bodyLayout()->addLayout(form);
    setupPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("当前模型能力说明")));
    setupPanel->bodyLayout()->addWidget(trainingBackendHintLabel_);

    auto* advancedGroup = new QGroupBox(QStringLiteral("高级 / 诊断后端"));
    auto* advancedForm = new QFormLayout(advancedGroup);
    advancedForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    advancedForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    advancedForm->setHorizontalSpacing(14);
    advancedForm->setVerticalSpacing(10);
    advancedForm->addRow(QStringLiteral("能力插件"), pluginCombo_);
    advancedForm->addRow(QStringLiteral("Grid Size"), gridSizeEdit_);
    advancedForm->addRow(QStringLiteral("Resume"), resumeCheckpointEdit_);
    auto* augmentRow = new QWidget;
    auto* augmentLayout = new QHBoxLayout(augmentRow);
    augmentLayout->setContentsMargins(0, 0, 0, 0);
    augmentLayout->setSpacing(14);
    augmentLayout->addWidget(horizontalFlipCheck_);
    augmentLayout->addWidget(colorJitterCheck_);
    augmentLayout->addStretch();
    advancedForm->addRow(QStringLiteral("Augment"), augmentRow);
    setupPanel->bodyLayout()->addWidget(advancedGroup);
    setupPanel->bodyLayout()->addStretch();

    auto* setupScroll = new QScrollArea;
    setupScroll->setWidget(setupPanel);
    setupScroll->setWidgetResizable(true);
    setupScroll->setFrameShape(QFrame::NoFrame);
    setupScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setupScroll->setMinimumWidth(360);

    auto* monitorPanel = new InfoPanel(QStringLiteral("训练监控"));
    monitorPanel->setMinimumWidth(0);
    progressBar_ = new QProgressBar;
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    metricsWidget_ = new MetricsWidget;
    monitorPanel->bodyLayout()->addWidget(progressBar_);
    monitorPanel->bodyLayout()->addWidget(metricsWidget_, 1);

    auto* artifactPanel = new InfoPanel(QStringLiteral("任务与产物"));
    artifactPanel->setMinimumWidth(0);
    auto* artifactGuideLabel = mutedLabel(QStringLiteral("运行后会记录 checkpoint、训练报告、ONNX、预览图和请求参数。完整产物浏览请进入“任务与产物”。"));
    auto* artifactBoundaryLabel = mutedLabel(QStringLiteral("主流程优先使用官方 YOLO / PaddleOCR 后端；PaddleOCR System 产物来自官方工具链，不代表 C++ DB 后处理已经接入。"));
    allowLabelToShrink(artifactGuideLabel);
    allowLabelToShrink(artifactBoundaryLabel);
    artifactPanel->bodyLayout()->addWidget(artifactGuideLabel);
    artifactPanel->bodyLayout()->addWidget(artifactBoundaryLabel);
    latestCheckpointLabel_ = mutedLabel(QStringLiteral("最新 checkpoint：暂无"));
    latestPreviewPathLabel_ = mutedLabel(QStringLiteral("最新预览：暂无"));
    allowLabelToShrink(latestCheckpointLabel_);
    allowLabelToShrink(latestPreviewPathLabel_);
    latestPreviewImageLabel_ = new QLabel(QStringLiteral("暂无预览图"));
    latestPreviewImageLabel_->setObjectName(QStringLiteral("MutedText"));
    latestPreviewImageLabel_->setAlignment(Qt::AlignCenter);
    latestPreviewImageLabel_->setMinimumHeight(120);
    latestPreviewImageLabel_->setFrameShape(QFrame::StyledPanel);
    latestPreviewImageLabel_->setScaledContents(false);
    artifactPanel->bodyLayout()->addWidget(latestCheckpointLabel_);
    artifactPanel->bodyLayout()->addWidget(latestPreviewPathLabel_);
    artifactPanel->bodyLayout()->addWidget(latestPreviewImageLabel_);
    artifactPanel->bodyLayout()->addStretch();

    auto* logPanel = new InfoPanel(QStringLiteral("训练日志"));
    logPanel->setMinimumWidth(0);
    logEdit_ = new QTextEdit;
    logEdit_->setObjectName(QStringLiteral("LogView"));
    logEdit_->setReadOnly(true);
    logEdit_->setLineWrapMode(QTextEdit::WidgetWidth);
    logEdit_->setMinimumWidth(0);
    logEdit_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
    logPanel->bodyLayout()->addWidget(logEdit_);

    auto* detailTabs = new QTabWidget;
    detailTabs->setObjectName(QStringLiteral("TrainingDetailTabs"));
    detailTabs->setDocumentMode(true);
    detailTabs->addTab(logPanel, QStringLiteral("训练日志"));
    detailTabs->addTab(artifactPanel, QStringLiteral("任务与产物"));

    auto* rightSplitter = new QSplitter(Qt::Vertical);
    rightSplitter->setMinimumWidth(0);
    rightSplitter->addWidget(monitorPanel);
    rightSplitter->addWidget(detailTabs);
    rightSplitter->setStretchFactor(0, 1);
    rightSplitter->setStretchFactor(1, 1);
    rightSplitter->setSizes(QList<int>() << 330 << 330);

    auto* bodySplitter = new QSplitter(Qt::Horizontal);
    bodySplitter->addWidget(setupScroll);
    bodySplitter->addWidget(rightSplitter);
    bodySplitter->setStretchFactor(0, 4);
    bodySplitter->setStretchFactor(1, 7);
    bodySplitter->setSizes(QList<int>() << 390 << 620);

    layout->addWidget(headerPanel);
    layout->addWidget(bodySplitter, 1);
    return page;
}

QWidget* MainWindow::buildTaskQueuePage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerRefreshButton = primaryButton(QStringLiteral("刷新历史"));
    connect(headerRefreshButton, &QPushButton::clicked, this, &MainWindow::updateRecentTasks);

    auto* refreshButton = primaryButton(QStringLiteral("刷新历史"));
    auto* cancelButton = dangerButton(QStringLiteral("取消选中任务"));
    auto* reproduceButton = new QPushButton(QStringLiteral("复现实验"));
    taskKindFilterCombo_ = new QComboBox;
    taskKindFilterCombo_->setMinimumWidth(140);
    taskKindFilterCombo_->addItem(uiText("全部类别"), QString());
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Train), QStringLiteral("train"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Validate), QStringLiteral("validate"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Export), QStringLiteral("export"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Infer), QStringLiteral("infer"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Evaluate), QStringLiteral("evaluate"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Benchmark), QStringLiteral("benchmark"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Curate), QStringLiteral("curate"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Snapshot), QStringLiteral("snapshot"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Pipeline), QStringLiteral("pipeline"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Report), QStringLiteral("report"));
    taskStateFilterCombo_ = new QComboBox;
    taskStateFilterCombo_->setMinimumWidth(140);
    taskStateFilterCombo_->addItem(uiText("全部状态"), QString());
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Queued), QStringLiteral("queued"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Running), QStringLiteral("running"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Completed), QStringLiteral("completed"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Failed), QStringLiteral("failed"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Canceled), QStringLiteral("canceled"));
    taskSearchEdit_ = new QLineEdit;
    taskSearchEdit_->setMinimumWidth(260);
    taskSearchEdit_->setPlaceholderText(QStringLiteral("搜索任务、后端、消息"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::updateRecentTasks);
    connect(cancelButton, &QPushButton::clicked, this, &MainWindow::cancelSelectedTask);
    connect(reproduceButton, &QPushButton::clicked, this, &MainWindow::reproduceSelectedTrainingTask);
    connect(taskKindFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::applyTaskFilters);
    connect(taskStateFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::applyTaskFilters);
    connect(taskSearchEdit_, &QLineEdit::textChanged, this, &MainWindow::applyTaskFilters);

    auto* toolbar = new InfoPanel(QStringLiteral("历史操作"));
    toolbar->bodyLayout()->setSpacing(6);

    auto* controlStrip = new QFrame;
    controlStrip->setObjectName(QStringLiteral("TaskControlStrip"));
    auto* actionLayout = new QGridLayout(controlStrip);
    actionLayout->setContentsMargins(12, 8, 12, 8);
    actionLayout->setHorizontalSpacing(10);
    actionLayout->setVerticalSpacing(6);
    auto* actionCaption = new QLabel(uiText("操作"));
    actionCaption->setObjectName(QStringLiteral("TaskFilterLabel"));
    actionLayout->addWidget(actionCaption, 0, 0);
    actionLayout->addWidget(refreshButton, 0, 1);
    actionLayout->addWidget(cancelButton, 0, 2);
    actionLayout->addWidget(reproduceButton, 0, 3);

    auto* categoryLabel = new QLabel(QStringLiteral("类别"));
    categoryLabel->setObjectName(QStringLiteral("TaskFilterLabel"));
    auto* stateLabel = new QLabel(QStringLiteral("状态"));
    stateLabel->setObjectName(QStringLiteral("TaskFilterLabel"));
    auto* searchLabel = new QLabel(uiText("搜索"));
    searchLabel->setObjectName(QStringLiteral("TaskFilterLabel"));
    actionLayout->addWidget(categoryLabel, 1, 0);
    actionLayout->addWidget(taskKindFilterCombo_, 1, 1);
    actionLayout->addWidget(stateLabel, 1, 2);
    actionLayout->addWidget(taskStateFilterCombo_, 1, 3);
    actionLayout->addWidget(searchLabel, 1, 4);
    actionLayout->addWidget(taskSearchEdit_, 1, 5);
    actionLayout->setColumnStretch(5, 1);

    toolbar->bodyLayout()->addWidget(controlStrip);
    toolbar->bodyLayout()->addWidget(mutedLabel(QStringLiteral("这里统一追踪训练、校验、划分、导出和推理任务；运行产物在下方详情区集中查看。")));

    auto* tablePanel = new InfoPanel(QStringLiteral("任务历史"));
    tablePanel->setMinimumWidth(420);
    tablePanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    taskQueueTable_ = new QTableWidget(0, 7);
    taskQueueTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务")
        << QStringLiteral("类别")
        << QStringLiteral("插件")
        << QStringLiteral("类型")
        << QStringLiteral("状态")
        << QStringLiteral("更新时间")
        << QStringLiteral("消息"));
    configureTable(taskQueueTable_);
    taskQueueTable_->setWordWrap(true);
    taskQueueTable_->setMinimumHeight(180);
    taskQueueTable_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    taskQueueTable_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    taskQueueTable_->verticalHeader()->setDefaultSectionSize(42);
    taskQueueTable_->horizontalHeader()->setStretchLastSection(false);
    taskQueueTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    taskQueueTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    taskQueueTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    taskQueueTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    taskQueueTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    taskQueueTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    taskQueueTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    taskQueueTable_->setColumnHidden(2, true);
    taskQueueTable_->setColumnHidden(6, true);
    connect(taskQueueTable_, &QTableWidget::itemSelectionChanged, this, &MainWindow::updateSelectedTaskDetails);
    tablePanel->bodyLayout()->addWidget(taskQueueTable_);

    auto* detailPanel = new InfoPanel(QStringLiteral("任务详情与产物"));
    detailPanel->setMinimumWidth(640);
    detailPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    selectedTaskSummaryLabel_ = inlineStatusLabel(QStringLiteral("请选择一个任务查看产物、指标和导出记录。"));
    selectedTaskSummaryLabel_->setObjectName(QStringLiteral("TaskDetailSummary"));
    selectedTaskSummaryLabel_->setMinimumHeight(40);
    selectedTaskSummaryLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
    taskArtifactTable_ = new QTableWidget(0, 4);
    taskArtifactTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("类型")
        << QStringLiteral("路径")
        << QStringLiteral("说明")
        << QStringLiteral("时间"));
    configureTable(taskArtifactTable_);
    taskArtifactTable_->setWordWrap(true);
    taskArtifactTable_->setMinimumHeight(220);
    taskArtifactTable_->verticalHeader()->setDefaultSectionSize(42);
    taskArtifactTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    taskArtifactTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    taskArtifactTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    taskArtifactTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    connect(taskArtifactTable_, &QTableWidget::itemSelectionChanged, this, &MainWindow::updateArtifactPreviewFromSelection);

    taskMetricTable_ = new QTableWidget(0, 4);
    taskMetricTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("指标")
        << QStringLiteral("值")
        << QStringLiteral("Step")
        << QStringLiteral("Epoch"));
    configureTable(taskMetricTable_);
    taskMetricTable_->setMinimumHeight(210);
    taskMetricTable_->verticalHeader()->setDefaultSectionSize(38);
    taskMetricTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    taskMetricTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    taskMetricTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    taskMetricTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);

    taskExportTable_ = new QTableWidget(0, 3);
    taskExportTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("格式")
        << QStringLiteral("路径")
        << QStringLiteral("时间"));
    configureTable(taskExportTable_);
    taskExportTable_->setWordWrap(true);
    taskExportTable_->setMinimumHeight(210);
    taskExportTable_->verticalHeader()->setDefaultSectionSize(42);
    taskExportTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    taskExportTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    taskExportTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);

    artifactImagePreviewLabel_ = new QLabel(QStringLiteral("暂无产物预览"));
    artifactImagePreviewLabel_->setObjectName(QStringLiteral("ArtifactPreviewCanvas"));
    artifactImagePreviewLabel_->setAlignment(Qt::AlignCenter);
    artifactImagePreviewLabel_->setMinimumHeight(220);
    artifactImagePreviewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    artifactPreviewText_ = new QPlainTextEdit;
    artifactPreviewText_->setObjectName(QStringLiteral("ArtifactPreviewText"));
    artifactPreviewText_->setReadOnly(true);
    artifactPreviewText_->setMinimumHeight(160);
    artifactPreviewText_->setPlainText(QStringLiteral("选择一个产物后显示摘要。"));
    auto* artifactDefaultPreview = new QWidget;
    auto* artifactDefaultLayout = new QVBoxLayout(artifactDefaultPreview);
    artifactDefaultLayout->setContentsMargins(0, 0, 0, 0);
    artifactDefaultLayout->setSpacing(10);
    artifactDefaultLayout->addWidget(artifactImagePreviewLabel_, 1);
    artifactDefaultLayout->addWidget(artifactPreviewText_, 2);
    artifactPreviewStack_ = new QStackedWidget;
    artifactPreviewStack_->setMinimumHeight(220);
    artifactPreviewStack_->addWidget(artifactDefaultPreview);
    artifactEvaluationReportView_ = new EvaluationReportView;
    auto* artifactEvaluationScroll = new QScrollArea;
    artifactEvaluationScroll->setWidget(artifactEvaluationReportView_);
    artifactEvaluationScroll->setWidgetResizable(true);
    artifactEvaluationScroll->setFrameShape(QFrame::NoFrame);
    artifactEvaluationScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    artifactEvaluationScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    artifactPreviewStack_->addWidget(artifactEvaluationScroll);

    auto* actionGridFrame = new QFrame;
    actionGridFrame->setObjectName(QStringLiteral("ArtifactActionGrid"));
    actionGridFrame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
    auto* actionGrid = new QGridLayout(actionGridFrame);
    actionGrid->setContentsMargins(12, 10, 12, 10);
    actionGrid->setHorizontalSpacing(12);
    actionGrid->setVerticalSpacing(8);
    auto* openDirButton = new QPushButton(QStringLiteral("打开目录"));
    auto* copyPathButton = new QPushButton(QStringLiteral("复制路径"));
    auto* useInferButton = new QPushButton(QStringLiteral("用作推理模型"));
    auto* useExportButton = new QPushButton(QStringLiteral("用作导出输入"));
    auto* registerModelButton = new QPushButton(QStringLiteral("注册模型版本"));
    auto* evaluateButton = new QPushButton(QStringLiteral("评估"));
    auto* benchmarkButton = new QPushButton(QStringLiteral("基准"));
    auto* reportButton = new QPushButton(QStringLiteral("交付报告"));
    connect(openDirButton, &QPushButton::clicked, this, &MainWindow::openSelectedArtifactDirectory);
    connect(copyPathButton, &QPushButton::clicked, this, &MainWindow::copySelectedArtifactPath);
    connect(useInferButton, &QPushButton::clicked, this, &MainWindow::useSelectedArtifactForInference);
    connect(useExportButton, &QPushButton::clicked, this, &MainWindow::useSelectedArtifactForExport);
    connect(registerModelButton, &QPushButton::clicked, this, &MainWindow::registerSelectedArtifactAsModelVersion);
    connect(evaluateButton, &QPushButton::clicked, this, &MainWindow::evaluateSelectedArtifact);
    connect(benchmarkButton, &QPushButton::clicked, this, &MainWindow::benchmarkSelectedArtifact);
    connect(reportButton, &QPushButton::clicked, this, &MainWindow::generateDeliveryReportFromSelectedArtifact);
    actionGrid->addWidget(openDirButton, 0, 0);
    actionGrid->addWidget(copyPathButton, 0, 1);
    actionGrid->addWidget(useInferButton, 0, 2);
    actionGrid->addWidget(useExportButton, 0, 3);
    actionGrid->addWidget(registerModelButton, 1, 0);
    actionGrid->addWidget(evaluateButton, 1, 1);
    actionGrid->addWidget(benchmarkButton, 1, 2);
    actionGrid->addWidget(reportButton, 1, 3);
    for (int column = 0; column < 4; ++column) {
        actionGrid->setColumnStretch(column, 1);
    }

    auto* artifactTab = new QWidget;
    auto* artifactTabLayout = new QVBoxLayout(artifactTab);
    artifactTabLayout->setContentsMargins(0, 0, 0, 0);
    artifactTabLayout->addWidget(taskArtifactTable_);
    auto* metricTab = new QWidget;
    auto* metricTabLayout = new QVBoxLayout(metricTab);
    metricTabLayout->setContentsMargins(0, 0, 0, 0);
    metricTabLayout->addWidget(taskMetricTable_);
    auto* exportTab = new QWidget;
    auto* exportTabLayout = new QVBoxLayout(exportTab);
    exportTabLayout->setContentsMargins(0, 0, 0, 0);
    exportTabLayout->addWidget(taskExportTable_);
    auto* previewTab = new QWidget;
    auto* previewTabLayout = new QVBoxLayout(previewTab);
    previewTabLayout->setContentsMargins(0, 0, 0, 0);
    previewTabLayout->addWidget(artifactPreviewStack_);

    auto* detailTabs = new QTabWidget;
    detailTabs->setObjectName(QStringLiteral("TaskDetailTabs"));
    detailTabs->addTab(artifactTab, uiText("产物"));
    detailTabs->addTab(metricTab, uiText("指标"));
    detailTabs->addTab(exportTab, uiText("导出"));
    detailTabs->addTab(previewTab, uiText("预览"));

    detailTabs->setMinimumHeight(420);
    detailTabs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    artifactPreviewStack_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    detailPanel->bodyLayout()->setSpacing(12);
    detailPanel->bodyLayout()->addWidget(selectedTaskSummaryLabel_);
    detailPanel->bodyLayout()->addWidget(detailTabs, 1);
    detailPanel->bodyLayout()->addWidget(actionGridFrame);

    auto* bodySplitter = new QSplitter(Qt::Horizontal);
    bodySplitter->addWidget(tablePanel);
    bodySplitter->addWidget(detailPanel);
    bodySplitter->setChildrenCollapsible(false);
    bodySplitter->setStretchFactor(0, 2);
    bodySplitter->setStretchFactor(1, 3);
    bodySplitter->setSizes(QList<int>() << 520 << 840);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("TASK ARTIFACT CENTER"),
        QStringLiteral("任务与产物工作台"),
        QStringLiteral("按任务追踪 Worker 产物、指标、导出和评估报告；选中产物后进入推理、导出、注册、评估或交付报告。"),
        headerRefreshButton,
        QStringList()
            << QStringLiteral("Task History")
            << QStringLiteral("Artifacts")
            << QStringLiteral("Metrics")
            << QStringLiteral("Reports")));
    layout->addWidget(toolbar);
    layout->addWidget(bodySplitter, 1);
    return page;
}

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

QWidget* MainWindow::buildConversionPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerExportButton = primaryButton(QStringLiteral("开始导出"));

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
    auto* kicker = new QLabel(QStringLiteral("MODEL DELIVERY WORKBENCH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("模型导出"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("从训练产物生成 ONNX、NCNN、TensorRT 或诊断 JSON，并把报告写入任务与产物。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(headerExportButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* sourceLabel = new QLabel(QStringLiteral("输入"));
    sourceLabel->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* policyLabel = new QLabel(QStringLiteral("策略"));
    policyLabel->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* sourceStatus = inlineStatusLabel(QStringLiteral("优先从“任务与产物”选择 checkpoint、ONNX 或官方训练产物。"));
    sourceStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    auto* policyStatus = inlineStatusLabel(QStringLiteral("导出只通过 Worker 执行；TensorRT 仍需 RTX / SM 75+ 外部验收。"));
    policyStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(sourceStatus);
    allowLabelToShrink(policyStatus);
    headerGrid->addWidget(sourceLabel, 0, 0);
    headerGrid->addWidget(sourceStatus, 0, 1);
    headerGrid->addWidget(policyLabel, 1, 0);
    headerGrid->addWidget(policyStatus, 1, 1);
    headerRoot->addLayout(headerGrid);

    auto* mainSplitter = new QSplitter(Qt::Horizontal);

    auto* setupPanel = new InfoPanel(QStringLiteral("导出设置"));
    conversionCheckpointEdit_ = new QLineEdit;
    conversionCheckpointEdit_->setPlaceholderText(QStringLiteral("从任务产物带入，或选择 checkpoint / ONNX / AITrain export"));
    auto* chooseCheckpointButton = new QPushButton(QStringLiteral("选择模型产物"));
    connect(chooseCheckpointButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择模型产物"), currentProjectPath_, QStringLiteral("AITrain model (*.aitrain *.json *.onnx *.pt *.pdparams *.engine *.plan);;All files (*.*)"));
        if (!file.isEmpty()) {
            conversionCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });

    conversionFormatCombo_ = new QComboBox;
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("onnx")), QStringLiteral("onnx"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("ncnn")), QStringLiteral("ncnn"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("tiny_detector_json")), QStringLiteral("tiny_detector_json"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("tensorrt")), QStringLiteral("tensorrt"));

    conversionOutputEdit_ = new QLineEdit;
    conversionOutputEdit_->setPlaceholderText(QStringLiteral("留空则输出到项目 models/exported；未打开项目时使用输入同目录"));
    auto* chooseOutputButton = new QPushButton(QStringLiteral("选择输出"));
    connect(chooseOutputButton, &QPushButton::clicked, this, [this]() {
        const QString format = conversionFormatCombo_
            ? conversionFormatCombo_->currentData().toString()
            : QStringLiteral("onnx");
        const QString inputPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
        const QString defaultDir = !currentProjectPath_.isEmpty()
            ? QDir(currentProjectPath_).filePath(QStringLiteral("models/exported"))
            : QFileInfo(inputPath).absolutePath();
        const QString selected = QFileDialog::getSaveFileName(
            this,
            uiText("选择导出路径"),
            QDir(defaultDir).filePath(defaultExportFileName(format)),
            exportFileFilter(format));
        if (!selected.isEmpty()) {
            conversionOutputEdit_->setText(QDir::toNativeSeparators(selected));
        }
    });
    auto* exportButton = primaryButton(QStringLiteral("开始导出"));
    connect(headerExportButton, &QPushButton::clicked, this, &MainWindow::startModelExport);
    connect(exportButton, &QPushButton::clicked, this, &MainWindow::startModelExport);

    auto* inputRow = new QWidget;
    auto* inputLayout = new QHBoxLayout(inputRow);
    inputLayout->setContentsMargins(0, 0, 0, 0);
    inputLayout->setSpacing(8);
    inputLayout->addWidget(conversionCheckpointEdit_, 1);
    inputLayout->addWidget(chooseCheckpointButton);

    auto* outputRow = new QWidget;
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    outputLayout->setSpacing(8);
    outputLayout->addWidget(conversionOutputEdit_, 1);
    outputLayout->addWidget(chooseOutputButton);

    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    form->addRow(QStringLiteral("模型输入"), inputRow);
    form->addRow(QStringLiteral("目标格式"), conversionFormatCombo_);
    form->addRow(QStringLiteral("输出路径"), outputRow);
    setupPanel->bodyLayout()->addLayout(form);

    auto* sourceHelp = emptyStateLabel(QStringLiteral("从“任务与产物”中选中 best.onnx、checkpoint 或官方导出目录后，可点击“用作导出输入”自动带入这里。"));
    allowLabelToShrink(sourceHelp);
    setupPanel->bodyLayout()->addWidget(sourceHelp);

    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(10);
    auto* exportActionHint = mutedLabel(QStringLiteral("导出任务会记录到任务历史，完成后可直接作为推理输入。"));
    allowLabelToShrink(exportActionHint);
    actionLayout->addWidget(exportActionHint, 1);
    actionLayout->addWidget(exportButton);
    setupPanel->bodyLayout()->addWidget(actionStrip);
    setupPanel->bodyLayout()->addStretch();

    auto* rightStack = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightStack);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(16);

    auto* matrixPanel = new InfoPanel(QStringLiteral("格式矩阵"));
    auto* matrixTable = new QTableWidget(4, 4);
    matrixTable->setWordWrap(true);
    matrixTable->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("格式")
        << QStringLiteral("输入")
        << QStringLiteral("产物")
        << QStringLiteral("状态"));
    configureTable(matrixTable);
    matrixTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    matrixTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    matrixTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    matrixTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    matrixTable->setMinimumHeight(170);
    matrixTable->verticalHeader()->setDefaultSectionSize(44);
    const QStringList formats = {
        QStringLiteral("onnx"),
        QStringLiteral("ncnn"),
        QStringLiteral("tiny_detector_json"),
        QStringLiteral("tensorrt")
    };
    for (int row = 0; row < formats.size(); ++row) {
        const QString format = formats.at(row);
        matrixTable->setItem(row, 0, new QTableWidgetItem(exportFormatLabel(format)));
        matrixTable->setItem(row, 1, new QTableWidgetItem(
            format == QStringLiteral("tiny_detector_json")
                ? QStringLiteral("tiny detector checkpoint")
                : uiText("checkpoint 或 ONNX")));
        matrixTable->setItem(row, 2, new QTableWidgetItem(defaultExportFileName(format)));
        matrixTable->setItem(row, 3, new QTableWidgetItem(exportFormatNote(format)));
    }
    matrixPanel->bodyLayout()->addWidget(matrixTable);

    auto* resultPanel = new InfoPanel(QStringLiteral("运行状态"));
    exportResultLabel_ = inlineStatusLabel(QStringLiteral("暂无导出任务。"));
    resultPanel->bodyLayout()->addWidget(exportResultLabel_);
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("ONNX 会写入 AITrain sidecar；NCNN 依赖本机 onnx2ncnn；TensorRT 在当前 GTX 1060 / SM 61 上保持 hardware-blocked。")));
    resultPanel->bodyLayout()->addStretch();

    rightLayout->addWidget(matrixPanel);
    rightLayout->addWidget(resultPanel, 1);

    mainSplitter->addWidget(setupPanel);
    mainSplitter->addWidget(rightStack);
    mainSplitter->setChildrenCollapsible(false);
    mainSplitter->setStretchFactor(0, 3);
    mainSplitter->setStretchFactor(1, 4);
    mainSplitter->setSizes(QList<int>() << 520 << 680);

    layout->addWidget(headerPanel);
    layout->addWidget(mainSplitter, 1);
    return page;
}

QWidget* MainWindow::buildInferencePage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerInferButton = primaryButton(QStringLiteral("开始推理"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("InferenceHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(16, 14, 16, 14);
    headerRoot->setSpacing(11);

    auto* headerTop = new QHBoxLayout;
    headerTop->setSpacing(14);
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(4);
    auto* kicker = new QLabel(QStringLiteral("INFERENCE VALIDATION"));
    kicker->setObjectName(QStringLiteral("InferenceKicker"));
    auto* title = new QLabel(QStringLiteral("推理验证工作台"));
    title->setObjectName(QStringLiteral("InferenceTitle"));
    auto* subtitle = new QLabel(QStringLiteral("选择模型和图片，运行 Worker 推理并查看 prediction JSON 与 overlay。这里不直接承载模型后处理逻辑。"));
    subtitle->setObjectName(QStringLiteral("InferenceMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    auto* badgeRow = new QWidget;
    auto* badgeLayout = new QHBoxLayout(badgeRow);
    badgeLayout->setContentsMargins(0, 4, 0, 0);
    badgeLayout->setSpacing(7);
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("ONNX Runtime")));
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("Worker")));
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("Prediction JSON")));
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("Overlay")));
    badgeLayout->addStretch();
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    titleLayout->addWidget(badgeRow);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(headerInferButton, 0, Qt::AlignTop);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* executionLabel = new QLabel(QStringLiteral("执行"));
    executionLabel->setObjectName(QStringLiteral("InferenceMeta"));
    auto* boundaryLabel = new QLabel(QStringLiteral("边界"));
    boundaryLabel->setObjectName(QStringLiteral("InferenceMeta"));
    auto* executionStatus = inlineStatusLabel(QStringLiteral("Worker 隔离执行推理；任务、产物和失败原因写入任务历史。"));
    executionStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    auto* boundaryStatus = inlineStatusLabel(QStringLiteral("C++ ONNX 支持 detection / segmentation / OCR Rec；PaddleOCR System 仍查看官方工具链产物。"));
    boundaryStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(executionStatus);
    allowLabelToShrink(boundaryStatus);
    headerGrid->addWidget(executionLabel, 0, 0);
    headerGrid->addWidget(executionStatus, 0, 1);
    headerGrid->addWidget(boundaryLabel, 1, 0);
    headerGrid->addWidget(boundaryStatus, 1, 1);
    headerGrid->setColumnStretch(1, 1);
    headerRoot->addLayout(headerGrid);

    auto* mainSplitter = new QSplitter(Qt::Horizontal);

    auto* leftStack = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftStack);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("验证输入"));
    auto* inferForm = new QFormLayout;
    inferenceCheckpointEdit_ = new QLineEdit;
    inferenceImageEdit_ = new QLineEdit;
    inferenceOutputEdit_ = new QLineEdit;
    inferenceCheckpointEdit_->setPlaceholderText(QStringLiteral("从任务产物带入，或选择 ONNX / AITrain export / TensorRT engine"));
    inferenceImageEdit_->setPlaceholderText(QStringLiteral("选择验证图片"));
    inferenceOutputEdit_->setPlaceholderText(QStringLiteral("输出目录；留空则输出到模型同目录 inference"));
    auto* chooseModelButton = new QPushButton(QStringLiteral("选择模型文件"));
    auto* chooseImageButton = new QPushButton(QStringLiteral("选择图片"));
    auto* chooseOutputButton = new QPushButton(QStringLiteral("选择输出目录"));
    auto* inferButton = primaryButton(QStringLiteral("开始推理"));
    connect(chooseModelButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择模型文件"), currentProjectPath_, QStringLiteral("AITrain model (*.aitrain *.json *.onnx *.engine *.plan);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    connect(chooseImageButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择图片"), currentProjectPath_, QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceImageEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    connect(chooseOutputButton, &QPushButton::clicked, this, [this]() {
        const QString modelPath = QDir::fromNativeSeparators(inferenceCheckpointEdit_ ? inferenceCheckpointEdit_->text().trimmed() : QString());
        const QString currentOutput = QDir::fromNativeSeparators(inferenceOutputEdit_ ? inferenceOutputEdit_->text().trimmed() : QString());
        const QString defaultDir = !currentOutput.isEmpty()
            ? currentOutput
            : (!currentProjectPath_.isEmpty()
                ? QDir(currentProjectPath_).filePath(QStringLiteral("inference"))
                : QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("inference")));
        const QString dir = QFileDialog::getExistingDirectory(this, uiText("选择推理输出目录"), defaultDir);
        if (!dir.isEmpty() && inferenceOutputEdit_) {
            inferenceOutputEdit_->setText(QDir::toNativeSeparators(dir));
        }
    });
    connect(inferButton, &QPushButton::clicked, this, &MainWindow::startInference);
    auto* modelRow = new QWidget;
    auto* modelLayout = new QHBoxLayout(modelRow);
    modelLayout->setContentsMargins(0, 0, 0, 0);
    modelLayout->addWidget(inferenceCheckpointEdit_);
    modelLayout->addWidget(chooseModelButton);
    auto* imageRow = new QWidget;
    auto* imageLayout = new QHBoxLayout(imageRow);
    imageLayout->setContentsMargins(0, 0, 0, 0);
    imageLayout->setSpacing(8);
    imageLayout->addWidget(inferenceImageEdit_);
    imageLayout->addWidget(chooseImageButton);
    modelLayout->setSpacing(8);
    auto* outputRow = new QWidget;
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    outputLayout->setSpacing(8);
    outputLayout->addWidget(inferenceOutputEdit_);
    outputLayout->addWidget(chooseOutputButton);
    inferForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    inferForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    inferForm->setHorizontalSpacing(14);
    inferForm->setVerticalSpacing(10);
    inferForm->addRow(QStringLiteral("模型路径"), modelRow);
    inferForm->addRow(QStringLiteral("图片路径"), imageRow);
    inferForm->addRow(QStringLiteral("推理输出"), outputRow);
    toolbar->bodyLayout()->addLayout(inferForm);
    auto* sourceHelp = emptyStateLabel(QStringLiteral("从“任务与产物”选中 ONNX、AITrain export 或 engine 后，可点击“用作推理模型”自动带入这里。输出目录留空会写到模型同目录 inference。"));
    allowLabelToShrink(sourceHelp);
    toolbar->bodyLayout()->addWidget(sourceHelp);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(10);
    auto* inferenceActionHint = mutedLabel(QStringLiteral("推理任务会记录到任务历史，完成后可在产物详情中复查 JSON、overlay 和耗时。"));
    allowLabelToShrink(inferenceActionHint);
    actionLayout->addWidget(inferenceActionHint, 1);
    actionLayout->addWidget(inferButton);
    toolbar->bodyLayout()->addWidget(actionStrip);
    toolbar->bodyLayout()->addStretch();

    auto* capabilityPanel = new InfoPanel(QStringLiteral("可解析结果"));
    auto* capabilityHint = mutedLabel(QStringLiteral("推理验证会根据 ONNX 模型族或 AITrain 导出信息选择后处理；scaffold / 官方工具链边界保持显式。"));
    allowLabelToShrink(capabilityHint);
    capabilityPanel->bodyLayout()->addWidget(capabilityHint);
    auto* capabilityGrid = new QGridLayout;
    capabilityGrid->setHorizontalSpacing(10);
    capabilityGrid->setVerticalSpacing(10);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("YOLO 检测"), QStringLiteral("box、类别、置信度、NMS 与 overlay。")), 0, 0);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("YOLO 分割"), QStringLiteral("box、mask、mask area 与半透明 overlay。")), 0, 1);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("OCR Rec"), QStringLiteral("CTC greedy decode、文本与置信度摘要。")), 1, 0);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("PaddleOCR System"), QStringLiteral("端到端结果仍通过官方工具链任务产物查看。")), 1, 1);
    capabilityGrid->setColumnStretch(0, 1);
    capabilityGrid->setColumnStretch(1, 1);
    capabilityPanel->bodyLayout()->addLayout(capabilityGrid);
    capabilityPanel->bodyLayout()->addStretch();

    leftLayout->addWidget(toolbar, 3);
    leftLayout->addWidget(capabilityPanel, 2);

    auto* rightStack = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightStack);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(16);

    auto* flowPanel = new InfoPanel(QStringLiteral("验证链路"));
    auto* flowGrid = new QGridLayout;
    flowGrid->setHorizontalSpacing(10);
    flowGrid->setVerticalSpacing(10);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("1"), QStringLiteral("模型产物"), QStringLiteral("ONNX / AITrain export / TensorRT engine")), 0, 0);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("2"), QStringLiteral("样本图片"), QStringLiteral("单张验证图进入预处理")), 0, 1);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("3"), QStringLiteral("Worker 推理"), QStringLiteral("隔离执行，不阻塞 GUI")), 1, 0);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("4"), QStringLiteral("结果归档"), QStringLiteral("prediction JSON + overlay")), 1, 1);
    flowGrid->setColumnStretch(0, 1);
    flowGrid->setColumnStretch(1, 1);
    flowPanel->bodyLayout()->addLayout(flowGrid);

    auto* preview = new QSplitter(Qt::Horizontal);
    auto* summaryPanel = new InfoPanel(QStringLiteral("结果摘要"));
    summaryPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("Worker 返回的 prediction JSON 会压缩显示任务类型、结果数量、首个类别 / 文本、耗时和结果文件路径。")));
    inferenceResultLabel_ = inlineStatusLabel(QStringLiteral("尚未推理。"));
    inferenceResultLabel_->setObjectName(QStringLiteral("InferenceResultSummary"));
    summaryPanel->bodyLayout()->addWidget(inferenceResultLabel_);
    summaryPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("完整原始 JSON 可在“任务与产物”的产物详情中查看。")));
    summaryPanel->bodyLayout()->addStretch();

    auto* resultPanel = new InfoPanel(QStringLiteral("Overlay 预览"));
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("完成后显示检测框、分割 mask 或 OCR 可视化图；失败时保留明确状态文本。")));
    inferenceOverlayLabel_ = new QLabel(QStringLiteral("暂无 overlay\n运行推理后显示可视化产物。"));
    inferenceOverlayLabel_->setObjectName(QStringLiteral("InferenceOverlayCanvas"));
    inferenceOverlayLabel_->setAlignment(Qt::AlignCenter);
    inferenceOverlayLabel_->setMinimumHeight(300);
    inferenceOverlayLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    inferenceOverlayLabel_->setFrameShape(QFrame::NoFrame);
    resultPanel->bodyLayout()->addWidget(inferenceOverlayLabel_);
    preview->addWidget(summaryPanel);
    preview->addWidget(resultPanel);
    preview->setStretchFactor(0, 1);
    preview->setStretchFactor(1, 2);
    preview->setSizes(QList<int>() << 300 << 620);

    rightLayout->addWidget(flowPanel);
    rightLayout->addWidget(preview, 1);

    mainSplitter->addWidget(leftStack);
    mainSplitter->addWidget(rightStack);
    mainSplitter->setChildrenCollapsible(false);
    mainSplitter->setStretchFactor(0, 3);
    mainSplitter->setStretchFactor(1, 5);
    mainSplitter->setSizes(QList<int>() << 460 << 760);

    connect(headerInferButton, &QPushButton::clicked, this, &MainWindow::startInference);

    layout->addWidget(headerPanel);
    layout->addWidget(mainSplitter, 1);
    return page;
}

QWidget* MainWindow::buildPluginsPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* refreshButton = primaryButton(QStringLiteral("重新扫描插件"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshPlugins);

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
    auto* kicker = new QLabel(QStringLiteral("PLUGIN CAPABILITY MATRIX"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("插件能力矩阵"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("扫描模型、数据集、导出和推理扩展能力；插件仍只通过公共接口暴露能力。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(refreshButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* scanCaption = new QLabel(QStringLiteral("扫描"));
    scanCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* pathCaption = new QLabel(QStringLiteral("路径"));
    pathCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    pluginConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("等待插件扫描。"));
    pluginConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    pluginSearchPathLabel_ = inlineStatusLabel(QStringLiteral("插件搜索路径：未初始化"));
    pluginSearchPathLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    pluginMarketplaceStatusLabel_ = inlineStatusLabel(QStringLiteral("插件市场：等待加载本地索引。"));
    pluginMarketplaceStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(pluginConsoleStatusLabel_);
    allowLabelToShrink(pluginSearchPathLabel_);
    allowLabelToShrink(pluginMarketplaceStatusLabel_);
    headerGrid->addWidget(scanCaption, 0, 0);
    headerGrid->addWidget(pluginConsoleStatusLabel_, 0, 1);
    headerGrid->addWidget(pathCaption, 1, 0);
    headerGrid->addWidget(pluginSearchPathLabel_, 1, 1);
    headerGrid->addWidget(new QLabel(QStringLiteral("市场")), 2, 0);
    headerGrid->addWidget(pluginMarketplaceStatusLabel_, 2, 1);
    headerRoot->addLayout(headerGrid);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* pluginCountCard = createMetricCard(QStringLiteral("已加载插件"), QStringLiteral("0"), QStringLiteral("manifest 已加载"));
    pluginCountSummaryLabel_ = pluginCountCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* datasetFormatCard = createMetricCard(QStringLiteral("数据集格式"), QStringLiteral("0"), QStringLiteral("可识别 / 校验格式"));
    pluginDatasetFormatSummaryLabel_ = datasetFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* exportFormatCard = createMetricCard(QStringLiteral("导出格式"), QStringLiteral("0"), QStringLiteral("插件声明的导出目标"));
    pluginExportFormatSummaryLabel_ = exportFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* gpuCard = createMetricCard(QStringLiteral("GPU 需求"), QStringLiteral("0"), QStringLiteral("声明需要 GPU 的插件"));
    pluginGpuSummaryLabel_ = gpuCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    summaryLayout->addWidget(pluginCountCard, 0, 0);
    summaryLayout->addWidget(datasetFormatCard, 0, 1);
    summaryLayout->addWidget(exportFormatCard, 0, 2);
    summaryLayout->addWidget(gpuCard, 0, 3);

    auto* tablePanel = new InfoPanel(QStringLiteral("插件扩展"));
    pluginTable_ = new QTableWidget(0, 7);
    pluginTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID")
        << QStringLiteral("名称")
        << QStringLiteral("版本")
        << QStringLiteral("任务")
        << QStringLiteral("数据集")
        << QStringLiteral("导出")
        << QStringLiteral("GPU"));
    configureTable(pluginTable_);
    pluginTable_->setWordWrap(true);
    pluginTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    pluginTable_->verticalHeader()->setDefaultSectionSize(42);

    const QString appDir = QApplication::applicationDirPath();
    pluginMarketplaceWidget_ = new PluginMarketplaceWidget(
        QDir(appDir).filePath(QStringLiteral("plugins/marketplace")),
        QDir(appDir).filePath(QStringLiteral("plugins/models")));
    connect(pluginMarketplaceWidget_, &PluginMarketplaceWidget::releasePluginLoadersRequested, this, [this]() {
        pluginManager_.scan(QStringList());
    });
    connect(pluginMarketplaceWidget_, &PluginMarketplaceWidget::pluginsChanged, this, &MainWindow::refreshPlugins);
    connect(pluginMarketplaceWidget_, &PluginMarketplaceWidget::statusChanged, this, [this](const QString& status) {
        if (pluginMarketplaceStatusLabel_) {
            pluginMarketplaceStatusLabel_->setText(compactTextForStatus(status, 108));
            pluginMarketplaceStatusLabel_->setToolTip(status);
        }
    });
    pluginMarketplaceWidget_->refreshInstalledPlugins();

    auto* tabs = new QTabWidget;
    auto* loadedTab = new QWidget;
    auto* loadedLayout = new QVBoxLayout(loadedTab);
    loadedLayout->setContentsMargins(0, 0, 0, 0);
    loadedLayout->addWidget(pluginTable_);

    tabs->addTab(loadedTab, QStringLiteral("已加载"));
    tabs->addTab(pluginMarketplaceWidget_, QStringLiteral("插件市场"));
    tablePanel->bodyLayout()->addWidget(tabs);

    layout->addWidget(headerPanel);
    layout->addWidget(summaryStrip);
    layout->addWidget(tablePanel, 1);
    updatePluginSummary();
    return page;
}

QWidget* MainWindow::buildEnvironmentPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* runButton = primaryButton(QStringLiteral("执行环境自检"));
    connect(runButton, &QPushButton::clicked, this, &MainWindow::runEnvironmentCheck);

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
    auto* kicker = new QLabel(QStringLiteral("RUNTIME HEALTH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("运行时健康面板"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("检查 NVIDIA 驱动、CUDA、TensorRT、ONNX Runtime、Qt 插件和 Worker 可用性。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(runButton);
    headerRoot->addLayout(headerTop);

    environmentConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("尚未执行环境自检。"));
    environmentConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(environmentConsoleStatusLabel_);
    headerRoot->addWidget(environmentConsoleStatusLabel_);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* okCard = createMetricCard(QStringLiteral("通过"), QStringLiteral("0"), QStringLiteral("可用依赖"));
    environmentOkSummaryLabel_ = okCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* warningCard = createMetricCard(QStringLiteral("警告"), QStringLiteral("0"), QStringLiteral("可继续但需关注"));
    environmentWarningSummaryLabel_ = warningCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* missingCard = createMetricCard(QStringLiteral("缺失"), QStringLiteral("0"), QStringLiteral("会阻塞相关能力"));
    environmentMissingSummaryLabel_ = missingCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* uncheckedCard = createMetricCard(QStringLiteral("未检测"), QStringLiteral("0"), QStringLiteral("等待 Worker 自检"));
    environmentUncheckedSummaryLabel_ = uncheckedCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    summaryLayout->addWidget(okCard, 0, 0);
    summaryLayout->addWidget(warningCard, 0, 1);
    summaryLayout->addWidget(missingCard, 0, 2);
    summaryLayout->addWidget(uncheckedCard, 0, 3);

    auto* panel = new InfoPanel(QStringLiteral("检查明细"));
    environmentTable_ = new QTableWidget(0, 3);
    environmentTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("检查项") << QStringLiteral("状态") << QStringLiteral("说明"));
    configureTable(environmentTable_);
    environmentTable_->setWordWrap(true);
    environmentTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    environmentTable_->verticalHeader()->setDefaultSectionSize(42);
    const QStringList rows = {
        QStringLiteral("NVIDIA Driver"),
        QStringLiteral("CUDA Runtime"),
        QStringLiteral("cuDNN"),
        QStringLiteral("TensorRT"),
        QStringLiteral("ONNX Runtime"),
        QStringLiteral("LibTorch"),
        QStringLiteral("Qt Plugins"),
        QStringLiteral("AITrain Plugins"),
        QStringLiteral("Worker")
    };
    for (const QString& rowName : rows) {
        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        environmentTable_->setItem(row, 0, new QTableWidgetItem(rowName));
        environmentTable_->setItem(row, 1, new QTableWidgetItem(uiText("未检测")));
        environmentTable_->setItem(row, 2, new QTableWidgetItem(uiText("点击执行环境自检。")));
    }
    panel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("TensorRT 真机验收仍需要 RTX / SM 75+。当前 GTX 1060 / SM 61 只能记录为 hardware-blocked。")));
    panel->bodyLayout()->addWidget(environmentTable_);
    layout->addWidget(headerPanel);
    layout->addWidget(summaryStrip);
    layout->addWidget(panel, 1);
    updateEnvironmentSummary();
    return page;
}

QWidget* MainWindow::buildSettingsPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

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
    auto* kicker = new QLabel(QStringLiteral("APPLICATION SETTINGS"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(uiText("系统设置"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(uiText("集中管理轻量级应用偏好、授权状态和本地系统入口；训练、导出和推理仍由各自页面和 Worker 执行。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerRoot->addLayout(headerTop);

    auto* languagePanel = new InfoPanel(uiText("界面语言"));
    auto* languageRow = new QFrame;
    languageRow->setObjectName(QStringLiteral("ActionStrip"));
    auto* languageLayout = new QHBoxLayout(languageRow);
    languageLayout->setContentsMargins(10, 8, 10, 8);
    languageLayout->setSpacing(10);
    auto* languageHint = mutedLabel(uiText("保存后需要重启 AITrain Studio 才会完全切换界面语言。"));
    allowLabelToShrink(languageHint);
    languageLayout->addWidget(languageHint, 1);
    auto* languageSwitch = new QFrame;
    languageSwitch->setObjectName(QStringLiteral("LanguageSwitch"));
    auto* switchLayout = new QHBoxLayout(languageSwitch);
    switchLayout->setContentsMargins(2, 2, 2, 2);
    switchLayout->setSpacing(0);
    settingsZhLanguageButton_ = new QToolButton;
    settingsZhLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    settingsZhLanguageButton_->setText(QStringLiteral("中"));
    settingsZhLanguageButton_->setCheckable(true);
    settingsZhLanguageButton_->setCursor(Qt::PointingHandCursor);
    settingsZhLanguageButton_->setToolTip(uiText("切换到中文，重启后生效"));
    settingsEnLanguageButton_ = new QToolButton;
    settingsEnLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    settingsEnLanguageButton_->setText(QStringLiteral("EN"));
    settingsEnLanguageButton_->setCheckable(true);
    settingsEnLanguageButton_->setCursor(Qt::PointingHandCursor);
    settingsEnLanguageButton_->setToolTip(uiText("切换到英文，重启后生效"));
    switchLayout->addWidget(settingsZhLanguageButton_);
    switchLayout->addWidget(settingsEnLanguageButton_);
    languageLayout->addWidget(languageSwitch);
    connect(settingsZhLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("zh_CN"));
    });
    connect(settingsEnLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("en_US"));
    });
    languagePanel->bodyLayout()->addWidget(languageRow);

    auto* projectPathPanel = new InfoPanel(uiText("默认项目目录"));
    auto* projectPathForm = new QFormLayout;
    projectPathForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    projectPathForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    projectPathForm->setHorizontalSpacing(14);
    projectPathForm->setVerticalSpacing(10);
    settingsDefaultProjectPathEdit_ = new QLineEdit(QDir::toNativeSeparators(configuredDefaultProjectPath()));
    settingsDefaultProjectPathEdit_->setPlaceholderText(QDir::toNativeSeparators(defaultProjectPath()));
    auto* browseDefaultProjectButton = new QPushButton(uiText("选择目录"));
    auto* defaultPathRow = new QWidget;
    auto* defaultPathLayout = new QHBoxLayout(defaultPathRow);
    defaultPathLayout->setContentsMargins(0, 0, 0, 0);
    defaultPathLayout->setSpacing(8);
    defaultPathLayout->addWidget(settingsDefaultProjectPathEdit_, 1);
    defaultPathLayout->addWidget(browseDefaultProjectButton);
    projectPathForm->addRow(uiText("默认项目目录"), defaultPathRow);
    projectPathPanel->bodyLayout()->addLayout(projectPathForm);
    auto* defaultPathActions = new QFrame;
    defaultPathActions->setObjectName(QStringLiteral("ActionStrip"));
    auto* defaultPathActionLayout = new QHBoxLayout(defaultPathActions);
    defaultPathActionLayout->setContentsMargins(10, 8, 10, 8);
    defaultPathActionLayout->setSpacing(10);
    settingsDefaultProjectPathStatusLabel_ = mutedLabel(uiText("默认项目目录会作为项目页的初始路径；不会自动打开或迁移现有项目。"));
    allowLabelToShrink(settingsDefaultProjectPathStatusLabel_);
    auto* saveDefaultPathButton = primaryButton(uiText("保存默认目录"));
    auto* resetDefaultPathButton = new QPushButton(uiText("恢复默认"));
    defaultPathActionLayout->addWidget(settingsDefaultProjectPathStatusLabel_, 1);
    defaultPathActionLayout->addWidget(resetDefaultPathButton);
    defaultPathActionLayout->addWidget(saveDefaultPathButton);
    projectPathPanel->bodyLayout()->addWidget(defaultPathActions);
    connect(browseDefaultProjectButton, &QPushButton::clicked, this, [this]() {
        const QString directory = QFileDialog::getExistingDirectory(
            this,
            uiText("请选择默认项目目录"),
            QDir::fromNativeSeparators(settingsDefaultProjectPathEdit_->text().trimmed()));
        if (!directory.isEmpty()) {
            settingsDefaultProjectPathEdit_->setText(QDir::toNativeSeparators(directory));
        }
    });
    connect(saveDefaultPathButton, &QPushButton::clicked, this, [this]() {
        storeDefaultProjectPathPreference(settingsDefaultProjectPathEdit_->text());
    });
    connect(resetDefaultPathButton, &QPushButton::clicked, this, [this]() {
        settingsDefaultProjectPathEdit_->setText(QDir::toNativeSeparators(defaultProjectPath()));
        storeDefaultProjectPathPreference(defaultProjectPath());
        if (settingsDefaultProjectPathStatusLabel_) {
            settingsDefaultProjectPathStatusLabel_->setText(uiText("默认项目目录已恢复。"));
        }
    });

    auto* licensePanel = new InfoPanel(uiText("授权状态"));
    auto* licenseForm = new QFormLayout;
    licenseForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    licenseForm->setHorizontalSpacing(14);
    licenseForm->setVerticalSpacing(8);
    licenseForm->addRow(uiText("授权用户"), new QLabel(licenseOwner_.isEmpty() ? uiText("已注册") : licenseOwner_));
    licenseForm->addRow(uiText("有效期"), new QLabel(licenseExpiry_.isEmpty() ? uiText("未记录") : licenseExpiry_));
    licensePanel->bodyLayout()->addLayout(licenseForm);
    auto* licenseHint = mutedLabel(uiText("离线授权已在启动时校验；这里只展示当前授权信息，不提供换绑或激活入口。"));
    allowLabelToShrink(licenseHint);
    licensePanel->bodyLayout()->addWidget(licenseHint);

    auto* entryPanel = new InfoPanel(uiText("系统入口"));
    auto* entryActions = new QFrame;
    entryActions->setObjectName(QStringLiteral("ActionStrip"));
    auto* entryLayout = new QGridLayout(entryActions);
    entryLayout->setContentsMargins(10, 8, 10, 8);
    entryLayout->setHorizontalSpacing(10);
    entryLayout->setVerticalSpacing(10);
    auto* openProjectButton = primaryButton(uiText("打开项目页"));
    auto* openPluginsButton = new QPushButton(uiText("打开插件页"));
    auto* openEnvironmentButton = new QPushButton(uiText("打开环境页"));
    auto* runEnvironmentButton = new QPushButton(uiText("执行环境自检"));
    connect(openProjectButton, &QPushButton::clicked, this, [this]() { showPage(ProjectPage, uiText("项目")); });
    connect(openPluginsButton, &QPushButton::clicked, this, [this]() { showPage(PluginsPage, uiText("插件")); });
    connect(openEnvironmentButton, &QPushButton::clicked, this, [this]() { showPage(EnvironmentPage, uiText("环境")); });
    connect(runEnvironmentButton, &QPushButton::clicked, this, [this]() {
        showPage(EnvironmentPage, uiText("环境"));
        runEnvironmentCheck();
    });
    entryLayout->addWidget(openProjectButton, 0, 0);
    entryLayout->addWidget(openPluginsButton, 0, 1);
    entryLayout->addWidget(openEnvironmentButton, 1, 0);
    entryLayout->addWidget(runEnvironmentButton, 1, 1);
    entryPanel->bodyLayout()->addWidget(entryActions);
    auto* entryHint = mutedLabel(uiText("环境自检仍通过 Worker 执行，当前有任务运行时会沿用现有 busy guard。"));
    allowLabelToShrink(entryHint);
    entryPanel->bodyLayout()->addWidget(entryHint);

    auto* pathsPanel = new InfoPanel(uiText("本地路径"));
    auto* pathsGrid = new QGridLayout;
    pathsGrid->setHorizontalSpacing(10);
    pathsGrid->setVerticalSpacing(8);
    const QString appDir = QApplication::applicationDirPath();
    const auto addStaticPathRow = [this, pathsGrid](int row, const QString& labelText, const QString& path) {
        auto* label = new QLabel(labelText);
        auto* edit = new QLineEdit(QDir::toNativeSeparators(path));
        edit->setReadOnly(true);
        auto* openButton = new QPushButton(uiText("打开目录"));
        auto* copyButton = new QPushButton(uiText("复制路径"));
        connect(openButton, &QPushButton::clicked, this, [this, path]() { openLocalDirectory(path); });
        connect(copyButton, &QPushButton::clicked, this, [this, path, labelText]() { copyLocalPath(path, labelText); });
        pathsGrid->addWidget(label, row, 0);
        pathsGrid->addWidget(edit, row, 1);
        pathsGrid->addWidget(openButton, row, 2);
        pathsGrid->addWidget(copyButton, row, 3);
    };
    addStaticPathRow(0, uiText("插件目录"), QDir(appDir).filePath(QStringLiteral("plugins/models")));
    addStaticPathRow(1, uiText("插件市场目录"), QDir(appDir).filePath(QStringLiteral("plugins/marketplace")));
    auto* currentProjectLabel = new QLabel(uiText("当前项目目录"));
    settingsCurrentProjectPathLabel_ = inlineStatusLabel(uiText("未打开项目"));
    allowLabelToShrink(settingsCurrentProjectPathLabel_);
    auto* openCurrentProjectButton = new QPushButton(uiText("打开目录"));
    auto* copyCurrentProjectButton = new QPushButton(uiText("复制路径"));
    connect(openCurrentProjectButton, &QPushButton::clicked, this, [this]() { openLocalDirectory(currentProjectPath_); });
    connect(copyCurrentProjectButton, &QPushButton::clicked, this, [this]() { copyLocalPath(currentProjectPath_, uiText("当前项目目录")); });
    pathsGrid->addWidget(currentProjectLabel, 2, 0);
    pathsGrid->addWidget(settingsCurrentProjectPathLabel_, 2, 1);
    pathsGrid->addWidget(openCurrentProjectButton, 2, 2);
    pathsGrid->addWidget(copyCurrentProjectButton, 2, 3);
    pathsGrid->setColumnStretch(1, 1);
    pathsPanel->bodyLayout()->addLayout(pathsGrid);

    layout->addWidget(headerPanel);
    layout->addWidget(languagePanel);
    layout->addWidget(projectPathPanel);
    layout->addWidget(licensePanel);
    layout->addWidget(entryPanel);
    layout->addWidget(pathsPanel);
    layout->addStretch();

    page->setWidget(content);
    updateLanguageButtonState();
    updateSettingsSummary();
    return page;
}

InfoPanel* MainWindow::createMetricCard(const QString& label, const QString& value, const QString& caption)
{
    auto* panel = new InfoPanel(label);
    auto* valueLabel = new QLabel(value);
    valueLabel->setObjectName(QStringLiteral("MetricValue"));
    auto* captionLabel = new QLabel(caption);
    captionLabel->setObjectName(QStringLiteral("MetricLabel"));
    captionLabel->setWordWrap(true);
    panel->bodyLayout()->addWidget(valueLabel);
    panel->bodyLayout()->addWidget(captionLabel);
    return panel;
}

void MainWindow::configureTable(QTableWidget* table) const
{
    table->setAlternatingRowColors(true);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setStretchLastSection(true);
    table->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    table->setShowGrid(false);
}
