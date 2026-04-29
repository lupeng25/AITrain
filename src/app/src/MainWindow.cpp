#include "MainWindow.h"

#include "InfoPanel.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSplitter>
#include <QStatusBar>
#include <QTime>
#include <QVBoxLayout>
#include <QUuid>

namespace {

QLabel* mutedLabel(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("MutedText"));
    label->setWordWrap(true);
    return label;
}

QPushButton* primaryButton(const QString& text)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("PrimaryButton"));
    button->setCursor(Qt::PointingHandCursor);
    return button;
}

QPushButton* dangerButton(const QString& text)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("DangerButton"));
    button->setCursor(Qt::PointingHandCursor);
    return button;
}

QString taskStateLabel(aitrain::TaskState state)
{
    switch (state) {
    case aitrain::TaskState::Queued: return QStringLiteral("排队中");
    case aitrain::TaskState::Running: return QStringLiteral("运行中");
    case aitrain::TaskState::Paused: return QStringLiteral("已暂停");
    case aitrain::TaskState::Completed: return QStringLiteral("已完成");
    case aitrain::TaskState::Failed: return QStringLiteral("失败");
    case aitrain::TaskState::Canceled: return QStringLiteral("已取消");
    }
    return QStringLiteral("未知");
}

QString taskKindLabel(aitrain::TaskKind kind)
{
    switch (kind) {
    case aitrain::TaskKind::Train: return QStringLiteral("训练");
    case aitrain::TaskKind::Validate: return QStringLiteral("校验");
    case aitrain::TaskKind::Export: return QStringLiteral("导出");
    case aitrain::TaskKind::Infer: return QStringLiteral("推理");
    }
    return QStringLiteral("任务");
}

QString environmentStatusLabel(const QString& status)
{
    if (status == QStringLiteral("ok")) {
        return QStringLiteral("通过");
    }
    if (status == QStringLiteral("warning")) {
        return QStringLiteral("警告");
    }
    if (status == QStringLiteral("missing")) {
        return QStringLiteral("缺失");
    }
    return QStringLiteral("未知");
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("AITrain Studio"));
    setMinimumSize(1180, 760);

    auto* central = new QWidget(this);
    auto* rootLayout = new QHBoxLayout(central);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    sidebar_ = new Sidebar;
    sidebar_->addItem(QStringLiteral("首页"), DashboardPage);
    sidebar_->addItem(QStringLiteral("项目"), ProjectPage);
    sidebar_->addItem(QStringLiteral("数据集"), DatasetPage);
    sidebar_->addItem(QStringLiteral("训练"), TrainingPage);
    sidebar_->addItem(QStringLiteral("任务队列"), TaskQueuePage);
    sidebar_->addItem(QStringLiteral("模型转换"), ConversionPage);
    sidebar_->addItem(QStringLiteral("推理验证"), InferencePage);
    sidebar_->addItem(QStringLiteral("插件"), PluginsPage);
    sidebar_->addItem(QStringLiteral("环境"), EnvironmentPage);
    rootLayout->addWidget(sidebar_);

    auto* content = new QWidget;
    auto* contentLayout = new QVBoxLayout(content);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(0);
    contentLayout->addWidget(buildTopBar());

    stack_ = new QStackedWidget;
    stack_->addWidget(buildDashboardPage());
    stack_->addWidget(buildProjectPage());
    stack_->addWidget(buildDatasetPage());
    stack_->addWidget(buildTrainingPage());
    stack_->addWidget(buildTaskQueuePage());
    stack_->addWidget(buildConversionPage());
    stack_->addWidget(buildInferencePage());
    stack_->addWidget(buildPluginsPage());
    stack_->addWidget(buildEnvironmentPage());
    contentLayout->addWidget(stack_, 1);

    rootLayout->addWidget(content, 1);
    setCentralWidget(central);

    statusBar()->showMessage(QStringLiteral("就绪"));

    connect(sidebar_, &Sidebar::pageRequested, this, &MainWindow::showPage);
    connect(&worker_, &WorkerClient::messageReceived, this, &MainWindow::handleWorkerMessage);
    connect(&worker_, &WorkerClient::logLine, this, &MainWindow::appendLog);
    connect(&worker_, &WorkerClient::connected, this, [this]() {
        workerPill_->setStatus(QStringLiteral("Worker 已连接"), StatusPill::Tone::Success);
    });
    connect(&worker_, &WorkerClient::idle, this, &MainWindow::startNextQueuedTask);
    connect(&worker_, &WorkerClient::finished, this, [this](bool ok, const QString& message) {
        progressBar_->setValue(ok ? 100 : progressBar_->value());
        workerPill_->setStatus(ok ? QStringLiteral("任务完成") : QStringLiteral("任务失败"),
            ok ? StatusPill::Tone::Success : StatusPill::Tone::Error);
        appendLog(ok ? QStringLiteral("任务完成：%1").arg(message) : QStringLiteral("任务失败：%1").arg(message));
        if (!currentTaskId_.isEmpty()) {
            QString error;
            repository_.updateTaskState(currentTaskId_, ok ? aitrain::TaskState::Completed : aitrain::TaskState::Failed, message, &error);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        startNextQueuedTask();
    });

    refreshPlugins();
    showPage(DashboardPage, QStringLiteral("首页"));
    updateHeaderState();
}

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
    pageTitle_ = new QLabel(QStringLiteral("首页"));
    pageTitle_->setObjectName(QStringLiteral("PageTitle"));
    pageCaption_ = new QLabel;
    pageCaption_->setObjectName(QStringLiteral("PageCaption"));
    titleLayout->addWidget(pageTitle_);
    titleLayout->addWidget(pageCaption_);

    headerProjectLabel_ = new QLabel(QStringLiteral("项目：未打开"));
    headerProjectLabel_->setObjectName(QStringLiteral("MutedText"));
    workerPill_ = new StatusPill;
    workerPill_->setStatus(QStringLiteral("Worker 空闲"), StatusPill::Tone::Neutral);
    pluginPill_ = new StatusPill;
    gpuPill_ = new StatusPill;
    gpuPill_->setStatus(QStringLiteral("GPU 未检测"), StatusPill::Tone::Warning);

    layout->addWidget(titleBlock, 1);
    layout->addWidget(headerProjectLabel_);
    layout->addWidget(workerPill_);
    layout->addWidget(pluginPill_);
    layout->addWidget(gpuPill_);
    return topBar;
}

QWidget* MainWindow::buildDashboardPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    projectLabel_ = new QLabel(QStringLiteral("当前项目：未打开"));
    gpuLabel_ = new QLabel(QStringLiteral("GPU：未执行环境自检"));

    auto* grid = new QGridLayout;
    grid->setSpacing(12);
    auto* projectCard = createMetricCard(QStringLiteral("当前项目"), QStringLiteral("未打开"), QStringLiteral("创建或打开项目后开始训练"));
    dashboardProjectValue_ = projectCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(projectCard, 0, 0);
    auto* taskCard = createMetricCard(QStringLiteral("任务记录"), QStringLiteral("0"), QStringLiteral("当前项目最近任务"));
    dashboardTaskValue_ = taskCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(taskCard, 0, 1);
    auto* pluginCard = createMetricCard(QStringLiteral("插件"), QStringLiteral("0"), QStringLiteral("已加载模型插件"));
    dashboardPluginValue_ = pluginCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(pluginCard, 0, 2);
    grid->addWidget(createMetricCard(QStringLiteral("环境"), QStringLiteral("待检测"), QStringLiteral("CUDA / TensorRT / Worker")), 0, 3);

    auto* bottom = new QSplitter(Qt::Horizontal);

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

    auto* quickPanel = new InfoPanel(QStringLiteral("快速操作"));
    auto* createProjectButton = primaryButton(QStringLiteral("创建 / 打开项目"));
    auto* validateButton = new QPushButton(QStringLiteral("校验数据集"));
    auto* startButton = new QPushButton(QStringLiteral("启动训练"));
    auto* queueButton = new QPushButton(QStringLiteral("任务队列"));
    auto* envButton = new QPushButton(QStringLiteral("查看环境"));
    connect(createProjectButton, &QPushButton::clicked, this, [this]() { showPage(ProjectPage, QStringLiteral("项目")); sidebar_->setCurrentIndex(ProjectPage); });
    connect(validateButton, &QPushButton::clicked, this, [this]() { showPage(DatasetPage, QStringLiteral("数据集")); sidebar_->setCurrentIndex(DatasetPage); });
    connect(startButton, &QPushButton::clicked, this, [this]() { showPage(TrainingPage, QStringLiteral("训练")); sidebar_->setCurrentIndex(TrainingPage); });
    connect(queueButton, &QPushButton::clicked, this, [this]() { showPage(TaskQueuePage, QStringLiteral("任务队列")); sidebar_->setCurrentIndex(TaskQueuePage); });
    connect(envButton, &QPushButton::clicked, this, [this]() { showPage(EnvironmentPage, QStringLiteral("环境")); sidebar_->setCurrentIndex(EnvironmentPage); });
    quickPanel->bodyLayout()->addWidget(createProjectButton);
    quickPanel->bodyLayout()->addWidget(validateButton);
    quickPanel->bodyLayout()->addWidget(startButton);
    quickPanel->bodyLayout()->addWidget(queueButton);
    quickPanel->bodyLayout()->addWidget(envButton);
    quickPanel->bodyLayout()->addStretch();

    bottom->addWidget(recentPanel);
    bottom->addWidget(quickPanel);
    bottom->setStretchFactor(0, 3);
    bottom->setStretchFactor(1, 1);

    layout->addWidget(projectLabel_);
    layout->addWidget(gpuLabel_);
    layout->addLayout(grid);
    layout->addWidget(bottom, 1);
    return page;
}

QWidget* MainWindow::buildProjectPage()
{
    auto* page = new QWidget;
    auto* layout = new QHBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* formPanel = new InfoPanel(QStringLiteral("项目设置"));
    auto* form = new QFormLayout;
    form->setLabelAlignment(Qt::AlignRight);
    form->setFormAlignment(Qt::AlignTop);
    projectNameEdit_ = new QLineEdit(QStringLiteral("demo_project"));
    projectRootEdit_ = new QLineEdit(QDir::toNativeSeparators(QDir::currentPath() + QStringLiteral("/demo_project")));
    auto* browseButton = new QPushButton(QStringLiteral("选择目录"));
    auto* createButton = primaryButton(QStringLiteral("创建 / 打开项目"));

    connect(browseButton, &QPushButton::clicked, this, [this]() {
        const QString directory = QFileDialog::getExistingDirectory(this, QStringLiteral("选择项目目录"));
        if (!directory.isEmpty()) {
            projectRootEdit_->setText(QDir::toNativeSeparators(directory));
        }
    });
    connect(createButton, &QPushButton::clicked, this, &MainWindow::createProject);

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(projectRootEdit_);
    pathLayout->addWidget(browseButton);
    form->addRow(QStringLiteral("项目名称"), projectNameEdit_);
    form->addRow(QStringLiteral("项目目录"), pathRow);
    formPanel->bodyLayout()->addLayout(form);
    formPanel->bodyLayout()->addWidget(createButton);
    formPanel->bodyLayout()->addStretch();

    auto* structurePanel = new InfoPanel(QStringLiteral("项目目录结构"));
    auto* structure = new QPlainTextEdit;
    structure->setReadOnly(true);
    structure->setPlainText(QStringLiteral("datasets/\n  raw/\n  normalized/\nruns/\n  <task-id>/\nmodels/\n  exported/\nproject.sqlite"));
    structurePanel->bodyLayout()->addWidget(structure);
    structurePanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("创建项目时会自动生成 datasets、runs、models 和 project.sqlite。")));

    layout->addWidget(formPanel, 2);
    layout->addWidget(structurePanel, 1);
    return page;
}

QWidget* MainWindow::buildDatasetPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* inputPanel = new InfoPanel(QStringLiteral("数据集导入与校验"));
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
    auto* validateButton = primaryButton(QStringLiteral("校验数据集"));
    connect(validateButton, &QPushButton::clicked, this, &MainWindow::validateDataset);
    form->addRow(QStringLiteral("数据集目录"), pathRow);
    form->addRow(QStringLiteral("格式"), datasetFormatCombo_);
    inputPanel->bodyLayout()->addLayout(form);
    inputPanel->bodyLayout()->addWidget(validateButton);

    auto* splitter = new QSplitter(Qt::Horizontal);
    auto* resultPanel = new InfoPanel(QStringLiteral("校验结果"));
    validationOutput_ = new QPlainTextEdit;
    validationOutput_->setReadOnly(true);
    validationOutput_->setPlainText(QStringLiteral("请选择数据集目录和格式，然后执行校验。"));
    resultPanel->bodyLayout()->addWidget(validationOutput_);

    auto* toolsPanel = new InfoPanel(QStringLiteral("标注工具与预览"));
    auto* toolPath = new QLineEdit;
    toolPath->setPlaceholderText(QStringLiteral("LabelMe / AnyLabeling 可执行文件路径"));
    toolsPanel->bodyLayout()->addWidget(toolPath);
    toolsPanel->bodyLayout()->addWidget(new QPushButton(QStringLiteral("启动标注工具")));
    toolsPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("第一轮 UI 重构保留外部标注工具入口，后续数据集阶段再实现样本预览和错误定位。")));
    toolsPanel->bodyLayout()->addStretch();

    splitter->addWidget(resultPanel);
    splitter->addWidget(toolsPanel);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 1);

    layout->addWidget(inputPanel);
    layout->addWidget(splitter, 1);
    return page;
}

QWidget* MainWindow::buildTrainingPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(12);

    auto* topSplitter = new QSplitter(Qt::Horizontal);

    auto* configPanel = new InfoPanel(QStringLiteral("训练配置"));
    auto* form = new QFormLayout;
    pluginCombo_ = new QComboBox;
    taskTypeCombo_ = new QComboBox;
    epochsEdit_ = new QLineEdit(QStringLiteral("20"));
    batchEdit_ = new QLineEdit(QStringLiteral("8"));
    imageSizeEdit_ = new QLineEdit(QStringLiteral("640"));
    connect(pluginCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        taskTypeCombo_->clear();
        auto* plugin = pluginManager_.pluginById(pluginCombo_->currentData().toString());
        if (plugin) {
            taskTypeCombo_->addItems(plugin->manifest().taskTypes);
        }
    });
    form->addRow(QStringLiteral("插件"), pluginCombo_);
    form->addRow(QStringLiteral("任务类型"), taskTypeCombo_);
    form->addRow(QStringLiteral("Epochs"), epochsEdit_);
    form->addRow(QStringLiteral("Batch Size"), batchEdit_);
    form->addRow(QStringLiteral("Image Size"), imageSizeEdit_);
    auto* startButton = primaryButton(QStringLiteral("启动训练"));
    auto* pauseButton = new QPushButton(QStringLiteral("暂停任务"));
    auto* resumeButton = new QPushButton(QStringLiteral("继续任务"));
    auto* cancelButton = dangerButton(QStringLiteral("取消任务"));
    connect(startButton, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(pauseButton, &QPushButton::clicked, &worker_, &WorkerClient::pause);
    connect(resumeButton, &QPushButton::clicked, &worker_, &WorkerClient::resume);
    connect(cancelButton, &QPushButton::clicked, &worker_, &WorkerClient::cancel);
    configPanel->bodyLayout()->addLayout(form);
    configPanel->bodyLayout()->addWidget(startButton);
    configPanel->bodyLayout()->addWidget(pauseButton);
    configPanel->bodyLayout()->addWidget(resumeButton);
    configPanel->bodyLayout()->addWidget(cancelButton);
    configPanel->bodyLayout()->addStretch();

    auto* monitorPanel = new InfoPanel(QStringLiteral("训练监控"));
    progressBar_ = new QProgressBar;
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    metricsWidget_ = new MetricsWidget;
    monitorPanel->bodyLayout()->addWidget(progressBar_);
    monitorPanel->bodyLayout()->addWidget(metricsWidget_, 1);

    auto* artifactPanel = new InfoPanel(QStringLiteral("任务与产物"));
    artifactPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("运行后将显示 checkpoint、request.json、metrics 和导出产物。")));
    artifactPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("当前训练核心仍为 Worker 流程模拟，真实 LibTorch 训练将在后续阶段接入。")));
    artifactPanel->bodyLayout()->addStretch();

    topSplitter->addWidget(configPanel);
    topSplitter->addWidget(monitorPanel);
    topSplitter->addWidget(artifactPanel);
    topSplitter->setStretchFactor(0, 1);
    topSplitter->setStretchFactor(1, 2);
    topSplitter->setStretchFactor(2, 1);

    auto* logPanel = new InfoPanel(QStringLiteral("训练日志"));
    logEdit_ = new QTextEdit;
    logEdit_->setObjectName(QStringLiteral("LogView"));
    logEdit_->setReadOnly(true);
    logPanel->bodyLayout()->addWidget(logEdit_);

    layout->addWidget(topSplitter, 3);
    layout->addWidget(logPanel, 2);
    return page;
}

QWidget* MainWindow::buildTaskQueuePage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("队列操作"));
    auto* row = new QHBoxLayout;
    auto* refreshButton = primaryButton(QStringLiteral("刷新队列"));
    auto* cancelButton = dangerButton(QStringLiteral("取消选中任务"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::updateRecentTasks);
    connect(cancelButton, &QPushButton::clicked, this, &MainWindow::cancelSelectedTask);
    row->addWidget(refreshButton);
    row->addWidget(cancelButton);
    row->addStretch();
    toolbar->bodyLayout()->addLayout(row);
    toolbar->bodyLayout()->addWidget(mutedLabel(QStringLiteral("当前版本本机同一时间只运行一个 Worker 任务；真实多任务调度将在后续阶段继续扩展。")));

    auto* tablePanel = new InfoPanel(QStringLiteral("任务队列与历史"));
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
    tablePanel->bodyLayout()->addWidget(taskQueueTable_);

    layout->addWidget(toolbar);
    layout->addWidget(tablePanel, 1);
    return page;
}

QWidget* MainWindow::buildConversionPage()
{
    auto* page = new QWidget;
    auto* layout = new QHBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* inputPanel = new InfoPanel(QStringLiteral("输入模型"));
    inputPanel->bodyLayout()->addWidget(new QLineEdit);
    inputPanel->bodyLayout()->addWidget(new QPushButton(QStringLiteral("选择 checkpoint")));
    inputPanel->bodyLayout()->addStretch();

    auto* exportPanel = new InfoPanel(QStringLiteral("导出配置"));
    exportPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("后续将在 IExporter 中接入 ONNX 和 TensorRT 真实导出。")));
    exportPanel->bodyLayout()->addWidget(new QPushButton(QStringLiteral("导出模型")));
    exportPanel->bodyLayout()->addStretch();

    auto* resultPanel = new InfoPanel(QStringLiteral("导出结果"));
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("ONNX / TensorRT engine / 校验报告会显示在这里。")));
    resultPanel->bodyLayout()->addStretch();

    layout->addWidget(inputPanel);
    layout->addWidget(exportPanel);
    layout->addWidget(resultPanel);
    return page;
}

QWidget* MainWindow::buildInferencePage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("推理输入"));
    auto* row = new QHBoxLayout;
    row->addWidget(new QLineEdit);
    row->addWidget(new QPushButton(QStringLiteral("选择模型")));
    row->addWidget(new QPushButton(QStringLiteral("选择图片 / 视频")));
    row->addWidget(primaryButton(QStringLiteral("开始推理")));
    toolbar->bodyLayout()->addLayout(row);

    auto* preview = new QSplitter(Qt::Horizontal);
    auto* sourcePanel = new InfoPanel(QStringLiteral("输入预览"));
    sourcePanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("原图、视频帧或 OCR 输入图像。")));
    auto* resultPanel = new InfoPanel(QStringLiteral("结果预览"));
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("检测框、分割 mask、OCR 文本和耗时会显示在这里。")));
    preview->addWidget(sourcePanel);
    preview->addWidget(resultPanel);

    layout->addWidget(toolbar);
    layout->addWidget(preview, 1);
    return page;
}

QWidget* MainWindow::buildPluginsPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("插件管理"));
    auto* refreshButton = primaryButton(QStringLiteral("重新扫描插件"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshPlugins);
    toolbar->bodyLayout()->addWidget(refreshButton);

    auto* tablePanel = new InfoPanel(QStringLiteral("已加载插件"));
    pluginTable_ = new QTableWidget(0, 6);
    pluginTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID")
        << QStringLiteral("名称")
        << QStringLiteral("版本")
        << QStringLiteral("任务")
        << QStringLiteral("数据集")
        << QStringLiteral("导出"));
    configureTable(pluginTable_);
    tablePanel->bodyLayout()->addWidget(pluginTable_);

    layout->addWidget(toolbar);
    layout->addWidget(tablePanel, 1);
    return page;
}

QWidget* MainWindow::buildEnvironmentPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* panel = new InfoPanel(QStringLiteral("环境自检"));
    auto* runButton = primaryButton(QStringLiteral("执行环境自检"));
    connect(runButton, &QPushButton::clicked, this, &MainWindow::runEnvironmentCheck);
    environmentTable_ = new QTableWidget(0, 3);
    environmentTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("检查项") << QStringLiteral("状态") << QStringLiteral("说明"));
    configureTable(environmentTable_);
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
        environmentTable_->setItem(row, 1, new QTableWidgetItem(QStringLiteral("未检测")));
        environmentTable_->setItem(row, 2, new QTableWidgetItem(QStringLiteral("点击执行环境自检。")));
    }
    panel->bodyLayout()->addWidget(runButton);
    panel->bodyLayout()->addWidget(environmentTable_);
    layout->addWidget(panel);
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

QString MainWindow::pageCaption(int pageIndex) const
{
    switch (pageIndex) {
    case DashboardPage: return QStringLiteral("项目、任务、GPU 与插件状态总览");
    case ProjectPage: return QStringLiteral("创建项目并管理工作目录");
    case DatasetPage: return QStringLiteral("导入、校验和准备训练数据");
    case TrainingPage: return QStringLiteral("配置训练任务并实时监控指标");
    case TaskQueuePage: return QStringLiteral("查看排队、运行和历史任务状态");
    case ConversionPage: return QStringLiteral("导出 ONNX / TensorRT 模型产物");
    case InferencePage: return QStringLiteral("验证模型推理效果和耗时");
    case PluginsPage: return QStringLiteral("扫描和诊断模型插件");
    case EnvironmentPage: return QStringLiteral("检查 GPU、CUDA、TensorRT 和运行时依赖");
    default: return {};
    }
}

void MainWindow::showPage(int pageIndex, const QString& title)
{
    stack_->setCurrentIndex(pageIndex);
    pageTitle_->setText(title);
    pageCaption_->setText(pageCaption(pageIndex));
    sidebar_->setCurrentIndex(pageIndex);
}

void MainWindow::createProject()
{
    currentProjectName_ = projectNameEdit_->text().trimmed();
    currentProjectPath_ = QDir::fromNativeSeparators(projectRootEdit_->text().trimmed());
    if (currentProjectName_.isEmpty() || currentProjectPath_.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("项目"), QStringLiteral("项目名称和目录不能为空。"));
        return;
    }

    ensureProjectSubdirs(currentProjectPath_);
    QString error;
    if (!repository_.open(QDir(currentProjectPath_).filePath(QStringLiteral("project.sqlite")), &error)
        || !repository_.upsertProject(currentProjectName_, currentProjectPath_, &error)) {
        QMessageBox::critical(this, QStringLiteral("项目"), error);
        return;
    }
    repository_.markInterruptedTasksFailed(QStringLiteral("上次会话结束时任务未正常完成，已标记为失败。"), &error);

    projectLabel_->setText(QStringLiteral("当前项目：%1").arg(currentProjectPath_));
    if (dashboardProjectValue_) {
        dashboardProjectValue_->setText(currentProjectName_);
    }
    updateHeaderState();
    updateRecentTasks();
    statusBar()->showMessage(QStringLiteral("项目已打开：%1").arg(currentProjectName_), 5000);
}

void MainWindow::browseDataset()
{
    const QString directory = QFileDialog::getExistingDirectory(this, QStringLiteral("选择数据集目录"));
    if (!directory.isEmpty()) {
        datasetPathEdit_->setText(QDir::toNativeSeparators(directory));
    }
}

void MainWindow::validateDataset()
{
    validationOutput_->clear();
    const QString format = datasetFormatCombo_->currentText();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_->text());
    if (format.isEmpty() || path.isEmpty()) {
        validationOutput_->setPlainText(QStringLiteral("请选择数据集目录和格式。"));
        return;
    }

    for (auto* plugin : pluginManager_.plugins()) {
        if (auto* adapter = plugin->datasetAdapter(format)) {
            const aitrain::DatasetValidationResult result = adapter->validateDataset(path, {});
            validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(result.toJson()).toJson(QJsonDocument::Indented)));
            statusBar()->showMessage(result.ok ? QStringLiteral("数据集校验通过") : QStringLiteral("数据集校验发现问题"), 5000);
            return;
        }
    }
    validationOutput_->setPlainText(QStringLiteral("没有插件支持该数据集格式。"));
}

void MainWindow::startTraining()
{
    if (currentProjectPath_.isEmpty()) {
        createProject();
        if (currentProjectPath_.isEmpty()) {
            return;
        }
    }
    if (pluginCombo_->currentData().toString().isEmpty() || taskTypeCombo_->currentText().isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("训练"), QStringLiteral("请选择可用插件和任务类型。"));
        return;
    }

    const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString runDir = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
    QDir().mkpath(runDir);

    QJsonObject parameters;
    parameters.insert(QStringLiteral("epochs"), epochsEdit_->text().toInt());
    parameters.insert(QStringLiteral("batchSize"), batchEdit_->text().toInt());
    parameters.insert(QStringLiteral("imageSize"), imageSizeEdit_->text().toInt());

    aitrain::TrainingRequest request;
    request.taskId = taskId;
    request.projectPath = currentProjectPath_;
    request.pluginId = pluginCombo_->currentData().toString();
    request.taskType = taskTypeCombo_->currentText();
    request.datasetPath = QDir::fromNativeSeparators(datasetPathEdit_->text());
    request.outputPath = runDir;
    request.parameters = parameters;

    aitrain::TaskRecord record;
    record.id = taskId;
    record.projectName = currentProjectName_;
    record.pluginId = request.pluginId;
    record.taskType = request.taskType;
    record.kind = aitrain::TaskKind::Train;
    record.state = aitrain::TaskState::Queued;
    record.workDir = runDir;
    record.message = worker_.isRunning() ? QStringLiteral("等待当前任务完成。") : QStringLiteral("等待 Worker 启动。");
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;
    QString error;
    if (!repository_.insertTask(record, &error)) {
        QMessageBox::critical(this, QStringLiteral("任务"), error);
        return;
    }

    if (worker_.isRunning()) {
        pendingTrainingTasks_.append(PendingTrainingTask{taskId, request});
        workerPill_->setStatus(QStringLiteral("任务已排队"), StatusPill::Tone::Info);
        appendLog(QStringLiteral("任务已加入队列：%1").arg(taskId));
        updateRecentTasks();
        return;
    }

    startQueuedTraining(taskId, request);
}

void MainWindow::cancelSelectedTask()
{
    if (!taskQueueTable_ || !repository_.isOpen()) {
        return;
    }

    const int row = taskQueueTable_->currentRow();
    if (row < 0 || !taskQueueTable_->item(row, 0)) {
        QMessageBox::information(this, QStringLiteral("任务队列"), QStringLiteral("请先选择一个任务。"));
        return;
    }

    const QString taskId = taskQueueTable_->item(row, 0)->data(Qt::UserRole).toString();
    if (taskId.isEmpty()) {
        return;
    }

    QString error;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(200, &error);
    for (const aitrain::TaskRecord& task : tasks) {
        if (task.id != taskId) {
            continue;
        }

        if (task.state == aitrain::TaskState::Queued) {
            for (int index = 0; index < pendingTrainingTasks_.size(); ++index) {
                if (pendingTrainingTasks_.at(index).taskId == taskId) {
                    pendingTrainingTasks_.remove(index);
                    break;
                }
            }
            if (!repository_.updateTaskState(taskId, aitrain::TaskState::Canceled, QStringLiteral("用户取消排队任务。"), &error)) {
                QMessageBox::warning(this, QStringLiteral("任务队列"), error);
            }
            updateRecentTasks();
            return;
        }

        if ((task.state == aitrain::TaskState::Running || task.state == aitrain::TaskState::Paused) && taskId == currentTaskId_) {
            worker_.cancel();
            return;
        }

        QMessageBox::information(this, QStringLiteral("任务队列"), QStringLiteral("只能取消排队任务或当前 Worker 正在运行的任务。"));
        return;
    }
}

void MainWindow::runEnvironmentCheck()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, QStringLiteral("环境自检"), QStringLiteral("Worker 正在执行任务，稍后再运行环境自检。"));
        return;
    }

    if (environmentTable_) {
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            environmentTable_->setItem(row, 1, new QTableWidgetItem(QStringLiteral("检测中")));
            environmentTable_->setItem(row, 2, new QTableWidgetItem(QStringLiteral("等待 Worker 返回结果。")));
        }
    }

    QString error;
    if (!worker_.requestEnvironmentCheck(workerExecutablePath(), &error)) {
        QMessageBox::critical(this, QStringLiteral("环境自检"), error);
        return;
    }
    workerPill_->setStatus(QStringLiteral("环境自检中"), StatusPill::Tone::Info);
}

void MainWindow::handleWorkerMessage(const QString& type, const QJsonObject& payload)
{
    if (type == QStringLiteral("progress")) {
        progressBar_->setValue(payload.value(QStringLiteral("percent")).toInt());
    } else if (type == QStringLiteral("metric")) {
        const QString name = payload.value(QStringLiteral("name")).toString();
        const double value = payload.value(QStringLiteral("value")).toDouble();
        metricsWidget_->addMetric(name, value);

        aitrain::MetricPoint point;
        point.taskId = currentTaskId_;
        point.name = name;
        point.value = value;
        point.step = payload.value(QStringLiteral("step")).toInt();
        point.epoch = payload.value(QStringLiteral("epoch")).toInt();
        point.createdAt = QDateTime::currentDateTimeUtc();
        QString error;
        repository_.insertMetric(point, &error);
    } else if (type == QStringLiteral("artifact")) {
        const QString path = payload.value(QStringLiteral("path")).toString();
        appendLog(QStringLiteral("产物：%1").arg(path));
        if (repository_.isOpen()) {
            aitrain::ArtifactRecord artifact;
            artifact.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
            artifact.kind = payload.value(QStringLiteral("kind")).toString();
            artifact.path = path;
            artifact.message = QStringLiteral("Worker 上报产物");
            artifact.createdAt = QDateTime::currentDateTimeUtc();
            QString error;
            repository_.insertArtifact(artifact, &error);
        }
    } else if (type == QStringLiteral("paused")) {
        QString error;
        repository_.updateTaskState(currentTaskId_, aitrain::TaskState::Paused, payload.value(QStringLiteral("message")).toString(), &error);
        workerPill_->setStatus(QStringLiteral("任务已暂停"), StatusPill::Tone::Warning);
        updateRecentTasks();
    } else if (type == QStringLiteral("resumed")) {
        QString error;
        repository_.updateTaskState(currentTaskId_, aitrain::TaskState::Running, payload.value(QStringLiteral("message")).toString(), &error);
        workerPill_->setStatus(QStringLiteral("训练运行中"), StatusPill::Tone::Info);
        updateRecentTasks();
    } else if (type == QStringLiteral("canceled")) {
        QString error;
        repository_.updateTaskState(currentTaskId_, aitrain::TaskState::Canceled, payload.value(QStringLiteral("message")).toString(), &error);
        workerPill_->setStatus(QStringLiteral("任务已取消"), StatusPill::Tone::Warning);
        appendLog(QStringLiteral("任务已取消：%1").arg(payload.value(QStringLiteral("message")).toString()));
        currentTaskId_.clear();
        updateRecentTasks();
        startNextQueuedTask();
    } else if (type == QStringLiteral("environmentCheck")) {
        updateEnvironmentTable(payload);
    }
}

void MainWindow::refreshPlugins()
{
    pluginManager_.scan(pluginSearchPaths());
    if (pluginTable_) {
        pluginTable_->setRowCount(0);
        for (auto* plugin : pluginManager_.plugins()) {
            const aitrain::PluginManifest manifest = plugin->manifest();
            const int row = pluginTable_->rowCount();
            pluginTable_->insertRow(row);
            pluginTable_->setItem(row, 0, new QTableWidgetItem(manifest.id));
            pluginTable_->setItem(row, 1, new QTableWidgetItem(manifest.name));
            pluginTable_->setItem(row, 2, new QTableWidgetItem(manifest.version));
            pluginTable_->setItem(row, 3, new QTableWidgetItem(manifest.taskTypes.join(QStringLiteral(", "))));
            pluginTable_->setItem(row, 4, new QTableWidgetItem(manifest.datasetFormats.join(QStringLiteral(", "))));
            pluginTable_->setItem(row, 5, new QTableWidgetItem(manifest.exportFormats.join(QStringLiteral(", "))));
        }
    }
    loadPluginCombos();
    updateHeaderState();
}

QString MainWindow::workerExecutablePath() const
{
    const QString name =
#if defined(Q_OS_WIN)
        QStringLiteral("aitrain_worker.exe");
#else
        QStringLiteral("aitrain_worker");
#endif
    const QString appDir = QApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).filePath(name),
        QDir(appDir).filePath(QStringLiteral("../worker/") + name),
        QDir(appDir).filePath(QStringLiteral("../bin/") + name)
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QDir::cleanPath(candidate);
        }
    }
    return QDir(appDir).filePath(name);
}

QStringList MainWindow::pluginSearchPaths() const
{
    const QString appDir = QApplication::applicationDirPath();
    return {
        QDir(appDir).filePath(QStringLiteral("plugins/models")),
        QDir(appDir).filePath(QStringLiteral("../plugins/models")),
        QDir(appDir).filePath(QStringLiteral("../../plugins/models"))
    };
}

void MainWindow::ensureProjectSubdirs(const QString& rootPath)
{
    QDir root(rootPath);
    root.mkpath(QStringLiteral("."));
    root.mkpath(QStringLiteral("datasets"));
    root.mkpath(QStringLiteral("runs"));
    root.mkpath(QStringLiteral("models"));
}

void MainWindow::appendLog(const QString& text)
{
    if (logEdit_) {
        logEdit_->append(QStringLiteral("[%1] %2").arg(QTime::currentTime().toString(QStringLiteral("HH:mm:ss")), text));
    }
}

void MainWindow::loadPluginCombos()
{
    const QString currentPlugin = pluginCombo_ ? pluginCombo_->currentData().toString() : QString();
    if (pluginCombo_) {
        pluginCombo_->clear();
    }
    if (datasetFormatCombo_) {
        datasetFormatCombo_->clear();
    }

    QStringList formats;
    for (auto* plugin : pluginManager_.plugins()) {
        const aitrain::PluginManifest manifest = plugin->manifest();
        if (pluginCombo_) {
            pluginCombo_->addItem(manifest.name, manifest.id);
        }
        for (const QString& format : manifest.datasetFormats) {
            if (!formats.contains(format)) {
                formats.append(format);
            }
        }
    }
    if (datasetFormatCombo_) {
        datasetFormatCombo_->addItems(formats);
    }
    if (pluginCombo_ && !currentPlugin.isEmpty()) {
        const int index = pluginCombo_->findData(currentPlugin);
        if (index >= 0) {
            pluginCombo_->setCurrentIndex(index);
        }
    }
}

void MainWindow::updateRecentTasks()
{
    if (!recentTasksTable_ || !repository_.isOpen()) {
        return;
    }
    QString error;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(20, &error);
    if (dashboardTaskValue_) {
        dashboardTaskValue_->setText(QString::number(tasks.size()));
    }
    updateTaskTable(recentTasksTable_, tasks);
    if (taskQueueTable_) {
        updateTaskTable(taskQueueTable_, tasks);
    }
}

void MainWindow::updateTaskTable(QTableWidget* table, const QVector<aitrain::TaskRecord>& tasks)
{
    if (!table) {
        return;
    }

    table->setRowCount(0);
    if (tasks.isEmpty()) {
        table->insertRow(0);
        auto* item = new QTableWidgetItem(QStringLiteral("暂无任务记录"));
        table->setItem(0, 0, item);
        for (int column = 1; column < table->columnCount(); ++column) {
            table->setItem(0, column, new QTableWidgetItem(QString()));
        }
        return;
    }

    for (const aitrain::TaskRecord& task : tasks) {
        const int row = table->rowCount();
        table->insertRow(row);
        auto* idItem = new QTableWidgetItem(task.id.left(8));
        idItem->setData(Qt::UserRole, task.id);
        table->setItem(row, 0, idItem);
        if (table->columnCount() == 5) {
            table->setItem(row, 1, new QTableWidgetItem(task.pluginId));
            table->setItem(row, 2, new QTableWidgetItem(task.taskType));
            table->setItem(row, 3, new QTableWidgetItem(taskStateLabel(task.state)));
            table->setItem(row, 4, new QTableWidgetItem(task.message));
        } else {
            table->setItem(row, 1, new QTableWidgetItem(taskKindLabel(task.kind)));
            table->setItem(row, 2, new QTableWidgetItem(task.pluginId));
            table->setItem(row, 3, new QTableWidgetItem(task.taskType));
            table->setItem(row, 4, new QTableWidgetItem(taskStateLabel(task.state)));
            table->setItem(row, 5, new QTableWidgetItem(task.updatedAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            table->setItem(row, 6, new QTableWidgetItem(task.message));
        }
    }
}

void MainWindow::updateHeaderState()
{
    headerProjectLabel_->setText(currentProjectPath_.isEmpty()
        ? QStringLiteral("项目：未打开")
        : QStringLiteral("项目：%1").arg(currentProjectName_));
    const int pluginCount = pluginManager_.plugins().size();
    pluginPill_->setStatus(QStringLiteral("插件 %1").arg(pluginCount), pluginCount > 0 ? StatusPill::Tone::Success : StatusPill::Tone::Warning);
    if (dashboardPluginValue_) {
        dashboardPluginValue_->setText(QString::number(pluginCount));
    }
}

void MainWindow::updateEnvironmentTable(const QJsonObject& payload)
{
    const QJsonArray checks = payload.value(QStringLiteral("checks")).toArray();
    if (!environmentTable_) {
        return;
    }

    bool hasMissing = false;
    bool hasWarning = false;
    environmentTable_->setRowCount(0);
    for (const QJsonValue& value : checks) {
        const QJsonObject check = value.toObject();
        const QString name = check.value(QStringLiteral("name")).toString();
        const QString status = check.value(QStringLiteral("status")).toString();
        const QString message = check.value(QStringLiteral("message")).toString();
        if (status == QStringLiteral("missing")) {
            hasMissing = true;
        } else if (status == QStringLiteral("warning")) {
            hasWarning = true;
        }

        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        environmentTable_->setItem(row, 0, new QTableWidgetItem(name));
        environmentTable_->setItem(row, 1, new QTableWidgetItem(environmentStatusLabel(status)));
        environmentTable_->setItem(row, 2, new QTableWidgetItem(message));

        if (repository_.isOpen()) {
            aitrain::EnvironmentCheckRecord record;
            record.name = name;
            record.status = status;
            record.message = message;
            record.detailsJson = QString::fromUtf8(QJsonDocument(check.value(QStringLiteral("details")).toObject()).toJson(QJsonDocument::Compact));
            record.checkedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
            QString error;
            repository_.insertEnvironmentCheck(record, &error);
        }
    }

    {
        const int pluginCount = pluginManager_.plugins().size();
        const QString status = pluginCount > 0 ? QStringLiteral("ok") : QStringLiteral("warning");
        const QString message = pluginCount > 0
            ? QStringLiteral("已加载 %1 个 AITrain 插件。").arg(pluginCount)
            : QStringLiteral("未加载 AITrain 插件，请检查 plugins/models 目录。");
        if (status == QStringLiteral("warning")) {
            hasWarning = true;
        }
        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        environmentTable_->setItem(row, 0, new QTableWidgetItem(QStringLiteral("AITrain Plugins")));
        environmentTable_->setItem(row, 1, new QTableWidgetItem(environmentStatusLabel(status)));
        environmentTable_->setItem(row, 2, new QTableWidgetItem(message));
        if (repository_.isOpen()) {
            aitrain::EnvironmentCheckRecord record;
            record.name = QStringLiteral("AITrain Plugins");
            record.status = status;
            record.message = message;
            record.checkedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
            QString error;
            repository_.insertEnvironmentCheck(record, &error);
        }
    }

    const StatusPill::Tone tone = hasMissing ? StatusPill::Tone::Error : (hasWarning ? StatusPill::Tone::Warning : StatusPill::Tone::Success);
    const QString text = hasMissing ? QStringLiteral("环境缺失") : (hasWarning ? QStringLiteral("环境警告") : QStringLiteral("环境通过"));
    gpuPill_->setStatus(text, tone);
    workerPill_->setStatus(QStringLiteral("Worker 空闲"), StatusPill::Tone::Neutral);
    gpuLabel_->setText(QStringLiteral("GPU / 运行时：%1").arg(text));
    statusBar()->showMessage(QStringLiteral("环境自检完成"), 5000);
}

void MainWindow::startQueuedTraining(const QString& taskId, const aitrain::TrainingRequest& request)
{
    metricsWidget_->clear();
    logEdit_->clear();
    progressBar_->setValue(0);
    currentTaskId_ = taskId;

    QString error;
    if (!worker_.startTraining(workerExecutablePath(), request, &error)) {
        repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, nullptr);
        currentTaskId_.clear();
        updateRecentTasks();
        QMessageBox::critical(this, QStringLiteral("Worker"), error);
        startNextQueuedTask();
        return;
    }

    repository_.updateTaskState(taskId, aitrain::TaskState::Running, QStringLiteral("started"), &error);
    workerPill_->setStatus(QStringLiteral("训练运行中"), StatusPill::Tone::Info);
    appendLog(QStringLiteral("任务已启动：%1").arg(taskId));
    updateRecentTasks();
}

void MainWindow::startNextQueuedTask()
{
    if (worker_.isRunning() || pendingTrainingTasks_.isEmpty()) {
        return;
    }

    const PendingTrainingTask next = pendingTrainingTasks_.takeFirst();
    startQueuedTraining(next.taskId, next.request);
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
