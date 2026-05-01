#include "MainWindow.h"

#include "InfoPanel.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QApplication>
#include <QCheckBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPixmap>
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

QString issueSeverityLabel(const QString& severity)
{
    if (severity == QStringLiteral("error")) {
        return QStringLiteral("错误");
    }
    if (severity == QStringLiteral("warning")) {
        return QStringLiteral("警告");
    }
    return QStringLiteral("信息");
}

QString inferenceTaskTypeLabel(const QString& taskType)
{
    if (taskType == QStringLiteral("segmentation")) {
        return QStringLiteral("分割");
    }
    if (taskType == QStringLiteral("ocr_recognition")) {
        return QStringLiteral("OCR 识别");
    }
    return QStringLiteral("检测");
}

QString confidencePercent(double confidence)
{
    return QStringLiteral("%1%").arg(QString::number(confidence * 100.0, 'f', 1));
}

QString inferenceSummaryFromPredictions(const QString& predictionsPath, const QJsonObject& fallback = {})
{
    const QString nativePath = QDir::toNativeSeparators(predictionsPath);
    QFile file(predictionsPath);
    if (!file.open(QIODevice::ReadOnly)) {
        const QString taskType = fallback.value(QStringLiteral("taskType")).toString(QStringLiteral("detection"));
        return QStringLiteral("%1：%2 个结果，%3 ms\n结果文件：%4")
            .arg(inferenceTaskTypeLabel(taskType))
            .arg(fallback.value(QStringLiteral("predictionCount")).toInt())
            .arg(fallback.value(QStringLiteral("elapsedMs")).toInt())
            .arg(nativePath);
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        return QStringLiteral("预测结果 JSON 无法解析：%1").arg(nativePath);
    }

    const QJsonObject root = document.object();
    const QString taskType = root.value(QStringLiteral("taskType")).toString(
        fallback.value(QStringLiteral("taskType")).toString(QStringLiteral("detection")));
    const QJsonArray predictions = root.value(QStringLiteral("predictions")).toArray();
    const int elapsedMs = root.value(QStringLiteral("elapsedMs")).toInt(fallback.value(QStringLiteral("elapsedMs")).toInt());
    QString detail;
    if (!predictions.isEmpty()) {
        const QJsonObject first = predictions.at(0).toObject();
        if (taskType == QStringLiteral("ocr_recognition")) {
            const QString text = first.value(QStringLiteral("text")).toString();
            detail = text.isEmpty()
                ? QStringLiteral("未识别出文本")
                : QStringLiteral("文本 \"%1\"").arg(text);
            if (first.contains(QStringLiteral("confidence"))) {
                detail.append(QStringLiteral("，置信度 %1").arg(confidencePercent(first.value(QStringLiteral("confidence")).toDouble())));
            }
        } else {
            const QString className = first.value(QStringLiteral("className")).toString(
                QStringLiteral("class %1").arg(first.value(QStringLiteral("classId")).toInt()));
            detail = QStringLiteral("首个 %1，置信度 %2")
                .arg(className)
                .arg(confidencePercent(first.value(QStringLiteral("confidence")).toDouble()));
            if (taskType == QStringLiteral("segmentation")) {
                detail.append(QStringLiteral("，mask area %1").arg(first.value(QStringLiteral("maskArea")).toInt()));
            }
        }
    } else {
        detail = QStringLiteral("无结果");
    }

    return QStringLiteral("%1：%2 个结果，%3，%4 ms\n结果文件：%5")
        .arg(inferenceTaskTypeLabel(taskType))
        .arg(predictions.size())
        .arg(detail)
        .arg(elapsedMs)
        .arg(nativePath);
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
        const QString kind;
        const QString path;
        if (!currentTaskId_.isEmpty()) {
            QString error;
            repository_.updateTaskState(currentTaskId_, ok ? aitrain::TaskState::Completed : aitrain::TaskState::Failed, message, &error);
            currentTaskId_.clear();
            updateRecentTasks();
        } else if (kind == QStringLiteral("export") && exportResultLabel_) {
            exportResultLabel_->setText(QStringLiteral("导出完成：%1").arg(QDir::toNativeSeparators(path)));
        } else if (kind == QStringLiteral("inference_overlay") && inferenceOverlayLabel_) {
            QPixmap overlay(path);
            if (!overlay.isNull()) {
                inferenceOverlayLabel_->setPixmap(overlay.scaled(
                    inferenceOverlayLabel_->size().boundedTo(QSize(520, 360)),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation));
            } else {
                inferenceOverlayLabel_->setText(QStringLiteral("推理 overlay 加载失败"));
            }
        } else if (kind == QStringLiteral("inference_predictions") && inferenceResultLabel_) {
            inferenceResultLabel_->setText(inferenceSummaryFromPredictions(path));
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

    splitOutputEdit_ = new QLineEdit;
    splitOutputEdit_->setPlaceholderText(QStringLiteral("默认输出到当前项目 datasets/normalized"));
    splitTrainRatioEdit_ = new QLineEdit(QStringLiteral("0.8"));
    splitValRatioEdit_ = new QLineEdit(QStringLiteral("0.2"));
    splitTestRatioEdit_ = new QLineEdit(QStringLiteral("0.0"));
    splitSeedEdit_ = new QLineEdit(QStringLiteral("42"));
    auto* splitButton = new QPushButton(QStringLiteral("划分数据集"));
    connect(splitButton, &QPushButton::clicked, this, &MainWindow::splitDataset);
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
    inputPanel->bodyLayout()->addWidget(validateButton);
    inputPanel->bodyLayout()->addWidget(splitButton);

    auto* splitter = new QSplitter(Qt::Horizontal);
    auto* resultPanel = new InfoPanel(QStringLiteral("校验结果"));
    validationSummaryLabel_ = mutedLabel(QStringLiteral("请选择数据集目录和格式，然后执行校验。"));
    validationIssuesTable_ = new QTableWidget(0, 5);
    validationIssuesTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("级别")
        << QStringLiteral("代码")
        << QStringLiteral("文件")
        << QStringLiteral("行号")
        << QStringLiteral("说明"));
    configureTable(validationIssuesTable_);
    validationOutput_ = new QPlainTextEdit;
    validationOutput_->setReadOnly(true);
    validationOutput_->setPlainText(QStringLiteral("校验报告 JSON 会显示在这里。"));
    resultPanel->bodyLayout()->addWidget(validationSummaryLabel_);
    resultPanel->bodyLayout()->addWidget(validationIssuesTable_, 2);
    resultPanel->bodyLayout()->addWidget(validationOutput_);

    auto* toolsPanel = new InfoPanel(QStringLiteral("标注工具与预览"));
    auto* toolPath = new QLineEdit;
    toolPath->setPlaceholderText(QStringLiteral("LabelMe / AnyLabeling 可执行文件路径"));
    datasetPreviewTable_ = new QTableWidget(0, 2);
    datasetPreviewTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("样本") << QStringLiteral("标签 / 说明"));
    configureTable(datasetPreviewTable_);
    toolsPanel->bodyLayout()->addWidget(toolPath);
    toolsPanel->bodyLayout()->addWidget(new QPushButton(QStringLiteral("启动标注工具")));
    toolsPanel->bodyLayout()->addWidget(datasetPreviewTable_, 1);
    toolsPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("划分会复制到新目录，不修改原始数据；当前划分先支持 YOLO 检测格式。")));
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
    gridSizeEdit_ = new QLineEdit(QStringLiteral("4"));
    resumeCheckpointEdit_ = new QLineEdit;
    resumeCheckpointEdit_->setPlaceholderText(QStringLiteral("可选：选择已有 tiny detector checkpoint 继续训练"));
    horizontalFlipCheck_ = new QCheckBox(QStringLiteral("水平翻转增强"));
    colorJitterCheck_ = new QCheckBox(QStringLiteral("亮度扰动增强"));
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
    form->addRow(QStringLiteral("Grid Size"), gridSizeEdit_);
    form->addRow(QStringLiteral("Resume"), resumeCheckpointEdit_);
    form->addRow(QStringLiteral("Augment"), horizontalFlipCheck_);
    form->addRow(QString(), colorJitterCheck_);
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
    artifactPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("Detection 已接入 tiny linear detector 占位训练；完整 LibTorch YOLO 训练将在后续接入。")));
    latestCheckpointLabel_ = mutedLabel(QStringLiteral("最新 checkpoint：暂无"));
    latestPreviewPathLabel_ = mutedLabel(QStringLiteral("最新预览：暂无"));
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
    conversionCheckpointEdit_ = new QLineEdit;
    conversionCheckpointEdit_->setPlaceholderText(QStringLiteral("tiny detector checkpoint path"));
    auto* chooseCheckpointButton = new QPushButton(QStringLiteral("选择 checkpoint"));
    connect(chooseCheckpointButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, QStringLiteral("选择 checkpoint"), currentProjectPath_, QStringLiteral("AITrain checkpoint (*.aitrain);;All files (*.*)"));
        if (!file.isEmpty()) {
            conversionCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    inputPanel->bodyLayout()->addWidget(conversionCheckpointEdit_);
    inputPanel->bodyLayout()->addWidget(chooseCheckpointButton);
    inputPanel->bodyLayout()->addStretch();

    auto* exportPanel = new InfoPanel(QStringLiteral("导出配置"));
    exportPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("当前支持 tiny detector JSON 和 tiny detector ONNX；TensorRT 为占位入口，会返回未支持。")));
    conversionFormatCombo_ = new QComboBox;
    conversionFormatCombo_->addItem(QStringLiteral("Tiny detector JSON 占位"), QStringLiteral("tiny_detector_json"));
    conversionFormatCombo_->addItem(QStringLiteral("ONNX 模型核心"), QStringLiteral("onnx"));
    conversionFormatCombo_->addItem(QStringLiteral("TensorRT Engine 占位"), QStringLiteral("tensorrt"));
    conversionOutputEdit_ = new QLineEdit;
    conversionOutputEdit_->setPlaceholderText(QStringLiteral("输出路径；留空则输出到 checkpoint 同目录"));
    auto* exportButton = primaryButton(QStringLiteral("导出模型"));
    connect(exportButton, &QPushButton::clicked, this, &MainWindow::startModelExport);
    exportPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("ONNX 产物包含 tiny detector 模型核心；图片预处理和 NMS 由 AITrain runtime 执行，完整 YOLO 后处理后续接入。")));
    exportPanel->bodyLayout()->addWidget(conversionFormatCombo_);
    exportPanel->bodyLayout()->addWidget(conversionOutputEdit_);
    exportPanel->bodyLayout()->addWidget(exportButton);
    exportPanel->bodyLayout()->addStretch();

    auto* resultPanel = new InfoPanel(QStringLiteral("导出结果"));
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("导出产物、ONNX shape 校验报告和后续 TensorRT engine 会显示在这里。")));
    exportResultLabel_ = mutedLabel(QStringLiteral("暂无导出结果。"));
    resultPanel->bodyLayout()->addWidget(exportResultLabel_);
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
    auto* inferForm = new QFormLayout;
    inferenceCheckpointEdit_ = new QLineEdit;
    inferenceImageEdit_ = new QLineEdit;
    inferenceOutputEdit_ = new QLineEdit;
    inferenceCheckpointEdit_->setPlaceholderText(QStringLiteral("tiny detector checkpoint/export JSON/ONNX/engine path"));
    inferenceImageEdit_->setPlaceholderText(QStringLiteral("image path"));
    inferenceOutputEdit_->setPlaceholderText(QStringLiteral("输出目录；留空则输出到 checkpoint 同目录 inference"));
    auto* chooseModelButton = new QPushButton(QStringLiteral("选择模型文件"));
    auto* chooseImageButton = new QPushButton(QStringLiteral("选择图片"));
    auto* inferButton = primaryButton(QStringLiteral("开始推理"));
    connect(chooseModelButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, QStringLiteral("选择模型文件"), currentProjectPath_, QStringLiteral("AITrain model (*.aitrain *.json *.onnx *.engine *.plan);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    connect(chooseImageButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, QStringLiteral("选择图片"), currentProjectPath_, QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceImageEdit_->setText(QDir::toNativeSeparators(file));
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
    imageLayout->addWidget(inferenceImageEdit_);
    imageLayout->addWidget(chooseImageButton);
    inferForm->addRow(QStringLiteral("Checkpoint"), modelRow);
    inferForm->addRow(QStringLiteral("Image"), imageRow);
    inferForm->addRow(QStringLiteral("Output"), inferenceOutputEdit_);
    toolbar->bodyLayout()->addLayout(inferForm);
    toolbar->bodyLayout()->addWidget(inferButton);

    auto* preview = new QSplitter(Qt::Horizontal);
    auto* sourcePanel = new InfoPanel(QStringLiteral("输入预览"));
    sourcePanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("原图、视频帧或 OCR 输入图像。")));
    auto* resultPanel = new InfoPanel(QStringLiteral("结果预览"));
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("检测框、分割 mask、OCR 文本和耗时会显示在这里。")));
    inferenceResultLabel_ = mutedLabel(QStringLiteral("暂无推理结果。"));
    sourcePanel->bodyLayout()->addWidget(inferenceResultLabel_);
    inferenceOverlayLabel_ = new QLabel(QStringLiteral("暂无 overlay"));
    inferenceOverlayLabel_->setObjectName(QStringLiteral("MutedText"));
    inferenceOverlayLabel_->setAlignment(Qt::AlignCenter);
    inferenceOverlayLabel_->setMinimumHeight(240);
    inferenceOverlayLabel_->setFrameShape(QFrame::StyledPanel);
    resultPanel->bodyLayout()->addWidget(inferenceOverlayLabel_);
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
        if (splitOutputEdit_ && currentProjectPath_.isEmpty()) {
            splitOutputEdit_->setText(QDir::toNativeSeparators(QDir(directory).absoluteFilePath(QStringLiteral("../normalized"))));
        }
        currentDatasetValid_ = false;
    }
}

void MainWindow::validateDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, QStringLiteral("数据集校验"), QStringLiteral("Worker 正在执行任务，稍后再校验数据集。"));
        return;
    }

    const QString format = datasetFormatCombo_->currentText();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_->text());
    if (format.isEmpty() || path.isEmpty()) {
        validationSummaryLabel_->setText(QStringLiteral("请选择数据集目录和格式。"));
        return;
    }

    currentDatasetValid_ = false;
    currentDatasetPath_ = path;
    currentDatasetFormat_ = format;
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
    }
    validationSummaryLabel_->setText(QStringLiteral("正在通过 Worker 校验数据集。"));
    validationOutput_->setPlainText(QStringLiteral("等待校验结果。"));

    QJsonObject options;
    options.insert(QStringLiteral("maxIssues"), 200);
    options.insert(QStringLiteral("maxFiles"), 5000);
    options.insert(QStringLiteral("allowEmptyLabels"), false);
    options.insert(QStringLiteral("maxTextLength"), 25);

    QString error;
    if (!worker_.requestDatasetValidation(workerExecutablePath(), path, format, options, &error)) {
        validationSummaryLabel_->setText(QStringLiteral("无法启动数据集校验：%1").arg(error));
        QMessageBox::critical(this, QStringLiteral("数据集校验"), error);
        return;
    }
    workerPill_->setStatus(QStringLiteral("数据集校验中"), StatusPill::Tone::Info);
}

void MainWindow::splitDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, QStringLiteral("数据集划分"), QStringLiteral("Worker 正在执行任务，稍后再划分数据集。"));
        return;
    }

    const QString format = datasetFormatCombo_->currentText();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_->text());
    if (path.isEmpty() || format.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("数据集划分"), QStringLiteral("请先选择数据集目录和格式。"));
        return;
    }
    if (format != QStringLiteral("yolo_detection") && format != QStringLiteral("yolo_txt")) {
        QMessageBox::warning(this, QStringLiteral("数据集划分"), QStringLiteral("当前划分先支持 YOLO 检测格式。"));
        return;
    }

    bool datasetReady = currentDatasetValid_ && currentDatasetPath_ == path && currentDatasetFormat_ == format;
    if (!datasetReady && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(path, &error);
        datasetReady = dataset.rootPath == path
            && dataset.format == format
            && dataset.validationStatus == QStringLiteral("valid");
    }
    if (!datasetReady) {
        QMessageBox::warning(this, QStringLiteral("数据集划分"), QStringLiteral("请先通过当前格式的数据集校验。"));
        return;
    }

    QString outputPath = QDir::fromNativeSeparators(splitOutputEdit_->text().trimmed());
    if (outputPath.isEmpty()) {
        const QString datasetName = QFileInfo(path).fileName();
        const QString basePath = currentProjectPath_.isEmpty()
            ? QDir(path).absoluteFilePath(QStringLiteral("../normalized"))
            : QDir(currentProjectPath_).filePath(QStringLiteral("datasets/normalized/%1").arg(datasetName));
        outputPath = QDir::cleanPath(basePath);
        splitOutputEdit_->setText(QDir::toNativeSeparators(outputPath));
    }

    QJsonObject options;
    options.insert(QStringLiteral("trainRatio"), splitTrainRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("valRatio"), splitValRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("testRatio"), splitTestRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("seed"), splitSeedEdit_->text().toInt());
    options.insert(QStringLiteral("maxIssues"), 200);
    options.insert(QStringLiteral("allowEmptyLabels"), false);

    QString error;
    if (!worker_.requestDatasetSplit(workerExecutablePath(), path, outputPath, format, options, &error)) {
        QMessageBox::critical(this, QStringLiteral("数据集划分"), error);
        return;
    }
    workerPill_->setStatus(QStringLiteral("数据集划分中"), StatusPill::Tone::Info);
    statusBar()->showMessage(QStringLiteral("正在划分数据集"), 3000);
}

void MainWindow::startModelExport()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, QStringLiteral("模型导出"), QStringLiteral("Worker 正在执行任务，稍后再导出模型。"));
        return;
    }
    const QString checkpointPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
    if (checkpointPath.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("模型导出"), QStringLiteral("请选择 checkpoint。"));
        return;
    }
    const QString format = conversionFormatCombo_
        ? conversionFormatCombo_->currentData().toString()
        : QStringLiteral("tiny_detector_json");
    QString outputPath = QDir::fromNativeSeparators(conversionOutputEdit_ ? conversionOutputEdit_->text().trimmed() : QString());
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(checkpointPath).absoluteDir().filePath(
            format == QStringLiteral("onnx")
                ? QStringLiteral("model.onnx")
                : (format == QStringLiteral("tensorrt") ? QStringLiteral("model.engine") : QStringLiteral("model.aitrain-export.json")));
    }

    QString taskId;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        aitrain::TaskRecord record;
        record.id = taskId;
        record.projectName = currentProjectName_.isEmpty() ? QStringLiteral("manual") : currentProjectName_;
        record.pluginId = QStringLiteral("com.aitrain.plugins.yolo_native");
        record.taskType = QStringLiteral("model_export");
        record.kind = aitrain::TaskKind::Export;
        record.state = aitrain::TaskState::Queued;
        record.workDir = QFileInfo(outputPath).absolutePath();
        record.message = QStringLiteral("等待 Worker 导出模型。");
        record.createdAt = QDateTime::currentDateTimeUtc();
        record.updatedAt = record.createdAt;
        QString taskError;
        if (!repository_.insertTask(record, &taskError)) {
            QMessageBox::critical(this, QStringLiteral("模型导出"), taskError);
            return;
        }
        if (!repository_.updateTaskState(taskId, aitrain::TaskState::Running, QStringLiteral("模型导出中。"), &taskError)) {
            QMessageBox::critical(this, QStringLiteral("模型导出"), taskError);
            return;
        }
        currentTaskId_ = taskId;
        updateRecentTasks();
    }

    QString error;
    if (!worker_.requestModelExport(workerExecutablePath(), checkpointPath, outputPath, format, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, QStringLiteral("模型导出"), error);
        return;
    }
    if (exportResultLabel_) {
        exportResultLabel_->setText(QStringLiteral("正在导出：%1").arg(QDir::toNativeSeparators(outputPath)));
    }
    workerPill_->setStatus(QStringLiteral("模型导出中"), StatusPill::Tone::Info);
}

void MainWindow::startInference()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, QStringLiteral("推理"), QStringLiteral("Worker 正在执行任务，稍后再推理。"));
        return;
    }
    const QString checkpointPath = QDir::fromNativeSeparators(inferenceCheckpointEdit_ ? inferenceCheckpointEdit_->text().trimmed() : QString());
    const QString imagePath = QDir::fromNativeSeparators(inferenceImageEdit_ ? inferenceImageEdit_->text().trimmed() : QString());
    QString outputPath = QDir::fromNativeSeparators(inferenceOutputEdit_ ? inferenceOutputEdit_->text().trimmed() : QString());
    if (checkpointPath.isEmpty() || imagePath.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("推理"), QStringLiteral("请选择模型文件和图片。"));
        return;
    }
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("inference"));
    }

    QString error;
    if (!worker_.requestInference(workerExecutablePath(), checkpointPath, imagePath, outputPath, &error)) {
        QMessageBox::critical(this, QStringLiteral("推理"), error);
        return;
    }
    if (inferenceResultLabel_) {
        inferenceResultLabel_->setText(QStringLiteral("正在推理：%1").arg(QDir::toNativeSeparators(imagePath)));
    }
    workerPill_->setStatus(QStringLiteral("推理中"), StatusPill::Tone::Info);
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
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_->text());
    const QString datasetFormat = datasetFormatCombo_->currentText();
    if (datasetPath.isEmpty() || datasetFormat.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("训练"), QStringLiteral("请先选择并校验数据集。"));
        return;
    }
    auto* selectedPlugin = pluginManager_.pluginById(pluginCombo_->currentData().toString());
    if (!selectedPlugin || !selectedPlugin->datasetAdapter(datasetFormat)) {
        QMessageBox::warning(this, QStringLiteral("训练"), QStringLiteral("当前训练插件不支持所选数据集格式。"));
        return;
    }
    bool datasetReady = currentDatasetValid_ && currentDatasetPath_ == datasetPath && currentDatasetFormat_ == datasetFormat;
    if (!datasetReady && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &error);
        datasetReady = dataset.rootPath == datasetPath
            && dataset.format == datasetFormat
            && dataset.validationStatus == QStringLiteral("valid");
    }
    if (!datasetReady) {
        QMessageBox::warning(this, QStringLiteral("训练"), QStringLiteral("数据集未通过当前格式校验，不能启动训练。"));
        return;
    }

    const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString runDir = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
    QDir().mkpath(runDir);

    QJsonObject parameters;
    parameters.insert(QStringLiteral("epochs"), epochsEdit_->text().toInt());
    parameters.insert(QStringLiteral("batchSize"), batchEdit_->text().toInt());
    parameters.insert(QStringLiteral("imageSize"), imageSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("gridSize"), gridSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("resumeCheckpointPath"), QDir::fromNativeSeparators(resumeCheckpointEdit_->text().trimmed()));
    parameters.insert(QStringLiteral("horizontalFlip"), horizontalFlipCheck_ && horizontalFlipCheck_->isChecked());
    parameters.insert(QStringLiteral("colorJitter"), colorJitterCheck_ && colorJitterCheck_->isChecked());

    aitrain::TrainingRequest request;
    request.taskId = taskId;
    request.projectPath = currentProjectPath_;
    request.pluginId = pluginCombo_->currentData().toString();
    request.taskType = taskTypeCombo_->currentText();
    request.datasetPath = datasetPath;
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
        if (!currentTaskId_.isEmpty()) {
            progressBar_->setValue(payload.value(QStringLiteral("percent")).toInt());
        }
        const QString message = payload.value(QStringLiteral("message")).toString();
        if (!message.isEmpty()) {
            statusBar()->showMessage(message, 3000);
        }
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
        const QString kind = payload.value(QStringLiteral("kind")).toString();
        appendLog(QStringLiteral("产物：%1").arg(path));
        if (kind == QStringLiteral("checkpoint") && latestCheckpointLabel_) {
            latestCheckpointLabel_->setText(QStringLiteral("最新 checkpoint：%1").arg(QDir::toNativeSeparators(path)));
        } else if (kind == QStringLiteral("preview") && latestPreviewPathLabel_) {
            latestPreviewPathLabel_->setText(QStringLiteral("最新预览：%1").arg(QDir::toNativeSeparators(path)));
            if (latestPreviewImageLabel_) {
                QPixmap preview(path);
                if (!preview.isNull()) {
                    latestPreviewImageLabel_->setPixmap(preview.scaled(
                        latestPreviewImageLabel_->size().boundedTo(QSize(320, 220)),
                        Qt::KeepAspectRatio,
                        Qt::SmoothTransformation));
                } else {
                    latestPreviewImageLabel_->setText(QStringLiteral("预览图加载失败"));
                }
            }
        }
        if (kind == QStringLiteral("export") && exportResultLabel_) {
            exportResultLabel_->setText(QStringLiteral("导出完成：%1").arg(QDir::toNativeSeparators(path)));
        } else if (kind == QStringLiteral("inference_overlay") && inferenceOverlayLabel_) {
            QPixmap overlay(path);
            if (!overlay.isNull()) {
                inferenceOverlayLabel_->setPixmap(overlay.scaled(
                    inferenceOverlayLabel_->size().boundedTo(QSize(520, 360)),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation));
            } else {
                inferenceOverlayLabel_->setText(QStringLiteral("推理 overlay 加载失败"));
            }
        } else if (kind == QStringLiteral("inference_predictions") && inferenceResultLabel_) {
            inferenceResultLabel_->setText(inferenceSummaryFromPredictions(path));
        }
        if (repository_.isOpen()) {
            aitrain::ArtifactRecord artifact;
            artifact.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
            artifact.kind = kind;
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
    } else if (type == QStringLiteral("datasetValidation")) {
        updateDatasetValidationResult(payload);
    } else if (type == QStringLiteral("datasetSplit")) {
        updateDatasetSplitResult(payload);
    } else if (type == QStringLiteral("modelExport")) {
        if (exportResultLabel_) {
            const QString exportPath = payload.value(QStringLiteral("exportPath")).toString();
            const QString reportPath = payload.value(QStringLiteral("reportPath")).toString();
            exportResultLabel_->setText(reportPath.isEmpty()
                ? QStringLiteral("导出完成：%1").arg(QDir::toNativeSeparators(exportPath))
                : QStringLiteral("导出完成：%1；报告：%2").arg(QDir::toNativeSeparators(exportPath), QDir::toNativeSeparators(reportPath)));
        }
        if (repository_.isOpen()) {
            const QJsonObject config = payload.value(QStringLiteral("config")).toObject();
            aitrain::ExportRecord exportRecord;
            exportRecord.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
            exportRecord.sourceCheckpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
            exportRecord.format = payload.value(QStringLiteral("format")).toString();
            exportRecord.path = payload.value(QStringLiteral("exportPath")).toString();
            exportRecord.configJson = QString::fromUtf8(QJsonDocument(config).toJson(QJsonDocument::Compact));
            exportRecord.inputShapeJson = QString::fromUtf8(QJsonDocument(config.value(QStringLiteral("input")).toObject()).toJson(QJsonDocument::Compact));
            exportRecord.outputShapeJson = QString::fromUtf8(QJsonDocument(QJsonObject{{QStringLiteral("outputs"), config.value(QStringLiteral("outputs")).toArray()}}).toJson(QJsonDocument::Compact));
            exportRecord.createdAt = QDateTime::currentDateTimeUtc();
            QString error;
            repository_.insertExport(exportRecord, &error);
        }
    } else if (type == QStringLiteral("inferenceResult")) {
        if (inferenceResultLabel_) {
            inferenceResultLabel_->setText(inferenceSummaryFromPredictions(
                payload.value(QStringLiteral("predictionsPath")).toString(),
                payload));
        }
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

void MainWindow::updateDatasetValidationResult(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const int sampleCount = payload.value(QStringLiteral("sampleCount")).toInt();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    const QJsonArray issues = payload.value(QStringLiteral("issues")).toArray();

    currentDatasetPath_ = datasetPath;
    currentDatasetFormat_ = format;
    currentDatasetValid_ = ok;

    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(ok
            ? QStringLiteral("校验通过：%1 个样本。").arg(sampleCount)
            : QStringLiteral("校验失败：发现 %1 个问题，训练已被阻止。").arg(issues.size()));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (datasetPreviewTable_) {
        datasetPreviewTable_->setRowCount(0);
        const QJsonArray previewSamples = payload.value(QStringLiteral("previewSamples")).toArray();
        if (previewSamples.isEmpty()) {
            datasetPreviewTable_->insertRow(0);
            datasetPreviewTable_->setItem(0, 0, new QTableWidgetItem(QStringLiteral("暂无样本预览")));
            datasetPreviewTable_->setItem(0, 1, new QTableWidgetItem(QString()));
        } else {
            for (const QJsonValue& sampleValue : previewSamples) {
                const QString sample = sampleValue.toString();
                const QStringList parts = sample.split(QLatin1Char('\t'));
                const int row = datasetPreviewTable_->rowCount();
                datasetPreviewTable_->insertRow(row);
                datasetPreviewTable_->setItem(row, 0, new QTableWidgetItem(parts.value(0)));
                datasetPreviewTable_->setItem(row, 1, new QTableWidgetItem(parts.size() > 1 ? parts.mid(1).join(QStringLiteral("\t")) : QStringLiteral("标注文件已匹配")));
            }
        }
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        if (issues.isEmpty()) {
            validationIssuesTable_->insertRow(0);
            validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(QStringLiteral("通过")));
            validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("ok")));
            validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(datasetPath));
            validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
            validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(QStringLiteral("未发现数据集问题。")));
        } else {
            for (const QJsonValue& value : issues) {
                const QJsonObject issue = value.toObject();
                const int row = validationIssuesTable_->rowCount();
                validationIssuesTable_->insertRow(row);
                validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(issueSeverityLabel(issue.value(QStringLiteral("severity")).toString())));
                validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(issue.value(QStringLiteral("code")).toString()));
                validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(issue.value(QStringLiteral("filePath")).toString()));
                const int line = issue.value(QStringLiteral("line")).toInt();
                validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(line > 0 ? QString::number(line) : QString()));
                validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(issue.value(QStringLiteral("message")).toString()));
            }
        }
    }

    if (repository_.isOpen()) {
        aitrain::DatasetRecord dataset;
        dataset.name = QFileInfo(datasetPath).fileName();
        dataset.format = format;
        dataset.rootPath = datasetPath;
        dataset.validationStatus = ok ? QStringLiteral("valid") : QStringLiteral("invalid");
        dataset.sampleCount = sampleCount;
        dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
        QString error;
        repository_.upsertDatasetValidation(dataset, &error);
    }

    workerPill_->setStatus(QStringLiteral("Worker 空闲"), StatusPill::Tone::Neutral);
    statusBar()->showMessage(ok ? QStringLiteral("数据集校验通过") : QStringLiteral("数据集校验失败"), 5000);
}

void MainWindow::updateDatasetSplitResult(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const int trainCount = payload.value(QStringLiteral("trainCount")).toInt();
    const int valCount = payload.value(QStringLiteral("valCount")).toInt();
    const int testCount = payload.value(QStringLiteral("testCount")).toInt();
    const QJsonArray errors = payload.value(QStringLiteral("errors")).toArray();

    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(ok
            ? QStringLiteral("划分完成：train %1 / val %2 / test %3。").arg(trainCount).arg(valCount).arg(testCount)
            : QStringLiteral("划分失败：%1 个错误。").arg(errors.size()));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        if (ok) {
            validationIssuesTable_->insertRow(0);
            validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(QStringLiteral("通过")));
            validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("split_ok")));
            validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(outputPath));
            validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
            validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(QStringLiteral("数据集已复制到标准划分目录。")));
        } else {
            for (const QJsonValue& value : errors) {
                const int row = validationIssuesTable_->rowCount();
                validationIssuesTable_->insertRow(row);
                validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(QStringLiteral("错误")));
                validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(QStringLiteral("split_error")));
                validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(outputPath));
                validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(QString()));
                validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(value.toString()));
            }
        }
    }

    if (ok && repository_.isOpen()) {
        aitrain::DatasetRecord dataset;
        dataset.name = QFileInfo(outputPath).fileName();
        dataset.format = QStringLiteral("yolo_detection");
        dataset.rootPath = outputPath;
        dataset.validationStatus = QStringLiteral("valid");
        dataset.sampleCount = trainCount + valCount + testCount;
        dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
        QString error;
        repository_.upsertDatasetValidation(dataset, &error);
        datasetPathEdit_->setText(QDir::toNativeSeparators(outputPath));
        datasetFormatCombo_->setCurrentText(QStringLiteral("yolo_detection"));
        currentDatasetPath_ = outputPath;
        currentDatasetFormat_ = QStringLiteral("yolo_detection");
        currentDatasetValid_ = true;
    }

    workerPill_->setStatus(QStringLiteral("Worker 空闲"), StatusPill::Tone::Neutral);
    statusBar()->showMessage(ok ? QStringLiteral("数据集划分完成") : QStringLiteral("数据集划分失败"), 5000);
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
