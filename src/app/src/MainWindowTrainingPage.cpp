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
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_rec_official")), QStringLiteral("paddleocr_rec_official"));
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
        << QStringLiteral("PP-OCRv4_mobile_rec"));
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
    trainingBackendHintLabel_ = mutedLabel(QStringLiteral("生产训练仅使用官方后端：Ultralytics YOLO 或 PaddleOCR official adapter。"));
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
