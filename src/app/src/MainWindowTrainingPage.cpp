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
#include <QVector>

using namespace aitrain_app;

namespace {
QString yoloArgObjectName(const QString& key)
{
    return QStringLiteral("YoloTrainArg_%1").arg(key);
}

QLineEdit* yoloArgLineEdit(const QString& key, const QString& placeholder = QString(), const QString& value = QString())
{
    auto* edit = new QLineEdit(value);
    edit->setObjectName(yoloArgObjectName(key));
    edit->setPlaceholderText(placeholder);
    edit->setMinimumWidth(0);
    return edit;
}

QComboBox* yoloArgComboBox(const QString& key, const QVector<QPair<QString, QString>>& items)
{
    auto* combo = new QComboBox;
    combo->setObjectName(yoloArgObjectName(key));
    combo->addItem(QStringLiteral("默认"), QString());
    for (const auto& item : items) {
        combo->addItem(item.first, item.second);
    }
    return combo;
}

QComboBox* yoloBoolComboBox(const QString& key)
{
    return yoloArgComboBox(key, {
        {QStringLiteral("true"), QStringLiteral("true")},
        {QStringLiteral("false"), QStringLiteral("false")}
    });
}

QComboBox* yoloEndToEndComboBox(const QString& objectName)
{
    auto* combo = new QComboBox;
    combo->setObjectName(objectName);
    combo->addItem(QStringLiteral("auto"), QStringLiteral("auto"));
    combo->addItem(QStringLiteral("true"), QStringLiteral("true"));
    combo->addItem(QStringLiteral("false"), QStringLiteral("false"));
    return combo;
}

QGroupBox* yoloArgGroup(const QString& title)
{
    auto* group = new QGroupBox(title);
    auto* form = new QFormLayout(group);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(8);
    return group;
}

void addYoloRow(QGroupBox* group, const QString& label, QWidget* field)
{
    if (auto* form = qobject_cast<QFormLayout*>(group->layout())) {
        form->addRow(label, field);
    }
}

QWidget* buildYoloOfficialArgsPanel()
{
    auto* container = new QWidget;
    auto* root = new QVBoxLayout(container);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(10);

    auto* deviceGroup = yoloArgGroup(QStringLiteral("数据与设备"));
    addYoloRow(deviceGroup, QStringLiteral("seed"), yoloArgLineEdit(QStringLiteral("seed"), QStringLiteral("42"), QStringLiteral("42")));
    addYoloRow(deviceGroup, QStringLiteral("device"), yoloArgLineEdit(QStringLiteral("device"), QStringLiteral("cpu / 0 / 0,1")));
    addYoloRow(deviceGroup, QStringLiteral("workers"), yoloArgLineEdit(QStringLiteral("workers"), QStringLiteral("0")));
    addYoloRow(deviceGroup, QStringLiteral("cache"), yoloArgComboBox(QStringLiteral("cache"), {
        {QStringLiteral("false"), QStringLiteral("false")},
        {QStringLiteral("true"), QStringLiteral("true")},
        {QStringLiteral("ram"), QStringLiteral("ram")},
        {QStringLiteral("disk"), QStringLiteral("disk")}
    }));
    addYoloRow(deviceGroup, QStringLiteral("deterministic"), yoloBoolComboBox(QStringLiteral("deterministic")));
    addYoloRow(deviceGroup, QStringLiteral("amp"), yoloBoolComboBox(QStringLiteral("amp")));
    addYoloRow(deviceGroup, QStringLiteral("pretrained"), yoloArgComboBox(QStringLiteral("pretrained"), {
        {QStringLiteral("true"), QStringLiteral("true")},
        {QStringLiteral("false"), QStringLiteral("false")}
    }));
    addYoloRow(deviceGroup, QStringLiteral("resume"), yoloBoolComboBox(QStringLiteral("resume")));
    addYoloRow(deviceGroup, QStringLiteral("save_period"), yoloArgLineEdit(QStringLiteral("save_period"), QStringLiteral("-1 / 10")));
    addYoloRow(deviceGroup, QStringLiteral("fraction"), yoloArgLineEdit(QStringLiteral("fraction"), QStringLiteral("0.0-1.0")));
    addYoloRow(deviceGroup, QStringLiteral("rect"), yoloBoolComboBox(QStringLiteral("rect")));
    addYoloRow(deviceGroup, QStringLiteral("multi_scale"), yoloArgLineEdit(QStringLiteral("multi_scale"), QStringLiteral("0.5")));
    addYoloRow(deviceGroup, QStringLiteral("single_cls"), yoloBoolComboBox(QStringLiteral("single_cls")));
    addYoloRow(deviceGroup, QStringLiteral("classes"), yoloArgLineEdit(QStringLiteral("classes"), QStringLiteral("0,1,2")));
    addYoloRow(deviceGroup, QStringLiteral("freeze"), yoloArgLineEdit(QStringLiteral("freeze"), QStringLiteral("10 或 0,1,2")));

    auto* optimizerGroup = yoloArgGroup(QStringLiteral("优化器与学习率"));
    addYoloRow(optimizerGroup, QStringLiteral("optimizer"), yoloArgComboBox(QStringLiteral("optimizer"), {
        {QStringLiteral("auto"), QStringLiteral("auto")},
        {QStringLiteral("SGD"), QStringLiteral("SGD")},
        {QStringLiteral("Adam"), QStringLiteral("Adam")},
        {QStringLiteral("AdamW"), QStringLiteral("AdamW")},
        {QStringLiteral("RMSProp"), QStringLiteral("RMSProp")}
    }));
    addYoloRow(optimizerGroup, QStringLiteral("lr0"), yoloArgLineEdit(QStringLiteral("lr0"), QStringLiteral("0.01")));
    addYoloRow(optimizerGroup, QStringLiteral("lrf"), yoloArgLineEdit(QStringLiteral("lrf"), QStringLiteral("0.01")));
    addYoloRow(optimizerGroup, QStringLiteral("momentum"), yoloArgLineEdit(QStringLiteral("momentum"), QStringLiteral("0.937")));
    addYoloRow(optimizerGroup, QStringLiteral("weight_decay"), yoloArgLineEdit(QStringLiteral("weight_decay"), QStringLiteral("0.0005")));
    addYoloRow(optimizerGroup, QStringLiteral("warmup_epochs"), yoloArgLineEdit(QStringLiteral("warmup_epochs"), QStringLiteral("3.0")));
    addYoloRow(optimizerGroup, QStringLiteral("cos_lr"), yoloBoolComboBox(QStringLiteral("cos_lr")));
    addYoloRow(optimizerGroup, QStringLiteral("box"), yoloArgLineEdit(QStringLiteral("box"), QStringLiteral("7.5")));
    addYoloRow(optimizerGroup, QStringLiteral("cls"), yoloArgLineEdit(QStringLiteral("cls"), QStringLiteral("0.5")));
    addYoloRow(optimizerGroup, QStringLiteral("dfl"), yoloArgLineEdit(QStringLiteral("dfl"), QStringLiteral("1.5")));
    addYoloRow(optimizerGroup, QStringLiteral("nbs"), yoloArgLineEdit(QStringLiteral("nbs"), QStringLiteral("64")));

    auto* augmentGroup = yoloArgGroup(QStringLiteral("增强"));
    for (const QString& key : {
             QStringLiteral("hsv_h"), QStringLiteral("hsv_s"), QStringLiteral("hsv_v"),
             QStringLiteral("degrees"), QStringLiteral("translate"), QStringLiteral("scale"),
             QStringLiteral("shear"), QStringLiteral("perspective"), QStringLiteral("flipud"),
             QStringLiteral("fliplr"), QStringLiteral("mosaic"), QStringLiteral("mixup"),
             QStringLiteral("cutmix"), QStringLiteral("copy_paste"), QStringLiteral("close_mosaic")}) {
        addYoloRow(augmentGroup, key, yoloArgLineEdit(key));
    }

    auto* segmentationGroup = yoloArgGroup(QStringLiteral("分割专属"));
    addYoloRow(segmentationGroup, QStringLiteral("copy_paste_mode"), yoloArgComboBox(QStringLiteral("copy_paste_mode"), {
        {QStringLiteral("flip"), QStringLiteral("flip")},
        {QStringLiteral("mixup"), QStringLiteral("mixup")}
    }));
    addYoloRow(segmentationGroup, QStringLiteral("overlap_mask"), yoloBoolComboBox(QStringLiteral("overlap_mask")));
    addYoloRow(segmentationGroup, QStringLiteral("mask_ratio"), yoloArgLineEdit(QStringLiteral("mask_ratio"), QStringLiteral("4")));

    auto* validationGroup = yoloArgGroup(QStringLiteral("验证与导出"));
    addYoloRow(validationGroup, QStringLiteral("val"), yoloBoolComboBox(QStringLiteral("val")));
    addYoloRow(validationGroup, QStringLiteral("plots"), yoloBoolComboBox(QStringLiteral("plots")));
    addYoloRow(validationGroup, QStringLiteral("max_det"), yoloArgLineEdit(QStringLiteral("max_det"), QStringLiteral("300")));
    addYoloRow(validationGroup, QStringLiteral("patience"), yoloArgLineEdit(QStringLiteral("patience"), QStringLiteral("100")));
    auto* exportDynamic = new QCheckBox(QStringLiteral("dynamic"));
    exportDynamic->setObjectName(QStringLiteral("YoloTrainExportArg_dynamic"));
    auto* exportHalf = new QCheckBox(QStringLiteral("half"));
    exportHalf->setObjectName(QStringLiteral("YoloTrainExportArg_half"));
    auto* exportInt8 = new QCheckBox(QStringLiteral("int8 TensorRT"));
    exportInt8->setObjectName(QStringLiteral("YoloTrainExportArg_int8"));
    auto* exportEndToEnd = yoloEndToEndComboBox(QStringLiteral("YoloTrainExportArg_end2end"));
    auto* exportFlags = new QWidget;
    auto* exportFlagsLayout = new QHBoxLayout(exportFlags);
    exportFlagsLayout->setContentsMargins(0, 0, 0, 0);
    exportFlagsLayout->setSpacing(8);
    exportFlagsLayout->addWidget(exportDynamic);
    exportFlagsLayout->addWidget(exportHalf);
    exportFlagsLayout->addWidget(exportInt8);
    exportFlagsLayout->addWidget(new QLabel(QStringLiteral("end2end")));
    exportFlagsLayout->addWidget(exportEndToEnd);
    exportFlagsLayout->addStretch();
    addYoloRow(validationGroup, QStringLiteral("export"), exportFlags);

    root->addWidget(deviceGroup);
    root->addWidget(optimizerGroup);
    root->addWidget(augmentGroup);
    root->addWidget(segmentationGroup);
    root->addWidget(validationGroup);
    return container;
}
} // namespace

QLabel* MainWindow::trainingLiveValueLabel(const QString& objectName) const
{
    return findChild<QLabel*>(objectName);
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
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_rec_official")), QStringLiteral("paddleocr_rec_official"));
    modelPresetCombo_ = new QComboBox;
    modelPresetCombo_->setEditable(true);
    modelPresetCombo_->addItems(modelPresetItemsForBackend(trainingBackendCombo_->currentData().toString()));
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
            const QString backend = trainingBackendCombo_->currentData().toString();
            {
                QSignalBlocker block(modelPresetCombo_);
                modelPresetCombo_->clear();
                modelPresetCombo_->addItems(modelPresetItemsForBackend(backend));
                modelPresetCombo_->setCurrentText(defaultModelForBackend(backend));
            }
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
    auto* yoloOfficialArgsGroup = new QGroupBox(QStringLiteral("YOLO 官方高级参数"));
    auto* yoloOfficialArgsLayout = new QVBoxLayout(yoloOfficialArgsGroup);
    yoloOfficialArgsLayout->setContentsMargins(10, 8, 10, 8);
    yoloOfficialArgsLayout->setSpacing(8);
    yoloOfficialArgsLayout->addWidget(buildYoloOfficialArgsPanel());
    setupPanel->bodyLayout()->addWidget(yoloOfficialArgsGroup);
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
    trainingPhaseLabel_ = inlineStatusLabel(QStringLiteral("阶段：等待启动"));
    trainingPhaseLabel_->setObjectName(QStringLiteral("TrainingPhaseStatus"));
    monitorPanel->bodyLayout()->addWidget(trainingPhaseLabel_);

    auto* liveGrid = new QGridLayout;
    liveGrid->setContentsMargins(0, 0, 0, 0);
    liveGrid->setHorizontalSpacing(8);
    liveGrid->setVerticalSpacing(0);
    auto addLiveCard = [liveGrid](int row, int column, const QString& caption, const QString& valueObjectName, QLabel** valueLabel) {
        auto* frame = new QFrame;
        frame->setObjectName(QStringLiteral("TrainingLivePanel_%1").arg(valueObjectName));
        frame->setProperty("trainingLiveRole", QStringLiteral("panel"));
        frame->setMinimumHeight(46);
        frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
        auto* layout = new QVBoxLayout(frame);
        layout->setContentsMargins(8, 5, 8, 5);
        layout->setSpacing(1);
        auto* value = new QLabel(QStringLiteral("--"), frame);
        value->setObjectName(valueObjectName);
        value->setProperty("trainingLiveRole", QStringLiteral("value"));
        value->setMinimumWidth(0);
        value->setMinimumHeight(20);
        value->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        value->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        auto* label = new QLabel(caption, frame);
        label->setObjectName(QStringLiteral("TrainingLiveCaption_%1").arg(valueObjectName));
        label->setProperty("trainingLiveRole", QStringLiteral("caption"));
        label->setMinimumWidth(0);
        label->setMinimumHeight(15);
        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        layout->addWidget(value);
        layout->addWidget(label);
        liveGrid->addWidget(frame, row, column);
        *valueLabel = value;
    };
    for (int column = 0; column < 3; ++column) {
        liveGrid->setColumnStretch(column, 1);
    }
    addLiveCard(0, 0, QStringLiteral("Epoch"), QStringLiteral("TrainingEpochValue"), &trainingEpochValueLabel_);
    addLiveCard(0, 1, QStringLiteral("Batch"), QStringLiteral("TrainingBatchValue"), &trainingBatchValueLabel_);
    addLiveCard(0, 2, QStringLiteral("ETA"), QStringLiteral("TrainingEtaValue"), &trainingEtaValueLabel_);
    addLiveCard(1, 0, QStringLiteral("Device"), QStringLiteral("TrainingDeviceValue"), &trainingDeviceValueLabel_);
    addLiveCard(1, 1, QStringLiteral("Loss"), QStringLiteral("TrainingLossValue"), &trainingLossValueLabel_);
    addLiveCard(1, 2, QStringLiteral("mAP"), QStringLiteral("TrainingMapValue"), &trainingMapValueLabel_);
    monitorPanel->bodyLayout()->addLayout(liveGrid);

    progressBar_ = new QProgressBar;
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    monitorPanel->bodyLayout()->addWidget(progressBar_);
    monitorPanel->bodyLayout()->addStretch();

    auto* artifactPanel = new InfoPanel(QStringLiteral("任务与产物"));
    artifactPanel->setMinimumWidth(0);
    auto* artifactGuideLabel = mutedLabel(QStringLiteral("运行后会记录 checkpoint、训练报告、ONNX、预览图和请求参数。完整产物浏览请进入“任务与产物”。"));
    auto* artifactBoundaryLabel = mutedLabel(QStringLiteral("主流程优先使用官方 YOLO / PaddleOCR 后端；PaddleOCR System 产物来自官方工具链，不代表 C++ DB 后处理已经接入。"));
    allowLabelToShrink(artifactGuideLabel);
    allowLabelToShrink(artifactBoundaryLabel);
    artifactPanel->bodyLayout()->addWidget(artifactGuideLabel);
    artifactPanel->bodyLayout()->addWidget(artifactBoundaryLabel);
    latestCheckpointLabel_ = mutedLabel(QStringLiteral("最新 checkpoint：暂无"));
    latestOnnxLabel_ = mutedLabel(QStringLiteral("最新 ONNX：暂无"));
    latestReportLabel_ = mutedLabel(QStringLiteral("训练报告：暂无"));
    latestPreviewPathLabel_ = mutedLabel(QStringLiteral("最新预览：暂无"));
    allowLabelToShrink(latestCheckpointLabel_);
    allowLabelToShrink(latestOnnxLabel_);
    allowLabelToShrink(latestReportLabel_);
    allowLabelToShrink(latestPreviewPathLabel_);
    latestPreviewImageLabel_ = new QLabel(QStringLiteral("暂无预览图"));
    latestPreviewImageLabel_->setObjectName(QStringLiteral("MutedText"));
    latestPreviewImageLabel_->setAlignment(Qt::AlignCenter);
    latestPreviewImageLabel_->setMinimumHeight(120);
    latestPreviewImageLabel_->setFrameShape(QFrame::StyledPanel);
    latestPreviewImageLabel_->setScaledContents(false);
    artifactPanel->bodyLayout()->addWidget(latestCheckpointLabel_);
    artifactPanel->bodyLayout()->addWidget(latestOnnxLabel_);
    artifactPanel->bodyLayout()->addWidget(latestReportLabel_);
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

    auto* metricsPanel = new InfoPanel(QStringLiteral("指标曲线"));
    metricsPanel->setMinimumWidth(0);
    metricsWidget_ = new MetricsWidget;
    metricsPanel->bodyLayout()->addWidget(metricsWidget_, 1);

    auto* detailTabs = new QTabWidget;
    detailTabs->setObjectName(QStringLiteral("TrainingDetailTabs"));
    detailTabs->setDocumentMode(true);
    detailTabs->addTab(metricsPanel, QStringLiteral("指标曲线"));
    detailTabs->addTab(logPanel, QStringLiteral("训练日志"));
    detailTabs->addTab(artifactPanel, QStringLiteral("任务与产物"));

    auto* rightSplitter = new QSplitter(Qt::Vertical);
    rightSplitter->setMinimumWidth(0);
    rightSplitter->addWidget(monitorPanel);
    rightSplitter->addWidget(detailTabs);
    rightSplitter->setStretchFactor(0, 2);
    rightSplitter->setStretchFactor(1, 1);
    rightSplitter->setSizes(QList<int>() << 430 << 230);

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
