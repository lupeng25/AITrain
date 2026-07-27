#include "MainWindow.h"
#include "TrainingPage.h"
#include "TrainingPageController.h"
#include "TaskRuntimeController.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"

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
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextDocument>
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

QString smpArgObjectName(const QString& key)
{
    return QStringLiteral("SmpTrainArg_%1").arg(key);
}

QString anomalyArgObjectName(const QString& key)
{
    return QStringLiteral("AnomalyTrainArg_%1").arg(key);
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
    combo->addItem(uiText("默认"), QString());
    for (const auto& item : items) {
        combo->addItem(item.first, item.second);
    }
    return combo;
}

QLineEdit* smpArgLineEdit(const QString& key, const QString& placeholder = QString(), const QString& value = QString())
{
    auto* edit = new QLineEdit(value);
    edit->setObjectName(smpArgObjectName(key));
    edit->setPlaceholderText(placeholder);
    edit->setMinimumWidth(0);
    return edit;
}

QLineEdit* anomalyArgLineEdit(const QString& key, const QString& placeholder = QString(), const QString& value = QString())
{
    auto* edit = new QLineEdit(value);
    edit->setObjectName(anomalyArgObjectName(key));
    edit->setPlaceholderText(placeholder);
    edit->setMinimumWidth(0);
    return edit;
}

QComboBox* smpArgComboBox(const QString& key, const QVector<QPair<QString, QString>>& items, const QString& defaultValue)
{
    auto* combo = new QComboBox;
    combo->setObjectName(smpArgObjectName(key));
    for (const auto& item : items) {
        combo->addItem(item.first, item.second);
    }
    const int index = combo->findData(defaultValue);
    if (index >= 0) {
        combo->setCurrentIndex(index);
    }
    return combo;
}

QComboBox* anomalyArgComboBox(const QString& key, const QVector<QPair<QString, QString>>& items, const QString& defaultValue)
{
    auto* combo = new QComboBox;
    combo->setObjectName(anomalyArgObjectName(key));
    for (const auto& item : items) {
        combo->addItem(item.first, item.second);
    }
    const int index = combo->findData(defaultValue);
    if (index >= 0) {
        combo->setCurrentIndex(index);
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

void addSmpRow(QGroupBox* group, const QString& label, QWidget* field)
{
    if (auto* form = qobject_cast<QFormLayout*>(group->layout())) {
        form->addRow(label, field);
    }
}

void addAnomalyRow(QGroupBox* group, const QString& label, QWidget* field)
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

    auto* deviceGroup = yoloArgGroup(uiText("数据与设备"));
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
    addYoloRow(deviceGroup, QStringLiteral("save_period"), yoloArgLineEdit(QStringLiteral("save_period"), QStringLiteral("-1 / 10")));
    addYoloRow(deviceGroup, QStringLiteral("fraction"), yoloArgLineEdit(QStringLiteral("fraction"), QStringLiteral("0.0-1.0")));
    addYoloRow(deviceGroup, QStringLiteral("rect"), yoloBoolComboBox(QStringLiteral("rect")));
    addYoloRow(deviceGroup, QStringLiteral("multi_scale"), yoloArgLineEdit(QStringLiteral("multi_scale"), QStringLiteral("0.5")));
    addYoloRow(deviceGroup, QStringLiteral("single_cls"), yoloBoolComboBox(QStringLiteral("single_cls")));
    addYoloRow(deviceGroup, QStringLiteral("classes"), yoloArgLineEdit(QStringLiteral("classes"), QStringLiteral("0,1,2")));
    addYoloRow(deviceGroup, QStringLiteral("freeze"), yoloArgLineEdit(QStringLiteral("freeze"), uiText("10 或 0,1,2")));

    auto* optimizerGroup = yoloArgGroup(uiText("优化器与学习率"));
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

    auto* augmentGroup = yoloArgGroup(uiText("增强"));
    for (const QString& key : {
             QStringLiteral("hsv_h"), QStringLiteral("hsv_s"), QStringLiteral("hsv_v"),
             QStringLiteral("degrees"), QStringLiteral("translate"), QStringLiteral("scale"),
             QStringLiteral("shear"), QStringLiteral("perspective"), QStringLiteral("flipud"),
             QStringLiteral("fliplr"), QStringLiteral("mosaic"), QStringLiteral("mixup"),
             QStringLiteral("cutmix"), QStringLiteral("copy_paste"), QStringLiteral("close_mosaic")}) {
        addYoloRow(augmentGroup, key, yoloArgLineEdit(key));
    }

    auto* segmentationGroup = yoloArgGroup(uiText("分割专属"));
    addYoloRow(segmentationGroup, QStringLiteral("copy_paste_mode"), yoloArgComboBox(QStringLiteral("copy_paste_mode"), {
        {QStringLiteral("flip"), QStringLiteral("flip")},
        {QStringLiteral("mixup"), QStringLiteral("mixup")}
    }));
    addYoloRow(segmentationGroup, QStringLiteral("overlap_mask"), yoloBoolComboBox(QStringLiteral("overlap_mask")));
    addYoloRow(segmentationGroup, QStringLiteral("mask_ratio"), yoloArgLineEdit(QStringLiteral("mask_ratio"), QStringLiteral("4")));

    auto* validationGroup = yoloArgGroup(uiText("验证与导出"));
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

QWidget* buildSmpArgsPanel()
{
    auto* container = new QWidget;
    auto* root = new QVBoxLayout(container);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(10);

    auto* trainGroup = yoloArgGroup(uiText("训练参数"));
    addSmpRow(trainGroup, QStringLiteral("seed"), smpArgLineEdit(QStringLiteral("seed"), QStringLiteral("42"), QStringLiteral("42")));
    addSmpRow(trainGroup, QStringLiteral("device"), smpArgLineEdit(QStringLiteral("device"), QStringLiteral("cpu / cuda / 0"), QStringLiteral("cpu")));
    addSmpRow(trainGroup, QStringLiteral("workers"), smpArgLineEdit(QStringLiteral("workers"), QStringLiteral("0"), QStringLiteral("0")));
    addSmpRow(trainGroup, QStringLiteral("learningRate"), smpArgLineEdit(QStringLiteral("learningRate"), QStringLiteral("0.0003"), QStringLiteral("0.0003")));
    addSmpRow(trainGroup, QStringLiteral("optimizer"), smpArgComboBox(QStringLiteral("optimizer"), {
        {QStringLiteral("adamw"), QStringLiteral("adamw")}
    }, QStringLiteral("adamw")));
    addSmpRow(trainGroup, QStringLiteral("loss"), smpArgComboBox(QStringLiteral("loss"), {
        {QStringLiteral("dice_ce"), QStringLiteral("dice_ce")}
    }, QStringLiteral("dice_ce")));

    auto* dataGroup = yoloArgGroup(uiText("Mask 与 encoder"));
    addSmpRow(dataGroup, QStringLiteral("encoderWeights"), smpArgComboBox(QStringLiteral("encoderWeights"), {
        {QStringLiteral("none"), QStringLiteral("none")},
        {QStringLiteral("imagenet"), QStringLiteral("imagenet")}
    }, QStringLiteral("none")));
    addSmpRow(dataGroup, QStringLiteral("ignoreIndex"), smpArgLineEdit(QStringLiteral("ignoreIndex"), QStringLiteral("255"), QStringLiteral("255")));

    root->addWidget(trainGroup);
    root->addWidget(dataGroup);
    return container;
}

QWidget* buildAnomalyArgsPanel()
{
    auto* container = new QWidget;
    auto* root = new QVBoxLayout(container);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(10);

    auto* runtimeGroup = yoloArgGroup(uiText("运行与阈值"));
    addAnomalyRow(runtimeGroup, QStringLiteral("seed"), anomalyArgLineEdit(QStringLiteral("seed"), QStringLiteral("42"), QStringLiteral("42")));
    addAnomalyRow(runtimeGroup, QStringLiteral("device"), anomalyArgLineEdit(QStringLiteral("device"), QStringLiteral("cpu / cuda / 0"), QStringLiteral("cpu")));
    addAnomalyRow(runtimeGroup, QStringLiteral("workers"), anomalyArgLineEdit(QStringLiteral("workers"), QStringLiteral("0"), QStringLiteral("0")));
    addAnomalyRow(runtimeGroup, QStringLiteral("thresholdStrategy"), anomalyArgComboBox(QStringLiteral("thresholdStrategy"), {
        {QStringLiteral("quantile"), QStringLiteral("quantile")},
        {QStringLiteral("adaptive"), QStringLiteral("adaptive")},
        {QStringLiteral("manual"), QStringLiteral("manual")}
    }, QStringLiteral("quantile")));
    addAnomalyRow(runtimeGroup, QStringLiteral("quantile"), anomalyArgLineEdit(QStringLiteral("quantile"), QStringLiteral("0.995"), QStringLiteral("0.995")));

    auto* patchCoreGroup = yoloArgGroup(QStringLiteral("PatchCore"));
    addAnomalyRow(patchCoreGroup, QStringLiteral("backbone"), anomalyArgLineEdit(QStringLiteral("backbone"), QStringLiteral("wide_resnet50_2"), QStringLiteral("wide_resnet50_2")));
    addAnomalyRow(patchCoreGroup, QStringLiteral("layers"), anomalyArgLineEdit(QStringLiteral("layers"), QStringLiteral("layer2,layer3"), QStringLiteral("layer2,layer3")));
    addAnomalyRow(patchCoreGroup, QStringLiteral("coresetSamplingRatio"), anomalyArgLineEdit(QStringLiteral("coresetSamplingRatio"), QStringLiteral("0.1"), QStringLiteral("0.1")));
    addAnomalyRow(patchCoreGroup, QStringLiteral("numNeighbors"), anomalyArgLineEdit(QStringLiteral("numNeighbors"), QStringLiteral("9"), QStringLiteral("9")));

    auto* efficientAdGroup = yoloArgGroup(QStringLiteral("EfficientAD"));
    addAnomalyRow(efficientAdGroup, QStringLiteral("modelSize"), anomalyArgComboBox(QStringLiteral("modelSize"), {
        {QStringLiteral("small"), QStringLiteral("small")},
        {QStringLiteral("medium"), QStringLiteral("medium")}
    }, QStringLiteral("small")));
    addAnomalyRow(efficientAdGroup, QStringLiteral("lr"), anomalyArgLineEdit(QStringLiteral("lr"), QStringLiteral("0.0001"), QStringLiteral("0.0001")));
    addAnomalyRow(efficientAdGroup, QStringLiteral("weightDecay"), anomalyArgLineEdit(QStringLiteral("weightDecay"), QStringLiteral("0.00001"), QStringLiteral("0.00001")));
    addAnomalyRow(efficientAdGroup, QStringLiteral("imagenetDir"), anomalyArgLineEdit(QStringLiteral("imagenetDir"), QStringLiteral(".deps/anomalib/imagenette")));

    root->addWidget(runtimeGroup);
    root->addWidget(patchCoreGroup);
    root->addWidget(efficientAdGroup);
    return container;
}
} // namespace

QWidget* MainWindow::buildTrainingPage()
{
    trainingPage_ = new TrainingWorkspacePage;
    auto* page = trainingPage_;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(12);

    auto* flowRail = new QFrame;
    flowRail->setObjectName(QStringLiteral("FlowRail"));
    auto* flowLayout = new QHBoxLayout(flowRail);
    flowLayout->setContentsMargins(10, 0, 10, 0);
    flowLayout->setSpacing(0);
    const QVector<QStringList> flowSteps = {
        {QStringLiteral("1"), uiText("数据准备"), uiText("数据集已通过")},
        {QStringLiteral("2"), uiText("训练实验"), uiText("配置并监控运行")},
        {QStringLiteral("3"), uiText("评估与模型"), uiText("等待当前训练")},
        {QStringLiteral("4"), uiText("部署验证"), uiText("ONNX · 待运行")}
    };
    for (int index = 0; index < flowSteps.size(); ++index) {
        auto* step = new QFrame;
        step->setObjectName(index == 1 ? QStringLiteral("FlowStepActive")
                                      : (index < 1 ? QStringLiteral("FlowStepDone") : QStringLiteral("FlowStep")));
        auto* stepLayout = new QHBoxLayout(step);
        stepLayout->setContentsMargins(12, 9, 12, 9);
        stepLayout->setSpacing(8);
        auto* number = new QLabel(flowSteps[index][0]);
        number->setObjectName(QStringLiteral("FlowStepNumber"));
        number->setAlignment(Qt::AlignCenter);
        number->setFixedSize(24, 24);
        auto* textBlock = new QWidget;
        auto* textLayout = new QVBoxLayout(textBlock);
        textLayout->setContentsMargins(0, 0, 0, 0);
        textLayout->setSpacing(0);
        auto* name = new QLabel(flowSteps[index][1]);
        name->setObjectName(QStringLiteral("FlowStepTitle"));
        auto* detail = new QLabel(flowSteps[index][2]);
        detail->setObjectName(QStringLiteral("FlowStepDetail"));
        textLayout->addWidget(name);
        textLayout->addWidget(detail);
        stepLayout->addWidget(number);
        stepLayout->addWidget(textBlock, 1);
        flowLayout->addWidget(step, 1);
    }
    layout->addWidget(flowRail);

    auto* capabilityCombo = new QComboBox;
    capabilityCombo->setObjectName(QStringLiteral("TrainingCapability"));
    auto* taskTypeCombo = new QComboBox;
    taskTypeCombo->setObjectName(QStringLiteral("TrainingTaskType"));
    auto* trainingBackendCombo = new QComboBox;
    trainingBackendCombo->setObjectName(QStringLiteral("TrainingBackend"));
    for (const QString& backend : {
             QStringLiteral("ultralytics_yolo_detect"),
             QStringLiteral("ultralytics_yolo_segment"),
             QStringLiteral("ultralytics_yolo_obb"),
             QStringLiteral("smp_semantic_segmentation"),
             QStringLiteral("anomalib_patchcore"),
             QStringLiteral("anomalib_efficientad"),
             QStringLiteral("paddleocr_det_official"),
             QStringLiteral("paddleocr_rec_official")}) {
        trainingBackendCombo->addItem(backendLabel(backend), backend);
    }
    auto* modelPresetCombo = new QComboBox;
    modelPresetCombo->setObjectName(QStringLiteral("TrainingModelPreset"));
    modelPresetCombo->setEditable(true);
    auto* epochsEdit = new QLineEdit(QStringLiteral("20"));
    epochsEdit->setObjectName(QStringLiteral("TrainingEpochs"));
    auto* batchEdit = new QLineEdit(QStringLiteral("8"));
    batchEdit->setObjectName(QStringLiteral("TrainingBatchSize"));
    auto* imageSizeEdit = new QLineEdit(QStringLiteral("640"));
    imageSizeEdit->setObjectName(QStringLiteral("TrainingImageSize"));
    auto* gridSizeEdit = new QLineEdit(QStringLiteral("4"));
    gridSizeEdit->setObjectName(QStringLiteral("TrainingGridSize"));
    auto* horizontalFlipCheck = new QCheckBox(QStringLiteral("水平翻转增强"));
    horizontalFlipCheck->setObjectName(QStringLiteral("TrainingHorizontalFlip"));
    auto* colorJitterCheck = new QCheckBox(QStringLiteral("亮度扰动增强"));
    colorJitterCheck->setObjectName(QStringLiteral("TrainingColorJitter"));
    auto* trainingDatasetSummaryLabel = inlineStatusLabel(
        QStringLiteral("当前数据集：未选择。请先在数据集页导入并通过校验。"));
    trainingDatasetSummaryLabel->setMinimumHeight(34);
    trainingDatasetSummaryLabel->setWordWrap(true);
    allowLabelToShrink(trainingDatasetSummaryLabel);
    auto* trainingBackendHintLabel = mutedLabel(
        QStringLiteral("生产训练仅使用产品合同声明的官方后端。"));
    trainingBackendHintLabel->setObjectName(QStringLiteral("TrainingBackendHint"));
    trainingBackendHintLabel->setWordWrap(true);
    allowLabelToShrink(trainingBackendHintLabel);
    auto* trainingRunSummaryLabel =
        inlineStatusLabel(QStringLiteral("等待配置训练实验。"));
    trainingRunSummaryLabel->setMinimumHeight(42);
    allowLabelToShrink(trainingRunSummaryLabel);

    auto* startButton = primaryButton(QStringLiteral("启动训练"));
    auto* cancelButton = dangerButton(QStringLiteral("取消任务"));
    connect(startButton, &QPushButton::clicked,
        page, &TrainingWorkspacePage::startRequested);
    connect(cancelButton, &QPushButton::clicked,
        page, &TrainingWorkspacePage::cancelRequested);

    trainingDatasetSummaryLabel->setObjectName(QStringLiteral("TrainingDatasetNote"));
    trainingRunSummaryLabel->setObjectName(QStringLiteral("TrainingRunNote"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("TrainingRunHeader"));
    auto* headerRoot = new QHBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 10, 14, 10);
    headerRoot->setSpacing(12);
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QGridLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setHorizontalSpacing(8);
    titleLayout->setVerticalSpacing(2);
    auto* runStatus = new QLabel(uiText("待启动"));
    runStatus->setObjectName(QStringLiteral("RunStatus"));
    auto* title = new QLabel(QStringLiteral("yolo11n-bearing-v3"));
    title->setObjectName(QStringLiteral("TrainingRunTitle"));
    auto* subtitle = new QLabel(QStringLiteral("Ultralytics YOLO Detection · yolo11n.pt · bearing-v3"));
    subtitle->setObjectName(QStringLiteral("TrainingRunMeta"));
    titleLayout->addWidget(runStatus, 0, 0);
    titleLayout->addWidget(title, 0, 1);
    titleLayout->addWidget(subtitle, 1, 1);
    titleLayout->setColumnStretch(1, 1);

    auto* actionLayout = new QHBoxLayout;
    actionLayout->setContentsMargins(0, 0, 0, 0);
    actionLayout->setSpacing(10);
    auto* editConfigButton = new QPushButton(uiText("编辑配置"));
    editConfigButton->setObjectName(QStringLiteral("SecondaryButton"));
    actionLayout->addWidget(editConfigButton);
    actionLayout->addWidget(startButton);
    actionLayout->addWidget(cancelButton);
    headerRoot->addWidget(titleBlock, 1);
    headerRoot->addLayout(actionLayout);

    auto* setupPanel = new InfoPanel(uiText("训练配置"));
    setupPanel->setMinimumWidth(0);
    setupPanel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    setupPanel->bodyLayout()->addWidget(mutedLabel(uiText("实验参数快照")));
    setupPanel->bodyLayout()->addWidget(trainingDatasetSummaryLabel);
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setRowWrapPolicy(QFormLayout::WrapLongRows);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    form->addRow(QStringLiteral("任务类型"), taskTypeCombo);
    form->addRow(QStringLiteral("训练后端"), trainingBackendCombo);
    form->addRow(QStringLiteral("模型预设"), modelPresetCombo);
    form->addRow(QStringLiteral("Epochs"), epochsEdit);
    form->addRow(QStringLiteral("Batch Size"), batchEdit);
    form->addRow(QStringLiteral("Image Size"), imageSizeEdit);
    setupPanel->bodyLayout()->addLayout(form);
    auto* yoloOfficialArgsGroup = new QGroupBox(uiText("YOLO 官方高级参数"));
    yoloOfficialArgsGroup->setObjectName(QStringLiteral("YoloOfficialArgsGroup"));
    auto* yoloOfficialArgsLayout = new QVBoxLayout(yoloOfficialArgsGroup);
    yoloOfficialArgsLayout->setContentsMargins(10, 8, 10, 8);
    yoloOfficialArgsLayout->setSpacing(8);
    yoloOfficialArgsLayout->addWidget(buildYoloOfficialArgsPanel());
    setupPanel->bodyLayout()->addWidget(yoloOfficialArgsGroup);
    auto* smpArgsGroup = new QGroupBox(uiText("SMP 语义分割参数"));
    smpArgsGroup->setObjectName(QStringLiteral("SmpSemanticArgsGroup"));
    auto* smpArgsLayout = new QVBoxLayout(smpArgsGroup);
    smpArgsLayout->setContentsMargins(10, 8, 10, 8);
    smpArgsLayout->setSpacing(8);
    smpArgsLayout->addWidget(buildSmpArgsPanel());
    setupPanel->bodyLayout()->addWidget(smpArgsGroup);
    auto* anomalyArgsGroup = new QGroupBox(uiText("Anomalib 异常检测参数"));
    anomalyArgsGroup->setObjectName(QStringLiteral("AnomalyDetectionArgsGroup"));
    auto* anomalyArgsLayout = new QVBoxLayout(anomalyArgsGroup);
    anomalyArgsLayout->setContentsMargins(10, 8, 10, 8);
    anomalyArgsLayout->setSpacing(8);
    anomalyArgsLayout->addWidget(buildAnomalyArgsPanel());
    setupPanel->bodyLayout()->addWidget(anomalyArgsGroup);
    const QString normalizedBackend =
        trainingBackendCombo->currentData().toString().trimmed().toLower();
    yoloOfficialArgsGroup->setVisible(normalizedBackend.startsWith(QStringLiteral("ultralytics_yolo")));
    smpArgsGroup->setVisible(normalizedBackend == QStringLiteral("smp_semantic_segmentation"));
    anomalyArgsGroup->setVisible(normalizedBackend == QStringLiteral("anomalib_patchcore")
        || normalizedBackend == QStringLiteral("anomalib_efficientad"));
    setupPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("当前模型能力说明")));
    setupPanel->bodyLayout()->addWidget(trainingBackendHintLabel);

    auto* advancedGroup = new QGroupBox(QStringLiteral("高级 / 诊断后端"));
    auto* advancedForm = new QFormLayout(advancedGroup);
    advancedForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    advancedForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    advancedForm->setHorizontalSpacing(14);
    advancedForm->setVerticalSpacing(10);
    advancedForm->addRow(QStringLiteral("内置能力"), capabilityCombo);
    advancedForm->addRow(QStringLiteral("Grid Size"), gridSizeEdit);
    auto* augmentRow = new QWidget;
    auto* augmentLayout = new QHBoxLayout(augmentRow);
    augmentLayout->setContentsMargins(0, 0, 0, 0);
    augmentLayout->setSpacing(14);
    augmentLayout->addWidget(horizontalFlipCheck);
    augmentLayout->addWidget(colorJitterCheck);
    augmentLayout->addStretch();
    advancedForm->addRow(QStringLiteral("Augment"), augmentRow);
    auto* advancedToggleButton = new QPushButton(uiText("展开高级参数"));
    advancedToggleButton->setCheckable(true);
    advancedToggleButton->setObjectName(QStringLiteral("AdvancedToggle"));
    setupPanel->bodyLayout()->addWidget(advancedToggleButton);
    setupPanel->bodyLayout()->addWidget(advancedGroup);
    yoloOfficialArgsGroup->setVisible(false);
    yoloOfficialArgsGroup->setProperty("advancedExpanded", false);
    smpArgsGroup->setVisible(false);
    smpArgsGroup->setProperty("advancedExpanded", false);
    anomalyArgsGroup->setVisible(false);
    anomalyArgsGroup->setProperty("advancedExpanded", false);
    advancedGroup->setVisible(false);
    connect(advancedToggleButton, &QPushButton::toggled, this,
        [advancedToggleButton, advancedGroup, yoloOfficialArgsGroup,
            smpArgsGroup, anomalyArgsGroup, trainingBackendCombo](bool expanded) {
            advancedToggleButton->setText(expanded ? uiText("收起高级参数") : uiText("展开高级参数"));
            advancedGroup->setVisible(expanded);
            yoloOfficialArgsGroup->setProperty("advancedExpanded", expanded);
            smpArgsGroup->setProperty("advancedExpanded", expanded);
            anomalyArgsGroup->setProperty("advancedExpanded", expanded);
            const QString backend =
                trainingBackendCombo->currentData().toString().trimmed().toLower();
            yoloOfficialArgsGroup->setVisible(expanded && backend.startsWith(QStringLiteral("ultralytics_yolo")));
            smpArgsGroup->setVisible(expanded && backend == QStringLiteral("smp_semantic_segmentation"));
            anomalyArgsGroup->setVisible(expanded && (backend == QStringLiteral("anomalib_patchcore")
                || backend == QStringLiteral("anomalib_efficientad")));
        });
    setupPanel->bodyLayout()->addStretch();

    auto* setupScroll = new QScrollArea;
    setupScroll->setWidget(setupPanel);
    setupScroll->setWidgetResizable(true);
    setupScroll->setFrameShape(QFrame::NoFrame);
    setupScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setupScroll->setMinimumWidth(240);
    setupScroll->setMaximumWidth(280);
    connect(editConfigButton, &QPushButton::clicked, this,
        [setupScroll, taskTypeCombo]() {
        setupScroll->ensureWidgetVisible(taskTypeCombo);
        taskTypeCombo->setFocus();
    });

    auto* monitorPanel = new InfoPanel(uiText("训练监控"));
    monitorPanel->setMinimumWidth(0);
    auto* trainingPhaseLabel = inlineStatusLabel(uiText("阶段：等待启动"));
    trainingPhaseLabel->setObjectName(QStringLiteral("TrainingPhaseStatus"));
    monitorPanel->bodyLayout()->addWidget(trainingPhaseLabel);

    auto* liveGrid = new QGridLayout;
    liveGrid->setContentsMargins(0, 0, 0, 0);
    liveGrid->setHorizontalSpacing(8);
    liveGrid->setVerticalSpacing(0);
    auto addLiveCard = [liveGrid](int row, int column,
                           const QString& caption,
                           const QString& valueObjectName) {
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
    };
    for (int column = 0; column < 6; ++column) {
        liveGrid->setColumnStretch(column, 1);
    }
    addLiveCard(0, 0, QStringLiteral("Epoch"), QStringLiteral("TrainingEpochValue"));
    addLiveCard(0, 1, QStringLiteral("Batch"), QStringLiteral("TrainingBatchValue"));
    addLiveCard(0, 2, QStringLiteral("ETA"), QStringLiteral("TrainingEtaValue"));
    addLiveCard(0, 3, QStringLiteral("Device"), QStringLiteral("TrainingDeviceValue"));
    addLiveCard(0, 4, QStringLiteral("Loss"), QStringLiteral("TrainingLossValue"));
    addLiveCard(0, 5, QStringLiteral("mAP"), QStringLiteral("TrainingMapValue"));
    monitorPanel->bodyLayout()->addLayout(liveGrid);

    auto* progressBar = new QProgressBar;
    progressBar->setObjectName(QStringLiteral("TrainingProgress"));
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    monitorPanel->bodyLayout()->addWidget(progressBar);

    auto* artifactPanel = new InfoPanel(QStringLiteral("任务与产物"));
    artifactPanel->setMinimumWidth(0);
    auto* artifactGuideLabel = mutedLabel(QStringLiteral("运行后会记录 checkpoint、训练报告、ONNX、预览图和请求参数。完整产物浏览请进入“任务与产物”。"));
    auto* artifactBoundaryLabel = mutedLabel(QStringLiteral("主流程优先使用官方 YOLO / PaddleOCR 后端；PaddleOCR System 产物来自官方工具链，不代表 C++ DB 后处理已经接入。"));
    allowLabelToShrink(artifactGuideLabel);
    allowLabelToShrink(artifactBoundaryLabel);
    artifactPanel->bodyLayout()->addWidget(artifactGuideLabel);
    artifactPanel->bodyLayout()->addWidget(artifactBoundaryLabel);
    auto* latestCheckpointLabel =
        mutedLabel(QStringLiteral("最新 checkpoint：暂无"));
    latestCheckpointLabel->setObjectName(
        QStringLiteral("TrainingLatestCheckpoint"));
    auto* latestOnnxLabel = mutedLabel(uiText("最新 ONNX：暂无"));
    latestOnnxLabel->setObjectName(QStringLiteral("TrainingLatestOnnx"));
    auto* latestReportLabel = mutedLabel(uiText("训练报告：暂无"));
    latestReportLabel->setObjectName(QStringLiteral("TrainingLatestReport"));
    auto* latestPreviewLabel = mutedLabel(QStringLiteral("最新预览：暂无"));
    latestPreviewLabel->setObjectName(QStringLiteral("TrainingLatestPreview"));
    allowLabelToShrink(latestCheckpointLabel);
    allowLabelToShrink(latestOnnxLabel);
    allowLabelToShrink(latestReportLabel);
    allowLabelToShrink(latestPreviewLabel);
    auto* latestPreviewImageLabel = new QLabel(QStringLiteral("暂无预览图"));
    latestPreviewImageLabel->setObjectName(QStringLiteral("MutedText"));
    latestPreviewImageLabel->setAlignment(Qt::AlignCenter);
    latestPreviewImageLabel->setMinimumHeight(120);
    latestPreviewImageLabel->setFrameShape(QFrame::StyledPanel);
    latestPreviewImageLabel->setScaledContents(false);
    artifactPanel->bodyLayout()->addWidget(latestCheckpointLabel);
    artifactPanel->bodyLayout()->addWidget(latestOnnxLabel);
    artifactPanel->bodyLayout()->addWidget(latestReportLabel);
    artifactPanel->bodyLayout()->addWidget(latestPreviewLabel);
    artifactPanel->bodyLayout()->addWidget(latestPreviewImageLabel);
    artifactPanel->bodyLayout()->addStretch();

    auto* logPanel = new InfoPanel(QStringLiteral("训练日志"));
    logPanel->setMinimumWidth(0);
    auto* logEdit = new QTextEdit;
    logEdit->setObjectName(QStringLiteral("LogView"));
    logEdit->setReadOnly(true);
    logEdit->setLineWrapMode(QTextEdit::WidgetWidth);
    logEdit->document()->setMaximumBlockCount(2000);
    logEdit->setMinimumWidth(0);
    logEdit->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
    logPanel->bodyLayout()->addWidget(logEdit);

    auto* metricsPanel = new InfoPanel(uiText("指标曲线"));
    metricsPanel->setMinimumWidth(0);
    auto* metricsWidget = new MetricsWidget;
    metricsPanel->bodyLayout()->addWidget(metricsWidget, 1);

    auto* detailTabs = new QTabWidget;
    detailTabs->setObjectName(QStringLiteral("TrainingDetailTabs"));
    detailTabs->setDocumentMode(true);
    detailTabs->addTab(metricsPanel, uiText("指标曲线"));
    detailTabs->addTab(logPanel, QStringLiteral("训练日志"));
    detailTabs->addTab(artifactPanel, uiText("Checkpoint 与产物"));

    auto* rightSplitter = new QSplitter(Qt::Vertical);
    rightSplitter->setMinimumWidth(0);
    rightSplitter->addWidget(monitorPanel);
    rightSplitter->addWidget(detailTabs);
    rightSplitter->setStretchFactor(0, 0);
    rightSplitter->setStretchFactor(1, 1);
    rightSplitter->setSizes(QList<int>() << 150 << 520);

    auto* bodySplitter = new QSplitter(Qt::Horizontal);
    bodySplitter->addWidget(setupScroll);
    bodySplitter->addWidget(rightSplitter);
    bodySplitter->setStretchFactor(0, 0);
    bodySplitter->setStretchFactor(1, 1);
    bodySplitter->setSizes(QList<int>() << 220 << 760);

    layout->addWidget(headerPanel);
    layout->addWidget(bodySplitter, 1);
    trainingPageController_->attach(page);
    return page;
}
