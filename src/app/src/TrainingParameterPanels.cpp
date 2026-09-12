#include "TrainingParameterPanels.h"
#include "MainWindowSupport.h"
#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QStackedWidget>
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
    auto* form = new QGridLayout(group);
    form->setColumnStretch(1, 1);
    form->setColumnStretch(3, 1);
    form->setVerticalSpacing(10);
    form->setAlignment(Qt::AlignTop);
    return group;
}

void addYoloRow(QGroupBox* group, const QString& label, QWidget* field)
{
    auto* grid = qobject_cast<QGridLayout*>(group->layout());
    const int index = group->property("fieldCount").toInt();
    group->setProperty("fieldCount", index + 1);
    const int row = index / 2, column = (index % 2) * 2;
    auto* caption = new QLabel(label);
    caption->setWordWrap(true);
    field->setMinimumWidth(0);
    if (label == QStringLiteral("export")) {
        grid->addWidget(caption, row + 1, 0);
        grid->addWidget(field, row + 1, 1, 1, 3);
    } else {
        grid->addWidget(caption, row, column);
        grid->addWidget(field, row, column + 1);
    }
}

void addSmpRow(QGroupBox* group, const QString& label, QWidget* field)
{
    auto* grid = qobject_cast<QGridLayout*>(group->layout());
    const int index = group->property("fieldCount").toInt();
    group->setProperty("fieldCount", index + 1);
    const int row = index / 2, column = (index % 2) * 2;
    auto* caption = new QLabel(label);
    caption->setWordWrap(true);
    field->setMinimumWidth(0);
    if (label == QStringLiteral("export")) {
        grid->addWidget(caption, row + 1, 0);
        grid->addWidget(field, row + 1, 1, 1, 3);
    } else {
        grid->addWidget(caption, row, column);
        grid->addWidget(field, row, column + 1);
    }
}

void addAnomalyRow(QGroupBox* group, const QString& label, QWidget* field)
{
    auto* grid = qobject_cast<QGridLayout*>(group->layout());
    const int index = group->property("fieldCount").toInt();
    group->setProperty("fieldCount", index + 1);
    const int row = index / 2, column = (index % 2) * 2;
    auto* caption = new QLabel(label);
    caption->setWordWrap(true);
    field->setMinimumWidth(0);
    if (label == QStringLiteral("export")) {
        grid->addWidget(caption, row + 1, 0);
        grid->addWidget(field, row + 1, 1, 1, 3);
    } else {
        grid->addWidget(caption, row, column);
        grid->addWidget(field, row, column + 1);
    }
}

QWidget* buildYoloOfficialArgsPanel()
{
    auto* container = new QWidget;
    auto* root = new QHBoxLayout(container);
    root->setContentsMargins(0, 0, 0, 0);
    auto* groups = new QListWidget;
    groups->setFixedWidth(150);
    groups->setWordWrap(true);
    groups->setResizeMode(QListView::Adjust);
    groups->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* stack = new QStackedWidget;
    root->addWidget(groups);
    root->addWidget(stack, 1);
    QObject::connect(groups, &QListWidget::currentRowChanged, stack, &QStackedWidget::setCurrentIndex);
    const auto addGroup = [groups, stack](QGroupBox* group) {
        groups->addItem(group->title());
        stack->addWidget(group);
        if (groups->currentRow() < 0) groups->setCurrentRow(0);
    };

    auto* deviceGroup = yoloArgGroup(uiText("数据与设备"));
    addYoloRow(deviceGroup, QStringLiteral("seed"), yoloArgLineEdit(QStringLiteral("seed"), QStringLiteral("42"), QStringLiteral("42")));
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

    addGroup(deviceGroup);
    addGroup(optimizerGroup);
    addGroup(augmentGroup);
    addGroup(segmentationGroup);
    addGroup(validationGroup);
    return container;
}

QWidget* buildSmpArgsPanel()
{
    auto* container = new QWidget;
    auto* root = new QHBoxLayout(container);
    root->setContentsMargins(0, 0, 0, 0);
    auto* groups = new QListWidget;
    groups->setFixedWidth(150);
    groups->setWordWrap(true);
    groups->setResizeMode(QListView::Adjust);
    groups->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* stack = new QStackedWidget;
    root->addWidget(groups);
    root->addWidget(stack, 1);
    QObject::connect(groups, &QListWidget::currentRowChanged, stack, &QStackedWidget::setCurrentIndex);
    const auto addGroup = [groups, stack](QGroupBox* group) {
        groups->addItem(group->title());
        stack->addWidget(group);
        if (groups->currentRow() < 0) groups->setCurrentRow(0);
    };

    auto* trainGroup = yoloArgGroup(uiText("训练参数"));
    addSmpRow(trainGroup, QStringLiteral("seed"), smpArgLineEdit(QStringLiteral("seed"), QStringLiteral("42"), QStringLiteral("42")));
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

    addGroup(trainGroup);
    addGroup(dataGroup);
    return container;
}

QWidget* buildAnomalyArgsPanel()
{
    auto* container = new QWidget;
    auto* root = new QHBoxLayout(container);
    root->setContentsMargins(0, 0, 0, 0);
    auto* groups = new QListWidget;
    groups->setFixedWidth(150);
    groups->setWordWrap(true);
    groups->setResizeMode(QListView::Adjust);
    groups->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* stack = new QStackedWidget;
    root->addWidget(groups);
    root->addWidget(stack, 1);
    QObject::connect(groups, &QListWidget::currentRowChanged, stack, &QStackedWidget::setCurrentIndex);
    const auto addGroup = [groups, stack](QGroupBox* group) {
        groups->addItem(group->title());
        stack->addWidget(group);
        if (groups->currentRow() < 0) groups->setCurrentRow(0);
    };

    auto* runtimeGroup = yoloArgGroup(uiText("运行与阈值"));
    addAnomalyRow(runtimeGroup, QStringLiteral("seed"), anomalyArgLineEdit(QStringLiteral("seed"), QStringLiteral("42"), QStringLiteral("42")));
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

    addGroup(runtimeGroup);
    addGroup(patchCoreGroup);
    addGroup(efficientAdGroup);
    return container;
}
} // namespace

QWidget* buildTrainingParameterPanel(const QString& family)
{
    if (family == QStringLiteral("yolo")) return buildYoloOfficialArgsPanel();
    if (family == QStringLiteral("smp")) return buildSmpArgsPanel();
    return buildAnomalyArgsPanel();
}

