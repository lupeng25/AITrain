#include "WorkbenchTranslation.h"
#include "TrainingPage.h"
#include "TrainingParameterPanels.h"
#include "MetricsWidget.h"
#include <QComboBox>
#include <QGridLayout>
#include <QIntValidator>
#include <QLineEdit>
#include <QProgressBar>
#include <QTextEdit>
#include <QTextDocument>
using namespace aitrain_app;

void TrainingWorkspacePage::buildLayout()
{
    auto* create = workbenchButton(aitrain_app::workbenchText(QStringLiteral("新建训练")), QStringLiteral("TrainingNew"), true);
    auto* refresh = workbenchButton(aitrain_app::workbenchText(QStringLiteral("刷新")));
    toolbar->addWidget(refresh);
    toolbar->addWidget(create);
    connect(create, &QPushButton::clicked, this, [this]() { setMode(Configuration); });
    connect(refresh, &QPushButton::clicked, this, &TrainingWorkspacePage::refreshHistoryRequested);
    auto* catalog = addMode(aitrain_app::workbenchText(QStringLiteral("训练记录")));
    catalog->addWidget(catalogSearchField(QStringLiteral("TrainingHistorySearch"), aitrain_app::workbenchText(QStringLiteral("搜索整个项目：任务、后端、数据或模型配置"))));
    historyStatus = workbenchHint(aitrain_app::workbenchText(QStringLiteral("打开项目后查看训练记录。")));
    catalog->addWidget(historyStatus);
    historyTable = workbenchTable({aitrain_app::workbenchText(QStringLiteral("训练")), aitrain_app::workbenchText(QStringLiteral("数据版本")), aitrain_app::workbenchText(QStringLiteral("模型 / 算法")), aitrain_app::workbenchText(QStringLiteral("状态")), aitrain_app::workbenchText(QStringLiteral("更新时间"))});
    historyTable->setObjectName(QStringLiteral("TrainingHistoryTable"));
    catalog->addWidget(historyTable, 1);
    auto* historyActions = new QHBoxLayout;
    auto* details = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看训练")));
    moreHistory = workbenchButton(aitrain_app::workbenchText(QStringLiteral("继续查找记录")));
    historyActions->addWidget(details); historyActions->addStretch(); historyActions->addWidget(moreHistory);
    catalog->addLayout(historyActions);
    const auto openSelected = [this]() {
        const auto* item = historyTable->item(historyTable->currentRow(), 0);
        if (item) emit historyRequested(item->data(Qt::UserRole).toString());
    };
    connect(details, &QPushButton::clicked, this, openSelected);
    connect(historyTable, &QTableWidget::cellDoubleClicked, this, [openSelected](int, int) { openSelected(); });
    connect(moreHistory, &QPushButton::clicked, this, &TrainingWorkspacePage::moreHistoryRequested);

    auto* config = addMode(aitrain_app::workbenchText(QStringLiteral("配置训练")));
    auto* dataset = workbenchHint(); dataset->setObjectName(QStringLiteral("TrainingDatasetNote"));
    auto* choose = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择数据版本与样本")), QStringLiteral("TrainingSelectDataset"));
    auto* datasetRow = new QHBoxLayout; datasetRow->addWidget(dataset, 1); datasetRow->addWidget(choose);
    config->addLayout(datasetRow);
    connect(choose, &QPushButton::clicked, this, &TrainingWorkspacePage::selectDatasetRequested);
    auto* sample = workbenchHint(aitrain_app::workbenchText(QStringLiteral("交付检查样本：尚未选择")));
    sample->setObjectName(QStringLiteral("TrainingSampleNote")); config->addWidget(sample);
    auto* grid = new QGridLayout; grid->setColumnStretch(1, 1); grid->setColumnStretch(3, 1);
    grid->setVerticalSpacing(14); grid->setHorizontalSpacing(14);
    auto* capability = new QComboBox(this); capability->setObjectName(QStringLiteral("TrainingCapability")); capability->hide();
    auto* task = new QComboBox(this); task->setObjectName(QStringLiteral("TrainingTaskType")); task->hide();
    auto* backend = new QComboBox; backend->setObjectName(QStringLiteral("TrainingBackend"));
    auto* model = new QComboBox; model->setEditable(true); model->setObjectName(QStringLiteral("TrainingModelPreset"));
    const auto add = [grid](int row, int col, const QString& label, QWidget* field) {
        auto* caption = new QLabel(label); caption->setWordWrap(true);
        field->setMinimumWidth(0); field->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        grid->addWidget(caption, row, col * 2); grid->addWidget(field, row, col * 2 + 1);
    };
    add(0, 0, aitrain_app::workbenchText(QStringLiteral("任务 / 算法")), backend);
    add(0, 1, aitrain_app::workbenchText(QStringLiteral("模型 / 配置")), model);
    const auto integer = [this](const QString& name, const QString& value) {
        auto* edit = new QLineEdit(value); edit->setObjectName(name);
        edit->setValidator(new QIntValidator(1, 1000000, edit)); return edit;
    };
    add(1, 0, aitrain_app::workbenchText(QStringLiteral("训练轮数")), integer(QStringLiteral("TrainingEpochs"), QStringLiteral("20")));
    add(1, 1, aitrain_app::workbenchText(QStringLiteral("批次大小")), integer(QStringLiteral("TrainingBatchSize"), QStringLiteral("8")));
    add(2, 0, aitrain_app::workbenchText(QStringLiteral("输入尺寸")), integer(QStringLiteral("TrainingImageSize"), QStringLiteral("640")));
    auto* devices = new QStackedWidget; devices->setObjectName(QStringLiteral("TrainingDeviceStack"));
    for (const QString& prefix : {QStringLiteral("YoloTrainArg_"), QStringLiteral("SmpTrainArg_"), QStringLiteral("AnomalyTrainArg_")}) {
        auto* edit = new QLineEdit(prefix.startsWith(QStringLiteral("Yolo")) ? QString() : QStringLiteral("cpu"));
        edit->setObjectName(prefix + QStringLiteral("device")); edit->setPlaceholderText(aitrain_app::workbenchText(QStringLiteral("自动 / cpu / GPU 编号"))); devices->addWidget(edit);
    }
    devices->addWidget(workbenchHint(aitrain_app::workbenchText(QStringLiteral("由 PaddleOCR 官方配置指定"))));
    add(2, 1, aitrain_app::workbenchText(QStringLiteral("计算设备")), devices);
    config->addLayout(grid);
    auto* hint = workbenchHint(); hint->setObjectName(QStringLiteral("TrainingBackendHint")); config->addWidget(hint);
    auto* summary = workbenchHint(); summary->setObjectName(QStringLiteral("TrainingRunNote")); config->addWidget(summary);
    auto* error = workbenchHint(); error->setObjectName(QStringLiteral("TrainingFormError")); config->addWidget(error);
    auto* draftRow = new QHBoxLayout;
    auto* draftStatus = workbenchHint(aitrain_app::workbenchText(QStringLiteral("草稿按项目自动保存，可在重启后恢复。"))); draftStatus->setObjectName(QStringLiteral("TrainingDraftStatus"));
    auto* saveDraft = workbenchButton(aitrain_app::workbenchText(QStringLiteral("保存草稿")), QStringLiteral("TrainingSaveDraft"));
    auto* discardDraft = workbenchButton(aitrain_app::workbenchText(QStringLiteral("丢弃草稿")), QStringLiteral("TrainingDiscardDraft"));
    draftRow->addWidget(draftStatus, 1); draftRow->addWidget(saveDraft); draftRow->addWidget(discardDraft); config->addLayout(draftRow);
    config->addStretch();
    auto* actions = new QHBoxLayout;
    auto* advanced = workbenchButton(aitrain_app::workbenchText(QStringLiteral("高级参数")), QStringLiteral("AdvancedToggle"));
    auto* cancel = workbenchButton(aitrain_app::workbenchText(QStringLiteral("返回训练记录")));
    auto* start = workbenchButton(aitrain_app::workbenchText(QStringLiteral("开始训练")), QStringLiteral("TrainingStart"), true);
    actions->addWidget(advanced); actions->addStretch(); actions->addWidget(cancel); actions->addWidget(start); config->addLayout(actions);
    connect(advanced, &QPushButton::clicked, this, &TrainingWorkspacePage::advancedRequested);
    connect(start, &QPushButton::clicked, this, &TrainingWorkspacePage::startRequested);
    connect(cancel, &QPushButton::clicked, this, [this]() { setMode(Catalog); });

    auto* monitor = addMode(aitrain_app::workbenchText(QStringLiteral("训练详情")));
    auto* monitorContext = workbenchHint(); monitorContext->setObjectName(QStringLiteral("TrainingMonitorContext")); monitor->addWidget(monitorContext);
    auto* phase = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未选择训练。"))); phase->setObjectName(QStringLiteral("TrainingPhaseStatus")); monitor->addWidget(phase);
    auto* progress = new QProgressBar; progress->setObjectName(QStringLiteral("TrainingProgress")); progress->setValue(0); monitor->addWidget(progress);
    auto* live = new QHBoxLayout;
    const QStringList keys = {QStringLiteral("Epoch"), QStringLiteral("Batch"), QStringLiteral("Eta"), QStringLiteral("Device"), QStringLiteral("Loss"), QStringLiteral("Map")};
    const QStringList captions = {aitrain_app::workbenchText(QStringLiteral("轮数")), aitrain_app::workbenchText(QStringLiteral("批次")), aitrain_app::workbenchText(QStringLiteral("预计剩余")), aitrain_app::workbenchText(QStringLiteral("设备")), aitrain_app::workbenchText(QStringLiteral("损失 / 误差")), aitrain_app::workbenchText(QStringLiteral("评价指标"))};
    for (int i = 0; i < keys.size(); ++i) {
        auto* column = new QVBoxLayout; column->addWidget(workbenchHint(captions[i]));
        auto* value = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未提供"))); value->setObjectName(QStringLiteral("Training%1Value").arg(keys[i])); column->addWidget(value); live->addLayout(column, 1);
    }
    monitor->addLayout(live); monitor->addWidget(new MetricsWidget, 1);
    auto* latest = new QTextEdit; latest->setReadOnly(true); latest->setObjectName(QStringLiteral("TrainingLatestLog"));
    latest->document()->setMaximumBlockCount(6); latest->setMaximumHeight(100); monitor->addWidget(latest);
    auto* monitorActions = new QHBoxLayout;
    auto* configuration = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看配置")));
    auto* copy = workbenchButton(aitrain_app::workbenchText(QStringLiteral("复制配置新建训练")), QStringLiteral("TrainingCopyConfiguration"));
    monitorActions->addWidget(configuration); monitorActions->addWidget(copy);
    connect(configuration, &QPushButton::clicked, this, &TrainingWorkspacePage::configurationRequested);
    connect(copy, &QPushButton::clicked, this, &TrainingWorkspacePage::copyConfigurationRequested);
    auto* fullLog = workbenchButton(aitrain_app::workbenchText(QStringLiteral("完整日志")));
    auto* artifacts = workbenchButton(aitrain_app::workbenchText(QStringLiteral("任务与产物")));
    auto* models = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看模型"))); modelsButton = models; models->setEnabled(false);
    cancelTaskButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("取消任务")), QStringLiteral("TrainingCancel")); cancelTaskButton->setEnabled(false);
    monitorActions->addWidget(fullLog); monitorActions->addWidget(artifacts); monitorActions->addWidget(models); monitorActions->addStretch(); monitorActions->addWidget(cancelTaskButton); monitor->addLayout(monitorActions);
    connect(cancelTaskButton, &QPushButton::clicked, this, &TrainingWorkspacePage::cancelRequested);
    connect(fullLog, &QPushButton::clicked, this, &TrainingWorkspacePage::logRequested);
    connect(artifacts, &QPushButton::clicked, this, &TrainingWorkspacePage::currentTaskRequested);
    connect(models, &QPushButton::clicked, this, &TrainingWorkspacePage::modelsRequested);

    auto* advancedView = addMode(aitrain_app::workbenchText(QStringLiteral("高级参数")));
    for (const auto& pair : {qMakePair(QStringLiteral("yolo"), QStringLiteral("YoloOfficialArgsGroup")), qMakePair(QStringLiteral("smp"), QStringLiteral("SmpSemanticArgsGroup")), qMakePair(QStringLiteral("anomaly"), QStringLiteral("AnomalyDetectionArgsGroup"))}) {
        auto* panel = buildTrainingParameterPanel(pair.first); panel->setObjectName(pair.second); advancedView->addWidget(panel, 1);
    }
    auto* ocr = workbenchHint(aitrain_app::workbenchText(QStringLiteral("PaddleOCR 使用所选官方模型配置。轮数、批次和尺寸沿用现有适配器映射；其他参数由官方配置定义。")));
    ocr->setObjectName(QStringLiteral("OcrOfficialArgsGroup")); advancedView->addWidget(ocr);
    auto* advancedActions = new QHBoxLayout; advancedActions->addStretch();
    auto* discard = workbenchButton(aitrain_app::workbenchText(QStringLiteral("取消修改"))); auto* apply = workbenchButton(aitrain_app::workbenchText(QStringLiteral("应用参数")), {}, true);
    advancedActions->addWidget(discard); advancedActions->addWidget(apply); advancedView->addLayout(advancedActions);
    connect(discard, &QPushButton::clicked, this, &TrainingWorkspacePage::cancelAdvancedRequested);
    connect(apply, &QPushButton::clicked, this, &TrainingWorkspacePage::applyAdvancedRequested);
    auto* logs = addMode(aitrain_app::workbenchText(QStringLiteral("完整日志")));
    auto* log = new QTextEdit; log->setObjectName(QStringLiteral("LogView")); log->setReadOnly(true); log->document()->setMaximumBlockCount(2000); logs->addWidget(log, 1);
    auto* result = addMode(aitrain_app::workbenchText(QStringLiteral("训练产物")));
    for (const QString& name : {QStringLiteral("Checkpoint"), QStringLiteral("Onnx"), QStringLiteral("Report"), QStringLiteral("Preview")}) {
        auto* label = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚无产物"))); label->setObjectName(QStringLiteral("TrainingLatest%1").arg(name)); result->addWidget(label);
    }
    result->addStretch();
    connect(views, &QStackedWidget::currentChanged, this, [this, create, refresh](int index) {
        create->setVisible(index == Catalog); refresh->setVisible(index == Catalog);
        if (index == Advanced) backButton->hide();
    });
    setMode(Catalog);
}
