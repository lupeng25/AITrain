#include "MainWindow.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "TaskArtifactPanel.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QTableWidget>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildTaskQueuePage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
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
    taskSearchEdit_->setMinimumWidth(0);
    taskSearchEdit_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
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
    detailPanel->bodyLayout()->setSpacing(12);
    taskArtifactPanel_ = new TaskArtifactPanel;
    connect(taskArtifactPanel_, &TaskArtifactPanel::openDirectoryRequested, this, &MainWindow::openSelectedArtifactDirectory);
    connect(taskArtifactPanel_, &TaskArtifactPanel::copyPathRequested, this, &MainWindow::copySelectedArtifactPath);
    connect(taskArtifactPanel_, &TaskArtifactPanel::useForInferenceRequested, this, &MainWindow::useSelectedArtifactForInference);
    connect(taskArtifactPanel_, &TaskArtifactPanel::useForExportRequested, this, &MainWindow::useSelectedArtifactForExport);
    connect(taskArtifactPanel_, &TaskArtifactPanel::registerModelRequested, this, &MainWindow::registerSelectedArtifactAsModelVersion);
    connect(taskArtifactPanel_, &TaskArtifactPanel::evaluateRequested, this, &MainWindow::evaluateSelectedArtifact);
    connect(taskArtifactPanel_, &TaskArtifactPanel::benchmarkRequested, this, &MainWindow::benchmarkSelectedArtifact);
    connect(taskArtifactPanel_, &TaskArtifactPanel::deliveryReportRequested, this, &MainWindow::generateDeliveryReportFromSelectedArtifact);
    detailPanel->bodyLayout()->addWidget(taskArtifactPanel_, 1);
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
    page->setWidget(content);
    return page;
}
