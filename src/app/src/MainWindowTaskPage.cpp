#include "MainWindow.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "TaskArtifactPanel.h"
#include "TaskArtifactPresenter.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDir>
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
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    auto* headerRefreshButton = primaryButton(QStringLiteral("刷新历史"));
    connect(headerRefreshButton, &QPushButton::clicked, this, &MainWindow::updateRecentTasks);

    auto* refreshButton = primaryButton(QStringLiteral("刷新历史"));
    taskCancelButton_ = dangerButton(QStringLiteral("取消当前任务"));
    taskCancelButton_->setObjectName(QStringLiteral("TaskCancelButton"));
    taskCancelButton_->setEnabled(false);
    taskCancelButton_->setToolTip(uiText("只允许取消当前 Worker 活动任务；历史任务只读。"));
    taskKindFilterCombo_ = new QComboBox;
    taskKindFilterCombo_->setMinimumWidth(140);
    taskKindFilterCombo_->addItem(uiText("全部类别"), QString());
    taskKindFilterCombo_->addItem(uiText(" 持久化任务"), QStringLiteral(""));
    taskStateFilterCombo_ = new QComboBox;
    taskStateFilterCombo_->setMinimumWidth(140);
    taskStateFilterCombo_->addItem(uiText("全部状态"), QString());
    taskStateFilterCombo_->addItem(uiText("排队中"), QStringLiteral("queued"));
    taskStateFilterCombo_->addItem(uiText("运行中"), QStringLiteral("running"));
    taskStateFilterCombo_->addItem(uiText("失败"), QStringLiteral("failed"));
    taskStateFilterCombo_->addItem(uiText("已取消"), QStringLiteral("canceled"));
    taskStateFilterCombo_->addItem(uiText("已创建"), QStringLiteral("created"));
    taskStateFilterCombo_->addItem(uiText("启动中"), QStringLiteral("starting"));
    taskStateFilterCombo_->addItem(uiText("取消中"), QStringLiteral("cancel_requested"));
    taskStateFilterCombo_->addItem(uiText("已完成"), QStringLiteral("succeeded"));
    taskSearchEdit_ = new QLineEdit;
    taskSearchEdit_->setMinimumWidth(0);
    taskSearchEdit_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    taskSearchEdit_->setPlaceholderText(QStringLiteral("搜索任务、后端、消息"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::updateRecentTasks);
    connect(taskCancelButton_, &QPushButton::clicked, this, &MainWindow::cancelSelectedTask);
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
    actionLayout->addWidget(taskCancelButton_, 0, 2);

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
    toolbar->bodyLayout()->addWidget(mutedLabel(QStringLiteral("这里只读展示已持久化任务；已提交产物、指标和工作流步骤在下方集中查看。")));

    auto* tablePanel = new InfoPanel(QStringLiteral("任务历史"));
    tablePanel->setMinimumWidth(300);
    tablePanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    taskQueueTable_ = new QTableWidget(0, 7);
    taskQueueTable_->setObjectName(QStringLiteral("TaskQueueTable"));
    taskQueueTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务")
        << QStringLiteral("类别")
        << QStringLiteral("内置能力")
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
    detailPanel->setMinimumWidth(480);
    detailPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    detailPanel->bodyLayout()->setSpacing(12);
    taskArtifactPanel_ = new TaskArtifactPanel;
    taskArtifactPanel_->setObjectName(QStringLiteral("TaskArtifactPanel"));
    taskArtifactPanel_->setPresenter(taskArtifactPresenter_);
    if (taskArtifactPresenter_) {
        connect(taskArtifactPresenter_, &TaskArtifactPresenter::detailsChanged, this, [this]() {
            if (taskArtifactPanel_) taskArtifactPanel_->setDetails(taskArtifactPresenter_->details());
        });
    }
    detailPanel->bodyLayout()->addWidget(taskArtifactPanel_, 1);
    auto* bodySplitter = new QSplitter(Qt::Horizontal);
    bodySplitter->addWidget(tablePanel);
    bodySplitter->addWidget(detailPanel);
    bodySplitter->setChildrenCollapsible(false);
    bodySplitter->setStretchFactor(0, 2);
    bodySplitter->setStretchFactor(1, 3);
    bodySplitter->setSizes(QList<int>() << 380 << 680);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("TASK ARTIFACT CENTER"),
        uiText("任务与产物工作台"),
        uiText("按任务追踪已提交产物、指标和工作流步骤；页面不读取 Worker 原始消息，也不暴露 Artifact Store 裸路径。"),
        headerRefreshButton,
        QStringList()
            << uiText("任务历史")
            << uiText("产物")
            << uiText("指标")
            << uiText("报告")));
    layout->addWidget(toolbar);
    layout->addWidget(bodySplitter, 1);
    page->setWidget(content);
    return page;
}

void MainWindow::cancelSelectedTask()
{
    if (!activeTaskId_.isEmpty() && worker_.isRunning()) {
        // Worker/Core 是运行任务取消的唯一写入方；GUI 不再直接改写同一 Task
        // 的 CancelRequested，避免双写和 Worker/Core 状态竞态。
        worker_.cancel();
        return;
    }
    QMessageBox::information(this, uiText("任务队列"),
        uiText("只能取消当前 GUI 会话派发且仍在运行的任务。历史任务为只读。"));
}

void MainWindow::updateTaskCancelButton()
{
    if (!taskCancelButton_) {
        return;
    }
    const bool canCancel = !activeTaskId_.isEmpty() && worker_.isRunning();
    taskCancelButton_->setEnabled(canCancel);
    taskCancelButton_->setToolTip(canCancel
        ? uiText("取消当前 Worker 活动任务。")
        : uiText("只允许取消当前 Worker 活动任务；历史任务只读。"));
}
