#include "TaskArtifactPage.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "TaskArtifactPanel.h"
#include "TaskArtifactTableModels.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QTableView>
#include <QVBoxLayout>

using namespace aitrain_app;

TaskArtifactPage::TaskArtifactPage(QWidget* parent)
    : QScrollArea(parent)
{
    setWidgetResizable(true);
    setFrameShape(QFrame::NoFrame);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    auto* headerRefreshButton = primaryButton(QStringLiteral("刷新历史"));
    connect(headerRefreshButton, &QPushButton::clicked,
        this, &TaskArtifactPage::refreshRequested);

    auto* refreshButton = primaryButton(QStringLiteral("刷新历史"));
    connect(refreshButton, &QPushButton::clicked,
        this, &TaskArtifactPage::refreshRequested);
    taskCancelButton_ = dangerButton(QStringLiteral("取消当前任务"));
    taskCancelButton_->setObjectName(QStringLiteral("TaskCancelButton"));
    taskCancelButton_->setEnabled(false);
    taskCancelButton_->setToolTip(uiText("只允许取消当前 Worker 活动任务；历史任务只读。"));
    connect(taskCancelButton_, &QPushButton::clicked,
        this, &TaskArtifactPage::cancelRequested);

    taskKindFilterCombo_ = new QComboBox;
    taskKindFilterCombo_->setMinimumWidth(140);
    taskKindFilterCombo_->addItem(uiText("全部类别"), QString());
    taskKindFilterCombo_->addItem(uiText("持久化任务"), QStringLiteral(""));
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
    connect(taskKindFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, [this]() { emitFilterChanged(); });
    connect(taskStateFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, [this]() { emitFilterChanged(); });
    connect(taskSearchEdit_, &QLineEdit::textChanged,
        this, [this]() { emitFilterChanged(); });

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
    toolbar->bodyLayout()->addWidget(mutedLabel(
        QStringLiteral("这里只读展示已持久化任务；已提交产物、指标和工作流步骤在下方集中查看。")));

    auto* tablePanel = new InfoPanel(QStringLiteral("任务历史"));
    tablePanel->setMinimumWidth(300);
    tablePanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    taskQueueTable_ = new QTableView;
    taskQueueTable_->setObjectName(QStringLiteral("TaskQueueTable"));
    taskListTableModel_ = new TaskListTableModel(taskQueueTable_);
    taskListFilterModel_ = new TaskListFilterProxyModel(taskQueueTable_);
    taskListFilterModel_->setSourceModel(taskListTableModel_);
    taskQueueTable_->setModel(taskListFilterModel_);
    taskQueueTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    taskQueueTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    taskQueueTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    taskQueueTable_->verticalHeader()->setVisible(false);
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
    connect(taskQueueTable_->selectionModel(), &QItemSelectionModel::selectionChanged,
        this, [this]() { emitSelectedTaskChanged(); });
    tablePanel->bodyLayout()->addWidget(taskQueueTable_);
    taskLoadMoreButton_ = new QPushButton(uiText("加载更多"));
    taskLoadMoreButton_->setObjectName(QStringLiteral("TaskLoadMoreButton"));
    taskLoadMoreButton_->setEnabled(false);
    connect(taskLoadMoreButton_, &QPushButton::clicked,
        this, &TaskArtifactPage::loadMoreRequested);
    tablePanel->bodyLayout()->addWidget(taskLoadMoreButton_, 0, Qt::AlignHCenter);

    auto* detailPanel = new InfoPanel(QStringLiteral("任务详情与产物"));
    detailPanel->setMinimumWidth(480);
    detailPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    detailPanel->bodyLayout()->setSpacing(12);
    taskArtifactPanel_ = new TaskArtifactPanel;
    taskArtifactPanel_->setObjectName(QStringLiteral("TaskArtifactPanel"));
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
        QStringList() << uiText("任务历史") << uiText("产物")
                      << uiText("指标") << uiText("报告")));
    layout->addWidget(toolbar);
    layout->addWidget(bodySplitter, 1);
    setWidget(content);
}

void TaskArtifactPage::setPresenter(TaskArtifactPresenter* presenter)
{
    taskArtifactPanel_->setPresenter(presenter);
}

void TaskArtifactPage::setRows(const QVector<TaskListItem>& rows, bool hasMore)
{
    const QString previousTaskId = selectedTaskId();
    taskListTableModel_->setRows(rows);
    taskLoadMoreButton_->setEnabled(hasMore);
    applyFilter(taskKindFilterCombo_->currentData().toString(),
        taskStateFilterCombo_->currentData().toString(),
        taskSearchEdit_->text());
    if (!previousTaskId.isEmpty()) {
        for (int row = 0; row < taskListFilterModel_->rowCount(); ++row) {
            const QModelIndex index = taskListFilterModel_->index(row, 0);
            if (index.data(TaskListTableModel::TaskIdRole).toString() == previousTaskId) {
                const QSignalBlocker blocker(taskQueueTable_->selectionModel());
                taskQueueTable_->selectRow(row);
                taskQueueTable_->setCurrentIndex(index);
                return;
            }
        }
    }
    ensureVisibleSelection();
}

void TaskArtifactPage::setDetails(const TaskArtifactDetails& details)
{
    taskArtifactPanel_->setDetails(details);
}

void TaskArtifactPage::clearDetails()
{
    taskArtifactPanel_->clear();
}

void TaskArtifactPage::setCancelable(bool cancelable)
{
    taskCancelButton_->setEnabled(cancelable);
    taskCancelButton_->setToolTip(cancelable
        ? uiText("取消当前 Worker 活动任务。")
        : uiText("只允许取消当前 Worker 活动任务；历史任务只读。"));
}

void TaskArtifactPage::applyFilter(const QString& taskKind, const QString& taskState,
    const QString& query)
{
    taskListFilterModel_->setTaskKind(taskKind);
    taskListFilterModel_->setTaskState(taskState);
    taskListFilterModel_->setQuery(query);
    ensureVisibleSelection();
}

QString TaskArtifactPage::selectedTaskId() const
{
    const QModelIndex current = taskQueueTable_->currentIndex();
    return current.isValid()
        ? current.data(TaskListTableModel::TaskIdRole).toString()
        : QString();
}

void TaskArtifactPage::ensureVisibleSelection()
{
    const QModelIndex current = taskQueueTable_->currentIndex();
    if (current.isValid() && current.row() < taskListFilterModel_->rowCount()) {
        return;
    }
    const QSignalBlocker blocker(taskQueueTable_->selectionModel());
    taskQueueTable_->clearSelection();
    if (taskListFilterModel_->rowCount() > 0) {
        taskQueueTable_->selectRow(0);
        taskQueueTable_->setCurrentIndex(taskListFilterModel_->index(0, 0));
    }
    emitSelectedTaskChanged();
}

void TaskArtifactPage::emitFilterChanged()
{
    emit filterChanged(taskKindFilterCombo_->currentData().toString(),
        taskStateFilterCombo_->currentData().toString(), taskSearchEdit_->text());
}

void TaskArtifactPage::emitSelectedTaskChanged()
{
    emit selectedTaskChanged(selectedTaskId());
}
