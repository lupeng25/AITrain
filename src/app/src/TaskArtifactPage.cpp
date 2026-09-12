#include "WorkbenchTranslation.h"
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

TaskArtifactPage::TaskArtifactPage(QWidget* parent) : WorkspaceViewHost(parent)
{
    auto* refresh = workbenchButton(aitrain_app::workbenchText(QStringLiteral("刷新历史")));
    toolbar->addWidget(refresh);
    connect(refresh, &QPushButton::clicked, this, &TaskArtifactPage::refreshRequested);
    auto* catalog = addMode(aitrain_app::workbenchText(QStringLiteral("任务记录")));
    auto* filters = new QHBoxLayout;
    taskKindFilterCombo_ = new QComboBox(this); taskKindFilterCombo_->addItem(aitrain_app::workbenchText(QStringLiteral("全部任务")), QString()); taskKindFilterCombo_->hide();
    taskStateFilterCombo_ = new QComboBox;
    const QStringList labels = {aitrain_app::workbenchText(QStringLiteral("全部状态")), aitrain_app::workbenchText(QStringLiteral("已创建")), aitrain_app::workbenchText(QStringLiteral("启动中")), aitrain_app::workbenchText(QStringLiteral("运行中")), aitrain_app::workbenchText(QStringLiteral("取消中")), aitrain_app::workbenchText(QStringLiteral("已完成")), aitrain_app::workbenchText(QStringLiteral("失败")), aitrain_app::workbenchText(QStringLiteral("已取消"))};
    const QStringList states = {QString(), QStringLiteral("created"), QStringLiteral("starting"), QStringLiteral("running"), QStringLiteral("cancel_requested"), QStringLiteral("succeeded"), QStringLiteral("failed"), QStringLiteral("canceled")};
    for (int i = 0; i < states.size(); ++i) taskStateFilterCombo_->addItem(labels[i], states[i]);
    taskSearchEdit_ = new QLineEdit; taskSearchEdit_->setPlaceholderText(aitrain_app::workbenchText(QStringLiteral("搜索整个项目：任务、后端或配置"))); taskSearchEdit_->setMinimumWidth(0);
    filters->addWidget(taskStateFilterCombo_); filters->addWidget(taskSearchEdit_, 1); catalog->addLayout(filters);
    connect(taskStateFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() { emitFilterChanged(); });
    taskSearchEdit_->setMaxLength(200); taskSearchEdit_->setClearButtonEnabled(true);
    bindCatalogSearch(taskSearchEdit_, this, [this](const QString&) { emitFilterChanged(); });
    catalog->addWidget(workbenchHint(aitrain_app::workbenchText(QStringLiteral("搜索和状态筛选覆盖整个项目；结果按更新时间分页。"))));
    taskQueueTable_ = new QTableView; taskQueueTable_->setObjectName(QStringLiteral("TaskQueueTable"));
    taskListTableModel_ = new TaskListTableModel(taskQueueTable_); taskListFilterModel_ = new TaskListFilterProxyModel(taskQueueTable_); taskListFilterModel_->setSourceModel(taskListTableModel_);
    taskQueueTable_->setModel(taskListFilterModel_); taskQueueTable_->setSelectionMode(QAbstractItemView::SingleSelection); taskQueueTable_->setSelectionBehavior(QAbstractItemView::SelectRows); taskQueueTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    taskQueueTable_->verticalHeader()->hide(); taskQueueTable_->verticalHeader()->setDefaultSectionSize(40); taskQueueTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch); taskQueueTable_->setMinimumSize(0, 0);
    taskQueueTable_->setColumnHidden(1, true); taskQueueTable_->setColumnHidden(2, true); taskQueueTable_->setColumnHidden(6, true); taskQueueTable_->setShowGrid(false); catalog->addWidget(taskQueueTable_, 1);
    connect(taskQueueTable_->selectionModel(), &QItemSelectionModel::selectionChanged, this, [this]() { emitSelectedTaskChanged(); });
    auto* actions = new QHBoxLayout; auto* details = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看详情"))); taskLoadMoreButton_ = workbenchButton(aitrain_app::workbenchText(QStringLiteral("载入更多")), QStringLiteral("TaskLoadMoreButton"));
    actions->addWidget(details); actions->addStretch(); actions->addWidget(taskLoadMoreButton_); catalog->addLayout(actions);
    connect(taskLoadMoreButton_, &QPushButton::clicked, this, &TaskArtifactPage::loadMoreRequested);
    const auto open = [this]() { if (!selectedTaskId().isEmpty()) { emitSelectedTaskChanged(); setMode(1); } };
    connect(details, &QPushButton::clicked, this, open); connect(taskQueueTable_, &QTableView::doubleClicked, this, [open](const QModelIndex&) { open(); });
    auto* detail = addMode(aitrain_app::workbenchText(QStringLiteral("任务详情")));
    taskArtifactPanel_ = new TaskArtifactPanel; taskArtifactPanel_->setObjectName(QStringLiteral("TaskArtifactPanel")); detail->addWidget(taskArtifactPanel_, 1);
    taskCancelButton_ = workbenchButton(aitrain_app::workbenchText(QStringLiteral("取消当前任务")), QStringLiteral("TaskCancelButton")); taskCancelButton_->setEnabled(false); detail->addWidget(taskCancelButton_, 0, Qt::AlignRight);
    connect(taskCancelButton_, &QPushButton::clicked, this, &TaskArtifactPage::cancelRequested);
    setMode(0);
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
    Q_UNUSED(query)
    taskListFilterModel_->setQuery({});
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

QString TaskArtifactPage::selectedArtifactMember() const
{
    const auto* files = taskArtifactPanel_->findChild<QTableView*>(QStringLiteral("TaskArtifactFileTable"));
    return files ? files->currentIndex().data(ArtifactFileTableModel::RelativePathRole).toString() : QString();
}

void TaskArtifactPage::restoreArtifactMember(const QString& member)
{
    auto* files = taskArtifactPanel_->findChild<QTableView*>(QStringLiteral("TaskArtifactFileTable"));
    if (!files || member.isEmpty()) return;
    for (int row = 0; row < files->model()->rowCount(); ++row)
        if (files->model()->index(row, 0).data(ArtifactFileTableModel::RelativePathRole).toString() == member) { files->selectRow(row); return; }
}
