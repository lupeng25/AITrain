#include "DashboardPage.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"

#include <QAbstractItemView>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

using namespace aitrain_app;

namespace {

InfoPanel* metricCard(const QString& label, const QString& value,
    const QString& caption)
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

void configureTable(QTableWidget* table)
{
    table->setAlternatingRowColors(true);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setStretchLastSection(true);
    table->horizontalHeader()->setDefaultAlignment(
        Qt::AlignLeft | Qt::AlignVCenter);
    table->setShowGrid(false);
}

} // namespace

DashboardWorkspacePage::DashboardWorkspacePage(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    projectStatusLabel_ = inlineStatusLabel(QStringLiteral(
        "未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    projectStatusLabel_->setObjectName(QStringLiteral("ProjectWorkspaceStatus"));
    gpuStatusLabel_ = inlineStatusLabel(QStringLiteral("GPU / 运行时：未执行环境自检"));
    allowLabelToShrink(projectStatusLabel_);
    allowLabelToShrink(gpuStatusLabel_);

    auto* grid = new QGridLayout;
    grid->setSpacing(10);
    auto* projectCard = metricCard(QStringLiteral("项目"),
        QStringLiteral("未打开"), QStringLiteral("当前本地工作目录"));
    projectValueLabel_ = projectCard->findChild<QLabel*>(
        QStringLiteral("MetricValue"));
    grid->addWidget(projectCard, 0, 0);
    auto* datasetCard = metricCard(QStringLiteral("数据集"),
        QStringLiteral("0"), QStringLiteral("快照 / 数据集"));
    datasetValueLabel_ = datasetCard->findChild<QLabel*>(
        QStringLiteral("MetricValue"));
    datasetValueLabel_->setObjectName(QStringLiteral("DashboardDatasetSummary"));
    grid->addWidget(datasetCard, 0, 1);
    auto* taskCard = metricCard(QStringLiteral("任务"), QStringLiteral("0"),
        QStringLiteral("训练、校验、导出、推理记录"));
    taskValueLabel_ = taskCard->findChild<QLabel*>(
        QStringLiteral("MetricValue"));
    taskValueLabel_->setObjectName(QStringLiteral("DashboardTaskSummary"));
    grid->addWidget(taskCard, 0, 2);

    auto* bottom = new QWidget;
    auto* bottomLayout = new QHBoxLayout(bottom);
    bottomLayout->setContentsMargins(0, 0, 0, 0);
    bottomLayout->setSpacing(12);

    auto* workflowPanel = new InfoPanel(QStringLiteral("下一步"));
    nextStepLabel_ = emptyStateLabel(QStringLiteral(
        "打开项目后，按 数据集 -> 训练实验 -> 任务与产物 -> 部署验证 的顺序完成本机训练闭环。"));
    allowLabelToShrink(nextStepLabel_);
    workflowPanel->bodyLayout()->addWidget(nextStepLabel_);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QGridLayout(actionStrip);
    actionLayout->setContentsMargins(12, 12, 12, 12);
    actionLayout->setSpacing(10);
    const QVector<QPair<QString, DashboardRoute>> actions{
        {QStringLiteral("打开项目"), DashboardRoute::Project},
        {QStringLiteral("导入 / 校验数据"), DashboardRoute::Dataset},
        {QStringLiteral("启动训练实验"), DashboardRoute::Training},
        {QStringLiteral("查看任务与产物"), DashboardRoute::TaskArtifact},
        {QStringLiteral("模型库"), DashboardRoute::ModelRegistry},
        {QStringLiteral("部署验证"), DashboardRoute::RuntimeDelivery}};
    for (int index = 0; index < actions.size(); ++index) {
        QPushButton* button = index == 0
            ? primaryButton(actions.at(index).first)
            : new QPushButton(actions.at(index).first);
        const DashboardRoute route = actions.at(index).second;
        connect(button, &QPushButton::clicked, this,
            [this, route]() { emit routeRequested(route); });
        actionLayout->addWidget(button, index / 2, index % 2);
    }
    workflowPanel->bodyLayout()->addWidget(actionStrip);
    workflowPanel->bodyLayout()->addStretch();

    auto* recentPanel = new InfoPanel(QStringLiteral("最近任务"));
    recentTasksTable_ = new QTableWidget(0, 5);
    recentTasksTable_->setObjectName(QStringLiteral("DashboardRecentTasks"));
    recentTasksTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务") << QStringLiteral("内置能力")
        << QStringLiteral("类型") << QStringLiteral("状态")
        << QStringLiteral("消息"));
    configureTable(recentTasksTable_);
    recentTasksTable_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(
        0, QHeaderView::Stretch);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(
        1, QHeaderView::Stretch);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(
        2, QHeaderView::ResizeToContents);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(
        3, QHeaderView::ResizeToContents);
    recentTasksTable_->horizontalHeader()->setSectionResizeMode(
        4, QHeaderView::Stretch);
    recentPanel->bodyLayout()->addWidget(recentTasksTable_);
    bottomLayout->addWidget(workflowPanel, 3, Qt::AlignTop);
    bottomLayout->addWidget(recentPanel, 4);

    layout->addWidget(projectStatusLabel_);
    layout->addWidget(gpuStatusLabel_);
    layout->addLayout(grid);
    layout->addWidget(bottom, 1);
}

void DashboardWorkspacePage::render(const DashboardViewModel& viewModel)
{
    const ProjectSummaryViewModel& summary = viewModel.summary;
    const bool ready = viewModel.projectOpen && summary.available;
    projectValueLabel_->setText(ready && !viewModel.projectName.isEmpty()
        ? viewModel.projectName : uiText("未打开"));
    projectStatusLabel_->setText(ready
        ? uiText("当前项目：%1（工作区已就绪）").arg(
            viewModel.projectName.isEmpty() ? uiText("已打开")
                                            : viewModel.projectName)
        : uiText("未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    gpuStatusLabel_->setText(viewModel.gpuStatus.isEmpty()
        ? uiText("GPU / 运行时：未执行环境自检")
        : viewModel.gpuStatus);
    datasetValueLabel_->setText(ready
        ? QStringLiteral("%1 / %2")
            .arg(summary.datasetSnapshotCount).arg(summary.datasetCount)
        : QStringLiteral("0"));
    datasetValueLabel_->setToolTip(uiText("快照 / 数据集；版本 %1")
        .arg(summary.datasetVersionCount));
    taskValueLabel_->setText(QString::number(summary.taskCount));
    taskValueLabel_->setToolTip(
        uiText("活动任务 %1；成功 %2；失败 %3；取消 %4")
            .arg(summary.activeTaskCount).arg(summary.succeededTaskCount)
            .arg(summary.failedTaskCount).arg(summary.canceledTaskCount));

    QString nextStep;
    if (!ready) {
        nextStep = uiText("先创建或打开一个本地项目。项目目录会集中保存数据集索引、任务历史、训练报告和模型产物。");
    } else if (summary.datasetSnapshotCount == 0) {
        nextStep = uiText("下一步：导入数据并创建数据集快照。训练工作流只消费已登记的不可变快照。");
    } else if (summary.taskCount == 0) {
        nextStep = uiText("下一步：进入训练实验，选择已登记的数据集快照并启动官方后端工作流。");
    } else if (summary.modelPackageCount == 0) {
        nextStep = uiText("下一步：在任务与产物中检查工作流产物，并完成模型包登记后进入部署验证。");
    } else {
        nextStep = uiText("项目已记录数据集快照、任务与模型包。可继续进入部署验证或追加实验。");
    }
    nextStepLabel_->setText(nextStep);
    nextStepLabel_->setToolTip(uiText("内置能力 %1；环境：%2")
        .arg(viewModel.capabilityCount).arg(viewModel.environmentStatus));

    recentTasksTable_->setRowCount(0);
    for (const TaskListItem& task : viewModel.recentTasks) {
        const int row = recentTasksTable_->rowCount();
        recentTasksTable_->insertRow(row);
        recentTasksTable_->setItem(row, 0,
            new QTableWidgetItem(task.taskId.left(8)));
        recentTasksTable_->setItem(row, 1,
            new QTableWidgetItem(task.capabilityId));
        recentTasksTable_->setItem(row, 2,
            new QTableWidgetItem(task.taskType));
        recentTasksTable_->setItem(row, 3,
            new QTableWidgetItem(task.stateLabel));
        recentTasksTable_->setItem(row, 4,
            new QTableWidgetItem(task.message));
    }
}
