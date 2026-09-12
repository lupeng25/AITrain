#include "WorkbenchTranslation.h"
#include "DashboardPage.h"

#include "InfoPanel.h"
#include "WorkbenchWidgets.h"
#include "WorkbenchLabels.h"
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

DashboardWorkspacePage::DashboardWorkspacePage(QWidget* parent) : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this); layout->setContentsMargins(20, 0, 20, 20); layout->setSpacing(12);
    projectStatusLabel_ = workbenchHint(); projectStatusLabel_->setObjectName(QStringLiteral("ProjectWorkspaceStatus")); layout->addWidget(projectStatusLabel_);
    gpuStatusLabel_ = workbenchHint(); layout->addWidget(gpuStatusLabel_);
    auto* summary = new QHBoxLayout;
    projectValueLabel_ = workbenchHint(); datasetValueLabel_ = workbenchHint(); taskValueLabel_ = workbenchHint();
    datasetValueLabel_->setObjectName(QStringLiteral("DashboardDatasetSummary")); taskValueLabel_->setObjectName(QStringLiteral("DashboardTaskSummary"));
    summary->addWidget(new QLabel(aitrain_app::workbenchText(QStringLiteral("项目")))); summary->addWidget(projectValueLabel_, 1); summary->addWidget(new QLabel(aitrain_app::workbenchText(QStringLiteral("快照 / 数据集")))); summary->addWidget(datasetValueLabel_, 1); summary->addWidget(new QLabel(aitrain_app::workbenchText(QStringLiteral("任务")))); summary->addWidget(taskValueLabel_, 1); layout->addLayout(summary);
    nextStepLabel_ = workbenchHint(); layout->addWidget(nextStepLabel_);
    recentTasksTable_ = workbenchTable({aitrain_app::workbenchText(QStringLiteral("最近任务")), aitrain_app::workbenchText(QStringLiteral("能力")), aitrain_app::workbenchText(QStringLiteral("类型")), aitrain_app::workbenchText(QStringLiteral("状态")), aitrain_app::workbenchText(QStringLiteral("说明"))}); recentTasksTable_->setObjectName(QStringLiteral("DashboardRecentTasks")); recentTasksTable_->setColumnHidden(1, true); recentTasksTable_->setColumnHidden(2, true); layout->addWidget(recentTasksTable_, 1);
    auto* actions = new QHBoxLayout;
    const QVector<QPair<QString, DashboardRoute>> routes{{aitrain_app::workbenchText(QStringLiteral("数据集")), DashboardRoute::Dataset}, {aitrain_app::workbenchText(QStringLiteral("训练")), DashboardRoute::Training}, {aitrain_app::workbenchText(QStringLiteral("模型")), DashboardRoute::ModelRegistry}, {aitrain_app::workbenchText(QStringLiteral("任务记录")), DashboardRoute::TaskArtifact}};
    for (const auto& route : routes) { auto* button = workbenchButton(route.first); actions->addWidget(button); connect(button, &QPushButton::clicked, this, [this, route]() { emit routeRequested(route.second); }); }
    actions->addStretch(); layout->addLayout(actions);
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
        nextStep = uiText("下一步：进入训练，选择已登记的数据集快照并启动官方后端工作流。");
    } else if (summary.modelPackageCount == 0) {
        nextStep = uiText("下一步：在任务记录中检查工作流产物，并完成模型包登记后进入部署验证。");
    } else {
        nextStep = uiText("项目已记录数据集快照、任务与模型包。可继续进入模型的验证与交付，或追加训练。");
    }
    nextStepLabel_->setText(nextStep);
    nextStepLabel_->setToolTip(uiText("内置能力 %1；环境：%2")
        .arg(viewModel.capabilityCount).arg(viewModel.environmentStatus));

    recentTasksTable_->setRowCount(0);
    for (const TaskListItem& task : viewModel.recentTasks) {
        const int row = recentTasksTable_->rowCount();
        recentTasksTable_->insertRow(row);
        recentTasksTable_->setItem(row, 0,
            new QTableWidgetItem(taskDisplayName(task.taskType)));
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
