#include "WorkbenchTranslation.h"
#include "MainWindow.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "WorkbenchWidgets.h"

#include <QAbstractItemView>
#include <QFrame>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QMenu>
#include <QPushButton>
#include <QSizePolicy>
#include <QToolButton>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildTopBar()
{
    auto* bar = new QFrame;
    bar->setObjectName(QStringLiteral("TopBar"));
    bar->setFixedHeight(52);
    auto* layout = new QHBoxLayout(bar);
    layout->setContentsMargins(20, 7, 20, 7);
    layout->setSpacing(10);

    auto* projectButton = new QToolButton;
    projectButton->setObjectName(QStringLiteral("ProjectMenuButton"));
    projectButton->setText(aitrain_app::workbenchText(QStringLiteral("项目")));
    projectButton->setPopupMode(QToolButton::InstantPopup);
    auto* menu = new QMenu(projectButton);
    menu->addAction(aitrain_app::workbenchText(QStringLiteral("打开、新建或管理项目")), this, [this]() {
        showPage(ProjectPage, aitrain_app::workbenchText(QStringLiteral("项目")));
    });
    menu->addAction(aitrain_app::workbenchText(QStringLiteral("项目概况")), this, [this]() {
        showPage(DashboardPage, aitrain_app::workbenchText(QStringLiteral("项目概况")));
        updateDashboardSummary();
    });
    projectButton->setMenu(menu);
    layout->addWidget(projectButton);
    headerProjectLabel_ = new ElidingLabel(aitrain_app::workbenchText(QStringLiteral("未打开项目")));
    headerProjectLabel_->setObjectName(QStringLiteral("TopbarProject"));
    headerProjectLabel_->setMaximumWidth(250);
    headerProjectLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    layout->addWidget(headerProjectLabel_, 1);
    layout->addStretch(1);

    auto* tasks = new QPushButton(aitrain_app::workbenchText(QStringLiteral("任务记录")));
    tasks->setObjectName(QStringLiteral("GlobalTaskButton"));
    connect(tasks, &QPushButton::clicked, this, [this]() {
        showPage(TaskQueuePage, aitrain_app::workbenchText(QStringLiteral("任务记录")));
    });
    layout->addWidget(tasks);
    gpuPill_ = new StatusPill;
    gpuPill_->setStatus(aitrain_app::workbenchText(QStringLiteral("环境待检查")), StatusPill::Tone::Neutral);
    layout->addWidget(gpuPill_);
    auto* environment = new QPushButton(aitrain_app::workbenchText(QStringLiteral("环境")));
    environment->setObjectName(QStringLiteral("GlobalEnvironmentButton"));
    connect(environment, &QPushButton::clicked, this, [this]() {
        showPage(EnvironmentPage, aitrain_app::workbenchText(QStringLiteral("环境与诊断")));
    });
    layout->addWidget(environment);

    // 任务反馈始终可见，长消息由状态栏承载，详细结果在任务记录中定位。
    workerPill_ = new StatusPill;
    workerPill_->setMaximumWidth(320);
    workerPill_->setMinimumWidth(0);
    workerPill_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    workerPill_->setStatus(aitrain_app::workbenchText(QStringLiteral("空闲")), StatusPill::Tone::Neutral);
    return bar;
}

QWidget* MainWindow::buildPageHeading()
{
    auto* heading = new QFrame;
    heading->setObjectName(QStringLiteral("PageHeading"));
    auto* layout = new QHBoxLayout(heading);
    layout->setContentsMargins(20, 14, 20, 10);
    layout->setSpacing(12);
    returnToWorkspaceButton_ = new QPushButton(aitrain_app::workbenchText(QStringLiteral("返回工作区")));
    returnToWorkspaceButton_->setObjectName(QStringLiteral("ReturnToWorkspaceButton"));
    returnToWorkspaceButton_->hide();
    connect(returnToWorkspaceButton_, &QPushButton::clicked, this, [this]() {
        const QString title = lastWorkspacePage_ == TrainingPage ? aitrain_app::workbenchText(QStringLiteral("训练"))
            : lastWorkspacePage_ == ModelRegistryPage ? aitrain_app::workbenchText(QStringLiteral("模型")) : aitrain_app::workbenchText(QStringLiteral("数据集"));
        showPage(lastWorkspacePage_, title);
    });
    layout->addWidget(returnToWorkspaceButton_);
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    pageCaption_ = new ElidingLabel;
    pageCaption_->setMinimumWidth(0);
    pageCaption_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    pageCaption_->setObjectName(QStringLiteral("PageEyebrow"));
    pageTitle_ = new QLabel(aitrain_app::workbenchText(QStringLiteral("项目")));
    pageTitle_->setObjectName(QStringLiteral("PageTitle"));
    titleLayout->addWidget(pageCaption_);
    titleLayout->addWidget(pageTitle_);
    layout->addWidget(titleBlock, 1);
    return heading;
}

InfoPanel* MainWindow::createMetricCard(const QString& label, const QString& value, const QString& caption)
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

void MainWindow::configureTable(QTableWidget* table) const
{
    table->setAlternatingRowColors(false);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->verticalHeader()->setVisible(false);
    table->verticalHeader()->setDefaultSectionSize(40);
    table->horizontalHeader()->setStretchLastSection(true);
    table->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    table->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    table->setShowGrid(false);
}
