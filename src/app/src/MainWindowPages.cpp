#include "MainWindow.h"
#include "SettingsPageController.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"

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
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QStyle>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildTopBar()
{
    auto* topBar = new QFrame;
    topBar->setObjectName(QStringLiteral("TopBar"));
    topBar->setFixedHeight(56);

    auto* layout = new QHBoxLayout(topBar);
    layout->setContentsMargins(18, 7, 14, 7);
    layout->setSpacing(10);

    auto* projectContext = new QWidget;
    auto* projectLayout = new QHBoxLayout(projectContext);
    projectLayout->setContentsMargins(0, 0, 0, 0);
    projectLayout->setSpacing(9);
    auto* projectCaption = new QLabel(uiText("项目"));
    projectCaption->setObjectName(QStringLiteral("TopbarCaption"));
    headerProjectLabel_ = new QLabel(uiText("未打开项目"));
    headerProjectLabel_->setObjectName(QStringLiteral("TopbarProject"));
    projectLayout->addWidget(projectCaption);
    projectLayout->addWidget(headerProjectLabel_);
    projectContext->setMaximumWidth(320);

    workerPill_ = new StatusPill;
    workerPill_->setStatus(tr("Worker 空闲"), StatusPill::Tone::Neutral);
    capabilityPill_ = new StatusPill;
    gpuPill_ = new StatusPill;
    gpuPill_->setStatus(tr("GPU 未检测"), StatusPill::Tone::Warning);
    licensePill_ = new StatusPill;
    licensePill_->setStatus(licenseOwner_.isEmpty()
            ? tr("已注册")
            : tr("授权：%1").arg(licenseOwner_),
        StatusPill::Tone::Success);
    licensePill_->setToolTip(licenseExpiry_.isEmpty()
            ? tr("离线授权已验证")
            : tr("授权有效期：%1").arg(licenseExpiry_));
    auto* languageSwitch = new QFrame;
    languageSwitch->setObjectName(QStringLiteral("LanguageSwitch"));
    auto* languageLayout = new QHBoxLayout(languageSwitch);
    languageLayout->setContentsMargins(2, 2, 2, 2);
    languageLayout->setSpacing(0);
    topBarZhLanguageButton_ = new QToolButton;
    topBarZhLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    topBarZhLanguageButton_->setText(QStringLiteral("中"));
    topBarZhLanguageButton_->setCheckable(true);
    topBarZhLanguageButton_->setCursor(Qt::PointingHandCursor);
    topBarZhLanguageButton_->setToolTip(uiText("切换到中文，重启后生效"));
    topBarEnLanguageButton_ = new QToolButton;
    topBarEnLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    topBarEnLanguageButton_->setText(QStringLiteral("EN"));
    topBarEnLanguageButton_->setCheckable(true);
    topBarEnLanguageButton_->setCursor(Qt::PointingHandCursor);
    topBarEnLanguageButton_->setToolTip(uiText("切换到英文，重启后生效"));
    languageLayout->addWidget(topBarZhLanguageButton_);
    languageLayout->addWidget(topBarEnLanguageButton_);
    connect(topBarZhLanguageButton_, &QToolButton::clicked, this, [this]() {
        settingsPageController_->setLanguageCode(QStringLiteral("zh_CN"));
    });
    connect(topBarEnLanguageButton_, &QToolButton::clicked, this, [this]() {
        settingsPageController_->setLanguageCode(QStringLiteral("en_US"));
    });

    inspectorToggleButton_ = new QToolButton;
    inspectorToggleButton_->setObjectName(QStringLiteral("InspectorToggle"));
    inspectorToggleButton_->setIcon(style()->standardIcon(QStyle::SP_FileDialogDetailedView));
    inspectorToggleButton_->setToolTip(uiText("显示或隐藏检查器"));
    inspectorToggleButton_->setCheckable(true);
    inspectorToggleButton_->setChecked(true);
    connect(inspectorToggleButton_, &QToolButton::toggled, this, [this](bool checked) {
        if (!applyingResponsiveChrome_) {
            inspectorUserOverride_ = true;
        }
        if (inspector_) inspector_->setVisible(checked);
    });

    layout->addWidget(projectContext);
    layout->addStretch(1);
    layout->addWidget(workerPill_);
    layout->addWidget(gpuPill_);
    layout->addWidget(languageSwitch);
    layout->addWidget(inspectorToggleButton_);
    return topBar;
}

QWidget* MainWindow::buildPageHeading()
{
    auto* heading = new QFrame;
    heading->setObjectName(QStringLiteral("PageHeading"));
    auto* layout = new QHBoxLayout(heading);
    layout->setContentsMargins(18, 12, 18, 8);
    layout->setSpacing(10);
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(1);
    pageCaption_ = new QLabel;
    pageCaption_->setObjectName(QStringLiteral("PageEyebrow"));
    pageTitle_ = new QLabel(tr("总览"));
    pageTitle_->setObjectName(QStringLiteral("PageTitle"));
    titleLayout->addWidget(pageCaption_);
    titleLayout->addWidget(pageTitle_);
    pageContextPill_ = new StatusPill;
    pageContextPill_->setStatus(uiText("项目未打开"), StatusPill::Tone::Neutral);
    layout->addWidget(titleBlock, 1);
    layout->addWidget(pageContextPill_, 0, Qt::AlignVCenter);
    return heading;
}

QWidget* MainWindow::buildInspector()
{
    auto* inspector = new QFrame;
    inspector->setObjectName(QStringLiteral("Inspector"));
    inspector->setMinimumWidth(0);
    inspector->setFixedWidth(272);

    auto* layout = new QVBoxLayout(inspector);
    layout->setContentsMargins(14, 14, 14, 12);
    layout->setSpacing(12);

    auto* header = new QWidget;
    auto* headerLayout = new QVBoxLayout(header);
    headerLayout->setContentsMargins(0, 0, 0, 0);
    headerLayout->setSpacing(2);
    auto* title = new QLabel(uiText("检查器"));
    title->setObjectName(QStringLiteral("InspectorTitle"));
    auto* subtitle = new QLabel(uiText("当前工作上下文"));
    subtitle->setObjectName(QStringLiteral("InspectorSubtitle"));
    headerLayout->addWidget(title);
    headerLayout->addWidget(subtitle);
    layout->addWidget(header);

    auto* identity = new QFrame;
    identity->setObjectName(QStringLiteral("InspectorIdentity"));
    auto* identityLayout = new QVBoxLayout(identity);
    identityLayout->setContentsMargins(10, 10, 10, 10);
    identityLayout->setSpacing(4);
    inspectorProjectLabel_ = new QLabel(uiText("未打开项目"));
    inspectorProjectLabel_->setObjectName(QStringLiteral("InspectorProject"));
    inspectorCapabilityLabel_ = new QLabel(uiText("内置能力未加载"));
    inspectorCapabilityLabel_->setObjectName(QStringLiteral("InspectorDetail"));
    inspectorCapabilityLabel_->setWordWrap(true);
    identityLayout->addWidget(inspectorProjectLabel_);
    identityLayout->addWidget(inspectorCapabilityLabel_);
    layout->addWidget(identity);

    auto* resourcePanel = new QFrame;
    resourcePanel->setObjectName(QStringLiteral("InspectorSection"));
    auto* resourceLayout = new QVBoxLayout(resourcePanel);
    resourceLayout->setContentsMargins(10, 10, 10, 10);
    resourceLayout->setSpacing(7);
    auto* resourceTitle = new QLabel(uiText("运行资源"));
    resourceTitle->setObjectName(QStringLiteral("InspectorSectionTitle"));
    inspectorWorkerLabel_ = new QLabel(uiText("Worker：等待连接"));
    inspectorWorkerLabel_->setObjectName(QStringLiteral("InspectorDetail"));
    inspectorWorkerLabel_->setWordWrap(true);
    inspectorGpuLabel_ = new QLabel(uiText("GPU：等待环境检查"));
    inspectorGpuLabel_->setObjectName(QStringLiteral("InspectorDetail"));
    inspectorGpuLabel_->setWordWrap(true);
    resourceLayout->addWidget(resourceTitle);
    resourceLayout->addWidget(inspectorWorkerLabel_);
    resourceLayout->addWidget(inspectorGpuLabel_);
    layout->addWidget(resourcePanel);

    auto* shortcuts = new QFrame;
    shortcuts->setObjectName(QStringLiteral("InspectorSection"));
    auto* shortcutLayout = new QVBoxLayout(shortcuts);
    shortcutLayout->setContentsMargins(10, 10, 10, 10);
    shortcutLayout->setSpacing(3);
    auto* shortcutTitle = new QLabel(uiText("快捷入口"));
    shortcutTitle->setObjectName(QStringLiteral("InspectorSectionTitle"));
    auto* taskButton = new QPushButton(uiText("查看任务与产物"));
    taskButton->setObjectName(QStringLiteral("InspectorShortcut"));
    auto* environmentButton = new QPushButton(uiText("检查运行环境"));
    environmentButton->setObjectName(QStringLiteral("InspectorShortcut"));
    connect(taskButton, &QPushButton::clicked, this, [this]() { showPage(TaskQueuePage, uiText("任务与产物")); });
    connect(environmentButton, &QPushButton::clicked, this, [this]() { showPage(EnvironmentPage, uiText("环境")); });
    shortcutLayout->addWidget(shortcutTitle);
    shortcutLayout->addWidget(taskButton);
    shortcutLayout->addWidget(environmentButton);
    layout->addWidget(shortcuts);
    layout->addStretch(1);

    auto* footer = new QLabel(uiText("本地工作站 · 数据仅保留在本机"));
    footer->setObjectName(QStringLiteral("InspectorFooter"));
    footer->setWordWrap(true);
    layout->addWidget(footer);
    return inspector;
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
    table->setAlternatingRowColors(true);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setStretchLastSection(true);
    table->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    table->setShowGrid(false);
}
