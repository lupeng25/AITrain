#include "MainWindow.h"

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
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildSystemSettingsPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("SYSTEM SETTINGS"),
        uiText("系统设置"),
        uiText("管理内置能力、界面语言、默认目录、授权状态和本地路径。"),
        nullptr,
        QStringList()
            << uiText("内置能力")
            << uiText("偏好设置")
            << uiText("本地路径")));

    systemSettingsTabs_ = new QTabWidget;
    systemSettingsTabs_->setObjectName(QStringLiteral("SystemSettingsTabs"));
    systemSettingsTabs_->addTab(buildCapabilitiesPanel(), uiText("内置能力"));
    systemSettingsTabs_->addTab(buildApplicationSettingsPanel(), uiText("应用设置"));
    layout->addWidget(systemSettingsTabs_, 1);
    updateCapabilitySummary();
    updateSettingsSummary();
    return page;
}

QWidget* MainWindow::buildCapabilitiesPanel()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* refreshButton = primaryButton(QStringLiteral("刷新能力摘要"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshBuiltInCapabilities);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("WorkspaceToolbar"));
    headerPanel->setMaximumHeight(52);
    auto* headerRoot = new QHBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 10, 14, 10);
    headerRoot->setSpacing(12);
    auto* contextLabel = new QLabel(QStringLiteral("编译期内置注册表"));
    contextLabel->setObjectName(QStringLiteral("WorkspaceToolbarTitle"));
    capabilityConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("等待读取内置能力注册表。"));
    capabilityConsoleStatusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus"));
    capabilitySourceLabel_ = inlineStatusLabel(QStringLiteral("能力来源：编译期内置注册表"));
    capabilitySourceLabel_->setObjectName(QStringLiteral("WorkspaceToolbarMeta"));
    allowLabelToShrink(capabilityConsoleStatusLabel_);
    allowLabelToShrink(capabilitySourceLabel_);
    headerRoot->addWidget(contextLabel);
    headerRoot->addWidget(capabilityConsoleStatusLabel_);
    headerRoot->addWidget(capabilitySourceLabel_, 1);
    headerRoot->addWidget(refreshButton);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* capabilityCountCard = createMetricCard(QStringLiteral("内置能力"), QStringLiteral("0"), QStringLiteral("编译期注册"));
    capabilityCountSummaryLabel_ = capabilityCountCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* datasetFormatCard = createMetricCard(QStringLiteral("数据集格式"), QStringLiteral("0"), QStringLiteral("可识别 / 校验格式"));
    capabilityDatasetFormatSummaryLabel_ = datasetFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* exportFormatCard = createMetricCard(QStringLiteral("导出格式"), QStringLiteral("0"), QStringLiteral("能力声明的导出目标"));
    capabilityExportFormatSummaryLabel_ = exportFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* gpuCard = createMetricCard(QStringLiteral("GPU 策略"), QStringLiteral("0"), QStringLiteral("GPU 推荐或必需能力"));
    capabilityGpuSummaryLabel_ = gpuCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    summaryLayout->addWidget(capabilityCountCard, 0, 0);
    summaryLayout->addWidget(datasetFormatCard, 0, 1);
    summaryLayout->addWidget(exportFormatCard, 0, 2);
    summaryLayout->addWidget(gpuCard, 0, 3);

    auto* tablePanel = new InfoPanel(uiText("内置能力"));
    capabilityTable_ = new QTableWidget(0, 7);
    capabilityTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID")
        << QStringLiteral("名称")
        << QStringLiteral("来源")
        << QStringLiteral("任务")
        << QStringLiteral("数据集")
        << QStringLiteral("后端")
        << QStringLiteral("运行策略"));
    configureTable(capabilityTable_);
    capabilityTable_->setWordWrap(true);
    capabilityTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    capabilityTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    capabilityTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    capabilityTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    capabilityTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    capabilityTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
    capabilityTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    capabilityTable_->verticalHeader()->setDefaultSectionSize(42);

    tablePanel->bodyLayout()->addWidget(capabilityTable_);

    layout->addWidget(headerPanel);
    layout->addWidget(summaryStrip);
    layout->addWidget(tablePanel, 1);
    updateCapabilitySummary();
    return page;
}

QWidget* MainWindow::buildEnvironmentPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    auto* runButton = primaryButton(QStringLiteral("执行环境自检"));
    connect(runButton, &QPushButton::clicked, this, &MainWindow::runEnvironmentCheck);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("WorkspaceToolbar"));
    auto* headerRoot = new QHBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 10, 14, 10);
    headerRoot->setSpacing(12);
    auto* contextLabel = new QLabel(QStringLiteral("运行时与交付证据"));
    contextLabel->setObjectName(QStringLiteral("WorkspaceToolbarTitle"));
    environmentConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("尚未执行环境自检。"));
    environmentConsoleStatusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus"));
    allowLabelToShrink(environmentConsoleStatusLabel_);
    headerRoot->addWidget(contextLabel);
    headerRoot->addWidget(environmentConsoleStatusLabel_, 1);
    headerRoot->addWidget(runButton);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* okCard = createMetricCard(QStringLiteral("通过"), QStringLiteral("0"), QStringLiteral("可用依赖"));
    environmentOkSummaryLabel_ = okCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* warningCard = createMetricCard(QStringLiteral("警告"), QStringLiteral("0"), QStringLiteral("可继续但需关注"));
    environmentWarningSummaryLabel_ = warningCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* missingCard = createMetricCard(QStringLiteral("缺失"), QStringLiteral("0"), QStringLiteral("会阻塞相关能力"));
    environmentMissingSummaryLabel_ = missingCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* uncheckedCard = createMetricCard(QStringLiteral("未检测"), QStringLiteral("0"), QStringLiteral("等待 Worker 自检"));
    environmentUncheckedSummaryLabel_ = uncheckedCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    summaryLayout->addWidget(okCard, 0, 0);
    summaryLayout->addWidget(warningCard, 0, 1);
    summaryLayout->addWidget(missingCard, 0, 2);
    summaryLayout->addWidget(uncheckedCard, 0, 3);

    auto* panel = new InfoPanel(QStringLiteral("检查明细"));
    environmentTable_ = new QTableWidget(0, 3);
    environmentTable_->setObjectName(QStringLiteral("EnvironmentTable"));
    environmentTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("检查项") << QStringLiteral("状态") << QStringLiteral("说明"));
    configureTable(environmentTable_);
    environmentTable_->setWordWrap(true);
    environmentTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    environmentTable_->verticalHeader()->setDefaultSectionSize(42);
    refreshEnvironmentReportView();
    panel->bodyLayout()->addWidget(mutedLabel(uiText("状态来自 Worker 已提交的 Environment Check 报告；未提交外部硬件验收证据不会被视为通过。")));
    panel->bodyLayout()->addWidget(environmentTable_);
    auto* runtimeTab = new QWidget;
    auto* runtimeLayout = new QVBoxLayout(runtimeTab);
    runtimeLayout->setContentsMargins(0, 0, 0, 0);
    runtimeLayout->setSpacing(16);
    runtimeLayout->addWidget(summaryStrip);
    runtimeLayout->addWidget(panel, 1);

    auto* tabs = new QTabWidget;
    tabs->setObjectName(QStringLiteral("EnvironmentTabs"));
    tabs->addTab(runtimeTab, uiText("运行环境"));
    tabs->addTab(buildDeliveryEvidencePanel(), uiText("交付证据"));

    layout->addWidget(headerPanel);
    layout->addWidget(tabs, 1);
    updateEnvironmentSummary();
    return page;
}

QWidget* MainWindow::buildApplicationSettingsPanel()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* languagePanel = new InfoPanel(uiText("界面语言"));
    auto* languageRow = new QFrame;
    languageRow->setObjectName(QStringLiteral("ActionStrip"));
    auto* languageLayout = new QHBoxLayout(languageRow);
    languageLayout->setContentsMargins(10, 8, 10, 8);
    languageLayout->setSpacing(10);
    auto* languageHint = mutedLabel(uiText("保存后需要重启 AITrain Studio 才会完全切换界面语言。"));
    allowLabelToShrink(languageHint);
    languageLayout->addWidget(languageHint, 1);
    auto* languageSwitch = new QFrame;
    languageSwitch->setObjectName(QStringLiteral("LanguageSwitch"));
    auto* switchLayout = new QHBoxLayout(languageSwitch);
    switchLayout->setContentsMargins(2, 2, 2, 2);
    switchLayout->setSpacing(0);
    settingsZhLanguageButton_ = new QToolButton;
    settingsZhLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    settingsZhLanguageButton_->setText(QStringLiteral("中"));
    settingsZhLanguageButton_->setCheckable(true);
    settingsZhLanguageButton_->setCursor(Qt::PointingHandCursor);
    settingsZhLanguageButton_->setToolTip(uiText("切换到中文，重启后生效"));
    settingsEnLanguageButton_ = new QToolButton;
    settingsEnLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    settingsEnLanguageButton_->setText(QStringLiteral("EN"));
    settingsEnLanguageButton_->setCheckable(true);
    settingsEnLanguageButton_->setCursor(Qt::PointingHandCursor);
    settingsEnLanguageButton_->setToolTip(uiText("切换到英文，重启后生效"));
    switchLayout->addWidget(settingsZhLanguageButton_);
    switchLayout->addWidget(settingsEnLanguageButton_);
    languageLayout->addWidget(languageSwitch);
    connect(settingsZhLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("zh_CN"));
    });
    connect(settingsEnLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("en_US"));
    });
    languagePanel->bodyLayout()->addWidget(languageRow);

    auto* projectPathPanel = new InfoPanel(uiText("默认项目目录"));
    auto* projectPathForm = new QFormLayout;
    projectPathForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    projectPathForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    projectPathForm->setHorizontalSpacing(14);
    projectPathForm->setVerticalSpacing(10);
    settingsDefaultProjectPathEdit_ = new QLineEdit(QDir::toNativeSeparators(configuredDefaultProjectPath()));
    settingsDefaultProjectPathEdit_->setPlaceholderText(QDir::toNativeSeparators(defaultProjectPath()));
    auto* browseDefaultProjectButton = new QPushButton(uiText("选择目录"));
    auto* defaultPathRow = new QWidget;
    auto* defaultPathLayout = new QHBoxLayout(defaultPathRow);
    defaultPathLayout->setContentsMargins(0, 0, 0, 0);
    defaultPathLayout->setSpacing(8);
    defaultPathLayout->addWidget(settingsDefaultProjectPathEdit_, 1);
    defaultPathLayout->addWidget(browseDefaultProjectButton);
    projectPathForm->addRow(uiText("默认项目目录"), defaultPathRow);
    projectPathPanel->bodyLayout()->addLayout(projectPathForm);
    auto* defaultPathActions = new QFrame;
    defaultPathActions->setObjectName(QStringLiteral("ActionStrip"));
    auto* defaultPathActionLayout = new QHBoxLayout(defaultPathActions);
    defaultPathActionLayout->setContentsMargins(10, 8, 10, 8);
    defaultPathActionLayout->setSpacing(10);
    settingsDefaultProjectPathStatusLabel_ = mutedLabel(uiText("默认项目目录会作为项目页的初始路径；不会自动打开或迁移现有项目。"));
    allowLabelToShrink(settingsDefaultProjectPathStatusLabel_);
    auto* saveDefaultPathButton = primaryButton(uiText("保存默认目录"));
    auto* resetDefaultPathButton = new QPushButton(uiText("恢复默认"));
    defaultPathActionLayout->addWidget(settingsDefaultProjectPathStatusLabel_, 1);
    defaultPathActionLayout->addWidget(resetDefaultPathButton);
    defaultPathActionLayout->addWidget(saveDefaultPathButton);
    projectPathPanel->bodyLayout()->addWidget(defaultPathActions);
    connect(browseDefaultProjectButton, &QPushButton::clicked, this, [this]() {
        const QString directory = QFileDialog::getExistingDirectory(
            this,
            uiText("请选择默认项目目录"),
            QDir::fromNativeSeparators(settingsDefaultProjectPathEdit_->text().trimmed()));
        if (!directory.isEmpty()) {
            settingsDefaultProjectPathEdit_->setText(QDir::toNativeSeparators(directory));
        }
    });
    connect(saveDefaultPathButton, &QPushButton::clicked, this, [this]() {
        storeDefaultProjectPathPreference(settingsDefaultProjectPathEdit_->text());
    });
    connect(resetDefaultPathButton, &QPushButton::clicked, this, [this]() {
        settingsDefaultProjectPathEdit_->setText(QDir::toNativeSeparators(defaultProjectPath()));
        storeDefaultProjectPathPreference(defaultProjectPath());
        if (settingsDefaultProjectPathStatusLabel_) {
            settingsDefaultProjectPathStatusLabel_->setText(uiText("默认项目目录已恢复。"));
        }
    });

    auto* licensePanel = new InfoPanel(uiText("授权状态"));
    auto* licenseForm = new QFormLayout;
    licenseForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    licenseForm->setHorizontalSpacing(14);
    licenseForm->setVerticalSpacing(8);
    licenseForm->addRow(uiText("授权用户"), new QLabel(licenseOwner_.isEmpty() ? uiText("已注册") : licenseOwner_));
    licenseForm->addRow(uiText("有效期"), new QLabel(licenseExpiry_.isEmpty() ? uiText("未记录") : licenseExpiry_));
    licensePanel->bodyLayout()->addLayout(licenseForm);
    auto* licenseHint = mutedLabel(uiText("离线授权已在启动时校验；这里只展示当前授权信息，不提供换绑或激活入口。"));
    allowLabelToShrink(licenseHint);
    licensePanel->bodyLayout()->addWidget(licenseHint);

    auto* entryPanel = new InfoPanel(uiText("系统入口"));
    auto* entryActions = new QFrame;
    entryActions->setObjectName(QStringLiteral("ActionStrip"));
    auto* entryLayout = new QGridLayout(entryActions);
    entryLayout->setContentsMargins(10, 8, 10, 8);
    entryLayout->setHorizontalSpacing(10);
    entryLayout->setVerticalSpacing(10);
    auto* openProjectButton = primaryButton(uiText("打开项目页"));
    auto* openCapabilitiesButton = new QPushButton(uiText("打开内置能力设置"));
    auto* openEnvironmentButton = new QPushButton(uiText("打开环境页"));
    auto* runEnvironmentButton = new QPushButton(uiText("执行环境自检"));
    connect(openProjectButton, &QPushButton::clicked, this, [this]() { showPage(ProjectPage, uiText("项目")); });
    connect(openCapabilitiesButton, &QPushButton::clicked, this, [this]() { showSystemSettingsTab(0); });
    connect(openEnvironmentButton, &QPushButton::clicked, this, [this]() { showPage(EnvironmentPage, uiText("环境")); });
    connect(runEnvironmentButton, &QPushButton::clicked, this, [this]() {
        showPage(EnvironmentPage, uiText("环境"));
        runEnvironmentCheck();
    });
    entryLayout->addWidget(openProjectButton, 0, 0);
    entryLayout->addWidget(openCapabilitiesButton, 0, 1);
    entryLayout->addWidget(openEnvironmentButton, 1, 0);
    entryLayout->addWidget(runEnvironmentButton, 1, 1);
    entryPanel->bodyLayout()->addWidget(entryActions);
    auto* entryHint = mutedLabel(uiText("环境自检仍通过 Worker 执行，当前有任务运行时会沿用现有 busy guard。"));
    allowLabelToShrink(entryHint);
    entryPanel->bodyLayout()->addWidget(entryHint);

    auto* pathsPanel = new InfoPanel(uiText("本地路径"));
    auto* pathsGrid = new QGridLayout;
    pathsGrid->setHorizontalSpacing(10);
    pathsGrid->setVerticalSpacing(8);
    const auto addStaticPathRow = [this, pathsGrid](int row, const QString& labelText, const QString& path) {
        auto* label = new QLabel(labelText);
        auto* edit = new QLineEdit(QDir::toNativeSeparators(path));
        edit->setReadOnly(true);
        auto* openButton = new QPushButton(uiText("打开目录"));
        auto* copyButton = new QPushButton(uiText("复制路径"));
        connect(openButton, &QPushButton::clicked, this, [this, path]() { openLocalDirectory(path); });
        connect(copyButton, &QPushButton::clicked, this, [this, path, labelText]() { copyLocalPath(path, labelText); });
        pathsGrid->addWidget(label, row, 0);
        pathsGrid->addWidget(edit, row, 1);
        pathsGrid->addWidget(openButton, row, 2);
        pathsGrid->addWidget(copyButton, row, 3);
    };
    auto* currentProjectLabel = new QLabel(uiText("当前项目目录"));
    settingsCurrentProjectPathLabel_ = inlineStatusLabel(uiText("未打开项目"));
    allowLabelToShrink(settingsCurrentProjectPathLabel_);
    auto* openCurrentProjectButton = new QPushButton(uiText("打开目录"));
    auto* copyCurrentProjectButton = new QPushButton(uiText("复制路径"));
    connect(openCurrentProjectButton, &QPushButton::clicked, this, [this]() { openLocalDirectory(currentProjectPath_); });
    connect(copyCurrentProjectButton, &QPushButton::clicked, this, [this]() { copyLocalPath(currentProjectPath_, uiText("当前项目目录")); });
    pathsGrid->addWidget(currentProjectLabel, 0, 0);
    pathsGrid->addWidget(settingsCurrentProjectPathLabel_, 0, 1);
    pathsGrid->addWidget(openCurrentProjectButton, 0, 2);
    pathsGrid->addWidget(copyCurrentProjectButton, 0, 3);
    pathsGrid->setColumnStretch(1, 1);
    pathsPanel->bodyLayout()->addLayout(pathsGrid);

    layout->addWidget(languagePanel);
    layout->addWidget(projectPathPanel);
    layout->addWidget(licensePanel);
    layout->addWidget(entryPanel);
    layout->addWidget(pathsPanel);
    layout->addStretch();

    page->setWidget(content);
    updateLanguageButtonState();
    updateSettingsSummary();
    return page;
}
