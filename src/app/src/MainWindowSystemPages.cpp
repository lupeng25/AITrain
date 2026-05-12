#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"

#include <QAbstractItemView>
#include <QApplication>
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

QWidget* MainWindow::buildPluginsPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* refreshButton = primaryButton(QStringLiteral("重新扫描插件"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshPlugins);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("PLUGIN CAPABILITY MATRIX"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("插件能力矩阵"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("扫描模型、数据集、导出和推理扩展能力；插件仍只通过公共接口暴露能力。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(refreshButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* scanCaption = new QLabel(QStringLiteral("扫描"));
    scanCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* pathCaption = new QLabel(QStringLiteral("路径"));
    pathCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    pluginConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("等待插件扫描。"));
    pluginConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    pluginSearchPathLabel_ = inlineStatusLabel(QStringLiteral("插件搜索路径：未初始化"));
    pluginSearchPathLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    pluginMarketplaceStatusLabel_ = inlineStatusLabel(QStringLiteral("插件市场：等待加载本地索引。"));
    pluginMarketplaceStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(pluginConsoleStatusLabel_);
    allowLabelToShrink(pluginSearchPathLabel_);
    allowLabelToShrink(pluginMarketplaceStatusLabel_);
    headerGrid->addWidget(scanCaption, 0, 0);
    headerGrid->addWidget(pluginConsoleStatusLabel_, 0, 1);
    headerGrid->addWidget(pathCaption, 1, 0);
    headerGrid->addWidget(pluginSearchPathLabel_, 1, 1);
    headerGrid->addWidget(new QLabel(QStringLiteral("市场")), 2, 0);
    headerGrid->addWidget(pluginMarketplaceStatusLabel_, 2, 1);
    headerRoot->addLayout(headerGrid);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* pluginCountCard = createMetricCard(QStringLiteral("已加载插件"), QStringLiteral("0"), QStringLiteral("manifest 已加载"));
    pluginCountSummaryLabel_ = pluginCountCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* datasetFormatCard = createMetricCard(QStringLiteral("数据集格式"), QStringLiteral("0"), QStringLiteral("可识别 / 校验格式"));
    pluginDatasetFormatSummaryLabel_ = datasetFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* exportFormatCard = createMetricCard(QStringLiteral("导出格式"), QStringLiteral("0"), QStringLiteral("插件声明的导出目标"));
    pluginExportFormatSummaryLabel_ = exportFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* gpuCard = createMetricCard(QStringLiteral("GPU 需求"), QStringLiteral("0"), QStringLiteral("声明需要 GPU 的插件"));
    pluginGpuSummaryLabel_ = gpuCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    summaryLayout->addWidget(pluginCountCard, 0, 0);
    summaryLayout->addWidget(datasetFormatCard, 0, 1);
    summaryLayout->addWidget(exportFormatCard, 0, 2);
    summaryLayout->addWidget(gpuCard, 0, 3);

    auto* tablePanel = new InfoPanel(QStringLiteral("插件扩展"));
    pluginTable_ = new QTableWidget(0, 7);
    pluginTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID")
        << QStringLiteral("名称")
        << QStringLiteral("版本")
        << QStringLiteral("任务")
        << QStringLiteral("数据集")
        << QStringLiteral("导出")
        << QStringLiteral("GPU"));
    configureTable(pluginTable_);
    pluginTable_->setWordWrap(true);
    pluginTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    pluginTable_->verticalHeader()->setDefaultSectionSize(42);

    const QString appDir = QApplication::applicationDirPath();
    pluginMarketplaceWidget_ = new PluginMarketplaceWidget(
        QDir(appDir).filePath(QStringLiteral("plugins/marketplace")),
        QDir(appDir).filePath(QStringLiteral("plugins/models")));
    connect(pluginMarketplaceWidget_, &PluginMarketplaceWidget::releasePluginLoadersRequested, this, [this](const QStringList& activeFiles) {
        pluginManager_.releasePluginFiles(activeFiles);
    });
    connect(pluginMarketplaceWidget_, &PluginMarketplaceWidget::pluginsChanged, this, &MainWindow::refreshPlugins);
    connect(pluginMarketplaceWidget_, &PluginMarketplaceWidget::statusChanged, this, [this](const QString& status) {
        if (pluginMarketplaceStatusLabel_) {
            pluginMarketplaceStatusLabel_->setText(compactTextForStatus(status, 108));
            pluginMarketplaceStatusLabel_->setToolTip(status);
        }
    });
    pluginMarketplaceWidget_->refreshInstalledPlugins();

    auto* tabs = new QTabWidget;
    auto* loadedTab = new QWidget;
    auto* loadedLayout = new QVBoxLayout(loadedTab);
    loadedLayout->setContentsMargins(0, 0, 0, 0);
    loadedLayout->addWidget(pluginTable_);

    tabs->addTab(loadedTab, QStringLiteral("已加载"));
    tabs->addTab(pluginMarketplaceWidget_, QStringLiteral("插件市场"));
    tablePanel->bodyLayout()->addWidget(tabs);

    layout->addWidget(headerPanel);
    layout->addWidget(summaryStrip);
    layout->addWidget(tablePanel, 1);
    updatePluginSummary();
    return page;
}

QWidget* MainWindow::buildEnvironmentPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* runButton = primaryButton(QStringLiteral("执行环境自检"));
    connect(runButton, &QPushButton::clicked, this, &MainWindow::runEnvironmentCheck);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("RUNTIME HEALTH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("运行时健康面板"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("检查 NVIDIA 驱动、CUDA、TensorRT、ONNX Runtime、Qt 插件和 Worker 可用性。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(runButton);
    headerRoot->addLayout(headerTop);

    environmentConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("尚未执行环境自检。"));
    environmentConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(environmentConsoleStatusLabel_);
    headerRoot->addWidget(environmentConsoleStatusLabel_);

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
    environmentTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("检查项") << QStringLiteral("状态") << QStringLiteral("说明"));
    configureTable(environmentTable_);
    environmentTable_->setWordWrap(true);
    environmentTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    environmentTable_->verticalHeader()->setDefaultSectionSize(42);
    const QStringList rows = {
        QStringLiteral("NVIDIA Driver"),
        QStringLiteral("CUDA Runtime"),
        QStringLiteral("cuDNN"),
        QStringLiteral("TensorRT"),
        QStringLiteral("ONNX Runtime"),
        QStringLiteral("LibTorch"),
        QStringLiteral("Qt Plugins"),
        QStringLiteral("AITrain Plugins"),
        QStringLiteral("Worker")
    };
    for (const QString& rowName : rows) {
        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        environmentTable_->setItem(row, 0, new QTableWidgetItem(rowName));
        environmentTable_->setItem(row, 1, new QTableWidgetItem(uiText("未检测")));
        environmentTable_->setItem(row, 2, new QTableWidgetItem(uiText("点击执行环境自检。")));
    }
    panel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("TensorRT 真机验收仍需要 RTX / SM 75+。当前 GTX 1060 / SM 61 只能记录为 hardware-blocked。")));
    panel->bodyLayout()->addWidget(environmentTable_);
    layout->addWidget(headerPanel);
    layout->addWidget(summaryStrip);
    layout->addWidget(panel, 1);
    updateEnvironmentSummary();
    return page;
}

QWidget* MainWindow::buildSettingsPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("APPLICATION SETTINGS"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(uiText("系统设置"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(uiText("集中管理轻量级应用偏好、授权状态和本地系统入口；训练、导出和推理仍由各自页面和 Worker 执行。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerRoot->addLayout(headerTop);

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
    auto* openPluginsButton = new QPushButton(uiText("打开插件页"));
    auto* openEnvironmentButton = new QPushButton(uiText("打开环境页"));
    auto* runEnvironmentButton = new QPushButton(uiText("执行环境自检"));
    connect(openProjectButton, &QPushButton::clicked, this, [this]() { showPage(ProjectPage, uiText("项目")); });
    connect(openPluginsButton, &QPushButton::clicked, this, [this]() { showPage(PluginsPage, uiText("插件")); });
    connect(openEnvironmentButton, &QPushButton::clicked, this, [this]() { showPage(EnvironmentPage, uiText("环境")); });
    connect(runEnvironmentButton, &QPushButton::clicked, this, [this]() {
        showPage(EnvironmentPage, uiText("环境"));
        runEnvironmentCheck();
    });
    entryLayout->addWidget(openProjectButton, 0, 0);
    entryLayout->addWidget(openPluginsButton, 0, 1);
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
    const QString appDir = QApplication::applicationDirPath();
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
    addStaticPathRow(0, uiText("插件目录"), QDir(appDir).filePath(QStringLiteral("plugins/models")));
    addStaticPathRow(1, uiText("插件市场目录"), QDir(appDir).filePath(QStringLiteral("plugins/marketplace")));
    auto* currentProjectLabel = new QLabel(uiText("当前项目目录"));
    settingsCurrentProjectPathLabel_ = inlineStatusLabel(uiText("未打开项目"));
    allowLabelToShrink(settingsCurrentProjectPathLabel_);
    auto* openCurrentProjectButton = new QPushButton(uiText("打开目录"));
    auto* copyCurrentProjectButton = new QPushButton(uiText("复制路径"));
    connect(openCurrentProjectButton, &QPushButton::clicked, this, [this]() { openLocalDirectory(currentProjectPath_); });
    connect(copyCurrentProjectButton, &QPushButton::clicked, this, [this]() { copyLocalPath(currentProjectPath_, uiText("当前项目目录")); });
    pathsGrid->addWidget(currentProjectLabel, 2, 0);
    pathsGrid->addWidget(settingsCurrentProjectPathLabel_, 2, 1);
    pathsGrid->addWidget(openCurrentProjectButton, 2, 2);
    pathsGrid->addWidget(copyCurrentProjectButton, 2, 3);
    pathsGrid->setColumnStretch(1, 1);
    pathsPanel->bodyLayout()->addLayout(pathsGrid);

    layout->addWidget(headerPanel);
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
