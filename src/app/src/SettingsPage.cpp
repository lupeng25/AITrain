#include "WorkbenchTranslation.h"
#include "SettingsPage.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"

#include <QAbstractItemView>
#include <QDir>
#include <QComboBox>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QListWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QToolButton>
#include <QVBoxLayout>

using namespace aitrain_app;

namespace {

void configureReadOnlyTable(QTableWidget* table)
{
    table->setAlternatingRowColors(true);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->verticalHeader()->setVisible(false);
}

QLabel* metricValue(InfoPanel* card)
{
    return card->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
}

} // namespace

SettingsWorkspacePage::SettingsWorkspacePage(const QString& licenseOwner,
    const QString& licenseExpiry, QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);
    auto* body = new QHBoxLayout;
    auto* categories = new QListWidget; categories->addItems({aitrain_app::workbenchText(QStringLiteral("内置能力")), aitrain_app::workbenchText(QStringLiteral("应用偏好与许可"))}); categories->setFixedWidth(150);
    categories->setWordWrap(true);
    categories->setResizeMode(QListView::Adjust);
    categories->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    for (int row = 0; row < categories->count(); ++row) categories->item(row)->setToolTip(categories->item(row)->text());
    tabs_ = new QStackedWidget; tabs_->setObjectName(QStringLiteral("SettingsViews"));
    tabs_->addWidget(buildCapabilitiesPanel()); tabs_->addWidget(buildApplicationSettingsPanel(licenseOwner, licenseExpiry));
    body->addWidget(categories); body->addWidget(tabs_, 1); layout->addLayout(body, 1);
    connect(categories, &QListWidget::currentRowChanged, tabs_, &QStackedWidget::setCurrentIndex);
    connect(tabs_, &QStackedWidget::currentChanged, categories, QOverload<int>::of(&QListWidget::setCurrentRow));
    categories->setCurrentRow(1);

}

QWidget* SettingsWorkspacePage::buildCapabilitiesPanel()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* refreshButton = primaryButton(tr("刷新能力摘要"));
    connect(refreshButton, &QPushButton::clicked,
        this, &SettingsWorkspacePage::refreshCapabilitiesRequested);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("WorkspaceToolbar"));
    headerPanel->setMaximumHeight(52);
    auto* headerLayout = new QHBoxLayout(headerPanel);
    headerLayout->setContentsMargins(14, 10, 14, 10);
    headerLayout->setSpacing(12);
    auto* contextLabel = new QLabel(tr("编译期内置注册表"));
    contextLabel->setObjectName(QStringLiteral("WorkspaceToolbarTitle"));
    capabilityStatusLabel_ = inlineStatusLabel(tr("等待读取内置能力注册表。"));
    capabilityStatusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus"));
    capabilitySourceLabel_ = inlineStatusLabel(tr("能力来源：编译期内置注册表"));
    capabilitySourceLabel_->setObjectName(QStringLiteral("WorkspaceToolbarMeta"));
    allowLabelToShrink(capabilityStatusLabel_);
    allowLabelToShrink(capabilitySourceLabel_);
    headerLayout->addWidget(contextLabel);
    headerLayout->addWidget(capabilityStatusLabel_);
    headerLayout->addWidget(capabilitySourceLabel_, 1);
    headerLayout->addWidget(refreshButton);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* capabilityCard =
        createCompactSummaryCard(tr("内置能力"), QStringLiteral("0"), tr("编译期注册"));
    auto* datasetCard =
        createCompactSummaryCard(tr("数据集格式"), QStringLiteral("0"), tr("可识别 / 校验格式"));
    auto* exportCard =
        createCompactSummaryCard(tr("导出格式"), QStringLiteral("0"), tr("能力声明的导出目标"));
    auto* gpuCard =
        createCompactSummaryCard(tr("GPU 策略"), QStringLiteral("0"), tr("GPU 推荐或必需能力"));
    capabilityCountLabel_ = metricValue(capabilityCard);
    datasetFormatCountLabel_ = metricValue(datasetCard);
    exportFormatCountLabel_ = metricValue(exportCard);
    gpuCapabilityCountLabel_ = metricValue(gpuCard);
    summaryLayout->addWidget(capabilityCard, 0, 0);
    summaryLayout->addWidget(datasetCard, 0, 1);
    summaryLayout->addWidget(exportCard, 0, 2);
    summaryLayout->addWidget(gpuCard, 0, 3);

    auto* tablePanel = new InfoPanel(tr("内置能力"));
    capabilityTable_ = new QTableWidget(0, 7);
    capabilityTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID") << tr("名称") << tr("来源") << tr("任务")
        << tr("数据集") << tr("后端") << tr("运行策略"));
    configureReadOnlyTable(capabilityTable_);
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
    return page;
}

QWidget* SettingsWorkspacePage::buildApplicationSettingsPanel(
    const QString& licenseOwner, const QString& licenseExpiry)
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* languagePanel = new InfoPanel(tr("界面语言"));
    auto* languageRow = new QFrame;
    languageRow->setObjectName(QStringLiteral("ActionStrip"));
    auto* languageLayout = new QHBoxLayout(languageRow);
    languageLayout->setContentsMargins(10, 8, 10, 8);
    languageLayout->setSpacing(10);
    auto* languageHint = mutedLabel(tr("保存后需要重启 AITrain Studio 才会完全切换界面语言。"));
    allowLabelToShrink(languageHint);
    languageLayout->addWidget(languageHint, 1);
    auto* languageSwitch = new QFrame;
    languageSwitch->setObjectName(QStringLiteral("LanguageSwitch"));
    auto* switchLayout = new QHBoxLayout(languageSwitch);
    switchLayout->setContentsMargins(2, 2, 2, 2);
    switchLayout->setSpacing(0);
    zhLanguageButton_ = new QToolButton;
    zhLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    zhLanguageButton_->setText(QStringLiteral("中"));
    zhLanguageButton_->setCheckable(true);
    zhLanguageButton_->setToolTip(tr("切换到中文，重启后生效"));
    enLanguageButton_ = new QToolButton;
    enLanguageButton_->setObjectName(QStringLiteral("LanguageSwitchButton"));
    enLanguageButton_->setText(QStringLiteral("EN"));
    enLanguageButton_->setCheckable(true);
    enLanguageButton_->setToolTip(tr("切换到英文，重启后生效"));
    switchLayout->addWidget(zhLanguageButton_);
    switchLayout->addWidget(enLanguageButton_);
    languageLayout->addWidget(languageSwitch);
    connect(zhLanguageButton_, &QToolButton::clicked, this,
        [this]() { emit languageRequested(QStringLiteral("zh_CN")); });
    connect(enLanguageButton_, &QToolButton::clicked, this,
        [this]() { emit languageRequested(QStringLiteral("en_US")); });
    languagePanel->bodyLayout()->addWidget(languageRow);
    auto* themeRow = new QHBoxLayout;
    themeRow->addWidget(new QLabel(tr("外观主题")));
    auto* theme = new QComboBox; theme->setObjectName(QStringLiteral("SettingsTheme"));
    theme->addItem(tr("浅色"), QStringLiteral("light")); theme->addItem(tr("深色"), QStringLiteral("dark"));
    themeRow->addWidget(theme); themeRow->addWidget(mutedLabel(tr("立即生效，重启后保留。")), 1);
    languagePanel->bodyLayout()->addLayout(themeRow);
    connect(theme, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, theme]() { emit themeRequested(theme->currentData().toString()); });

    auto* projectPathPanel = new InfoPanel(tr("默认项目目录"));
    auto* projectPathForm = new QFormLayout;
    projectPathForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    defaultProjectPathEdit_ = new QLineEdit;
    auto* browseButton = new QPushButton(tr("选择目录"));
    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->setSpacing(8);
    pathLayout->addWidget(defaultProjectPathEdit_, 1);
    pathLayout->addWidget(browseButton);
    projectPathForm->addRow(tr("默认项目目录"), pathRow);
    projectPathPanel->bodyLayout()->addLayout(projectPathForm);
    auto* pathActions = new QFrame;
    pathActions->setObjectName(QStringLiteral("ActionStrip"));
    auto* pathActionLayout = new QHBoxLayout(pathActions);
    pathActionLayout->setContentsMargins(10, 8, 10, 8);
    pathActionLayout->setSpacing(10);
    defaultProjectPathStatusLabel_ =
        mutedLabel(tr("默认项目目录会作为项目页的初始路径；不会自动打开或迁移现有项目。"));
    allowLabelToShrink(defaultProjectPathStatusLabel_);
    auto* resetButton = new QPushButton(tr("恢复默认"));
    auto* saveButton = primaryButton(tr("保存默认目录"));
    pathActionLayout->addWidget(defaultProjectPathStatusLabel_, 1);
    pathActionLayout->addWidget(resetButton);
    pathActionLayout->addWidget(saveButton);
    projectPathPanel->bodyLayout()->addWidget(pathActions);
    connect(browseButton, &QPushButton::clicked,
        this, &SettingsWorkspacePage::browseDefaultProjectPathRequested);
    connect(saveButton, &QPushButton::clicked,
        this, &SettingsWorkspacePage::saveDefaultProjectPathRequested);
    connect(resetButton, &QPushButton::clicked,
        this, &SettingsWorkspacePage::resetDefaultProjectPathRequested);

    auto* licensePanel = new InfoPanel(tr("授权状态"));
    auto* licenseForm = new QFormLayout;
    licenseForm->addRow(tr("授权用户"),
        new QLabel(licenseOwner.isEmpty() ? tr("已注册") : licenseOwner));
    licenseForm->addRow(tr("有效期"),
        new QLabel(licenseExpiry.isEmpty() ? tr("未记录") : licenseExpiry));
    licensePanel->bodyLayout()->addLayout(licenseForm);
    licensePanel->bodyLayout()->addWidget(
        mutedLabel(tr("离线授权已在启动时校验；这里只展示当前授权信息，不提供换绑或激活入口。")));

    auto* entryPanel = new InfoPanel(tr("系统入口"));
    auto* entryActions = new QFrame;
    entryActions->setObjectName(QStringLiteral("ActionStrip"));
    auto* entryLayout = new QGridLayout(entryActions);
    auto* openProjectButton = primaryButton(tr("打开项目页"));
    auto* openCapabilitiesButton = new QPushButton(tr("打开内置能力设置"));
    auto* openEnvironmentButton = new QPushButton(tr("打开环境页"));
    auto* runEnvironmentButton = new QPushButton(tr("执行环境自检"));
    connect(openProjectButton, &QPushButton::clicked,
        this, &SettingsWorkspacePage::openProjectRequested);
    connect(openCapabilitiesButton, &QPushButton::clicked,
        this, [this]() { showTab(0); });
    connect(openEnvironmentButton, &QPushButton::clicked,
        this, &SettingsWorkspacePage::openEnvironmentRequested);
    connect(runEnvironmentButton, &QPushButton::clicked,
        this, &SettingsWorkspacePage::runEnvironmentRequested);
    entryLayout->addWidget(openProjectButton, 0, 0);
    entryLayout->addWidget(openCapabilitiesButton, 0, 1);
    entryLayout->addWidget(openEnvironmentButton, 1, 0);
    entryLayout->addWidget(runEnvironmentButton, 1, 1);
    entryPanel->bodyLayout()->addWidget(entryActions);
    entryPanel->bodyLayout()->addWidget(
        mutedLabel(tr("环境自检仍通过 Worker 执行，当前有任务运行时会沿用现有 busy guard。")));

    auto* boundaryPanel = new InfoPanel(tr("文件路径边界"));
    boundaryPanel->bodyLayout()->addWidget(mutedLabel(tr(
        "系统设置不展示、打开或复制项目文件的物理路径。数据集、模型和报告必须通过导入边界或任务与产物页按登记身份访问。")));

    layout->addWidget(languagePanel);
    layout->addWidget(projectPathPanel);
    layout->addWidget(licensePanel);
    delete entryPanel;
    delete boundaryPanel;
    layout->addStretch();

    return page;
}

QString SettingsWorkspacePage::defaultProjectPathText() const
{
    return defaultProjectPathEdit_ ? defaultProjectPathEdit_->text() : QString();
}

void SettingsWorkspacePage::setDefaultProjectPathText(const QString& path)
{
    if (defaultProjectPathEdit_) {
        defaultProjectPathEdit_->setText(QDir::toNativeSeparators(path));
    }
}

void SettingsWorkspacePage::setDefaultProjectPathStatus(const QString& status)
{
    if (defaultProjectPathStatusLabel_) {
        defaultProjectPathStatusLabel_->setText(status);
    }
}

void SettingsWorkspacePage::setLanguageCode(const QString& languageCode)
{
    const auto apply = [&languageCode](QToolButton* button, const QString& code) {
        const QSignalBlocker blocker(button);
        button->setChecked(languageCode == code);
    };
    apply(zhLanguageButton_, QStringLiteral("zh_CN"));
    apply(enLanguageButton_, QStringLiteral("en_US"));
}

void SettingsWorkspacePage::setCapabilities(
    const QVector<SettingsCapabilityRow>& rows,
    const SettingsCapabilitySummary& summary)
{
    capabilityTable_->setRowCount(0);
    if (rows.isEmpty()) {
        capabilityTable_->setRowCount(1);
        capabilityTable_->setItem(0, 0, new QTableWidgetItem(tr("暂无内置能力")));
        for (int column = 1; column < capabilityTable_->columnCount(); ++column) {
            capabilityTable_->setItem(0, column,
                new QTableWidgetItem(tr("内置能力注册表为空。")));
        }
    }
    for (const SettingsCapabilityRow& capability : rows) {
        const int row = capabilityTable_->rowCount();
        capabilityTable_->insertRow(row);
        capabilityTable_->setItem(row, 0, new QTableWidgetItem(capability.id));
        capabilityTable_->setItem(row, 1, new QTableWidgetItem(aitrain_app::workbenchText(capability.displayName)));
        capabilityTable_->setItem(row, 2, new QTableWidgetItem(tr("内置")));
        capabilityTable_->setItem(row, 3, new QTableWidgetItem(capability.taskTypes));
        capabilityTable_->setItem(row, 4, new QTableWidgetItem(capability.datasetFormats));
        capabilityTable_->setItem(row, 5, new QTableWidgetItem(capability.backendIds));
        capabilityTable_->setItem(row, 6, new QTableWidgetItem(tr("内置")));
    }
    capabilityStatusLabel_->setText(rows.isEmpty()
        ? tr("内置能力注册表为空。")
        : tr("已注册 %1 个内置能力。").arg(rows.size()));
    capabilitySourceLabel_->setText(tr("能力来源：编译期内置注册表"));
    capabilitySourceLabel_->setToolTip(tr("能力由编译期注册表提供。"));
    capabilityCountLabel_->setText(QString::number(summary.capabilityCount));
    datasetFormatCountLabel_->setText(QString::number(summary.datasetFormatCount));
    datasetFormatCountLabel_->setToolTip(summary.datasetFormats);
    exportFormatCountLabel_->setText(QString::number(summary.exportFormatCount));
    exportFormatCountLabel_->setToolTip(summary.exportFormats);
    gpuCapabilityCountLabel_->setText(QString::number(summary.gpuCapabilityCount));
}

void SettingsWorkspacePage::showTab(int tabIndex)
{
    if (tabs_ && tabIndex >= 0 && tabIndex < tabs_->count()) {
        tabs_->setCurrentIndex(tabIndex);
    }
}

void SettingsWorkspacePage::setThemeCode(const QString& theme)
{
    auto* combo = findChild<QComboBox*>(QStringLiteral("SettingsTheme"));
    const QSignalBlocker blocker(combo); combo->setCurrentIndex(qMax(0, combo->findData(theme)));
}
