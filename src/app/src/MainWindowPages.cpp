#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"

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

QWidget* MainWindow::buildTopBar()
{
    auto* topBar = new QFrame;
    topBar->setObjectName(QStringLiteral("TopBar"));
    topBar->setFixedHeight(72);

    auto* layout = new QHBoxLayout(topBar);
    layout->setContentsMargins(22, 10, 22, 10);
    layout->setSpacing(16);

    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    pageTitle_ = new QLabel(tr("首页"));
    pageTitle_->setObjectName(QStringLiteral("PageTitle"));
    pageCaption_ = new QLabel;
    pageCaption_->setObjectName(QStringLiteral("PageCaption"));
    titleLayout->addWidget(pageTitle_);
    titleLayout->addWidget(pageCaption_);

    headerProjectLabel_ = new QLabel(tr("项目：未打开"));
    headerProjectLabel_->setObjectName(QStringLiteral("MutedText"));
    workerPill_ = new StatusPill;
    workerPill_->setStatus(tr("Worker 空闲"), StatusPill::Tone::Neutral);
    pluginPill_ = new StatusPill;
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
    updateLanguageButtonState();
    languageLayout->addWidget(topBarZhLanguageButton_);
    languageLayout->addWidget(topBarEnLanguageButton_);
    connect(topBarZhLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("zh_CN"));
    });
    connect(topBarEnLanguageButton_, &QToolButton::clicked, this, [this]() {
        storeLanguagePreference(QStringLiteral("en_US"));
    });

    layout->addWidget(titleBlock, 1);
    layout->addWidget(headerProjectLabel_);
    layout->addWidget(workerPill_);
    layout->addWidget(pluginPill_);
    layout->addWidget(gpuPill_);
    layout->addWidget(licensePill_);
    layout->addWidget(languageSwitch);
    return topBar;
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
