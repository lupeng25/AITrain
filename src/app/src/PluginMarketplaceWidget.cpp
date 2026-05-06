#include "PluginMarketplaceWidget.h"

#include <QAbstractItemView>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QMessageBox>
#include <QPushButton>
#include <QTabWidget>
#include <QTableWidgetItem>
#include <QUrl>
#include <QVBoxLayout>

PluginMarketplaceWidget::PluginMarketplaceWidget(
    const QString& marketplaceRoot,
    const QString& activePluginDirectory,
    QWidget* parent)
    : QWidget(parent)
    , marketplaceRoot_(QFileInfo(marketplaceRoot).absoluteFilePath())
    , activePluginDirectory_(QFileInfo(activePluginDirectory).absoluteFilePath())
{
    buildUi();
    refreshInstalledPlugins();
}

void PluginMarketplaceWidget::loadIndex()
{
    const QString source = sourceEdit_ ? sourceEdit_->text().trimmed() : QString();
    aitrain::PluginMarketplaceReport report;
    const QVector<aitrain::MarketplacePluginEntry> entries = marketplace().loadIndex(source, &report);
    if (marketplaceTable_) {
        marketplaceTable_->setRowCount(0);
        if (entries.isEmpty()) {
            marketplaceTable_->setRowCount(1);
            marketplaceTable_->setItem(0, 0, new QTableWidgetItem(QStringLiteral("暂无市场插件")));
            for (int column = 1; column < marketplaceTable_->columnCount(); ++column) {
                marketplaceTable_->setItem(0, column, new QTableWidgetItem(report.message.isEmpty()
                    ? QStringLiteral("请加载 plugins/marketplace/marketplace.json 或内网静态索引。")
                    : report.message));
            }
        }
        for (const aitrain::MarketplacePluginEntry& entry : entries) {
            const int row = marketplaceTable_->rowCount();
            marketplaceTable_->insertRow(row);
            marketplaceTable_->setItem(row, 0, new QTableWidgetItem(entry.id));
            marketplaceTable_->setItem(row, 1, new QTableWidgetItem(entry.name));
            marketplaceTable_->setItem(row, 2, new QTableWidgetItem(entry.version));
            marketplaceTable_->setItem(row, 3, new QTableWidgetItem(entry.category));
            marketplaceTable_->setItem(row, 4, new QTableWidgetItem(entry.publisher));
            marketplaceTable_->setItem(row, 5, new QTableWidgetItem(entry.installedState));
            marketplaceTable_->setItem(row, 6, new QTableWidgetItem(entry.downloadUrl));
            marketplaceTable_->setItem(row, 7, new QTableWidgetItem(entry.compatibilityMessage.isEmpty() ? entry.description : entry.compatibilityMessage));
        }
    }
    setStatus(report.message);
}

void PluginMarketplaceWidget::importPackage()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("选择插件包"),
        marketplaceRoot_,
        QStringLiteral("AITrain 插件包 (*.aitrain-plugin.zip *.zip);;所有文件 (*.*)"));
    if (path.isEmpty()) {
        return;
    }

    emit releasePluginLoadersRequested();
    const aitrain::PluginMarketplaceReport report = marketplace().installPackage(path, true);
    appendReport(report);
    if (!report.ok) {
        QMessageBox::warning(this, QStringLiteral("插件市场"), report.errors.isEmpty() ? report.message : report.errors.join(QStringLiteral("\n")));
    } else {
        setStatus(report.message);
    }
    refreshInstalledPlugins();
    emit pluginsChanged();
}

void PluginMarketplaceWidget::enableSelectedPlugin()
{
    const QString id = selectedInstalledPluginId();
    const QString version = selectedInstalledPluginVersion();
    if (id.isEmpty() || version.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("插件市场"), QStringLiteral("请先在“已安装”表格中选择插件。"));
        return;
    }
    emit releasePluginLoadersRequested();
    const aitrain::PluginMarketplaceReport report = marketplace().enablePlugin(id, version);
    appendReport(report);
    if (!report.ok) {
        QMessageBox::warning(this, QStringLiteral("插件市场"), report.errors.isEmpty() ? report.message : report.errors.join(QStringLiteral("\n")));
    } else {
        setStatus(QStringLiteral("插件已启用：%1 %2").arg(id, version));
    }
    refreshInstalledPlugins();
    emit pluginsChanged();
}

void PluginMarketplaceWidget::disableSelectedPlugin()
{
    const QString id = selectedInstalledPluginId();
    if (id.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("插件市场"), QStringLiteral("请先在“已安装”表格中选择插件。"));
        return;
    }
    emit releasePluginLoadersRequested();
    const aitrain::PluginMarketplaceReport report = marketplace().disablePlugin(id);
    appendReport(report);
    if (!report.ok) {
        QMessageBox::warning(this, QStringLiteral("插件市场"), report.errors.isEmpty() ? report.message : report.errors.join(QStringLiteral("\n")));
    } else {
        setStatus(QStringLiteral("插件已禁用：%1").arg(id));
    }
    refreshInstalledPlugins();
    emit pluginsChanged();
}

void PluginMarketplaceWidget::uninstallSelectedPlugin()
{
    const QString id = selectedInstalledPluginId();
    const QString version = selectedInstalledPluginVersion();
    if (id.isEmpty() || version.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("插件市场"), QStringLiteral("请先在“已安装”表格中选择插件。"));
        return;
    }
    emit releasePluginLoadersRequested();
    const aitrain::PluginMarketplaceReport report = marketplace().uninstallPlugin(id, version);
    appendReport(report);
    if (!report.ok) {
        QMessageBox::warning(this, QStringLiteral("插件市场"), report.errors.isEmpty() ? report.message : report.errors.join(QStringLiteral("\n")));
    } else {
        setStatus(QStringLiteral("插件已卸载：%1 %2").arg(id, version));
    }
    refreshInstalledPlugins();
    emit pluginsChanged();
}

void PluginMarketplaceWidget::refreshInstalledPlugins()
{
    const aitrain::PluginMarketplace currentMarketplace = marketplace();
    aitrain::PluginMarketplaceReport report;
    const QVector<aitrain::InstalledPluginRecord> installed = currentMarketplace.installedPlugins(&report);
    if (installedTable_) {
        installedTable_->setRowCount(0);
        if (installed.isEmpty()) {
            installedTable_->setRowCount(1);
            installedTable_->setItem(0, 0, new QTableWidgetItem(QStringLiteral("暂无已安装市场插件")));
            for (int column = 1; column < installedTable_->columnCount(); ++column) {
                installedTable_->setItem(0, column, new QTableWidgetItem(QStringLiteral("可通过“导入插件包”安装本地插件包。")));
            }
        }
        for (const aitrain::InstalledPluginRecord& record : installed) {
            const int row = installedTable_->rowCount();
            installedTable_->insertRow(row);
            auto* idItem = new QTableWidgetItem(record.id);
            idItem->setData(Qt::UserRole, record.id);
            auto* versionItem = new QTableWidgetItem(record.version);
            versionItem->setData(Qt::UserRole, record.version);
            installedTable_->setItem(row, 0, idItem);
            installedTable_->setItem(row, 1, new QTableWidgetItem(record.name));
            installedTable_->setItem(row, 2, versionItem);
            installedTable_->setItem(row, 3, new QTableWidgetItem(record.enabled ? QStringLiteral("是") : QStringLiteral("否")));
            installedTable_->setItem(row, 4, new QTableWidgetItem(record.verificationStatus));
            installedTable_->setItem(row, 5, new QTableWidgetItem(QDir::toNativeSeparators(record.installPath)));
            installedTable_->setItem(row, 6, new QTableWidgetItem(record.message));
        }
    }
    setStatus(report.ok
        ? QStringLiteral("插件市场：已安装 %1 个，状态文件 %2").arg(installed.size()).arg(QDir::toNativeSeparators(currentMarketplace.statePath()))
        : report.message);
}

aitrain::PluginMarketplace PluginMarketplaceWidget::marketplace() const
{
    return aitrain::PluginMarketplace(marketplaceRoot_, activePluginDirectory_);
}

void PluginMarketplaceWidget::buildUi()
{
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(8);

    sourceEdit_ = new QLineEdit(QDir(marketplaceRoot_).filePath(QStringLiteral("marketplace.json")));
    auto* loadIndexButton = new QPushButton(QStringLiteral("加载索引"));
    loadIndexButton->setCursor(Qt::PointingHandCursor);
    connect(loadIndexButton, &QPushButton::clicked, this, &PluginMarketplaceWidget::loadIndex);

    auto* marketplaceToolbar = new QHBoxLayout;
    marketplaceToolbar->addWidget(new QLabel(QStringLiteral("市场索引")));
    marketplaceToolbar->addWidget(sourceEdit_, 1);
    marketplaceToolbar->addWidget(loadIndexButton);

    marketplaceTable_ = new QTableWidget(0, 8);
    marketplaceTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID")
        << QStringLiteral("名称")
        << QStringLiteral("版本")
        << QStringLiteral("分类")
        << QStringLiteral("发布者")
        << QStringLiteral("状态")
        << QStringLiteral("来源")
        << QStringLiteral("说明"));
    configureTable(marketplaceTable_);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::Stretch);
    marketplaceTable_->horizontalHeader()->setSectionResizeMode(7, QHeaderView::Stretch);

    installedTable_ = new QTableWidget(0, 7);
    installedTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID")
        << QStringLiteral("名称")
        << QStringLiteral("版本")
        << QStringLiteral("启用")
        << QStringLiteral("校验")
        << QStringLiteral("安装目录")
        << QStringLiteral("消息"));
    configureTable(installedTable_);
    installedTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    installedTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    installedTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    installedTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    installedTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    installedTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
    installedTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::Stretch);

    auto* importButton = new QPushButton(QStringLiteral("导入插件包"));
    auto* enableButton = new QPushButton(QStringLiteral("启用"));
    auto* disableButton = new QPushButton(QStringLiteral("禁用"));
    auto* uninstallButton = new QPushButton(QStringLiteral("卸载"));
    auto* openMarketplaceDirButton = new QPushButton(QStringLiteral("打开市场目录"));
    importButton->setCursor(Qt::PointingHandCursor);
    enableButton->setCursor(Qt::PointingHandCursor);
    disableButton->setCursor(Qt::PointingHandCursor);
    uninstallButton->setCursor(Qt::PointingHandCursor);
    openMarketplaceDirButton->setCursor(Qt::PointingHandCursor);
    connect(importButton, &QPushButton::clicked, this, &PluginMarketplaceWidget::importPackage);
    connect(enableButton, &QPushButton::clicked, this, &PluginMarketplaceWidget::enableSelectedPlugin);
    connect(disableButton, &QPushButton::clicked, this, &PluginMarketplaceWidget::disableSelectedPlugin);
    connect(uninstallButton, &QPushButton::clicked, this, &PluginMarketplaceWidget::uninstallSelectedPlugin);
    connect(openMarketplaceDirButton, &QPushButton::clicked, this, [this]() {
        QDesktopServices::openUrl(QUrl::fromLocalFile(marketplaceRoot_));
    });

    auto* installedToolbar = new QHBoxLayout;
    installedToolbar->addWidget(importButton);
    installedToolbar->addWidget(enableButton);
    installedToolbar->addWidget(disableButton);
    installedToolbar->addWidget(uninstallButton);
    installedToolbar->addStretch();
    installedToolbar->addWidget(openMarketplaceDirButton);

    installLogTable_ = new QTableWidget(0, 4);
    installLogTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("时间")
        << QStringLiteral("状态")
        << QStringLiteral("消息")
        << QStringLiteral("详情"));
    configureTable(installLogTable_);
    installLogTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    installLogTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    installLogTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    installLogTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);

    auto* marketplaceTab = new QWidget;
    auto* marketplaceLayout = new QVBoxLayout(marketplaceTab);
    marketplaceLayout->setContentsMargins(0, 0, 0, 0);
    marketplaceLayout->setSpacing(8);
    marketplaceLayout->addLayout(marketplaceToolbar);
    marketplaceLayout->addWidget(marketplaceTable_);

    auto* installedTab = new QWidget;
    auto* installedLayout = new QVBoxLayout(installedTab);
    installedLayout->setContentsMargins(0, 0, 0, 0);
    installedLayout->setSpacing(8);
    installedLayout->addLayout(installedToolbar);
    installedLayout->addWidget(installedTable_);

    auto* logTab = new QWidget;
    auto* logLayout = new QVBoxLayout(logTab);
    logLayout->setContentsMargins(0, 0, 0, 0);
    logLayout->addWidget(installLogTable_);

    auto* tabs = new QTabWidget;
    tabs->addTab(installedTab, QStringLiteral("已安装"));
    tabs->addTab(marketplaceTab, QStringLiteral("市场"));
    tabs->addTab(logTab, QStringLiteral("安装记录"));
    rootLayout->addWidget(tabs);
}

void PluginMarketplaceWidget::configureTable(QTableWidget* table) const
{
    if (!table) {
        return;
    }
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->verticalHeader()->setVisible(false);
    table->setAlternatingRowColors(true);
    table->setWordWrap(true);
}

void PluginMarketplaceWidget::appendReport(const aitrain::PluginMarketplaceReport& report)
{
    if (!installLogTable_) {
        return;
    }
    const int row = installLogTable_->rowCount();
    installLogTable_->insertRow(row);
    installLogTable_->setItem(row, 0, new QTableWidgetItem(QDateTime::currentDateTime().toString(Qt::ISODate)));
    installLogTable_->setItem(row, 1, new QTableWidgetItem(report.status));
    installLogTable_->setItem(row, 2, new QTableWidgetItem(report.message));
    installLogTable_->setItem(row, 3, new QTableWidgetItem(QString::fromUtf8(QJsonDocument(report.toJson()).toJson(QJsonDocument::Compact))));
}

void PluginMarketplaceWidget::setStatus(const QString& status)
{
    const QString text = status.isEmpty() ? QStringLiteral("插件市场：等待加载本地索引。") : status;
    if (statusLabel_) {
        statusLabel_->setText(text);
    }
    emit statusChanged(text);
}

QString PluginMarketplaceWidget::selectedInstalledPluginId() const
{
    if (!installedTable_ || installedTable_->currentRow() < 0) {
        return {};
    }
    auto* item = installedTable_->item(installedTable_->currentRow(), 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

QString PluginMarketplaceWidget::selectedInstalledPluginVersion() const
{
    if (!installedTable_ || installedTable_->currentRow() < 0) {
        return {};
    }
    auto* item = installedTable_->item(installedTable_->currentRow(), 2);
    return item ? item->data(Qt::UserRole).toString() : QString();
}
