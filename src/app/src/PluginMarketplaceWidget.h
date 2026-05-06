#pragma once

#include "aitrain/core/PluginMarketplace.h"

#include <QLabel>
#include <QLineEdit>
#include <QTableWidget>
#include <QWidget>

class PluginMarketplaceWidget : public QWidget {
    Q_OBJECT

public:
    explicit PluginMarketplaceWidget(
        const QString& marketplaceRoot,
        const QString& activePluginDirectory,
        QWidget* parent = nullptr);

public slots:
    void loadIndex();
    void importPackage();
    void enableSelectedPlugin();
    void disableSelectedPlugin();
    void uninstallSelectedPlugin();
    void refreshInstalledPlugins();

signals:
    void pluginsChanged();
    void statusChanged(const QString& status);
    void releasePluginLoadersRequested();

private:
    aitrain::PluginMarketplace marketplace() const;
    void buildUi();
    void configureTable(QTableWidget* table) const;
    void appendReport(const aitrain::PluginMarketplaceReport& report);
    void setStatus(const QString& status);
    QString selectedInstalledPluginId() const;
    QString selectedInstalledPluginVersion() const;

    QLabel* statusLabel_ = nullptr;
    QLineEdit* sourceEdit_ = nullptr;
    QTableWidget* marketplaceTable_ = nullptr;
    QTableWidget* installedTable_ = nullptr;
    QTableWidget* installLogTable_ = nullptr;
    QString marketplaceRoot_;
    QString activePluginDirectory_;
};
