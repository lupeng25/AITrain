#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

struct MarketplacePluginEntry {
    QString id;
    QString name;
    QString version;
    QString description;
    QString publisher;
    QString license;
    QString category;
    QStringList capabilities;
    QString minAitrainVersion;
    QString qtAbi;
    QString msvcRuntime;
    bool requiresGpu = false;
    QJsonArray dependencies;
    QString packageSha256;
    QString signature;
    QString downloadUrl;
    QString installedState;
    QString compatibilityMessage;

    QJsonObject toJson() const;
    static MarketplacePluginEntry fromJson(const QJsonObject& object);
};

struct PluginPackageManifest {
    int schemaVersion = 0;
    QString id;
    QString name;
    QString version;
    QString description;
    QString publisher;
    QString license;
    QString category;
    QStringList capabilities;
    QString qtModelPlugin;
    QString minAitrainVersion;
    QString qtAbi;
    QString msvcRuntime;
    bool requiresGpu = false;
    QJsonArray files;
    QJsonObject hashes;
    QJsonObject raw;

    QJsonObject toJson() const;
    static PluginPackageManifest fromJson(const QJsonObject& object);
};

struct InstalledPluginRecord {
    QString id;
    QString name;
    QString version;
    bool enabled = false;
    QString installPath;
    QString sourcePath;
    QString installedAt;
    QString verificationStatus;
    QString message;
    QStringList activeFiles;
    QJsonObject packageManifest;

    QJsonObject toJson() const;
    static InstalledPluginRecord fromJson(const QJsonObject& object);
};

struct PluginMarketplaceReport {
    bool ok = false;
    QString status;
    QString message;
    QStringList warnings;
    QStringList errors;
    QJsonObject details;

    QJsonObject toJson() const;
};

class PluginMarketplace {
public:
    PluginMarketplace(const QString& marketplaceRoot, const QString& activePluginDirectory, const QString& statePath = QString());

    QString marketplaceRoot() const;
    QString activePluginDirectory() const;
    QString statePath() const;

    QVector<MarketplacePluginEntry> loadIndex(const QString& source, PluginMarketplaceReport* report = nullptr) const;
    QVector<InstalledPluginRecord> installedPlugins(PluginMarketplaceReport* report = nullptr) const;

    PluginMarketplaceReport inspectPackage(const QString& packagePath, PluginPackageManifest* manifest = nullptr) const;
    PluginMarketplaceReport installPackage(const QString& packagePath, bool enableAfterInstall = true);
    PluginMarketplaceReport enablePlugin(const QString& id, const QString& version);
    PluginMarketplaceReport disablePlugin(const QString& id);
    PluginMarketplaceReport uninstallPlugin(const QString& id, const QString& version);

    static int compareSemver(const QString& left, const QString& right);
    static QString currentAitrainVersion();
    static QString currentQtAbi();
    static QString currentMsvcRuntime();

private:
    QJsonObject loadState(QString* error = nullptr) const;
    bool saveState(const QJsonObject& state, QString* error = nullptr) const;
    PluginMarketplaceReport validateManifest(const QString& packageRoot, const PluginPackageManifest& manifest) const;
    PluginMarketplaceReport validateQtEntrypoint(const QString& packageRoot, const PluginPackageManifest& manifest) const;
    bool updateInstalledRecord(const InstalledPluginRecord& record, QString* error);
    bool removeInstalledRecord(const QString& id, const QString& version, QString* error);
    QString packageInstallPath(const QString& id, const QString& version) const;
    QString normalizedPackageRoot(const QString& packagePath, QString* temporaryRoot, PluginMarketplaceReport* report) const;

    QString marketplaceRoot_;
    QString activePluginDirectory_;
    QString statePath_;
};

} // namespace aitrain
