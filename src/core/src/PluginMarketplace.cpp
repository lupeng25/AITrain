#include "aitrain/core/PluginMarketplace.h"

#include "aitrain/core/PluginManager.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QEventLoop>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QRegularExpression>
#include <QTimer>
#include <QTemporaryDir>

namespace aitrain {

namespace {

QStringList readStringList(const QJsonArray& array)
{
    QStringList values;
    for (const QJsonValue& value : array) {
        values.append(value.toString());
    }
    return values;
}

QJsonArray stringListToJson(const QStringList& values)
{
    return QJsonArray::fromStringList(values);
}

QString nowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
}

bool readJsonObjectFile(const QString& path, QJsonObject* object, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("Cannot open JSON file: %1").arg(path);
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Invalid JSON object in %1: %2").arg(path, parseError.errorString());
        }
        return false;
    }
    if (object) {
        *object = document.object();
    }
    return true;
}

bool readJsonObjectUrl(const QString& url, QJsonObject* object, QString* error)
{
    QNetworkAccessManager manager;
    QNetworkReply* reply = manager.get(QNetworkRequest(QUrl(url)));
    QEventLoop loop;
    QTimer timer;
    timer.setSingleShot(true);
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    QObject::connect(&timer, &QTimer::timeout, &loop, &QEventLoop::quit);
    timer.start(15000);
    loop.exec();
    if (timer.isActive()) {
        timer.stop();
    } else {
        reply->abort();
    }

    if (reply->error() != QNetworkReply::NoError) {
        if (error) {
            *error = QStringLiteral("Cannot load marketplace index %1: %2").arg(url, reply->errorString());
        }
        reply->deleteLater();
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(reply->readAll(), &parseError);
    reply->deleteLater();
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) {
            *error = QStringLiteral("Invalid JSON object from %1: %2").arg(url, parseError.errorString());
        }
        return false;
    }
    if (object) {
        *object = document.object();
    }
    return true;
}

bool writeJsonObjectFile(const QString& path, const QJsonObject& object, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Cannot create directory for %1").arg(path);
        }
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = QStringLiteral("Cannot write JSON file: %1").arg(path);
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

bool copyDirectory(const QString& sourcePath, const QString& targetPath, QString* error)
{
    const QDir source(sourcePath);
    if (!source.exists()) {
        if (error) {
            *error = QStringLiteral("Source directory does not exist: %1").arg(sourcePath);
        }
        return false;
    }
    if (!QDir().mkpath(targetPath)) {
        if (error) {
            *error = QStringLiteral("Cannot create target directory: %1").arg(targetPath);
        }
        return false;
    }

    const QFileInfoList entries = source.entryInfoList(QDir::NoDotAndDotDot | QDir::Files | QDir::Dirs);
    for (const QFileInfo& entry : entries) {
        const QString target = QDir(targetPath).filePath(entry.fileName());
        if (entry.isDir()) {
            if (!copyDirectory(entry.absoluteFilePath(), target, error)) {
                return false;
            }
            continue;
        }
        QFile::remove(target);
        if (!QFile::copy(entry.absoluteFilePath(), target)) {
            if (error) {
                *error = QStringLiteral("Cannot copy %1 to %2").arg(entry.absoluteFilePath(), target);
            }
            return false;
        }
    }
    return true;
}

QString fileSha256(const QString& path, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("Cannot open file for hashing: %1").arg(path);
        }
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        hash.addData(file.read(1024 * 1024));
    }
    return QString::fromLatin1(hash.result().toHex());
}

QString cleanRelativePath(const QString& path)
{
    QString normalized = QDir::fromNativeSeparators(path).trimmed();
    while (normalized.startsWith(QLatin1Char('/'))) {
        normalized.remove(0, 1);
    }
    return QDir::cleanPath(normalized);
}

bool removeDirectorySafely(const QString& path, QString* error)
{
    QDir dir(path);
    if (!dir.exists()) {
        return true;
    }
    if (!dir.removeRecursively()) {
        if (error) {
            *error = QStringLiteral("Cannot remove directory: %1").arg(path);
        }
        return false;
    }
    return true;
}

QJsonObject objectOrNestedMetadata(const QJsonObject& object)
{
    if (object.contains(QStringLiteral("metadata")) && object.value(QStringLiteral("metadata")).isObject()) {
        QJsonObject merged = object.value(QStringLiteral("metadata")).toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it) {
            if (!merged.contains(it.key()) && it.key() != QStringLiteral("metadata")) {
                merged.insert(it.key(), it.value());
            }
        }
        return merged;
    }
    return object;
}

QStringList activeDllSuffixes()
{
#if defined(Q_OS_WIN)
    return QStringList() << QStringLiteral("*.dll");
#elif defined(Q_OS_MAC)
    return QStringList() << QStringLiteral("*.dylib") << QStringLiteral("*.so");
#else
    return QStringList() << QStringLiteral("*.so");
#endif
}

} // namespace

QJsonObject MarketplacePluginEntry::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("id"), id);
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("version"), version);
    object.insert(QStringLiteral("description"), description);
    object.insert(QStringLiteral("publisher"), publisher);
    object.insert(QStringLiteral("license"), license);
    object.insert(QStringLiteral("category"), category);
    object.insert(QStringLiteral("capabilities"), stringListToJson(capabilities));
    object.insert(QStringLiteral("minAitrainVersion"), minAitrainVersion);
    object.insert(QStringLiteral("qtAbi"), qtAbi);
    object.insert(QStringLiteral("msvcRuntime"), msvcRuntime);
    object.insert(QStringLiteral("requiresGpu"), requiresGpu);
    object.insert(QStringLiteral("dependencies"), dependencies);
    object.insert(QStringLiteral("packageSha256"), packageSha256);
    object.insert(QStringLiteral("signature"), signature);
    object.insert(QStringLiteral("downloadUrl"), downloadUrl);
    object.insert(QStringLiteral("installedState"), installedState);
    object.insert(QStringLiteral("compatibilityMessage"), compatibilityMessage);
    return object;
}

MarketplacePluginEntry MarketplacePluginEntry::fromJson(const QJsonObject& object)
{
    const QJsonObject source = objectOrNestedMetadata(object);
    MarketplacePluginEntry entry;
    entry.id = source.value(QStringLiteral("id")).toString();
    entry.name = source.value(QStringLiteral("name")).toString();
    entry.version = source.value(QStringLiteral("version")).toString();
    entry.description = source.value(QStringLiteral("description")).toString();
    entry.publisher = source.value(QStringLiteral("publisher")).toString();
    entry.license = source.value(QStringLiteral("license")).toString();
    entry.category = source.value(QStringLiteral("category")).toString();
    entry.capabilities = readStringList(source.value(QStringLiteral("capabilities")).toArray());
    entry.minAitrainVersion = source.value(QStringLiteral("minAitrainVersion")).toString();
    entry.qtAbi = source.value(QStringLiteral("qtAbi")).toString();
    entry.msvcRuntime = source.value(QStringLiteral("msvcRuntime")).toString();
    entry.requiresGpu = source.value(QStringLiteral("requiresGpu")).toBool();
    entry.dependencies = source.value(QStringLiteral("dependencies")).toArray();
    entry.packageSha256 = source.value(QStringLiteral("packageSha256")).toString();
    entry.signature = source.value(QStringLiteral("signature")).toString();
    entry.downloadUrl = source.value(QStringLiteral("downloadUrl")).toString();
    return entry;
}

QJsonObject PluginPackageManifest::toJson() const
{
    QJsonObject object = raw;
    object.insert(QStringLiteral("schemaVersion"), schemaVersion);
    object.insert(QStringLiteral("id"), id);
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("version"), version);
    object.insert(QStringLiteral("description"), description);
    object.insert(QStringLiteral("publisher"), publisher);
    object.insert(QStringLiteral("license"), license);
    object.insert(QStringLiteral("category"), category);
    object.insert(QStringLiteral("capabilities"), stringListToJson(capabilities));
    object.insert(QStringLiteral("entrypoints"), QJsonObject{{QStringLiteral("qtModelPlugin"), qtModelPlugin}});
    object.insert(QStringLiteral("compatibility"), QJsonObject{
        {QStringLiteral("minAitrainVersion"), minAitrainVersion},
        {QStringLiteral("qtAbi"), qtAbi},
        {QStringLiteral("msvcRuntime"), msvcRuntime},
        {QStringLiteral("requiresGpu"), requiresGpu}
    });
    object.insert(QStringLiteral("files"), files);
    object.insert(QStringLiteral("hashes"), hashes);
    return object;
}

PluginPackageManifest PluginPackageManifest::fromJson(const QJsonObject& object)
{
    PluginPackageManifest manifest;
    manifest.raw = object;
    manifest.schemaVersion = object.value(QStringLiteral("schemaVersion")).toInt();
    manifest.id = object.value(QStringLiteral("id")).toString();
    manifest.name = object.value(QStringLiteral("name")).toString();
    manifest.version = object.value(QStringLiteral("version")).toString();
    manifest.description = object.value(QStringLiteral("description")).toString();
    manifest.publisher = object.value(QStringLiteral("publisher")).toString();
    manifest.license = object.value(QStringLiteral("license")).toString();
    manifest.category = object.value(QStringLiteral("category")).toString();
    manifest.capabilities = readStringList(object.value(QStringLiteral("capabilities")).toArray());
    const QJsonObject entrypoints = object.value(QStringLiteral("entrypoints")).toObject();
    manifest.qtModelPlugin = cleanRelativePath(entrypoints.value(QStringLiteral("qtModelPlugin")).toString());
    const QJsonObject compatibility = object.value(QStringLiteral("compatibility")).toObject();
    manifest.minAitrainVersion = compatibility.value(QStringLiteral("minAitrainVersion")).toString();
    manifest.qtAbi = compatibility.value(QStringLiteral("qtAbi")).toString();
    manifest.msvcRuntime = compatibility.value(QStringLiteral("msvcRuntime")).toString();
    manifest.requiresGpu = compatibility.value(QStringLiteral("requiresGpu")).toBool();
    manifest.files = object.value(QStringLiteral("files")).toArray();
    manifest.hashes = object.value(QStringLiteral("hashes")).toObject();
    return manifest;
}

QJsonObject InstalledPluginRecord::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("id"), id);
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("version"), version);
    object.insert(QStringLiteral("enabled"), enabled);
    object.insert(QStringLiteral("installPath"), installPath);
    object.insert(QStringLiteral("sourcePath"), sourcePath);
    object.insert(QStringLiteral("installedAt"), installedAt);
    object.insert(QStringLiteral("verificationStatus"), verificationStatus);
    object.insert(QStringLiteral("message"), message);
    object.insert(QStringLiteral("activeFiles"), stringListToJson(activeFiles));
    object.insert(QStringLiteral("packageManifest"), packageManifest);
    return object;
}

InstalledPluginRecord InstalledPluginRecord::fromJson(const QJsonObject& object)
{
    InstalledPluginRecord record;
    record.id = object.value(QStringLiteral("id")).toString();
    record.name = object.value(QStringLiteral("name")).toString();
    record.version = object.value(QStringLiteral("version")).toString();
    record.enabled = object.value(QStringLiteral("enabled")).toBool();
    record.installPath = object.value(QStringLiteral("installPath")).toString();
    record.sourcePath = object.value(QStringLiteral("sourcePath")).toString();
    record.installedAt = object.value(QStringLiteral("installedAt")).toString();
    record.verificationStatus = object.value(QStringLiteral("verificationStatus")).toString();
    record.message = object.value(QStringLiteral("message")).toString();
    record.activeFiles = readStringList(object.value(QStringLiteral("activeFiles")).toArray());
    record.packageManifest = object.value(QStringLiteral("packageManifest")).toObject();
    return record;
}

QJsonObject PluginMarketplaceReport::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("ok"), ok);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("message"), message);
    object.insert(QStringLiteral("warnings"), stringListToJson(warnings));
    object.insert(QStringLiteral("errors"), stringListToJson(errors));
    object.insert(QStringLiteral("details"), details);
    return object;
}

PluginMarketplace::PluginMarketplace(const QString& marketplaceRoot, const QString& activePluginDirectory, const QString& statePath)
    : marketplaceRoot_(QFileInfo(marketplaceRoot).absoluteFilePath())
    , activePluginDirectory_(QFileInfo(activePluginDirectory).absoluteFilePath())
    , statePath_(statePath.isEmpty()
          ? QDir(QFileInfo(marketplaceRoot_).absoluteFilePath()).filePath(QStringLiteral("plugin_marketplace_state.json"))
          : QFileInfo(statePath).absoluteFilePath())
{
}

QString PluginMarketplace::marketplaceRoot() const
{
    return marketplaceRoot_;
}

QString PluginMarketplace::activePluginDirectory() const
{
    return activePluginDirectory_;
}

QString PluginMarketplace::statePath() const
{
    return statePath_;
}

QVector<MarketplacePluginEntry> PluginMarketplace::loadIndex(const QString& source, PluginMarketplaceReport* report) const
{
    QVector<MarketplacePluginEntry> entries;
    PluginMarketplaceReport localReport;
    localReport.status = QStringLiteral("loaded");

    QJsonObject root;
    QString error;
    const QString indexPath = source.isEmpty()
        ? QDir(marketplaceRoot_).filePath(QStringLiteral("marketplace.json"))
        : source;
    const bool isRemote = indexPath.startsWith(QStringLiteral("http://"), Qt::CaseInsensitive)
        || indexPath.startsWith(QStringLiteral("https://"), Qt::CaseInsensitive);
    const QString resolvedSource = isRemote ? indexPath : QFileInfo(indexPath).absoluteFilePath();
    if (!(isRemote ? readJsonObjectUrl(indexPath, &root, &error) : readJsonObjectFile(resolvedSource, &root, &error))) {
        localReport.ok = false;
        localReport.status = QStringLiteral("missing");
        localReport.message = error;
        localReport.errors.append(error);
        if (report) {
            *report = localReport;
        }
        return entries;
    }

    const QJsonArray plugins = root.value(QStringLiteral("plugins")).toArray();
    const QVector<InstalledPluginRecord> installed = installedPlugins();
    for (const QJsonValue& value : plugins) {
        MarketplacePluginEntry entry = MarketplacePluginEntry::fromJson(value.toObject());
        entry.installedState = QStringLiteral("available");
        if (!entry.minAitrainVersion.isEmpty()
            && compareSemver(currentAitrainVersion(), entry.minAitrainVersion) < 0) {
            entry.installedState = QStringLiteral("incompatible");
            entry.compatibilityMessage = QStringLiteral("AITrain version %1 is lower than required %2.")
                .arg(currentAitrainVersion(), entry.minAitrainVersion);
        } else if (!entry.qtAbi.isEmpty() && entry.qtAbi != currentQtAbi()) {
            entry.installedState = QStringLiteral("incompatible");
            entry.compatibilityMessage = QStringLiteral("Qt ABI mismatch: requires %1, current %2.")
                .arg(entry.qtAbi, currentQtAbi());
        } else {
            for (const InstalledPluginRecord& record : installed) {
                if (record.id != entry.id) {
                    continue;
                }
                if (record.version == entry.version) {
                    entry.installedState = record.enabled ? QStringLiteral("installed-enabled") : QStringLiteral("installed-disabled");
                    break;
                }
                if (compareSemver(record.version, entry.version) < 0) {
                    entry.installedState = QStringLiteral("update-available");
                }
            }
        }
        entries.append(entry);
    }

    localReport.ok = true;
    localReport.message = QStringLiteral("Loaded %1 marketplace plugin entries.").arg(entries.size());
    localReport.details.insert(QStringLiteral("source"), resolvedSource);
    localReport.details.insert(QStringLiteral("count"), entries.size());
    if (report) {
        *report = localReport;
    }
    return entries;
}

QVector<InstalledPluginRecord> PluginMarketplace::installedPlugins(PluginMarketplaceReport* report) const
{
    QVector<InstalledPluginRecord> records;
    QString error;
    const QJsonObject state = loadState(&error);
    if (!error.isEmpty()) {
        if (report) {
            report->ok = false;
            report->status = QStringLiteral("state-error");
            report->message = error;
            report->errors.append(error);
        }
        return records;
    }
    const QJsonArray installed = state.value(QStringLiteral("installed")).toArray();
    for (const QJsonValue& value : installed) {
        records.append(InstalledPluginRecord::fromJson(value.toObject()));
    }
    if (report) {
        report->ok = true;
        report->status = QStringLiteral("loaded");
        report->message = QStringLiteral("Loaded %1 installed plugin records.").arg(records.size());
    }
    return records;
}

PluginMarketplaceReport PluginMarketplace::inspectPackage(const QString& packagePath, PluginPackageManifest* manifest) const
{
    PluginMarketplaceReport report;
    QString temporaryRoot;
    const QString root = normalizedPackageRoot(packagePath, &temporaryRoot, &report);
    if (root.isEmpty()) {
        return report;
    }
    const auto cleanupTemporaryRoot = [&temporaryRoot]() {
        if (!temporaryRoot.isEmpty()) {
            removeDirectorySafely(temporaryRoot, nullptr);
        }
    };

    QJsonObject manifestJson;
    QString error;
    const QString manifestPath = QDir(root).filePath(QStringLiteral("plugin.json"));
    if (!readJsonObjectFile(manifestPath, &manifestJson, &error)) {
        report.ok = false;
        report.status = QStringLiteral("invalid");
        report.message = error;
        report.errors.append(error);
        cleanupTemporaryRoot();
        return report;
    }
    const PluginPackageManifest parsed = PluginPackageManifest::fromJson(manifestJson);
    report = validateManifest(root, parsed);
    if (report.ok) {
        const PluginMarketplaceReport entrypointReport = validateQtEntrypoint(root, parsed);
        if (!entrypointReport.ok) {
            report = entrypointReport;
        }
    }
    report.details.insert(QStringLiteral("packageRoot"), root);
    report.details.insert(QStringLiteral("temporaryRoot"), temporaryRoot);
    report.details.insert(QStringLiteral("manifest"), parsed.toJson());
    if (manifest) {
        *manifest = parsed;
    }
    cleanupTemporaryRoot();
    return report;
}

PluginMarketplaceReport PluginMarketplace::installPackage(const QString& packagePath, bool enableAfterInstall)
{
    PluginPackageManifest manifest;
    PluginMarketplaceReport report = inspectPackage(packagePath, &manifest);
    if (!report.ok) {
        return report;
    }

    QString temporaryRoot;
    PluginMarketplaceReport rootReport;
    const QString sourceRoot = normalizedPackageRoot(packagePath, &temporaryRoot, &rootReport);
    if (sourceRoot.isEmpty()) {
        return rootReport;
    }
    const auto cleanupTemporaryRoot = [&temporaryRoot]() {
        if (!temporaryRoot.isEmpty()) {
            removeDirectorySafely(temporaryRoot, nullptr);
        }
    };

    QString error;
    QDir().mkpath(marketplaceRoot_);
    const QString targetPath = packageInstallPath(manifest.id, manifest.version);
    const QString stagingPath = QStringLiteral("%1.staging.%2")
        .arg(targetPath, QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMddHHmmsszzz")));
    removeDirectorySafely(stagingPath, nullptr);
    if (!copyDirectory(sourceRoot, stagingPath, &error)) {
        report.ok = false;
        report.status = QStringLiteral("install-failed");
        report.message = error;
        report.errors.append(error);
        cleanupTemporaryRoot();
        return report;
    }
    removeDirectorySafely(targetPath, nullptr);
    QDir targetParent(QFileInfo(targetPath).absolutePath());
    if (!targetParent.rename(QFileInfo(stagingPath).fileName(), QFileInfo(targetPath).fileName())) {
        if (!copyDirectory(stagingPath, targetPath, &error)) {
            report.ok = false;
            report.status = QStringLiteral("install-failed");
            report.message = QStringLiteral("Cannot move staged plugin into install directory: %1").arg(error);
            report.errors.append(report.message);
            cleanupTemporaryRoot();
            return report;
        }
        removeDirectorySafely(stagingPath, nullptr);
    }
    cleanupTemporaryRoot();

    InstalledPluginRecord record;
    record.id = manifest.id;
    record.name = manifest.name;
    record.version = manifest.version;
    record.enabled = false;
    record.installPath = targetPath;
    record.sourcePath = QFileInfo(packagePath).absoluteFilePath();
    record.installedAt = nowIso();
    record.verificationStatus = QStringLiteral("verified-local-unsigned");
    record.message = QStringLiteral("Installed from local package. Signature enforcement is reserved for a future release.");
    record.packageManifest = manifest.toJson();
    if (!updateInstalledRecord(record, &error)) {
        report.ok = false;
        report.status = QStringLiteral("state-error");
        report.message = error;
        report.errors.append(error);
        return report;
    }

    if (enableAfterInstall) {
        report = enablePlugin(manifest.id, manifest.version);
        if (!report.ok) {
            return report;
        }
    } else {
        report.ok = true;
        report.status = QStringLiteral("installed");
        report.message = QStringLiteral("Plugin installed but not enabled: %1 %2").arg(manifest.id, manifest.version);
    }
    report.details.insert(QStringLiteral("installPath"), targetPath);
    return report;
}

PluginMarketplaceReport PluginMarketplace::enablePlugin(const QString& id, const QString& version)
{
    PluginMarketplaceReport report;
    QVector<InstalledPluginRecord> records = installedPlugins(&report);
    if (!report.ok && report.status == QStringLiteral("state-error")) {
        return report;
    }

    int targetIndex = -1;
    for (int index = 0; index < records.size(); ++index) {
        if (records.at(index).id == id && records.at(index).version == version) {
            targetIndex = index;
            break;
        }
    }
    if (targetIndex < 0) {
        report.ok = false;
        report.status = QStringLiteral("not-installed");
        report.message = QStringLiteral("Plugin is not installed: %1 %2").arg(id, version);
        report.errors.append(report.message);
        return report;
    }

    QString error;
    QDir().mkpath(activePluginDirectory_);
    for (InstalledPluginRecord& record : records) {
        if (record.id == id && record.enabled) {
            disablePlugin(record.id);
            record.enabled = false;
            record.activeFiles.clear();
        }
    }

    InstalledPluginRecord target = records.at(targetIndex);
    const PluginPackageManifest manifest = PluginPackageManifest::fromJson(target.packageManifest);
    const QString sourceDll = QDir(target.installPath).filePath(manifest.qtModelPlugin);
    if (!QFileInfo::exists(sourceDll)) {
        report.ok = false;
        report.status = QStringLiteral("missing-entrypoint");
        report.message = QStringLiteral("Plugin entrypoint is missing: %1").arg(sourceDll);
        report.errors.append(report.message);
        return report;
    }

    const QString targetDll = QDir(activePluginDirectory_).filePath(QFileInfo(sourceDll).fileName());
    QFile::remove(targetDll);
    if (!QFile::copy(sourceDll, targetDll)) {
        report.ok = false;
        report.status = QStringLiteral("enable-failed");
        report.message = QStringLiteral("Cannot copy plugin entrypoint to active directory: %1. The target DLL may be loaded by the running application or blocked by file permissions.").arg(targetDll);
        report.errors.append(report.message);
        return report;
    }
    QFile::setPermissions(targetDll,
        QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner
        | QFile::ReadUser | QFile::WriteUser | QFile::ExeUser
        | QFile::ReadGroup | QFile::ExeGroup
        | QFile::ReadOther | QFile::ExeOther);

    target.enabled = true;
    target.activeFiles = QStringList() << targetDll;
    target.message = QStringLiteral("Plugin enabled.");
    if (!updateInstalledRecord(target, &error)) {
        report.ok = false;
        report.status = QStringLiteral("state-error");
        report.message = error;
        report.errors.append(error);
        return report;
    }

    report.ok = true;
    report.status = QStringLiteral("enabled");
    report.message = QStringLiteral("Plugin enabled: %1 %2").arg(id, version);
    report.details.insert(QStringLiteral("activeFile"), targetDll);
    return report;
}

PluginMarketplaceReport PluginMarketplace::disablePlugin(const QString& id)
{
    PluginMarketplaceReport report;
    QVector<InstalledPluginRecord> records = installedPlugins(&report);
    if (!report.ok && report.status == QStringLiteral("state-error")) {
        return report;
    }
    bool changed = false;
    QString error;
    for (InstalledPluginRecord& record : records) {
        if (record.id != id || !record.enabled) {
            continue;
        }
        for (const QString& activeFile : record.activeFiles) {
            QFile::remove(activeFile);
        }
        record.enabled = false;
        record.activeFiles.clear();
        record.message = QStringLiteral("Plugin disabled.");
        if (!updateInstalledRecord(record, &error)) {
            report.ok = false;
            report.status = QStringLiteral("state-error");
            report.message = error;
            report.errors.append(error);
            return report;
        }
        changed = true;
    }
    report.ok = true;
    report.status = changed ? QStringLiteral("disabled") : QStringLiteral("not-enabled");
    report.message = changed
        ? QStringLiteral("Plugin disabled: %1").arg(id)
        : QStringLiteral("Plugin was not enabled: %1").arg(id);
    return report;
}

PluginMarketplaceReport PluginMarketplace::uninstallPlugin(const QString& id, const QString& version)
{
    PluginMarketplaceReport report;
    disablePlugin(id);
    const QString installPath = packageInstallPath(id, version);
    QString error;
    if (!removeDirectorySafely(installPath, &error)) {
        report.ok = false;
        report.status = QStringLiteral("uninstall-failed");
        report.message = error;
        report.errors.append(error);
        return report;
    }
    if (!removeInstalledRecord(id, version, &error)) {
        report.ok = false;
        report.status = QStringLiteral("state-error");
        report.message = error;
        report.errors.append(error);
        return report;
    }
    report.ok = true;
    report.status = QStringLiteral("uninstalled");
    report.message = QStringLiteral("Plugin uninstalled: %1 %2").arg(id, version);
    return report;
}

int PluginMarketplace::compareSemver(const QString& left, const QString& right)
{
    const auto parse = [](const QString& value) {
        QVector<int> parts;
        const QStringList tokens = value.split(QRegularExpression(QStringLiteral("[^0-9]+")), QString::SkipEmptyParts);
        for (int index = 0; index < 3; ++index) {
            parts.append(index < tokens.size() ? tokens.at(index).toInt() : 0);
        }
        return parts;
    };
    const QVector<int> a = parse(left);
    const QVector<int> b = parse(right);
    for (int index = 0; index < 3; ++index) {
        if (a.at(index) < b.at(index)) return -1;
        if (a.at(index) > b.at(index)) return 1;
    }
    return 0;
}

QString PluginMarketplace::currentAitrainVersion()
{
    return QStringLiteral("0.1.0");
}

QString PluginMarketplace::currentQtAbi()
{
    return QStringLiteral("Qt%1.%2").arg(QT_VERSION_MAJOR).arg(QT_VERSION_MINOR);
}

QString PluginMarketplace::currentMsvcRuntime()
{
#if defined(_MSC_VER)
    return QStringLiteral("msvc%1").arg(_MSC_VER);
#else
    return {};
#endif
}

QJsonObject PluginMarketplace::loadState(QString* error) const
{
    if (!QFileInfo::exists(statePath_)) {
        return QJsonObject{{QStringLiteral("schemaVersion"), 1}, {QStringLiteral("installed"), QJsonArray()}};
    }
    QJsonObject state;
    if (!readJsonObjectFile(statePath_, &state, error)) {
        return {};
    }
    if (!state.contains(QStringLiteral("installed"))) {
        state.insert(QStringLiteral("installed"), QJsonArray());
    }
    return state;
}

bool PluginMarketplace::saveState(const QJsonObject& state, QString* error) const
{
    return writeJsonObjectFile(statePath_, state, error);
}

PluginMarketplaceReport PluginMarketplace::validateManifest(const QString& packageRoot, const PluginPackageManifest& manifest) const
{
    PluginMarketplaceReport report;
    report.ok = true;
    report.status = QStringLiteral("valid");
    const auto require = [&report](bool condition, const QString& message) {
        if (!condition) {
            report.ok = false;
            report.status = QStringLiteral("invalid");
            report.errors.append(message);
        }
    };

    require(manifest.schemaVersion == 1, QStringLiteral("plugin.json schemaVersion must be 1."));
    require(!manifest.id.isEmpty(), QStringLiteral("plugin.json id is required."));
    require(!manifest.version.isEmpty(), QStringLiteral("plugin.json version is required."));
    require(!manifest.qtModelPlugin.isEmpty(), QStringLiteral("entrypoints.qtModelPlugin is required."));
    require(QFileInfo::exists(QDir(packageRoot).filePath(QStringLiteral("payload"))), QStringLiteral("payload/ directory is required."));
    require(QFileInfo::exists(QDir(packageRoot).filePath(QStringLiteral("LICENSE"))), QStringLiteral("LICENSE file is required."));
    if (!manifest.minAitrainVersion.isEmpty()
        && compareSemver(currentAitrainVersion(), manifest.minAitrainVersion) < 0) {
        report.ok = false;
        report.status = QStringLiteral("incompatible");
        report.errors.append(QStringLiteral("AITrain version %1 is lower than required %2.")
            .arg(currentAitrainVersion(), manifest.minAitrainVersion));
    }
    if (!manifest.qtAbi.isEmpty() && manifest.qtAbi != currentQtAbi()) {
        report.ok = false;
        report.status = QStringLiteral("incompatible");
        report.errors.append(QStringLiteral("Qt ABI mismatch: requires %1, current %2.")
            .arg(manifest.qtAbi, currentQtAbi()));
    }

    for (auto it = manifest.hashes.constBegin(); it != manifest.hashes.constEnd(); ++it) {
        const QString relativePath = cleanRelativePath(it.key());
        const QString expected = it.value().toString().trimmed().toLower();
        if (expected.isEmpty()) {
            continue;
        }
        QString error;
        const QString actual = fileSha256(QDir(packageRoot).filePath(relativePath), &error);
        if (!error.isEmpty() || actual.toLower() != expected) {
            report.ok = false;
            report.status = QStringLiteral("hash-mismatch");
            report.errors.append(error.isEmpty()
                    ? QStringLiteral("SHA256 mismatch for %1.").arg(relativePath)
                    : error);
        }
    }

    if (report.ok) {
        report.message = QStringLiteral("Plugin package manifest is valid.");
        report.warnings.append(QStringLiteral("Signature verification is not enforced in v1; local package is marked unsigned."));
    } else {
        report.message = report.errors.join(QStringLiteral(" "));
    }
    return report;
}

PluginMarketplaceReport PluginMarketplace::validateQtEntrypoint(const QString& packageRoot, const PluginPackageManifest& manifest) const
{
    PluginMarketplaceReport report;
    const QString entrypoint = QDir(packageRoot).filePath(manifest.qtModelPlugin);
    if (!QFileInfo::exists(entrypoint)) {
        report.ok = false;
        report.status = QStringLiteral("missing-entrypoint");
        report.message = QStringLiteral("Qt model plugin entrypoint does not exist: %1").arg(entrypoint);
        report.errors.append(report.message);
        return report;
    }

    QTemporaryDir dir;
    if (!dir.isValid()) {
        report.ok = false;
        report.status = QStringLiteral("temp-error");
        report.message = QStringLiteral("Cannot create temporary directory for plugin load validation.");
        report.errors.append(report.message);
        return report;
    }
    const QString copiedEntrypoint = QDir(dir.path()).filePath(QFileInfo(entrypoint).fileName());
    if (!QFile::copy(entrypoint, copiedEntrypoint)) {
        report.ok = false;
        report.status = QStringLiteral("copy-failed");
        report.message = QStringLiteral("Cannot stage plugin entrypoint for validation: %1").arg(entrypoint);
        report.errors.append(report.message);
        return report;
    }
    QFile::setPermissions(copiedEntrypoint,
        QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner
        | QFile::ReadUser | QFile::WriteUser | QFile::ExeUser
        | QFile::ReadGroup | QFile::ExeGroup
        | QFile::ReadOther | QFile::ExeOther);
    PluginManager manager;
    manager.scan(QStringList() << dir.path());
    for (IModelPlugin* plugin : manager.plugins()) {
        if (!plugin) {
            continue;
        }
        const QString loadedId = plugin->manifest().id;
        if (loadedId == manifest.id) {
            report.ok = true;
            report.status = QStringLiteral("loadable");
            report.message = QStringLiteral("Qt plugin entrypoint loads and matches id %1.").arg(manifest.id);
            return report;
        }
        report.errors.append(QStringLiteral("Plugin manifest id mismatch: package=%1 dll=%2.").arg(manifest.id, loadedId));
    }
    report.ok = false;
    report.status = QStringLiteral("load-failed");
    report.message = QStringLiteral("Qt plugin entrypoint did not load as AITrain IModelPlugin.");
    report.errors.append(manager.errors());
    if (report.errors.isEmpty()) {
        report.errors.append(report.message);
    }
    return report;
}

bool PluginMarketplace::updateInstalledRecord(const InstalledPluginRecord& record, QString* error)
{
    QJsonObject state = loadState(error);
    if (error && !error->isEmpty()) {
        return false;
    }
    QJsonArray installed = state.value(QStringLiteral("installed")).toArray();
    bool replaced = false;
    for (int index = 0; index < installed.size(); ++index) {
        const QJsonObject object = installed.at(index).toObject();
        if (object.value(QStringLiteral("id")).toString() == record.id
            && object.value(QStringLiteral("version")).toString() == record.version) {
            installed.replace(index, record.toJson());
            replaced = true;
            break;
        }
    }
    if (!replaced) {
        installed.append(record.toJson());
    }
    state.insert(QStringLiteral("schemaVersion"), 1);
    state.insert(QStringLiteral("updatedAt"), nowIso());
    state.insert(QStringLiteral("installed"), installed);
    return saveState(state, error);
}

bool PluginMarketplace::removeInstalledRecord(const QString& id, const QString& version, QString* error)
{
    QJsonObject state = loadState(error);
    if (error && !error->isEmpty()) {
        return false;
    }
    QJsonArray next;
    const QJsonArray installed = state.value(QStringLiteral("installed")).toArray();
    for (const QJsonValue& value : installed) {
        const QJsonObject object = value.toObject();
        if (object.value(QStringLiteral("id")).toString() == id
            && object.value(QStringLiteral("version")).toString() == version) {
            continue;
        }
        next.append(object);
    }
    state.insert(QStringLiteral("updatedAt"), nowIso());
    state.insert(QStringLiteral("installed"), next);
    return saveState(state, error);
}

QString PluginMarketplace::packageInstallPath(const QString& id, const QString& version) const
{
    QString safeId = id;
    QString safeVersion = version;
    safeId.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9_.-]")), QStringLiteral("_"));
    safeVersion.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9_.-]")), QStringLiteral("_"));
    return QDir(marketplaceRoot_).filePath(QStringLiteral("%1/%2").arg(safeId, safeVersion));
}

QString PluginMarketplace::normalizedPackageRoot(const QString& packagePath, QString* temporaryRoot, PluginMarketplaceReport* report) const
{
    if (temporaryRoot) {
        temporaryRoot->clear();
    }
    const QFileInfo info(packagePath);
    if (!info.exists()) {
        if (report) {
            report->ok = false;
            report->status = QStringLiteral("missing");
            report->message = QStringLiteral("Plugin package does not exist: %1").arg(packagePath);
            report->errors.append(report->message);
        }
        return {};
    }
    if (info.isDir()) {
        return info.absoluteFilePath();
    }

    if (!info.fileName().endsWith(QStringLiteral(".aitrain-plugin.zip"), Qt::CaseInsensitive)
        && !info.fileName().endsWith(QStringLiteral(".zip"), Qt::CaseInsensitive)) {
        if (report) {
            report->ok = false;
            report->status = QStringLiteral("unsupported-package");
            report->message = QStringLiteral("Plugin package must be an expanded directory or .aitrain-plugin.zip file.");
            report->errors.append(report->message);
        }
        return {};
    }

    QTemporaryDir tempDir;
    tempDir.setAutoRemove(false);
    if (!tempDir.isValid()) {
        if (report) {
            report->ok = false;
            report->status = QStringLiteral("temp-error");
            report->message = QStringLiteral("Cannot create temporary directory for plugin package.");
            report->errors.append(report->message);
        }
        return {};
    }
    QProcess process;
    QTemporaryDir scriptDir;
    if (!scriptDir.isValid()) {
        if (report) {
            report->ok = false;
            report->status = QStringLiteral("temp-error");
            report->message = QStringLiteral("Cannot create temporary directory for plugin package extractor.");
            report->errors.append(report->message);
        }
        removeDirectorySafely(tempDir.path(), nullptr);
        return {};
    }
    const QString scriptPath = QDir(scriptDir.path()).filePath(QStringLiteral("extract_plugin_package.ps1"));
    QFile scriptFile(scriptPath);
    if (!scriptFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (report) {
            report->ok = false;
            report->status = QStringLiteral("temp-error");
            report->message = QStringLiteral("Cannot write temporary plugin package extractor script.");
            report->errors.append(report->message);
        }
        removeDirectorySafely(tempDir.path(), nullptr);
        return {};
    }
    scriptFile.write(
        "param([Parameter(Mandatory=$true)][string]$PackagePath,"
        "[Parameter(Mandatory=$true)][string]$DestinationPath)\n"
        "Expand-Archive -LiteralPath $PackagePath -DestinationPath $DestinationPath -Force\n");
    scriptFile.close();
    process.start(QStringLiteral("powershell"),
        QStringList()
            << QStringLiteral("-NoProfile")
            << QStringLiteral("-ExecutionPolicy")
            << QStringLiteral("Bypass")
            << QStringLiteral("-File")
            << scriptPath
            << info.absoluteFilePath()
            << tempDir.path());
    if (!process.waitForStarted(5000) || !process.waitForFinished(30000)
        || process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        const QString stderrText = QString::fromLocal8Bit(process.readAllStandardError()).trimmed();
        if (report) {
            report->ok = false;
            report->status = QStringLiteral("extract-failed");
            report->message = stderrText.isEmpty()
                ? QStringLiteral("Cannot extract plugin package with PowerShell Expand-Archive.")
                : stderrText;
            report->errors.append(report->message);
        }
        removeDirectorySafely(tempDir.path(), nullptr);
        return {};
    }
    if (temporaryRoot) {
        *temporaryRoot = tempDir.path();
    }
    QString root = tempDir.path();
    if (!QFileInfo::exists(QDir(root).filePath(QStringLiteral("plugin.json")))) {
        const QFileInfoList childDirs = QDir(root).entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
        if (childDirs.size() == 1 && QFileInfo::exists(QDir(childDirs.first().absoluteFilePath()).filePath(QStringLiteral("plugin.json")))) {
            root = childDirs.first().absoluteFilePath();
        }
    }
    return root;
}

} // namespace aitrain
