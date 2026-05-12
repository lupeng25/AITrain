#include "TestSupport.h"

class PlatformTests : public QObject {
    Q_OBJECT

private slots:
    void taskStateTransitions()
    {
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Running));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Queued, aitrain::TaskState::Failed));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Paused));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Paused, aitrain::TaskState::Running));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Paused, aitrain::TaskState::Canceled));
        QVERIFY(aitrain::isValidTaskStateTransition(aitrain::TaskState::Running, aitrain::TaskState::Completed));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Completed, aitrain::TaskState::Running));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Failed, aitrain::TaskState::Running));
        QVERIFY(!aitrain::isValidTaskStateTransition(aitrain::TaskState::Canceled, aitrain::TaskState::Running));
    }

    void protocolRoundTrip()
    {
        QJsonObject payload;
        payload.insert(QStringLiteral("taskId"), QStringLiteral("abc"));
        payload.insert(QStringLiteral("value"), 42);

        const QByteArray encoded = aitrain::protocol::encodeMessage(QStringLiteral("metric"), payload, QStringLiteral("req-1"));
        QVERIFY(encoded.endsWith('\n'));

        QString type;
        QJsonObject decodedPayload;
        QString requestId;
        QString error;
        QVERIFY(aitrain::protocol::decodeMessage(encoded, &type, &decodedPayload, &requestId, &error));
        QCOMPARE(type, QStringLiteral("metric"));
        QCOMPARE(requestId, QStringLiteral("req-1"));
        QCOMPARE(decodedPayload.value(QStringLiteral("taskId")).toString(), QStringLiteral("abc"));
        QCOMPARE(decodedPayload.value(QStringLiteral("value")).toInt(), 42);
        QVERIFY(error.isEmpty());
    }

    void pluginMarketplaceParsesIndexAndInstallsExpandedPackage()
    {
        const QString pluginDll = builtPluginPath(QStringLiteral("DatasetInteropPlugin.dll"));
        if (pluginDll.isEmpty()) {
            QSKIP("Built DatasetInteropPlugin.dll is not available for marketplace fixture.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString packageRoot = writeMarketplacePackageFixture(
            dir.path(),
            pluginDll,
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            QStringLiteral("0.1.1"));
        QVERIFY(!packageRoot.isEmpty());

        const QString indexPath = dir.filePath(QStringLiteral("marketplace.json"));
        QJsonObject index;
        index.insert(QStringLiteral("schemaVersion"), 1);
        index.insert(QStringLiteral("plugins"), QJsonArray{
            QJsonObject{
                {QStringLiteral("id"), QStringLiteral("com.aitrain.plugins.dataset_interop")},
                {QStringLiteral("name"), QStringLiteral("Dataset Interop")},
                {QStringLiteral("version"), QStringLiteral("0.1.1")},
                {QStringLiteral("description"), QStringLiteral("Dataset marketplace fixture")},
                {QStringLiteral("publisher"), QStringLiteral("AITrain Tests")},
                {QStringLiteral("license"), QStringLiteral("Test")},
                {QStringLiteral("category"), QStringLiteral("dataset_interop")},
                {QStringLiteral("capabilities"), QJsonArray{QStringLiteral("dataset_interop")}},
                {QStringLiteral("minAitrainVersion"), QStringLiteral("0.1.0")},
                {QStringLiteral("qtAbi"), aitrain::PluginMarketplace::currentQtAbi()},
                {QStringLiteral("downloadUrl"), packageRoot}
            }
        });
        QFile indexFile(indexPath);
        QVERIFY(indexFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
        indexFile.write(QJsonDocument(index).toJson(QJsonDocument::Indented));
        indexFile.close();

        aitrain::PluginMarketplace marketplace(
            dir.filePath(QStringLiteral("marketplace")),
            dir.filePath(QStringLiteral("plugins/models")),
            dir.filePath(QStringLiteral("plugin_marketplace_state.json")));

        aitrain::PluginMarketplaceReport indexReport;
        const QVector<aitrain::MarketplacePluginEntry> entries = marketplace.loadIndex(indexPath, &indexReport);
        QVERIFY2(indexReport.ok, qPrintable(indexReport.message));
        QCOMPARE(entries.size(), 1);
        QCOMPARE(entries.first().installedState, QStringLiteral("available"));

        aitrain::PluginPackageManifest manifest;
        const aitrain::PluginMarketplaceReport inspectReport = marketplace.inspectPackage(packageRoot, &manifest);
        QVERIFY2(inspectReport.ok, qPrintable(inspectReport.errors.join(QStringLiteral("\n"))));
        QCOMPARE(manifest.id, QStringLiteral("com.aitrain.plugins.dataset_interop"));

        const aitrain::PluginMarketplaceReport installReport = marketplace.installPackage(packageRoot, true);
        QVERIFY2(installReport.ok, qPrintable(installReport.errors.join(QStringLiteral("\n"))));
        QCOMPARE(installReport.status, QStringLiteral("enabled"));

        const QVector<aitrain::InstalledPluginRecord> installed = marketplace.installedPlugins();
        QCOMPARE(installed.size(), 1);
        QVERIFY(installed.first().enabled);
        QVERIFY(QFileInfo::exists(QDir(marketplace.activePluginDirectory()).filePath(QStringLiteral("DatasetInteropPlugin.dll"))));

        QProcess smoke;
        smoke.start(workerExecutablePath(), QStringList() << QStringLiteral("--plugin-smoke") << marketplace.activePluginDirectory());
        QVERIFY(smoke.waitForStarted(5000));
        QVERIFY(smoke.waitForFinished(15000));
        QCOMPARE(smoke.exitCode(), 4);
        const QJsonObject smokeJson = QJsonDocument::fromJson(smoke.readAllStandardOutput().trimmed()).object();
        QCOMPARE(smokeJson.value(QStringLiteral("pluginCount")).toInt(), 1);

        const aitrain::PluginMarketplaceReport disableReport = marketplace.disablePlugin(QStringLiteral("com.aitrain.plugins.dataset_interop"));
        QVERIFY2(disableReport.ok, qPrintable(disableReport.message));
        QVERIFY(!QFileInfo::exists(QDir(marketplace.activePluginDirectory()).filePath(QStringLiteral("DatasetInteropPlugin.dll"))));

        const aitrain::PluginMarketplaceReport uninstallReport = marketplace.uninstallPlugin(QStringLiteral("com.aitrain.plugins.dataset_interop"), QStringLiteral("0.1.1"));
        QVERIFY2(uninstallReport.ok, qPrintable(uninstallReport.message));
        QVERIFY(marketplace.installedPlugins().isEmpty());
    }

    void pluginMarketplaceRejectsHashMismatch()
    {
        const QString pluginDll = builtPluginPath(QStringLiteral("DatasetInteropPlugin.dll"));
        if (pluginDll.isEmpty()) {
            QSKIP("Built DatasetInteropPlugin.dll is not available for marketplace fixture.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString packageRoot = writeMarketplacePackageFixture(
            dir.path(),
            pluginDll,
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            QStringLiteral("0.1.2"));
        QVERIFY(!packageRoot.isEmpty());

        QJsonObject manifest = readJsonObject(QDir(packageRoot).filePath(QStringLiteral("plugin.json")));
        const QString relativeDll = manifest.value(QStringLiteral("entrypoints")).toObject().value(QStringLiteral("qtModelPlugin")).toString();
        manifest.insert(QStringLiteral("hashes"), QJsonObject{{relativeDll, QStringLiteral("0000")}});
        QFile manifestFile(QDir(packageRoot).filePath(QStringLiteral("plugin.json")));
        QVERIFY(manifestFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
        manifestFile.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented));
        manifestFile.close();

        aitrain::PluginMarketplace marketplace(
            dir.filePath(QStringLiteral("marketplace")),
            dir.filePath(QStringLiteral("plugins/models")),
            dir.filePath(QStringLiteral("plugin_marketplace_state.json")));
        const aitrain::PluginMarketplaceReport report = marketplace.inspectPackage(packageRoot);
        QVERIFY(!report.ok);
        QCOMPARE(report.status, QStringLiteral("hash-mismatch"));
    }

    void pluginMarketplaceKeepsStateWhenDisableCannotRemoveActiveFile()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString marketplaceRoot = dir.filePath(QStringLiteral("marketplace"));
        const QString activeDir = dir.filePath(QStringLiteral("plugins/models"));
        const QString statePath = dir.filePath(QStringLiteral("plugin_marketplace_state.json"));
        QVERIFY(QDir().mkpath(marketplaceRoot));
        QVERIFY(QDir().mkpath(activeDir));

        const QString blockedActivePath = QDir(activeDir).filePath(QStringLiteral("blocked-active-entry"));
        QVERIFY(QDir().mkpath(blockedActivePath));

        QJsonObject manifest;
        manifest.insert(QStringLiteral("schemaVersion"), 1);
        manifest.insert(QStringLiteral("id"), QStringLiteral("com.aitrain.plugins.dataset_interop"));
        manifest.insert(QStringLiteral("name"), QStringLiteral("Marketplace Fixture"));
        manifest.insert(QStringLiteral("version"), QStringLiteral("0.1.4"));
        manifest.insert(QStringLiteral("entrypoints"), QJsonObject{{QStringLiteral("qtModelPlugin"), QStringLiteral("payload/plugins/models/DatasetInteropPlugin.dll")}});

        QJsonObject record;
        record.insert(QStringLiteral("id"), QStringLiteral("com.aitrain.plugins.dataset_interop"));
        record.insert(QStringLiteral("name"), QStringLiteral("Marketplace Fixture"));
        record.insert(QStringLiteral("version"), QStringLiteral("0.1.4"));
        record.insert(QStringLiteral("enabled"), true);
        record.insert(QStringLiteral("installPath"), QDir(marketplaceRoot).filePath(QStringLiteral("com.aitrain.plugins.dataset_interop/0.1.4")));
        record.insert(QStringLiteral("sourcePath"), QStringLiteral("fixture"));
        record.insert(QStringLiteral("installedAt"), QStringLiteral("2026-05-12T00:00:00.000Z"));
        record.insert(QStringLiteral("verificationStatus"), QStringLiteral("verified-local-unsigned"));
        record.insert(QStringLiteral("message"), QStringLiteral("Plugin enabled."));
        record.insert(QStringLiteral("activeFiles"), QJsonArray{blockedActivePath});
        record.insert(QStringLiteral("packageManifest"), manifest);

        QFile stateFile(statePath);
        QVERIFY(stateFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
        stateFile.write(QJsonDocument(QJsonObject{
            {QStringLiteral("schemaVersion"), 1},
            {QStringLiteral("installed"), QJsonArray{record}},
            {QStringLiteral("updatedAt"), QStringLiteral("2026-05-12T00:00:00.000Z")}
        }).toJson(QJsonDocument::Indented));
        stateFile.close();

        aitrain::PluginMarketplace marketplace(marketplaceRoot, activeDir, statePath);
        const aitrain::PluginMarketplaceReport report =
            marketplace.disablePlugin(QStringLiteral("com.aitrain.plugins.dataset_interop"));
        QVERIFY(!report.ok);
        QCOMPARE(report.status, QStringLiteral("disable-failed"));
        QVERIFY(report.errors.join(QStringLiteral("\n")).contains(blockedActivePath));

        const QVector<aitrain::InstalledPluginRecord> installed = marketplace.installedPlugins();
        QCOMPARE(installed.size(), 1);
        QVERIFY(installed.first().enabled);
        QCOMPARE(installed.first().activeFiles, QStringList{blockedActivePath});
        QVERIFY(QFileInfo(blockedActivePath).isDir());
    }

    void pluginMarketplaceInstallsZipPackage()
    {
        const QString pluginDll = builtPluginPath(QStringLiteral("DatasetInteropPlugin.dll"));
        if (pluginDll.isEmpty()) {
            QSKIP("Built DatasetInteropPlugin.dll is not available for marketplace fixture.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString packageRoot = writeMarketplacePackageFixture(
            dir.path(),
            pluginDll,
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            QStringLiteral("0.1.3"));
        QVERIFY(!packageRoot.isEmpty());

        const QString zipPath = dir.filePath(QStringLiteral("DatasetInterop.aitrain-plugin.zip"));
        QString zipError;
        if (!zipDirectoryForTest(packageRoot, zipPath, &zipError)) {
            QSKIP(qPrintable(QStringLiteral("Cannot create zip package fixture: %1").arg(zipError)));
        }

        aitrain::PluginMarketplace marketplace(
            dir.filePath(QStringLiteral("marketplace")),
            dir.filePath(QStringLiteral("plugins/models")),
            dir.filePath(QStringLiteral("plugin_marketplace_state.json")));

        aitrain::PluginPackageManifest manifest;
        const aitrain::PluginMarketplaceReport inspectReport = marketplace.inspectPackage(zipPath, &manifest);
        QVERIFY2(inspectReport.ok, qPrintable(inspectReport.errors.join(QStringLiteral("\n"))));
        QCOMPARE(manifest.id, QStringLiteral("com.aitrain.plugins.dataset_interop"));
        QVERIFY(!inspectReport.details.value(QStringLiteral("temporaryRoot")).toString().isEmpty());

        const aitrain::PluginMarketplaceReport installReport = marketplace.installPackage(zipPath, true);
        QVERIFY2(installReport.ok, qPrintable(installReport.errors.join(QStringLiteral("\n"))));
        QCOMPARE(installReport.status, QStringLiteral("enabled"));
        QVERIFY(QFileInfo::exists(QDir(marketplace.activePluginDirectory()).filePath(QStringLiteral("DatasetInteropPlugin.dll"))));

        const QVector<aitrain::InstalledPluginRecord> installed = marketplace.installedPlugins();
        QCOMPARE(installed.size(), 1);
        QCOMPARE(installed.first().sourcePath, QFileInfo(zipPath).absoluteFilePath());
        QVERIFY(installed.first().enabled);
    }

    void offlineLicenseTokensValidate()
    {
        if (!aitrain::licenseCryptoAvailable()) {
            QSKIP("Offline ECDSA license tests require the Windows CNG implementation");
        }

        QString error;
        aitrain::LicenseKeyPair keyPair;
        QVERIFY2(aitrain::generateLicenseKeyPair(&keyPair, &error), qPrintable(error));
        QVERIFY(!keyPair.publicKeyBase64.isEmpty());
        QVERIFY(!keyPair.privateKeyBase64.isEmpty());
        QCOMPARE(aitrain::publicKeyFromPrivateKey(keyPair.privateKeyBase64, &error), keyPair.publicKeyBase64);

        const QDateTime now = QDateTime::fromString(QStringLiteral("2026-05-01T00:00:00Z"), Qt::ISODate);
        aitrain::LicensePayload payload;
        payload.product = aitrain::licenseProductName();
        payload.customer = QStringLiteral("Smoke Customer");
        payload.machineCode = QStringLiteral("ABCD-EF12-3456-7890-CAFE");
        payload.licenseId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        payload.issuedAt = now.addDays(-1);
        payload.expiresAt = now.addDays(30);

        const QString token = aitrain::createLicenseToken(payload, keyPair.privateKeyBase64, &error);
        QVERIFY2(!token.isEmpty(), qPrintable(error));

        const aitrain::LicenseValidationResult valid =
            aitrain::validateLicenseToken(token, keyPair.publicKeyBase64, payload.machineCode, now);
        QVERIFY2(valid.isValid(), qPrintable(valid.message));
        QCOMPARE(valid.payload.customer, payload.customer);

        const aitrain::LicenseValidationResult mismatch =
            aitrain::validateLicenseToken(token, keyPair.publicKeyBase64, QStringLiteral("0000-0000-0000-0000-0000"), now);
        QCOMPARE(static_cast<int>(mismatch.status), static_cast<int>(aitrain::LicenseStatus::MachineMismatch));

        const aitrain::LicenseValidationResult tampered =
            aitrain::validateLicenseToken(tokenWithCustomer(token, QStringLiteral("Mallory")), keyPair.publicKeyBase64, payload.machineCode, now);
        QCOMPARE(static_cast<int>(tampered.status), static_cast<int>(aitrain::LicenseStatus::SignatureInvalid));

        aitrain::LicensePayload expiredPayload = payload;
        expiredPayload.licenseId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        expiredPayload.expiresAt = now.addSecs(-1);
        const QString expiredToken = aitrain::createLicenseToken(expiredPayload, keyPair.privateKeyBase64, &error);
        QVERIFY2(!expiredToken.isEmpty(), qPrintable(error));
        const aitrain::LicenseValidationResult expired =
            aitrain::validateLicenseToken(expiredToken, keyPair.publicKeyBase64, payload.machineCode, now);
        QCOMPARE(static_cast<int>(expired.status), static_cast<int>(aitrain::LicenseStatus::Expired));

        QCOMPARE(static_cast<int>(aitrain::validateLicenseToken(QString(), keyPair.publicKeyBase64, payload.machineCode, now).status),
            static_cast<int>(aitrain::LicenseStatus::MissingToken));
        QCOMPARE(static_cast<int>(aitrain::validateLicenseToken(QStringLiteral("bad-token"), keyPair.publicKeyBase64, payload.machineCode, now).status),
            static_cast<int>(aitrain::LicenseStatus::MalformedToken));
    }

    void packagingLayoutUsesPhaseSevenInstallShape()
    {
        const QString root = QDir::cleanPath(QDir(QDir::tempPath()).filePath(QStringLiteral("AITrainStudioPackage")));
        const aitrain::PackagingLayout layout = aitrain::packagingLayoutForRoot(root);
        const QString appExecutableName =
#ifdef Q_OS_WIN
            QStringLiteral("AITrainStudio.exe");
#else
            QStringLiteral("AITrainStudio");
#endif
        const QString workerExecutableName =
#ifdef Q_OS_WIN
            QStringLiteral("aitrain_worker.exe");
#else
            QStringLiteral("aitrain_worker");
#endif

        QCOMPARE(layout.rootPath, root);
        QCOMPARE(layout.appExecutablePath, QDir(root).filePath(appExecutableName));
        QCOMPARE(layout.workerExecutablePath, QDir(root).filePath(workerExecutableName));
        QCOMPARE(layout.pluginModelsDirectory, QDir(root).filePath(QStringLiteral("plugins/models")));
        QCOMPARE(layout.onnxRuntimeDirectory, QDir(root).filePath(QStringLiteral("runtimes/onnxruntime")));
        QCOMPARE(layout.tensorRtRuntimeDirectory, QDir(root).filePath(QStringLiteral("runtimes/tensorrt")));
        QCOMPARE(layout.examplesDirectory, QDir(root).filePath(QStringLiteral("examples")));
        QCOMPARE(layout.docsDirectory, QDir(root).filePath(QStringLiteral("docs")));
        QCOMPARE(layout.toJson().value(QStringLiteral("appExecutablePath")).toString(), layout.appExecutablePath);
    }

    void runtimeDependencyCheckReportsSearchPaths()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QStringList paths = aitrain::runtimeSearchPaths(dir.path());
        QVERIFY(paths.contains(QDir(dir.path()).filePath(QStringLiteral("runtimes/onnxruntime"))));
        QVERIFY(paths.contains(QDir(dir.path()).filePath(QStringLiteral("runtimes/tensorrt"))));

        const aitrain::RuntimeDependencyCheck check = aitrain::checkRuntimeDependency(
            QStringLiteral("Missing Runtime"),
            QStringList() << QStringLiteral("aitrain_definitely_missing_runtime"),
            QStringLiteral("expected missing runtime message."),
            dir.path());
        QCOMPARE(check.status, QStringLiteral("missing"));
        QVERIFY(check.message.contains(QStringLiteral("Missing Runtime")));
        QVERIFY(check.message.contains(QStringLiteral("expected missing runtime message")));

        const QString json = QString::fromUtf8(QJsonDocument(check.toJson()).toJson(QJsonDocument::Compact));
        QVERIFY(json.contains(QStringLiteral("runtimes/onnxruntime")));
        QVERIFY(json.contains(QStringLiteral("runtimes/tensorrt")));
        QVERIFY(json.contains(QStringLiteral("aitrain_definitely_missing_runtime")));
    }

};

QTEST_MAIN(PlatformTests)
#include "tst_platform.moc"
