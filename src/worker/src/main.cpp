#include "WorkerSession.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/runtime/RuntimeCapabilityMatrix.h"
#include "aitrain/storage/ProjectStore.h"
#include "aitrain/workflow/ProjectWorkspace.h"

#include <QDir>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QTextStream>

namespace {

void writeJsonLine(const QJsonObject& object)
{
    QTextStream stream(stdout);
    stream.setCodec("UTF-8");
    stream << QString::fromUtf8(QJsonDocument(object).toJson(QJsonDocument::Compact))
           << QLatin1Char('\n');
    stream.flush();
}

int runSelfCheck()
{
    QJsonArray checks;
    const QVector<aitrain::RuntimeDependencyCheck> runtimeChecks =
        aitrain::defaultRuntimeDependencyChecks(QCoreApplication::applicationDirPath());
    bool hasMissing = false;
    bool hasWarning = false;
    for (const aitrain::RuntimeDependencyCheck& check : runtimeChecks) {
        checks.append(check.toJson());
        hasMissing = hasMissing || check.status == QStringLiteral("missing");
        hasWarning = hasWarning || check.status == QStringLiteral("warning");
    }

    QJsonObject result;
    // A missing required runtime dependency is a failed self-check, not a
    // warning-only report. Callers use the exit code to gate package startup.
    result.insert(QStringLiteral("ok"), !hasMissing);
    result.insert(QStringLiteral("status"), hasMissing
        ? QStringLiteral("missing")
        : (hasWarning ? QStringLiteral("warning") : QStringLiteral("ok")));
    result.insert(QStringLiteral("applicationDir"), QCoreApplication::applicationDirPath());
    result.insert(QStringLiteral("ncnnBackend"), aitrain::ncnnBackendStatus().toJson());
    result.insert(QStringLiteral("tensorRtBackend"), aitrain::tensorRtBackendStatus().toJson());
    result.insert(QStringLiteral("builtinCapabilities"), aitrain::BuiltinCapabilityRegistry::instance().toJson());
    result.insert(QStringLiteral("runtimeCapabilityMatrix"), aitrain::RuntimeCapabilityMatrix().toJson());
    result.insert(QStringLiteral("checks"), checks);
    writeJsonLine(result);
    return hasMissing ? 4 : 0;
}

int runBuiltinCapabilityCheck()
{
    const auto& registry = aitrain::BuiltinCapabilityRegistry::instance();
    QJsonObject result;
    result.insert(QStringLiteral("ok"), !registry.capabilities().isEmpty() && !registry.backends().isEmpty());
    result.insert(QStringLiteral("capabilityCount"), registry.capabilities().size());
    result.insert(QStringLiteral("backendCount"), registry.backends().size());
    result.insert(QStringLiteral("registry"), registry.toJson());
    writeJsonLine(result);
    return result.value(QStringLiteral("ok")).toBool() ? 0 : 4;
}

int runWorkspaceSelfCheck(const QString& requestedRoot)
{
    const QString root = QDir::cleanPath(QDir::fromNativeSeparators(requestedRoot.trimmed()));
    QJsonObject result;
    result.insert(QStringLiteral("schemaVersion"), aitrain::ProjectStore::schemaVersion());
    if (root.isEmpty()) {
        result.insert(QStringLiteral("ok"), false);
        result.insert(QStringLiteral("error"), QStringLiteral("缺少 --workspace 参数。"));
        writeJsonLine(result);
        return 2;
    }
    const QFileInfo rootInfo(root);
    if (rootInfo.exists() && !rootInfo.isDir()) {
        result.insert(QStringLiteral("ok"), false);
        result.insert(QStringLiteral("error"), QStringLiteral("workspace 必须是目录。"));
        writeJsonLine(result);
        return 2;
    }
    if (rootInfo.exists()
        && !QDir(root).entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries,
            QDir::Name).isEmpty()) {
        result.insert(QStringLiteral("ok"), false);
        result.insert(QStringLiteral("error"), QStringLiteral("workspace-self-check 只允许在空目录执行。"));
        writeJsonLine(result);
        return 2;
    }

    const auto fail = [&result](const QString& message) {
        result.insert(QStringLiteral("ok"), false);
        result.insert(QStringLiteral("error"), message);
        writeJsonLine(result);
        return 4;
    };

    aitrain::ProjectWorkspace firstWorkspace;
    QString error;
    if (!firstWorkspace.open(root, &error)) {
        return fail(QStringLiteral("首次打开工作区失败：%1").arg(error));
    }
    const QString metadataRoot = firstWorkspace.workspacePath();
    firstWorkspace.close();
    result.insert(QStringLiteral("firstOpen"), true);

    const QString databasePath = QDir(metadataRoot).filePath(QStringLiteral("project.sqlite"));
    const QString artifactRoot = QDir(metadataRoot).filePath(QStringLiteral("artifacts"));
    const QString stagingRoot = QDir(artifactRoot).filePath(QStringLiteral(".staging"));
    const QString stagingMetadataRoot = QDir(artifactRoot).filePath(QStringLiteral(".staging-meta"));
    const QString committedRoot = QDir(artifactRoot).filePath(QStringLiteral("committed"));
    const QString runtimeStagingRoot = QDir(metadataRoot).filePath(QStringLiteral(".runtime-staging"));
    const bool layoutValid = QFileInfo::exists(databasePath)
        && QFileInfo(databasePath).isFile()
        && QDir(artifactRoot).exists()
        && QDir(stagingRoot).exists()
        && QDir(stagingMetadataRoot).exists()
        && QDir(committedRoot).exists()
        && QDir(runtimeStagingRoot).exists();
    result.insert(QStringLiteral("layoutValid"), layoutValid);
    if (!layoutValid) {
        return fail(QStringLiteral("首次打开后工作区目录结构不完整。"));
    }

    const QString connectionName = QStringLiteral("aitrain_workspace_self_check_%1")
        .arg(QUuid::createUuid().toString(QUuid::Id128));
    int storedSchemaVersion = 0;
    bool hasProjectsTable = false;
    {
        QSqlDatabase database = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
        database.setDatabaseName(databasePath);
        if (!database.open()) {
            return fail(QStringLiteral("无法读取首启数据库：%1").arg(database.lastError().text()));
        }
        QSqlQuery schemaQuery(database);
        if (!schemaQuery.exec(QStringLiteral("select version from schema_info limit 1"))
            || !schemaQuery.next()) {
            return fail(QStringLiteral("首启数据库缺少 schema 版本记录。"));
        }
        storedSchemaVersion = schemaQuery.value(0).toInt();
        QSqlQuery projectsQuery(database);
        if (!projectsQuery.exec(QStringLiteral(
                "select 1 from sqlite_master where type = 'table' and name = 'projects'"))) {
            return fail(QStringLiteral("无法检查 projects 死表：%1").arg(projectsQuery.lastError().text()));
        }
        hasProjectsTable = projectsQuery.next();
        database.close();
    }
    QSqlDatabase::removeDatabase(connectionName);
    result.insert(QStringLiteral("storedSchemaVersion"), storedSchemaVersion);
    result.insert(QStringLiteral("projectsTablePresent"), hasProjectsTable);
    if (storedSchemaVersion != aitrain::ProjectStore::schemaVersion() || hasProjectsTable) {
        return fail(QStringLiteral("首启数据库 schema 或死表检查失败。"));
    }

    aitrain::ProjectWorkspace secondWorkspace;
    if (!secondWorkspace.open(root, &error)) {
        return fail(QStringLiteral("关闭后再次打开工作区失败：%1").arg(error));
    }
    secondWorkspace.close();
    result.insert(QStringLiteral("secondOpen"), true);

    const bool stagingClean = QDir(stagingRoot).entryInfoList(
        QDir::NoDotAndDotDot | QDir::AllEntries, QDir::Name).isEmpty()
        && QDir(stagingMetadataRoot).entryInfoList(
            QDir::NoDotAndDotDot | QDir::AllEntries, QDir::Name).isEmpty()
        && QDir(runtimeStagingRoot).entryInfoList(
            QDir::NoDotAndDotDot | QDir::AllEntries, QDir::Name).isEmpty();
    result.insert(QStringLiteral("stagingClean"), stagingClean);
    result.insert(QStringLiteral("ok"), stagingClean);
    if (!stagingClean) {
        result.insert(QStringLiteral("error"), QStringLiteral("首启/二次打开后发现未清理的 staging。"));
        writeJsonLine(result);
        return 4;
    }
    writeJsonLine(result);
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("aitrain_worker"));

    QCommandLineParser parser;
    parser.addHelpOption();
    QCommandLineOption serverOption(
        QStringLiteral("server"), QStringLiteral("Protocol  本地服务名。"), QStringLiteral("name"));
    QCommandLineOption requestIdOption(
        QStringLiteral("request-id"), QStringLiteral("Protocol  RequestId。"), QStringLiteral("uuid"));
    QCommandLineOption taskIdOption(
        QStringLiteral("task-id"), QStringLiteral("Protocol  TaskId。"), QStringLiteral("uuid"));
    QCommandLineOption controlTokenOption(
        QStringLiteral("control-token"), QStringLiteral("Protocol  控制令牌。"), QStringLiteral("token"));
    QCommandLineOption selfCheckOption(
        QStringLiteral("self-check"), QStringLiteral("执行运行时依赖自检并输出 JSON。"));
    QCommandLineOption capabilityCheckOption(
        QStringLiteral("builtin-capabilities"), QStringLiteral("输出内置能力注册表。"));
    QCommandLineOption workspaceSelfCheckOption(
        QStringLiteral("workspace-self-check"), QStringLiteral("验证空项目首次打开、二次打开及工作区目录。"));
    QCommandLineOption workspaceOption(
        QStringLiteral("workspace"), QStringLiteral("workspace-self-check 使用的空项目目录。"), QStringLiteral("path"));
    parser.addOption(serverOption);
    parser.addOption(requestIdOption);
    parser.addOption(taskIdOption);
    parser.addOption(controlTokenOption);
    parser.addOption(selfCheckOption);
    parser.addOption(capabilityCheckOption);
    parser.addOption(workspaceSelfCheckOption);
    parser.addOption(workspaceOption);
    parser.process(app);

    if (parser.isSet(selfCheckOption)) {
        return runSelfCheck();
    }
    if (parser.isSet(capabilityCheckOption)) {
        return runBuiltinCapabilityCheck();
    }
    if (parser.isSet(workspaceSelfCheckOption)) {
        return runWorkspaceSelfCheck(parser.value(workspaceOption));
    }

    const QString serverName = parser.value(serverOption);
    if (serverName.isEmpty()) {
        qCritical("缺少 --server 参数。");
        return 2;
    }

    aitrain::RequestId requestId;
    aitrain::TaskId taskId;
    QString identityError;
    if (!aitrain::RequestId::parse(parser.value(requestIdOption), &requestId, &identityError)) {
        qCritical().noquote() << QStringLiteral("无效或缺少 --request-id：%1").arg(identityError);
        return 2;
    }
    if (!aitrain::TaskId::parse(parser.value(taskIdOption), &taskId, &identityError)) {
        qCritical().noquote() << QStringLiteral("无效或缺少 --task-id：%1").arg(identityError);
        return 2;
    }
    const QString controlToken = parser.value(controlTokenOption).trimmed();
    if (controlToken.isEmpty()) {
        qCritical("缺少 --control-token 参数。");
        return 2;
    }

    WorkerSession session;
    if (!session.connectToServer(serverName, requestId, taskId, controlToken)) {
        qCritical("无法连接到控制端。");
        return 3;
    }
    return app.exec();
}
