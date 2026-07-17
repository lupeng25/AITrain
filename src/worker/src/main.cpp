#include "WorkerSession.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/runtime/RuntimeCapabilityMatrix.h"

#include <QCommandLineParser>
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
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
    parser.addOption(serverOption);
    parser.addOption(requestIdOption);
    parser.addOption(taskIdOption);
    parser.addOption(controlTokenOption);
    parser.addOption(selfCheckOption);
    parser.addOption(capabilityCheckOption);
    parser.process(app);

    if (parser.isSet(selfCheckOption)) {
        return runSelfCheck();
    }
    if (parser.isSet(capabilityCheckOption)) {
        return runBuiltinCapabilityCheck();
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
