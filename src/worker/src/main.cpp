#include "WorkerSession.h"

#include <QCoreApplication>
#include <QCommandLineParser>

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("aitrain_worker"));

    QCommandLineParser parser;
    parser.addHelpOption();
    QCommandLineOption serverOption(QStringLiteral("server"), QStringLiteral("QLocalServer name."), QStringLiteral("name"));
    parser.addOption(serverOption);
    parser.process(app);

    const QString serverName = parser.value(serverOption);
    if (serverName.isEmpty()) {
        qCritical("Missing --server argument.");
        return 2;
    }

    WorkerSession session;
    if (!session.connectToServer(serverName)) {
        qCritical("Failed to connect to controller.");
        return 3;
    }

    return app.exec();
}

