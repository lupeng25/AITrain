#include "aitrain/v2/AdapterEventServerV2.h"
#include "aitrain/v2/ProcessTreeSupervisor.h"
#include "aitrain/v2/PythonAdapterHostV2.h"

#include <QJsonDocument>
#include <QFile>
#include <QProcess>
#include <QStandardPaths>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTest>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

QString pythonExecutable()
{
    const QString configured = qEnvironmentVariable("PYTHON");
    return configured.isEmpty() ? QStandardPaths::findExecutable(QStringLiteral("python.exe")) : configured;
}

QString writeProcessTreeFixture(const QTemporaryDir& directory)
{
    const QString path = directory.filePath(QStringLiteral("process_tree_fixture.py"));
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return {};
    }
    file.write(
        "import os, signal, subprocess, sys, time\n"
        "pid_path, mode = sys.argv[1], sys.argv[2]\n"
        "time.sleep(0.35)\n"
        "child = 'import os,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], \\\"w\\\", encoding=\\\"ascii\\\").write(str(os.getpid())); time.sleep(60)'\n"
        "subprocess.Popen([sys.executable, '-c', child, pid_path])\n"
        "deadline = time.time() + 5\n"
        "while not os.path.exists(pid_path) and time.time() < deadline: time.sleep(0.01)\n"
        "if mode == 'crash': os._exit(23)\n"
        "time.sleep(60)\n");
    file.close();
    return path;
}

qint64 readPid(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return 0;
    }
    bool ok = false;
    const qint64 pid = file.readAll().trimmed().toLongLong(&ok);
    return ok ? pid : 0;
}

bool processIsAlive(qint64 pid)
{
#ifdef Q_OS_WIN
    HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (!handle) {
        return false;
    }
    DWORD exitCode = 0;
    const bool alive = GetExitCodeProcess(handle, &exitCode) && exitCode == STILL_ACTIVE;
    CloseHandle(handle);
    return alive;
#else
    Q_UNUSED(pid)
    return false;
#endif
}

aitrain::v2::PythonAdapterLaunchV2 fixtureLaunch(const QString& python,
    const QString& fixture,
    const QString& pidPath,
    const QString& mode)
{
    aitrain::v2::PythonAdapterLaunchV2 launch;
    launch.program = python;
    launch.arguments = QStringList{fixture, pidPath, mode};
    launch.cancellationGraceMs = 50;
    return launch;
}

} // namespace

class V2ProcessTreeTests : public QObject {
    Q_OBJECT

private slots:
    void jobObjectOwnsAndTerminatesAttachedProcessTree();
    void adapterEventChannelAuthenticatesAndForwardsBoundEvent();
    void adapterEventChannelRejectsBadTokenAndCrossTaskEvent();
    void adapterEventChannelRejectsUnterminatedOversizedBuffer();
    void pythonAdapterHostInjectsEnvironmentAndForcesCancellation();
    void cancellationKillsGrandchildThatIgnoresCooperativeSignal();
    void adapterCrashCannotLeaveGrandchildRunning();
    void hostDestructionCannotLeaveProcessTreeRunning();
    void supervisorCloseSimulatesWorkerKillAndKillsDescendants();
};

void V2ProcessTreeTests::jobObjectOwnsAndTerminatesAttachedProcessTree()
{
#ifdef Q_OS_WIN
    aitrain::v2::ProcessTreeSupervisor supervisor;
    QString error;
    QVERIFY2(supervisor.create(&error), qPrintable(error));
    QVERIFY(supervisor.isCreated());

    QProcess process;
    process.start(QStringLiteral("cmd.exe"), QStringList() << QStringLiteral("/c") << QStringLiteral("ping 127.0.0.1 -n 20 > nul"));
    QVERIFY2(process.waitForStarted(3000), qPrintable(process.errorString()));
    QVERIFY2(supervisor.attach(&process, &error), qPrintable(error));

    aitrain::v2::ProcessTreeStats stats;
    QVERIFY2(supervisor.statistics(&stats, &error), qPrintable(error));
    QVERIFY(stats.activeProcessCount >= 1);
    QVERIFY2(supervisor.terminate(&error), qPrintable(error));
    QVERIFY(process.waitForFinished(5000));
#else
    QSKIP("Windows Job Object only");
#endif
}

void V2ProcessTreeTests::adapterEventChannelAuthenticatesAndForwardsBoundEvent()
{
    const aitrain::v2::RequestId requestId = aitrain::v2::RequestId::create();
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::AdapterEventServerV2 server;
    QList<aitrain::v2::ProtocolEnvelope> received;
    server.setEventHandler([&received](const aitrain::v2::ProtocolEnvelope& event) {
        received.append(event);
    });
    QString error;
    QVERIFY2(server.start(requestId, taskId, &error), qPrintable(error));
    QVERIFY(server.isListening());
    const aitrain::v2::AdapterEventEndpointV2 endpoint = server.endpoint();
    QCOMPARE(endpoint.host, QStringLiteral("127.0.0.1"));
    QVERIFY(endpoint.port > 0);
    QVERIFY(endpoint.token.size() >= 16);

    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    const QByteArray handshake = QByteArrayLiteral("{\"channel\":\"aitrain.adapter.v2\",\"token\":\"")
        + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n");
    QCOMPARE(socket.write(handshake), handshake.size());
    QVERIFY(socket.waitForBytesWritten(3000));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    const QJsonObject reply = QJsonDocument::fromJson(socket.readLine()).object();
    QCOMPARE(reply.value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));

    aitrain::v2::ProtocolEnvelope event;
    event.messageId = aitrain::v2::MessageId::create();
    event.requestId = requestId;
    event.taskId = taskId;
    event.sequence = 1;
    event.kind = QStringLiteral("event.progress");
    event.timestamp = QDateTime::currentDateTimeUtc();
    event.payload = QJsonObject{{QStringLiteral("percent"), 10}};
    const QByteArray encoded = aitrain::v2::encodeProtocolV2Message(event, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    QCOMPARE(socket.write(encoded), encoded.size());
    QVERIFY(socket.waitForBytesWritten(3000));
    QTRY_COMPARE(received.size(), 1);
    QCOMPARE(received.constFirst().messageId.toString(), event.messageId.toString());
    QCOMPARE(received.constFirst().payload, event.payload);
}

void V2ProcessTreeTests::adapterEventChannelRejectsBadTokenAndCrossTaskEvent()
{
    const aitrain::v2::RequestId requestId = aitrain::v2::RequestId::create();
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::AdapterEventServerV2 server;
    QString error;
    QVERIFY2(server.start(requestId, taskId, &error), qPrintable(error));
    const aitrain::v2::AdapterEventEndpointV2 endpoint = server.endpoint();

    QTcpSocket badTokenSocket;
    badTokenSocket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(badTokenSocket.waitForConnected(3000));
    badTokenSocket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter.v2\",\"token\":\"incorrect-token\"}\n"));
    QTRY_VERIFY(badTokenSocket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(badTokenSocket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("rejected"));
    QTRY_COMPARE(badTokenSocket.state(), QAbstractSocket::UnconnectedState);

    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    const QByteArray handshake = QByteArrayLiteral("{\"channel\":\"aitrain.adapter.v2\",\"token\":\"")
        + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n");
    socket.write(handshake);
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));

    aitrain::v2::ProtocolEnvelope crossTask;
    crossTask.messageId = aitrain::v2::MessageId::create();
    crossTask.requestId = requestId;
    crossTask.taskId = aitrain::v2::TaskId::create();
    crossTask.sequence = 1;
    crossTask.kind = QStringLiteral("event.log");
    crossTask.timestamp = QDateTime::currentDateTimeUtc();
    crossTask.payload = QJsonObject{{QStringLiteral("message"), QStringLiteral("wrong task")}};
    const QByteArray encoded = aitrain::v2::encodeProtocolV2Message(crossTask, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    socket.write(encoded);
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("code")).toString(), QStringLiteral("protocol_violation"));
    QTRY_COMPARE(socket.state(), QAbstractSocket::UnconnectedState);
    QVERIFY(server.lastError().contains(QStringLiteral("active request/task")));
}

void V2ProcessTreeTests::adapterEventChannelRejectsUnterminatedOversizedBuffer()
{
    const aitrain::v2::RequestId requestId = aitrain::v2::RequestId::create();
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::AdapterEventServerV2 server;
    QString error;
    QVERIFY2(server.start(requestId, taskId, &error), qPrintable(error));
    const aitrain::v2::AdapterEventEndpointV2 endpoint = server.endpoint();
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    const QByteArray oversized(aitrain::v2::kProtocolV2MaxControlMessageBytes + 1, 'x');
    QCOMPARE(socket.write(oversized), oversized.size());
    QVERIFY(socket.waitForBytesWritten(5000));
    QTRY_COMPARE_WITH_TIMEOUT(socket.state(), QAbstractSocket::UnconnectedState, 7000);
    QVERIFY(server.lastError().contains(QStringLiteral("message_too_large")));
}

void V2ProcessTreeTests::pythonAdapterHostInjectsEnvironmentAndForcesCancellation()
{
#ifdef Q_OS_WIN
    aitrain::v2::PythonAdapterLaunchV2 launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments.append(QStringLiteral("/c"));
    launch.arguments.append(QStringLiteral("if \"%AITRAIN_EVENT_PORT%\"==\"\" (exit /b 9) else (ping 127.0.0.1 -n 20 > nul)"));
    launch.cancellationGraceMs = 50;

    aitrain::v2::PythonAdapterHostV2 host;
    QList<aitrain::v2::PythonAdapterExitV2> exits;
    QString error;
    QVERIFY2(host.start(launch, aitrain::v2::RequestId::create(), aitrain::v2::TaskId::create(), {},
        [&exits](const aitrain::v2::PythonAdapterExitV2& outcome) { exits.append(outcome); }, &error), qPrintable(error));
    QVERIFY(host.endpoint().port > 0);
    QVERIFY2(host.requestCancellation(&error), qPrintable(error));
    QTRY_COMPARE(exits.size(), 1);
    QVERIFY(exits.constFirst().cancelRequested);
    QVERIFY(exits.constFirst().forceTerminated);
    QVERIFY(!host.isRunning());
    QVERIFY(exits.constFirst().exitCode != 9);
#else
    QSKIP("V2 Python Adapter Host cancellation integration currently uses Windows Job Object.");
#endif
}

void V2ProcessTreeTests::cancellationKillsGrandchildThatIgnoresCooperativeSignal()
{
#ifdef Q_OS_WIN
    const QString python = pythonExecutable();
    if (python.isEmpty()) QSKIP("Python executable is required for process-tree integration.");
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString fixture = writeProcessTreeFixture(directory);
    QVERIFY(!fixture.isEmpty());
    const QString pidPath = directory.filePath(QStringLiteral("grandchild.pid"));

    aitrain::v2::PythonAdapterHostV2 host;
    QList<aitrain::v2::PythonAdapterExitV2> exits;
    QString error;
    QVERIFY2(host.start(fixtureLaunch(python, fixture, pidPath, QStringLiteral("live")),
        aitrain::v2::RequestId::create(), aitrain::v2::TaskId::create(), {},
        [&exits](const aitrain::v2::PythonAdapterExitV2& outcome) { exits.append(outcome); }, &error), qPrintable(error));
    QTRY_VERIFY_WITH_TIMEOUT(readPid(pidPath) > 0, 7000);
    const qint64 grandchildPid = readPid(pidPath);
    QVERIFY(processIsAlive(grandchildPid));
    QVERIFY2(host.requestCancellation(&error), qPrintable(error));
    QTRY_COMPARE_WITH_TIMEOUT(exits.size(), 1, 7000);
    QTRY_VERIFY_WITH_TIMEOUT(!processIsAlive(grandchildPid), 7000);
    QVERIFY(exits.constFirst().cancelRequested);
    QVERIFY(exits.constFirst().forceTerminated);
#else
    QSKIP("Windows Job Object only");
#endif
}

void V2ProcessTreeTests::adapterCrashCannotLeaveGrandchildRunning()
{
#ifdef Q_OS_WIN
    const QString python = pythonExecutable();
    if (python.isEmpty()) QSKIP("Python executable is required for process-tree integration.");
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString fixture = writeProcessTreeFixture(directory);
    const QString pidPath = directory.filePath(QStringLiteral("crash-grandchild.pid"));
    aitrain::v2::PythonAdapterHostV2 host;
    QList<aitrain::v2::PythonAdapterExitV2> exits;
    QString error;
    QVERIFY2(host.start(fixtureLaunch(python, fixture, pidPath, QStringLiteral("crash")),
        aitrain::v2::RequestId::create(), aitrain::v2::TaskId::create(), {},
        [&exits](const aitrain::v2::PythonAdapterExitV2& outcome) { exits.append(outcome); }, &error), qPrintable(error));
    QTRY_VERIFY_WITH_TIMEOUT(readPid(pidPath) > 0, 7000);
    const qint64 grandchildPid = readPid(pidPath);
    QTRY_COMPARE_WITH_TIMEOUT(exits.size(), 1, 7000);
    QTRY_VERIFY_WITH_TIMEOUT(!processIsAlive(grandchildPid), 7000);
    QCOMPARE(exits.constFirst().exitCode, 23);
    QVERIFY(!exits.constFirst().terminalEventSeen);
#else
    QSKIP("Windows Job Object only");
#endif
}

void V2ProcessTreeTests::hostDestructionCannotLeaveProcessTreeRunning()
{
#ifdef Q_OS_WIN
    const QString python = pythonExecutable();
    if (python.isEmpty()) QSKIP("Python executable is required for process-tree integration.");
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString fixture = writeProcessTreeFixture(directory);
    const QString pidPath = directory.filePath(QStringLiteral("destroy-grandchild.pid"));
    qint64 grandchildPid = 0;
    {
        aitrain::v2::PythonAdapterHostV2 host;
        QString error;
        QVERIFY2(host.start(fixtureLaunch(python, fixture, pidPath, QStringLiteral("live")),
            aitrain::v2::RequestId::create(), aitrain::v2::TaskId::create(), {}, {}, &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(readPid(pidPath) > 0, 7000);
        grandchildPid = readPid(pidPath);
        QVERIFY(processIsAlive(grandchildPid));
    }
    QTRY_VERIFY_WITH_TIMEOUT(!processIsAlive(grandchildPid), 7000);
#else
    QSKIP("Windows Job Object only");
#endif
}

void V2ProcessTreeTests::supervisorCloseSimulatesWorkerKillAndKillsDescendants()
{
#ifdef Q_OS_WIN
    const QString python = pythonExecutable();
    if (python.isEmpty()) QSKIP("Python executable is required for process-tree integration.");
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString fixture = writeProcessTreeFixture(directory);
    const QString pidPath = directory.filePath(QStringLiteral("worker-kill-grandchild.pid"));
    QProcess root;
    root.start(python, QStringList{fixture, pidPath, QStringLiteral("live")});
    QVERIFY2(root.waitForStarted(3000), qPrintable(root.errorString()));
    aitrain::v2::ProcessTreeSupervisor supervisor;
    QString error;
    QVERIFY2(supervisor.create(&error), qPrintable(error));
    QVERIFY2(supervisor.attach(&root, &error), qPrintable(error));
    QTRY_VERIFY_WITH_TIMEOUT(readPid(pidPath) > 0, 7000);
    const qint64 grandchildPid = readPid(pidPath);
    QVERIFY(processIsAlive(grandchildPid));
    supervisor.reset();
    QVERIFY(root.waitForFinished(7000));
    QTRY_VERIFY_WITH_TIMEOUT(!processIsAlive(grandchildPid), 7000);
#else
    QSKIP("Windows Job Object only");
#endif
}

QTEST_MAIN(V2ProcessTreeTests)
#include "tst_v2_process_tree.moc"
