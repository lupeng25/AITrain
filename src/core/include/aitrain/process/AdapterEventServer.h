#pragma once

#include "aitrain/protocol/Protocol.h"

#include <QString>

#include <QHash>
#include <QSet>
#include <functional>
#include <memory>

class QTcpServer;
class QTcpSocket;
class QTimer;

namespace aitrain {

struct AdapterEventEndpoint final {
    QString host;
    quint16 port = 0;
    QString token;
    RequestId requestId;
    TaskId taskId;
};

class AdapterEventServer final {
public:
    // true means the event has been accepted by the downstream durable
    // consumer. Terminal events are acknowledged to the Adapter only then.
    using EventHandler = std::function<bool(const ProtocolEnvelope&)>;

    AdapterEventServer();
    ~AdapterEventServer();

    AdapterEventServer(const AdapterEventServer&) = delete;
    AdapterEventServer& operator=(const AdapterEventServer&) = delete;

    bool start(const RequestId& requestId, const TaskId& taskId, QString* error = nullptr);
    void stop();
    bool isListening() const;
    AdapterEventEndpoint endpoint() const;
    QString lastError() const;
    void setEventHandler(EventHandler handler);

private:
    void acceptPendingConnections();
    void readSocket(QTcpSocket* socket);
    void rejectSocket(QTcpSocket* socket, const QString& code);
    void removeHandshakeTimer(QTcpSocket* socket);
    void setFailure(const QString& error);

    std::unique_ptr<QTcpServer> server_;
    QTcpSocket* activeSocket_ = nullptr;
    QSet<QTcpSocket*> pendingSockets_;
    QHash<QTcpSocket*, QTimer*> handshakeTimers_;
    AdapterEventEndpoint endpoint_;
    ProtocolSequenceTracker sequenceTracker_;
    EventHandler eventHandler_;
    bool authenticated_ = false;
    bool terminalEventSeen_ = false;
    QString lastError_;
};

} // namespace aitrain
