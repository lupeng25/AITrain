#pragma once

#include "aitrain/v2/ProtocolV2.h"

#include <QString>

#include <functional>
#include <memory>

class QTcpServer;
class QTcpSocket;

namespace aitrain::v2 {

struct AdapterEventEndpointV2 final {
    QString host;
    quint16 port = 0;
    QString token;
    RequestId requestId;
    TaskId taskId;
};

class AdapterEventServerV2 final {
public:
    using EventHandler = std::function<void(const ProtocolEnvelope&)>;

    AdapterEventServerV2();
    ~AdapterEventServerV2();

    AdapterEventServerV2(const AdapterEventServerV2&) = delete;
    AdapterEventServerV2& operator=(const AdapterEventServerV2&) = delete;

    bool start(const RequestId& requestId, const TaskId& taskId, QString* error = nullptr);
    void stop();
    bool isListening() const;
    AdapterEventEndpointV2 endpoint() const;
    QString lastError() const;
    void setEventHandler(EventHandler handler);

private:
    void acceptPendingConnections();
    void readSocket(QTcpSocket* socket);
    void rejectSocket(QTcpSocket* socket, const QString& code);
    void setFailure(const QString& error);

    std::unique_ptr<QTcpServer> server_;
    QTcpSocket* activeSocket_ = nullptr;
    AdapterEventEndpointV2 endpoint_;
    ProtocolV2SequenceTracker sequenceTracker_;
    EventHandler eventHandler_;
    bool authenticated_ = false;
    QString lastError_;
};

} // namespace aitrain::v2
