#include "aitrain/v2/AdapterEventServerV2.h"

#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QTcpServer>
#include <QTcpSocket>
#include <QUuid>

namespace aitrain::v2 {
namespace {

constexpr auto kChannelName = "aitrain.adapter.v2";

QByteArray handshakeReply(const char* status, const QString& code = {})
{
    QJsonObject payload;
    payload.insert(QStringLiteral("status"), QString::fromLatin1(status));
    if (!code.isEmpty()) {
        payload.insert(QStringLiteral("code"), code);
    }
    QByteArray bytes = QJsonDocument(payload).toJson(QJsonDocument::Compact);
    bytes.append('\n');
    return bytes;
}

} // namespace

AdapterEventServerV2::AdapterEventServerV2() = default;

AdapterEventServerV2::~AdapterEventServerV2()
{
    stop();
}

bool AdapterEventServerV2::start(const RequestId& requestId, const TaskId& taskId, QString* error)
{
    stop();
    if (!requestId.isValid() || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral("启动 Adapter 事件服务需要有效 requestId 和 taskId。");
        }
        return false;
    }

    auto server = std::make_unique<QTcpServer>();
    if (!server->listen(QHostAddress(QHostAddress::LocalHost), 0)) {
        if (error) {
            *error = QStringLiteral("无法监听 Adapter loopback 事件端口：%1").arg(server->errorString());
        }
        return false;
    }

    endpoint_.host = QStringLiteral("127.0.0.1");
    endpoint_.port = server->serverPort();
    endpoint_.token = QUuid::createUuid().toString(QUuid::WithoutBraces);
    endpoint_.requestId = requestId;
    endpoint_.taskId = taskId;
    sequenceTracker_.clear();
    lastError_.clear();
    authenticated_ = false;
    activeSocket_ = nullptr;
    server_ = std::move(server);
    QObject::connect(server_.get(), &QTcpServer::newConnection, server_.get(), [this] {
        acceptPendingConnections();
    });
    return true;
}

void AdapterEventServerV2::stop()
{
    if (activeSocket_) {
        activeSocket_->disconnectFromHost();
        activeSocket_ = nullptr;
    }
    authenticated_ = false;
    sequenceTracker_.clear();
    if (server_) {
        server_->close();
        server_.reset();
    }
    endpoint_ = {};
}

bool AdapterEventServerV2::isListening() const
{
    return server_ && server_->isListening();
}

AdapterEventEndpointV2 AdapterEventServerV2::endpoint() const
{
    return endpoint_;
}

QString AdapterEventServerV2::lastError() const
{
    return lastError_;
}

void AdapterEventServerV2::setEventHandler(EventHandler handler)
{
    eventHandler_ = std::move(handler);
}

void AdapterEventServerV2::acceptPendingConnections()
{
    while (server_ && server_->hasPendingConnections()) {
        QTcpSocket* socket = server_->nextPendingConnection();
        if (!socket) {
            continue;
        }
        if (activeSocket_) {
            rejectSocket(socket, QStringLiteral("connection_already_active"));
            continue;
        }

        activeSocket_ = socket;
        authenticated_ = false;
        // Bound the unread loopback buffer even if a broken/malicious adapter
        // never terminates a JSONL frame.  One extra byte lets readSocket()
        // distinguish an exactly-at-limit control frame from overflow.
        socket->setReadBufferSize(kProtocolV2MaxControlMessageBytes + 1);
        QObject::connect(socket, &QTcpSocket::readyRead, socket, [this, socket] {
            readSocket(socket);
        });
        QObject::connect(socket, &QTcpSocket::disconnected, socket, [this, socket] {
            if (activeSocket_ == socket) {
                activeSocket_ = nullptr;
                authenticated_ = false;
            }
            socket->deleteLater();
        });
    }
}

void AdapterEventServerV2::readSocket(QTcpSocket* socket)
{
    if (!socket || socket != activeSocket_) {
        return;
    }
    if (socket->bytesAvailable() > kProtocolV2MaxControlMessageBytes) {
        rejectSocket(socket, QStringLiteral("message_too_large"));
        return;
    }

    while (socket->canReadLine()) {
        const QByteArray line = socket->readLine(kProtocolV2MaxControlMessageBytes + 1);
        if (line.isEmpty() || line.size() > kProtocolV2MaxControlMessageBytes || !line.endsWith('\n')) {
            rejectSocket(socket, QStringLiteral("message_too_large"));
            return;
        }

        if (!authenticated_) {
            QJsonParseError parseError;
            const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
            const QJsonObject handshake = document.isObject() ? document.object() : QJsonObject{};
            if (parseError.error != QJsonParseError::NoError
                || handshake.value(QStringLiteral("channel")).toString() != QString::fromLatin1(kChannelName)
                || handshake.value(QStringLiteral("token")).toString() != endpoint_.token) {
                rejectSocket(socket, QStringLiteral("authentication_failed"));
                return;
            }
            authenticated_ = true;
            socket->write(handshakeReply("accepted"));
            continue;
        }

        ProtocolEnvelope envelope;
        QString protocolError;
        if (!decodeProtocolV2Message(line, &envelope, &protocolError)
            || !envelope.kind.startsWith(QStringLiteral("event."))
            || !sequenceTracker_.observe(envelope, endpoint_.requestId, endpoint_.taskId, &protocolError)) {
            setFailure(protocolError.isEmpty()
                    ? QStringLiteral("Adapter event channel received a non-event message.")
                    : protocolError);
            rejectSocket(socket, QStringLiteral("protocol_violation"));
            return;
        }
        if (eventHandler_) {
            eventHandler_(envelope);
        }
    }
}

void AdapterEventServerV2::rejectSocket(QTcpSocket* socket, const QString& code)
{
    if (!socket) {
        return;
    }
    if (lastError_.isEmpty()) {
        setFailure(code);
    }
    socket->write(handshakeReply("rejected", code));
    socket->flush();
    socket->disconnectFromHost();
}

void AdapterEventServerV2::setFailure(const QString& error)
{
    lastError_ = error;
}

} // namespace aitrain::v2
