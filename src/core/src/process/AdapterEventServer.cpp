#include "aitrain/process/AdapterEventServer.h"

#include <QHostAddress>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>
#include <QUuid>

namespace aitrain {
namespace {

constexpr auto kChannelName = "aitrain.adapter";
constexpr int kHandshakeTimeoutMs = 5000;

bool isTerminalEventKind(const QString& kind)
{
    return kind == QStringLiteral("event.succeeded")
        || kind == QStringLiteral("event.failed")
        || kind == QStringLiteral("event.canceled");
}

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

AdapterEventServer::AdapterEventServer() = default;

AdapterEventServer::~AdapterEventServer()
{
    stop();
}

bool AdapterEventServer::start(const RequestId& requestId, const TaskId& taskId, QString* error)
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
    pendingSockets_.clear();
    handshakeTimers_.clear();
    server_ = std::move(server);
    terminalEventSeen_ = false;
    QObject::connect(server_.get(), &QTcpServer::newConnection, server_.get(), [this] {
        acceptPendingConnections();
    });
    return true;
}

void AdapterEventServer::stop()
{
    if (activeSocket_) {
        activeSocket_->disconnectFromHost();
        activeSocket_ = nullptr;
    }
    const auto pendingSockets = pendingSockets_.values();
    for (QTcpSocket* socket : pendingSockets) {
        if (socket) {
            socket->disconnectFromHost();
        }
    }
    pendingSockets_.clear();
    for (QTimer* timer : handshakeTimers_) {
        if (timer) {
            timer->stop();
            timer->deleteLater();
        }
    }
    handshakeTimers_.clear();
    authenticated_ = false;
    terminalEventSeen_ = false;
    sequenceTracker_.clear();
    if (server_) {
        server_->close();
        server_.reset();
    }
    endpoint_ = {};
}

bool AdapterEventServer::isListening() const
{
    return server_ && server_->isListening();
}

AdapterEventEndpoint AdapterEventServer::endpoint() const
{
    return endpoint_;
}

QString AdapterEventServer::lastError() const
{
    return lastError_;
}

void AdapterEventServer::setEventHandler(EventHandler handler)
{
    eventHandler_ = std::move(handler);
}

void AdapterEventServer::acceptPendingConnections()
{
    while (server_ && server_->hasPendingConnections()) {
        QTcpSocket* socket = server_->nextPendingConnection();
        if (!socket) {
            continue;
        }
        if (activeSocket_ || pendingSockets_.size() >= 4) {
            QObject::connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
            rejectSocket(socket, QStringLiteral("connection_already_active"));
            continue;
        }

        pendingSockets_.insert(socket);
        // Bound the unread loopback buffer even if a broken/malicious adapter
        // never terminates a JSONL frame.  One extra byte lets readSocket()
        // distinguish an exactly-at-limit control frame from overflow.
        socket->setReadBufferSize(kProtocolMaxControlMessageBytes + 1);
        QObject::connect(socket, &QTcpSocket::readyRead, socket, [this, socket] {
            readSocket(socket);
        });
        QObject::connect(socket, &QTcpSocket::disconnected, socket, [this, socket] {
            if (activeSocket_ == socket) {
                activeSocket_ = nullptr;
                authenticated_ = false;
            }
            pendingSockets_.remove(socket);
            removeHandshakeTimer(socket);
            socket->deleteLater();
        });
        QTimer* handshakeTimer = new QTimer(socket);
        handshakeTimer->setSingleShot(true);
        QObject::connect(handshakeTimer, &QTimer::timeout, socket, [this, socket] {
            if (pendingSockets_.contains(socket)) {
                rejectSocket(socket, QStringLiteral("handshake_timeout"));
            }
        });
        handshakeTimers_.insert(socket, handshakeTimer);
        handshakeTimer->start(kHandshakeTimeoutMs);
    }
}

void AdapterEventServer::readSocket(QTcpSocket* socket)
{
    if (!socket || (socket != activeSocket_ && !pendingSockets_.contains(socket))) {
        return;
    }
    if (socket->bytesAvailable() > kProtocolMaxControlMessageBytes) {
        rejectSocket(socket, QStringLiteral("message_too_large"));
        return;
    }

    while (socket->canReadLine()) {
        const QByteArray line = socket->readLine(kProtocolMaxControlMessageBytes + 1);
        if (line.isEmpty() || line.size() > kProtocolMaxControlMessageBytes || !line.endsWith('\n')) {
            rejectSocket(socket, QStringLiteral("message_too_large"));
            return;
        }

        if (socket != activeSocket_) {
            QJsonParseError parseError;
            const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
            const QJsonObject handshake = document.isObject() ? document.object() : QJsonObject{};
            if (parseError.error != QJsonParseError::NoError
                || handshake.value(QStringLiteral("channel")).toString() != QString::fromLatin1(kChannelName)
                || handshake.value(QStringLiteral("token")).toString() != endpoint_.token) {
                rejectSocket(socket, QStringLiteral("authentication_failed"));
                return;
            }
            if (activeSocket_) {
                rejectSocket(socket, QStringLiteral("connection_already_active"));
                return;
            }
            pendingSockets_.remove(socket);
            activeSocket_ = socket;
            authenticated_ = true;
            removeHandshakeTimer(socket);
            socket->write(handshakeReply("accepted"));
            continue;
        }

        ProtocolEnvelope envelope;
        QString protocolError;
        if (!decodeProtocolMessage(line, &envelope, &protocolError)
            || !envelope.kind.startsWith(QStringLiteral("event."))
            || !sequenceTracker_.observe(envelope, endpoint_.requestId, endpoint_.taskId, &protocolError)) {
            setFailure(protocolError.isEmpty()
                    ? QStringLiteral("Adapter event channel received a non-event message.")
                    : protocolError);
            rejectSocket(socket, QStringLiteral("protocol_violation"));
            return;
        }
        if (terminalEventSeen_) {
            setFailure(QStringLiteral("terminal_event_already_seen"));
            rejectSocket(socket, QStringLiteral("terminal_event_already_seen"));
            return;
        }
        if (eventHandler_ && !eventHandler_(envelope)) {
            setFailure(QStringLiteral("downstream_event_rejected"));
            rejectSocket(socket, QStringLiteral("downstream_event_rejected"));
            return;
        }
        if (isTerminalEventKind(envelope.kind)) {
            terminalEventSeen_ = true;
            QJsonObject acknowledgment;
            acknowledgment.insert(QStringLiteral("status"), QStringLiteral("accepted"));
            acknowledgment.insert(QStringLiteral("sequence"), QString::number(envelope.sequence));
            acknowledgment.insert(QStringLiteral("terminal"), true);
            QByteArray bytes = QJsonDocument(acknowledgment).toJson(QJsonDocument::Compact);
            bytes.append('\n');
            socket->write(bytes);
            socket->flush();
        }
    }
}

void AdapterEventServer::rejectSocket(QTcpSocket* socket, const QString& code)
{
    if (!socket) {
        return;
    }
    if (lastError_.isEmpty()) {
        setFailure(code);
    }
    removeHandshakeTimer(socket);
    socket->write(handshakeReply("rejected", code));
    socket->flush();
    socket->disconnectFromHost();
}

void AdapterEventServer::removeHandshakeTimer(QTcpSocket* socket)
{
    auto iterator = handshakeTimers_.find(socket);
    if (iterator == handshakeTimers_.end()) {
        return;
    }
    QTimer* timer = iterator.value();
    handshakeTimers_.erase(iterator);
    if (timer) {
        timer->stop();
        timer->deleteLater();
    }
}

void AdapterEventServer::setFailure(const QString& error)
{
    lastError_ = error;
    qWarning().noquote() << QStringLiteral("[adapter channel failure] %1").arg(error);
}

} // namespace aitrain
