#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QDateTime>
#include <QHash>
#include <QJsonObject>
#include <QQueue>
#include <QSet>

namespace aitrain {

inline constexpr int kProtocolVersion = 2;
inline constexpr qsizetype kProtocolMaxControlMessageBytes = 1024 * 1024;
inline constexpr qsizetype kProtocolMaxLogMessageBytes = 64 * 1024;

struct ProtocolEnvelope final {
    MessageId messageId;
    RequestId requestId;
    TaskId taskId;
    // 仅用于 GUI↔Worker 控制面；Adapter 事件通道可为空。令牌不进入业务 payload，
    // 由双方在解包业务前校验，避免本地 Socket 被其他进程注入命令或事件。
    QString controlToken;
    quint64 sequence = 0;
    QString kind;
    QDateTime timestamp;
    QJsonObject payload;
};

bool isKnownProtocolKind(const QString& kind);
bool isProtocolLogKind(const QString& kind);
bool validateProtocolEnvelope(const ProtocolEnvelope& envelope, QString* error = nullptr);
QByteArray encodeProtocolMessage(const ProtocolEnvelope& envelope, QString* error = nullptr);
bool decodeProtocolMessage(const QByteArray& bytes, ProtocolEnvelope* envelope, QString* error = nullptr);

class ProtocolSequenceTracker final {
public:
    bool observe(const ProtocolEnvelope& envelope,
        const RequestId& expectedRequestId,
        const TaskId& expectedTaskId,
        QString* error = nullptr);
    void reset(const RequestId& requestId);
    void clear();

private:
    QHash<QString, quint64> lastSequenceByRequest_;
    QSet<QString> observedMessageIds_;
    QQueue<QString> observedMessageOrder_;
};

} // namespace aitrain
