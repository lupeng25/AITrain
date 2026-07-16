#pragma once

#include "aitrain/v2/DomainTypes.h"

#include <QDateTime>
#include <QHash>
#include <QJsonObject>
#include <QSet>

namespace aitrain::v2 {

inline constexpr int kProtocolV2Version = 2;
inline constexpr qsizetype kProtocolV2MaxControlMessageBytes = 1024 * 1024;
inline constexpr qsizetype kProtocolV2MaxLogMessageBytes = 64 * 1024;

struct ProtocolEnvelope final {
    MessageId messageId;
    RequestId requestId;
    TaskId taskId;
    quint64 sequence = 0;
    QString kind;
    QDateTime timestamp;
    QJsonObject payload;
};

bool isKnownProtocolV2Kind(const QString& kind);
bool isProtocolV2LogKind(const QString& kind);
bool validateProtocolV2Envelope(const ProtocolEnvelope& envelope, QString* error = nullptr);
QByteArray encodeProtocolV2Message(const ProtocolEnvelope& envelope, QString* error = nullptr);
bool decodeProtocolV2Message(const QByteArray& bytes, ProtocolEnvelope* envelope, QString* error = nullptr);

class ProtocolV2SequenceTracker final {
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
};

} // namespace aitrain::v2
