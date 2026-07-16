#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QDateTime>
#include <QHash>
#include <QJsonObject>
#include <QSet>

namespace aitrain {

inline constexpr int kProtocolVersion = 1;
inline constexpr qsizetype kProtocolMaxControlMessageBytes = 1024 * 1024;
inline constexpr qsizetype kProtocolMaxLogMessageBytes = 64 * 1024;

struct ProtocolEnvelope final {
    MessageId messageId;
    RequestId requestId;
    TaskId taskId;
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
};

} // namespace aitrain
