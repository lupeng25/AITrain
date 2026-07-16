#include "aitrain/protocol/Protocol.h"

#include <QJsonDocument>
#include <QTest>

namespace {

aitrain::ProtocolEnvelope makeEnvelope(const QString& kind = QStringLiteral("event.progress"), quint64 sequence = 1)
{
    aitrain::ProtocolEnvelope envelope;
    envelope.messageId = aitrain::MessageId::create();
    envelope.requestId = aitrain::RequestId::create();
    envelope.taskId = aitrain::TaskId::create();
    envelope.sequence = sequence;
    envelope.kind = kind;
    envelope.timestamp = QDateTime::currentDateTimeUtc();
    envelope.payload = QJsonObject{{QStringLiteral("percent"), 42}};
    return envelope;
}

} // namespace

class ProtocolTests : public QObject {
    Q_OBJECT

private slots:
    void envelopeRoundTrips();
    void invalidEnvelopeIsRejected();
    void malformedPayloadIsRejected();
    void unsupportedVersionAndOversizedMessageAreRejected();
    void logEnvelopeHasSmallerDedicatedLimit();
    void trackerRejectsDuplicateAndOutOfOrderMessages();
    void trackerRejectsCrossTaskMessage();
    void artifactCandidateIsDistinctFromCommittedArtifact();
};

void ProtocolTests::envelopeRoundTrips()
{
    const aitrain::ProtocolEnvelope source = makeEnvelope();
    QString error;
    const QByteArray encoded = aitrain::encodeProtocolMessage(source, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    QVERIFY(encoded.endsWith('\n'));

    aitrain::ProtocolEnvelope decoded;
    QVERIFY2(aitrain::decodeProtocolMessage(encoded, &decoded, &error), qPrintable(error));
    QVERIFY(decoded.messageId == source.messageId);
    QVERIFY(decoded.requestId == source.requestId);
    QVERIFY(decoded.taskId == source.taskId);
    QCOMPARE(decoded.sequence, source.sequence);
    QCOMPARE(decoded.kind, source.kind);
    QCOMPARE(decoded.payload, source.payload);
}

void ProtocolTests::invalidEnvelopeIsRejected()
{
    aitrain::ProtocolEnvelope envelope = makeEnvelope(QStringLiteral("event.unknown"));
    QString error;
    QVERIFY(aitrain::encodeProtocolMessage(envelope, &error).isEmpty());
    QVERIFY(error.contains(QStringLiteral("Unsupported")));

    envelope = makeEnvelope();
    envelope.sequence = 0;
    QVERIFY(aitrain::encodeProtocolMessage(envelope, &error).isEmpty());
}

void ProtocolTests::malformedPayloadIsRejected()
{
    const aitrain::ProtocolEnvelope envelope = makeEnvelope();
    QJsonObject object;
    object.insert(QStringLiteral("protocol"), aitrain::kProtocolVersion);
    object.insert(QStringLiteral("messageId"), envelope.messageId.toString());
    object.insert(QStringLiteral("requestId"), envelope.requestId.toString());
    object.insert(QStringLiteral("taskId"), envelope.taskId.toString());
    object.insert(QStringLiteral("sequence"), QStringLiteral("1"));
    object.insert(QStringLiteral("kind"), QStringLiteral("event.progress"));
    object.insert(QStringLiteral("timestamp"), envelope.timestamp.toString(Qt::ISODateWithMs));
    object.insert(QStringLiteral("payload"), QStringLiteral("not-an-object"));

    aitrain::ProtocolEnvelope decoded;
    QString error;
    QVERIFY(!aitrain::decodeProtocolMessage(QJsonDocument(object).toJson(QJsonDocument::Compact), &decoded, &error));
    QVERIFY(error.contains(QStringLiteral("payload")));
}

void ProtocolTests::unsupportedVersionAndOversizedMessageAreRejected()
{
    const aitrain::ProtocolEnvelope envelope = makeEnvelope();
    QJsonObject object;
    object.insert(QStringLiteral("protocol"), aitrain::kProtocolVersion + 1);
    object.insert(QStringLiteral("messageId"), envelope.messageId.toString());
    object.insert(QStringLiteral("requestId"), envelope.requestId.toString());
    object.insert(QStringLiteral("taskId"), envelope.taskId.toString());
    object.insert(QStringLiteral("sequence"), QStringLiteral("1"));
    object.insert(QStringLiteral("kind"), QStringLiteral("event.progress"));
    object.insert(QStringLiteral("timestamp"), envelope.timestamp.toString(Qt::ISODateWithMs));
    object.insert(QStringLiteral("payload"), QJsonObject{});

    aitrain::ProtocolEnvelope decoded;
    QString error;
    QVERIFY(!aitrain::decodeProtocolMessage(QJsonDocument(object).toJson(QJsonDocument::Compact), &decoded, &error));
    QVERIFY(error.contains(QStringLiteral("version")));

    const QByteArray oversized(aitrain::kProtocolMaxControlMessageBytes + 1, 'x');
    QVERIFY(!aitrain::decodeProtocolMessage(oversized, &decoded, &error));
    QVERIFY(error.contains(QStringLiteral("size")));
}

void ProtocolTests::logEnvelopeHasSmallerDedicatedLimit()
{
    aitrain::ProtocolEnvelope log = makeEnvelope(QStringLiteral("event.log"));
    log.payload = QJsonObject{{QStringLiteral("message"),
        QString(aitrain::kProtocolMaxLogMessageBytes, QLatin1Char('x'))}};
    QString error;
    QVERIFY(aitrain::encodeProtocolMessage(log, &error).isEmpty());
    QVERIFY(error.contains(QStringLiteral("event.log")));

    log.kind = QStringLiteral("event.progress");
    QVERIFY2(!aitrain::encodeProtocolMessage(log, &error).isEmpty(), qPrintable(error));
}

void ProtocolTests::trackerRejectsDuplicateAndOutOfOrderMessages()
{
    aitrain::ProtocolEnvelope first = makeEnvelope();
    aitrain::ProtocolSequenceTracker tracker;
    QString error;
    QVERIFY2(tracker.observe(first, first.requestId, first.taskId, &error), qPrintable(error));
    QVERIFY(!tracker.observe(first, first.requestId, first.taskId, &error));
    QVERIFY(error.contains(QStringLiteral("duplicate")));

    aitrain::ProtocolEnvelope outOfOrder = first;
    outOfOrder.messageId = aitrain::MessageId::create();
    QVERIFY(!tracker.observe(outOfOrder, first.requestId, first.taskId, &error));
    QVERIFY(error.contains(QStringLiteral("increasing")));
}

void ProtocolTests::trackerRejectsCrossTaskMessage()
{
    const aitrain::ProtocolEnvelope envelope = makeEnvelope();
    aitrain::ProtocolSequenceTracker tracker;
    QString error;
    QVERIFY(!tracker.observe(envelope, aitrain::RequestId::create(), envelope.taskId, &error));
    QVERIFY(error.contains(QStringLiteral("active request/task")));
}

void ProtocolTests::artifactCandidateIsDistinctFromCommittedArtifact()
{
    aitrain::ProtocolEnvelope candidate = makeEnvelope(QStringLiteral("event.artifact_candidate"));
    candidate.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("report")}, {QStringLiteral("path"), QStringLiteral("staging/report.json")}};
    QString error;
    QVERIFY2(!aitrain::encodeProtocolMessage(candidate, &error).isEmpty(), qPrintable(error));
}

QTEST_MAIN(ProtocolTests)
#include "tst_protocol.moc"
