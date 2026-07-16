#include "aitrain/v2/ProtocolV2.h"

#include <QJsonDocument>
#include <QTest>

namespace {

aitrain::v2::ProtocolEnvelope makeEnvelope(const QString& kind = QStringLiteral("event.progress"), quint64 sequence = 1)
{
    aitrain::v2::ProtocolEnvelope envelope;
    envelope.messageId = aitrain::v2::MessageId::create();
    envelope.requestId = aitrain::v2::RequestId::create();
    envelope.taskId = aitrain::v2::TaskId::create();
    envelope.sequence = sequence;
    envelope.kind = kind;
    envelope.timestamp = QDateTime::currentDateTimeUtc();
    envelope.payload = QJsonObject{{QStringLiteral("percent"), 42}};
    return envelope;
}

} // namespace

class V2ProtocolTests : public QObject {
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

void V2ProtocolTests::envelopeRoundTrips()
{
    const aitrain::v2::ProtocolEnvelope source = makeEnvelope();
    QString error;
    const QByteArray encoded = aitrain::v2::encodeProtocolV2Message(source, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    QVERIFY(encoded.endsWith('\n'));

    aitrain::v2::ProtocolEnvelope decoded;
    QVERIFY2(aitrain::v2::decodeProtocolV2Message(encoded, &decoded, &error), qPrintable(error));
    QVERIFY(decoded.messageId == source.messageId);
    QVERIFY(decoded.requestId == source.requestId);
    QVERIFY(decoded.taskId == source.taskId);
    QCOMPARE(decoded.sequence, source.sequence);
    QCOMPARE(decoded.kind, source.kind);
    QCOMPARE(decoded.payload, source.payload);
}

void V2ProtocolTests::invalidEnvelopeIsRejected()
{
    aitrain::v2::ProtocolEnvelope envelope = makeEnvelope(QStringLiteral("event.unknown"));
    QString error;
    QVERIFY(aitrain::v2::encodeProtocolV2Message(envelope, &error).isEmpty());
    QVERIFY(error.contains(QStringLiteral("Unsupported")));

    envelope = makeEnvelope();
    envelope.sequence = 0;
    QVERIFY(aitrain::v2::encodeProtocolV2Message(envelope, &error).isEmpty());
}

void V2ProtocolTests::malformedPayloadIsRejected()
{
    const aitrain::v2::ProtocolEnvelope envelope = makeEnvelope();
    QJsonObject object;
    object.insert(QStringLiteral("protocol"), aitrain::v2::kProtocolV2Version);
    object.insert(QStringLiteral("messageId"), envelope.messageId.toString());
    object.insert(QStringLiteral("requestId"), envelope.requestId.toString());
    object.insert(QStringLiteral("taskId"), envelope.taskId.toString());
    object.insert(QStringLiteral("sequence"), 1);
    object.insert(QStringLiteral("kind"), QStringLiteral("event.progress"));
    object.insert(QStringLiteral("timestamp"), envelope.timestamp.toString(Qt::ISODateWithMs));
    object.insert(QStringLiteral("payload"), QStringLiteral("not-an-object"));

    aitrain::v2::ProtocolEnvelope decoded;
    QString error;
    QVERIFY(!aitrain::v2::decodeProtocolV2Message(QJsonDocument(object).toJson(QJsonDocument::Compact), &decoded, &error));
    QVERIFY(error.contains(QStringLiteral("payload")));
}

void V2ProtocolTests::unsupportedVersionAndOversizedMessageAreRejected()
{
    const aitrain::v2::ProtocolEnvelope envelope = makeEnvelope();
    QJsonObject object;
    object.insert(QStringLiteral("protocol"), 1);
    object.insert(QStringLiteral("messageId"), envelope.messageId.toString());
    object.insert(QStringLiteral("requestId"), envelope.requestId.toString());
    object.insert(QStringLiteral("taskId"), envelope.taskId.toString());
    object.insert(QStringLiteral("sequence"), 1);
    object.insert(QStringLiteral("kind"), QStringLiteral("event.progress"));
    object.insert(QStringLiteral("timestamp"), envelope.timestamp.toString(Qt::ISODateWithMs));
    object.insert(QStringLiteral("payload"), QJsonObject{});

    aitrain::v2::ProtocolEnvelope decoded;
    QString error;
    QVERIFY(!aitrain::v2::decodeProtocolV2Message(QJsonDocument(object).toJson(QJsonDocument::Compact), &decoded, &error));
    QVERIFY(error.contains(QStringLiteral("version")));

    const QByteArray oversized(aitrain::v2::kProtocolV2MaxControlMessageBytes + 1, 'x');
    QVERIFY(!aitrain::v2::decodeProtocolV2Message(oversized, &decoded, &error));
    QVERIFY(error.contains(QStringLiteral("size")));
}

void V2ProtocolTests::logEnvelopeHasSmallerDedicatedLimit()
{
    aitrain::v2::ProtocolEnvelope log = makeEnvelope(QStringLiteral("event.log"));
    log.payload = QJsonObject{{QStringLiteral("message"),
        QString(aitrain::v2::kProtocolV2MaxLogMessageBytes, QLatin1Char('x'))}};
    QString error;
    QVERIFY(aitrain::v2::encodeProtocolV2Message(log, &error).isEmpty());
    QVERIFY(error.contains(QStringLiteral("event.log")));

    log.kind = QStringLiteral("event.progress");
    QVERIFY2(!aitrain::v2::encodeProtocolV2Message(log, &error).isEmpty(), qPrintable(error));
}

void V2ProtocolTests::trackerRejectsDuplicateAndOutOfOrderMessages()
{
    aitrain::v2::ProtocolEnvelope first = makeEnvelope();
    aitrain::v2::ProtocolV2SequenceTracker tracker;
    QString error;
    QVERIFY2(tracker.observe(first, first.requestId, first.taskId, &error), qPrintable(error));
    QVERIFY(!tracker.observe(first, first.requestId, first.taskId, &error));
    QVERIFY(error.contains(QStringLiteral("duplicate")));

    aitrain::v2::ProtocolEnvelope outOfOrder = first;
    outOfOrder.messageId = aitrain::v2::MessageId::create();
    QVERIFY(!tracker.observe(outOfOrder, first.requestId, first.taskId, &error));
    QVERIFY(error.contains(QStringLiteral("increasing")));
}

void V2ProtocolTests::trackerRejectsCrossTaskMessage()
{
    const aitrain::v2::ProtocolEnvelope envelope = makeEnvelope();
    aitrain::v2::ProtocolV2SequenceTracker tracker;
    QString error;
    QVERIFY(!tracker.observe(envelope, aitrain::v2::RequestId::create(), envelope.taskId, &error));
    QVERIFY(error.contains(QStringLiteral("active request/task")));
}

void V2ProtocolTests::artifactCandidateIsDistinctFromCommittedArtifact()
{
    aitrain::v2::ProtocolEnvelope candidate = makeEnvelope(QStringLiteral("event.artifact_candidate"));
    candidate.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("report")}, {QStringLiteral("path"), QStringLiteral("staging/report.json")}};
    QString error;
    QVERIFY2(!aitrain::v2::encodeProtocolV2Message(candidate, &error).isEmpty(), qPrintable(error));
}

QTEST_MAIN(V2ProtocolTests)
#include "tst_v2_protocol.moc"
