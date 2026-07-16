#include "aitrain/workflow/EvidenceBundle.h"
#include "aitrain/workflow/EvidenceRenderer.h"

#include <QDateTime>
#include <QJsonDocument>
#include <QTest>

namespace {

aitrain::EvidenceBundle validBundle()
{
    aitrain::EvidenceBundle bundle;
    bundle.projectIdentity = QStringLiteral("project:demo");
    bundle.task.id = aitrain::TaskId::create();
    bundle.task.requestId = aitrain::RequestId::create();
    bundle.task.state = aitrain::TaskState::Succeeded;
    bundle.task.capabilityId = QStringLiteral("runtime.aitrain_onnxruntime");
    bundle.task.taskType = QStringLiteral("inference");
    bundle.task.createdAt = QDateTime::currentDateTimeUtc().addSecs(-3);
    bundle.task.updatedAt = QDateTime::currentDateTimeUtc();
    bundle.workflowRunId = aitrain::WorkflowRunId::create();
    bundle.datasetSnapshotId = aitrain::SnapshotId::create();
    bundle.backendEnvironment = {{QStringLiteral("backend"), QStringLiteral("aitrain_onnxruntime")},
        {QStringLiteral("onnxRuntimeVersion"), QStringLiteral("1.x")}};
    bundle.parameters = {{QStringLiteral("scoreThreshold"), 0.25}};
    bundle.metrics = {{QStringLiteral("latencyMs"), 4.5}};
    bundle.runtimeStatus = {{QStringLiteral("route"), QStringLiteral("aitrain_onnxruntime")},
        {QStringLiteral("state"), QStringLiteral("supported")}};
    bundle.evaluation = {{QStringLiteral("available"), false}};
    bundle.benchmark = {{QStringLiteral("available"), false}};
    bundle.artifacts.append({aitrain::ArtifactId::create(), QStringLiteral("runtime_output_bundle"),
        {{QStringLiteral("sha256"), QString(64, QLatin1Char('a'))}}});
    bundle.limitations = QStringList{QStringLiteral("这是 ONNX Runtime 推理 Smoke，不构成客户域验收。")};
    bundle.createdAt = QDateTime::currentDateTimeUtc();
    return bundle;
}

} // namespace

class EvidenceTests : public QObject {
    Q_OBJECT

private slots:
    void bundleRoundTripsAndAllRenderersShareFacts();
    void rejectsInconsistentTerminalFailureFacts();
    void rejectsNonSuccessFailureWithoutMessageOrOccurredAt();
};

void EvidenceTests::bundleRoundTripsAndAllRenderersShareFacts()
{
    const aitrain::EvidenceBundle source = validBundle();
    QString error;
    const QJsonObject encoded = aitrain::encodeEvidenceBundle(source, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    QCOMPARE(encoded.value(QStringLiteral("kind")).toString(), QStringLiteral("aitrain_evidence_bundle"));
    aitrain::EvidenceBundle decoded;
    QVERIFY2(aitrain::decodeEvidenceBundle(encoded, &decoded, &error), qPrintable(error));
    QCOMPARE(decoded.task.id, source.task.id);
    QCOMPARE(decoded.workflowRunId, source.workflowRunId);
    QCOMPARE(decoded.artifacts.constFirst().artifactId, source.artifacts.constFirst().artifactId);

    const QByteArray json = aitrain::EvidenceRenderer::renderJson(decoded, &error);
    QVERIFY2(!json.isEmpty(), qPrintable(error));
    const QString markdown = aitrain::EvidenceRenderer::renderMarkdown(decoded, &error);
    QVERIFY2(!markdown.isEmpty(), qPrintable(error));
    const QString html = aitrain::EvidenceRenderer::renderHtml(decoded, &error);
    QVERIFY2(!html.isEmpty(), qPrintable(error));
    const QByteArray modelCard = aitrain::EvidenceRenderer::renderModelCard(decoded, &error);
    QVERIFY2(!modelCard.isEmpty(), qPrintable(error));
    const QString taskState = aitrain::taskStateToString(source.task.state);
    QVERIFY(markdown.contains(taskState));
    QVERIFY(html.contains(taskState));
    const QJsonObject modelCardObject = QJsonDocument::fromJson(modelCard).object();
    QCOMPARE(modelCardObject.value(QStringLiteral("evidence")).toObject(), encoded);
}

void EvidenceTests::rejectsInconsistentTerminalFailureFacts()
{
    aitrain::EvidenceBundle bundle = validBundle();
    bundle.task.failure = {aitrain::FailureCode::InternalError,
        QStringLiteral("success must not have a failure"), {}, QDateTime::currentDateTimeUtc()};
    QString error;
    QVERIFY(!aitrain::validateEvidenceBundle(bundle, &error));
    QVERIFY(error.contains(QStringLiteral("成功")));

    bundle = validBundle();
    bundle.task.state = aitrain::TaskState::Failed;
    QVERIFY(!aitrain::validateEvidenceBundle(bundle, &error));
    bundle.task.failure = {aitrain::FailureCode::ArtifactIncomplete,
        QStringLiteral("运行产物不完整"), QStringLiteral("重新运行"), QDateTime::currentDateTimeUtc()};
    QVERIFY2(aitrain::validateEvidenceBundle(bundle, &error), qPrintable(error));
}

void EvidenceTests::rejectsNonSuccessFailureWithoutMessageOrOccurredAt()
{
    aitrain::EvidenceBundle bundle = validBundle();
    bundle.task.state = aitrain::TaskState::Canceled;
    bundle.task.failure.code = aitrain::FailureCode::Canceled;
    QString error;

    QVERIFY(!aitrain::validateEvidenceBundle(bundle, &error));
    QVERIFY(error.contains(QStringLiteral("消息")));

    bundle.task.failure.message = QStringLiteral("用户取消任务");
    QVERIFY(!aitrain::validateEvidenceBundle(bundle, &error));
    QVERIFY(error.contains(QStringLiteral("发生时间")));

    bundle.task.failure.occurredAt = QDateTime::currentDateTimeUtc();
    QVERIFY2(aitrain::validateEvidenceBundle(bundle, &error), qPrintable(error));

    const QJsonObject encoded = aitrain::encodeEvidenceBundle(bundle, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    aitrain::EvidenceBundle decoded;
    QVERIFY2(aitrain::decodeEvidenceBundle(encoded, &decoded, &error), qPrintable(error));
    QCOMPARE(decoded.task.failure.message, bundle.task.failure.message);
    QCOMPARE(decoded.task.failure.occurredAt, bundle.task.failure.occurredAt);
}

QTEST_MAIN(EvidenceTests)
#include "tst_evidence.moc"
