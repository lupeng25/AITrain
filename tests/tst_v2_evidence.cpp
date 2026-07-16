#include "aitrain/v2/EvidenceBundleV2.h"
#include "aitrain/v2/EvidenceRendererV2.h"

#include <QDateTime>
#include <QJsonDocument>
#include <QTest>

namespace {

aitrain::v2::EvidenceBundleV2 validBundle()
{
    aitrain::v2::EvidenceBundleV2 bundle;
    bundle.projectIdentity = QStringLiteral("project:demo-v2");
    bundle.task.id = aitrain::v2::TaskId::create();
    bundle.task.requestId = aitrain::v2::RequestId::create();
    bundle.task.state = aitrain::v2::TaskState::Succeeded;
    bundle.task.capabilityId = QStringLiteral("runtime.aitrain_onnxruntime");
    bundle.task.taskType = QStringLiteral("inference");
    bundle.task.createdAt = QDateTime::currentDateTimeUtc().addSecs(-3);
    bundle.task.updatedAt = QDateTime::currentDateTimeUtc();
    bundle.workflowRunId = aitrain::v2::WorkflowRunId::create();
    bundle.datasetSnapshotId = aitrain::v2::SnapshotId::create();
    bundle.backendEnvironment = {{QStringLiteral("backend"), QStringLiteral("aitrain_onnxruntime")},
        {QStringLiteral("onnxRuntimeVersion"), QStringLiteral("1.x")}};
    bundle.parameters = {{QStringLiteral("scoreThreshold"), 0.25}};
    bundle.metrics = {{QStringLiteral("latencyMs"), 4.5}};
    bundle.runtimeStatus = {{QStringLiteral("route"), QStringLiteral("aitrain_onnxruntime")},
        {QStringLiteral("state"), QStringLiteral("supported")}};
    bundle.evaluation = {{QStringLiteral("available"), false}};
    bundle.benchmark = {{QStringLiteral("available"), false}};
    bundle.artifacts.append({aitrain::v2::ArtifactId::create(), QStringLiteral("runtime_output_bundle"),
        {{QStringLiteral("sha256"), QString(64, QLatin1Char('a'))}}});
    bundle.limitations = QStringList{QStringLiteral("这是 ONNX Runtime 推理 Smoke，不构成客户域验收。")};
    bundle.createdAt = QDateTime::currentDateTimeUtc();
    return bundle;
}

} // namespace

class V2EvidenceTests : public QObject {
    Q_OBJECT

private slots:
    void bundleRoundTripsAndAllRenderersShareFacts();
    void rejectsInconsistentTerminalFailureFacts();
    void rejectsNonSuccessFailureWithoutMessageOrOccurredAt();
};

void V2EvidenceTests::bundleRoundTripsAndAllRenderersShareFacts()
{
    const aitrain::v2::EvidenceBundleV2 source = validBundle();
    QString error;
    const QJsonObject encoded = aitrain::v2::encodeEvidenceBundleV2(source, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    QCOMPARE(encoded.value(QStringLiteral("kind")).toString(), QStringLiteral("aitrain_evidence_bundle_v2"));
    aitrain::v2::EvidenceBundleV2 decoded;
    QVERIFY2(aitrain::v2::decodeEvidenceBundleV2(encoded, &decoded, &error), qPrintable(error));
    QCOMPARE(decoded.task.id, source.task.id);
    QCOMPARE(decoded.workflowRunId, source.workflowRunId);
    QCOMPARE(decoded.artifacts.constFirst().artifactId, source.artifacts.constFirst().artifactId);

    const QByteArray json = aitrain::v2::EvidenceRendererV2::renderJson(decoded, &error);
    QVERIFY2(!json.isEmpty(), qPrintable(error));
    const QString markdown = aitrain::v2::EvidenceRendererV2::renderMarkdown(decoded, &error);
    QVERIFY2(!markdown.isEmpty(), qPrintable(error));
    const QString html = aitrain::v2::EvidenceRendererV2::renderHtml(decoded, &error);
    QVERIFY2(!html.isEmpty(), qPrintable(error));
    const QByteArray modelCard = aitrain::v2::EvidenceRendererV2::renderModelCard(decoded, &error);
    QVERIFY2(!modelCard.isEmpty(), qPrintable(error));
    const QString taskState = aitrain::v2::taskStateToString(source.task.state);
    QVERIFY(markdown.contains(taskState));
    QVERIFY(html.contains(taskState));
    const QJsonObject modelCardObject = QJsonDocument::fromJson(modelCard).object();
    QCOMPARE(modelCardObject.value(QStringLiteral("evidence")).toObject(), encoded);
}

void V2EvidenceTests::rejectsInconsistentTerminalFailureFacts()
{
    aitrain::v2::EvidenceBundleV2 bundle = validBundle();
    bundle.task.failure = {aitrain::v2::FailureCode::InternalError,
        QStringLiteral("success must not have a failure"), {}, QDateTime::currentDateTimeUtc()};
    QString error;
    QVERIFY(!aitrain::v2::validateEvidenceBundleV2(bundle, &error));
    QVERIFY(error.contains(QStringLiteral("成功")));

    bundle = validBundle();
    bundle.task.state = aitrain::v2::TaskState::Failed;
    QVERIFY(!aitrain::v2::validateEvidenceBundleV2(bundle, &error));
    bundle.task.failure = {aitrain::v2::FailureCode::ArtifactIncomplete,
        QStringLiteral("运行产物不完整"), QStringLiteral("重新运行"), QDateTime::currentDateTimeUtc()};
    QVERIFY2(aitrain::v2::validateEvidenceBundleV2(bundle, &error), qPrintable(error));
}

void V2EvidenceTests::rejectsNonSuccessFailureWithoutMessageOrOccurredAt()
{
    aitrain::v2::EvidenceBundleV2 bundle = validBundle();
    bundle.task.state = aitrain::v2::TaskState::Canceled;
    bundle.task.failure.code = aitrain::v2::FailureCode::Canceled;
    QString error;

    QVERIFY(!aitrain::v2::validateEvidenceBundleV2(bundle, &error));
    QVERIFY(error.contains(QStringLiteral("消息")));

    bundle.task.failure.message = QStringLiteral("用户取消任务");
    QVERIFY(!aitrain::v2::validateEvidenceBundleV2(bundle, &error));
    QVERIFY(error.contains(QStringLiteral("发生时间")));

    bundle.task.failure.occurredAt = QDateTime::currentDateTimeUtc();
    QVERIFY2(aitrain::v2::validateEvidenceBundleV2(bundle, &error), qPrintable(error));

    const QJsonObject encoded = aitrain::v2::encodeEvidenceBundleV2(bundle, &error);
    QVERIFY2(!encoded.isEmpty(), qPrintable(error));
    aitrain::v2::EvidenceBundleV2 decoded;
    QVERIFY2(aitrain::v2::decodeEvidenceBundleV2(encoded, &decoded, &error), qPrintable(error));
    QCOMPARE(decoded.task.failure.message, bundle.task.failure.message);
    QCOMPARE(decoded.task.failure.occurredAt, bundle.task.failure.occurredAt);
}

QTEST_MAIN(V2EvidenceTests)
#include "tst_v2_evidence.moc"
