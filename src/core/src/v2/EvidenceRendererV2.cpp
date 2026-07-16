#include "aitrain/v2/EvidenceRendererV2.h"

#include <QJsonArray>
#include <QJsonDocument>

namespace aitrain::v2 {
namespace {

QString compactJson(const QJsonObject& object)
{
    return QString::fromUtf8(QJsonDocument(object).toJson(QJsonDocument::Compact));
}

QString failureText(const Failure& failure)
{
    if (!failure.isFailure()) return QStringLiteral("none");
    QString text = failureCodeToString(failure.code);
    if (!failure.message.isEmpty()) text += QStringLiteral(": %1").arg(failure.message);
    if (!failure.suggestedAction.isEmpty()) text += QStringLiteral("\nSuggested action: %1").arg(failure.suggestedAction);
    return text;
}

QString artifactsMarkdown(const QVector<EvidenceArtifactV2>& artifacts)
{
    if (artifacts.isEmpty()) return QStringLiteral("- none\n");
    QString output;
    for (const EvidenceArtifactV2& artifact : artifacts) {
        output += QStringLiteral("- `%1` — %2\n").arg(artifact.artifactId.toString(), artifact.kind);
    }
    return output;
}

QString externalInputsMarkdown(const QVector<EvidenceExternalInputV2>& inputs)
{
    if (inputs.isEmpty()) return QStringLiteral("- none\n");
    QString output;
    for (const EvidenceExternalInputV2& input : inputs) {
        output += QStringLiteral("- `%1`: producer task `%2`, artifact `%3`, dataset `%4`, snapshot `%5`, version `%6`, manifest `%7`, root `%8`\n")
            .arg(input.role, input.producerTaskId.toString(), input.artifactId.toString(),
                input.datasetId.toString(), input.datasetSnapshotId.toString(),
                input.datasetVersionId.toString(), input.manifestSha256, input.rootHash);
    }
    return output;
}

} // namespace

QByteArray EvidenceRendererV2::renderJson(const EvidenceBundleV2& bundle, QString* error)
{
    const QJsonObject object = encodeEvidenceBundleV2(bundle, error);
    return object.isEmpty() ? QByteArray{} : QJsonDocument(object).toJson(QJsonDocument::Indented);
}

QString EvidenceRendererV2::renderMarkdown(const EvidenceBundleV2& bundle, QString* error)
{
    if (!validateEvidenceBundleV2(bundle, error)) return {};
    QString markdown;
    markdown += QStringLiteral("# AITrain Evidence Bundle V2\n\n");
    markdown += QStringLiteral("- Project: `%1`\n").arg(bundle.projectIdentity);
    markdown += QStringLiteral("- Task: `%1`\n").arg(bundle.task.id.toString());
    markdown += QStringLiteral("- Task state: `%1`\n").arg(taskStateToString(bundle.task.state));
    markdown += QStringLiteral("- Capability: `%1`\n").arg(bundle.task.capabilityId);
    markdown += QStringLiteral("- Task type: `%1`\n").arg(bundle.task.taskType);
    markdown += QStringLiteral("- Created at: `%1`\n\n").arg(bundle.createdAt.toUTC().toString(Qt::ISODateWithMs));
    markdown += QStringLiteral("## Backend and environment\n\n```json\n%1\n```\n\n").arg(compactJson(bundle.backendEnvironment));
    markdown += QStringLiteral("## Runtime status\n\n```json\n%1\n```\n\n").arg(compactJson(bundle.runtimeStatus));
    markdown += QStringLiteral("## External inputs\n\n%1\n").arg(externalInputsMarkdown(bundle.externalInputs));
    markdown += QStringLiteral("## Artifacts\n\n%1\n").arg(artifactsMarkdown(bundle.artifacts));
    markdown += QStringLiteral("## Limitations\n\n");
    if (bundle.limitations.isEmpty()) {
        markdown += QStringLiteral("- none\n");
    } else {
        for (const QString& limitation : bundle.limitations) markdown += QStringLiteral("- %1\n").arg(limitation);
    }
    markdown += QStringLiteral("\n## Failure\n\n```text\n%1\n```\n").arg(failureText(bundle.task.failure));
    return markdown;
}

QString EvidenceRendererV2::renderHtml(const EvidenceBundleV2& bundle, QString* error)
{
    if (!validateEvidenceBundleV2(bundle, error)) return {};
    QString artifactItems;
    for (const EvidenceArtifactV2& artifact : bundle.artifacts) {
        artifactItems += QStringLiteral("<li><code>%1</code> — %2</li>")
            .arg(artifact.artifactId.toString().toHtmlEscaped(), artifact.kind.toHtmlEscaped());
    }
    if (artifactItems.isEmpty()) artifactItems = QStringLiteral("<li>none</li>");
    QString externalInputItems;
    for (const EvidenceExternalInputV2& input : bundle.externalInputs) {
        externalInputItems += QStringLiteral(
            "<li><code>%1</code>: producer task <code>%2</code>, artifact <code>%3</code>, "
            "dataset <code>%4</code>, snapshot <code>%5</code>, version <code>%6</code>, "
            "manifest <code>%7</code>, root <code>%8</code></li>")
            .arg(input.role.toHtmlEscaped(), input.producerTaskId.toString().toHtmlEscaped(),
                input.artifactId.toString().toHtmlEscaped(), input.datasetId.toString().toHtmlEscaped(),
                input.datasetSnapshotId.toString().toHtmlEscaped(), input.datasetVersionId.toString().toHtmlEscaped(),
                input.manifestSha256.toHtmlEscaped(), input.rootHash.toHtmlEscaped());
    }
    if (externalInputItems.isEmpty()) externalInputItems = QStringLiteral("<li>none</li>");
    QString limitationItems;
    for (const QString& limitation : bundle.limitations) {
        limitationItems += QStringLiteral("<li>%1</li>").arg(limitation.toHtmlEscaped());
    }
    if (limitationItems.isEmpty()) limitationItems = QStringLiteral("<li>none</li>");
    QString html = QStringLiteral(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>AITrain Evidence Bundle V2</title></head><body>"
        "<h1>AITrain Evidence Bundle V2</h1>"
        "<dl><dt>Project</dt><dd><code>%1</code></dd><dt>Task</dt><dd><code>%2</code></dd>"
        "<dt>Task state</dt><dd><code>%3</code></dd><dt>Capability</dt><dd><code>%4</code></dd>"
        "<dt>Task type</dt><dd><code>%5</code></dd></dl>"
        "<h2>Backend and environment</h2><pre>%6</pre>"
        "<h2>Runtime status</h2><pre>%7</pre>"
        "<h2>External inputs</h2><ul>%8</ul><h2>Artifacts</h2><ul>%9</ul>"
        "<h2>Limitations</h2><ul>%10</ul><h2>Failure</h2><pre>%11</pre></body></html>");
    html = html.arg(bundle.projectIdentity.toHtmlEscaped())
               .arg(bundle.task.id.toString().toHtmlEscaped())
               .arg(taskStateToString(bundle.task.state).toHtmlEscaped())
               .arg(bundle.task.capabilityId.toHtmlEscaped())
               .arg(bundle.task.taskType.toHtmlEscaped())
               .arg(compactJson(bundle.backendEnvironment).toHtmlEscaped())
               .arg(compactJson(bundle.runtimeStatus).toHtmlEscaped())
               .arg(externalInputItems)
               .arg(artifactItems)
               .arg(limitationItems)
               .arg(failureText(bundle.task.failure).toHtmlEscaped());
    return html;
}

QByteArray EvidenceRendererV2::renderModelCard(const EvidenceBundleV2& bundle, QString* error)
{
    const QJsonObject evidence = encodeEvidenceBundleV2(bundle, error);
    if (evidence.isEmpty()) return {};
    // Model Card 不复制或重新计算 task/runtime 结论，只嵌入同一事实 Bundle。
    return QJsonDocument(QJsonObject{{QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("kind"), QStringLiteral("aitrain_model_card_v2")},
        {QStringLiteral("evidence"), evidence}}).toJson(QJsonDocument::Indented);
}

} // namespace aitrain::v2
