#include "DiagnosticBundlePresenterV2.h"
#include "EnvironmentCheckPresenterV2.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/v2/ProjectQueryServiceV2.h"
#include "aitrain/v2/ProjectWorkspaceV2.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>

namespace wp = aitrain::worker_protocol;

namespace {

bool startDiagnosticsTask(aitrain::v2::ProjectWorkspaceV2* workspace,
    const QString& root, aitrain::v2::TaskId* taskId, QString* error)
{
    if (!workspace->open(root, error)) return false;
    *taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    return workspace->startTask(*taskId, QStringLiteral("diagnostics.bundle.v2"),
        QStringLiteral("diagnostics"), &task, error);
}

bool jsonContainsAbsolutePath(const QJsonValue& value)
{
    if (value.isString()) {
        const QString text = value.toString();
        return QDir::isAbsolutePath(text)
            || (text.size() > 2 && text.at(1) == QLatin1Char(':')
                && (text.at(2) == QLatin1Char('/') || text.at(2) == QLatin1Char('\\')));
    }
    if (value.isArray()) {
        for (const QJsonValue& item : value.toArray()) {
            if (jsonContainsAbsolutePath(item)) return true;
        }
    } else if (value.isObject()) {
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it) {
            if (jsonContainsAbsolutePath(it.value())) return true;
        }
    }
    return false;
}

} // namespace

class DiagnosticsV2Tests : public QObject {
    Q_OBJECT

private slots:
    void coreSuccessCommitsFactsBundleAndEvidence();
    void coreFailureStillCommitsEvidence();
    void coreCancellationStillCommitsEvidence();
    void protocolAndPresenterExposeIdsWithoutPaths();
    void environmentWorkflowAndPresenterUseCommittedSanitizedReport();
    void environmentInvalidFactsAndCancellationStillCommitEvidence();
};

void DiagnosticsV2Tests::coreSuccessCommitsFactsBundleAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    aitrain::v2::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    aitrain::v2::DiagnosticsWorkflowRequestV2 request;
    request.options = {{QStringLiteral("probeTimeoutMs"), 500},
        {QStringLiteral("probeOutputBytes"), 1024}};
    aitrain::v2::DiagnosticsWorkflowResultV2 result;
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, request, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Succeeded);
    QVERIFY(result.workflowRunId.isValid());
    QVERIFY(result.factsArtifactId.isValid());
    QVERIFY(result.diagnosticsArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());

    const QVector<aitrain::v2::ArtifactSnapshotV2> artifacts = workspace.artifactsForTask(taskId, &error);
    QStringList kinds;
    for (const auto& artifact : artifacts) kinds.append(artifact.kind);
    QVERIFY(kinds.contains(QStringLiteral("diagnostic_facts_v2")));
    QVERIFY(kinds.contains(QStringLiteral("diagnostic_bundle_v2")));
    QVERIFY(kinds.contains(QStringLiteral("evidence_bundle_v2")));
}

void DiagnosticsV2Tests::coreFailureStillCommitsEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    aitrain::v2::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    bool tampered = false;
    const auto tamperAfterCollect = [&]() {
        if (tampered) return false;
        QDirIterator iterator(QDir(directory.path()).filePath(QStringLiteral(".aitrain-v2/artifact-store/artifacts")),
            QStringList{QStringLiteral("diagnostic_facts_v2.json")}, QDir::Files, QDirIterator::Subdirectories);
        if (!iterator.hasNext()) return false;
        const QString factsPath = iterator.next();
        QFile::setPermissions(factsPath,
            QFile::permissions(factsPath) | QFileDevice::WriteOwner | QFileDevice::WriteUser);
        QFile file(factsPath);
        if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            file.write("{\"tampered\":true}");
            file.close();
            tampered = true;
        }
        return false;
    };
    aitrain::v2::DiagnosticsWorkflowRequestV2 request;
    request.options = {{QStringLiteral("probeTimeoutMs"), 500}};
    aitrain::v2::DiagnosticsWorkflowResultV2 result;
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, request, &result, &error, tamperAfterCollect), qPrintable(error));
    QVERIFY(tampered);
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(result.factsArtifactId.isValid());
    QVERIFY(!result.diagnosticsArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(result.failure.code, aitrain::v2::FailureCode::ArtifactIncompatible);
}

void DiagnosticsV2Tests::coreCancellationStillCommitsEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    aitrain::v2::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    aitrain::v2::DiagnosticsWorkflowResultV2 result;
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, {}, &result, &error, []() { return true; }), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Canceled);
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(result.failure.code, aitrain::v2::FailureCode::Canceled);
}

void DiagnosticsV2Tests::protocolAndPresenterExposeIdsWithoutPaths()
{
    const QJsonObject request = wp::diagnosticsWorkflowV2Request(
        QStringLiteral("task-id"), QStringLiteral("C:/project"), QJsonObject{{QStringLiteral("taskLimit"), 10}});
    QCOMPARE(request.size(), 3);
    QVERIFY(request.contains(QStringLiteral("taskId")));
    QVERIFY(request.contains(QStringLiteral("projectRoot")));
    QVERIFY(request.contains(QStringLiteral("options")));
    QVERIFY(!request.contains(QStringLiteral("outputPath")));
    QVERIFY(!request.contains(QStringLiteral("context")));

    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    aitrain::v2::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    aitrain::v2::DiagnosticsWorkflowResultV2 result;
    aitrain::v2::DiagnosticsWorkflowRequestV2 workflowRequest;
    workflowRequest.options = {{QStringLiteral("probeTimeoutMs"), 500}};
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, workflowRequest, &result, &error), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    DiagnosticBundlePresenterV2 presenter(&query);
    QVERIFY2(presenter.selectTask(taskId.toString()), qPrintable(presenter.lastError()));
    const DiagnosticBundleViewModelV2 model = presenter.viewModel();
    QCOMPARE(model.taskId, taskId.toString());
    QCOMPARE(model.diagnosticsArtifactId, result.diagnosticsArtifactId.toString());
    QCOMPARE(model.evidenceArtifactId, result.evidenceArtifactId.toString());
    const QJsonObject exposed{{QStringLiteral("taskId"), model.taskId},
        {QStringLiteral("state"), model.state},
        {QStringLiteral("diagnosticsArtifactId"), model.diagnosticsArtifactId},
        {QStringLiteral("evidenceArtifactId"), model.evidenceArtifactId}};
    QVERIFY(!jsonContainsAbsolutePath(exposed));
}

void DiagnosticsV2Tests::environmentWorkflowAndPresenterUseCommittedSanitizedReport()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.path(), &error), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("environment.check.v2"),
        QStringLiteral("environment_check"), &task, &error), qPrintable(error));

    aitrain::v2::EnvironmentCheckWorkflowRequestV2 request;
    request.facts = QJsonObject{
        {QStringLiteral("checkedAt"), QStringLiteral("2026-07-16T00:00:00.000Z")},
        {QStringLiteral("checks"), QJsonArray{QJsonObject{
            {QStringLiteral("name"), QStringLiteral("Python")},
            {QStringLiteral("status"), QStringLiteral("ok")},
            {QStringLiteral("message"), QStringLiteral("available")},
            {QStringLiteral("details"), QJsonObject{{QStringLiteral("executable"), QStringLiteral("C:/secret/python.exe")}}}}}},
        {QStringLiteral("profiles"), QJsonObject{{QStringLiteral("yolo"), QJsonObject{
            {QStringLiteral("title"), QStringLiteral("YOLO")},
            {QStringLiteral("status"), QStringLiteral("ok")},
            {QStringLiteral("repairHints"), QJsonArray{QStringLiteral("none")}},
            {QStringLiteral("checks"), QJsonArray{QJsonObject{
                {QStringLiteral("name"), QStringLiteral("torch")},
                {QStringLiteral("status"), QStringLiteral("ok")},
                {QStringLiteral("message"), QStringLiteral("ready")},
                {QStringLiteral("details"), QJsonObject{{QStringLiteral("path"), QStringLiteral("D:/secret")}}}}}}}}}}};
    aitrain::v2::EnvironmentCheckWorkflowResultV2 result;
    QVERIFY2(workspace.runEnvironmentCheckWorkflow(taskId, request, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Succeeded);
    QVERIFY(result.reportArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    EnvironmentCheckPresenterV2 presenter(&query);
    QVERIFY2(presenter.selectTask(taskId.toString()), qPrintable(presenter.lastError()));
    const EnvironmentCheckViewModelV2 model = presenter.viewModel();
    QCOMPARE(model.reportArtifactId, result.reportArtifactId.toString());
    QCOMPARE(model.evidenceArtifactId, result.evidenceArtifactId.toString());
    QVERIFY(!jsonContainsAbsolutePath(model.report));
    QVERIFY(!model.report.value(QStringLiteral("checks")).toArray().first().toObject()
        .contains(QStringLiteral("details")));
}

void DiagnosticsV2Tests::environmentInvalidFactsAndCancellationStillCommitEvidence()
{
    for (const bool cancel : {false, true}) {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        aitrain::v2::ProjectWorkspaceV2 workspace;
        QString error;
        QVERIFY2(workspace.open(directory.path(), &error), qPrintable(error));
        const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
        aitrain::v2::TaskSnapshot task;
        QVERIFY2(workspace.startTask(taskId, QStringLiteral("environment.check.v2"),
            QStringLiteral("environment_check"), &task, &error), qPrintable(error));
        aitrain::v2::EnvironmentCheckWorkflowRequestV2 request;
        request.facts = QJsonObject{{QStringLiteral("checks"), QJsonArray{}},
            {QStringLiteral("profiles"), QJsonObject{}}};
        aitrain::v2::EnvironmentCheckWorkflowResultV2 result;
        QVERIFY2(workspace.runEnvironmentCheckWorkflow(taskId, request, &result, &error,
            cancel ? aitrain::CancellationCallback([]() { return true; }) : aitrain::CancellationCallback{}),
            qPrintable(error));
        QCOMPARE(result.terminalState, cancel
            ? aitrain::v2::TaskState::Canceled : aitrain::v2::TaskState::Failed);
        QVERIFY(result.evidenceArtifactId.isValid());
        QCOMPARE(result.failure.code, cancel
            ? aitrain::v2::FailureCode::Canceled : aitrain::v2::FailureCode::InvalidRequest);
    }
}

QTEST_MAIN(DiagnosticsV2Tests)
#include "tst_diagnostics_v2.moc"
