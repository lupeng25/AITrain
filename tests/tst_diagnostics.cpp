#include "DiagnosticBundlePresenter.h"
#include "EnvironmentCheckPresenter.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/workflow/ProjectQueryService.h"
#include "aitrain/workflow/ProjectWorkspace.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>

namespace wp = aitrain::worker_protocol;

namespace {

bool startDiagnosticsTask(aitrain::ProjectWorkspace* workspace,
    const QString& root, aitrain::TaskId* taskId, QString* error)
{
    if (!workspace->open(root, error)) return false;
    *taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    return workspace->startTask(*taskId, QStringLiteral("diagnostics.bundle"),
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

class DiagnosticsTests : public QObject {
    Q_OBJECT

private slots:
    void coreSuccessCommitsFactsBundleAndEvidence();
    void coreFailureStillCommitsEvidence();
    void coreCancellationStillCommitsEvidence();
    void protocolAndPresenterExposeIdsWithoutPaths();
    void environmentWorkflowAndPresenterUseCommittedSanitizedReport();
    void environmentInvalidFactsAndCancellationStillCommitEvidence();
};

void DiagnosticsTests::coreSuccessCommitsFactsBundleAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    aitrain::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    aitrain::DiagnosticsWorkflowRequest request;
    request.options = {{QStringLiteral("probeTimeoutMs"), 500},
        {QStringLiteral("probeOutputBytes"), 1024}};
    aitrain::DiagnosticsWorkflowResult result;
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, request, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Succeeded);
    QVERIFY(result.workflowRunId.isValid());
    QVERIFY(result.factsArtifactId.isValid());
    QVERIFY(result.diagnosticsArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());

    const QVector<aitrain::ArtifactSnapshot> artifacts = workspace.artifactsForTask(taskId, &error);
    QStringList kinds;
    for (const auto& artifact : artifacts) kinds.append(artifact.kind);
    QVERIFY(kinds.contains(QStringLiteral("diagnostic_facts")));
    QVERIFY(kinds.contains(QStringLiteral("diagnostic_bundle")));
    QVERIFY(kinds.contains(QStringLiteral("evidence_bundle")));
}

void DiagnosticsTests::coreFailureStillCommitsEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    aitrain::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    bool tampered = false;
    const auto tamperAfterCollect = [&]() {
        if (tampered) return false;
        QDirIterator iterator(QDir(directory.path()).filePath(QStringLiteral(".aitrain/artifacts/artifacts")),
            QStringList{QStringLiteral("diagnostic_facts.json")}, QDir::Files, QDirIterator::Subdirectories);
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
    aitrain::DiagnosticsWorkflowRequest request;
    request.options = {{QStringLiteral("probeTimeoutMs"), 500}};
    aitrain::DiagnosticsWorkflowResult result;
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, request, &result, &error, tamperAfterCollect), qPrintable(error));
    QVERIFY(tampered);
    QCOMPARE(result.terminalState, aitrain::TaskState::Failed);
    QVERIFY(result.factsArtifactId.isValid());
    QVERIFY(!result.diagnosticsArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(result.failure.code, aitrain::FailureCode::ArtifactIncompatible);
}

void DiagnosticsTests::coreCancellationStillCommitsEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    aitrain::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    aitrain::DiagnosticsWorkflowResult result;
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, {}, &result, &error, []() { return true; }), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Canceled);
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(result.failure.code, aitrain::FailureCode::Canceled);
}

void DiagnosticsTests::protocolAndPresenterExposeIdsWithoutPaths()
{
    const QJsonObject request = wp::diagnosticsWorkflowRequest(
        QStringLiteral("task-id"), QStringLiteral("C:/project"), QJsonObject{{QStringLiteral("taskLimit"), 10}});
    QCOMPARE(request.size(), 3);
    QVERIFY(request.contains(QStringLiteral("taskId")));
    QVERIFY(request.contains(QStringLiteral("projectRoot")));
    QVERIFY(request.contains(QStringLiteral("options")));
    QVERIFY(!request.contains(QStringLiteral("outputPath")));
    QVERIFY(!request.contains(QStringLiteral("context")));

    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    aitrain::TaskId taskId;
    QString error;
    QVERIFY2(startDiagnosticsTask(&workspace, directory.path(), &taskId, &error), qPrintable(error));
    aitrain::DiagnosticsWorkflowResult result;
    aitrain::DiagnosticsWorkflowRequest workflowRequest;
    workflowRequest.options = {{QStringLiteral("probeTimeoutMs"), 500}};
    QVERIFY2(workspace.runDiagnosticsWorkflow(taskId, workflowRequest, &result, &error), qPrintable(error));

    aitrain::ProjectQueryService query(&workspace);
    DiagnosticBundlePresenter presenter(&query);
    QVERIFY2(presenter.selectTask(taskId.toString()), qPrintable(presenter.lastError()));
    const DiagnosticBundleViewModel model = presenter.viewModel();
    QCOMPARE(model.taskId, taskId.toString());
    QCOMPARE(model.diagnosticsArtifactId, result.diagnosticsArtifactId.toString());
    QCOMPARE(model.evidenceArtifactId, result.evidenceArtifactId.toString());
    const QJsonObject exposed{{QStringLiteral("taskId"), model.taskId},
        {QStringLiteral("state"), model.state},
        {QStringLiteral("diagnosticsArtifactId"), model.diagnosticsArtifactId},
        {QStringLiteral("evidenceArtifactId"), model.evidenceArtifactId}};
    QVERIFY(!jsonContainsAbsolutePath(exposed));
}

void DiagnosticsTests::environmentWorkflowAndPresenterUseCommittedSanitizedReport()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.path(), &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("environment.check"),
        QStringLiteral("environment_check"), &task, &error), qPrintable(error));

    aitrain::EnvironmentCheckWorkflowRequest request;
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
    aitrain::EnvironmentCheckWorkflowResult result;
    QVERIFY2(workspace.runEnvironmentCheckWorkflow(taskId, request, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Succeeded);
    QVERIFY(result.reportArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());

    aitrain::ProjectQueryService query(&workspace);
    EnvironmentCheckPresenter presenter(&query);
    QVERIFY2(presenter.selectTask(taskId.toString()), qPrintable(presenter.lastError()));
    const EnvironmentCheckViewModel model = presenter.viewModel();
    QCOMPARE(model.reportArtifactId, result.reportArtifactId.toString());
    QCOMPARE(model.evidenceArtifactId, result.evidenceArtifactId.toString());
    QVERIFY(!jsonContainsAbsolutePath(model.report));
    QVERIFY(!model.report.value(QStringLiteral("checks")).toArray().first().toObject()
        .contains(QStringLiteral("details")));
}

void DiagnosticsTests::environmentInvalidFactsAndCancellationStillCommitEvidence()
{
    for (const bool cancel : {false, true}) {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        aitrain::ProjectWorkspace workspace;
        QString error;
        QVERIFY2(workspace.open(directory.path(), &error), qPrintable(error));
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        aitrain::TaskSnapshot task;
        QVERIFY2(workspace.startTask(taskId, QStringLiteral("environment.check"),
            QStringLiteral("environment_check"), &task, &error), qPrintable(error));
        aitrain::EnvironmentCheckWorkflowRequest request;
        request.facts = QJsonObject{{QStringLiteral("checks"), QJsonArray{}},
            {QStringLiteral("profiles"), QJsonObject{}}};
        aitrain::EnvironmentCheckWorkflowResult result;
        QVERIFY2(workspace.runEnvironmentCheckWorkflow(taskId, request, &result, &error,
            cancel ? aitrain::CancellationCallback([]() { return true; }) : aitrain::CancellationCallback{}),
            qPrintable(error));
        QCOMPARE(result.terminalState, cancel
            ? aitrain::TaskState::Canceled : aitrain::TaskState::Failed);
        QVERIFY(result.evidenceArtifactId.isValid());
        QCOMPARE(result.failure.code, cancel
            ? aitrain::FailureCode::Canceled : aitrain::FailureCode::InvalidRequest);
    }
}

QTEST_MAIN(DiagnosticsTests)
#include "tst_diagnostics.moc"
