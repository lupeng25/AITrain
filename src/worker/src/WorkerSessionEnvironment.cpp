#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/DatasetValidators.h"
#include "aitrain/core/AnnotationIntegration.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/runtime/RuntimeCapabilityMatrix.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QStandardPaths>
#include <QThread>

#include <utility>

using namespace worker_support;
namespace wp = aitrain::worker_protocol;

void WorkerSession::runEnvironmentCheckWorkflow(const wp::EnvironmentCheckCommand& command)
{
    const aitrain::TaskId taskId = command.context.taskId;
    const QString taskIdText = taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    QString error;
    if (!taskId.isValid()) {
        error = QStringLiteral("TaskId 无效。");
    }
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        fail(QStringLiteral("Environment Check  请求缺少有效项目或 TaskId：%1").arg(error));
        return;
    }
    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 Environment Check  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId, QStringLiteral("environment.check"),
            QStringLiteral("environment_check"), &task, &error)) {
        fail(QStringLiteral("无法创建 Environment Check  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 Environment Check 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();
    activeTaskId_ = taskIdText;
    const QString reportDir = activeWorkspace->runtimeStagingPath(taskId);
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 5},
        {wp::field::message(), QStringLiteral("Environment Check  正在探测本机依赖。")}});

    QJsonArray checks;
    checks.append(nvidiaSmiCheck());
    const QVector<aitrain::RuntimeDependencyCheck> runtimeChecks =
        aitrain::defaultRuntimeDependencyChecks(QCoreApplication::applicationDirPath());
    for (const aitrain::RuntimeDependencyCheck& check : runtimeChecks) {
        checks.append(check.toJson());
    }
    const PythonExecutableResolution yoloPython =
        resolvePythonExecutable(QStringLiteral("yolo"));
    const PythonExecutableResolution smpPython =
        resolvePythonExecutable(QStringLiteral("smp_semantic_segmentation"));
    const PythonExecutableResolution anomalyPython =
        resolvePythonExecutable(QStringLiteral("anomaly_detection"));
    const PythonExecutableResolution ocrPython =
        resolvePythonExecutable(QStringLiteral("ocr"));
    checks.append(runPythonCommandCheck(
        QStringLiteral("Python"),
        yoloPython.executable,
        QStringList() << QStringLiteral("--version"),
        5000,
        yoloPython.message.isEmpty()
            ? QStringLiteral("Python executable is unavailable.") : yoloPython.message));

    const aitrain::WorkflowResult xAnyEnvironment =
        aitrain::inspectXAnyLabelingEnvironment(reportDir, {}, pollingCancellationCallback(20));
    if (activeWorkflow_.cancellationRequested()) {
        aitrain::EnvironmentCheckWorkflowRequest canceledRequest;
        canceledRequest.facts = QJsonObject{
            {QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
            {QStringLiteral("checks"), checks},
            {QStringLiteral("profiles"), QJsonObject{{QStringLiteral("canceled"), QJsonObject{
                {QStringLiteral("title"), QStringLiteral("Canceled")},
                {QStringLiteral("status"), QStringLiteral("warning")}}}}}};
        aitrain::EnvironmentCheckWorkflowResult canceledResult;
        activeWorkspace->runEnvironmentCheckWorkflow(taskId, canceledRequest, &canceledResult,
            &error, []() { return true; });
        publishPersistedTerminal(taskId);
        return;
    }
    const QJsonObject xAnyPayload = xAnyEnvironment.payload;
    const QString xAnyStatus = xAnyPayload.value(QStringLiteral("status")).toString(
        xAnyEnvironment.ok ? QStringLiteral("ok") : QStringLiteral("missing"));
    const QString xAnyMessage = xAnyPayload.value(QStringLiteral("message")).toString(xAnyEnvironment.error);
    checks.append(checkObject(
        QStringLiteral("X-AnyLabeling"),
        xAnyStatus,
        xAnyMessage,
        xAnyPayload));
    checks.append(checkObject(QStringLiteral("Worker"), QStringLiteral("ok"), QStringLiteral("Worker 环境自检命令可用。")));

    QJsonObject profiles;
    profiles.insert(QStringLiteral("yolo"), yoloEnvironmentProfile(yoloPython.executable));
    profiles.insert(QStringLiteral("smp_semantic_segmentation"),
        smpEnvironmentProfile(smpPython.executable));
    profiles.insert(QStringLiteral("anomaly_detection"),
        anomalibEnvironmentProfile(anomalyPython.executable));
    profiles.insert(QStringLiteral("ocr"), ocrEnvironmentProfile(ocrPython.executable));
    profiles.insert(QStringLiteral("tensorrt"), tensorRtEnvironmentProfile(checks));
    {
        QJsonArray profileChecks;
        QJsonObject details;
        details.insert(QStringLiteral("executable"), xAnyPayload.value(QStringLiteral("executable")).toString());
        details.insert(QStringLiteral("licenseBoundary"), xAnyPayload.value(QStringLiteral("licenseBoundary")).toString());
        details.insert(QStringLiteral("redistributionReviewRequired"), xAnyPayload.value(QStringLiteral("redistributionReviewRequired")).toBool(true));
        profileChecks.append(profileCheck(
            QStringLiteral("xanylabelingExecutable"),
            xAnyStatus,
            xAnyMessage,
            details));
        QJsonArray repairHints;
        repairHints.append(QStringLiteral("Set `AITRAIN_XANYLABELING_EXE` to the local X-AnyLabeling executable."));
        repairHints.append(QStringLiteral("Or place X-AnyLabeling under `.deps/tools/annotation-tools/X-AnyLabeling`."));
        repairHints.append(QStringLiteral("Keep X-AnyLabeling as a local external dependency unless redistribution has a separate license/package review."));
        profiles.insert(
            QStringLiteral("xanylabeling"),
            makeProfile(QStringLiteral("xanylabeling"), QStringLiteral("X-AnyLabeling Profile"), profileChecks, repairHints));
    }

    QJsonObject result;
    result.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    result.insert(QStringLiteral("checks"), checks);
    result.insert(QStringLiteral("profiles"), profiles);
    result.insert(QStringLiteral("runtimeCapabilityMatrix"), aitrain::RuntimeCapabilityMatrix().toJson());

    aitrain::EnvironmentCheckWorkflowRequest request;
    request.facts = result;
    aitrain::EnvironmentCheckWorkflowResult workflowResult;
    const bool executed = activeWorkspace->runEnvironmentCheckWorkflow(
        taskId, request, &workflowResult, &error, pollingCancellationCallback(0));
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Environment Check  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查  工作区和 Evidence 后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            activeWorkspace->finalizeTask(taskId, aitrain::TaskState::Failed, failure, nullptr);
        }
        publishPersistedTerminal(taskId);
        return;
    }
    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), workflowResult.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(workflowResult.terminalState)},
        {QStringLiteral("reportArtifactId"), workflowResult.reportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), workflowResult.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), workflowResult.summary}};
    send(wp::event::environmentCheckWorkflow(), response);
    publishPersistedTerminal(taskId,
        QStringLiteral("Environment Check  completed"));
}
