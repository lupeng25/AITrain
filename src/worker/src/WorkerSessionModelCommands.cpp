#include "WorkerSession.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/model/ModelManifest.h"

#include <QJsonArray>

#include <utility>

namespace wp = aitrain::worker_protocol;

void WorkerSession::importModel(const wp::ModelImportCommand& command)
{
    const QString taskId = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    const QString sourceFilePath = command.sourceFilePath.trimmed();
    const QJsonObject manifestDraft = command.manifestDraft;
    if (taskId.isEmpty() || projectRoot.isEmpty() || sourceFilePath.isEmpty() || manifestDraft.isEmpty()) {
        fail(QStringLiteral("importModel 需要 taskId、项目目录、模型文件和用户确认的 Manifest 草稿。"));
        return;
    }
    aitrain::TaskId parsedTaskId;
    QString error;
    if (!aitrain::TaskId::parse(taskId, &parsedTaskId, &error) || parsedTaskId != controlTaskId_) {
        const QString reason = error.isEmpty()
            ? QStringLiteral("payload taskId 与 Protocol 控制 TaskId 不一致。")
            : error;
        fail(QStringLiteral("importModel 要求 payload taskId 为有效 UUID 且与 Protocol 控制 TaskId 一致：%1")
                .arg(reason));
        return;
    }
    aitrain::ModelManifest manifest;
    if (!aitrain::decodeModelManifestImportDraft(manifestDraft, &manifest, &error)) {
        fail(QStringLiteral("importModel 拒绝无效 Manifest 草稿：%1").arg(error));
        return;
    }
    activeTaskId_ = taskId;
    QJsonObject started;
    started.insert(wp::field::taskId(), taskId);
    started.insert(QStringLiteral("percent"), 0);
    started.insert(wp::field::message(), QStringLiteral(" 模型导入开始：正在复制并计算 SHA-256。"));
    send(wp::event::progress(), started);

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开  项目工作区：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), parsedTaskId, &error)) {
        fail(QStringLiteral("无法绑定模型导入活动任务：%1").arg(error));
        return;
    }
    aitrain::ModelImportRequest request;
    request.taskId = parsedTaskId;
    request.sourceFilePath = sourceFilePath;
    request.manifest = manifest;
    aitrain::ModelImportResult result;
    if (!activeWorkflow_.workspace()->importModel(
            request, &result, &error, pollingCancellationCallback(0))) {
        publishPersistedTerminal(parsedTaskId);
        return;
    }
    QJsonObject response;
    response.insert(wp::field::taskId(), taskId);
    response.insert(QStringLiteral("modelPackageId"), result.modelPackage.manifest.modelPackageId.toString());
    response.insert(QStringLiteral("modelFamily"), result.modelPackage.manifest.modelFamily);
    response.insert(QStringLiteral("taskType"), result.modelPackage.manifest.taskType);
    response.insert(QStringLiteral("runtimeRoutes"), QJsonArray::fromStringList(result.modelPackage.manifest.runtimeRoutes));
    response.insert(wp::field::message(), QStringLiteral(" 模型导入完成，模型包已原子登记。"));
    send(wp::event::modelImport(), response);
    publishPersistedTerminal(parsedTaskId,
        QStringLiteral(" model import completed"));
}
