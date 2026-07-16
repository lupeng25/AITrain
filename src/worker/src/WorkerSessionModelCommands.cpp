#include "WorkerSession.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/v2/ProjectWorkspaceV2.h"
#include "aitrain/v2/ModelManifestV2.h"

#include <QJsonArray>

namespace wp = aitrain::worker_protocol;

void WorkerSession::importModelV2(const QJsonObject& payload)
{
    const QString taskId = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    const QString sourceFilePath = payload.value(QStringLiteral("sourceFilePath")).toString().trimmed();
    const QJsonObject manifestDraft = payload.value(QStringLiteral("manifestDraft")).toObject();
    if (taskId.isEmpty() || projectRoot.isEmpty() || sourceFilePath.isEmpty() || manifestDraft.isEmpty()) {
        fail(QStringLiteral("importModelV2 需要 taskId、项目目录、模型文件和用户确认的 Manifest 草稿。"));
        return;
    }
    aitrain::v2::ModelManifestV2 manifest;
    QString error;
    if (!aitrain::v2::decodeModelManifestImportDraftV2(manifestDraft, &manifest, &error)) {
        fail(QStringLiteral("importModelV2 拒绝无效 Manifest 草稿：%1").arg(error));
        return;
    }
    activeTaskId_ = taskId;
    canceled_ = false;
    running_ = true;
    QJsonObject started;
    started.insert(wp::field::taskId(), taskId);
    started.insert(QStringLiteral("percent"), 0);
    started.insert(wp::field::message(), QStringLiteral("V2 模型导入开始：正在复制并计算 SHA-256。"));
    send(wp::event::progress(), started);

    aitrain::v2::ProjectWorkspaceV2 workspace;
    if (!workspace.open(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 V2 项目工作区：%1").arg(error));
        return;
    }
    aitrain::v2::ModelImportRequestV2 request;
    if (!aitrain::v2::TaskId::parse(taskId, &request.taskId, &error)) {
        fail(QStringLiteral("importModelV2 要求 taskId 为有效 UUID：%1").arg(error));
        return;
    }
    request.sourceFilePath = sourceFilePath;
    request.manifest = manifest;
    aitrain::v2::ModelImportResultV2 result;
    if (!workspace.importModel(request, &result, &error, pollingCancellationCallback(0))) {
        if (canceled_) {
            sendCanceledAndFinish(taskId, error.isEmpty() ? QStringLiteral("Canceled by user") : error);
            return;
        }
        fail(QStringLiteral("V2 模型导入失败：%1").arg(error));
        return;
    }
    QJsonObject response;
    response.insert(wp::field::taskId(), taskId);
    response.insert(QStringLiteral("modelPackageId"), result.modelPackage.manifest.modelPackageId.toString());
    response.insert(QStringLiteral("artifactPath"), result.artifactPath);
    response.insert(QStringLiteral("modelFamily"), result.modelPackage.manifest.modelFamily);
    response.insert(QStringLiteral("taskType"), result.modelPackage.manifest.taskType);
    response.insert(QStringLiteral("runtimeRoutes"), QJsonArray::fromStringList(result.modelPackage.manifest.runtimeRoutes));
    response.insert(wp::field::message(), QStringLiteral("V2 模型导入完成，模型包已原子登记。"));
    send(wp::event::modelImportV2(), response);
    QJsonObject completed;
    completed.insert(wp::field::taskId(), taskId);
    completed.insert(wp::field::message(), QStringLiteral("V2 model import completed"));
    running_ = false;
    send(wp::event::completed(), completed);
    finishSession();
}
