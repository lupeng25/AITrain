#pragma once

#include "aitrain/artifact/ArtifactStore.h"
#include "aitrain/workflow/TaskCoordinator.h"
#include "aitrain/core/Cancellation.h"

namespace aitrain {

struct ModelImportRequest final {
    // Worker/GUI 已分配的任务 ID 可直接成为  任务 ID，禁止为同一业务动作创建双身份。
    TaskId taskId;
    QString sourceFilePath;
    ModelManifest manifest;
};

struct ModelImportResult final {
    TaskSnapshot task;
    ModelPackageSnapshot modelPackage;
    QString artifactPath;
};

class ModelImportService final {
public:
    ModelImportService(TaskCoordinator* coordinator, ArtifactStore* artifactStore);

    bool importModel(const ModelImportRequest& request,
        ModelImportResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});

private:
    TaskCoordinator* coordinator_ = nullptr;
    ArtifactStore* artifactStore_ = nullptr;
};

} // namespace aitrain
