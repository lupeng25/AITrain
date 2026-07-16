#pragma once

#include "aitrain/v2/ArtifactStoreV2.h"
#include "aitrain/v2/TaskCoordinator.h"
#include "aitrain/core/Cancellation.h"

namespace aitrain::v2 {

struct ModelImportRequestV2 final {
    // Worker/GUI 已分配的任务 ID 可直接成为 V2 任务 ID，禁止为同一业务动作创建双身份。
    TaskId taskId;
    QString sourceFilePath;
    ModelManifestV2 manifest;
};

struct ModelImportResultV2 final {
    TaskSnapshot task;
    ModelPackageSnapshotV2 modelPackage;
    QString artifactPath;
};

class ModelImportServiceV2 final {
public:
    ModelImportServiceV2(TaskCoordinator* coordinator, ArtifactStoreV2* artifactStore);

    bool importModel(const ModelImportRequestV2& request,
        ModelImportResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});

private:
    TaskCoordinator* coordinator_ = nullptr;
    ArtifactStoreV2* artifactStore_ = nullptr;
};

} // namespace aitrain::v2
