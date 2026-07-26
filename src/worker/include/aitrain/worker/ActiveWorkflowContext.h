#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QString>

#include <memory>

namespace aitrain {

class ProjectWorkspace;

enum class TaskCommandKind {
    EnvironmentCheck,
    DatasetSplit,
    DatasetConversion,
    DataQuality,
    Diagnostics,
    ExternalEvidenceImport,
    AnnotationCreate,
    AnnotationSync,
    DatasetSnapshotImport,
    OcrReportImport,
    OcrAcceptance,
    RuntimeDelivery,
    ModelImport,
    Training
};

enum class ActiveWorkflowPhase {
    Inactive,
    Running,
    CancelRequested,
    DurableTerminal
};

class ActiveWorkflowContext final {
public:
    ActiveWorkflowContext();
    ~ActiveWorkflowContext();
    ActiveWorkflowContext(const ActiveWorkflowContext&) = delete;
    ActiveWorkflowContext& operator=(const ActiveWorkflowContext&) = delete;

    bool begin(TaskCommandKind kind, QString* error = nullptr);
    bool bind(std::unique_ptr<ProjectWorkspace> workspace,
        const TaskId& taskId, QString* error = nullptr);
    bool requestCancel();
    bool markDurableTerminal(QString* error = nullptr);
    void clear();

    ActiveWorkflowPhase phase() const;
    TaskCommandKind commandKind() const;
    bool cancellationRequested() const;
    ProjectWorkspace* workspace() const;
    const TaskId& taskId() const;

private:
    TaskCommandKind commandKind_ = TaskCommandKind::EnvironmentCheck;
    ActiveWorkflowPhase phase_ = ActiveWorkflowPhase::Inactive;
    bool cancellationRequested_ = false;
    std::unique_ptr<ProjectWorkspace> workspace_;
    TaskId taskId_;
};

} // namespace aitrain
