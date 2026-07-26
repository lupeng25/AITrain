#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

// 单个 Workflow 步骤的静态路由合同。artifactCandidates 使用 Artifact
// manifest 中的“kind/相对路径”表示法，并按优先级排列；空列表表示该步骤
// 不从上游产物中选择输入，而不是允许任意产物。
struct TrainingWorkflowStepProfile final {
    QString kind;
    QString backend;
    QString script;
    QStringList artifactCandidates;
};

// 训练 Workflow Profile 只保存由产品代码维护的静态可信事实。
// 它不读取用户请求，也不探测模型文件；Worker/Workspace 后续可据此选择
// Adapter、验证数据集/模型合同并生成准确的产品限制。
struct TrainingWorkflowProfile final {
    QString trainingBackend;
    QString templateId;
    QString capabilityTaskType;
    QString adapterTaskType;
    QString datasetFormat;
    QString modelFamily;
    QString decoder;
    QString artifactFormat;
    QString trainScript;
    QString evaluationBackend;
    QString evaluationScript;
    QString exportBackend;
    QString exportScript;
    QString deploymentBackend;
    QStringList runtimeRoutes;
    QString pythonProfileId;
    QStringList limitations;
    QVector<TrainingWorkflowStepProfile> steps;
};

// 返回编译期注册的规范 Profile。调用方不得修改返回内容。
const QVector<TrainingWorkflowProfile>& trainingWorkflowProfiles();

// backend 匹配不区分大小写并忽略首尾空白；调用方必须使用注册表中的规范 ID。
bool resolveTrainingWorkflowProfile(const QString& trainingBackend,
    TrainingWorkflowProfile* result,
    QString* error = nullptr);
bool hasTrainingWorkflowProfile(const QString& trainingBackend);

} // namespace aitrain
