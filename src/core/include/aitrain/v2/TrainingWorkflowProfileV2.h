#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain::v2 {

// 单个 Workflow 步骤的静态路由合同。artifactCandidates 使用 Artifact
// manifest 中的“kind/相对路径”表示法，并按优先级排列；空列表表示该步骤
// 不从上游产物中选择输入，而不是允许任意产物。
struct TrainingWorkflowStepProfileV2 final {
    QString kind;
    QString backend;
    QString script;
    QStringList artifactCandidates;
};

// 训练 Workflow Profile 只保存由产品代码维护的静态可信事实。
// 它不读取用户请求，也不探测模型文件；Worker/Workspace 后续可据此选择
// Adapter、验证数据集/模型合同并生成准确的产品限制。
struct TrainingWorkflowProfileV2 final {
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
    QStringList limitations;
    QVector<TrainingWorkflowStepProfileV2> steps;
};

// 返回编译期注册的规范 Profile。调用方不得修改返回内容。
const QVector<TrainingWorkflowProfileV2>& trainingWorkflowProfilesV2();

// backend 匹配不区分大小写并忽略首尾空白；历史的 ultralytics_yolo
// 别名解析到规范 ultralytics_yolo_detect Profile。
bool resolveTrainingWorkflowProfileV2(const QString& trainingBackend,
    TrainingWorkflowProfileV2* result,
    QString* error = nullptr);
bool hasTrainingWorkflowProfileV2(const QString& trainingBackend);

} // namespace aitrain::v2
