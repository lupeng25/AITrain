#pragma once

#include "aitrain/v2/StorageV2.h"

#include <QJsonObject>
#include <QStringList>
#include <QVector>

namespace aitrain::v2 {

inline constexpr int kEvidenceBundleV2SchemaVersion = 2;

// Evidence Artifact 只索引已经提交的 Artifact；facts 可以携带文件摘要、来源和
// 运行时观察值，但不能把暂存目录或未提交路径当作可交付证据。
struct EvidenceArtifactV2 final {
    ArtifactId artifactId;
    QString kind;
    QJsonObject facts;
};

// 跨任务输入只记录已持久化身份与摘要，不暴露 Artifact Store 或源数据集的
// 绝对路径。producerTaskId 是产出该不可变输入的任务，而非当前 Evidence 任务。
struct EvidenceExternalInputV2 final {
    QString role;
    TaskId producerTaskId;
    ArtifactId artifactId;
    DatasetId datasetId;
    DatasetVersionId datasetVersionId;
    SnapshotId datasetSnapshotId;
    QString manifestSha256;
    QString rootHash;
};

// Evidence Bundle 是报告的唯一事实输入。Renderer 只能序列化/展示这些字段，
// 不得通过 GUI 状态、磁盘猜测或能力分支重新推导产品结论。
struct EvidenceBundleV2 final {
    int schemaVersion = kEvidenceBundleV2SchemaVersion;
    QString projectIdentity;
    TaskSnapshot task;
    WorkflowRunId workflowRunId;
    SnapshotId datasetSnapshotId;
    QJsonObject backendEnvironment;
    QJsonObject parameters;
    QJsonObject metrics;
    QJsonObject runtimeStatus;
    QJsonObject evaluation;
    QJsonObject benchmark;
    QVector<EvidenceExternalInputV2> externalInputs;
    QVector<EvidenceArtifactV2> artifacts;
    QStringList limitations;
    QDateTime createdAt;
};

bool validateEvidenceBundleV2(const EvidenceBundleV2& bundle, QString* error = nullptr);
QJsonObject encodeEvidenceBundleV2(const EvidenceBundleV2& bundle, QString* error = nullptr);
bool decodeEvidenceBundleV2(const QJsonObject& object, EvidenceBundleV2* bundle, QString* error = nullptr);

} // namespace aitrain::v2
