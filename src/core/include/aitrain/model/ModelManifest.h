#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QJsonObject>
#include <QStringList>
#include <QVector>

namespace aitrain {

inline constexpr int kModelManifestSchemaVersion = 3;

enum class ModelManifestStatus {
    Valid,
    Unclassified,
    Invalid
};

struct TensorContract final {
    QString name;
    QString layout;
    QVector<qint64> shape;
};

struct ModelManifest final {
    ModelPackageId modelPackageId;
    QString modelFamily;
    QString taskType;
    QString sourceBackend;
    TaskId sourceTaskId;
    SnapshotId sourceSnapshotId;
    QString sourceArtifactSha256;
    QString artifactEntryPath;
    // 模型包的判别字段。onnx 包必须携带 tensor/opset 合同；
    // ncnn / tensorrt_engine 包必须携带明确 tensor/blob 合同且 opset=0；
    // anomalib_bundle / paddleocr_inference_bundle 以官方 Python sidecar 为入口，
    // 不伪造 ONNX 元数据。
    QString artifactFormat = QStringLiteral("onnx");
    QVector<TensorContract> inputs;
    QVector<TensorContract> outputs;
    QJsonObject preprocessing;
    QJsonObject postprocessing;
    QString decoder;
    QStringList classNames;
    int opset = 0;
    QString exporterVersion;
    QStringList runtimeRoutes;
    bool verified = false;
    QStringList limitations;
};

bool validateModelManifest(const ModelManifest& manifest, QString* error = nullptr);
QJsonObject encodeModelManifest(const ModelManifest& manifest, QString* error = nullptr);
bool decodeModelManifest(const QJsonObject& object, ModelManifest* manifest, QString* error = nullptr);
// 导入草稿不携带由导入任务生成的 sourceTaskId 和 sourceArtifactSha256；
// 其余字段仍遵循完整 Manifest 的严格校验，最终值仅在 Artifact 提交后写入。
bool decodeModelManifestImportDraft(const QJsonObject& object, ModelManifest* manifest, QString* error = nullptr);
ModelManifestStatus modelManifestStatus(const QJsonObject* object, QString* error = nullptr);
bool canUseModelManifestForRuntime(const QJsonObject* object, const QString& runtimeRoute, QString* error = nullptr);

} // namespace aitrain
