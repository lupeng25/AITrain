#pragma once

#include "aitrain/v2/DomainTypes.h"

#include <QJsonObject>
#include <QStringList>
#include <QVector>

namespace aitrain::v2 {

inline constexpr int kModelManifestV2SchemaVersion = 3;

enum class ModelManifestStatusV2 {
    Valid,
    Unclassified,
    Invalid
};

struct TensorContractV2 final {
    QString name;
    QString layout;
    QVector<qint64> shape;
};

struct ModelManifestV2 final {
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
    QVector<TensorContractV2> inputs;
    QVector<TensorContractV2> outputs;
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

bool validateModelManifestV2(const ModelManifestV2& manifest, QString* error = nullptr);
QJsonObject encodeModelManifestV2(const ModelManifestV2& manifest, QString* error = nullptr);
bool decodeModelManifestV2(const QJsonObject& object, ModelManifestV2* manifest, QString* error = nullptr);
// 导入草稿不携带由导入任务生成的 sourceTaskId 和 sourceArtifactSha256；
// 其余字段仍遵循完整 Manifest 的严格校验，最终值仅在 Artifact 提交后写入。
bool decodeModelManifestImportDraftV2(const QJsonObject& object, ModelManifestV2* manifest, QString* error = nullptr);
ModelManifestStatusV2 modelManifestStatusV2(const QJsonObject* object, QString* error = nullptr);
bool canUseModelManifestV2ForRuntime(const QJsonObject* object, const QString& runtimeRoute, QString* error = nullptr);

} // namespace aitrain::v2
