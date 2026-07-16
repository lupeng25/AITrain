#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/v2/ArtifactStoreV2.h"
#include "aitrain/v2/DatasetDriverV2.h"

#include <QJsonObject>

#include <functional>

namespace aitrain::v2 {

struct DatasetConversionRequestV2 final {
    QString sourcePath;
    QString sourceFormat;
    QString targetFormat;
    QJsonObject options;
};

struct DatasetConversionArtifactV2 final {
    ArtifactId artifactId;
    QString artifactPath;
    QString planPath;
    QString conversionReportPath;
    QJsonObject plan;
    QJsonObject conversionReport;
    DatasetValidationResult targetValidation;
};

enum class DatasetConversionFailPointV2 {
    AfterPlan,
    AfterMaterialize,
    AfterTargetValidation
};

enum class DatasetConversionIoOperationV2 {
    MaterializeWrite,
    ReportWrite,
    Commit
};

// 测试注入器也可在返回 false 前改变源文件，用于验证 plan 后源数据变化。
using DatasetConversionFailureInjectorV2 =
    std::function<bool(DatasetConversionFailPointV2 point, const QString& stagingPath)>;

// 仅用于稳定复现平台难以可靠制造的 I/O 故障。返回非空稳定错误码时，
// 转换必须在对应写入/提交边界停止，且不得登记正式 Artifact。
using DatasetConversionIoFailureInjectorV2 =
    std::function<QString(DatasetConversionIoOperationV2 operation, const QString& path)>;

// 转换结果只能进入 Artifact staging。目标 Driver 校验通过后，ArtifactStoreV2
// 才会把目录与 SQLite 文件清单一起原子提交。
class DatasetConversionServiceV2 final {
public:
    DatasetConversionServiceV2(ArtifactStoreV2* artifactStore,
        StorageV2* storage,
        const DatasetDriverRegistryV2* drivers,
        DatasetConversionFailureInjectorV2 failureInjector = {},
        DatasetConversionIoFailureInjectorV2 ioFailureInjector = {});

    bool convert(const TaskId& taskId,
        const DatasetConversionRequestV2& request,
        DatasetConversionArtifactV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {},
        const std::function<void(int percent, const QString& message)>& progress = {}) const;

private:
    ArtifactStoreV2* artifactStore_ = nullptr;
    StorageV2* storage_ = nullptr;
    const DatasetDriverRegistryV2* drivers_ = nullptr;
    DatasetConversionFailureInjectorV2 failureInjector_;
    DatasetConversionIoFailureInjectorV2 ioFailureInjector_;
};

} // namespace aitrain::v2
