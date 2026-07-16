#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/artifact/ArtifactStore.h"
#include "aitrain/dataset/DatasetDriver.h"

#include <QJsonObject>

#include <functional>

namespace aitrain {

struct DatasetArtifactConversionRequest final {
    QString sourcePath;
    QString sourceFormat;
    QString targetFormat;
    QJsonObject options;
};

struct DatasetArtifactConversion final {
    ArtifactId artifactId;
    QString artifactPath;
    QString planPath;
    QString conversionReportPath;
    QJsonObject plan;
    QJsonObject conversionReport;
    DatasetDriverValidationResult targetValidation;
};

enum class DatasetConversionFailPoint {
    AfterPlan,
    AfterMaterialize,
    AfterTargetValidation
};

enum class DatasetConversionIoOperation {
    MaterializeWrite,
    ReportWrite,
    Commit
};

// 测试注入器也可在返回 false 前改变源文件，用于验证 plan 后源数据变化。
using DatasetConversionFailureInjector =
    std::function<bool(DatasetConversionFailPoint point, const QString& stagingPath)>;

// 仅用于稳定复现平台难以可靠制造的 I/O 故障。返回非空稳定错误码时，
// 转换必须在对应写入/提交边界停止，且不得登记正式 Artifact。
using DatasetConversionIoFailureInjector =
    std::function<QString(DatasetConversionIoOperation operation, const QString& path)>;

// 转换结果只能进入 Artifact staging。目标 Driver 校验通过后，ArtifactStore
// 才会把目录与 SQLite 文件清单一起原子提交。
class DatasetConversionService final {
public:
    DatasetConversionService(ArtifactStore* artifactStore,
        ProjectStore* storage,
        const DatasetDriverRegistry* drivers,
        DatasetConversionFailureInjector failureInjector = {},
        DatasetConversionIoFailureInjector ioFailureInjector = {});

    bool convert(const TaskId& taskId,
        const DatasetArtifactConversionRequest& request,
        DatasetArtifactConversion* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {},
        const std::function<void(int percent, const QString& message)>& progress = {}) const;

private:
    ArtifactStore* artifactStore_ = nullptr;
    ProjectStore* storage_ = nullptr;
    const DatasetDriverRegistry* drivers_ = nullptr;
    DatasetConversionFailureInjector failureInjector_;
    DatasetConversionIoFailureInjector ioFailureInjector_;
};

} // namespace aitrain
