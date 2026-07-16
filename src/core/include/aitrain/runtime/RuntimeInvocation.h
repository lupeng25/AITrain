#pragma once

#include "aitrain/runtime/RuntimeAdapter.h"

#include <QJsonObject>

namespace aitrain {

// Worker 边界使用的、已经完成模型准入的推理调用描述。它不承担模型发现或类型推断；
// 调用方必须先从 ModelPackageId 解析出 RuntimeModelLocation。
struct RuntimeInvocation final {
    RuntimeModelLocation model;
    QString runtimeRoute;
    QString imagePath;
    QString outputPath;
    QJsonObject options;
};

QJsonObject encodeRuntimeInvocation(const RuntimeInvocation& invocation, QString* error = nullptr);
bool decodeRuntimeInvocation(const QJsonObject& object, RuntimeInvocation* invocation, QString* error = nullptr);

} // namespace aitrain
