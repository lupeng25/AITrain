#pragma once

#include "aitrain/v2/RuntimeAdapterV2.h"

#include <QJsonObject>

namespace aitrain::v2 {

// Worker 边界使用的、已经完成模型准入的推理调用描述。它不承担模型发现或类型推断；
// 调用方必须先从 ModelPackageId 解析出 RuntimeModelLocationV2。
struct RuntimeInvocationV2 final {
    RuntimeModelLocationV2 model;
    QString runtimeRoute;
    QString imagePath;
    QString outputPath;
    QJsonObject options;
};

QJsonObject encodeRuntimeInvocationV2(const RuntimeInvocationV2& invocation, QString* error = nullptr);
bool decodeRuntimeInvocationV2(const QJsonObject& object, RuntimeInvocationV2* invocation, QString* error = nullptr);

} // namespace aitrain::v2
