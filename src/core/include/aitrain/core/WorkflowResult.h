#pragma once

#include <QJsonObject>
#include <QString>

namespace aitrain {

// 外部工具探测等 Core 边界返回的结构化结果； 产品工作流使用各自的 Result。
struct WorkflowResult {
    bool ok = false;
    QString error;
    QString reportPath;
    QJsonObject payload;
};

} // namespace aitrain
