#pragma once

#include <QJsonObject>

namespace aitrain {
namespace protocol {

// 对跨进程协议中可能携带的物理文件系统字段做递归脱敏。
// 相对 Artifact 成员（relativePath / *RelativePath）属于协议契约，必须保留；
// 其余 path、dir、directory、root 结尾的字段会被移除。数组和嵌套对象同样处理。
QJsonObject redactPhysicalPathFields(const QJsonObject& payload);

} // namespace protocol
} // namespace aitrain
