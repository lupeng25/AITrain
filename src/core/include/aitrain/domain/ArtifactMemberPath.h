#pragma once

#include <QString>

namespace aitrain {

// Artifact 包内成员使用与宿主平台无关的规范相对路径。该函数是
// ArtifactStore、Storage、Manifest 与查询边界的唯一校验入口。
bool normalizeArtifactMemberPath(const QString& value,
    QString* normalized,
    QString* error = nullptr);

} // namespace aitrain
