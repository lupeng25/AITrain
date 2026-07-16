#pragma once

#include <QImage>

namespace aitrain {

// 语义分割类别由原始灰度值或 palette index 表示；不得从显示颜色反推。
bool isSupportedSemanticMask(const QImage& mask);
int semanticMaskClassId(const QImage& mask, int x, int y);

} // namespace aitrain
