#include "aitrain/core/SemanticMask.h"

namespace aitrain {

bool isSupportedSemanticMask(const QImage& mask)
{
    return !mask.isNull()
        && (mask.format() == QImage::Format_Grayscale8
            || mask.format() == QImage::Format_Indexed8);
}

int semanticMaskClassId(const QImage& mask, int x, int y)
{
    if (!isSupportedSemanticMask(mask) || x < 0 || y < 0 || x >= mask.width() || y >= mask.height()) {
        return -1;
    }
    // Grayscale8 stores the gray value and Indexed8 stores the palette index.
    return static_cast<int>(mask.constScanLine(y)[x]);
}

} // namespace aitrain
