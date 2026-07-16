#pragma once

#include "aitrain/v2/ValidatedDatasetDriverV2.h"

namespace aitrain::v2 {
class PaddleOcrDetDatasetDriverV2 final : public ValidatedDatasetDriverV2 {
public:
    PaddleOcrDetDatasetDriverV2();
};
} // namespace aitrain::v2
