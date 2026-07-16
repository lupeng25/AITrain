#pragma once

#include "aitrain/v2/ValidatedDatasetDriverV2.h"

namespace aitrain::v2 {
class PaddleOcrRecDatasetDriverV2 final : public ValidatedDatasetDriverV2 {
public:
    PaddleOcrRecDatasetDriverV2();
};
} // namespace aitrain::v2
