#pragma once

#include "aitrain/dataset/ValidatedDatasetDriver.h"

namespace aitrain {
class PaddleOcrRecDatasetDriver final : public ValidatedDatasetDriver {
public:
    PaddleOcrRecDatasetDriver();
};
} // namespace aitrain
