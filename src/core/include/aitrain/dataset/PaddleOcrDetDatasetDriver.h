#pragma once

#include "aitrain/dataset/ValidatedDatasetDriver.h"

namespace aitrain {
class PaddleOcrDetDatasetDriver final : public ValidatedDatasetDriver {
public:
    PaddleOcrDetDatasetDriver();
};
} // namespace aitrain
