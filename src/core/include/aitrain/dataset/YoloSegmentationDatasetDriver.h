#pragma once

#include "aitrain/dataset/ValidatedDatasetDriver.h"

namespace aitrain {
class YoloSegmentationDatasetDriver final : public ValidatedDatasetDriver {
public:
    YoloSegmentationDatasetDriver();
};
} // namespace aitrain
