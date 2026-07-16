#pragma once

#include "aitrain/v2/ValidatedDatasetDriverV2.h"

namespace aitrain::v2 {
class YoloSegmentationDatasetDriverV2 final : public ValidatedDatasetDriverV2 {
public:
    YoloSegmentationDatasetDriverV2();
};
} // namespace aitrain::v2
