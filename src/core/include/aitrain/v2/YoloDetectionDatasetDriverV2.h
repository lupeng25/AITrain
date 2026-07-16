#pragma once

#include "aitrain/v2/ValidatedDatasetDriverV2.h"

namespace aitrain::v2 {
class YoloDetectionDatasetDriverV2 final : public ValidatedDatasetDriverV2 {
public:
    YoloDetectionDatasetDriverV2();
};
} // namespace aitrain::v2
