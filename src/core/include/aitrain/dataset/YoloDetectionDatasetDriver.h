#pragma once

#include "aitrain/dataset/ValidatedDatasetDriver.h"

namespace aitrain {
class YoloDetectionDatasetDriver final : public ValidatedDatasetDriver {
public:
    YoloDetectionDatasetDriver();
};
} // namespace aitrain
