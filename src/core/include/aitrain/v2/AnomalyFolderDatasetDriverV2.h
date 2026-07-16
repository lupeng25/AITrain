#pragma once

#include "aitrain/v2/ValidatedDatasetDriverV2.h"

namespace aitrain::v2 {
class AnomalyFolderDatasetDriverV2 final : public ValidatedDatasetDriverV2 {
public:
    AnomalyFolderDatasetDriverV2();
};
} // namespace aitrain::v2
