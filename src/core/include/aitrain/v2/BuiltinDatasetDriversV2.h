#pragma once

#include "aitrain/v2/DatasetDriverV2.h"

namespace aitrain::v2 {

bool registerBuiltinDatasetDriversV2(DatasetDriverRegistryV2* registry, QString* error = nullptr);

} // namespace aitrain::v2
