#pragma once

#include "aitrain/dataset/DatasetDriver.h"

namespace aitrain {

bool registerBuiltinDatasetDrivers(DatasetDriverRegistry* registry, QString* error = nullptr);

} // namespace aitrain
