#pragma once

#include <functional>

namespace aitrain {

using CancellationCallback = std::function<bool()>;

inline bool isCancellationRequested(const CancellationCallback& callback)
{
    return callback && callback();
}

} // namespace aitrain
