#pragma once

#include "aitrain/runtime/RuntimeAdapter.h"

#include <QJsonObject>

#include <functional>

namespace aitrain {

struct RuntimeBenchmarkOptions final {
    int warmupIterations = 3;
    int measuredIterations = 20;
};

struct RuntimeBenchmarkResult final {
    RuntimeOperationResult lastOperation;
    QJsonObject report;
    bool canceled = false;
};

using RuntimeBenchmarkInvocationFactory =
    std::function<QJsonObject(bool warmup, int oneBasedIteration)>;
using RuntimeBenchmarkCancellation = std::function<bool()>;

class RuntimeBenchmarkRunner final {
public:
    RuntimeBenchmarkResult run(const RuntimeAdapter& adapter,
        const RuntimeModelLocation& model,
        const RuntimeBenchmarkInvocationFactory& invocationFactory,
        const RuntimeBenchmarkOptions& options = {},
        const RuntimeBenchmarkCancellation& cancellation = {}) const;
};

} // namespace aitrain
