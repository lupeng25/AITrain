#include "aitrain/runtime/RuntimeBenchmarkRunner.h"

#include <QElapsedTimer>
#include <QJsonArray>

#include <algorithm>
#include <numeric>

namespace aitrain {
namespace {

double percentile(const QVector<double>& sorted, double fraction)
{
    if (sorted.isEmpty()) return 0.0;
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const int lower = static_cast<int>(position);
    const int upper = qMin(lower + 1, sorted.size() - 1);
    const double weight = position - lower;
    return sorted.at(lower) * (1.0 - weight) + sorted.at(upper) * weight;
}

} // namespace

RuntimeBenchmarkResult RuntimeBenchmarkRunner::run(const RuntimeAdapter& adapter,
    const RuntimeModelLocation& model,
    const RuntimeBenchmarkInvocationFactory& invocationFactory,
    const RuntimeBenchmarkOptions& options,
    const RuntimeBenchmarkCancellation& cancellation) const
{
    RuntimeBenchmarkResult result;
    if (!invocationFactory || options.warmupIterations < 0
        || options.measuredIterations < 1) {
        result.lastOperation.status = RuntimeStatus::ArtifactIncompatible;
        result.lastOperation.message = QStringLiteral("Runtime Benchmark 参数无效。");
        return result;
    }
    for (int index = 1; index <= options.warmupIterations; ++index) {
        if (cancellation && cancellation()) {
            result.canceled = true;
            result.lastOperation.status = RuntimeStatus::RuntimeNotImplemented;
            result.lastOperation.message = QStringLiteral("Runtime Benchmark 预热阶段已取消。");
            return result;
        }
        result.lastOperation = adapter.infer(model, invocationFactory(true, index));
        if (result.lastOperation.status != RuntimeStatus::Available) return result;
    }

    QVector<double> samples;
    samples.reserve(options.measuredIterations);
    QJsonArray sampleJson;
    for (int index = 1; index <= options.measuredIterations; ++index) {
        if (cancellation && cancellation()) {
            result.canceled = true;
            result.lastOperation.status = RuntimeStatus::RuntimeNotImplemented;
            result.lastOperation.message = QStringLiteral("Runtime Benchmark 采样阶段已取消。");
            return result;
        }
        QElapsedTimer timer;
        timer.start();
        result.lastOperation = adapter.infer(model, invocationFactory(false, index));
        const double elapsedMs = static_cast<double>(timer.nsecsElapsed()) / 1000000.0;
        if (result.lastOperation.status != RuntimeStatus::Available) return result;
        samples.append(elapsedMs);
        sampleJson.append(elapsedMs);
    }
    std::sort(samples.begin(), samples.end());
    const double total = std::accumulate(samples.cbegin(), samples.cend(), 0.0);
    const double mean = total / samples.size();
    result.report = {
        {QStringLiteral("benchmarkKind"), QStringLiteral("smoke_timing")},
        {QStringLiteral("timingDefinition"),
            QStringLiteral("Runtime Adapter 推理及结果解析；不含模型加载、报告落盘和 overlay 渲染。")},
        {QStringLiteral("setupMs"), 0.0},
        {QStringLiteral("warmupIterations"), options.warmupIterations},
        {QStringLiteral("measuredIterations"), options.measuredIterations},
        {QStringLiteral("samplesMs"), sampleJson},
        {QStringLiteral("minMs"), samples.first()},
        {QStringLiteral("meanMs"), mean},
        {QStringLiteral("p50Ms"), percentile(samples, 0.50)},
        {QStringLiteral("p95Ms"), percentile(samples, 0.95)},
        {QStringLiteral("p99Ms"), percentile(samples, 0.99)},
        {QStringLiteral("maxMs"), samples.last()},
        {QStringLiteral("throughputPerSecond"), mean > 0.0 ? 1000.0 / mean : 0.0}
    };
    return result;
}

} // namespace aitrain
