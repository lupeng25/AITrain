#pragma once

#include <QtGlobal>

class QProcess;

namespace aitrain::v2 {

struct ProcessTreeStats final {
    quint64 activeProcessCount = 0;
    quint64 totalUserTimeMs = 0;
    quint64 totalKernelTimeMs = 0;
    quint64 peakProcessMemoryBytes = 0;
};

class ProcessTreeSupervisor final {
public:
    ProcessTreeSupervisor();
    ~ProcessTreeSupervisor();

    ProcessTreeSupervisor(const ProcessTreeSupervisor&) = delete;
    ProcessTreeSupervisor& operator=(const ProcessTreeSupervisor&) = delete;

    bool create(QString* error = nullptr);
    void reset();
    bool attach(QProcess* process, QString* error = nullptr);
    bool terminate(QString* error = nullptr);
    bool statistics(ProcessTreeStats* result, QString* error = nullptr) const;
    bool isCreated() const;

private:
    void* jobHandle_ = nullptr;
};

} // namespace aitrain::v2
