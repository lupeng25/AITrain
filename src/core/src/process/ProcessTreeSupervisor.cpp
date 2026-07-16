#include "aitrain/process/ProcessTreeSupervisor.h"

#include <QProcess>

#ifdef Q_OS_WIN
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace aitrain {

ProcessTreeSupervisor::ProcessTreeSupervisor() = default;

ProcessTreeSupervisor::~ProcessTreeSupervisor()
{
    reset();
}

void ProcessTreeSupervisor::reset()
{
#ifdef Q_OS_WIN
    if (jobHandle_) {
        CloseHandle(static_cast<HANDLE>(jobHandle_));
        jobHandle_ = nullptr;
    }
#endif
}

bool ProcessTreeSupervisor::create(QString* error)
{
    if (jobHandle_) {
        return true;
    }
#ifdef Q_OS_WIN
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job) {
        if (error) {
            *error = QStringLiteral("无法创建 Windows Job Object：%1").arg(GetLastError());
        }
        return false;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        if (error) {
            *error = QStringLiteral("无法设置 Job Object 关闭时终止进程树：%1").arg(GetLastError());
        }
        CloseHandle(job);
        return false;
    }
    jobHandle_ = job;
    return true;
#else
    if (error) {
        *error = QStringLiteral("ProcessTreeSupervisor 目前仅支持 Windows Job Object。");
    }
    return false;
#endif
}

bool ProcessTreeSupervisor::attach(QProcess* process, QString* error)
{
    if (!process || !jobHandle_ || process->processId() == 0) {
        if (error) {
            *error = QStringLiteral("绑定进程树需要已创建 Job Object 和已启动的根进程。");
        }
        return false;
    }
#ifdef Q_OS_WIN
    HANDLE processHandle = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
        static_cast<DWORD>(process->processId()));
    if (!processHandle) {
        if (error) {
            *error = QStringLiteral("无法打开根进程以绑定 Job Object：%1").arg(GetLastError());
        }
        return false;
    }
    const BOOL assigned = AssignProcessToJobObject(static_cast<HANDLE>(jobHandle_), processHandle);
    const DWORD lastError = assigned ? ERROR_SUCCESS : GetLastError();
    CloseHandle(processHandle);
    if (!assigned) {
        if (error) {
            *error = QStringLiteral("无法将根进程绑定到 Job Object：%1").arg(lastError);
        }
        return false;
    }
    return true;
#else
    Q_UNUSED(error)
    return false;
#endif
}

bool ProcessTreeSupervisor::resume(QProcess* process, QString* error)
{
    if (!process || !jobHandle_ || process->processId() == 0) {
        if (error) {
            *error = QStringLiteral("恢复进程需要已创建 Job Object 和已启动的根进程。");
        }
        return false;
    }
#ifdef Q_OS_WIN
    const DWORD processId = static_cast<DWORD>(process->processId());
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        if (error) {
            *error = QStringLiteral("无法枚举根进程线程以恢复启动：%1").arg(GetLastError());
        }
        return false;
    }

    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    bool resumed = false;
    DWORD failureCode = ERROR_SUCCESS;
    if (Thread32First(snapshot, &entry)) {
        do {
            if (entry.th32OwnerProcessID != processId) {
                continue;
            }
            HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_QUERY_LIMITED_INFORMATION,
                FALSE, entry.th32ThreadID);
            if (!thread) {
                failureCode = GetLastError();
                continue;
            }
            const DWORD previousCount = ResumeThread(thread);
            if (previousCount == static_cast<DWORD>(-1)) {
                failureCode = GetLastError();
            } else {
                resumed = true;
            }
            CloseHandle(thread);
        } while (Thread32Next(snapshot, &entry));
    } else {
        failureCode = GetLastError();
    }
    CloseHandle(snapshot);

    // A process can create helper threads between CREATE_SUSPENDED and this
    // enumeration.  Those threads may already be runnable or may deny the
    // limited query handle; the root thread being resumed is the invariant we
    // need.  Do not fail the entire adapter merely because a best-effort
    // helper-thread resume was unavailable.
    if (!resumed) {
        if (error) {
            *error = QStringLiteral("无法恢复 Job Object 根进程：%1").arg(failureCode);
        }
        return false;
    }
    return true;
#else
    Q_UNUSED(error)
    return false;
#endif
}

bool ProcessTreeSupervisor::terminate(QString* error)
{
    if (!jobHandle_) {
        if (error) {
            *error = QStringLiteral("Job Object 尚未创建。");
        }
        return false;
    }
#ifdef Q_OS_WIN
    if (!TerminateJobObject(static_cast<HANDLE>(jobHandle_), ERROR_CANCELLED)) {
        if (error) {
            *error = QStringLiteral("无法终止 Job Object 进程树：%1").arg(GetLastError());
        }
        return false;
    }
    return true;
#else
    Q_UNUSED(error)
    return false;
#endif
}

bool ProcessTreeSupervisor::statistics(ProcessTreeStats* result, QString* error) const
{
    if (!result || !jobHandle_) {
        if (error) {
            *error = QStringLiteral("读取进程树统计需要已创建 Job Object 和输出对象。");
        }
        return false;
    }
#ifdef Q_OS_WIN
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting{};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION extended{};
    if (!QueryInformationJobObject(static_cast<HANDLE>(jobHandle_), JobObjectBasicAccountingInformation,
            &accounting, sizeof(accounting), nullptr)
        || !QueryInformationJobObject(static_cast<HANDLE>(jobHandle_), JobObjectExtendedLimitInformation,
            &extended, sizeof(extended), nullptr)) {
        if (error) {
            *error = QStringLiteral("无法读取 Job Object 统计：%1").arg(GetLastError());
        }
        return false;
    }
    result->activeProcessCount = static_cast<quint64>(accounting.ActiveProcesses);
    result->totalUserTimeMs = static_cast<quint64>(accounting.TotalUserTime.QuadPart / 10000);
    result->totalKernelTimeMs = static_cast<quint64>(accounting.TotalKernelTime.QuadPart / 10000);
    result->peakProcessMemoryBytes = static_cast<quint64>(extended.PeakProcessMemoryUsed);
    return true;
#else
    Q_UNUSED(error)
    return false;
#endif
}

bool ProcessTreeSupervisor::isCreated() const
{
    return jobHandle_ != nullptr;
}

} // namespace aitrain
