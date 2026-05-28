#pragma once

#include <dlfcn.h>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <utility>

// ---- Memory ----------------------------------------------------------------

struct MemSnapshot {
    long rss_mb      = 0;  // current resident set size
    long peak_rss_mb = 0;  // peak RSS since process start (ru_maxrss)
};

inline MemSnapshot read_mem() {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);

    long rss_mb = 0;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            rss_mb = std::stol(line.substr(6)) / 1024; // kB → MB
            break;
        }
    }
    return { rss_mb, r.ru_maxrss / 1024 }; // ru_maxrss is kB on Linux
}

// ---- CPU -------------------------------------------------------------------

struct CpuTime {
    double user_s = 0.0;
    double sys_s  = 0.0;
    double total() const { return user_s + sys_s; }
};

inline CpuTime read_cpu() {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return {
        r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6,
        r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6
    };
}

// ---- CPU core count --------------------------------------------------------

// Returns physical core count by counting unique (physical_id, core_id) pairs
// in /proc/cpuinfo, which excludes hyperthreading duplicates.
// Falls back to logical CPU count when topology fields are absent (VMs, ARM).
inline int cpu_core_count() {
    auto trim = [](const std::string& s) {
        const auto a = s.find_first_not_of(" \t");
        const auto b = s.find_last_not_of(" \t");
        return a == std::string::npos ? std::string{} : s.substr(a, b - a + 1);
    };

    std::set<std::pair<int, int>> seen;
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    int phys = -1, core = -1;

    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            if (phys >= 0 && core >= 0) seen.insert({phys, core});
            phys = core = -1;
            continue;
        }
        const std::string key = trim(line.substr(0, colon));
        const std::string val = trim(line.substr(colon + 1));
        if      (key == "physical id") phys = std::stoi(val);
        else if (key == "core id")     core = std::stoi(val);
    }
    if (phys >= 0 && core >= 0) seen.insert({phys, core});

    if (!seen.empty()) return static_cast<int>(seen.size());

    const long n = sysconf(_SC_NPROCESSORS_ONLN);  // fallback: logical CPUs
    return n > 0 ? static_cast<int>(n) : 1;
}

// ---- GPU (NVML via dlopen) -------------------------------------------------

struct GpuSample {
    unsigned int util_pct        = 0;  // GPU core utilisation %
    unsigned int proc_mem_mb     = 0;  // VRAM used by this process
};

class GpuMonitor {
    struct NvmlUtilization { unsigned int gpu, memory; };

    // Must match nvmlProcessInfo_t layout in the NVML binary (24 bytes on modern drivers):
    //   unsigned int pid              (offset 0)
    //   [4 bytes implicit padding]
    //   unsigned long long usedGpuMem (offset 8)
    //   unsigned int gpuInstanceId    (offset 16)
    //   unsigned int computeInstanceId(offset 20)
    struct NvmlProcessInfo {
        unsigned int       pid;
        unsigned long long usedGpuMemory;  // compiler inserts 4-byte pad before this
        unsigned int       gpuInstanceId;
        unsigned int       computeInstanceId;
    };
    static_assert(sizeof(NvmlProcessInfo) == 24, "NvmlProcessInfo layout mismatch");

    using FnVoid      = int(*)();
    using FnGetDevice = int(*)(unsigned int, void**);
    using FnGetUtil   = int(*)(void*, NvmlUtilization*);
    using FnGetProcs  = int(*)(void*, unsigned int*, NvmlProcessInfo*);

public:
    GpuMonitor() {
        handle_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!handle_) handle_ = dlopen("libnvidia-ml.so", RTLD_LAZY);
        if (!handle_) return;

        auto sym = [&](const char* n) { return dlsym(handle_, n); };

        void* init_sym = sym("nvmlInit_v2");
        if (!init_sym)  init_sym = sym("nvmlInit");
        void* dev_sym  = sym("nvmlDeviceGetHandleByIndex_v2");
        if (!dev_sym)   dev_sym  = sym("nvmlDeviceGetHandleByIndex");

        auto init   = reinterpret_cast<FnVoid>(init_sym);
        get_device_ = reinterpret_cast<FnGetDevice>(dev_sym);
        get_util_   = reinterpret_cast<FnGetUtil>(sym("nvmlDeviceGetUtilizationRates"));
        get_procs_  = reinterpret_cast<FnGetProcs>(sym("nvmlDeviceGetComputeRunningProcesses"));
        shutdown_   = reinterpret_cast<FnVoid>(sym("nvmlShutdown"));

        if (!init || !get_device_ || !get_util_ || !get_procs_) {
            dlclose(handle_); handle_ = nullptr; return;
        }
        if (init() != 0 || get_device_(0, &device_) != 0) {
            if (shutdown_) shutdown_();
            dlclose(handle_); handle_ = nullptr; return;
        }
        available_ = true;
    }

    ~GpuMonitor() {
        if (available_ && shutdown_) shutdown_();
        if (handle_) dlclose(handle_);
    }

    bool available() const { return available_; }

    std::optional<GpuSample> sample() const {
        if (!available_) return std::nullopt;

        NvmlUtilization util{};
        if (get_util_(device_, &util) != 0) return std::nullopt;

        // Find VRAM used by this process specifically
        unsigned int proc_mem_mb = 0;
        const auto my_pid = static_cast<unsigned int>(getpid());
        NvmlProcessInfo procs[64];
        unsigned int count = 64;
        if (get_procs_(device_, &count, procs) == 0) {
            for (unsigned int i = 0; i < count; ++i) {
                if (procs[i].pid == my_pid) {
                    proc_mem_mb = static_cast<unsigned int>(procs[i].usedGpuMemory >> 20);
                    break;
                }
            }
        }

        return GpuSample{ util.gpu, proc_mem_mb };
    }

private:
    void*       handle_     = nullptr;
    void*       device_     = nullptr;
    bool        available_  = false;
    FnGetDevice get_device_ = nullptr;
    FnGetUtil   get_util_   = nullptr;
    FnGetProcs  get_procs_  = nullptr;
    FnVoid      shutdown_   = nullptr;
};
