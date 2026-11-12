#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace atspmv {
// Snapshotted at conversion time: changing the environment cannot invalidate
// an already packed matrix. prefetch=0 keeps the calibrated legacy kernel.
struct KernelConfig {
    int tiles_per_task = 8;
    int warps_per_block = 4;
    int prefetch = 0;
    bool cta_reduce = false;
    bool baseline() const {
        return tiles_per_task == 8 && warps_per_block == 4 && prefetch == 0 && !cta_reduce;
    }
    void validate() const {
        if ((tiles_per_task != 4 && tiles_per_task != 8 && tiles_per_task != 16 && tiles_per_task != 32) ||
            (warps_per_block != 2 && warps_per_block != 4 && warps_per_block != 8) ||
            (prefetch != 0 && prefetch != 1 && prefetch != 2 && prefetch != 4 && prefetch != 8))
            throw std::invalid_argument("invalid SpMV kernel configuration");
    }
};
inline KernelConfig kernelConfigFromEnvironment() {
    KernelConfig c;
    auto read = [](const char* name, int fallback) {
        const char* value = std::getenv(name);
        if (!value) return fallback;
        char* end = nullptr;
        long n = std::strtol(value, &end, 10);
        if (end == value || *end || n < 0 || n > 32) throw std::invalid_argument(name);
        return int(n);
    };
    c.tiles_per_task = read("ATSPMV_TUNE_CAP", 8);
    c.warps_per_block = read("ATSPMV_TUNE_WARPS", 4);
    c.prefetch = read("ATSPMV_TUNE_PREFETCH", 0);
    const int reduce = read("ATSPMV_TUNE_CTA_REDUCE", 0);
    if (reduce > 1) throw std::invalid_argument("ATSPMV_TUNE_CTA_REDUCE");
    c.cta_reduce = reduce;
    c.validate();
    return c;
}
} // namespace atspmv
