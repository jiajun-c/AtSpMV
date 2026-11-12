#pragma once

#include "tileModel.h"
#include <cuda_runtime.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace atspmv {

inline void modelCudaCheck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// One pass over sorted row lengths evaluates all six layouts. Only 144 bytes
// (six TileStats) are copied to the host; the matrix stays on the device.
static __global__ void collectTileStats(const int* lengths, int rows, TileStats* stats, int cap) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 6; ++i) {
        const int m = 1 << i;
        const int k = kTileSlots / m;
        if (group >= (static_cast<long long>(rows) + m - 1) / m) continue;
        const int last = static_cast<int>(min(static_cast<long long>(rows) - 1,
                                              (static_cast<long long>(group) + 1) * m - 1));
        const int tiles = static_cast<int>((static_cast<long long>(lengths[last]) + k - 1) / k);
        atomicAdd(&stats[i].tiles, static_cast<unsigned long long>(tiles));
        atomicAdd(&stats[i].split_warps,
                  static_cast<unsigned long long>((static_cast<long long>(tiles) + cap - 1) / cap));
        atomicMax(&stats[i].max_tiles, tiles);
    }
}

inline ModelParameters modelParametersFromEnvironment(const HardwareModel& hw) {
    const char* cache_env = std::getenv("ATSPMV_MODEL_CACHE");
    const std::string cache_mode = cache_env ? cache_env : "warm";
    if (cache_mode != "warm" && cache_mode != "cold")
        throw std::invalid_argument("ATSPMV_MODEL_CACHE must be warm or cold");
    ModelParameters p = calibratedModelParameters(hw, cache_mode == "cold");
    if (std::getenv("ATSPMV_MODEL_MEMORY_GBPS")) p.calibrated = false;
    struct Setting { const char* name; double* value; bool allow_zero; };
    Setting settings[] = {
        {"ATSPMV_MODEL_BW_EFFICIENCY", &p.bandwidth_efficiency, false},

        {"ATSPMV_MODEL_TILE_CYCLES", &p.tile_issue_cycles, false},
        {"ATSPMV_MODEL_WARP_CYCLES", &p.warp_setup_cycles, true},
        {"ATSPMV_MODEL_SHUFFLE_CYCLES", &p.shuffle_step_cycles, true},
        {"ATSPMV_MODEL_ATOMIC_NS", &p.atomic_ns, true},
        {"ATSPMV_MODEL_ATOMIC_SERIAL_NS", &p.atomic_serial_ns, true},
        {"ATSPMV_MODEL_L2_MULTIPLIER", &p.l2_bandwidth_multiplier, false},
        {"ATSPMV_MODEL_CACHE_FRACTION", &p.cache_capacity_fraction, false},
        {"ATSPMV_MODEL_CACHE_TRANSITION", &p.cache_transition, false},
        {"ATSPMV_MODEL_LAUNCH_NS", &p.launch_ns, true},
        {"ATSPMV_MODEL_X_BYTES", &p.x_bytes_per_slot, true},
        {"ATSPMV_MODEL_SATURATION_WARPS", &p.saturation_warps, false},
    };
    if (const char* text = std::getenv("ATSPMV_MODEL_TILE_NS")) {
        char* end = nullptr;
        const double value = std::strtod(text, &end);
        if (end == text || *end || !std::isfinite(value) || value <= 0)
            throw std::invalid_argument("ATSPMV_MODEL_TILE_NS");
        std::fill(p.tile_dependency_ns, p.tile_dependency_ns + 6, value);
        p.calibrated = false;
    }
    for (auto& setting : settings) {
        if (const char* text = std::getenv(setting.name)) {
            char* end = nullptr;
            errno = 0;
            const double value = std::strtod(text, &end);
            if (end == text || *end || errno || !std::isfinite(value) ||
                (setting.allow_zero ? value < 0 : value <= 0))
                throw std::invalid_argument(setting.name);
            *setting.value = value;
            p.calibrated = false;
        }
    }
    if (p.bandwidth_efficiency > 1) throw std::invalid_argument("ATSPMV_MODEL_BW_EFFICIENCY > 1");
    return p;
}

inline HardwareModel currentHardware() {
    int device;
    cudaDeviceProp prop{};
    modelCudaCheck(cudaGetDevice(&device));
    modelCudaCheck(cudaGetDeviceProperties(&prop, device));
    HardwareModel hw;
    hw.sm_count = prop.multiProcessorCount;
    hw.l2_bytes = prop.l2CacheSize;
    hw.cc_major = prop.major;
    hw.cc_minor = prop.minor;
    hw.clock_cycles_per_ns = prop.clockRate / 1.e6;
    hw.bandwidth_bytes_per_ns = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.e6;
    // Optional calibrated bandwidth for the actual cache/DRAM regime (GB/s).
    if (const char* text = std::getenv("ATSPMV_MODEL_MEMORY_GBPS")) {
        char* end = nullptr;
        errno = 0;
        const double bandwidth = std::strtod(text, &end);
        if (end == text || *end || errno || !std::isfinite(bandwidth) || bandwidth <= 0)
            throw std::invalid_argument("ATSPMV_MODEL_MEMORY_GBPS");
        hw.bandwidth_bytes_per_ns = bandwidth;
    }
    // FP32 lanes/SM for the supported SM80+ families. Future architectures
    // should calibrate the throughput model rather than assume this mapping.
    const int fp32_lanes = (prop.major == 8 && prop.minor == 0) ? 64 : 128;
    hw.fp64_lanes_per_cycle = double(fp32_lanes) /
                              std::max(1, prop.singleToDoublePrecisionPerfRatio);
    return hw;
}

struct TileSelection {
    TileStats stats[6];
    Prediction candidates[12];
    Prediction best;
    bool calibrated = false;
};

// residentWarps(index, balanced, int16) queries each concrete kernel's occupancy.
template<class Occupancy>
inline TileSelection selectTileByTime(const int* sorted_lengths, int rows, int index_bytes,
                                     Occupancy residentWarps, int forced_tile = -1,
                                     int forced_balance = -1, int cols = 0, std::int64_t nnz = -1,
                                     const KernelConfig& config = {}, const TileStats* cached_stats = nullptr) {
    config.validate();
    if (rows <= 0 || forced_tile < -1 || forced_tile > 5 ||
        forced_balance < -1 || forced_balance > 1)
        throw std::invalid_argument("invalid tile selection request");
    TileSelection selection{};
    if (cached_stats) std::copy(cached_stats,cached_stats+6,selection.stats);
    else {
        TileStats* device_stats = nullptr;
        modelCudaCheck(cudaMalloc(&device_stats, sizeof(selection.stats)));
        // Free scratch space even when collection or the host copy fails.
        try {
            modelCudaCheck(cudaMemset(device_stats, 0, sizeof(selection.stats)));
            collectTileStats<<<(static_cast<long long>(rows) + 255) / 256, 256>>>(sorted_lengths, rows, device_stats, config.tiles_per_task);
            modelCudaCheck(cudaGetLastError());
            modelCudaCheck(cudaMemcpy(selection.stats, device_stats, sizeof(selection.stats), cudaMemcpyDeviceToHost));
        } catch (...) {
            cudaFree(device_stats);
            throw;
        }
        modelCudaCheck(cudaFree(device_stats));
    }
    auto hw = currentHardware();
    auto params = modelParametersFromEnvironment(hw);
    params.calibrated = params.calibrated && config.baseline();
    selection.calibrated = params.calibrated;
    const bool trace = std::getenv("ATSPMV_MODEL_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "tile_model calibrated=%d cold_cache=%d sm=%d cc=%d.%d l2_bytes=%d\n",
        params.calibrated, params.cold_cache, hw.sm_count, hw.cc_major, hw.cc_minor, hw.l2_bytes);
    for (int i = 0; i < 6; ++i) {
        for (int balance = 0; balance < 2; ++balance) {
            hw.resident_warps = residentWarps(i, balance != 0, index_bytes == 2);
            auto prediction = predictTile(selection.stats[i], rows, i, balance, index_bytes, hw, params, cols, nnz, config);
            // Existing packing/kernel offsets are signed 32-bit integers.
            if (selection.stats[i].tiles > INT_MAX / kTileSlots || prediction.warps > INT_MAX)
                prediction.total_ns = std::numeric_limits<double>::infinity();
            selection.candidates[2 * i + balance] = prediction;
            if (trace) std::fprintf(stderr,
                "tile_model m=%d k=%d balanced=%d tiles=%llu warps=%llu max_tiles=%d predicted_us=%.6f memory_us=%.6f issue_us=%.6f dependency_us=%.6f atomic_us=%.6f\n",
                1 << i, 32 >> i, balance, selection.stats[i].tiles,
                static_cast<unsigned long long>(prediction.warps), selection.stats[i].max_tiles,
                prediction.total_ns / 1000, prediction.memory_ns / 1000,
                prediction.issue_ns / 1000, prediction.dependency_ns / 1000, prediction.atomic_ns / 1000);
            if ((forced_tile < 0 || forced_tile == i) &&
                (forced_balance < 0 || forced_balance == balance) &&
                prediction.total_ns < selection.best.total_ns)
                selection.best = prediction;
        }
    }
    if (!std::isfinite(selection.best.total_ns))
        throw std::overflow_error("no tile layout fits the current 32-bit kernel offsets");
    return selection;
}

} // namespace atspmv
