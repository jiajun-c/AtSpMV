#pragma once

#include "kernelConfig.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace atspmv {

constexpr int kTileSlots = 32;
constexpr int kTilesPerTask = 8;
constexpr int kWarpsPerBlock = 4;

struct TileStats {
    unsigned long long tiles = 0;
    unsigned long long split_warps = 0;
    int max_tiles = 0;
};

// Rates are effective estimates, not architectural peak-performance promises.
// All time parameters use nanoseconds. Calibrate on the target GPU/workload.
struct ModelParameters {
    double bandwidth_efficiency = 0.7;
    double l2_bandwidth_multiplier = 8.0;
    double tile_dependency_ns[6] = {80, 80, 80, 80, 80, 80};
    double tile_issue_cycles = 24.0;
    double warp_setup_cycles = 32.0;
    double shuffle_step_cycles = 8.0;
    double atomic_ns = 2.0;
    double atomic_serial_ns = 12.0;
    double launch_ns = 1500.0;
    double x_bytes_per_slot = 8.0; // Per nonzero; padding only reads matrix/index data.
    double saturation_warps = 16.0;
    double cache_capacity_fraction = 0.85;
    double cache_transition = 8.0;
    bool calibrated = false;
    bool cold_cache = false;
};

struct HardwareModel {
    int sm_count = 1;
    int resident_warps = 4; // Measured by the occupancy API for this exact kernel.
    double bandwidth_bytes_per_ns = 1.0; // Whole device, theoretical DRAM bandwidth.
    double clock_cycles_per_ns = 1.0;
    double fp64_lanes_per_cycle = 1.0;
    int l2_bytes = 0;
    int cc_major = 0;
    int cc_minor = 0;
};

struct Prediction {
    int tile_index = 0;
    bool balanced = false;
    std::uint64_t warps = 0;
    double memory_ns = 0;
    double issue_ns = 0;
    double dependency_ns = 0;
    double atomic_ns = 0;
    double total_ns = std::numeric_limits<double>::infinity();
};

inline Prediction predictTile(const TileStats& stats, int rows, int tile_index,
                              bool balanced, int index_bytes,
                              const HardwareModel& hw,
                              const ModelParameters& p = {}, int cols = 0, std::int64_t nnz = -1,
                              const KernelConfig& config = {}) {
    config.validate();
    if (cols < 0 || rows < 0 || tile_index < 0 || tile_index > 5 ||
        (index_bytes != 2 && index_bytes != 4) || hw.sm_count <= 0 ||
        hw.resident_warps <= 0 || hw.bandwidth_bytes_per_ns <= 0 ||
        hw.clock_cycles_per_ns <= 0 || hw.fp64_lanes_per_cycle <= 0) {
        throw std::invalid_argument("invalid tile-model input");
    }
    Prediction result;
    result.tile_index = tile_index;
    result.balanced = balanced;
    const int m = 1 << tile_index;
    result.warps = balanced ? stats.split_warps : (std::uint64_t(rows) + m - 1) / m;
    if (stats.tiles == 0) {
        result.total_ns = 0; // Empty matrix is handled without launching SpMV.
        return result;
    }
    if (!result.warps) return result;

    // Blocks, not individual warps, are assigned to SMs.
    const double blocks = std::ceil(double(result.warps) / config.warps_per_block);
    const double active_sms = std::min(double(hw.sm_count), blocks);
    const double warps_per_sm = double(result.warps) / active_sms;
    const double concurrent_warps = std::min(double(hw.resident_warps), warps_per_sm);
    const double tiles_per_sm = double(stats.tiles) / active_sms;
    const double occupancy_factor = std::min(1.0, concurrent_warps / p.saturation_warps);
    const double matrix_bytes = double(stats.tiles) * kTileSlots * (8 + index_bytes);
    const double footprint = matrix_bytes + 8.0 * (double(rows) + cols);
    const double ratio = footprint / std::max(1, hw.l2_bytes);
    // Smooth cache-capacity transition for repeated warm SpMV. This is an
    // effective traffic model, not a claim to know individual cache misses.
    const double exponent = std::clamp(-p.cache_transition *
                                       (ratio - p.cache_capacity_fraction), -50.0, 50.0);
    const double miss_fraction = p.cold_cache ? 1.0 : 1.0 / (1.0 + std::exp(exponent));
    const double nonzeros = nnz < 0 ? double(stats.tiles) * kTileSlots : double(nnz);
    const double bytes = matrix_bytes + p.x_bytes_per_slot * nonzeros +
                         double(result.warps) * (balanced ? 12 : 8) + double(rows) * 12;
    result.memory_ns = bytes / hw.bandwidth_bytes_per_ns *
        (miss_fraction / p.bandwidth_efficiency +
         (1.0 - miss_fraction) / p.l2_bandwidth_multiplier) *
        hw.sm_count / active_sms / occupancy_factor;

    const double reduction_cycles = (5 - tile_index) * p.shuffle_step_cycles;
    const double tile_cycles = std::max(p.tile_issue_cycles,
                                        kTileSlots / hw.fp64_lanes_per_cycle);
    result.issue_ns = (tiles_per_sm * tile_cycles + warps_per_sm *
                      (p.warp_setup_cycles + reduction_cycles)) / hw.clock_cycles_per_ns;
    // Throughput bound plus the indivisible longest task: N alone misses long tails.
    const double longest = balanced ? std::min(stats.max_tiles, config.tiles_per_task) : stats.max_tiles;
    result.dependency_ns = std::max(tiles_per_sm / concurrent_warps, longest) *
                          p.tile_dependency_ns[tile_index];
    if (balanced) {
        const double updates_per_sm = double(result.warps) * m / active_sms;
        const double same_row_updates = std::ceil(double(stats.max_tiles) / config.tiles_per_task);
        result.atomic_ns = updates_per_sm * p.atomic_ns +
                           same_row_updates * m / 32.0 * p.atomic_serial_ns;
    }
    result.total_ns = p.launch_ns + std::max({result.memory_ns, result.issue_ns,
                                             result.dependency_ns}) + result.atomic_ns;
    return result;
}

} // namespace atspmv

#include "tileModelCalibration.h"
