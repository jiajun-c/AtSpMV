#include "tileModel.h"
#include "hybridModelFeatures.h"
#include "unifiedSpmvModel.h"
#include <cstdlib>
#include <iostream>

static void require(bool value, const char* message) {
    if (!value) { std::cerr << message << '\n'; std::exit(1); }
}
int main() {
    using namespace atspmv;
    {
        const auto f=collectHybridModelFeatures({0,0,1,34,234,235},1000,100);
        require(f.rows==5&&f.nnz==235&&f.long_rows==1&&f.long_nnz==200&&f.short_nnz==35,"hybrid partition statistics");
        require(f.short_tiles[0]==4&&f.short_max_tiles[0]==2,"hybrid short M=1 tiles");
        require(f.short_tiles[1]==4&&f.short_max_tiles[1]==3,"hybrid short M=2 tiles");
        require(f.longest==200&&f.longest_short==33&&f.long_length_squared==40000,"hybrid tail statistics");
        HybridModelFeatures gate;gate.rows=100;gate.long_rows=9;
        require(unifiedHybridEligible(gate),"sparse long rows should be eligible");
        gate.long_rows=10;require(!unifiedHybridEligible(gate),"dense long rows should be gated");
        gate.long_rows=0;require(!unifiedHybridEligible(gate),"no-long hybrid should be gated");
    }
    HardwareModel hw{80, 32, 1500, 1.4, 32};
    TileStats regular{32000, 4000, 8};
    const auto regular_time = predictTile(regular, 4000, 0, false, 4, hw);
    TileStats long_tail{32000, 4000, 32000};
    const auto tail = predictTile(long_tail, 4000, 0, false, 4, hw);
    const auto split = predictTile(long_tail, 4000, 0, true, 4, hw);
    require(tail.total_ns > regular_time.total_ns, "equal tile counts must distinguish long tails");
    require(split.total_ns < tail.total_ns, "splitting should mitigate a dominant long row");
    const auto int16 = predictTile(regular, 4000, 0, false, 2, hw);
    require(int16.memory_ns < regular_time.memory_ns, "16-bit indices reduce traffic");
    auto slower = hw; slower.bandwidth_bytes_per_ns /= 2;
    require(predictTile(regular, 4000, 0, false, 4, slower).memory_ns > regular_time.memory_ns,
            "bandwidth must have physical units");
    auto lower_occupancy = hw; lower_occupancy.resident_warps = 4;
    require(predictTile(regular, 4000, 0, false, 4, lower_occupancy).total_ns >= regular_time.total_ns,
            "lower occupancy should not improve the estimate");
    ModelParameters p; p.atomic_ns = 100;
    require(predictTile(regular, 4000, 0, true, 4, hw, p).total_ns >
            predictTile(regular, 4000, 0, true, 4, hw).total_ns, "atomic cost must be explicit");
    require(predictTile({}, 0, 0, false, 4, hw).total_ns == 0, "empty input");
    auto cached = hw; cached.l2_bytes = 1000000000;
    const auto warm = predictTile(regular, 4000, 0, false, 4, cached, {}, 4000, 1000000);
    const auto cold = predictTile(regular, 4000, 0, false, 4, hw, {}, 4000, 1000000);
    require(warm.memory_ns < cold.memory_ns, "cache residency should lower traffic service time");
    auto fewer_sms = hw; fewer_sms.sm_count = 40; fewer_sms.bandwidth_bytes_per_ns /= 2;
    require(predictTile(regular, 4000, 0, false, 4, fewer_sms).total_ns >= regular_time.total_ns,
            "half the SMs and bandwidth should not improve execution time");
    require(!calibratedModelParameters(hw).calibrated, "unknown hardware must not claim calibration");
    HardwareModel ada{110, 48, 960, 2.46, 2.0, 100663296, 8, 9};
    require(calibratedModelParameters(ada).calibrated, "validated warm hardware profile");
    require(calibratedModelParameters(ada, true).calibrated, "validated cold hardware profile");
    require(calibratedModelParameters(ada, true).cold_cache, "cold mode must be explicit");
    TileStats huge{10000000000ULL, 1250000000ULL, 100};
    require(std::isfinite(predictTile(huge, 1000000000, 5, true, 4, hw).total_ns), "64-bit counts");
    bool rejected = false;
    try { predictTile(regular, 4000, 6, false, 4, hw); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "reject invalid tile IDs");
    KernelConfig cap16; cap16.tiles_per_task = 16;
    TileStats one_row{128, 8, 128};
    const auto longer_task = predictTile(one_row, 1, 0, true, 4, hw, {}, 0, -1, cap16);
    const auto shorter_task = predictTile(TileStats{128, 16, 128}, 1, 0, true, 4, hw);
    require(longer_task.dependency_ns > shorter_task.dependency_ns,
            "configured cap must affect the longest-task bound");
    KernelConfig wide; wide.warps_per_block = 8;
    TileStats small_grid{32, 4, 4};
    require(predictTile(small_grid, 8, 0, false, 4, hw, {}, 0, -1, wide).issue_ns >
            predictTile(small_grid, 8, 0, false, 4, hw).issue_ns,
            "larger CTA can reduce active SMs for a small grid");
    rejected = false;
    try { KernelConfig bad; bad.prefetch = 3; bad.validate(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "reject unsupported prefetch depth");
    std::cout << "tile model tests passed\n";
}
