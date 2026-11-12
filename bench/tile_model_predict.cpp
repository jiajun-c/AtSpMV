#include "tileModel.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Evaluate exported feature tables without a GPU; also checks Python/C++ parity.
static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: tile_model_predict FEATURES.csv");
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("cannot open feature table");
        std::string line;
        std::getline(file, line);
        auto header = split(line);
        std::map<std::string, std::size_t> index;
        for (std::size_t i = 0; i < header.size(); ++i) index[header[i]] = i;
        std::cout << "prediction_us,calibrated\n" << std::setprecision(17);
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            const auto fields = split(line);
            auto get = [&](const char* key) { return std::stod(fields.at(index.at(key))); };
            atspmv::HardwareModel hw;
            hw.sm_count = get("sm_count"); hw.resident_warps = get("resident_warps");
            hw.l2_bytes = get("l2_bytes"); hw.cc_major = get("cc_major"); hw.cc_minor = get("cc_minor");
            hw.clock_cycles_per_ns = get("clock_khz") / 1.e6;
            hw.bandwidth_bytes_per_ns = 2 * get("memory_clock_khz") * (get("memory_bus_bits") / 8) / 1.e6;
            hw.fp64_lanes_per_cycle = ((hw.cc_major == 8 && hw.cc_minor == 0) ? 64 : 128) / get("fp64_ratio");
            atspmv::TileStats stats;
            stats.tiles = get("tiles"); stats.split_warps = get("split_tasks"); stats.max_tiles = get("max_tiles");
            const bool cold = index.count("cache_mode") && fields.at(index.at("cache_mode")) == "cold";
            const auto parameters = atspmv::calibratedModelParameters(hw, cold);
            const auto prediction = atspmv::predictTile(stats, get("rows"), std::lround(std::log2(get("m"))),
                get("balanced") != 0, get("index_bytes"), hw, parameters, get("cols"), get("nnz"));
            std::cout << prediction.total_ns / 1000 << ',' << parameters.calibrated << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
