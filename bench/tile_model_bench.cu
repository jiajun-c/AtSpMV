#include "atspmvprod.h"
#include "csrReader.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <array>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> static T* upload(const std::vector<T>& values) {
    T* result;
    check(cudaMalloc(&result, std::max(std::size_t(1), values.size()) * sizeof(T)));
    if (!values.empty()) check(cudaMemcpy(result, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return result;
}
static void release(AtBsr<double, int32_t>& a) {
    for (void* pointer : {static_cast<void*>(a.blcVal), static_cast<void*>(a.blcIdx),
                         static_cast<void*>(a.blcIdxInt16), static_cast<void*>(a.blcPtr),
                         static_cast<void*>(a.recorder), static_cast<void*>(a.targetRowByWarp),
                         static_cast<void*>(a.warpBlcPtr)}) check(cudaFree(pointer));
}

__global__ void evictCache(double* buffer, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) {
        double value;
        asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(value) : "l"(buffer + i));
        buffer[i] = value + 1.0;
    }
}

static void run(const std::string& name, const std::vector<int>& offsets,
                const std::vector<int>& columns, const std::vector<double>& values,
                int ncols, int repeat) {
    const int rows = static_cast<int>(offsets.size()) - 1;
    std::vector<double> x(ncols), output(rows);
    std::vector<long double> reference(rows, 0), absolute_sum(rows, 0);
    for (int i = 0; i < ncols; ++i) x[i] = 0.5 + (i % 17) * 0.03125;
    for (int row = 0; row < rows; ++row)
        for (int j = offsets[row]; j < offsets[row + 1]; ++j) {
            const long double product = static_cast<long double>(values[j]) * x[columns[j]];
            reference[row] += product;
            absolute_sum[row] += std::abs(product);
        }
    CSRMatrix<double, int32_t> csr;
    csr.nrows = rows; csr.ncols = ncols; csr.nnz = values.size();
    csr.row_ptr = upload(offsets); csr.col_idx = upload(columns); csr.val = upload(values);
    double* dx = upload(x);
    double* dy = upload(std::vector<double>(rows, std::numeric_limits<double>::quiet_NaN()));
    const std::string cache_mode = std::getenv("ATSPMV_BENCH_CACHE") ? std::getenv("ATSPMV_BENCH_CACHE") : "warm";
    if (cache_mode != "warm" && cache_mode != "cold") throw std::invalid_argument("invalid ATSPMV_BENCH_CACHE");
    const bool cold = cache_mode == "cold";
    if (const char* requested = std::getenv("ATSPMV_MODEL_CACHE")) {
        if (cache_mode != requested)
            throw std::invalid_argument("ATSPMV_MODEL_CACHE must match ATSPMV_BENCH_CACHE for timing validation");
    }
    if (setenv("ATSPMV_MODEL_CACHE", cache_mode.c_str(), 1) != 0)
        throw std::runtime_error("cannot configure benchmark cache mode");
    AtBsr<double, int32_t> automatic;
    if (CSRToAtBSRProduct(csr, automatic, -1)) throw std::runtime_error("conversion failed");
    const int selected_m = automatic.TILE_M;
    const bool selected_balance = automatic.useBalance;
    release(automatic);

    int device;
    cudaDeviceProp prop{};
    check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&prop, device));
    int clock_khz=0,memory_clock_khz=0,memory_bus_bits=0,fp32_to_fp64=0;
    check(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,device));
    check(cudaDeviceGetAttribute(&memory_clock_khz,cudaDevAttrMemoryClockRate,device));
    check(cudaDeviceGetAttribute(&memory_bus_bits,cudaDevAttrGlobalMemoryBusWidth,device));
    check(cudaDeviceGetAttribute(&fp32_to_fp64,cudaDevAttrSingleToDoublePrecisionPerfRatio,device));
    double* flush = nullptr;
    const std::size_t flush_count = std::max(std::size_t(32 * 1024 * 1024), std::size_t(prop.l2CacheSize) * 2) / sizeof(double);
    if (cold) {
        check(cudaMalloc(&flush, flush_count * sizeof(double)));
        check(cudaMemset(flush, 0, flush_count * sizeof(double)));
    }
    std::vector<int> sorted_lengths(rows);
    for (int r = 0; r < rows; ++r) sorted_lengths[r] = offsets[r+1] - offsets[r];
    std::sort(sorted_lengths.begin(), sorted_lengths.end());
    std::vector<int> order(12);
    std::iota(order.begin(), order.end(), 0);
    unsigned seed = 20261027;
    if (const char* value = std::getenv("ATSPMV_BENCH_SEED")) seed = std::stoul(value);
    std::mt19937 random(seed);
    std::shuffle(order.begin(), order.end(), random);
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    double best_us = std::numeric_limits<double>::infinity(), selected_us = 0;
    for (int candidate : order) {
        const int tile = candidate / 2;
        const int balance = candidate % 2;
        {
            AtBsr<double, int32_t> a;
            if (CSRToAtBSRProduct(csr, a, tile, balance)) throw std::runtime_error("conversion failed");
            // Poison output to ensure beta=0 really initializes it, including empty rows.
            check(cudaMemset(dy, 0xff, rows * sizeof(double)));
            AtSpmvProduct(a, dx, dy, 1.0, 0.0);
            // Reusing the output must not accumulate results across beta=0 calls.
            AtSpmvProduct(a, dx, dy, 1.0, 0.0);
            check(cudaGetLastError());
            check(cudaMemcpy(output.data(), dy, rows * sizeof(double), cudaMemcpyDeviceToHost));
            long double max_backward_error = 0;
            for (int row = 0; row < rows; ++row) {
                const long double difference = std::abs(output[row] - reference[row]);
                max_backward_error = std::max(max_backward_error,
                    difference / std::max(absolute_sum[row], 1.e-300L));
                // Forward-error bound scaled by sum |a*x|, robust to cancellation.
                const long double bound = 8 * std::numeric_limits<double>::epsilon() *
                    (offsets[row+1] - offsets[row] + 32.0L) * absolute_sum[row] + 1.e-12L;
                if (!std::isfinite(output[row]) || difference > bound)
                    throw std::runtime_error(name + ": incorrect result at row " + std::to_string(row) +
                                             ", tile=" + std::to_string(tile) + ", balance=" + std::to_string(balance) +
                                             ", gpu=" + std::to_string(output[row]) + ", cpu=" + std::to_string(reference[row]));
            }
            for (int j = 0; j < 5; ++j) AtSpmvProduct(a, dx, dy, 1.0, 0.0);
            check(cudaDeviceSynchronize());
            // Capture a bounded number of kernel nodes, then replay them without
            // CPU launch gaps. beta=1 times only SpMV; correctness above uses beta=0.
            check(cudaEventRecord(start, stream));
            AtSpmvProduct(a, dx, dy, 1.0, 1.0, stream);
            check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
            float pilot_ms;
            check(cudaEventElapsedTime(&pilot_ms, start, stop));
            const int nodes = cold ? 1 : std::max(1, std::min(repeat, int(1.0 / std::max(0.001f, pilot_ms))));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int j = 0; j < nodes; ++j) AtSpmvProduct(a, dx, dy, 1.0, 1.0, stream);
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
            std::vector<double> timings;
            for (int batch = 0; batch < 9; ++batch) {
                check(cudaMemsetAsync(dy, 0, rows * sizeof(double), stream));
                if (cold) evictCache<<<prop.multiProcessorCount * 8, 256, 0, stream>>>(flush, flush_count);
                check(cudaEventRecord(start, stream));
                check(cudaGraphLaunch(executable, stream));
                check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
                float ms;
                check(cudaEventElapsedTime(&ms, start, stop));
                timings.push_back(ms * 1000.0 / nodes);
            }
            check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
            std::sort(timings.begin(), timings.end());
            const double measured = timings[timings.size() / 2];
            const bool selected = a.TILE_M == selected_m && a.useBalance == selected_balance;
            if (selected) selected_us = measured;
            best_us = std::min(best_us, measured);
            const int m = 1 << tile, k = 32 >> tile;
            std::vector<double> group_tiles;
            double sum_sq = 0, block_max_sum = 0;
            double block_max = 0;
            std::uint64_t active_groups = 0, split_tasks = 0;
            for (int first = 0; first < rows; first += m) {
                const double count = (sorted_lengths[std::min(first+m, rows)-1] + k - 1) / k;
                group_tiles.push_back(count);
                sum_sq += count * count;
                block_max = std::max(block_max, count);
                if (group_tiles.size() % a.kernelConfig.warps_per_block == 0 || first + m >= rows) {
                    block_max_sum += block_max; block_max = 0;
                }
                active_groups += count > 0;
                split_tasks += static_cast<std::uint64_t>(std::ceil(count / a.kernelConfig.tiles_per_task));
            }
            auto quantile = [&](double q) {
                return group_tiles.empty() ? 0.0 : group_tiles[std::min(group_tiles.size()-1,
                                               std::size_t(q * group_tiles.size()))];
            };
            std::cout << name << ',' << rows << ',' << ncols << ',' << values.size() << ','
                      << m << ',' << k << ',' << balance << ','
                      << a.blockCount << ',' << a.warpCount << ',' << a.predictedTimeUs << ','
                      << measured << ',' << selected << ",1," << timings[1] << ',' << timings[7]
                      << ',' << nodes << ',' << seed << ',' << prop.multiProcessorCount
                      << ',' << prop.l2CacheSize << ',' << clock_khz << ',' << memory_clock_khz
                      << ',' << memory_bus_bits << ',' << prop.major << ',' << prop.minor
                      << ',' << fp32_to_fp64 << ',' << a.modelResidentWarps
                      << ',' << (a.useColIdxInt16 ? 2 : 4) << ',' << quantile(1)
                      << ',' << quantile(.5) << ',' << quantile(.9) << ',' << quantile(.99)
                      << ',' << quantile(.999) << ',' << sum_sq << ',' << block_max_sum
                      << ',' << active_groups << ',' << split_tasks << ',' << double(max_backward_error) << ',' << cache_mode << ',' << a.kernelConfig.tiles_per_task << ',' << a.kernelConfig.warps_per_block
                      << ',' << a.kernelConfig.prefetch << ',' << a.kernelConfig.cta_reduce
                      << ',' << a.modelCalibrated << '\n';
            release(a);
        }
    }
    std::cerr << name << ": selected=" << selected_m << 'x' << 32 / selected_m
              << " balanced=" << selected_balance << " selected_us=" << selected_us
              << " oracle_us=" << best_us << " ratio=" << selected_us / best_us << '\n';
    check(cudaFree(flush));
    check(cudaStreamDestroy(stream));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaFree(csr.row_ptr)); check(cudaFree(csr.col_idx)); check(cudaFree(csr.val));
    check(cudaFree(dx)); check(cudaFree(dy));
}

static void synthetic(const std::string& name, const std::vector<int>& lengths, int cols, int repeat) {
    std::vector<int> offsets(1, 0), columns;
    std::vector<double> values;
    for (int row = 0; row < static_cast<int>(lengths.size()); ++row) {
        for (int j = 0; j < lengths[row]; ++j) {
            columns.push_back((static_cast<long long>(row) * 19 + j * 7) % cols);
            values.push_back((j % 7 - 3) * 0.125);
        }
        offsets.push_back(values.size());
    }
    run(name, offsets, columns, values, cols, repeat);
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: " << argv[0] << " MATRIX.mtx|--synthetic [repeat=50]\n";
            return 2;
        }
        const int repeat = argc == 3 ? std::stoi(argv[2]) : 50;
        if (repeat <= 0) throw std::invalid_argument("repeat must be positive");
        std::cout << std::setprecision(8)
                  << "matrix,rows,cols,nnz,m,k,balanced,tiles,warps,predicted_us,measured_us,selected,correct,p10_us,p90_us,graph_nodes,seed,sm_count,l2_bytes,clock_khz,memory_clock_khz,memory_bus_bits,cc_major,cc_minor,fp64_ratio,resident_warps,index_bytes,max_tiles,p50_tiles,p90_tiles,p99_tiles,p999_tiles,sum_tiles_sq,block_max_sum,active_groups,split_tasks,max_backward_error,cache_mode,cap,warps_per_block,prefetch,cta_reduce,model_calibrated\n";
        if (std::string(argv[1]) == "--synthetic") {
            synthetic("empty", {0, 0, 0}, 32, repeat);
            synthetic("partial_int16", {0, 1, 17, 0, 33, 3, 65}, 100, repeat);
            synthetic("partial_int32", {0, 1, 17, 0, 33, 3, 65}, 40000, repeat);
            std::vector<int> boundaries(137);
            for (int i = 0; i < 137; ++i) boundaries[i] = (i % 11 == 0) ? 0 : 1 + (i * 137) % 4097;
            synthetic("merge_boundaries16", boundaries, 8192, repeat);
            synthetic("merge_boundaries32", boundaries, 40000, repeat);
            synthetic("uniform_short", std::vector<int>(32768, 4), 16384, repeat);
            synthetic("uniform_long", std::vector<int>(8192, 256), 40000, repeat);
            auto tail = std::vector<int>(8192, 4); tail.back() = 131072;
            synthetic("long_tail", tail, 40000, repeat);
        } else {
            auto a = read_mtx(argv[1]);
            run(std::filesystem::path(argv[1]).stem().string(), std::vector<int>(a.row_ptr, a.row_ptr + a.nrows + 1),
                std::vector<int>(a.col_idx, a.col_idx + a.nnz),
                std::vector<double>(a.val, a.val + a.nnz), a.ncols, repeat);
            delete[] a.row_ptr; delete[] a.col_idx; delete[] a.val;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
