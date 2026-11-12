#include "dsmSpmv.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> static T* upload(const std::vector<T>& v) {
    T* ptr = nullptr;
    check(cudaMalloc(&ptr, std::max(std::size_t(1), v.size()) * sizeof(T)));
    if (!v.empty()) check(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}
static void release(AtBsr<double, int32_t>& a) {
    for (void* ptr : {static_cast<void*>(a.blcVal), static_cast<void*>(a.blcIdx),
                     static_cast<void*>(a.blcIdxInt16), static_cast<void*>(a.blcPtr),
                     static_cast<void*>(a.recorder), static_cast<void*>(a.targetRowByWarp),
                     static_cast<void*>(a.warpBlcPtr)}) check(cudaFree(ptr));
}

static int checked = 0;
static int graph_checked = 0;
static void run(const std::string& name, const std::vector<int>& lengths,
                int cols, const std::vector<int>& cluster_sizes, bool quick) {
    std::vector<int> offsets(1, 0), columns;
    std::vector<double> values, x(cols), output(lengths.size());
    // Nonuniform values and columns across every shard expose owner/offset bugs.
    for (int row = 0; row < int(lengths.size()); ++row) {
        for (int j = 0; j < lengths[row]; ++j) {
            columns.push_back((static_cast<long long>(row) * 7919 + j * 104729 + j % 7) % cols);
            values.push_back(((j * 17 + row * 13) % 31 - 15) * 0.03125);
        }
        offsets.push_back(values.size());
    }
    CSRMatrix<double, int32_t> csr;
    csr.nrows = lengths.size(); csr.ncols = cols; csr.nnz = values.size();
    csr.row_ptr = upload(offsets); csr.col_idx = upload(columns); csr.val = upload(values);
    double* dx = upload(x);
    double* dy = upload(output);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int tile : (quick ? std::vector<int>{0, 5} : std::vector<int>{0, 1, 2, 3, 4, 5})) {
        for (int balance : {0, 1}) {
            AtBsr<double, int32_t> a;
            CSRToAtBSRProduct(csr, a, tile, balance);
            if(csr.nnz)require(a.useColIdxInt16 ? (a.blcIdxInt16 && !a.blcIdx) : (a.blcIdx && !a.blcIdxInt16),
                               "packing must allocate only the active index width");
            for (int cluster : cluster_sizes) {
                for (auto storage : {atspmv::XStorage::Global, atspmv::XStorage::DistributedShared})
                for (bool single_block : {false, true})
                for (int warps : (single_block ? std::vector<int>{0,8,16,32} : std::vector<int>{0})) {
                    // Cap at two clusters: exercises multiple clusters, grid-stride
                    // reuse, and CTAs with no work that must remain alive for DSM.
                    const auto plan = atspmv::makeDsmSpmvPlan(a, cluster, storage, 2, single_block, warps);
                    if (a.nnz) require(plan.gridBlocks() % cluster == 0, "unaligned cluster grid");
                    for (int version = 0; version < 2; ++version) {
                        for (int col = 0; col < cols; ++col)
                            x[col] = ((col * 23LL + version * 31) % 97 - 48) * 0.015625;
                        std::vector<long double> ref(lengths.size(), 0), magnitude(lengths.size(), 0);
                        for (int row = 0; row < csr.nrows; ++row)
                            for (int j = offsets[row]; j < offsets[row+1]; ++j) {
                                const long double v = static_cast<long double>(values[j]) * x[columns[j]];
                                ref[row] += v; magnitude[row] += std::abs(v);
                            }
                        check(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
                        check(cudaMemsetAsync(dy, 0xff, output.size() * sizeof(double), stream));
                        atspmv::launchDsmSpmv(plan, dx, dy, stream);
                        // A second beta=0 call must not accumulate, and the next
                        // iteration changes x so stale shared data cannot pass.
                        atspmv::launchDsmSpmv(plan, dx, dy, stream);
                        check(cudaStreamSynchronize(stream));
                        check(cudaMemcpy(output.data(), dy, output.size() * sizeof(double), cudaMemcpyDeviceToHost));
                        auto verify = [&] {
                            for (int row = 0; row < csr.nrows; ++row) {
                                const long double bound = 8 * std::numeric_limits<double>::epsilon() *
                                    (lengths[row] + 32.0L) * magnitude[row] + 1.e-12L;
                                if (!std::isfinite(output[row]) || std::abs(output[row] - ref[row]) > bound)
                                    throw std::runtime_error(name + ": mismatch tile=" + std::to_string(tile) +
                                        " balance=" + std::to_string(balance) + " cluster=" + std::to_string(cluster) +
                                        " row=" + std::to_string(row));
                            }
                        };
                        verify(); ++checked;
                        if (version == 1 && storage == atspmv::XStorage::DistributedShared) {
                            // Capture includes x staging and balanced output clearing.
                            cudaGraph_t graph;
                            cudaGraphExec_t executable;
                            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                            atspmv::launchDsmSpmv(plan, dx, dy, stream);
                            check(cudaStreamEndCapture(stream, &graph));
                            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                            check(cudaMemsetAsync(dy, 0xff, output.size() * sizeof(double), stream));
                            check(cudaGraphLaunch(executable, stream));
                            check(cudaStreamSynchronize(stream));
                            check(cudaMemcpy(output.data(), dy, output.size() * sizeof(double), cudaMemcpyDeviceToHost));
                            verify(); ++graph_checked;
                            check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
                        }
                    }
                }
            }
            release(a);
        }
    }
    check(cudaStreamDestroy(stream));
    for (void* ptr : {static_cast<void*>(csr.row_ptr), static_cast<void*>(csr.col_idx),
                     static_cast<void*>(csr.val), static_cast<void*>(dx), static_cast<void*>(dy)}) check(cudaFree(ptr));
    std::cout << name << " passed\n";
}

static void rejectOversize() {
    int device, limit;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // Capacity is validated before launch or any vector access. A fake shape
    // suffices, avoiding allocations of an intentionally unsupported x.
    AtBsr<double, int32_t> a;
    a.nrows = a.nnz = a.warpCount = 1;
    a.ncols = (limit / int(sizeof(double))) * 8 + 1;
    a.TILE_M = 1; a.TILE_K = 32;
    bool rejected = false;
    try { atspmv::makeDsmSpmvPlan(a, 8); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "oversized x was not rejected");
    rejected = false;
    try { atspmv::makeDsmSpmvPlan(a, 3); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unsupported cluster size was not rejected");
}

int main(int argc, char** argv) {
    try {
        const bool quick = argc == 2 && std::string(argv[1]) == "--quick";
        if (argc > 1 && !quick) throw std::invalid_argument("Usage: dsm_spmv_test [--quick] (correctness only)");
        int device, supported;
        check(cudaGetDevice(&device));
        check(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device));
        if (!supported) { std::cout << "SKIP: SM90+ cluster support required\n"; return 77; }
        // Isolate the experiment from inherited tuning options and model cache settings.
        setenv("ATSPMV_TUNE_CAP", "8", 1); setenv("ATSPMV_TUNE_WARPS", "4", 1);
        setenv("ATSPMV_TUNE_PREFETCH", "0", 1); setenv("ATSPMV_TUNE_CTA_REDUCE", "0", 1);
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, device));
        std::cout << "Correctness only; no timing. Device: " << prop.name << '\n';
        const auto clusters = quick ? std::vector<int>{2, 8} : std::vector<int>{1, 2, 4, 8};
        std::vector<int> lengths(137);
        for (int row = 0; row < 137; ++row) lengths[row] = row % 9 == 0 ? 0 : (row * 137) % 1025;
        lengths.back() = 8193;
        run("ragged_int16", lengths, 8191, clusters, quick);
        run("ragged_int32", lengths, 40003, quick ? std::vector<int>{2, 8} : std::vector<int>{2, 4, 8}, quick);
        run("partial_cta", {0, 1, 0, 3, 65}, 17, clusters, quick);
        run("empty", {0, 0, 0}, 17, {2}, quick);
        run("zero_rows", {}, 17, {2}, quick);
        // >1 MiB vector: only 8-CTA DSM can hold this complete vector on H100.
        run("large_x", {0, 1, 31, 257, 19}, 200003, {8}, quick);
        if (!quick) {
            setenv("ATSPMV_TUNE_CAP", "4", 1); setenv("ATSPMV_TUNE_WARPS", "2", 1);
            run("cap4_warps2", lengths, 8191, {2}, true);
            setenv("ATSPMV_TUNE_CAP", "32", 1); setenv("ATSPMV_TUNE_WARPS", "8", 1);
            run("cap32_warps8", lengths, 8191, {4}, true);
        }
        rejectOversize();
        std::cout << "PASS: " << checked << " direct checks, " << graph_checked
                  << " graph checks; capacity/configuration rejection passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
