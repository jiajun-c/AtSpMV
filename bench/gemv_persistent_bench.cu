#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void checkBlas(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure " + std::to_string(e));
}

// Row-major A, identical dot product and block reduction in all three kernels.
// Layout 0: CTA b computes only row b (ordinary launch with M CTAs).
// Layout 1: CTA b computes b, b+B, b+2B, ... (cyclic persistent assignment).
// Layout 2: CTA b computes [floor(M*b/B), floor(M*(b+1)/B)) (contiguous).
// Both persistent layouts assign floor(M/B) or ceil(M/B) equal-length rows.
template<int Layout>
__global__ void gemv(const double* a, const double* x, double* y, int m, int n) {
    __shared__ double partial[32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int first = Layout == 2 ? int((long long)m * blockIdx.x / gridDim.x) : blockIdx.x;
    const int end = Layout == 0 ? first + 1 : (Layout == 2 ? int((long long)m * (blockIdx.x + 1) / gridDim.x) : m);
    const int step = Layout == 1 ? gridDim.x : 1;
    for (int row = first; row < end; row += step) {
        double sum = 0;
        for (int col = threadIdx.x; col < n; col += blockDim.x)
            sum += a[(std::size_t)row * n + col] * x[col];
        for (int offset = 16; offset; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < blockDim.x / 32 ? partial[lane] : 0;
            for (int offset = 16; offset; offset /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane == 0) y[row] = sum;
        }
        // Only persistent kernels reuse partial[] for another row.
        if constexpr (Layout != 0) __syncthreads();
    }
}

struct Candidate {
    std::string name;
    int threads = 0, blocks = 0, resident_limit = 0, min_rows = 0, max_rows = 0;
    std::function<void()> launch;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    double error = 0;
    std::vector<double> times;
};

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 5)
            throw std::invalid_argument("Usage: gemv_persistent_bench M N [seed=202610290] [--check-only]");
        const int m = std::stoi(argv[1]), n = std::stoi(argv[2]);
        const unsigned seed = argc >= 4 ? std::stoul(argv[3]) : 202610290;
        const bool check_only = argc == 5 && std::string(argv[4]) == "--check-only";
        if (argc == 5 && !check_only) throw std::invalid_argument("unknown option");
        if (m <= 0 || n <= 0 || (long long)m * n > 134217728)
            throw std::invalid_argument("positive dimensions and A <= 1 GiB required");
        int device;
        check(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 9) { std::cout << "SKIP: this experiment is built for SM90+\n"; return 77; }
        std::vector<double> a((std::size_t)m * n), x(n), output(m);
        std::vector<long double> reference(m), magnitude(m);
        for (int col = 0; col < n; ++col) x[col] = ((col * 23LL) % 97 - 48) / 61.;
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                const auto index = (std::size_t)row * n + col;
                a[index] = (int((index * 17 + row * 7) % 127) - 63) / 79.;
                const long double product = (long double)a[index] * x[col];
                reference[row] += product;
                magnitude[row] += std::abs(product);
            }
        }
        double *da, *dx, *dy;
        check(cudaMalloc(&da, a.size() * sizeof(double)));
        check(cudaMalloc(&dx, x.size() * sizeof(double)));
        check(cudaMalloc(&dy, output.size() * sizeof(double)));
        check(cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::vector<Candidate> candidates;
        for (int threads : {128, 256, 512, 1024}) {
            int active = 0;
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, gemv<0>, threads, 0));
            Candidate normal;
            normal.name = "one_row_per_cta"; normal.threads = threads; normal.blocks = m;
            normal.resident_limit = active * prop.multiProcessorCount;
            normal.min_rows = normal.max_rows = 1;
            normal.launch = [=] { gemv<0><<<m, threads, 0, stream>>>(da, dx, dy, m, n); };
            candidates.push_back(normal);
            for (int layout : {1, 2}) {
                if (layout == 1) check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, gemv<1>, threads, 0));
                else check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, gemv<2>, threads, 0));
                std::set<int> used;
                for (int factor : {1, 2, 4, 8, active}) {
                    const int blocks = std::min(m, std::min(active, factor) * prop.multiProcessorCount);
                    if (!used.insert(blocks).second) continue;
                    Candidate c;
                    c.name = layout == 1 ? "persistent_cyclic" : "persistent_contiguous";
                    c.threads = threads; c.blocks = blocks;
                    c.resident_limit = active * prop.multiProcessorCount;
                    c.min_rows = m / blocks; c.max_rows = (m + blocks - 1) / blocks;
                    c.launch = [=] {
                        if (layout == 1) gemv<1><<<blocks, threads, 0, stream>>>(da, dx, dy, m, n);
                        else gemv<2><<<blocks, threads, 0, stream>>>(da, dx, dy, m, n);
                    };
                    candidates.push_back(c);
                }
            }
        }
        cublasHandle_t handle;
        checkBlas(cublasCreate(&handle));
        checkBlas(cublasSetStream(handle, stream));
        const double alpha = 1, beta = 0;
        Candidate blas;
        blas.name = "cublas";
        // Row-major MxN is column-major NxM; transpose that view.
        blas.launch = [&] { checkBlas(cublasDgemv(handle, CUBLAS_OP_T, n, m, &alpha, da, n, dx, 1, &beta, dy, 1)); };
        candidates.push_back(blas);
        constexpr int nodes = 8, batches = 15;
        for (auto& c : candidates) {
            auto verify = [&] {
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(output.data(), dy, output.size() * sizeof(double), cudaMemcpyDeviceToHost));
                for (int row = 0; row < m; ++row) {
                    const long double error = std::abs(output[row] - reference[row]);
                    const long double bound = 8 * std::numeric_limits<double>::epsilon() * (n + 32.L) * magnitude[row] + 1e-12L;
                    if (!std::isfinite(output[row]) || error > bound) throw std::runtime_error("incorrect " + c.name);
                    c.error = std::max(c.error, (double)error);
                }
            };
            check(cudaMemsetAsync(dy, 0xff, output.size() * sizeof(double), stream));
            c.launch(); verify();
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int k = 0; k < nodes; ++k) c.launch();
            check(cudaStreamEndCapture(stream, &c.graph));
            check(cudaGraphInstantiate(&c.executable, c.graph, nullptr, nullptr, 0));
            check(cudaMemsetAsync(dy, 0xff, output.size() * sizeof(double), stream));
            check(cudaGraphLaunch(c.executable, stream)); verify();
        }
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        std::vector<int> order(candidates.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(seed);
        for (int batch = -3; !check_only && batch < batches; ++batch) {
            std::shuffle(order.begin(), order.end(), random);
            for (int id : order) {
                auto& c = candidates[id];
                check(cudaGraphLaunch(c.executable, stream));
                check(cudaEventRecord(start, stream));
                check(cudaGraphLaunch(c.executable, stream));
                check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
                float ms; check(cudaEventElapsedTime(&ms, start, stop));
                if (batch >= 0) c.times.push_back(ms * 1000. / nodes);
            }
        }
        if (!check_only) std::cout << "m,n,a_bytes,sm_count,kind,threads,blocks,resident_limit,min_rows_per_cta,max_rows_per_cta,median_us,p10_us,p90_us,max_abs_error,seed\n";
        for (auto& c : candidates) {
            if (!check_only) {
                std::sort(c.times.begin(), c.times.end());
                std::cout << std::setprecision(10) << m << ',' << n << ',' << a.size()*8 << ',' << prop.multiProcessorCount << ','
                          << c.name << ',' << c.threads << ',' << c.blocks << ',' << c.resident_limit << ',' << c.min_rows << ',' << c.max_rows << ','
                          << c.times[7] << ',' << c.times[1] << ',' << c.times[13] << ',' << c.error << ',' << seed << '\n';
            }
            check(cudaGraphExecDestroy(c.executable)); check(cudaGraphDestroy(c.graph));
        }
        if (check_only) std::cout << "PASS: " << candidates.size() << " direct and graph GEMV candidates\n";
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
        checkBlas(cublasDestroy(handle)); check(cudaStreamDestroy(stream));
        check(cudaFree(da)); check(cudaFree(dx)); check(cudaFree(dy));
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
