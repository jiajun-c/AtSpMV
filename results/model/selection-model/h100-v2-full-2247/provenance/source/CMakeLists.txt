cmake_minimum_required(VERSION 3.22)

# Build for SM80 by default; override for the target GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
endif()
project(AtSpMV VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
include(FetchContent)
FetchContent_Declare(fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
    GIT_TAG b6172c96ef73d16b2c10917373bacec1a7583a31
)
FetchContent_MakeAvailable(fast_matrix_market)

# Compile each implementation once. The core and baseline libraries are
# independent; the historical libatspmv.a remains a combined compatibility
# archive for callers that link it directly outside CMake.
add_library(atspmv_core_objects OBJECT
    src/csrReader.cpp
    src/atspmv.cu
    src/atspmvProduct.cu
)
target_include_directories(atspmv_core_objects PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(atspmv_core_objects PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(atspmv_core_objects PUBLIC
    CUDA::cudart CUDA::cusparse CUDA::cublas
    OpenMP::OpenMP_CXX fast_matrix_market::fast_matrix_market
)
target_compile_options(atspmv_core_objects PUBLIC
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=-fopenmp>"
)
add_library(atspmv_core STATIC $<TARGET_OBJECTS:atspmv_core_objects>)
target_link_libraries(atspmv_core PUBLIC atspmv_core_objects)

add_library(atspmv_baseline_objects OBJECT EXCLUDE_FROM_ALL
    src/baselines/amgT.cu
    src/baselines/dasp.cu
    src/baselines/cusparseCSR.cu
)
target_link_libraries(atspmv_baseline_objects PUBLIC atspmv_core)
add_library(atspmv_baselines STATIC EXCLUDE_FROM_ALL $<TARGET_OBJECTS:atspmv_baseline_objects>)
target_link_libraries(atspmv_baselines PUBLIC atspmv_core)

add_library(atspmv STATIC EXCLUDE_FROM_ALL
    $<TARGET_OBJECTS:atspmv_core_objects>
    $<TARGET_OBJECTS:atspmv_baseline_objects>
)
target_link_libraries(atspmv PUBLIC atspmv_core_objects)

# Unified owner/dispatcher, usable without the optional SM90 experiment.
add_library(atspmv_plan STATIC
    src/spmvPlan.cu
    src/selection/spmvPlanModel.cpp
    src/selection/vectorModelCalibration.cpp
    src/selection/vectorModelCalibrationV2.cpp
    include/spmvPlan.h
    include/detail/spmvPlanKernels.cuh
    include/detail/planModelSelection.h
    include/detail/planTimingSelection.h
    include/detail/vectorModelFeaturesV2.h
    include/detail/planStorage.cuh
    include/detail/mixedTile.cuh
    include/detail/spmvPlanDsm.h
    include/unifiedSpmvModel.h
    include/hybridModelFeatures.h
    include/hybridModelCalibration.h
    include/vectorModelCalibration.h
    include/vectorModelCalibrationV2.h
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
    src/spmvPlan.cu src/selection/spmvPlanModel.cpp src/selection/vectorModelCalibration.cpp src/selection/vectorModelCalibrationV2.cpp
    include/spmvPlan.h include/detail/spmvPlanKernels.cuh include/detail/planModelSelection.h include/detail/planTimingSelection.h include/detail/vectorModelFeaturesV2.h include/detail/planStorage.cuh include/detail/mixedTile.cuh
    include/detail/spmvPlanDsm.h include/unifiedSpmvModel.h include/hybridModelFeatures.h
    include/hybridModelCalibration.h include/vectorModelCalibration.h include/vectorModelCalibrationV2.h
)
target_link_libraries(atspmv_plan PUBLIC atspmv_core)

option(ATSPMV_BUILD_BENCHMARKS "Build experiment executables" ON)
if(ATSPMV_BUILD_BENCHMARKS)
    add_executable(spmv_plan_bench bench/spmv_plan_bench.cu)
    target_link_libraries(spmv_plan_bench PRIVATE atspmv_plan)
    add_executable(plan_model_replay tools/plan_model_replay.cpp)
    target_link_libraries(plan_model_replay PRIVATE atspmv_plan)
    add_executable(spmv_hybrid_validation bench/spmv_hybrid_validation.cu)
    target_link_libraries(spmv_hybrid_validation PRIVATE atspmv_plan)
    add_executable(cusparse_buffer_probe bench/cusparse_buffer_probe.cu)
    target_link_libraries(cusparse_buffer_probe PRIVATE atspmv_core)
    add_executable(mixed_tile_bench bench/mixed_tile_bench.cu)
    target_link_libraries(mixed_tile_bench PRIVATE atspmv_core)
    add_executable(tile_model_predict bench/tile_model_predict.cpp)
    target_include_directories(tile_model_predict PRIVATE include)
    target_compile_features(tile_model_predict PRIVATE cxx_std_17)
    add_executable(spmv_double bench/spmv_double.cpp)
    target_link_libraries(spmv_double PRIVATE atspmv)
    add_executable(tile_model_bench bench/tile_model_bench.cu)
    target_link_libraries(tile_model_bench PRIVATE atspmv_core)
endif()

# Keep application targets independently configurable from microbenchmarks.
# The default follows ATSPMV_BUILD_BENCHMARKS to preserve existing builds.
option(ATSPMV_BUILD_APPS "Build PageRank and Jacobi applications" ${ATSPMV_BUILD_BENCHMARKS})
if(ATSPMV_BUILD_APPS)
    foreach(app IN ITEMS jacobi pagerank pagerank_atspmv)
        add_executable(${app} apps/${app}.cu)
        target_link_libraries(${app} PRIVATE atspmv_core)
    endforeach()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(plan_timing_selection_test tests/plan_timing_selection_test.cpp)
    target_include_directories(plan_timing_selection_test PRIVATE include)
    target_link_libraries(plan_timing_selection_test PRIVATE CUDA::cudart)
    target_compile_features(plan_timing_selection_test PRIVATE cxx_std_17)
    add_test(NAME plan_timing_selection COMMAND plan_timing_selection_test)
    add_executable(tile_model_test tests/tile_model_test.cpp)
    target_include_directories(tile_model_test PRIVATE include)
    target_compile_features(tile_model_test PRIVATE cxx_std_17)
    add_test(NAME tile_model COMMAND tile_model_test)
    add_executable(spmv_plan_test tests/spmv_plan_test.cu)
    target_link_libraries(spmv_plan_test PRIVATE atspmv_plan)
    add_executable(spmv_regression tests/spmv_regression.cu)
    target_link_libraries(spmv_regression PRIVATE atspmv_core)
    add_executable(baseline_regression EXCLUDE_FROM_ALL tests/baseline_regression.cpp)
    target_link_libraries(baseline_regression PRIVATE atspmv)
    option(ATSPMV_ENABLE_GPU_TESTS "Run GPU correctness tests in CTest" OFF)
    if(ATSPMV_ENABLE_GPU_TESTS)
        set_target_properties(baseline_regression PROPERTIES EXCLUDE_FROM_ALL FALSE)
        add_test(NAME spmv_plan COMMAND spmv_plan_test)
        set_tests_properties(spmv_plan PROPERTIES TIMEOUT 300)
        add_test(NAME baseline_regression COMMAND baseline_regression "${CMAKE_CURRENT_SOURCE_DIR}/datasets/matrices/test.mtx")
        set_tests_properties(baseline_regression PROPERTIES TIMEOUT 300)
        add_test(NAME spmv_regression COMMAND spmv_regression)
        add_test(NAME spmv_tuned_regression COMMAND spmv_regression)
        set_tests_properties(spmv_tuned_regression PROPERTIES ENVIRONMENT
            "ATSPMV_TUNE_CAP=16;ATSPMV_TUNE_WARPS=2;ATSPMV_TUNE_PREFETCH=4;ATSPMV_TUNE_CTA_REDUCE=1")
        if(ATSPMV_BUILD_BENCHMARKS)
            add_test(NAME tile_model_gpu COMMAND tile_model_bench --synthetic 2)
            add_test(NAME tile_tuned_merge_gpu COMMAND tile_model_bench --synthetic 2)
            set_tests_properties(tile_tuned_merge_gpu PROPERTIES ENVIRONMENT
                "ATSPMV_TUNE_CAP=4;ATSPMV_TUNE_WARPS=8;ATSPMV_TUNE_PREFETCH=8;ATSPMV_TUNE_CTA_REDUCE=1")
            add_test(NAME tile_tuned_prefetch_gpu COMMAND tile_model_bench --synthetic 2)
            set_tests_properties(tile_tuned_prefetch_gpu PROPERTIES ENVIRONMENT
                "ATSPMV_TUNE_CAP=32;ATSPMV_TUNE_WARPS=2;ATSPMV_TUNE_PREFETCH=2;ATSPMV_TUNE_CTA_REDUCE=0")
        endif()
    endif()
endif()


# DSM is isolated from the default SM80 library and never selected implicitly.
option(ATSPMV_BUILD_DSM_EXPERIMENT "Build experimental SM90 distributed shared memory paths" OFF)
if(ATSPMV_BUILD_DSM_EXPERIMENT)
    if(CUDAToolkit_VERSION VERSION_LESS 12.0)
        message(FATAL_ERROR "The DSM experiment requires CUDA 12 or newer")
    endif()
    add_library(atspmv_plan_dsm OBJECT src/spmvPlanDsm.cu)
    set_target_properties(atspmv_plan_dsm PROPERTIES CUDA_ARCHITECTURES 90)
    target_link_libraries(atspmv_plan_dsm PRIVATE atspmv_core)
    target_sources(atspmv_plan PRIVATE $<TARGET_OBJECTS:atspmv_plan_dsm>)
    target_compile_definitions(atspmv_plan PRIVATE ATSPMV_PLAN_HAS_DSM=1)
    # Keep ordinary-device images and add native SM90 for the two-stage
    # comparator as well; enabling DSM must not handicap that comparator.
    if(CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(all|all-major|native)$")
        set(plan_architectures ${CMAKE_CUDA_ARCHITECTURES} 90)
        list(REMOVE_DUPLICATES plan_architectures)
        set_target_properties(atspmv_plan PROPERTIES CUDA_ARCHITECTURES "${plan_architectures}")
    endif()
    add_library(atspmv_dsm STATIC src/dsmSpmv.cu src/dsmPageRank.cu)
    set_target_properties(atspmv_dsm PROPERTIES CUDA_ARCHITECTURES 90)
    target_link_libraries(atspmv_dsm PUBLIC atspmv_core)
    if(ATSPMV_BUILD_BENCHMARKS)
        add_executable(dsm_cluster_sweep bench/dsm_cluster_sweep.cu)
        set_target_properties(dsm_cluster_sweep PROPERTIES CUDA_ARCHITECTURES 90)
        target_link_libraries(dsm_cluster_sweep PRIVATE CUDA::cudart)
        add_executable(dsm_hbm_latency bench/dsm_hbm_latency.cu)
        set_target_properties(dsm_hbm_latency PROPERTIES CUDA_ARCHITECTURES 90)
        target_link_libraries(dsm_hbm_latency PRIVATE CUDA::cudart)
        if(BUILD_TESTING AND ATSPMV_ENABLE_GPU_TESTS)
            add_test(NAME dsm_hbm_latency_correctness COMMAND dsm_hbm_latency 64 1 202610271 1)
            set_tests_properties(dsm_hbm_latency_correctness PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
            add_test(NAME dsm_cluster_sweep_correctness COMMAND dsm_cluster_sweep 64 5 202610271 --check-only)
            set_tests_properties(dsm_cluster_sweep_correctness PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
        endif()
        add_executable(dsm_latency_bench bench/dsm_latency_bench.cu)
        set_target_properties(dsm_latency_bench PROPERTIES CUDA_ARCHITECTURES 90)
        target_link_libraries(dsm_latency_bench PRIVATE CUDA::cudart)
        if(BUILD_TESTING AND ATSPMV_ENABLE_GPU_TESTS)
            add_test(NAME dsm_latency_correctness COMMAND dsm_latency_bench 64 5 202610271 --check-only)
            set_tests_properties(dsm_latency_correctness PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
        endif()
        add_executable(dsm_long_row_bench bench/dsm_long_row_bench.cu)
        set_target_properties(dsm_long_row_bench PROPERTIES CUDA_ARCHITECTURES 90)
        target_link_libraries(dsm_long_row_bench PRIVATE atspmv_dsm)
        if(BUILD_TESTING AND ATSPMV_ENABLE_GPU_TESTS)
            foreach(case IN ITEMS empty tail mixed)
                if(case STREQUAL "empty")
                    set(long_row_args 7 0)
                elseif(case STREQUAL "tail")
                    set(long_row_args 137 1003)
                else()
                    set(long_row_args mixed:137 16387)
                endif()
                add_test(NAME dsm_long_row_${case} COMMAND dsm_long_row_bench ${long_row_args} 202610301 --check-only)
                set_tests_properties(dsm_long_row_${case} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
            endforeach()
        endif()
        add_executable(gemv_persistent_bench bench/gemv_persistent_bench.cu)
        set_target_properties(gemv_persistent_bench PROPERTIES CUDA_ARCHITECTURES 90)
        target_compile_features(gemv_persistent_bench PRIVATE cuda_std_17)
        target_link_libraries(gemv_persistent_bench PRIVATE CUDA::cudart CUDA::cublas)
        if(BUILD_TESTING AND ATSPMV_ENABLE_GPU_TESTS)
            add_test(NAME gemv_persistent_correctness COMMAND gemv_persistent_bench 137 1003 202610290 --check-only)
            set_tests_properties(gemv_persistent_correctness PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
        endif()
        add_executable(dsm_spmv_bench bench/dsm_spmv_bench.cu)
        set_target_properties(dsm_spmv_bench PROPERTIES CUDA_ARCHITECTURES 90)
        target_link_libraries(dsm_spmv_bench PRIVATE atspmv_dsm)
        add_executable(dsm_pagerank_bench bench/dsm_pagerank_bench.cu)
        set_target_properties(dsm_pagerank_bench PROPERTIES CUDA_ARCHITECTURES 90)
        target_link_libraries(dsm_pagerank_bench PRIVATE atspmv_dsm)
        if(BUILD_TESTING AND ATSPMV_ENABLE_GPU_TESTS)
            add_test(NAME dsm_pagerank_correctness COMMAND dsm_pagerank_bench --synthetic 7 20261030 --check-only)
            set_tests_properties(dsm_pagerank_correctness PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
            # All-dangling, cached-row boundary, and uncached long-row paths.
            foreach(size IN ITEMS 1 513 4097)
                foreach(iterations IN ITEMS 0 1 2 7)
                    set(test_name dsm_pagerank_n${size}_i${iterations})
                    add_test(NAME ${test_name} COMMAND dsm_pagerank_bench
                        --synthetic:${size} ${iterations} 20261104 --check-only)
                    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
                endforeach()
            endforeach()
        endif()
    endif()
    if(BUILD_TESTING)
        add_executable(dsm_spmv_test tests/dsm_spmv_test.cu)
        set_target_properties(dsm_spmv_test PROPERTIES CUDA_ARCHITECTURES 90)
        target_link_libraries(dsm_spmv_test PRIVATE atspmv_dsm)
        if(ATSPMV_ENABLE_GPU_TESTS)
            add_test(NAME dsm_spmv_correctness COMMAND dsm_spmv_test)
            set_tests_properties(dsm_spmv_correctness PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
        endif()
    endif()
endif()
