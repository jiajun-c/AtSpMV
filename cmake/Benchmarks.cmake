# Included from the repository root; paths remain relative to that source directory.
option(ATSPMV_BUILD_BENCHMARKS "Build experiment executables" ON)
if(ATSPMV_BUILD_BENCHMARKS)
    add_executable(sptrsv_plan_bench bench/sptrsv_plan_bench.cu)
    target_link_libraries(sptrsv_plan_bench PRIVATE atspmv_sptrsv_plan atspmv_core)
    add_executable(sptrsv_plan_setup_bench bench/sptrsv_plan_setup_bench.cu)
    target_link_libraries(sptrsv_plan_setup_bench PRIVATE atspmv_sptrsv_plan atspmv_core)
    add_executable(sptrsv_level_bench bench/sptrsv_level_bench.cu)
    target_link_libraries(sptrsv_level_bench PRIVATE atspmv_core)
    add_executable(spmv_plan_bench bench/spmv_plan_bench.cu)
    target_link_libraries(spmv_plan_bench PRIVATE atspmv_plan)
    add_executable(plan_model_replay tools/plan_model_replay.cpp)
    target_link_libraries(plan_model_replay PRIVATE atspmv_plan)
    add_executable(medium_row_model_predict tools/medium_row_model_predict.cpp)
    target_link_libraries(medium_row_model_predict PRIVATE atspmv_plan)
    add_executable(selection_model_cpu_bench bench/selection_model_cpu_bench.cpp)
    target_link_libraries(selection_model_cpu_bench PRIVATE atspmv_plan)
    add_executable(cusparse_spmm_core_compare bench/cusparse_spmm_core_compare.cu)
    target_link_libraries(cusparse_spmm_core_compare PRIVATE CUDA::cudart CUDA::cusparse)
    add_executable(absolute_reference_plan_replay tools/absolute_reference_plan_replay.cpp)
    target_link_libraries(absolute_reference_plan_replay PRIVATE atspmv_plan)
    add_executable(spmv_hybrid_validation
        bench/spmv_hybrid_validation.cu
        bench/detail/spmvBenchmarkSupport.cuh
        bench/detail/spmvBenchmarkOptions.h
        bench/detail/spmvRowOrder.cuh
        bench/detail/spmvBenchmarkValidation.h
    )
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
