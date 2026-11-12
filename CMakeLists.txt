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
    include/detail/rowLengths.cuh
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

# Build the triangular-solve experiment only when explicitly requested or used
# by its benchmarks/tests; a core-only build does not compile unrelated work.
add_library(atspmv_sptrsv_plan STATIC EXCLUDE_FROM_ALL src/sptrsvPlan.cu)
target_include_directories(atspmv_sptrsv_plan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(atspmv_sptrsv_plan PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(atspmv_sptrsv_plan PUBLIC CUDA::cudart CUDA::cusparse)

# Unified owner/dispatcher, usable without the optional SM90 experiment.
add_library(atspmv_plan STATIC
    src/spmvPlan.cu
    src/selection/spmvPlanModel.cpp
    src/selection/vectorModelCalibration.cpp
    src/selection/vectorModelCalibrationV2.cpp
    src/selection/vectorModelCalibrationGraph.cpp
    src/selection/mediumRowModel.cpp
    src/selection/absoluteReferenceModel.cpp
    include/spmvPlan.h
    include/detail/spmvPlanKernels.cuh
    include/detail/planValueUpdate.cuh
    include/detail/compressedTileColumns.cuh
    include/detail/groupedTileColumns.cuh
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
    include/mediumRowModel.h
    include/absoluteReferenceModel.h
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
    src/spmvPlan.cu src/selection/spmvPlanModel.cpp src/selection/vectorModelCalibration.cpp src/selection/vectorModelCalibrationV2.cpp src/selection/vectorModelCalibrationGraph.cpp src/selection/mediumRowModel.cpp src/selection/absoluteReferenceModel.cpp
    include/spmvPlan.h include/detail/spmvPlanKernels.cuh include/detail/planValueUpdate.cuh include/detail/planModelSelection.h include/detail/planTimingSelection.h include/detail/vectorModelFeaturesV2.h include/detail/planStorage.cuh include/detail/mixedTile.cuh
    include/detail/spmvPlanDsm.h include/unifiedSpmvModel.h include/hybridModelFeatures.h
    include/hybridModelCalibration.h include/vectorModelCalibration.h include/vectorModelCalibrationV2.h include/mediumRowModel.h include/absoluteReferenceModel.h
)
target_link_libraries(atspmv_plan PUBLIC atspmv_core)

# Keep target names and option defaults stable; separate experiment registration
# from the libraries used by callers. Includes intentionally share root scope.
include(cmake/Benchmarks.cmake)
include(cmake/Applications.cmake)
include(cmake/Tests.cmake)
include(cmake/DsmExperiment.cmake)
