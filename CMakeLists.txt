cmake_minimum_required(VERSION 3.12)
project(OOC_Tensor_Engine VERSION 1.0 LANGUAGES C)

# --- Compiler Settings ---
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Treat warnings as errors during development; apply to all targets below.
add_compile_options(-Wall -Wextra)

# --- Dependencies ---
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Threads REQUIRED)

# --- Core Library ---
# tensor_store, registry, memory, and engine are compiled as one static lib
# so every executable avoids duplicating them.
add_library(tensor_core STATIC
    src/tensor_store.c
    src/registry.c
    src/memory.c
    src/engine.c
    src/einsum.c
    src/odometer.c
    src/write_queue.c
    src/metal_backend.m
    src/tensor_engine.c
)
# metal_backend.m must be compiled as Objective-C (not plain C).
set_source_files_properties(src/metal_backend.m PROPERTIES
    COMPILE_FLAGS "-x objective-c")
target_include_directories(tensor_core PUBLIC include)
target_include_directories(tensor_core PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(tensor_core PRIVATE ${HDF5_C_LIBRARIES} Threads::Threads m)

# Metal frameworks (macOS / Apple Silicon only).
if(APPLE)
    find_library(METAL_FW       Metal)
    find_library(FOUNDATION_FW  Foundation)
    if(METAL_FW AND FOUNDATION_FW)
        target_link_libraries(tensor_core PRIVATE
            ${METAL_FW} ${FOUNDATION_FW})
        target_compile_definitions(tensor_core PRIVATE HAVE_METAL)
        message(STATUS "  Metal       : enabled (${METAL_FW})")
    else()
        message(STATUS "  Metal       : not found – GPU worker disabled")
    endif()
endif()

# --- BLAS ---
#
# Set -DBLAS_BACKEND=Accelerate|OpenBLAS|None  (default: auto-detect)
#
# Auto-detect on macOS: Accelerate → OpenBLAS → None
# Auto-detect on Linux: OpenBLAS  → None
#
set(BLAS_BACKEND "Auto" CACHE STRING
    "BLAS backend: Auto (default) | Accelerate | MKL | OpenBLAS | None")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS Auto Accelerate MKL OpenBLAS None)

# Normalise "Auto" to the best available backend.
if(BLAS_BACKEND STREQUAL "Auto")
    if(APPLE)
        set(BLAS_BACKEND "Accelerate")
    else()
        # Prefer MKL when MKLROOT is set or the oneAPI default path exists.
        find_path(_mkl_probe mkl_cblas.h
            PATHS
                "$ENV{MKLROOT}/include"
                /opt/intel/oneapi/mkl/latest/include
                /opt/intel/mkl/include
            NO_DEFAULT_PATH)
        if(_mkl_probe)
            set(BLAS_BACKEND "MKL")
        else()
            set(BLAS_BACKEND "OpenBLAS")
        endif()
        unset(_mkl_probe CACHE)
    endif()
endif()

if(BLAS_BACKEND STREQUAL "Accelerate")
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_compile_definitions(tensor_core PRIVATE
            HAVE_CBLAS USE_ACCELERATE ACCELERATE_NEW_LAPACK)
        target_link_libraries(tensor_core PRIVATE ${ACCELERATE_FRAMEWORK})
        message(STATUS "  BLAS        : Apple Accelerate (AMX/vecLib)")
    else()
        message(WARNING "  BLAS        : Accelerate requested but not found"
                        " – falling back to None")
    endif()

elseif(BLAS_BACKEND STREQUAL "MKL")
    find_path(MKL_INCLUDE_DIR mkl_cblas.h
        PATHS
            "$ENV{MKLROOT}/include"
            /opt/intel/oneapi/mkl/latest/include
            /opt/intel/mkl/include
        NO_DEFAULT_PATH
    )
    find_library(MKL_CORE_LIB mkl_core
        PATHS
            "$ENV{MKLROOT}/lib"
            "$ENV{MKLROOT}/lib/intel64"
            /opt/intel/oneapi/mkl/latest/lib
            /opt/intel/oneapi/mkl/latest/lib/intel64
            /opt/intel/mkl/lib/intel64
        NO_DEFAULT_PATH
    )
    # Intel OpenMP runtime — required by mkl_intel_thread.
    # Search order: CMPLR_ROOT env var (standard oneAPI activation), the
    # Spack view layout (view/compiler/VERSION beside view/mkl/VERSION),
    # and the conventional bare-metal oneAPI install path.
    # Version-specific subdirs are tried after the generic "latest" symlink
    # so newer oneAPI releases work without changes here.
    get_filename_component(_mkl_view_root "$ENV{MKLROOT}/../.." ABSOLUTE)
    find_library(IOMP5_LIB iomp5
        PATHS
            "$ENV{CMPLR_ROOT}/linux/compiler/lib/intel64_lin"
            "$ENV{CMPLR_ROOT}/lib"
            "${_mkl_view_root}/compiler/latest/lib"
            "${_mkl_view_root}/compiler/2025.3/lib"
            "${_mkl_view_root}/compiler/2025.2/lib"
            "${_mkl_view_root}/compiler/2025.1/lib"
            /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin
    )
    if(MKL_INCLUDE_DIR AND MKL_CORE_LIB)
        get_filename_component(MKL_LIB_DIR "${MKL_CORE_LIB}" DIRECTORY)
        target_compile_definitions(tensor_core PRIVATE HAVE_CBLAS USE_MKL)
        target_include_directories(tensor_core PRIVATE "${MKL_INCLUDE_DIR}")
        target_link_directories(tensor_core PRIVATE "${MKL_LIB_DIR}")
        if(IOMP5_LIB)
            get_filename_component(IOMP5_LIB_DIR "${IOMP5_LIB}" DIRECTORY)
            # mkl_intel_thread: multi-threaded via Intel OpenMP (iomp5).
            # -fopenmp initialises the runtime; without it MKL silently
            # falls back to 1 thread even when MKL_NUM_THREADS > 1.
            target_compile_options(tensor_core PRIVATE -fopenmp)
            target_link_directories(tensor_core PRIVATE "${IOMP5_LIB_DIR}")
            target_link_libraries(tensor_core PRIVATE
                mkl_intel_lp64 mkl_intel_thread mkl_core iomp5
                pthread dl
            )
            message(STATUS "  BLAS        : Intel MKL threaded (iomp5, ${MKL_LIB_DIR})")
        else()
            # iomp5 not found — fall back to sequential layer.
            target_link_libraries(tensor_core PRIVATE
                mkl_intel_lp64 mkl_sequential mkl_core
                pthread dl
            )
            message(STATUS "  BLAS        : Intel MKL sequential (iomp5 not found, ${MKL_LIB_DIR})")
        endif()
    else()
        message(WARNING "  BLAS        : MKL requested but not found"
                        " – falling back to None")
    endif()

elseif(BLAS_BACKEND STREQUAL "OpenBLAS")
    # find_package(BLAS) locates the library; we then separately find cblas.h
    # because not all installs place the header in a standard location.
    # CMAKE_PREFIX_PATH is honoured first so Spack/vcpkg/Homebrew prefixes work
    # without hardcoding machine-specific paths here.
    find_package(BLAS)
    if(BLAS_FOUND)
        find_path(CBLAS_INCLUDE_DIR cblas.h
            PATH_SUFFIXES openblas
            PATHS
                /usr/local/include
                /opt/homebrew/include
                /opt/homebrew/include/openblas
                /usr/include/openblas
                /usr/include
        )
        if(CBLAS_INCLUDE_DIR)
            target_compile_definitions(tensor_core PRIVATE HAVE_CBLAS)
            target_include_directories(tensor_core PRIVATE ${CBLAS_INCLUDE_DIR})
            target_link_libraries(tensor_core PRIVATE ${BLAS_LIBRARIES})
            message(STATUS "  BLAS        : OpenBLAS  cblas.h in ${CBLAS_INCLUDE_DIR}")
        else()
            message(STATUS "  BLAS        : OpenBLAS library found but cblas.h missing"
                           " – using fallback kernel")
        endif()
    else()
        message(STATUS "  BLAS        : OpenBLAS not found – using fallback kernel")
    endif()

else() # None
    message(STATUS "  BLAS        : None – using fallback kernel")
endif()

# --- Executables ---

# Main contraction driver (thin wrapper around run_contraction).
add_executable(engine_app src/main.c)
target_link_libraries(engine_app PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
target_include_directories(engine_app PRIVATE ${HDF5_INCLUDE_DIRS})

# Test data generator.
add_executable(gen_data src/gen_data.c)
target_link_libraries(gen_data PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
target_include_directories(gen_data PRIVATE ${HDF5_INCLUDE_DIRS})

# --- Test Drivers ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor_store.c)
    add_executable(test_tensor_store tests/test_tensor_store.c)
    target_link_libraries(test_tensor_store PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(test_tensor_store PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_io.c)
    add_executable(test_io tests/test_io.c)
    target_link_libraries(test_io PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(test_io PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_high_rank.c)
    add_executable(test_high_rank tests/test_high_rank.c)
    target_link_libraries(test_high_rank PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(test_high_rank PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_einsum.c)
    add_executable(test_einsum tests/test_einsum.c)
    target_link_libraries(test_einsum PRIVATE tensor_core)
    target_include_directories(test_einsum PRIVATE include)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_odometer.c)
    add_executable(test_odometer tests/test_odometer.c)
    target_link_libraries(test_odometer PRIVATE tensor_core m)
    target_include_directories(test_odometer PRIVATE include)
endif()

# --- Generic complex tensor generator ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_complex_tensor.c)
    add_executable(gen_complex_tensor tools/gen_complex_tensor.c)
    target_link_libraries(gen_complex_tensor PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(gen_complex_tensor PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "  gen_complex_tensor: enabled")
endif()

# --- Small contraction benchmark ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_small_contraction.c)
    add_executable(bench_small_contraction tests/bench_small_contraction.c)
    target_link_libraries(bench_small_contraction PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(bench_small_contraction PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "  bench_small_contraction: enabled")
endif()

# --- 100 GB complex data generator ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_100gb_complex.c)
    add_executable(gen_100gb_complex tools/gen_100gb_complex.c)
    target_link_libraries(gen_100gb_complex PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(gen_100gb_complex PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "  gen_100gb_complex: enabled")
endif()

# --- 100 GB complex benchmark ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_100gb_contraction.c)
    add_executable(bench_100gb_contraction tests/bench_100gb_contraction.c)
    target_link_libraries(bench_100gb_contraction PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(bench_100gb_contraction PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "  bench_100gb_contraction: enabled")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_odd_strides_40gb.c)
    add_executable(bench_odd_strides_40gb tests/bench_odd_strides_40gb.c)
    target_link_libraries(bench_odd_strides_40gb PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(bench_odd_strides_40gb PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "  bench_odd_strides_40gb: enabled")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_compute_bound_40gb.c)
    add_executable(bench_compute_bound_40gb tests/bench_compute_bound_40gb.c)
    target_link_libraries(bench_compute_bound_40gb PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(bench_compute_bound_40gb PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "  bench_compute_bound_40gb: enabled")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mixed_rank.c)
    add_executable(test_mixed_rank tests/test_mixed_rank.c)
    target_link_libraries(test_mixed_rank PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(test_mixed_rank PRIVATE ${HDF5_INCLUDE_DIRS})
    message(STATUS "  test_mixed_rank: enabled")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_100gb_contraction.c)
    add_executable(test_100gb_contraction tests/test_100gb_contraction.c)
    target_link_libraries(test_100gb_contraction PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(test_100gb_contraction PRIVATE ${HDF5_INCLUDE_DIRS})
    # cmake -DSMALL_TEST=1 → GLOBAL_DIM=40 for a quick smoke test (~820 MB / tensor).
    # Default (no flag) → GLOBAL_DIM=340 (≈106 GB / tensor, production scale).
    if(SMALL_TEST)
        target_compile_definitions(test_100gb_contraction PRIVATE SMALL_TEST=1)
        message(STATUS "  test_100gb_contraction: SMALL_TEST=1 (40^4 tensors)")
    else()
        message(STATUS "  test_100gb_contraction: full scale (340^4 ≈ 106 GB / tensor)")
    endif()
endif()

# --- Consolidated benchmark suite ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/run_all.c)
    add_executable(bench_run_all bench/run_all.c)
    target_link_libraries(bench_run_all PRIVATE tensor_core ${HDF5_C_LIBRARIES} m)
    target_include_directories(bench_run_all PRIVATE include ${HDF5_INCLUDE_DIRS})
    message(STATUS "  bench_run_all: enabled")
endif()

# --- Diagnostics ---
message(STATUS "Build config:")
message(STATUS "  C standard  : ${CMAKE_C_STANDARD}")
message(STATUS "  HDF5 found  : ${HDF5_FOUND}")
message(STATUS "  HDF5 libs   : ${HDF5_C_LIBRARIES}")
