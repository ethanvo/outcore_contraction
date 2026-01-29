cmake_minimum_required(VERSION 3.16)
project(outcore_contraction LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(outcore_contraction
  src/engine.cpp
  src/tensor_store.c
)

target_include_directories(outcore_contraction PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(HDF5 REQUIRED hdf5)
target_link_libraries(outcore_contraction PUBLIC ${HDF5_LIBRARIES} m)
target_include_directories(outcore_contraction PUBLIC ${HDF5_INCLUDE_DIRS})

add_executable(outcore_tests
  tests/engine_tests.cpp
)

target_link_libraries(outcore_tests PRIVATE outcore_contraction)
