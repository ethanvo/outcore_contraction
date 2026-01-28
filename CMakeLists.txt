cmake_minimum_required(VERSION 3.16)
project(outcore_contraction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(outcore_contraction
  src/engine.cpp
)

target_include_directories(outcore_contraction PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(outcore_tests
  tests/engine_tests.cpp
)

target_link_libraries(outcore_tests PRIVATE outcore_contraction)
