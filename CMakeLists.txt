cmake_minimum_required(VERSION 3.16)
project(outcore_contraction LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find MPI
find_package(MPI QUIET)
if(MPI_FOUND)
  message(STATUS "MPI found")
endif()

# Try to find parallel HDF5
find_package(HDF5 QUIET COMPONENTS C HL)
if(HDF5_FOUND)
  # Check if we can build with parallel HDF5 features
  if(MPI_FOUND AND HDF5_IS_PARALLEL)
    message(STATUS "Using parallel HDF5 with MPI support")
  else()
    message(STATUS "Using serial HDF5")
  endif()
else()
  message(STATUS "HDF5 not found, building without HDF5 support")
endif()

# Create library with core files
add_library(outcore_contraction
  src/engine.cpp
 )

# Set include directories
target_include_directories(outcore_contraction PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
 )

# Add tensor_store.c to library only if HDF5 is found
if(HDF5_FOUND)
  target_sources(outcore_contraction PRIVATE src/tensor_store.c)
  target_link_libraries(outcore_contraction PUBLIC ${HDF5_LIBRARIES})
  target_include_directories(outcore_contraction PUBLIC ${HDF5_INCLUDE_DIRS})
  target_compile_definitions(outcore_contraction PUBLIC HAVE_HDF5)
endif()

# Link MPI if it was found
if(MPI_FOUND)
  target_link_libraries(outcore_contraction PUBLIC MPI::MPI_C)
  target_compile_definitions(outcore_contraction PUBLIC HAVE_MPI)
endif()

add_executable(outcore_tests
  tests/engine_tests.cpp
 )

target_link_libraries(outcore_tests PRIVATE outcore_contraction)

# Add a test for tensor_store functions instead of executable
if(HDF5_FOUND)
  add_executable(test_tensor_store tests/test_tensor_store.c)
  target_link_libraries(test_tensor_store ${HDF5_LIBRARIES} m outcore_contraction)
  target_include_directories(test_tensor_store PUBLIC ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  
  # If MPI is available, link with MPI
  if(MPI_FOUND)
    target_link_libraries(test_tensor_store ${MPI_LIBRARIES})
    target_compile_definitions(test_tensor_store PUBLIC HAVE_MPI)
  endif()
endif()
