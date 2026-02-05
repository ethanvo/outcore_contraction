cmake_minimum_required(VERSION 3.12)
project(OOC_Tensor_Engine VERSION 1.0 LANGUAGES C)

# --- 1. Compiler Settings ---
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Useful for VSCode/Clangd intellisense

# --- 2. Dependencies ---
# We find HDF5 (C Component) and the Math library
find_package(HDF5 REQUIRED COMPONENTS C)

# --- 3. The Core Library (The "Engine") ---
# We compile the components into a static library to avoid repeating them
add_library(tensor_core STATIC
    src/tensor_store.c
    src/registry.c
    src/memory.c
)

# Allow the library to find headers in the current root directory
target_include_directories(tensor_core PUBLIC include)

# Link HDF5 and Math (-lm) to the core
target_link_libraries(tensor_core PRIVATE ${HDF5_C_LIBRARIES} m)
target_include_directories(tensor_core PRIVATE ${HDF5_INCLUDE_DIRS})

# --- 5. Test Drivers ---
# This matches the test_io.c you uploaded
if(EXISTS tests/test_io.c)
    add_executable(test_io tests/test_io.c)
    target_link_libraries(test_io PRIVATE tensor_core)
    target_link_libraries(test_io PRIVATE ${HDF5_C_LIBRARIES} m)
    target_include_directories(test_io PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# (Optional) Tensor Store Test
if(EXISTS tests/test_tensor_store.c)
    add_executable(test_tensor_store tests/test_tensor_store.c)
    target_link_libraries(test_tensor_store PRIVATE tensor_core)
    target_link_libraries(test_tensor_store PRIVATE ${HDF5_C_LIBRARIES} m)
    target_include_directories(test_tensor_store PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# --- 6. Output Config Info ---
message(STATUS "Build Config:")
message(STATUS "  HDF5 Found: ${HDF5_FOUND}")
message(STATUS "  HDF5 Libs:  ${HDF5_C_LIBRARIES}")

add_executable(gen_data src/gen_data.c)
target_link_libraries(gen_data PRIVATE tensor_core)
target_link_libraries(gen_data PRIVATE ${HDF5_C_LIBRARIES} m)
target_include_directories(gen_data PRIVATE ${HDF5_INCLUDE_DIRS})

add_executable(engine_app src/engine.c)
target_link_libraries(engine_app PRIVATE tensor_core)
target_link_libraries(engine_app PRIVATE ${HDF5_C_LIBRARIES} m)
target_include_directories(engine_app PRIVATE ${HDF5_INCLUDE_DIRS})
