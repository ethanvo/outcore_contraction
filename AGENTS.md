# AGENTS.md

## Build, Lint, and Test Commands

### Build Commands
```bash
# Build the project with CMake
mkdir -p build && cd build
cmake ..
make

# Build with specific flags
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Build tests specifically
make outcore_tests
```

### Lint Commands
```bash
# Run clang-tidy (if available)
clang-tidy src/*.cpp -- -std=c++17 -I include

# Run clang-format check
find src tests -name "*.cpp" -o -name "*.h" | xargs clang-format -style=file -Werror --dry-run
```

### Test Commands
```bash
# Run all tests
./build/outcore_tests

# Run a single test file
cd build && ./outcore_tests

# Run with verbose output
./build/outcore_tests --verbose

# Run with specific test filter (if supported by test framework)
./build/outcore_tests --gtest_filter="*TestName*"
```

## Code Style Guidelines

### General
- Follow C++17 standard
- Use modern C++ features (smart pointers, range-based loops, etc.)
- Keep functions small and focused (preferably under 50 lines)
- Prefer const correctness and noexcept where appropriate

### Naming Conventions
- Classes: PascalCase (e.g., `OutcoreEngine`, `LruCache`)
- Functions: camelCase (e.g., `RegisterBlock`, `QueuePrefetch`)
- Variables: camelCase (e.g., `cacheBytes`, `elementCount`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_BUFFER_SIZE`)
- Private members: `camelCase_` (e.g., `cacheBytes_`)

### Formatting
- Use 2-space indentation (no tabs)
- Braces on the same line for functions, newlines for control structures
- One statement per line
- No trailing whitespace
- Space after commas, no space before semicolons

### Includes
- Group includes by type: system headers, third-party, local headers
- Order within groups alphabetically
- Use angle brackets for system headers, quotes for local headers
- Minimize include dependencies

### Types and Variables
- Prefer `auto` when type is obvious
- Use `const` and `constexpr` where appropriate
- Use `std::size_t` for sizes and counts
- Use `std::optional` for nullable values
- Use `std::unique_ptr`/`std::shared_ptr` for dynamic allocation

### Error Handling
- Use `std::optional` for operations that may fail
- Use exceptions for error conditions that shouldn't happen
- Prefer RAII over manual resource management
- Check for overflow conditions where appropriate

### Thread Safety
- Use mutexes for shared mutable state
- Prefer `std::mutex` and `std::lock_guard` over raw locking
- Use atomic operations for simple shared flags
- Avoid race conditions in concurrent code

### Memory Management
- Prefer heap allocation over stack when possible
- Use RAII for automatic resource management
- Prefer raw `new`/`delete` except in special cases
- Use `std::unique_ptr` for ownership transfer

### Documentation
- Use Doxygen-style comments for public APIs
- Document complex algorithms with comments
- Add class-level documentation for public interfaces
- Document function parameters and return values
