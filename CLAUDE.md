# Modernization Guide for GIZA++ and mkcls

This document outlines recommendations for updating the GIZA++ and mkcls codebases to modern C++ standards while preserving their algorithmic correctness and improving maintainability, performance, and developer experience.

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [C++ Standards Upgrade](#c-standards-upgrade)
3. [Build System Modernization](#build-system-modernization)
4. [Code Modernization Recommendations](#code-modernization-recommendations)
5. [Performance Improvements](#performance-improvements)
6. [Code Quality and Maintainability](#code-quality-and-maintainability)
7. [Testing Infrastructure](#testing-infrastructure)
8. [Documentation](#documentation)
9. [Phased Implementation Plan](#phased-implementation-plan)

---

## Current State Assessment

### Codebase Characteristics

**Positive Aspects:**
- Well-structured algorithm implementations
- Clear separation of concerns (models, tables, data structures)
- Extensive parameter system for experimentation
- Proven correctness (widely used in MT community for 20+ years)

**Areas for Modernization:**
- **C++ Standard**: Uses C++98/03 with some TR1 features
- **Hash Maps**: Uses deprecated `hash_map` with compiler-specific compatibility layer
- **Memory Management**: Primarily raw pointers, manual memory management
- **Build System**: GNU Make with manual dependency tracking
- **Compiler Compatibility**: Extensive preprocessor hacks for GCC 2/3/4+ compatibility
- **STL Usage**: Limited use of modern STL features
- **Error Handling**: Primarily assertions, limited exception safety
- **Code Style**: Inconsistent, pre-modern C++ conventions

### Compatibility Layer Analysis

From `GIZA++-v2/mystl.h:22-35`:
```cpp
#if __GNUC__==2
#include <hash_map>
#elif __GNUC__==3
#include <ext/hash_map>
using __gnu_cxx::hash_map;
#else
#if __cplusplus < 201103L && !defined(_LIBCPP_VERSION)
#include <tr1/unordered_map>
using namespace std::tr1;
#else
#include <unordered_map>
#endif
#define hash_map unordered_map
#endif
```

This shows the codebase still supports very old compilers (GCC 2/3) which are no longer relevant.

---

## C++ Standards Upgrade

### Recommended Target: C++17

**Rationale:**
- Widely supported by all modern compilers (GCC 7+, Clang 5+, MSVC 2017+)
- Provides significant quality-of-life improvements over C++11/14
- Not as cutting-edge as C++20/23, ensuring broader compatibility
- Includes all essential modern features needed for this codebase

**Alternative:** C++14 as minimum, C++20 as optional

### Key C++11/14/17 Features to Adopt

#### 1. Auto Type Deduction

**Before:**
```cpp
for(typename leda_h_array<A,B>::const_iterator __jj__=(d).begin();
    __jj__!=(d).end();
    ++__jj__)
```

**After:**
```cpp
for(const auto& [key, value] : d)  // C++17 structured bindings
```

#### 2. Range-Based For Loops

**Before:**
```cpp
for(unsigned int i=0; i<a.length(); i++)
```

**After:**
```cpp
for(const auto& elem : a)
```

#### 3. Smart Pointers

Replace raw pointers with:
- `std::unique_ptr<T>` for exclusive ownership
- `std::shared_ptr<T>` for shared ownership (use sparingly)
- `std::weak_ptr<T>` for non-owning references

This eliminates manual `new`/`delete` and prevents memory leaks.

#### 4. Standard Containers

**Replace:**
- `hash_map` → `std::unordered_map`
- Custom `leda_h_array` → `std::unordered_map` with default values via wrapper if needed
- C-style arrays → `std::array` (fixed size) or `std::vector` (dynamic)

#### 5. Move Semantics

Enable efficient transfers of large objects (e.g., probability tables) without copying:
```cpp
class TTables {
    std::unordered_map<WordPair, Probability> table_;
public:
    TTables(TTables&&) = default;  // Move constructor
    TTables& operator=(TTables&&) = default;  // Move assignment
};
```

#### 6. nullptr

Replace `NULL` and `0` for pointers with `nullptr`:
```cpp
if (ptr == nullptr) { ... }
```

#### 7. Enum Classes

**Before:**
```cpp
enum { INIT_RAN, INIT_AIO, INIT_FREQ, INIT_LWRW };
```

**After:**
```cpp
enum class InitMethod { Random, AllInOne, Frequency, LocalWordRW };
```

#### 8. constexpr and const

Mark compile-time constants and immutable functions:
```cpp
constexpr int MAX_FERTILITY = 10;
constexpr double compute_threshold(int k) { return k * 0.01; }
```

#### 9. Lambda Functions

Replace function pointers and functors:
```cpp
std::sort(words.begin(), words.end(),
          [](const auto& a, const auto& b) { return a.frequency > b.frequency; });
```

#### 10. std::optional (C++17)

For values that may or may not be present:
```cpp
std::optional<Alignment> findBestAlignment(const SentencePair& sp);
```

---

## Build System Modernization

### Replace GNU Make with CMake

**Benefits:**
- Cross-platform support (Windows, macOS, Linux)
- Modern dependency tracking (automatic)
- Better IDE integration (CLion, VS Code, Visual Studio)
- Easier to add tests and external dependencies
- Standard in modern C++ projects

**Proposed CMakeLists.txt Structure:**

```cmake
cmake_minimum_required(VERSION 3.15)
project(giza-pp VERSION 2.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0>
)

# GIZA++ executable
file(GLOB GIZA_SOURCES "GIZA++-v2/*.cpp")
list(REMOVE_ITEM GIZA_SOURCES
     "${CMAKE_CURRENT_SOURCE_DIR}/GIZA++-v2/plain2snt.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/GIZA++-v2/snt2plain.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/GIZA++-v2/snt2cooc.cpp")

add_executable(GIZA++ ${GIZA_SOURCES})

# Utility tools
add_executable(plain2snt GIZA++-v2/plain2snt.cpp)
add_executable(snt2plain GIZA++-v2/snt2plain.cpp)
add_executable(snt2cooc GIZA++-v2/snt2cooc.cpp)

# mkcls executable
file(GLOB MKCLS_SOURCES "mkcls-v2/*.cpp")
add_executable(mkcls ${MKCLS_SOURCES})

# Installation
install(TARGETS GIZA++ plain2snt snt2plain snt2cooc mkcls
        RUNTIME DESTINATION bin)

# Testing (optional)
enable_testing()
add_subdirectory(tests)
```

**Migration Path:**
1. Create `CMakeLists.txt` alongside existing `Makefile`
2. Test CMake build in parallel
3. Eventually deprecate GNU Make once CMake is stable

---

## Code Modernization Recommendations

### 1. Replace Custom Hash Map Wrapper

**Current:** `leda_h_array<K,V>` in `mystl.h`

**Modernized Replacement:**

```cpp
template<typename K, typename V>
class DefaultMap {
private:
    std::unordered_map<K, V> map_;
    V default_value_;

public:
    explicit DefaultMap(V default_val = V{})
        : default_value_(std::move(default_val)) {}

    const V& operator[](const K& key) const {
        auto it = map_.find(key);
        return (it != map_.end()) ? it->second : default_value_;
    }

    V& operator[](const K& key) {
        return map_.try_emplace(key, default_value_).first->second;
    }

    bool contains(const K& key) const { return map_.contains(key); }  // C++20
    // Or for C++17: bool contains(const K& key) const { return map_.count(key) > 0; }

    auto begin() { return map_.begin(); }
    auto end() { return map_.end(); }
    auto begin() const { return map_.begin(); }
    auto end() const { return map_.end(); }

    size_t size() const { return map_.size(); }
};
```

**Benefits:**
- Uses standard `std::unordered_map` internally
- Modern interface with range-based for loop support
- Type-safe with automatic type deduction

### 2. Replace Custom Macros with Range-Based Loops

**Remove macros from `mystl.h:42-47`:**
```cpp
#define over_string(a,i) for(unsigned int i=0;i<a.length();i++)
#define over_array(a,i) for(i=(a).low();i<=(a).high();i++)
```

**Replace with:**
```cpp
for (auto& elem : container) { ... }
for (size_t i = 0; i < container.size(); ++i) { ... }
```

### 3. Modernize Parameter System

**Current:** Global macros in `Parameter.h`

**Proposed:** Type-safe configuration class

```cpp
class AlignmentConfig {
public:
    struct ModelIterations {
        int model1 = 5;
        int model2 = 0;
        int hmm = 5;
        int model3 = 5;
        int model4 = 5;
        int model5 = 0;
    };

    ModelIterations iterations;
    std::filesystem::path source_vocab;  // C++17
    std::filesystem::path target_vocab;
    std::filesystem::path corpus;
    std::string output_prefix = "alignment";

    // Load from config file
    static AlignmentConfig from_file(const std::filesystem::path& config_path);

    // Load from command-line arguments
    static AlignmentConfig from_args(int argc, char* argv[]);
};
```

**Benefits:**
- Type-safe (no macro errors)
- Easy to serialize/deserialize
- Clear structure and documentation
- Unit testable

### 4. Improve Table Classes with Modern C++

**Example: TTables Modernization**

**Before (current):**
```cpp
class TTables {
    hash_map<wordPairIds, LpPair<COUNT, PROB>> table;
    // ...
};
```

**After:**
```cpp
class TTables {
public:
    using WordPair = std::pair<WordIndex, WordIndex>;
    using Probability = double;
    using Count = double;

    struct Entry {
        Count count;
        Probability prob;
    };

private:
    std::unordered_map<WordPair, Entry> table_;
    Probability smoothing_floor_ = 1e-12;

public:
    // Modern interface
    Probability get_prob(WordIndex source, WordIndex target) const;
    void set_count(WordIndex source, WordIndex target, Count count);
    void normalize();

    // Iteration support
    auto begin() const { return table_.begin(); }
    auto end() const { return table_.end(); }

    // Serialization
    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);

    // Move semantics for efficiency
    TTables(TTables&&) = default;
    TTables& operator=(TTables&&) = default;

    // Delete copy for large tables (force move)
    TTables(const TTables&) = delete;
    TTables& operator=(const TTables&) = delete;
};
```

### 5. Error Handling Modernization

**Current:** Assertions via `massert()`, `iassert()` macros

**Proposed:** Mix of exceptions and assertions

```cpp
// For recoverable errors (e.g., file I/O, invalid input)
class AlignmentError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class FileIOError : public AlignmentError {
public:
    explicit FileIOError(const std::filesystem::path& path)
        : AlignmentError("Failed to access file: " + path.string()) {}
};

// For programming errors (keep assertions)
#include <cassert>
assert(fertility >= 0 && fertility <= MAX_FERTILITY);

// Or use contracts in C++20/23 (future)
```

**Usage:**
```cpp
void TTables::load(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw FileIOError(path);
    }
    // ...
}
```

### 6. Alignment Class Modernization

**Current:** Separate alignment representation per model

**Proposed:** Unified alignment class

```cpp
class Alignment {
public:
    using Position = size_t;
    using AlignmentLinks = std::vector<std::vector<Position>>;

private:
    AlignmentLinks links_;  // links_[source_pos] = {target_pos1, target_pos2, ...}
    size_t source_length_;
    size_t target_length_;
    double log_probability_;

public:
    Alignment(size_t source_len, size_t target_len)
        : links_(source_len),
          source_length_(source_len),
          target_length_(target_len),
          log_probability_(-std::numeric_limits<double>::infinity()) {}

    void add_link(Position source_pos, Position target_pos);
    void remove_link(Position source_pos, Position target_pos);

    const std::vector<Position>& targets_for_source(Position source_pos) const {
        return links_.at(source_pos);
    }

    size_t fertility(Position source_pos) const {
        return links_[source_pos].size();
    }

    // Iteration
    struct Link {
        Position source;
        Position target;
    };

    std::vector<Link> all_links() const;

    // Probability
    double log_prob() const { return log_probability_; }
    void set_log_prob(double lp) { log_probability_ = lp; }

    // I/O
    void save(std::ostream& out) const;
    static Alignment load(std::istream& in);
};
```

### 7. Model Interface Standardization

Create abstract base class for all models:

```cpp
class AlignmentModel {
public:
    virtual ~AlignmentModel() = default;

    // Core training interface
    virtual void train_iteration(const SentencePairCollection& corpus) = 0;
    virtual Alignment viterbi_alignment(const SentencePair& sp) const = 0;
    virtual double log_probability(const SentencePair& sp, const Alignment& a) const = 0;

    // Model transfer
    virtual void initialize_from(const AlignmentModel& previous_model) = 0;

    // Serialization
    virtual void save(const std::filesystem::path& base_path) const = 0;
    virtual void load(const std::filesystem::path& base_path) = 0;

    // Statistics
    virtual double perplexity(const SentencePairCollection& corpus) const = 0;
};

class Model1 : public AlignmentModel {
    TTables ttables_;
public:
    void train_iteration(const SentencePairCollection& corpus) override;
    // ...
};
```

**Benefits:**
- Polymorphic interface enables generic training loops
- Easier testing (mock implementations)
- Clear contracts via pure virtual functions

---

## Performance Improvements

### 1. Parallelization

**Modern C++ Threading:**

Replace single-threaded training with parallel processing:

```cpp
#include <execution>  // C++17 parallel algorithms
#include <thread>
#include <mutex>

void Model1::train_iteration(const SentencePairCollection& corpus) {
    std::mutex count_mutex;
    CountTable local_counts;

    // Parallel E-step
    std::for_each(std::execution::par,
                  corpus.begin(), corpus.end(),
                  [&](const SentencePair& sp) {
        auto local_count = compute_posteriors(sp);

        std::lock_guard lock(count_mutex);
        merge_counts(local_counts, local_count);
    });

    // M-step (normalization)
    normalize_counts(local_counts);
}
```

**Alternative:** Use thread pool for finer control

```cpp
#include <BS_thread_pool.hpp>  // External library

BS::thread_pool pool(std::thread::hardware_concurrency());

for (const auto& sp : corpus) {
    pool.push_task([&, sp]() {
        // Process sentence pair
    });
}
pool.wait_for_tasks();
```

**Reference:** mgiza (https://github.com/moses-smt/mgiza) already implements multi-threading for GIZA++

### 2. SIMD Vectorization

For inner loops in probability computation:

```cpp
#include <immintrin.h>  // AVX instructions

// Example: vectorized probability summation
double sum_probabilities(const std::vector<double>& probs) {
    // Modern compilers can auto-vectorize, but explicit SIMD available if needed
    return std::reduce(std::execution::par_unseq,
                      probs.begin(), probs.end(), 0.0);
}
```

### 3. Memory Optimization

**Use std::pmr (C++17) for custom allocators:**

```cpp
#include <memory_resource>

class Model3 {
    std::pmr::monotonic_buffer_resource pool_{1024 * 1024 * 100};  // 100MB
    std::pmr::polymorphic_allocator<SentencePair> alloc_{&pool_};

    // Fast allocation during iteration, bulk deallocation after
};
```

**Benefits:**
- Reduced allocation overhead
- Better cache locality
- Faster iteration times

---

## Code Quality and Maintainability

### 1. Adopt clang-format for Code Formatting

**Create `.clang-format`:**

```yaml
BasedOnStyle: Google
IndentWidth: 2
ColumnLimit: 100
PointerAlignment: Left
AllowShortFunctionsOnASingleLine: Empty
```

**Usage:**
```bash
clang-format -i GIZA++-v2/*.cpp GIZA++-v2/*.h
```

### 2. Static Analysis

**Enable in CMake:**

```cmake
# clang-tidy
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

# cppcheck
find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)
```

**CI Integration:** Run on every commit via GitHub Actions

### 3. Sanitizers for Runtime Checking

**CMake configuration:**

```cmake
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()
```

### 4. Documentation with Doxygen

**Generate API documentation:**

```cpp
/**
 * @brief Computes Viterbi alignment for a sentence pair using IBM Model 1.
 *
 * @param source_sentence Source language sentence (word indices)
 * @param target_sentence Target language sentence (word indices)
 * @return Alignment object with maximum probability alignment
 *
 * @note Complexity: O(l * m) where l = source length, m = target length
 *
 * @see Model1::log_probability for computing alignment probability
 */
Alignment Model1::viterbi_alignment(
    const std::vector<WordIndex>& source_sentence,
    const std::vector<WordIndex>& target_sentence) const;
```

**Doxyfile for generation:**
```bash
doxygen -g  # Generate template
doxygen     # Generate docs
```

### 5. Namespace Organization

Avoid `using namespace std` globally:

```cpp
namespace gizapp {
namespace alignment {

class Model1 { ... };
class Model2 { ... };

}  // namespace alignment

namespace clustering {

class MKCls { ... };

}  // namespace clustering
}  // namespace gizapp
```

---

## Testing Infrastructure

### Unit Tests with Google Test

**CMake setup:**

```cmake
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(giza_tests
    tests/model1_test.cpp
    tests/ttables_test.cpp
    tests/alignment_test.cpp
)
target_link_libraries(giza_tests gtest_main)
include(GoogleTest)
gtest_discover_tests(giza_tests)
```

**Example test:**

```cpp
#include <gtest/gtest.h>
#include "model1.h"

TEST(Model1Test, ViterbiAlignmentSimple) {
    Model1 model;
    // Initialize with known probabilities
    model.set_prob(/*source=*/1, /*target=*/1, 0.8);
    model.set_prob(/*source=*/1, /*target=*/2, 0.2);

    std::vector<WordIndex> source = {1};
    std::vector<WordIndex> target = {1, 2};

    Alignment alignment = model.viterbi_alignment(source, target);

    EXPECT_EQ(alignment.targets_for_source(0).size(), 1);
    EXPECT_EQ(alignment.targets_for_source(0)[0], 0);  // Aligns to target word 1
}

TEST(Model1Test, TrainingConvergence) {
    // Test that Model 1 EM converges on toy data
    // ...
}
```

### Integration Tests

**Test end-to-end alignment:**

```cpp
TEST(IntegrationTest, Model1ToModel3Pipeline) {
    SentencePairCollection corpus = load_test_corpus("data/test.txt");

    Model1 m1;
    m1.train(corpus, /*iterations=*/5);

    Model3 m3;
    m3.initialize_from(m1);
    m3.train(corpus, /*iterations=*/5);

    // Check alignment quality metrics
    double aer = compute_alignment_error_rate(m3, corpus);
    EXPECT_LT(aer, 0.3);  // Expect < 30% error on toy data
}
```

### Continuous Integration

**GitHub Actions workflow (`.github/workflows/ci.yml`):**

```yaml
name: CI

on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        compiler: [g++, clang++]
        build_type: [Debug, Release]

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        # Install build tools

    - name: Configure CMake
      run: cmake -B build -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}

    - name: Build
      run: cmake --build build

    - name: Test
      run: cd build && ctest --output-on-failure

    - name: Run sanitizers
      if: matrix.build_type == 'Debug'
      run: |
        cmake -B build-asan -DENABLE_ASAN=ON
        cmake --build build-asan
        cd build-asan && ctest
```

---

## Documentation

### 1. Update README.md

**Modern README structure:**

```markdown
# GIZA++ and mkcls

Statistical word alignment and word clustering for machine translation.

## Quick Start

### Building

    mkdir build && cd build
    cmake ..
    make -j$(nproc)

### Usage

    # Train alignment models
    ./GIZA++ config.txt

    # Cluster words
    ./mkcls -c50 -n10 -pinput.txt -Voutput.classes

## Documentation

- [Algorithm Overview](OVERVIEW.md)
- [API Documentation](https://your-project.github.io/giza-pp/)
- [User Guide](docs/user_guide.md)

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15+

## Citation

    @article{och2003systematic,
      title={A systematic comparison of various statistical alignment models},
      author={Och, Franz Josef and Ney, Hermann},
      journal={Computational linguistics},
      year={2003}
    }
```

### 2. API Documentation

- Generate with Doxygen
- Host on GitHub Pages
- Include usage examples

### 3. Tutorial and Examples

**Create `examples/` directory:**

```
examples/
├── basic_alignment/
│   ├── README.md
│   ├── train.cpp
│   └── data/
├── custom_model/
│   └── ...
└── python_binding/
    └── ...
```

---

## Phased Implementation Plan

### Phase 1: Foundation (2-4 weeks)

**Goals:** Modernize build system and establish new coding standards

**Tasks:**
1. Create CMake build system alongside existing Makefiles
2. Set up CI pipeline (GitHub Actions)
3. Add clang-format configuration
4. Create initial test framework with a few unit tests
5. Update README with modern build instructions

**Deliverables:**
- Working CMake build
- Basic CI running on PRs
- Formatted code (at least in new files)

### Phase 2: Core Modernization (6-8 weeks)

**Goals:** Migrate core data structures to modern C++

**Tasks:**
1. Replace `hash_map` with `std::unordered_map` throughout
2. Update `mystl.h` to use standard library containers
3. Modernize TTables, NTables, ATables, D4Tables
4. Replace macros with range-based for loops
5. Add smart pointers where appropriate
6. Update to C++17 standard
7. Add comprehensive unit tests for updated components

**Deliverables:**
- Core tables using `std::unordered_map`
- Modern iteration patterns
- 50%+ test coverage of core components

### Phase 3: Model Modernization (6-8 weeks)

**Goals:** Refactor model implementations with modern C++

**Tasks:**
1. Create `AlignmentModel` base class interface
2. Refactor Model1, Model2, HMM to use modern C++
3. Refactor Model3, Model4, Model5
4. Standardize alignment representation
5. Improve error handling (exceptions + assertions)
6. Add integration tests

**Deliverables:**
- Unified model interface
- Modern model implementations
- End-to-end integration tests

### Phase 4: Performance Optimization (4-6 weeks)

**Goals:** Improve performance through parallelization and optimization

**Tasks:**
1. Profile current implementation
2. Implement multi-threaded training (parallel E-step)
3. Optimize memory allocation patterns
4. Add SIMD optimizations where beneficial
5. Benchmark against baseline

**Deliverables:**
- Multi-threaded implementation
- Performance benchmarks
- 2-4x speedup on multi-core systems

### Phase 5: Polish and Documentation (3-4 weeks)

**Goals:** Production-ready codebase with excellent documentation

**Tasks:**
1. Complete test coverage (aim for 80%+)
2. Generate API documentation with Doxygen
3. Write user guide and tutorials
4. Add Python bindings (optional)
5. Deprecate old Makefiles
6. Create release artifacts

**Deliverables:**
- Comprehensive documentation
- Release v2.0
- Migration guide from v1.x

### Total Timeline: 21-30 weeks (5-7 months)

---

## Backward Compatibility Considerations

### Option 1: Clean Break (Recommended)

- Version 2.0 with modern codebase
- Maintain v1.x branch for bug fixes only
- Provide migration guide
- Archive old code as `legacy/` directory

**Rationale:** Clean slate enables best practices without technical debt

### Option 2: Gradual Migration

- Maintain compatibility layer during transition
- Support both old and new APIs simultaneously
- Deprecation warnings for old APIs
- Remove after 1-2 release cycles

**Rationale:** Easier for existing users, but slows modernization

### Recommended: Option 1 (Clean Break)

The codebase is mature and stable. Users can continue using v1.x while transitioning to v2.0 on their own schedule.

---

## Migration Guide for Users

**For users upgrading from GIZA++ v1.x to v2.0:**

### Build System

**Old:**
```bash
cd GIZA++-v2
make
```

**New:**
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Configuration Files

**Minimal changes required:**
- Configuration file format remains the same
- Output file formats remain compatible
- Model files from v1.x can be loaded in v2.0 (with conversion utility)

### Command-Line Interface

**Mostly compatible:**
```bash
# Works in both v1.x and v2.0
./GIZA++ -S source.vcb -T target.vcb -C corpus.snt

# New in v2.0: cleaner configuration
./GIZA++ --config config.yaml
```

---

## Conclusion

Modernizing GIZA++ and mkcls to C++17 will significantly improve:

1. **Maintainability:** Modern C++ is easier to read, write, and maintain
2. **Performance:** Parallelization and better memory management
3. **Portability:** CMake enables easier cross-platform builds
4. **Safety:** Smart pointers and RAII reduce memory errors
5. **Developer Experience:** Better tooling, IDE support, and debugging
6. **Future-Proofing:** Positioned to adopt C++20/23 features as needed

The phased approach allows incremental progress while maintaining a working codebase at each stage. The estimated 5-7 month timeline is aggressive but achievable with dedicated effort.

**Key Success Factors:**
- Comprehensive testing to ensure algorithmic correctness
- Performance benchmarking to validate improvements
- Community engagement for feedback and adoption
- Clear documentation for both users and developers

**Next Steps:**
1. Review and approve this modernization plan
2. Set up development branch (`develop-v2`)
3. Begin Phase 1 (Foundation)
4. Establish regular review checkpoints

---

## Additional Resources

### Learning Modern C++

- **"Effective Modern C++" by Scott Meyers** - Essential C++11/14 patterns
- **"C++17 - The Complete Guide" by Nicolai Josuttis**
- **CppCon talks** - Annual C++ conference with excellent talks on YouTube

### Tools

- **CMake:** https://cmake.org/
- **Google Test:** https://github.com/google/googletest
- **clang-format:** https://clang.llvm.org/docs/ClangFormat.html
- **clang-tidy:** https://clang.llvm.org/extra/clang-tidy/
- **Compiler Explorer:** https://godbolt.org/ (great for exploring assembly/optimization)

### Related Projects

- **mgiza:** https://github.com/moses-smt/mgiza (multi-threaded GIZA++)
- **fast_align:** https://github.com/clab/fast_align (modern, fast alignment)
- **eflomal:** https://github.com/robertostling/eflomal (efficient low-memory aligner)

These projects demonstrate various approaches to modernizing and optimizing alignment algorithms.
