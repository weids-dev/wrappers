cmake_minimum_required(VERSION 3.10)
project(Wrappers)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Silence developer warnings that come from third-party code
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "" FORCE)

# --- libff & libfqfft -----------------------------------------------
# We only want the library, no sub-modules or tests (legacy libff 2014 CMake)
# set(BUILD_TESTING OFFCACHE BOOL "" FORCE)       # standard CMake knob
# set(BUILD_TESTS   OFF CACHE BOOL "" FORCE)       # libff's own switch
add_subdirectory(external/libfqfft)

# Re-enable testsing for current project
set(BUILD_TESTING ON CACHE BOOL "" FORCE)

# --- GoogleTest -----------------------------------------------------
find_package(GTest REQUIRED)

add_executable(wrappers
  src/wrappers.cpp)

add_executable(test_sumcheck_multilinear
    tests/test_sumcheck_multilinear.cpp)

add_executable(test_gkr_dense
    tests/test_gkr_dense.cpp)

# Headers
target_include_directories(test_sumcheck_multilinear PRIVATE
    external/libfqfft
    external/libfqfft/depends/libff
    src)

target_include_directories(wrappers PRIVATE
    external/libfqfft
    external/libfqfft/depends/libff
    src)

# Headers
target_include_directories(test_gkr_dense PRIVATE
    external/libfqfft
    external/libfqfft/depends/libff
    src)

target_link_libraries(test_sumcheck_multilinear PRIVATE
    ff
    gmpxx gmp
    GTest::GTest
    GTest::Main)

target_link_libraries(test_gkr_dense PRIVATE
    ff
    gmpxx gmp
    GTest::GTest
    GTest::Main)

# --- CTest ----------------------------------------------------------
enable_testing()
add_test(NAME SumCheckMultilinearTests COMMAND test_sumcheck_multilinear)
add_test(NAME GKRDenseTests COMMAND test_gkr_dense)
