cmake_minimum_required(VERSION 3.10)
project(Wrappers)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- libff ----------------------------------------------------------
# We only want the library, no sub-modules or tests (legacy libff 2014 CMake)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)       # standard CMake knob
set(BUILD_TESTS   OFF CACHE BOOL "" FORCE)       # libff's own switch
# set(IS_LIBFF_PARENT ON CACHE BOOL "" FORCE)     # skip dependences
add_subdirectory(external/libff) # looking for CMakeLists.txt in dependencies

# Re-enable testsing for current project
set(BUILD_TESTING ON CACHE BOOL "" FORCE)

# --- GoogleTest -----------------------------------------------------
find_package(GTest REQUIRED)

add_executable(test_sumcheck_multilinear
    tests/test_sumcheck_multilinear.cpp)

target_include_directories(test_sumcheck_multilinear PRIVATE
    external/libff
    src)

target_link_libraries(test_sumcheck_multilinear PRIVATE
    ff
    gmpxx       
    GTest::GTest
    GTest::Main)

# --- CTest ----------------------------------------------------------
enable_testing()
add_test(NAME SumCheckMultilinearTests COMMAND test_sumcheck_multilinear)