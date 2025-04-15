cmake_minimum_required(VERSION 3.10)
project(SumCheckProtocol)

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(EXPORT_COMPILE_COMMANDS ON)

include_directories(src)

# Add test executable
add_executable(test_sumcheck_multilinear tests/test_sumcheck_multilinear.cpp)

# Link against Google Test libraries
target_link_libraries(test_sumcheck_multilinear GTest::GTest GTest::Main)

# Enable testing with CTest
enable_testing()
add_test(NAME SumCheckMultilinearTests COMMAND test_sumcheck_multilinear)