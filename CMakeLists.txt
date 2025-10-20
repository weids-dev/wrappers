cmake_minimum_required(VERSION 3.10)
project(Wrappers)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Silence developer warnings that come from third-party code
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "" FORCE)

# --- procps / proc2 compatibility -----------------------------------
# Some vendored deps still do: pkg_check_modules(PROCPS REQUIRED libprocps)
# On newer distros the package is libproc2. We create a tiny pkg-config shim
# named "libprocps.pc" that forwards to libproc2, so no edits under external/.
find_package(PkgConfig REQUIRED)

# Prefer libproc2 if present; otherwise fall back to legacy libprocps.
pkg_check_modules(PROC2 QUIET libproc2)
if (PROC2_FOUND)
  # Create a local pkg-config directory with a shim for "libprocps"
  set(_shim_dir "${CMAKE_BINARY_DIR}/pkgconfig")
  file(MAKE_DIRECTORY "${_shim_dir}")
  # Write a minimal .pc that forwards Cflags/Libs to libproc2
  file(WRITE "${_shim_dir}/libprocps.pc"
"Name: libprocps (shim)
Description: Compatibility shim mapping libprocps -> libproc2
Version: 1.0
Libs: ${PROC2_LDFLAGS}
Cflags: ${PROC2_CFLAGS}
")
  # Ensure vendored projects see our shim first
  if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT \"$ENV{PKG_CONFIG_PATH}\" STREQUAL \"\")
    set(ENV{PKG_CONFIG_PATH} \"${_shim_dir}:$ENV{PKG_CONFIG_PATH}\")
  else()
    set(ENV{PKG_CONFIG_PATH} \"${_shim_dir}\")
  endif()
else()
  # Old distros: make sure libprocps exists so vendored checks pass
  pkg_check_modules(PROCPS REQUIRED libprocps)
endif()
# --------------------------------------------------------------------

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
