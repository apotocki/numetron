cmake_minimum_required(VERSION 3.15)
project(numetron_tests LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Enable folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Enable testing
include(CTest)
enable_testing()

# The back end (the compiled algorithms and kernels, the src/arch assembly), built once as the
# static "numetron" library; numetron_tests and numetron_bench_mul link it instead of listing its
# sources, so new sources only need to be added in lib/CMakeLists.txt.
add_subdirectory(lib)

# The library's C++ (the compiled algorithms, NUMETRON_COMPILED) takes the same target flags as
# the tests: the FFT kernel follows them.
get_target_property(_numetron_lib_type numetron TYPE)
if (NOT _numetron_lib_type STREQUAL "INTERFACE_LIBRARY" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(numetron PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# Collect test sources
set (NUMETRON_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/float16_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/basic_integer_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/basic_decimal_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/ct_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/mul_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/mpn_mul_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/mul_shapes_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/udiv_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/assign_mul_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.cpp
)

# Benchmark sources (standalone executables, not wired into CTest)
set (NUMETRON_BENCH_MUL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/mul_bench.cpp
)

# If we are using vcpkg, rely on CMake config packages
if(DEFINED ENV{VCPKG_ROOT})
    message(STATUS "Using vcpkg from: $ENV{VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

    find_package(ZLIB REQUIRED)
    find_package(ICU REQUIRED COMPONENTS i18n uc data)
    find_package(BZip2 REQUIRED)
    find_package(LZ4 CONFIG REQUIRED)
    find_package(LibLZMA REQUIRED)
    find_package(GTest CONFIG REQUIRED)

    set(DEPS
        GTest::gtest
        GTest::gtest_main
    )
    set(DEPS_BENCH "")

else()
    message(STATUS "No vcpkg detected. Using system packages.")

    # GTest fallback search
    find_package(GTest REQUIRED)

    # LZ4 fallback search
    find_path(GMP_INCLUDE_DIR NAMES gmp.h)
    find_library(GMP_LIBRARY NAMES libgmp.so)
    if(NOT GMP_INCLUDE_DIR)
        message(FATAL_ERROR "GMP_INCLUDE_DIR not found. Install libgmp3-dev.")
    endif()
    if(NOT GMP_LIBRARY)
        message(FATAL_ERROR "GMP_LIBRARY not found. Install libgmp3-dev.")
    endif()

    set(DEPS
        GTest::GTest
        GTest::Main
        ${GMP_LIBRARY}
    )
    set(DEPS_BENCH ${GMP_LIBRARY})
endif()

# Define the test executable
add_executable(numetron_tests ${NUMETRON_TEST_SOURCES})
target_include_directories(numetron_tests PRIVATE include ${GMP_INCLUDE_DIR})

# Nice warnings for GCC/Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(numetron_tests PRIVATE -Wall -Wextra -Wpedantic -march=native -O3)
endif()

# Link dependencies
target_link_libraries(numetron_tests PRIVATE numetron ${DEPS})

# The tests of the plans call the multiplication chain directly: that unit is compiled with the
# library's configuration (numetron_internal's definitions), the others as an application.
get_target_property(_numetron_internal_definitions numetron_internal INTERFACE_COMPILE_DEFINITIONS)
if (_numetron_internal_definitions)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/tests/mul_shapes_test.cpp
    PROPERTIES COMPILE_DEFINITIONS "${_numetron_internal_definitions}")
endif()

# Add the test to CTest
add_test(NAME numetron_tests COMMAND numetron_tests)

# Define the mul benchmark executable (Numetron vs GMP, standalone, not run via ctest).
# It links numetron_internal: the threshold tuner in it runs the multiplication chain directly,
# with the library's configuration.
add_executable(numetron_bench_mul ${NUMETRON_BENCH_MUL_SOURCES})
target_include_directories(numetron_bench_mul PRIVATE include ${GMP_INCLUDE_DIR})

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(numetron_bench_mul PRIVATE -Wall -Wextra -Wpedantic -march=native -O3)
endif()

target_link_libraries(numetron_bench_mul PRIVATE numetron_internal ${DEPS_BENCH})
