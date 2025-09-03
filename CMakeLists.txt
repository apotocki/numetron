cmake_minimum_required(VERSION 3.15)
project(numetron_tests LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Enable folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Enable testing
include(CTest)
enable_testing()

# Collect test sources
set (NUMETRON_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/basic_integer_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/basic_decimal_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/ct_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/mul_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/mpn_mul_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.cpp
    
    # MASM source(s)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arch/x86_64/alderlake/mul_basecase.s
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arch/x86_64/k8/mul_basecase.s
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arch/x86_64/core2/mul_basecase.s
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arch/x86_64/detect_platform.s
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arch/x86_64/detect_mul_basecase.s
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
endif()

# Define the test executable
add_executable(numetron_tests ${NUMETRON_TEST_SOURCES})
target_include_directories(numetron_tests PRIVATE include ${GMP_INCLUDE_DIR})

# Nice warnings for GCC/Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(numetron_tests PRIVATE -Wall -Wextra -Wpedantic -march=native -O3)
endif()

# Link dependencies
target_link_libraries(numetron_tests PRIVATE ${DEPS})

# Add the test to CTest
add_test(NAME numetron_tests COMMAND numetron_tests)
