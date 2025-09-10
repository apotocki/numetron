// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include <iostream>

using namespace numetron;

#ifdef NUMETRON_BOOST_AUTO_TEST_REGISTRATION
void numetron_test_registrar()
{
    //register_test(BOOST_TEST_CASE(&mul_test));
    register_test(BOOST_TEST_CASE(&basic_integer_test0));
    register_test(BOOST_TEST_CASE(&basic_decimal_test0));
    register_test(BOOST_TEST_CASE_SILENT(&ct_test));
    register_test(BOOST_TEST_CASE(&test_float16_basic_operations));
    register_test(BOOST_TEST_CASE(&test_float16_special_values));
    register_test(BOOST_TEST_CASE(&test_float16_zeros));
    register_test(BOOST_TEST_CASE(&test_float16_small_values));
    register_test(BOOST_TEST_CASE(&test_float16_precision_limits));
    register_test(BOOST_TEST_CASE(&test_float16_arithmetic_properties));
    register_test(BOOST_TEST_CASE(&test_float16_negative_values));
}
AUTOTEST(numetron_test_registrar)
#else

TEST(NumetronTest, float16_basic_operations) { test_float16_basic_operations(); }
TEST(NumetronTest, float16_special_values) { test_float16_special_values(); }
TEST(NumetronTest, float16_zeros) { test_float16_zeros(); }
TEST(NumetronTest, float16_small_values) { test_float16_small_values(); }
TEST(NumetronTest, float16_precision_limits) { test_float16_precision_limits(); }
TEST(NumetronTest, float16_arithmetic_properties) { test_float16_arithmetic_properties(); }
TEST(NumetronTest, float16_negative_values) { test_float16_negative_values(); }
TEST(NumetronTest, float16_integral_constructor) { test_float16_integral_constructor(); }
TEST(NumetronTest, float16_static_values) { test_float16_static_values(); }
TEST(NumetronTest, float16_static_values_ordering) { test_float16_static_values_ordering(); }
TEST(NumetronTest, float16_static_values_bit_patterns) { test_float16_static_values_bit_patterns(); }
TEST(NumetronTest, float16_integral_equality) { test_float16_integral_equality(); }
TEST(NumetronTest, float16_integral_equality_special_values) { test_float16_integral_equality_special_values(); }
TEST(NumetronTest, float16_integral_equality_precision) { test_float16_integral_equality_precision(); }
TEST(NumetronTest, float16_floating_point_equality) { test_float16_floating_point_equality(); }
TEST(NumetronTest, float16_floating_point_equality_special_values) { test_float16_floating_point_equality_special_values(); }
TEST(NumetronTest, float16_floating_point_equality_precision_limits) { test_float16_floating_point_equality_precision_limits(); }
TEST(NumetronTest, float16_floating_point_mixed_precision) { test_float16_floating_point_mixed_precision(); }
TEST(NumetronTest, float16_double_constructor) { test_float16_double_constructor(); }
TEST(NumetronTest, float16_double_precision_conversion) { test_float16_double_precision_conversion(); }
TEST(NumetronTest, float16_double_comparison) { test_float16_double_comparison(); }
TEST(NumetronTest, float16_integral_three_way_comparison) { test_float16_integral_three_way_comparison(); }
TEST(NumetronTest, float16_integral_three_way_comparison_special_values) { test_float16_integral_three_way_comparison_special_values(); }
TEST(NumetronTest, float16_integral_three_way_comparison_boundary_values) { test_float16_integral_three_way_comparison_boundary_values(); }
TEST(NumetronTest, float16_integral_three_way_comparison_precision) { test_float16_integral_three_way_comparison_precision(); }
TEST(NumetronTest, float16_integral_three_way_comparison_derived_operators) { test_float16_integral_three_way_comparison_derived_operators(); }
TEST(NumetronTest, float16_floating_point_three_way_comparison) { test_float16_floating_point_three_way_comparison(); }
TEST(NumetronTest, float16_floating_point_three_way_comparison_special_values) { test_float16_floating_point_three_way_comparison_special_values(); }
TEST(NumetronTest, float16_floating_point_three_way_comparison_precision) { test_float16_floating_point_three_way_comparison_precision(); }
TEST(NumetronTest, float16_floating_point_three_way_comparison_derived_operators) { test_float16_floating_point_three_way_comparison_derived_operators(); }

TEST(NumetronTest, basic_integer) { basic_integer_test0(); }
TEST(NumetronTest, basic_decimal) { basic_decimal_test0(); }
TEST(NumetronTest, compile_time) { ct_test(); }

//TEST(NumetronTest, mul) { mul_test(); }
//TEST(NumetronTest, mpn_mul) { mpn_mul_test(); }

#if !defined(NDEBUG) && defined(_WIN32)
#   include "windows.h"
#   define _CRTDBG_MAP_ALLOC //to get more details
#   include <crtdbg.h>   //for malloc and free
#endif

int main(int argc, char** argv)
{
#if !defined(NDEBUG) && defined(_WIN32)
    // Enable automatic leak detection at program exit
    int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    // Turn on leak-checking bit.  
    tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
    //// Turn off CRT block checking bit.  
    //tmpFlag &= ~_CRTDBG_CHECK_CRT_DF;
    _CrtSetDbgFlag(tmpFlag);

    // Configure CRT debug reporting to output to stdout
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
#endif

    ::testing::InitGoogleTest(&argc, argv);
    
    return RUN_ALL_TESTS();
}
#endif