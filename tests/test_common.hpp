// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

//#include <string>
//#include <vector>
//#include <utility>
//#include <iterator>
//#include <iostream>
//#include <span>

#include <gtest/gtest.h>

#define CHECK(condition) EXPECT_TRUE(condition)
#define CHECK_EQUAL(lhs, rhs) EXPECT_EQ((lhs), (rhs))
#define CHECK_LE(lhs, rhs) EXPECT_LE((lhs), (rhs))
#define CHECK_LT(lhs, rhs) EXPECT_LT((lhs), (rhs))
#define CHECK_NE(lhs, rhs) EXPECT_NE((lhs), (rhs))
#define CHECK_GE(lhs, rhs) EXPECT_GE((lhs), (rhs))
#define CHECK_GT(lhs, rhs) EXPECT_GT((lhs), (rhs))

#define NUMETRON_NO30 true
#define NUMETRON_TEST_COUNT 16// (16 * 256)//16//(48*5000) //16// (48*5000)
#define NUMETRON_USZCOND(sz) (((sz) & 3) == 2)
namespace numetron {

void basic_integer_test0();
void basic_decimal_test0();
void ct_test();

void mul_test();
void mpn_mul_test();

void test_float16_basic_operations();
void test_float16_special_values();
void test_float16_zeros();
void test_float16_small_values();
void test_float16_precision_limits();
void test_float16_arithmetic_properties();
void test_float16_negative_values();
void test_float16_integral_constructor();
void test_float16_static_values();
void test_float16_static_values_ordering();
void test_float16_static_values_bit_patterns();
void test_float16_integral_equality();
void test_float16_integral_equality_special_values();
void test_float16_integral_equality_precision();
void test_float16_floating_point_equality();
void test_float16_floating_point_equality_special_values();
void test_float16_floating_point_equality_precision_limits();
void test_float16_floating_point_mixed_precision();
void test_float16_double_constructor();
void test_float16_double_precision_conversion();
void test_float16_double_comparison();
void test_float16_integral_three_way_comparison();
void test_float16_integral_three_way_comparison_special_values();
void test_float16_integral_three_way_comparison_boundary_values();
void test_float16_integral_three_way_comparison_precision();
void test_float16_integral_three_way_comparison_derived_operators();
void test_float16_floating_point_three_way_comparison();
void test_float16_floating_point_three_way_comparison_special_values();
void test_float16_floating_point_three_way_comparison_precision();
void test_float16_floating_point_three_way_comparison_derived_operators();
}
