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

namespace numetron {

void basic_integer_test0();
void basic_decimal_test0();
void ct_test();

}
