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
}
AUTOTEST(numetron_test_registrar)
#else

TEST(NumetronTest, basic_integer) { basic_integer_test0(); }
TEST(NumetronTest, basic_decimal) { basic_decimal_test0(); }
TEST(NumetronTest, compile_time) { ct_test(); }

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif
