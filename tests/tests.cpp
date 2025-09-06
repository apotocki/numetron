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