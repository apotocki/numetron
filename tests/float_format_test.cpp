// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include "numetron/float_format.hpp"
#include "numetron/integer_view.hpp"

#include <bit>
#include <limits>
#include <sstream>
#include <string_view>

namespace numetron {

void test_hex_float_string()
{
    // double
    CHECK_EQUAL(to_hex_float_string(1.0), "0x1p+0");
    CHECK_EQUAL(to_hex_float_string(3.0), "0x1.8p+1");
    CHECK_EQUAL(to_hex_float_string(-0.5), "-0x1p-1");
    CHECK_EQUAL(to_hex_float_string(0.1), "0x1.999999999999Ap-4");
    CHECK_EQUAL(to_hex_float_string(0.1, false), "0x1.999999999999ap-4");
    CHECK_EQUAL(to_hex_float_string(0.0), "0x0p+0");
    CHECK_EQUAL(to_hex_float_string(-0.0), "-0x0p+0");
    CHECK_EQUAL(to_hex_float_string(std::bit_cast<double>(uint64_t{ 1 })), "0x0.0000000000001p-1022"); // the least subnormal
    CHECK_EQUAL(to_hex_float_string(std::numeric_limits<double>::infinity()), "inf");
    CHECK_EQUAL(to_hex_float_string(-std::numeric_limits<double>::infinity()), "-inf");
    CHECK_EQUAL(to_hex_float_string(std::numeric_limits<double>::quiet_NaN()), "NaN");

    // float
    CHECK_EQUAL(to_hex_float_string(1.0f), "0x1p+0");
    CHECK_EQUAL(to_hex_float_string(0.1f), "0x1.99999Ap-4");
    CHECK_EQUAL(to_hex_float_string(std::bit_cast<float>(uint32_t{ 1 })), "0x0.000002p-126");

    // float16
    CHECK_EQUAL(to_hex_float_string(float16::from_bits(0x3C00)), "0x1p+0");
    CHECK_EQUAL(to_hex_float_string(float16::from_bits(0x7BFF)), "0x1.FFCp+15"); // max, 65504
    CHECK_EQUAL(to_hex_float_string(float16::from_bits(0x0001)), "0x0.004p-14"); // the least subnormal
    CHECK_EQUAL(to_hex_float_string(float16::from_bits(0xFC00)), "-inf");
    CHECK_EQUAL(to_hex_float_string(float16::from_bits(0x7E00)), "NaN");
}

void test_general_string()
{
    // double: plain for a decimal exponent in [-5, 15]
    CHECK_EQUAL(to_general_string(1.0), "1.0");
    CHECK_EQUAL(to_general_string(0.1), "0.1");
    CHECK_EQUAL(to_general_string(-2.5), "-2.5");
    CHECK_EQUAL(to_general_string(123.456), "123.456");
    CHECK_EQUAL(to_general_string(1500.0), "1500.0");
    CHECK_EQUAL(to_general_string(0.00015), "0.00015");
    CHECK_EQUAL(to_general_string(0.000015), "0.000015");
    CHECK_EQUAL(to_general_string(1e15), "1000000000000000.0");
    // scientific outside it
    CHECK_EQUAL(to_general_string(1e16), "1e+16");
    CHECK_EQUAL(to_general_string(1.5e20), "1.5e+20");
    CHECK_EQUAL(to_general_string(1e-7), "1e-7");
    CHECK_EQUAL(to_general_string(-1.25e-300), "-1.25e-300");
    // zeros and non-finite
    CHECK_EQUAL(to_general_string(0.0), "0.0");
    CHECK_EQUAL(to_general_string(-0.0), "-0.0");
    CHECK_EQUAL(to_general_string(std::numeric_limits<double>::infinity()), "inf");
    CHECK_EQUAL(to_general_string(std::numeric_limits<double>::quiet_NaN()), "NaN");
    // a custom range
    CHECK_EQUAL(to_general_string(1500.0, -5, 2), "1.5e+3");

    // float: its own resolution
    CHECK_EQUAL(to_general_string(0.1f), "0.1");
    CHECK_EQUAL(to_general_string(16777216.0f), "16777216.0");
    CHECK_EQUAL(to_general_string(-0.0f), "-0.0");

    // float16
    CHECK_EQUAL(to_general_string(float16::from_bits(0x3C00)), "1.0");
    CHECK_EQUAL(to_general_string(float16::from_bits(0x7BFF)), "65504.0");
    CHECK_EQUAL(to_general_string(float16::from_bits(0x7BFF), decimal_shortest_mode::native), "65500.0");
    CHECK_EQUAL(to_general_string(float16::from_bits(0x2E66)), "0.0999755859375"); // 0.1 rounded to float16
    CHECK_EQUAL(to_general_string(float16::from_bits(0x2E66), decimal_shortest_mode::native), "0.1");
    CHECK_EQUAL(to_general_string(float16::from_bits(0x8000)), "-0.0");
    CHECK_EQUAL(to_general_string(float16::from_bits(0x7C00)), "inf");
}

void test_fancy_print_min_digits()
{
    auto print = [](uint64_t v, unsigned int base, uint8_t group_sz, size_t min_digits) {
        std::ostringstream os;
        fancy_print(os, integer_view{ v }, base, group_sz, std::string_view{ " " }, base == 16, min_digits);
        return os.str();
    };
    CHECK_EQUAL(print(1, 16, 4, 8), "0x0000 0001");
    CHECK_EQUAL(print(0xDBFF0000010001ull, 16, 4, 16), "0x00db ff00 0001 0001");
    CHECK_EQUAL(print(1234567, 10, 3, 0), "1 234 567");
    CHECK_EQUAL(print(12345, 16, 4, 2), "0x3039"); // longer than min_digits: unchanged
}

}
