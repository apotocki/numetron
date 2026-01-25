// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include "numetron/basic_decimal.hpp"
#include "numetron/basic_integer.hpp"
#include "numetron/float16.hpp"

#include <vector>
#include <iostream>
#include <cfloat>

namespace numetron {

void basic_decimal_test0()
{
    using dec_t = basic_decimal<uint64_t, 1, 8>;
    using mdec_t = basic_decimal<uint8_t, 1, 2>;
    using decv_t = basic_decimal_view<uint64_t>;

    decimal d = std::numeric_limits<int64_t>::min();
    decimal d2 = d + decimal{ 1 };
    CHECK_EQUAL((int64_t)d2, std::numeric_limits<int64_t>::min() + 1);
    d = d + decimal{ 1 };
    CHECK_EQUAL((int64_t)d, std::numeric_limits<int64_t>::min() + 1);

    using namespace std::string_view_literals;
    using namespace numetron::literals;

    CHECK_EQUAL(to_string(decv_t{ 42.00 }), "42");
    CHECK_EQUAL(to_string(decv_t{ -42.01 }), "-42.01");
    CHECK_EQUAL(to_string(dec_t{ 42.00 }), "42");
    CHECK_EQUAL(to_string(dec_t{ -42.01 }), "-42.01");
    CHECK_EQUAL(to_string(dec_t{ "42.00"sv }), "42");
    CHECK_EQUAL(to_string(dec_t{ "42.1"sv }), "42.1");
    CHECK_EQUAL(to_string(dec_t{ "42.001"sv }), "42.001");
    CHECK_EQUAL(to_string(dec_t{ "42.001000"sv }), "42.001");
    CHECK_EQUAL(to_string(dec_t{ ".0042"sv }), "0.0042");
    CHECK_EQUAL(to_string(dec_t{ "0.0042"sv }), "0.0042");
    CHECK_EQUAL(to_string(dec_t{ ".0042000"sv }), "0.0042");
    CHECK_EQUAL(to_string(dec_t{ "00.0042000"sv }), "0.0042");
    CHECK_EQUAL(to_string(dec_t{ ".0042000e-2"sv }), "0.000042");
    CHECK_EQUAL(to_string(dec_t{ ".0042000e4"sv }), "42");

    CHECK_EQUAL(dec_t{ ".022"sv }.exponent_as<int>(), -3);
    CHECK_EQUAL(dec_t{ "-.022"sv }.significand(), -22);
    CHECK_EQUAL(dec_t{ "-0.022"sv }.significand(), -22);

    CHECK(!dec_t{});
    CHECK(dec_t{ 1 });
    CHECK(dec_t{ 10000000000ll }.is_inplaced());
    CHECK_EQUAL(dec_t{ 10000000000ll }.significand(), 1);
    CHECK_EQUAL(dec_t{ 10000000000ll }.exponent_as<int>(), 10);

    CHECK(!dec_t{ "0"sv });
    CHECK_EQUAL(dec_t{ "-10000000000"sv }.significand(), -1);
    CHECK_EQUAL(dec_t{ "-10000000000"sv }.exponent_as<int>(), 10);
    CHECK_EQUAL(dec_t{ "0.1"sv }.exponent_as<int>(), -1);
    CHECK_EQUAL(dec_t{ ".1"sv }.exponent_as<int>(), -1);

    CHECK_EQUAL(dec_t{ "0.1"sv }.exponent_as<int>(), -1);
    CHECK_EQUAL(dec_t{ "0.1"sv }.significand(), 1);
    CHECK_EQUAL(dec_t{ "00.2"sv }.exponent_as<int>(), -1);
    CHECK_EQUAL(dec_t{ "00.2"sv }.significand(), 2);
    CHECK_EQUAL(dec_t{ ".3"sv }.exponent_as<int>(), -1);
    CHECK_EQUAL(dec_t{ ".3"sv }.significand(), 3);
   
    CHECK_EQUAL(dec_t{ "9.00000000990000000000001"sv }.significand(), 900000000990000000000001_bi);
    CHECK_EQUAL(mdec_t{ "9.00000000990000000000001"sv }.significand(), (basic_integer<uint8_t, 1>("900000000990000000000001"sv)));
    
    CHECK_EQUAL(dec_t{ "3.1e5"sv }.significand(), 31);
    CHECK_EQUAL(dec_t{ "3.1e5"sv }.exponent_as<int>(), 4);
    CHECK_EQUAL(to_string(dec_t{ "3.1e5"sv }), "310000");
    CHECK_EQUAL(to_string(dec_t{ "3.1e-5"sv }), "0.000031");
    
    CHECK_LE(std::abs((float)dec_t{ "3.1234e-5"sv } - 0.000031234f), FLT_EPSILON);
    CHECK_LE(std::abs((double)dec_t { "3.1e-5"sv } - 0.000031), DBL_EPSILON);
    
    CHECK_EQUAL(dec_t{ "31e5"sv }, dec_t{ "3.1e6"sv });
    CHECK_EQUAL(dec_t{ "-3.1e6"sv }, dec_t{ "-31e5"sv });
    CHECK_EQUAL(dec_t{ "31e5"sv }, 3100000);
    
    CHECK_LT(dec_t{ "3.11e5"sv }, dec_t{ "3.1e6"sv });
    CHECK_GT(dec_t{ "3.1e6"sv }, dec_t{ "3.11e5"sv });
    CHECK_GT(dec_t{ "3110000"sv }, dec_t{ "3.1e6"sv });
    CHECK_LT(dec_t{ "3.1e6"sv }, dec_t{ "3110000"sv });

    CHECK_GT(dec_t{ "3.11000000000000000000000000000001e6"sv }, dec_t{ "3110000"sv });
    CHECK_LT(dec_t{ "-3.11000000000000000000000000000001e6"sv }, dec_t{ "-3110000"sv });
    CHECK_LT(dec_t{ "3110000"sv }, dec_t{ "3.11000000000000000000000000000001e6"sv });
    CHECK_GT(dec_t{ "-3110000"sv }, dec_t{ "-3.11000000000000000000000000000001e6"sv });
    CHECK_LT(dec_t{ "3.10999999999999999999999999999999e6"sv }, dec_t{ "3110000"sv });
    CHECK_GT(dec_t{ "-3.10999999999999999999999999999999e6"sv }, dec_t{ "-3110000"sv });
    CHECK_GT(dec_t{ "3110000"sv }, dec_t{ "3.10999999999999999999999999999999e6"sv });
    CHECK_LT(dec_t{ "-3110000"sv }, dec_t{ "-3.10999999999999999999999999999999e6"sv });

    CHECK_EQUAL((size_t)dec_t{ "3e5"sv }, 300000);
    CHECK_EQUAL((int16_t)dec_t { "-2e2"sv }, -200);
    CHECK_EQUAL((int16_t)dec_t { "2.19399878273287837459238450239485023985748738458787"sv }, 2);

    
    //x0 /= 10000000000000000000ul;
    //std::cout << x0 << "\n";
    auto x1 = "2193998782732"_bi;
    
    //x0 /= 10000000000000000000ul;
    //x0 /= 10000000000000000000ul;
    //CHECK_EQUAL("219399878273287837459238450239485023985748738458787"_bi / "0xffffFFFFffffFFFE"_bi, "-0xffffFFFFffffFFFF"_bi);

    // ==================== float16 -> decimal_view tests ====================
    
    // Basic integer values
    CHECK_EQUAL(to_string(decv_t{ float16{0} }), "0");
    CHECK_EQUAL(to_string(decv_t{ float16{1} }), "1");
    CHECK_EQUAL(to_string(decv_t{ float16{-1} }), "-1");
    CHECK_EQUAL(to_string(decv_t{ float16{10} }), "10");
    CHECK_EQUAL(to_string(decv_t{ float16{100} }), "100");
    CHECK_EQUAL(to_string(decv_t{ float16{1000} }), "1000");
    CHECK_EQUAL(to_string(decv_t{ float16{-1000} }), "-1000");
    
    // Fractional values - float16 has limited precision (about 3-4 decimal digits)
    CHECK_EQUAL(to_string(decv_t{ float16{0.5f} }), "0.5");
    CHECK_EQUAL(to_string(decv_t{ float16{-0.5f} }), "-0.5");
    CHECK_EQUAL(to_string(decv_t{ float16{0.25f} }), "0.25");
    CHECK_EQUAL(to_string(decv_t{ float16{0.125f} }), "0.125");
    CHECK_EQUAL(to_string(decv_t{ float16{1.5f} }), "1.5");
    CHECK_EQUAL(to_string(decv_t{ float16{-1.5f} }), "-1.5");
    
    // Powers of 2 (exact in binary floating point)
    CHECK_EQUAL(to_string(decv_t{ float16{2.0f} }), "2");
    CHECK_EQUAL(to_string(decv_t{ float16{4.0f} }), "4");
    CHECK_EQUAL(to_string(decv_t{ float16{8.0f} }), "8");
    CHECK_EQUAL(to_string(decv_t{ float16{16.0f} }), "16");
    CHECK_EQUAL(to_string(decv_t{ float16{0.0625f} }), "0.0625"); // 1/16
    
    // Special float16 values - max finite value (65504)
    CHECK_EQUAL(to_string(decv_t{ (float16::max)() }), "65504");
    CHECK_EQUAL(to_string(decv_t{ float16::lowest() }), "-65504");
    
    // Small values near min normal
    auto min_normal = (float16::min)();
    auto min_normal_dv = decv_t{ min_normal };
    CHECK(min_normal_dv.significand() > 0); // should be positive
    CHECK(min_normal_dv.exponent() < 0);    // should have negative exponent
    
    // Subnormal (denormalized) values
    auto denorm_min = float16::denorm_min();
    auto denorm_dv = decv_t{ denorm_min };
    CHECK(denorm_dv.significand() > 0); // should be positive
    CHECK(denorm_dv.exponent() < 0);    // should have very negative exponent
    
    // Negative zero should equal positive zero
    CHECK_EQUAL(to_string(decv_t{ float16::zero() }), "0");
    CHECK_EQUAL(to_string(decv_t{ float16::negative_zero() }), "0");
    
    // Values that are exact in float16
    CHECK_EQUAL(to_string(decv_t{ float16{1024} }), "1024");
    CHECK_EQUAL(to_string(decv_t{ float16{2048} }), "2048");
    
    // Check significand and exponent directly for some values
    auto dv_half = decv_t{ float16{0.5f} };
    CHECK_EQUAL(dv_half.significand(), 5);
    CHECK_EQUAL(dv_half.exponent(), -1);
    
    auto dv_quarter = decv_t{ float16{0.25f} };
    CHECK_EQUAL(dv_quarter.significand(), 25);
    CHECK_EQUAL(dv_quarter.exponent(), -2);
    
    auto dv_ten = decv_t{ float16{10} };
    CHECK_EQUAL(dv_ten.significand(), 1);
    CHECK_EQUAL(dv_ten.exponent(), 1);
    
    auto dv_hundred = decv_t{ float16{100} };
    CHECK_EQUAL(dv_hundred.significand(), 1);
    CHECK_EQUAL(dv_hundred.exponent(), 2);

    // pretty-printing tests
    //CHECK_EQUAL(to_string(decv_t{ float16{42.1} }), "42.1");
    //CHECK_EQUAL(to_string(decv_t{ float16{0.1023} }), "0.1023");
    //CHECK_EQUAL(to_string(decv_t{ float16{0.1022} }), "0.1022");

    // Test that infinity/NaN throws
    EXPECT_THROW(decv_t{ float16::infinity() }, std::invalid_argument);
    EXPECT_THROW(decv_t{ float16::negative_infinity() }, std::invalid_argument);
    EXPECT_THROW(decv_t{ float16::quiet_NaN() }, std::invalid_argument);
    EXPECT_THROW(decv_t{ float16::signaling_NaN() }, std::invalid_argument);
}

}

