// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include "numetron/basic_decimal.hpp"
#include "numetron/basic_integer.hpp"

#include <vector>
#include <iostream>
#include <cfloat>

namespace numetron {

void basic_decimal_test0()
{
    using dec_t = basic_decimal<uint64_t, 1, 8>;
    using mdec_t = basic_decimal<uint8_t, 1, 2>;

    //mdec_t d0{ "9.00000000990000000000001"sv };

    //std::cout << mdec_t::storage_type::sig_limb_count_bits << "\n";
    //using biv_t = basic_integer_view<uint64_t>;
    using namespace std::string_view_literals;
    using namespace numetron::literals;

    //auto vvv = -.022e-12;

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
}

}

