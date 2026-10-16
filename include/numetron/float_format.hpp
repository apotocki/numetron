// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <bit>
#include <concepts>

#include "float16.hpp"
#include "decimal_view.hpp"

// Text forms of IEEE 754 binary16 / binary32 / binary64 values that are the same on every platform (unlike printf's
// "%a" / "%g", whose output differs between C runtimes, and which have no float16 at all):
// - to_hex_float_string: the C99 "%a" form built from the bits;
// - to_general_string: "%g"-like - the shortest round-trip decimal (Dragonbox, basic_decimal_view) laid out plain for
//   a decimal exponent in [min_exp, max_exp], otherwise in scientific notation.
// Non-finite values are "inf", "-inf" and "NaN" in both (Rust's spelling).

namespace numetron {

namespace detail {

inline std::string non_finite_float_string(bool negative, bool nan)
{
    if (nan) return "NaN";
    return negative ? "-inf" : "inf";
}

// "%a" of a binary floating value given by its bits and its exponent / mantissa field widths: "0x1.<mantissa>p<exp>",
// the mantissa's hex digits without the trailing zeros (none - no point: "0x1p+3"); a subnormal "0x0.<mantissa>p<min exp>";
// a zero "0x0p+0". The digits are upper case if `uppercase`, "0x" and "p" are always lower case.
inline std::string hex_float_string_from_bits(uint64_t bits, int exp_bits, int mant_bits, bool uppercase)
{
    int const total = 1 + exp_bits + mant_bits;
    bool const negative = ((bits >> (total - 1)) & 1) != 0;
    uint64_t const exp_max = (uint64_t{ 1 } << exp_bits) - 1;
    uint64_t const exp = (bits >> mant_bits) & exp_max;
    uint64_t const mant = bits & ((uint64_t{ 1 } << mant_bits) - 1);
    if (exp == exp_max) return non_finite_float_string(negative, mant != 0);

    std::string result = negative ? "-" : "";
    if (!exp && !mant) return result + "0x0p+0";

    int const bias = (1 << (exp_bits - 1)) - 1;
    int const hex_digits = (mant_bits + 3) / 4;
    uint64_t m = mant << (4 * hex_digits - mant_bits);
    char const* const digits = uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
    std::string frac(static_cast<size_t>(hex_digits), '0');
    for (int i = hex_digits - 1; i >= 0; --i, m >>= 4) {
        frac[static_cast<size_t>(i)] = digits[m & 0xF];
    }
    while (!frac.empty() && frac.back() == '0') frac.pop_back();

    int64_t const e = exp ? static_cast<int64_t>(exp) - bias : 1 - bias;
    result += exp ? "0x1" : "0x0";
    if (!frac.empty()) {
        result.push_back('.');
        result += frac;
    }
    result.push_back('p');
    result.push_back(e < 0 ? '-' : '+');
    result += std::to_string(e < 0 ? -e : e);
    return result;
}

}

inline std::string to_hex_float_string(float16 value, bool uppercase = true)
{
    return detail::hex_float_string_from_bits(value.to_bits(), 5, 10, uppercase);
}

inline std::string to_hex_float_string(float value, bool uppercase = true)
{
    return detail::hex_float_string_from_bits(std::bit_cast<uint32_t>(value), 8, 23, uppercase);
}

inline std::string to_hex_float_string(double value, bool uppercase = true)
{
    return detail::hex_float_string_from_bits(std::bit_cast<uint64_t>(value), 11, 52, uppercase);
}

// A decimal laid out as "%g" does, but with all its digits (no precision cut): plain ("1500.0", "0.00015", an integral
// value keeps ".0") when the decimal exponent of its first significant digit is in [min_exp, max_exp], otherwise
// scientific ("1.5e+20", "1e-7"). Zero is "0.0".
template <std::unsigned_integral LimbT>
std::string to_general_string(basic_decimal_view<LimbT> const& value, int64_t min_exp = -5, int64_t max_exp = 15)
{
    std::string digits;
    bool reversed = false;
    value.significand().with_limbs([&digits, &reversed](std::span<const LimbT> sp, int) { to_string(sp, std::back_inserter(digits), reversed); });
    if (reversed) std::reverse(digits.begin(), digits.end());

    // value = digits * 10^e; the trailing zeros moved into the exponent
    int64_t e = static_cast<int64_t>(value.exponent());
    size_t begin = digits.find_first_not_of('0');
    if (begin == std::string::npos) return "0.0";
    digits.erase(0, begin);
    while (digits.size() > 1 && digits.back() == '0') {
        digits.pop_back();
        ++e;
    }

    std::string result = value.sgn() < 0 ? "-" : "";
    int64_t const n = static_cast<int64_t>(digits.size());
    int64_t const x = n - 1 + e; // digits[0].digits[1..] * 10^x
    if (x >= min_exp && x <= max_exp) {
        if (x < 0) {
            result += "0.";
            result.append(static_cast<size_t>(-x - 1), '0');
            result += digits;
        } else if (n <= x + 1) {
            result += digits;
            result.append(static_cast<size_t>(x + 1 - n), '0');
            result += ".0";
        } else {
            result.append(digits, 0, static_cast<size_t>(x + 1));
            result.push_back('.');
            result.append(digits, static_cast<size_t>(x + 1));
        }
        return result;
    }
    result.push_back(digits[0]);
    if (n > 1) {
        result.push_back('.');
        result.append(digits, 1);
    }
    result.push_back('e');
    result.push_back(x < 0 ? '-' : '+');
    result += std::to_string(x < 0 ? -x : x);
    return result;
}

// The shortest round-trip decimal of a float value as to_general_string lays it out; a negative zero is "-0.0".
// float16: `mode` as basic_decimal_view's (decimal_view.hpp): widened - the shortest of the value widened to double, i.e.
// its exact value ("65504", but 0.1 is "0.0999755859375"); native - the fewest digits reading back to the same float16
// ("0.1", but 65504 is "65500"). float: native resolution.
inline std::string to_general_string(float16 value, decimal_shortest_mode mode = decimal_shortest_mode::widened, int64_t min_exp = -5, int64_t max_exp = 15)
{
    uint16_t const bits = value.to_bits();
    bool const negative = (bits & 0x8000) != 0;
    if (!value.is_finit()) return detail::non_finite_float_string(negative, (bits & 0x3FF) != 0);
    if ((bits & 0x7FFF) == 0) return negative ? "-0.0" : "0.0";
    return to_general_string(decimal_view{ value, mode }, min_exp, max_exp);
}

template <std::floating_point T>
requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
inline std::string to_general_string(T value, int64_t min_exp = -5, int64_t max_exp = 15)
{
    if (!std::isfinite(value)) return detail::non_finite_float_string(std::signbit(value), std::isnan(value));
    if (value == T{ 0 }) return std::signbit(value) ? "-0.0" : "0.0";
    return to_general_string(decimal_view{ value }, min_exp, max_exp);
}

}
