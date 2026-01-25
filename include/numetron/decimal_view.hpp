// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstdint>
#include <utility>
#include <concepts>
#include <algorithm>
#include <iterator>
#include <string>
#include <iosfwd>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "integer_view.hpp"
#include "basic_integer.hpp"
#include "float16.hpp"
#include "detail/hash.hpp"

#include "external/dragonbox.h"

namespace numetron {

template <std::unsigned_integral LimbT>
class basic_decimal_view
{
    basic_integer_view<LimbT> significand_;
    basic_integer_view<LimbT> exponent_;

public:
    using limb_type = LimbT;

    basic_decimal_view() noexcept = default;

    template <typename ST, typename ET>
    inline basic_decimal_view(ST && s, ET && e) noexcept
        : significand_{ std::forward<ST>(s) }, exponent_{ std::forward<ET>(e) }
    {}

    template <std::integral T>
    inline basic_decimal_view(T value) noexcept
    {
        uint8_t e = 0;
        while (!(value % 10)) {
            value /= 10; ++e;
        }
        significand_ = value;
        exponent_ = e;
    }

    template <std::floating_point T>
    explicit basic_decimal_view(T value)
    {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("floating-point value must be finite");
        }
        
        if (value == 0.0) {
            significand_ = 0;
            exponent_ = 0;
            return;
        }
        
        auto result = jkj::dragonbox::to_decimal(value, jkj::dragonbox::policy::trailing_zero::remove);
        
        auto sig = result.significand;
        int64_t exp = result.exponent;
        
        significand_ = sig;
        if (result.is_negative) {
            significand_ = -significand_;
        }
        exponent_ = exp;
    }

    explicit basic_decimal_view(float16 value);
    
    inline bool is_negative() const noexcept { return significand_.is_negative(); }
    inline int sgn() const noexcept { return significand_.sgn(); }

    template <std::integral T>
    explicit operator T() const noexcept
    {
        if (!exponent_) return (T)significand_;
        constexpr size_t bisz = (sizeof(T) + sizeof(LimbT) - 1) / sizeof(LimbT);
        return (T)(basic_integer<LimbT, bisz>)*this;
    }

    template <std::floating_point T>
    inline explicit operator T() const
    {
        return (T)((T)significand_ * std::pow(10.0, (int64_t)exponent_));
    }

    template <size_t N, typename AllocatorT>
    explicit operator basic_integer<LimbT, N, AllocatorT>() const
    {
        if (!exponent_.template is_fit<int>()) {
            if (exponent_.sgn() > 0) {
                throw std::invalid_argument("exponent is too large");
            } else {
                return basic_integer<LimbT, N, AllocatorT>{0};
            }
        }

        basic_integer<LimbT, N, AllocatorT> result{ significand_ };
        int intexp = (int)exponent_;
        if (intexp) {
            basic_integer<LimbT, N, AllocatorT> val10{ 10 };
            auto expm = pow(val10, (unsigned int)std::abs(intexp));

            if (intexp > 0) {
                result *= expm;
            } else {
                result /= expm;
            }
        }

        return result;
    }

    inline basic_integer_view<LimbT> const& significand() const noexcept { return significand_; }
    inline basic_integer_view<LimbT> const& exponent() const noexcept { return exponent_; }

    [[nodiscard]] inline basic_decimal_view operator- () const noexcept
    {
        return basic_decimal_view{ -significand_, exponent_ };
    }
};

using decimal_view = basic_decimal_view<uint64_t>;

template <typename T> struct is_basic_decimal_view : std::false_type {};
template <typename LimbT> struct is_basic_decimal_view<basic_decimal_view<LimbT>> : std::true_type {};
template <typename T> constexpr bool is_basic_decimal_view_v = is_basic_decimal_view<T>::value;


// expected normilized value
template <std::unsigned_integral LimbT>
bool operator== (basic_decimal_view<LimbT> const& lhs, basic_decimal_view<LimbT> const& rhs) noexcept
{
    return lhs.exponent() == rhs.exponent() && lhs.significand() == rhs.significand();
}

template <std::unsigned_integral LimbT, std::integral T>
bool operator ==(basic_decimal_view<LimbT> const& lhs, T rhs) noexcept
{
    if (!rhs) return !lhs;
    if (!lhs || lhs.exponent().is_negative()) return false;
    size_t exp = 0;
    for (;;) {
        auto [q, r] = numetron::arithmetic::div1(rhs, 10);
        if (r) break;
        rhs = q;
        ++exp;
    }
    return lhs.significand() == rhs && lhs.exponent() == exp;
}

template <std::unsigned_integral LimbT>
std::strong_ordering operator<=> (basic_decimal_view<LimbT> const& lhs, basic_decimal_view<LimbT> const& rhs)
{
    int lsgn = lhs.significand().sgn();
    if (lsgn < 0) {
        if (rhs.significand().sgn() >= 0) return std::strong_ordering::less;
    } else if (!lsgn) {
        int rsgn = rhs.significand().sgn();
        return !rsgn ? std::strong_ordering::equal : (rsgn < 0 ? std::strong_ordering::greater : std::strong_ordering::less);
    } else {
        if (rhs.significand().sgn() <= 0) return std::strong_ordering::greater;
    }

    auto r = basic_integer<LimbT, 1>{rhs.exponent() } - lhs.exponent(); // can throw bad_alloc
    constexpr size_t big_base_digits_per_limb = std::numeric_limits<LimbT>::digits10;
    constexpr LimbT big_base = numetron::arithmetic::ipow<LimbT>(10, big_base_digits_per_limb);
    if (!r) {
        return lhs.significand() <=> rhs.significand();
    }
    auto [lsa, rsa, less_res] = r.is_negative()
        ? std::tuple{ rhs.significand().abs(), lhs.significand().abs(), lsgn > 0 ? std::strong_ordering::greater : std::strong_ordering::less }
        : std::tuple{ lhs.significand().abs(), rhs.significand().abs(), lsgn < 0 ? std::strong_ordering::greater : std::strong_ordering::less };
    
    if (r.is_negative()) {
        r.negate();
    }

    basic_integer<LimbT, 2> operand; // for now div needs more space for result, so it's optimization for 1-limb values
    
    if (lsa < rsa) return less_res;
    operand = lsa;
    for (;;) {
        if (auto res = r <=> big_base_digits_per_limb; res == std::strong_ordering::less || res == std::strong_ordering::equal) {
            operand /= (res == std::strong_ordering::equal ? big_base : numetron::arithmetic::ipow<LimbT>(10, (size_t)r)); // can throw bad_alloc
            return operand >= rsa ? 0 <=> less_res : less_res;
        } else {
            operand /= big_base;
            r -= big_base_digits_per_limb;
            if (operand < rsa) return less_res;
        }
    }
}

#if 1
template <std::unsigned_integral LimbT>
basic_decimal_view<LimbT>::basic_decimal_view(float16 value)
{
    uint16_t bits = value.to_bits();

    bool is_negative = (bits & 0x8000) != 0;
    uint16_t exp_bits = (bits >> 10) & 0x1F;
    uint16_t mant_bits = bits & 0x3FF;

    if (exp_bits == 0x1F) {
        throw std::invalid_argument("floating-point value must be finite");
    }

    if (exp_bits == 0 && mant_bits == 0) {
        significand_ = 0;
        exponent_ = 0;
        return;
    }

    int binary_exp;
    uint32_t significand;

    if (exp_bits == 0) {
        // Subnormal
        significand = mant_bits;
        binary_exp = -24;
    } else {
        // Normal
        significand = 1024 + mant_bits;
        binary_exp = static_cast<int>(exp_bits) - 25;
    }

    // Remove trailing binary zeros
    while ((significand & 1) == 0) {
        significand >>= 1;
        ++binary_exp;
    }

    int64_t decimal_exp = 0;
    uint64_t sig = significand;

    if (binary_exp >= 0) {
        sig <<= binary_exp;
    } else {
        int neg_exp = -binary_exp;
        decimal_exp = binary_exp;

        // Compute 5^neg_exp (max neg_exp = 24, 5^24 fits in uint64_t)
        static constexpr uint64_t pow5_table[] = {
            1ULL, 5ULL, 25ULL, 125ULL, 625ULL, 3125ULL, 15625ULL, 78125ULL,
            390625ULL, 1953125ULL, 9765625ULL, 48828125ULL, 244140625ULL,
            1220703125ULL, 6103515625ULL, 30517578125ULL, 152587890625ULL,
            762939453125ULL, 3814697265625ULL, 19073486328125ULL,
            95367431640625ULL, 476837158203125ULL, 2384185791015625ULL,
            11920928955078125ULL, 59604644775390625ULL
        };
        sig *= pow5_table[neg_exp];
    }

    // Remove trailing decimal zeros
    while (sig && (sig % 10) == 0) {
        sig /= 10;
        ++decimal_exp;
    }

    significand_ = static_cast<LimbT>(sig);
    if (is_negative) {
        significand_ = -significand_;
    }
    exponent_ = decimal_exp;
}

#else

template <std::unsigned_integral LimbT>
basic_decimal_view<LimbT>::basic_decimal_view(float16 value)
{
    // Compute 5^neg_exp (max neg_exp = 24, 5^24 fits in uint64_t)
    static constexpr uint64_t pow5_table[] = {
        1ULL, 5ULL, 25ULL, 125ULL, 625ULL, 3125ULL, 15625ULL, 78125ULL,
        390625ULL, 1953125ULL, 9765625ULL, 48828125ULL, 244140625ULL,
        1220703125ULL, 6103515625ULL, 30517578125ULL, 152587890625ULL,
        762939453125ULL, 3814697265625ULL, 19073486328125ULL,
        95367431640625ULL, 476837158203125ULL, 2384185791015625ULL,
        11920928955078125ULL, 59604644775390625ULL
    };

    auto exact = [](float16 v) -> std::tuple<uint64_t, int64_t, bool> {
        uint16_t bits = v.to_bits();
        bool is_negative = (bits & 0x8000) != 0;
        uint16_t exp_bits = (bits >> 10) & 0x1F;
        
        if (exp_bits == 0x1F) {
            throw std::invalid_argument("floating-point value must be finite");
        }

        uint16_t mant_bits = bits & 0x3FF;
        
        if (exp_bits == 0 && mant_bits == 0) return { 0, 0, false };

        int binary_exp;
        uint32_t significand;

        if (exp_bits == 0) {
            // Subnormal
            significand = mant_bits;
            binary_exp = -24;
        } else {
            // Normal
            significand = 1024 + mant_bits;
            binary_exp = static_cast<int>(exp_bits) - 25;
        }

        // Remove trailing binary zeros
        while ((significand & 1) == 0) {
            significand >>= 1;
            ++binary_exp;
        }

        int64_t decimal_exp = 0;
        uint64_t sig = significand;

        if (binary_exp >= 0) {
            sig <<= binary_exp;
        } else {
            int neg_exp = -binary_exp;
            decimal_exp = binary_exp;
            sig *= pow5_table[neg_exp];
        }

        return { sig, decimal_exp, is_negative };
    };

    auto to_float = [](uint64_t s, int64_t e, bool is_negative) -> float {
        float val = static_cast<float>(s);
        if (e > 0) {
            for (int64_t i = 0; i < e; ++i) val *= 10.0f;
        } else {
            for (int64_t i = 0; i < -e; ++i) val /= 10.0f;
        }
        return is_negative ? -val : val;
    };

    auto [sig, decimal_exp, is_negative] = exact(value); // get exact decimal representation first
    float orig = to_float(sig, decimal_exp, is_negative);

    // Try to shorten by removing last digit
    while (sig >= 10) {
        uint64_t base = sig / 10;
        int remainder = sig % 10;
        int64_t new_exp = decimal_exp + 1;
        if (!remainder) {
            sig = base;
            decimal_exp = new_exp;
            continue;
        } else if (remainder >= 5) {
            float16 nup = is_negative ? value.next_down() : value.next_up();
            if (!nup.is_finit()) break;
            if (std::abs((float)nup - orig) < to_float(1, decimal_exp, false)) break;
            float rounded = to_float(base + 1, new_exp, is_negative);
            
            if (std::abs(rounded - orig) > std::abs(orig - (float)nup) / 2) break;
            sig = base + 1;
            decimal_exp = new_exp;
            continue;
        } else { // remainder < 5
            float16 ndown = is_negative ? value.next_up() : value.next_down();
            if (!ndown.is_finit()) break;
            if (std::abs((float)ndown - orig) < to_float(1, decimal_exp, false)) break;
            float rounded = to_float(base, new_exp, is_negative);
            if (std::abs(rounded - orig) > std::abs(orig - (float)ndown) / 2) break;
            sig = base;
            decimal_exp = new_exp;
            continue;
        }
    }

    significand_ = sig;
    if (is_negative) {
        significand_ = -significand_;
    }
    exponent_ = decimal_exp;
}

#endif

template <std::unsigned_integral LimbT>
inline std::string to_string(basic_decimal_view<LimbT> const& val)
{
    std::string result;
    bool reversed;

    int sgn = val.sgn();
    if (sgn < 0) result.push_back('-');
    size_t offset = result.size();

    val.significand().with_limbs([&result, &reversed](std::span<const LimbT> sp, int) { to_string(sp, std::back_inserter(result), reversed); });

    if (reversed) {
        std::reverse(result.begin() + offset, result.end());
    }
    int64_t e = (int64_t)val.exponent();
    if (e >= 0) {
        result.resize(result.size() + e, '0');
    } else {
        int pos = sgn < 0 ? 1 : 0;
        int64_t zpadcount = -e - (int64_t)result.size() + pos + 1;
        if (zpadcount > 0) {
            result.insert(result.begin() + pos, zpadcount, '0');
        }
        result.insert(result.begin() + result.size() + e, '.');
    }
    return result;
}

template <typename Elem, typename Traits, std::unsigned_integral LimbT>
inline std::basic_ostream<Elem, Traits>& operator <<(std::basic_ostream<Elem, Traits>& os, basic_decimal_view<LimbT> const& dv)
{
    return os << to_string(dv);
}

template <std::unsigned_integral LimbT>
inline size_t hash_value(basic_decimal_view<LimbT> const& v) noexcept
{
    return detail::hasher{}(v.significand(), v.exponent());
}

using decimal_view = basic_decimal_view<uint64_t>;

}
