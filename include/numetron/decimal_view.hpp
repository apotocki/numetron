// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <bit>
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

    // Number of significant bits in |v| (0 for v == 0). Used by the floating-point conversion
    // operator below to size/align its exact division instead of narrowing a wide significand into
    // T before the base-10 scaling is even applied (see that operator's own comment).
    static inline int64_t bit_length(basic_integer_view<LimbT> v) noexcept
    {
        auto [limbs, mask, sign] = v.decompose();
        if (limbs.empty()) return 0;
        LimbT top = limbs.back() & mask;
        if (!top) return 0;
        return static_cast<int64_t>(limbs.size() - 1) * std::numeric_limits<LimbT>::digits + std::bit_width(top);
    }

    // Largest n such that 5^n fits in a native (single-limb-safe) divisor for this LimbT.
    static constexpr int max_pow5_step = []() constexpr {
        uint64_t v = 1;
        uint64_t limit;
        // `if constexpr`, not a runtime `?:` -- with a `?:` both branches are compiled for every
        // LimbT (only which one *runs* depends on the condition), so for LimbT=uint64_t the
        // untaken branch's `1 << digits` (`1 << 64`) is still a real, invalid shift-by-full-width
        // expression the compiler warns about (C4293) even though it's dead. `if constexpr`
        // discards the untaken branch entirely for each instantiation, so it's never compiled.
        if constexpr (sizeof(LimbT) >= sizeof(uint64_t)) {
            limit = (std::numeric_limits<uint64_t>::max)();
        } else {
            limit = (uint64_t{ 1 } << std::numeric_limits<LimbT>::digits) - 1;
        }
        int steps = 0;
        while (v <= limit / 5) { v *= 5; ++steps; }
        return steps;
    }();

    // floor(A / 5^k), exact, without ever dividing by a divisor wider than one limb: numetron's
    // general bigint division only implements a single-limb divisor (limb_arithmetic::udiv throws
    // "not implemented" otherwise -- see BUGFIXES.md), and 5^k is itself multi-limb for any k
    // beyond max_pow5_step (27 for a 64-bit LimbT). Chunks the division into steps of at most
    // max_pow5_step factors of 5 each (always native/single-limb divisors, using the working fast
    // path), relying on floor(floor(a/d1)/d2) == floor(a/(d1*d2)) for positive integers -- exact,
    // not an approximation. The exact remainder relative to the *full* 5^k is not tracked through
    // these chunked steps; the caller recovers it separately in one shot via multiplication and
    // subtraction (both fully supported for any size), which is simpler than accumulating it here.
    static basic_integer<LimbT> floor_div_pow5(basic_integer<LimbT> const& A, uint64_t k)
    {
        basic_integer<LimbT> q = A;
        while (k > 0) {
            uint64_t step = (std::min)(k, static_cast<uint64_t>(max_pow5_step));
            uint64_t chunk = 1;
            for (uint64_t i = 0; i < step; ++i) chunk *= 5;
            q = q / chunk; // native divisor, single-limb -- the working fast path
            k -= step;
        }
        return q;
    }

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
        if (value) {
            while (!(value % 10)) {
                value /= 10; ++e;
            }
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

    // Correctly-rounded (round-half-to-even) conversion to T, computed with exact bigint
    // arithmetic throughout -- not `(T)significand_ * pow(10, exponent)`, which narrows the
    // (possibly much wider than T's mantissa) significand into T *before* the base-10 scaling is
    // even applied. That's only accidentally exact when the significand happens to need no more
    // significant bits than T's mantissa provides (e.g. short literals like 1.5, or values like
    // FLT_MAX/DBL_MAX whose significant bits happen to fit) -- for a fully-precise value (e.g. the
    // correctly-rounded nearest f32/f64 to pi or e), the significand genuinely needs more bits than
    // T holds, and narrowing it first loses precision the later multiply can't recover.
    //
    // The algorithm splits |significand * 10^exponent| = |significand| * 5^exponent * 2^exponent:
    // the 2^exponent factor is folded straight into the final binary exponent (std::ldexp handles
    // that part exactly, no bigint division involved), leaving only a division by 5^|exponent| when
    // exponent < 0 -- the one part that can require actual bigint division, done via
    // floor_div_pow5's single-limb-safe chunking (see its own comment for why: general multi-limb
    // bigint division isn't implemented in this library). `p` (an extra binary left-shift folded
    // into that division) is chosen so the quotient `q` lands on exactly numeric_limits<T>::digits
    // significant bits (T's mantissa width including the implicit leading bit) -- q is then the
    // correctly-rounded mantissa, combined with the matching binary exponent via std::ldexp
    // (mant * 2^exp), itself exact for a mantissa that already fits T.
    //
    // `p` is capped so the result's exponent never drops below T's smallest *normal* exponent --
    // letting q have fewer significant bits instead once capped, exactly mirroring how IEEE-754
    // subnormals trade mantissa bits for exponent range. This keeps rounding a single pass at the
    // correct final precision; letting ldexp re-round an over-precise q a second time in the
    // subnormal range would risk a classic double-rounding error. Magnitudes beyond T's range still
    // work with no special-casing: std::ldexp itself correctly overflows to +-infinity or underflows
    // to +-0 when the final exponent falls out of range either way.
    template <std::floating_point T>
    inline explicit operator T() const
    {
        if (!significand_) return T{0};

        bool neg = significand_.is_negative();
        basic_integer<LimbT> S{ significand_.abs() };
        int64_t e10 = static_cast<int64_t>(exponent_);

        basic_integer<LimbT> num;
        basic_integer<LimbT> den5{ 1 };
        uint64_t k = 0;
        if (e10 >= 0) {
            num = S * numetron::pow(basic_integer<LimbT>{5}, static_cast<unsigned int>(e10));
        } else {
            num = S;
            k = static_cast<uint64_t>(-e10);
            den5 = numetron::pow(basic_integer<LimbT>{5}, static_cast<unsigned int>(k));
        }

        constexpr int bits = std::numeric_limits<T>::digits;
        constexpr int64_t min_normal_exp = static_cast<int64_t>(std::numeric_limits<T>::min_exponent) - 1;

        int64_t num_bits = bit_length(basic_integer_view<LimbT>{ num });
        int64_t den5_bits = bit_length(basic_integer_view<LimbT>{ den5 });

        int64_t natural_p = static_cast<int64_t>(bits) - (num_bits - den5_bits);
        int64_t max_p = e10 - min_normal_exp; // largest p keeping the final exponent (e10 - p) >= min_normal_exp
        int64_t p = (std::min)(natural_p, max_p);

        // q = floor(num * 2^p / (5^k)), and its exact remainder r against that same divisor --
        // computed without ever dividing by a multi-limb value (see floor_div_pow5): p >= 0 folds
        // into the dividend (a left-shift, always exact); p < 0 instead divides by 5^k first and
        // then peels the extra 2^(-p) off the *quotient* via an exact right-shift, recombining the
        // two remainders into the one exact remainder against the full divisor 5^k * 2^(-p).
        basic_integer<LimbT> q, r;
        auto compute_at = [&](int64_t pp) {
            if (pp >= 0) {
                basic_integer<LimbT> dividend = num << static_cast<unsigned int>(pp);
                q = floor_div_pow5(dividend, k);
                r = dividend - q * den5;
            } else {
                uint64_t extra2 = static_cast<uint64_t>(-pp);
                basic_integer<LimbT> q5 = floor_div_pow5(num, k);
                basic_integer<LimbT> r5 = num - q5 * den5;
                q = q5 >> static_cast<unsigned int>(extra2);
                basic_integer<LimbT> shifted_out = q5 - (q << static_cast<unsigned int>(extra2));
                r = shifted_out * den5 + r5;
            }
        };
        auto divisor_at = [&](int64_t pp) -> basic_integer<LimbT> {
            return pp >= 0 ? den5 : (den5 << static_cast<unsigned int>(-pp));
        };

        compute_at(p);
        int64_t q_bits = bit_length(basic_integer_view<LimbT>{ q });
        // Nudge p so q lands on exactly `bits` significant bits -- the bit-length-difference
        // estimate above can be off by one -- unless capped by max_p (subnormal range), where fewer
        // bits is correct and expected. Bounded: the estimate is never off by more than one or two
        // steps, so this is not an unbounded search.
        for (int guard = 0; q_bits > bits && guard < 4; ++guard) {
            --p;
            compute_at(p);
            q_bits = bit_length(basic_integer_view<LimbT>{ q });
        }
        for (int guard = 0; q_bits < bits && p < max_p && guard < 4; ++guard) {
            ++p;
            compute_at(p);
            q_bits = bit_length(basic_integer_view<LimbT>{ q });
        }

        // Round half-to-even on the truncated quotient using its own exact remainder, mirroring
        // divide_decimal_rounded's (numeric_promotion.cpp) round-half-even pattern -- base-2
        // alignment here instead of that function's base-10 scale. A rollover from e.g. 2^bits-1 to
        // 2^bits needs no special handling: a pure power of two is always exactly representable in T
        // (needs only the implicit leading bit), so std::ldexp below stays exact either way.
        basic_integer<LimbT> divisor = divisor_at(p);
        basic_integer<LimbT> twice_r = r * basic_integer<LimbT>{2};
        basic_integer_view<LimbT> twice_r_v{ twice_r };
        basic_integer_view<LimbT> divisor_v{ divisor };
        if (twice_r_v > divisor_v || (twice_r_v == divisor_v && (q % 2))) {
            q += 1;
        }

        uint64_t mant = static_cast<uint64_t>(basic_integer_view<LimbT>{ q }); // q has <= bits+1 (<=54) significant bits, always fits
        T result = std::ldexp(static_cast<T>(mant), static_cast<int>(e10 - p));
        return neg ? -result : result;
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


template <std::unsigned_integral LimbT>
std::strong_ordering operator<=> (basic_decimal_view<LimbT> const& lhs, basic_decimal_view<LimbT> const& rhs);

template <std::unsigned_integral LimbT>
bool operator== (basic_decimal_view<LimbT> const& lhs, basic_decimal_view<LimbT> const& rhs) noexcept
{
    // decimal_view is assumed always normalized (significand stripped of trailing decimal zeros) --
    // every real producer of one (basic_decimal_view's own integral/floating constructors, from_blob<
    // basic_decimal_view<LimbT>>, sonia-prime's invocation.hpp) upholds that invariant, normalizing even
    // a bigint-sourced view rather than handing back an un-stripped (significand, exponent) pair -- so two
    // views representing the same value always agree on both fields directly, with no need to align
    // differing exponents the way operator<=> does for ordering. Comparing them directly here is both
    // simpler and cheaper (no risk of the bad_alloc operator<=> can throw while aligning differently-scaled
    // significands) than routing equality through operator<=>.
    return lhs.significand() == rhs.significand() && lhs.exponent() == rhs.exponent();
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

    auto r = basic_integer<LimbT, 1>{ rhs.exponent() } - lhs.exponent(); // can throw bad_alloc
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
    // Tracks whether every chunked division performed below has had a zero remainder. Scaling `operand`
    // down by 10^r in chunks is only an exact match for `rsa` (rather than merely landing on the same
    // floor()'d quotient) if *no* digit was ever discarded along the way -- a single nonzero remainder,
    // anywhere in the chain, proves operand isn't a clean multiple of 10^r and the true (unrounded)
    // comparison must be strictly greater than rsa*10^r, never equal to it.
    bool exact = true;
    for (;;) {
        if (auto res = r <=> big_base_digits_per_limb; res == std::strong_ordering::less || res == std::strong_ordering::equal) {
            LimbT divisor = (res == std::strong_ordering::equal ? big_base : numetron::arithmetic::ipow<LimbT>(10, (size_t)r));
            if (operand % divisor) exact = false; // can throw bad_alloc
            operand /= divisor; // can throw bad_alloc
            if (operand == rsa) {
                // Equal quotients alone aren't enough: if any division along the way (this one included)
                // dropped a nonzero remainder, `operand` is a floor()'d approximation that merely landed
                // on rsa by coincidence -- the true value was strictly greater before rounding down.
                return exact ? std::strong_ordering::equal : (0 <=> less_res);
            }
            return operand > rsa ? (0 <=> less_res) : less_res;
        } else {
            if (operand % big_base) exact = false; // can throw bad_alloc
            operand /= big_base; // can throw bad_alloc
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
