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

// Which "shortest round-trip" is meant for a float16/float32 source (see basic_decimal_view<LimbT>'s
// float16 and float constructors below): a narrower float format's own resolution can be coarse
// enough that a decimal shorter than the "obvious" one still validly round-trips back to the same
// value, e.g. "65500" for float16::max() (65504) -- both round-trip correctly, but "65500" has fewer
// significant digits (see BUGFIXES.md/RESOLVED.md). `widened` widens to `double` first and prints
// *its* shortest round-trip, whose much finer resolution near the same value essentially always
// reproduces the "obvious" decimal instead. `native` is the honest "fewest digits that round-trip
// through the source type's own resolution" variant.
//
// The right default differs by type, which is why each constructor picks its own rather than this
// enum picking one: for float16, `widened` is the default -- virtually no float16 library actually
// implements a native-resolution shortest-round-trip algorithm, they all widen first, so `widened`
// is what matches real-world expectations (e.g. seeing "65504" rather than "65500"). For float32,
// it's the reverse: Dragonbox itself operates directly on float32's own resolution, and that *is*
// what every mainstream float32 pretty-printer already does (there's no widen-first convention to
// match), so `native` is the default there -- and also the pre-existing behavior, so adding this
// parameter changes nothing for existing callers. float64 gets no such parameter at all: `double` is
// already the widest floating type this library deals with, so there's nothing to widen *to*.
enum class decimal_shortest_mode {
    native,
    widened,
};

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

    // Shared by the templated floating-point constructor below and by the float16/float
    // constructors' `decimal_shortest_mode::widened` path (which widens to double and delegates
    // here) -- factored out so every path goes through the exact same Dragonbox call rather than
    // duplicating it.
    template <std::floating_point T>
    inline void init_dragonbox_shortest(T value)
    {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("floating-point value must be finite");
        }

        if (value == T{0}) {
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

    template <std::floating_point T>
    explicit basic_decimal_view(T value)
    {
        init_dragonbox_shortest(value);
    }

    // Dedicated (non-template) overload for `float` specifically -- preferred by overload
    // resolution over the templated constructor above for an actual `float` argument (a non-template
    // function wins over an equally-good template specialization), so this is the one that actually
    // runs for `float`; the template above still handles `double` (and any other floating type this
    // library might gain). Defaults to `native`, matching both the pre-existing behavior (Dragonbox
    // on `float` directly) and what every mainstream float32 pretty-printer already does -- see
    // decimal_shortest_mode's own comment above for why float32's default is the opposite of
    // float16's.
    explicit basic_decimal_view(float value, decimal_shortest_mode mode = decimal_shortest_mode::native)
    {
        if (mode == decimal_shortest_mode::widened) {
            init_dragonbox_shortest(static_cast<double>(value));
        } else {
            init_dragonbox_shortest(value);
        }
    }

    explicit basic_decimal_view(float16 value, decimal_shortest_mode mode = decimal_shortest_mode::widened);
    
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

namespace detail {

// Computes the exact signed decimal value of a finite float16 as (significand, exponent), via
// bit-decomposition (sign/exponent/mantissa, strip trailing binary zeros, fold 2^binary_exp into
// decimal via 5^-binary_exp for a negative exponent). Returns bigint `basic_integer<LimbT>` rather
// than a native integral type -- deliberately: the exact significand can need more bits than a
// single LimbT (verified: float16's smallest-normal-exponent bucket with a fully-odd mantissa
// needs ~67 bits for LimbT = uint64_t, e.g. exp_bits=1, mant_bits=0x3FF), which a fixed-width
// uint64_t accumulator (an earlier version of this file's exact-value logic used one) silently
// overflows for exactly that class of input -- see BUGFIXES.md. A bigint has nowhere to overflow
// to. Shared by basic_decimal_view<LimbT>::basic_decimal_view(float16) below (the shortest-decimal
// search needs the exact values of `value` and its neighbors to compare against) and
// numetron::exact_decimal(float16) (basic_decimal.hpp), which just returns this directly.
template <std::unsigned_integral LimbT>
std::pair<basic_integer<LimbT>, int64_t> exact_signed_decimal_from_float16(float16 v)
{
    uint16_t bits = v.to_bits();
    bool negative = (bits & 0x8000) != 0;
    uint16_t exp_bits = (bits >> 10) & 0x1F;
    uint16_t mant_bits = bits & 0x3FF;
    if (exp_bits == 0x1F) {
        throw std::invalid_argument("floating-point value must be finite");
    }
    if (exp_bits == 0 && mant_bits == 0) {
        return { basic_integer<LimbT>{0}, 0 };
    }

    int binary_exp;
    uint32_t significand;
    if (exp_bits == 0) {
        significand = mant_bits; // subnormal
        binary_exp = -24;
    } else {
        significand = 1024 + mant_bits; // normal
        binary_exp = static_cast<int>(exp_bits) - 25;
    }
    while ((significand & 1) == 0) { significand >>= 1; ++binary_exp; }

    basic_integer<LimbT> sig{ significand };
    int64_t decimal_exp = 0;
    if (binary_exp >= 0) {
        sig <<= static_cast<unsigned int>(binary_exp);
    } else {
        decimal_exp = binary_exp;
        sig *= numetron::pow(basic_integer<LimbT>{5}, static_cast<unsigned int>(-binary_exp));
    }
    while (sig && !(sig % 10)) { sig /= 10; ++decimal_exp; }

    if (negative) sig = -sig;
    return { std::move(sig), decimal_exp };
}

// Binary exponent of `v`'s least-significant mantissa bit -- what the function above computes as
// `binary_exp` *before* its trailing-zero-stripping loop runs. Needed separately by
// basic_decimal_view<LimbT>::basic_decimal_view(float16) below to size the *analytic* neighbor at
// the extremal (max/lowest finite) values, where the real neighbor one step further from zero
// overflows to infinity rather than existing as a finite float16.
inline int raw_binary_exponent_of_float16(float16 v)
{
    uint16_t exp_bits = (v.to_bits() >> 10) & 0x1F;
    return exp_bits == 0 ? -24 : static_cast<int>(exp_bits) - 25;
}

} // namespace detail

// Constructs the *shortest* decimal that still round-trips back to `value` -- the same guarantee
// the templated (f32/f64) constructor above gets from Dragonbox. `mode` picks which "shortest" is
// meant (see decimal_shortest_mode above this class): `widened` (the default, matching what most
// other float16 tooling does) widens to double and delegates to the same Dragonbox call the
// templated constructor uses, via `init_dragonbox_shortest` -- double's resolution near any float16
// value is so much finer than float16's own that this always reproduces the value's "obvious"
// decimal, e.g. "65504" for float16::max(). `native` instead computes the true fewest-digit
// decimal that round-trips through float16's own (much coarser) resolution, which can legitimately
// be shorter yet still round-trip correctly, e.g. "65500" for that same value (see
// BUGFIXES.md/RESOLVED.md) -- computed by hand since Dragonbox itself only targets IEEE-754
// binary32/64, not this library's float16. (Either way, see numetron::exact_decimal(float16),
// basic_decimal.hpp, for the *exact*, non-shortened value, needed wherever exactness rather than
// brevity matters, e.g. numeric comparisons.)
//
// `native` algorithm: get the exact decimal value of `value` and of its immediate neighbors
// (next_down()/next_up(), whichever are finite), align them to a common decimal exponent, and
// search from the fewest significant digits upward for the first rounded candidate that still
// falls strictly within the interval bounded by the midpoints to those neighbors -- any decimal in
// that interval parses back to exactly `value`. A candidate landing exactly on a midpoint is
// accepted only when `value`'s own raw bit pattern is even, matching IEEE round-half-to-even
// (adjacent float16 bit patterns always alternate parity, so this is equivalent to, but cheaper
// than, comparing mantissas). Every comparison here is an exact bigint comparison -- no float
// re-parsing anywhere, unlike an earlier version of this constructor, which compared approximate
// `float` reconstructions and could not fully guarantee round-trip correctness at a tie.
template <std::unsigned_integral LimbT>
basic_decimal_view<LimbT>::basic_decimal_view(float16 value, decimal_shortest_mode mode)
{
    using integer_t = basic_integer<LimbT>;
    using view_t = basic_integer_view<LimbT>;

    if (!value.is_finit()) {
        throw std::invalid_argument("floating-point value must be finite");
    }
    if (value.to_bits() == 0 || value.to_bits() == 0x8000) { // +0 or -0
        significand_ = 0;
        exponent_ = 0;
        return;
    }
    if (mode == decimal_shortest_mode::widened) {
        init_dragonbox_shortest(static_cast<double>(value));
        return;
    }

    bool value_wins_ties = (value.to_bits() & 1) == 0;

    auto [v_sig, v_exp] = detail::exact_signed_decimal_from_float16<LimbT>(value);

    bool have_lo = value.next_down().is_finit();
    bool have_hi = value.next_up().is_finit();

    integer_t lo_sig{ 0 }; int64_t lo_exp = v_exp;
    integer_t hi_sig{ 0 }; int64_t hi_exp = v_exp;
    if (have_lo) std::tie(lo_sig, lo_exp) = detail::exact_signed_decimal_from_float16<LimbT>(value.next_down());
    if (have_hi) std::tie(hi_sig, hi_exp) = detail::exact_signed_decimal_from_float16<LimbT>(value.next_up());

    if (!have_lo || !have_hi) {
        // `value` is the largest-magnitude finite float16 of its sign -- the neighbor one step
        // further from zero doesn't exist (it overflows to +/-infinity), but the boundary that
        // decides what still round-trips to `value` is well-defined regardless: it's exactly one
        // ULP away, same as if that neighbor *were* representable (IEEE overflow-to-infinity is
        // itself defined by rounding against that same, merely non-representable, extrapolated
        // value). Compute it analytically rather than leaving that side of the interval
        // unconstrained -- an earlier version did exactly that, via `!have_lo ||`/`!have_hi ||`
        // unconditionally satisfying its own side of the interval check, which let the
        // shortest-digit search shorten without limit there: float16::max() (65504) came out as
        // "70000", which overflows back to infinity on parsing instead of round-tripping to
        // 65504 -- caught by CI, not local testing; see BUGFIXES.md.
        int raw_exp = detail::raw_binary_exponent_of_float16(value);
        integer_t delta_sig{ 1 };
        int64_t delta_exp = 0;
        if (raw_exp >= 0) {
            delta_sig <<= static_cast<unsigned int>(raw_exp);
        } else {
            delta_exp = raw_exp;
            delta_sig *= numetron::pow(integer_t{ 5 }, static_cast<unsigned int>(-raw_exp));
        }

        int64_t align_exp = (std::min)(v_exp, delta_exp);
        integer_t v_aligned = v_sig;
        if (v_exp > align_exp) v_aligned *= numetron::pow(integer_t{ 10 }, static_cast<uint64_t>(v_exp - align_exp));
        integer_t delta_aligned = delta_sig;
        if (delta_exp > align_exp) delta_aligned *= numetron::pow(integer_t{ 10 }, static_cast<uint64_t>(delta_exp - align_exp));

        if (!have_hi) { // value > 0: missing neighbor is further from zero, i.e. larger
            hi_sig = v_aligned + delta_aligned;
            hi_exp = align_exp;
            have_hi = true;
        } else { // value < 0: missing neighbor is further from zero, i.e. more negative
            lo_sig = v_aligned - delta_aligned;
            lo_exp = align_exp;
            have_lo = true;
        }
    }

    int64_t e = v_exp;
    if (have_lo) e = (std::min)(e, lo_exp);
    if (have_hi) e = (std::min)(e, hi_exp);

    auto align = [e](integer_t sig, int64_t exp) {
        if (exp > e) sig *= numetron::pow(integer_t{10}, static_cast<uint64_t>(exp - e));
        return sig;
    };

    integer_t V = align(v_sig, v_exp);
    integer_t twice_lower_mid = have_lo ? (integer_t)(V + align(lo_sig, lo_exp)) : integer_t{0};
    integer_t twice_upper_mid = have_hi ? (integer_t)(V + align(hi_sig, hi_exp)) : integer_t{0};

    bool neg_V = V.is_negative();

    // Shorten one decimal digit at a time, starting from the exact value (`cur_mag`/`e`, always a
    // valid answer -- it's the interval's own center) and stopping at the first digit removal that
    // no longer satisfies the interval, keeping the last one that did. Deliberately never divides
    // by anything wider than a single digit (`10`, always single-limb) -- an earlier version of
    // this loop rounded straight to an arbitrary target scale (`magV / 10^drop`), which for a
    // `drop` past ~19 is itself a multi-limb divisor and would have hit
    // `numetron::limb_arithmetic::udiv`'s own "not implemented" case (see `BUGFIXES.md`), the exact
    // failure mode this whole session's `to_fixed_string`/`divide_rounded` work went to some length
    // to avoid elsewhere. Scaling a candidate back up to exponent `e` for the interval comparison is
    // a multiply, not a divide, so it's unaffected by that limitation regardless of magnitude.
    integer_t cur_mag = neg_V ? -V : V; // basic_integer has no .abs() -- that's a basic_integer_view-only method
    int64_t cur_exp = e;
    for (;;) {
        integer_t next_mag = cur_mag / integer_t{10};
        if (!next_mag) break; // fewer than two digits left -- can't shorten further

        // Round this one digit off to nearest (ties broken arbitrarily -- half up -- since the
        // interval check below is what actually decides correctness; getting this internal
        // tie-break "wrong" only risks stopping one digit earlier than the true minimum, never
        // producing a wrong answer).
        if ((view_t)(cur_mag % integer_t{10}) >= view_t{5}) next_mag += 1;
        int64_t next_exp = cur_exp + 1;

        integer_t signed_candidate = neg_V ? -next_mag : next_mag;
        integer_t twice_candidate = signed_candidate * integer_t{2};
        if (next_exp > e) twice_candidate *= numetron::pow(integer_t{10}, static_cast<uint64_t>(next_exp - e));

        bool lower_ok = !have_lo || (view_t)twice_candidate > (view_t)twice_lower_mid ||
                        ((view_t)twice_candidate == (view_t)twice_lower_mid && value_wins_ties);
        bool upper_ok = !have_hi || (view_t)twice_candidate < (view_t)twice_upper_mid ||
                        ((view_t)twice_candidate == (view_t)twice_upper_mid && value_wins_ties);
        if (!(lower_ok && upper_ok)) break;

        cur_mag = std::move(next_mag);
        cur_exp = next_exp;
    }

    integer_t result_sig = neg_V ? -cur_mag : cur_mag;
    int64_t result_exp = cur_exp;

    // Strip any incidental trailing zeros the rounding above may have introduced (e.g. rounding
    // 95 to the nearest 10 gives 100) -- keeps the result normalized like every other producer in
    // this file. Only ever divides by 10, same single-limb-safe reasoning as the search above.
    while (result_sig && !(result_sig % 10)) { result_sig /= 10; ++result_exp; }

    // Assign via a plain scalar, not `(view_t)result_sig` -- `basic_integer::operator
    // basic_integer_view()` returns a view wrapping `result_sig`'s *own* inplace limb storage
    // (basic_integer.hpp's `significand()`), not a copy of it; `result_sig` is a local about to be
    // destroyed when this constructor returns, so a view referencing its storage would dangle the
    // moment it did (a real, Valgrind-caught bug the first version of this fix had). The shortest
    // round-tripping decimal for a float16 is always a handful of digits, comfortably within a
    // native `int64_t`, so extracting it as a scalar and letting `basic_integer_view`'s own
    // integral constructor copy *that* into its inplace storage (exactly how the exponent below,
    // and the original pre-rewrite version of this constructor, already did it) is both safe and
    // sufficient -- never a case of the value being too wide to fit.
    significand_ = static_cast<int64_t>(result_sig);
    exponent_ = result_exp;
}

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

// Tie-break convention for to_fixed_string below. This library's own enum -- numetron has no
// dependency on any host language, so it doesn't mirror any particular one's naming; a caller whose
// own language has a symbolic rounding-mode type (e.g. Annium's bootstrap.ann `rounding_mode`)
// converts at its own language boundary instead (Annium's own C++ code uses this enum directly,
// with no separate mirror of it on the Annium side -- see numeric_promotion.hpp's
// divide_decimal_rounded comment; only the runtime `.ann`/C++ boundary itself, which has no shared
// symbolic enum type to begin with, crosses as a bare integer ordinal that gets cast straight to
// this enum). Only half_even and half_up are implemented by either to_fixed_string overload below;
// every other value is declared for a stable signature but throws std::runtime_error if selected --
// same incremental-implementation approach limb_arithmetic::udiv's own "not implemented" case uses.
enum class decimal_round_mode
{
    half_even,
    half_up,
    half_down,
    up,
    down,
    ceiling,
    floor,
};

namespace detail {

// Formats a non-negative bigint magnitude `sig`, already scaled so it *is* the target value *
// 10^digits (rounded), into a "int.frac" (or bare integer at digits == 0) string -- left-padding
// the integer part with zeros so there's always at least one digit ahead of the decimal point
// (e.g. sig == 5 at digits == 2 must read "0.05", not split "5" into nothing). Shared by both
// to_fixed_string overloads below -- once a value's been rounded down to an exact bigint at the
// target scale, the string layout is identical regardless of which base the rounding itself used.
// `sig` is assumed non-negative (its sign is tracked separately via `negative`), so the existing
// friend `to_string(basic_integer const&, base, show_base)` -- which would prepend its own '-' for
// a negative operand -- never gets the chance to.
template <std::unsigned_integral LimbT, size_t N, typename AllocatorT>
std::string to_fixed_digits_string(basic_integer<LimbT, N, AllocatorT> const& sig, bool negative, int64_t digits)
{
    std::string digit_str = to_string(sig);

    std::string result;
    if (negative) result.push_back('-');

    if (digits == 0) {
        result += digit_str;
        return result;
    }

    if (digit_str.size() <= static_cast<size_t>(digits)) {
        digit_str.insert(0, static_cast<size_t>(digits) + 1 - digit_str.size(), '0');
    }
    size_t split = digit_str.size() - static_cast<size_t>(digits);
    result += digit_str.substr(0, split);
    result.push_back('.');
    result += digit_str.substr(split);
    return result;
}

} // namespace detail

// Formats `d` to exactly `digits` fractional digits (zero-padded, never trimmed), correctly rounded
// per `mode`, via exact bigint significand/exponent arithmetic (no double round-trip, so this stays
// exact even for a significand too wide to survive a double conversion intact). `digits` must be
// >= 0. Scales the significand to the target exponent `-digits`: an exact widening multiply if
// `exponent + digits >= 0`; otherwise a narrowing bigint division with a remainder tie-break per
// `mode` (`half_up`: `2r >= den` always rounds away from zero on a tie; `half_even`: `2r > den`
// rounds up, `2r == den` checks the quotient's own parity).
template <std::unsigned_integral LimbT>
std::string to_fixed_string(basic_decimal_view<LimbT> const& d, int64_t digits, decimal_round_mode mode)
{
    if (mode != decimal_round_mode::half_even && mode != decimal_round_mode::half_up) {
        throw std::runtime_error("to_fixed_string: only decimal_round_mode::half_even and ::half_up are implemented so far");
    }

    basic_integer<LimbT> sig{ d.significand().abs() };

    int64_t e = (int64_t)d.exponent() + digits;
    if (e >= 0) {
        sig *= numetron::pow(basic_integer<LimbT>{ 10 }, static_cast<uint64_t>(e));
    } else {
        basic_integer<LimbT> den = numetron::pow(basic_integer<LimbT>{ 10 }, static_cast<uint64_t>(-e));
        basic_integer<LimbT> q = sig / den;
        basic_integer<LimbT> r = sig % den;
        basic_integer<LimbT> twice_r = r * basic_integer<LimbT>{ 2 };
        basic_integer_view<LimbT> twice_r_v{ twice_r };
        basic_integer_view<LimbT> den_v{ den };
        if (mode == decimal_round_mode::half_up) {
            if (twice_r_v >= den_v) q += 1;
        } else if (twice_r_v > den_v) {
            q += 1;
        } else if (twice_r_v == den_v && (q % 2)) {
            q += 1; // exact tie, half_even: round to the even neighbor, and q is currently odd
        }
        sig = std::move(q);
    }

    return detail::to_fixed_digits_string(sig, d.is_negative(), digits);
}

// Rounds a finite floating-point value's *exact* binary value to `digits` fractional decimal
// digits, honoring `mode`, entirely in base 2 -- deliberately not via an exact base-10 decimal
// (this class's own floating-point constructor above uses Dragonbox, the *shortest*
// round-tripping decimal, not the exact value; and even a genuinely exact base-10 conversion,
// folding 2^binexp == 5^-binexp / 10^binexp the way the float16 constructor above does, keeps
// *binexp itself* as the resulting decimal's exponent -- rounding that down to a handful of digits
// then needs a base-10 divisor whose bit width is binexp's magnitude times log2(10), reliably
// wider than one limb for any ordinary double, which is exactly what limb_arithmetic::udiv's
// multi-limb-divisor case doesn't support). `value`'s bits are exactly int_mantissa * 2^binexp
// (frexp/ldexp are exact for finite floats), so value * 10^digits == int_mantissa * 5^digits *
// 2^(binexp + digits): when binexp + digits >= 0 that's an exact multiply (folded into a left
// shift); otherwise it's a right shift by -(binexp + digits) bits, with the exact remainder
// recovered by subtraction rather than a second division. Shifts operate directly on limbs (they
// never call udiv), so this works for any magnitude -- unlike the equivalent base-10 division,
// there's no "normal-range value only" caveat here.
template <std::unsigned_integral LimbT = uint64_t, std::floating_point T>
std::string to_fixed_string(T value, int64_t digits, decimal_round_mode mode)
{
    if (mode != decimal_round_mode::half_even && mode != decimal_round_mode::half_up) {
        throw std::runtime_error("to_fixed_string: only decimal_round_mode::half_even and ::half_up are implemented so far");
    }

    bool negative = std::signbit(value);
    if (value == T{ 0 }) {
        return detail::to_fixed_digits_string(basic_integer<LimbT>{ 0 }, negative, digits);
    }

    int exp2;
    T mantissa = std::frexp(value, &exp2);
    constexpr int mantissa_bits = std::numeric_limits<T>::digits;
    int64_t int_mantissa = static_cast<int64_t>(std::ldexp(std::fabs(mantissa), mantissa_bits));
    int64_t binexp = static_cast<int64_t>(exp2) - mantissa_bits;

    basic_integer<LimbT> sig{ int_mantissa };
    sig *= numetron::pow(basic_integer<LimbT>{ 5 }, static_cast<uint64_t>(digits));

    int64_t k = binexp + digits;
    if (k >= 0) {
        sig <<= static_cast<unsigned int>(k);
    } else {
        unsigned int shift = static_cast<unsigned int>(-k);
        basic_integer<LimbT> q = sig >> shift;
        basic_integer<LimbT> r = sig - (q << shift); // exact remainder -- subtraction, not division
        basic_integer<LimbT> den = basic_integer<LimbT>{ 1 } << shift; // == 2^shift
        basic_integer<LimbT> twice_r = r * basic_integer<LimbT>{ 2 };
        basic_integer_view<LimbT> twice_r_v{ twice_r };
        basic_integer_view<LimbT> den_v{ den };
        if (mode == decimal_round_mode::half_up) {
            if (twice_r_v >= den_v) q += 1;
        } else if (twice_r_v > den_v) {
            q += 1;
        } else if (twice_r_v == den_v && (q % 2)) {
            q += 1; // exact tie, half_even: round to the even neighbor, and q is currently odd
        }
        sig = std::move(q);
    }

    return detail::to_fixed_digits_string(sig, negative, digits);
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
