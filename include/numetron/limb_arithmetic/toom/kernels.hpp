// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <tuple>
#include <limits>
#include <cstddef>
#include <concepts>
#include <algorithm>

#include "numetron/detail/assert.hpp"
#include "numetron/limb_arithmetic/uadd.hpp"
#include "numetron/limb_arithmetic/usub.hpp"

// Limb-vector kernels shared by the Toom implementations (the hand-written ones and the
// plan-driven engine), so both run on exactly the same primitives. Unless noted, r may coincide
// with an input.

namespace numetron::limb_arithmetic::detail::toom_kernels {

template <std::unsigned_integral LimbT>
inline LimbT add_n(LimbT* r, LimbT const* a, LimbT const* b, size_t n) noexcept
{
    LimbT const* ap = a;
    LimbT* rp = r;
    return uadd_partial_unchecked(ap, b, b + n, rp);
}

template <std::unsigned_integral LimbT>
inline LimbT sub_n(LimbT* r, LimbT const* a, LimbT const* b, size_t n) noexcept
{
    LimbT const* ap = a;
    LimbT* rp = r;
    return usub_partial_unchecked(ap, b, b + n, rp);
}

// r[0..n) = a[0..n), a != r. The callers here copy the untouched top of an operand after a carry
// or borrow died out, which is nearly always 0-3 limbs: those are stored directly, since
// std::copy turns into an out-of-line memmove call that costs more than the copy itself.
template <std::unsigned_integral LimbT>
inline void copy_tail(LimbT* r, LimbT const* a, size_t n) noexcept
{
    switch (n) {
    case 3: r[2] = a[2]; [[fallthrough]];
    case 2: r[1] = a[1]; [[fallthrough]];
    case 1: r[0] = a[0]; [[fallthrough]];
    case 0: return;
    default: std::copy(a, a + n, r);
    }
}

// r[0..n) = a[0..n) + c, returns the carry out.
template <std::unsigned_integral LimbT>
inline LimbT add_1(LimbT* r, LimbT const* a, size_t n, LimbT c) noexcept
{
    size_t i = 0;
    for (; c && i < n; ++i) {
        std::tie(c, r[i]) = arithmetic::uadd1(a[i], c);
    }
    if (r != a) copy_tail(r + i, a + i, n - i);
    return c;
}

// r[0..n) = a[0..n) - c, returns the borrow out.
template <std::unsigned_integral LimbT>
inline LimbT sub_1(LimbT* r, LimbT const* a, size_t n, LimbT c) noexcept
{
    size_t i = 0;
    for (; c && i < n; ++i) {
        std::tie(c, r[i]) = arithmetic::usub1(a[i], c);
    }
    if (r != a) copy_tail(r + i, a + i, n - i);
    return c;
}

template <std::unsigned_integral LimbT>
inline int cmp_n(LimbT const* a, LimbT const* b, size_t n) noexcept
{
    while (n--) {
        if (a[n] != b[n]) return a[n] < b[n] ? -1 : 1;
    }
    return 0;
}

// (l << K) | (prev >> (limb bits - K)), 0 < K < limb bits: limb l of a source shifted left by K,
// prev being the source limb below it. A double-width shift: MSVC is given shld through the
// intrinsic (its own code for the expression -- shl, shr, or and a copy -- measured slower);
// GCC does better with the plain expression than with an __int128 shift, so it gets that.
template <std::unsigned_integral LimbT, unsigned K>
NUMETRON_FORCEINLINE LimbT shl_pair(LimbT prev, LimbT l) noexcept
{
    constexpr unsigned bits = std::numeric_limits<LimbT>::digits;
    static_assert(K > 0 && K < bits);
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
    if constexpr (bits == 64) return __shiftleft128(prev, l, static_cast<unsigned char>(K));
    else
#endif
    return static_cast<LimbT>((l << K) | (prev >> (bits - K)));
}

// (lo >> K) | (hi << (limb bits - K)), 0 < K < limb bits: shrd, see shl_pair().
template <std::unsigned_integral LimbT, unsigned K>
NUMETRON_FORCEINLINE LimbT shr_pair(LimbT lo, LimbT hi) noexcept
{
    constexpr unsigned bits = std::numeric_limits<LimbT>::digits;
    static_assert(K > 0 && K < bits);
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
    if constexpr (bits == 64) return __shiftright128(lo, hi, static_cast<unsigned char>(K));
    else
#endif
    return static_cast<LimbT>((lo >> K) | (hi << (bits - K)));
}

// r[0..n) = a[0..n) << 1, returns the bit shifted out at the top. n > 0.
template <std::unsigned_integral LimbT>
inline LimbT lshift1(LimbT* r, LimbT const* a, size_t n) noexcept
{
    constexpr int top = std::numeric_limits<LimbT>::digits - 1;
    const LimbT out = a[n - 1] >> top;
    for (size_t i = n - 1; i > 0; --i) { // high to low, so r == a works
        r[i] = shl_pair<LimbT, 1>(a[i - 1], a[i]);
    }
    r[0] = a[0] << 1;
    return out;
}

// r[0..n) = a[0..n) >> 1, returns the bit shifted out at the bottom. n > 0.
template <std::unsigned_integral LimbT>
inline LimbT rshift1(LimbT* r, LimbT const* a, size_t n) noexcept
{
    const LimbT out = a[0] & 1;
    for (size_t i = 0; i + 1 < n; ++i) { // low to high, so r == a works
        r[i] = shr_pair<LimbT, 1>(a[i], a[i + 1]);
    }
    r[n - 1] = a[n - 1] >> 1;
    return out;
}

// r[0..n) = a[0..n) << k (0 < k < limb bits), returns the bits shifted out at the top. n > 0.
template <std::unsigned_integral LimbT>
inline LimbT lshift(LimbT* r, LimbT const* a, size_t n, unsigned k) noexcept
{
    const unsigned rk = std::numeric_limits<LimbT>::digits - k;
    const LimbT out = a[n - 1] >> rk;
    for (size_t i = n - 1; i > 0; --i) { // high to low, so r == a works
        r[i] = (a[i] << k) | (a[i - 1] >> rk);
    }
    r[0] = a[0] << k;
    return out;
}

// r[0..n) = a[0..n) >> k (0 < k < limb bits), returns the k bits shifted out at the bottom. n > 0.
template <std::unsigned_integral LimbT>
inline LimbT rshift(LimbT* r, LimbT const* a, size_t n, unsigned k) noexcept
{
    const unsigned rk = std::numeric_limits<LimbT>::digits - k;
    const LimbT out = a[0] & ((LimbT{ 1 } << k) - 1);
    for (size_t i = 0; i + 1 < n; ++i) { // low to high, so r == a works
        r[i] = (a[i] >> k) | (a[i + 1] << rk);
    }
    r[n - 1] = a[n - 1] >> k;
    return out;
}

// a - b - br with the borrow in/out in br.
template <std::unsigned_integral LimbT>
inline LimbT sbb1(LimbT a, LimbT b, unsigned char& br) noexcept
{
#if defined(_M_X64) || defined(__x86_64__)
    if constexpr (sizeof(LimbT) == 8) {
        unsigned long long d;
        br = _subborrow_u64(br, a, b, &d);
        return d;
    } else
#endif
    {
        auto [nb, d] = arithmetic::usub1c(a, b, LimbT{ br });
        br = static_cast<unsigned char>(nb);
        return d;
    }
}

// r[0..n) = a[0..n) + (b[0..n) << k), 0 < k < limb bits; hi <- the bits of b shifted out at the
// top; returns the carry. One streaming pass: b[i] is read before r[i] is written and its high
// bits are carried to the next limb in a register, so r may be a or b.
template <std::unsigned_integral LimbT>
inline LimbT addlsh_n(LimbT* r, LimbT const* a, LimbT const* b, size_t n, unsigned k, LimbT& hi) noexcept
{
    const unsigned rk = std::numeric_limits<LimbT>::digits - k;
    LimbT prev = 0;
    unsigned char c = 0;
    for (size_t i = 0; i < n; ++i) {
        const LimbT bi = b[i];
        const LimbT s = (bi << k) | prev;
        prev = bi >> rk;
        r[i] = arithmetic::uadd1c(a[i], s, c);
    }
    hi = prev;
    return c;
}

// r[0..n) = a[0..n) - (b[0..n) << k), 0 < k < limb bits; hi <- the bits of b shifted out at the
// top; returns the borrow. Streaming like addlsh_n(), r may be a or b.
template <std::unsigned_integral LimbT>
inline LimbT sublsh_n(LimbT* r, LimbT const* a, LimbT const* b, size_t n, unsigned k, LimbT& hi) noexcept
{
    const unsigned rk = std::numeric_limits<LimbT>::digits - k;
    LimbT prev = 0;
    unsigned char br = 0;
    for (size_t i = 0; i < n; ++i) {
        const LimbT bi = b[i];
        const LimbT s = (bi << k) | prev;
        prev = bi >> rk;
        r[i] = sbb1(a[i], s, br);
    }
    hi = prev;
    return br;
}

// r[0..n) = a[0..n) / D, a known to be a multiple of D, for any D dividing B-1 (B = 2^limb bits;
// for 64-bit limbs e.g. 3, 5, 15, 17, 51, 85, 255, 257, ...).
// 2-adic (Hensel) division: with m = (B-1)/D, D*m = B-1 = -1 (mod B), so -m is the inverse of D
// modulo B, and subtracting a*m limb by limb -- low part at this position, high part at the
// next -- builds the quotient from the bottom up, each limb depending only on the one below.
// The multiplications a[i]*m don't depend on the running value, so they come off the critical
// path, leaving just the subtraction chain on it (unlike multiplying the running difference by
// the inverse, which would put a multiply latency on every limb). For an exact division the
// 2-adic quotient mod B^n is the true quotient.
template <std::unsigned_integral LimbT, unsigned D>
inline void divexact_by(LimbT* r, LimbT const* a, size_t n) noexcept
{
    static_assert(D > 1 && (std::numeric_limits<LimbT>::max)() % D == 0, "D must divide B - 1");
    constexpr LimbT m = (std::numeric_limits<LimbT>::max)() / D;
    LimbT h = 0;
    for (size_t i = 0; i < n; ++i) {
        auto [p1, p0] = arithmetic::umul1(a[i], m);
        // h - p0, then minus p1 and that borrow: sub + sbb, the only dependent steps per limb
        unsigned char br = 0;
        const LimbT q = sbb1(h, p0, br);
        r[i] = q;
        h = sbb1(q, p1, br);
    }
}

template <std::unsigned_integral LimbT>
inline void divexact_by3(LimbT* r, LimbT const* a, size_t n) noexcept
{
    divexact_by<LimbT, 3>(r, a, n);
}

// Compile-time description of a fused linear combination (see lincomb()):
//   r = ((sum over j of +-(src_j << shift[j])) >> rshift) / div
struct lincomb_desc
{
    static constexpr unsigned max_terms = 4;

    // number of terms, 1..max_terms
    unsigned char count = 0;
    // term j is src_j << shift[j], shift[j] < limb bits
    unsigned char shift[max_terms] = {};
    // term j is subtracted; term 0 must be added
    bool neg[max_terms] = {};
    // Toom engine only: term j is additionally negated when its slot holds a negative value
    // (resolved to neg[] before lincomb() is instantiated)
    bool slot_sign[max_terms] = {};
    // source j is a two's complement number: sign-extended, not zero-extended, above its limbs
    bool tc[max_terms] = {};
    // the result may be negative and is written in two's complement (w limbs); otherwise it
    // must be non-negative (checked in debug builds)
    bool tc_result = false;
    // the sum is shifted right by rshift bits (the bits shifted out must be zero; arithmetic
    // shift for a two's complement sum)
    unsigned char rshift = 0;
    // and then divided exactly by div, any odd number: a divisor of B - 1, or a product of two
    // such, runs as one or two divexact_by() stages; anything else (e.g. 7, 189) as a Hensel
    // division by the inverse of div modulo B, which is slower (a multiply latency per limb;
    // lincomb_dual() hides half of it)
    unsigned div = 1;
};

template <std::unsigned_integral LimbT>
struct lincomb_src
{
    LimbT const* ptr;
    // limbs of the source; it is zero- (or, for a tc term, sign-) extended above them. n <= the
    // result width.
    size_t n;
};

namespace lincomb_detail {

// How lincomb() divides by an odd D: up to two stages dividing by divisors of B - 1 (a, b) and
// a Hensel stage for whatever doesn't factor that way (g). 1 = stage not used.
struct div_stages
{
    unsigned a = 1, b = 1, g = 1;
};

template <std::unsigned_integral LimbT>
consteval div_stages plan_div(unsigned d)
{
    constexpr LimbT mx = (std::numeric_limits<LimbT>::max)();
    if (d == 1) return {};
    if (d % 2 == 0) throw "lincomb: div must be odd (powers of two go into rshift)";
    if (mx % d == 0) return { d, 1, 1 };
    for (unsigned x = 3; x * x <= d; x += 2) {
        if (d % x == 0 && mx % x == 0 && mx % (d / x) == 0) return { x, d / x, 1 };
    }
    return { 1, 1, d };
}

// d^-1 mod B for odd d (Newton: each step doubles the correct low bits).
template <std::unsigned_integral LimbT>
consteval LimbT binvert(LimbT d)
{
    LimbT inv = d; // correct to 3 bits: d*d = 1 mod 8
    for (int i = 0; i < 6; ++i) inv = static_cast<LimbT>(inv * static_cast<LimbT>(LimbT{ 2 } - static_cast<LimbT>(d * inv)));
    if (static_cast<LimbT>(d * inv) != 1) throw "binvert: failed";
    return inv;
}

// src limb << K (see shl_pair()); prev holds the previous limb of that source (0 before the first one).
template <std::unsigned_integral LimbT, unsigned K>
NUMETRON_FORCEINLINE LimbT shifted(LimbT l, LimbT& prev) noexcept
{
    if constexpr (K == 0) {
        return l;
    } else {
        const LimbT t = shl_pair<LimbT, K>(prev, l);
        prev = l;
        return t;
    }
}

template <std::unsigned_integral LimbT, bool Neg>
NUMETRON_FORCEINLINE LimbT accumulate(LimbT x, LimbT t, unsigned char& c) noexcept
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    // GCC keeps these chains' carries (and _subborrow_u64's result) in memory when several are
    // live; the asm pins the carry to a register. See also chain4().
    if constexpr (sizeof(LimbT) == 8) {
        if constexpr (Neg) {
            __asm__("addb $0xFF, %[c]\n\tsbbq %[t], %[x]\n\tsetc %[c]" : [x] "+r"(x), [c] "+q"(c) : [t] "rm"(t) : "cc");
        } else {
            __asm__("addb $0xFF, %[c]\n\tadcq %[t], %[x]\n\tsetc %[c]" : [x] "+r"(x), [c] "+q"(c) : [t] "rm"(t) : "cc");
        }
        return x;
    } else
#endif
    {
        if constexpr (Neg) return sbb1(x, t, c);
        else return arithmetic::uadd1c(x, t, c);
    }
}

// x0..x3 +-= t0..t3 along one carry chain: four back-to-back adc/sbb with the carry in the flags
// throughout, restored from / saved to c once. MSVC makes that of four intrinsic steps by itself;
// GCC doesn't (it saves and restores the flag around every step, through memory), so it gets the
// sequence spelled out.
template <std::unsigned_integral LimbT, bool Neg>
NUMETRON_FORCEINLINE void chain4(LimbT& x0, LimbT& x1, LimbT& x2, LimbT& x3, LimbT t0, LimbT t1, LimbT t2, LimbT t3, unsigned char& c) noexcept
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if constexpr (sizeof(LimbT) == 8) {
        if constexpr (Neg) {
            __asm__(
                "addb $0xFF, %[c]\n\t"
                "sbbq %[t0], %[x0]\n\t"
                "sbbq %[t1], %[x1]\n\t"
                "sbbq %[t2], %[x2]\n\t"
                "sbbq %[t3], %[x3]\n\t"
                "setc %[c]"
                : [x0] "+r"(x0), [x1] "+r"(x1), [x2] "+r"(x2), [x3] "+r"(x3), [c] "+q"(c)
                : [t0] "rm"(t0), [t1] "rm"(t1), [t2] "rm"(t2), [t3] "rm"(t3)
                : "cc");
        } else {
            __asm__(
                "addb $0xFF, %[c]\n\t"
                "adcq %[t0], %[x0]\n\t"
                "adcq %[t1], %[x1]\n\t"
                "adcq %[t2], %[x2]\n\t"
                "adcq %[t3], %[x3]\n\t"
                "setc %[c]"
                : [x0] "+r"(x0), [x1] "+r"(x1), [x2] "+r"(x2), [x3] "+r"(x3), [c] "+q"(c)
                : [t0] "rm"(t0), [t1] "rm"(t1), [t2] "rm"(t2), [t3] "rm"(t3)
                : "cc");
        }
        return;
    } else
#endif
    {
        x0 = accumulate<LimbT, Neg>(x0, t0, c);
        x1 = accumulate<LimbT, Neg>(x1, t1, c);
        x2 = accumulate<LimbT, Neg>(x2, t2, c);
        x3 = accumulate<LimbT, Neg>(x3, t3, c);
    }
}

// Running state of lincomb(): every term has its own shift state and carry chain (so the chains
// are independent and overlap), the output goes through the optional right shift (one limb of
// delay) and the optional exact division.
template <std::unsigned_integral LimbT, lincomb_desc Desc>
struct lincomb_state
{
    static constexpr unsigned N = Desc.count;
    static constexpr unsigned bits = std::numeric_limits<LimbT>::digits;
    static constexpr unsigned R = Desc.rshift;
    static constexpr div_stages DS = plan_div<LimbT>(Desc.div);
    static constexpr LimbT dma = DS.a > 1 ? (std::numeric_limits<LimbT>::max)() / DS.a : LimbT{ 0 };
    static constexpr LimbT dmb = DS.b > 1 ? (std::numeric_limits<LimbT>::max)() / DS.b : LimbT{ 0 };
    static constexpr LimbT ginv = DS.g > 1 ? binvert<LimbT>(static_cast<LimbT>(DS.g)) : LimbT{ 1 };

    LimbT* rp;
    // previous source limb of each shifted term (see shifted())
    LimbT hi0 = 0, hi1 = 0, hi2 = 0, hi3 = 0;
    unsigned char c1 = 0, c2 = 0, c3 = 0;
    LimbT pending = 0; // R > 0: the previous sum limb, waiting for the low bits of the next one
    LimbT ha = 0, hb = 0; // divexact_by() states of the division stages a and b
    LimbT gb = 0;         // Hensel stage: borrow into the next limb

    // One divexact_by() step (see there): sub, then sbb on the borrow.
    template <LimbT M>
    NUMETRON_FORCEINLINE static LimbT dbm1_step(LimbT y, LimbT& h) noexcept
    {
        auto [p1, p0] = arithmetic::umul1(y, M);
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        if constexpr (sizeof(LimbT) == 8) {
            // spelled out for GCC, which otherwise routes the borrow through memory (see chain4())
            LimbT q;
            __asm__("movq %[h], %[q]\n\tsubq %[p0], %[q]\n\tmovq %[q], %[h]\n\tsbbq %[p1], %[h]"
                : [q] "=&r"(q), [h] "+r"(h) : [p0] "rm"(p0), [p1] "rm"(p1) : "cc");
            return q;
        } else
#endif
        {
            unsigned char br = 0;
            const LimbT q = sbb1(h, p0, br);
            h = sbb1(q, p1, br);
            return q;
        }
    }

    // One step of the Hensel division by DS.g: the quotient limb is (y - borrow) * g^-1 mod B,
    // and q * g reaches back into the next limb by its high part.
    NUMETRON_FORCEINLINE LimbT hensel_step(LimbT y) noexcept
    {
        unsigned char br = 0;
        const LimbT s = sbb1(y, gb, br);
        const LimbT q = static_cast<LimbT>(s * ginv);
        auto [qh, ql] = arithmetic::umul1(q, static_cast<LimbT>(DS.g));
        (void)ql; // == s
        gb = static_cast<LimbT>(qh + br);
        return q;
    }

    // The sum limb for the given source limbs.
    NUMETRON_FORCEINLINE LimbT combine(LimbT l0, [[maybe_unused]] LimbT l1, [[maybe_unused]] LimbT l2, [[maybe_unused]] LimbT l3) noexcept
    {
        LimbT x = shifted<LimbT, Desc.shift[0]>(l0, hi0);
        if constexpr (N > 1) x = accumulate<LimbT, Desc.neg[1]>(x, shifted<LimbT, Desc.shift[1]>(l1, hi1), c1);
        if constexpr (N > 2) x = accumulate<LimbT, Desc.neg[2]>(x, shifted<LimbT, Desc.shift[2]>(l2, hi2), c2);
        if constexpr (N > 3) x = accumulate<LimbT, Desc.neg[3]>(x, shifted<LimbT, Desc.shift[3]>(l3, hi3), c3);
        return x;
    }

    // One term over 4 consecutive limbs: the 4 shifted source limbs first, then 4 back-to-back
    // steps of the term's carry chain, so the compiler can keep the carry in the flags across
    // them instead of saving and restoring it per limb (which interleaving the terms limb by limb
    // forces, the chains being separate).
    template <unsigned K>
    NUMETRON_FORCEINLINE static void shifted4(LimbT const* q, LimbT& prev, LimbT& t0, LimbT& t1, LimbT& t2, LimbT& t3) noexcept
    {
        const LimbT l0 = q[0], l1 = q[1], l2 = q[2], l3 = q[3];
        if constexpr (K == 0) {
            t0 = l0; t1 = l1; t2 = l2; t3 = l3;
        } else {
            t0 = shl_pair<LimbT, K>(prev, l0);
            t1 = shl_pair<LimbT, K>(l0, l1);
            t2 = shl_pair<LimbT, K>(l1, l2);
            t3 = shl_pair<LimbT, K>(l2, l3);
            prev = l3;
        }
    }

    template <unsigned K, bool Neg>
    NUMETRON_FORCEINLINE void apply4(LimbT const* q, LimbT& prev, unsigned char& c, LimbT& x0, LimbT& x1, LimbT& x2, LimbT& x3) noexcept
    {
        LimbT t0, t1, t2, t3;
        shifted4<K>(q, prev, t0, t1, t2, t3);
        chain4<LimbT, Neg>(x0, x1, x2, x3, t0, t1, t2, t3, c);
    }

    // Sum limbs i..i+3, all sources present there.
    NUMETRON_FORCEINLINE void combine4(LimbT const* q0, [[maybe_unused]] LimbT const* q1, [[maybe_unused]] LimbT const* q2, [[maybe_unused]] LimbT const* q3,
        LimbT& x0, LimbT& x1, LimbT& x2, LimbT& x3) noexcept
    {
        shifted4<Desc.shift[0]>(q0, hi0, x0, x1, x2, x3);
        if constexpr (N > 1) apply4<Desc.shift[1], Desc.neg[1]>(q1, hi1, c1, x0, x1, x2, x3);
        if constexpr (N > 2) apply4<Desc.shift[2], Desc.neg[2]>(q2, hi2, c2, x0, x1, x2, x3);
        if constexpr (N > 3) apply4<Desc.shift[3], Desc.neg[3]>(q3, hi3, c3, x0, x1, x2, x3);
    }

    // Result limb through the division stages. Each is a 2-adic exact division working from the
    // bottom up, so they chain limb by limb; the quotient is exact, for a two's complement
    // (negative) dividend too.
    NUMETRON_FORCEINLINE void out(LimbT y) noexcept
    {
        if constexpr (DS.a > 1) y = dbm1_step<dma>(y, ha);
        if constexpr (DS.b > 1) y = dbm1_step<dmb>(y, hb);
        if constexpr (DS.g > 1) y = hensel_step(y);
        *rp++ = y;
    }

    // Takes sum limb i; with a right shift, emits result limb i - 1 (i > 0).
    NUMETRON_FORCEINLINE void put(LimbT x) noexcept
    {
        if constexpr (R == 0) {
            out(x);
        } else {
            out(shr_pair<LimbT, R>(pending, x));
            pending = x;
        }
    }

    // Signed sum of the final carries/borrows: 0 when the combination fits (debug check).
    int net_carry() const noexcept
    {
        int net = 0;
        if constexpr (N > 1) net += Desc.neg[1] ? -int(c1) : int(c1);
        if constexpr (N > 2) net += Desc.neg[2] ? -int(c2) : int(c2);
        if constexpr (N > 3) net += Desc.neg[3] ? -int(c3) : int(c3);
        return net;
    }
};

}

namespace lincomb_detail {

// One lincomb() stream: the state plus the sources, walked by the drivers below in blocks of 4
// limbs while every source has limbs there, then limb by limb.
template <std::unsigned_integral LimbT, lincomb_desc Desc>
struct lincomb_stream
{
    using state_t = lincomb_state<LimbT, Desc>;
    static constexpr unsigned N = Desc.count;
    static constexpr unsigned R = Desc.rshift;
    static_assert(N >= 1 && N <= lincomb_desc::max_terms, "lincomb: bad term count");
    static_assert(!Desc.neg[0], "lincomb: the first term must be added");
    static_assert(Desc.shift[0] < state_t::bits && Desc.shift[1] < state_t::bits
        && Desc.shift[2] < state_t::bits && Desc.shift[3] < state_t::bits, "lincomb: shift out of range");
    static_assert(R < state_t::bits, "lincomb: rshift out of range");

    state_t st;
    LimbT* r;
    size_t w;
    LimbT const* p0; LimbT const* p1; LimbT const* p2; LimbT const* p3;
    size_t n0, n1, n2, n3;
    // what each source reads as above its limbs: 0, or the sign for a two's complement term
    LimbT f0, f1, f2, f3;
    // limbs where every source is present
    size_t m;

    static LimbT fill(bool tc, LimbT const* p, size_t n) noexcept
    {
        return tc && n && (p[n - 1] >> (state_t::bits - 1)) ? static_cast<LimbT>(~LimbT{ 0 }) : LimbT{ 0 };
    }

    static LimbT at(LimbT const* p, size_t n, size_t i, LimbT f) noexcept { return i < n ? p[i] : f; }

    // Reads the fills before anything is written (r may be a source).
    NUMETRON_FORCEINLINE lincomb_stream(LimbT* r_, size_t w_, lincomb_src<LimbT> const* src) noexcept
        : st{ r_ }, r{ r_ }, w{ w_ }
        , p0{ src[0].ptr }, p1{ N > 1 ? src[1].ptr : nullptr }, p2{ N > 2 ? src[2].ptr : nullptr }, p3{ N > 3 ? src[3].ptr : nullptr }
        , n0{ src[0].n }, n1{ N > 1 ? src[1].n : w_ }, n2{ N > 2 ? src[2].n : w_ }, n3{ N > 3 ? src[3].n : w_ }
    {
        NUMETRON_ASSERT(w > 0);
        NUMETRON_ASSERT(n0 <= w && n1 <= w && n2 <= w && n3 <= w);
        f0 = fill(Desc.tc[0], p0, n0);
        f1 = N > 1 ? fill(Desc.tc[1], p1, n1) : LimbT{ 0 };
        f2 = N > 2 ? fill(Desc.tc[2], p2, n2) : LimbT{ 0 };
        f3 = N > 3 ? fill(Desc.tc[3], p3, n3) : LimbT{ 0 };
        m = (std::min)((std::min)(n0, n1), (std::min)(n2, n3));
    }

    // The limb index the walk starts at: with a right shift the first sum limb only primes it.
    NUMETRON_FORCEINLINE size_t start() noexcept
    {
        if constexpr (R > 0) {
            st.pending = st.combine(at(p0, n0, 0, f0), N > 1 ? at(p1, n1, 0, f1) : 0, N > 2 ? at(p2, n2, 0, f2) : 0, N > 3 ? at(p3, n3, 0, f3) : 0);
            return 1;
        } else {
            return 0;
        }
    }

    // Limbs i..i+3, all sources present there (i + 4 <= m).
    NUMETRON_FORCEINLINE void block4(size_t i) noexcept
    {
        LimbT x0, x1, x2, x3;
        st.combine4(p0 + i, N > 1 ? p1 + i : nullptr, N > 2 ? p2 + i : nullptr, N > 3 ? p3 + i : nullptr, x0, x1, x2, x3);
        st.put(x0);
        st.put(x1);
        st.put(x2);
        st.put(x3);
    }

    NUMETRON_FORCEINLINE void single(size_t i) noexcept
    {
        st.put(st.combine(at(p0, n0, i, f0), N > 1 ? at(p1, n1, i, f1) : 0, N > 2 ? at(p2, n2, i, f2) : 0, N > 3 ? at(p3, n3, i, f3) : 0));
    }

    // From limb i on: the remaining blocks, the single limbs, and the top.
    NUMETRON_FORCEINLINE void rest(size_t i) noexcept
    {
        for (; i + 4 <= m; i += 4) block4(i);
        for (; i < w; ++i) single(i);
        finish();
    }

    // Sum limb w: the bits shifted out of the sources' tops (and their sign extensions) and the
    // carries.
    NUMETRON_FORCEINLINE void finish() noexcept
    {
        const LimbT top = st.combine(f0, f1, f2, f3);
        if constexpr (R > 0) st.put(top);
        if constexpr (!Desc.tc_result) {
            // Non-negative and fitting: nothing left above the w result limbs, and no net carry
            // out of the w + 1 limbs -- except that a negative two's complement term stands there
            // as B^(w+1) + value, which one carry (added) or borrow (subtracted) takes back out.
            if constexpr (R > 0) {
                NUMETRON_ASSERT(!(top >> R));
            } else {
                NUMETRON_ASSERT(!top);
            }
#ifndef NDEBUG
            int expected = f0 ? 1 : 0;
            if constexpr (N > 1) expected += f1 ? (Desc.neg[1] ? -1 : 1) : 0;
            if constexpr (N > 2) expected += f2 ? (Desc.neg[2] ? -1 : 1) : 0;
            if constexpr (N > 3) expected += f3 ? (Desc.neg[3] ? -1 : 1) : 0;
            NUMETRON_ASSERT(st.net_carry() == expected);
#endif
        }
        NUMETRON_ASSERT(st.rp == r + w);
    }
};

}

// r[0..w) = ((sum over j of +-(src_j << shift[j])) >> rshift) / div, all in one streaming pass;
// Desc (see lincomb_desc) fixes the shape at compile time. The combination must be a multiple
// of 2^rshift * div and the result must fit in w limbs -- non-negative, or with tc_result as a
// two's complement number; the sum itself may take one limb more than that (it's formed over
// w + 1 limbs). w > 0.
// Source limb i is read before result limb i is written, so r may coincide with any source
// (same pointer); otherwise r must not overlap a source.
template <std::unsigned_integral LimbT, lincomb_desc Desc>
inline void lincomb(LimbT* r, size_t w, lincomb_src<LimbT> const* src) noexcept
{
    lincomb_detail::lincomb_stream<LimbT, Desc> a{ r, w, src };
    a.rest(a.start());
}

// Two independent lincomb()s of the same shape and width in one loop: ra from srca and rb from
// srcb, interleaved block by block. The two streams' dependency chains -- the carry chains, and
// above all a Hensel division stage, whose multiply latency otherwise bounds the pass -- then
// overlap. Neither result may overlap the other stream's sources or result.
// Interleaving pays off where a pass is latency-bound: a Hensel stage by far (about 40% off the
// pair), but also the plain and shifted shapes (15-25%). The exceptions, measured on x86-64:
// four unshifted terms, and two divexact_by() stages without a Hensel one, whose two streams'
// state no longer fits the registers -- those run one after the other.
template <std::unsigned_integral LimbT, lincomb_desc Desc>
consteval bool lincomb_dual_interleaves()
{
    constexpr auto ds = lincomb_detail::plan_div<LimbT>(Desc.div);
    if (ds.g > 1) return true;
    if (ds.b > 1) return false;
    if (Desc.count == 4 && !Desc.rshift && !Desc.shift[0] && !Desc.shift[1] && !Desc.shift[2] && !Desc.shift[3]) return false;
    return true;
}

template <std::unsigned_integral LimbT, lincomb_desc Desc>
inline void lincomb_dual(LimbT* ra, lincomb_src<LimbT> const* srca, LimbT* rb, lincomb_src<LimbT> const* srcb, size_t w) noexcept
{
    if constexpr (!lincomb_dual_interleaves<LimbT, Desc>()) {
        lincomb<LimbT, Desc>(ra, w, srca);
        lincomb<LimbT, Desc>(rb, w, srcb);
    } else {
        lincomb_detail::lincomb_stream<LimbT, Desc> a{ ra, w, srca };
        lincomb_detail::lincomb_stream<LimbT, Desc> b{ rb, w, srcb };
        size_t i = a.start();
        b.start();
        const size_t m = (std::min)(a.m, b.m);
        for (; i + 4 <= m; i += 4) {
            a.block4(i);
            b.block4(i);
        }
        a.rest(i);
        b.rest(i);
    }
}

}
