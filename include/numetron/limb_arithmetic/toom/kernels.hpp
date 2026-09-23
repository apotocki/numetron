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

// r[0..n) = a[0..n) << 1, returns the bit shifted out at the top. n > 0.
template <std::unsigned_integral LimbT>
inline LimbT lshift1(LimbT* r, LimbT const* a, size_t n) noexcept
{
    constexpr int top = std::numeric_limits<LimbT>::digits - 1;
    const LimbT out = a[n - 1] >> top;
    for (size_t i = n - 1; i > 0; --i) { // high to low, so r == a works
        r[i] = (a[i] << 1) | (a[i - 1] >> top);
    }
    r[0] = a[0] << 1;
    return out;
}

// r[0..n) = a[0..n) >> 1, returns the bit shifted out at the bottom. n > 0.
template <std::unsigned_integral LimbT>
inline LimbT rshift1(LimbT* r, LimbT const* a, size_t n) noexcept
{
    constexpr int top = std::numeric_limits<LimbT>::digits - 1;
    const LimbT out = a[0] & 1;
    for (size_t i = 0; i + 1 < n; ++i) { // low to high, so r == a works
        r[i] = (a[i] >> 1) | (a[i + 1] << top);
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
        const LimbT borrow = h < p0;
        h -= p0;
        r[i] = h;
        h = h - p1 - borrow;
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
    // the sum is shifted right by rshift bits (the bits shifted out must be zero)
    unsigned char rshift = 0;
    // and then divided exactly by div: 1, or a divisor of B - 1 (see divexact_by())
    unsigned short div = 1;
};

template <std::unsigned_integral LimbT>
struct lincomb_src
{
    LimbT const* ptr;
    // limbs of the source; it is zero-extended above them. n <= the result width.
    size_t n;
};

namespace lincomb_detail {

// src limb << K with the bits shifted out of the previous limb in hi.
template <std::unsigned_integral LimbT, unsigned K>
NUMETRON_FORCEINLINE LimbT shifted(LimbT l, LimbT& hi) noexcept
{
    if constexpr (K == 0) {
        return l;
    } else {
        const LimbT t = (l << K) | hi;
        hi = l >> (std::numeric_limits<LimbT>::digits - K);
        return t;
    }
}

template <std::unsigned_integral LimbT, bool Neg>
NUMETRON_FORCEINLINE LimbT accumulate(LimbT x, LimbT t, unsigned char& c) noexcept
{
    if constexpr (Neg) return sbb1(x, t, c);
    else return arithmetic::uadd1c(x, t, c);
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
    static constexpr unsigned D = Desc.div;
    static constexpr LimbT dm = D > 1 ? (std::numeric_limits<LimbT>::max)() / D : LimbT{ 0 };

    LimbT* rp;
    LimbT hi0 = 0, hi1 = 0, hi2 = 0, hi3 = 0;
    unsigned char c1 = 0, c2 = 0, c3 = 0;
    LimbT pending = 0; // R > 0: the previous sum limb, waiting for the low bits of the next one
    LimbT h = 0;       // D > 1: divexact_by() state

    // The sum limb for the given source limbs.
    NUMETRON_FORCEINLINE LimbT combine(LimbT l0, [[maybe_unused]] LimbT l1, [[maybe_unused]] LimbT l2, [[maybe_unused]] LimbT l3) noexcept
    {
        LimbT x = shifted<LimbT, Desc.shift[0]>(l0, hi0);
        if constexpr (N > 1) x = accumulate<LimbT, Desc.neg[1]>(x, shifted<LimbT, Desc.shift[1]>(l1, hi1), c1);
        if constexpr (N > 2) x = accumulate<LimbT, Desc.neg[2]>(x, shifted<LimbT, Desc.shift[2]>(l2, hi2), c2);
        if constexpr (N > 3) x = accumulate<LimbT, Desc.neg[3]>(x, shifted<LimbT, Desc.shift[3]>(l3, hi3), c3);
        return x;
    }

    NUMETRON_FORCEINLINE void out(LimbT y) noexcept
    {
        if constexpr (D == 1) {
            *rp++ = y;
        } else {
            auto [p1, p0] = arithmetic::umul1(y, dm);
            const LimbT borrow = h < p0;
            h -= p0;
            *rp++ = h;
            h = h - p1 - borrow;
        }
    }

    // Takes sum limb i; with a right shift, emits result limb i - 1 (i > 0).
    NUMETRON_FORCEINLINE void put(LimbT x) noexcept
    {
        if constexpr (R == 0) {
            out(x);
        } else {
            out((pending >> R) | (x << (bits - R)));
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

// r[0..w) = ((sum over j of +-(src_j << shift[j])) >> rshift) / div, all in one streaming pass;
// Desc (see lincomb_desc) fixes the shape at compile time. The combination must be
// non-negative, a multiple of 2^rshift * div, and the result must fit in w limbs; the sum itself
// may take one limb more than that (it's formed over w + 1 limbs). w > 0.
// Source limb i is read before result limb i is written, so r may coincide with any source
// (same pointer); otherwise r must not overlap a source.
template <std::unsigned_integral LimbT, lincomb_desc Desc>
inline void lincomb(LimbT* r, size_t w, lincomb_src<LimbT> const* src) noexcept
{
    using state_t = lincomb_detail::lincomb_state<LimbT, Desc>;
    constexpr unsigned N = Desc.count;
    constexpr unsigned R = Desc.rshift;
    static_assert(N >= 1 && N <= lincomb_desc::max_terms, "lincomb: bad term count");
    static_assert(!Desc.neg[0], "lincomb: the first term must be added");
    static_assert(Desc.shift[0] < state_t::bits && Desc.shift[1] < state_t::bits
        && Desc.shift[2] < state_t::bits && Desc.shift[3] < state_t::bits, "lincomb: shift out of range");
    static_assert(R < state_t::bits, "lincomb: rshift out of range");
    static_assert(Desc.div == 1 || (Desc.div > 1 && (std::numeric_limits<LimbT>::max)() % Desc.div == 0),
        "lincomb: div must divide B - 1");
    NUMETRON_ASSERT(w > 0);

    LimbT const* const p0 = src[0].ptr;
    LimbT const* const p1 = N > 1 ? src[1].ptr : nullptr;
    LimbT const* const p2 = N > 2 ? src[2].ptr : nullptr;
    LimbT const* const p3 = N > 3 ? src[3].ptr : nullptr;
    const size_t n0 = src[0].n;
    const size_t n1 = N > 1 ? src[1].n : w;
    const size_t n2 = N > 2 ? src[2].n : w;
    const size_t n3 = N > 3 ? src[3].n : w;
    NUMETRON_ASSERT(n0 <= w && n1 <= w && n2 <= w && n3 <= w);
    const size_t m = (std::min)((std::min)(n0, n1), (std::min)(n2, n3));

    state_t st{ r };
    auto at = [](LimbT const* p, size_t n, size_t i) noexcept -> LimbT { return i < n ? p[i] : LimbT{ 0 }; };

    size_t i = 0;
    if constexpr (R > 0) {
        // the first sum limb only primes the shift
        st.pending = st.combine(at(p0, n0, 0), N > 1 ? at(p1, n1, 0) : 0, N > 2 ? at(p2, n2, 0) : 0, N > 3 ? at(p3, n3, 0) : 0);
        i = 1;
    }
    // all sources present
    for (; i < m; ++i) {
        st.put(st.combine(p0[i], N > 1 ? p1[i] : 0, N > 2 ? p2[i] : 0, N > 3 ? p3[i] : 0));
    }
    // some exhausted
    for (; i < w; ++i) {
        st.put(st.combine(at(p0, n0, i), N > 1 ? at(p1, n1, i) : 0, N > 2 ? at(p2, n2, i) : 0, N > 3 ? at(p3, n3, i) : 0));
    }
    // sum limb w: just the bits shifted out of the sources' tops and the carries
    const LimbT top = st.combine(0, 0, 0, 0);
    if constexpr (R > 0) {
        st.put(top);
        NUMETRON_ASSERT(!(top >> R));
    } else {
        NUMETRON_ASSERT(!top);
    }
    NUMETRON_ASSERT(st.net_carry() == 0);
    NUMETRON_ASSERT(st.rp == r + w);
}

}
