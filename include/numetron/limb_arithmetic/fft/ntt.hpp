// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "numetron/arithmetic.hpp"
#include "numetron/detail/assert.hpp"
#include "numetron/limb_arithmetic/platform.hpp" // NUMETRON_FORCEINLINE

// Number-theoretic transform over word-size primes: the modular arithmetic, the primes, the
// root-of-unity tables and the forward / inverse transforms that umul_fft.hpp builds the
// multi-prime FFT multiplication from. See docs/fft.md.
//
// Arithmetic. Every prime p is below 2^62, so values can be kept lazily in [0, 2p) (and sums of
// two in [0, 4p)) without leaving a 64-bit word:
//  - multiplication by a fixed root w uses Shoup's method: with w' = floor(w * 2^64 / p),
//    a * w - hi(a * w') * p lies in [0, 2p) for any 64-bit a;
//  - the pointwise products and the loading of the operands use Montgomery reduction (R = 2^64):
//    redc(t) = t / R mod p, in [0, 2p) for t < p * R. Operands are loaded as u * R (Montgomery
//    form), the pointwise product of two such is (u*v) * R, and the final scaling of the inverse
//    transform multiplies by 1 / (L * R) to take both the transform length and that R out.
//
// Transforms. Length L = m or L = 3m, m = 2^k. The forward transform is decimation in frequency:
// for L = 3m one radix-3 step splits the input into three sequences of m, then each goes through
// radix-2 DIF butterflies, leaving the spectrum in a permuted (bit-reversed per block) order. The
// inverse is the exact mirror (radix-2 DIT per block, then the radix-3 step), so it takes that
// order back to natural order without any explicit permutation -- pointwise products don't care
// about the order in between.

namespace numetron::limb_arithmetic::detail::ntt {

using u64 = std::uint64_t;

// The transform lengths go up to 3 * 2^max_log2.
inline constexpr unsigned max_log2 = 40;

// A prime p = c * 3 * 2^40 + 1 < 2^62 with its Montgomery constants and a primitive root.
struct prime
{
    u64 p;
    u64 pinv;   // -p^-1 mod 2^64
    u64 r2;     // 2^128 mod p (to Montgomery form: redc(x * r2) = x * 2^64 mod p)
    u64 r3;     // 2^192 mod p (redc(x * r3) = x * 2^128 mod p: the high limb of a 2-limb value)
    u64 g;      // a primitive root mod p
};

constexpr prime make_prime(u64 p, u64 g) noexcept
{
    u64 x = p; // p * p = 1 (mod 8); each Newton step doubles the correct low bits
    for (int i = 0; i < 5; ++i) x *= 2 - p * x;
    u64 r = (~u64{ 0 } % p + 1) % p; // 2^64 mod p
    auto times_2_64 = [p](u64 v) {
        for (int i = 0; i < 64; ++i) {
            v <<= 1;
            if (v >= p) v -= p;
        }
        return v;
    };
    const u64 r2 = times_2_64(r);
    return { p, u64{ 0 } - x, r2, times_2_64(r2), g };
}

// The largest primes of that form below 2^62, in descending order (scratch search, Miller-Rabin
// with the 12 deterministic 64-bit bases; primitive roots checked against every prime factor of
// p - 1). The FFT (umul_fft.hpp, 2 limbs per coefficient) uses all five: their product ~2^310
// covers coefficients up to L * 2^256 for any L < 2^54.
inline constexpr prime primes[5] = {
    make_prime(0x3fffc00000000001ull, 11), // 1398080 * 3 * 2^40 + 1
    make_prime(0x3fff840000000001ull, 19), // 1398060 * 3 * 2^40 + 1
    make_prime(0x3fff810000000001ull, 5),  // 1398059 * 3 * 2^40 + 1
    make_prime(0x3fff540000000001ull, 5),  // 1398044 * 3 * 2^40 + 1
    make_prime(0x3fff450000000001ull, 10), // 1398039 * 3 * 2^40 + 1
};
inline constexpr size_t prime_count = 5;

// t / 2^64 mod p for t = hi * 2^64 + lo < p * 2^64; the result is in [0, 2p).
NUMETRON_FORCEINLINE u64 redc(u64 hi, u64 lo, prime const& P) noexcept
{
    const u64 m = lo * P.pinv;
    const u64 mh = arithmetic::umul1(m, P.p).first;
    // lo + lo(m * p) is 0 mod 2^64, with a carry out unless lo is 0
    return hi + mh + (lo != 0);
}

// a * b / 2^64 mod p, for a * b < p * 2^64 (e.g. both below 2p); in [0, 2p).
NUMETRON_FORCEINLINE u64 mont_mul(u64 a, u64 b, prime const& P) noexcept
{
    auto [h, l] = arithmetic::umul1(a, b);
    return redc(h, l, P);
}

// floor(w * 2^64 / p), the Shoup companion of a root w < p.
inline u64 shoup_const(u64 w, u64 p) noexcept
{
    return arithmetic::udiv2by1(w, u64{ 0 }, p).first;
}

// a * w mod p, for any 64-bit a, w < p with its Shoup companion wq; in [0, 2p).
NUMETRON_FORCEINLINE u64 shoup_mul(u64 a, u64 w, u64 wq, u64 p) noexcept
{
    const u64 q = arithmetic::umul1(a, wq).first;
    return a * w - q * p;
}

// x + y for x, y in [0, 2p), back into [0, 2p).
NUMETRON_FORCEINLINE u64 add2p(u64 x, u64 y, u64 p2) noexcept
{
    const u64 s = x + y;
    return s >= p2 ? s - p2 : s;
}

// [0, 2p) -> [0, p)
NUMETRON_FORCEINLINE u64 reduce(u64 x, u64 p) noexcept
{
    return x >= p ? x - p : x;
}

// a * b mod p for plain (non-Montgomery) a, b < p; in [0, p).
inline u64 mul_mod(u64 a, u64 b, prime const& P) noexcept
{
    // redc(a*b) = a*b/R; times r2 = R^2 and reduced again: a*b
    return reduce(mont_mul(reduce(mont_mul(a, b, P), P.p), P.r2, P), P.p);
}

inline u64 pow_mod(u64 a, u64 e, prime const& P) noexcept
{
    u64 r = 1;
    for (; e; e >>= 1) {
        if (e & 1) r = mul_mod(r, a, P);
        a = mul_mod(a, a, P);
    }
    return r;
}

// The roots of unity used everywhere come from one chain per prime: G = g^((p-1) / (3*2^40)) of
// order 3*2^40, the root of order 3*2^k is G^(2^(40-k)) and the root of order 2^k is
// G^(3*2^(40-k)) -- the cube of the former, which is what the radix-3 step needs.
inline u64 root_pow2(size_t pi, unsigned k) noexcept
{
    prime const& P = primes[pi];
    const u64 G = pow_mod(P.g, (P.p - 1) / (u64{ 3 } << max_log2), P);
    return pow_mod(G, u64{ 3 } << (max_log2 - k), P);
}

inline u64 root_3pow2(size_t pi, unsigned k) noexcept
{
    prime const& P = primes[pi];
    const u64 G = pow_mod(P.g, (P.p - 1) / (u64{ 3 } << max_log2), P);
    return pow_mod(G, u64{ 1 } << (max_log2 - k), P);
}

// Root tables, one per prime, kind and k, built once on first use and kept for the life of the
// process; readers take the pointer with acquire and never lock, building is serialized per
// prime. A table doesn't depend on the transform length, so every length shares them.
//  - radix-2 level k (butterfly span 2^k): half = 2^(k-1) roots w^j of the order-2^k root w, their
//    Shoup companions, then the same for w^-1 (4 * half words);
//  - radix-3 step for L = 3 * 2^k (m = 2^k): for j < m the twiddles w^j, w^2j with companions,
//    then w^-j, w^-2j (w the root of order 3m), then the cube roots of unity e1 = w^m,
//    e2 = w^2m = e1^-1 with companions (8m + 4 words).
class root_tables
{
public:
    static u64 const* level(size_t pi, unsigned k)
    {
        NUMETRON_ASSERT(pi < prime_count && k >= 1 && k <= max_log2);
        return get(pi, 0, k);
    }

    static u64 const* radix3(size_t pi, unsigned k)
    {
        NUMETRON_ASSERT(pi < prime_count && k <= max_log2);
        return get(pi, 1, k);
    }

private:
    struct prime_state
    {
        std::atomic<u64 const*> tables[2][max_log2 + 1] = {};
        std::unique_ptr<u64[]> owned[2][max_log2 + 1];
        std::mutex mutex;
    };

    static prime_state& state(size_t pi)
    {
        static prime_state states[prime_count];
        return states[pi];
    }

    static u64 const* get(size_t pi, int kind, unsigned k)
    {
        auto& s = state(pi);
        if (u64 const* t = s.tables[kind][k].load(std::memory_order_acquire)) return t;
        std::lock_guard<std::mutex> lock(s.mutex);
        if (u64 const* t = s.tables[kind][k].load(std::memory_order_relaxed)) return t;
        std::unique_ptr<u64[]> t = kind ? build_radix3(pi, k) : build_level(pi, k);
        u64 const* result = t.get();
        s.owned[kind][k] = std::move(t);
        s.tables[kind][k].store(result, std::memory_order_release);
        return result;
    }

    // r[j] = w^j, r[n + j] = its Shoup companion, for j < n
    static void powers(u64* r, size_t n, u64 w, prime const& P)
    {
        const u64 wq = shoup_const(w, P.p);
        u64 x = 1;
        for (size_t j = 0; j < n; ++j) {
            r[j] = x;
            r[n + j] = shoup_const(x, P.p);
            x = reduce(shoup_mul(x, w, wq, P.p), P.p);
        }
    }

    static std::unique_ptr<u64[]> build_level(size_t pi, unsigned k)
    {
        prime const& P = primes[pi];
        const size_t half = size_t{ 1 } << (k - 1);
        auto t = std::make_unique<u64[]>(4 * half);
        const u64 w = root_pow2(pi, k);
        powers(t.get(), half, w, P);
        powers(t.get() + 2 * half, half, pow_mod(w, (u64{ 1 } << k) - 1, P), P);
        return t;
    }

    static std::unique_ptr<u64[]> build_radix3(size_t pi, unsigned k)
    {
        prime const& P = primes[pi];
        const size_t m = size_t{ 1 } << k;
        auto t = std::make_unique<u64[]>(8 * m + 4);
        const u64 w = root_3pow2(pi, k);
        const u64 wi = pow_mod(w, 3 * m - 1, P);
        powers(t.get(), m, w, P);
        powers(t.get() + 2 * m, m, mul_mod(w, w, P), P);
        powers(t.get() + 4 * m, m, wi, P);
        powers(t.get() + 6 * m, m, mul_mod(wi, wi, P), P);
        const u64 e1 = pow_mod(w, m, P), e2 = mul_mod(e1, e1, P);
        u64* e = t.get() + 8 * m;
        e[0] = e1;
        e[1] = shoup_const(e1, P.p);
        e[2] = e2;
        e[3] = shoup_const(e2, P.p);
        return t;
    }
};

// The butterflies. Values in [0, 2p) in and out; u - v is formed as u - v + 2p in (0, 4p).
// DIF: (u, v) -> (u + v, (u - v) * w)
NUMETRON_FORCEINLINE void dif_bf(u64& x, u64& y, u64 w, u64 wq, u64 p, u64 p2) noexcept
{
    const u64 u = x, v = y;
    x = add2p(u, v, p2);
    y = shoup_mul(u - v + p2, w, wq, p);
}

// DIT: (u, v) -> (u + v * w, u - v * w)
NUMETRON_FORCEINLINE void dit_bf(u64& x, u64& y, u64 w, u64 wq, u64 p, u64 p2) noexcept
{
    const u64 u = x;
    const u64 t = shoup_mul(y, w, wq, p);
    x = add2p(u, t, p2);
    y = add2p(u, p2 - t, p2);
}

// No-twiddle butterfly: (u, v) -> (u + v, u - v)
NUMETRON_FORCEINLINE void bf1(u64& x, u64& y, u64 p2) noexcept
{
    const u64 u = x, v = y;
    x = add2p(u, v, p2);
    y = add2p(u, p2 - v, p2);
}

// One radix-2 DIF level (span 2^lk) over a[0..m).
inline void dif_level(u64* a, size_t m, unsigned lk, size_t pi) noexcept
{
    const u64 p = primes[pi].p, p2 = 2 * p;
    const size_t half = size_t{ 1 } << (lk - 1);
    u64 const* w = root_tables::level(pi, lk);
    u64 const* wq = w + half;
    for (size_t s = 0; s < m; s += 2 * half) {
        u64* x = a + s;
        for (size_t j = 0; j < half; ++j) dif_bf(x[j], x[j + half], w[j], wq[j], p, p2);
    }
}

// The last two DIF levels (spans 4 and 2): the only root that isn't 1 is i = w4.
inline void dif_last2(u64* a, size_t m, size_t pi) noexcept
{
    const u64 p = primes[pi].p, p2 = 2 * p;
    u64 const* w4 = root_tables::level(pi, 2);
    const u64 i = w4[1], iq = w4[3];
    for (size_t s = 0; s < m; s += 4) {
        u64* x = a + s;
        bf1(x[0], x[2], p2);
        dif_bf(x[1], x[3], i, iq, p, p2);
        bf1(x[0], x[1], p2);
        bf1(x[2], x[3], p2);
    }
}

// Radix-2 DIF over a[0..2^k), k >= 1: natural order in, bit-reversed out.
inline void dif2(u64* a, unsigned k, size_t pi) noexcept
{
    const size_t m = size_t{ 1 } << k;
    if (k == 1) {
        bf1(a[0], a[1], 2 * primes[pi].p);
        return;
    }
    // (fusing two levels into one pass over memory measured slower on both compilers)
    for (unsigned lk = k; lk > 2; --lk) dif_level(a, m, lk, pi);
    dif_last2(a, m, pi);
}

// One radix-2 DIT level with the inverse roots (span 2^lk) over a[0..m).
inline void dit_level(u64* a, size_t m, unsigned lk, size_t pi) noexcept
{
    const u64 p = primes[pi].p, p2 = 2 * p;
    const size_t half = size_t{ 1 } << (lk - 1);
    u64 const* w = root_tables::level(pi, lk) + 2 * half;
    u64 const* wq = w + half;
    for (size_t s = 0; s < m; s += 2 * half) {
        u64* x = a + s;
        for (size_t j = 0; j < half; ++j) dit_bf(x[j], x[j + half], w[j], wq[j], p, p2);
    }
}

// The first two DIT levels (spans 2 and 4): the only root that isn't 1 is w4^-1.
inline void dit_first2(u64* a, size_t m, size_t pi) noexcept
{
    const u64 p = primes[pi].p, p2 = 2 * p;
    u64 const* w4 = root_tables::level(pi, 2);
    const u64 ii = w4[5], iiq = w4[7];
    for (size_t s = 0; s < m; s += 4) {
        u64* x = a + s;
        bf1(x[0], x[1], p2);
        bf1(x[2], x[3], p2);
        bf1(x[0], x[2], p2);
        dit_bf(x[1], x[3], ii, iiq, p, p2);
    }
}

// Radix-2 DIT with the inverse roots over a[0..2^k), k >= 1: bit-reversed in, natural order out,
// times 2^k.
inline void dit2_inverse(u64* a, unsigned k, size_t pi) noexcept
{
    const size_t m = size_t{ 1 } << k;
    if (k == 1) {
        bf1(a[0], a[1], 2 * primes[pi].p);
        return;
    }
    dit_first2(a, m, pi);
    for (unsigned lk = 3; lk <= k; ++lk) dit_level(a, m, lk, pi);
}

// The transform length for a product with n coefficients: the smallest 2^k or 3 * 2^k >= n.
// Returns {L, k, radix3} with L = 2^k, or L = 3 * 2^k when radix3.
struct length
{
    size_t L;
    unsigned k;
    bool radix3;
};

inline length choose_length(size_t n) noexcept
{
    unsigned k = 0;
    while ((size_t{ 1 } << k) < n) ++k;
    if (k >= 2 && (size_t{ 3 } << (k - 2)) >= n) return { size_t{ 3 } << (k - 2), k - 2, true };
    return { size_t{ 1 } << k, k, false };
}

// Forward transform of a[0..L), values in [0, 2p) in and out.
inline void forward(u64* a, length const& len, size_t pi) noexcept
{
    const size_t m = size_t{ 1 } << len.k;
    if (len.radix3) {
        const u64 p = primes[pi].p, p2 = 2 * p;
        u64 const* t = root_tables::radix3(pi, len.k);
        u64 const* w1 = t;
        u64 const* w1q = t + m;
        u64 const* w2 = t + 2 * m;
        u64 const* w2q = t + 3 * m;
        u64 const* e = t + 8 * m;
        const u64 e1 = e[0], e1q = e[1], e2 = e[2], e2q = e[3];
        u64* a0 = a;
        u64* a1 = a + m;
        u64* a2 = a + 2 * m;
        for (size_t j = 0; j < m; ++j) {
            const u64 x0 = a0[j], x1 = a1[j], x2 = a2[j];
            a0[j] = add2p(x0, add2p(x1, x2, p2), p2);
            const u64 y1 = add2p(add2p(x0, shoup_mul(x1, e1, e1q, p), p2), shoup_mul(x2, e2, e2q, p), p2);
            const u64 y2 = add2p(add2p(x0, shoup_mul(x1, e2, e2q, p), p2), shoup_mul(x2, e1, e1q, p), p2);
            a1[j] = shoup_mul(y1, w1[j], w1q[j], p);
            a2[j] = shoup_mul(y2, w2[j], w2q[j], p);
        }
        if (len.k) {
            dif2(a0, len.k, pi);
            dif2(a1, len.k, pi);
            dif2(a2, len.k, pi);
        }
    } else if (len.k) {
        dif2(a, len.k, pi);
    }
}

// Inverse of forward(), times L; values in [0, 2p) in and out.
inline void inverse(u64* a, length const& len, size_t pi) noexcept
{
    const size_t m = size_t{ 1 } << len.k;
    if (len.radix3) {
        const u64 p = primes[pi].p, p2 = 2 * p;
        u64* a0 = a;
        u64* a1 = a + m;
        u64* a2 = a + 2 * m;
        if (len.k) {
            dit2_inverse(a0, len.k, pi);
            dit2_inverse(a1, len.k, pi);
            dit2_inverse(a2, len.k, pi);
        }
        u64 const* t = root_tables::radix3(pi, len.k);
        u64 const* w1 = t + 4 * m;
        u64 const* w1q = t + 5 * m;
        u64 const* w2 = t + 6 * m;
        u64 const* w2q = t + 7 * m;
        u64 const* e = t + 8 * m;
        const u64 e1 = e[0], e1q = e[1], e2 = e[2], e2q = e[3]; // e1^-1 = e2
        for (size_t j = 0; j < m; ++j) {
            const u64 z0 = a0[j];
            const u64 z1 = shoup_mul(a1[j], w1[j], w1q[j], p);
            const u64 z2 = shoup_mul(a2[j], w2[j], w2q[j], p);
            a0[j] = add2p(add2p(z0, z1, p2), z2, p2);
            a1[j] = add2p(add2p(z0, shoup_mul(z1, e2, e2q, p), p2), shoup_mul(z2, e1, e1q, p), p2);
            a2[j] = add2p(add2p(z0, shoup_mul(z1, e1, e1q, p), p2), shoup_mul(z2, e2, e2q, p), p2);
        }
    } else if (len.k) {
        dit2_inverse(a, len.k, pi);
    }
}

}
