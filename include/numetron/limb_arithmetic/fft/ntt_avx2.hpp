// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include <immintrin.h>

#include "ntt.hpp"

// The number-theoretic transform in double precision, four lanes at a time with AVX2 + FMA (the
// approach of FLINT's fft_small), for umul_fft.hpp. See docs/fft.md.
//
// Arithmetic. The primes are below 2^49 and every value is an integer held in a double, signed,
// |x| < 2p between operations (so far from the 2^53 limit of exact doubles):
//  - a * b mod p: h = a*b rounded, l = fma(a, b, -h) the exact rest (a*b = h + l),
//    q = round(h / p), r = fma(-q, p, h) + l. Both steps are exact (h - q*p and r are integers
//    well below 2^53), so r = a*b - q*p exactly, and with |a*b / p| <= 2^51 the rounded q is within
//    0.75 of a*b / p: |r| <= 1.25p. Used with |a| < 4p, |b| < 2p... one of them < p, or both < 2p.
//  - reduce(x) = x - round(x / p) * p, |result| <= p/2 (for |x| up to ~2^52).
// Sums are reduced after every butterfly, products are never reduced further; so every value
// stays below 2p in magnitude.
//
// Transforms: the same structure as ntt.hpp (radix-3 step for 3*2^k, radix-2 DIF forward, DIT
// inverse, no permutation). Levels with a span of 8 or more run four butterflies per vector; the
// last two DIF (first two DIT) levels, whose butterflies sit inside blocks of four, run on four
// blocks at a time transposed into four vectors. Transforms of fewer than 16 points use the same
// arithmetic on scalars.

namespace numetron::limb_arithmetic::detail::ntt_avx2 {

using ntt::u64;
using ntt::prime;

// The transform lengths go up to 3 * 2^max_log2.
inline constexpr unsigned max_log2 = 32;

// The largest primes p = c * 3 * 2^32 + 1 below 2^49, in descending order (scratch search as for
// ntt::primes). Their product is ~2^294 > L * 2^256 for L < 2^37: two limbs per coefficient.
inline constexpr prime primes[6] = {
    ntt::make_prime(0x0001fffe00000001ull, 13), // 43690 * 3 * 2^32 + 1
    ntt::make_prime(0x0001ff9200000001ull, 10), // 43654 * 3 * 2^32 + 1
    ntt::make_prime(0x0001ff8c00000001ull, 5),  // 43652 * 3 * 2^32 + 1
    ntt::make_prime(0x0001ff7700000001ull, 38), // 43645 * 3 * 2^32 + 1
    ntt::make_prime(0x0001ff6b00000001ull, 19), // 43641 * 3 * 2^32 + 1
    ntt::make_prime(0x0001ff5600000001ull, 5),  // 43634 * 3 * 2^32 + 1
};
inline constexpr size_t prime_count = 6;

// The root of order 2^k, or of order 3 * 2^k (radix3), from the chain G = g^((p-1) / (3*2^32)):
// the order-2^k root is the cube of the order-3*2^k one, as the radix-3 step needs.
inline u64 root(size_t pi, unsigned k, bool radix3) noexcept
{
    prime const& P = primes[pi];
    const u64 G = ntt::pow_mod(P.g, (P.p - 1) / (u64{ 3 } << max_log2), P);
    return ntt::pow_mod(G, (radix3 ? u64{ 1 } : u64{ 3 }) << (max_log2 - k), P);
}

// Root tables as doubles in [0, p), one per prime, kind and k, cached like ntt::root_tables:
//  - radix-2 level k: the half = 2^(k-1) roots w^j of the order-2^k root, then w^-j;
//  - radix-3 step for L = 3 * 2^k: w^j, w^2j, w^-j, w^-2j for j < m = 2^k (w of order 3m), then
//    the cube roots of unity e1 = w^m and e2 = w^2m = e1^-1.
class root_tables
{
public:
    static double const* level(size_t pi, unsigned k)
    {
        NUMETRON_ASSERT(pi < prime_count && k >= 1 && k <= max_log2);
        return get(pi, 0, k);
    }

    static double const* radix3(size_t pi, unsigned k)
    {
        NUMETRON_ASSERT(pi < prime_count && k <= max_log2);
        return get(pi, 1, k);
    }

private:
    struct prime_state
    {
        std::atomic<double const*> tables[2][max_log2 + 1] = {};
        std::unique_ptr<double[]> owned[2][max_log2 + 1];
        std::mutex mutex;
    };

    static prime_state& state(size_t pi)
    {
        static prime_state states[prime_count];
        return states[pi];
    }

    static double const* get(size_t pi, int kind, unsigned k)
    {
        auto& s = state(pi);
        if (double const* t = s.tables[kind][k].load(std::memory_order_acquire)) return t;
        std::lock_guard<std::mutex> lock(s.mutex);
        if (double const* t = s.tables[kind][k].load(std::memory_order_relaxed)) return t;
        std::unique_ptr<double[]> t = kind ? build_radix3(pi, k) : build_level(pi, k);
        double const* result = t.get();
        s.owned[kind][k] = std::move(t);
        s.tables[kind][k].store(result, std::memory_order_release);
        return result;
    }

    // r[j] = w^j for j < n
    static void powers(double* r, size_t n, u64 w, prime const& P)
    {
        u64 x = 1;
        for (size_t j = 0; j < n; ++j) {
            r[j] = static_cast<double>(x);
            x = ntt::mul_mod(x, w, P);
        }
    }

    static std::unique_ptr<double[]> build_level(size_t pi, unsigned k)
    {
        prime const& P = primes[pi];
        const size_t half = size_t{ 1 } << (k - 1);
        auto t = std::make_unique<double[]>(2 * half);
        const u64 w = root(pi, k, false);
        powers(t.get(), half, w, P);
        powers(t.get() + half, half, ntt::pow_mod(w, (u64{ 1 } << k) - 1, P), P);
        return t;
    }

    static std::unique_ptr<double[]> build_radix3(size_t pi, unsigned k)
    {
        prime const& P = primes[pi];
        const size_t m = size_t{ 1 } << k;
        auto t = std::make_unique<double[]>(4 * m + 2);
        const u64 w = root(pi, k, true);
        const u64 wi = ntt::pow_mod(w, 3 * m - 1, P);
        powers(t.get(), m, w, P);
        powers(t.get() + m, m, ntt::mul_mod(w, w, P), P);
        powers(t.get() + 2 * m, m, wi, P);
        powers(t.get() + 3 * m, m, ntt::mul_mod(wi, wi, P), P);
        const u64 e1 = ntt::pow_mod(w, m, P);
        t[4 * m] = static_cast<double>(e1);
        t[4 * m + 1] = static_cast<double>(ntt::mul_mod(e1, e1, P));
        return t;
    }
};

// ---- arithmetic -------------------------------------------------------------------------------

inline constexpr int round_nearest = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

NUMETRON_FORCEINLINE __m256d v_mul(__m256d a, __m256d b, __m256d p, __m256d pinv) noexcept
{
    const __m256d h = _mm256_mul_pd(a, b);
    const __m256d l = _mm256_fmsub_pd(a, b, h);
    const __m256d q = _mm256_round_pd(_mm256_mul_pd(h, pinv), round_nearest);
    return _mm256_add_pd(_mm256_fnmadd_pd(q, p, h), l);
}

NUMETRON_FORCEINLINE __m256d v_red(__m256d x, __m256d p, __m256d pinv) noexcept
{
    const __m256d q = _mm256_round_pd(_mm256_mul_pd(x, pinv), round_nearest);
    return _mm256_fnmadd_pd(q, p, x);
}

NUMETRON_FORCEINLINE double s_mul(double a, double b, double p, double pinv) noexcept
{
    const __m128d A = _mm_set_sd(a), B = _mm_set_sd(b);
    const __m128d h = _mm_mul_sd(A, B);
    const __m128d l = _mm_fmsub_sd(A, B, h);
    const __m128d q = _mm_round_sd(h, _mm_mul_sd(h, _mm_set_sd(pinv)), round_nearest);
    return _mm_cvtsd_f64(_mm_add_sd(_mm_fnmadd_sd(q, _mm_set_sd(p), h), l));
}

NUMETRON_FORCEINLINE double s_red(double x, double p, double pinv) noexcept
{
    const __m128d X = _mm_set_sd(x);
    const __m128d q = _mm_round_sd(X, _mm_mul_sd(X, _mm_set_sd(pinv)), round_nearest);
    return _mm_cvtsd_f64(_mm_fnmadd_sd(q, _mm_set_sd(p), X));
}

// c_k = element k of each of the four blocks r_0..r_3 (and back: the transpose is an involution)
NUMETRON_FORCEINLINE void transpose4(__m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3) noexcept
{
    const __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    const __m256d t1 = _mm256_unpackhi_pd(r0, r1);
    const __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    const __m256d t3 = _mm256_unpackhi_pd(r2, r3);
    r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

// ---- radix-2 ----------------------------------------------------------------------------------

// Radix-2 DIF over a[0..2^k): natural order in, bit-reversed out.
inline void dif2(double* a, unsigned k, size_t pi, double p, double pinv) noexcept
{
    const size_t m = size_t{ 1 } << k;
    if (k < 4) {
        for (unsigned lk = k; lk >= 1; --lk) {
            const size_t half = size_t{ 1 } << (lk - 1);
            double const* w = root_tables::level(pi, lk);
            for (size_t s = 0; s < m; s += 2 * half) {
                for (size_t j = 0; j < half; ++j) {
                    const double u = a[s + j], v = a[s + j + half];
                    a[s + j] = s_red(u + v, p, pinv);
                    a[s + j + half] = s_mul(u - v, w[j], p, pinv);
                }
            }
        }
        return;
    }
    const __m256d P = _mm256_set1_pd(p), PI = _mm256_set1_pd(pinv);
    for (unsigned lk = k; lk >= 3; --lk) {
        const size_t half = size_t{ 1 } << (lk - 1);
        double const* w = root_tables::level(pi, lk);
        for (size_t s = 0; s < m; s += 2 * half) {
            double* x = a + s;
            double* y = x + half;
            for (size_t j = 0; j < half; j += 4) {
                const __m256d u = _mm256_loadu_pd(x + j), v = _mm256_loadu_pd(y + j);
                _mm256_storeu_pd(x + j, v_red(_mm256_add_pd(u, v), P, PI));
                _mm256_storeu_pd(y + j, v_mul(_mm256_sub_pd(u, v), _mm256_loadu_pd(w + j), P, PI));
            }
        }
    }
    // spans 4 and 2: in each block of four, (0, 2) root 1, (1, 3) root i, then (0, 1), (2, 3)
    const __m256d I = _mm256_set1_pd(root_tables::level(pi, 2)[1]);
    for (size_t s = 0; s < m; s += 16) {
        __m256d c0 = _mm256_loadu_pd(a + s), c1 = _mm256_loadu_pd(a + s + 4);
        __m256d c2 = _mm256_loadu_pd(a + s + 8), c3 = _mm256_loadu_pd(a + s + 12);
        transpose4(c0, c1, c2, c3);
        const __m256d d0 = v_red(_mm256_add_pd(c0, c2), P, PI), d2 = v_red(_mm256_sub_pd(c0, c2), P, PI);
        const __m256d d1 = v_red(_mm256_add_pd(c1, c3), P, PI), d3 = v_mul(_mm256_sub_pd(c1, c3), I, P, PI);
        c0 = v_red(_mm256_add_pd(d0, d1), P, PI);
        c1 = v_red(_mm256_sub_pd(d0, d1), P, PI);
        c2 = v_red(_mm256_add_pd(d2, d3), P, PI);
        c3 = v_red(_mm256_sub_pd(d2, d3), P, PI);
        transpose4(c0, c1, c2, c3);
        _mm256_storeu_pd(a + s, c0);
        _mm256_storeu_pd(a + s + 4, c1);
        _mm256_storeu_pd(a + s + 8, c2);
        _mm256_storeu_pd(a + s + 12, c3);
    }
}

// Radix-2 DIT with the inverse roots over a[0..2^k): bit-reversed in, natural order out, times 2^k.
inline void dit2_inverse(double* a, unsigned k, size_t pi, double p, double pinv) noexcept
{
    const size_t m = size_t{ 1 } << k;
    if (k < 4) {
        for (unsigned lk = 1; lk <= k; ++lk) {
            const size_t half = size_t{ 1 } << (lk - 1);
            double const* w = root_tables::level(pi, lk) + half;
            for (size_t s = 0; s < m; s += 2 * half) {
                for (size_t j = 0; j < half; ++j) {
                    const double u = a[s + j];
                    const double t = s_mul(a[s + j + half], w[j], p, pinv);
                    a[s + j] = s_red(u + t, p, pinv);
                    a[s + j + half] = s_red(u - t, p, pinv);
                }
            }
        }
        return;
    }
    const __m256d P = _mm256_set1_pd(p), PI = _mm256_set1_pd(pinv);
    // spans 2 and 4: in each block of four, (0, 1), (2, 3) root 1, then (0, 2) root 1, (1, 3) root i^-1
    const __m256d II = _mm256_set1_pd(root_tables::level(pi, 2)[3]);
    for (size_t s = 0; s < m; s += 16) {
        __m256d c0 = _mm256_loadu_pd(a + s), c1 = _mm256_loadu_pd(a + s + 4);
        __m256d c2 = _mm256_loadu_pd(a + s + 8), c3 = _mm256_loadu_pd(a + s + 12);
        transpose4(c0, c1, c2, c3);
        const __m256d d0 = v_red(_mm256_add_pd(c0, c1), P, PI), d1 = v_red(_mm256_sub_pd(c0, c1), P, PI);
        const __m256d d2 = v_red(_mm256_add_pd(c2, c3), P, PI), d3 = v_red(_mm256_sub_pd(c2, c3), P, PI);
        const __m256d t = v_mul(d3, II, P, PI);
        c0 = v_red(_mm256_add_pd(d0, d2), P, PI);
        c2 = v_red(_mm256_sub_pd(d0, d2), P, PI);
        c1 = v_red(_mm256_add_pd(d1, t), P, PI);
        c3 = v_red(_mm256_sub_pd(d1, t), P, PI);
        transpose4(c0, c1, c2, c3);
        _mm256_storeu_pd(a + s, c0);
        _mm256_storeu_pd(a + s + 4, c1);
        _mm256_storeu_pd(a + s + 8, c2);
        _mm256_storeu_pd(a + s + 12, c3);
    }
    for (unsigned lk = 3; lk <= k; ++lk) {
        const size_t half = size_t{ 1 } << (lk - 1);
        double const* w = root_tables::level(pi, lk) + half;
        for (size_t s = 0; s < m; s += 2 * half) {
            double* x = a + s;
            double* y = x + half;
            for (size_t j = 0; j < half; j += 4) {
                const __m256d u = _mm256_loadu_pd(x + j);
                const __m256d t = v_mul(_mm256_loadu_pd(y + j), _mm256_loadu_pd(w + j), P, PI);
                _mm256_storeu_pd(x + j, v_red(_mm256_add_pd(u, t), P, PI));
                _mm256_storeu_pd(y + j, v_red(_mm256_sub_pd(u, t), P, PI));
            }
        }
    }
}

// ---- whole transforms -------------------------------------------------------------------------

// Forward transform of a[0..L) (ntt::length: L = 2^k or 3 * 2^k), values |x| < 2p in and out.
inline void forward(double* a, ntt::length const& len, size_t pi, double p, double pinv) noexcept
{
    const size_t m = size_t{ 1 } << len.k;
    if (len.radix3) {
        double const* t = root_tables::radix3(pi, len.k);
        double const* w1 = t;
        double const* w2 = t + m;
        const double e1 = t[4 * m], e2 = t[4 * m + 1];
        double* a0 = a;
        double* a1 = a + m;
        double* a2 = a + 2 * m;
        size_t j = 0;
        if (m >= 4) {
            const __m256d P = _mm256_set1_pd(p), PI = _mm256_set1_pd(pinv);
            const __m256d E1 = _mm256_set1_pd(e1), E2 = _mm256_set1_pd(e2);
            for (; j < m; j += 4) {
                const __m256d x0 = _mm256_loadu_pd(a0 + j), x1 = _mm256_loadu_pd(a1 + j), x2 = _mm256_loadu_pd(a2 + j);
                _mm256_storeu_pd(a0 + j, v_red(_mm256_add_pd(x0, _mm256_add_pd(x1, x2)), P, PI));
                const __m256d y1 = v_red(_mm256_add_pd(x0, _mm256_add_pd(v_mul(x1, E1, P, PI), v_mul(x2, E2, P, PI))), P, PI);
                const __m256d y2 = v_red(_mm256_add_pd(x0, _mm256_add_pd(v_mul(x1, E2, P, PI), v_mul(x2, E1, P, PI))), P, PI);
                _mm256_storeu_pd(a1 + j, v_mul(y1, _mm256_loadu_pd(w1 + j), P, PI));
                _mm256_storeu_pd(a2 + j, v_mul(y2, _mm256_loadu_pd(w2 + j), P, PI));
            }
        }
        for (; j < m; ++j) {
            const double x0 = a0[j], x1 = a1[j], x2 = a2[j];
            a0[j] = s_red(x0 + x1 + x2, p, pinv);
            const double y1 = s_red(x0 + s_mul(x1, e1, p, pinv) + s_mul(x2, e2, p, pinv), p, pinv);
            const double y2 = s_red(x0 + s_mul(x1, e2, p, pinv) + s_mul(x2, e1, p, pinv), p, pinv);
            a1[j] = s_mul(y1, w1[j], p, pinv);
            a2[j] = s_mul(y2, w2[j], p, pinv);
        }
        dif2(a0, len.k, pi, p, pinv);
        dif2(a1, len.k, pi, p, pinv);
        dif2(a2, len.k, pi, p, pinv);
    } else {
        dif2(a, len.k, pi, p, pinv);
    }
}

// Inverse of forward(), times L; values |x| < 2p in and out.
inline void inverse(double* a, ntt::length const& len, size_t pi, double p, double pinv) noexcept
{
    const size_t m = size_t{ 1 } << len.k;
    if (len.radix3) {
        double* a0 = a;
        double* a1 = a + m;
        double* a2 = a + 2 * m;
        dit2_inverse(a0, len.k, pi, p, pinv);
        dit2_inverse(a1, len.k, pi, p, pinv);
        dit2_inverse(a2, len.k, pi, p, pinv);
        double const* t = root_tables::radix3(pi, len.k);
        double const* w1 = t + 2 * m;
        double const* w2 = t + 3 * m;
        const double e1 = t[4 * m], e2 = t[4 * m + 1]; // e1^-1 = e2
        size_t j = 0;
        if (m >= 4) {
            const __m256d P = _mm256_set1_pd(p), PI = _mm256_set1_pd(pinv);
            const __m256d E1 = _mm256_set1_pd(e1), E2 = _mm256_set1_pd(e2);
            for (; j < m; j += 4) {
                const __m256d z0 = _mm256_loadu_pd(a0 + j);
                const __m256d z1 = v_mul(_mm256_loadu_pd(a1 + j), _mm256_loadu_pd(w1 + j), P, PI);
                const __m256d z2 = v_mul(_mm256_loadu_pd(a2 + j), _mm256_loadu_pd(w2 + j), P, PI);
                _mm256_storeu_pd(a0 + j, v_red(_mm256_add_pd(z0, _mm256_add_pd(z1, z2)), P, PI));
                _mm256_storeu_pd(a1 + j, v_red(_mm256_add_pd(z0, _mm256_add_pd(v_mul(z1, E2, P, PI), v_mul(z2, E1, P, PI))), P, PI));
                _mm256_storeu_pd(a2 + j, v_red(_mm256_add_pd(z0, _mm256_add_pd(v_mul(z1, E1, P, PI), v_mul(z2, E2, P, PI))), P, PI));
            }
        }
        for (; j < m; ++j) {
            const double z0 = a0[j];
            const double z1 = s_mul(a1[j], w1[j], p, pinv);
            const double z2 = s_mul(a2[j], w2[j], p, pinv);
            a0[j] = s_red(z0 + z1 + z2, p, pinv);
            a1[j] = s_red(z0 + s_mul(z1, e2, p, pinv) + s_mul(z2, e1, p, pinv), p, pinv);
            a2[j] = s_red(z0 + s_mul(z1, e1, p, pinv) + s_mul(z2, e2, p, pinv), p, pinv);
        }
    } else {
        dit2_inverse(a, len.k, pi, p, pinv);
    }
}

// a[i] = a[i] * b[i] (or a[i]^2 when b is null) mod p for i < L.
inline void pointwise(double* a, double const* b, size_t L, double p, double pinv) noexcept
{
    size_t i = 0;
    const __m256d P = _mm256_set1_pd(p), PI = _mm256_set1_pd(pinv);
    if (b) {
        for (; i + 4 <= L; i += 4) _mm256_storeu_pd(a + i, v_mul(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), P, PI));
        for (; i < L; ++i) a[i] = s_mul(a[i], b[i], p, pinv);
    } else {
        for (; i + 4 <= L; i += 4) {
            const __m256d x = _mm256_loadu_pd(a + i);
            _mm256_storeu_pd(a + i, v_mul(x, x, P, PI));
        }
        for (; i < L; ++i) a[i] = s_mul(a[i], a[i], p, pinv);
    }
}

// ---- operands in, residues out ----------------------------------------------------------------

// 2^52: a double in [0, 2^52) plus this has exactly its value as the mantissa bits, and the
// other way round.
inline constexpr double two52 = 4503599627370496.0;
inline constexpr u64 two52_bits = 0x4330000000000000ull;

// x mod p for |x| < 2p (after v_red: |x| <= p/2), into [0, p)
NUMETRON_FORCEINLINE __m256d v_normalize(__m256d x, __m256d p, __m256d pinv) noexcept
{
    x = v_red(x, p, pinv);
    return _mm256_add_pd(x, _mm256_and_pd(_mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ), p));
}

// The 2-limb coefficients of an n-limb operand mod p, |x| <= p/2, into dst[0..ceil(n/2)),
// dst[ceil(n/2)..L) = 0. Each limb is split into 32-bit halves (exact as doubles), and a
// coefficient lo + hi * 2^64 is lo_0 + lo_1 * 2^32 + hi_0 * 2^64 + hi_1 * 2^96 with the powers of
// two taken mod p; four coefficients (eight limbs) per step.
inline void load(double* dst, u64 const* src, size_t n, size_t L, prime const& P, double p, double pinv) noexcept
{
    const u64 c32 = u64{ 1 } << 32;                  // < p
    const u64 c64 = (~u64{ 0 } % P.p + 1) % P.p;
    const u64 c96 = ntt::mul_mod(c64, c32, P);
    const double d32 = static_cast<double>(c32), d64 = static_cast<double>(c64), d96 = static_cast<double>(c96);
    size_t c = 0, i = 0;
    {
        const __m256d Pv = _mm256_set1_pd(p), PI = _mm256_set1_pd(pinv);
        // lanes of four limbs (lo, hi, lo, hi): low halves times A, high halves times B
        const __m256d A = _mm256_setr_pd(1.0, d64, 1.0, d64);
        const __m256d B = _mm256_setr_pd(d32, d96, d32, d96);
        const __m256i mask32 = _mm256_set1_epi64x(0xffffffffll);
        const __m256i magic_i = _mm256_set1_epi64x(static_cast<long long>(two52_bits));
        const __m256d magic = _mm256_set1_pd(two52);
        auto to_double = [&](__m256i x) { return _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(x, magic_i)), magic); };
        auto limbs = [&](u64 const* s) {
            const __m256i l = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(s));
            return _mm256_add_pd(v_mul(to_double(_mm256_and_si256(l, mask32)), A, Pv, PI),
                v_mul(to_double(_mm256_srli_epi64(l, 32)), B, Pv, PI));
        };
        for (; i + 8 <= n; i += 8, c += 4) {
            // hadd: (c0, c2, c1, c3), then back in order
            const __m256d s = _mm256_hadd_pd(limbs(src + i), limbs(src + i + 4));
            _mm256_storeu_pd(dst + c, v_red(_mm256_permute4x64_pd(s, 0xD8), Pv, PI));
        }
    }
    for (; i < n; i += 2, ++c) {
        const u64 lo = src[i], hi = i + 1 < n ? src[i + 1] : 0;
        const double x = static_cast<double>(lo & 0xffffffffu) + s_mul(static_cast<double>(lo >> 32), d32, p, pinv)
            + s_mul(static_cast<double>(hi & 0xffffffffu), d64, p, pinv) + s_mul(static_cast<double>(hi >> 32), d96, p, pinv);
        dst[c] = s_red(x, p, pinv);
    }
    for (; c < L; ++c) dst[c] = 0.0;
}

// a[i] = a[i] * s mod p in [0, p), for i < n.
inline void finish(double* a, size_t n, double s, double p, double pinv) noexcept
{
    size_t i = 0;
    const __m256d P = _mm256_set1_pd(p), PI = _mm256_set1_pd(pinv), S = _mm256_set1_pd(s);
    for (; i + 4 <= n; i += 4) _mm256_storeu_pd(a + i, v_normalize(v_mul(_mm256_loadu_pd(a + i), S, P, PI), P, PI));
    for (; i < n; ++i) {
        double y = s_red(s_mul(a[i], s, p, pinv), p, pinv);
        a[i] = y < 0 ? y + p : y;
    }
}

// Garner's constants: p_j^-1 mod p_s (j < s) as doubles.
struct garner_constants
{
    double inv[prime_count][prime_count];

    static garner_constants const& get() noexcept
    {
        static const garner_constants c = [] {
            garner_constants k{};
            for (size_t s = 1; s < prime_count; ++s) {
                prime const& Ps = primes[s];
                for (size_t j = 0; j < s; ++j) {
                    k.inv[s][j] = static_cast<double>(ntt::pow_mod(primes[j].p % Ps.p, Ps.p - 2, Ps));
                }
            }
            return k;
        }();
        return c;
    }
};

// The mixed-radix digits t[s] (x = t0 + p0 (t1 + p1 (t2 + ...)), t_s < p_s) of the coefficients
// i .. i+3 from their residues r[s][i..i+4) in [0, p_s): t[s][q] for coefficient i + q.
inline void garner4(double const* const* r, size_t i, u64 (*t)[4]) noexcept
{
    garner_constants const& k = garner_constants::get();
    const __m256i magic_i = _mm256_set1_epi64x(static_cast<long long>(two52_bits));
    const __m256d magic = _mm256_set1_pd(two52);
    __m256d T[prime_count];
    T[0] = _mm256_loadu_pd(r[0] + i);
    for (size_t s = 1; s < prime_count; ++s) {
        const double ps = static_cast<double>(primes[s].p);
        const __m256d P = _mm256_set1_pd(ps), PI = _mm256_set1_pd(1.0 / ps);
        __m256d x = _mm256_loadu_pd(r[s] + i);
        for (size_t j = 0; j < s; ++j) {
            // |x - t_j| < 1.25 p_s + p_j < 4 p_s
            x = v_mul(_mm256_sub_pd(x, T[j]), _mm256_set1_pd(k.inv[s][j]), P, PI);
        }
        T[s] = v_normalize(x, P, PI);
    }
    for (size_t s = 0; s < prime_count; ++s) {
        const __m256i bits = _mm256_xor_si256(_mm256_castpd_si256(_mm256_add_pd(T[s], magic)), magic_i);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(t[s]), bits);
    }
}

// garner4() for the one coefficient i.
inline void garner1(double const* const* r, size_t i, u64* t) noexcept
{
    garner_constants const& k = garner_constants::get();
    double T[prime_count];
    T[0] = r[0][i];
    for (size_t s = 1; s < prime_count; ++s) {
        const double ps = static_cast<double>(primes[s].p), psinv = 1.0 / ps;
        double x = r[s][i];
        for (size_t j = 0; j < s; ++j) x = s_mul(x - T[j], k.inv[s][j], ps, psinv);
        x = s_red(x, ps, psinv);
        T[s] = x < 0 ? x + ps : x;
    }
    for (size_t s = 0; s < prime_count; ++s) t[s] = static_cast<u64>(T[s]);
}

}
