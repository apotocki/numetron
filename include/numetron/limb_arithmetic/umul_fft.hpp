// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <algorithm>
#include <memory>
#include <span>
#include <tuple>
#include <concepts>
#include <type_traits>

#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/assert.hpp"
#include "numetron/arithmetic.hpp"
#include "numetron/config/implementation.hpp" // NUMETRON_FFT_IMPL

#include "fft/ntt.hpp"
#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
#   include "fft/ntt_avx2.hpp"
#endif

// FFT multiplication: a multi-prime number-theoretic transform (docs/fft.md). Every two limbs of
// an operand are one coefficient; the product's coefficients (< min(cu, cv) * 2^256) are computed
// modulo several word-size primes by transform - pointwise product - inverse transform, recovered
// exactly by CRT (Garner) and summed into the result with a running carry. Two transform kernels
// (NUMETRON_FFT_IMPL): the portable scalar one over five 62-bit primes (fft/ntt.hpp), and the
// AVX2 + FMA one in double precision over six 49-bit primes (fft/ntt_avx2.hpp). (One limb per
// coefficient with three primes was measured 9-18% slower at every size.)

namespace numetron::limb_arithmetic {

namespace detail::ntt {

// The 2-limb coefficients of an n-limb operand in Montgomery form mod P into dst[0..ceil(n/2)),
// in [0, 2p): lo + hi * 2^64 becomes lo * R + hi * R^2 = redc(lo * r2) + redc(hi * r3).
// store(i, x) receives each one; returns the number of coefficients.
template <typename StoreF>
NUMETRON_FORCEINLINE size_t load_coefficients(u64 const* src, size_t n, prime const& P, StoreF&& store) noexcept
{
    const u64 p2 = 2 * P.p;
    size_t c = 0;
    for (; 2 * c + 1 < n; ++c) {
        auto [h0, l0] = arithmetic::umul1(src[2 * c], P.r2);
        auto [h1, l1] = arithmetic::umul1(src[2 * c + 1], P.r3);
        store(c, add2p(redc(h0, l0, P), redc(h1, l1, P), p2));
    }
    if (2 * c < n) { // odd n: the top coefficient has one limb
        auto [h, l] = arithmetic::umul1(src[2 * c], P.r2);
        store(c++, redc(h, l, P));
    }
    return c;
}

// The scalar kernel: the cyclic convolution of u and v (or of u with itself when v is null) mod
// primes[pi] into a[0..nc): the first nc coefficients of the product, reduced to [0, p). b is
// L words of scratch (unused when squaring).
inline void convolve(u64* a, u64* b, u64 const* u, size_t un, u64 const* v, size_t vn, size_t nc, length const& len, size_t pi) noexcept
{
    prime const& P = primes[pi];
    auto load = [&](u64* dst, u64 const* src, size_t n) {
        const size_t c = load_coefficients(src, n, P, [dst](size_t i, u64 x) { dst[i] = x; });
        std::fill(dst + c, dst + len.L, u64{ 0 });
    };
    load(a, u, un);
    forward(a, len, pi);
    if (v) {
        load(b, v, vn);
        forward(b, len, pi);
        for (size_t i = 0; i < len.L; ++i) a[i] = mont_mul(a[i], b[i], P);
    } else {
        for (size_t i = 0; i < len.L; ++i) a[i] = mont_mul(a[i], a[i], P);
    }
    inverse(a, len, pi);
    // times 1 / (L * R): the transform's L and the Montgomery R of the pointwise product.
    // L^-1 = p - (p - 1) / L (as L * (p - 1) / L = -1), and redc(x) = x / R.
    const u64 s = reduce(redc(0, P.p - (P.p - 1) / len.L, P), P.p);
    const u64 sq = shoup_const(s, P.p);
    for (size_t i = 0; i < nc; ++i) a[i] = reduce(shoup_mul(a[i], s, sq, P.p), P.p);
}

// Garner's constants for a set of NP primes (in descending order, each within a factor 2 of the
// others): p_j^-1 mod p_i (j < i) with Shoup companions.
template <size_t NP, prime const* Primes>
struct crt_constants
{
    u64 inv[NP][NP], invq[NP][NP];

    static crt_constants const& get() noexcept
    {
        static const crt_constants c = [] {
            crt_constants k{};
            for (size_t i = 1; i < NP; ++i) {
                prime const& Pi = Primes[i];
                for (size_t j = 0; j < i; ++j) {
                    k.inv[i][j] = pow_mod(Primes[j].p % Pi.p, Pi.p - 2, Pi);
                    k.invq[i][j] = shoup_const(k.inv[i][j], Pi.p);
                }
            }
            return k;
        }();
        return c;
    }
};

// Sums the coefficients x_i * B^(2i) into the result from their mixed-radix digits
// (x = t0 + p0 (t1 + p1 (t2 + ...)), t_s < p_s), two limbs out per coefficient. The product of
// the primes must be below 2^320 and above every x_i (< min(cu, cv) * 2^256).
template <size_t NP, prime const* Primes>
struct crt_accumulator
{
    static_assert(NP >= 5 && NP <= 6);

    // running carry: x_i < 2^320 minus a bit, so after two limbs out it stays below 2^192 -- 3 limbs
    u64 c0 = 0, c1 = 0, c2 = 0;

    // out[0..2) = the next two limbs of the result
    NUMETRON_FORCEINLINE void push(u64 const* t, u64* out) noexcept
    {
        // Horner: y = t_last; y = y * p_s + t_s for s = NP-2 .. 0
        u64 y[NP] = { t[NP - 1] };
        for (size_t s = NP - 1, yl = 1; s-- > 0; ++yl) {
            u64 carry = t[s];
            for (size_t l = 0; l < yl; ++l) {
                auto [h, lo] = arithmetic::umul1(y[l], Primes[s].p);
                unsigned char cy = 0;
                y[l] = arithmetic::uadd1c(lo, carry, cy);
                carry = h + cy;
            }
            y[yl] = carry;
        }
        if constexpr (NP > 5) NUMETRON_ASSERT(y[5] == 0); // x < 2^320
        // (carry + x): emit two limbs, keep three
        unsigned char cy = 0;
        out[0] = arithmetic::uadd1c(y[0], c0, cy);
        out[1] = arithmetic::uadd1c(y[1], c1, cy);
        c0 = arithmetic::uadd1c(y[2], c2, cy);
        c1 = arithmetic::uadd1c(y[3], u64{ 0 }, cy);
        c2 = y[4] + cy;
    }

    // rb[o..rn) = the rest of the carry, rn - o <= 2
    void flush(u64* rb, size_t o, size_t rn) noexcept
    {
        if (o < rn) rb[o++] = c0, c0 = c1, c1 = c2, c2 = 0;
        if (o < rn) rb[o++] = c0, c0 = c1, c1 = c2, c2 = 0;
        NUMETRON_ASSERT(o == rn && c0 == 0 && c1 == 0 && c2 == 0); // the product fits in rn limbs
    }
};

// rb[0..rn) = sum over i < nc of x_i * B^(2i), x_i the CRT of the residues r[0..NP)[i] (each in
// [0, p)); 2 * nc <= rn <= 2 * nc + 2.
template <size_t NP, prime const* Primes>
inline void crt_compose(u64* rb, size_t rn, u64 const* const* r, size_t nc) noexcept
{
    auto const& k = crt_constants<NP, Primes>::get();
    crt_accumulator<NP, Primes> acc;
    for (size_t i = 0; i < nc; ++i) {
        u64 t[NP];
        t[0] = r[0][i];
        for (size_t s = 1; s < NP; ++s) {
            const u64 ps = Primes[s].p;
            u64 x = r[s][i];                                    // [0, p_s), then [0, 2 p_s)
            for (size_t j = 0; j < s; ++j) {
                const u64 tj = t[j] >= ps ? t[j] - ps : t[j];   // p_j > p_s: t_j < 2 p_s
                x = shoup_mul(x - tj + ps, k.inv[s][j], k.invq[s][j], ps);
            }
            t[s] = reduce(x, ps);
        }
        acc.push(t, rb + 2 * i);
    }
    acc.flush(rb, 2 * nc, rn);
}

} // namespace detail::ntt

namespace detail {

// FFT multiplication cores: rb[0..un+vn) = u * v, un >= vn >= 1, rb not overlapping u, v.
// Square with one forward transform per prime when u and v are the same operand. Return
// rb + un + vn. n = ceil(un/2) + ceil(vn/2) - 1 coefficients, L the transform length,
// n <= L <= 4/3 n.

// Scalar, five 62-bit primes. Scratch 2L + 4n limbs.
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(sizeof(LimbT) == 8)
LimbT* umul_fft_scalar_impl(LimbT const* u, size_t un, LimbT const* v, size_t vn, LimbT* rb, AllocatorT alloc)
{
    using namespace ntt;
    constexpr size_t NP = 5;
    NUMETRON_ASSERT(un >= vn && vn >= 1);

    const size_t n = (un + 1) / 2 + (vn + 1) / 2 - 1;
    const length len = choose_length(n);
    const bool square = u == v && un == vn;

    const size_t scratch_sz = 2 * len.L + (NP - 1) * n;
    LimbT* scratch = std::allocator_traits<AllocatorT>::allocate(alloc, scratch_sz);
    NUMETRON_SCOPE_EXIT([&] {
        std::allocator_traits<AllocatorT>::deallocate(alloc, scratch, scratch_sz);
    });
    u64* a = reinterpret_cast<u64*>(scratch);   // L: transform, then the residues mod the last prime
    u64* b = a + len.L;                         // L: the second operand's transform
    u64* r[NP];                                 // n each: the residues mod the other primes
    for (size_t pi = 0; pi + 1 < NP; ++pi) r[pi] = b + len.L + pi * n;
    r[NP - 1] = a;

    u64 const* uu = reinterpret_cast<u64 const*>(u);
    u64 const* vv = square ? nullptr : reinterpret_cast<u64 const*>(v);
    for (size_t pi = 0; pi < NP; ++pi) {
        convolve(a, b, uu, un, vv, vn, n, len, pi);
        if (pi + 1 < NP) std::copy(a, a + n, r[pi]);
    }
    crt_compose<NP, primes>(reinterpret_cast<u64*>(rb), un + vn, r, n);
    return rb + un + vn;
}

#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2

// AVX2 + FMA in double precision, six 49-bit primes. Scratch L + NP * max(L, n) doubles.
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(sizeof(LimbT) == 8)
LimbT* umul_fft_avx2_impl(LimbT const* u, size_t un, LimbT const* v, size_t vn, LimbT* rb, AllocatorT alloc)
{
    namespace nt = ntt_avx2;
    using ntt::u64;
    constexpr size_t NP = nt::prime_count;
    NUMETRON_ASSERT(un >= vn && vn >= 1);

    const size_t n = (un + 1) / 2 + (vn + 1) / 2 - 1;
    const ntt::length len = ntt::choose_length(n);
    NUMETRON_ASSERT(len.k <= nt::max_log2);
    const bool square = u == v && un == vn;

    // r[pi]: the transform of u mod primes[pi], then its residues (the first n values); b: v's
    const size_t scratch_sz = (NP + 1) * len.L;
    LimbT* scratch = std::allocator_traits<AllocatorT>::allocate(alloc, scratch_sz);
    NUMETRON_SCOPE_EXIT([&] {
        std::allocator_traits<AllocatorT>::deallocate(alloc, scratch, scratch_sz);
    });
    double* base = reinterpret_cast<double*>(scratch);
    double* b = base + NP * len.L;
    double* r[NP];
    for (size_t pi = 0; pi < NP; ++pi) r[pi] = base + pi * len.L;

    u64 const* uu = reinterpret_cast<u64 const*>(u);
    u64 const* vv = reinterpret_cast<u64 const*>(v);
    for (size_t pi = 0; pi < NP; ++pi) {
        ntt::prime const& P = nt::primes[pi];
        const double p = static_cast<double>(P.p), pinv = 1.0 / p;
        double* a = r[pi];
        nt::load(a, uu, un, len.L, P, p, pinv);
        nt::forward(a, len, pi, p, pinv);
        if (!square) {
            nt::load(b, vv, vn, len.L, P, p, pinv);
            nt::forward(b, len, pi, p, pinv);
        }
        nt::pointwise(a, square ? nullptr : b, len.L, p, pinv);
        nt::inverse(a, len, pi, p, pinv);
        // times 1 / L: L^-1 = p - (p - 1) / L (as L * (p - 1) / L = -1)
        nt::finish(a, n, static_cast<double>(P.p - (P.p - 1) / len.L), p, pinv);
    }

    ntt::crt_accumulator<NP, nt::primes> acc;
    u64* out = reinterpret_cast<u64*>(rb);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        u64 t[NP][4];
        nt::garner4(r, i, t);
        for (size_t q = 0; q < 4; ++q) {
            u64 d[NP];
            for (size_t s = 0; s < NP; ++s) d[s] = t[s][q];
            acc.push(d, out + 2 * (i + q));
        }
    }
    for (; i < n; ++i) {
        u64 d[NP];
        nt::garner1(r, i, d);
        acc.push(d, out + 2 * i);
    }
    acc.flush(out, 2 * n, un + vn);
    return rb + un + vn;
}

#endif

// The FFT core the dispatch uses (NUMETRON_FFT_IMPL).
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(sizeof(LimbT) == 8)
LimbT* umul_fft_impl(LimbT const* u, size_t un, LimbT const* v, size_t vn, LimbT* rb, AllocatorT alloc)
{
#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
    return umul_fft_avx2_impl(u, un, v, vn, rb, std::move(alloc));
#else
    return umul_fft_scalar_impl(u, un, v, vn, rb, std::move(alloc));
#endif
}

} // namespace detail

// FFT multiplication. Preconditions: un >= vn > 0.
// Allocates the result buffer via alloc, the scratch via scratch_alloc. Returns {ptr, size, capacity}.
template <std::unsigned_integral LimbT, typename AllocatorT, typename ScratchAllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type> && sizeof(LimbT) == 8)
inline std::tuple<LimbT*, size_t, size_t> umul_fft(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc, ScratchAllocatorT scratch_alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();
    NUMETRON_ASSERT(un > 0 && vn > 0 && un >= vn);

    const size_t alloc_sz = un + vn;
    LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
    try {
        LimbT* re = detail::umul_fft_impl(u.data(), un, v.data(), vn, rb, scratch_alloc);
        while (re != rb && *(re - 1) == 0) --re;
        return { rb, static_cast<size_t>(re - rb), alloc_sz };
    }
    catch (...) {
        std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
        throw;
    }
}

}
