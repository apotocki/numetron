// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <tuple>
#include <concepts>
#include <type_traits>

#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/assert.hpp"
#include "numetron/arithmetic.hpp"
#include "numetron/config/implementation.hpp" // NUMETRON_FFT_IMPL
#include "toom/thresholds.hpp" // fft_threshold()

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

// v transformed once, mod every prime, at the transform length a pl x vn product takes, for
// multiplying many pl-limb pieces of u by it (umul_sliced over fft_slice_length()'s pieces): each
// piece then needs its own forward and the inverse transform only, two of the three. Holds
// NP transforms of v and NP of the piece (2 NP L words) from alloc for its lifetime; construct
// it before and destroy it after anything else that allocates from the same stack allocator.
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(sizeof(LimbT) == 8)
class fft_fixed_v
{
public:
#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
    static constexpr size_t NP = ntt_avx2::prime_count;
    using word = double;
#else
    static constexpr size_t NP = ntt::prime_count;
    using word = ntt::u64;
#endif
    static_assert(sizeof(word) == sizeof(LimbT));

    // pl even (a piece is whole coefficients)
    fft_fixed_v(LimbT const* v, size_t vn, size_t pl, AllocatorT alloc)
        : alloc_{ std::move(alloc) }, vn_{ vn }, pl_{ pl }
        , n_{ pl / 2 + (vn + 1) / 2 - 1 }, len_{ ntt::choose_length(n_) }
        , scratch_sz_{ 2 * NP * len_.L }
    {
        NUMETRON_ASSERT(vn >= 1 && pl >= vn && pl % 2 == 0);
        scratch_ = std::allocator_traits<AllocatorT>::allocate(alloc_, scratch_sz_);
        word* base = reinterpret_cast<word*>(scratch_);
        for (size_t pi = 0; pi < NP; ++pi) {
            vt_[pi] = base + pi * len_.L;
            r_[pi] = base + (NP + pi) * len_.L;
        }
        ntt::u64 const* vv = reinterpret_cast<ntt::u64 const*>(v);
        for (size_t pi = 0; pi < NP; ++pi) {
#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
            namespace nt = ntt_avx2;
            ntt::prime const& P = nt::primes[pi];
            const double p = static_cast<double>(P.p), pinv = 1.0 / p;
            nt::load(vt_[pi], vv, vn, len_.L, P, p, pinv);
            nt::forward(vt_[pi], len_, pi, p, pinv);
#else
            ntt::prime const& P = ntt::primes[pi];
            word* b = vt_[pi];
            const size_t c = ntt::load_coefficients(vv, vn, P, [b](size_t i, ntt::u64 x) { b[i] = x; });
            std::fill(b + c, b + len_.L, ntt::u64{ 0 });
            ntt::forward(b, len_, pi);
#endif
        }
    }

    fft_fixed_v(fft_fixed_v const&) = delete;
    fft_fixed_v& operator=(fft_fixed_v const&) = delete;

    ~fft_fixed_v()
    {
        std::allocator_traits<AllocatorT>::deallocate(alloc_, scratch_, scratch_sz_);
    }

    // dst[0..pl+vn) = a[0..pl) * v
    void mul(LimbT* dst, LimbT const* a) noexcept
    {
        ntt::u64 const* aa = reinterpret_cast<ntt::u64 const*>(a);
        ntt::u64* out = reinterpret_cast<ntt::u64*>(dst);
#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
        namespace nt = ntt_avx2;
        for (size_t pi = 0; pi < NP; ++pi) {
            ntt::prime const& P = nt::primes[pi];
            const double p = static_cast<double>(P.p), pinv = 1.0 / p;
            double* x = r_[pi];
            nt::load(x, aa, pl_, len_.L, P, p, pinv);
            nt::forward(x, len_, pi, p, pinv);
            nt::pointwise(x, vt_[pi], len_.L, p, pinv);
            nt::inverse(x, len_, pi, p, pinv);
            nt::finish(x, n_, static_cast<double>(P.p - (P.p - 1) / len_.L), p, pinv);
        }
        ntt::crt_accumulator<NP, nt::primes> acc;
        double const* const* r = r_;
        size_t i = 0;
        for (; i + 4 <= n_; i += 4) {
            ntt::u64 t[NP][4];
            nt::garner4(r, i, t);
            for (size_t q = 0; q < 4; ++q) {
                ntt::u64 d[NP];
                for (size_t s = 0; s < NP; ++s) d[s] = t[s][q];
                acc.push(d, out + 2 * (i + q));
            }
        }
        for (; i < n_; ++i) {
            ntt::u64 d[NP];
            nt::garner1(r, i, d);
            acc.push(d, out + 2 * i);
        }
        acc.flush(out, 2 * n_, pl_ + vn_);
#else
        for (size_t pi = 0; pi < NP; ++pi) {
            ntt::prime const& P = ntt::primes[pi];
            word* x = r_[pi];
            const size_t c = ntt::load_coefficients(aa, pl_, P, [x](size_t i, ntt::u64 y) { x[i] = y; });
            std::fill(x + c, x + len_.L, ntt::u64{ 0 });
            ntt::forward(x, len_, pi);
            word const* b = vt_[pi];
            for (size_t i = 0; i < len_.L; ++i) x[i] = ntt::mont_mul(x[i], b[i], P);
            ntt::inverse(x, len_, pi);
            // times 1 / (L * R), as in ntt::convolve()
            const ntt::u64 s = ntt::reduce(ntt::redc(0, P.p - (P.p - 1) / len_.L, P), P.p);
            const ntt::u64 sq = ntt::shoup_const(s, P.p);
            for (size_t i = 0; i < n_; ++i) x[i] = ntt::reduce(ntt::shoup_mul(x[i], s, sq, P.p), P.p);
        }
        ntt::u64 const* r[NP];
        for (size_t pi = 0; pi < NP; ++pi) r[pi] = r_[pi];
        ntt::crt_compose<NP, ntt::primes>(out, pl_ + vn_, r, n_);
#endif
    }

private:
    AllocatorT alloc_;
    size_t vn_, pl_, n_;
    ntt::length len_;
    size_t scratch_sz_;
    LimbT* scratch_ = nullptr;
    word* vt_[NP];
    word* r_[NP];
};

// Piece length for cutting u into FFT products of pl x vn limbs (umul_sliced) when that is
// cheaper than one transform over the whole un x vn, or 0. The transform length rounds the
// coefficient count up to 2^k or 3 * 2^(k-2), so one transform over a long u pays for up to a
// third more than it needs, while pieces can fill a length exactly (pl = 2 (L + 1 - ceil(vn/2))),
// v transformed once for all of them (fft_fixed_v). A product of n coefficients at transform
// length L is costed as L log2 L for its three transforms (forward u, forward v, inverse) + 6 n
// (loading, CRT: a piece's n exceeds its share of u by vn / 2); a piece pays two of the three
// transforms, v's third once. That picks the measured best piece length or one within a few
// percent of it (GCC, vn = 2688..16384, un/vn = 2..32; docs/fft.md); the pieces have to be
// clearly (3%) cheaper than the whole.
// Whether a square of n limbs (n >= sqr_fft_threshold()) takes the FFT. Its time is a staircase
// in the transform length (2^k or 3 * 2^(k-2), up to a third more than the coefficients need),
// the Toom chain's a smooth curve, so the FFT wins in the upper part of each length's range and
// loses just above each step. Measured (squares of 1920..6400 limbs, 2026-09-28, with the
// squaring basecase's straight-line code up to 32 limbs; the lowest winning fill of the length,
// coefficients / L, GCC / MSVC): 3072 1.00 / 1.00 (by 3% / 1%), 4096 0.86 / 0.89, 6144 0.77 /
// 0.84, 8192 from the start of its range (fill 0.76) on both. Hence a fill of 0.78 (GCC) / 0.84
// (MSVC) or more, and any fill from length 8192 up; sqr_fft_threshold() cuts off below the
// 4096 range. Measured with NUMETRON_USE_ASM only; the header-only builds, whose FFT starts far
// lower (293..1766 limbs), keep the single threshold.
inline bool fft_square_fills([[maybe_unused]] size_t n) noexcept
{
#if defined(NUMETRON_USE_ASM)
#   if defined(_MSC_VER) && !defined(__clang__)
    constexpr size_t min_fill_percent = 84;
#   else
    constexpr size_t min_fill_percent = 78;
#   endif
    const size_t c = (n + 1) / 2 * 2 - 1; // the coefficients of a square
    const size_t L = ntt::choose_length(c).L;
    return L >= 8192 || 100 * c >= min_fill_percent * L;
#else
    return true;
#endif
}

inline size_t fft_slice_length(size_t un, size_t vn) noexcept
{
    NUMETRON_ASSERT(un >= vn && vn >= 1);
    auto cost_of = [](size_t n) {
        const double L = static_cast<double>(ntt::choose_length(n).L);
        return L * std::log2(L) + 6.0 * static_cast<double>(n);
    };
    auto product_cost = [&](size_t an, size_t bn) { return cost_of((an + 1) / 2 + (bn + 1) / 2 - 1); };
    auto transforms = [](size_t L) { return static_cast<double>(L) * std::log2(static_cast<double>(L)); };

    double best = 0.97 * product_cost(un, vn);
    size_t best_pl = 0;
    for (unsigned k = 1; k < 8 * sizeof(size_t) - 2; ++k) {
        for (size_t L : { k >= 2 ? size_t{ 3 } << (k - 2) : size_t{ 0 }, size_t{ 1 } << k }) {
            if (L + 1 < (vn + 1) / 2) continue;
            const size_t pl = 2 * (L + 1 - (vn + 1) / 2);
            if (pl < vn) continue;
            if (pl >= un) return best_pl; // longer pieces: one product
            const size_t pieces = un / pl, rest = un - pieces * pl;
            // a remainder below the FFT threshold takes the Toom chain, which the model doesn't
            // cost (MSVC, vn = 4096, un = 3.5 vn: 3vn + 0.5vn was 1.02-1.04 of one transform)
            if (rest && rest < fft_threshold()) continue;
            // a piece fills its length: n = L
            double cost = transforms(L) / 3 + static_cast<double>(pieces) * (2 * transforms(L) / 3 + 6.0 * static_cast<double>(L));
            if (rest) cost += rest >= vn ? product_cost(rest, vn) : product_cost(vn, rest);
            if (cost < best) {
                best = cost;
                best_pl = pl;
            }
        }
    }
    return best_pl;
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
