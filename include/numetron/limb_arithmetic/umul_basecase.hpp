// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <tuple>
#include <atomic>

#include "platform.hpp"
#include "umul1.hpp"
#include "umul_basecase_variants.hpp" // the C++ basecase alternatives (NUMETRON_CXX_BASECASE)
#include "toom/thresholds.hpp"          // sqr_basecase_threshold()

namespace numetron::limb_arithmetic {

// Closed-form (loop-free) multiply of a 2-limb u = (u1:u0) by a v that is either 1 limb (v0
// alone, pass v1 == 0) or 2 limbs (v1:v0, v1 != 0). v1 == 0 doubles as "v has no high limb"
// instead of "v has one and it happens to be zero" -- both mean the exact same product (a zero
// high limb contributes nothing), the only difference being whether the result comes back
// pre-trimmed to 3 limbs or written out to the full 4 with a trailing zero. Both of this
// function's callers are fine with that: umul_basecase_unrolled() below already returns a
// possibly-shorter-than-un+vn size elsewhere, and limb_arithmetic::mul()'s <=2-limb short path
// trims trailing zero limbs itself regardless.
// This is the dedicated ucnt==2 case of umul_basecase_unrolled() below, pulled out so it can
// also be called directly -- with scalar operands rather than a {pointer, count} pair -- from
// limb_arithmetic::mul()'s own <=2-limb short path, which needs to feed it top-limb-masked
// values without materializing a temporary buffer just to satisfy a span/pointer interface.
// Returns one past the last limb written: rb+3 when v1 == 0, rb+4 otherwise.
template <std::unsigned_integral LimbT>
inline LimbT* umul_basecase_2x(LimbT u0, LimbT u1, LimbT v0, LimbT v1, LimbT* rb) noexcept
{
    auto [h00, l00] = arithmetic::umul1(u0, v0);
    *rb = l00;
    auto [h01, l01] = arithmetic::umul1(u1, v0);
    LimbT r1 = arithmetic::uadd1ca(l01, h00, h01);
    if (!v1) {
        *(rb + 1) = r1;
        *(rb + 2) = h01;
        return rb + 3;
    }
    auto [h10, l10] = arithmetic::umul1(u0, v1);
    *(rb + 1) = arithmetic::uadd1ca(r1, l10, h10);
    auto [h11, l11] = arithmetic::umul1(u1, v1);
    h10 = arithmetic::uadd1ca(h10, h01, h11);
    *(rb + 2) = arithmetic::uadd1ca(l11, h10, h11); // h11 can not overflow: 4 limbs are always enough to hold a 2x2-limb product
    *(rb + 3) = h11;
    return rb + 4;
}

// Perhaps one day, a C++ compiler will be able to optimise this to the same extent as hand-written assembly code.
template <std::unsigned_integral LimbT>
inline LimbT* umul_basecase_unrolled(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT* rb) noexcept
{
    const auto ucnt = ue - ub;
    const auto vcnt = ve - vb;
    if (ucnt == 2) {
        return umul_basecase_2x<LimbT>(*ub, *(ub + 1), *vb, vcnt >= 2 ? *(vb + 1) : LimbT{0}, rb);
    }

    auto [h0, l0] = arithmetic::umul1(*ub, *vb);
    auto [h1, l1] = arithmetic::umul1(*(ub + 1), *vb);
    auto [h2, l2] = arithmetic::umul1(*(ub + 2), *vb);
    
    if (ucnt & 1) {
        if (ucnt & 2) {
            for (auto un = ucnt; un != 3; un -= 4) { // ucnt = 7, 11, 15, ...
                *rb = l0;
                auto [h3, l3] = arithmetic::umul1(*(ub + 3), *vb);
                *(rb + 1) = arithmetic::uadd1ca(h0, l1, h1);
                std::tie(h0, l0) = arithmetic::umul1(*(ub + 4), *vb);
                *(rb + 2) = arithmetic::uadd1ca(h1, l2, h2);
                std::tie(h1, l1) = arithmetic::umul1(*(ub + 5), *vb);
                *(rb + 3) = arithmetic::uadd1ca(h2, l3, h3);
                std::tie(h2, l2) = arithmetic::umul1(*(ub + 6), *vb);
                l0 = arithmetic::uadd1ca(h3, l0, h0);
                ub += 4; rb += 4; 
            }

            *rb = l0;
            *(rb + 1) = arithmetic::uadd1ca(h0, l1, h1);
            *(rb + 2) = arithmetic::uadd1ca(h1, l2, h2);
            *(rb + 3) = h2;
            while (++vb != ve) {
                rb += 4 - ucnt;
                ub += 3 - ucnt;
                
                auto [h0, l0] = arithmetic::umul1(*ub, *vb);
                auto [h1, l1] = arithmetic::umul1(*(ub + 1), *vb);
                auto [h2, l2] = arithmetic::umul1(*(ub + 2), *vb);
                auto [cov, l0_] = arithmetic::uadd1(*rb, l0);

                for (auto un = ucnt; un != 3; un -= 4) { // ucnt = 7, 11, 15, ...
                    *rb = l0_;
                    l0_ = arithmetic::umul4add<LimbT>(ub + 3, *vb, rb + 1, cov, h0, h1, l1, h2, l2);
                    ub += 4; rb += 4;
                }
                *rb = l0_;
                l1 = arithmetic::uadd1ca(h0, l1, h1);
                *(rb + 1) = arithmetic::uadd1c(*(rb + 1), l1, cov);
                l2 = arithmetic::uadd1ca(h1, l2, h2);
                *(rb + 2) = arithmetic::uadd1c(*(rb + 2), l2, cov);
                *(rb + 3) = h2 + cov;
            }
            return rb + 4;
        } else {
            // (ucnt = 5, 9, 13, ...)
            ++ub; --rb;
            LimbT h3, l3;
            for (auto un = ucnt;;) {
                *(rb + 1) = l0;
                std::tie(h3, l3) = arithmetic::umul1(*(ub + 2), *vb); // h3=r12, l3=r13
                *(rb + 2) = arithmetic::uadd1ca(h0, l1, h1);
                std::tie(h0, l0) = arithmetic::umul1(*(ub + 3), *vb);
                *(rb + 3) = arithmetic::uadd1ca(h1, l2, h2);
                ub += 4; rb += 4; un -= 4;
                if (un < 2) break;
                std::tie(h1, l1) = arithmetic::umul1(*ub, *vb);
                *rb = arithmetic::uadd1ca(h2, l3, h3);
                std::tie(h2, l2) = arithmetic::umul1(*(ub + 1), *vb);
                l0 = arithmetic::uadd1ca(h3, l0, h0);
            }
            *(rb) = arithmetic::uadd1ca(h2, l3, h3); // h3 = rbx
            *(rb + 1) = arithmetic::uadd1ca(h3, l0, h0);
            *(rb + 2) = h0;
            while (++vb != ve) {
                rb += 2 - ucnt;
                ub += 1 - ucnt;
                
                auto [h0, l0] = arithmetic::umul1(*(ub - 1), *vb);
                auto [h1, l1] = arithmetic::umul1(*ub, *vb);
                auto [h2, l2] = arithmetic::umul1(*(ub + 1), *vb);
                auto [cov, l0_] = arithmetic::uadd1(*(rb + 1), l0);
                for (auto un = ucnt; un != 5; un -= 4) { // (ucnt = 5, 9, 13, ...)
                    *(rb + 1) = l0_;
                    l0_ = arithmetic::umul4add<LimbT>(ub + 2, *vb, rb + 2, cov, h0, h1, l1, h2, l2);
                    ub += 4; rb += 4;
                }
                *(rb + 1) = l0_;

                auto [h3, l3] = arithmetic::umul1(*(ub + 2), *vb);
                l1 = arithmetic::uadd1ca(h0, l1, h1);
                l2 = arithmetic::uadd1ca(h1, l2, h2);
                l3 = arithmetic::uadd1ca(h2, l3, h3);
                *(rb + 2) = arithmetic::uadd1c(*(rb + 2), l1, cov);
                *(rb + 3) = arithmetic::uadd1c(*(rb + 3), l2, cov);
                *(rb + 4) = arithmetic::uadd1c(*(rb + 4), l3, cov);
                std::tie(h0, l0) = arithmetic::umul1(*(ub + 3), *vb);
                l0 = arithmetic::uadd1ca(h3, l0, h0);
                *(rb + 5) = arithmetic::uadd1c(*(rb + 5), l0, cov);
                *(rb + 6) = h0 + cov;

                rb += 4;
                ub += 4;
            }
            return rb + 3;
        }
    } else {
        auto [h3, l3] = arithmetic::umul1(*(ub + 3), *vb);
        if (!(ucnt & 2)) {
            // ucnt = 4, 8, 12, ...
            for (auto un = ucnt; un != 4; un -= 4) {
                ub += 4;
                *rb = l0;
                *(rb + 1) = arithmetic::uadd1ca(h0, l1, h1);
                std::tie(h0, l0) = arithmetic::umul1(*ub, *vb);
                *(rb + 2) = arithmetic::uadd1ca(h1, l2, h2);
                std::tie(h1, l1) = arithmetic::umul1(*(ub + 1), *vb);
                *(rb + 3) = arithmetic::uadd1ca(h2, l3, h3);
                std::tie(h2, l2) = arithmetic::umul1(*(ub + 2), *vb);
                l0 = arithmetic::uadd1ca(h3, l0, h0);
                std::tie(h3, l3) = arithmetic::umul1(*(ub + 3), *vb);
                rb += 4; 
            }

            *rb = l0;
            *(rb + 1) = arithmetic::uadd1ca(h0, l1, h1);
            *(rb + 2) = arithmetic::uadd1ca(h1, l2, h2);
            *(rb + 3) = arithmetic::uadd1ca(h2, l3, h3);
            *(rb + 4) = h3;

            while (++vb != ve) {
                rb += 5 - ucnt;
                ub += 4 - ucnt;
                
                auto [h0, l0] = arithmetic::umul1(*ub, *vb);
                auto [h1, l1] = arithmetic::umul1(*(ub + 1), *vb);
                auto [h2, l2] = arithmetic::umul1(*(ub + 2), *vb);
                auto [cov, l0_] = arithmetic::uadd1(*rb, l0);
                for (auto un = ucnt; un != 4; un -= 4) {
                    *rb = l0_;
                    ub += 4;
                    l0_ = arithmetic::umul4add<LimbT>(ub - 1, *vb, rb + 1, cov, h0, h1, l1, h2, l2);
                    rb += 4;
                }
                *rb = l0_;
                l1 = arithmetic::uadd1ca(h0, l1, h1);
                *(rb + 1) = arithmetic::uadd1c(*(rb + 1), l1, cov);
                l2 = arithmetic::uadd1ca(h1, l2, h2);
                *(rb + 2) = arithmetic::uadd1c(*(rb + 2), l2, cov);
                auto [h3, l3] = arithmetic::umul1(*(ub + 3), *vb);
                l3 = arithmetic::uadd1ca(h2, l3, h3);
                *(rb + 3) = arithmetic::uadd1c(*(rb + 3), l3, cov);
                *(rb + 4) = h3 + cov;
            }
            return rb + 5;
        } else {
            // ucnt = 6, 10, 14, ...
            for (auto un = ucnt;;) {
                *rb = l0;
                *(rb + 1) = arithmetic::uadd1ca(h0, l1, h1);
                *(rb + 2) = arithmetic::uadd1ca(h1, l2, h2);
                *(rb + 3) = arithmetic::uadd1ca(h2, l3, h3);
                std::tie(h0, l0) = arithmetic::umul1(*(ub + 4), *vb);
                std::tie(h1, l1) = arithmetic::umul1(*(ub + 5), *vb);
                l0 = arithmetic::uadd1ca(h3, l0, h0);
                ub += 4; rb += 4; un -= 4;
                if (un == 2) break;
                std::tie(h2, l2) = arithmetic::umul1(*(ub + 2), *vb);
                std::tie(h3, l3) = arithmetic::umul1(*(ub + 3), *vb);
            }
            *rb = l0;
            *(rb + 1) = arithmetic::uadd1ca(h0, l1, h1);
            *(rb + 2) = h1;
            while (++vb != ve) {
                rb += 3 - ucnt; // rb = r0 + 1
                ub += 2 - ucnt; // ub = u0
                
                auto [h0, l0] = arithmetic::umul1(*ub, *vb);
                auto [h1, l1] = arithmetic::umul1(*(ub + 1), *vb);
                auto [h2, l2] = arithmetic::umul1(*(ub + 2), *vb);
                auto [cov, l0_] = arithmetic::uadd1(*rb, l0);
                for (auto un = ucnt; un != 6; un -= 4) { // (ucnt = 6, 10, 14, ...)
                    *rb = l0_;
                    l0_ = arithmetic::umul4add<LimbT>(ub + 3, *vb, rb + 1, cov, h0, h1, l1, h2, l2);
                    ub += 4; rb += 4;
                }
                *rb = l0_;
                std::tie(h3, l3) = arithmetic::umul1(*(ub + 3), *vb);
                l1 = arithmetic::uadd1ca(h0, l1, h1);
                l2 = arithmetic::uadd1ca(h1, l2, h2);
                l3 = arithmetic::uadd1ca(h2, l3, h3);

                *(rb + 1) = arithmetic::uadd1c(*(rb + 1), l1, cov);
                *(rb + 2) = arithmetic::uadd1c(*(rb + 2), l2, cov);
                *(rb + 3) = arithmetic::uadd1c(*(rb + 3), l3, cov);

                ub += 4; rb += 4;
                std::tie(h0, l0) = arithmetic::umul1(*ub, *vb);
                std::tie(h1, l1) = arithmetic::umul1(*(ub + 1), *vb);
                l0 = arithmetic::uadd1ca(h3, l0, h0);
                
                *rb = arithmetic::uadd1c(*rb, l0, cov);
                l1 = arithmetic::uadd1ca(h0, l1, h1);
                *(rb + 1) = arithmetic::uadd1c(*(rb + 1), l1, cov);
                *(rb + 2) = h1 + cov;
            }
            return rb + 3;
        }
    }
}

// The C++ basecase: NUMETRON_CXX_BASECASE for four limbs and more, the reference below that
// (faster there). The unrolled loop needs un >= 2, so the 1 x 1 product is done here.
template <std::unsigned_integral LimbT>
requires (sizeof(LimbT) == 8)
inline LimbT* umul_basecase_cxx(LimbT const* ub, size_t un, LimbT const* vb, size_t vn, LimbT* rb) noexcept
{
#if NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
    if (un >= 4) return umul_basecase_adx<LimbT>(ub, un, vb, vn, rb);
#elif NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
    if (un >= 4) return umul_basecase_blocked<LimbT>(ub, un, vb, vn, rb);
#endif
    if (un == 1) {
        auto [h, l] = arithmetic::umul1(*ub, *vb);
        rb[0] = l;
        rb[1] = h;
        return rb + 2;
    }
    return umul_basecase_unrolled<LimbT>(ub, ub + un, vb, vb + vn, rb);
}

#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64)) && defined(NUMETRON_PLATFORM_AUTODETECT)
namespace detail {

// The C++ basecase behind the mul_basecase signature, for CPUs no assembly basecase is picked for
// (also called from the assembly Karatsuba, through numetron_karatsuba_ctx). That contract is all
// un + vn limbs written; the C++ basecase may stop short of a zero top limb.
inline void mul_basecase_cxx(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn) noexcept
{
    for (uint64_t* e = umul_basecase_cxx<uint64_t>(up, un, vp, vn, rp), *re = rp + un + vn; e != re; ++e) *e = 0;
}

// The mul_basecase for this CPU: with NUMETRON_ASM_LICENSE_GMP_LGPL the GMP-derived routine for
// the CPU family; otherwise (or for a CPU that has none) our own numetron_mul_basecase_adx when
// the CPU has BMI2 + ADX; the C++ basecase as the last resort.
inline detect_mul_basecase_type select_mul_basecase() noexcept
{
#   if NUMETRON_ASM_LICENSE == NUMETRON_ASM_LICENSE_GMP_LGPL
    if (auto gmp = detect_mul_basecase(numetron_detect_platform())) return gmp;
#   endif
    if (cpu_has_bmi2_adx()) return &numetron_mul_basecase_adx;
    return &mul_basecase_cxx;
}

// select_mul_basecase()'s choice, made on the first call. A plain atomic rather than a
// function-local static: the latter's thread-safe initialization guard is checked on every call,
// through TLS on MSVC (~1 ns of every basecase product). Threads racing through the first call all
// store the same pointer.
inline std::atomic<detect_mul_basecase_type> mul_basecase_fn{ nullptr };

// The first call's choice, out of line so that the check below inlines into every caller (GCC
// kept the whole function out of line in some of them: one more call per small product).
NUMETRON_NOINLINE inline detect_mul_basecase_type init_mul_basecase() noexcept
{
    detect_mul_basecase_type fn = select_mul_basecase();
    mul_basecase_fn.store(fn, std::memory_order_relaxed);
    return fn;
}

NUMETRON_FORCEINLINE detect_mul_basecase_type detected_mul_basecase() noexcept
{
    detect_mul_basecase_type fn = mul_basecase_fn.load(std::memory_order_relaxed);
    if (!fn) [[unlikely]] fn = init_mul_basecase();
    return fn;
}

}
#endif

#if (defined(__GNUC__) || defined(__clang__)) && defined(__SIZEOF_INT128__)
#   define NUMETRON_SMALL_BASECASE_FIXED 1
namespace detail {

// Fixed-size schoolbook products for un in {3, 4}, vn <= 3, fully unrolled, through the compiler's
// 128-bit arithmetic (mul / adc). Behind a run-time switch they are 0.2-0.6 ns faster than the
// asm basecase there (its setup is most of such a product), on a par at 4 x 4 (not taken); un <= 2
// has its own paths already (umul1, umul_basecase_2x). GCC / Clang only: MSVC's _addcarry_u64
// chains made the same kernels up to 2x slower than the asm from 3 x 2 on (docs/multiplication.md).
__extension__ typedef unsigned __int128 umul_small_u128;

template <size_t UN, size_t VN>
NUMETRON_FORCEINLINE void umul_basecase_fixed(uint64_t const* u, uint64_t const* v, uint64_t* r) noexcept
{
    uint64_t c = 0;
    [&]<size_t... I>(std::index_sequence<I...>) {
        ((void)[&] { umul_small_u128 t = (umul_small_u128)u[I] * v[0] + c; r[I] = (uint64_t)t; c = (uint64_t)(t >> 64); }(), ...);
    }(std::make_index_sequence<UN>{});
    r[UN] = c;
    [&]<size_t... J>(std::index_sequence<J...>) {
        ((void)[&] {
            constexpr size_t j = J + 1;
            uint64_t cc = 0;
            [&]<size_t... I>(std::index_sequence<I...>) {
                ((void)[&] { umul_small_u128 t = (umul_small_u128)u[I] * v[j] + r[I + j] + cc; r[I + j] = (uint64_t)t; cc = (uint64_t)(t >> 64); }(), ...);
            }(std::make_index_sequence<UN>{});
            r[UN + j] = cc;
        }(), ...);
    }(std::make_index_sequence<VN - 1>{});
}

// rb[0..un+vn) = u * v for un in {3, 4}, 1 <= vn <= 3
inline void umul_basecase_small(uint64_t const* u, size_t un, uint64_t const* v, size_t vn, uint64_t* r) noexcept
{
    switch (un * 4 + vn) {
    case 13: umul_basecase_fixed<3, 1>(u, v, r); break;
    case 14: umul_basecase_fixed<3, 2>(u, v, r); break;
    case 15: umul_basecase_fixed<3, 3>(u, v, r); break;
    case 17: umul_basecase_fixed<4, 1>(u, v, r); break;
    case 18: umul_basecase_fixed<4, 2>(u, v, r); break;
    default: umul_basecase_fixed<4, 3>(u, v, r); break;
    }
}

// Fixed-size squares, r[0..2N) = u^2, the way usqr_basecase_rows() does it (the products above
// the diagonal once, then doubled and the diagonal added), unrolled through the compiler's 128-bit
// arithmetic: N (N - 1) / 2 + N products instead of N^2, without the rows' per-call cost.
#define NUMETRON_SMALL_SQR_FIXED 1
template <size_t N>
NUMETRON_FORCEINLINE void usqr_basecase_fixed(uint64_t const* u, uint64_t* r) noexcept
{
    static_assert(N >= 2);
    r[0] = 0;
    {
        uint64_t c = 0;
#pragma GCC unroll 16
        for (size_t i = 1; i < N; ++i) {
            umul_small_u128 t = (umul_small_u128)u[i] * u[0] + c;
            r[i] = (uint64_t)t;
            c = (uint64_t)(t >> 64);
        }
        r[N] = c;
    }
#pragma GCC unroll 16
    for (size_t j = 1; j + 1 < N; ++j) {
        uint64_t c = 0;
#pragma GCC unroll 16
        for (size_t i = j + 1; i < N; ++i) {
            umul_small_u128 t = (umul_small_u128)u[i] * u[j] + r[i + j] + c;
            r[i + j] = (uint64_t)t;
            c = (uint64_t)(t >> 64);
        }
        r[N + j] = c;
    }
    r[2 * N - 1] = 0;

    uint64_t shift = 0; // the bit shifted out of the previous limb
    uint64_t c = 0;
#pragma GCC unroll 16
    for (size_t i = 0; i < N; ++i) {
        const umul_small_u128 d = (umul_small_u128)u[i] * u[i];
        const uint64_t lo = r[2 * i], hi = r[2 * i + 1];
        umul_small_u128 s = (umul_small_u128)((lo << 1) | shift) + (uint64_t)d + c;
        r[2 * i] = (uint64_t)s;
        s = (umul_small_u128)((hi << 1) | (lo >> 63)) + (uint64_t)(d >> 64) + (uint64_t)(s >> 64);
        r[2 * i + 1] = (uint64_t)s;
        c = (uint64_t)(s >> 64);
        shift = hi >> 63;
    }
}

// r[0..2n) = u^2 for 3 <= n <= 8
inline void usqr_basecase_small(uint64_t const* u, size_t n, uint64_t* r) noexcept
{
    switch (n) {
    case 3: usqr_basecase_fixed<3>(u, r); break;
    case 4: usqr_basecase_fixed<4>(u, r); break;
    case 5: usqr_basecase_fixed<5>(u, r); break;
    case 6: usqr_basecase_fixed<6>(u, r); break;
    case 7: usqr_basecase_fixed<7>(u, r); break;
    default: usqr_basecase_fixed<8>(u, r); break;
    }
}

}
#endif

// Inlined into its callers: with NUMETRON_USE_ASM it is a call through the detected routine's
// pointer, and as a function of its own it was one more call level (its prologue and epilogue
// ~5% of a 4 x 4 product on MSVC, VTune).
template <std::unsigned_integral LimbT>
requires (sizeof(LimbT) == 8)
NUMETRON_FORCEINLINE LimbT* umul_basecase(LimbT const* ub, size_t un, LimbT const* vb, size_t vn, LimbT* rb) noexcept
{
#if defined(NUMETRON_SMALL_BASECASE_FIXED)
    if (un - 3 < 2 && vn - 1 < 3) { // un in {3, 4}, vn in {1, 2, 3}
        detail::umul_basecase_small(reinterpret_cast<uint64_t const*>(ub), un, reinterpret_cast<uint64_t const*>(vb), vn, reinterpret_cast<uint64_t*>(rb));
        return rb + un + vn;
    }
#endif
#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
#   if defined(NUMETRON_PLATFORM_AUTODETECT)
    detail::detected_mul_basecase()(rb, ub, un, vb, vn);
#   else
    NUMETRON_mul_basecase(rb, ub, un, vb, vn);
#   endif
    return rb + un + vn;
#else
    // Pure C++ (header-only build, or no assembly for this target).
    return umul_basecase_cxx<LimbT>(ub, un, vb, vn, rb);
#endif
}

namespace detail {

// r[0..2n) = u^2, n >= 2, from two row kernels: Mul1(rp, up, k, v) -> rp[0..k) = up * v, returns
// the high limb; AddMul1(rp, up, k, v) -> rp[0..k) += up * v, returns the carry out.
//   1. The products above the diagonal, once: row i adds u[i] * u[i+1..n) at r[2i+1] (rows of
//      n - 1, n - 2, ..., 1 limbs, each's top limb into r[i+n]).
//   2. One pass doubling that (a shift by one bit) and adding the diagonal u[i]^2 at r[2i].
// Half the products of u * u, which is what a square saves (GMP's sqr_basecase does the same).
template <auto Mul1, auto AddMul1>
inline void usqr_basecase_rows(uint64_t* r, uint64_t const* u, size_t n) noexcept
{
    r[0] = 0;
    r[n] = Mul1(r + 1, u + 1, n - 1, u[0]);
    for (size_t i = 1; i + 1 < n; ++i) r[i + n] = AddMul1(r + 2 * i + 1, u + i + 1, n - i - 1, u[i]);
    r[2 * n - 1] = 0;

    uint64_t shift = 0; // the bit shifted out of the previous limb
    unsigned char c = 0;
    for (size_t i = 0; i < n; ++i) {
        auto [dh, dl] = arithmetic::umul1(u[i], u[i]);
        const uint64_t lo = r[2 * i], hi = r[2 * i + 1];
        const uint64_t lo2 = (lo << 1) | shift;
        const uint64_t hi2 = (hi << 1) | (lo >> 63);
        shift = hi >> 63;
        r[2 * i] = arithmetic::uadd1c(lo2, dl, c);
        r[2 * i + 1] = arithmetic::uadd1c(hi2, dh, c);
    }
}

#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
// Whether numetron_sqr_basecase_adx (BMI2 + ADX) may run here: checked once.
inline std::atomic<int> sqr_kernel_state{ 0 }; // 0 not checked yet, 1 yes, 2 no

inline bool sqr_kernel_available() noexcept
{
#   if defined(NUMETRON_PLATFORM_AUTODETECT)
    int s = sqr_kernel_state.load(std::memory_order_relaxed);
    if (!s) [[unlikely]] {
        s = cpu_has_bmi2_adx() ? 1 : 2;
        sqr_kernel_state.store(s, std::memory_order_relaxed);
    }
    return s == 1;
#   elif defined(NUMETRON_PLATFORM_ADX)
    return true;
#   else
    return false;
#   endif
}
#elif defined(NUMETRON_BASECASE_ADX_ASM)
// r[0..2n) = u^2 by the inline-assembly rows. Out of line: usqr_basecase() itself is inlined into
// its callers, and below the threshold it is just a call of the product basecase.
NUMETRON_NOINLINE inline void usqr_basecase_by_rows(uint64_t* r, uint64_t const* u, size_t n) noexcept
{
    usqr_basecase_rows<&addmul_1_adx<false>, &addmul_1_adx<true>>(r, u, n);
}
#endif

}

// rb[0..2n) = u^2, returns rb + 2n, from sqr_basecase_threshold() up with a squaring kernel:
// numetron_sqr_basecase_adx with NUMETRON_USE_ASM on x86-64 (on CPUs with BMI2 + ADX; faster than
// mpn_sqr up to its Karatsuba range), the inline-assembly rows of the header-only GCC / Clang
// build targeting ADX, the fixed-size GCC / Clang kernels for 3..8 limbs. Otherwise, and below
// the threshold, as a general product.
template <std::unsigned_integral LimbT>
requires (sizeof(LimbT) == 8)
NUMETRON_FORCEINLINE LimbT* usqr_basecase(LimbT const* ub, size_t n, LimbT* rb) noexcept
{
#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
    if (n >= 2 && n >= sqr_basecase_threshold() && detail::sqr_kernel_available()) {
        numetron_sqr_basecase_adx(reinterpret_cast<uint64_t*>(rb), reinterpret_cast<uint64_t const*>(ub), n);
        return rb + 2 * n;
    }
#endif
#if defined(NUMETRON_SMALL_SQR_FIXED)
    if (n - 3 < 6) { // 3 <= n <= 8
        detail::usqr_basecase_small(reinterpret_cast<uint64_t const*>(ub), n, reinterpret_cast<uint64_t*>(rb));
        return rb + 2 * n;
    }
#endif
#if !(defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))) && defined(NUMETRON_BASECASE_ADX_ASM)
    if (n >= sqr_basecase_threshold()) {
        detail::usqr_basecase_by_rows(reinterpret_cast<uint64_t*>(rb), reinterpret_cast<uint64_t const*>(ub), n);
        return rb + 2 * n;
    }
#endif
    return umul_basecase(ub, n, ub, n, rb);
}

}
