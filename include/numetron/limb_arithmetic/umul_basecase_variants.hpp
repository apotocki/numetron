// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <concepts>

#include "numetron/arithmetic.hpp"
#include "numetron/limb_arithmetic/platform.hpp" // NUMETRON_FORCEINLINE

#if defined(_MSC_VER) && defined(_M_X64)
#   include <intrin.h>
#elif defined(__x86_64__)
#   include <immintrin.h>
#endif

// Alternative C++ basecase multiplications for the header-only build (no NUMETRON_USE_ASM);
// umul_basecase() picks one by NUMETRON_CXX_BASECASE (config/implementation.hpp), and
// umul_basecase_unrolled() in umul_basecase.hpp stays the reference. Same contract as it:
// rb[0..un+vn) = u[0..un) * v[0..vn), un >= vn >= 1, rb not overlapping u, v; returns rb + un + vn.
// Measured in docs/multiplication.md (§ 9, header-only build).

namespace numetron::limb_arithmetic {

namespace detail {

// (c2:c1:c0) += a * b
template <std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE void mul_acc3(LimbT& c0, LimbT& c1, LimbT& c2, LimbT a, LimbT b) noexcept
{
    if constexpr (sizeof(LimbT) == 8) {
#if defined(__SIZEOF_INT128__)
        // GCC / Clang: add, adc, adc from plain 128-bit arithmetic
        __extension__ typedef unsigned __int128 u128; // __extension__: no -Wpedantic warning
        const u128 p = static_cast<u128>(a) * b;
        const u128 acc = ((static_cast<u128>(c1) << 64) | c0) + p;
        c2 += acc < p;
        c0 = static_cast<LimbT>(acc);
        c1 = static_cast<LimbT>(acc >> 64);
        return;
#elif defined(_MSC_VER) && defined(_M_X64)
        unsigned long long h;
        const unsigned long long l = _umul128(a, b, &h);
        unsigned long long r0, r1;
        unsigned char c = _addcarry_u64(0, c0, l, &r0);
        c = _addcarry_u64(c, c1, h, &r1);
        c0 = r0;
        c1 = r1;
        c2 += c;
        return;
#endif
    }
    auto [h, l] = arithmetic::umul1(a, b);
    unsigned char c = 0;
    c0 = arithmetic::uadd1c(c0, l, c);
    c1 = arithmetic::uadd1c(c1, h, c);
    c2 += c;
}

}

// Product scanning (Comba): the result is produced column by column, each column k the sum of
// u[i] * v[k-i] over the valid i, accumulated in three words (c0, c1, c2). A product adds its low
// half to c0 and its high half plus that carry to c1, the last carry going to c2 -- the carries
// stay within one product (add, adc, adc), so there is no carry chain running along a whole row
// as in the operand-scanning (row by row) basecase.
template <std::unsigned_integral LimbT>
inline LimbT* umul_basecase_comba(LimbT const* u, size_t un, LimbT const* v, size_t vn, LimbT* rb) noexcept
{
    LimbT c0 = 0, c1 = 0, c2 = 0;
    const size_t cols = un + vn - 1;
    for (size_t k = 0; k < cols; ++k) {
        // i in [max(0, k - vn + 1), min(k, un - 1)], j = k - i
        const size_t i0 = k + 1 > vn ? k + 1 - vn : 0;
        const size_t i1 = k < un - 1 ? k : un - 1;
        LimbT const* up = u + i0;
        LimbT const* vp = v + (k - i0);
        for (size_t i = i0; i <= i1; ++i, ++up, --vp) detail::mul_acc3(c0, c1, c2, *up, *vp);
        rb[k] = c0;
        c0 = c1;
        c1 = c2;
        c2 = 0;
    }
    rb[cols] = c0;
    return rb + un + vn;
}

// ---- Blocked rows: the two carry chains of a row taken four limbs at a time --------------------
//
// A row r[0..n) += u[0..n) * v is two additions: the low halves of the products into r, and the
// high halves into the next position. With one carry flag they can't run interleaved limb by limb
// without saving and restoring the flag every time. Here a block of four limbs does its four
// products, then the whole high-half chain (four adc), then the whole r chain (four adc): the flag
// is saved twice per block instead of twice per limb. Plain x64 (mul, adc), no BMI2 / ADX.

#if defined(_M_X64) || defined(__x86_64__)

namespace detail {

NUMETRON_FORCEINLINE unsigned char addcarry64(unsigned char c, std::uint64_t a, std::uint64_t b, std::uint64_t& r) noexcept
{
    unsigned long long t;
    c = _addcarry_u64(c, a, b, &t);
    r = t;
    return c;
}

NUMETRON_FORCEINLINE std::uint64_t mul64(std::uint64_t a, std::uint64_t b, std::uint64_t& hi) noexcept
{
    auto [h, l] = arithmetic::umul1(a, b);
    hi = h;
    return l;
}

// r[0..n) += u[0..n) * v (Add) or r[0..n) = u[0..n) * v, n >= 1; returns the limb above.
template <bool Add>
NUMETRON_FORCEINLINE std::uint64_t addmul_1_blocked(std::uint64_t* rp, std::uint64_t const* up, size_t n, std::uint64_t v) noexcept
{
    unsigned char c1 = 0, c2 = 0;
    std::uint64_t hi = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        std::uint64_t h0, h1, h2, h3;
        std::uint64_t l0 = mul64(up[i], v, h0);
        std::uint64_t l1 = mul64(up[i + 1], v, h1);
        std::uint64_t l2 = mul64(up[i + 2], v, h2);
        std::uint64_t l3 = mul64(up[i + 3], v, h3);
        c1 = addcarry64(c1, l0, hi, l0);
        c1 = addcarry64(c1, l1, h0, l1);
        c1 = addcarry64(c1, l2, h1, l2);
        c1 = addcarry64(c1, l3, h2, l3);
        if constexpr (Add) {
            c2 = addcarry64(c2, l0, rp[i], l0);
            c2 = addcarry64(c2, l1, rp[i + 1], l1);
            c2 = addcarry64(c2, l2, rp[i + 2], l2);
            c2 = addcarry64(c2, l3, rp[i + 3], l3);
        }
        rp[i] = l0;
        rp[i + 1] = l1;
        rp[i + 2] = l2;
        rp[i + 3] = l3;
        hi = h3;
    }
    for (; i < n; ++i) {
        std::uint64_t h;
        std::uint64_t l = mul64(up[i], v, h);
        c1 = addcarry64(c1, l, hi, l);
        if constexpr (Add) c2 = addcarry64(c2, l, rp[i], l);
        rp[i] = l;
        hi = h;
    }
    return hi + c1 + c2;
}

}

template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
inline LimbT* umul_basecase_blocked(LimbT const* u, size_t un, LimbT const* v, size_t vn, LimbT* rb) noexcept
{
    auto* r = reinterpret_cast<std::uint64_t*>(rb);
    auto const* up = reinterpret_cast<std::uint64_t const*>(u);
    r[un] = detail::addmul_1_blocked<false>(r, up, un, v[0]);
    for (size_t j = 1; j < vn; ++j) r[un + j] = detail::addmul_1_blocked<true>(r + j, up, un, v[j]);
    return rb + un + vn;
}

#endif

// ---- ADX (mulx + adcx/adox): two independent carry chains per row --------------------------
//
// Operand scanning like umul_basecase_unrolled(), but each row r[j..j+un] += u * v[j] runs the two
// additions it consists of -- the low halves of the products into r, the high halves into the
// next position -- as two separate carry chains: adox on the overflow flag and adcx on the carry
// flag. mulx leaves the flags alone, so both chains stay in the flags for the whole row. Needs
// BMI2 and ADX (Broadwell / Zen and later).

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__ADX__) && defined(__BMI2__)
#   define NUMETRON_BASECASE_ADX_ASM
#endif
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#   define NUMETRON_BASECASE_ADX_INTRIN // the caller must know the CPU has ADX (not checked here)
#endif

#if defined(NUMETRON_BASECASE_ADX_ASM) || defined(NUMETRON_BASECASE_ADX_INTRIN)

namespace detail {

#if defined(NUMETRON_BASECASE_ADX_ASM)

// r[0..n) += u[0..n) * v (Add = true) or r[0..n) = u[0..n) * v (Add = false), n >= 1; returns the
// limb above. GCC / Clang inline assembly: n % 4 single limbs first, then blocks of four; the
// loop control uses lea and jrcxz only, which leave CF and OF alone.
template <bool Add>
NUMETRON_FORCEINLINE std::uint64_t addmul_1_adx(std::uint64_t* rp, std::uint64_t const* up, size_t n, std::uint64_t v) noexcept
{
    std::uint64_t hi, lo, t;
    size_t singles = n & 3, blocks = n >> 2;
    if constexpr (Add) {
        __asm__ volatile(
            "xorl   %k[hi], %k[hi]\n\t"             // hi = 0, CF = OF = 0
            "jrcxz  2f\n"
            "1:\n\t"
            "mulx   (%[up]), %[lo], %[t]\n\t"
            "adcx   %[hi], %[lo]\n\t"               // + previous high half (CF chain)
            "adox   (%[rp]), %[lo]\n\t"             // + r[i] (OF chain)
            "movq   %[lo], (%[rp])\n\t"
            "movq   %[t], %[hi]\n\t"
            "leaq   8(%[up]), %[up]\n\t"
            "leaq   8(%[rp]), %[rp]\n\t"
            "leaq   -1(%%rcx), %%rcx\n\t"
            "jrcxz  2f\n\t"
            "jmp    1b\n"
            "2:\n\t"
            "movq   %[blocks], %%rcx\n\t"
            "jrcxz  4f\n"
            "3:\n\t"
            "mulx   (%[up]), %[lo], %[t]\n\t"
            "adcx   %[hi], %[lo]\n\t"
            "adox   (%[rp]), %[lo]\n\t"
            "movq   %[lo], (%[rp])\n\t"
            "mulx   8(%[up]), %[lo], %[hi]\n\t"
            "adcx   %[t], %[lo]\n\t"
            "adox   8(%[rp]), %[lo]\n\t"
            "movq   %[lo], 8(%[rp])\n\t"
            "mulx   16(%[up]), %[lo], %[t]\n\t"
            "adcx   %[hi], %[lo]\n\t"
            "adox   16(%[rp]), %[lo]\n\t"
            "movq   %[lo], 16(%[rp])\n\t"
            "mulx   24(%[up]), %[lo], %[hi]\n\t"
            "adcx   %[t], %[lo]\n\t"
            "adox   24(%[rp]), %[lo]\n\t"
            "movq   %[lo], 24(%[rp])\n\t"
            "leaq   32(%[up]), %[up]\n\t"
            "leaq   32(%[rp]), %[rp]\n\t"
            "leaq   -1(%%rcx), %%rcx\n\t"
            "jrcxz  4f\n\t"
            "jmp    3b\n"
            "4:\n\t"
            "movl   $0, %k[lo]\n\t"                 // mov keeps the flags
            "adcx   %[lo], %[hi]\n\t"
            "adox   %[lo], %[hi]\n\t"
            : [hi] "=&r"(hi), [lo] "=&r"(lo), [t] "=&r"(t), [up] "+r"(up), [rp] "+r"(rp), "+c"(singles)
            : "d"(v), [blocks] "r"(blocks)
            : "cc", "memory");
    } else {
        __asm__ volatile(
            "xorl   %k[hi], %k[hi]\n\t"             // hi = 0, CF = 0
            "jrcxz  2f\n"
            "1:\n\t"
            "mulx   (%[up]), %[lo], %[t]\n\t"
            "adcx   %[hi], %[lo]\n\t"
            "movq   %[lo], (%[rp])\n\t"
            "movq   %[t], %[hi]\n\t"
            "leaq   8(%[up]), %[up]\n\t"
            "leaq   8(%[rp]), %[rp]\n\t"
            "leaq   -1(%%rcx), %%rcx\n\t"
            "jrcxz  2f\n\t"
            "jmp    1b\n"
            "2:\n\t"
            "movq   %[blocks], %%rcx\n\t"
            "jrcxz  4f\n"
            "3:\n\t"
            "mulx   (%[up]), %[lo], %[t]\n\t"
            "adcx   %[hi], %[lo]\n\t"
            "movq   %[lo], (%[rp])\n\t"
            "mulx   8(%[up]), %[lo], %[hi]\n\t"
            "adcx   %[t], %[lo]\n\t"
            "movq   %[lo], 8(%[rp])\n\t"
            "mulx   16(%[up]), %[lo], %[t]\n\t"
            "adcx   %[hi], %[lo]\n\t"
            "movq   %[lo], 16(%[rp])\n\t"
            "mulx   24(%[up]), %[lo], %[hi]\n\t"
            "adcx   %[t], %[lo]\n\t"
            "movq   %[lo], 24(%[rp])\n\t"
            "leaq   32(%[up]), %[up]\n\t"
            "leaq   32(%[rp]), %[rp]\n\t"
            "leaq   -1(%%rcx), %%rcx\n\t"
            "jrcxz  4f\n\t"
            "jmp    3b\n"
            "4:\n\t"
            "movl   $0, %k[lo]\n\t"
            "adcx   %[lo], %[hi]\n\t"
            : [hi] "=&r"(hi), [lo] "=&r"(lo), [t] "=&r"(t), [up] "+r"(up), [rp] "+r"(rp), "+c"(singles)
            : "d"(v), [blocks] "r"(blocks)
            : "cc", "memory");
    }
    return hi;
}

#else // NUMETRON_BASECASE_ADX_INTRIN

// The same with intrinsics: two interleaved _addcarryx_u64 chains, which MSVC can map to adcx /
// adox (whether it does is what the benchmark tells).
template <bool Add>
NUMETRON_FORCEINLINE std::uint64_t addmul_1_adx(std::uint64_t* rp, std::uint64_t const* up, size_t n, std::uint64_t v) noexcept
{
    unsigned char c1 = 0, c2 = 0;
    unsigned long long hi = 0;
    size_t i = 0;
    auto step = [&](size_t k) {
        unsigned long long h;
        unsigned long long l = _mulx_u64(up[k], v, &h);
        c1 = _addcarryx_u64(c1, l, hi, &l);
        if constexpr (Add) c2 = _addcarryx_u64(c2, l, rp[k], &l);
        rp[k] = l;
        hi = h;
    };
    for (; i + 4 <= n; i += 4) {
        step(i);
        step(i + 1);
        step(i + 2);
        step(i + 3);
    }
    for (; i < n; ++i) step(i);
    return hi + c1 + c2;
}

#endif

}

// Operand-scanning basecase on the ADX rows: the first row writes r, the others add into it.
template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
inline LimbT* umul_basecase_adx(LimbT const* u, size_t un, LimbT const* v, size_t vn, LimbT* rb) noexcept
{
    auto* r = reinterpret_cast<std::uint64_t*>(rb);
    auto const* up = reinterpret_cast<std::uint64_t const*>(u);
    r[un] = detail::addmul_1_adx<false>(r, up, un, v[0]);
    for (size_t j = 1; j < vn; ++j) r[un + j] = detail::addmul_1_adx<true>(r + j, up, un, v[j]);
    return rb + un + vn;
}

#endif

}
