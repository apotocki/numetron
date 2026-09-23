// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>
#include <cstddef>
#include <concepts>

#include "platform.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

#if defined(_M_X64) || defined(__x86_64__)
// r[0..n) = u[0..n) - v[0..n), returns the borrow out; the inline C++ kernel. Same structure,
// reasons and aliasing rules as add_n_x64_inline() in uadd.hpp: unrolled by 4 so the borrow
// stays in CF through a block, results via locals to keep strict aliasing intact; r may
// coincide with u or v, or trail them.
template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
NUMETRON_FORCEINLINE unsigned char sub_n_x64_inline(LimbT* r, LimbT const* u, LimbT const* v, size_t n) noexcept
{
    using ull = unsigned long long;
    unsigned char b = 0;
    for (size_t i = n >> 2; i; --i, r += 4, u += 4, v += 4) {
        ull r0, r1, r2, r3;
        b = _subborrow_u64(b, u[0], v[0], &r0);
        b = _subborrow_u64(b, u[1], v[1], &r1);
        b = _subborrow_u64(b, u[2], v[2], &r2);
        b = _subborrow_u64(b, u[3], v[3], &r3);
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
    for (n &= 3; n; --n, ++r, ++u, ++v) {
        ull d;
        b = _subborrow_u64(b, *u, *v, &d);
        *r = d;
    }
    return b;
}

// r[0..n) = u[0..n) - v[0..n), returns the borrow out: the assembly numetron_sub_n from
// asm_add_sub_n_min_limbs on, the inline kernel above below that. Force-inlined for the same
// reason as add_n_x64() in uadd.hpp.
template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
NUMETRON_FORCEINLINE unsigned char sub_n_x64(LimbT* r, LimbT const* u, LimbT const* v, size_t n) noexcept
{
#if defined(NUMETRON_USE_ASM)
    if (n >= asm_add_sub_n_min_limbs) {
        return static_cast<unsigned char>(numetron_sub_n(
            reinterpret_cast<uint64_t*>(r), reinterpret_cast<uint64_t const*>(u), reinterpret_cast<uint64_t const*>(v), n));
    }
#endif
    return sub_n_x64_inline<LimbT>(r, u, v, n);
}
#endif

}

// Unsigned subtract u[0..n) - v[0..n) inplace, assuming u >= v. where n is ve - v.
template <std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE LimbT usub_inplace(LimbT* u, LimbT const* v, LimbT const* ve) noexcept
{
#if defined(_M_X64) || defined(__x86_64__)
    if constexpr (sizeof(LimbT) == 8) {
        return detail::sub_n_x64<LimbT>(u, u, v, static_cast<size_t>(ve - v));
    } else
#endif
    {
        LimbT borrow = 0;
        for (; v != ve; ++u, ++v)
            std::tie(borrow, *u) = arithmetic::usub1c(*u, *v, borrow);
        return borrow;
    }
}

// Propagate carry/borrow c into a[0..n): a -= c. Returns borrow.
template <std::unsigned_integral LimbT>
inline LimbT usub_limb(LimbT* u, LimbT* ue, LimbT c) noexcept
{
    for (; c && u != ue; ++u) {
        std::tie(c, *u) = arithmetic::usub1(*u, c);
    }
    return c;
}

// u size must be >= v size
template <std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE LimbT usub_partial_unchecked(LimbT const*& ub, LimbT const* vb, LimbT const* ve, LimbT*& rb)
{
#if defined(_M_X64) || defined(__x86_64__)
    if constexpr (sizeof(LimbT) == 8) {
        const size_t n = static_cast<size_t>(ve - vb);
        LimbT c = detail::sub_n_x64<LimbT>(rb, ub, vb, n);
        ub += n;
        rb += n;
        return c;
    } else
#endif
    {
        LimbT c = 0;
        for (; vb != ve; ++ub, ++vb, ++rb) {
            std::tie(c, *rb) = numetron::arithmetic::usub1c(*ub, *vb, c);
        }
        return c;
    }
}

// u size must be >= v size
template <std::unsigned_integral LimbT>
inline void usub_unchecked(LimbT const*& ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT*& rb)
{
    assert(ue - ub >= ve - vb);
    LimbT c = usub_partial_unchecked(ub, vb, ve, rb);
    for (; ub != ue; ++ub, ++rb) {
        std::tie(c, *rb) = numetron::arithmetic::usub1(*ub, c);
    }
}

template <std::unsigned_integral LimbT>
inline LimbT usub_partial_limb(LimbT const*& ub, LimbT const* ue, LimbT c, LimbT*& rb) noexcept
{
    for (; c && ub != ue; ++ub, ++rb) {
        std::tie(c, *rb) = numetron::arithmetic::usub1(*ub, c);
    }
    return c;
}

// u size must be >= v size
template <std::unsigned_integral LimbT>
inline LimbT usub_unchecked(LimbT last_u, LimbT const* ub, LimbT const* ue, LimbT last_v, LimbT const* vb, LimbT const* ve, LimbT*& rb)
{
    LimbT c = usub_partial_unchecked(ub, vb, ve, rb);
    if (ub != ue) {
        std::tie(c, *rb++) = numetron::arithmetic::usub1c(*ub++, last_v, c);
        for (; ub != ue; ++ub, ++rb) {
            std::tie(c, *rb) = numetron::arithmetic::usub1(*ub, c);
        }
        std::tie(c, *rb) = numetron::arithmetic::usub1(last_u, c);
    } else {
        std::tie(c, *rb) = numetron::arithmetic::usub1c(last_u, last_v, c);
    }
    return c;
}

template <std::unsigned_integral LimbT, typename RIteratorT>
inline LimbT usub_unchecked(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, RIteratorT& rb)
{
    return usub_unchecked<LimbT>(uh, u.data(), u.data() + u.size(), vh, v.data(), v.data() + v.size(), rb);
}

}