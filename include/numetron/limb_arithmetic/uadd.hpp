// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>
#include <cstddef>
#include <concepts>
#include <type_traits>

#include "platform.hpp"

namespace numetron::limb_arithmetic {

// prereq: size(r) >= u.size()
// returns carry
template <std::unsigned_integral LimbT, typename InputIteratorT, typename ResultIteratorT>
inline LimbT uadd1(InputIteratorT u, size_t usz, LimbT v, ResultIteratorT r, LimbT c = 0) noexcept
{
    if (!usz) {
        auto [hc, s] = numetron::arithmetic::uadd1(v, c);
        *r = s;
        return hc;
    }
    auto [hc, s] = numetron::arithmetic::uadd1<LimbT>(*u, v, c);
    *r = s;
    while (hc) {
        if (!--usz) break;
        ++u;
        ++r;
        std::tie(hc, *r) = numetron::arithmetic::uadd1<LimbT>(*u, hc);
    }
    return hc;
}

namespace detail {

#if defined(_M_X64) || defined(__x86_64__)
// r[0..n) = u[0..n) + v[0..n), returns the carry out; the inline C++ kernel.
// Unrolled by 4 so the carry can stay in CF through a whole block: a per-limb loop has to
// spill it to a register and reload it every iteration, because the loop's own end-of-range
// compare clobbers CF, putting a setc/add/adc chain (~3 cycles) on the critical path of every
// limb. Results go through locals rather than straight into r: _addcarry_u64 takes
// unsigned long long*, which isn't LimbT (unsigned long) on LP64 targets, and writing through
// a cast pointer would break strict aliasing. Each block reads all its inputs before writing,
// so r may coincide with u or v, or trail them, exactly as with a plain forward loop.
template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
NUMETRON_FORCEINLINE unsigned char add_n_x64_inline(LimbT* r, LimbT const* u, LimbT const* v, size_t n) noexcept
{
    using ull = unsigned long long;
    unsigned char c = 0;
    for (size_t i = n >> 2; i; --i, r += 4, u += 4, v += 4) {
        ull r0, r1, r2, r3;
        c = _addcarry_u64(c, u[0], v[0], &r0);
        c = _addcarry_u64(c, u[1], v[1], &r1);
        c = _addcarry_u64(c, u[2], v[2], &r2);
        c = _addcarry_u64(c, u[3], v[3], &r3);
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    }
    for (n &= 3; n; --n, ++r, ++u, ++v) {
        ull s;
        c = _addcarry_u64(c, *u, *v, &s);
        *r = s;
    }
    return c;
}

// r[0..n) = u[0..n) + v[0..n), returns the carry out. From asm_add_sub_n_min_limbs on this is
// the assembly numetron_add_n (src/arch/x86_64/add_sub_n.*), which keeps the carry in CF across
// the whole run; below that the inline kernel above, where the call would cost more than the
// work. Force-inlined, like everything between the callers and the kernel, so a short run costs
// no call at all -- otherwise the compiler may keep this dispatch out of line and a 4-limb add
// pays for a call it was supposed to avoid.
template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
NUMETRON_FORCEINLINE unsigned char add_n_x64(LimbT* r, LimbT const* u, LimbT const* v, size_t n) noexcept
{
#if defined(NUMETRON_USE_ASM)
    if (n >= asm_add_sub_n_min_limbs) {
        return static_cast<unsigned char>(numetron_add_n(
            reinterpret_cast<uint64_t*>(r), reinterpret_cast<uint64_t const*>(u), reinterpret_cast<uint64_t const*>(v), n));
    }
#endif
    return add_n_x64_inline<LimbT>(r, u, v, n);
}
#endif

}

// Unsigned add u[0..n) += v[0..n) inplace, where n is ve - v. Returns carry.
template <std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE LimbT uadd_inplace(LimbT* u, LimbT const* v, LimbT const* ve) noexcept
{
#if defined(_M_X64) || defined(__x86_64__)
    if constexpr (sizeof(LimbT) == 8) {
        return detail::add_n_x64<LimbT>(u, u, v, static_cast<size_t>(ve - v));
    } else
#endif
    {
        unsigned char c = 0;
        for (; v != ve; ++u, ++v)
            *u = arithmetic::uadd1c(*u, *v, c);
        return (LimbT)c;
    }
}

// Propagate carry/borrow c into a[0..n): a += c. Returns carry.
template <std::unsigned_integral LimbT>
inline LimbT uadd_limb(LimbT* u, LimbT* ue, LimbT c) noexcept
{
    for (; c && u != ue; ++u) {
        std::tie(c, *u) = arithmetic::uadd1(*u, c);
    }
    return c;
}

// u size must be >= v size
template <typename UIteratorT, typename VIteratorT, typename RIteratorT>
NUMETRON_FORCEINLINE unsigned char uadd_partial_unchecked(UIteratorT& ub, VIteratorT vb, VIteratorT ve, RIteratorT& rb) noexcept
{
#if defined(_M_X64) || defined(__x86_64__)
    using limb_t = std::remove_cvref_t<decltype(*rb)>;
    if constexpr (std::is_pointer_v<UIteratorT> && std::is_pointer_v<VIteratorT> && std::is_pointer_v<RIteratorT>
        && std::unsigned_integral<limb_t> && sizeof(limb_t) == 8
        && std::is_same_v<std::remove_cv_t<std::remove_pointer_t<UIteratorT>>, limb_t>
        && std::is_same_v<std::remove_cv_t<std::remove_pointer_t<VIteratorT>>, limb_t>) {
        const size_t n = static_cast<size_t>(ve - vb);
        unsigned char c = detail::add_n_x64<limb_t>(rb, ub, vb, n);
        ub += n;
        rb += n;
        return c;
    } else
#endif
    {
        unsigned char c = 0;
        for (; vb != ve; ++ub, ++vb, ++rb) {
            *rb = numetron::arithmetic::uadd1c(*ub, *vb, c);
        }
        return c;
    }
}

// u size must be >= v size
template <std::unsigned_integral LimbT, typename UIteratorT, typename VIteratorT, typename RIteratorT>
inline unsigned char uadd_unchecked(LimbT uh, UIteratorT ub, UIteratorT ue, LimbT vh, VIteratorT vb, VIteratorT ve, RIteratorT& rb) noexcept
{
    unsigned char c = uadd_partial_unchecked(ub, vb, ve, rb);
    if (ub != ue) {
        *rb = numetron::arithmetic::uadd1c(*ub, vh, c);
        ++rb;
        while (++ub != ue) {
            if (!c) {
                for (;;) {
                    *rb = *ub;
                    ++ub; ++rb;
                    if (ub == ue) break;
                }
                *rb = uh;
                ++rb;
                return 0;
            }
            *rb = numetron::arithmetic::uincx(*ub, c);
            ++rb;
        }
        std::tie(c, *rb) = numetron::arithmetic::uadd1(uh, (LimbT)c);
    } else {
        *rb = numetron::arithmetic::uadd1c(uh, vh, c);
    }
    ++rb;
    return c;
}

template <std::unsigned_integral LimbT, typename RIteratorT>
inline unsigned char uadd_unchecked(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, RIteratorT& rb) noexcept
{
    return uadd_unchecked<LimbT>(uh, u.data(), u.data() + u.size(), vh, v.data(), v.data() + v.size(), rb);
}


}
