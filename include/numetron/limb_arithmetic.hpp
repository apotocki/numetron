// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>

#include "arithmetic.hpp"
#include "ct.hpp"

#include "detail/small_array.hpp"

#define NUMETRON_USE_ASM
#define NUMETRON_PLATFORM_AUTODETECT

//#define NUMETRON_PLATFORM_ALDERLAKE
//#define NUMETRON_PLATFORM_CORE2
//#define NUMETRON_PLATFORM_K8

#ifndef NUMETRON_DC_DIV_QR_THRESHOLD
#   define NUMETRON_DC_DIV_QR_THRESHOLD 50
#endif

#ifndef NUMETRON_INPLACE_LIMB_RESERVE_COUNT
#   define NUMETRON_INPLACE_LIMB_RESERVE_COUNT 8
#endif

#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))

#if defined(NUMETRON_PLATFORM_AUTODETECT)
#include <mutex>
typedef void (*detect_mul_basecase_type)(uint64_t*, const uint64_t*, size_t, const uint64_t*, size_t);
inline std::once_flag mul_basecase_init_flag;
extern "C" uint64_t numetron_detect_platform();
extern "C" detect_mul_basecase_type detect_mul_basecase(uint64_t);
inline void (*__mul_basecase_ptr)(uint64_t*, const uint64_t*, size_t, const uint64_t*, size_t) = nullptr;
//extern "C" void __alderlake_mul_basecase(uint64_t* rp, const uint64_t* up, size_t un, uint64_t const* vp, uint64_t vn);
//extern "C" void __k8_mul_basecase(uint64_t* rp, const uint64_t* up, size_t un, uint64_t const* vp, uint64_t vn);
#define NUMETRON_mul_basecase __mul_basecase_ptr
#endif

#if defined(NUMETRON_PLATFORM_K8)
#   if defined(NUMETRON_mul_basecase)
#       error "NUMETRON_mul_basecase already defined"
#   endif
#   define NUMETRON_mul_basecase __k8_mul_basecase
#endif

#if defined(NUMETRON_PLATFORM_ALDERLAKE)
#   if defined(NUMETRON_mul_basecase)
#       error "NUMETRON_mul_basecase already defined"
#   endif
#   define NUMETRON_mul_basecase __alderlake_mul_basecase
#endif

#if defined(NUMETRON_PLATFORM_CORE2)
#   if defined(NUMETRON_mul_basecase)
#       error "NUMETRON_mul_basecase already defined"
#   endif
#define NUMETRON_mul_basecase __core2_mul_basecase 
#endif

#if !defined(NUMETRON_PLATFORM_AUTODETECT)
extern "C" void NUMETRON_mul_basecase(uint64_t* rp, const uint64_t* up, size_t un, uint64_t const* vp, uint64_t vn);
#endif

#endif // NUMETRON_USE_ASM

namespace numetron::limb_arithmetic {

using numetron::detail::small_array;

template <std::unsigned_integral LimbT>
using composition = std::tuple<std::span<const LimbT>, LimbT, int>; // limbs, mask for the last limb, sign (+1 or -1)

template <std::unsigned_integral LimbT>
[[nodiscard]] inline bool equal_decomposed(composition<LimbT> const& l, composition<LimbT> const& r) noexcept
{
    auto& [llimbs, lmask, lsign] = l;
    auto& [rlimbs, rmask, rsign] = r;

    if (lsign != rsign || llimbs.size() != rlimbs.size()) return false;
    if (llimbs.empty()) return true; // both zero (perhaps not a real case)

    if ((llimbs.back() & lmask) != (rlimbs.back() & rmask)) return false;
    return !std::memcmp(llimbs.data(), rlimbs.data(), (llimbs.size() - 1) * sizeof(LimbT));
}

template <std::unsigned_integral LimbT>
[[nodiscard]] int compare(std::span<const LimbT> llimbs, LimbT lmask, std::span<const LimbT> rlimbs, LimbT rmask, int lsign) noexcept
{
    if (llimbs.size() < rlimbs.size()) return -lsign;
    if (llimbs.size() > rlimbs.size()) return lsign;
    if (llimbs.empty()) return 0; // both zero (perhaps not a real case)

    LimbT llast = llimbs.back() & lmask;
    LimbT rlast = rlimbs.back() & rmask;
    if (llast < rlast) return -lsign;
    if (llast > rlast) return lsign;
    for (size_t i = llimbs.size() - 1; i-- > 0;) {
        if (llimbs[i] < rlimbs[i]) return -lsign;
        if (llimbs[i] > rlimbs[i]) return lsign;
    }
    return 0;
}

template <std::unsigned_integral LimbT>
[[nodiscard]] inline int compare_decomposed(composition<LimbT> const& l, composition<LimbT> const& r) noexcept
{
    auto& [llimbs, lmask, lsign] = l;
    auto& [rlimbs, rmask, rsign] = r;

    if (lsign < 0 && rsign > 0) return -1;
    if (lsign > 0 && rsign < 0) return 1;

    return compare(llimbs, lmask, rlimbs, rmask, lsign);
}

// u size must be >= v size
template <typename UIteratorT, typename VIteratorT, typename RIteratorT>
inline unsigned char uadd_partial_unchecked(UIteratorT& ub, VIteratorT vb, VIteratorT ve, RIteratorT& rb) noexcept
{
    unsigned char c = 0;
    for (; vb != ve; ++ub, ++vb, ++rb) {
        *rb = numetron::arithmetic::uadd1c(*ub, *vb, c);
    }
    return c;
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
        *rb = numetron::arithmetic::uincx(*ub, c);
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

// u size must be >= v size
template <std::unsigned_integral LimbT>
inline LimbT usub_partial_unchecked(LimbT const*& ub, LimbT const* vb, LimbT const* ve, LimbT*& rb)
{
    LimbT c = 0;
    for (; vb != ve; ++ub, ++vb, ++rb) {
        std::tie(c, *rb) = numetron::arithmetic::usub1c(*ub, *vb, c);
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

template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<std::remove_cvref_t<AllocatorT>>::value_type>)
[[nodiscard]] std::tuple<LimbT*, size_t, size_t, int> copy(composition<LimbT> const& arg, AllocatorT&& alloc)
{
    std::tuple<LimbT*, size_t, size_t, int> result;
    auto [limbs, mask, sign] = arg;
    if (!limbs.size() || (!(limbs.back() & mask) && limbs.size() == 1)) [[unlikely]] {
        result = { nullptr, 0, 0, 1 };
    } else {
        using alloc_traits_t = std::allocator_traits<std::remove_cvref_t<AllocatorT>>;
        result = {
            alloc_traits_t::allocate(alloc, limbs.size()),
            limbs.size(),
            limbs.size(),
            sign
        };
        std::memcpy(get<0>(result), limbs.data(), limbs.size() * sizeof(LimbT));
        *(get<0>(result) + limbs.size() - 1) &= mask;
    }
    return result;
}

template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<std::remove_cvref_t<AllocatorT>>::value_type>)
[[nodiscard]] std::tuple<LimbT*, size_t, size_t, int> add(composition<LimbT> const& l, composition<LimbT> const& r, AllocatorT&& alloc)
{
    using alloc_traits_t = std::allocator_traits<std::remove_cvref_t<AllocatorT>>;
    
    std::tuple<LimbT*, size_t, size_t, int> result;

    auto [llimbs, lmask, lsign] = l;
    if (llimbs.empty()) [[unlikely]] {
        result = copy(r, std::forward<AllocatorT>(alloc));
        return result;
    }
    auto [rlimbs, rmask, rsign] = r;
    if (rlimbs.empty()) [[unlikely]] {
        result = copy(l, std::forward<AllocatorT>(alloc));
        return result;
    }
    LimbT last_l = llimbs.back() & lmask;
    LimbT last_r = rlimbs.back() & rmask;

    size_t maxargsz = (std::max)(llimbs.size(), rlimbs.size());
    llimbs = llimbs.first(llimbs.size() - 1);
    rlimbs = rlimbs.first(rlimbs.size() - 1);

    if (lsign == rsign) {
        get<3>(result) = lsign;
        get<1>(result) = get<2>(result) = maxargsz + 1;
        LimbT* res = get<0>(result) = alloc_traits_t::allocate(alloc, get<2>(result));

        LimbT c = (llimbs.size() >= rlimbs.size()) ?
            uadd_unchecked(last_l, llimbs, last_r, rlimbs, res):
            uadd_unchecked(last_r, rlimbs, last_l, llimbs, res);
        if (c) [[unlikely]] {
            *res = c;
        } else {
            --get<1>(result);
        }
    } else {
        bool do_swap = false;
        if (llimbs.size() == rlimbs.size()) {
            while (last_l == last_r) {
                if (llimbs.empty()) {
                    result = { nullptr, 0, 0, 1 };
                    return result;
                }
                llimbs = llimbs.first(llimbs.size() - 1);
                rlimbs = rlimbs.first(llimbs.size() - 1);
                last_l = llimbs.back();
                last_r = rlimbs.back();
            }
            do_swap = last_l < last_r;
        }

        get<1>(result) = get<2>(result) = maxargsz;
        LimbT* res = get<0>(result) = alloc_traits_t::allocate(alloc, maxargsz);
        
        LimbT c;
        if ((llimbs.size() >= rlimbs.size()) ^ do_swap) {
            c = usub_unchecked(last_l, llimbs, last_r, rlimbs, res);
            get<3>(result) = lsign;
        } else {
            c = usub_unchecked(last_r, rlimbs, last_l, llimbs, res);
            get<3>(result) = rsign;
        }
        (void)c; // to avoid unused variable warning
        assert(!c);
        for (; !*res; --res) {
            assert(res != get<0>(result));
            --get<1>(result);
        }
    }
    return result;
}

template <std::unsigned_integral LimbT>
inline void ushift_right(std::span<const LimbT> u, unsigned int shift, LimbT* r) noexcept
{
    assert(!u.empty());
    assert(shift < std::numeric_limits<LimbT>::digits);

    size_t lshift = std::numeric_limits<LimbT>::digits - shift;

    LimbT const* ub = u.data(), * ue = &u.back();
    LimbT* re = r + u.size() - 1;

    LimbT result = *ue << lshift;
    *re = (*ue) >> shift;
    while (ub != ue) {
        *re-- |= (*--ue) << lshift;
        *re = (*ue) >> shift;
    }
    return result;
}

// inplace
template <std::unsigned_integral LimbT>
inline LimbT ushift_right(std::span<LimbT> u, unsigned int shift) noexcept
{
    return ushift_right<LimbT>(u, shift, u.data());
}

template <std::unsigned_integral LimbT>
inline LimbT ushift_right(LimbT& uh, std::span<const LimbT> u, unsigned int shift, LimbT* rb) noexcept
{
    assert(shift < std::numeric_limits<LimbT>::digits);
    size_t lshift = std::numeric_limits<LimbT>::digits - shift;
    LimbT result;
    
    if (u.empty()) {
        result = uh << lshift;
    } else {
        LimbT const* ub = u.data(), * ue = &u.back();

        result = *ub << lshift;
        *rb = (*ub) >> shift;
        while (ub != ue) {
            *rb++ |= (*++ub) << lshift;
            *rb = (*ub) >> shift;
        }
        *rb |= uh << lshift;
    }
    uh >>= shift;

    return result;
}

//// r = [U / 2 ^ shift]
//template <std::unsigned_integral LimbT>
//LimbT ushift_right(LimbT& uh, std::span<const LimbT> ul, unsigned int shift, LimbT* rl)
//{
//    assert(shift < std::numeric_limits<LimbT>::digits);
//    size_t lshift = std::numeric_limits<LimbT>::digits - shift;
//    LimbT rll = uh << lshift;
//    uh >>= shift;
//    if (!ul.empty()) {
//        LimbT const* ub = ul.data(), * ue = ub + ul.size() - 1;
//        LimbT* re = rl + ul.size() - 1;
//        for (;; --re, --ue) {
//            LimbT tmp = rll | ((*ue) >> shift);
//            rll = (*ue) << lshift;
//            *re = tmp;
//            if (ub == ue) [[unlikely]] break;
//        }
//    }
//    return rll;
//}

template <std::unsigned_integral LimbT>
inline LimbT ushift_right(LimbT& uh, std::span<LimbT> u, unsigned int shift) noexcept
{
    return ushift_right<LimbT>(uh, u, shift, u.data());
}

// r = U * 2 ^ shift
template <std::unsigned_integral LimbT>
LimbT ushift_left(LimbT& uh, std::span<const LimbT> ul, unsigned int shift, LimbT* rl) noexcept
{
    assert(shift < std::numeric_limits<LimbT>::digits);
    size_t rshift = std::numeric_limits<LimbT>::digits - shift;
    LimbT uhh = uh >> rshift;
    uh <<= shift;
    if (!ul.empty()) {
        uh |= ul.back() >> rshift;
        LimbT const* ub = ul.data(), * ue = ub + ul.size() - 1;
        LimbT* re = rl + ul.size() - 1;
        for (;; --re) {
            *re = (*ue) << shift;
            if (ub == ue) [[unlikely]] break;
            --ue;
            (*re) |= (*ue) >> rshift;
        }
    }
    return uhh;
}

template <std::unsigned_integral LimbT>
inline LimbT ushift_left(std::span<LimbT> u, unsigned int shift) noexcept
{
    return ushift_left<LimbT>(u.back(), u.first(u.size() - 1), shift, u.data());
}

template <std::unsigned_integral LimbT>
inline void uor(std::span<const LimbT> u, std::span<const LimbT> v, std::span<LimbT> r) noexcept
{
    assert(r.size() >= (std::max)(u.size(), v.size()));
    LimbT const* ub = u.data(), * ue = ub + u.size();
    LimbT const* vb = v.data(), * ve = vb + v.size();
    LimbT* rb = r.data(), * re = rb + r.size();

    for (;; ++ub, ++vb, ++rb) {
        if (ub != ue) {
            if (vb != ve) {
                *rb = *ub | *vb;
                continue;
            } else {
                do {
                    *rb = *ub;
                    ++ub; ++rb;
                } while (ub != ue);
            }
        } else {
            for (; vb != ve; ++vb, ++rb) {
                *rb = *vb;
            }
        }
        break;
    }
}

// prereq: size(r) >= max(size(u), size(v))
template <std::unsigned_integral LimbT, typename OpT>
inline void perlimb_op(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, LimbT * rb, OpT && op) noexcept
{
    LimbT const* ub = u.data(), * ue = ub + u.size();
    LimbT const* vb = v.data(), * ve = vb + v.size();

    for (;; ++ub, ++vb, ++rb) {
        if (ub != ue) {
            if (vb != ve) {
                *rb = std::forward<OpT>(op)(*ub, *vb);
                continue;
            } else {
                *rb++ = std::forward<OpT>(op)(*ub++, vh);
                while (ub != ue) {
                    *rb++ = *ub++;
                }
                *rb = uh;
            }
        } else if (vb != ve) {
            *rb++ = std::forward<OpT>(op)(*vb++, uh);
            while (vb != ve) {
                *rb++ = *vb++;
            }
            *rb = vh;
        } else {
            *rb = std::forward<OpT>(op)(uh, vh);
        }
        break;
    }
}

template <std::unsigned_integral LimbT>
inline void uor(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, LimbT* r) noexcept
{
    perlimb_op<LimbT>(uh, u, vh, v, r, [](LimbT l, LimbT r)->LimbT { return l | r; });
}

template <std::unsigned_integral LimbT>
inline void uxor(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, LimbT* r) noexcept
{
    perlimb_op<LimbT>(uh, u, vh, v, r, [](LimbT l, LimbT r)->LimbT { return l ^ r; });
}

template <std::unsigned_integral LimbT>
inline void uxor(std::span<const LimbT> u, std::span<const LimbT> v, std::span<LimbT> r)
{
    assert(r.size() >= (std::max)(u.size(), v.size()));
    LimbT const* ub = u.data(), * ue = ub + u.size();
    LimbT const* vb = v.data(), * ve = vb + v.size();
    LimbT* rb = r.data(), * re = rb + r.size();

    for (;; ++ub, ++vb, ++rb) {
        if (ub != ue) {
            if (vb != ve) {
                *rb = *ub ^ *vb;
                continue;
            } else {
                do {
                    *rb = *ub;
                    ++ub; ++rb;
                } while (ub != ue);
            }
        } else {
            for (; vb != ve; ++vb, ++rb) {
                *rb = *vb;
            }
        }
        break;
    }
}

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

// inplace version
template <std::unsigned_integral LimbT>
LimbT uadd_partial_unchecked(LimbT *& ub, LimbT const* vb, LimbT const* ve) noexcept
{
    LimbT c = 0;
    for (; vb != ve; ++ub, ++vb) {
        *ub = numetron::arithmetic::uadd1c(*ub, *vb, c);
    }
    return c;
}

// u size must be >= v size
template <std::unsigned_integral LimbT>
inline LimbT uadd_unchecked(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT* rb)
{
    LimbT c = uadd_partial_unchecked(ub, vb, ve, rb);
    if (c) {
        if (ub == ue) return c;
        do {
            std::tie(c, *rb++) = numetron::arithmetic::uadd1(*ub++, c);
            if (ub == ue) return c;
        } while (c);
    }
    std::copy(ub, ue, rb);
    return c;
}

// u size must be >= v size
// inplace version
template <std::unsigned_integral LimbT>
inline LimbT uadd_unchecked(LimbT * ub, LimbT const* ue, LimbT const* vb, LimbT const* ve)
{
    LimbT c = uadd_partial_unchecked(ub, vb, ve);
    if (c && ub != ue) {
        do {
            std::tie(c, *ub) = numetron::arithmetic::uadd1(*ub, c);
            if (!c) return c;
            ++ub;
        } while (ub != ue);
    }
    return c;
}

template <std::unsigned_integral LimbT>
inline LimbT uadd(std::span<const LimbT> u, std::span<const LimbT> v, std::span<LimbT> r) noexcept
{
    assert(r.size() >= (std::max)(u.size(), v.size()));
    LimbT const* ub = u.data(), * ue = ub + u.size();
    LimbT const* vb = v.data(), * ve = vb + v.size();
    LimbT* rb = r.data();

    if (u.size() >= v.size()) {
        return uadd_unchecked(ub, ue, vb, ve, rb);
    } else {
        return uadd_unchecked(vb, ve, ub, ue, rb);
    }
}

// inplace +=
template <std::unsigned_integral LimbT>
inline LimbT uadd(std::span<LimbT> u, std::span<const LimbT> v) noexcept
{
    return uadd<LimbT>(u, v, u);
}

template <std::unsigned_integral LimbT>
LimbT usub(std::span<const LimbT> u, std::span<const LimbT> v, std::span<LimbT> r) noexcept
{
    assert(r.size() >= (std::max)(u.size(), v.size()));
    LimbT const* ub = u.data(), * ue = ub + u.size();
    LimbT const* vb = v.data(), * ve = vb + v.size();
    LimbT* rb = r.data(); // , * re = rb + r.size();
    LimbT c = 0;
    for (;; ++ub, ++vb, ++rb) {
        if (ub != ue) {
            if (vb != ve) {
                std::tie(c , *rb) = numetron::arithmetic::usub1c(*ub, *vb, c);
                continue;
            } else if (c) {
                do {
                    std::tie(c, *rb) = numetron::arithmetic::usub1(*ub, c);
                    if (!c) break;
                    ++ub; ++rb;
                } while (ub != ue);
            }
        } else {
            for (; vb != ve; ++vb, ++rb) {
                std::tie(c, *rb) = numetron::arithmetic::usub1c(LimbT{0}, *vb, c);
            }
        }
        return c;
    }
}

// (c, [u]) <- [u] * v + cl; returns c
template <std::unsigned_integral LimbT, typename UIteratorT>
inline LimbT umul1_inplace(UIteratorT ub, UIteratorT ue, LimbT v, LimbT cl = 0) noexcept
{
    while (ub != ue) {
        auto [h, l] = arithmetic::umul1(*ub, v);
        *ub = arithmetic::uadd1ca(l, cl, h);
        cl = h;
        ++ub;
    }
    return cl;
}

template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
inline LimbT umul1S4(UIteratorT ub, UIteratorT ue, LimbT v, RIteratorT& r) noexcept
{
    assert(!(2 & (ue - ub)));
    // unrolled first iteration
    auto [h, r0] = arithmetic::umul1(*ub, v);
    auto [h1, l1] = arithmetic::umul1(*(ub + 1), v);
    auto [h2, l2] = arithmetic::umul1(*(ub + 2), v);
    auto [rcl, l3] = arithmetic::umul1(*(ub + 3), v);

    *r = r0;
    *(r + 1) = arithmetic::uadd1ca(l1, h, h1);
    *(r + 2) = arithmetic::uadd1ca(l2, h1, h2);
    auto[c, r3] = arithmetic::uadd1(l3, h2);
    *(r + 3) = r3;
    
    r += 4;
    ub += 4;

    // remaining iterations
    while (ub != ue) {
        auto [h, l] = arithmetic::umul1(*ub, v);
        auto [h1, l1] = arithmetic::umul1(*(ub + 1), v);
        auto [h2, l2] = arithmetic::umul1(*(ub + 2), v);
        auto [h3, l3] = arithmetic::umul1(*(ub + 3), v);
        *r = arithmetic::uadd1cca(l, rcl, c, h); c = 0;
        *(r + 1) = arithmetic::uadd1ca(l1, h, h1);
        *(r + 2) = arithmetic::uadd1ca(l2, h1, h2);
        *(r + 3) = arithmetic::uadd1ca(l3, h2, h3);
        rcl = h3;
        ub += 4;
        r += 4;
    }
    return rcl + c;
}

template <std::unsigned_integral LimbT,  typename UIteratorT, typename RIteratorT>
inline LimbT umul1S2(UIteratorT ub, UIteratorT ue, LimbT v, RIteratorT& r) noexcept
{
    assert(!(1&(ue - ub)));
    // unrolled first iteration
    auto [r1c, r0] = arithmetic::umul1(*ub, v);
    auto [rcl, r1_0] = arithmetic::umul1(*(ub + 1), v);
    auto [c, r1] = arithmetic::uadd1(r1_0, r1c);
    *r = r0; ++r;
    *r = r1; ++r;

    // remaining iterations
    while ((ub+=2) != ue) {
        auto [h, l] = arithmetic::umul1(*ub, v);
        auto [h1, l1] = arithmetic::umul1(*(ub + 1), v);
        *r = arithmetic::uadd1c(l, rcl, c); ++r;
        *r = arithmetic::uadd1c(l1, h, c); ++r;
        rcl = h1;
    }
    return rcl + c;
}

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
inline LimbT umul1S2(LimbT const* ub, LimbT const* ue, LimbT v, LimbT*& r) noexcept
{
    assert(!(1 & (ue - ub)));
    // BMI2 path: use mulx (rdx is implicit source)

    LimbT rcl;
    LimbT l, l1, h, h1; // temporaries
    unsigned char c; // carry (temporary)

    __asm__(
        "mov  %[v],  %%rdx\n\t"   // rdx = v (implicit for mulx)

        // Unrolled first pair:
        "mulx (%[ub]), %[l], %[h]\n\t"      // [h, l] = arithmetic::umul1(*ub, v);
        "mulx 8(%[ub]), %[l1], %[rcl]\n\t"   // [rcl, l1] = arithmetic::umul1(*(ub + 1), v);

        "mov  %[l], (%[r])\n\t"               // *r = l;

        "add  %[h], %[l1]\n\t"             // l1 = l1 + h
        "mov  %[l1], 8(%[r])\n\t"            // *(r+1) = r1
        "setc %b[c]\n\t"                        // c = carry

        "add  $16, %[r]\n\t" //"lea 16(%[r]) %[r]\n\t" // r  += 2 //  "add  $16, %[r]\n\t"
        "add  $16, %[ub]\n\t" //"lea 16(%[ub]) %[ub]\n\t" // ub += 2 // "add  $16, %[ub]\n\t"
        
        // If done, skip loop
        "cmp  %[ub], %[ue]\n\t"
        "je   .Lexit%=\n\t"

        ".Lnext%=:\n\t"
        
        "mulx (%[ub]), %[l], %[h]\n\t"      // [h, l] = arithmetic::umul1(*ub, v);
        "mulx 8(%[ub]), %[l1], %[h1]\n\t"   // [h1, l1] = arithmetic::umul1(*(ub + 1), v);
        
        "add $0xFF, %[c]\n\t"
        "adcq %[rcl], %[l]\n\t" // arithmetic::uadd1c(l, rcl, c) ->l
        "adcq %[h], %[l1]\n\t"  //arithmetic::uadd1c(l1, h, c) -> l1
        "setc %b[c]\n\t"
        "mov  %[h1], %[rcl]\n\t"

        "mov  %[l], (%[r])\n\t"
        "mov  %[l1], 8(%[r])\n\t"

        "add  $16, %[r]\n\t"
        "add  $16, %[ub]\n\t"
        "cmp  %[ub], %[ue]\n\t"
        "jne  .Lnext%=\n\t"
        
        ".Lexit%=:\n\t"
        "add $0xFF, %[c]\n\t"
        "adc $0, %[rcl]\n\t"
        
        : "=&r"(ub), "=&r"(r), [c] "=&qm" (c), [rcl]"=&r"(rcl), [l]"=&r"(l), [h]"=&r"(h), [l1]"=&r"(l1), [h1]"=&r"(h1)
        : [ub] "0"(ub), [ue] "r"(ue), [v]"r"(v), [r]"1"(r)
        : "rdx", "cc", "memory"
    );
    return rcl;
}
#endif

// (rh, r[u.size()]) <- [u] * v; returns rh
template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
inline LimbT umul1(UIteratorT ub, UIteratorT ue, LimbT v, RIteratorT& r) noexcept
{
    assert(ub != ue);
    // unrolled first iteration
    auto [h, l] = arithmetic::umul1(*ub, v);
    *r = l; ++r;
    // unrolled second iteration
    
    if (++ub == ue) return h;
    
    auto [h1, l1] = arithmetic::umul1(*ub, v);
    *r = arithmetic::uadd1ca(l1, h, h1);
    ++r;
    
    // remaining iterations
    while (++ub != ue) {
        auto [h, l] = arithmetic::umul1(*ub, v);
        *r = arithmetic::uadd1ca(l, h1, h);
        ++r;
        h1 = h;
    }
    return h1;
}

template <std::unsigned_integral LimbT, typename UIteratorT, typename ResultIteratorT>
inline LimbT umul1_addS2(UIteratorT ub, UIteratorT ue, LimbT v, ResultIteratorT& r, LimbT climb, unsigned char c) noexcept
{
    assert(ue != ub);
    assert(!(1 & (ue - ub)));
    unsigned char c2 = 0;
    do {
        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
        auto [h1, l1] = numetron::arithmetic::umul1(*(ub + 1), v);
        *r = numetron::arithmetic::uadd1c2(l, climb, *r, c, c2);
        *(r + 1) = numetron::arithmetic::uadd1c2(l1, h, *(r + 1), c, c2);
        r += 2;
        climb = h1;
    } while ((ub += 2) != ue);
    return climb + c + c2;
}

//template <std::unsigned_integral LimbT, typename UIteratorT, typename ResultIteratorT>
//inline LimbT umul1_addS4(UIteratorT ub, UIteratorT ue, LimbT v, ResultIteratorT& r) noexcept
//{
//    assert(!(3 & (ue - ub)));
//    unsigned char c = 0;
//    auto [h, l] = numetron::arithmetic::umul1(*ub, v);
//    auto [h1, l1] = arithmetic::umul1(*(ub + 1), v);
//    auto [h2, l2] = arithmetic::umul1(*(ub + 2), v);
//    auto [rcl, l3] = arithmetic::umul1(*(ub + 3), v);
//
//    auto [c1, r0] = numetron::arithmetic::uadd1(l, *r); *r = r0;
//    *(r + 1) = numetron::arithmetic::uadd1c2(l1, h, *(r + 1), c, c1);
//    *(r + 2) = numetron::arithmetic::uadd1c2(l2, h1, *(r + 2), c, c1);
//    *(r + 3) = numetron::arithmetic::uadd1c2(l3, h2, *(r + 3), c, c1);
//    r += 4;
//    ub += 4;
//    while (ub != ue) {
//        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
//        auto [h1, l1] = numetron::arithmetic::umul1(*(ub + 1), v);
//        auto [h2, l2] = arithmetic::umul1(*(ub + 2), v);
//        auto [h3, l3] = arithmetic::umul1(*(ub + 3), v);
//        *r = numetron::arithmetic::uadd1c2(l, rcl, *r, c, c1);
//        *(r + 1) = numetron::arithmetic::uadd1c2(l1, h, *(r + 1), c, c1);
//        *(r + 2) = numetron::arithmetic::uadd1c2(l2, h1, *(r + 2), c, c1);
//        *(r + 3) = numetron::arithmetic::uadd1c2(l3, h2, *(r + 3), c, c1);
//        r += 4;
//        rcl = h3;
//        ub += 4;
//    }
//    return rcl + c + c1;
//}

template <std::unsigned_integral LimbT, typename UIteratorT, typename ResultIteratorT>
inline LimbT umul1_addS4(UIteratorT ub, UIteratorT ue, LimbT v, ResultIteratorT& r) noexcept
{
    assert(!(3 & (ue - ub)));
    LimbT rcl = 0;
    unsigned char c = 0;
    //unsigned char c1 = 0;
    
    do {
        auto [h, l] = arithmetic::umul1(*ub, v);
        auto [h1, l1] = arithmetic::umul1(*(ub + 1), v);
        auto [h2, l2] = arithmetic::umul1(*(ub + 2), v);
        auto [h3, l3] = arithmetic::umul1(*(ub + 3), v);

        auto tmp = numetron::arithmetic::uadd1cca(l, rcl, c, h);
        auto [c0, r0] = numetron::arithmetic::uadd1(tmp, *r);
        *r = r0;

        tmp = numetron::arithmetic::uadd1cca(l1, h, c0, h1);
        auto [c1, r1] = numetron::arithmetic::uadd1(tmp, *(r + 1));
        *(r + 1) = r1;

        tmp = numetron::arithmetic::uadd1cca(l2, h1, c1, h2);
        auto [c2, r2] = numetron::arithmetic::uadd1(tmp, *(r + 2));
        *(r + 2) = r2;

        tmp = numetron::arithmetic::uadd1cca(l3, h2, c2, h3);
        auto [c3, r3] = numetron::arithmetic::uadd1(tmp, *(r + 3));
        *(r + 3) = r3;

        rcl = h3;
        c = c3;
        r += 4;
        ub += 4;
        //*r = numetron::arithmetic::uadd1c2(l, rcl, *r, c, c1);
        //*(r + 1) = numetron::arithmetic::uadd1c2(l1, h, *(r + 1), c, c1);
        //*(r + 2) = numetron::arithmetic::uadd1c2(l2, h1, *(r + 2), c, c1);
        //*(r + 3) = numetron::arithmetic::uadd1c2(l3, h2, *(r + 3), c, c1);
        //r += 4;
        //rcl = h3;
        //ub += 4;
    } while (ub != ue);
    //return rcl + c + c1;
    return rcl + c;
}
#if 0
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
template <std::unsigned_integral LimbT>
requires(sizeof(LimbT) == 8)
__attribute__((always_inline)) inline LimbT umul1_addS2(LimbT const* ub, LimbT const* ue, LimbT v, LimbT*& r) noexcept
{
    assert(!(1 & (ue - ub)));
    // BMI2 path: use mulx (rdx is implicit source)

    LimbT rcl;
    LimbT l, l1, h, h1; // temporaries
    unsigned char c, c1; // carris (temporaries)

    __asm__(
        "mov  %[v], %%rdx\n\t"   // rdx = v (implicit for mulx)

        // Unrolled first pair:
        "mulx (%[ub]), %[l], %[h]\n\t"      // [h, l] = arithmetic::umul1(*ub, v);
        "mulx 8(%[ub]), %[l1], %[rcl]\n\t"   // [rcl, l1] = arithmetic::umul1(*(ub + 1), v);

        "addq  %[l], (%[r])\n\t"            // *r += l;

        "adcq  %[h], %[l1]\n\t"             // l1 += h +(c)
        "setc %b[c]\n\t"
        "addq  %[l1], 8(%[r])\n\t"            // *(r+1) += l1
        "setc %b[c1]\n\t"                        // c = carry
        
        "add  $16, %[r]\n\t" //"lea 16(%[r]) %[r]\n\t" // r  += 2 //  "add  $16, %[r]\n\t"
        "add  $16, %[ub]\n\t" //"lea 16(%[ub]) %[ub]\n\t" // ub += 2 // "add  $16, %[ub]\n\t"
        
        // If done, skip loop
        "cmp  %[ub], %[ue]\n\t"
        "je   .Lexit%=\n\t"

        ".Lnext%=:\n\t"
        
        "mulx (%[ub]), %[l], %[h]\n\t"      // [h, l] = arithmetic::umul1(*ub, v);
        "mulx 8(%[ub]), %[l1], %[h1]\n\t"   // [h1, l1] = arithmetic::umul1(*(ub + 1), v);
        
        "add  $0xFF, %[c]\n\t"
        "adcq %[rcl], %[l]\n\t" // arithmetic::uadd1c(l, rcl, c) ->l, c
        "adcq %[h], %[l1]\n\t" // arithmetic::uadd1c(l1, h, c) ->l1, c
        "setc %b[c]\n\t"
        "add  $0xFF, %[c1]\n\t"
        "adcq  %[l], (%[r])\n\t" // *r += l
        "adcq  %[l1], 8(%[r])\n\t" // *r += l1
        "setc %b[c1]\n\t"
        "mov  %[h1], %[rcl]\n\t"

        "add  $16, %[r]\n\t"
        "add  $16, %[ub]\n\t"
        "cmp  %[ub], %[ue]\n\t"
        "jne  .Lnext%=\n\t"
       
        ".Lexit%=:\n\t"
        "add %b[c1], %b[c]\n\t"
        "movzx %b[c], %[l]\n\t"
        "addq %[l], %[rcl]\n\t"
        
        : "=&r"(ub), "=&r"(r), [c] "=&q" (c), [c1] "=&q" (c1), [rcl]"=&r"(rcl), [l]"=&r"(l), [h]"=&r"(h), [l1]"=&r"(l1), [h1]"=&r"(h1)
        : [ub] "0"(ub), [ue] "r"(ue), [v]"r"(v), [r]"1"(r)
        : "rdx", "cc", "memory"
    );
    //if constexpr (sizeof(LimbT) == 8) {
    //    std::cout << "r0: " << std::hex << *(r - 2);
    //    std::cout << ", r1: " << std::hex << *(r - 1);
    //    std::cout << ", rcl: " << std::hex << rcl;
    //    std::cout << ", c: " << (int)c << ", c1: " << (int)c1 << "\n";
    //}
    return rcl;
    //while (ub != ue) {
    //    auto [h, l] = numetron::arithmetic::umul1(*ub, v);
    //    auto [h1, l1] = numetron::arithmetic::umul1(*(ub + 1), v);
    //    *r = numetron::arithmetic::uadd1c2(l, rcl, *r, c, c1);
    //    *(r + 1) = numetron::arithmetic::uadd1c2(l1, h, *(r + 1), c, c1);
    //    r += 2;
    //    rcl = h1;
    //    ub += 2;
    //    //std::cout << std::hex << "r0: " << *(r - 2);
    //    //std::cout << ", r1: " << std::hex << *(r - 1);
    //    //std::cout << ", rcl: " << std::hex << rcl << "\n";
    //}
    //return rcl + c + c1;
}
#endif
#endif
template <std::unsigned_integral LimbT, typename UIteratorT, typename ResultIteratorT>
inline LimbT umul1_addS2(UIteratorT ub, UIteratorT ue, LimbT v, ResultIteratorT& r) noexcept
{
    assert(!(1 & (ue - ub)));
    unsigned char c = 0;
    auto [h, l] = numetron::arithmetic::umul1(*ub, v);
    auto [rcl, l1] = numetron::arithmetic::umul1(*(ub + 1), v);
    auto [c1, r0] = numetron::arithmetic::uadd1(l, *r);
    *r = r0;
    *(r+1) = numetron::arithmetic::uadd1c2(l1, h, *(r+1), c, c1);
    r += 2;
    //if constexpr (sizeof(LimbT) == 8) {
    //    std::cout << std::hex << "r0: " << *(r - 2);
    //    std::cout << ", r1: " << std::hex << *(r - 1);
    //    std::cout << ", rcl: " << std::hex << rcl;
    //    std::cout << ", c: " << (int) c << ", c1: " << (int) c1 << "\n";
    //}
    
    while ((ub+=2) != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
        auto [h1, l1] = numetron::arithmetic::umul1(*(ub+1), v);
        *r = numetron::arithmetic::uadd1c2(l, rcl, *r, c, c1);
        *(r + 1) = numetron::arithmetic::uadd1c2(l1,  h, *(r + 1), c, c1);
        r += 2;
        rcl = h1;
        //std::cout << std::hex << "r0: " << *(r - 2);
        //std::cout << ", r1: " << std::hex << *(r - 1);
        //std::cout << ", rcl: " << std::hex << rcl << "\n";
    }
    return rcl + c + c1;
}

// (c, p[u.size()]) <- [u] * v + p[u.size()]; returns [c, p + u.size()]
template <std::unsigned_integral LimbT, typename UIteratorT, typename ResultIteratorT>
inline LimbT umul1_add(UIteratorT ub, UIteratorT ue, LimbT v, ResultIteratorT & r) noexcept
{
    assert(ub != ue);
#if 0
    auto [h, l] = numetron::arithmetic::umul1(*ub, v);
    auto [cr, rl] = numetron::arithmetic::uadd1(l, *r);
    *r = rl; ++r;
    LimbT rh = h + cr;
    while (++ub != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
        auto [cr, rl] = numetron::arithmetic::uadd1(l, *r, rh);
        *r = rl; ++r;
        rh = h + cr;
        assert(rh >= h); // if l > 1 => h < 0xff...fe => no cl overflow
    }
    return rh;
#else
    unsigned char cl = 0;
    auto [previous, l] = numetron::arithmetic::umul1(*ub, v);
    auto [cl2, current] = numetron::arithmetic::uadd1(l, *r);
    *r = current; ++r;
    while (++ub != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
        *r = numetron::arithmetic::uadd1c2(l, *r, previous, cl, cl2);
        ++r;
        previous = h;
    }
    return previous + cl + cl2;
#endif
}

// (uh, [ul]) * v + cl -> (rh, r[ul.size() + 1]); returns rh
template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
inline LimbT umul1(LimbT uh, UIteratorT ulb, UIteratorT ule, LimbT v, RIteratorT& r) noexcept
{
    LimbT pcl = umul1<LimbT>(ulb, ule, v, r);
    auto [h, ph] = numetron::arithmetic::umul1(uh, v);
    auto [c, res] = numetron::arithmetic::uadd1(ph, pcl);
    *r = res; ++r;
    return h + c;
}

// (c, ph, p[ul.size()]) <- (uhh, uh, [ul]) * v + cl; returns (c, ph)
template <std::unsigned_integral LimbT>
inline std::tuple<LimbT, LimbT, LimbT> umul1(LimbT uhh, LimbT uh, std::span<const LimbT> ul, LimbT v, LimbT* p, LimbT cl = 0) noexcept
{
    LimbT pcl = umul1<LimbT>(ul, v, p, cl);
    auto [h, ph] = numetron::arithmetic::umul1<LimbT>(uh, v);
    ph += pcl;
    pcl = (ph < pcl) + h;
    auto [hh, phh] = numetron::arithmetic::umul1<LimbT>(uhh, v);
    phh += pcl;
    pcl = (phh < pcl) + hh;
    return { pcl, phh, ph };
}

// (c, p[size(ul) + 2]) <- (uhh, uh, [ul]) * v + p[size(ul)] + cl; returns c
template <std::unsigned_integral LimbT>
inline LimbT umul1_add(LimbT uhh, LimbT uh, std::span<const LimbT> ul, LimbT v, LimbT* p, LimbT cl = 0) noexcept
{
    LimbT pcl = umul1_sum<LimbT>(ul, v, p, cl);
    p += ul.size();
    auto [h, ph] = numetron::arithmetic::umul1<LimbT>(uh, v);
    auto [rc, rh] = numetron::arithmetic::uadd1(ph, *p, pcl);
    *p++ = rh;
    auto [hh, phh] = numetron::arithmetic::umul1<LimbT>(uhh, v);
    auto [rch, rhh] = numetron::arithmetic::uadd1(phh, *p, rc);
    *p++ = rh;
    return rch;
}

//template <std::unsigned_integral LimbT>
//inline void umul(std::span<const LimbT> u, std::span<const LimbT> v, std::span<LimbT> r) noexcept
//{
//    if (!u.size() || !v.size()) [[unlikely]] return;
//
//    LimbT const* ub = u.data(), * ue = ub + u.size();
//    LimbT const* vb = v.data(), * ve = vb + v.size();
//    LimbT* rb = r.data(), * re = rb + r.size();
//    assert(r.size() >= u.size() + v.size());
//
//    // first line
//    LimbT cl = 0;
//    do {
//        auto [h, l] = numetron::arithmetic::umul1(*ub, *vb);
//        l += cl;
//        cl = (l < cl) + h;
//        *rb = l;
//        ++ub; ++rb;
//    } while (ub != ue);
//    if (cl) { *rb++ = cl; }
//    while (rb != re) { *rb++ = 0; }
//    rb = r.data() + u.size();
//
//    // next lines
//    for (++vb; vb != ve; ++vb) {
//        cl = 0;
//        ub = u.data();
//        rb -= u.size() - 1;
//        do {
//            auto [h, l] = numetron::arithmetic::umul1(*ub, *vb);
//            auto [h1, l1] = numetron::arithmetic::uadd1c(l, *rb, cl);
//            cl = h + h1;
//            *rb = l1;
//            ++ub; ++rb;
//        } while (ub != ue);
//        if (cl) {
//            *rb = numetron::arithmetic::uadd1(*rb, cl).second;
//        }
//    }
//}
#if 0
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
template <std::unsigned_integral LimbT>
requires std::is_same_v<LimbT, uint64_t>
inline uint64_t* umul(uint64_t const* ub, uint64_t const* ue, uint64_t const* vb, uint64_t const* ve, uint64_t * rb) noexcept
{
    __asm_mul_basecase(rb, const_cast<uint64_t*>(ub), (ue - ub), const_cast<uint64_t*>(vb), (ve - vb));
    return rb + (ue - ub) + (ve - vb);
}
#endif
#endif
// base case mul: {u} * {v} -> {rb, re}
// returns re
#if 0
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, ResultIteratorT rb) noexcept
{
    assert(ub != ue && vb != ve);
    //if (ub == ue || vb == ve) [[unlikely]] return r;

    size_t dec_usz = ue - ub - 1;

    // first line
    
    if ((dec_usz & 3) == 3) {
        LimbT rh = umul1S4<LimbT>(ub, ue, *vb, rb);
        *rb = rh;
        for (++vb; vb != ve; ++vb) {
            rb -= dec_usz;
            rh = umul1_addS4(ub, ue, *vb, rb);
            *rb = rh;
        }
    } else {/*
        
        if (dec_usz & 1) {
        LimbT rh = umul1S2<LimbT>(ub, ue, *vb, rb);
        *rb = rh;
        for (++vb; vb != ve; ++vb) {
            rb -= dec_usz;
            rh = umul1_addS2(ub, ue, *vb, rb);
            *rb = rh;
        }
    } else {*/
#if 1
        LimbT rh = umul1<LimbT>(ub, ue, *vb, rb);
        *rb = rh;
        for (++vb; vb != ve; ++vb) {
            rb -= dec_usz;
            rh = umul1_add(ub, ue, *vb, rb);
            *rb = rh;
        }
#else
        LimbT rh = umul1<LimbT>(ub, ue, *vb, rb);
        *rb = rh;
        for (++vb; vb != ve; ++vb) {
            rb -= dec_usz;
            auto [r1c, r0_0] = numetron::arithmetic::umul1(*ub, *vb);
            auto [cl, r0] = numetron::arithmetic::uadd1(r0_0, *rb);
            *rb = r0;
            rh = umul1_addS2(ub + 1, ue, *vb, ++rb, r1c, cl);
            *rb = rh;
        }
#endif
    }
    return rb + 1;
    // next lines
    
    //for (++vb; vb != ve; ++vb) {
    //    rb -= dec_usz;
    //    LimbT rh = umul1_add(ub, ue, *vb, rb);
    //    *rb = rh;

    //    // first iteration unrolled
    //    //unsigned char cl = 0;
    //    //auto ub_it = ub;
    //    //LimbT vb_val = *vb;
    //    //auto [previous, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
    //    //auto [cl2, current] = numetron::arithmetic::uadd1(l, *rb);
    //    //*rb = current; ++rb;
    //    //while (++ub_it != ue) {
    //    //    auto [h, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
    //    //    *rb = numetron::arithmetic::uadd1c2(l, *rb, previous, cl, cl2);
    //    //    ++rb;
    //    //    previous = h;
    //    //}
    //    //*rb = previous + cl + cl2;
    //}
    //return rb + 1;
}
#endif



// base case mul: {u} * {v} -> {rb, re}
// returns re
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, ResultIteratorT r) noexcept
{
    assert(ub != ue && vb != ve);
    //if (ub == ue || vb == ve) [[unlikely]] return r;

    ResultIteratorT rb = r;

    // first line
#if 0
    unsigned char cl = 0;
    LimbT previous = 0;
    do {
        auto [h, l] = numetron::arithmetic::umul1(*ub, *vb);
        auto [nc, current] = numetron::arithmetic::uadd1cl(l, previous, cl);
        *rb = current; ++rb;
        cl = nc;
        previous = h;
    } while (++ub != ue);
    *rb = previous + cl;
#else
    unsigned char cl = 0;
    // unrolled first iteration
    auto [previous, current] = numetron::arithmetic::umul1(*ub, *vb);
    *rb = current; ++rb;
    // unrolled second iteration
    auto ub_it = ub;
    if (++ub_it != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub_it, *vb);
        auto [nc, current] = numetron::arithmetic::uadd1(l, previous);
        *rb = current; ++rb;
        cl = nc;
        previous = h;
        while (++ub_it != ue) {
            auto [h, l] = numetron::arithmetic::umul1(*ub_it, *vb);
            *rb = numetron::arithmetic::uadd1c(l, previous, cl);
            ++rb;
            previous = h;
        }
    }
    *rb = previous + cl;
#endif
    // next lines
    //for (++vb; vb != ve; ++vb) {
    //    LimbT cl = 0;
    //    ub = u.data();
    //    rb -= u.size() - 1;
    //    do {
    //        auto [h, l] = numetron::arithmetic::umul1(*ub, *vb);
    //        auto [h1, l1] = numetron::arithmetic::uadd1c(l, *rb, cl);
    //        cl = h + h1;
    //        *rb = l1;
    //        ++ub; ++rb;
    //    } while (ub != ue);
    //    *rb = cl;
    //}
    //return rb + 1;

    //for (++vb; vb != ve; ++vb) {
    //    cl = 0;
    //    unsigned char cl2 = 0;
    //    previous = 0;
    //    ub = u.data();
    //    rb -= u.size() - 1;
    //    do {
    //        auto [h, l] = numetron::arithmetic::umul1(*ub, *vb);
    //        //auto [nc, tmp] = numetron::arithmetic::uadd1cl(l, *rb, cl);
    //        //auto [nc2, current] = numetron::arithmetic::uadd1cl(tmp, previous, cl2);
    //        //auto [nc, nc2, current] = numetron::arithmetic::uadd1cl2(l, *rb, previous, cl, cl2);
    //        *rb = numetron::arithmetic::uadd1cl2x(l, *rb, previous, cl, cl2);
    //        ++rb;
    //        //*rb = current; ++rb;
    //        //cl = nc; cl2 = nc2; 
    //        previous = h;
    //        ++ub;
    //    } while (ub != ue);
    //    *rb = previous + cl + cl2;
    //}
    //return rb + 1;
#if 0
    for (++vb; vb != ve; ++vb) {
        cl = 0;
        ub = u.data();
        rb -= u.size() - 1;

        // first iteration unrolled
        auto [previous, l] = numetron::arithmetic::umul1(*ub, *vb);
        auto [cl2, current] = numetron::arithmetic::uadd1(l, *rb);
        *rb = current; ++rb; ++ub;
        while (ub != ue) {
            auto [h, l] = numetron::arithmetic::umul1(*ub, *vb);
            *rb = numetron::arithmetic::uadd1cl2x(l, *rb, previous, cl, cl2);
            ++rb;
            previous = h;
            ++ub;
        } 
        *rb = previous + cl + cl2;
    }
    return rb + 1;
#else
    size_t dec_usz = ue - ub - 1;
    for (++vb; vb != ve; ++vb) {
        cl = 0;
        ub_it = ub;
        rb -= dec_usz;
        LimbT vb_val = *vb;
        // first iteration unrolled
        auto [previous, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
        auto [cl2, current] = numetron::arithmetic::uadd1(l, *rb);
        *rb = current; ++rb;
        while (++ub_it != ue) {
            auto [h, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
            *rb = numetron::arithmetic::uadd1c2(l, *rb, previous, cl, cl2);
            ++rb;
            previous = h;
        } 
        *rb = previous + cl + cl2;
    }
    return rb + 1;

    //for (++vb; vb != ve; ++vb) {
    //    ub = u.data();
    //    rb -= u.size() - 1;
    //    LimbT vb_val = *vb;
    //    // first iteration unrolled
    //    auto [previous, l] = numetron::arithmetic::umul1(*ub, vb_val);
    //    auto [c, current] = numetron::arithmetic::uadd1(l, *rb);
    //    *rb = current; ++rb; previous += c;
    //    while (++ub != ue) {
    //        auto [h, l] = numetron::arithmetic::umul1(*ub, vb_val);
    //        auto [c, current] = numetron::arithmetic::uadd1(l, *rb, previous);
    //        *rb = current; ++rb;
    //        previous = h + c;
    //    }
    //    *rb = previous;
    //}
    //return rb + 1;
#endif
}

template <std::unsigned_integral LimbT>
void fallback_mul_basecase(LimbT* rb, LimbT const* ub, size_t usz, LimbT const* vb, size_t vsz) noexcept
{
    umul<LimbT, LimbT*>(ub, ub + usz, vb, vb + vsz, rb);
}

#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
template <std::unsigned_integral LimbT>
requires (sizeof(LimbT) == 8)
inline LimbT* umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT* rb) noexcept
{
#if defined(NUMETRON_PLATFORM_AUTODETECT)
    std::call_once(mul_basecase_init_flag, []() {
        uint64_t platform_descriptor = numetron_detect_platform();
        std::cout << "PLATFROM: " << std::hex << "0x" << platform_descriptor << std::dec << std::endl;
        __mul_basecase_ptr = detect_mul_basecase(platform_descriptor);
        if (!__mul_basecase_ptr) 
            __mul_basecase_ptr = &fallback_mul_basecase<LimbT>;
    });
    __mul_basecase_ptr(rb, ub, (ue - ub), vb, (ve - vb));
#else
    NUMETRON_mul_basecase(rb, ub, (ue - ub), vb, (ve - vb));
#endif
    return rb + (ue - ub) + (ve - vb);
}

#else
// befor 3586
//#if defined(_MSC_VER) && defined(_M_X64)
//#if (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)) && (defined(__x86_64__) || defined(_M_X64))
template <std::unsigned_integral LimbT>
inline LimbT* umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT* rb) noexcept
{
    const auto ucnt = ue - ub;
    const auto vcnt = ve - vb;
    if (ucnt == 2) {
        auto [h00, l00] = arithmetic::umul1(*ub, *vb);
        *rb = l00;
        auto [h01, l01] = arithmetic::umul1(*(ub + 1), *vb);
        uint64_t r1 = arithmetic::uadd1ca(l01, h00, h01);
        if (vcnt < 2) {
            *(rb + 1) = r1;
            *(rb + 2) = h01;
            return rb + 3;
        }
        ++vb;
        auto [h10, l10] = arithmetic::umul1(*ub, *vb);
        *(rb + 1) = arithmetic::uadd1ca(r1, l10, h10);
        auto [h11, l11] = arithmetic::umul1(*(ub + 1), *vb);
        h10 = arithmetic::uadd1ca(h10, h01, h11);
        *(rb + 2) = arithmetic::uadd1ca(l11, h10, h11); // h11 can not overflow
        *(rb + 3) = h11;
        return rb + 4;
    }
    if ((ucnt & 3) != 3) return umul<LimbT, LimbT*>(ub, ue, vb, ve, rb);
    
    if (ucnt & 1) {
        if (ucnt & 2) {
            // .Lb11
            auto [h0, l0] = arithmetic::umul1(*ub, *vb);
            auto [h1, l1] = arithmetic::umul1(*(ub + 1), *vb);
            auto [h2, l2] = arithmetic::umul1(*(ub + 2), *vb);
            
            for (auto un = ucnt; un != 3; un -= 4) { // un = 7, 11, 15, ...
                // .Lmtp3
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
            //.Lmed3:
            *rb = l0;
            *(rb + 1) = arithmetic::uadd1ca(h0, l1, h1);
            *(rb + 2) = arithmetic::uadd1ca(h1, l2, h2);
            *(rb + 3) = h2;
            while (++vb != ve) {
                // .Lout3
                rb += 4 - ucnt;
                ub += 3 - ucnt;
                
                auto [h0, l0] = arithmetic::umul1(*ub, *vb);
                auto [h1, l1] = arithmetic::umul1(*(ub + 1), *vb);
                auto [h2, l2] = arithmetic::umul1(*(ub + 2), *vb);
                auto [cov, l0_] = arithmetic::uadd1(*rb, l0);

                for (auto un = ucnt; un != 3; un -= 4) { // un = 7, 11, 15, ...
                    //.Ltp3:
                    *rb = l0_;
                    l0_ = arithmetic::umul4add<LimbT>(ub + 3, *vb, rb + 1, cov, h0, h1, l1, h2, l2);
                    ub += 4; rb += 4;
                }
                //.Led3
                *rb = l0_;
                l1 = arithmetic::uadd1ca(h0, l1, h1);
                *(rb + 1) = arithmetic::uadd1c(*(rb + 1), l1, cov); // ov
                l2 = arithmetic::uadd1ca(h1, l2, h2);
                *(rb + 2) = arithmetic::uadd1c(*(rb + 2), l2, cov); // ov
                *(rb + 3) = arithmetic::uadd1c(LimbT{0}, h2, cov); // ov
            }
            return rb + 4;
        }
    //.Lb01
    ////////////////// h0=r10, l0=r11, h1=r12, l1=r13, h2=rax, l2=rbx, h3=r8, l3=r9
    }
    //.Lbx0

    return umul<LimbT, LimbT*>(ub, ue, vb, ve, rb);
    //return rb + (ue - ub) + (ve - vb);
}
//#endif
#endif
// base case mul with explicit high limbs uh and vh
// returns iterator to one past the last written limb (r + u.size() + v.size() + 2)
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, ResultIteratorT r) noexcept
{
    const size_t n = u.size();
    const size_t m = v.size();

    if (v.empty()) {
        assert(!u.empty());
        LimbT rh = umul1(uh, u.data(), u.data() + u.size(), vh, r);
        if (rh) { *r = rh; ++r; }

        //if (u.empty()) {
        //    auto [h, ph] = numetron::arithmetic::umul1(uh, vh);
        //    *r = ph; ++r;
        //    *r = h; ++r;
        //} else {
        //    LimbT rh = umul1(uh, u.data(), u.data() + u.size(), vh, r);
        //    if (rh) { *r = rh; ++r; }
        //}
        return r;
    } 
    assert(!u.empty()); // because |u| > |v| and |v| != 0

    //else if (!u.size()) {
    //    auto [pcl, ph] = umul1(vh, v, uh, r);

    //    if (ph) *r++ = ph;
    //    if (pcl) {
    //        assert(!pcl); // because cl = 0 in umul1 call
    //    }
    //    assert(!pcl); // because cl = 0 in umul1 call
    //    return r;
    //}

    // Low product
    ResultIteratorT re = umul<LimbT>(u.data(), u.data() + u.size(), v.data(), v.data() + v.size(), r); // may return r if one is empty

    if (!uh && !vh) return re;

    // Ensure top two limbs exist and start from zero
    ResultIteratorT top = r; top += (n + m);
        
    // Clear intermediate limbs if any
    while (re != top) {
        *re++ = LimbT{0};
    }

    if (uh) {
        *re = LimbT{0};
        ++re;
        // Add uh * v at offset n
        auto rvhu = r + u.size();
        LimbT c = umul1_add(v.data(), v.data() + v.size(), uh, rvhu);
        c = uadd1<LimbT>(rvhu, re - rvhu, c, rvhu);
        assert(c == 0);
    }
    if (vh) {
        *re = LimbT{0};
        ++re;
        // Add vh * u at offset m
        auto rvhv = r + v.size();
        LimbT c = umul1_add(u.data(), u.data() + u.size(), vh, rvhv);
        c = uadd1<LimbT>(rvhv, re - rvhv, c, rvhv);
        assert(c == 0);
    
        // Add uh * vh at offset n + m
        if (uh) {
            auto [h, l] = numetron::arithmetic::umul1<LimbT>(uh, vh);
            r += u.size() + v.size();
            auto [c, s] = numetron::arithmetic::uadd1<LimbT>(*r, l);
            *r = s; ++r;
            auto [c2, s2] = numetron::arithmetic::uadd1<LimbT>(*r, h, c);
            *r = s2;
            assert(c2 == 0);
        }
    }

    return re;
}

template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<std::remove_cvref_t<AllocatorT>>::value_type>)
[[nodiscard]] std::tuple<std::remove_cv_t<LimbT>*, size_t, size_t, int> mul(composition<LimbT> const& l, composition<LimbT> const& r, AllocatorT&& alloc)
{
    using alloc_traits_t = std::allocator_traits<std::remove_cvref_t<AllocatorT>>;

    std::tuple<LimbT*, size_t, size_t, int> result;

    if (get<0>(l).empty() || get<0>(r).empty()) {
        get<0>(result) = nullptr;
        get<1>(result) = get<2>(result) = 0;
        get<3>(result) = 0;
        return result;
    }

    auto& [llimbs, lmask, lsign] = l;
    auto& [rlimbs, rmask, rsign] = r;

    get<3>(result) = !(lsign + rsign) ? -1 : 1;

    if (llimbs.size() == 1) [[unlikely]] { // short path
        if (rlimbs.size() == 1) [[unlikely]] {
            LimbT hl = llimbs.back() & lmask;
            LimbT hr = rlimbs.back() & rmask;
            auto [h, l] = arithmetic::umul1(hl, hr);
            if (h) {
                get<1>(result) = get<2>(result) = 2;
                get<0>(result) = alloc_traits_t::allocate(alloc, 2);
                *(get<0>(result) + 1) = h;
            } else {
                get<1>(result) = get<2>(result) = 1;
                get<0>(result) = alloc_traits_t::allocate(alloc, 1);
            }
            *get<0>(result) = l;
            return result;
        }
    }

    size_t margsz = llimbs.size() + rlimbs.size();
    get<1>(result) = get<2>(result) = margsz;
    get<0>(result) = alloc_traits_t::allocate(alloc, get<2>(result));
    
    //int cmpres = compare(llimbs, lmask, rlimbs, rmask, 1);
    intptr_t cmpres = (intptr_t)llimbs.size() - (intptr_t)rlimbs.size();
    LimbT const* lb = llimbs.data(), * le = lb + llimbs.size();
    LimbT const* rb = rlimbs.data(), * re = rb + rlimbs.size();
    LimbT* rese;
    if (lmask == std::numeric_limits<LimbT>::max() && rmask == std::numeric_limits<LimbT>::max()) {
        if (cmpres > 0) {
            rese = umul<LimbT>(lb, le, rb, re, get<0>(result));
        } else {
            rese = umul<LimbT>(rb, re, lb, le, get<0>(result));
        }
    } else {
        LimbT hl = llimbs.back() & lmask;
        LimbT hr = rlimbs.back() & rmask;
        auto lls = llimbs.first(llimbs.size() - 1);
        auto lrs = rlimbs.first(rlimbs.size() - 1);
        if (cmpres > 0) {
            rese = umul(hl, lls, hr, lrs, get<0>(result));
        } else {
            rese = umul(hr, lrs, hl, lls, get<0>(result));
        }
    }
    get<1>(result) = rese - get<0>(result);
    while (get<1>(result) && !*(get<0>(result) + get<1>(result) - 1)) {
        --get<1>(result);
    }
    return result;
}

// returns residual
template <std::unsigned_integral LimbT>
auto udivby1(std::span<LimbT> ls, LimbT d, LimbT invd, int l) -> LimbT
{
    using numetron::arithmetic::udiv2by1;

    int shift = std::numeric_limits<LimbT>::digits - l;

    LimbT ahigh = ls.back();
    LimbT* qp = &ls.back();

    LimbT r;

    if (!shift) {
        LimbT qhigh = (ahigh >= d);
        r = (qhigh ? ahigh - d : ahigh);

        *qp = qhigh;
        if (qp != ls.data()) {
            --qp;
            for (;;) {
                LimbT n0 = *qp;
                udiv2by1<LimbT>(*qp, r, r, n0, d, invd);
                if (qp == ls.data()) break;
                --qp;
            }
        }
    } else {
        r = 0;
        for (;;) {
            if (ahigh < d) {
                r = ahigh << shift;
                *qp = 0;
                if (qp == ls.data()) break;
                --qp;
            }
            
            LimbT n1 = *qp;
            r |= n1 >> l;
            LimbT dnorm = d << shift;
            while (qp != ls.data()) {
                --qp;
                LimbT n0 = *qp;
                udiv2by1<LimbT>(*(qp + 1), r, r, (n1 << shift) | (n0 >> l), dnorm, invd);
                n1 = n0;
            }
            udiv2by1<LimbT>(*qp, r, r, n1 << shift, dnorm, invd);
            break;
        }
    }
    return r >> shift;
}



// returns residual
template <std::unsigned_integral LimbT>
auto udivby1(std::span<const LimbT> ls, LimbT d, std::span<LimbT> r) -> LimbT
{
    assert(r.size() >= ls.size());
    std::copy(ls.begin(), ls.end(), r.begin());
    int shift = numetron::arithmetic::count_leading_zeros(d);
    int l = std::numeric_limits<LimbT>::digits - shift;
    LimbT u1 = (shift ? (LimbT{ 1 } << l) : 0) - d;
    auto [invd, dummy] = numetron::arithmetic::udiv2by1<LimbT>(u1, 0, d);

    return udivby1(r, d, invd, l);
}

template <std::unsigned_integral LimbT, LimbT d, bool ProcR = true>
auto udivby1(std::span<LimbT> ls) -> std::pair<LimbT, LimbT>
{
    using numetron::arithmetic::udiv2by1;
    using ct::W;

    constexpr uint32_t limb_bit_count = std::numeric_limits<LimbT>::digits;
    constexpr int l = 1 + numetron::arithmetic::consteval_log2<LimbT, limb_bit_count>(d);
    constexpr auto d_c = W<d>;
    using m_ct = decltype(upow(W<2>, W<limb_bit_count>) * (upow(W<2>, W<l>) - d_c) / d_c);
    constexpr auto invd = ct::front<m_ct>;
    constexpr uint32_t shift = limb_bit_count - l;

    LimbT ahigh = ls.back();
    LimbT* qp = &ls.back();

    LimbT r;
    LimbT dnorm = d << shift;

    if constexpr(!shift) {
        LimbT qhigh = (ahigh >= d);
        r = (qhigh ? ahigh - d : ahigh);

        *qp = qhigh;
        if (qp != ls.data()) {
            --qp;
            for (;;) {
                LimbT n0 = *qp;
                udiv2by1<LimbT>(*qp, r, r, n0, d, invd);
                if (qp == ls.data()) break;
                --qp;
            }
        }
    } else {
        r = 0;
        for (;;) {
            if (ahigh < d) {
                r = ahigh << shift;
                *qp = 0;
                if (qp == ls.data()) break;
                --qp;
            }

            LimbT n1 = *qp;
            r |= n1 >> l;
            
            while (qp != ls.data()) {
                --qp;
                LimbT n0 = *qp;
                udiv2by1<LimbT>(*(qp + 1), r, r, (n1 << shift) | (n0 >> l), dnorm, invd);
                n1 = n0;
            }
            udiv2by1<LimbT>(*qp, r, r, n1 << shift, dnorm, invd);
            break;
        }
    }

    if constexpr (ProcR) {
        LimbT frac;
        udiv2by1<LimbT>(frac, r, r, 0, dnorm, invd);
        return std::pair{ frac, r >> shift };
    } else {
        return r >> shift;
    }
}

template <std::unsigned_integral LimbT, typename QOutputIteratorT>
inline LimbT udivby1(LimbT& uh, std::span<LimbT> ul, LimbT d, QOutputIteratorT q)
{
    throw std::runtime_error("not implemented");
#if 0
    if (!uh) {
        ul.front() = udivby1((std::span<const LimbT>)ul, d, q);
        std::fill(ul.begin() + 1, ul.end(), 0);
        return uh;
    }
    throw std::runtime_error("not implemented");
#endif
}

// base case u / d
// prereqs: u < d * B^m, d normilized, where m = size(u) - size(d) in limbs, B = 2^bitsize(limb)
// returns {rhh, rh}; [rl] -> u
template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
std::pair<LimbT, LimbT> udiv_bc_unorm(LimbT* puhh, LimbT* puh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl, QOutputIteratorT qit, AllocatorT& alloc)
{
    assert(*puhh <= dh);

    size_t m = ul.size() - dl.size() + 1;
    
    if (m) {
        small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> daux{ dl.size(), alloc };

#if defined(NUMETRON_ARITHMETIC_USE_INVINT_DIV)
        auto [dinv, _] = numetron::arithmetic::udiv2by1<LimbT>(~dh + 1, 0, dh);
#endif
        do {
            LimbT dummy;
            LimbT qj;
            for (;;)
            {
                if (*puhh < dh) {
#if defined(NUMETRON_ARITHMETIC_USE_INVINT_DIV)
                    numetron::arithmetic::udiv2by1<LimbT>(qj, dummy, *puhh, *puh, dh, dinv);
#else
                    std::tie(qj, dummy) = numetron::arithmetic::udiv2by1norm<LimbT>(*puhh, *puh, dh);
#endif
                    if (!qj) break;
                } else {
                    qj = (std::numeric_limits<LimbT>::max)();
                }
            
                auto [mdhh, mdh] = umul1<LimbT>(dh, dl, qj, daux.data());
                
                // u - dh * B^j
                auto usp = ul.last(dl.size());
                LimbT uc = usub<LimbT>(usp, daux, usp);
                std::tie(uc, *puh) = numetron::arithmetic::usub1c(*puh, mdh, uc);
                std::tie(uc, *puhh) = numetron::arithmetic::usub1c(*puhh, mdhh, uc);

                if (uc) {
                    do {
                        --qj;
                        uc = uadd<LimbT>(usp, dl, usp);
                        std::tie(uc, *puh) = numetron::arithmetic::uadd1(*puh, dh, uc);
                        std::tie(uc, *puhh) = numetron::arithmetic::uadd1(*puhh, uc);
                    } while (!uc);
                }
                break;
            }
            assert(!*puhh);
            puhh = puh;
            puh = &ul.back();
            ul = ul.first(ul.size() - 1);
            *qit = qj; --qit;
        } while (--m);
    }
    return { *puhh, *puh };
}

template <std::unsigned_integral LimbT>
LimbT do_udiv_unorm(LimbT* puhh, LimbT* puh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl)
{
    LimbT uc, tmphh, tmph;
    std::tie(uc, tmphh) = numetron::arithmetic::usub1(*puhh, dh);
    if (!uc) { // dh <= *puhh
        if (dl.empty()) {
            *puhh = tmphh;
        } else {
            //if (dl.size() == 1) {
            //    std::tie(uc, tmph) = numetron::arithmetic::usub1(*puh, dl.front());
            //    std::tie(uc, tmphh) = numetron::arithmetic::usub1(tmphh, uc);
            //    if (!uc) {
            //        *puhh = tmphh;
            //        *puh = tmph;
            //    }
            //}

            auto dl1 = dl.first(dl.size() - 1);
            auto ul1 = ul.last(dl1.size());

            uc = usub<LimbT>(ul1, dl1, ul1);
            std::tie(uc, tmph) = numetron::arithmetic::usub1c(*puh, dl.back(), uc);
            std::tie(uc, tmphh) = numetron::arithmetic::usub1(tmphh, uc);
            if (!uc) {
                *puhh = tmphh;
                *puh = tmph;
            } else {
                uadd<LimbT>(ul1, dl1, ul1);
            }
        }
    }
    return 1 - uc;
}

// base case u / d
// prereqs: u < d * B^m, d normilized, where m = size(u) - size(d) in limbs, B = 2^bitsize(limb)
// returns {rhh, rh}; [rl] -> u
template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
std::pair<LimbT, LimbT> udiv_bc(LimbT* puhh, LimbT* puh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl, QOutputIteratorT qit, AllocatorT& alloc)
{
    *qit-- = do_udiv_unorm(puhh, puh, ul, dh, dl);
    return udiv_bc_unorm<LimbT>(puhh, puh, ul, dh, dl, std::move(qit), alloc);
}

//
template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
std::pair<LimbT, LimbT> udiv_svoboda(LimbT* puhh, LimbT* puh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl, QOutputIteratorT qit, AllocatorT& alloc)
{
    //using allocator_type = std::remove_cvref_t<AllocatorT>;
    //using alloc_traits_t = std::allocator_traits<allocator_type>;

    *qit-- = do_udiv_unorm(puhh, puh, ul, dh, dl);

    size_t m = ul.size() + 1 - dl.size();
    if (!m) return { *puhh, *puh };

    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> auxbuff(dl.size() + 1 /*dh*/ + 2, alloc);
    std::fill(auxbuff.begin(), auxbuff.end() - 1, 0);
    auxbuff.back() = 1;
    
    LimbT k[3];
    auto tmpsp = auxbuff.span().first(dl.size() + 1);
    udiv_bc(&auxbuff.back(), &auxbuff.back() - 1, tmpsp, dh, dl, k + 2, alloc);
    // ceiling: if reminder != 0 => k = k+1
    for (LimbT const* pr = tmpsp.data() + dl.size();;) {
        if (*pr) {
            LimbT uc;
            std::tie(uc, k[0]) = numetron::arithmetic::uadd1<LimbT>(k[0], 1);
            if (uc) ++k[1];
            break;
        }
        if (pr == tmpsp.data()) break;
        --pr;
    }

    //d1 = k*d, reuse auxbuff for d1
    auto d1 = auxbuff.span();
    
    //(dh*B^(dl.size()) + dl) * k = dh * k + dl * k
    LimbT kdh[3];
    umul<LimbT>(dl, { k, 2 }, d1); // dl * k
    kdh[2] = umul1<LimbT>({ k, 2 }, dh, kdh);
    uadd<LimbT>({ d1.data() + dl.size(), d1.size() - dl.size() }, { kdh, 3 });
    assert(d1[dl.size() + 1] == 0);
    assert(d1[dl.size() + 2] == 1);
    auto d1l = std::span{ d1.data(), d1.size() - 2 };
    //LimbT d1h = d1.back();

    size_t m1 = m - 1;
    
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> q1{ m1, alloc };
    if (m1) {
        LimbT* pq1 = &q1.back();
        auto q1sp = q1.span();

        // daux stores qj * d1
        small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> daux{ d1.size(), alloc };
        LimbT& dauxhh = daux.back();
        LimbT& dauxh = *(&dauxhh - 1);
        auto dauxl = daux.first(d1.size() - 2);

        for (;;) {
            // q(j) = a(n+j)
            LimbT qj = *puhh;
        
            // A <- A - qj*d1*B^(j-1)
            LimbT uc = umul1<LimbT>(d1, qj, dauxl.data());
            assert(!uc);
        
            auto usp = ul.last(dauxl.size());
        
            uc = usub<LimbT>(usp, dauxl, usp);
            std::tie(uc, *puh) = numetron::arithmetic::usub1c(*puh, dauxh, uc);
            std::tie(uc, *puhh) = numetron::arithmetic::usub1c(*puhh, dauxhh, uc);
#if 0 // my version
            if (*puhh) { // <=> if uc != 0 then gj <- gj - 1, A <- A + d1*B^(j-1)
                --qj;
                uc = uadd<LimbT>(usp, d1l, usp);
                std::tie(uc, *puh) = numetron::arithmetic::uadd1<LimbT>(*puh, uc); // d1h = 0
                //std::tie(uc, *puhh) = numetron::arithmetic::uadd1c<LimbT>(*puhh, 1, uc); // d1hh = 1
                //assert(uc);
                *puhh = 0;
            }
#else // gpt fix
            if (uc) {
                do {
                    --qj;
                    uc = uadd<LimbT>(usp, d1l, usp);
                    std::tie(uc, *puh) = numetron::arithmetic::uadd1<LimbT>(*puh, uc); // d1h = 0
                    //std::tie(uc, *puhh) = numetron::arithmetic::uadd1c<LimbT>(*puhh, 1, uc); // d1hh = 1
                    //assert(uc);
                    *puhh = 0;
                } while (!uc);
            }
#endif
            puhh = puh;
            puh = &ul.back();
            ul = ul.first(ul.size() - 1);
            *pq1 = qj;
            if (pq1 == q1.data()) break;
            --pq1;
        }
    }
    // q0 = r1 div d
    LimbT q0arr[2];
    auto qr = udiv_bc<LimbT>(puhh, puh, ul, dh, dl, q0arr + 1, alloc);
    
    // q = k * q1, we need aux buffer for q to reorder q limbs
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> q(m1 + 2, alloc);
    LimbT* qb = q.data();
    LimbT *qe = umul<LimbT>(q1, { k, 2 }, qb) - 1;
    auto [q0, qbval] = numetron::arithmetic::uadd1(*qb, q0arr[0]);
    *qb = qbval; q0 += q0arr[1];
    if (q0) {
        LimbT* qit = qb;
        do {
            ++qit;
            std::tie(q0, *qit) = numetron::arithmetic::uadd1(*qit, q0);
        } while (q0);
    }

    while (qe != qb) {
        *qit-- = *--qe;
    }
    return qr;
}


// divide and conquer
// prereqs: u >= d, d normilized
template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
inline void udiv_dv(LimbT* puhh, LimbT* puh, std::span<LimbT>& ul, std::span<LimbT> d, QOutputIteratorT qit, AllocatorT& alloc)
{
    // B (base) = 2^std::numetic_limits<LimbT>::digits;

    using allocator_type = std::remove_cvref_t<AllocatorT>;
    using alloc_traits_t = std::allocator_traits<allocator_type>;

    size_t m = ul.size() - d.size() + 2;
    if (m < NUMETRON_DC_DIV_QR_THRESHOLD || d.size() < 2)
        return udiv_bc(puh, puhh, ul, d, std::move(qit));

    size_t k = m / 2;
    size_t thr = 2 * k;;
    assert(ul.size() >= thr);
    auto ul1 = ul.subspan(thr);
    auto d1 = d.subspan(k);
    auto d0 = d.first(k);

    size_t q1sz = 2 + ul1.size() - d1.size();
    LimbT* q1 = alloc_traits_t::allocate(alloc, q1sz);
    SCOPE_EXIT([&alloc, q1, q1sz] { alloc_traits_t::deallocate(alloc, q1, q1sz); });

    LimbT r1h = udiv_dv<LimbT>(puhh, puh, ul1, d1, q1 + q1sz - 1, alloc);
    size_t realq1sz = q1sz;
    if (!q1[q1sz - 1]) --realq1sz;
    // here u1 = r1
    ul1 = ul.first(thr + ul1.size());
    // now u1 = r1*B^2k + (u mod B^2k)
    
    // u1 - q1 * d0 * B^k
    size_t q1d0sz = realq1sz * d0.size();
    LimbT* q1d0 = alloc_traits_t::allocate(alloc, q1d0sz);
    SCOPE_EXIT([&alloc, q1d0, q1d0sz] { alloc_traits_t::deallocate(alloc, q1d0, q1d0sz); });
    umul<LimbT>({ q1, realq1sz }, d0, { q1d0, q1d0sz });
    
    auto ul2 = ul1.subspan(k); // ul2 = u1 div B^k
    LimbT c = usub<LimbT>(r1h, ul2, { q1d0, q1d0sz });
}

template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
LimbT udiv(LimbT uh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl, QOutputIteratorT qit, AllocatorT && alloc)
{
    //using allocator_type = std::remove_cvref_t<AllocatorT>;
    //using alloc_traits_t = std::allocator_traits<allocator_type>;

    LimbT* puh;
    if (!uh) {
        puh = &ul.back();
        ul = ul.first(ul.size() - 1);
    } else {
        puh = &uh;
    }

    // normalization
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> optdnorm(0, alloc);
    const LimbT* dlnorm;
    LimbT uhhstore = 0;
    int shift = numetron::arithmetic::count_leading_zeros(dh);
    if (shift) {
        optdnorm.reset(dl.size());
        ushift_left<LimbT>(dh, dl, shift, optdnorm.data()); // returns 0
        uhhstore = ushift_left<LimbT>(*puh, ul, shift, ul.data());
        dlnorm = optdnorm.data();
    } else {
        dlnorm = dl.data();
    }
    
    LimbT* puhh = &uhhstore;
    if (!uhhstore) {
        puhh = puh;
        puh = &ul.back();
        ul = ul.first(ul.size() - 1);
    }
    assert(*puhh);
    //auto [rhh, rh] = udiv_dv(puhh, puh, ul, dh, { dlnorm, dl.size() }, std::move(qit), alloc);
    auto [rhh, rh] = udiv_svoboda(puhh, puh, ul, dh, { dlnorm, dl.size() }, std::move(qit), alloc);
    //auto [rhh, rh] = udiv_bc(puhh, puh, ul, dh, { dlnorm, dl.size() }, std::move(qit), alloc);
    if (shift) {
        ushift_right<LimbT>(rh, ul, shift); // returns 0
        if (rhh) {
            rh |= (rhh << (std::numeric_limits<LimbT>::digits - shift));
        }
    } else if (rhh) {
        ul = { ul.data(), ul.size() + 1 };
        assert(ul.back() == rh);
        rh = rhh;
    }
    
    return rh;
}



// prereqs: u >= d, d.back() > 0
// {uh, ul} / d -> q(from high to low); rl -> ul, returns rh
// uh can be 0, daux.size() >= d.size()
template <std::unsigned_integral LimbT, typename QOutputIteratorT>
LimbT udiv2(LimbT& uh, std::span<LimbT> & ul, std::span<LimbT> d, std::span<LimbT> daux, QOutputIteratorT qit)
{
    assert(d.back());
    assert(daux.size() >= d.size());

    if (d.size() == 1) {
        return udivby1(uh, ul, d.back(), qit);
    }

    LimbT* puh;
    if (!uh) {
        puh = &ul.back();
        ul = ul.first(ul.size() - 1);
    } else {
        puh = &uh;
    }

    // normalization
    LimbT uhhstore = 0;
    int shift = numetron::arithmetic::count_leading_zeros(d.back());
    if (shift) {
        uhhstore = ushift_left<LimbT>(*puh, ul, shift, ul.data());
        ushift_left<LimbT>(d, shift); // returns 0
    }
    LimbT* puhh = &uhhstore;
    if (!uhhstore) {
        puhh = puh;
        puh = &ul.back();
        ul = ul.first(ul.size() - 1);
    }
    assert(*puhh);

    // ul.size() + 2 = n + m, n = d.size(); => m = ul.size() + 2 - d.size() = ul.size() - d.size() + 2
    // compare U and d * B^m
    size_t m = ul.size() - d.size() + 2;
    
    LimbT const* db = d.data(), * de = db + d.size() - 1;
    LimbT const* ub = ul.data(), * ue = ub + ul.size() - 1;
    bool u_gt_d = *puhh > *de;
    if (!u_gt_d && *puhh == *de) {
        --de;
        u_gt_d = *puh > *de;
        if (!u_gt_d && *puh == *de) {
            u_gt_d = true;
            while (db != de) {
                --de;
                if (*ue != *de) { u_gt_d = *ue > *de; break; }
                --ue;
            }
        }
    }

    if (u_gt_d) {
        //  u <- u - B^m, q(m) <- 1
        auto dsp = d.first(d.size() - 2);
        auto usp = ul.last(dsp.size());
        LimbT uc;
        if (!dsp.empty()) {
            uc = usub<LimbT>(usp, dsp, usp);
            std::tie(uc, *puh) = numetron::arithmetic::usub1c(*puh, d[d.size() - 2], uc);
        } else {
            std::tie(uc, *puh) = numetron::arithmetic::usub1(*puh, d[d.size() - 2]);
        }
        std::tie(uc, *puhh) = numetron::arithmetic::usub1c(*puhh, d.back(), uc);
        *qit = 1;
    } else {
        *qit = 0;
    }
    --qit;

    if (m) {
        auto dsp = d.first(d.size() - 1);
        auto dauxsp = daux.first(daux.size() - 1);

#if defined(NUMETRON_ARITHMETIC_USE_INVINT_DIV)
        auto [dinv, _] = numetron::arithmetic::udiv2by1<LimbT>(~d.back() + 1, 0, d.back());
#endif
        do {
            LimbT dummy;
            LimbT qj;
            for (;;)
            {
                if (*puhh < d.back()) {
#if defined(NUMETRON_ARITHMETIC_USE_INVINT_DIV)
                    numetron::arithmetic::udiv2by1<LimbT>(qj, dummy, *puhh, *puh, d.back(), dinv);
#else
                    std::tie(qj, dummy) = numetron::arithmetic::udiv2by1norm<LimbT>(*puhh, *puh, d.back());
#endif
                    if (!qj) break;
                } else {
                    qj = (std::numeric_limits<LimbT>::max)();
                }
            
                LimbT dh = umul1<LimbT>(d, qj, daux.data());
            
                // u - dh * B^j
                auto usp = ul.last(dauxsp.size());
                LimbT uc = usub<LimbT>(usp, dauxsp, usp);
                std::tie(uc, *puh) = numetron::arithmetic::usub1c(*puh, daux.back(), uc);
                std::tie(uc, *puhh) = numetron::arithmetic::usub1c(*puhh, dh, uc);

                if (uc) {
                    do {
                        --qj;
                        uc = uadd<LimbT>(usp, dsp, usp);
                        std::tie(uc, *puh) = numetron::arithmetic::uadd1(*puh, d.back(), uc);
                        std::tie(uc, *puhh) = numetron::arithmetic::uadd1(*puhh, uc);
                    } while (!uc);
                }
                break;
            }
            assert(!*puhh);
            puhh = puh;
            puh = &ul.back();
            ul = ul.first(ul.size() - 1);
            *qit = qj; --qit;
        } while (--m);
    }
    if (*puhh) {
        puh = puhh;
        ul = { ul.data(), ul.size() + 1 };
    }
    if (shift) {
        ushift_right<LimbT>(*puh, ul, shift, ul.data());
    }
    return *puh;
}

template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
LimbT udiv2(LimbT& uh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl, QOutputIteratorT qit, AllocatorT&& alloc)
{
    using allocator_type = std::remove_cvref_t<AllocatorT>;
    using alloc_traits_t = std::allocator_traits<allocator_type>;

    std::span<LimbT> d{ alloc_traits_t::allocate(alloc, dl.size() + 1), dl.size() + 1 };
    std::span<LimbT> daux{ alloc_traits_t::allocate(alloc, dl.size() + 1), dl.size() + 1 };
    std::copy(dl.begin(), dl.end(), d.data()); d.back() = dh;
    SCOPE_EXIT([&alloc, d, daux] {
        alloc_traits_t::deallocate(alloc, d.data(), d.size());
        alloc_traits_t::deallocate(alloc, daux.data(), daux.size());
        });

    return udiv2<LimbT>(uh, ul, d, daux, std::move(qit));
}

template <std::unsigned_integral LimbT>
inline void udiv(std::span<const LimbT> u, std::span<const LimbT> v, std::span<LimbT> q, std::span<LimbT> r)
{
    LimbT const* vb = v.data(), * ve = vb + v.size();
    for (;; --ve) {
        if (vb == ve) [[unlikely]] {
            throw std::runtime_error("division by zero");
        }
        if (*(ve - 1)) break;
    }
    if (1 == ve - vb) {
        r.front() = udivby1(u, *vb, q);
        std::fill(r.begin() + 1, r.end(), 0);
        return;
    }
    throw std::runtime_error("not implemented");
}

template <std::unsigned_integral LimbT>
inline void ushift_left(std::span<const LimbT> u, size_t shift, std::span<LimbT> r)
{
    assert(r.size() > u.size());
    if (u.empty()) [[unlikely]] {
        return;
    }

    LimbT const* ub = u.data(), * ue = ub + u.size();
    r.front() = (*ub) << shift;
    
    LimbT * rb = r.data(), * re = rb + u.size();
    for (LimbT const* nub = ub + 1;;++ub, ++nub) {
        ++rb;
        if (nub == ue) [[unlikely]] {
            *rb = *ub >> (sizeof(LimbT) * CHAR_BIT - shift);
            return;
        }
        *rb = (*nub << shift) | (*ub >> (sizeof(LimbT) * CHAR_BIT - shift));
    }
}



template <std::unsigned_integral LimbT>
auto sqrt_rem(std::span<const LimbT> m, std::span<LimbT> s) noexcept
{
    size_t l = (m.size() - 1) / 4;
    if (!l) {
        s[0] = numetron::arithmetic::sqrt<LimbT>(m[0]);
    }
}

}
