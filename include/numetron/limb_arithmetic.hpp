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

// (rh, r[u.size()]) <- [u] * v; returns rh
template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
inline LimbT umul1(UIteratorT ub, UIteratorT ue, LimbT v, RIteratorT&& rraw) noexcept
{
    std::conditional_t<std::is_reference_v<RIteratorT>, RIteratorT, std::decay_t<RIteratorT>> r = std::forward<RIteratorT>(rraw);
    if (ub == ue) {
        assert(ub != ue);
    }
    assert(ub != ue);
    // unroll first iteration
    auto [h, l] = arithmetic::umul1(*ub, v);
    *r = l; ++r;
    
    if (++ub == ue) return h;
    
    // unroll second iteration
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
    auto [h0, l0] = numetron::arithmetic::umul1(*ub, v);
    auto [cl2, current] = numetron::arithmetic::uadd1(l0, *r);
    *r = current; ++r;
    while (++ub != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
        *r = numetron::arithmetic::uadd1c2(l, *r, h0, cl, cl2);
        ++r;
        h0 = h;
    }
    return h0 + cl + cl2;
#endif
}

// (uh, [ul]) * v + cl -> (rh, r[ul.size() + 1]); returns rh
//template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
//inline LimbT umul1(LimbT uh, UIteratorT ulb, UIteratorT ule, LimbT v, RIteratorT&& rraw) noexcept
//{
//    std::conditional_t<std::is_reference_v<RIteratorT>, RIteratorT, std::decay_t<RIteratorT>> r = std::forward<RIteratorT>(rraw);
//    LimbT pcl = umul1<LimbT>(ulb, ule, v, r);
//    auto [h, ph] = numetron::arithmetic::umul1(uh, v);
//    auto [c, res] = numetron::arithmetic::uadd1(ph, pcl);
//    *r = res; ++r;
//    return h + c;
//}

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

// base case mul: {u} * {v} -> {rb, re}
// returns re
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, ResultIteratorT r) noexcept
{
    assert(ub != ue && vb != ve);

    ResultIteratorT rb = r;

    // first line
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

    // next lines
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
}

// Perhaps one day, a C++ compiler will be able to optimise this to the same extent as hand-written assembly code.
template <std::unsigned_integral LimbT>
inline LimbT* umul_basecase_unrolled(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT* rb) noexcept
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

template <std::unsigned_integral LimbT>
requires (sizeof(LimbT) == 8)
inline LimbT* umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT* rb) noexcept
{
    if constexpr (sizeof(LimbT) == 8) {
#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
#   if defined(NUMETRON_PLATFORM_AUTODETECT)
        std::call_once(mul_basecase_init_flag, []() {
            uint64_t platform_descriptor = numetron_detect_platform();
            //std::cout << "PLATFROM: " << std::hex << "0x" << platform_descriptor << std::dec << std::endl;
            __mul_basecase_ptr = detect_mul_basecase(platform_descriptor);
        });
        if (__mul_basecase_ptr) __mul_basecase_ptr(rb, ub, (ue - ub), vb, (ve - vb));
        return rb + (ue - ub) + (ve - vb);
#   else
        NUMETRON_mul_basecase(rb, ub, (ue - ub), vb, (ve - vb));
        return rb + (ue - ub) + (ve - vb);
#   endif
#endif
    }
    return umul_basecase_unrolled<LimbT>(ub, ue, vb, ve, rb);
    //return umul<LimbT, LimbT*>(ub, ue, vb, ve, rb);
}


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
    if (lmask == (std::numeric_limits<LimbT>::max)() && rmask == (std::numeric_limits<LimbT>::max)()) {
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
