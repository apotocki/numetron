// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>

#include "arithmetic.hpp"
#include "ct.hpp"

#include "detail/small_array.hpp"

#define NUMETRON_USE_ASM
#define NUMETRON_PLATFORM_AUTODETECT

#include "limb_arithmetic/usub.hpp"
#include "limb_arithmetic/umul_karatsuba.hpp"
#include "limb_arithmetic/toom_plan.hpp"
#include "limb_arithmetic/toom_engine.hpp"
#include "limb_arithmetic/umul.hpp"

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
#   if defined(NUMETRON_PLATFORM_AUTODETECT)
extern "C" uint64_t numetron_detect_platform();
#   endif
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

template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
[[nodiscard]] std::tuple<std::remove_cv_t<LimbT>*, size_t, size_t, int> mul(composition<LimbT> const& l, composition<LimbT> const& r, AllocatorT alloc)
{
    using alloc_traits_t = std::allocator_traits<AllocatorT>;

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

    //size_t margsz = llimbs.size() + rlimbs.size();
    //get<1>(result) = get<2>(result) = margsz;
    //get<0>(result) = alloc_traits_t::allocate(alloc, get<2>(result));
    
    //int cmpres = compare(llimbs, lmask, rlimbs, rmask, 1);
    intptr_t cmpres = (intptr_t)llimbs.size() - (intptr_t)rlimbs.size();
    std::tuple<LimbT*, size_t, size_t> rese;
    if (lmask == (std::numeric_limits<LimbT>::max)() && rmask == (std::numeric_limits<LimbT>::max)()) {
        if (cmpres > 0) {
            rese = umul<LimbT>(llimbs, rlimbs, std::move(alloc));
        } else {
            rese = umul<LimbT>(rlimbs, llimbs, std::move(alloc));
        }
    } else {
        LimbT hl = llimbs.back() & lmask;
        LimbT hr = rlimbs.back() & rmask;
        auto lls = llimbs.first(llimbs.size() - 1);
        auto lrs = rlimbs.first(rlimbs.size() - 1);
        if (cmpres > 0) {
            rese = umul(hl, lls, hr, lrs, std::move(alloc));
        } else {
            rese = umul(hr, lrs, hl, lls, std::move(alloc));
        }
    }
    get<0>(result) = get<0>(rese);
    get<1>(result) = get<1>(rese);
    get<2>(result) = get<2>(rese);
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
