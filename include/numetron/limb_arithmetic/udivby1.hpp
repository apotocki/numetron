// Numetron — Compile - time and runtime arbitrary - precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <concepts>
#include <limits>
#include <span>

#include "numetron/arithmetic.hpp"
#include "numetron/ct.hpp"

namespace numetron::limb_arithmetic {

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

#if 0 // not tested
template <std::unsigned_integral LimbT>
auto udivby1(LimbT uh, std::span<LimbT> ul, LimbT d, LimbT invd, int l) -> LimbT
{
    using numetron::arithmetic::udiv2by1;

    int shift = std::numeric_limits<LimbT>::digits - l;

    LimbT r;

    if (!shift) {
        // d is already normalized (high bit set)
        // Process the high limb uh first
        LimbT qhigh = (uh >= d);
        r = (qhigh ? uh - d : uh);

        if (ul.empty()) {
            return r;
        }

        LimbT* qp = &ul.back();
        for (;;) {
            LimbT n0 = *qp;
            udiv2by1<LimbT>(*qp, r, r, n0, d, invd);
            if (qp == ul.data()) break;
            --qp;
        }
    } else {
        // d needs normalization
        LimbT dnorm = d << shift;

        if (ul.empty()) {
            // Only uh to process
            if (uh < d) {
                return uh;
            }
            // uh >= d, but we have no place to store the quotient
            // This case means uh is the only limb and result goes nowhere
            // Just compute remainder
            r = uh << shift;
            LimbT q_unused;
            udiv2by1<LimbT>(q_unused, r, r, 0, dnorm, invd);
            return r >> shift;
        }

        LimbT* qp = &ul.back();
        LimbT ahigh = uh;

        r = 0;
        if (ahigh < d) {
            r = ahigh << shift;
            // Continue with ul limbs
            LimbT n1 = *qp;
            r |= n1 >> l;

            while (qp != ul.data()) {
                --qp;
                LimbT n0 = *qp;
                udiv2by1<LimbT>(*(qp + 1), r, r, (n1 << shift) | (n0 >> l), dnorm, invd);
                n1 = n0;
            }
            udiv2by1<LimbT>(*qp, r, r, n1 << shift, dnorm, invd);
        } else {
            // ahigh >= d
            // First, divide {ahigh_shifted, ul.back() high bits} 
            LimbT n1 = *qp;
            r = (ahigh << shift) | (n1 >> l);

            while (qp != ul.data()) {
                --qp;
                LimbT n0 = *qp;
                udiv2by1<LimbT>(*(qp + 1), r, r, (n1 << shift) | (n0 >> l), dnorm, invd);
                n1 = n0;
            }
            udiv2by1<LimbT>(*qp, r, r, n1 << shift, dnorm, invd);
        }
    }
    return r >> shift;
}
#endif

// returns remainder
template <std::unsigned_integral LimbT>
auto udivby1(std::span<const LimbT> ls, LimbT d, std::span<LimbT> q) -> LimbT
{
    assert(d);
    assert(q.size() >= ls.size());
    assert(!ls.empty());

    // Special case: division by 1
    if (d == 1) {
        std::memcpy(q.data(), ls.data(), ls.size() * sizeof(LimbT));
        return 0;
    }
    
    constexpr int limb_bits = std::numeric_limits<LimbT>::digits;
    int zcnt = numetron::arithmetic::count_leading_zeros(d);
    
     // Special case: d is a power of two
    if ((d & (d - 1)) == 0) {
        LimbT remainder_mask = d - 1;
        int shift = limb_bits - 1 - zcnt;

        // Remainder is the lowest 'shift' bits of the number
        LimbT remainder = ls.front() & remainder_mask;

        // Quotient is the number shifted right by 'shift' bits
        int lshift = limb_bits - shift;
        
        LimbT const* src = ls.data();
        LimbT* dst = q.data();
        size_t n = ls.size();

        for (size_t i = 0; i < n - 1; ++i) {
            dst[i] = (src[i] >> shift) | (src[i + 1] << lshift);
        }
        dst[n - 1] = src[n - 1] >> shift;
        
        return remainder;
    }
    
    std::memcpy(q.data(), ls.data(), ls.size() * sizeof(LimbT));
    
    int l = limb_bits - zcnt;
    LimbT u1 = (zcnt ? (LimbT{ 1 } << l) : 0) - d;
    auto [invd, dummy] = numetron::arithmetic::udiv2by1<LimbT>(u1, 0, d);

    return udivby1(q, d, invd, l);
}

// returns remainder
template <std::unsigned_integral LimbT>
auto udivby1(LimbT uh, std::span<const LimbT> ul, LimbT d, std::span<LimbT> q) -> LimbT
{
    assert(d);
    assert(q.size() >= 1 + ul.size());

    if (ul.empty()) {
        // only uh to process
        auto [qval, r] = numetron::arithmetic::div1(uh, d);
        q[0] = qval;
        return r;
    }

    // Special case: division by 1
    if (d == 1) {
        std::memcpy(q.data(), ul.data(), ul.size() * sizeof(LimbT));
        q[ul.size()] = uh;
        return 0;
    }

    q = q.first(ul.size() + 1);

    constexpr int limb_bits = std::numeric_limits<LimbT>::digits;
    int zcnt = numetron::arithmetic::count_leading_zeros(d);

    // Special case: d is a power of two
    if ((d & (d - 1)) == 0) {
        LimbT remainder_mask = d - 1;
        int shift = limb_bits - 1 - zcnt;

        // Remainder is the lowest 'shift' bits of the number
        LimbT remainder = ul.front() & remainder_mask;

        // Quotient is the number shifted right by 'shift' bits
        int lshift = limb_bits - shift;

        LimbT const* src = ul.data(), * src_end = ul.data() + ul.size();
        LimbT const* next_src = src + 1;
        LimbT* dst = q.data();
        
        while (next_src != src_end) {
            *dst++ = (*src++ >> shift) | (*next_src++ << lshift);
        }
        *dst++ = (*src >> shift) | (uh << lshift);
        *dst = uh >> shift;

        return remainder;
    }

    std::memcpy(q.data(), ul.data(), ul.size() * sizeof(LimbT));
    q[ul.size()] = uh;
    int l = limb_bits - zcnt;
    LimbT u1 = (zcnt ? (LimbT{ 1 } << l) : 0) - d;
    auto [invd, dummy] = numetron::arithmetic::udiv2by1<LimbT>(u1, 0, d);

    return udivby1(q, d, invd, l);
}

template <std::unsigned_integral LimbT, LimbT d, bool ProcR = true>
auto udivby1(std::span<LimbT> ls) // -> std::pair<LimbT, LimbT>
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

}
