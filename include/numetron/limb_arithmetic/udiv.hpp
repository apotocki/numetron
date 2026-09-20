// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "udivby1.hpp"

namespace numetron::limb_arithmetic {

// base case u / d
// prereqs: u < d * B^m, d normilized, where m = size(u) - size(d) in limbs, B = 2^bitsize(limb)
// returns {rhh, rh}; [rl] -> u
template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
std::pair<LimbT, LimbT> udiv_bc_unorm(LimbT* puhh, LimbT* puh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl, QOutputIteratorT qit, AllocatorT& alloc)
{
    assert(*puhh <= dh);

    size_t m = ul.size() - dl.size() + 1;
    
    if (m) {
        // umul1 writes dl.size() + 1 limbs: the low part plus the high limb of qj * dl
        small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> daux{ dl.size() + 1, alloc };

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
            
                LimbT * tmpr = daux.data();
                LimbT mdhh = umul1<LimbT>(dh, dl.data(), dl.data() + dl.size(), qj, tmpr);
                LimbT mdh = *(tmpr - 1);
                // u - dh * B^j
                auto usp = ul.last(dl.size());
                LimbT uc = usub<LimbT>(usp, daux.first(dl.size()), usp);
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
LimbT do_udiv_unorm(LimbT* puhh, LimbT* puh, std::span<LimbT> const ul, LimbT dh, std::span<const LimbT> dl)
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

// Svoboda division: same contract as udiv_bc_unorm(), but scales the divisor up front so that the
// per-digit 2/1 division disappears -- the top limb of the partial remainder *is* the digit
// estimate. k = ceil(B^(n+1) / d) gives d1 = k * d with B^(n+1) <= d1 < B^(n+1) + B^n, u / d1 is
// computed digit by digit, and the result is unscaled at the end via q = k * q1 + (r1 / d).
// prereqs: u < d * B^m, d normalized, where m = size(u) - size(d) + 1 in limbs
// returns {rhh, rh}; [rl] -> u
template <std::unsigned_integral LimbT, typename QOutputIteratorT, typename AllocatorT>
std::pair<LimbT, LimbT> udiv_svoboda_unorm(LimbT* puhh, LimbT* puh, std::span<LimbT>& ul, LimbT dh, std::span<const LimbT> dl, QOutputIteratorT qit, AllocatorT& alloc)
{
    assert(*puhh <= dh);

    size_t const nl = dl.size();
    size_t const n = nl + 1;             // divisor size in limbs
    size_t const m = ul.size() + 1 - nl; // quotient digits to produce

    // scaling the divisor costs O(n); with a single digit there is nothing to amortize it over
    if (m < 2) return udiv_bc_unorm<LimbT>(puhh, puh, ul, dh, dl, std::move(qit), alloc);

    // aux holds d1 (n + 2 limbs, initially B^(n+1)) followed by a contiguous copy of d
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> aux(2 * n + 2, alloc);
    auto d1 = aux.first(n + 2);
    auto dfull = aux.last(n);
    std::copy(dl.begin(), dl.end(), dfull.data());
    dfull[nl] = dh;

    // k = ceil(B^(n+1) / d); d normalized means B^n / 2 <= d < B^n, so B < k <= 2 * B
    LimbT k[3];
    std::fill(d1.begin(), d1.end() - 1, LimbT{ 0 });
    d1.back() = 1;
    {
        auto tmpsp = d1.first(n);
        udiv_bc<LimbT>(&d1.back(), d1.data() + n, tmpsp, dh, dl, k + 2, alloc);
        assert(!k[2]);
        // round up: a non-zero remainder, left in d1[0 .. n), means k = floor(...) + 1
        for (LimbT const* pr = d1.data() + n; pr-- != d1.data();) {
            if (*pr) {
                LimbT c;
                std::tie(c, k[0]) = numetron::arithmetic::uadd1<LimbT>(k[0], 1);
                k[1] += c;
                break;
            }
        }
    }

    // d1 = d * k, reusing the buffer that held B^(n+1)
    {
        LimbT* p = d1.data();
        LimbT c0 = umul1<LimbT>(dfull.data(), dfull.data() + n, k[0], p); // p advances by n
        *p = c0;
        LimbT* p1 = d1.data() + 1;
        d1[n + 1] = umul1_add<LimbT>(dfull.data(), dfull.data() + n, k[1], p1);
    }
    assert(!d1[n] && d1[n + 1] == 1); // B^(n+1) <= d1 < B^(n+1) + B^n
    auto d1l = d1.first(n);           // d1 - B^(n+1)

    size_t const m1 = m - 1;          // digits of q1 = u / d1
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> q1(m1, alloc);
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> daux(n + 2, alloc);
    auto dauxsp = daux.span();

    // A < d1 * B^j is the loop invariant, so A occupies n + 2 limbs plus one extra bit, kept in
    // atop: the limb above *puhh, which the frame shift below would otherwise drop
    LimbT atop = 0;
    for (LimbT* pq1 = &q1.back();;) {
        LimbT qj = atop ? (std::numeric_limits<LimbT>::max)() : *puhh;

        // A -= qj * d1 * B^(j-1)
        LimbT* tmpr = dauxsp.data();
        LimbT c = umul1<LimbT>(d1.data(), d1.data() + d1.size(), qj, tmpr);
        assert(!c); // qj * d1 < B^(n+2)
        (void)c;
        
        auto usp = ul.last(n);
        LimbT uc = usub<LimbT>(usp, dauxsp.first(n), usp);
        std::tie(uc, *puh) = numetron::arithmetic::usub1c(*puh, dauxsp[n], uc);
        std::tie(uc, *puhh) = numetron::arithmetic::usub1c(*puhh, dauxsp[n + 1], uc);

        if (uc > atop) { // went negative: the estimate was exactly one too large
            --qj;
            uc = uadd<LimbT>(usp, d1l, usp);
            std::tie(uc, *puh) = numetron::arithmetic::uadd1<LimbT>(*puh, uc);           // d1[n] == 0
            std::tie(uc, *puhh) = numetron::arithmetic::uadd1<LimbT>(*puhh, LimbT{ 1 }, uc); // d1[n+1] == 1
            assert(uc); // cancels the borrow
        } else {
            assert(uc == atop); // atop == 1 forces a borrow, the result being below d1
        }

        // the remainder is below d1 < B^(n+1) + B^n, so the limb above the shifted frame is 0 or 1
        atop = *puhh;
        assert(atop <= 1);

        puhh = puh;
        puh = &ul.back();
        ul = ul.first(ul.size() - 1);
        *pq1 = qj;
        if (pq1 == q1.data()) break;
        --pq1;
    }

    // q0 = r1 / d, with r1 < d1 <= 2 * B * d, so q0 needs two limbs. The frame is widened by one
    // limb to carry atop; *puh is contiguous with ul, having just been taken from it.
    assert(puh == ul.data() + ul.size());
    LimbT q0arr[3];
    std::span<LimbT> ulw{ ul.data(), ul.size() + 1 };
    auto qr = udiv_bc<LimbT>(&atop, puhh, ulw, dh, dl, q0arr + 2, alloc);
    ul = ulw;
    assert(!q0arr[2]);

    // q = k * q1 + q0
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, AllocatorT> qbuf(m1 + 2, alloc);
    {
        LimbT* qb = qbuf.data();
        LimbT* p = qb;
        LimbT c0 = umul1<LimbT>(q1.data(), q1.data() + m1, k[0], p); // p advances by m1
        *p = c0;
        LimbT* p1 = qb + 1;
        *(qb + m1 + 1) = umul1_add<LimbT>(q1.data(), q1.data() + m1, k[1], p1);

        LimbT c;
        std::tie(c, *qb) = numetron::arithmetic::uadd1<LimbT>(*qb, q0arr[0]);
        std::tie(c, *(qb + 1)) = numetron::arithmetic::uadd1<LimbT>(*(qb + 1), q0arr[1], c);
        for (size_t i = 2; c; ++i) {
            assert(i < m1 + 2);
            std::tie(c, *(qb + i)) = numetron::arithmetic::uadd1<LimbT>(*(qb + i), c);
        }
        assert(!*(qb + m1 + 1)); // q < B^m = B^(m1+1)

        for (size_t i = m1 + 1; i-- > 0;) {
            *qit = *(qb + i); --qit;
        }
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
    
    // Normalization may push the dividend one limb higher. The working frame is then one limb
    // longer than the quotient the caller asked for, and the extra leading digit it would produce
    // is provably 0 (q < B^(size(u) - size(d) + 1)), so it must not be written out: doing so would
    // shift the whole quotient one position below the caller's buffer.
    LimbT* puhh = &uhhstore;
    bool const extra_limb = uhhstore != 0;
    if (!extra_limb) {
        puhh = puh;
        puh = &ul.back();
        ul = ul.first(ul.size() - 1);
    }
    assert(*puhh);

    std::span<const LimbT> dnorm{ dlnorm, dl.size() };
    LimbT qtop = do_udiv_unorm(puhh, puh, ul, dh, dnorm);
    if (extra_limb) {
        assert(!qtop);
    } else {
        *qit = qtop; --qit;
    }
    auto [rhh, rh] = (ul.size() + 1 - dl.size() >= NUMETRON_SVOBODA_DIV_THRESHOLD)
        ? udiv_svoboda_unorm<LimbT>(puhh, puh, ul, dh, dnorm, std::move(qit), alloc)
        : udiv_bc_unorm<LimbT>(puhh, puh, ul, dh, dnorm, std::move(qit), alloc);

    // The remainder is below d, so it needs at most size(d) limbs and always fits into the
    // dividend's own storage; hand it back as [ul] plus the returned high limb. rh and rhh may
    // live in locals (uh / uhhstore), hence the explicit move into the buffer.
    if (rhh) {
        // rhh != 0 means the remainder needs more than ul.size() + 1 limbs, so this slot is inside
        // the dividend's storage
        ul = { ul.data(), ul.size() + 1 };
        ul.back() = rh;
        rh = rhh;
    }
    if (shift) {
        ushift_right<LimbT>(rh, ul, shift); // returns the discarded low bits
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

// u / v -> q, u % v -> r; q and r are fully written (zero padded above the significant limbs)
// prereqs: q.size() >= u.size(), r.size() >= v.size()
template <std::unsigned_integral LimbT>
inline void udiv(std::span<const LimbT> u, std::span<const LimbT> v, std::span<LimbT> q, std::span<LimbT> r)
{
    assert(q.size() >= u.size() && r.size() >= v.size());

    LimbT const* vb = v.data(), * ve = vb + v.size();
    for (;; --ve) {
        if (vb == ve) [[unlikely]] {
            throw std::runtime_error("division by zero");
        }
        if (*(ve - 1)) break;
    }
    if (1 == ve - vb) {
        r.front() = udivby1(u, *vb, q);
        std::memset(r.data() + 1, 0, (r.size() - 1) * sizeof(LimbT));
        return;
    }

    LimbT const* ue = u.data() + u.size();
    while (ue != u.data() && !*(ue - 1)) --ue;

    size_t const usz = ue - u.data(), vsz = ve - vb;
    if (usz < vsz) { // u < v => q = 0, r = u
        std::memcpy(r.data(), u.data(), usz * sizeof(LimbT));
        std::memset(r.data() + usz, 0, (r.size() - usz) * sizeof(LimbT));
        std::memset(q.data(), 0, q.size() * sizeof(LimbT));
        return;
    }

    // udiv() computes the remainder in place, so the dividend needs a scratch copy
    using alloc_t = std::allocator<LimbT>;
    small_array<LimbT, NUMETRON_INPLACE_LIMB_RESERVE_COUNT, alloc_t> uaux{ usz, alloc_t{} };
    std::memcpy(uaux.data(), u.data(), usz * sizeof(LimbT));

    size_t const qsz = usz - vsz + 1;
    std::span<LimbT> ul = uaux.first(usz - 1);
    LimbT rh = udiv<LimbT>(uaux.back(), ul, *(ve - 1), { vb, vsz - 1 }, q.data() + qsz - 1, alloc_t{});

    std::memset(q.data() + qsz, 0, (q.size() - qsz) * sizeof(LimbT));

    size_t rsz = ul.size();
    std::memcpy(r.data(), ul.data(), rsz * sizeof(LimbT));
    if (rh) r[rsz++] = rh;
    assert(rsz <= r.size());
    std::memset(r.data() + rsz, 0, (r.size() - rsz) * sizeof(LimbT));
}

}
