// Numetron — Compile - time and runtime arbitrary - precision arithmetic
// (c) 2025 Alexander Pototskiy
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
            
                LimbT * tmpr = daux.data();
                LimbT mdhh = umul1<LimbT>(dh, dl.data(), dl.data() + dl.size(), qj, tmpr);
                LimbT mdh = *(tmpr - 1);
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
    umul<LimbT>(dl.data(), dl.data() + dl.size(), k, k + 2, d1.data()); // dl * k
    kdh[2] = umul1<LimbT>(k, k + 2, dh, (LimbT*)kdh);
    uadd<LimbT>({ d1.data() + dl.size(), d1.size() - dl.size() }, { kdh, 3 });
    if (d1[dl.size() + 1] != 0) {
        assert(d1[dl.size() + 1] == 0);
    }
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
            LimbT* tmpr = dauxl.data();
            LimbT uc = umul1<LimbT>(d1.data(), d1.data() + d1.size(), qj, tmpr);
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
    LimbT *qe = umul<LimbT>(q1.data(), q1.data() + q1.size(), k, k + 2, qb) - 1;
    auto [q0, qbval] = numetron::arithmetic::uadd1(*qb, q0arr[0]);
    *qb = qbval; q0 += static_cast<unsigned char>(q0arr[1]);
    if (q0) {
        LimbT* qit = qb;
        do {
            ++qit;
            std::tie(q0, *qit) = numetron::arithmetic::uadd1<LimbT>(*qit, q0);
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
        std::memset(r.data() + 1, 0, (r.size() - 1) * sizeof(LimbT));
        return;
    }
    throw std::runtime_error("not implemented");
}

}
