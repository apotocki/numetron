// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "platform.hpp"
#include "umul1.hpp"

namespace numetron::limb_arithmetic {

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
inline LimbT* umul_basecase(LimbT const* ub, size_t un, LimbT const* vb, size_t vn, LimbT* rb) noexcept
{
    if constexpr (sizeof(LimbT) == 8) {
#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
#   if defined(NUMETRON_PLATFORM_AUTODETECT)
        std::call_once(mul_basecase_init_flag, []() {
            uint64_t platform_descriptor = numetron_detect_platform();
            //std::cout << "PLATFROM: " << std::hex << "0x" << platform_descriptor << std::dec << std::endl;
            __mul_basecase_ptr = detect_mul_basecase(platform_descriptor);
        });
        if (__mul_basecase_ptr) __mul_basecase_ptr(rb, ub, un, vb, vn);
        return rb + un + vn;
#   else
        NUMETRON_mul_basecase(rb, ub, un, vb, vn);
        return rb + un + vn;
#   endif
#endif
    }
    return umul_basecase_unrolled<LimbT>(ub, ub + un, vb, vb + vn, rb);
    //return umul<LimbT, LimbT*>(ub, ue, vb, ve, rb);
}

}
