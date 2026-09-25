// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

// Products of many shapes u x v (un >= vn) against GMP's mpn_mul, through both entry points:
// limb_arithmetic::umul() (allocates the result, picks the top-level algorithm) and umul_dispatch()
// (what the Karatsuba/Toom recursion calls). The shapes are built around the current thresholds,
// so they cross every algorithm boundary and the slicing of u into vn-limb pieces (un >= 2vn):
// exact multiples of vn, one limb more / less, half a piece left over.

#include "test_common.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <vector>

#ifdef _WIN32
#   pragma warning(disable : 4244 4146)
#endif
#include "gmp.h"

#include "numetron/limb_arithmetic.hpp"

namespace numetron {

namespace {

using limb = std::uint64_t;

// kind 0: random; 1: all ones (longest carry chains); 2: random with the top limbs of every
// vn-limb piece of u zero (umul_dispatch strips them, the piece products come back short);
// 3: random with the whole second piece of u zero.
void fill_operands(std::mt19937_64& rng, int kind, std::vector<limb>& u, std::vector<limb>& v)
{
    const size_t vn = v.size();
    for (auto& x : u) x = kind == 1 ? ~limb{ 0 } : rng();
    for (auto& x : v) x = kind == 1 ? ~limb{ 0 } : rng();
    if (kind == 2) {
        for (size_t i = 0; i < u.size(); ++i)
            if (i % vn >= vn - (std::min)(vn / 2, size_t{ 3 })) u[i] = 0;
    }
    if (kind == 3) {
        for (size_t i = vn; i < (std::min)(u.size(), 2 * vn); ++i) u[i] = 0;
    }
    u.back() |= 1; // keep the top limbs non-zero: un and vn stay what the shape says
    v.back() |= 1;
}

}

void mul_shapes_test()
{
    namespace la = numetron::limb_arithmetic;

    std::mt19937_64 rng{ 0x5A1CEDULL };
    size_t checked = 0, failed = 0;

    auto run_pass = [&](char const* label) {
        std::vector<size_t> v_sizes;
        auto add_around = [&](size_t t) {
            if (t == ~size_t{ 0 } || t > 800) return; // FFT or off: out of range here
            for (size_t d : { t - 1, t, t + 1 }) if (d >= 2) v_sizes.push_back(d);
        };
        add_around(la::karatsuba_threshold());
        add_around(la::slicing_threshold());
        add_around(la::toom32_threshold());
        add_around(la::toom42_threshold());
        add_around(la::toom76_threshold());
        add_around(la::toom63_threshold());
        add_around(la::toom98_threshold());
        add_around(la::toom107_threshold());
        add_around(la::toom116_threshold());
        add_around(la::toom54_threshold());
        add_around(la::toom53_threshold());
        add_around(la::toom43_threshold());
        if (la::toom116_threshold() != ~size_t{ 0 }) add_around(la::toom116_threshold() + la::toom116_threshold() / 4 * 3); // 11 x 6 near 2 vn
        add_around(la::toom3_threshold());
        add_around(la::toom4_threshold());
        add_around(la::toom6h_threshold());
        for (size_t n : { 2, 5, 8, 17, 64, 100, 257 }) v_sizes.push_back(n);
        std::sort(v_sizes.begin(), v_sizes.end());
        v_sizes.erase(std::unique(v_sizes.begin(), v_sizes.end()), v_sizes.end());

        for (size_t vn : v_sizes) {
            std::vector<size_t> u_sizes;
            for (size_t m : { 1, 2, 3, 5, 8 }) {
                for (size_t d : { size_t{ 0 }, size_t{ 1 }, vn / 2, vn - 1 }) u_sizes.push_back(m * vn + d);
            }
            // the toom32 window [1.25 vn, 1.75 vn) and where its chunk switches from ceil(vn/2)
            // to ceil(un/3) (un ~ 1.5 vn)
            // and the toom42 window [1.75 vn, 2 vn) with its chunk switch (un ~ 2 vn), and toom76's
            // (vn, 1.4 vn) with its chunk switch (un ~ 7/6 vn)
            for (size_t un : { (5 * vn + 3) / 4 - 1, (5 * vn + 3) / 4, vn + vn / 2 - 1, vn + vn / 2, vn + vn / 2 + 1,
                               (7 * vn + 3) / 4 - 1, (7 * vn + 3) / 4, (15 * vn) / 8, 2 * vn - 2, 2 * vn - 1,
                               (7 * vn) / 6 - 1, (7 * vn) / 6 + 1, (6 * vn) / 5 + 1, (13 * vn) / 10, (7 * vn) / 5 - 1,
                               // the Toom-8.5 N x (17 - N) windows and their chunk switches: 9 x 8 on
                               // [1.08 vn, 1.2 vn) (switch at 9/8 vn), 10 x 7 on [1.32 vn, 5/3 vn)
                               // (10/7 vn), 11 x 6 on (5/3 vn, 2 vn) (11/6 vn)
                               (27 * vn + 24) / 25, (9 * vn) / 8, (9 * vn) / 8 + 1, (6 * vn) / 5 - 1,
                               (33 * vn + 24) / 25, (10 * vn) / 7, (10 * vn) / 7 + 1, (5 * vn) / 3 - 1, (5 * vn) / 3 + 1,
                               (11 * vn) / 6, (11 * vn) / 6 + 1, (39 * vn + 19) / 20 - 1, (39 * vn + 19) / 20,
                               // toom54 on [1.2 vn, 1.45 vn) (chunk switch at 5/4 vn), toom53 on
                               // [1.55 vn, 1.85 vn) (5/3 vn, above)
                               (6 * vn + 4) / 5, (5 * vn) / 4, (5 * vn) / 4 + 1, (29 * vn + 19) / 20 - 1,
                               (31 * vn + 19) / 20, (37 * vn + 19) / 20 - 1,
                               // toom43 on [1.3 vn, 1.45 vn) (chunk switch at 4/3 vn)
                               (13 * vn + 9) / 10, (4 * vn) / 3, (4 * vn) / 3 + 1 }) {
                if (un >= vn) u_sizes.push_back(un);
            }
            if (vn <= 64) u_sizes.push_back(40 * vn + 3);
            std::sort(u_sizes.begin(), u_sizes.end());
            u_sizes.erase(std::unique(u_sizes.begin(), u_sizes.end()), u_sizes.end());

            for (size_t un : u_sizes) {
                for (int kind = 0; kind < 4; ++kind) {
                    std::vector<limb> u(un), v(vn);
                    fill_operands(rng, kind, u, v);

                    std::vector<limb> expected(un + vn);
                    mpn_mul(reinterpret_cast<mp_limb_t*>(expected.data()),
                        reinterpret_cast<const mp_limb_t*>(u.data()), static_cast<mp_size_t>(un),
                        reinterpret_cast<const mp_limb_t*>(v.data()), static_cast<mp_size_t>(vn));
                    size_t expected_size = un + vn;
                    while (expected_size && !expected[expected_size - 1]) --expected_size;

                    // umul(): may leave leading zero limbs in the size it returns (its callers trim)
                    {
                        std::allocator<limb> alloc;
                        auto [r, rsize, rcap] = la::umul<limb>(std::span<const limb>{ u }, std::span<const limb>{ v }, alloc);
                        while (rsize && !r[rsize - 1]) --rsize;
                        ++checked;
                        if (rsize != expected_size || !std::equal(r, r + rsize, expected.data())) {
                            if (++failed <= 10) std::cout << label << ": umul MISMATCH " << un << " x " << vn << " kind " << kind << "\n";
                        }
                        alloc.deallocate(r, rcap);
                    }
                    // umul_dispatch(): writes up to what it returns; the rest is zero
                    {
                        std::vector<limb> r(un + vn + 1, 0xABABABABABABABABULL); // one guard limb
                        limb* e = la::umul_dispatch(u.data(), un, v.data(), vn, r.data(), numetron::detail::stack_allocator<limb>{});
                        std::fill(e, r.data() + un + vn, limb{ 0 });
                        ++checked;
                        if (!std::equal(expected.begin(), expected.end(), r.begin()) || r[un + vn] != 0xABABABABABABABABULL) {
                            if (++failed <= 10) std::cout << label << ": umul_dispatch MISMATCH " << un << " x " << vn << " kind " << kind << "\n";
                        }
                    }
                }
            }
        }
    };

    run_pass("default thresholds");

    // The unbalanced plans from their minimum: they take every node in their windows, down to a
    // few limbs.
    const size_t saved_toom32 = la::toom32_threshold();
    const size_t saved_toom42 = la::toom42_threshold();
    const size_t saved_toom76 = la::toom76_threshold();
    const size_t saved_toom63 = la::toom63_threshold();
    const size_t saved_toom98 = la::toom98_threshold();
    const size_t saved_toom107 = la::toom107_threshold();
    const size_t saved_toom116 = la::toom116_threshold();
    const size_t saved_toom54 = la::toom54_threshold();
    const size_t saved_toom53 = la::toom53_threshold();
    const size_t saved_toom43 = la::toom43_threshold();
    // toom54 / toom53 / toom43 off here: below the Toom-6.5 threshold they'd take the others' windows
    la::set_toom54_threshold(~size_t{ 0 });
    la::set_toom53_threshold(~size_t{ 0 });
    la::set_toom43_threshold(~size_t{ 0 });
    la::set_toom32_threshold(la::min_toom32_threshold);
    la::set_toom42_threshold(la::min_toom42_threshold);
    la::set_toom76_threshold(la::min_toom76_threshold);
    la::set_toom63_threshold(la::min_toom63_threshold);
    la::set_toom98_threshold(la::min_toom98_threshold);
    la::set_toom107_threshold(la::min_toom107_threshold);
    la::set_toom116_threshold(la::min_toom116_threshold);
    run_pass("toom32/42/63/76/98/107/116 from the minimum");
    la::set_toom32_threshold(saved_toom32);
    la::set_toom42_threshold(saved_toom42);
    la::set_toom76_threshold(saved_toom76);
    la::set_toom63_threshold(saved_toom63);
    la::set_toom98_threshold(saved_toom98);
    la::set_toom107_threshold(saved_toom107);
    la::set_toom116_threshold(saved_toom116);

    // Toom-5/4, Toom-5/3 and Toom-4/3 from their minimum, everything else at its default.
    la::set_toom54_threshold(la::min_toom54_threshold);
    la::set_toom53_threshold(la::min_toom53_threshold);
    la::set_toom43_threshold(la::min_toom43_threshold);
    run_pass("toom54/53/43 from the minimum");
    la::set_toom54_threshold(saved_toom54);
    la::set_toom53_threshold(saved_toom53);
    la::set_toom43_threshold(saved_toom43);

    // The FFT from 64 limbs: the shapes above with vn >= 64 take it, and the long ones are cut
    // into pieces that fill a transform length (fft_slice_length()), with a remainder piece.
    const size_t saved_fft = la::fft_threshold();
    la::set_fft_threshold(64);
    run_pass("FFT from 64 limbs");
    la::set_fft_threshold(saved_fft);

    std::cout << "mul shapes: checked " << checked << ", failed " << failed << "\n";
    CHECK_EQUAL(failed, 0u);
}

}
