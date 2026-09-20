// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include <random>
#include <span>
#include <vector>

#ifdef _WIN32
#   pragma warning(disable : 4244 4146)
#endif
#include "gmp.h"

#include "numetron/basic_integer.hpp"
#include "numetron/limb_arithmetic/udiv.hpp"

namespace numetron {

namespace {

using limb_t = uint64_t;

constexpr limb_t guard_value = 0xDEADBEEFCAFEBABEull;

void to_mpz(mpz_t r, std::span<const limb_t> v)
{
    mpz_import(r, v.size(), -1, sizeof(limb_t), 0, 0, v.data());
}

// u / d and u % d against GMP; q and r carry a guard limb to catch out-of-range writes
void check_division(std::vector<limb_t> const& u, std::vector<limb_t> const& d)
{
    std::vector<limb_t> q(u.size() + 1, guard_value), r(d.size() + 1, guard_value);
    limb_arithmetic::udiv<limb_t>({ u.data(), u.size() }, { d.data(), d.size() },
        { q.data(), u.size() }, { r.data(), d.size() });

    CHECK_EQUAL(q.back(), guard_value);
    CHECK_EQUAL(r.back(), guard_value);

    mpz_t mu, md, mq, mr, mqe, mre;
    mpz_inits(mu, md, mq, mr, mqe, mre, nullptr);
    to_mpz(mu, u);
    to_mpz(md, d);
    to_mpz(mq, { q.data(), u.size() });
    to_mpz(mr, { r.data(), d.size() });
    mpz_tdiv_qr(mqe, mre, mu, md);

    CHECK_EQUAL(mpz_cmp(mq, mqe), 0);
    CHECK_EQUAL(mpz_cmp(mr, mre), 0);

    mpz_clears(mu, md, mq, mr, mqe, mre, nullptr);
}

}

void udiv_test()
{
    std::mt19937_64 rng{ 20250920 };

    for (size_t dsz = 2; dsz <= 8; ++dsz) {
        for (size_t usz = dsz; usz <= dsz + 6; ++usz) {
            for (int iter = 0; iter < 24; ++iter) {
                std::vector<limb_t> u(usz), d(dsz);
                for (limb_t& x : u) x = rng();
                for (limb_t& x : d) x = rng();

                switch (iter % 6) {
                case 0: break;                                  // random divisor
                case 1: d.back() |= (limb_t{ 1 } << 63); break; // already normalized: no shift
                case 2: d.back() = 1; break;                    // maximal normalization shift
                case 3:                                         // all ones: maximal correction pressure
                    std::fill(d.begin(), d.end(), ~limb_t{ 0 });
                    break;
                case 4:                                         // u shares its high limbs with d
                    std::copy(d.begin(), d.end(), u.begin() + (usz - dsz));
                    break;
                case 5:                                         // divisor just above a power of the base
                    std::fill(d.begin(), d.end() - 1, limb_t{ 0 });
                    d.front() = 1;
                    d.back() = limb_t{ 1 } << (rng() % 64);
                    break;
                }
                if (!u.back()) u.back() = 1;
                if (!d.back()) d.back() = 1;

                check_division(u, d);
            }
        }
    }

    // u < d with equal limb counts: quotient 0, remainder u
    check_division({ 1, 1 }, { 2, 2 });
    // exact division
    check_division({ 0, 0, 1 }, { 0, 1 });

    using namespace numetron::literals;

    // the same through the public type
    CHECK_EQUAL("340282366920938463463374607431768211456"_bi / "18446744073709551617"_bi,
        "18446744073709551615"_bi);
    CHECK_EQUAL("340282366920938463463374607431768211456"_bi % "18446744073709551617"_bi, 1);
    CHECK_EQUAL("219399878273287837459238450239485023985748738458787"_bi / "123456789012345678901234567890"_bi,
        "1777139030007882755090"_bi);
    CHECK_EQUAL("219399878273287837459238450239485023985748738458787"_bi % "123456789012345678901234567890"_bi,
        "24860483783064888429890398687"_bi);
}

}
