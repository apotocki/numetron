// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include <limits>
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

// Svoboda is off by default, so the tests that cover it pass a threshold of their own rather than
// relying on the tunable. Doing it with a macro would not work: udiv() is an inline template, and
// redefining the tunable in this translation unit alone leaves the linker free to pick another
// one's body.
constexpr size_t svoboda_always = 2;
constexpr size_t svoboda_default = NUMETRON_SVOBODA_DIV_THRESHOLD;

// u / d and u % d against GMP; q and r carry a guard limb to catch out-of-range writes
void check_division(std::vector<limb_t> const& u, std::vector<limb_t> const& d,
    size_t svoboda_threshold = svoboda_default)
{
    std::vector<limb_t> q(u.size() + 1, guard_value), r(d.size() + 1, guard_value);
    limb_arithmetic::udiv<limb_t>({ u.data(), u.size() }, { d.data(), d.size() },
        { q.data(), u.size() }, { r.data(), d.size() }, svoboda_threshold);

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

constexpr size_t limb_bits = std::numeric_limits<limb_t>::digits;

void from_mpz(mpz_t v, std::vector<limb_t>& out)
{
    out.assign((mpz_sizeinbase(v, 2) + limb_bits - 1) / limb_bits + 1, 0);
    size_t count = 0;
    mpz_export(out.data(), &count, -1, sizeof(limb_t), 0, 0, v);
    out.resize(count ? count : 1);
}

void random_limbs(std::mt19937_64& rng, std::vector<limb_t>& v, size_t sz)
{
    v.resize(sz);
    for (limb_t& x : v) x = rng();
}

// Dividends whose Svoboda partial remainder lands in [B^(n+1), d1), where d1 = ceil(B^(n+1)/d) * d
// is the scaled divisor: the corrected remainder then needs one limb more than the working frame.
// Random operands reach this only with probability ~2^-64 per digit, so it has to be constructed.
void check_svoboda_carry(std::mt19937_64& rng, size_t n, size_t m1)
{
    std::vector<limb_t> d, tmp;
    random_limbs(rng, d, n);
    d.back() |= (limb_t{ 1 } << (limb_bits - 1)); // normalized, so udiv does not shift it

    mpz_t md, bn1, k, d1, window, r, q1, u;
    mpz_inits(md, bn1, k, d1, window, r, q1, u, nullptr);
    to_mpz(md, d);
    mpz_setbit(bn1, static_cast<mp_bitcnt_t>((n + 1) * limb_bits));
    mpz_cdiv_q(k, bn1, md);
    mpz_mul(d1, k, md);
    mpz_sub(window, d1, bn1);

    if (mpz_sgn(window) > 0) {
        random_limbs(rng, tmp, n);
        to_mpz(r, tmp);
        mpz_mod(r, r, window);
        mpz_add(r, r, bn1); // r in [B^(n+1), d1)

        // q1 < B^m1 / 2 keeps u below d * B^m, so that the leading quotient digit stays 0
        random_limbs(rng, tmp, m1);
        tmp.back() = (tmp.back() >> 1) | (limb_t{ 1 } << (limb_bits - 2));
        to_mpz(q1, tmp);

        mpz_mul(u, q1, d1);
        mpz_add(u, u, r);
        from_mpz(u, tmp);
        check_division(tmp, d, svoboda_always); // the last step ends holding the extra limb

        mpz_mul_2exp(u, u, 2 * limb_bits);
        mpz_add_ui(u, u, static_cast<unsigned long>(rng() & 0xFFFFFFFFu));
        from_mpz(u, tmp);
        check_division(tmp, d, svoboda_always); // ... and here a later step consumes it
    }

    mpz_clears(md, bn1, k, d1, window, r, q1, u, nullptr);
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

    // the same through Svoboda's division, including its rare extra-limb branch
    for (size_t dsz : { size_t{ 2 }, size_t{ 3 }, size_t{ 9 } }) {
        for (int iter = 0; iter < 8; ++iter) {
            std::vector<limb_t> u, d;
            random_limbs(rng, u, dsz + 2 + (rng() % 40));
            random_limbs(rng, d, dsz);
            switch (iter % 4) {
            case 1: d.back() = 1; break;                            // maximal normalization shift
            case 2: std::fill(d.begin(), d.end(), ~limb_t{ 0 }); break;
            case 3:                                                 // scaled divisor is exact: k * d == B^(n+1)
                std::fill(d.begin(), d.end() - 1, limb_t{ 0 });
                d.back() = limb_t{ 1 } << 63;
                break;
            }
            if (!u.back()) u.back() = 1;
            if (!d.back()) d.back() = 1;
            check_division(u, d, svoboda_always);
        }
        for (int iter = 0; iter < 4; ++iter) check_svoboda_carry(rng, dsz, 8);
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
