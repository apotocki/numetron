// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#ifdef _WIN32
#   pragma warning(disable : 4244 4146)
#endif
#include "gmp.h"

#include "numetron/basic_integer.hpp"

#include <cstdlib>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

long g_new_calls = 0;
long g_delete_calls = 0;
bool g_count_allocs = false;

std::string random_hex_operand(std::mt19937_64& rng, size_t limb_count)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::uniform_int_distribution<int> nonzero_digit(1, 15);
    std::uniform_int_distribution<int> any_digit(0, 15);

    std::string s;
    s.reserve(limb_count * 16);
    s.push_back(digits[nonzero_digit(rng)]); // top digit non-zero: keeps the operand exactly limb_count limbs wide
    for (size_t i = 1; i < limb_count * 16; ++i) {
        s.push_back(digits[any_digit(rng)]);
    }
    return s;
}

std::string gmp_hex_mul(std::string const& a, std::string const& b)
{
    mpz_t x, y, r;
    mpz_init_set_str(x, a.c_str(), 16);
    mpz_init_set_str(y, b.c_str(), 16);
    mpz_init(r);
    mpz_mul(r, x, y);
    std::unique_ptr<char, void(*)(void*)> s(mpz_get_str(nullptr, 16, r), [](void* p) { std::free(p); });
    std::string result = s.get();
    mpz_clears(x, y, r, nullptr);
    return result;
}

} // namespace

// Counts (de)allocations while g_count_allocs is set, otherwise behaves exactly like the
// default global operators -- used by assign_mul_no_realloc_test() below to verify that
// repeated assign_mul() calls stop allocating once the destination is big enough.
void* operator new(std::size_t sz)
{
    if (g_count_allocs) ++g_new_calls;
    void* p = std::malloc(sz);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept { if (g_count_allocs) ++g_delete_calls; std::free(p); }
void operator delete(void* p, std::size_t) noexcept { if (g_count_allocs) ++g_delete_calls; std::free(p); }

namespace numetron {

void assign_mul_test()
{
    using integer_t = integer;
    std::mt19937_64 rng{ 20260920 };

    // Correctness across sizes spanning the SBO boundary, the Karatsuba threshold (70 limbs)
    // and the Toom-3 threshold (160 limbs), into a destination that starts fresh each time.
    for (size_t limbs : { 1uz, 2uz, 3uz, 8uz, 16uz, 69uz, 70uz, 71uz, 159uz, 160uz, 161uz, 500uz }) {
        std::string a = random_hex_operand(rng, limbs), b = random_hex_operand(rng, limbs);
        integer_t u{ a, 16 }, v{ b, 16 };
        integer_t dst;
        dst.assign_mul(u, v);
        CHECK_EQUAL(to_string(dst, 16, false), gmp_hex_mul(a, b));
    }

    // Growing then shrinking sizes on the same destination -- exercises the reuse-or-grow
    // allocator's inplace<->heap transitions, not just a single fixed shape.
    {
        integer_t dst;
        for (size_t limbs : { 1uz, 2uz, 200uz, 5uz, 300uz, 1uz, 400uz, 2uz }) {
            std::string a = random_hex_operand(rng, limbs), b = random_hex_operand(rng, limbs);
            integer_t u{ a, 16 }, v{ b, 16 };
            dst.assign_mul(u, v);
            CHECK_EQUAL(to_string(dst, 16, false), gmp_hex_mul(a, b));
        }
    }

    // Aliasing: the destination may be one (or both) of the operands -- assign_mul() must fall
    // back to the always-correct allocating path rather than reading from storage it is
    // concurrently overwriting.
    {
        std::string a = random_hex_operand(rng, 40), b = random_hex_operand(rng, 40);
        integer_t x{ a, 16 }, y{ b, 16 };
        std::string expect = gmp_hex_mul(a, b);
        x.assign_mul(x, y);
        CHECK_EQUAL(to_string(x, 16, false), expect);
    }
    {
        std::string a = random_hex_operand(rng, 40), b = random_hex_operand(rng, 40);
        integer_t x{ a, 16 }, y{ b, 16 };
        std::string expect = gmp_hex_mul(a, b);
        y.assign_mul(x, y);
        CHECK_EQUAL(to_string(y, 16, false), expect);
    }
    {
        std::string a = random_hex_operand(rng, 40);
        integer_t x{ a, 16 };
        std::string expect = gmp_hex_mul(a, a);
        x.assign_mul(x, x);
        CHECK_EQUAL(to_string(x, 16, false), expect);
    }

    // A zero operand after the destination is already heap-backed must still free that buffer
    // (init_zero() alone doesn't -- see free_if_replaced()).
    {
        integer_t dst;
        std::string a = random_hex_operand(rng, 300), b = random_hex_operand(rng, 300);
        integer_t u{ a, 16 }, v{ b, 16 };
        dst.assign_mul(u, v);
        integer_t zero{ 0 };
        dst.assign_mul(u, zero);
        CHECK_EQUAL(to_string(dst, 16, false), "0");
    }
}

// Regression test for the whole point of assign_mul(): repeated calls into the same,
// already-large-enough destination must stop allocating -- otherwise it is no better than plain
// operator*, which always builds a fresh result (see the "1" vs "1*" rows in
// bench/mul_bench.cpp, and CLAUDE.md's note on basic_integer's small-buffer optimization).
void assign_mul_no_realloc_test()
{
    using integer_t = integer;
    std::mt19937_64 rng{ 20260920 };

    // Covers the plain basecase path (64 limbs) and the Karatsuba/Toom-3 paths (100/250 limbs),
    // which allocate their own transient scratch buffer through the SAME allocator instance --
    // this also verifies that scratch use is never mistaken for the reusable result buffer.
    for (size_t limbs : { 64uz, 100uz, 250uz }) {
        integer_t dst;
        std::vector<std::pair<integer_t, integer_t>> ops;
        ops.reserve(50);
        for (int i = 0; i < 50; ++i) {
            ops.emplace_back(integer_t{ random_hex_operand(rng, limbs), 16 }, integer_t{ random_hex_operand(rng, limbs), 16 });
        }
        dst.assign_mul(ops[0].first, ops[0].second); // warm-up: this call may allocate

        g_count_allocs = true;
        long before = g_new_calls;
        for (auto& op : ops) {
            dst.assign_mul(op.first, op.second);
        }
        g_count_allocs = false;

        CHECK_EQUAL(g_new_calls - before, 0);
    }
}

}
