// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

// Standalone benchmark comparing Numetron's basic_integer multiplication against GMP's
// mpz_mul across a range of operand sizes (in 64-bit limbs). Not part of the GoogleTest
// suite -- build the numetron_bench_mul target and run the resulting executable directly.
//
// The library is header-only (pure C++) unless NUMETRON_USE_ASM is defined. This target links
// the src/arch assembly and gets NUMETRON_USE_ASM with it (the CMake `numetron` target exports
// it; msvc/numetron_bench_mul.vcxproj sets it), so it measures Numetron's best runtime-selected
// mul_basecase implementation, not the plain C++ fallback. The other implementation choices (Karatsuba,
// Toom-3) are defaults from the same header, overridable per build with compiler flags, e.g.
// -DNUMETRON_KARATSUBA_IMPL=NUMETRON_KARATSUBA_IMPL_FUSED; the output header names the ones in use.
//
// Modes: --balanced (un == vn), --unbalanced (un > vn, several vn x un/vn); neither: both.
// --large: the unbalanced shapes with vn = 4096 .. 16384 (the FFT range) instead.
// --sqr: squares u * u (the same object on both sides) against GMP, and against u * v.
// --small: a dense grid of small operands (products n x n and un x vn, squares, add/sub; 1..32
// limbs), where the calls around the basecase cost most; alone unless other modes are given too.
// --max-limbs=N: --balanced and --sqr only up to N limbs.
// --tune[=N] [--trace] retunes the thresholds first; --add benchmarks limb add/sub instead.
// --csv=FILE: every measured point also as a row "section,shape,column,ns" in FILE (the
// configuration first, as '#' lines), for comparing builds with tools/bench_compare.py.

#ifdef _WIN32
#   pragma warning(disable : 4244 4146)
#endif
#include "gmp.h"

#include "numetron/basic_integer.hpp"
#include "numetron/limb_arithmetic/mul_tuning.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

// Written to from every timed multiplication below, so the optimizer can't prove the
// result is dead and drop the call.
volatile std::uint64_t g_sink = 0;

// --csv=FILE: the machine-readable copy of every measured point; g_section names the table the
// rows belong to (balanced, unbalanced, sqr, add, small_*).
std::ofstream g_csv;
std::string g_section;

void csv_row(std::string const& shape, char const* column, double ns)
{
    if (g_csv.is_open()) g_csv << g_section << ',' << shape << ',' << column << ',' << ns << '\n';
}

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

struct operand_pair
{
    std::string u_hex, v_hex;
};

std::vector<operand_pair> make_operands(std::mt19937_64& rng, size_t u_limbs, size_t v_limbs, size_t count)
{
    std::vector<operand_pair> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back({ random_hex_operand(rng, u_limbs), random_hex_operand(rng, v_limbs) });
    }
    return result;
}

std::string to_hex16(uint64_t value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string s(16, '0');
    for (int i = 15; i >= 0 && value; --i, value >>= 4) {
        s[i] = digits[value & 0xf];
    }
    return s;
}

// Both factors are capped to 31 bits, so u*v < 2^62 always fits in a single 64-bit limb. This
// isolates the raw multiply cost from the heap allocation numetron::integer (1-limb SBO) pays
// whenever a 1x1-limb product overflows into 2 limbs -- which is the common case for the
// full-range random operands the "1" row (limb_counts[0]) uses, since two ~64-bit factors
// almost always produce a ~128-bit product.
std::vector<operand_pair> make_small_operands(std::mt19937_64& rng, size_t count)
{
    std::uniform_int_distribution<uint64_t> small(1, 0x7fffffffULL);
    std::vector<operand_pair> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back({ to_hex16(small(rng)), to_hex16(small(rng)) });
    }
    return result;
}

// Cross-checks every sample against GMP before timing it, so a benchmark result is never
// silently measuring a wrong (or differently-shaped) multiplication.
void verify_against_gmp(std::vector<operand_pair> const& operands)
{
    using integer = numetron::integer;

    mpz_t gu, gv, gr;
    mpz_inits(gu, gv, gr, nullptr);

    for (auto const& o : operands) {
        integer u{ o.u_hex, 16 };
        integer v{ o.v_hex, 16 };
        integer product = u * v;

        mpz_set_str(gu, o.u_hex.c_str(), 16);
        mpz_set_str(gv, o.v_hex.c_str(), 16);
        mpz_mul(gr, gu, gv);

        std::unique_ptr<char, void(*)(void*)> gmp_hex(mpz_get_str(nullptr, 16, gr), [](void* p) { std::free(p); });
        std::string numetron_hex = to_string(product, 16, false);

        if (numetron_hex != gmp_hex.get()) {
            std::cerr << "MISMATCH for " << (o.u_hex.size() / 16) << " x " << (o.v_hex.size() / 16) << "-limb operands:\n"
                      << "  numetron: " << numetron_hex << "\n"
                      << "  gmp     : " << gmp_hex.get() << "\n";
            std::exit(1);
        }
    }

    mpz_clears(gu, gv, gr, nullptr);
}

double time_numetron_mul(std::vector<operand_pair> const& operands, int repeats)
{
    using integer = numetron::integer;

    std::vector<std::pair<integer, integer>> ops;
    ops.reserve(operands.size());
    for (auto const& o : operands) {
        ops.emplace_back(integer{ o.u_hex, 16 }, integer{ o.v_hex, 16 });
    }

    auto start = clock_type::now();
    for (int r = 0; r < repeats; ++r) {
        for (auto const& op : ops) {
            integer result = op.first * op.second;
            auto [high, low_limbs] = result.limbs();
            g_sink ^= low_limbs.empty() ? high : low_limbs.front();
        }
    }
    auto finish = clock_type::now();

    return std::chrono::duration<double, std::nano>(finish - start).count() / (double(repeats) * double(ops.size()));
}

// Same as time_numetron_mul(), but writes every product into one result object reused across
// the whole loop via assign_mul() instead of building a fresh basic_integer every call -- the
// Numetron-side equivalent of how time_gmp_mul() below keeps reusing its single mpz_t r. This
// is the fair, apples-to-apples comparison against GMP; time_numetron_mul() instead measures
// the cost of plain operator*'s everyday (always-allocating) usage.
double time_numetron_mul_reuse(std::vector<operand_pair> const& operands, int repeats)
{
    using integer = numetron::integer;

    std::vector<std::pair<integer, integer>> ops;
    ops.reserve(operands.size());
    for (auto const& o : operands) {
        ops.emplace_back(integer{ o.u_hex, 16 }, integer{ o.v_hex, 16 });
    }

    integer result;
    auto start = clock_type::now();
    for (int r = 0; r < repeats; ++r) {
        for (auto const& op : ops) {
            result.assign_mul(op.first, op.second);
            auto [high, low_limbs] = result.limbs();
            g_sink ^= low_limbs.empty() ? high : low_limbs.front();
        }
    }
    auto finish = clock_type::now();

    return std::chrono::duration<double, std::nano>(finish - start).count() / (double(repeats) * double(ops.size()));
}

double time_gmp_mul(std::vector<operand_pair> const& operands, int repeats)
{
    std::vector<std::pair<mpz_t, mpz_t>> ops(operands.size());
    for (size_t i = 0; i < operands.size(); ++i) {
        mpz_init_set_str(ops[i].first, operands[i].u_hex.c_str(), 16);
        mpz_init_set_str(ops[i].second, operands[i].v_hex.c_str(), 16);
    }

    mpz_t r;
    mpz_init(r);

    auto start = clock_type::now();
    for (int rep = 0; rep < repeats; ++rep) {
        for (auto& op : ops) {
            mpz_mul(r, op.first, op.second);
            g_sink ^= mpz_getlimbn(r, 0);
        }
    }
    auto finish = clock_type::now();

    mpz_clear(r);
    for (auto& op : ops) {
        mpz_clear(op.first);
        mpz_clear(op.second);
    }

    return std::chrono::duration<double, std::nano>(finish - start).count() / (double(repeats) * double(ops.size()));
}

// Repeat count per (limb_count, attempt), scaled so every tier does roughly comparable
// total work. The n^2 estimate is only a rough guide for keeping the run within a few
// seconds -- it doesn't need to model Numetron's actual (sub-quadratic for large operands)
// complexity.
int repeats_for(size_t u_limbs, size_t v_limbs, double budget = 4.0e8)
{
    double estimate = budget / (double(u_limbs) * double(v_limbs) + 1.0);
    int repeats = std::clamp(int(estimate), 1, 20000);

    // At <=16 limbs a single multiply takes only a few nanoseconds, close enough to the clock's
    // resolution/call overhead that the measurement noise floor starts to matter -- run 10x more
    // repeats there so the timed interval stays comfortably above it.
    if (u_limbs <= 16) repeats *= 10;

    return repeats;
}

int repeats_for(size_t limb_count) { return repeats_for(limb_count, limb_count); }

double ns_to_us(double ns) { return ns / 1000.0; }

static constexpr int attempts = 3;

// with_plain = false leaves out the plain operator* column (and its timing): --unbalanced shows
// only reuse, the allocation-free product that compares with GMP's mpz_mul into a reused result.
void run_tier(std::string const& limbs_label, std::string const& bits_label, std::vector<operand_pair> const& operands, int repeats,
    int label_width = 10, bool with_plain = true)
{
    verify_against_gmp(operands);

    double best_numetron_ns = std::numeric_limits<double>::infinity();
    double best_reuse_ns = std::numeric_limits<double>::infinity();
    double best_gmp_ns = std::numeric_limits<double>::infinity();
    for (int attempt = 0; attempt < attempts; ++attempt) {
        if (with_plain) best_numetron_ns = std::min(best_numetron_ns, time_numetron_mul(operands, repeats));
        best_reuse_ns = std::min(best_reuse_ns, time_numetron_mul_reuse(operands, repeats));
        best_gmp_ns = std::min(best_gmp_ns, time_gmp_mul(operands, repeats));
    }

    if (with_plain) csv_row(limbs_label, "plain", best_numetron_ns);
    csv_row(limbs_label, "reuse", best_reuse_ns);
    csv_row(limbs_label, "gmp", best_gmp_ns);

    std::cout << std::setw(label_width) << limbs_label
               << std::setw(12) << bits_label;
    if (with_plain) std::cout << std::setw(16) << std::fixed << std::setprecision(3) << ns_to_us(best_numetron_ns);
    std::cout << std::setw(16) << std::fixed << std::setprecision(3) << ns_to_us(best_reuse_ns)
               << std::setw(16) << std::fixed << std::setprecision(3) << ns_to_us(best_gmp_ns)
               << std::setw(14) << std::fixed << std::setprecision(2) << (best_gmp_ns / best_reuse_ns)
               << "\n";
}

// --sqr: u * u with the same object on both sides (GMP takes its squaring path only then, on
// mpz_mul(r, u, u); the same for numetron), against GMP and against the general product of two
// different operands of the size -- gmp mul/sqr is what squaring buys GMP, i.e. what there is to
// gain.
double time_numetron_sqr_reuse(std::vector<operand_pair> const& operands, int repeats)
{
    using integer = numetron::integer;
    std::vector<integer> ops;
    ops.reserve(operands.size());
    for (auto const& o : operands) ops.emplace_back(o.u_hex, 16);

    integer result;
    auto start = clock_type::now();
    for (int r = 0; r < repeats; ++r) {
        for (auto const& op : ops) {
            result.assign_mul(op, op);
            auto [high, low_limbs] = result.limbs();
            g_sink ^= low_limbs.empty() ? high : low_limbs.front();
        }
    }
    auto finish = clock_type::now();
    return std::chrono::duration<double, std::nano>(finish - start).count() / (double(repeats) * double(ops.size()));
}

double time_gmp_sqr(std::vector<operand_pair> const& operands, int repeats)
{
    std::vector<mpz_t> ops(operands.size());
    for (size_t i = 0; i < operands.size(); ++i) mpz_init_set_str(ops[i], operands[i].u_hex.c_str(), 16);
    mpz_t r;
    mpz_init(r);
    auto start = clock_type::now();
    for (int rep = 0; rep < repeats; ++rep) {
        for (auto& op : ops) {
            mpz_mul(r, op, op);
            g_sink ^= mpz_getlimbn(r, 0);
        }
    }
    auto finish = clock_type::now();
    mpz_clear(r);
    for (auto& op : ops) mpz_clear(op);
    return std::chrono::duration<double, std::nano>(finish - start).count() / (double(repeats) * double(ops.size()));
}

// The squares u * u of the operands' u against GMP (the u * v products are checked by
// verify_against_gmp()).
void verify_sqr_against_gmp(size_t limbs, std::vector<operand_pair> const& operands)
{
    using integer = numetron::integer;
    mpz_t g, r;
    mpz_inits(g, r, nullptr);
    for (auto const& o : operands) {
        integer u{ o.u_hex, 16 };
        integer p;
        p.assign_mul(u, u);
        mpz_set_str(g, o.u_hex.c_str(), 16);
        mpz_mul(r, g, g);
        std::unique_ptr<char, void(*)(void*)> gmp_hex(mpz_get_str(nullptr, 16, r), [](void* q) { std::free(q); });
        if (to_string(p, 16, false) != gmp_hex.get()) {
            std::cerr << "MISMATCH for the square of a " << limbs << "-limb operand\n";
            std::exit(1);
        }
    }
    mpz_clears(g, r, nullptr);
}

void run_sqr_tier(size_t limbs, std::vector<operand_pair> const& operands, int repeats)
{
    verify_sqr_against_gmp(limbs, operands);

    double sqr = std::numeric_limits<double>::infinity(), gsqr = sqr, mul = sqr, gmul = sqr;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        sqr = std::min(sqr, time_numetron_sqr_reuse(operands, repeats));
        gsqr = std::min(gsqr, time_gmp_sqr(operands, repeats));
        mul = std::min(mul, time_numetron_mul_reuse(operands, repeats));
        gmul = std::min(gmul, time_gmp_mul(operands, repeats));
    }
    const std::string shape = std::to_string(limbs);
    csv_row(shape, "sqr", sqr);
    csv_row(shape, "gmp_sqr", gsqr);
    csv_row(shape, "mul", mul);
    csv_row(shape, "gmp_mul", gmul);
    std::cout << std::setw(10) << limbs << std::fixed
              << std::setw(14) << std::setprecision(3) << ns_to_us(sqr)
              << std::setw(14) << ns_to_us(gsqr)
              << std::setw(12) << std::setprecision(2) << (gsqr / sqr)
              << std::setw(14) << std::setprecision(3) << ns_to_us(mul)
              << std::setw(14) << ns_to_us(gmul)
              << std::setw(12) << std::setprecision(2) << (gmul / gsqr)
              << std::setw(12) << (mul / sqr)
              << "\n";
}

template <typename OpT>
double best_ns_per_limb(size_t n, OpT&& op)
{
    const size_t reps = (std::max)(size_t{ 1 }, size_t{ 20'000'000 } / n);
    double best = std::numeric_limits<double>::infinity();
    for (int attempt = 0; attempt < 5; ++attempt) {
        auto start = clock_type::now();
        for (size_t r = 0; r < reps; ++r) op();
        auto finish = clock_type::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(finish - start).count() / (double(reps) * double(n)));
    }
    return best;
}

// --add: Numetron's in-place limb add/sub against GMP's mpn_add_n/mpn_sub_n (same in-place
// form, rp == s1p), per limb -- isolates the linear primitives Karatsuba/Toom interpolation
// is built from.
void run_add_bench()
{
    static constexpr size_t sizes[] = { 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
    std::mt19937_64 rng{ 0xADD5EEDULL };
    g_section = "add";

    std::cout << "In-place add/sub, ns per limb (best of 5)\n\n"
              << std::right
              << std::setw(8) << "limbs"
              << std::setw(14) << "uadd_inplace" << std::setw(12) << "mpn_add_n" << std::setw(10) << "ratio"
              << std::setw(14) << "usub_inplace" << std::setw(12) << "mpn_sub_n" << std::setw(10) << "ratio"
              << "\n";

    for (size_t n : sizes) {
        std::vector<std::uint64_t> u(n), v(n);
        for (auto& x : u) x = rng();
        for (auto& x : v) x = rng();
        std::vector<mp_limb_t> gu(u.begin(), u.end()), gv(v.begin(), v.end());
        const auto gn = static_cast<mp_size_t>(n);

        double add = best_ns_per_limb(n, [&] { g_sink ^= numetron::limb_arithmetic::uadd_inplace(u.data(), v.data(), v.data() + n); });
        double gadd = best_ns_per_limb(n, [&] { g_sink ^= mpn_add_n(gu.data(), gu.data(), gv.data(), gn); });
        double sub = best_ns_per_limb(n, [&] { g_sink ^= numetron::limb_arithmetic::usub_inplace(u.data(), v.data(), v.data() + n); });
        double gsub = best_ns_per_limb(n, [&] { g_sink ^= mpn_sub_n(gu.data(), gu.data(), gv.data(), gn); });

        const std::string shape = std::to_string(n);
        csv_row(shape, "add", add);
        csv_row(shape, "gmp_add", gadd);
        csv_row(shape, "sub", sub);
        csv_row(shape, "gmp_sub", gsub);

        std::cout << std::setw(8) << n << std::fixed << std::setprecision(3)
                  << std::setw(14) << add << std::setw(12) << gadd << std::setw(10) << std::setprecision(2) << (add / gadd)
                  << std::setw(14) << std::setprecision(3) << sub << std::setw(12) << gsub << std::setw(10) << std::setprecision(2) << (sub / gsub)
                  << "\n";
    }
    std::cout << "\nratio: numetron / gmp (lower is better, 1.00 = parity)\n"
              << "(sink: " << g_sink << ")\n";
}

// --small: every size 1..32 rather than the powers of two of the other tables, times in ns --
// the range below and around basecase_limit(), where the path from basic_integer to the basecase
// kernel (inlined calls, the kernel's dispatch) is a large part of the cost. More attempts than
// the other tables: differences of a fraction of a nanosecond are what it is for.
void run_small_bench()
{
    static constexpr size_t max_limbs = 32;
    static constexpr size_t samples = 6;
    static constexpr int small_attempts = 5;
    std::mt19937_64 rng{ 0x5A11EEDULL }; // own seed: the same operands whatever else runs

    auto best_of = [](auto&& timed) {
        double best = std::numeric_limits<double>::infinity();
        for (int attempt = 0; attempt < small_attempts; ++attempt) best = std::min(best, timed());
        return best;
    };

    // products: plain operator*, assign_mul into a reused result, GMP's mpz_mul
    auto mul_table = [&](char const* section, char const* title, auto&& shapes) {
        g_section = section;
        std::cout << "\n" << title << "\n\n" << std::right
                  << std::setw(10) << "un x vn" << std::setw(14) << "plain(ns)" << std::setw(14) << "reuse(ns)"
                  << std::setw(14) << "gmp(ns)" << std::setw(12) << "gmp/reuse" << "\n";
        for (auto [un, vn] : shapes) {
            auto operands = make_operands(rng, un, vn, samples);
            verify_against_gmp(operands);
            const int repeats = repeats_for(un, vn);
            const double plain = best_of([&] { return time_numetron_mul(operands, repeats); });
            const double reuse = best_of([&] { return time_numetron_mul_reuse(operands, repeats); });
            const double gmp = best_of([&] { return time_gmp_mul(operands, repeats); });
            const std::string shape = un == vn ? std::to_string(un) : std::to_string(un) + "x" + std::to_string(vn);
            csv_row(shape, "plain", plain);
            csv_row(shape, "reuse", reuse);
            csv_row(shape, "gmp", gmp);
            std::cout << std::setw(10) << shape << std::fixed << std::setprecision(2)
                      << std::setw(14) << plain << std::setw(14) << reuse << std::setw(14) << gmp
                      << std::setw(12) << (gmp / reuse) << "\n";
        }
    };

    std::vector<std::pair<size_t, size_t>> square_shapes, rect_shapes;
    for (size_t n = 1; n <= max_limbs; ++n) square_shapes.emplace_back(n, n);
    // un x vn with a short vn: the fixed-size kernels (3..4 x 1..3) and the rows of the basecase
    for (size_t un = 3; un <= 16; ++un) {
        for (size_t vn = 1; vn <= 4 && vn < un; ++vn) rect_shapes.emplace_back(un, vn);
    }
    mul_table("small_mul", "small products (un == vn)", square_shapes);
    mul_table("small_rect", "small products (un > vn)", rect_shapes);

    g_section = "small_sqr";
    std::cout << "\nsmall squares (assign_mul(u, u) into a reused result vs mpz_mul(r, u, u))\n\n" << std::right
              << std::setw(10) << "limbs" << std::setw(14) << "sqr(ns)" << std::setw(14) << "gmp sqr(ns)"
              << std::setw(12) << "gmp/sqr" << "\n";
    for (size_t n = 1; n <= max_limbs; ++n) {
        auto operands = make_operands(rng, n, n, samples);
        verify_sqr_against_gmp(n, operands);
        const int repeats = repeats_for(n);
        const double sqr = best_of([&] { return time_numetron_sqr_reuse(operands, repeats); });
        const double gsqr = best_of([&] { return time_gmp_sqr(operands, repeats); });
        csv_row(std::to_string(n), "sqr", sqr);
        csv_row(std::to_string(n), "gmp_sqr", gsqr);
        std::cout << std::setw(10) << n << std::fixed << std::setprecision(2)
                  << std::setw(14) << sqr << std::setw(14) << gsqr << std::setw(12) << (gsqr / sqr) << "\n";
    }

    // in-place add/sub per call (not per limb as --add): the inline kernel below
    // asm_add_sub_n_min_limbs, the out-of-line one from there
    g_section = "small_add";
    std::cout << "\nsmall in-place add/sub, ns per call\n\n" << std::right
              << std::setw(10) << "limbs" << std::setw(14) << "uadd(ns)" << std::setw(14) << "mpn_add(ns)"
              << std::setw(14) << "usub(ns)" << std::setw(14) << "mpn_sub(ns)" << "\n";
    for (size_t n = 1; n <= max_limbs; ++n) {
        std::vector<std::uint64_t> u(n), v(n);
        for (auto& x : u) x = rng();
        for (auto& x : v) x = rng();
        std::vector<mp_limb_t> gu(u.begin(), u.end()), gv(v.begin(), v.end());
        const auto gn = static_cast<mp_size_t>(n);
        const double dn = double(n);

        const double add = dn * best_ns_per_limb(n, [&] { g_sink ^= numetron::limb_arithmetic::uadd_inplace(u.data(), v.data(), v.data() + n); });
        const double gadd = dn * best_ns_per_limb(n, [&] { g_sink ^= mpn_add_n(gu.data(), gu.data(), gv.data(), gn); });
        const double sub = dn * best_ns_per_limb(n, [&] { g_sink ^= numetron::limb_arithmetic::usub_inplace(u.data(), v.data(), v.data() + n); });
        const double gsub = dn * best_ns_per_limb(n, [&] { g_sink ^= mpn_sub_n(gu.data(), gu.data(), gv.data(), gn); });
        const std::string shape = std::to_string(n);
        csv_row(shape, "add", add);
        csv_row(shape, "gmp_add", gadd);
        csv_row(shape, "sub", sub);
        csv_row(shape, "gmp_sub", gsub);
        std::cout << std::setw(10) << n << std::fixed << std::setprecision(2)
                  << std::setw(14) << add << std::setw(14) << gadd << std::setw(14) << sub << std::setw(14) << gsub << "\n";
    }
}

// Which implementations this build runs (numetron/config/implementation.hpp) and the thresholds
// in effect, so bench outputs of different builds can be told apart.
void print_configuration(std::ostream& os)
{
    namespace la = numetron::limb_arithmetic;
    os << "implementations: karatsuba " << numetron::config::karatsuba_impl_name
              << ", toom3 " << numetron::config::toom3_impl_name
              << ", fft " << numetron::config::fft_impl_name
              << ", mul_basecase " << numetron::config::mul_basecase_name << "\n"
              << "thresholds (limbs): karatsuba " << la::karatsuba_threshold()
              << ", toom3 " << la::toom3_threshold()
              << ", toom4 " << la::toom4_threshold()
              << ", toom6h " << la::toom6h_threshold()
              << ", toom8h " << la::toom8h_threshold()
              << ", toom32 " << la::toom32_threshold()
              << ", toom42 " << la::toom42_threshold()
              << ", toom76 " << la::toom76_threshold()
              << ", toom63 " << la::toom63_threshold()
              << ", toom98 " << la::toom98_threshold()
              << ", toom107 " << la::toom107_threshold()
              << ", toom116 " << la::toom116_threshold()
              << ", toom54 " << la::toom54_threshold()
              << ", toom53 " << la::toom53_threshold()
              << ", toom43 " << la::toom43_threshold()
              << ", slicing " << la::slicing_threshold()
              << ", fft ";
    if (la::fft_threshold() == ~size_t{ 0 }) os << "off\n";
    else os << la::fft_threshold() << "\n";
    os << "squares: basecase " << la::sqr_basecase_threshold()
              << ", karatsuba " << la::sqr_karatsuba_threshold()
              << ", toom3 " << la::sqr_toom3_threshold()
              << ", toom4 " << la::sqr_toom4_threshold()
              << ", toom6h " << la::sqr_toom6h_threshold()
              << ", toom8h " << la::sqr_toom8h_threshold()
              << ", fft ";
    if (la::sqr_fft_threshold() == ~size_t{ 0 }) os << "off\n";
    else os << la::sqr_fft_threshold() << "\n";
}

// The configuration on stdout and, with --csv, as '#' lines at the top of the file.
void report_configuration()
{
    std::ostringstream text;
    print_configuration(text);
    std::cout << text.str();
    if (g_csv.is_open()) {
        std::istringstream lines{ text.str() };
        for (std::string line; std::getline(lines, line);) g_csv << "# " << line << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    // Flush after every output operation: when stdout is a pipe (Docker, `| tee`) it is fully
    // buffered otherwise, and rows would only appear in large chunks. Output is never timed.
    std::cout << std::unitbuf;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--csv=", 0) != 0) continue;
        g_csv.open(arg.substr(6));
        if (!g_csv) {
            std::cerr << "can't open " << arg.substr(6) << " for writing\n";
            return 1;
        }
        g_csv << std::setprecision(9);
    }

    for (int i = 1; i < argc; ++i) {
        if (std::string{ argv[i] } == "--add") {
            report_configuration();
            run_add_bench();
            return 0;
        }
    }

    static constexpr size_t limb_counts[] = {
        1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384
    };
    static constexpr size_t samples_per_tier = 6;

    // --tune[=N]: retune the multiplication thresholds on this machine before benchmarking,
    // taking the best of N samples per probe (default: mul_tuning_options' default).
    // --tune-unbalanced[=N]: the same for the unbalanced thresholds alone (Toom-3/2 .. Toom-8.5
    // 11 x 6; the balanced ones stay the defaults).
    // --tune-squares[=N]: the same for the squaring thresholds alone.
    // --trace (with --tune): print every probe: size, lower / higher algorithm time, ratio.
    bool trace = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string{ argv[i] } == "--trace") trace = true;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--tune", 0) != 0) continue;
        numetron::limb_arithmetic::mul_tuning_options opts;
        // --tune-unbalanced[=N]: only the unbalanced thresholds, against the balanced defaults
        if (arg.rfind("--tune-unbalanced", 0) == 0) opts.tune_balanced = false;
        // --tune-squares[=N]: only the squaring thresholds
        if (arg.rfind("--tune-squares", 0) == 0) opts.squares_only = true;
        if (auto eq = arg.find('='); eq != std::string::npos) {
            opts.samples = static_cast<unsigned>(std::stoul(arg.substr(eq + 1)));
        }
        if (trace) {
            opts.trace = [](char const* algorithm, size_t n, double lower_ns, double higher_ns) {
                const auto flags = std::cout.flags();
                const auto precision = std::cout.precision();
                NUMETRON_SCOPE_EXIT([&] { std::cout.flags(flags); std::cout.precision(precision); });
                std::cout << "  " << std::left << std::setw(10) << algorithm << std::right
                          << std::setw(6) << n
                          << std::setw(14) << std::fixed << std::setprecision(0) << lower_ns
                          << std::setw(14) << higher_ns
                          << std::setw(8) << std::setprecision(3) << higher_ns / lower_ns << "\n";
            };
            opts.trace_candidate = [](char const* algorithm, size_t threshold, double relative_time, bool chosen) {
                const auto flags = std::cout.flags();
                const auto precision = std::cout.precision();
                NUMETRON_SCOPE_EXIT([&] { std::cout.flags(flags); std::cout.precision(precision); });
                std::cout << "  " << std::left << std::setw(10) << algorithm << std::right
                          << " candidate " << std::setw(6) << threshold
                          << "  full-product time vs off: " << std::fixed << std::setprecision(3) << relative_time
                          << (chosen ? "  <- chosen" : "") << "\n";
            };
            std::cout << "  algorithm      n      lower(ns)    higher(ns)   ratio\n";
        }
        auto before_k = numetron::limb_arithmetic::karatsuba_threshold();
        auto before_t = numetron::limb_arithmetic::toom3_threshold();
        auto before_t4 = numetron::limb_arithmetic::toom4_threshold();
        auto before_t6 = numetron::limb_arithmetic::toom6h_threshold();
        auto before_t8 = numetron::limb_arithmetic::toom8h_threshold();
        auto before_fft = numetron::limb_arithmetic::fft_threshold();
        auto before_t32 = numetron::limb_arithmetic::toom32_threshold();
        auto before_t42 = numetron::limb_arithmetic::toom42_threshold();
        auto before_t76 = numetron::limb_arithmetic::toom76_threshold();
        auto before_t63 = numetron::limb_arithmetic::toom63_threshold();
        auto before_t98 = numetron::limb_arithmetic::toom98_threshold();
        auto before_t107 = numetron::limb_arithmetic::toom107_threshold();
        auto before_t116 = numetron::limb_arithmetic::toom116_threshold();
        auto before_t54 = numetron::limb_arithmetic::toom54_threshold();
        auto before_t53 = numetron::limb_arithmetic::toom53_threshold();
        auto before_t43 = numetron::limb_arithmetic::toom43_threshold();
        auto before_sk = numetron::limb_arithmetic::sqr_karatsuba_threshold();
        auto before_s3 = numetron::limb_arithmetic::sqr_toom3_threshold();
        auto before_s4 = numetron::limb_arithmetic::sqr_toom4_threshold();
        auto before_s6 = numetron::limb_arithmetic::sqr_toom6h_threshold();
        auto before_s8 = numetron::limb_arithmetic::sqr_toom8h_threshold();
        auto before_sf = numetron::limb_arithmetic::sqr_fft_threshold();
        auto tuned = numetron::limb_arithmetic::tune_mul_thresholds(opts);
        auto show = [](size_t t) { return t == ~size_t{ 0 } ? std::string{ "off" } : std::to_string(t); };
        constexpr bool fill_bound = numetron::limb_arithmetic::detail::fft_threshold_is_fill_bound;
        std::cout << "tuned thresholds (limbs):\n"
                  << "  karatsuba: " << before_k << " -> " << tuned.karatsuba_threshold
                  << (tuned.karatsuba_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom3:     " << before_t << " -> " << tuned.toom3_threshold
                  << (tuned.toom3_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom4:     " << before_t4 << " -> " << tuned.toom4_threshold
                  << (tuned.toom4_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom6h:    " << before_t6 << " -> " << tuned.toom6h_threshold
                  << (tuned.toom6h_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom8h:    " << before_t8 << " -> " << tuned.toom8h_threshold
                  << (tuned.toom8h_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom32:    " << before_t32 << " -> " << tuned.toom32_threshold
                  << (tuned.toom32_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom42:    " << before_t42 << " -> " << tuned.toom42_threshold
                  << (tuned.toom42_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom76:    " << before_t76 << " -> " << tuned.toom76_threshold
                  << (tuned.toom76_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom63:    " << before_t63 << " -> " << tuned.toom63_threshold
                  << (tuned.toom63_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom98:    " << before_t98 << " -> " << tuned.toom98_threshold
                  << (tuned.toom98_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom107:   " << before_t107 << " -> " << tuned.toom107_threshold
                  << (tuned.toom107_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom116:   " << before_t116 << " -> " << tuned.toom116_threshold
                  << (tuned.toom116_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom54:    " << before_t54 << " -> " << tuned.toom54_threshold
                  << (tuned.toom54_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom53:    " << before_t53 << " -> " << tuned.toom53_threshold
                  << (tuned.toom53_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  toom43:    " << before_t43 << " -> " << tuned.toom43_threshold
                  << (tuned.toom43_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  sqr kara:   " << before_sk << " -> " << tuned.sqr_karatsuba_threshold
                  << (tuned.sqr_karatsuba_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  sqr toom3:  " << before_s3 << " -> " << tuned.sqr_toom3_threshold
                  << (tuned.sqr_toom3_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  sqr toom4:  " << before_s4 << " -> " << tuned.sqr_toom4_threshold
                  << (tuned.sqr_toom4_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  sqr toom6h: " << before_s6 << " -> " << tuned.sqr_toom6h_threshold
                  << (tuned.sqr_toom6h_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  sqr toom8h: " << before_s8 << " -> " << tuned.sqr_toom8h_threshold
                  << (tuned.sqr_toom8h_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  sqr fft:    " << show(before_sf) << " -> " << show(tuned.sqr_fft_threshold)
                  << (fill_bound ? " (the fill rule's lower bound, kept)" : tuned.sqr_fft_found ? "" : " (no crossover found, kept)") << "\n"
                  << "  fft:       " << show(before_fft) << " -> " << show(tuned.fft_threshold)
                  << (fill_bound ? " (the fill rule's lower bound, kept)" : tuned.fft_found ? "" : " (no crossover found, kept)") << "\n\n";
    }

    // --balanced: un == vn over limb_counts; --unbalanced: un > vn over unbalanced_v_limbs x
    // unbalanced_ratios. Neither given: both, balanced first. --large: the unbalanced shapes over
    // unbalanced_large_v_limbs (the FFT range) instead.
    // --sqr: squares u * u over limb_counts (alone unless --balanced / --unbalanced are given too).
    // --small: run_small_bench() (alone unless other modes are given too).
    bool balanced = false, unbalanced = false, large = false, sqr = false, small = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--balanced") balanced = true;
        else if (arg == "--unbalanced") unbalanced = true;
        else if (arg == "--large") large = true;
        else if (arg == "--sqr") sqr = true;
        else if (arg == "--small") small = true;
    }
    if (!balanced && !unbalanced && !sqr && !small) balanced = unbalanced = true;

    // --max-limbs=N: --balanced and --sqr only up to N limbs (the large sizes take most of a run)
    size_t max_limbs = ~size_t{ 0 };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--max-limbs=", 0) == 0) max_limbs = std::stoul(arg.substr(12));
    }

    std::mt19937_64 rng{ 0x5EED1234ULL };

    std::cout << "Numetron vs GMP multiplication benchmark\n";
    std::cout << "(" << samples_per_tier << " random operand pairs per tier, best of " << attempts << " attempts)\n";
    report_configuration();

    if (small) run_small_bench();

    if (balanced) {
        g_section = "balanced";
        std::cout << "\nbalanced (un == vn)\n\n";
        std::cout << std::right
                   << std::setw(10) << "limbs"
                   << std::setw(12) << "bits"
                   << std::setw(16) << "numetron(us)"
                   << std::setw(16) << "reuse(us)"
                   << std::setw(16) << "gmp(us)"
                   << std::setw(14) << "gmp/reuse"
                   << "\n";

        // "1*" isolates the raw single-limb multiply from the allocation the plain "1" row below
        // pays for its (almost always 2-limb) product -- see make_small_operands() for why.
        run_tier("1*", "<=62", make_small_operands(rng, samples_per_tier), repeats_for(1));

        for (size_t limb_count : limb_counts) {
            if (limb_count > max_limbs) break;
            auto operands = make_operands(rng, limb_count, limb_count, samples_per_tier);
            run_tier(std::to_string(limb_count), std::to_string(limb_count * 64), operands, repeats_for(limb_count));
        }
    }

    if (unbalanced) {
        // vn from below the Karatsuba threshold to below the FFT one (where the FFT takes any
        // shape); un/vn from the Toom-3 range (1.5, 2) to far past it, where only slicing u into
        // vn-sized pieces keeps the product sub-quadratic.
        static constexpr size_t unbalanced_v_limbs[] = { 16, 24, 32, 64, 128, 256, 512, 1024, 2048 };
        // --large: from the FFT threshold up (the FFT takes any shape there)
        static constexpr size_t unbalanced_large_v_limbs[] = { 4096, 8192, 16384 };
        static constexpr double unbalanced_ratios[] = { 1.25, 1.35, 1.5, 1.75, 1.9, 2, 3, 4, 8, 16, 32 };

        std::mt19937_64 urng{ 0x0B1A5EEDULL }; // own seed: the same operands with or without --balanced
        g_section = large ? "unbalanced_large" : "unbalanced";

        std::cout << "\nunbalanced (un > vn)\n\n";
        std::cout << std::right
                   << std::setw(14) << "un x vn"
                   << std::setw(12) << "un/vn"
                   << std::setw(16) << "reuse(us)"
                   << std::setw(16) << "gmp(us)"
                   << std::setw(14) << "gmp/reuse"
                   << "\n";

        for (size_t vn : large ? std::span<const size_t>{ unbalanced_large_v_limbs } : std::span<const size_t>{ unbalanced_v_limbs }) {
            for (double ratio : unbalanced_ratios) {
                const auto un = static_cast<size_t>(double(vn) * ratio + 0.5);
                std::ostringstream ratio_label;
                ratio_label << ratio;
                auto operands = make_operands(urng, un, vn, samples_per_tier);
                // A quarter of the balanced budget: 99 shapes, and the product sizes grow with un/vn.
                run_tier(std::to_string(un) + "x" + std::to_string(vn), ratio_label.str(), operands, repeats_for(un, vn, 1.0e8), 14, false);
            }
        }
    }

    if (sqr) {
        std::mt19937_64 srng{ 0x5A0A5EEDULL }; // own seed: the same operands whatever else runs
        g_section = "sqr";
        std::cout << "\nsquares (u * u, the same object on both sides; reuse vs GMP's mpz_mul(r, u, u))\n\n"
                  << std::right
                  << std::setw(10) << "limbs"
                  << std::setw(14) << "sqr(us)"
                  << std::setw(14) << "gmp sqr(us)"
                  << std::setw(12) << "gmp/sqr"
                  << std::setw(14) << "mul(us)"
                  << std::setw(14) << "gmp mul(us)"
                  << std::setw(12) << "gmp mul/sqr"
                  << std::setw(12) << "mul/sqr"
                  << "\n";
        for (size_t limb_count : limb_counts) {
            if (limb_count > max_limbs) break;
            run_sqr_tier(limb_count, make_operands(srng, limb_count, limb_count, samples_per_tier), repeats_for(limb_count));
        }
    }

    std::cout << "\n";
    if (sqr) std::cout << "sqr / mul: assign_mul(u, u) / assign_mul(u, v) into a reused result; gmp: mpz_mul(r, u, u) / mpz_mul(r, u, v)\n"
                       << "gmp mul/sqr: what squaring buys GMP; mul/sqr: what it buys numetron (1.00: nothing)\n";
    if (balanced) std::cout << "* both factors <= 31 bits: product guaranteed to fit in 1 limb, no result allocation\n";
    if (balanced) std::cout << "numetron(us): plain operator* (builds a fresh result every call, like u * v)\n";
    std::cout << "reuse(us):    assign_mul() into one result reused across the whole run, like GMP's mpz_mul(r, u, v)\n";
    std::cout << "(sink: " << g_sink << ")\n"; // keeps g_sink itself from looking unused to -Wunused warnings
    return 0;
}
