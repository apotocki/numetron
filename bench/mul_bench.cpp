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

#ifdef _WIN32
#   pragma warning(disable : 4244 4146)
#endif
#include "gmp.h"

#define NUMETRON_KARATSUBA_ASM

#include "numetron/basic_integer.hpp"
#include "numetron/limb_arithmetic/mul_tuning.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

// Written to from every timed multiplication below, so the optimizer can't prove the
// result is dead and drop the call.
volatile std::uint64_t g_sink = 0;

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

std::vector<operand_pair> make_operands(std::mt19937_64& rng, size_t limb_count, size_t count)
{
    std::vector<operand_pair> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back({ random_hex_operand(rng, limb_count), random_hex_operand(rng, limb_count) });
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
            std::cerr << "MISMATCH for " << (o.u_hex.size() / 16) << "-limb operands:\n"
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
int repeats_for(size_t limb_count)
{
    double budget = 4.0e8;
    double estimate = budget / (double(limb_count) * double(limb_count) + 1.0);
    int repeats = std::clamp(int(estimate), 1, 20000);

    // At <=16 limbs a single multiply takes only a few nanoseconds, close enough to the clock's
    // resolution/call overhead that the measurement noise floor starts to matter -- run 10x more
    // repeats there so the timed interval stays comfortably above it.
    if (limb_count <= 16) repeats *= 10;

    return repeats;
}

double ns_to_us(double ns) { return ns / 1000.0; }

static constexpr int attempts = 3;

void run_tier(std::string const& limbs_label, std::string const& bits_label, std::vector<operand_pair> const& operands, int repeats)
{
    verify_against_gmp(operands);

    double best_numetron_ns = std::numeric_limits<double>::infinity();
    double best_reuse_ns = std::numeric_limits<double>::infinity();
    double best_gmp_ns = std::numeric_limits<double>::infinity();
    for (int attempt = 0; attempt < attempts; ++attempt) {
        best_numetron_ns = std::min(best_numetron_ns, time_numetron_mul(operands, repeats));
        best_reuse_ns = std::min(best_reuse_ns, time_numetron_mul_reuse(operands, repeats));
        best_gmp_ns = std::min(best_gmp_ns, time_gmp_mul(operands, repeats));
    }

    std::cout << std::setw(10) << limbs_label
               << std::setw(12) << bits_label
               << std::setw(16) << std::fixed << std::setprecision(3) << ns_to_us(best_numetron_ns)
               << std::setw(16) << std::fixed << std::setprecision(3) << ns_to_us(best_reuse_ns)
               << std::setw(16) << std::fixed << std::setprecision(3) << ns_to_us(best_gmp_ns)
               << std::setw(14) << std::fixed << std::setprecision(2) << (best_gmp_ns / best_reuse_ns)
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

        std::cout << std::setw(8) << n << std::fixed << std::setprecision(3)
                  << std::setw(14) << add << std::setw(12) << gadd << std::setw(10) << std::setprecision(2) << (add / gadd)
                  << std::setw(14) << std::setprecision(3) << sub << std::setw(12) << gsub << std::setw(10) << std::setprecision(2) << (sub / gsub)
                  << "\n";
    }
    std::cout << "\nratio: numetron / gmp (lower is better, 1.00 = parity)\n"
              << "(sink: " << g_sink << ")\n";
}

// Which implementations this build runs (numetron/config/implementation.hpp) and the thresholds
// in effect, so bench outputs of different builds can be told apart.
void print_configuration()
{
    namespace la = numetron::limb_arithmetic;
    std::cout << "implementations: karatsuba " << numetron::config::karatsuba_impl_name
              << ", toom3 " << numetron::config::toom3_impl_name
              << ", mul_basecase " << numetron::config::mul_basecase_name << "\n"
              << "thresholds (limbs): karatsuba " << la::karatsuba_threshold()
              << ", toom3 " << la::toom3_threshold()
              << ", toom4 " << la::toom4_threshold()
              << ", toom6h " << la::toom6h_threshold()
              << ", toom8h " << la::toom8h_threshold() << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        if (std::string{ argv[i] } == "--add") {
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
    // --trace (with --tune): print every probe: size, lower / higher algorithm time, ratio.
    bool trace = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string{ argv[i] } == "--trace") trace = true;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--tune", 0) != 0) continue;
        numetron::limb_arithmetic::mul_tuning_options opts;
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
        auto tuned = numetron::limb_arithmetic::tune_mul_thresholds(opts);
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
                  << (tuned.toom8h_found ? "" : " (no crossover found, kept)") << "\n\n";
    }

    std::mt19937_64 rng{ 0x5EED1234ULL };

    std::cout << "Numetron vs GMP multiplication benchmark\n";
    std::cout << "(" << samples_per_tier << " random operand pairs per tier, best of " << attempts << " attempts)\n";
    print_configuration();
    std::cout << "\n";
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
        auto operands = make_operands(rng, limb_count, samples_per_tier);
        run_tier(std::to_string(limb_count), std::to_string(limb_count * 64), operands, repeats_for(limb_count));
    }

    std::cout << "\n* both factors <= 31 bits: product guaranteed to fit in 1 limb, no result allocation\n";
    std::cout << "numetron(us): plain operator* (builds a fresh result every call, like u * v)\n";
    std::cout << "reuse(us):    assign_mul() into one result reused across the whole run, like GMP's mpz_mul(r, u, v)\n";
    std::cout << "(sink: " << g_sink << ")\n"; // keeps g_sink itself from looking unused to -Wunused warnings
    return 0;
}
