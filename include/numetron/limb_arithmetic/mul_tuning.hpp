// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <chrono>
#include <random>
#include <vector>
#include <limits>
#include <cstdint>
#include <optional>
#include <algorithm>

#include "numetron/limb_arithmetic.hpp"
#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/stack_allocator.hpp"

namespace numetron::limb_arithmetic {

struct mul_tuning_options
{
    // Timed samples per measured (algorithm, size) point; the fastest is kept, so more samples
    // filter out more interference (interrupts, frequency changes) at the cost of a longer run.
    unsigned samples = 5;

    // Minimum duration of one sample: the multiplication is repeated in a batch sized so that a
    // sample lasts at least this long, keeping timer resolution and clock-read overhead out of
    // small-size measurements.
    std::chrono::nanoseconds min_sample_time = std::chrono::microseconds{ 200 };

    // A crossover is accepted only once the higher algorithm wins at this many consecutive
    // probed sizes, so a single noisy point can't set a threshold.
    unsigned confirmations = 3;

    // The higher algorithm counts as winning only when it is faster by at least this fraction.
    // Near a crossover the two curves are nearly flat against each other, so switching a bit
    // late costs almost nothing, while switching early on a marginal (or biased) win can cost
    // noticeably at those sizes.
    double min_gain = 0.02;

    // Distinct operand pairs cycled through within each timed batch. Karatsuba/Toom branch on
    // the data (which half is larger, sign of the middle product, carries); timing one fixed
    // pair lets the branch predictor learn all of that and flatters them against basecase,
    // whose branches depend only on sizes. Capped per size so the operands of a batch stay
    // within ~64 KiB and a small-size probe isn't turned into a cache-miss benchmark.
    size_t operand_pairs = 16;

    // Upper ends of the searched ranges, in limbs. If no crossover is confirmed within a range,
    // that threshold is left as it was and reported as not found.
    size_t karatsuba_max = 256;
    size_t toom3_max = 2048;

    // Install the found thresholds; when false they are only returned and the previous values
    // are restored.
    bool apply = true;
};

struct mul_tuning_result
{
    size_t karatsuba_threshold;
    size_t toom3_threshold;
    bool karatsuba_found;
    bool toom3_found;
};

namespace mul_tuning_detail {

using limb_type = std::uint64_t;
using clock_type = std::chrono::steady_clock;

class workload
{
public:
    workload(size_t max_limbs, size_t max_pairs)
        : max_pairs_((std::max)(max_pairs, size_t{ 1 }))
        , u_(max_pairs_ * max_limbs), v_(max_pairs_ * max_limbs), r_(2 * max_limbs)
    {
        std::mt19937_64 rng{ 0x6E756D6574726F6EULL };
        // Every limb non-zero, so umul_dispatch's leading-zero stripping never shortens an
        // operand and an n-limb probe really multiplies n x n limbs.
        for (auto& x : u_) x = rng() | 1;
        for (auto& x : v_) x = rng() | 1;
    }

    double batch_ns(size_t n, size_t reps)
    {
        // Pairs are packed back to back (pair k at offset k*n), not at a fixed max_limbs stride:
        // a power-of-two-ish stride would map every pair onto the same L1 sets.
        constexpr size_t operand_bytes_budget = 64 * 1024;
        const size_t pairs = (std::clamp)(operand_bytes_budget / (2 * n * sizeof(limb_type)), size_t{ 1 }, max_pairs_);

        size_t k = 0;
        auto start = clock_type::now();
        for (size_t i = 0; i < reps; ++i) {
            umul_dispatch(u_.data() + k * n, n, v_.data() + k * n, n, r_.data(), numetron::detail::stack_allocator<limb_type>{});
            if (++k == pairs) k = 0;
        }
        auto finish = clock_type::now();
        sink_ = r_[0];
        return std::chrono::duration<double, std::nano>(finish - start).count();
    }

    // Best per-multiplication time of `samples` batches, each batch sized to last at least
    // min_sample_time, with whatever thresholds are currently installed.
    double measure_ns(size_t n, mul_tuning_options const& opts)
    {
        const double min_ns = std::chrono::duration<double, std::nano>(opts.min_sample_time).count();

        batch_ns(n, 1); // warm-up: caches, stack allocator slabs

        size_t reps = 1;
        double elapsed = batch_ns(n, reps);
        while (elapsed < min_ns && reps < (size_t{ 1 } << 30)) {
            reps *= 2;
            elapsed = batch_ns(n, reps);
        }

        double best = elapsed / static_cast<double>(reps);
        for (unsigned i = 1; i < opts.samples; ++i) {
            best = (std::min)(best, batch_ns(n, reps) / static_cast<double>(reps));
        }
        return best;
    }

private:
    size_t max_pairs_;
    std::vector<limb_type> u_, v_, r_;
    volatile limb_type sink_ = 0;
};

// Probes sizes lo..hi in steps of ~1/16 and returns the first size of the first run of
// `confirmations` consecutive probes at which higher_wins(n) holds.
template <typename HigherWinsF>
std::optional<size_t> find_crossover(size_t lo, size_t hi, unsigned confirmations, HigherWinsF&& higher_wins)
{
    confirmations = (std::max)(confirmations, 1u);
    size_t streak_start = 0;
    unsigned streak = 0;
    for (size_t n = lo; n <= hi; n += (std::max)(size_t{ 1 }, n / 16)) {
        if (higher_wins(n)) {
            if (!streak++) streak_start = n;
            if (streak >= confirmations) return streak_start;
        } else {
            streak = 0;
        }
    }
    return std::nullopt;
}

}

// Measures where each multiplication algorithm starts beating the one below it on this machine
// and (by default) installs the results as the runtime thresholds. Karatsuba is tuned first
// against basecase, then Toom-3 against whatever the tuned Karatsuba threshold selects below it.
// Each size is compared one level deep, the way GMP's tuneup does it: with the threshold set
// to n the top-level n x n product uses the higher algorithm while its sub-products (smaller
// than n) fall back to the lower one, versus the threshold at n+1 where the whole product uses
// the lower one.
//
// While it runs, the global thresholds are temporarily forced to other values. Multiplications
// on other threads still produce correct results but may run at suboptimal speed, and their
// load disturbs the measurements -- call this at startup or from an otherwise idle process.
inline mul_tuning_result tune_mul_thresholds(mul_tuning_options const& opts = {})
{
    const size_t prev_karatsuba = karatsuba_threshold();
    const size_t prev_toom3 = toom3_threshold();

    bool committed = false;
    NUMETRON_SCOPE_EXIT([&] {
        if (!committed) {
            set_karatsuba_threshold(prev_karatsuba);
            set_toom3_threshold(prev_toom3);
        }
    });

    mul_tuning_detail::workload work{ (std::max)(opts.karatsuba_max, opts.toom3_max), opts.operand_pairs };
    mul_tuning_options sample_opts = opts;
    sample_opts.samples = (std::max)(opts.samples, 1u);

    const double win_factor = 1.0 - (std::clamp)(opts.min_gain, 0.0, 0.99);

    mul_tuning_result result{};

    set_toom3_threshold((std::numeric_limits<size_t>::max)());
    auto karatsuba = mul_tuning_detail::find_crossover(min_karatsuba_threshold, opts.karatsuba_max, opts.confirmations,
        [&](size_t n) {
            set_karatsuba_threshold(n + 1);
            const double lower = work.measure_ns(n, sample_opts);
            set_karatsuba_threshold(n);
            const double higher = work.measure_ns(n, sample_opts);
            return higher < lower * win_factor;
        });
    result.karatsuba_found = karatsuba.has_value();
    result.karatsuba_threshold = karatsuba.value_or(prev_karatsuba);
    set_karatsuba_threshold(result.karatsuba_threshold);

    auto toom3 = mul_tuning_detail::find_crossover((std::max)(min_toom3_threshold, result.karatsuba_threshold), opts.toom3_max, opts.confirmations,
        [&](size_t n) {
            set_toom3_threshold(n + 1);
            const double lower = work.measure_ns(n, sample_opts);
            set_toom3_threshold(n);
            const double higher = work.measure_ns(n, sample_opts);
            return higher < lower * win_factor;
        });
    result.toom3_found = toom3.has_value();
    result.toom3_threshold = toom3.value_or(prev_toom3);

    if (opts.apply) {
        set_karatsuba_threshold(result.karatsuba_threshold);
        set_toom3_threshold(result.toom3_threshold);
        committed = true;
    }
    return result;
}

}
