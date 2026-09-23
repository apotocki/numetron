// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <limits>
#include <cstdint>
#include <optional>
#include <algorithm>
#include <functional>

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

    // Required average advantage of the higher algorithm in the one-level scan's estimate of the
    // best threshold (see mul_tuning_detail::tune_threshold()): every probe above the threshold
    // is charged this fraction extra.
    double min_gain = 0.01;

    // Candidate thresholds whose validated full-product time (geometric mean over the validation
    // sizes) is within this fraction of the best count as tied; the largest of them is chosen.
    double tie_tolerance = 0.003;

    // Optional progress report, called for every size of the one-level scan with the algorithm
    // being tuned ("karatsuba", "toom3", "toom4"), the size, and the measured per-multiplication
    // times with the lower and the higher algorithm at the top level.
    std::function<void(char const* algorithm, size_t n, double lower_ns, double higher_ns)> trace;

    // Optional report of the validation phase, called for every candidate threshold with the
    // geometric mean, over the validation sizes, of full-product time with that threshold
    // relative to the higher algorithm switched off (< 1: faster), and whether it was chosen.
    std::function<void(char const* algorithm, size_t threshold, double relative_time, bool chosen)> trace_candidate;

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
    size_t toom4_max = 4096;

    // Install the found thresholds; when false they are only returned and the previous values
    // are restored.
    bool apply = true;
};

struct mul_tuning_result
{
    size_t karatsuba_threshold;
    size_t toom3_threshold;
    size_t toom4_threshold;
    bool karatsuba_found;
    bool toom3_found;
    bool toom4_found;
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

    // Best per-multiplication times of n x n products under each of several threshold settings
    // (install(k) installs setting k), `samples` batches per setting, every batch sized to last at
    // least min_sample_time. The settings are measured interleaved, in alternately forward and
    // backward order (ABBA...), so that a drift of the machine's speed during the probe (clock
    // boost, thermal throttling, background load) hits all of them alike instead of biasing the
    // comparison.
    template <typename InstallF>
    std::vector<double> measure_ns(size_t n, size_t settings, mul_tuning_options const& opts, InstallF&& install)
    {
        const double min_ns = std::chrono::duration<double, std::nano>(opts.min_sample_time).count();

        // warm-up of every path: caches, stack allocator slabs
        for (size_t k = 0; k < settings; ++k) {
            install(k);
            batch_ns(n, 1);
        }

        size_t reps = 1;
        double elapsed = batch_ns(n, reps);
        while (elapsed < min_ns && reps < (size_t{ 1 } << 30)) {
            reps *= 2;
            elapsed = batch_ns(n, reps);
        }

        const double r = static_cast<double>(reps);
        std::vector<double> best(settings, std::numeric_limits<double>::infinity());
        const unsigned samples = (std::max)(opts.samples, 1u);
        for (unsigned i = 0; i < samples; ++i) {
            for (size_t j = 0; j < settings; ++j) {
                const size_t k = (i & 1) ? settings - 1 - j : j;
                install(k);
                best[k] = (std::min)(best[k], batch_ns(n, reps) / r);
            }
        }
        return best;
    }

private:
    size_t max_pairs_;
    std::vector<limb_type> u_, v_, r_;
    volatile limb_type sink_ = 0;
};

// Tunes the threshold of one algorithm ("higher") over the one below it ("lower", whatever the
// already installed lower thresholds select), searching lo..hi. Two phases:
//
// 1. Scan, one level deep (like GMP's tuneup): at each size n (steps of ~1/16) the top-level
//    n x n product is timed with the higher algorithm (threshold n) and with the lower one
//    (threshold n + 1), the sub-products taking the lower one either way. The ratio of the two
//    is not a smooth curve: as n grows, the sub-products of the two algorithms (n/3 vs n/2, ...)
//    cross the lower thresholds at different sizes, so the ratio swings around 1 in bands about
//    an octave wide. A single threshold can't follow those swings, and past the first few bands
//    the one-level ratio no longer describes the real product (whose sub-products then use the
//    higher algorithm too). So the scan only proposes candidates: the start of every band where
//    the higher algorithm wins, plus the threshold minimizing the scan's own estimate of the
//    total time, sum over n >= T of log(ratio(n) / (1 - min_gain)).
//
// 2. Validation: each candidate threshold, and "higher algorithm off", is installed in turn and
//    full products (all recursion levels using it) are timed at sizes from the smallest
//    candidate to hi (steps of ~1/8), interleaved per size. The candidate with the least total
//    log time over those sizes wins -- that is the quantity a threshold actually decides.
//
// Returns the threshold, or nothing when leaving the higher algorithm off is best.
template <typename SetThresholdF>
std::optional<size_t> tune_threshold(workload& work, mul_tuning_options const& opts, char const* name,
    SetThresholdF&& set_threshold, size_t lo, size_t hi)
{
    constexpr size_t off = (std::numeric_limits<size_t>::max)();
    constexpr size_t max_candidates = 8;

    // Phase 1: scan.
    std::vector<size_t> sizes;
    std::vector<double> ratios;
    for (size_t n = lo; n <= hi; n += (std::max)(size_t{ 1 }, n / 16)) {
        auto t = work.measure_ns(n, 2, opts, [&](size_t k) { set_threshold(k ? n : n + 1); });
        if (opts.trace) opts.trace(name, n, t[0], t[1]);
        sizes.push_back(n);
        ratios.push_back(t[1] / t[0]);
    }
    if (sizes.empty()) return std::nullopt;

    // suffix sums of the charged log ratios: the scan's estimate of the total for threshold sizes[i]
    const double charge = -std::log(1.0 - (std::clamp)(opts.min_gain, 0.0, 0.99));
    std::vector<double> suffix(sizes.size() + 1, 0.0);
    for (size_t i = sizes.size(); i-- > 0;) suffix[i] = suffix[i + 1] + std::log(ratios[i]) + charge;

    std::vector<size_t> cand; // indices into sizes
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (ratios[i] < 1.0 && (i == 0 || ratios[i - 1] >= 1.0)) cand.push_back(i);
    }
    const size_t best_scan = static_cast<size_t>(std::min_element(suffix.begin(), suffix.end() - 1) - suffix.begin());
    if (suffix[best_scan] < 0.0 && std::find(cand.begin(), cand.end(), best_scan) == cand.end()) cand.push_back(best_scan);
    if (cand.empty()) return std::nullopt;
    if (cand.size() > max_candidates) {
        std::sort(cand.begin(), cand.end(), [&](size_t a, size_t b) { return suffix[a] < suffix[b]; });
        cand.resize(max_candidates);
    }
    std::sort(cand.begin(), cand.end());

    // Phase 2: validation with full products. Setting k < cand.size() is threshold
    // sizes[cand[k]], the last one is "off".
    std::vector<size_t> thresholds;
    for (size_t i : cand) thresholds.push_back(sizes[i]);
    thresholds.push_back(off);

    std::vector<double> score(thresholds.size(), 0.0);
    size_t points = 0;
    for (size_t n = thresholds.front(); n <= hi; n += (std::max)(size_t{ 1 }, n / 8)) {
        auto t = work.measure_ns(n, thresholds.size(), opts, [&](size_t k) { set_threshold(thresholds[k]); });
        for (size_t k = 0; k < t.size(); ++k) score[k] += std::log(t[k]);
        ++points;
    }

    // Among the candidates within tie_tolerance of the fastest, the largest threshold (the least
    // use of the higher algorithm) is taken: they are equally good, and this keeps the choice
    // from flipping between run-to-run noise.
    const double best_score = *std::min_element(score.begin(), score.end());
    const double tie = points ? std::log1p((std::max)(opts.tie_tolerance, 0.0)) * static_cast<double>(points) : 0.0;
    size_t best = 0;
    for (size_t k = 0; k < score.size(); ++k) {
        if (score[k] <= best_score + tie) best = k; // thresholds ascend, "off" last
    }
    if (opts.trace_candidate && points) {
        for (size_t k = 0; k + 1 < thresholds.size(); ++k) {
            opts.trace_candidate(name, thresholds[k], std::exp((score[k] - score.back()) / static_cast<double>(points)), k == best);
        }
    }
    if (thresholds[best] == off) return std::nullopt;
    return thresholds[best];
}

}

// Measures where each multiplication algorithm starts paying off over the one below it on this
// machine and (by default) installs the results as the runtime thresholds. Karatsuba is tuned
// first against basecase, then Toom-3 against whatever the tuned Karatsuba threshold selects
// below it, then Toom-4 against the tuned Toom-3/Karatsuba below it. See
// mul_tuning_detail::tune_threshold() for how each threshold is chosen.
//
// While it runs, the global thresholds are temporarily forced to other values. Multiplications
// on other threads still produce correct results but may run at suboptimal speed, and their
// load disturbs the measurements -- call this at startup or from an otherwise idle process.
inline mul_tuning_result tune_mul_thresholds(mul_tuning_options const& opts = {})
{
    const size_t prev_karatsuba = karatsuba_threshold();
    const size_t prev_toom3 = toom3_threshold();
    const size_t prev_toom4 = toom4_threshold();

    bool committed = false;
    NUMETRON_SCOPE_EXIT([&] {
        if (!committed) {
            set_karatsuba_threshold(prev_karatsuba);
            set_toom3_threshold(prev_toom3);
            set_toom4_threshold(prev_toom4);
        }
    });

    mul_tuning_detail::workload work{ (std::max)({ opts.karatsuba_max, opts.toom3_max, opts.toom4_max }), opts.operand_pairs };

    constexpr size_t off = (std::numeric_limits<size_t>::max)();
    mul_tuning_result result{};

    // Algorithms above the one being tuned are switched off until their own turn; one that ends
    // up not found stays off for the tuning of the next one, and is restored afterwards.
    set_toom4_threshold(off);
    set_toom3_threshold(off);
    auto karatsuba = mul_tuning_detail::tune_threshold(work, opts, "karatsuba", &set_karatsuba_threshold,
        min_karatsuba_threshold, opts.karatsuba_max);
    result.karatsuba_found = karatsuba.has_value();
    result.karatsuba_threshold = karatsuba.value_or(prev_karatsuba);
    set_karatsuba_threshold(karatsuba.value_or(off));

    auto toom3 = mul_tuning_detail::tune_threshold(work, opts, "toom3", &set_toom3_threshold,
        (std::max)(min_toom3_threshold, karatsuba.value_or(min_toom3_threshold)), opts.toom3_max);
    result.toom3_found = toom3.has_value();
    result.toom3_threshold = toom3.value_or(prev_toom3);
    set_toom3_threshold(toom3.value_or(off));

    auto toom4 = mul_tuning_detail::tune_threshold(work, opts, "toom4", &set_toom4_threshold,
        (std::max)(min_toom4_threshold, toom3.value_or(min_toom4_threshold)), opts.toom4_max);
    result.toom4_found = toom4.has_value();
    result.toom4_threshold = toom4.value_or(prev_toom4);

    if (opts.apply) {
        set_karatsuba_threshold(result.karatsuba_threshold);
        set_toom3_threshold(result.toom3_threshold);
        set_toom4_threshold(result.toom4_threshold);
        committed = true;
    }
    return result;
}

}
