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
#include "numetron/limb_arithmetic/umul_dispatch.hpp" // the chain it tunes and times
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
    size_t toom6h_max = 8192;
    size_t toom8h_max = 16384;
    size_t fft_max = 16384;
    size_t toom32_max = 1024; // vn, with un = 1.5 vn
    size_t toom42_max = 1024; // vn, with un = 1.875 vn
    size_t toom76_max = 4096; // vn, with un = 1.3 vn
    size_t toom63_max = 4096; // vn, with un = 1.875 vn
    size_t toom98_max = 4096;  // vn, with un = 1.15 vn
    size_t toom107_max = 4096; // vn, with un = 1.5 vn
    size_t toom116_max = 4096; // vn, with un = 1.875 vn
    size_t toom54_max = 1024;  // vn, with un = 1.3 vn (it stops at the Toom-6.5 threshold)
    size_t toom53_max = 1024;  // vn, with un = 1.7 vn (likewise)
    size_t toom43_max = 1024;  // vn, with un = 1.375 vn (likewise)
    size_t sqr_karatsuba_max = 256; // the squaring stages (u * u)
    size_t sqr_toom3_max = 2048;
    size_t sqr_toom4_max = 4096;
    size_t sqr_toom6h_max = 8192;
    size_t sqr_toom8h_max = 16384;
    size_t sqr_fft_max = 16384;

    // Install the found thresholds; when false they are only returned and the previous values
    // are restored.
    bool apply = true;

    // false: tune only the unbalanced stages (Toom-3/2 .. Toom-8.5 11 x 6, Toom-5/4, Toom-5/3,
    // Toom-4/3), against the balanced thresholds as they are installed (e.g. the defaults); the
    // balanced ones and the FFT are returned unchanged and reported as found.
    bool tune_balanced = true;

    // Tune the squaring thresholds (Karatsuba, Toom-3 .. Toom-8.5 and the FFT on u * u) too; only
    // with tune_balanced.
    bool tune_squares = true;

    // Tune the squaring thresholds alone; the multiplication's are returned unchanged and
    // reported as found.
    bool squares_only = false;
};

struct mul_tuning_result
{
    size_t karatsuba_threshold;
    size_t toom3_threshold;
    size_t toom4_threshold;
    size_t toom6h_threshold;
    size_t toom8h_threshold;
    size_t fft_threshold;
    size_t toom32_threshold;
    size_t toom42_threshold;
    size_t toom76_threshold;
    size_t toom63_threshold;
    size_t toom98_threshold;
    size_t toom107_threshold;
    size_t toom116_threshold;
    size_t toom54_threshold;
    size_t toom53_threshold;
    size_t toom43_threshold;
    size_t sqr_karatsuba_threshold;
    size_t sqr_toom3_threshold;
    size_t sqr_toom4_threshold;
    size_t sqr_toom6h_threshold;
    size_t sqr_toom8h_threshold;
    size_t sqr_fft_threshold;
    bool karatsuba_found;
    bool toom3_found;
    bool toom4_found;
    bool toom6h_found;
    bool toom8h_found;
    bool fft_found;
    bool toom32_found;
    bool toom42_found;
    bool toom76_found;
    bool toom63_found;
    bool toom98_found;
    bool toom107_found;
    bool toom116_found;
    bool toom54_found;
    bool toom53_found;
    bool toom43_found;
    bool sqr_karatsuba_found;
    bool sqr_toom3_found;
    bool sqr_toom4_found;
    bool sqr_toom6h_found;
    bool sqr_toom8h_found;
    bool sqr_fft_found;
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

    // true: squares, u * u with the same pointer on both sides (un == vn).
    bool square = false;

    // un x vn products (un >= vn, both <= max_limbs).
    double batch_ns(size_t un, size_t vn, size_t reps)
    {
        // Pairs are packed back to back (pair k at offsets k*un, k*vn), not at a fixed max_limbs
        // stride: a power-of-two-ish stride would map every pair onto the same L1 sets.
        constexpr size_t operand_bytes_budget = 64 * 1024;
        const size_t pairs = (std::clamp)(operand_bytes_budget / ((un + vn) * sizeof(limb_type)), size_t{ 1 }, max_pairs_);

        size_t k = 0;
        auto start = clock_type::now();
        for (size_t i = 0; i < reps; ++i) {
            umul_dispatch(u_.data() + k * un, un, square ? u_.data() + k * un : v_.data() + k * vn, vn, r_.data(), numetron::detail::stack_allocator<limb_type>{});
            if (++k == pairs) k = 0;
        }
        auto finish = clock_type::now();
        sink_ = r_[0];
        return std::chrono::duration<double, std::nano>(finish - start).count();
    }

    // Best per-multiplication times of un x vn products under each of several threshold settings
    // (install(k) installs setting k), `samples` batches per setting, every batch sized to last at
    // least min_sample_time. The settings are measured interleaved, in alternately forward and
    // backward order (ABBA...), so that a drift of the machine's speed during the probe (clock
    // boost, thermal throttling, background load) hits all of them alike instead of biasing the
    // comparison.
    template <typename InstallF>
    std::vector<double> measure_ns(size_t un, size_t vn, size_t settings, mul_tuning_options const& opts, InstallF&& install)
    {
        const double min_ns = std::chrono::duration<double, std::nano>(opts.min_sample_time).count();

        // warm-up of every path: caches, stack allocator slabs
        for (size_t k = 0; k < settings; ++k) {
            install(k);
            batch_ns(un, vn, 1);
        }

        size_t reps = 1;
        double elapsed = batch_ns(un, vn, reps);
        while (elapsed < min_ns && reps < (size_t{ 1 } << 30)) {
            reps *= 2;
            elapsed = batch_ns(un, vn, reps);
        }

        const double r = static_cast<double>(reps);
        std::vector<double> best(settings, std::numeric_limits<double>::infinity());
        const unsigned samples = (std::max)(opts.samples, 1u);
        for (unsigned i = 0; i < samples; ++i) {
            for (size_t j = 0; j < settings; ++j) {
                const size_t k = (i & 1) ? settings - 1 - j : j;
                install(k);
                best[k] = (std::min)(best[k], batch_ns(un, vn, reps) / r);
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
// The products are n x n, or (un_num / un_den) n x n for an unbalanced algorithm (Toom-3/2: 3/2);
// n, and so the threshold, is vn either way.
//
// Returns the threshold, or nothing when leaving the higher algorithm off is best.
template <typename SetThresholdF>
std::optional<size_t> tune_threshold(workload& work, mul_tuning_options const& opts, char const* name,
    SetThresholdF&& set_threshold, size_t lo, size_t hi, size_t un_num = 1, size_t un_den = 1)
{
    constexpr size_t off = (std::numeric_limits<size_t>::max)();
    constexpr size_t max_candidates = 8;
    auto un_of = [&](size_t n) { return n * un_num / un_den; };

    // Phase 1: scan.
    std::vector<size_t> sizes;
    std::vector<double> ratios;
    for (size_t n = lo; n <= hi; n += (std::max)(size_t{ 1 }, n / 16)) {
        auto t = work.measure_ns(un_of(n), n, 2, opts, [&](size_t k) { set_threshold(k ? n : n + 1); });
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
        auto t = work.measure_ns(un_of(n), n, thresholds.size(), opts, [&](size_t k) { set_threshold(thresholds[k]); });
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
// below it, then Toom-4 against the tuned Toom-3/Karatsuba below it, then Toom-6.5 and Toom-8.5
// against all of those below them, then the unbalanced Toom-3/2 on 1.5n x n products against
// Karatsuba (whose halves take the tuned balanced chain), and last the FFT against the whole Toom
// chain. See mul_tuning_detail::tune_threshold() for how each threshold is chosen.
//
// While it runs, the global thresholds are temporarily forced to other values. Multiplications
// on other threads still produce correct results but may run at suboptimal speed, and their
// load disturbs the measurements -- call this at startup or from an otherwise idle process.
inline mul_tuning_result tune_mul_thresholds(mul_tuning_options const& opts = {})
{
    const size_t prev_karatsuba = karatsuba_threshold();
    const size_t prev_toom3 = toom3_threshold();
    const size_t prev_toom4 = toom4_threshold();
    const size_t prev_toom6h = toom6h_threshold();
    const size_t prev_toom8h = toom8h_threshold();
    const size_t prev_fft = fft_threshold();
    const size_t prev_toom32 = toom32_threshold();
    const size_t prev_toom42 = toom42_threshold();
    const size_t prev_toom76 = toom76_threshold();
    const size_t prev_toom63 = toom63_threshold();
    const size_t prev_toom98 = toom98_threshold();
    const size_t prev_toom107 = toom107_threshold();
    const size_t prev_toom116 = toom116_threshold();
    const size_t prev_toom54 = toom54_threshold();
    const size_t prev_toom53 = toom53_threshold();
    const size_t prev_toom43 = toom43_threshold();
    const size_t prev_sqr_karatsuba = sqr_karatsuba_threshold();
    const size_t prev_sqr_toom3 = sqr_toom3_threshold();
    const size_t prev_sqr_toom4 = sqr_toom4_threshold();
    const size_t prev_sqr_toom6h = sqr_toom6h_threshold();
    const size_t prev_sqr_toom8h = sqr_toom8h_threshold();
    const size_t prev_sqr_fft = sqr_fft_threshold();

    bool committed = false;
    NUMETRON_SCOPE_EXIT([&] {
        if (!committed) {
            set_toom63_threshold(prev_toom63);
            set_toom98_threshold(prev_toom98);
            set_toom107_threshold(prev_toom107);
            set_toom116_threshold(prev_toom116);
            set_toom54_threshold(prev_toom54);
            set_toom53_threshold(prev_toom53);
            set_toom43_threshold(prev_toom43);
            set_toom32_threshold(prev_toom32);
            set_toom42_threshold(prev_toom42);
            set_toom76_threshold(prev_toom76);
            set_karatsuba_threshold(prev_karatsuba);
            set_toom3_threshold(prev_toom3);
            set_toom4_threshold(prev_toom4);
            set_toom6h_threshold(prev_toom6h);
            set_toom8h_threshold(prev_toom8h);
            set_fft_threshold(prev_fft);
            set_sqr_karatsuba_threshold(prev_sqr_karatsuba);
            set_sqr_toom3_threshold(prev_sqr_toom3);
            set_sqr_toom4_threshold(prev_sqr_toom4);
            set_sqr_toom6h_threshold(prev_sqr_toom6h);
            set_sqr_toom8h_threshold(prev_sqr_toom8h);
            set_sqr_fft_threshold(prev_sqr_fft);
        }
    });

    mul_tuning_detail::workload work{ (std::max)({ opts.karatsuba_max, opts.toom3_max, opts.toom4_max, opts.toom6h_max, opts.toom8h_max, opts.fft_max,
        opts.toom32_max * 3 / 2, opts.toom42_max * 15 / 8, opts.toom76_max * 13 / 10, opts.toom63_max * 15 / 8,
        opts.toom98_max * 23 / 20, opts.toom107_max * 3 / 2, opts.toom116_max * 15 / 8,
        opts.toom54_max * 13 / 10, opts.toom53_max * 17 / 10, opts.toom43_max * 11 / 8,
        opts.sqr_karatsuba_max, opts.sqr_toom3_max, opts.sqr_toom4_max, opts.sqr_toom6h_max, opts.sqr_toom8h_max, opts.sqr_fft_max }), opts.operand_pairs };

    constexpr size_t off = (std::numeric_limits<size_t>::max)();
    mul_tuning_result result{};

    auto keep_squares = [&] {
        result.sqr_karatsuba_threshold = prev_sqr_karatsuba;
        result.sqr_toom3_threshold = prev_sqr_toom3;
        result.sqr_toom4_threshold = prev_sqr_toom4;
        result.sqr_toom6h_threshold = prev_sqr_toom6h;
        result.sqr_toom8h_threshold = prev_sqr_toom8h;
        result.sqr_fft_threshold = prev_sqr_fft;
        result.sqr_karatsuba_found = result.sqr_toom3_found = result.sqr_toom4_found = true;
        result.sqr_toom6h_found = result.sqr_toom8h_found = result.sqr_fft_found = true;
    };

    // The squaring chain (u * u), each stage against the ones below it like the balanced chain.
    // It is independent of the multiplication thresholds (its Toom products are squares again)
    // except the FFT's, which it shares and which is off here.
    auto tune_squares = [&] {
        work.square = true;
        NUMETRON_SCOPE_EXIT([&] { work.square = false; });
        set_sqr_fft_threshold(off);
        set_sqr_toom8h_threshold(off);
        set_sqr_toom6h_threshold(off);
        set_sqr_toom4_threshold(off);
        set_sqr_toom3_threshold(off);
        auto karatsuba = mul_tuning_detail::tune_threshold(work, opts, "sqr_kara", &set_sqr_karatsuba_threshold,
            (std::max)(min_sqr_karatsuba_threshold, sqr_basecase_threshold()), opts.sqr_karatsuba_max);
        result.sqr_karatsuba_found = karatsuba.has_value();
        result.sqr_karatsuba_threshold = karatsuba.value_or(prev_sqr_karatsuba);
        set_sqr_karatsuba_threshold(karatsuba.value_or(off));

        auto toom3 = mul_tuning_detail::tune_threshold(work, opts, "sqr_toom3", &set_sqr_toom3_threshold,
            (std::max)(min_sqr_toom3_threshold, karatsuba.value_or(min_sqr_toom3_threshold)), opts.sqr_toom3_max);
        result.sqr_toom3_found = toom3.has_value();
        result.sqr_toom3_threshold = toom3.value_or(prev_sqr_toom3);
        set_sqr_toom3_threshold(toom3.value_or(off));

        auto toom4 = mul_tuning_detail::tune_threshold(work, opts, "sqr_toom4", &set_sqr_toom4_threshold,
            (std::max)(min_sqr_toom4_threshold, toom3.value_or(min_sqr_toom4_threshold)), opts.sqr_toom4_max);
        result.sqr_toom4_found = toom4.has_value();
        result.sqr_toom4_threshold = toom4.value_or(prev_sqr_toom4);
        set_sqr_toom4_threshold(toom4.value_or(off));

        auto toom6h = mul_tuning_detail::tune_threshold(work, opts, "sqr_toom6h", &set_sqr_toom6h_threshold,
            (std::max)(min_sqr_toom6h_threshold, toom4.value_or(toom3.value_or(min_sqr_toom6h_threshold))), opts.sqr_toom6h_max);
        result.sqr_toom6h_found = toom6h.has_value();
        result.sqr_toom6h_threshold = toom6h.value_or(prev_sqr_toom6h);
        set_sqr_toom6h_threshold(toom6h.value_or(off));

        auto toom8h = mul_tuning_detail::tune_threshold(work, opts, "sqr_toom8h", &set_sqr_toom8h_threshold,
            (std::max)(min_sqr_toom8h_threshold, toom6h.value_or(toom4.value_or(toom3.value_or(min_sqr_toom8h_threshold)))), opts.sqr_toom8h_max);
        result.sqr_toom8h_found = toom8h.has_value();
        result.sqr_toom8h_threshold = toom8h.value_or(prev_sqr_toom8h);
        set_sqr_toom8h_threshold(toom8h.value_or(off));

        // the FFT squares by itself from its own threshold, searched like the multiplication's
        // (kept when it is the fill rule's lower bound, detail::fft_threshold_is_fill_bound)
        std::optional<size_t> fft = prev_sqr_fft;
        if constexpr (!detail::fft_threshold_is_fill_bound) {
            fft = mul_tuning_detail::tune_threshold(work, opts, "sqr_fft", &set_sqr_fft_threshold,
                (std::max)(min_sqr_fft_threshold, toom4.value_or(toom3.value_or(min_sqr_toom4_threshold))), opts.sqr_fft_max);
        }
        result.sqr_fft_found = fft.has_value();
        result.sqr_fft_threshold = fft.value_or(prev_sqr_fft);
        set_sqr_fft_threshold(result.sqr_fft_threshold);
    };

    // The unbalanced stages, against whatever balanced thresholds are installed, the FFT off. Each
    // on shapes from the middle of its window, against what takes those shapes otherwise: the
    // Karatsuba level, and below the Karatsuba threshold the basecase (which the Toom plans may
    // beat too), so each from its own minimum up. Toom-3/2 (1.25 <= un/vn < 1.75) on 1.5n x n,
    // then Toom-4/2 (1.75 <= un/vn < 2) on 1.875n x n; their windows don't overlap.
    auto tune_unbalanced = [&] {
        auto toom32 = mul_tuning_detail::tune_threshold(work, opts, "toom32", &set_toom32_threshold,
            min_toom32_threshold, opts.toom32_max, 3, 2);
        result.toom32_found = toom32.has_value();
        result.toom32_threshold = toom32.value_or(prev_toom32);
        set_toom32_threshold(result.toom32_threshold);

        auto toom42 = mul_tuning_detail::tune_threshold(work, opts, "toom42", &set_toom42_threshold,
            min_toom42_threshold, opts.toom42_max, 15, 8);
        result.toom42_found = toom42.has_value();
        result.toom42_threshold = toom42.value_or(prev_toom42);
        set_toom42_threshold(result.toom42_threshold);

        // Toom-6/3 (same window as toom42, checked before it) on 1.875n x n, against toom42 and
        // below, from the toom42 threshold up (the header-only builds won already at the Toom-3
        // threshold, where the search used to start).
        auto toom63 = mul_tuning_detail::tune_threshold(work, opts, "toom63", &set_toom63_threshold,
            (std::max)(min_toom63_threshold, result.toom42_threshold), opts.toom63_max, 15, 8);
        result.toom63_found = toom63.has_value();
        result.toom63_threshold = toom63.value_or(prev_toom63);
        set_toom63_threshold(result.toom63_threshold);

        // Toom-6.5 7 x 6 (un/vn < 1.4) on 1.3n x n, against Toom-4 / toom32 below it, from the
        // Toom-3 threshold up (it is a large-operand plan; from the Toom-4 one it won everywhere).
        auto toom76 = mul_tuning_detail::tune_threshold(work, opts, "toom76", &set_toom76_threshold,
            (std::max)(min_toom76_threshold, toom3_threshold()), opts.toom76_max, 13, 10);
        result.toom76_found = toom76.has_value();
        result.toom76_threshold = toom76.value_or(prev_toom76);
        set_toom76_threshold(result.toom76_threshold);

        // Toom-8.5 N x (17 - N), each in the middle of its window against what takes it otherwise
        // (toom76 / Toom-6.5 / Toom-8.5, toom32, toom63), from the Toom-4 threshold up: 9 x 8 on
        // 1.15n x n, 10 x 7 on 1.5n x n, 11 x 6 on 1.875n x n.
        auto toom98 = mul_tuning_detail::tune_threshold(work, opts, "toom98", &set_toom98_threshold,
            (std::max)(min_toom98_threshold, toom4_threshold()), opts.toom98_max, 23, 20);
        result.toom98_found = toom98.has_value();
        result.toom98_threshold = toom98.value_or(prev_toom98);
        set_toom98_threshold(result.toom98_threshold);

        auto toom107 = mul_tuning_detail::tune_threshold(work, opts, "toom107", &set_toom107_threshold,
            (std::max)(min_toom107_threshold, toom4_threshold()), opts.toom107_max, 3, 2);
        result.toom107_found = toom107.has_value();
        result.toom107_threshold = toom107.value_or(prev_toom107);
        set_toom107_threshold(result.toom107_threshold);

        auto toom116 = mul_tuning_detail::tune_threshold(work, opts, "toom116", &set_toom116_threshold,
            (std::max)(min_toom116_threshold, toom4_threshold()), opts.toom116_max, 15, 8);
        result.toom116_found = toom116.has_value();
        result.toom116_threshold = toom116.value_or(prev_toom116);
        set_toom116_threshold(result.toom116_threshold);

        // Toom-5/4 and Toom-5/3 (checked first below the Toom-6.5 threshold) on 1.3n x n and
        // 1.7n x n, against everything else, from their minimum up.
        auto toom54 = mul_tuning_detail::tune_threshold(work, opts, "toom54", &set_toom54_threshold,
            min_toom54_threshold, opts.toom54_max, 13, 10);
        result.toom54_found = toom54.has_value();
        result.toom54_threshold = toom54.value_or(prev_toom54);
        set_toom54_threshold(result.toom54_threshold);

        auto toom53 = mul_tuning_detail::tune_threshold(work, opts, "toom53", &set_toom53_threshold,
            min_toom53_threshold, opts.toom53_max, 17, 10);
        result.toom53_found = toom53.has_value();
        result.toom53_threshold = toom53.value_or(prev_toom53);
        set_toom53_threshold(result.toom53_threshold);

        // Toom-4/3 (checked before toom54) on 1.375n x n, against everything else.
        auto toom43 = mul_tuning_detail::tune_threshold(work, opts, "toom43", &set_toom43_threshold,
            min_toom43_threshold, opts.toom43_max, 11, 8);
        result.toom43_found = toom43.has_value();
        result.toom43_threshold = toom43.value_or(prev_toom43);
        set_toom43_threshold(result.toom43_threshold);
    };

    if (opts.squares_only) {
        result.karatsuba_threshold = prev_karatsuba;
        result.toom3_threshold = prev_toom3;
        result.toom4_threshold = prev_toom4;
        result.toom6h_threshold = prev_toom6h;
        result.toom8h_threshold = prev_toom8h;
        result.fft_threshold = prev_fft;
        result.toom32_threshold = prev_toom32;
        result.toom42_threshold = prev_toom42;
        result.toom76_threshold = prev_toom76;
        result.toom63_threshold = prev_toom63;
        result.toom98_threshold = prev_toom98;
        result.toom107_threshold = prev_toom107;
        result.toom116_threshold = prev_toom116;
        result.toom54_threshold = prev_toom54;
        result.toom53_threshold = prev_toom53;
        result.toom43_threshold = prev_toom43;
        result.karatsuba_found = result.toom3_found = result.toom4_found = result.toom6h_found = true;
        result.toom8h_found = result.fft_found = result.toom32_found = result.toom42_found = true;
        result.toom76_found = result.toom63_found = result.toom98_found = result.toom107_found = true;
        result.toom116_found = result.toom54_found = result.toom53_found = result.toom43_found = true;
        tune_squares(); // the multiplication's thresholds don't take part (the squares' products are squares)
        if (opts.apply) {
            set_sqr_karatsuba_threshold(result.sqr_karatsuba_threshold);
            set_sqr_toom3_threshold(result.sqr_toom3_threshold);
            set_sqr_toom4_threshold(result.sqr_toom4_threshold);
            set_sqr_toom6h_threshold(result.sqr_toom6h_threshold);
            set_sqr_toom8h_threshold(result.sqr_toom8h_threshold);
            set_sqr_fft_threshold(result.sqr_fft_threshold);
            committed = true;
        }
        return result;
    }

    set_toom32_threshold(off);
    set_toom42_threshold(off);
    set_toom76_threshold(off);
    set_toom63_threshold(off);
    set_toom98_threshold(off);
    set_toom107_threshold(off);
    set_toom116_threshold(off);
    set_toom54_threshold(off);
    set_toom53_threshold(off);
    set_toom43_threshold(off);
    set_fft_threshold(off);

    if (!opts.tune_balanced) {
        tune_unbalanced();
        keep_squares();
        result.karatsuba_threshold = prev_karatsuba;
        result.toom3_threshold = prev_toom3;
        result.toom4_threshold = prev_toom4;
        result.toom6h_threshold = prev_toom6h;
        result.toom8h_threshold = prev_toom8h;
        result.fft_threshold = prev_fft;
        result.karatsuba_found = result.toom3_found = result.toom4_found = true;
        result.toom6h_found = result.toom8h_found = result.fft_found = true;
        set_fft_threshold(prev_fft);
        if (opts.apply) committed = true; // else the scope exit restores the previous toom32/42
        return result;
    }

    // Algorithms above the one being tuned are switched off until their own turn; one that ends
    // up not found stays off for the tuning of the next one, and is restored afterwards.
    set_fft_threshold(off);
    set_toom8h_threshold(off);
    set_toom6h_threshold(off);
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
    set_toom4_threshold(toom4.value_or(off));

    auto toom6h = mul_tuning_detail::tune_threshold(work, opts, "toom6h", &set_toom6h_threshold,
        (std::max)(min_toom6h_threshold, toom4.value_or(toom3.value_or(min_toom6h_threshold))), opts.toom6h_max);
    result.toom6h_found = toom6h.has_value();
    result.toom6h_threshold = toom6h.value_or(prev_toom6h);
    set_toom6h_threshold(toom6h.value_or(off));

    auto toom8h = mul_tuning_detail::tune_threshold(work, opts, "toom8h", &set_toom8h_threshold,
        (std::max)(min_toom8h_threshold, toom6h.value_or(toom4.value_or(toom3.value_or(min_toom8h_threshold)))), opts.toom8h_max);
    result.toom8h_found = toom8h.has_value();
    result.toom8h_threshold = toom8h.value_or(prev_toom8h);
    set_toom8h_threshold(toom8h.value_or(off));

    tune_unbalanced(); // the FFT is still off
    if (opts.tune_squares) tune_squares();
    else keep_squares();

    // The FFT takes any size (and has no sub-products), so it is searched from the Toom-4
    // threshold up against everything below; kept when it is the fill rule's lower bound
    // (detail::fft_threshold_is_fill_bound).
    std::optional<size_t> fft = prev_fft;
    if constexpr (!detail::fft_threshold_is_fill_bound) {
        fft = mul_tuning_detail::tune_threshold(work, opts, "fft", &set_fft_threshold,
            (std::max)(min_fft_threshold, toom4.value_or(toom3.value_or(min_toom4_threshold))), opts.fft_max);
    }
    result.fft_found = fft.has_value();
    result.fft_threshold = fft.value_or(prev_fft);

    if (opts.apply) {
        set_karatsuba_threshold(result.karatsuba_threshold);
        set_toom3_threshold(result.toom3_threshold);
        set_toom4_threshold(result.toom4_threshold);
        set_toom6h_threshold(result.toom6h_threshold);
        set_toom8h_threshold(result.toom8h_threshold);
        set_fft_threshold(result.fft_threshold);
        set_toom32_threshold(result.toom32_threshold);
        set_toom42_threshold(result.toom42_threshold);
        set_toom76_threshold(result.toom76_threshold);
        set_toom63_threshold(result.toom63_threshold);
        set_toom98_threshold(result.toom98_threshold);
        set_toom107_threshold(result.toom107_threshold);
        set_toom116_threshold(result.toom116_threshold);
        set_toom54_threshold(result.toom54_threshold);
        set_toom53_threshold(result.toom53_threshold);
        set_toom43_threshold(result.toom43_threshold);
        set_sqr_karatsuba_threshold(result.sqr_karatsuba_threshold);
        set_sqr_toom3_threshold(result.sqr_toom3_threshold);
        set_sqr_toom4_threshold(result.sqr_toom4_threshold);
        set_sqr_toom6h_threshold(result.sqr_toom6h_threshold);
        set_sqr_toom8h_threshold(result.sqr_toom8h_threshold);
        set_sqr_fft_threshold(result.sqr_fft_threshold);
        committed = true;
    }
    return result;
}

}
