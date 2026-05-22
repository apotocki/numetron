// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <array>
#include <cstddef>
#include "thresholds.hpp"

namespace numetron::limb_arithmetic {

struct toom_plan_descriptor
{
    size_t n;
    size_t m;
    size_t eval_point_count;
    size_t product_count;
    size_t split_den;
    size_t split_rounding_bias;
    size_t d_u_max_limbs_mul_n2;
    size_t d_u_max_limbs_mul_u_hi;
    size_t scratch_extra_mul_n2;
    size_t scratch_eval_buf_mul_chunk;
    size_t scratch_interp_buf_mul_chunk;
    bool has_signed_middle_term;
};

inline constexpr std::array toom_plan_table{
    toom_plan_descriptor{
        2,
        2,
        3,
        3,
        2,
        1,
        1,
        1,
        4,
        0,
        0,
        true,
    },
    toom_plan_descriptor{
        3,
        3,
        5,
        5,
        3,
        2,
        1,
        1,
        8,
        8,
        8,
        false,
    },
};

namespace detail {

template <size_t N, size_t M>
consteval bool has_toom_plan() noexcept
{
    for (auto const& plan : toom_plan_table)
        if (plan.n == N && plan.m == M)
            return true;
    return false;
}

template <size_t>
inline constexpr bool always_false_v = false;

template <size_t N, size_t M>
consteval toom_plan_descriptor get_toom_plan_descriptor() noexcept
{
    if constexpr (!has_toom_plan<N, M>()) {
        static_assert(always_false_v<N + M>,
            "numetron: toom_plan is not implemented for this (N, M) pair. "
            "Add a descriptor row to toom_plan_table.");
        return {};
    } else {
        for (auto const& plan : toom_plan_table)
            if (plan.n == N && plan.m == M)
                return plan;
        return {};
    }
}

} // namespace detail

template <size_t N, size_t M>
struct toom_plan
{
    static constexpr toom_plan_descriptor descriptor = detail::get_toom_plan_descriptor<N, M>();

    static constexpr size_t pieces_u = descriptor.n;
    static constexpr size_t pieces_v = descriptor.m;
    static constexpr size_t eval_point_count = descriptor.eval_point_count;
    static constexpr size_t product_count = descriptor.product_count;

    static constexpr size_t split_den = descriptor.split_den;
    static constexpr size_t split_rounding_bias = descriptor.split_rounding_bias;
    static constexpr size_t d_u_max_limbs_mul_n2 = descriptor.d_u_max_limbs_mul_n2;
    static constexpr size_t d_u_max_limbs_mul_u_hi = descriptor.d_u_max_limbs_mul_u_hi;
    static constexpr size_t scratch_extra_mul_n2 = descriptor.scratch_extra_mul_n2;
    static constexpr size_t scratch_eval_buf_mul_chunk = descriptor.scratch_eval_buf_mul_chunk;
    static constexpr size_t scratch_interp_buf_mul_chunk = descriptor.scratch_interp_buf_mul_chunk;
    static constexpr bool has_signed_middle_term = descriptor.has_signed_middle_term;
};

} // namespace numetron::limb_arithmetic
