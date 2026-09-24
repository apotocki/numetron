// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <memory>
#include <span>
#include <tuple>

#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/assert.hpp"

#include "platform.hpp"
#include "toom/thresholds.hpp"

// Experimental Karatsuba running entirely in assembly (src/arch/x86_64/karatsuba_mul.s): the
// recursion, |a0-a1|, the interpolation with vm1 folded in, and calls into the mul_basecase
// picked for the CPU. Selected with NUMETRON_KARATSUBA_IMPL = NUMETRON_KARATSUBA_IMPL_ASM
// (numetron/config/implementation.hpp); x86-64 (karatsuba_mul.s for SysV, karatsuba_mul.asm for
// Microsoft x64).

#if !(defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64)))
#   error "umul_karatsuba_asm.hpp needs NUMETRON_USE_ASM on x86-64"
#endif

namespace numetron::limb_arithmetic {

namespace detail {

// The mul_basecase the assembly calls for products below the Karatsuba threshold.
inline auto karatsuba_asm_basecase() noexcept -> decltype(numetron_karatsuba_ctx::mul_basecase)
{
#if defined(NUMETRON_PLATFORM_AUTODETECT)
    static const auto fn = reinterpret_cast<decltype(numetron_karatsuba_ctx::mul_basecase)>(
        detect_mul_basecase(numetron_detect_platform()));
    return fn;
#else
    return reinterpret_cast<decltype(numetron_karatsuba_ctx::mul_basecase)>(&NUMETRON_mul_basecase);
#endif
}

// Scratch limbs numetron_karatsuba_mul needs for an un-limb product: a node takes 2n
// (n = ceil(un/2)) and passes the rest to its children, none larger than n.
inline size_t karatsuba_asm_scratch(size_t un) noexcept
{
    size_t sz = 0;
    while (un >= min_karatsuba_threshold) {
        const size_t n = un - un / 2;
        sz += 2 * n;
        un = n;
    }
    return sz;
}

// Karatsuba multiplication core; same contract as umul_karatsuba_impl, except that the scratch is
// allocated once for the whole recursion.
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(sizeof(LimbT) == 8)
LimbT* umul_karatsuba_asm_impl(std::span<const LimbT> u, std::span<const LimbT> v,
    LimbT* rb,
    AllocatorT alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();
    const size_t threshold = karatsuba_threshold();

    assert(un >= vn && vn >= threshold && 2 * vn > un);

    const size_t scratch_sz = karatsuba_asm_scratch(un);
    LimbT* scratch = std::allocator_traits<AllocatorT>::allocate(alloc, scratch_sz);
    NUMETRON_SCOPE_EXIT([&] {
        std::allocator_traits<AllocatorT>::deallocate(alloc, scratch, scratch_sz);
    });

    const numetron_karatsuba_ctx ctx{ karatsuba_asm_basecase(), threshold, reinterpret_cast<uint64_t*>(scratch) };
    numetron_karatsuba_mul(reinterpret_cast<uint64_t*>(rb),
        reinterpret_cast<const uint64_t*>(u.data()), un,
        reinterpret_cast<const uint64_t*>(v.data()), vn, &ctx);
    return rb + un + vn;
}

} // namespace detail

// umul_karatsuba() running umul_karatsuba_asm_impl. Same contract.
template <std::unsigned_integral LimbT, typename AllocatorT, typename ScratchAllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul_karatsuba_asm(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc, ScratchAllocatorT scratch_alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();

    assert(un > 0 && vn > 0 && un >= vn);

    const size_t alloc_sz = un + vn;
    LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
    try {
        LimbT* re = detail::umul_karatsuba_asm_impl(u, v, rb, scratch_alloc);
        while (re != rb && *(re - 1) == 0) --re;
        return { rb, static_cast<size_t>(re - rb), alloc_sz };
    }
    catch (...) {
        std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
        throw;
    }
}

}
