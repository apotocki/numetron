// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

// What the front end takes from the library for NUMETRON_COMPILED (see config/implementation.hpp)
// besides the compiled algorithms: the tunable thresholds (limb_arithmetic/toom/thresholds.hpp)
// and the choice of the assembly kernels for the CPU (limb_arithmetic/backend.hpp), all from this
// library's configuration. Without the define this unit is empty, so a build may always compile
// it.

#include "numetron/limb_arithmetic.hpp"

#if defined(NUMETRON_COMPILED)

#if !defined(NUMETRON_BUILDING_LIBRARY)
#   error "the numetron library's units are compiled with NUMETRON_BUILDING_LIBRARY"
#endif

namespace numetron::limb_arithmetic::detail {

#define NUMETRON_DETAIL_THRESHOLD_DEFINE(name, NAME) std::atomic<size_t> name##_threshold_value{ NUMETRON_DETAIL_THRESHOLD_INIT(name, NAME) };
NUMETRON_DETAIL_THRESHOLDS(NUMETRON_DETAIL_THRESHOLD_DEFINE)
#undef NUMETRON_DETAIL_THRESHOLD_DEFINE
std::atomic<size_t> basecase_limit_value{ NUMETRON_DETAIL_BASECASE_LIMIT_INIT };

}

#if defined(NUMETRON_USE_ASM)

namespace numetron::limb_arithmetic::detail {

namespace {

// The C++ basecase behind the mul_basecase contract, for CPUs no assembly basecase is picked for
// (also called from the assembly Karatsuba, through numetron_karatsuba_ctx). That contract is all
// un + vn limbs written; the C++ basecase may stop short of a zero top limb.
[[maybe_unused]] void mul_basecase_cxx(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn) noexcept
{
    for (uint64_t* e = umul_basecase_cxx<uint64_t>(up, un, vp, vn, rp), *re = rp + un + vn; e != re; ++e) *e = 0;
}

// The mul_basecase for this CPU: pinned (NUMETRON_PLATFORM_*), or picked from CPUID -- with
// NUMETRON_ASM_LICENSE_GMP_LGPL the GMP-derived routine for the CPU family; otherwise (or for a
// CPU that has none) our own numetron_mul_basecase_adx when the CPU has BMI2 + ADX; the C++
// basecase as the last resort.
mul_basecase_fn select_mul_basecase() noexcept
{
#   if defined(NUMETRON_PLATFORM_AUTODETECT)
#       if NUMETRON_ASM_LICENSE == NUMETRON_ASM_LICENSE_GMP_LGPL
    if (auto gmp = detect_mul_basecase(numetron_detect_platform())) return gmp;
#       endif
    if (cpu_has_bmi2_adx()) return &numetron_mul_basecase_adx;
    return &mul_basecase_cxx;
#   else
    return reinterpret_cast<mul_basecase_fn>(&NUMETRON_mul_basecase);
#   endif
}

// Whether numetron_sqr_basecase_adx may run on this CPU: 1 yes, 2 no. Pinned: yes with our own
// basecase (NUMETRON_PLATFORM_ADX), no with a GMP-derived one.
int select_sqr_kernel_state() noexcept
{
#   if defined(NUMETRON_PLATFORM_AUTODETECT)
    return cpu_has_bmi2_adx() ? 1 : 2;
#   elif defined(NUMETRON_PLATFORM_ADX)
    return 1;
#   else
    return 2;
#   endif
}

}

std::atomic<mul_basecase_fn> mul_basecase_fn_value{ nullptr };

NUMETRON_NOINLINE mul_basecase_fn init_mul_basecase() noexcept
{
    const mul_basecase_fn fn = select_mul_basecase();
    mul_basecase_fn_value.store(fn, std::memory_order_relaxed);
    return fn;
}

std::atomic<int> sqr_kernel_state{ 0 };

NUMETRON_NOINLINE int init_sqr_kernel_state() noexcept
{
    const int s = select_sqr_kernel_state();
    sqr_kernel_state.store(s, std::memory_order_relaxed);
    return s;
}

}

#elif defined(__x86_64__) || defined(_M_X64)

// A library without the assembly, on x86-64: what a front end compiled with NUMETRON_BACKEND_ASM
// by mistake refers to (backend.hpp), in C++, so that such a mismatch costs speed but still links
// and computes right. A front end compiled right (without NUMETRON_BACKEND_ASM) calls none of it.

namespace numetron::limb_arithmetic::detail {

namespace {

void mul_basecase_cxx(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn) noexcept
{
    for (uint64_t* e = umul_basecase_cxx<uint64_t>(up, un, vp, vn, rp), *re = rp + un + vn; e != re; ++e) *e = 0;
}

}

using mul_basecase_fn = void (*)(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn);

std::atomic<mul_basecase_fn> mul_basecase_fn_value{ &mul_basecase_cxx };

NUMETRON_NOINLINE mul_basecase_fn init_mul_basecase() noexcept
{
    return &mul_basecase_cxx;
}

std::atomic<int> sqr_kernel_state{ 2 }; // no squaring kernel: squares go to the product

NUMETRON_NOINLINE int init_sqr_kernel_state() noexcept
{
    return 2;
}

}

extern "C" uint64_t numetron_add_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept
{
    return numetron::limb_arithmetic::detail::add_n_x64_inline<uint64_t>(rp, up, vp, n);
}

extern "C" uint64_t numetron_sub_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept
{
    return numetron::limb_arithmetic::detail::sub_n_x64_inline<uint64_t>(rp, up, vp, n);
}

// Never called (sqr_kernel_state says there is no kernel), but correct if called.
extern "C" void numetron_sqr_basecase_adx(uint64_t* rp, const uint64_t* up, size_t n) noexcept
{
    numetron::limb_arithmetic::detail::mul_basecase_cxx(rp, up, n, up, n);
}

#endif

#endif
