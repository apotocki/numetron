// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

// The assembly kernels the front end calls in a library that has them (NUMETRON_ASM_KERNELS:
// NUMETRON_COMPILED with NUMETRON_BACKEND_ASM, see config/implementation.hpp). The inline code
// calling them is the one the headers had when the application defined NUMETRON_USE_ASM itself:
// the lowest level, called directly or through a pointer picked once, never through a wrapper (a
// basecase product of a few limbs costs a few nanoseconds, one call level more is several percent
// of it). Which kernel runs on this CPU is the library's choice (src/backend.cpp: CPUID, the
// license, a pinned platform), made on the first call.

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "numetron/config/implementation.hpp" // NUMETRON_ASM_KERNELS
#include "platform.hpp"                       // NUMETRON_FORCEINLINE, NUMETRON_NOINLINE

#if defined(NUMETRON_ASM_KERNELS)

// r[0..n) = u[0..n) +/- v[0..n), returning the carry/borrow out (src/arch/x86_64/add_sub_n.*).
// rp may coincide with up or vp, or trail them.
extern "C" uint64_t numetron_add_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept;
extern "C" uint64_t numetron_sub_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept;

// rp[0..2n) = up[0..n)^2 for n >= 2 with mulx + adcx/adox (src/arch/x86_64/sqr_basecase_adx.*):
// the rows above the diagonal in one call, then the doubling and the diagonal. Needs BMI2 + ADX
// (detail::sqr_kernel_available()).
extern "C" void numetron_sqr_basecase_adx(uint64_t* rp, const uint64_t* up, size_t n) noexcept;

namespace numetron::limb_arithmetic::detail {

// Below this length the inline C++ loop wins over numetron_add_n / numetron_sub_n: the call itself
// costs about as much as the few limbs it would process (measured with numetron_bench_mul --add
// against GMP's mpn_add_n, which is also an out-of-line call: parity at 4 limbs, the call ahead
// from 8).
inline constexpr size_t asm_add_sub_n_min_limbs = 8;

// rp[0..un+vn) = up[0..un) * vp[0..vn), un >= vn >= 1 (GMP's mul_basecase contract: all un + vn
// limbs written).
using mul_basecase_fn = void (*)(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn);

// The basecase product for this CPU, chosen on the first call (null until then). A plain atomic
// rather than a function-local static: the latter's thread-safe initialization guard is checked
// on every call, through TLS on MSVC (~1 ns of every basecase product). Threads racing through
// the first call all store the same pointer. The first call's choice is out of line, and must
// stay so under link-time optimization too (MSVC's /GL inlined it, CPUID and all, into
// basic_integer::assign_mul: a larger frame there cost every small product up to 20%).
extern std::atomic<mul_basecase_fn> mul_basecase_fn_value;
NUMETRON_NOINLINE mul_basecase_fn init_mul_basecase() noexcept;

NUMETRON_FORCEINLINE mul_basecase_fn detected_mul_basecase() noexcept
{
    mul_basecase_fn fn = mul_basecase_fn_value.load(std::memory_order_relaxed);
    if (!fn) [[unlikely]] fn = init_mul_basecase();
    return fn;
}

// Whether numetron_sqr_basecase_adx may run here: 0 not checked yet, 1 yes, 2 no; checked on the
// first call.
extern std::atomic<int> sqr_kernel_state;
NUMETRON_NOINLINE int init_sqr_kernel_state() noexcept;

NUMETRON_FORCEINLINE bool sqr_kernel_available() noexcept
{
    int s = sqr_kernel_state.load(std::memory_order_relaxed);
    if (!s) [[unlikely]] s = init_sqr_kernel_state();
    return s == 1;
}

}

#endif // NUMETRON_ASM_KERNELS
