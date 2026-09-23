// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

// For the thin layers in front of the limb add/sub kernels: a short run is only a handful of
// instructions, so a call that the compiler declined to inline would cost as much as the work.
#ifndef NUMETRON_FORCEINLINE
#   if defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_FORCEINLINE __forceinline
#   else
#       define NUMETRON_FORCEINLINE inline __attribute__((always_inline))
#   endif
#endif

#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))

// src/arch/x86_64/add_sub_n.{asm,s}: r[0..n) = u[0..n) +/- v[0..n), returning the carry/borrow
// out. One generic x86-64 implementation (no per-CPU variants, so no dispatch through platform
// detection). rp may coincide with up or vp, or trail them.
extern "C" uint64_t numetron_add_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept;
extern "C" uint64_t numetron_sub_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept;

namespace numetron::limb_arithmetic::detail {
// Below this length the inline C++ loop wins: the call itself costs about as much as the few
// limbs it would process (measured with numetron_bench_mul --add against GMP's mpn_add_n,
// which is also an out-of-line call: parity at 4 limbs, the call ahead from 8).
inline constexpr size_t asm_add_sub_n_min_limbs = 8;
}

#if defined(NUMETRON_PLATFORM_AUTODETECT)
typedef void (*detect_mul_basecase_type)(uint64_t*, const uint64_t*, size_t, const uint64_t*, size_t);
extern "C" uint64_t numetron_detect_platform();
extern "C" detect_mul_basecase_type detect_mul_basecase(uint64_t);
#define NUMETRON_mul_basecase detected_mul_basecase_ptr
#endif

#if defined(NUMETRON_PLATFORM_K8)
#   if defined(NUMETRON_mul_basecase)
#       error "NUMETRON_mul_basecase already defined"
#   endif
#   define NUMETRON_mul_basecase __k8_mul_basecase
#endif

#if defined(NUMETRON_PLATFORM_ALDERLAKE)
#   if defined(NUMETRON_mul_basecase)
#       error "NUMETRON_mul_basecase already defined"
#   endif
#   define NUMETRON_mul_basecase __alderlake_mul_basecase
#endif

#if defined(NUMETRON_PLATFORM_CORE2)
#   if defined(NUMETRON_mul_basecase)
#       error "NUMETRON_mul_basecase already defined"
#   endif
#define NUMETRON_mul_basecase __core2_mul_basecase 
#endif

#if !defined(NUMETRON_PLATFORM_AUTODETECT)
extern "C" void NUMETRON_mul_basecase(uint64_t* rp, const uint64_t* up, size_t un, uint64_t const* vp, uint64_t vn);
#endif

#endif // NUMETRON_USE_ASM
