// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include "numetron/config/implementation.hpp" // NUMETRON_USE_ASM, NUMETRON_PLATFORM_*

// For the thin layers in front of the limb add/sub kernels: a short run is only a handful of
// instructions, so a call that the compiler declined to inline would cost as much as the work.
#ifndef NUMETRON_FORCEINLINE
#   if defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_FORCEINLINE __forceinline
#   else
#       define NUMETRON_FORCEINLINE inline __attribute__((always_inline))
#   endif
#endif

// The opposite, for code that is reached from many call sites and should exist once: e.g. the
// Toom engine's op implementations, one copy per op type rather than one per plan instruction
// (keeping the hot code small enough not to push the assembly kernels out of the uop cache).
#ifndef NUMETRON_NOINLINE
#   if defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_NOINLINE __declspec(noinline)
#   else
#       define NUMETRON_NOINLINE __attribute__((noinline))
#   endif
#endif

#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))

#if defined(_MSC_VER) && !defined(__clang__)
#   include <intrin.h> // __cpuid, __cpuidex
#else
#   include <cpuid.h>  // __get_cpuid_count
#endif

// src/arch/x86_64/add_sub_n.{asm,s}: r[0..n) = u[0..n) +/- v[0..n), returning the carry/borrow
// out. One generic x86-64 implementation (no per-CPU variants, so no dispatch through platform
// detection). rp may coincide with up or vp, or trail them.
extern "C" uint64_t numetron_add_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept;
extern "C" uint64_t numetron_sub_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n) noexcept;

// src/arch/x86_64/karatsuba_interp.{asm,s}: the fused middle-column pass of the Karatsuba
// interpolation (see detail::karatsuba_interp in umul_karatsuba_fused.hpp). Returns the carry into
// limb 2n in bits 0..31 and the carry into limb 3n in bits 32..63.
extern "C" uint64_t numetron_karatsuba_interp(uint64_t* rp, size_t n, size_t h) noexcept;

// src/arch/x86_64/karatsuba_mul.{asm,s}: the whole recursive Karatsuba in assembly, selected by
// NUMETRON_KARATSUBA_IMPL_ASM (see umul_karatsuba_asm.hpp).
struct numetron_karatsuba_ctx
{
    void (*mul_basecase)(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn);
    size_t threshold;
    uint64_t* scratch;
};
extern "C" void numetron_karatsuba_mul(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn, const numetron_karatsuba_ctx* ctx) noexcept;

// src/arch/x86_64/mul_basecase_adx.{asm,s}: our own (MIT) schoolbook multiplication with mulx +
// adcx/adox, the same contract as the GMP-derived mul_basecase routines; needs BMI2 + ADX. The
// runtime choice on such CPUs unless NUMETRON_ASM_LICENSE_GMP_LGPL; pinned with
// NUMETRON_PLATFORM_ADX.
extern "C" void numetron_mul_basecase_adx(uint64_t* rp, const uint64_t* up, size_t un, const uint64_t* vp, size_t vn);

namespace numetron::limb_arithmetic::detail {
// Below this length the inline C++ loop wins: the call itself costs about as much as the few
// limbs it would process (measured with numetron_bench_mul --add against GMP's mpn_add_n,
// which is also an out-of-line call: parity at 4 limbs, the call ahead from 8).
inline constexpr size_t asm_add_sub_n_min_limbs = 8;
}

#if defined(NUMETRON_PLATFORM_AUTODETECT)
// The runtime choice is made once, by detail::detected_mul_basecase() (umul_basecase.hpp).
typedef void (*detect_mul_basecase_type)(uint64_t*, const uint64_t*, size_t, const uint64_t*, size_t);
#   if NUMETRON_ASM_LICENSE == NUMETRON_ASM_LICENSE_GMP_LGPL
// src/arch/x86_64/detect_{platform,mul_basecase}: the GMP-derived routine for the CPU, or null.
extern "C" uint64_t numetron_detect_platform();
extern "C" detect_mul_basecase_type detect_mul_basecase(uint64_t);
#   endif

namespace numetron::limb_arithmetic::detail {
// CPUID.(EAX=7, ECX=0):EBX bit 8 (BMI2) and bit 19 (ADX): what numetron_mul_basecase_adx needs.
inline bool cpu_has_bmi2_adx() noexcept
{
#   if defined(_MSC_VER) && !defined(__clang__)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 7, 0);
    const unsigned ebx = static_cast<unsigned>(r[1]);
#   else
    unsigned a, ebx, c, d;
    if (!__get_cpuid_count(7, 0, &a, &ebx, &c, &d)) return false;
#   endif
    return (ebx & (1u << 8)) && (ebx & (1u << 19));
}
}
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

#if defined(NUMETRON_PLATFORM_ADX)
#   if defined(NUMETRON_mul_basecase)
#       error "NUMETRON_mul_basecase already defined"
#   endif
#   define NUMETRON_mul_basecase numetron_mul_basecase_adx
#endif

#if !defined(NUMETRON_PLATFORM_AUTODETECT) && !defined(NUMETRON_PLATFORM_ADX)
extern "C" void NUMETRON_mul_basecase(uint64_t* rp, const uint64_t* up, size_t un, uint64_t const* vp, uint64_t vn);
#endif

#endif // NUMETRON_USE_ASM
