// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))

#if defined(NUMETRON_PLATFORM_AUTODETECT)
#include <mutex>
typedef void (*detect_mul_basecase_type)(uint64_t*, const uint64_t*, size_t, const uint64_t*, size_t);
inline std::once_flag mul_basecase_init_flag;
extern "C" uint64_t numetron_detect_platform();
extern "C" detect_mul_basecase_type detect_mul_basecase(uint64_t);
inline void (*__mul_basecase_ptr)(uint64_t*, const uint64_t*, size_t, const uint64_t*, size_t) = nullptr;
#define NUMETRON_mul_basecase __mul_basecase_ptr
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
