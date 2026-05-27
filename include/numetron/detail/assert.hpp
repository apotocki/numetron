// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once
#include <cstdio>
#include <cstdlib>

namespace numetron {
    inline void assert_failed(char const* expr_str, char const* file, int line) {
        std::fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", expr_str, file, line);
        std::abort();
    }
}

#ifndef NUMETRON_ASSERT
#   ifndef NDEBUG
#     define NUMETRON_ASSERT(expr) if(!(expr)) numetron::assert_failed(#expr, __FILE__, __LINE__)
#     define NUMETRON_ASSERT_GE(a, b) if(!((a) >= (b))) numetron::assert_failed(#a " >= " #b, __FILE__, __LINE__)
#   else
#     define NUMETRON_ASSERT(expr) ((void)(expr))
#     define NUMETRON_ASSERT_GE(a, b) ((void)(0))
#   endif
#endif

#ifndef CONSTEVAL_STATIC_ASSERT
#   define CONSTEVAL_STATIC_ASSERT(c, msg) do { if (!(c)) throw msg; } while(false)
#endif

using namespace std::literals;

template <typename LimbT>
void print_limbs(LimbT const* limbs, size_t sz, std::string_view prefix)
{
    std::cout << "\n\n";
    for (size_t i = 0; i < sz; ++i) {
        std::cout << prefix << "["sv << std::dec << i << "]= 0x"sv << std::hex << limbs[i] << "\n";
    }
}