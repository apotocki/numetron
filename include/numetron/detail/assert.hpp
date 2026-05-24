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
#     define NUMETRON_ASSERT(expr) ((void(0)))
#     define NUMETRON_ASSERT_GE(a, b) ((void(0)))
#   endif
#endif
