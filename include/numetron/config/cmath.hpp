// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once
#include <cmath>

#if defined(__GNUC__) && __GNUC__ <= 13

namespace std {
    // GCC 13 and earlier do not support long double logl
    inline long double logl(long double x) { return log(x); }
    inline long double powl(long double x, long double y) { return pow(x, y); }
}

#endif
