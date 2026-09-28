// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

// The multiplication chain above the basecase (limb_arithmetic/umul_dispatch.hpp), compiled once
// for NUMETRON_COMPILED (see config/implementation.hpp): the entry points umul.hpp declares.
// Without the define this unit is empty, so a build may always compile it.

#include "numetron/limb_arithmetic.hpp"

#if defined(NUMETRON_COMPILED)

#include "numetron/limb_arithmetic/umul_dispatch.hpp"

namespace numetron::limb_arithmetic::detail {

uint64_t* umul_large(const uint64_t* u, size_t un, const uint64_t* v, size_t vn, uint64_t* rb)
{
    return umul_large_impl(u, un, v, vn, rb);
}

}

#endif
