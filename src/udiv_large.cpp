// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

// The division (limb_arithmetic/udiv.hpp: the basecase and Svoboda's), compiled once for
// NUMETRON_COMPILED (see config/implementation.hpp): the entry point udiv.hpp declares.
// Without the define this unit is empty, so a build may always compile it.

#include "numetron/limb_arithmetic/udiv.hpp"

#if defined(NUMETRON_COMPILED)

#if !defined(NUMETRON_BUILDING_LIBRARY)
#   error "the numetron library's units are compiled with NUMETRON_BUILDING_LIBRARY"
#endif

namespace numetron::limb_arithmetic::detail {

uint64_t udiv_large(uint64_t uh, std::span<uint64_t>& ul, uint64_t dh, std::span<const uint64_t> dl, uint64_t* qit,
    size_t svoboda_threshold)
{
    return udiv_impl<uint64_t>(uh, ul, dh, dl, qit, std::allocator<uint64_t>{}, svoboda_threshold);
}

}

#endif
