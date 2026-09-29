set (LIBRARY_NAME numetron)
set (LIBRARY_HOME ${CMAKE_CURRENT_SOURCE_DIR}/..)
set (NUMETRON_LIB_SOURCES)

# The numetron library is the back end of the headers (see numetron/config/implementation.hpp):
# with NUMETRON_COMPILED (exported) they take the heavy algorithms, the basecase kernels and the
# tunable thresholds from it. The options below configure the library itself -- the assembly, its
# license, NUMETRON_SQR_STRAIGHT_MAX -- and stay private to it: what links `numetron` sees only
# NUMETRON_COMPILED and, when the library has the assembly kernels, NUMETRON_BACKEND_ASM. Code that
# runs the multiplication chain directly (the threshold tuner, tests of the plans) links
# `numetron_internal` instead, which carries the library's configuration too.

# On (the default): the heavy algorithms (NUMETRON_COMPILED_SOURCES below; now the multiplication
# chain above the basecase -- Karatsuba, the Toom engine and plans, the FFT --, the division, the
# thresholds and the kernel choice of src/backend.cpp) are compiled once, into this library, and
# NUMETRON_COMPILED is exported, so the headers only declare them instead of every translation
# unit that uses them compiling them. On every architecture, assembly or not.
# Off: nothing is built, and whatever links `numetron` gets the pure C++ header-only library. The
# library's C++ is compiled with whatever flags reach this target: the FFT kernel and the C++
# basecase follow its target (AVX2 + FMA, ADX + BMI2 or not), so give it the intended ones (e.g.
# target_compile_options(numetron PRIVATE -march=...)), as the top-level CMakeLists.txt of this
# repository does.
option(NUMETRON_COMPILED "Compile the heavy algorithms into the numetron library" ON)
set (NUMETRON_COMPILED_SOURCES
    ${LIBRARY_HOME}/src/backend.cpp
    ${LIBRARY_HOME}/src/umul_large.cpp
    ${LIBRARY_HOME}/src/udiv_large.cpp
)

# On (the default): the library uses the src/arch assembly on x86-64 (mul_basecase, sqr_basecase,
# add/sub_n, the Karatsuba kernels). Off: its C++ fallbacks (NUMETRON_NO_ASM). Needs
# NUMETRON_COMPILED: the assembly is part of the compiled back end.
# NUMETRON_USE_ASM, the former name of this option, is still read.
if (DEFINED NUMETRON_USE_ASM AND NOT DEFINED NUMETRON_ASM)
    message(DEPRECATION "Numetron: NUMETRON_USE_ASM is now NUMETRON_ASM")
    set(NUMETRON_ASM ${NUMETRON_USE_ASM} CACHE BOOL "")
endif()
option(NUMETRON_ASM "Use the src/arch assembly in the numetron library" ON)

# Off (the default): the library is numetron's own assembly only, MIT. On: the GMP-derived
# mul_basecase routines (src/arch/x86_64/{alderlake,core2,k8}, LGPL) are built in as well
# (NUMETRON_USE_GMP_LGPL), so they are picked per CPU; the result is subject to the LGPL.
option(NUMETRON_GMP_LGPL "Use the GMP-derived (LGPL) mul_basecase assembly" OFF)

# The largest n the squaring basecase (src/arch/x86_64/sqr_basecase_adx.s) has straight-line code
# for, 2..32: a smaller one trades speed at 17..32 limbs for code size (n = 17..32 are ~174 KB,
# which in a larger program compete for the instruction caches). Given to the library's C++ as
# well, so that its default squaring thresholds match it.
set(NUMETRON_SQR_STRAIGHT_MAX 32 CACHE STRING "Largest n with straight-line squaring code (2..32)")

# CMake cross-compile toolchains for Apple platforms (iOS/simulator via Xcode generator) don't
# populate CMAKE_SYSTEM_PROCESSOR - the target arch(es) only show up in CMAKE_OSX_ARCHITECTURES,
# which is a list (a universal/fat build can request more than one arch at once). Fall back to it
# on Apple platforms; everywhere else CMAKE_SYSTEM_PROCESSOR is still what's populated.
if (CMAKE_OSX_ARCHITECTURES)
    set (_numetron_target_archs ${CMAKE_OSX_ARCHITECTURES})
else()
    set (_numetron_target_archs ${CMAKE_SYSTEM_PROCESSOR})
endif()
list (REMOVE_DUPLICATES _numetron_target_archs)
list (LENGTH _numetron_target_archs _numetron_arch_count)

if (NOT NUMETRON_COMPILED)
    message(STATUS "Numetron: NUMETRON_COMPILED is OFF; nothing to build (the pure C++, header-only library).")
elseif (NOT NUMETRON_ASM)
    message(STATUS "Numetron: NUMETRON_ASM is OFF; building the library without ASM sources.")
elseif (_numetron_arch_count GREATER 1)
    message(STATUS "Numetron: multiple target archs (${_numetron_target_archs}); building without ASM sources.")
else()
    string(TOLOWER "${_numetron_target_archs}" _numetron_proc)
    if (_numetron_proc MATCHES "^(x86_64|amd64)$" OR CMAKE_GENERATOR_PLATFORM MATCHES "^[xX]64$")
        list (APPEND NUMETRON_LIB_SOURCES
            # GAS sources, MIT
            ${LIBRARY_HOME}/src/arch/x86_64/add_sub_n.s
            ${LIBRARY_HOME}/src/arch/x86_64/karatsuba_interp.s
            ${LIBRARY_HOME}/src/arch/x86_64/karatsuba_mul.s
            ${LIBRARY_HOME}/src/arch/x86_64/mul_basecase_adx.s
            ${LIBRARY_HOME}/src/arch/x86_64/sqr_basecase_adx.s
        )
        if (NUMETRON_GMP_LGPL)
            list (APPEND NUMETRON_LIB_SOURCES
                # GMP-derived, LGPL
                ${LIBRARY_HOME}/src/arch/x86_64/alderlake/mul_basecase.s
                ${LIBRARY_HOME}/src/arch/x86_64/k8/mul_basecase.s
                ${LIBRARY_HOME}/src/arch/x86_64/core2/mul_basecase.s
                # the CPU-family choice among them (MIT, but references the LGPL routines)
                ${LIBRARY_HOME}/src/arch/x86_64/detect_platform.s
                ${LIBRARY_HOME}/src/arch/x86_64/detect_mul_basecase.s
            )
        endif()
        set_source_files_properties(${LIBRARY_HOME}/src/arch/x86_64/sqr_basecase_adx.s
            PROPERTIES COMPILE_OPTIONS "-Wa,--defsym,NUMETRON_SQR_STRAIGHT_MAX=${NUMETRON_SQR_STRAIGHT_MAX}")
        set (_numetron_x64_asm ON)
    elseif (_numetron_proc MATCHES "^(arm|arm64|aarch64)$")
        #message(STATUS "Numetron: ARM assembly not available yet; building without ASM sources.")
        set (_numetron_aarch64_detect_src ${LIBRARY_HOME}/src/arch/aarch64/detect_mul_basecase.s)
        list (APPEND NUMETRON_LIB_SOURCES
            # MASM source(s)
            ${_numetron_aarch64_detect_src}
        )
        if (APPLE)
            # Mach-O needs a leading underscore on the C symbol and has no ELF-only
            # directives (.type/.size/.note.GNU-stack); the file picks between the two
            # ABIs via #ifdef __APPLE__, which needs cpp - .s (lowercase) isn't
            # preprocessed by default the way .S is, so force it here.
            set_source_files_properties(${_numetron_aarch64_detect_src}
                PROPERTIES COMPILE_OPTIONS "-x;assembler-with-cpp")
        endif()
    else()
        message(STATUS "Numetron: unknown arch '${_numetron_target_archs}'; building without ASM sources.")
    endif()
endif()

add_library(numetron_internal INTERFACE)

if (NUMETRON_COMPILED)
    list (APPEND NUMETRON_LIB_SOURCES ${NUMETRON_COMPILED_SOURCES})
    add_library(${LIBRARY_NAME} STATIC ${NUMETRON_LIB_SOURCES})

    # The library's configuration: private to it, and carried by numetron_internal. Without the
    # x86-64 assembly in the library, NUMETRON_NO_ASM tells its C++ so (on x86-64 the headers
    # would assume the assembly otherwise).
    set (_numetron_backend_definitions NUMETRON_SQR_STRAIGHT_MAX=${NUMETRON_SQR_STRAIGHT_MAX})
    if (NOT _numetron_x64_asm)
        list (APPEND _numetron_backend_definitions NUMETRON_NO_ASM)
    elseif (NUMETRON_GMP_LGPL)
        list (APPEND _numetron_backend_definitions NUMETRON_USE_GMP_LGPL)
    endif()
    target_compile_definitions(${LIBRARY_NAME} PRIVATE NUMETRON_BUILDING_LIBRARY ${_numetron_backend_definitions})
    target_compile_definitions(${LIBRARY_NAME} PUBLIC NUMETRON_COMPILED)
    # What the headers need to know of the library: that it has the assembly kernels (they call
    # them instead of their own C++ ones). A fact about this build, told to what links it.
    if (_numetron_x64_asm)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC NUMETRON_BACKEND_ASM)
    endif()
    target_include_directories(${LIBRARY_NAME} PUBLIC ${LIBRARY_HOME}/include)
    target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_23)

    target_compile_definitions(numetron_internal INTERFACE NUMETRON_BACKEND_INTERNAL ${_numetron_backend_definitions})
else()
    add_library(${LIBRARY_NAME} INTERFACE) # nothing to build; linking it stays valid
endif()
target_link_libraries(numetron_internal INTERFACE ${LIBRARY_NAME})

# Whichever project embeds this target (annium-lang, standalone or nested under sonia-ui) may set
# directory-scoped C++ flags/defines (e.g. add_compile_definitions(SONIA_PRIME_STATIC ...)) that
# the .s sources inherit too, though they're meaningless for a pure assembler compile - clang's
# assembler driver flags every one of them as unused. Silence that rather than fighting CMake's
# directory-scope flag inheritance from the parent project.
if (NUMETRON_COMPILED AND CMAKE_ASM_COMPILER_ID MATCHES "Clang")
    target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:ASM>:-Wno-unused-command-line-argument>)
endif()
