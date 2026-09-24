set (LIBRARY_NAME numetron)
set (LIBRARY_HOME ${CMAKE_CURRENT_SOURCE_DIR}/..)
set (NUMETRON_LIB_SOURCES)

# Off (the default): the library is numetron's own assembly only, MIT. On: the GMP-derived
# mul_basecase routines (src/arch/x86_64/{alderlake,core2,k8}, LGPL) are built in as well and
# NUMETRON_USE_GMP_LGPL is exported, so they are picked per CPU; the result is subject to the LGPL.
option(NUMETRON_GMP_LGPL "Use the GMP-derived (LGPL) mul_basecase assembly" OFF)

# Off: no assembly at all -- the target carries nothing, and whatever links it gets the pure C++
# (header-only) library.
option(NUMETRON_USE_ASM "Build the src/arch assembly and export NUMETRON_USE_ASM" ON)

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

if (NOT NUMETRON_USE_ASM)
    message(STATUS "Numetron: NUMETRON_USE_ASM is OFF; building without ASM sources (header-only C++).")
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

if (NUMETRON_LIB_SOURCES)
    add_library(${LIBRARY_NAME} STATIC ${NUMETRON_LIB_SOURCES})
else()
    add_library(${LIBRARY_NAME} INTERFACE) # nothing to build; linking it stays valid
endif()

# Without NUMETRON_USE_ASM the headers are pure C++ (header-only); linking the x86-64 assembly
# is what turns it on, for everything that links this target.
if (_numetron_x64_asm)
    target_compile_definitions(${LIBRARY_NAME} INTERFACE NUMETRON_USE_ASM)
    if (NUMETRON_GMP_LGPL)
        target_compile_definitions(${LIBRARY_NAME} INTERFACE NUMETRON_USE_GMP_LGPL)
    endif()
endif()

# This target only ever compiles .s sources. Whichever project embeds it (annium-lang,
# standalone or nested under sonia-ui) may set directory-scoped C++ flags/defines
# (e.g. add_compile_definitions(SONIA_PRIME_STATIC ...)) that get inherited here even
# though they're meaningless for a pure assembler compile - clang's assembler driver
# flags every one of them as unused. Silence that rather than fighting CMake's
# directory-scope flag inheritance from the parent project.
if (NUMETRON_LIB_SOURCES AND CMAKE_ASM_COMPILER_ID MATCHES "Clang")
    target_compile_options(${LIBRARY_NAME} PRIVATE -Wno-unused-command-line-argument)
endif()