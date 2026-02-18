set (LIBRARY_NAME numetron)
set (LIBRARY_HOME ${CMAKE_CURRENT_SOURCE_DIR}/..)
set (NUMETRON_LIB_SOURCES)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _numetron_proc)
if (_numetron_proc MATCHES "^(x86_64|amd64)$" OR CMAKE_GENERATOR_PLATFORM MATCHES "^[xX]64$")
    list (APPEND NUMETRON_LIB_SOURCES
        # MASM source(s)
        ${LIBRARY_HOME}/src/arch/x86_64/alderlake/mul_basecase.s
        ${LIBRARY_HOME}/src/arch/x86_64/k8/mul_basecase.s
        ${LIBRARY_HOME}/src/arch/x86_64/core2/mul_basecase.s
        ${LIBRARY_HOME}/src/arch/x86_64/detect_platform.s
        ${LIBRARY_HOME}/src/arch/x86_64/detect_mul_basecase.s
    )
elseif (_numetron_proc MATCHES "^(arm|arm64|aarch64)$")
    message(STATUS "Numetron: ARM assembly not available yet; building without ASM sources.")
    list (APPEND NUMETRON_LIB_SOURCES
        # MASM source(s)
        ${LIBRARY_HOME}/src/arch/aarch64/detect_mul_basecase.s
    )
else()
    message(STATUS "Numetron: unknown arch '${CMAKE_SYSTEM_PROCESSOR}'; building without ASM sources.")
endif()

add_library(${LIBRARY_NAME} STATIC ${NUMETRON_LIB_SOURCES})